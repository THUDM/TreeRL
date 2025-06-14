from typing import Optional, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
# from optimum.bettertransformer import BetterTransformer
# from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
# from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from .utils import log_probs_from_logits, replace_rope_embedding
import os

# https://github.com/microsoft/DeepSpeed/issues/4932
replace_rope_embedding()


class Actor(nn.Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        target_modules=None,
        ds_config=None,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            # assert not use_flash_attention_2
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Patch for https://github.com/huggingface/transformers/issues/28052
            def _autoset_attn_implementation_monkeypatch(cls, config, *args, **kwargs):  # type: ignore
                config._attn_implementation = attn_implementation
                return config

            PreTrainedModel._autoset_attn_implementation = classmethod(_autoset_attn_implementation_monkeypatch)

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            nf4_config = None

            print(f"########## loading actor model from: {pretrain_or_model} ##########")
            if "qwen" in pretrain_or_model.lower():
                if attn_implementation != "flash_attention_2":
                    attn_implementation = "sdpa"
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrain_or_model,
                    attn_implementation=attn_implementation,
                    quantization_config=nf4_config,
                    torch_dtype="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrain_or_model,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    quantization_config=nf4_config,
                    torch_dtype="auto",
                    empty_init=False,
                )

            # Mixtral 8x7b - balancing loss
            if "output_router_logits" in self.model.config.to_dict():
                print("[Mixtral 8x7b] set output_router_logits as True")
                self.model.config.output_router_logits = True
                deepspeed.utils.set_z3_leaf_modules(self.model, [MixtralSparseMoeBlock])
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, **kwargs
    ) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            # "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            # "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens ", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        for seq in sequences:
            if seq[-1].item() != pad_token_id:
                seq[-1] = eos_token_id

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        # attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        attention_mask = sequences.ne(pad_token_id).to(dtype=torch.long)

        # seq_length = attention_mask.size(1)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        
        exceeded = attention_mask[:, -1] > 0
        end_token_replace = torch.full((attention_mask.size(0),), pad_token_id, device=sequences.device)
        end_token_replace[exceeded] = eos_token_id
        sequences[:, -1] = end_token_replace
        
        ##############
        # eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        
        # end_tokens = torch.gather(sequences, dim=1, index=eos_indices - 1)
        # indices = (end_tokens == eos_token_id)
        # eos_indices[indices] -= 1
                
        # attention_mask.scatter_(dim=1, index=eos_indices, value=1)
        # sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)
        ##############
        
        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        # we only calculate the loss of state_i != eos | pad
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        # action_mask = state_seq.ne(pad_token_id)
        
        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        # https://github.com/OpenLLMAI/OpenRLHF/issues/217
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if return_output:
            return output if num_actions is None else (log_probs[:, -num_actions:], output)
        else:
            return log_probs[:, -num_actions:]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def to_bettertransformer(self):
        self.model = BetterTransformer.transform(self.model)

    def reverse_bettertransformer(self):
        self.model = BetterTransformer.reverse(self.model)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

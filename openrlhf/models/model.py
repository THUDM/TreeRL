from typing import Optional

import deepspeed
import torch
import torch.nn as nn
# from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
# from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from openrlhf.utils.logging import init_logger

from .utils import log_probs_from_logits, replace_rope_embedding
import pdb
# https://github.com/microsoft/DeepSpeed/issues/4932
replace_rope_embedding()

logger = init_logger(__name__)


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1310
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    device_map=None,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        normalize_reward (bool, optional): Whether normalize reward. Defaults to False.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert (
        model_type in ["reward", "reward_mix", "critic"]
    ), f"invalid model_type: {model_type}, should be critic or reward or reward_mix."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    try:
        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class)
        elif model_type == "reward_mix":
            cls_class = _get_reward_model_mix(base_pretrained_class, base_class)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class)
        print("Failed to load from AutoModel, construct from modelling file.")
        print(base_class, base_pretrained_class, cls_class)
        exit(0)
    except Exception as e:
        print("Failed to load from AutoModel, construct from modelling file.")
        module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(".")

        # special case
        if causal_model_name == "QWenLMHeadModel":
            auto_model_name = "QWenModel"
            pretrained_model_name = "QWenPreTrainedModel"
        elif causal_model_name == "InternLMForCausalLM":
            auto_model_name = "InternLMModel"
            pretrained_model_name = "InternLMPreTrainedModel"
        else:
            if "AutoModel" not in config.auto_map:
                auto_model_name = causal_model_name.split("For")[0] + "Model"
            else:
                auto_model_name = config.auto_map["AutoModel"].split(".")[1]
            pretrained_model_name = causal_model_name.split("For")[0] + "PreTrainedModel"

        logger.info(f"BASE_MODEL_CLASS: {auto_model_name}, PRETRAINED_MODEL_CLASS: {pretrained_model_name}")

        base_pretrained_class = get_class_from_dynamic_module(
            f"{module_file}.{pretrained_model_name}", model_name_or_path
        )
        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class)
        elif model_type == "reward_mix":
            cls_class = _get_reward_model_mix(base_pretrained_class, base_class)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
        empty_init = False
    else:
        dschf = None
        empty_init = True

    # assert empty_init is False
    # if load_in_4bit:
    #     assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
    #     nf4_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #     )
    # else:
    nf4_config = None

    if "glm" in model_name_or_path.lower():
        config.empty_init = empty_init
        model = cls_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
            quantization_config=nf4_config,
            device_map=device_map,
            # empty_init=empty_init,
            **kwargs,
        )
    else:
        model = cls_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
            quantization_config=nf4_config,
            device_map=device_map,
            # empty_init=empty_init,
            **kwargs,
        )

    # LoRA
    # if lora_rank > 0:
    #     model.enable_input_require_grads()
    #     lora_config = LoraConfig(
    #         r=lora_rank,
    #         lora_alpha=lora_alpha,
    #         target_modules=target_modules or find_all_linear_names(model, load_in_4bit),
    #         lora_dropout=0,
    #         bias="none",
    #     )
    #     model = get_peft_model(model, lora_config)

    #     if load_in_4bit:
    #         for name, module in model.named_modules():
    #             if isinstance(module, LoraLayer):
    #                 module = module.to(torch.bfloat16)
    #             if "norm" in name:
    #                 module = module.to(torch.float32)
    #             if "value_head" in name or "embed_tokens" in name:
    #                 if hasattr(module, "weight"):
    #                     module = module.to(torch.bfloat16)

    # Mixtral 8x7b - balancing loss
    if "output_router_logits" in model.config.to_dict():
        print("[Mixtral 8x7b] set output_router_logits as True")
        model.config.output_router_logits = True
        deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model_mix(base_pretrained_model, base_llm_model):
    class LLMForSequenceRegression(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            # config.empty_init = True
            if "glm" in str(base_llm_model):
                self._model_name = "chatglm" 
                setattr(
                    self, 
                    self.base_model_prefix, 
                    getattr(base_llm_model(config), self.base_model_prefix)
                )    
            else:
                self._model_name = "general"
                setattr(self, self.base_model_prefix, base_llm_model(config))
        
            self.value_head = nn.Linear(config.hidden_size, 2, bias=False)

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        @classmethod
        def _autoset_attn_implementation(cls, config, *args, **kwargs):
            logger.info(
                "Monkey patch for Flash Attention, see https://github.com/huggingface/transformers/issues/28052"
            )
            return config

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            process_supervision=False
        ) -> torch.Tensor:
            # https://github.com/OpenLLMAI/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]

            batch_size = input_ids.shape[0]
            if "glm" in self._model_name:
                if last_hidden_states.shape[0] != batch_size:
                    last_hidden_states = last_hidden_states.transpose(0, 1)

            values = self.value_head(last_hidden_states).squeeze(-1)

            VALID_INDEX = 1
            if not process_supervision:
                # left padding in training mode
                if self.training:
                    # reward = values[:, -1]
                    reward = values[..., VALID_INDEX]
                    reward = reward[:, -1]
                else:
                    values = values[..., VALID_INDEX]
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                    # normalize reward in eval mode
                    if self.normalize_reward:
                        reward = (reward - self.mean) / self.std
            else:
                if self.training:    
                    if return_output:
                        return values, outputs
                    else:
                        return values
                else:
                    values = values[..., VALID_INDEX].squeeze(-1)
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                    # normalize reward in eval mode
                    if self.normalize_reward:
                        reward = (reward - self.mean) / self.std
                    
                
            if return_output:
                return reward, outputs
            else:
                return reward

    return LLMForSequenceRegression


def _get_reward_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceRegression(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            # config.empty_init = True
            if "glm" in str(base_llm_model):
                self._model_name = "chatglm" 
                setattr(
                    self, 
                    self.base_model_prefix, 
                    getattr(base_llm_model(config), self.base_model_prefix)
                )    
            else:
                self._model_name = "general"
                setattr(self, self.base_model_prefix, base_llm_model(config))
        
            self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        @classmethod
        def _autoset_attn_implementation(cls, config, *args, **kwargs):
            logger.info(
                "Monkey patch for Flash Attention, see https://github.com/huggingface/transformers/issues/28052"
            )
            return config

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            process_supervision=False
        ) -> torch.Tensor:
            # https://github.com/OpenLLMAI/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]

            # if "glm" in self._model_name:
                # last_hidden_states = last_hidden_states.transpose(0, 1)
            batch_size = input_ids.shape[0]
            if "glm" in self._model_name:
                if last_hidden_states.shape[0] != batch_size:
                    last_hidden_states = last_hidden_states.transpose(0, 1)
                    
            values = self.value_head(last_hidden_states).squeeze(-1)

            if not process_supervision:
                # left padding in training mode
                if self.training:
                    reward = values[:, -1]
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                    # normalize reward in eval mode
                    if self.normalize_reward:
                        reward = (reward - self.mean) / self.std
            else:
                if return_output:
                    return values, outputs
                else:
                    return values
            
            if return_output:
                return reward, outputs
            else:
                return reward

    return LLMForSequenceRegression


def _get_critic_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceRegression(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            if "glm" in str(base_llm_model):
                self._model_name = "chatglm" 
                setattr(
                    self, 
                    self.base_model_prefix, 
                    getattr(base_llm_model(config), self.base_model_prefix)
                )    
            else:
                self._model_name = "general"
                setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        @classmethod
        def _autoset_attn_implementation(cls, config, *args, **kwargs):
            logger.info(
                "Monkey patch for Flash Attention, see https://github.com/huggingface/transformers/issues/28052"
            )
            return config

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            # https://github.com/OpenLLMAI/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            last_hidden_states = outputs["last_hidden_state"]
            # if "glm" in self._model_name:
            #     last_hidden_states = last_hidden_states.transpose(0, 1)
            batch_size = input_ids.shape[0]
            if "glm" in self._model_name:
                if last_hidden_states.shape[0] != batch_size:
                    last_hidden_states = last_hidden_states.transpose(0, 1)
                
            values = self.value_head(last_hidden_states).squeeze(-1)[:, :-1]
            num_actions = action_mask.size(1)

            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            if return_output:
                return outputs if num_actions is None else (values[:, -num_actions:], outputs)
            else:
                return values[:, -num_actions:]

    return LLMForSequenceRegression

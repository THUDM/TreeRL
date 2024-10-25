import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, masked_mean
from openrlhf.utils.logging import init_logger
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker, Experience


logger = init_logger(__name__)


class RemoteExperienceMakerPPO(RemoteExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines


    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        experiences = self.sample_responses(
            prompts,
            getattr(self.strategy.args, "num_trace_per_sample", 1),  
            **generate_kwargs
        )
        action_mask = experiences["action_mask"]
        attention_mask = experiences["attention_mask"]
        
        reward, kl = compute_reward(
            experiences["reward"],
            self.kl_ctl.value,
            experiences["action_log_probs"],
            experiences["base_action_log_probs"],
            action_mask=action_mask,
            process_reward=self.strategy.args.process_supervision
        )
        advantage, returns = self.get_advantages_and_returns(
            experiences["value"],
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )
        
        # action_mask = action_mask.float()
        # mask loss for eos_token
        # eos_indices = action_mask.float().shape[1] - 1 - action_mask.float().fliplr().argmax(dim=1)
        # mask = torch.arange(action_mask.size(1), device=action_mask.device).unsqueeze(0) == eos_indices.unsqueeze(1)
        # advantage = torch.where((advantage < 0) * mask, torch.zeros_like(advantage), advantage)
        
        def reformat_reward_for_info(piece):
            if len(piece.shape) == 1:
                return piece
            elif self.strategy.args.process_supervision:
                mask = piece != 0
                piece = masked_mean(piece, mask, dim=-1)
            elif piece.shape[1] > 1:
                piece = masked_mean(piece, action_mask, dim=-1)
            return piece
        
        response_entropy = -(experiences["action_log_probs"] * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
        # response_entropy = -response_entropy.mean(dim=1

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": reformat_reward_for_info(experiences["raw_reward"]),
            "reward_normalized": reformat_reward_for_info(experiences["reward"]),
            "return": (reward * action_mask).sum(dim=-1),
            "advantage": (advantage * action_mask).sum(-1) / action_mask.sum(-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
            "response_overlong_ratio": (action_mask[:, -1] == 0).float(),
            "pass_rate": experiences["pass_rate"] if "pass_rate" in experiences else 0,
            "response_entropy": response_entropy,
        }
        if self.strategy.args.perf:
            info = self.log_perf(info, experiences, getattr(self.strategy.args, "num_trace_per_sample", 1))

        experience = Experience(
            experiences["sequences"],
            experiences["action_log_probs"],
            experiences["value"],
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience

    @torch.no_grad()
    def _independent_make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()

        # generate sequence
        
        num_trace_per_sample = getattr(self.strategy.args, "num_trace_per_sample", 1)
        
        _wait_time = 0
        _generate_time = 0
        _actor_time = 0
        # _advantage = []
        # _returns = []
        _sequences = []
        _action_log_probs = []
        _value = []
        _attention_mask = []
        _action_mask = []
        # _kl = []
        _raw_r = []
        # _reward = []
        _base_action_log_probs = []
        
        if isinstance(prompts, list) and len(prompts) == 2: 
            batch_size = len(prompts[0])
        else:
            batch_size = len(prompts)
        
        for _ in range(num_trace_per_sample):
            start = time.time()
            
            sequences, attention_mask, action_mask = (
                self._generate_local(prompts, **generate_kwargs)
                if self.vllm_engines is None
                else self._generate_vllm(prompts, **generate_kwargs)
            )
            assert batch_size == int(sequences.shape[0]), f"len(prompts): {len(prompts)}---{len(prompts[0])}, sequences.shape[0]: {sequences.shape[0]}, {type(prompts)}, {prompts}"
            # print(f"sequences.shape: {sequences.shape}, num_prompts: {len(prompts)}")
            
            generate_time = time.time() - start

            num_actions = action_mask.size(1)
            sequences_cpu, attention_mask_cpu, action_mask_cpu = (
                sequences.to("cpu"),
                attention_mask.to("cpu"),
                action_mask.to("cpu"),
            )

            # init log probs
            base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

            # values
            value_ref = self.critic.forward.remote(sequences_cpu, action_mask_cpu, attention_mask_cpu)

            # rewards
            r_refs = []
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu))

            # log probs
            start = time.time()
            action_log_probs = self.actor(sequences, num_actions, attention_mask)
            actor_time = time.time() - start

            # wait initial/critic/reward model done
            start = time.time()
            ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
            wait_time = time.time() - start

            base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
            base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
            rewards = [r.to(device) for r in rewards]
            r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

            _base_action_log_probs.append(base_action_log_probs)

            # reward, kl = compute_reward(
            #     r,
            #     self.kl_ctl.value,
            #     action_log_probs,
            #     base_action_log_probs,
            #     action_mask=action_mask,
            # )

            # advantage, returns = self.get_advantages_and_returns(
            #     value,
            #     reward,
            #     action_mask,
            #     generate_kwargs["gamma"],
            #     generate_kwargs["lambd"],
            # )
            
            # _advantage.append(advantage)
            # _returns.append(returns)

            _attention_mask.append(attention_mask)
            _action_mask.append(action_mask)
            _value.append(value)
            _sequences.append(sequences)
            _action_log_probs.append(action_log_probs)
            # _kl.append(kl)
            _raw_r.append(r)
            # _reward.append(reward)

            _wait_time += wait_time
            _generate_time += generate_time
            _actor_time += actor_time
            
        action_mask = zero_pad_batch(_action_mask, side="right")
        # advantage = zero_pad_batch(_advantage, side="right")
        # returns = zero_pad_batch(_returns, side="right")
        # kl = zero_pad_batch(_kl, side="right")

        attention_mask = zero_pad_batch(_attention_mask, side="right")
        value = zero_pad_batch(_value, side="right")
        sequences = zero_pad_batch(_sequences, side="right")
        action_log_probs = zero_pad_batch(_action_log_probs, side="right")
        base_action_log_probs = zero_pad_batch(_base_action_log_probs, side="right")
        r = torch.cat(_raw_r, dim=0)

        _raw_reward = r
        if num_trace_per_sample > 1 and self.strategy.args.normalize_reward_from_multi_traces:
            r = normalize_reward_from_multi_traces(r, batch_size, num_trace_per_sample)

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask
        )

        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        # reward = zero_pad_batch(_reward, side="right")
        generate_time, actor_time, wait_time = _generate_time, _actor_time, _wait_time
        
        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": _raw_reward,
            "reward_normalized": r,
            "return": reward.sum(dim=-1),
            "advantage": (advantage * action_mask).sum(-1) / action_mask.sum(-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }

        if self.strategy.args.perf:
            # batch_size = 1 if isinstance(prompts, str) else len(prompts)
            # batch_size = len()
            batch_size = batch_size * num_trace_per_sample
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience
    
    @torch.no_grad()
    def _past_make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()

        # generate sequence
        start = time.time()
        
        num_trace_per_sample = getattr(self.strategy.args, "num_trace_per_sample", 1)
        if num_trace_per_sample > 1:
            # prompts = prompts * num_trace_per_sample
            _prompts = []
            for item in prompts:
                if isinstance(item, list):
                    _prompts.append(item * num_trace_per_sample)
                elif torch.is_tensor(item):
                    _prompts.append(item.repeat(num_trace_per_sample))
                else:
                    raise ValueError(f"Unsupported type: {type(item)}")
            prompts = _prompts
                    
        sequences, attention_mask, action_mask = (
            self._generate_local(prompts, **generate_kwargs)
            if self.vllm_engines is None
            else self._generate_vllm(prompts, **generate_kwargs)
        )
        generate_time = time.time() - start

        num_actions = action_mask.size(1)
        sequences_cpu, attention_mask_cpu, action_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
            action_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

        # values
        value_ref = self.critic.forward.remote(sequences_cpu, action_mask_cpu, attention_mask_cpu)

        # rewards
        r_refs = []
        for rm in self.reward_model:
            r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu))

        # log probs
        start = time.time()
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        actor_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )
        

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }

        if self.strategy.args.perf:
            batch_size = 1 if isinstance(prompts, str) else len(prompts)
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        return self.actor.generate(**inputs, **kwargs)

    # def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     from vllm import SamplingParams

    #     # round-robin load balance
    #     rank = torch.distributed.get_rank()
    #     llm = self.vllm_engines[rank % len(self.vllm_engines)]

    #     sampling_params = SamplingParams(
    #         temperature=kwargs.get("temperature", 1.0),
    #         top_p=kwargs.get("top_p", 1.0),
    #         top_k=kwargs.get("top_k", -1),
    #         max_tokens=kwargs.get("max_new_tokens", 16),
    #         stop_token_ids=[self.tokenizer.convert_tokens_to_ids("<|user|>"), self.tokenizer.convert_tokens_to_ids("<|observation|>")],
    #         # stop=["<|user|>", "<|observation|>"],
    #     )

    #     # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
    #     input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
    #     assert self.tokenizer.padding_side == "left", f"tokenizer padding_size should be left"
    #     pad_indices = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.int).argmax(dim=-1)
    #     prompt_token_ids = []
    #     for i, pad_index in enumerate(pad_indices.numpy()):
    #         prompt_token_ids.append(input_ids[i][pad_index:].tolist())
    #     outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

    #     # NOTE: concat all outputs to following format:
    #     #
    #     # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
    #     # | token token token token token | token token [EOS] [PAD] |
    #     # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
    #     # |<---------- prompt ----------->|<-------- answer ------->|
    #     if "glm" in self.current_model:
    #         eos_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")
    #         eos_token_set = (self.tokenizer.convert_tokens_to_ids("<|user|>"), self.tokenizer.convert_tokens_to_ids("<|observation|>"), self.tokenizer.eos_token_id)
    #     else:
    #         eos_token_id = self.tokenizer.eos_token_id
    #         eos_token_id_set = (self.tokenizer.eos_token_id)

    #     max_input_len, max_output_len = 0, 0
    #     for output in outputs:
    #         # TODO: how to force vLLM generate at least one token?
    #         output_token_ids = output.outputs[0].token_ids
    #         if output_token_ids[0] == eos_token_id:
    #             logger.warning(f"Only EOS output for prompt: {output.prompt}")
    #             output.outputs[0].token_ids = [self.tokenizer.unk_token_id, eos_token_id]

    #         max_input_len = max(max_input_len, len(output.prompt_token_ids))
    #         max_output_len = max(max_output_len, len(output_token_ids))

    #     pad_token_id = self.tokenizer.pad_token_id
    #     # pad_token_id = self.tokenizer.pad_token_idp
    #     # eos_token_id = 
    #     sequences = []
    #     for output in outputs:
    #         # left padding input
    #         input_len = len(output.prompt_token_ids)
    #         input_ids = [pad_token_id] * (max_input_len - input_len) + output.prompt_token_ids

    #         # right padding output
    #         output_len = len(output.outputs[0].token_ids)
    #         output_ids = output.outputs[0].token_ids + [pad_token_id] * (max_output_len - output_len)
    #         if int(output_ids[output_len - 1]) not in eos_token_set:
    #             assert output_len == max_output_len, f"output_len: {output_len}, max_output_len: {max_output_len}, output_ids: {output_ids[output_len-5:output_len+1]}"
    #             output_ids[-1] = eos_token_id

    #         # concat input and output
    #         sequences.append(input_ids + output_ids)

    #     sequences = torch.tensor(sequences)
    #     sequences, attention_mask, action_mask = self.actor.process_sequences(
    #         sequences, max_input_len, eos_token_id, pad_token_id
    #     )
    #     return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None

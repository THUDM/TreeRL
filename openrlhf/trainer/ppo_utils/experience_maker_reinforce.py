import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm
import json

# from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, compute_reward_naive, masked_mean
from openrlhf.utils.logging import init_logger
from .experience_maker import RemoteExperienceMaker, Experience


logger = init_logger(__name__)
        

class RemoteExperienceMakerReinforce(RemoteExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, **kwargs):
        # if "critic" not in kwargs:
            # kwargs["critic"] = None
        assert args[1] == None, f"critic should be None but got {args[1]}"
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines


    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        # device = torch.cuda.current_device()

        # num_trace_per_sample = getattr(self.strategy.args, "num_trace_per_sample", 1)
        # if num_trace_per_sample > 1:
        #     prompts = prompts * num_trace_per_sample
        
        # # generate sequence
        # start = time.time()
        # sequences, attention_mask, action_mask = (
        #     self._generate_local(prompts, **generate_kwargs)
        #     if self.vllm_engines is None
        #     else self._generate_vllm(prompts, **generate_kwargs)
        # )
        # generate_time = time.time() - start

        # num_actions = action_mask.size(1)
        # sequences_cpu, attention_mask_cpu = (
        #     sequences.to("cpu"),
        #     attention_mask.to("cpu"),
        #     # action_mask.to("cpu"),
        # )

        # # init log probs
        # base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

        # # rewards
        # r_refs = []
        # for rm in self.reward_model:
        #     r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu))

        # # log probs
        # start = time.time()
        # action_log_probs = self.actor(sequences, num_actions, attention_mask)
        # actor_time = time.time() - start

        # # wait initial/critic/reward model done
        # start = time.time()
        # ref_values = ray.get([base_action_log_probs_ref] + r_refs)
        # wait_time = time.time() - start

        # base_action_log_probs, rewards = ref_values[0], ref_values[1:]
        # base_action_log_probs = base_action_log_probs.to(device)
        # rewards = [r.to(device) for r in rewards]
        # r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        experiences = self.sample_responses(
            prompts,
            getattr(self.strategy.args, "num_trace_per_sample", 1), 
            **generate_kwargs
        )
        action_mask = experiences["action_mask"]
        attention_mask = experiences["attention_mask"]
        
            
        experience_reward = experiences["reward"]
        # experience_reward[experience_reward < 0] *= 0.5
        reward, kl = compute_reward_naive(
            experience_reward,
            self.kl_ctl.value,
            experiences["action_log_probs"],
            experiences["base_action_log_probs"],
            action_mask=action_mask,
            kl_as_reward=True,
            process_reward=self.strategy.args.process_supervision
        )
                
        # * debuging
        if self.strategy.get_rank() == 0:
            print(f"----------- normalized_rewards: {experience_reward}, reward_with_kl: {reward}")
        
        def reformat_reward_for_info(piece):
            if len(piece.shape) == 1:
                return piece
            elif self.strategy.args.process_supervision:
                mask = piece != 0
                piece = masked_mean(piece, mask, dim=-1)
            elif piece.shape[1] > 1:
                piece = masked_mean(piece, action_mask, dim=-1)
            return piece

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": reformat_reward_for_info(experiences["raw_reward"]),
            "reward_normalized": reformat_reward_for_info(experiences["reward"]),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
            "response_overlong_ratio": (action_mask[:, -1] == 0).float(),
        }

        if self.strategy.args.perf:
            info = self.log_perf(info, experiences, getattr(self.strategy.args, "num_trace_per_sample", 1))

        # mask loss for eos_token
        # eos_indices = action_mask.float().shape[1] - 1 - action_mask.float().fliplr().argmax(dim=1)
        # mask = torch.arange(action_mask.size(1), device=action_mask.device).unsqueeze(0) == eos_indices.unsqueeze(1)
        # reward = torch.where((reward < 0) * mask, torch.zeros_like(reward), reward)

        experience = Experience(
            sequences=experiences["sequences"],
            action_log_probs=experiences["action_log_probs"],
            advantages=reward,
            attention_mask=attention_mask,
            action_mask=action_mask,
            info=info,
            kl=kl
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        # self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience

    def flush(self):
        "Ensure all experience has been send to critic"
        # ray.get(self._ref)
        self._ref = None

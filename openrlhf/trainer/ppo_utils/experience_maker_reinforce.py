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
from .experience_maker import RemoteExperienceMaker, Experience, NaiveExperienceMaker


logger = init_logger(__name__)


class NaiveExperienceMakerReinforce(NaiveExperienceMaker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    @torch.no_grad()
    def _make_experience(self, prompts: Union[str, List[str]],**generate_kwargs) -> Experience:
        self.actor.eval()
        use_mcts = getattr(generate_kwargs, "use_mcts", 0)

        if use_mcts:
            print("using mcts!!")
            experiences = self.sample_responses_bymcts(
            prompts,
            getattr(self.strategy.args, "num_trace_per_sample", 1), 
            **generate_kwargs
        )
        else:
            print("not using mcts!!")
            experiences = self.sample_responses(
                prompts,
                getattr(self.strategy.args, "num_trace_per_sample", 1), 
                **generate_kwargs
            )
        action_mask = experiences["action_mask"]
        attention_mask = experiences["attention_mask"]
        
        experience_reward = experiences["reward"]
        overlong_mask = experiences["overlong_mask"]
        # with open("/workspace/lurui/openrlhf-glm/logs/outputs/overlong_mask.jsonl","a") as f:
        #     f.write(json.dumps(overlong_mask.tolist()) + "\n")
        print("logging overlong mask",experiences["overlong_mask"], overlong_mask.sum() / overlong_mask.numel())
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

        # sample_kl = (kl.abs() * action_mask).sum(1) / action_mask.sum(1)
        # sample_kl_mask = (sample_kl <= 0.3).view(-1, 1).float()
        # reward = reward * sample_kl_mask
        # reward = reward.clamp(min=-0.5)

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

        response_entropy = -(experiences["action_log_probs"] * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
        print("action_mask_lastdigit",action_mask[:,-1])
        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": reformat_reward_for_info(experiences["raw_reward"]),
            "reward_normalized": reformat_reward_for_info(experiences["reward"]),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
            # "response_overlong_ratio": (action_mask[:, -1] == 0).float(),
            "response_overlong_ratio": overlong_mask.sum() / overlong_mask.numel(),
            "pass_rate": experiences["pass_rate"] if "pass_rate" in experiences else 0,
            "response_entropy": response_entropy,
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
    
    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        return self._make_experience(prompts, **generate_kwargs)


class RemoteExperienceMakerReinforce(RemoteExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, **kwargs):
        # if "critic" not in kwargs:
            # kwargs["critic"] = None
        assert args[1] == None, f"critic should be None but got {args[1]}"
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines


    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], use_mcts, use_vinevalue, use_sentence_level_value, **generate_kwargs) -> Experience:
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
        # use_mcts = getattr(generate_kwargs, "use_mcts", 1)
        # use_vinevalue = getattr(generate_kwargs, "use_vinevalue", 0)
        print("use_mcts", use_mcts)
        print("use_vinevalue", use_vinevalue)
        print("file_name", str(getattr(self.strategy.args, "wandb_run_name", "test"))+ ".jsonl")

        if use_mcts:
            if use_vinevalue:
                experiences = self.sample_responses_bymcts_use_vinevalue(
                    prompts,
                    getattr(self.strategy.args, "num_trace_per_sample", 1), 
                    file_name = "logs/samplings/"+str(getattr(self.strategy.args, "wandb_run_name", "test"))+ ".jsonl",
                    use_sentence_level_value = use_sentence_level_value,
                    **generate_kwargs
                )
            else:
                experiences = self.sample_responses_bymcts(
                    prompts,
                    getattr(self.strategy.args, "num_trace_per_sample", 1), 
                    file_name = "logs/samplings/"+str(getattr(self.strategy.args, "wandb_run_name", "test"))+ ".jsonl",
                    **generate_kwargs
            )
        else:
            print("not use mcts!!")
            experiences = self.sample_responses(
                prompts,
                getattr(self.strategy.args, "num_trace_per_sample", 1), 
                file_name = "logs/samplings/"+str(getattr(self.strategy.args, "wandb_run_name", "test"))+ ".jsonl",
                **generate_kwargs
            )
        action_mask = experiences["action_mask"]
        attention_mask = experiences["attention_mask"]
        
        experience_reward = experiences["reward"]
        overlong_mask = experiences["overlong_mask"]
        # experience_reward[experience_reward < 0] *= 0.5
        if use_vinevalue and (not use_sentence_level_value):
            print("reward: ",experience_reward)
            print("value: ",experiences["values"])
            reward, kl = compute_reward(
                experiences["reward"],
                self.kl_ctl.value,
                experiences["action_log_probs"],
                experiences["base_action_log_probs"],
                action_mask=action_mask,
                process_reward=self.strategy.args.process_supervision,
                value = experiences["values"]
            )
            advantage, returns = self.get_advantages_and_returns(
                experiences["values"],
                reward,
                action_mask,
                generate_kwargs["gamma"],
                generate_kwargs["lambd"],
                experiences["sequences"]
            )
        elif use_vinevalue and use_sentence_level_value:
            print("reward: ",experience_reward)
            print("value: ",experiences["values"])
            advantage, returns = self.get_advantages_and_returns(
                experiences["values"],
                reward,
                action_mask,
                generate_kwargs["gamma"],
                generate_kwargs["lambd"],
            )
            # print("advantage",advantage)
            # with open("/workspace/lurui/openrlhf-glm/logs/outputs/advantage.jsonl","a") as f:
            #     f.write(json.dumps({"advantage":advantage.tolist()}) + "\n")
        else:
            reward, kl = compute_reward_naive(
                experience_reward,
                self.kl_ctl.value,
                experiences["action_log_probs"],
                experiences["base_action_log_probs"],
                action_mask=action_mask,
                kl_as_reward=True,
                process_reward=self.strategy.args.process_supervision
            )
        # sample_kl = (kl.abs() * action_mask).sum(1) / action_mask.sum(1)
        # sample_kl_mask = (sample_kl <= 0.3).view(-1, 1).float()
        # reward = reward * sample_kl_mask
        # reward = reward.clamp(min=-0.5)

        # * debuging
        if self.strategy.get_rank() == 0:
            print(f"----------- normalized_rewards: {experience_reward}, reward_with_kl: {reward}")
        
        def reformat_reward_for_info(piece):
            piece_origin = piece
            if len(piece.shape) == 1:
                return piece
            elif self.strategy.args.process_supervision:
                mask = piece != 0
                piece = masked_mean(piece, mask, dim=-1)
            elif piece.shape[1] > 1:
                piece = masked_mean(piece, action_mask, dim=-1)
            print("reward to log",piece_origin, piece)
            return piece

        response_entropy = -(experiences["action_log_probs"] * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
        # with open("/workspace/lurui/openrlhf-glm/logs/outputs/advantage.jsonl","a") as f:
        #     sequences = experiences["sequences"]
        #     num_actions = action_mask.size(1)
        #     for s in range(sequences.shape[0]):
        #         match_list = []
        #         for i in range(reward[0].shape[0]):
        #             str_seq = self.tokenizer.decode([sequences[s][-num_actions+i].to("cpu").tolist()], skip_special_tokens=True)
        #             match_list.append({"reward":reward[s][i].item(),"content":str_seq})
        #         f.write(json.dumps(match_list) + "\n")
        
        if use_vinevalue:
            info = {
                "kl": masked_mean(kl, action_mask, dim=-1),
                "reward": reformat_reward_for_info(experiences["raw_reward"]),
                "value": reformat_reward_for_info(experiences["values"]),
                "advantage": reformat_reward_for_info(advantage),
                "returns": reformat_reward_for_info(returns),
                "reward_normalized": reformat_reward_for_info(experiences["reward"]),
                "response_length": action_mask.float().sum(dim=-1),
                "total_length": attention_mask.float().sum(dim=-1),
                # "response_overlong_ratio": (action_mask[:, -1] == 0).float(),
                "response_overlong_ratio": overlong_mask.float(),
                "pass_rate": experiences["pass_rate"] if "pass_rate" in experiences else 0,
                "pass_at_1": experiences["pass_at_1"] if "pass_at_1" in experiences else 0,
                "response_entropy": response_entropy,
            }
        elif use_mcts:
            #输出reformat_reward_for_info(experiences["raw_reward"]是否有NAN
            print("reward to log",experiences["raw_reward"].shape,any(torch.isnan(reformat_reward_for_info(experiences["raw_reward"]))) )
            info = {
                "kl": masked_mean(kl, action_mask, dim=-1),
                "reward": reformat_reward_for_info(experiences["raw_reward"]),
                "reward_normalized": reformat_reward_for_info(experiences["reward"]),
                "response_length": action_mask.float().sum(dim=-1),
                "total_length": attention_mask.float().sum(dim=-1),
                # "response_overlong_ratio": (action_mask[:, -1] == 0).float(),
                "response_overlong_ratio": overlong_mask.float(),
                "pass_rate": experiences["pass_rate"] if "pass_rate" in experiences else 0,
                "pass_at_1": experiences["pass_at_1"] if "pass_at_1" in experiences else 0,
                "response_entropy": response_entropy,
            }
        else:
            info = {
                "kl": masked_mean(kl, action_mask, dim=-1),
                "reward": reformat_reward_for_info(experiences["raw_reward"]),
                "reward_normalized": reformat_reward_for_info(experiences["reward"]),
                "response_length": action_mask.float().sum(dim=-1),
                "total_length": attention_mask.float().sum(dim=-1),
                # "response_overlong_ratio": (action_mask[:, -1] == 0).float(),
                "response_overlong_ratio": overlong_mask.float(),
                "pass_rate": experiences["pass_rate"] if "pass_rate" in experiences else 0,
                "pass_at_1": experiences["pass_at_1"] if "pass_at_1" in experiences else 0,
                "response_entropy": response_entropy,
            }

        if self.strategy.args.perf:
            info = self.log_perf(info, experiences, getattr(self.strategy.args, "num_trace_per_sample", 1))

        # mask loss for eos_token
        # eos_indices = action_mask.float().shape[1] - 1 - action_mask.float().fliplr().argmax(dim=1)
        # mask = torch.arange(action_mask.size(1), device=action_mask.device).unsqueeze(0) == eos_indices.unsqueeze(1)
        # reward = torch.where((reward < 0) * mask, torch.zeros_like(reward), reward)
        if use_vinevalue:
            experience = Experience(
                sequences=experiences["sequences"],
                action_log_probs=experiences["action_log_probs"],
                values = experiences["values"],
                advantages = advantage,
                returns=returns,
                attention_mask=attention_mask,
                action_mask=action_mask,
                info=info,
                kl=kl
            )
        else:
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


def segment_actions(sequences, attention_mask, action_mask, tokenizer):
    for seq, attn_mask, act_mask in zip(sequences, attention_mask, action_mask):
        seq = seq[attn_mask.bool()]
        text = tokenizer.decode(seq.tolist(), skip_special_tokens=False)
        
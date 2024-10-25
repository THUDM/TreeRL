from itertools import chain
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
import torch.utils
from tqdm import tqdm
import json

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, masked_mean
from openrlhf.utils.logging import init_logger
from openrlhf.datasets.reward_dataset import (
    reformat_response_into_steps, 
    # revert_special_tokens, 
    get_process_flag_tokens, 
    get_process_flag
)

logger = init_logger(__name__)


def get_eos_token_id(tokenizer):
    return tokenizer.convert_tokens_to_ids("<|user|>")

def get_eos_token(tokenizer):
    return "<|user|>"


# normalize reward
def normalize_reward_from_multi_traces(
        reward, 
        batch_size, 
        num_trace_per_sample,
        min_threshold=0,
        batch_first=False,
    ):
    reward_raw = reward.clone()

    if not batch_first:
        if reward.numel() != batch_size * num_trace_per_sample:
            print(f"******* Problem: {reward.shape} != {batch_size} * {num_trace_per_sample}") 
            reward = reward.view(num_trace_per_sample, -1)
        else:
            reward = reward.view(num_trace_per_sample, batch_size)
        
        mean = reward.mean(dim=0).view(1, -1)
        std = reward.std(dim=0).view(1, -1)
        reward = reward - mean
        reward = reward / (std + 1e-6)

        if min_threshold > 0:
            reward_raw = reward_raw.view(num_trace_per_sample, batch_size)
            reward_raw -= mean
            reward_mask = (reward_raw.abs() > min_threshold) | (reward_raw > 0)
            reward_mask = reward_mask.float()
            reward = reward * reward_mask
        
        reward = reward.view(-1)
    else:
        reward = reward.view(batch_size, num_trace_per_sample)
        mean = reward.mean(dim=1).view(-1, 1)
        std = reward.std(dim=1).view(-1, 1)
        reward = reward - mean
        reward = reward / (std + 1e-6)
        if min_threshold > 0:
            reward_raw = reward_raw.view(batch_size, num_trace_per_sample)
            reward_raw -= mean
            reward_mask = (reward_raw.abs() > min_threshold) | (reward_raw > 0)
            reward_mask = reward_mask.float()
            reward = reward * reward_mask
        reward = reward.view(-1)
    return reward
    
    # reward = reward.view(batch_size, num_trace_per_sample)
    # mean = reward.mean(dim=1).view(-1, 1)
    # std = reward.std(dim=1).view(-1, 1)
    # reward = reward - mean
    # reward = reward / (std + 1e-6)
    # reward = reward.view(-1)

    return reward
    

def zero_pad_batch(sequences: List[torch.Tensor], side: str = "right", pad_token_id: int = 0) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=pad_token_id))
    return torch.cat(padded_sequences, dim=0)


def _tokenize_fn_chatglm(tokenizer, prompt, history, max_length):
    input_ids = []
    history = json.loads(history)
    
    if history:
        for item in history:
            input_ids.extend(tokenizer.build_single_message("user", "", item["prompt"]))
            input_ids.extend(tokenizer.build_single_message("assistant", "", item["response"]))
    
    sample_input_ids = tokenizer.build_single_message("user", "", prompt)
    sample_input_ids = input_ids + sample_input_ids

    # sample_input_ids = sample_input_ids[-max_length:] \
        # + [tokenizer.get_command("<|assistant|>")] \
        # + tokenizer.encode("\n", add_special_tokens=False)
    sample_input_ids = sample_input_ids[-max_length:] + tokenizer.encode("<|assistant|>\n", add_special_tokens=False)
    # sample = self.tokenizer.batch_encode_plus([sample_input_ids], return_tensors="pt", is_split_into_words=True)
    # return sample["input_ids"][0], sample["attention_mask"][0]
    return sample_input_ids


def _tokenize_fn_llama(tokenizer, prompt, history, max_length):
    conversation = []

    if history:
        for x in history:
            conversation.append({"role": "user", "content": x["prompt"]})
            conversation.append({"role": "assistant", "content": x["response"]})
    conversation.append({"role": "user", "content": prompt})
    sample_input_ids = tokenizer.apply_chat_template(conversation)
    sample_input_ids = sample_input_ids[-max_length:] + tokenizer.encode("<|im_start|>assistant\n")
    return sample_input_ids


def tokenize_fn_llama(tokenizer, texts, max_length, device):
    batch = [_tokenize_fn_llama(tokenizer, prompt=_prompt, history=_history, max_length=max_length) for _prompt, _history in zip(*texts)]

    batch_length = max([len(x) for x in batch])
    pad_token_id = tokenizer.pad_token_id
    max_length = max(max_length, batch_length)

    def batch_encode_plus(input_ids):
        sample_len = len(input_ids)
        if sample_len < max_length:
            attention_mask = [0] * (max_length - sample_len) + [1] * sample_len
            input_ids = [pad_token_id] * (max_length - sample_len) + input_ids
        else:
            attention_mask = [1] * max_length
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    batch = [batch_encode_plus(x) for x in batch]
    return {k: v.to(device) for k, v in batch.items()}


def tokenize_fn_chatglm(tokenizer, texts, max_length, device):
    batch = [_tokenize_fn_chatglm(tokenizer, prompt=_prompt, history=_history, max_length=max_length) for _prompt, _history in zip(*texts)]
    batch = tokenizer.batch_encode_plus(batch, return_tensors="pt", is_split_into_words=True, padding=True)
    return {k: v.to(device) for k, v in batch.items()}


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    info: Optional[dict] = None
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        if self.values is not None:
            self.values = self.values.to(device)
        if self.returns is not None:
            self.returns = self.returns.to(device)
        if self.advantages is not None:
            self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)
        if self.kl is not None:
            self.kl = self.kl.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        if self.values is not None:
            self.values = self.values.pin_memory()
        if self.returns is not None:
            self.returns = self.returns.pin_memory()
        if self.advantages is not None:
            self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        if self.kl is not None:
            self.kl = self.kl.pin_memory()
        return self


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: Optional[nn.Module],
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        reward_fn=None,
        tokenizer_reward=None
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.tokenizer_reward = tokenizer_reward
        self.current_model = "chatglm" if "glm" in strategy.args.pretrain else ""
        
    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        if "glm" in self.current_model:
            return tokenize_fn_chatglm(self.tokenizer, texts, max_length, device)
        if "qwen" or "llama" in self.current_model.lower():
            return tokenize_fn_llama(self.tokenizer, texts, max_length, device)

        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}
    
    # def _build_single_message(self, role, metadata, message):
    #     assert role in ["system", "user", "assistant", "observation"], role
    #     role_tokens = [self.tokenizer.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n", add_special_tokens=False)
    #     message_tokens = self.tokenizer.encode(message, add_special_tokens=False)
    #     tokens = role_tokens + message_tokens
    #     return tokens

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()

        self.initial_model.eval()
        self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        value = self.critic(sequences, action_mask, attention_mask)

        # rewards
        r = self.reward_model(sequences, attention_mask)

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
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.process_flag_tokens = get_process_flag_tokens(self.tokenizer)

    def to_train(self):
        self.actor.train()
        if self.critic:
            self.critic.train()
            
    def to_eval(self):
        self.actor.eval()
        if self.critic:
            self.critic.eval()
    
    def retokenize(self, input_ids, source_tokenizer, target_tokenzier):
        texts = source_tokenizer.batch_decode(input_ids.cpu(), skip_special_tokens=False)
        # print(texts[0])
        
        # batch_inputs = target_tokenzier(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.strategy.args.prompt_max_len + self.strategy.args.generate_max_len)

        max_len = 8192 + 1020
        
        # endtoken = target_tokenzier.encode("<|user|>", add_special_tokens=False)[0]
        endtoken = get_eos_token_id(target_tokenzier)
        batch_input_ids = [target_tokenzier.encode(e, add_special_tokens=False) for e in texts]
        
        for i, item in enumerate(batch_input_ids):
            if len(item) > max_len:
                batch_input_ids[i] = item[:max_len-1] + [endtoken]
                
        # input_ids = batch_inputs["input_ids"]
        # attention_mask = batch_inputs["attention_mask"]
        pad_token = target_tokenzier.pad_token_id

        batch = target_tokenzier.batch_encode_plus(batch_input_ids, return_tensors="pt", is_split_into_words=True, padding=True, add_special_tokens=False)

        output_ids = batch["input_ids"]
        shape = output_ids.shape
        output_ids = output_ids.view(-1)
        attention_mask = batch["attention_mask"].view(-1)
        attention_mask[output_ids == pad_token] = 0

        output_ids = output_ids.view(*shape)
        attention_mask = attention_mask.view(*shape)
        # print(f"output_ids: ", output_ids, "\nmax_value:", output_ids.max())
        assert int(output_ids.max()) < target_tokenzier.vocab_size, f"output_ids: {output_ids}, max_value: {output_ids.max()}"

        assert output_ids.shape[1] <= max_len, f"output_ids: {output_ids.shape}"
        # output_ids = output_ids[:, :max_len-1] + target_tokenzier.encode("<|user|>", add_special_tokens=False)

        return output_ids, attention_mask

    def process_reformat_and_tokenize(self, input_ids, attention_mask, action_mask):
        max_gen_len = self.strategy.args.generate_max_len
        device = input_ids.device
        # input_len = action_mask[0].argmax().item()
        input_len = input_ids.shape[1] - action_mask.shape[1]

        # if int(input_ids[0][input_len]) == self.tokenizer.convert_tokens_to_ids("<|assistant|>"):
            # input_len += 1
            # print(f"found <|assistant|> at prompt: {input_len}")
        chat_end_token = get_eos_token(self.tokenizer)
        # eos_token_id = self.tokenizer.convert_tokens_to_ids(chat_end_token)
        eos_token_id = get_eos_token_id(self.tokenizer)
        pad_token_id = self.tokenizer.pad_token_id
        
        actions = input_ids[:, input_len:]
        # assert actions.shape[1] <= max_gen_len, f"actions: {actions.shape}, max_gen_len: {max_gen_len}"

        # for item in actions:
        #     if len(item) == max_gen_len and item[-1] != self.tokenizer.pad_token_id:
        #         # item[-1] = eos_token_id
        #         assert item[-1] == eos_token_id, f"length: {len(item)}, item: {item}"
        # responses = self.tokenizer.decode(actions, skip_special_tokens=True)
        
        # if self.strategy.get_rank() == 0:
        # for resp in actions:
        #     assert eos_token_id in resp, f"eos_token_id: {eos_token_id}, resp_shape: {resp.shape}, resp: {resp.shape}"
        
        actions = [x[x != pad_token_id] for x in actions]
        num_tokens_per_action = [len(x) for x in actions]
        # responses = self.tokenizer.batch_decode(actions.tolist(), skip_special_tokens=True)
        responses = [self.tokenizer.decode(x.tolist(), skip_special_tokens=True) for x in actions]
        
        for x in responses:
            assert chat_end_token not in x, f"responses: {x}"
        responses = [x + chat_end_token for x in responses]
        
        # eos_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")
        # for resp in responses:
        
        process_flag = get_process_flag()[0]
        
        num_steps_per_response = []
        for i in range(len(responses)):
            # assert process_flag not in responses[i], f"responses: {responses[i]}, process_flag: {process_flag}"
            # if process_flag in responses[i]:
                # TODO: directly replace flag_token may not be correct
                # responses[i] = responses[i].replace(process_flag, "")
            responses[i], num_steps = reformat_response_into_steps(responses[i], use_nk=False, flag=process_flag, return_num_splits=True, join=True)

            num_steps_per_response.append(num_steps)
            assert process_flag in responses[i], f"resp={responses[i]}, flag_token={process_flag}"

        pad_token = self.tokenizer.pad_token_id
        responses = self.tokenizer(responses, add_special_tokens=False)
        num_tokens_per_action_retokenized = [len(x) for x in responses["input_ids"]]
        for nb, na, ns, resp in zip(num_tokens_per_action, num_tokens_per_action_retokenized, num_steps_per_response, responses):
            assert nb + ns == na, f"num_act_before: {nb}, num_act_after: {na}, num_steps: {ns}. resp={resp}"

        process_flag_token_id = self.tokenizer.encode(process_flag, add_special_tokens=False)[0]

        for i, item in enumerate(responses["input_ids"]):
            assert item[-1] == process_flag_token_id, f" --- item: {self.tokenizer.decode(item)}, process_token_ids: {process_flag_token_id} --- "
            
        # responses["input_ids"] = [x[:-1] + [eos_token_id] + [x[-1]] for x in responses["input_ids"]]
        # chat_end_token = self.tokenizer.convert_tokens_to_ids("<|user|>")
        # responses = [x + [chat_end_token] for x in responses]

        # responses = [self.tokenizer.encode(resp) for resp in responses]
        
        max_length = max([len(x) for x in responses["input_ids"]])
        reencode_input_ids = [x + [pad_token] * (max_length - len(x)) for x in responses["input_ids"]]
        reencode_action_mask = [x + [0] * (max_length - len(x)) for x in responses["attention_mask"]]
        reencode_input_ids = torch.tensor(reencode_input_ids).to(device)
        reencode_action_mask = torch.tensor(reencode_action_mask).to(device)
        
        input_ids = torch.cat([input_ids[:, :input_len], reencode_input_ids], dim=1)
        attention_mask = torch.cat([attention_mask[:, :input_len], reencode_action_mask], dim=1)
    
        # if self.strategy.get_rank() == 0:
            # print(f"********* reencodes sentences: ", self.tokenizer.decode(input_ids[i][input_ids[i] != pad_token].tolist(), skip_special_tokens=False))

        # action_mask = torch.cat([action_mask[:, :input_len], torch.ones_like(reencode_action_mask)], dim=1)
        action_mask = reencode_action_mask

        assert ((input_ids == get_process_flag_tokens(self.tokenizer)[0]).sum(1) > 0).all()

        return input_ids, attention_mask, action_mask
    
    def apply_process_rewards(self, actions, rewards, batch_size, is_ppo=False, normalize_multi_traces=False, original_num_actions=-1):
        assert actions.shape == rewards.shape, f"actions: {actions.shape}, rewards: {rewards.shape}"
        # input_len = action_mask.float()[0].argmax().item()
        # action_tokens = input_ids[:, input_len:]
        # rewards = rewards[:, input_len:]

        action_tokens = actions
        num_samples = actions.shape[0]
        action_tokens_reformatted = []

        # process_flag = torch.zeros_like(action_tokens)
        # for token in self.process_flag_tokens:
        # process_flag += (action_tokens == self.process_flag_tokens[0]).float()

        process_flag = (action_tokens == self.process_flag_tokens[0])
        # action_len = action_tokens.shape[-1]
        rewards = rewards * process_flag.float()

        assert (process_flag.sum(1) > 0).all(), f"process_flag: {process_flag.sum(1)}"
        
        if (rewards.abs().sum(1) == 0).any():
            print(f"rewards: {rewards.abs().sum(1)}, process_flag: {process_flag.sum(1)}")
            for item, action_item in zip(rewards, actions):
                if item.abs().sum() == 0:
                    print(f"************* reformatted_actions={self.tokenizer.decode(action_item[action_item != self.tokenizer.pad_token_id].tolist(), skip_special_tokens=False)}")
                    assert False

        assert (rewards.abs().sum(1) > 0).all(), f"rewards: {rewards.abs().sum(1)}, process_flag: {process_flag.sum(1)}"

        # print(f"num_reward_max: {max([x.nonzero()[-1] + 1 for x in rewards])}, reward_shape: {rewards.shape}, num_flags: {max([x.nonzero()[-1] + 1 for x in process_flag.float()])}, {(action_tokens != self.tokenizer.pad_token_id).sum(1)}")
        
        rewards_valid = []
        process_tokens_valid = []
        for reward_item, process_flag_item, action_item in zip(rewards, process_flag, actions):
            # reward_item[0:-1] = reward_item[1:].clone()
            if process_flag_item.dtype != torch.bool:
                process_flag_item = process_flag_item > 0

            reward_positions = torch.where(process_flag_item)[0]
            target_reward_positions = reward_positions - torch.arange(len(reward_positions), device=reward_positions.device) - 1
            
            _reward_item = reward_item[~process_flag_item]
            _reward_item[target_reward_positions] += reward_item[reward_positions]
            
            # reward_item = reward_item[1:][~process_flag_item[:-1]]
            rewards_valid.append(_reward_item)

            assert action_item[0] != self.tokenizer.pad_token_id, f"action_item: {self.tokenizer.decode(action_item)}"
            
            action_item = action_item[~process_flag_item]
            # print(f"************* reformatted_actions={self.tokenizer.decode(action_item[action_item != self.tokenizer.pad_token_id].tolist(), skip_special_tokens=False)}")
            

            # assert (reward_item != 0).float().nonzero().flatten()[-1] < original_num_actions, f"reward_item: {len(reward_item)}, original_num_actions: {original_num_actions}, process_flag_item: {(reward_item != 0).float().nonzero().flatten().max()}, {process_flag_item.sum()}, {(process_flag_item > 0).float().nonzero().flatten().max()}, action_tokens={action_item}, actions={self.tokenizer.decode(action_item[action_item != self.tokenizer.pad_token_id].tolist(), skip_special_tokens=False)}"

            _process_flag_item = (_reward_item != 0).float()
            process_tokens_valid.append(_process_flag_item)
            action_tokens_reformatted.append(action_item)

        rewards = torch.nn.utils.rnn.pad_sequence(rewards_valid, batch_first=True, padding_value=0)
        process_flag = torch.nn.utils.rnn.pad_sequence(process_tokens_valid, batch_first=True, padding_value=0)
        action_tokens_reformatted = torch.nn.utils.rnn.pad_sequence(action_tokens_reformatted, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        assert rewards.shape == action_tokens_reformatted.shape, f"rewards: {rewards.shape}, action_tokens_reformatted: {action_tokens_reformatted.shape}"

        assert ((process_flag > 0).sum(1) > 0).all(), f"process_flag: {process_flag.sum(1)}"
        assert ((rewards != 0).float().sum(1) > 0).all(), f"rewards: {rewards.sum(1)}"
        # max_length = max([(process_flag_item == 1).float().nonzero().view(-1)[-1] + 1 for x in rewards_valid])
        # print(f"max_len={max([(x == 1).float().nonzero().view(-1)[-1] + 1 for x in process_flag])}, padded_len={rewards.shape[1]}, original_max_len={process_flag.shape[1]}")            
        
        while (action_tokens_reformatted[:, -1] == self.tokenizer.pad_token_id).all():
            rewards = rewards[:, :-1]
            process_flag = process_flag[:, :-1]
            action_tokens_reformatted = action_tokens_reformatted[:, :-1]

        action_tokens_reformatted = action_tokens_reformatted.contiguous()
        rewards = rewards.contiguous()
        process_flag = process_flag.contiguous()
        
        # rewards = rewards[:, :max_length]
        # process_flag = process_flag[:, :max_length]
                    
        # process_flag = process_flag[:, 1:]        
        # valid_token_positions = (process_flag.flatten() == 0)

        if process_flag.sum(1).min() == 0:
            for i in range(len(process_flag)):
                if process_flag[i].sum() == 0:
                    print(f"************* reformatted_actions={self.tokenizer.decode(action_tokens_reformatted[i][action_tokens_reformatted[i] != self.tokenizer.pad_token_id].tolist(), skip_special_tokens=False)}")
                    
        assert (process_flag.sum(1) > 0).all(), f"num_flag={process_flag.sum(1)}"
        if normalize_multi_traces:
            rewards = rewards.reshape(batch_size, -1)
            _process_flag = process_flag.reshape(batch_size, -1)
            # assert (_process_flag.sum(1) > 0).all(), f"input_len: {input_len}, seq_len={input_ids.shape[1]}"
            
            reward_mean = (rewards.sum(dim=1) / _process_flag.sum(1)).view(-1, 1)
            # reward_std = rewards.std(dim=1).view(-1, 1)
            reward_std = (((rewards - reward_mean) ** 2) * _process_flag).sum(1) / (_process_flag.sum(1) + 1e-9)
            reward_std = reward_std.sqrt()
            reward_std = reward_std.view(-1, 1)
            
            normalized_rewards = (rewards - reward_mean) / (reward_std + 1e-7)
            normalized_rewards = normalized_rewards * _process_flag
            normalized_rewards = normalized_rewards.view(num_samples, -1)

            # normalized_rewards = normalized_rewards.view(-1, action_len)

            # normalized_rewards_valid = []
            # for reward_item, process_flag_item in zip(normalized_rewards, process_flag):
            #     reward_item[0:-1] = reward_item[1:].clone()
            #     reward_item = reward_item[process_flag_item[:-1]]
            #     normalized_rewards_valid.append(reward_item)

            # normalized_rewards = torch.utils.rnn.pad_sequence(normalized_rewards_valid, batch_first=True, padding_value=0)
            # assert normalized_rewards.shape == action_tokens.shape, f"normalized_rewards: {normalized_rewards.shape}, action_tokens: {action_tokens.shape}"

            # ---- failed implementation ----
            # normalized_rewards = normalized_rewards.view(-1)        
            # normalized_rewards[0:-1] = normalized_rewards[1:].clone()
            # # normalized_rewards = normalized_rewards[1:][valid_token_positions[:-1]]
            # normalized_rewards = normalized_rewards[valid_token_positions]
            # normalized_rewards = normalized_rewards.view(num_samples, -1)
    
            # print(f"---- reward_sum: {rewards.abs().sum(1)}")
            assert torch.isnan(normalized_rewards).sum() == 0, f"rewards: {rewards}, normalized_rewards: {normalized_rewards}"
        else:
            normalized_rewards = rewards.reshape(num_samples, -1)

        if is_ppo:
            # assert normalized_rewards.shape == process_flag.shape, f"normalized_rewards: {normalized_rewards.shape}, process_flag: {process_flag.shape}"
            # normalized_rewards = normalized_rewards * process_flag
            # prefix_pad = torch.zeros((input_ids.shape[0], input_len), device=input_ids.device)
            # normalized_rewards = torch.cat([prefix_pad, normalized_rewards], dim=1)
            assert normalized_rewards.shape == action_tokens_reformatted.shape, f"normalized_rewards: {normalized_rewards.shape}, action_tokens_reformatted: {action_tokens_reformatted.shape}"
            rewards = rewards.view(num_samples, -1)
            return normalized_rewards, rewards, action_tokens_reformatted
        
        target_rewards = torch.zeros_like(normalized_rewards).view(-1)
        normalized_rewards = normalized_rewards.view(-1)
        process_flag = process_flag.view(-1)
        flag_pos = torch.where(process_flag > 0)[0]
        flag_reward = normalized_rewards[flag_pos]

        for i in range(len(flag_pos)):
            if i == 0:
                target_rewards[:flag_pos[i]] = normalized_rewards[flag_reward[i]]
            else:
                target_rewards[flag_pos[i-1]:flag_pos[i]] = normalized_rewards[flag_reward[i]]

        # prefix = torch.zeros((input_ids.shape[0], input_len), device=input_ids.device)
        # prefix_pad = torch.zeros((input_ids.shape[0], input_len), device=input_ids.device)
        target_rewards = target_rewards.view(*rewards.shape)
        # target_rewards = torch.cat([prefix_pad, target_rewards], dim=1)
        return target_rewards, rewards, action_tokens_reformatted
        
    def sample_responses_prm(self, prompts: List[str], num_trace_per_sample: int = 1, **generate_kwargs):
        device = torch.cuda.current_device()

        _wait_time = 0
        _generate_time = 0
        _actor_time = 0
        _sequences = []
        _action_log_probs = []
        _value = []
        _attention_mask = []
        _action_mask = []
        _raw_r = []
        _base_action_log_probs = []
        _actions_for_rm = []
        _actions = []
        _inputs = []
        _attention_masks_actions = []
        _attention_masks_inputs = []
        # _process_rewards = []
        
        if isinstance(prompts, list) and len(prompts) == 2: 
            # [[prompt], [history]]
            micro_batch_size_roll_out = batch_size = len(prompts[0])
        else:
            micro_batch_size_roll_out = batch_size = len(prompts)

        forward_batch_size = getattr(self.strategy.args, "inference_batch_size", 4)
        
        start_overall = time.time()
        batch_first = True

        if isinstance(prompts, list) and len(prompts) == 2:
            # prompts = [prompts[0] * num_trace_per_sample, prompts[1] * num_trace_per_sample]
            prompts = [
                [x for x in prompts[0] for _ in range(num_trace_per_sample)],
                [x for x in prompts[1] for _ in range(num_trace_per_sample)]
            ]
        else:
            prompts = prompts * num_trace_per_sample

        generate_batch_size = getattr(self.strategy.args, "generation_batch_size", 16)
        generate_batch_size = min(generate_batch_size, micro_batch_size_roll_out * num_trace_per_sample)        

        assert generate_batch_size % forward_batch_size == 0, f"generate_batch_size: {generate_batch_size}, forward_batch_size: {forward_batch_size}"
         
        batch_size_multiplier = generate_batch_size // forward_batch_size
                
        for i in range(0, (micro_batch_size_roll_out * num_trace_per_sample), generate_batch_size):
            # start = time.time()
            
            # sequences, attention_mask, action_mask = (
            #     self._generate_local(prompts, **generate_kwargs)
            #     if self.vllm_engines is None
            #     else self._generate_vllm(prompts, **generate_kwargs)
            # )
            # assert batch_size == int(sequences.shape[0]), f"len(prompts): {len(prompts)}---{len(prompts[0])}, sequences.shape[0]: {sequences.shape[0]}, {type(prompts)}, {prompts}"
            # # print(f"sequences.shape: {sequences.shape}, num_prompts: {len(prompts)}")
            # ### re-tokenize
            
            if isinstance(prompts, list) and len(prompts) == 2:
                batch_prompts = [
                    prompts[0][i: i + generate_batch_size], 
                    prompts[1][i: i + generate_batch_size]
                ]
            else:
                batch_prompts = prompts[i: i + generate_batch_size]

            start = time.time()
            
            sequences, attention_mask, action_mask = (
                self._generate_local(batch_prompts, **generate_kwargs)
                if self.vllm_engines is None
                else self._generate_vllm(batch_prompts, **generate_kwargs)
            )
            
            # TODO: not aligned with read-generation distributions, just for debugging
            def reformat_sequences(sequences, attention_mask, action_mask):
                _num_actions = action_mask.shape[1]
                _input_len = sequences.shape[1] - _num_actions
                inputs = sequences[:, :_input_len]
                actions = sequences[:, _input_len:]
                generation_texts = self.tokenizer.batch_decode(actions.cpu(), skip_special_tokens=True)
                
                eos_token_id = get_eos_token_id(self.tokenizer)
                generation_texts = [reformat_response_into_steps(x, join=False) for x in generation_texts]
                actions_rfm = []
                for item in generation_texts:
                    seq = [self.tokenizer.encode(x, add_special_tokens=False) for x in item]
                    seq.append([eos_token_id])
                    seq = chain(*seq)
                    actions_rfm.append(torch.tensor(list(seq)).view(1, -1))
                actions_rfm = zero_pad_batch(actions_rfm, side="right", pad_token_id=self.tokenizer.pad_token_id)

                actions_rfm = actions_rfm.to(sequences.device)
                action_mask = (actions_rfm != self.tokenizer.pad_token_id).to(attention_mask.dtype)
                attention_mask = torch.cat([attention_mask[:, :_input_len], action_mask], dim=1)
                sequences = torch.cat([inputs, actions_rfm], dim=1)
                print(f"### increased seq-length: original: {_num_actions + _input_len}, rm_seq={sequences.shape}, actions: {actions_rfm.shape}, action_mask: {action_mask.shape}")
                return sequences, attention_mask, action_mask

            # TODO: not aligned with read-generation distributions, just for debugging
            sequences, attention_mask, action_mask = reformat_sequences(sequences, attention_mask, action_mask)
            
            sequences_for_rm, attention_mask_for_rm, action_mask_for_rm = self.process_reformat_and_tokenize(sequences, attention_mask, action_mask)
            # sequences = revert_special_tokens(self.tokenizer, sequences_for_rm)
            # print(f"### increased seq-length: original: {sequences.shape}, rm_sequence={sequences_for_rm.shape}, original_actions: {action_mask.shape}, rm_actions={action_mask_for_rm.shape}")
            
            _generate_time += time.time() - start

            num_actions = action_mask.size(1)
            num_rm_actions = action_mask_for_rm.shape[1]

            sequences_cpu, attention_mask_cpu, action_mask_cpu = (
                sequences.to("cpu").contiguous(),
                attention_mask.to("cpu").contiguous(),
                action_mask.to("cpu").contiguous(),
            )

            for i in range(batch_size_multiplier):
                micro_sequences = sequences[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_attention_mask = attention_mask[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_action_mask = action_mask[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_sequences_for_rm = sequences_for_rm[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_attention_mask_for_rm = attention_mask_for_rm[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_action_mask_for_rm = action_mask_for_rm[i * forward_batch_size: (i + 1) * forward_batch_size]

                # to cpu_device
                micro_sequences_cpu = sequences_cpu[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_attention_mask_cpu = attention_mask_cpu[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_action_mask_cpu = action_mask_cpu[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_sequences_for_rm_cpu = micro_sequences_for_rm.cpu()
                micro_attention_mask_for_rm_cpu = micro_attention_mask_for_rm.cpu()
                micro_action_mask_for_rm_cpu = micro_action_mask_for_rm.cpu()
                # micro_num_actions = micro_action_mask_cpu.size(1)
                
                base_action_log_probs_ref = self.initial_model.forward.remote(micro_sequences_cpu, num_actions, micro_attention_mask_cpu)

                # values
                if self.critic:
                    # value_ref = self.critic.forward.remote(micro_sequences_cpu, micro_action_mask_cpu, micro_attention_mask_cpu)
                    value_ref = self.critic.forward.remote(micro_sequences_cpu, micro_action_mask_cpu, micro_attention_mask_cpu)
                else:
                    value_ref = None
        
                # rewards
                r_refs = []
                for rm in self.reward_model:                    
                    micro_sequences_for_rm_cpu, micro_attention_mask_cpu_rm = micro_sequences_for_rm_cpu, micro_attention_mask_for_rm_cpu

                    r_refs.append(rm.forward.remote(micro_sequences_for_rm_cpu, micro_attention_mask_cpu_rm, False))

                # log probs
                start = time.time()
                action_log_probs = self.actor(micro_sequences, num_actions, micro_attention_mask)
                actor_time = time.time() - start

                # wait initial/critic/reward model done
                start = time.time()
                
                if self.critic:
                    ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
                    base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
                    base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
                else:
                    ref_values = ray.get([base_action_log_probs_ref] + r_refs)
                    base_action_log_probs, rewards = ref_values[0], ref_values[1:]
                    base_action_log_probs = base_action_log_probs.to(device)
                    value = None
                    
                wait_time = time.time() - start

                rewards = [r.to(device) for r in rewards]
                r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]
                
                _base_action_log_probs.append(base_action_log_probs)

                _attention_mask.append(micro_attention_mask)
                _action_mask.append(micro_action_mask)
                _value.append(value)
                _sequences.append(micro_sequences)
                _action_log_probs.append(action_log_probs)
                _actions.append(micro_sequences[:, -num_actions:])
                _inputs.append(micro_sequences[:, :-num_actions])
                _attention_masks_actions.append(micro_attention_mask[:, -num_actions:])
                _attention_masks_inputs.append(micro_attention_mask[:, :-num_actions])
                # _kl.append(kl)
                _raw_r.append(r)
                # _reward.append(reward)

                _wait_time += wait_time
                _actor_time += actor_time

        action_mask = zero_pad_batch(_action_mask, side="right")
        # advantage = zero_pad_batch(_advantage, side="right")
        # returns = zero_pad_batch(_returns, side="right")
        # kl = zero_pad_batch(_kl, side="right")
        rollout_time = time.time() - start_overall
        actions = zero_pad_batch(_actions, side="right", pad_token_id=self.tokenizer.pad_token_id)
        inputs = zero_pad_batch(_inputs, side="left", pad_token_id=self.tokenizer.pad_token_id)
        attention_mask_action = zero_pad_batch(_attention_masks_actions, side="right")
        attention_mask_input = zero_pad_batch(_attention_masks_inputs, side="left")

        # attention_mask = zero_pad_batch(_attention_mask, side="right")
        # sequences = zero_pad_batch(_sequences, side="right", pad_token_id=self.tokenizer.pad_token_id)
        sequences = torch.cat([inputs, actions], dim=1)
        attention_mask = torch.cat([attention_mask_input, attention_mask_action], dim=1)

        action_log_probs = zero_pad_batch(_action_log_probs, side="right")
        base_action_log_probs = zero_pad_batch(_base_action_log_probs, side="right")
        r = torch.cat(_raw_r, dim=0)
        
        if self.critic:
            value = zero_pad_batch(_value, side="right")
        else:
            value = None

        _raw_reward = r
        if num_trace_per_sample > 1 and self.strategy.args.normalize_reward_from_multi_traces:
            r = normalize_reward_from_multi_traces(
                r, 
                batch_size, 
                num_trace_per_sample,
                min_threshold=getattr(self.strategy.args, "min_reward_gap", 0.0),
                batch_first=batch_first
            )

        # else:
            # action_mask = actions_restored != self.tokenizer.pad_token_id
            # sequence_prompts = [s[:-m.shape[0]] for s, m in zip(_sequences, action_mask)]
            # sequence_prompts = zero_pad_batch(sequence_prompts, side="left", pad_token_id=self.tokenizer.pad_token_id)
            # sequences = torch.cat([sequence_prompts, actions_restored], dim=1)
            # attention_mask = (sequences != self.tokenizer.pad_token_id).float()
            # atention_mask = 
        
        return {
            "action_mask": action_mask,
            "attention_mask": attention_mask,
            "sequences": sequences,
            "action_log_probs": action_log_probs,
            "base_action_log_probs": base_action_log_probs,
            "value": value,
            "reward": process_rewards,
            "raw_reward": unormalized_process_reward,
            "generate_time": _generate_time,
            "actor_time": _actor_time,
            "wait_time": _wait_time,
            "rollout_time": rollout_time,
        }

    def sample_responses(self, prompts: List[str], num_trace_per_sample: int = 1, **generate_kwargs):
        if self.strategy.args.process_supervision:
            return self.sample_responses_prm(prompts, num_trace_per_sample, **generate_kwargs)
            
        device = torch.cuda.current_device()

        _wait_time = 0
        _generate_time = 0
        _actor_time = 0
        _sequences = []
        _action_log_probs = []
        _value = []
        _attention_mask = []
        _action_mask = []
        _raw_r = []
        _base_action_log_probs = []
        _actions = []
        _inputs = []
        _attention_masks_actions = []
        _attention_masks_inputs = []
        
        if isinstance(prompts, list) and len(prompts) == 2: 
            # [[prompt], [history]]
            micro_batch_size_roll_out = batch_size = len(prompts[0])
        else:
            micro_batch_size_roll_out = batch_size = len(prompts)

        # batch_size_multiplier = 4
        # batch_size_multiplier = self.strategy.args.roll_out_batch_size_multiplier

        # generate_batch_size = micro_batch_size_roll_out
        # forward_batch_size = micro_batch_size_roll_out // batch_size_multiplier
    
        forward_batch_size = getattr(self.strategy.args, "inference_batch_size", 4)
        
        start_overall = time.time()
        
        # ---- standard implementation -----
        # for _ in range(num_trace_per_sample):
        #     start = time.time()
            
        #     sequences, attention_mask, action_mask = (
        #         self._generate_local(prompts, **generate_kwargs)
        #         if self.vllm_engines is None
        #         else self._generate_vllm(prompts, **generate_kwargs)
        #     )
        #     assert batch_size == int(sequences.shape[0]), f"len(prompts): {len(prompts)}---{len(prompts[0])}, sequences.shape[0]: {sequences.shape[0]}, {type(prompts)}, {prompts}"
        #     # print(f"sequences.shape: {sequences.shape}, num_prompts: {len(prompts)}")
        # ------ standard implementation ------
        
        # ------ efficient implementation -----
        # prompts = prompts * num_trace_per_sample

        batch_first = False

        if batch_first:
            if isinstance(prompts, list) and len(prompts) == 2:
                prompts = [
                    [x for x in prompts[0] for _ in range(num_trace_per_sample)],
                    [x for x in prompts[1] for _ in range(num_trace_per_sample)]
                ]
            else:
                prompts = prompts * num_trace_per_sample
        else:
            if isinstance(prompts, list) and len(prompts) == 2:
                prompts = [prompts[0] * num_trace_per_sample, prompts[1] * num_trace_per_sample]
            else:
                prompts = prompts * num_trace_per_sample
            
        generate_batch_size = getattr(self.strategy.args, "generation_batch_size", 16)
        generate_batch_size = min(generate_batch_size, micro_batch_size_roll_out * num_trace_per_sample)
        # assert generate_batch_size % batch_size == 0, f"generate_batch_size: {generate_batch_size}, batch_size: {batch_size}"
        assert generate_batch_size % forward_batch_size == 0, f"generate_batch_size: {generate_batch_size}, forward_batch_size: {forward_batch_size}"
         
        batch_size_multiplier = generate_batch_size // forward_batch_size
        
        # for i in range(0, micro_batch_size_roll_out, generate_batch_size):
        for i in range(0, (micro_batch_size_roll_out * num_trace_per_sample), generate_batch_size):
        # for _ in range():
            if isinstance(prompts, list) and len(prompts) == 2:
                batch_prompts = [
                    prompts[0][i: i + generate_batch_size], 
                    prompts[1][i: i + generate_batch_size]
                ]
            else:
                batch_prompts = prompts[i: i + generate_batch_size]

            start = time.time()
            
            sequences, attention_mask, action_mask = (
                self._generate_local(batch_prompts, **generate_kwargs)
                if self.vllm_engines is None
                else self._generate_vllm(batch_prompts, **generate_kwargs)
            )
        # ------ efficient implementation -----

            _generate_time += time.time() - start

            num_actions = action_mask.size(1)
            sequences_cpu, attention_mask_cpu, action_mask_cpu = (
                sequences.to("cpu"),
                attention_mask.to("cpu"),
                action_mask.to("cpu"),
            )

            for i in range(batch_size_multiplier):
                micro_sequences = sequences[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_attention_mask = attention_mask[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_action_mask = action_mask[i * forward_batch_size: (i + 1) * forward_batch_size]
                
                micro_sequences_cpu = sequences_cpu[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_attention_mask_cpu = attention_mask_cpu[i * forward_batch_size: (i + 1) * forward_batch_size]
                micro_action_mask_cpu = action_mask_cpu[i * forward_batch_size: (i + 1) * forward_batch_size]
                # micro_num_actions = micro_action_mask_cpu.size(1)
                
                base_action_log_probs_ref = self.initial_model.forward.remote(micro_sequences_cpu, num_actions, micro_attention_mask_cpu)

                # values
                if self.critic:
                    value_ref = self.critic.forward.remote(micro_sequences_cpu, micro_action_mask_cpu, micro_attention_mask_cpu)
                else:
                    value_ref = None

                # rewards
                r_refs = []
                for rm in self.reward_model:
                    # if self.critic and self.strategy.args.critic_pretrain and (self.strategy.args.reward_pretrain != self.strategy.args.critic_pretrain):
                    #     assert False
                    #     micro_sequences_cpu_rm, micro_attention_mask_cpu_rm = self.retokenize(micro_sequences_cpu, self.tokenizer, self.tokenizer_reward)
                    #     assert micro_sequences_cpu_rm.max() < self.tokenizer_reward.vocab_size, f"micro_sequences_cpu_rm: {micro_sequences_cpu_rm}"
                    #     assert micro_sequences_cpu_rm.min() >= 0, f"micro_sequences_cpu_rm: {micro_sequences_cpu_rm}"
                    #     # print(f"micro_sequences_cpu_rm: {micro_sequences_cpu_rm.shape}")
                    #     assert micro_sequences_cpu_rm.shape[1] < 9216, f"micro_sequences_cpu_rm: {micro_sequences_cpu_rm.shape}"
                        
                    # else:
                        # print(self.strategy.args.process_supervision, self.strategy.args.critic_pretrain, self.strategy.args.reward_pretrain)
                        # assert False, f"not implemented yet"

                    micro_sequences_cpu_rm, micro_attention_mask_cpu_rm = micro_sequences_cpu, micro_attention_mask_cpu

                    r_refs.append(rm.forward.remote(micro_sequences_cpu_rm, micro_attention_mask_cpu_rm, False))

                # log probs
                start = time.time()
                action_log_probs = self.actor(micro_sequences, num_actions, micro_attention_mask)
                actor_time = time.time() - start

                # wait initial/critic/reward model done
                start = time.time()
                
                if self.critic:
                    ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
                    base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
                    base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
                else:
                    ref_values = ray.get([base_action_log_probs_ref] + r_refs)
                    base_action_log_probs, rewards = ref_values[0], ref_values[1:]
                    base_action_log_probs = base_action_log_probs.to(device)
                    value = None
                    
                wait_time = time.time() - start

                rewards = [r.to(device) for r in rewards]
                r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]
                
                _base_action_log_probs.append(base_action_log_probs)

                _attention_mask.append(micro_attention_mask)
                _action_mask.append(micro_action_mask)
                _value.append(value)
                _sequences.append(micro_sequences)
                _action_log_probs.append(action_log_probs)
                _actions.append(micro_sequences[:, -num_actions:])
                _inputs.append(micro_sequences[:, :-num_actions])
                _attention_masks_actions.append(micro_attention_mask[:, -num_actions:])
                _attention_masks_inputs.append(micro_attention_mask[:, :-num_actions])
                # _kl.append(kl)
                _raw_r.append(r)
                # _reward.append(reward)

                _wait_time += wait_time
                _actor_time += actor_time
                
        action_mask = zero_pad_batch(_action_mask, side="right")
        # advantage = zero_pad_batch(_advantage, side="right")
        # returns = zero_pad_batch(_returns, side="right")
        # kl = zero_pad_batch(_kl, side="right")
        rollout_time = time.time() - start_overall
        actions = zero_pad_batch(_actions, side="right", pad_token_id=self.tokenizer.pad_token_id)
        inputs = zero_pad_batch(_inputs, side="left", pad_token_id=self.tokenizer.pad_token_id)
        attention_mask_action = zero_pad_batch(_attention_masks_actions, side="right")
        attention_mask_input = zero_pad_batch(_attention_masks_inputs, side="left")

        # attention_mask = zero_pad_batch(_attention_mask, side="right")
        # sequences = zero_pad_batch(_sequences, side="right", pad_token_id=self.tokenizer.pad_token_id)
        sequences = torch.cat([inputs, actions], dim=1)
        attention_mask = torch.cat([attention_mask_input, attention_mask_action], dim=1)

        action_log_probs = zero_pad_batch(_action_log_probs, side="right")
        base_action_log_probs = zero_pad_batch(_base_action_log_probs, side="right")
        r = torch.cat(_raw_r, dim=0)
        
        if self.critic:
            value = zero_pad_batch(_value, side="right")
        else:
            value = None

        _raw_reward = r
        if num_trace_per_sample > 1 and self.strategy.args.normalize_reward_from_multi_traces:
            r = normalize_reward_from_multi_traces(
                r, 
                batch_size, 
                num_trace_per_sample,
                min_threshold=getattr(self.strategy.args, "min_reward_gap", 0.0),
                batch_first=batch_first
            )

        return {
            "action_mask": action_mask,
            "attention_mask": attention_mask,
            "sequences": sequences,
            "action_log_probs": action_log_probs,
            "base_action_log_probs": base_action_log_probs,
            "value": value,
            "reward": r,
            "raw_reward": _raw_reward,
            "generate_time": _generate_time,
            "actor_time": _actor_time,
            "wait_time": _wait_time,
            "rollout_time": rollout_time,
        }
        
    def log_perf(self, info, experiences, num_trace_per_sample):
        device = torch.cuda.current_device()
        # batch_size = 1 if isinstance(prompts, str) else len(prompts)
        # batch_size = len()
        batch_size = len(experiences["sequences"])
        # batch_size = batch_size * num_trace_per_sample
        info["generate_time"] = torch.full((batch_size,), experiences["generate_time"], device=device)
        info["actor_time"] = torch.full((batch_size,), experiences["actor_time"], device=device)
        info["wait_time"] = torch.full((batch_size,), experiences["wait_time"], device=device)
        info["rollout_time"] = torch.full((batch_size,), experiences["rollout_time"], device=device)
        return info

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
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
        
        start_overall = time.time()
        
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
            
        overall_time = time.time() - start_overall

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
            info["rollout_time"] = torch.full((batch_size,), overall_time, device=device)
            
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

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        if "glm" in self.current_model:
            eos_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")
            eos_token_set = (self.tokenizer.convert_tokens_to_ids("<|user|>"), self.tokenizer.convert_tokens_to_ids("<|observation|>"), self.tokenizer.eos_token_id)
        else:
            # assert False, "Not supported model except for ChatGLM."
            eos_token_id = self.tokenizer.eos_token_id
            eos_token_set = (self.tokenizer.eos_token_id)

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 16),
            # stop_token_ids=[self.tokenizer.convert_tokens_to_ids("<|user|>"), self.tokenizer.convert_tokens_to_ids("<|observation|>")],
            stop_token_ids=list(eos_token_set),
            min_tokens=kwargs.get("min_new_tokens ", 1),
            # stop=["<|user|>", "<|observation|>"],
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
        # assert self.tokenizer.padding_side == "left", f"tokenizer padding_size should be left"
        pad_indices = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.int).argmax(dim=-1)
        prompt_token_ids = []
        for i, pad_index in enumerate(pad_indices.numpy()):
            prompt_token_ids.append(input_ids[i][pad_index:].tolist())
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|

        max_input_len, max_output_len = 0, 0
        for output in outputs:
            # TODO: how to force vLLM generate at least one token?
            output_token_ids = output.outputs[0].token_ids
            if output_token_ids[0] == eos_token_id:
                logger.warning(f"Only EOS output for prompt: {output.prompt}")
                output.outputs[0].token_ids = [self.tokenizer.unk_token_id, eos_token_id]
            # elif output_token_ids[-1] != eos_token_id:
            #     # TODO: may be a bug
            #     output.outputs[0].token_ids[-1] = eos_token_id

            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output_token_ids))

        pad_token_id = self.tokenizer.pad_token_id
        # pad_token_id = self.tokenizer.pad_token_idp
        # eos_token_id = 
        sequences = []
        for output in outputs:
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + output.prompt_token_ids

            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = output.outputs[0].token_ids + [pad_token_id] * (max_output_len - output_len)
            if int(output_ids[output_len - 1]) not in eos_token_set:
                # assert output_len == max_output_len, f"output_len: {output_len}, max_output_len: {max_output_len}, output_ids: {output_ids[output_len-5:output_len+1]}"
                # output_ids[-1] = eos_token_id
                output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id
                
            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None

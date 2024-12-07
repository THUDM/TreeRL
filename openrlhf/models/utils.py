from typing import Optional, Tuple, Union, List

# import bitsandbytes as bnb
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    log_ratio = log_ratio.clamp(min=-0.5, max=0.5)
    
    return log_ratio * action_mask


def compute_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
):
    kl = (log_probs_base - log_probs).exp() - (log_probs_base - log_probs) - 1
    return kl * action_mask


def compute_reward_naive(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    clip_reward_range: float = 5,
    kl_as_reward: bool = True,
    process_reward: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kl_coef <= 0.0:
        kl_coef = 0.0
        
    clip_reward_range = 1.5

    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)

    # assert kl.shape == r.shape, f"kl shape: {kl.shape} != r shape: {r.shape}"

    print(f"kl reward shape: {kl.shape}, kl_coef: {kl_coef}, r shape: {r.shape},log_probs shape: {log_probs.shape}, log_probs_base shape: {log_probs_base.shape}, action_mask shape: {action_mask.shape}")
    # with open("/workspace/lurui/openrlhf-glm/logs/outputs/kl_reward.txt", "a") as f:
    #     f.write(str(kl) + str(r)+ "\n")

    # kl = compute_kl(log_probs, log_probs_base, action_mask=action_mask)

    kl_reward = -kl_coef * kl

    # TODO: ---- RESTORE THIS LINE ----
    # r = r.clamp(min=-clip_reward_range, max=clip_reward_range)

    # The following code is equivalent to:
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break

    # eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)

    # last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))
    if process_reward:
        reward = r
    else:
        reward = r.view(-1, 1)
    if kl_as_reward:
        reward = reward + kl_reward

    reward = (reward * action_mask).float()
        
    return reward, kl


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    clip_reward_range: float = 5,
    process_reward: bool = False,
    value: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    # kl = compute_kl(log_probs, log_probs_base, action_mask=action_mask)

    kl_reward = -kl_coef * kl

    r = r.clamp(min=-clip_reward_range, max=clip_reward_range)

    # The following code is equivalent to:
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    
    if process_reward:
        # !bug
        # RuntimeError: The size of tensor a (1709) must match the size of tensor b (1180) at non-singleton dimension 1
        # print(f"reward: {r.shape}, kl: {kl_reward.shape}, logp: {log_probs.shape}")
        reward = r + kl_reward
    else:
        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        # value 中 eos_indices 位置的值组成的 tensor
        if value is not None:
            value_position = value.gather(1, eos_indices)
            print(f"eos_indices: {eos_indices.shape}, r: {r.shape}, kl: {kl_reward.shape}, action_mask: {action_mask.shape},value position:{value_position}")
        if r.shape == eos_indices.shape:
            last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.to(kl.dtype))
        else:
            last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

        reward = last_reward + kl_reward    
    
    # reward = reward * action_mask.float()
    
    return reward, kl

# def compute_reward_sentence_advantage(
#     r: Union[torch.Tensor, float],
#     kl_coef: float,
#     log_probs: torch.Tensor,
#     log_probs_base: torch.Tensor,
#     action_mask: Optional[torch.Tensor] = None,
#     clip_reward_range: float = 5,
#     process_reward: bool = False,
#     value: Optional[torch.Tensor] = None,
#     seq_path_lens: Optional[List] = None

# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     if kl_coef <= 0.0:
#         kl_coef = 0.0

#     kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
#     # kl = compute_kl(log_probs, log_probs_base, action_mask=action_mask)

#     kl_reward = -kl_coef * kl

#     r = r.clamp(min=-clip_reward_range, max=clip_reward_range)

#     for 
    
#     if process_reward:
#         # !bug
#         # RuntimeError: The size of tensor a (1709) must match the size of tensor b (1180) at non-singleton dimension 1
#         # print(f"reward: {r.shape}, kl: {kl_reward.shape}, logp: {log_probs.shape}")
#         reward = r + kl_reward
#     else:
#         eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
#         # value 中 eos_indices 位置的值组成的 tensor
#         if value is not None:
#             value_position = value.gather(1, eos_indices)
#             print(f"eos_indices: {eos_indices.shape}, r: {r.shape}, kl: {kl_reward.shape}, action_mask: {action_mask.shape},value position:{value_position}")
#         if r.shape == eos_indices.shape:
#             last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.to(kl.dtype))
#         else:
#             last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

#         reward = last_reward + kl_reward    
    
#     # reward = reward * action_mask.float()
    
#     return reward, kl


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        return (tensor * mask).sum(axis=dim) / (mask.sum(axis=dim) + 1e-7)
    else:
        return (tensor * mask).sum() / (mask.sum()+ 1e-7)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


# def find_all_linear_names(model, load_in_4bit=False):
#     cls = bnb.nn.Linear4bit if load_in_4bit else nn.Linear
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, cls):
#             names = name.split(".")
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])

#     if "lm_head" in lora_module_names:  # needed for 16-bit
#         lora_module_names.remove("lm_head")
#     return list(lora_module_names)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        # deepseed.zero.init hooks torch.arange to run it on the GPU
        hooked_arange = torch.arange
        torch.arange = deepspeed.runtime.zero.partition_parameters._orig_torch_arange

        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

        self.inv_freq = self.inv_freq.to("cuda")
        self._cos_cached = self._cos_cached.to("cuda")
        self._sin_cached = self._sin_cached.to("cuda")
        torch.arange = hooked_arange

    @property
    def sin_cached(self):
        return self._sin_cached

    @property
    def cos_cached(self):
        return self._cos_cached

    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


# Patch for LLaMA RoPE embedding
# https://github.com/microsoft/DeepSpeed/issues/4932
def replace_rope_embedding():
    from transformers.models.llama import modeling_llama

    modeling_llama.LlamaRotaryEmbedding = LlamaRotaryEmbedding
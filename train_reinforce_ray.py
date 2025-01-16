import argparse
from datetime import datetime
from typing import List

import deepspeed

import ray
import torch
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import (
    ActorModelRayActorReinforce,
    ReinforceRayActorGroup,
    ReferenceModelRayActor,
    RewardModelRayActor,
    create_vllm_engines,
)
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


# NOTE: reward function for multiple reward models, replace this with your own function!
def reward_fn(rewards: List[torch.Tensor]):
    return torch.stack(rewards).sum(dim=0)


def multi_reward_fn(rewards: List[torch.Tensor]):
    return rewards


def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

    assert (
        actor_world_size & (actor_world_size - 1)
    ) == 0, f"actor_world_size must be power of 2, got {actor_world_size}"
    assert args.zero_stage != 3 or args.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"


def train(args):
    _validate_args(args)

    # configure strategy
    strategy = get_strategy(args)

    # if colocated, create placement group for actor and critic model explicitly.
    pg = None
    # if args.colocate_actor_critic:
    #     assert (
    #         args.actor_num_nodes == args.critic_num_nodes
    #         and args.actor_num_gpus_per_node == args.critic_num_gpus_per_node
    #     ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and critic model."

    #     bundles = [
    #         {"GPU": args.actor_num_gpus_per_node, "CPU": args.actor_num_gpus_per_node}
    #         for _ in range(args.actor_num_nodes)
    #     ]
    #     pg = placement_group(bundles, strategy="STRICT_SPREAD")
    #     ray.get(pg.ready())

    # NOTE(wuxibin): Why don't we allocate 0.5 gpu for each actor when colocate models?
    # Say we have 1 node with 4 GPUs, and num_gpus_per_node for each model is 4.
    # If we allocate 0.5 gpu for both actor and critic model, then gpu allocation is
    #   |actor|actor|actor|actor|critic|critic|critic|critic|
    #   |GPU0 |GPU0 |GPU1 |GPU1 | GPU2 | GPU2 | GPU3 | GPU3 |
    #
    # So 0.75/0.25 gpu is a tricky to let Ray spread all models evenly on all gpus.
    #   |actor|critic|actor|critic|actor|critic|actor|critic|
    #   |GPU0 | GPU0 |GPU1 | GPU1 |GPU2 | GPU2 |GPU3 | GPU3 |

    actor_model = ReinforceRayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        ActorModelRayActorReinforce,
        pg=pg,
        num_gpus_per_actor=1,
    )

    # if colocated, create placement group for reference and reward model explicitly.
    pg = None
    if args.colocate_ref_reward:
        assert (
            args.ref_num_nodes == args.reward_num_nodes and args.ref_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate reference and reward model."

        bundles = [
            {"GPU": args.ref_num_gpus_per_node, "CPU": args.ref_num_gpus_per_node} for _ in range(args.ref_num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())

    ref_model = ReinforceRayActorGroup(
        args.ref_num_nodes,
        args.ref_num_gpus_per_node,
        ReferenceModelRayActor,
        pg=pg,
        num_gpus_per_actor=1,
    )

    # multiple reward models
    if not args.remote_rm_url:
        reward_pretrains = args.reward_pretrain.split(",")
        reward_models = []
        for _ in reward_pretrains:
            reward_models.append(
                ReinforceRayActorGroup(
                    args.reward_num_nodes,
                    args.reward_num_gpus_per_node,
                    RewardModelRayActor,
                    pg=pg,
                    num_gpus_per_actor=1,
                )
            )
    else:
        reward_models = None

    # init reference/reward/actor model
    refs = []
    refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))
    if not args.remote_rm_url:
        for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
            refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))

    # init vLLM engine for text generation
    vllm_engines = None
    if args.vllm_num_engines is not None:
        vllm_engines = create_vllm_engines(
            args.vllm_num_engines, args.vllm_tensor_parallel_size, args.pretrain, args.seed,
        )

    # TODO: use first reward model as critic model
    # max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
    # refs.extend(critic_model.async_init_model_from_pretrained(strategy, reward_pretrains[0], max_steps))
    ray.get(refs)

    # train actor and critic mdoel
    refs = actor_model.async_fit_actor_model(
        ref_model, 
        reward_models, 
        args.remote_rm_url,
        reward_fn=reward_fn, 
        vllm_engines=vllm_engines
    )
    ray.get(refs)

    # save model
    ray.get(actor_model.async_save_actor_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=1, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=1, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_ref_reward",
        action="store_true",
        default=False,
        help="whether to colocate reference and reward model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=1, help="number of gpus per node for actor")

    # optional vLLM for text generation
    parser.add_argument("--vllm_num_engines", type=int, default=None, help="number of vLLM Engines")
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )

    parser.add_argument("--prompt_data", type=str, default=None, nargs="*")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default=None,
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_data", type=str, default=None)
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--load_ckpt", action="store_true", default=False)
    
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--generate_max_len", type=int, default=1024)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--ptx_coef", type=float, default=0.05)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--value_clip", type=float, default=0.2)
    parser.add_argument("--lambd", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_nodes", type=int, default=256)
    parser.add_argument("--max_node_per_depth", type=float, default=18)
    parser.add_argument("--max_time_use", type=float, default=360)
    parser.add_argument("--step_level_norm", action="store_true", default=False)
    parser.add_argument("--random_pick", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--min_actor_learning_rate_lr", type=float, default=0.1)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.02)
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=list, default=None)
    parser.add_argument("--input_template", type=str, default="Human: {}\nAssistant: ")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")

    parser.add_argument("--bos_token", type=str, default=None)
    parser.add_argument("--eos_token", type=str, default=None)
    parser.add_argument("--pad_token", type=str, default=None)
    parser.add_argument("--unk_token", type=str, default=None)

    # custom dataset key name
    parser.add_argument("--input_key", type=str, default=None)

    # evaluation
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    parser.add_argument("--use_mpi_init", action="store_true", default=False)
    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    # reward normalization
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--normalize_reward_from_multi_traces", action="store_true", default=False)
    parser.add_argument("--normalize_advantage", action="store_true", default=False)

    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_trace_per_sample", type=int, default=1)
    parser.add_argument("--task_type", type=str, default="130b_reinforce")
    parser.add_argument("--max_min_reward_samples", action="store_true", default=False)
    parser.add_argument("--save_ckpt", action="store_true", default=False)
    parser.add_argument("--ema_beta", type=float, default=0.992)
    parser.add_argument("--process_supervision", action="store_true", default=False)
    parser.add_argument("--activation_offload", action="store_true", default=False)
    parser.add_argument("--generation_batch_size", type=int, default=16)
    parser.add_argument("--inference_batch_size", type=int, default=4)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--min_reward_gap", type=float, default=0.0)
    parser.add_argument("--remote_rm_url", type=str, nargs="+", default=None)
    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--source_key", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--normalize_reward_from_multi_traces_with_rloo", action="store_true", default=False)
    parser.add_argument("--normalize_reward_mean_only", action="store_true", default=False)
    parser.add_argument("--mask_repeated_samples", action="store_true", default=False)
    parser.add_argument("--use_rule_based_reward", action="store_true", default=False)
    parser.add_argument("--mask_pass_confident_samples", action="store_true", default=False)
    parser.add_argument("--use_random_top_k_logits_sampling", action="store_true", default=False)
    parser.add_argument("--use_general_reward_for_stem", action="store_true", default=False)
    parser.add_argument("--random_temperature", action="store_true", default=False)
    parser.add_argument("--use_mcts", action="store_true", default=False)
    parser.add_argument("--use_vinevalue", action="store_true", default=False)
    parser.add_argument("--use_orm_reward", action="store_true", default=False)
    parser.add_argument("--select_correct_leaf", action="store_true", default=False)
    parser.add_argument("--use_chain_reward", action="store_true", default=False)
    parser.add_argument("--use_state_value_reward", action="store_true", default=False)
    parser.add_argument("--use_value_only", action="store_true", default=False)
    parser.add_argument("--use_pure_RM", action="store_true", default=False)
    parser.add_argument("--first_token_temperature", type=float, default=0)
    parser.add_argument("--use_pure_binary", action="store_true", default=False)
    parser.add_argument("--use_entropy_tree", action="store_true", default=False)
    parser.add_argument("--average_one_generation", action="store_true", default=False)
    parser.add_argument("--advantage_mix_allancestor", action="store_true", default=False)
    parser.add_argument("--use_weighted_value", action="store_true", default=False)
    parser.add_argument("--correct_bonus_ratio", type=float, default=1)
    parser.add_argument("--balance_ratio", type=float, default=0)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--a_coeff", type=float, default=0.5)
    parser.add_argument("--b_mean", type=float, default=2.898)
    
    parser.add_argument("--binary_judge_url", type=str, default=None)
    parser.add_argument("--reward_model_url", type=str, default=None)
    parser.add_argument("--extractor_url", type=str, default=None)
    
    parser.add_argument("--wandb_id", type=str, default=None)
 
    args = parser.parse_args()
    train(args)

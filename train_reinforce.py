import argparse
import itertools
import math
import os
from copy import deepcopy
from datetime import datetime

import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.trainer import PPOTrainer, ReinforceTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer, get_cosine_schedule_with_warmup


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    if not args.remote_rm_url:
        # reward_pretrains = args.reward_pretrain.split(",")
        reward_models = []
        reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_train_config(is_actor=False),
        )
        reward_models.append(reward_model)
    else:
        reward_models = None


    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy)
    # get_tokenizer(args.reward_pretrain, critic, "left", strategy)
    if reward_models is not None:
        get_tokenizer(args.reward_pretrain, reward_models[0], "left", strategy)

    strategy.print(actor)

    # load weights for reference actor
    initial_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=False),
    )
    get_tokenizer(args.pretrain, initial_model, "left", strategy)

    # strategy.print("reward normalization status: {}".format(args.normalize_reward))
    # strategy.print("mean: {}, std {}".format(reward_model.mean, reward_model.std))

    if args.enable_ema:
        ema_model = deepcopy(actor)
    else:
        ema_model = None

    # configure optimizer
    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
    )
    # critic_optim = strategy.create_optimizer(
    #     critic, lr=args.critic_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
    # )

    # prepare datasets
    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=False,
    )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)

    if args.pretrain_data:
        pretrain_data = blending_datasets(
            args.pretrain_data,
            args.pretrain_data_probs,
            strategy,
            args.seed,
            return_eval=False,
        )
        pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))),
            tokenizer,
            pretrain_max_len,
            strategy,
            pretrain_mode=True,
        )
        pretrain_dataloader = itertools.cycle(
            iter(
                strategy.setup_dataloader(
                    pretrain_dataset,
                    args.micro_train_batch_size,
                    True,
                    True,
                    pretrain_dataset.collate_fn,
                )
            )
        )
    else:
        pretrain_dataloader = None

    # configure scheduler
    num_update_steps_per_episodes = len(prompts_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)
    min_actor_learning_rate_lr = getattr(args, "min_actor_learning_rate_lr", 0.1)
        
    # actor_scheduler = get_scheduler(
    #     "cosine",
    #     actor_optim,
    #     num_warmup_steps=math.ceil(max_steps * 0.03),
    #     num_training_steps=max_steps,
    #     min_actor_learning_rate_lr=min_actor_learning_rate_lr
    # )

    if args.lr_scheduler_type == "cosine":
        actor_scheduler = get_cosine_schedule_with_warmup(actor_optim, num_warmup_steps=math.ceil(max_steps * 0.03), num_training_steps=max_steps, min_lr=min_actor_learning_rate_lr)
    else:
        actor_scheduler = get_scheduler(
            args.lr_scheduler_type,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # prepare models/optimizers...
    (
        (actor, actor_optim, actor_scheduler),
        initial_model,
    ) = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        initial_model,
        is_rlhf=True,
    )

    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)
        del ema_model._offload

    # load checkpoint
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    print(f"eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")

    # configure Trainer
    trainer = ReinforceTrainer(
        strategy,
        actor,
        reward_models,
        initial_model,
        ema_model,
        actor_optim,
        actor_scheduler,
        max_epochs=args.max_epochs,
        remote_reward_url=args.remote_rm_url,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        reward_fn=None,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ema_beta=0.992,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        # fro GPT generation
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    trainer.fit(
        prompts_dataloader,
        pretrain_dataloader,
        args,
    )

    # save model checkpoint after fitting on only rank0
    strategy.save_model(
        ema_model if args.enable_ema else actor,
        tokenizer,
        args.save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
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
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_trace_per_sample", type=int, default=1)


    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
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

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="reinforce_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )
    parser.add_argument("--save_ckpt", action="store_true", default=False)
    parser.add_argument("--use_mpi_init", action="store_true", default=False)
    parser.add_argument("--perf", action="store_true", default=False)

    # reward normalization
    parser.add_argument("--normalize_reward_from_multi_traces", action="store_true", default=False)
    parser.add_argument("--normalize_advantage", action="store_true", default=False)

    parser.add_argument("--task_type", type=str, default="130b_reinforce")
    parser.add_argument("--max_min_reward_samples", action="store_true", default=False)
    parser.add_argument("--ema_beta", type=float, default=0.992)
    parser.add_argument("--process_supervision", action="store_true", default=False)
    parser.add_argument("--activation_offload", action="store_true", default=False)
    parser.add_argument("--generation_batch_size", type=int, default=16)
    parser.add_argument("--inference_batch_size", type=int, default=4)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--min_reward_gap", type=float, default=0.0)
    parser.add_argument("--remote_rm_url", type=str, nargs="+", default=None)
    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--normalize_reward_from_multi_traces_with_rloo", action="store_true", default=False)
    parser.add_argument("--normalize_reward_mean_only", action="store_true", default=False)
    parser.add_argument("--min_actor_learning_rate_lr", type=float, default=0.1)
    
    args = parser.parse_args()
    train(args)

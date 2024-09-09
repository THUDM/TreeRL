import itertools
import math
import os
import socket
from copy import deepcopy
from typing import Callable, Dict, List, Tuple
# from torch.optim import 
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import ray
from functools import partial
import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import PPOTrainer
# from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMakerPPO as RemoteExperienceMaker
from openrlhf.trainer.ppo_utils import Experience
from openrlhf.trainer.ppo_utils import RemoteExperienceMakerPPO as RemoteExperienceMaker
from openrlhf.utils import DeepspeedStrategy, blending_datasets, get_tokenizer
from openrlhf.utils.deepspeed_utils import _z3_params_to_fetch
from openrlhf.utils.distributed_util import init_process_group

from .launcher import BasePPORole
from .utils import get_cosine_schedule_with_warmup


class ActorPPOTrainer(PPOTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        critic_train_remote: bool = False,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.critic_train_remote = critic_train_remote

        self.experience_maker = RemoteExperienceMaker(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.reward_fn,
            self.tokenizer_reward,
            vllm_engines=self.vllm_engines,
        )

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            
            refs = [
                engine.init_process_group.remote(
                    master_address, master_port, i * vllm_tensor_parallel_size + 1, world_size, "vllm"
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            self._model_update_group = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="vllm",
            )

            ray.get(refs)

        torch.distributed.barrier()

    def ppo_train(self, global_step):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()

        # 2. triger remote critic model training
        if self.critic_train_remote:
            critic_status_ref = self.critic.fit.remote()

        # 3. actor model training
        status = super().ppo_train(global_step)

        # 4. broadcast weights to vllm engines
        if self.vllm_engines is not None:
            # torch.distributed.barrier()
            self._broadcast_to_vllm()

        # 5. wait remote critic model training done
        if self.critic_train_remote:
            status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        return status

    def training_step(self, experience: Experience, frozen_step=False) -> Dict[str, float]:
        return self.training_step_actor(experience, frozen_step=frozen_step)

    def _broadcast_to_vllm(self):
        # avoid OOM
        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([param]), enabled=self.strategy.args.zero_stage == 3):
                if torch.distributed.get_rank() == 0:
                    torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                    ray.get(refs)
                    
    # def _broadcast_to_vllm(self):
    #     torch.cuda.empty_cache()
        
    #     model = self.actor.model.module
    #     count, num_params = 0, len(list(model.named_parameters()))
    #     for name, param in model.named_parameters():
    #         count += 1  # empty_cache at last param

    #         # Fire all vllm engines for broadcast
    #         if torch.distributed.get_rank() == 0:
    #             shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
    #             [
    #                 engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
    #                 for engine in self.vllm_engines
    #             ]

    #         if self.strategy.args.zero_stage != 3:
    #             # For ZeRO-1/2, broadcast parameter to all vllm engines by rank 0
    #             if torch.distributed.get_rank() == 0:
    #                 torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
    #         else:
    #             # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
    #             with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([param]), enabled=True):
    #                 if torch.distributed.get_rank() == 0:
    #                     torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
    #                     ray.get(refs)


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


# def _get_cosine_schedule_with_warmup_lr_lambda(
#     current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr: float = 0.0
# ):
#     if current_step < num_warmup_steps:
#         return float(current_step) / float(max(1, num_warmup_steps))
#     progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#     return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


# def get_cosine_schedule_with_warmup(
#     optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, min_lr: float = 0.0,
# ):
#     """
#     Create a schedule with a learning rate that decreases following the values of the cosine function between the
#     initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
#     initial lr set in the optimizer.

#     Args:
#         optimizer ([`~torch.optim.Optimizer`]):
#             The optimizer for which to schedule the learning rate.
#         num_warmup_steps (`int`):
#             The number of steps for the warmup phase.
#         num_training_steps (`int`):
#             The total number of training steps.
#         num_cycles (`float`, *optional*, defaults to 0.5):
#             The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
#             following a half-cosine).
#         last_epoch (`int`, *optional*, defaults to -1):
#             The index of the last epoch when resuming training.

#     Return:
#         `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
#     """

#     lr_lambda = partial(
#         _get_cosine_schedule_with_warmup_lr_lambda,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=num_training_steps,
#         num_cycles=num_cycles,
#         min_lr=min_lr
#     )
#     return LambdaLR(optimizer, lr_lambda, last_epoch)


@ray.remote(num_gpus=1)
class ActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            ds_config=strategy.get_ds_train_config(is_actor=True, activation_offload=getattr(strategy.args, "activation_offload", False)),
        )

        # configure tokenizer
        self.tokenizer_reward = get_tokenizer(strategy.args.reward_pretrain.split(",")[0], actor.model, "left", strategy)
        
        self.tokenizer = get_tokenizer(pretrain, actor.model, "left", strategy)
        
        strategy.print(actor)
        self.prepare_datasets()

        args = strategy.args

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
        )

        # configure scheduler
        num_update_steps_per_episodes = len(self.prompts_dataloader) * (args.micro_rollout_batch_size / args.micro_train_batch_size) * args.max_epochs // strategy.accumulated_gradient
        max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)
        self.max_steps = max_steps

        min_actor_learning_rate_lr = getattr(args, "min_actor_learning_rate_lr", 0.1)

        if args.lr_scheduler_type == "cosine":
            actor_scheduler = get_cosine_schedule_with_warmup(actor_optim, num_warmup_steps=math.ceil(max_steps * 0.03), num_training_steps=max_steps, min_lr=min_actor_learning_rate_lr)
        else:
            actor_scheduler = get_scheduler(
                args.lr_scheduler_type,
                actor_optim,
                num_warmup_steps=math.ceil(max_steps * 0.03),
                num_training_steps=max_steps,
            )

        # actor_scheduler = get_scheduler(
        #     self.strategy.args.lr_scheduler_type,
        #     actor_optim,
        #     num_warmup_steps=math.ceil(max_steps * 0.03),
        #     num_training_steps=max_steps,
        # )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        self.past_global_step = 0

        if strategy.args.load_ckpt:
            ckpt_path = strategy.args.ckpt_path
            exist_dirs = os.listdir(ckpt_path)
            exist_dirs = [d for d in exist_dirs if "_actor_ckpt_" in d]

            if len(exist_dirs) > 0:
                exist_dirs = sorted(exist_dirs, key=lambda x:int(x[len("_actor_ckpt_global_step"):]),  reverse=True)

                target_dir = os.path.join(ckpt_path, exist_dirs[0]) 
                self.past_global_step = int(exist_dirs[0][len("_actor_ckpt_global_step"):])

                strategy.load_ckpt(self.actor.model, load_dir=target_dir, load_lr_scheduler_states=True, load_optimizer_states=True, load_module_strict=True)
            else:
                print(f"************** failed to load ckpt from {ckpt_path} because no ckpt files found **************")


        if args.enable_ema:
            # ema_model = deepcopy(actor)
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                # lora_rank=strategy.args.lora_rank,
                # lora_alpha=strategy.args.lora_alpha,
                # target_modules=strategy.args.target_modules,
                ds_config=strategy.get_ds_eval_config(offload=True),
                # ds_config=strategy.get_ds_eval_config()
            )
        else:
            ema_model = None

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
            del ema_model._offload
        else:
            self.ema_model = None

        self.strategy.args.past_global_steps = self.past_global_step

        # self.reset_dataloader_state()

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

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
        prompts_dataset = PromptDataset(prompts_data, self.tokenizer, strategy, input_template=args.input_template)
        self.prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, pin_memory=True, shuffle=False)

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
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        pin_memory=True,
                        shuffle=False,
                        collate_fn=pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    # def reset_dataloader_state(self):
    #     if self.past_global_step > 0:
    #         past_steps = self.past_global_step * self.strategy.args.rollout_batch_size / self.strategy.args.micro_rollout_batch_size / self.strategy.world_size
            
    #         _past_steps = int(past_steps)
    #         # for _ in range(past_steps):
    #         for batch in self.prompts_dataloader:
    #             if _past_steps == 0:
    #                 break
    #             _past_steps -= 1
            
    #         _past_steps = int(past_steps)
    #         if self.pretrain_dataloader is not None:
    #             for batch in self.pretrain_dataloader:
    #                 if past_steps == 0:
    #                     break
    #                 past_steps -= 1
                
    def max_steps(self):
        """Return the maximum number of steps."""
        return self.max_steps

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        critic_train_remote: bool = False,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            tokenizer=self.tokenizer,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=args.ema_beta,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # fro GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            tokenizer_reward=self.tokenizer_reward,
        )

        trainer.fit(self.prompts_dataloader, self.pretrain_dataloader, args)

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )

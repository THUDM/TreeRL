import math
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, ReinforcePolicyLoss, SwitchBalancingLoss, ValueLoss, PolicyLoss
from openrlhf.models.utils import masked_mean

from .ppo_utils import AdaptiveKLController, FixedKLController, NaiveReplayBuffer
from .ppo_utils import Experience
from .ppo_utils.experience_maker import NaiveExperienceMaker
from .ppo_utils.experience_maker_reinforce import NaiveExperienceMakerReinforce


class ReinforceTrainer(ABC):
    """
        Trainer for PPO algorithm.

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        remote_rm_url: List[str] = None,
        tokenizer_reward: Optional[Callable[[Any], dict]] = None,
        remote_reward_url: Optional[str] = None,
        use_mcts: bool = False,
        use_vinevalue: bool = False,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn
        self.tokenizer_reward = tokenizer_reward
        self.remote_rm_url = remote_rm_url

        self.actor = actor
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.use_mcts = use_mcts
        self.use_vinevalue = use_vinevalue

        if use_vinevalue:
            self.actor_loss_fn = PolicyLoss(eps_clip)
        else:
            self.actor_loss_fn = ReinforcePolicyLoss(eps_clip)
        self.ptx_loss_fn = GPTLMLoss()

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        # self.experience_maker = NaiveExperienceMaker(
        #     actor, None, reward_model, initial_model, tokenizer, prompt_max_len, self.kl_ctl, strategy, reward_fn
        # )
        self.experience_maker = NaiveExperienceMakerReinforce(
            actor, None, reward_model, remote_reward_url, initial_model, tokenizer, prompt_max_len, self.kl_ctl, strategy, reward_fn
        )
        self.replay_buffer = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            # wandb.login(key=strategy.args.use_wandb)
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                # reinit=True,
                id=strategy.args.wandb_id,
                resume="allow",
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
            # self.sample_table = pd.DataFrame(columns=["prompt", "response"])
            

    def fit(
        self,
        prompts_dataloader,
        pretrain_dataloader,
        args,
    ) -> None:
        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)
        global_step = 1

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = prompts_dataloader.__len__() // update_timesteps  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # self.strategy.copy_model_weight(self.initial_model, self.actor)
        
        past_steps = 0
        
        if "_actor_global_step" in args.pretrain:
            past_steps = int(args.pretrain.split("_actor_global_step")[-1])
            print("*" * 20)
            print("past_steps: ", past_steps)
        else:
            print("No past steps")
            print("*" * 20)
        
        for episode in range(args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader:
                if past_steps > 0:
                    past_steps -= 1
                    global_step += 1
                    pbar.update()
                    continue
                # experience = self.experience_maker.make_experience(rand_prompts, use_mcts = self.use_mcts, use_vinevalue = self.use_vinevalue, sample_table = self.sample_table, **self.generate_kwargs)
                experience = self.experience_maker.make_experience(rand_prompts, use_mcts = self.use_mcts, use_vinevalue = self.use_vinevalue,use_sentence_level_value = False, **self.generate_kwargs)

                # print prompt/answer in each update step
                # if global_step % update_timesteps == 0:

                try:
                    sequences = experience.sequences.clone().detach()
                    if torch.is_tensor(sequences):
                        sequences = sequences.cpu().tolist()

                    for i, sequence in enumerate(sequences):
                        # sequence = sequences[0]
                        # sequence[sequence >= self.tokenizer.vocab_size] = self.tokenizer.eos_token_id

                        vocab_size = len(self.tokenizer.additional_special_tokens) + self.tokenizer.vocab_size
                        sequence = [min(x, vocab_size - 1) for x in sequence]
                        
                        output = self.tokenizer.decode(sequence, skip_special_tokens=False)
                        self.strategy.print(output.replace("<|endoftext|>", "").strip())
                        # output = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                        # self.strategy.print(output[0].replace("<|endoftext|>", ""))
                        if i > 3:
                            break

                except Exception as e:
                    self.strategy.print(str("Error in decoding prompt/answer: ") + str (e))
                        
                self.replay_buffer.append(experience)

                if global_step % update_timesteps == 0:
                    torch.cuda.empty_cache()
                    if self.strategy.args.normalize_advantage:
                        self.replay_buffer.normalize("advantages", self.strategy)
                    status = self.reinforce_train(global_step)
                    self.replay_buffer.clear()
                    torch.cuda.empty_cache()
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                    # logs/checkpoints
                    self.save_logs_and_checkpoints(args, global_step // update_timesteps, pbar, status)

                pbar.update()
                global_step = global_step + 1

    def reinforce_train(self, global_step):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=False,
            # shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()
        micro_steps = len(dataloader)
        
        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for i, experience in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience)

                # for DP
                # weighted mean for kl
                status["kl"] *= status["response_length"]
                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]

                status_list.append(status)
                self.save_logs_only(status, step=int(global_step * micro_steps + i))

                short_status = {
                    "pg": status["policy_loss"],
                    "rm": status["reward"],
                    # "ret": status["return"],
                    "glen": status["response_length"],
                    "tlen": status["total_length"],
                    "kl": status["kl"],
                }

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience) -> Dict[str, float]:
        status = self.training_step_actor(experience)
        return status

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()

        num_actions = experience.action_mask.size(1)
        # actor loss
        action_log_probs, output = self.actor(
            experience.sequences, num_actions, attention_mask=experience.attention_mask, return_output=True
        )

        # !!! potential BUG
        # loss function
        if self.use_vinevalue:
            actor_loss = self.actor_loss_fn(
                action_log_probs,
                experience.action_log_probs,
                experience.advantages,
                action_mask=experience.action_mask,
            )
        else:
            actor_loss = self.actor_loss_fn(
                action_log_probs,
                experience.action_log_probs,
                experience.advantages,
                action_mask=experience.action_mask,
                kl=experience.kl,
                kl_coef=self.strategy.args.init_kl_coef,
            )
        if self.strategy.args.l2_logits_loss_coeff > 0:
            print("l2_logits_loss_coeff",self.strategy.args.l2_logits_loss_coeff)
            logits = output["logits"].pow(2).mean(dim=-1)
            advantage = experience.advantages
            advantage_seq_avg = advantage.abs().mean(dim=-1)
            l2_loss = advantage_seq_avg.view(-1, 1) * logits
            l2_loss = l2_loss.mean()
        else:
            l2_loss = 0
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef

        if self.strategy.args.l2_logits_loss_coeff > 0:
            loss += l2_loss * self.strategy.args.l2_logits_loss_coeff
        
        self.strategy.backward(loss, self.actor, self.actor_optim)

        # grad norm calculation of deepspeed - by lurui
        # reference https://github.com/deepspeedai/DeepSpeed/issues/5883
        import deepspeed.utils
        grad_norm = 0.0
        for param in self.actor.model.module.parameters():
            grad_data = deepspeed.utils.safe_get_full_grad(param)
            # if self.strategy.is_rank_0():
                # print("grad_data.norm(2): ", grad_data.norm(2))
            grad_norm += grad_data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Print some debug information
        if self.strategy.is_rank_0():
            self.strategy.print(f"Debug - Gradient norm: {grad_norm}")
            
        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # status
        status = {
            "policy_loss": actor_loss.item(),
            "grad_norm": grad_norm,
        }
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            # if k == "reward":  
            #     print("here is reward",k, v,v.shape(), v.mean().item())
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            # elif (k == "reward") and self.strategy.args.process_supervision:
            #     if (v != 0).sum() == 0:
            #         status[k] = 0
            #     else:
            #         status[k] = (v.sum() / (v != 0).sum()).item()
            else:
                status[k] = v.mean().item()
        return status

    def save_logs_only(self, logs_dict, step):
        # if global_step % args.logging_steps == 0:
        # step bar
        # step_bar.set_postfix(logs_dict)
        # wandb
        if self._wandb is not None and self.strategy.is_rank_0():
            logs = {
                "train/%s" % k: v
                for k, v in {
                    **logs_dict,
                    "step": step,
                }.items()
            }
            self._wandb.log(logs)
                
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            step_bar.set_postfix(logs_dict)
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)
            pass
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            # self.strategy.save_ckpt(
            #     self.actor.model, os.path.join(args.ckpt_path, "_actor"), tag, args.max_ckpt_num, args.max_ckpt_mem
            # )
            self.strategy.save_model(
                self.actor.model, self.tokenizer, os.path.join(args.ckpt_path, f"_actor_{tag}")
            )

        if self.strategy.args.save_ckpt:
            os.makedirs(os.path.join(args.ckpt_path, f"_actor_ckpt_{tag}"), exist_ok=True)
            self.strategy.save_ckpt(
                self.actor.model, 
                os.path.join(args.ckpt_path, f"_actor_ckpt_{tag}"), 
                tag, 
                max_num=args.max_ckpt_num, 
                max_mem=args.max_ckpt_mem
            )
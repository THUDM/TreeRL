import math
from abc import ABC
import os
import pdb
# import loralib as lora
import torch

from torch import nn
import torch.nn.functional
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from openrlhf.models import (
    LogExpLoss, 
    PairWiseLoss, 
    SwitchBalancingLoss, 
    PointSigmoidLoss, 
    PointMSELoss,
    CrossEntropyLoss
)


class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.process_supervision = self.args.process_supervision or self.args.mix_supervision

        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss")
        elif loss == "logexp":
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss")
        elif loss == "pointsigmoid":
            self.loss_fn = PointSigmoidLoss()
        elif loss == "pointmse":
            self.loss_fn = PointMSELoss()
        elif loss == "crossentropy":
            self.loss_fn = CrossEntropyLoss()
        else:
            raise NotImplemented
        

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

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
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            acc_mean = 0
            loss_mean = 0
            for chosen_ids, c_mask, reject_ids, r_mask, margin, labels in self.train_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                labels = labels.to(torch.cuda.current_device()).view(-1)

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                        
                with torch.autograd.set_detect_anomaly(True):
                    if chosen_ids.numel() > 0 and reject_ids.numel() > 0: # pairwise loss
                        reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                        r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                        
                        chosen_reward, reject_reward, aux_loss = self.concatenated_forward(
                            self.model, chosen_ids, c_mask, reject_ids, r_mask
                        )
                        torch.cuda.empty_cache()

                        # loss function
                        if self.compute_fp32_loss:
                            chosen_reward = chosen_reward.float()
                            reject_reward = reject_reward.float()
                        
                        if chosen_ids.shape[0] > reject_ids.shape[0]:
                            inst_rewards = chosen_reward[len(reject_reward):]
                            _chosen_reward = chosen_reward[: len(reject_reward)]

                            inst_labels = labels[len(reject_reward):]
                            # inst_loss = (inst_rewards * inst_labels) + (inst_labels - 1).abs() / 2 
                            inst_loss = inst_rewards * inst_labels

                            assert (inst_labels.abs() > 0).all(), inst_labels
                            inst_predictions = ((inst_rewards.sigmoid() - 0.5) * inst_labels) > 0

                            # inst_labels[inst_labels < 0] = 0
                            preference_loss = self.loss_fn(_chosen_reward, reject_reward, margin)
                            # absolute_loss = torch.nn.functional.cross_entropy(inst_rewards, inst_labels)
                            absolute_loss = - torch.nn.functional.logsigmoid(inst_loss).mean()

                            preference_loss = (preference_loss + absolute_loss) / 2
                            
                            _acc_batch = torch.cat([_chosen_reward > reject_reward, inst_predictions])
                            acc_mean = acc_mean * 0.9 + 0.1 * _acc_batch.float().mean().item()
                        else:
                            preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                            acc_mean = acc_mean * 0.9 + 0.1 * (chosen_reward > reject_reward).float().mean().item()
                        
                            # regularization_loss = chosen_reward.pow_(2).mean() + reject_reward.pow_(2).mean()
                            # preference_loss = preference_loss + 0.001 * regularization_loss                
                    else:
                        rewards, aux_loss = self.non_concatenated_forward(self.model, chosen_ids, c_mask)
                        if self.compute_fp32_loss:
                            rewards = rewards.float()
                        
                        # inst_loss = (rewards.sigmoid() * labels) + (labels - 1).abs() / 2
                        inst_loss = rewards * labels

                        # inst_loss = rewards
                        # rewards_ = (rewards * labels) + (labels - 1).abs() / 2

                        # rejected_reward = torch.zeros_like(rewards)
                        assert (labels.abs() > 0).all(), labels
                        acc_mean = acc_mean * 0.9 + 0.1 * (((rewards.sigmoid() - 0.5) * labels) > 0).float().mean().item()
                        
                        # labels[labels < 0] = 0                      
                        # preference_loss = torch.nn.functional.cross_entropy(rewards, labels)
                        preference_loss = - torch.nn.functional.logsigmoid(inst_loss)
                        # preference_loss = - torch.log(inst_loss)
                        preference_loss = preference_loss.mean()

                        chosen_reward = rewards[labels == 1] if (labels == 1).float().sum() > 0 else torch.zeros_like(rewards)
                        reject_reward = rewards[labels == -1] if (labels == -1).float().sum() > 0 else torch.zeros_like(rewards)
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                torch.cuda.empty_cache()
                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                # optional rm info
                logs_dict = {
                    "preference_loss": preference_loss.item(),
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "acc_mean": acc_mean,
                    "loss_mean": loss_mean,
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            save_dir = args.save_path
            
            tag = f"global_step{global_step}"
            state_dir = os.path.join(save_dir, f"_ckpt_{tag}")
            os.makedirs(state_dir, exist_ok=True)

            self.strategy.save_ckpt(self.model, state_dir, tag, args.max_ckpt_num, args.max_ckpt_mem)
            
            model_path = os.path.join(save_dir, f"model_{tag}")
            os.makedirs(model_path, exist_ok=True)
            self.strategy.save_model(self.model, self.tokenizer, model_path)
            print("save ckpt at global step %d" % global_step)

        # eval
        # if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0
            for chosen_ids, c_mask, reject_ids, r_mask, margin, labels in eval_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                
                labels = labels.to(torch.cuda.current_device()).view(-1)
                margin = margin.to(torch.cuda.current_device())

                # if reject_ids.numel() > 0:
                #     reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                #     r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                # else:
                #     reject_ids = None
                #     r_mask = None
                
                if chosen_ids.numel() > 0 and reject_ids.numel() > 0:
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                                        
                    chosen_reward, reject_reward, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask
                    )
                    
                    if chosen_ids.shape[0] > reject_ids.shape[0]:
                        inst_rewards = chosen_reward[len(reject_reward):]
                        inst_labels = labels[len(reject_reward):]
                        inst_chosen_reward = (inst_rewards - 0.5) * inst_labels
                        inst_reject_reward = torch.zeros_like(inst_chosen_reward)
                        
                        preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                        # inst_loss = torch.nn.functional.cross_entropy(inst_rewards, inst_labels)
                        # inst_rewards = (inst_rewards * inst_labels) + (inst_labels - 1).abs() / 2
                        inst_rewards = inst_rewards * inst_labels

                        inst_loss = -torch.nn.functional.logsigmoid(inst_rewards).mean()

                        loss = (preference_loss + inst_loss) / 2
                        chosen_reward[len(reject_reward):] = inst_chosen_reward
                        reject_reward = torch.cat((reject_reward, inst_reject_reward))
                    else:
                        loss = self.loss_fn(chosen_reward, reject_reward, margin)
                else:
                    overall_rewards, aux_loss = self.non_concatenated_forward(self.model, chosen_ids, c_mask)
                    # rewards = (rewards * labels) + (labels - 1).abs() / 2
                    chosen_reward = (overall_rewards - 0.5) * labels
                    reject_reward = torch.zeros_like(chosen_reward)
                    # labels[labels < 0] = 0
                    
                    # loss = torch.nn.functional.cross_entropy(chosen_rewards, labels)
                    # loss = (overall_rewards * labels) + (labels - 1).abs() / 2
                    loss = -torch.nn.functional.logsigmoid(overall_rewards * labels).mean()

                rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                acc += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                step_bar.update()

            acc_mean = acc / self.eval_dataloader.__len__()
            loss_mean = loss_sum / self.eval_dataloader.__len__()

            rewards = torch.cat(rewards).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            # save mean std
            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, concat=True):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        if concat:
            input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
            all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
            chosen_rewards = all_values[:chosen_ids.shape[0]]
            rejected_rewards = all_values[chosen_ids.shape[0]:]
            aux_loss = output.aux_loss if "aux_loss" in output else []
        else:
            input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
            batch_size = input_ids.shape[0]
            all_values = []
            output = []
            for i in range(batch_size):
                values, _ = model(input_ids[i:i+1], attention_mask=att_masks[i:i+1], return_output=True)
                all_values.append(values)

            all_values = torch.cat(all_values, dim=0)
            chosen_rewards = all_values[:chosen_ids.shape[0]]
            rejected_rewards = all_values[chosen_ids.shape[0]:]
            
            aux_loss = []
        
        return chosen_rewards, rejected_rewards, aux_loss
    
    def non_concatenated_forward(self, model, input_ids, mask):
        rewards, output = model(input_ids, attention_mask=mask, return_output=True, process_supervision=self.process_supervision)
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return rewards, aux_loss

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks
    

class ProcessRewardModelTrainer(RewardModelTrainer):
    def non_concatenated_forward(self, model, input_ids, mask):
        rewards, output = model(input_ids, attention_mask=mask, return_output=True, process_supervision=True)
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return rewards, aux_loss
    
    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            acc_mean = 0
            loss_mean = 0
            for input_ids, attention_mask, labels in self.train_dataloader:                
                input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_mask.squeeze(1).to(torch.cuda.current_device())
                labels = labels.to(torch.cuda.current_device()).view(-1)

                rewards, aux_loss = self.non_concatenated_forward(self.model, input_ids, attention_mask)
                if self.compute_fp32_loss:
                    rewards = rewards.float()

                label_mask = (labels != 0).view(-1)
                rewards = rewards.view(-1)[label_mask]
                labels = labels.view(-1)[label_mask]
                # labels[labels < 0] = 0
                
                assert (labels.abs() > 0).all(), labels
                assert len(labels) > 0, len(labels)

                if isinstance(self.loss_fn, PointMSELoss):
                    acc_mean = 0.5
                else:
                    acc_mean = acc_mean * 0.9 + 0.1 * (((rewards.sigmoid() - 0.5) * labels) > 0).float().mean().item()
                  
                # reward_loss = rewards * labels
                # preference_loss = torch.nn.functional.cross_entropy(rewards, labels)
                # logit_loss = - torch.nn.functional.logsigmoid(reward_loss)
                logit_loss = self.loss_fn(rewards, labels)
                # preference_loss = - torch.log(inst_loss)
                logit_loss = logit_loss.mean()

                # chosen_reward = rewards[labels == 1] if (labels == 1).float().sum() > 0 else torch.zeros_like(rewards)
                # reject_reward = rewards[labels == -1] if (labels == -1).float().sum() > 0 else torch.zeros_like(rewards)

                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                loss = logit_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * logit_loss.item()
                # optional rm info
                logs_dict = {
                    "loss": logit_loss.item(),
                    # "chosen_reward": chosen_reward.mean().item(),
                    # "reject_reward": reject_reward.mean().item(),
                    "acc_mean": acc_mean,
                    "loss_mean": loss_mean,
                }
                
                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        print("start evaluation...")
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0
            for input_ids, attention_mask, labels in eval_dataloader:
                input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_mask.squeeze(1).to(torch.cuda.current_device())
                
                labels = labels.to(torch.cuda.current_device()).view(-1)
                
                overall_rewards, aux_loss = self.non_concatenated_forward(self.model, input_ids, attention_mask)
                # rewards = (rewards * labels) + (labels - 1).abs() / 2
                # labels[labels < 0] = 0
                
                # loss = torch.nn.functional.cross_entropy(chosen_rewards, labels)
                # loss = (overall_rewards * labels) + (labels - 1).abs() / 2
                
                labels_mask = labels != 0
                overall_rewards = overall_rewards.view(-1)
                labels = labels[labels_mask]
                overall_rewards = overall_rewards[labels_mask]
                
                # loss = -torch.nn.functional.logsigmoid(overall_rewards * labels).mean()
                loss = self.loss_fn(overall_rewards, labels)
                # rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                rewards.append(overall_rewards.flatten())
                
                # acc += (chosen_reward > reject_reward).float().mean().item()
                if isinstance(self.loss_fn, PointMSELoss):
                    acc += 0.5
                else:
                    acc += (((overall_rewards.sigmoid() - 0.5) * labels) > 0).float().mean().item()
                
                loss_sum += loss.item()
                step_bar.update()

            print("evaluation done...")

            acc_mean = acc / self.eval_dataloader.__len__()
            loss_mean = loss_sum / self.eval_dataloader.__len__()

            rewards = torch.cat(rewards).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            # save mean std
            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)

        self.model.train()  # reset model state
        

class RewardProcessMixModelTrainer(RewardModelTrainer):
    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            acc_mean = 0
            loss_mean = 0
            VALID_INDEX = 1
            p_loss, c_loss = None, None
            for chosen_ids, c_mask, reject_ids, r_mask, margin, labels in self.train_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                labels = labels.to(torch.cuda.current_device()).view(-1)

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                
                with torch.autograd.set_detect_anomaly(True):
                    if chosen_ids.numel() > 0 and reject_ids.numel() > 0: # pairwise loss
                        reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                        r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                        
                        chosen_reward, reject_reward, aux_loss = self.concatenated_forward(
                            self.model, chosen_ids, c_mask, reject_ids, r_mask
                        )
                        torch.cuda.empty_cache()

                        # loss function
                        if self.compute_fp32_loss:
                            chosen_reward = chosen_reward.float()
                            reject_reward = reject_reward.float()
                                                
                        if chosen_ids.shape[0] > reject_ids.shape[0]:
                            inst_rewards = chosen_reward[len(reject_reward):]
                            _chosen_reward = chosen_reward[: len(reject_reward)]

                            inst_labels = labels[len(reject_reward):]
                            # inst_loss = (inst_rewards * inst_labels) + (inst_labels - 1).abs() / 2 
                            inst_loss = inst_rewards * inst_labels

                            assert (inst_labels.abs() > 0).all(), inst_labels
                            inst_predictions = ((inst_rewards.sigmoid() - 0.5) * inst_labels) > 0

                            # inst_labels[inst_labels < 0] = 0
                            preference_loss = self.loss_fn(_chosen_reward, reject_reward, margin)
                            # absolute_loss = torch.nn.functional.cross_entropy(inst_rewards, inst_labels)
                            absolute_loss = - torch.nn.functional.logsigmoid(inst_loss).mean()

                            preference_loss = (preference_loss + absolute_loss) / 2
                            
                            _acc_batch = torch.cat([_chosen_reward > reject_reward, inst_predictions])
                            acc_mean = acc_mean * 0.9 + 0.1 * _acc_batch.float().mean().item()
                        else:
                            preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                            acc_mean = acc_mean * 0.9 + 0.1 * (chosen_reward > reject_reward).float().mean().item()
                        
                            # regularization_loss = chosen_reward.pow_(2).mean() + reject_reward.pow_(2).mean()
                            # preference_loss = preference_loss + 0.001 * regularization_loss
                        c_loss = preference_loss
                    else:
                        rewards, aux_loss = self.non_concatenated_forward(self.model, chosen_ids, c_mask)
                        if self.compute_fp32_loss:
                            rewards = rewards.float()
                        
                        # inst_loss = (rewards.sigmoid() * labels) + (labels - 1).abs() / 2
                        # inst_loss = rewards * labels
                        # labels = (labels + 1) // 2
                        labels_mask = labels != 0
                        overall_rewards = rewards.view(-1, *rewards.shape[2:])
                        labels = labels[labels_mask]
                        overall_rewards = overall_rewards[labels_mask]
                        labels[labels == -1] = 0
             
                        preference_loss = torch.nn.CrossEntropyLoss()(overall_rewards, labels.long())
                        # inst_loss = rewards
                        # rewards_ = (rewards * labels) + (labels - 1).abs() / 2

                        # rejected_reward = torch.zeros_like(rewards)
                        # assert (labels.abs() > 0).all(), labels
                        # acc_mean = acc_mean * 0.9 + 0.1 * (((rewards.sigmoid() - 0.5) * labels) > 0).float().mean().item()
                        acc_mean = acc_mean * 0.9 + 0.1 * (((overall_rewards.sigmoid()[:, VALID_INDEX] - 0.5) * labels) > 0).float().mean().item()
                
                        
                        # labels[labels < 0] = 0                      
                        # preference_loss = torch.nn.functional.cross_entropy(rewards, labels)
                        # preference_loss = - torch.nn.functional.logsigmoid(inst_loss)
                        # preference_loss = - torch.log(inst_loss)
                        # preference_loss = preference_loss.mean()

                        # chosen_reward = rewards[labels == 1] if (labels == 1).float().sum() > 0 else torch.zeros_like(rewards)
                        # chosen_reward = overall_rewards[labels == 1] if (labels == 1).float().sum() > 0 else torch.zeros_like(overall_rewards)
                        # reject_reward = overall_rewards[labels == -1] if (labels == -1).float().sum() > 0 else torch.zeros_like(overall_rewards)
                        # reject_reward = rewards[labels == -1] if (labels == -1).float().sum() > 0 else torch.zeros_like(rewards)
                        chosen_reward, reject_reward = torch.tensor([1.]), torch.tensor([1.])
                        p_loss = preference_loss
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                torch.cuda.empty_cache()
                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                # optional rm info
                logs_dict = {
                    "p_loss": p_loss.item() if p_loss is not None else 1,
                    "c_loss": c_loss.item() if c_loss is not None else 1,
                    "preference_loss": preference_loss.item(),
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "acc_mean": acc_mean,
                    "loss_mean": loss_mean,
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

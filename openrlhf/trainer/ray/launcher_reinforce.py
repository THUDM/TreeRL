import logging
import os
import socket
from typing import Callable, List, Optional, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import DeepspeedStrategy, get_tokenizer
from .launcher import BasePPORole, RewardModelRayActor, ReferenceModelRayActor


class ReinforceRayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BasePPORole]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BasePPORole],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [
                {"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)
            ]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor, num_gpus=num_gpus_per_actor
            ).remote(world_size, 0, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank // self._num_gpus_per_node,
                        ),
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor, num_gpus=num_gpus_per_actor
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_fit_actor_model(
        self,
        initial_model_group: "ReinforceRayActorGroup",
        reward_model_groups: List["ReinforceRayActorGroup"],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List = None,
    ):
        """Train actor model.

        Args:
            initial_model_group (PPORayActorGroup): reference model group.
            reward_model_groups (PPORayActorGroup): reward model groups.
            reward_fn: reward calculate function, must be specified if using multiple reward models.
            remote_rm_url: remote RM APIs,
            vllm_engines: vllm engines for text generation, if not specified, generate text by actor model directly.

        Returns:
            List: list of remote object refs.
        """
        assert (
            remote_rm_url or 
            len(reward_model_groups) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        initial_actors = initial_model_group._actor_handlers

        refs = []
        # TODO(wuxibin): actor model choose reward/initial model in a
        # round robin fashion, implement more efficient dispatching strategy.
        for i, actor in enumerate(self._actor_handlers):
            initial_actor = initial_actors[i % len(initial_actors)]

            reward_actors = []
            if not remote_rm_url:
                for reward_model_group in reward_model_groups:
                    actors = reward_model_group._actor_handlers
                    reward_actors.append(actors[i % len(actors)])

            refs.append(
                actor.fit.remote(
                    initial_model=initial_actor,
                    reward_model=reward_actors,
                    remote_rm_url=remote_rm_url,
                    reward_fn=reward_fn,
                    vllm_engines=vllm_engines,
                    # whether this actor should triger corresponding critic model training
                    # critic_train_remote=(i < len(critic_actors)),
                )
            )

        return refs

    def async_save_actor_model(self):
        """Save actor model on rank 0.

        Returns:
            List: list of remote object refs.
        """
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs

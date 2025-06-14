# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os

import deepspeed
import numpy as np
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def get_train_ds_config(
    offload,
    adam_offload=True,
    stage=2,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=False,
    activation_offload=False
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        # "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "offload_param": {
            "device": device,
            "pin_memory": True
        },
        # "sub_group_size": "auto",
        # "stage3_max_live_parameters": "auto",
        # "stage3_max_reuse_distance": "auto",
        # "stage3_param_persistence_threshold": "auto",
        # "stage3_prefetch_bucket_size": "auto",
        # "reduce_bucket_size": "auto",

        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_param_persistence_threshold": 1e8,
        "stage3_prefetch_bucket_size": 1e8,
        "reduce_bucket_size": 1e8,
        # "stage3_max_live_parameters": 1e8,
        # "stage3_max_reuse_distance": 1e8,
        # "stage3_param_persistence_threshold": 1e8,
        # "stage3_prefetch_bucket_size": 1e8,
        # "reduce_bucket_size": 1e8,
        # ZeRO++
        # "zero_hpz_partition_size": zpg,
        # "zero_quantized_weights": False,
        # "zero_quantized_gradients": False,
    }
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0

    ds_config = {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {
            "grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"
        },
    }
    
    if activation_offload:
        print("!!!###### ------ Activations offload enabled ------- ########")
        checkpointint_opt_dict = {
            "partition_activations": True,
            # "checkpoint_in_cpu": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize": False,
            "profile": False,
            "cpu_checkpointing": True,
            "checkpoint_in_cpu": True
        }
        ds_config["activation_checkpointing"] = checkpointint_opt_dict
        
    return  ds_config


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]

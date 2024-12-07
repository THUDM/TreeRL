set -x 
export PATH=$HOME/.local/bin/:$PATH

# ray start --head --node-ip-address=192.168.246.53 --port=6379 --block &
# ray start --address 192.168.246.53:6379 --block &

# pkill -9 ray
# pkill -9 python
# pkill -9 gcs_server
# sleep 5

# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

NUM_TRACE=8
KL=0.0001

TAG=glm9b-ppo-ms${NUM_TRACE}-approx-kl-${KL}-chain
SAVE_DIR=/workspace/lurui/openrlhf-glm/checkpoints/ppo/$TAG
mkdir -p $SAVE_DIR

# export LD_LIBRARY_PATH=/workspace/zhenyu/code/nccl/build/lib:$LD_LIBRARY_PATH
# "LD_LIBRARY_PATH": "/workspace/zhenyu/code/nccl/build/lib:${LD_LIBRARY_PATH}", 

    # --runtime-env-json='{"working_dir": "/workspace/zhenyu/code/OpenRLHF", "pip": "/workspace/zhenyu/code/OpenRLHF/requirements.txt", "excludes": ["*.ipynb", "tests", "logs", "wandb"], "env_vars": {"LD_LIBRARY_PATH": "/workspace/zhenyu/code/nccl/build/lib:/usr/local/cuda/compat/lib.real:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"}}' \

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
        "working_dir": "/workspace/lurui/openrlhf-glm",
        "pip": "/workspace/lurui/openrlhf-glm/requirements.txt",
        "excludes": ["*.ipynb", "tests", "log", "logs", "wandb", "checkpoints","reward","datasets"],
        "env_vars": {
            "NCCL_TIMEOUT_MS": "3600000"
        }
    }' \
    -- python train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --pretrain /data/o1-cloud/checkpoints/sft/glm_9b_1102 \
    --reward_pretrain /data/o1-cloud/checkpoints/sft/glm_9b_1102 \
    --critic_pretrain /data/o1-cloud/checkpoints/sft/glm_9b_1102 \
    --save_path $SAVE_DIR \
    --ckpt_path $SAVE_DIR \
    --micro_train_batch_size 4 \
    --train_batch_size $((NUM_TRACE * 16)) \
    --micro_rollout_batch_size 1 \
    --generation_batch_size 16 \
    --inference_batch_size 1 \
    --rollout_batch_size 16 \
    --actor_freeze_steps 100 \
    --max_epochs 1 \
    --max_samples 100000 \
    --prompt_max_len 1024 \
    --generate_max_len 8192 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 2e-6 \
    --critic_learning_rate 2e-6 \
    --lr_scheduler_type cosine \
    --min_actor_learning_rate_lr 1 \
    --l2 0.1 \
    --init_kl_coef $KL \
    --prompt_data /workspace/lurui/openrlhf-glm/openrlhf/datasets/train_30k.jsonl \
    --prompt_data_probs 1 \
    --normalize_reward \
    --normalize_reward_from_multi_traces \
    --top_p 0.9 \
    --temperature 0.95 \
    --actor_init_on_gpu \
    --num_trace_per_sample $NUM_TRACE \
    --gradient_checkpointing \
    --input_key text \
    --save_steps 20 \
    --perf \
    --load_ckpt \
    --wandb_run_name $TAG \
    --use_wandb lyj \
    --task_type 9b-glm4-ppo-chain \
    --label_key label \
    --source_key data_type \
    --remote_rm_url /workspace/lurui/openrlhf-glm/examples/tools/rm_urls.json \
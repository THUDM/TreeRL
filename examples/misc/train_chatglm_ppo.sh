set -x 
export PATH=$HOME/.local/bin/:$PATH


# ray start --head --node-ip-address=192.168.246.53 --port=6379 --block &
# ray start --address 192.168.246.53:6379 --block &

# pkill -9 ray
# pkill -9 python
# pkill -9 gcs_server
# sleep 5

# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8


DATA_DIR=/workspace/data
SAVE_DIR=/workspace/checkpoints

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/workspace/code/glm-openrlhf", "pip": "/workspace/code/glm-openrlhf/requirements.txt"}' \
    -- python train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --pretrain chatglm3-6b \
    --reward_pretrain <rm_path> \
    --save_path $SAVE_DIR \
    --ckpt_path $SAVE_DIR \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 1e-5 \
    --critic_learning_rate 1e-5 \
    --init_kl_coef 0.005 \
    --prompt_data <path_to_data> \
    --prompt_data_probs 1 \
    --max_samples 10000 \
    --normalize_reward \
    --normalize_advantage \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing \
    --input_key prompt \
    --perf \

ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8


ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/workspace/zhenyu/code/OpenRLHF", "pip": "/workspace/zhenyu/code/OpenRLHF/requirements.txt"}' \
    -- python examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --pretrain /workspace/zhenyu/checkpoints/llama/Llama-2-7b-chat-hf \
    --reward_pretrain /workspace/zhenyu/checkpoints/openrlhf/7b_llama_reward \
    --save_path /workspace/zhenyu/checkpoints/openrlhf/7b_llama_ppo \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data $DATA_DIR/Dahoas/full-hh-rlhf/train.jsonl,$DATA_DIR/Open-Orca/OpenOrca/train.jsonl \
    --prompt_data_probs 0.5,0.5 \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --use_wandb d0252b186de41ef07ac87f8a2ac2c0ce90c0e08f \
    --wandb_run_name 7b_llama_ppo_test \
    --gradient_checkpointing 
    # --flash_attn \

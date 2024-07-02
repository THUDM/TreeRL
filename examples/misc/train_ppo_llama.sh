set -x 

DATA_DIR="/workspace/zhenyu/data/hfdata_json/prompts"

read -r -d '' training_commands <<EOF
train_ppo.py \
    --pretrain /workspace/zhenyu/checkpoints/llama/Llama-2-7b-chat-hf \
    --reward_pretrain /workspace/zhenyu/checkpoints/openrlhf/7b_llama_reward \
    --save_path /workspace/zhenyu/checkpoints/openrlhf/7b_llama_ppo \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
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
    --flash_attn \
    --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS]
    # --prompt_data $DATA_DIR/Anthropic/hh-rlhf/train.jsonl,$DATA_DIR/tasksource/oasst1_pairwise_rlhf_reward/train.jsonl,$DATA_DIR/openai/webgpt_comparisons/train.jsonl \


if [[ ${1} != "slurm" ]]; then
    # export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi

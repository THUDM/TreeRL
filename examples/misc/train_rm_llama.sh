set -x 

DATA_DIR="/workspace/zhenyu/data/hfdata_json"

read -r -d '' training_commands <<EOF
train_rm.py \
     --save_path ./ckpt/7b_llama \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 512 \
     --micro_train_batch_size 4 \
     --pretrain /workspace/zhenyu/checkpoints/llama/Llama-2-7b-chat-hf \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset /workspace/zhenyu/data/pairwise_data/reward_data/reward_pair_1226/reward_pair_1226.jsonl,/workspace/zhenyu/data/subjects/math/math_dpo_1117_from_yifan.jsonl \
     --dataset_probs 0.95,0.05 \
     --flash_attn \
     --gradient_checkpointing
EOF
     # --wandb [WANDB_TOKENS]
     # --dataset Anthropic/hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward,openai/webgpt_comparisons \


# if [[ ${1} != "slurm" ]]; then
#     export PATH=$HOME/.local/bin/:$PATH
# deepspeed $training_commands

torchrun --nproc_per_node $MLP_GPU --master_addr $MLP_WORKER_0_HOST --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM $training_commands
# fi

     # --dataset $DATA_DIR/Anthropic/hh-rlhf/train.jsonl,$DATA_DIR/tasksource/oasst1_pairwise_rlhf_reward/train.jsonl,$DATA_DIR/openai/webgpt_comparisons/train.jsonl \

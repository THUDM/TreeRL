set -x 

read -r -d '' training_commands <<EOF
/workspace/ddn/openrlhf-glm/train_rm.py \
    --pretrain /workspace/ddn/modelscope/ZhipuAI/glm-4-9b-chat \
    --model_type reward_mix \
    --dataset /workspace/ddn/data/MATH_train/trainset/MATH_shepherd_LLAMA450k_0818/train_format_v2.jsonl \
    --dataset_probs 1 \
    --prompt_key prompt \
    --label_key labels \
    --save_path /workspace/ddn/models/glm_9B_rw_mix_MATH_shepherd_LLAMA450k \
    --logging_steps 1 \
    --max_epochs 1 \
    --train_batch_size 256 \
    --micro_train_batch_size 1 \
    --bf16 \
    --max_len 2048 \
    --zero_stage 2 \
    --learning_rate 1e-6 \
    --dataset_probs 1 \
    --gradient_checkpointing \
    --use_wandb 92294210a64bba75fe1a28448a625c4410321a0f \
    --mix_supervision
EOF

#if [[ ${1} != "slurm" ]]; then
#    deepspeed $training_commands
#fi

if [[ ${1} != "slurm" ]]; then
    deepspeed --include="localhost:0,1" $training_commands
fi

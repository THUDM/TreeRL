set -x 

# DATA_DIR="/workspace/zhenyu/data/hfdata_json"

# export PATH=$HOME/.local/bin/:$PATH

script="train_rm.py"
TAG=glm4-32b-general-prm-v1

DATA_DIR="/workspace/zhenyu/data/pairwise_data/reward_data"
# --pretrain /workspace/zhenyu/checkpoints/llama/Llama-2-7b-chat-hf/ \

# DATASET=$DATA_DIR/reward_pair0218/foropenrlhf/pair_0128.jsonl,$DATA_DIR/subjects/tiku_zhipu_8_2_deduped_dpo.jsonl,$DATA_DIR/subjects/afanti_sft/afanti_and_sft_0504.jsonl,$DATA_DIR/subjects/chatglm130-240104-mid-full-22k.jsonl,$DATA_DIR/subjects/ultraact.jsonl

# DATASET_RATIO=1,0.27,0.10,0.05,0.1
     # --dataset_probs $DATASET_RATIO \

DATASET="
     $DATA_DIR/0613/20240613v2-rm_path_used_0629.jsonl,1
"


ARGS="
     --save_path /workspace/zhenyu/checkpoints/openrlhf/reward/$TAG \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 512 \
     --pretrain /workspace/zhenyu/checkpoints/32b/sft/32b-sft-0527-v16/iter_0016000/chatglm-hf-rm \
     --micro_train_batch_size 2 \
     --bf16 \
     --max_epochs 1 \
     --max_len 8192 \
     --zero_stage 3 \
     --learning_rate 5e-6 \
     --dataset $DATASET \
     --max_sample 119258 \
     --gradient_checkpointing \
     --prompt_key prompt \
     --response_key response \
     --label_key labels \
     --flash_attn \
     --wandb_run_name $TAG \
     --use_wandb think2try-glm \
     --process_supervision \
     --loss pointmse \
"

     # --- rm v5 ---
     # --dataset $DATA_DIR/reward_pair0218/foropenrlhf/pair_0128.jsonl,$DATA_DIR/subjects/tiku_zhipu_8_2_dpo_release.jsonl,$DATA_DIR/subjects/chatglm130-240104-mid-full-22k.jsonl \
     # --dataset_probs 0.8,0.15,0.05 \

     # --dataset Anthropic/hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward,openai/webgpt_comparisons \
     # --pretrain /workspace/zhenyu/checkpoints/chatglm3-32b \



# deepspeed $scripts $ARGS
# torchrun --nproc_per_node $MLP_GPU --master_addr $MLP_WORKER_0_HOST --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM $script $ARGS

# | tee logs/${MLP_TASK_ID}/${MLP_ROLE_INDEX}.log


# if [[ ${1} != "slurm" ]]; then
# #     export PATH=$HOME/.local/bin/:$PATH
#     command="deepspeed $scripts $ARGS"
#     mkdir logs
#     eval ${command} 2>&1 | tee logs/output.log
# fi

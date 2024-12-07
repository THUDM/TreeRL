set -x 

# DATA_DIR="/workspace/zhenyu/data/hfdata_json"

# export PATH=$HOME/.local/bin/:$PATH

script="train_rm.py"
TAG=glm4-32b-general-prm-v1

DATA_DIR="/workspace/lurui/openrlhf-glm/datasets"

# DATASET=$DATA_DIR/reward_pair0218/foropenrlhf/pair_0128.jsonl,$DATA_DIR/subjects/tiku_zhipu_8_2_deduped_dpo.jsonl,$DATA_DIR/subjects/afanti_sft/afanti_and_sft_0504.jsonl,$DATA_DIR/subjects/chatglm130-240104-mid-full-22k.jsonl,$DATA_DIR/subjects/ultraact.jsonl

# DATASET_RATIO=1,0.27,0.10,0.05,0.1
     # --dataset_probs $DATASET_RATIO \

DATASET="
     $DATA_DIR/general_test.jsonl,1
"


ARGS="
     --save_path /workspace/zhenyu/checkpoints/openrlhf/reward/$TAG \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 512 \
     --pretrain "/data/share/checkpoint/glm-4-9b-chat" \
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
# pip install deepspeed==0.15.2,openai,ray
LOCAL_RANK=0 python $script $ARGS

set -x 

# DATA_DIR="/workspace/zhenyu/data/hfdata_json"

# export PATH=$HOME/.local/bin/:$PATH

script="train_rm.py"
TAG=glm4-32b-general-prm-v1

DATA_DIR="data/pairwise_data/reward_data"


DATASET="
     $DATA_DIR/0613/general-prm.jsonl,1
"

ARGS="
     --save_path checkpoints/openrlhf/reward/$TAG \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 512 \
     --pretrain chatglm-hf-rm \
     --micro_train_batch_size 2 \
     --bf16 \
     --max_epochs 1 \
     --max_len 8192 \
     --zero_stage 3 \
     --learning_rate 1e-6 \
     --dataset $DATASET \
     --max_sample 10000 \
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
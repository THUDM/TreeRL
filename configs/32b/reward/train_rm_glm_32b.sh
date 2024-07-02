set -x 

# DATA_DIR="/workspace/zhenyu/data/hfdata_json"

# export PATH=$HOME/.local/bin/:$PATH

script="train_rm.py"
TAG=glm4-32b-rm-v2-point

DATA_DIR="data/pairwise_data/reward_data"

DATASET="
     $DATA_DIR/reward_pair0218/foropenrlhf/pair_0128.jsonl,1
"


ARGS="
     --save_path reward/$TAG \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 512 \
     --pretrain chatglm-hf \
     --micro_train_batch_size 2 \
     --bf16 \
     --max_epochs 1 \
     --max_len 8192 \
     --zero_stage 3 \
     --learning_rate 5e-6 \
     --dataset $DATASET \
     --max_sample 10000 \
     --gradient_checkpointing \
     --prompt_key prompt \
     --chosen_key response_chosen \
     --rejected_key response_rejected \
     --flash_attn \
     --wandb_run_name $TAG \
     --use_wandb think2try \
"
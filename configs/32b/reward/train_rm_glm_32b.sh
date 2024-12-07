set -x 

# DATA_DIR="/workspace/zhenyu/data/hfdata_json"

# export PATH=$HOME/.local/bin/:$PATH

script="train_rm.py"
TAG=1030_9w

DATA_DIR="/workspace/lurui/openrlhf-glm/datasets"


DATASET="
     $DATA_DIR/1030_rmdata.jsonl,1
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
     --max_sample 1000000 \
     --gradient_checkpointing \
     --prompt_key prompt \
     --chosen_key response_chosen \
     --rejected_key response_rejected \
     --flash_attn \
     --wandb_run_name $TAG \
     --use_wandb think2try \
"
pip install deepspeed==0.15.2 openai ray
# LOCAL_RANK=0 python $script $ARGS
LOCAL_RANK=0 deepspeed --num_gpus=8 train_rm.py $ARGS

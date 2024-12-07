      
# pkill -f -9 'import spawn_main'

# # export CHECKPOINT_NAME="RLOO-9B-ms16-golden-kl-0.004-code-contest-1115-v1-resample"
# export CHECKPOINT_NAME=$1

# python3 auto_eval.py \
#     --base_dir /workspace/lurui/openrlhf-glm/checkpoints/reinforce \
#     --checkpoint_dirs $CHECKPOINT_NAME \

#!/bin/bash

# 设置间隔时间，半小时
INTERVAL=1800

# 设置CHECKPOINT_NAME
CHECKPOINT_NAME=$1

# 无限循环
while true; do
    # 杀死现有的 `auto_eval.py` 进程
    pkill -f -9 python3

    # 等待一小段时间确保进程已被终止
    sleep 5

    # 运行 `auto_eval.py`
    python3 auto_eval.py \
        --base_dir /workspace/lurui/openrlhf-glm/checkpoints/reinforce \
        --checkpoint_dirs $CHECKPOINT_NAME &

    # 等待指定的间隔时间
    sleep $INTERVAL
    echo "Restarting auto_eval.py..."
done
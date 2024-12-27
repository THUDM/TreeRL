      
# -------- parameter --------
export MODEL_PATH=$1
export MODEL_NAME=$2
# -------- parameter --------

source /workspace/lurui/miniconda3/etc/profile.d/conda.sh 
conda activate vllm

echo "up model"
for i in {0..7}
do
    export CUDA_VISIBLE_DEVICES=$i
    # 计算端口号
    port=$((8000 + i))
    # 在后台启动服务
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --served-model-name $MODEL_NAME \
        --max-model-len 8192 \
        --trust-remote-code \
        --host=0.0.0.0 --port=$port \
        --tensor-parallel-size 1 \
        --enforce-eager &
    sleep 1
done
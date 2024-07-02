source $1
# OPTIONS="LD_LIBRARY_PATH=/workspace/zhenyu/code/nccl/build/lib:$LD_LIBRARY_PATH"

# run_cmd="
torchrun --nproc_per_node $MLP_GPU --master_addr $MLP_WORKER_0_HOST --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM $script $ARGS


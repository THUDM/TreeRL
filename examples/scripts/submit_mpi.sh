#! /bin/bash

set -o pipefail

source $1

mkdir -p logs/${MLP_TASK_ID}

mpi_cmd="mpirun -np ${MLP_WORKER_NUM} \
        --hostfile ${MLP_MPI_HOSTFILE} \
        -npernode 1 -x OMPI_MCA_btl_tcp_if_include=$MLP_SOCKET_IFNAME \
        pip install datasets deepspeed einops jsonlines tqdm transformers accelerate wandb tiktoken mpi4py ray openai"

#  ray[default]

eval ${mpi_cmd}


if [[ -z "$NCCL_IB_QPS_PER_CONNECTION" ]]; then
  NCCL_IB_QPS_PER_CONNECTION=2
fi

echo "Start training"

export NCCL_IB_ENVS="-x NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1,mlx5_8:1,mlx5_10:1,mlx5_12:1,mlx5_14:1"


mpi_cmd="mpirun -np $((MLP_WORKER_NUM * MLP_GPU)) \
        --hostfile ${MLP_MPI_HOSTFILE} \
        --allow-run-as-root -oversubscribe -map-by ppr:8:node \
        -mca btl ^openib -x OMPI_MCA_btl_tcp_if_include=$MLP_SOCKET_IFNAME \
        --output-filename logs/${MLP_TASK_ID} \
        -x NCCL_PXN_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_NET_GDR_LEVEL=4 \
        -x NCCL_IB_RETRY_CNT=7 \
        -x NCCL_IB_TIMEOUT=50 \
        -x NCCL_IB_QPS_PER_CONNECTION=$NCCL_IB_QPS_PER_CONNECTION \
        -x NCCL_P2P_LEVEL=NVL \
        -x NCCL_DEBUG=VERSION \
        -x NCCL_IB_TC=106 \
        -x PATH \
        -x MASTER_ADDR=$MLP_WORKER_0_HOST \
        -x MASTER_PORT=$MLP_WORKER_0_PORT \
        -x GLOO_SOCKET_IFNAME=$MLP_SOCKET_IFNAME \
        -x NCCL_SOCKET_IFNAME=$MLP_SOCKET_IFNAME \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -x NCCL_NVLS_ENABLE=0 \
        -x NVTE_FWD_LAYERNORM_SM_MARGIN=8 \
        -x NVTE_BWD_LAYERNORM_SM_MARGIN=8 \
        -x LD_LIBRARY_PATH=$NCCL_LIBRARY_PATH:$LD_LIBRARY_PATH \
        -x NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1,mlx5_8:1,mlx5_10:1,mlx5_12:1,mlx5_14:1 \
        -x UCX_NET_DEVICES=mlx5_0:1 \
        python ${script} --use_mpi_init ${ARGS}"

        # -x LD_LIBRARY_PATH=$NCCL_LIBRARY_PATH:$LD_LIBRARY_PATH \


# mpi_cmd="mpirun -np $((MLP_WORKER_NUM * MLP_GPU)) \
#         --hostfile ${MLP_MPI_HOSTFILE} \
#         --allow-run-as-root -oversubscribe -map-by ppr:8:node \
#         -mca btl ^openib -x OMPI_MCA_btl_tcp_if_include=$MLP_SOCKET_IFNAME \
#         --output-filename ${MAIN_DIR}/logs \
#         -x NCCL_PXN_DISABLE=0 \
#         -x NCCL_IB_GID_INDEX=3 \
#         -x NCCL_NET_GDR_LEVEL=4 \
#         -x NCCL_IB_RETRY_CNT=7 \
#         -x NCCL_IB_TIMEOUT=32 \
#         -x NCCL_IB_QPS_PER_CONNECTION=$NCCL_IB_QPS_PER_CONNECTION \
#         -x NCCL_P2P_LEVEL=NVL \
#         -x NCCL_DEBUG=VERSION \
#         -x PATH \
#         -x MASTER_ADDR=$MLP_WORKER_0_HOST \
#         -x MASTER_PORT=$MLP_WORKER_0_PORT \
#         -x GLOO_SOCKET_IFNAME=$MLP_SOCKET_IFNAME \
#         -x NCCL_SOCKET_IFNAME=$MLP_SOCKET_IFNAME \
#         -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
#         -x NCCL_NVLS_ENABLE=0 \
#         -x NVTE_FWD_LAYERNORM_SM_MARGIN=8 \
#         -x NVTE_BWD_LAYERNORM_SM_MARGIN=8 \
#         -x LD_LIBRARY_PATH=/workspace/zhenyu/code/nccl/build/lib:$LD_LIBRARY_PATH \
#         python ${script} --use_mpi_init ${ARGS}"

echo ${mpi_cmd}
eval ${mpi_cmd} 2>&1 | tee logs/${MLP_TASK_ID}/output.log


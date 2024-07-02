# cd /workspace/zhenyu/code/OpenRLHF/examples/scripts
# bash build_openrlhf.sh


cd /workspace/zhenyu/code/OpenRLHF

hostip=$(env | grep MLP_HOST=)
hostip=${hostip#*=}
echo $hostip


pkill -9 ray
pkill -9 python
pkill -9 gcs_server

# export LD_LIBRARY_PATH=/workspace/zhenyu/code/nccl/build/lib:$LD_LIBRARY_PATH


pip install -e .
# pip install transformers==4.38.2

# cp /workspace/zhenyu/code/nccl/build/lib/libnccl.so.2.19.4 /root/.config/vllm/nccl/cu12/
# rm /root/.config/vllm/nccl/cu12/libnccl.so.2.18.1

if [ $hostip  = $MLP_WORKER_0_HOST ]; then
ray start --head --node-ip-address=$MLP_WORKER_0_HOST --port=6379 --block &
else
sleep 50
ray start --address $MLP_WORKER_0_HOST:6379 --block &
fi


# echo $1

# sleep 300
# cd /workspace/zhenyu/code/OpenRLHF

# if [ $MLP_ROLE_INDEX  = "0" ]; then
#     script=$1
#     bash $script
# else
#     sleep inf
# fi



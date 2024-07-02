# KEYS=("response_chosen", "response_rejected")

model_path=$1
# dataset_name=$2
source_file=$2


if [ -z "$dataset_name" ]; then
    dataset_name="mathbench_0312_dev"
    echo "dataset_name is not provided, use default dataset: ${dataset_name}"
fi


output_path="${source_file}.result"

# source_file=/workspace/zhenyu/data/evaluation/basebench/${dataset_name}/questions_code_v3_0303.jsonl
# judge_file="${model_path}/${dataset_name}_best_32_bon.jsonl"


# deepspeed --master_port 12345 --num_nodes 1 --num_gpus 8 

torchrun --nproc_per_node $MLP_GPU --master_addr $MLP_WORKER_0_HOST --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM  tools/reward_inference.py \
--eval_task rm \
--zero_stage 3 \
--bf16 \
--input_key prompt \
--output_key response \
--max_samples 3000000 \
--micro_batch_size 1 \
--prompt_max_len 4096 \
--max_len 8190 \
--tp_size 4 \
--model_path $model_path \
--dataset $source_file \
--dataset_probs 1 \
--output_path $output_path

# cd /workspace/zhenyu/code/OpenRLHF

# hostip=$(env | grep MLP_HOST=)
# hostip=${hostip#*=}
# echo $hostip

# if [ $hostip  = $MLP_WORKER_0_HOST ]; then
#     python tools/judge_mr_bench.py --input_file $judge_file --reference_file $source_file
# fi


# bash examples/scripts/reward_inf.sh /workspace/zhenyu/checkpoints/openrlhf/reward/32b_chatglm_rmv6 mathbench_0312_dev
# bash examples/scripts/reward_inf.sh /workspace/zhenyu/checkpoints/openrlhf/reward/glm-cv3-130b-rm-v6 mathbench_0312_dev
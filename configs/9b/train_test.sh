set -x 
export PATH=$HOME/.local/bin/:$PATH

# ray start --head --node-ip-address=192.168.246.53 --port=6379 --block &
# ray start --address 192.168.246.53:6379 --block &

# pkill -9 ray
# pkill -9 python
# pkill -9 gcs_server
# sleep 5

# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

NUM_TRACE=2
KL=0.04

DATA_DIR=/workspace/zhenyu/data/prompt_data
TAG=glm-9b-reinforce-v15-ms${NUM_TRACE}_rm32-v1-approx-kl-${KL}-run4
SAVE_DIR=/workspace/zhenyu/checkpoints/openrlhf/reinforce/$TAG
mkdir -p $SAVE_DIR

# export LD_LIBRARY_PATH=/workspace/zhenyu/code/nccl/build/lib:$LD_LIBRARY_PATH
# "LD_LIBRARY_PATH": "/workspace/zhenyu/code/nccl/build/lib:${LD_LIBRARY_PATH}", 

    # --runtime-env-json='{"working_dir": "/workspace/zhenyu/code/OpenRLHF", "pip": "/workspace/zhenyu/code/OpenRLHF/requirements.txt", "excludes": ["*.ipynb", "tests", "logs", "wandb"], "env_vars": {"LD_LIBRARY_PATH": "/workspace/zhenyu/code/nccl/build/lib:/usr/local/cuda/compat/lib.real:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"}}' \

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/workspace/zhenyu/code/OpenRLHF", "pip": "/workspace/zhenyu/code/OpenRLHF/requirements.txt", "excludes": ["*.ipynb", "tests", "logs", "wandb"]}' \
    -- python train_reinforce_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 2 \
    --pretrain /workspace/zhenyu/checkpoints/9b/sft/9b-sft-0526-v15/iter_0016000/chatglm-hf \
    --reward_pretrain /workspace/zhenyu/checkpoints/openrlhf/reward/glm4-32b-rm-v1-point/ \
    --save_path $SAVE_DIR \
    --ckpt_path $SAVE_DIR \
    --micro_train_batch_size 1 \
    --train_batch_size $((NUM_TRACE * 32)) \
    --micro_rollout_batch_size 4 \
    --roll_out_batch_size_multiplier 2 \
    --rollout_batch_size 128 \
    --max_epochs 1 \
    --max_samples 100000 \
    --prompt_max_len 3072 \
    --generate_max_len 4096 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --min_actor_learning_rate_lr 1 \
    --l2 0.1 \
    --init_kl_coef $KL \
    --prompt_data $DATA_DIR/0103_prompts/0103_prompts_dpo.jsonl,$DATA_DIR/prompt_0419/prompt_20240419_processed.jsonl,$DATA_DIR/others/math/tiku_all_part_0416_deduped_pair.jsonl \
    --prompt_data_probs 0.6,0.1,0.3 \
    --normalize_reward \
    --normalize_reward_from_multi_traces \
    --top_p 0.9 \
    --temperature 0.95 \
    --actor_init_on_gpu \
    --num_trace_per_sample $NUM_TRACE \
    --gradient_checkpointing \
    --input_key prompt \
    --save_steps 20 \
    --perf \
    --load_ckpt \
    --wandb_run_name $TAG \
    --use_wandb think2try \
    --task_type 9b-glm4-ppo \

    # --use_wandb think2try-glm \
    # --max_min_reward_samples \
    # --adam_offload \
    # --save_steps 40 \
    # --normalize_advantage \
    # --pretrain_data "" \
    # --ptx_coef 0.005 \
    # --wandb_run_name 32b_chatglm_ppo_test \
    # --flash_attn \

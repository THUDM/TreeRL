set -x 
# export PATH=$HOME/.local/bin/:$PATH

# ray start --head --node-ip-address=192.168.246.53 --port=6379 --block &
# ray start --address 192.168.246.53:6379 --block &

# pkill -9 ray
# pkill -9 python
# pkill -9 gcs_server
# sleep 5

# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

NUM_TRACE=16
KL=0.0001

TAG=RLOO-glm9b-o1sft-model-ms${NUM_TRACE}-kl-${KL}-math-chain-5epoch
SAVE_DIR=/workspace/lurui/openrlhf-glm/checkpoints/reinforce/$TAG
mkdir -p $SAVE_DIR

# /workspace/zhenyu/data/research_training/math_data/numina_used_1019.jsonl,1
# /workspace/lurui/openrlhf-glm/data/code_contest_rlhf.jsonl,1
# /workspace/lurui/openrlhf-glm/data/code_contest_3856_1117.jsonl

# DATASETS="
#     /data/share/openrlhf_data/math_1030_v1.jsonl,1
# "

DATASETS="
    /workspace/lurui/openrlhf-glm-data/train_30k.jsonl,1
"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
        "working_dir": "/workspace/lurui/openrlhf-glm",
        "pip": "/workspace/lurui/openrlhf-glm/requirements.txt",
        "excludes": ["*.ipynb", "tests", "log", "logs", "wandb", "checkpoints","reward",".git"],
        "env_vars": {
            "NCCL_TIMEOUT_MS": "3600000"
        }
    }' \
    -- python train_reinforce_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 8 \
    --actor_num_nodes 2 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --pretrain /workspace/lurui/glm-train_data/checkpoints/9b-sft-o1-mini-part-1212/hf_0000381 \
    --reward_pretrain /workspace/lurui/glm-train_data/checkpoints/9b-sft-o1-mini-part-1212/hf_0000381 \
    --save_path $SAVE_DIR \
    --ckpt_path $SAVE_DIR \
    --micro_train_batch_size 1 \
    --train_batch_size $((NUM_TRACE * 16)) \
    --micro_rollout_batch_size 1 \
    --generation_batch_size 16 \
    --inference_batch_size 1 \
    --rollout_batch_size 16 \
    --num_episodes 2 \
    --prompt_max_len 1024 \
    --generate_max_len 8192 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 1.5e-6 \
    --lr_scheduler_type cosine \
    --min_actor_learning_rate_lr 1 \
    --l2 0.1 \
    --init_kl_coef $KL \
    --prompt_data $DATASETS \
    --max_samples 300000 \
    --normalize_reward \
    --top_p 0.95 \
    --temperature 1.2 \
    --actor_init_on_gpu \
    --num_trace_per_sample $NUM_TRACE \
    --gradient_checkpointing \
    --input_key text \
    --save_steps 10 \
    --perf \
    --wandb_run_name $TAG \
    --use_wandb lyj \
    --task_type 9b-glm4-reinforce \
    --label_key label \
    --source_key data_type \
    --remote_rm_url /workspace/lurui/openrlhf-glm/examples/tools/rm_urls.json \
    --normalize_reward_from_multi_traces_with_rloo \
    --wandb_project openrlhf_code_rl \
    # --use_general_reward_for_stem \
    # --use_rule_based_reward \
    # --mask_repeated_samples \
    # --random_temperature \
    # --use_rule_based_reward \
    # --use_general_reward_for_stem \
    # --mask_pass_confident_samples \

    # tools/remote_model_rms_qwen.txt 

    # --use_wandb think2try-glm \
    # --max_min_reward_samples \
    # --adam_offload \
    # --save_steps 40 \
    # --normalize_advantage \
    # --pretrain_data "" \
    # --ptx_coef 0.005 \
    # --wandb_run_name 32b_chatglm_ppo_test \
    # --flash_attn \

    

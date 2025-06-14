set -x 

NUM_TRACE=16
KL=0.0

TAG=qwen14b-ms${NUM_TRACE}
SAVE_DIR=/workspace/lurui/rl_checkpoints/checkpoints/reinforce/$TAG
mkdir -p $SAVE_DIR

DATASETS="
    datasets/train/train_30k.jsonl,1
"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
        "working_dir": treerl/openrlhf",
        "pip": treerl/requirements.txt",
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
    --vllm_num_engines 16 \
    --vllm_tensor_parallel_size 1 \
    --pretrain qwen-14b-2.5-instruct \
    --reward_pretrain qwen-14b-2.5-instruct \
    --save_path $SAVE_DIR \
    --ckpt_path $SAVE_DIR \
    --micro_train_batch_size 1 \
    --train_batch_size $((NUM_TRACE * 16)) \
    --micro_rollout_batch_size 1 \
    --generation_batch_size 32 \
    --inference_batch_size 1 \
    --rollout_batch_size 16 \
    --num_episodes 2 \
    --prompt_max_len 1024 \
    --generate_max_len 8192 \
    --zero_stage 3 \
    --adam_offload \
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
    --task_type qwen-math-reinforce \
    --label_key label \
    --source_key data_type \
    --remote_rm_url ./remote_rm_urls.json \
    --normalize_reward_from_multi_traces_with_rloo \
    --wandb_project openrlhf_math_mcts \
    --use_mcts \
    --process_supervision \
    --mask_repeated_samples \
    --max_nodes 256 \
    --max_node_per_depth 18 \
    --max_time_use 480 \
    --random_pick \
    --use_state_value_reward \
    --use_pure_binary \
    --select_correct_leaf \
    --first_token_temperature 1 \
    --use_entropy_tree \
    --m 6 \
    --n 2 \
    --l 1 \
    --t 2 \
    --binary_judge_url "http://172.20.68.200:8000/v1" \
    --extractor_url "http://172.20.69.62:8000/v1" \
    --reward_model_url "http://172.20.68.44:8000/v1" \
    --correct_bonus_ratio 1 \
    --a_coeff 0.515 \
    --b_mean 0.221 \
    --correct_bonus_threshold 0 \
    --gradient_checkpointing \
    --gradient_checkpointing_use_reentrant \
    --adam_offload \
    --wandb_id $TAG \
    --resume \
    --use_weighted_value \
    --weighted_value_style sqrt \

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
    # --save_steps 40 \
    # --normalize_advantage \
    # --pretrain_data "" \
    # --ptx_coef 0.005 \
    # --wandb_run_name 32b_chatglm_ppo_test \
    # --flash_attn \
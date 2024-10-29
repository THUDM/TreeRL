set -x 

NUM_TRACE=16
KL=0.004

DATA_DIR=/data
TAG=RLOO-9B-ms${NUM_TRACE}-golden-kl-${KL}-1025-v2-rerun
SAVE_DIR=/checkpoints/reason/rl/9b/reinforce/$TAG
mkdir -p $SAVE_DIR

    # /workspace/zhenyu/data/research_training/math_data/numina_used_1019.jsonl,1

DATASETS="
    /workspace/zhenyu/data/research_training/math_data/math-train.jsonl,0.15
"


ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/workspace/zhenyu/code/OpenRLHF", "pip": "/workspace/zhenyu/code/OpenRLHF/requirements.txt", "excludes": ["*.ipynb", "tests", "logs", "wandb"]}' \
    -- python train_reinforce_ray.py \
    --ref_num_nodes 8 \
    --ref_num_gpus_per_node 8 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 8 \
    --actor_num_nodes 8 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 64 \
    --vllm_tensor_parallel_size 1 \
    --pretrain /checkpoints/reason/sft/9b/9b-sft-format-1022/iter_0001500/chatglm_hf \
    --reward_pretrain /checkpoints/reason/sft/9b/9b-sft-format-1022/iter_0001500/chatglm_hf \
    --save_path $SAVE_DIR \
    --ckpt_path $SAVE_DIR \
    --micro_train_batch_size 1 \
    --train_batch_size $((NUM_TRACE * 64)) \
    --micro_rollout_batch_size 4 \
    --generation_batch_size 32 \
    --inference_batch_size 1 \
    --rollout_batch_size 256 \
    --num_episodes 2 \
    --prompt_max_len 1024 \
    --generate_max_len 12000 \
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
    --input_key prompt \
    --save_steps 5 \
    --perf \
    --wandb_run_name $TAG \
    --use_wandb think2try-glm \
    --task_type 9b-glm4-reinforce \
    --remote_rm_url tools/remote_rm_urls.txt \
    --label_key label \
    --normalize_reward_from_multi_traces_with_rloo \
    --mask_repeated_samples \
    --use_random_top_k_logits_sampling \
    --min_p 0.05 \
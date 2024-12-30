      
# export WANDB_API_KEY=local-515145080efc923cf7bd1427ed76cda1c83a15c5
# export WANDB_BASE_URL=https://wandb.glm.ai
# export WANDB_ENTITY=glm-zero


# EVAL_DIR=/workspace/lurui/rm_simple_evals/RL_auto_results
EVAL_DIR=/workspace/lurui/rm_simple_evals/RL_auto_results
# example 
# TUNED_MODEL_NAME=ms32-mcts-advantage-value-binary-wototalnorm
# TUNED_MODEL_NAME=ms32-mcts-binary-chainonly-firsttokentemp
TUNED_MODEL_NAME=RLOO-glm9b-o1sft-model-ms32-kl-0.0001-math-mcts-advantage-value-binary-wototalnorm

# TUNED_MODEL_NAME=$1

python auto_upload.py \
    --base_path $EVAL_DIR/$TUNED_MODEL_NAME \
    --project_name openrlhf_math_mcts \
    --run_name $TUNED_MODEL_NAME-eval \
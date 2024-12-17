      
# export WANDB_API_KEY=local-515145080efc923cf7bd1427ed76cda1c83a15c5
# export WANDB_BASE_URL=https://wandb.glm.ai
# export WANDB_ENTITY=glm-zero


EVAL_DIR=/workspace/lurui/rm_simple_evals/RL_auto_results
# example 
TUNED_MODEL_NAME=RLOO-glm9b-o1sft-model-ms16-kl-0.0001-math-mcts-advantage-plus-orm-selectcorrect
# TUNED_MODEL_NAME=$1

python auto_upload.py \
    --base_path $EVAL_DIR/$TUNED_MODEL_NAME \
    --project_name openrlhf_code_rl \
    --run_name $TUNED_MODEL_NAME \
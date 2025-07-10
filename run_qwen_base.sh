export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="huggingface"
export WANDB_NAME="Qwen2.5-1.5B-Base-GRPO-Thinking"

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Base/grpo/config_demo.yaml \
    --vllm_mode colocate
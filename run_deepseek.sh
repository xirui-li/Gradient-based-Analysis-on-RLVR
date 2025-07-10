export CUDA_VISIBLE_DEVICES=5,6
export WANDB_PROJECT="huggingface"
export WANDB_NAME="DeepSeek-R1-Distill-Qwen-1.5B-GRPO-Thinking"

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
    --vllm_mode colocate
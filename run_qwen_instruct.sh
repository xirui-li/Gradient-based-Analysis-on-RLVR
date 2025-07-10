export CUDA_VISIBLE_DEVICES=2,3,4
export WANDB_PROJECT="huggingface"
export WANDB_NAME="Qwen2.5-Math-1.5B-GRPO-Thinking"

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
    --vllm_mode colocate
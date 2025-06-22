export CUDA_VISIBLE_DEVICES=5,6
export WANDB_NAME="Qwen2.5-Math-1.5B-GRPO-Nothinking"

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo.py --config recipes/Qwen2.5-Math-1.5B/grpo/config_demo.yaml \
    --vllm_mode colocate
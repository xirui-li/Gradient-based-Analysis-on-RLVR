export CUDA_VISIBLE_DEVICES=5,6
export WANDB_NAME="Qwen2.5-1.5B-Instruct-SFT"

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
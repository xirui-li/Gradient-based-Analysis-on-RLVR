export CUDA_VISIBLE_DEVICES=5,6
export WANDB_NAME="Qwen2.5-1.5B-Instruct-Filter"

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3_debug.yaml \
    src/open_r1/filter.py --config recipes/Qwen2.5-1.5B-Instruct/sft/config_gsm8k.yaml
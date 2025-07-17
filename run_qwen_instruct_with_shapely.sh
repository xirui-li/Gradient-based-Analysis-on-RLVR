#!/bin/bash
#SBATCH --job-name=qwen_grpo
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodelist=vulcan35
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load conda manually
# Load appropriate modules
# source /nfshomes/xiruili/anaconda3/etc/profile.d/conda.sh
# conda activate gradient-analysis

# Environment variables
export WANDB_API_KEY="7b8ef784250fec92ca3bf5f34c5c04834b9ec7c4" 
export WANDB_PROJECT="huggingface"
export WANDB_NAME="Qwen2.5-Math-1.5B-GRPO-Thinking"
export CUDA_VISIBLE_DEVICES=5,6

# Launch training
ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_shapley.yaml \
    --vllm_mode colocate

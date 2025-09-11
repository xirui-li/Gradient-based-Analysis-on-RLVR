# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path open-r1/Qwen2.5-Math-7B-RoPE-300k \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --eos_token '<|im_end|>' \
    --learning_rate 4.0e-5 \
    --num_train_epochs 5 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/OpenR1-Distill-7B
"""

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format
from open_r1.trainer import SFTTrainer_Filter


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    dataset = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")
    
    #     # Format into conversation
    # def make_conversation(example):
    #     prompt = []

    #     # if training_args.system_prompt is not None:
    #         # prompt.append({"role": "system", "content": training_args.system_prompt})
    #     prompt.append({"role": "user", "content": example["problem"]})
    #     return {"prompt": prompt, "completion": "The answer is " + example["answer"] + "."}

    # dataset = dataset.map(make_conversation)

    ############################
    # Initialize the SFT Trainer
    ############################
    # Modify training args for assessment only (keep existing settings where possible)
    # Create a copy of training args for assessment
    train_dataset = dataset['train']
    train_dataset = train_dataset.select(range(1000))

    from copy import deepcopy
    assessment_args = deepcopy(training_args)
    assessment_args.learning_rate = 0.0  # No actual training
    assessment_args.num_train_epochs = 1  # Only need one pass
    assessment_args.per_device_train_batch_size = 1  # Process one sample at a time
    assessment_args.gradient_accumulation_steps = 1
    assessment_args.save_steps = 99999  # Don't save checkpoints
    assessment_args.save_total_limit = 0
    assessment_args.evaluation_strategy = "no"
    assessment_args.logging_steps = 100
    assessment_args.output_dir = os.path.join(training_args.output_dir, "data_quality_assessment")
    assessment_args.report_to = []  # Disable wandb for assessment
    assessment_args.push_to_hub = False  # Don't push assessment runs
    
    # Format dataset for SFTTrainer
    def format_for_sft(examples):
        """Format GSM8K examples using the chat template."""
        formatted_texts = []
        
        # Process each example in the batch
        for i in range(len(examples['question'])):
            # Create a conversation format
            messages = [
                {"role": "user", "content": examples['question'][i]},
                {"role": "assistant", "content": examples['answer'][i]}
            ]
            # Apply chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
        
        return {"text": formatted_texts}
    
    # Apply formatting and remove original columns
    logger.info("Formatting dataset with chat template...")
    formatted_dataset = train_dataset.map(
        format_for_sft,
        batched=True,
        remove_columns=train_dataset.column_names,  # Remove 'question' and 'answer'
        desc="Formatting dataset"
    )
    
    def add_sample_ids(examples, indices):
        # Use the actual indices, not creating new strings
        examples['sample_ids'] = indices  # Just use the indices directly
        return examples

    logger.info("Adding sample IDs...")
    dataset_with_ids = formatted_dataset.map(
        add_sample_ids,
        batched=True,
        with_indices=True,
        desc="Adding sample IDs"
    )

    # Create quality assessment trainer
    logger.info("Initializing data quality assessment trainer...")
    
    # Build trainer kwargs with optional parameters
    trainer_kwargs = {
        "model": model,
        "args": assessment_args,
        "train_dataset": dataset_with_ids,
        "processing_class": tokenizer,
    }
    
    # Add optional parameters if they exist
    if hasattr(model_args, 'use_peft') and model_args.use_peft:
        trainer_kwargs["peft_config"] = get_peft_config(model_args)
    
    if hasattr(script_args, 'use_callbacks') and script_args.use_callbacks:
        trainer_kwargs["callbacks"] = get_callbacks(training_args)
    
    quality_assessor = SFTTrainer_Filter(**trainer_kwargs)

    # Run assessment (looks like training but doesn't update model)
    logger.info("Starting data quality assessment...")
    quality_assessor.train()
    logger.info("Assessment complete. Saving final quality scores...")
    
    # Save the final complete scores
    final_output_path = os.path.join(assessment_args.output_dir, "data_quality_scores_final.json")
    quality_assessor.export_quality_scores(final_output_path)
    
    # Get filtered sample IDs
    filter_percentage = getattr(script_args, 'quality_filter_percentage', 70)  # Default to 70%
    logger.info(f"Filtering dataset to keep top {filter_percentage}% of samples...")
    high_quality_ids = quality_assessor.get_high_quality_sample_ids(top_k_percent=filter_percentage)
    id_to_index = {}
    for i in range(len(train_dataset)):
        # Get the formatted sample to see what ID it would get
        sample = dataset_with_ids[i]
        if 'sample_ids' in sample:
            id_to_index[sample['sample_ids']] = i

    logger.info(f"ID to index mapping created: {len(id_to_index)} entries")


    if high_quality_ids:
        # Convert sample IDs back to indices
        kept_indices = []
        for sid in high_quality_ids:
            if isinstance(sid, str):
                # If it's a string like "sample_0", extract the number
                kept_indices.append(int(sid.split('_')[1]))
            elif isinstance(sid, int):
                # If it's already an int, use it directly
                kept_indices.append(sid)
            else:
                # If it's something else, try to convert
                kept_indices.append(int(sid))
        
        # Check if we have valid indices
        if kept_indices:
            filtered_dataset = train_dataset.select(kept_indices)
            
            logger.info(f"Original dataset size: {len(train_dataset)}")
            logger.info(f"Filtered dataset size: {len(filtered_dataset)}")
            logger.info(f"Removed {len(train_dataset) - len(filtered_dataset)} low-quality samples")
            
            # Save filtered dataset
            filtered_dataset_path = os.path.join(assessment_args.output_dir, "filtered_dataset")
            filtered_dataset.save_to_disk(filtered_dataset_path)
            logger.info(f"Filtered dataset saved to: {filtered_dataset_path}")
            
            # Save filtering metadata
            import json
            metadata = {
                "original_size": len(train_dataset),
                "filtered_size": len(filtered_dataset),
                "kept_indices": kept_indices,
                "filter_percentage": filter_percentage,
                "high_quality_ids": high_quality_ids
            }
            with open(os.path.join(assessment_args.output_dir, "filter_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            logger.warning("No valid indices extracted from high quality IDs")
    else:
        logger.warning("No quality scores computed. Cannot filter dataset.")

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
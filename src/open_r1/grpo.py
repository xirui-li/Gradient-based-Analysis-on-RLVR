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

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, TrlParser, get_peft_config
from open_r1.trainer import GRPOTrainer, GRPOTrainerWithShapely, GRPOTrainerWithNewReward, GRPOTrainerMonitor 


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
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

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
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

    # Load the dataset
    dataset = get_dataset(script_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    if 'math' in script_args.dataset_name.lower():
        def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
            prompt = []

            # if training_args.system_prompt is not None:
                # prompt.append({"role": "system", "content": training_args.system_prompt})

            if prompt_column not in example:
                raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
            if training_args.system_prompt is not None:
                prompt_suffix = training_args.system_prompt
            else:
                prompt_suffix = ""
            prompt.append({"role": "user", "content": example[prompt_column] + " " + prompt_suffix})
            return {"prompt": prompt, "solution": example["answer"]}
    elif 'gsm8k' in script_args.dataset_name.lower():
        def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
            prompt = []

            # if training_args.system_prompt is not None:
                # prompt.append({"role": "system", "content": training_args.system_prompt})

            if prompt_column not in example:
                raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
            if training_args.system_prompt is not None:
                prompt_suffix = training_args.system_prompt
            else:
                prompt_suffix = ""
            prompt.append({"role": "user", "content": example[prompt_column] + " " + prompt_suffix})
            answer = example["answer"].split("#### ")[-1].strip()
            return {"prompt": prompt, "solution": answer}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    ####################################
    # Set the shapely evaluation dataset
    ####################################
    eval_dataset = dataset[script_args.dataset_test_split]

    if script_args.eval_dataset_name is not None:
        if "gsm8k" in script_args.eval_dataset_name:
            
            eval_dataset = dataset[script_args.dataset_test_split]

        elif "MATH-500" in script_args.eval_dataset_name:

            eval_dataset_hf = datasets.load_dataset(script_args.eval_dataset_name, split="test")

            eval_dataset_hf = eval_dataset_hf.shuffle(seed=training_args.seed)

            # Format into {prompt, solution} with your make_conversation
            eval_dataset_hf = eval_dataset_hf.map(
                make_conversation,
                fn_kwargs={"prompt_column": "problem"},  # always use "question"
                desc="Formatting MATH-500 eval to {prompt, solution}"
            )

            if "messages" in eval_dataset_hf.column_names:
                eval_dataset_hf = eval_dataset_hf.remove_columns("messages")

            # Make sure order matches old eval (["prompt", "solution"])
            eval_dataset_hf = eval_dataset_hf.select_columns(eval_dataset.column_names)

            # Replace both dataset dict and eval_dataset reference
            eval_dataset = eval_dataset_hf

        elif "GPQA-Diamond" in script_args.eval_dataset_name:

            eval_dataset_hf = datasets.load_dataset(script_args.eval_dataset_name, split="test")

            eval_dataset_hf = eval_dataset_hf.shuffle(seed=training_args.seed)

            # Format into {prompt, solution}
            eval_dataset_hf = eval_dataset_hf.map(make_conversation, fn_kwargs={"prompt_column": "question"}, desc="GPQA→{prompt, solution}").select_columns(["prompt", "solution"])

            if "messages" in eval_dataset_hf.column_names:
                eval_dataset_hf = eval_dataset_hf.remove_columns("messages")

            eval_dataset_hf = eval_dataset_hf.select_columns(["prompt", "solution"])

            # Use only the local variable (do NOT mutate `dataset`)
            eval_dataset = eval_dataset_hf
    #############################
    # Initialize the GRPO trainer
    #############################
    train_dataset = dataset[script_args.dataset_train_split].select(range(1000))  # Use a smaller subset for quicker testing
    shapely = False
    new_reward = False
    monitor = True
    if shapely:
        trainer = GRPOTrainerWithShapely(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=(eval_dataset if training_args.eval_strategy != "no" else None),
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
            processing_class=tokenizer,
        )
    elif new_reward:
        trainer = GRPOTrainerWithNewReward(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
            processing_class=tokenizer,
        )
    elif monitor:
        trainer = GRPOTrainerMonitor(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
            processing_class=tokenizer,
        )
    else:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
            processing_class=tokenizer,
        )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    if shapely:
        train_result = trainer.train_with_shapley(resume_from_checkpoint=checkpoint)
    else:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

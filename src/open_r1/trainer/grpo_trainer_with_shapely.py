import os
import time
import shutil
import textwrap
import warnings
import collections
import functools
import contextlib
from functools import partial
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union, Dict

import numpy as np
import re
from collections import defaultdict
import pandas as pd
import json

import huggingface_hub.utils as hf_hub_utils
import datasets
import torch
import torch.utils.data
import transformers
import torch.distributed as dist
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import (seed_worker,
                                        enable_full_determinism,
                                        get_last_checkpoint,
                                        find_executable_batch_size,
                                        speed_metrics,
                                        TrainOutput)
from transformers.modeling_utils import unwrap_model
from transformers.training_args import (OptimizerNames,
                                        ParallelMode)
from transformers.utils import (is_datasets_available, 
                                is_peft_available, 
                                is_rich_available, 
                                is_apex_available,
                                is_sagemaker_mp_enabled,
                                is_torch_compile_available,
                                is_torch_mlu_available,
                                is_torch_mps_available,
                                is_torch_musa_available,
                                is_torch_neuroncore_available,
                                is_torch_npu_available,
                                is_torch_xla_available,
                                is_torch_xpu_available,
                                is_torchao_available,
                                is_accelerate_available,
                                logging)
from transformers.trainer_callback import (TrainerState,
                                            ExportableState,
                                            )
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.integrations.deepspeed import (deepspeed_init, 
                                                     deepspeed_load_checkpoint, 
                                                     is_deepspeed_available)

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import deepspeed

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
else:
    IS_XLA_FSDPV2_POST_2_2 = False

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )
if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

TRAINER_STATE_NAME = "trainer_state.json"
logger = logging.get_logger(__name__)

from .grpo_trainer import GRPOTrainer

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def shuffle_tensor_dict(tensor_dict: dict[str, Optional[torch.Tensor]]) -> dict[str, Optional[torch.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.

    Example:
        >>> x = torch.arange(6).reshape(3, 2)
        >>> y = torch.arange(3).reshape(3, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> shuffle_tensor_dict(tensor_dict)
        {'x': tensor([[2, 3],
                      [0, 1],
                      [4, 5]]),
         'y': tensor([[1],
                      [0],
                      [2]])}
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = torch.randperm(batch_size)
    return {key: tensor[permutation] if tensor is not None else None for key, tensor in tensor_dict.items()}

def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]

class GRPOTrainerWithShapely(GRPOTrainer):
    """
    GRPOTrainer with Shapely Value for enhanced gradient-based analysis.
    
    This class extends the GRPOTrainer to include Shapely functionalities,
    Apart from the train set training calculation, this class will also do the same pipeline to evaluation set.
    The shapely value is obtained by the dot product of the weight vector from train set and the evaluation set.
    Since the evaluation set is not used for training, it will not be used to update the model weights.
    And the reward functions for evaluaiton set will only keep the accuracy reward function.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization for Shapely can be added here if needed
        self.reward_func_names_evaluation = ["accuracy_reward"] # <-- hard coded for now, can be changed later
        self.reward_processing_classes_evaluation = [None] * len(self.reward_func_names_evaluation)
        if self.reward_funcs is not None:
            self.reward_funcs_evaluation = [
                func for name, func in zip(self.reward_func_names, self.reward_funcs)
                if name in self.reward_func_names_evaluation
                ]

        if self.reward_weights is not None:
            self.reward_weights_evaluation = [
                weight for name, weight in zip(self.reward_func_names, self.reward_weights)
                if name in self.reward_func_names_evaluation
            ]
        
        self._eval_batch_size = self.args.eval_batch_size

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer with Shapley: training requires a eval_dataset.")

        return self._get_dataloader(
            dataset=self.eval_dataset,
            description="Validation",
            batch_size=self._eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            is_training=True,
        )

    def _get_eval_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.eval_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def train_with_shapley(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", dict[str, Any], None] = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point with Shapley value computation.
        Similar to the standard train() but enables Shapley value calculation.
        
        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint. If `True`, load the last checkpoint.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*):
                Keys in model output to ignore when gathering predictions for evaluation.
            **kwargs: Additional keyword arguments.
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args
        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # Handle deprecated arguments and validation
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train_with_shapley() got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Use the inner training loop with Shapley modifications
        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop_with_shapley, self._train_batch_size, args.auto_find_batch_size
        )
        
        if args.push_to_hub:
            try:
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop_with_shapley(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        Modified inner training loop that incorporates Shapley value computation.
        """
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        
        # Handle batch size adjustments for auto_find_batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory
                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model
                
                if self.is_deepspeed_enabled:
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size

        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        
        # Data loaders
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()  # For Shapley computation
        
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader) # For Shapley computation

        # Set up evaluation batch for Shapley computation (reused across training steps)
        eval_batch = None
        if eval_dataloader is not None:
            eval_iterator = iter(eval_dataloader)
            eval_batch = next(eval_iterator)

        # Setup training parameters (similar to original)
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        # Token tracking setup
        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        # Model wrapping and preparation (similar to original)
        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Initialize state
        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size
        self.state.compute_steps(args, max_steps)

        # Gradient checkpointing
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)
        use_accelerator_prepare = True if model is self.model else False

        # Model preparation continues as in original...
        if use_accelerator_prepare and self.is_fsdp_enabled:
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Accelerator preparation
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        if model is not self.model:
            self.model_wrapped = model

        # Checkpoint loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # Training setup logging
        logger.info("***** Running Shapley training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")

        # Training state initialization
        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Resume from checkpoint handling
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

        # Update callback handler references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader
        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # Training variables
        tr_loss = torch.tensor(0.0, device=args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        # MAIN TRAINING LOOP WITH SHAPLEY MODIFICATIONS
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )

            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    # Token tracking
                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name in inputs:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()

                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # MODIFIED TRAINING STEP WITH SHAPLEY COMPUTATION
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        # Use our modified training step that computes Shapley values
                        tr_loss_step = self.training_step_with_shapley(model, inputs, eval_batch, num_items_in_batch)

                    # Loss handling
                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    # Synchronization step handling
                    if do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                        self.optimizer.step()
                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        # # Uncomment if you want to log/save/evaluate at the end of each step
                        # self._maybe_log_save_evaluate(
                        #     tr_loss,
                        #     grad_norm,
                        #     model,
                        #     trial,
                        #     epoch,
                        #     ignore_keys_for_eval,
                        #     start_time,
                        #     learning_rate=learning_rate,
                        # )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break

            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}!"
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # # Uncomment if you want to log/save/evaluate at the end of each epoch
            # self._maybe_log_save_evaluate(
            #     tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            # )

            if self.control.should_training_stop:
                break

        # Training completion (similar to original)
        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nShapley training completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()
            self._load_best_model()

        # Final metrics computation
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)

        # Cleanup checkpoints
        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        self._finish_current_push()

        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step_with_shapley(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], eval_inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """
        Modified training step that computes Shapley values using gradient dot products.
        Based on the actual training_step implementation with Shapley computation added.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        eval_inputs = self._prepare_inputs(eval_inputs, shapely=True)
        
        # Handle SageMaker MP if enabled
        if is_sagemaker_mp_enabled():
            print("is_sagemaker_mp_enabled")
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        # Compute loss for training batch
        with self.compute_loss_context_manager():
            train_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        
        # Compute loss for evaluation batch (for Shapley computation)
        with self.compute_loss_context_manager():
            eval_loss = self.compute_loss(model, eval_inputs, num_items_in_batch=num_items_in_batch)

        # Clean up inputs
        del inputs
        del eval_inputs

        # Cache management
        if self.args.torch_empty_cache_steps is not None and self.state.global_step % self.args.torch_empty_cache_steps == 0:
            torch.cuda.empty_cache()

        kwargs = {}
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        # Handle multi-GPU
        if self.args.n_gpu > 1:
            train_loss = train_loss.mean()
            eval_loss = eval_loss.mean()

        # SHAPLEY COMPUTATION: Calculate gradients for both losses
        # Compute gradients for training loss (without backward pass)
        if self.use_apex:
            with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
                train_gradients = torch.autograd.grad(
                    scaled_loss, 
                    model.parameters(), 
                    retain_graph=True, 
                    create_graph=False
                )
        else:
            train_loss_for_grad = train_loss / self.args.gradient_accumulation_steps if (not self.model_accepts_loss_kwargs and self.compute_loss_func is None) else train_loss
            train_gradients = torch.autograd.grad(
                train_loss_for_grad,
                model.parameters(),
                retain_graph=True,
                create_graph=False
            )

        # Compute gradients for evaluation loss
        if self.use_apex:
            with amp.scale_loss(eval_loss, self.optimizer) as scaled_loss:
                eval_gradients = torch.autograd.grad(
                    scaled_loss,
                    model.parameters(),
                    retain_graph=False,
                    create_graph=False
                )
        else:
            eval_loss_for_grad = eval_loss / self.args.gradient_accumulation_steps if (not self.model_accepts_loss_kwargs and self.compute_loss_func is None) else eval_loss
            eval_gradients = torch.autograd.grad(
                eval_loss_for_grad,
                model.parameters(),
                retain_graph=False,
                create_graph=False
            )
        
        del eval_loss
        torch.cuda.empty_cache()

        mode = "train" if self.model.training else "eval"
        # Compute Shapley value as dot product of gradients
        self.compute_shapley_from_gradients(train_gradients, eval_gradients, model, mode=mode)

        del train_gradients
        del eval_gradients
        torch.cuda.empty_cache()

        # Now perform the actual backward pass for training (following original training_step logic)
        if self.use_apex:
            with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                train_loss = train_loss / self.args.gradient_accumulation_steps
            
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            
            self.accelerator.backward(train_loss, **kwargs)
            
            # Extract gradients and collect stats (following original training_step)
            gradients = self._extract_global_gradients(self.accelerator, self.model)
            self._collect_gradient_stats_by_layers(gradients, mode)
            self._save_all_metrics_snapshot(
                    mode=mode
                )

        return train_loss.detach()

    def compute_shapley_from_gradients(
            self,
            train_gradients,  # tuple of gradients from torch.autograd.grad
            eval_gradients,   # tuple of gradients from torch.autograd.grad
            model,
            mode=None,  # "train" or "eval"
        ) -> None:
        use_cosine_similarity = True

        """Compute Shapley values as dot product of train and eval gradients."""
        if not self.accelerator.is_main_process:
            return

        if mode is None:
            mode = "train" if model.training else "eval"

        # Get parameter names to match with gradients
        param_names = [name for name, _ in model.named_parameters()]
        
        # Ensure we have matching lengths
        if len(param_names) != len(train_gradients) or len(param_names) != len(eval_gradients):
            print(f"[rank0] Warning: Parameter count mismatch - names: {len(param_names)}, "
                f"train_grads: {len(train_gradients)}, eval_grads: {len(eval_gradients)}")
            return

        # Setup metrics storage
        metrics_mode = self._metrics.setdefault(mode, defaultdict(list))
        
        # Compute Shapley values
        step = int(getattr(self.state, "global_step", 0))
        shapley_values = []
        computed_params = []
        
        for name, g_train, g_eval in zip(param_names, train_gradients, eval_gradients):
            if g_train is None or g_eval is None:
                continue
                
            try:
                # Ensure same device and dtype
                if g_train.device != g_eval.device:
                    g_eval = g_eval.to(g_train.device)
                if g_train.dtype != g_eval.dtype:
                    g_eval = g_eval.to(g_train.dtype)
                    
                # Compute dot product (Shapley value)
                if use_cosine_similarity:
                    gt = g_train.flatten()
                    ge = g_eval.flatten()
                    denom = (gt.norm() * ge.norm()).clamp_min(1e-12)
                    shapley_value = torch.dot(gt, ge) / denom
                    shapley_value = shapley_value.item()
                else:
                    shapley_value = torch.dot(g_train.flatten(), g_eval.flatten()).item()
                
                # Store the value
                metrics_mode[f"shapley_stats/params/{name}/dot_product"].append(float(shapley_value))
                shapley_values.append(shapley_value)
                computed_params.append(name)
                
            except RuntimeError as e:
                if self.args.debug:
                    print(f"Failed to compute Shapley for {name}: {str(e)}")
                continue
        
        # Compute and log aggregate statistics
        if shapley_values:
            shapley_tensor = torch.tensor(shapley_values)
            metrics_mode["shapley_stats/aggregate/mean"].append(float(shapley_tensor.mean()))
            metrics_mode["shapley_stats/aggregate/std"].append(float(shapley_tensor.std()))
            metrics_mode["shapley_stats/aggregate/max"].append(float(shapley_tensor.max()))
            metrics_mode["shapley_stats/aggregate/min"].append(float(shapley_tensor.min()))
            metrics_mode["shapley_stats/aggregate/num_params"].append(len(shapley_values))
            
            if self.args.debug:
                print(f"[rank0] Step {step}: Computed {len(shapley_values)}/{len(param_names)} "
                    f"Shapley values, mean={shapley_tensor.mean():.6f}, std={shapley_tensor.std():.6f}")
        else:
            print(f"[rank0] No Shapley values computed at step {step} "
                f"(total parameters: {len(param_names)})")


    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]], shapely: bool = False
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch, shapely=shapely)
                generation_batch = shuffle_tensor_dict(generation_batch)
                self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch, shapely=shapely)
        return inputs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]], shapely: bool = False
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs) # <- call grand parent method to handle any additional processing
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1,  # vLLM on each GPU generates only 1 in colocate mode
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        # 
        if not shapely:
            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

            # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

            for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
            ):
                with profiling_context(self, reward_func_name):
                    if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                        if is_conversational(inputs[0]):
                            messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                        else:
                            texts = [p + c for p, c in zip(prompts, completions)]
                        reward_inputs = reward_processing_class(
                            text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                        )
                        reward_inputs = super()._prepare_inputs(reward_inputs)
                        with torch.inference_mode():
                            rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                    else:
                        output_reward_func = reward_func(
                            prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                        )
                        # Convert None values to NaN
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            # If all reward functions return None for a given row, issue a detailed warning
            if torch.isnan(rewards_per_func).all(dim=1).any():
                nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
                row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
                row_reward_kwargs["prompt"] = prompts[nan_row_idx]
                row_reward_kwargs["completion"] = completions[nan_row_idx]
                warnings.warn(
                    f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                    "Please ensure that at least one reward function returns a valid reward."
                )

            # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
            # completions may be distributed across processes
            rewards_per_func = gather(rewards_per_func)

            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        
        else:
            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs_evaluation), device=device)

            # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

            for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
                zip(self.reward_funcs_evaluation, self.reward_processing_classes_evaluation, self.reward_func_names_evaluation)
            ):
                with profiling_context(self, reward_func_name):
                    if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                        if is_conversational(inputs[0]):
                            messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                        else:
                            texts = [p + c for p, c in zip(prompts, completions)]
                        reward_inputs = reward_processing_class(
                            text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                        )
                        reward_inputs = super()._prepare_inputs(reward_inputs)
                        with torch.inference_mode():
                            rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                    else:
                        output_reward_func = reward_func(
                            prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                        )
                        # Convert None values to NaN
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)


        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # H ≈ mean_t( - log πθ(y_t | y_<t, x) ) over completion tokens (masked after first EOS)
        # We compute per-token log-probs for the completion positions, then average -logp.
        with torch.no_grad():
            # If already computed above, reuse it; otherwise compute now.
            if old_per_token_logps is not None:
                per_token_logps = old_per_token_logps  # shape: (B, C)
            else:
                per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )  # shape: (B, C)

        # Mask: count only tokens up to & including the first EOS (and possibly drop truncated completions)
        # completion_mask: (B, C) with {0,1}
        mask = completion_mask.to(per_token_logps.dtype)

        # Monte-Carlo entropy per token: -log p(y_t)
        nll = -per_token_logps  # (B, C)

        # Global (all tokens across batch) entropy mean
        total_nll = (nll * mask).sum()
        total_tokens = mask.sum().clamp_min(1)
        entropy_mc_mean = (total_nll / total_tokens).item()

        # Per-sequence entropy (mean over tokens of that sequence)
        per_seq_tokens = mask.sum(dim=1).clamp_min(1)
        per_seq_entropy = (nll * mask).sum(dim=1) / per_seq_tokens  # (B,)

        # Gather to main process for logging consistent with other metrics
        agg_per_seq_entropy = self.accelerator.gather(per_seq_entropy)
        self._metrics[mode]["entropy/per_seq_mean"].append(agg_per_seq_entropy.float().mean().item())
        self._metrics[mode]["entropy/per_seq_min"].append(agg_per_seq_entropy.float().min().item())
        self._metrics[mode]["entropy/per_seq_max"].append(agg_per_seq_entropy.float().max().item())

        # Also log mean over all tokens (global)
        self._metrics[mode]["entropy/mc_token_nll_mean"].append(entropy_mc_mean)

        # (Optional) Restrict to sequences that terminated with EOS
        agg_is_terminated = self.accelerator.gather(is_eos.any(dim=1))  # (B_total,)
        if agg_is_terminated.any():
            term_mask_local = is_eos.any(dim=1)  # (B,)
            if term_mask_local.any():
                term_nll = (nll[term_mask_local] * mask[term_mask_local]).sum()
                term_tokens = mask[term_mask_local].sum().clamp_min(1)
                term_entropy_mc_mean = (term_nll / term_tokens).item()
                self._metrics[mode]["entropy/mc_token_nll_mean_terminated"].append(term_entropy_mc_mean)
            else:
                # Keep metric aligned in case of no terminated seq on this rank
                self._metrics[mode]["entropy/mc_token_nll_mean_terminated"].append(float("nan"))
        else:
            self._metrics[mode]["entropy/mc_token_nll_mean_terminated"].append(float("nan"))
        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }

    # ===================================
    # Below are legacy implementations
    # ===================================
    # def train_with_shapely(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
    #     """
    #     Main training entry point. This method will initialize the model, optimizer, and scheduler,
    #     set up the training loop, and handle the training process. At each step, two inputs will be passed to the model:
    #     1. The training set input, which will be used to update the model weights. Each training step will have a new training batch.
    #     2. The evaluation set input, which will be used to compute the Shapely value. Each training step will take the evaluation set as as batch and 
    #         use the same evaluation set input, which is the same for all training steps.

    #     Args:
    #         model_path (:obj:`str`, `optional`):
    #             Local path to the model if the model to train has been instantiated from a local path. If present,
    #             training will resume from the optimizer/scheduler states loaded here.
    #         trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
    #             The trial run or the hyperparameter dictionary for hyperparameter search.
    #     """
    #     # This might change the seed so needs to run first.
    #     self._hp_search_setup(trial)

    #     # Model re-init
    #     if self.model_init is not None:
    #         # Seed must be set before instantiating the model when using model_init.
    #         set_seed(self.args.seed)

    #         model = self.call_model_init(trial)

    #         self.model = model.to(self.args.device)

    #         # Reinitializes optimizer and scheduler
    #         self.optimizer, self.lr_scheduler = None, None

    #     # Keeping track whether we can can len() on the dataset or not
    #     train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
    #     eval_dataset_is_sized = isinstance(self.eval_dataset, collections.abc.Sized)

    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()
    #     eval_dataloader = self.get_eval_dataloader() # This is the evaluation set for Shapely Value

    #     # This is the evaluation batch for Shapely Value
    #     eval_batch = None
    #     if eval_dataloader is not None:
    #         # Get a single batch from evaluation dataset and reuse it
    #         eval_iterator = iter(eval_dataloader)
    #         eval_batch = next(eval_iterator)
    #         # Move to appropriate device
    #         if isinstance(eval_batch, dict):
    #             eval_batch = {k: v.to(self.args.device) for k, v in eval_batch.items()}
    #         else:
    #             eval_batch = eval_batch.to(self.args.device)

    #     # Check if evaluation dataset is bigger than a training batch
    #     if eval_dataloader is not None and eval_dataset_is_sized and len(eval_dataloader) > self.args.train_batch_size:
    #         raise ValueError(
    #             "The evaluation dataset is bigger than the training batch size. "
    #             "Please increase the training batch size or reduce the evaluation dataset size."
    #         )

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     if train_dataset_is_sized:
    #         num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         if self.args.max_steps > 0:
    #             max_steps = self.args.max_steps
    #             num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
    #                 self.args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #         else:
    #             max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(self.args.num_train_epochs)
    #     else:
    #         # see __init__. max_steps is set when the dataset has no __len__
    #         max_steps = self.args.max_steps
    #         num_train_epochs = 1
    #         num_update_steps_per_epoch = max_steps

    #     self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    #     self.state = TrainerState()
    #     self.state.is_hyper_param_search = trial is not None

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(model_path)

    #     # Mixed precision training with apex (torch < 1.6)
    #     model = self.model
    #     if self.args.fp16 and self.use_apex:
    #         if not is_apex_available():
    #             raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #         model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

    #     # Multi-gpu training (should be after apex fp16 initialization)
    #     if self.args.n_gpu > 1:
    #         model = torch.nn.DataParallel(model)

    #     # Distributed training (should be after apex fp16 initialization)
    #     if self.args.local_rank != -1:
    #         model = torch.nn.parallel.DistributedDataParallel(
    #             model,
    #             device_ids=[self.args.local_rank],
    #             output_device=self.args.local_rank,
    #             find_unused_parameters=(
    #                 not getattr(model.config, "gradient_checkpointing", False)
    #                 if isinstance(model, PreTrainedModel)
    #                 else True
    #             ),
    #         )
    #     # find_unused_parameters breaks checkpointing as per
    #     # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

    #     # Train!
    #     if is_torch_tpu_available():
    #         total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
    #     else:
    #         total_train_batch_size = (
    #             self.args.train_batch_size
    #             * self.args.gradient_accumulation_steps
    #             * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
    #         )

    #     num_examples = (
    #         self.num_examples(train_dataloader)
    #         if train_dataset_is_sized
    #         else total_train_batch_size * self.args.max_steps
    #     )

    #     logger.info("***** Running training *****")
    #     logger.info("  Num examples = %d", num_examples)
    #     logger.info("  Num Epochs = %d", num_train_epochs)
    #     logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
    #     logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
    #     logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
    #     logger.info("  Total optimization steps = %d", max_steps)

    #     self.state.epoch = 0
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0

    #     # Check if continuing training from a checkpoint
    #     if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
    #         self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
    #         epochs_trained = self.state.global_step // num_update_steps_per_epoch
    #         steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info("  Continuing training from epoch %d", epochs_trained)
    #         logger.info("  Continuing training from global step %d", self.state.global_step)
    #         logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
    #     self.state.trial_params = hp_params(trial) if trial is not None else None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()

    #     tr_loss = torch.tensor(0.0).to(self.args.device)
    #     self._logging_loss_scalar = 0
    #     self._globalstep_last_logged = 0
    #     self._total_flos = self.state.total_flos
    #     model.zero_grad()

    #     self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

    #     for epoch in range(epochs_trained, num_train_epochs):
    #         if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
    #             train_dataloader.sampler.set_epoch(epoch)

    #         if is_torch_tpu_available():
    #             parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
    #                 self.args.device
    #             )
    #             epoch_iterator = parallel_loader
    #         else:
    #             epoch_iterator = train_dataloader

    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if self.args.past_index >= 0:
    #             self._past = None

    #         steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
    #         self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

    #         for step, inputs in enumerate(epoch_iterator):

    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 continue

    #             if (step + 1) % self.args.gradient_accumulation_steps == 0:
    #                 self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

    #             if (
    #                 ((step + 1) % self.args.gradient_accumulation_steps != 0)
    #                 and self.args.local_rank != -1
    #                 and _use_ddp_no_sync
    #             ):
    #                 with model.no_sync():
    #                     tr_loss += self.training_step(model, inputs)
    #             else:
    #                 tr_loss += self.training_step(model, inputs)
    #             self._total_flos += self.floating_point_ops(inputs)

    #             if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
    #                 # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                 steps_in_epoch <= self.args.gradient_accumulation_steps
    #                 and (step + 1) == steps_in_epoch
    #             ):
    #                 if self.args.fp16 and _use_native_amp:
    #                     self.scaler.unscale_(self.optimizer)
    #                     torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
    #                 elif self.args.fp16 and _use_apex:
    #                     torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
    #                 else:
    #                     torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

    #                 if is_torch_tpu_available():
    #                     xm.optimizer_step(self.optimizer)
    #                 elif self.args.fp16 and _use_native_amp:
    #                     self.scaler.step(self.optimizer)
    #                     self.scaler.update()
    #                 else:
    #                     self.optimizer.step()

    #                 self.lr_scheduler.step()
    #                 model.zero_grad()
    #                 self.state.global_step += 1
    #                 self.state.epoch = epoch + (step + 1) / steps_in_epoch
    #                 self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

    #                 self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 break

    #         self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
    #         self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

    #         if self.args.tpu_metrics_debug or self.args.debug:
    #             if is_torch_tpu_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break

    #     if self.args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")

    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         logger.info(
    #             f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
    #         )
    #         if isinstance(model, PreTrainedModel):
    #             self.model = model.from_pretrained(self.state.best_model_checkpoint)
    #             self.model = self.model.to(self.args.device)
    #         else:
    #             state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
    #             self.model.load_state_dict(state_dict)

    #     if self._total_flos is not None:
    #         self.store_flos()
    #         self.log({"total_flos": self.state.total_flos})

    #     self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)

    #     return TrainOutput(self.state.global_step, tr_loss.item() / self.state.global_step)

    # def training_step_with_shapley(self, model: nn.Module, inputs: Dict[str, Any], eval_inputs: Dict[str, Any]) -> torch.Tensor:
    #     """
    #     Full training step with Shapley value computation using gradient dot products.
    #     Computes gradients on both training and evaluation batches for Shapley calculation.
    #     """
    #     model.train()
    #     if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
    #         self.optimizer.train()

    #     inputs = self._prepare_inputs(inputs)
    #     eval_inputs = self._prepare_inputs(eval_inputs)
        
    #     if is_sagemaker_mp_enabled():
    #         print("is_sagemaker_mp_enabled")
    #         # Note: SageMaker MP path would need custom handling for dual gradient computation
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)

    #     # Compute loss and gradients for training batch
    #     with self.compute_loss_context_manager():
    #         train_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        
    #     # Compute loss and gradients for evaluation batch
    #     with self.compute_loss_context_manager():
    #         eval_loss = self.compute_loss(model, eval_inputs, num_items_in_batch=num_items_in_batch)

    #     # Clean up inputs
    #     del inputs
    #     del eval_inputs

    #     if self.args.torch_empty_cache_steps is not None and self.state.global_step % self.args.torch_empty_cache_steps == 0:
    #         torch.cuda.empty_cache()

    #     kwargs = {}
    #     if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #         kwargs["learning_rate"] = self._get_learning_rate()

    #     if self.args.n_gpu > 1:
    #         train_loss = train_loss.mean()
    #         eval_loss = eval_loss.mean()

    #     # Compute gradients for training loss
    #     if self.use_apex:
    #         with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
    #             train_gradients = torch.autograd.grad(
    #                 scaled_loss, 
    #                 model.parameters(), 
    #                 retain_graph=True, 
    #                 create_graph=True
    #             )
    #     else:
    #         if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
    #             train_loss_scaled = train_loss / self.args.gradient_accumulation_steps
    #         else:
    #             train_loss_scaled = train_loss
                
    #         train_gradients = torch.autograd.grad(
    #             train_loss_scaled,
    #             model.parameters(),
    #             retain_graph=True,
    #             create_graph=False # First order gradients are sufficient for Shapley computation
    #         )

    #     # Compute gradients for evaluation loss
    #     if self.use_apex:
    #         with amp.scale_loss(eval_loss, self.optimizer) as scaled_loss:
    #             eval_gradients = torch.autograd.grad(
    #                 scaled_loss,
    #                 model.parameters(),
    #                 retain_graph=True,
    #                 create_graph=False  # First order gradients are sufficient for Shapley computation
    #             )
    #     else:
    #         if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
    #             eval_loss_scaled = eval_loss / self.args.gradient_accumulation_steps
    #         else:
    #             eval_loss_scaled = eval_loss
                
    #         eval_gradients = torch.autograd.grad(
    #             eval_loss_scaled,
    #             model.parameters(),
    #             retain_graph=True,
    #             create_graph=False
    #         )

    #     # Compute Shapley value as dot product of gradients
    #     shapley_value = self.compute_shapley_from_gradients(train_gradients, eval_gradients)
        
    #     # Log Shapley value for monitoring
    #     if self.state.global_step % self.args.logging_steps == 0:
    #         self.log({"shapley_value": shapley_value.item()})

    #     # Now perform the actual backward pass for training
    #     if self.use_apex:
    #         with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
    #             final_loss = train_loss / self.args.gradient_accumulation_steps
    #         else:
    #             final_loss = train_loss

    #         if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
    #             kwargs["scale_wrt_gas"] = False

    #         self.accelerator.backward(final_loss, **kwargs)

    #     # Extract gradients for any additional processing
    #     gradients = self._extract_global_gradients(self.accelerator, self.model)

    #     return train_loss.detach()

    # def compute_shapley_from_gradients(self, train_gradients, eval_gradients) -> torch.Tensor:
    #     """
    #     Compute Shapley value as dot product of training and evaluation gradients.
        
    #     Args:
    #         train_gradients: Tuple of gradient tensors from training batch
    #         eval_gradients: Tuple of gradient tensors from evaluation batch
        
    #     Returns:
    #         Shapley value as scalar tensor
    #     """
    #     shapley_value = torch.tensor(0.0, device=self.args.device)
        
    #     # Compute dot product for each parameter's gradients
    #     for train_grad, eval_grad in zip(train_gradients, eval_gradients):
    #         if train_grad is not None and eval_grad is not None:
    #             # Flatten gradients and compute dot product
    #             train_flat = train_grad.flatten()
    #             eval_flat = eval_grad.flatten()
    #             shapley_value += torch.dot(train_flat, eval_flat)
        
    #     return shapley_value
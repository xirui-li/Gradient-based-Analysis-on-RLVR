import os
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union, Dict

import numpy as np
import re
from collections import defaultdict
import pandas as pd
import json

import datasets
import torch
import torch.utils.data
import torch.distributed as dist
import transformers
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
from transformers.trainer_utils import seed_worker
from transformers.training_args import OptimizerNames
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
                                is_accelerate_available
                                )

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
from .grpo_trainer import GRPOTrainer
from ..utils.metrics import MetricsComputer

class GRPOTrainerMonitor(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Configure which metrics to compute and their intervals
        metric_config = {
            "layer_gradient_distribution": {"enabled": True, "interval": 10},
            "effective_rank": {"enabled": True, "interval": 10},
            "nuclear_norm": {"enabled": True, "interval": 10},
            "reasoning_emergence": {"enabled": False, "interval": 10},  # Enable if you have hidden states
            "global_gradient": {"enabled": False, "interval": 10}
        }
        
        # Create metrics computer
        self.metrics_computer = MetricsComputer(
            metric_config=metric_config,
            svd_min_params=1000,
            output_dir=f"metrics/{self.args.run_name}"  # or however you name your runs
        )

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """
        Full training step with gradient stats collection (Accelerate-compatible).
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            print("is_sagemaker_mp_enabled")
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            # To be checked
            # Extract hidden states if your model outputs them (for reasoning emergence metrics)
            hidden_states = None
            if hasattr(model, 'last_hidden_states'):
                hidden_states = model.last_hidden_states
            elif hasattr(self, 'last_outputs') and hasattr(self.last_outputs, 'hidden_states'):
                # Some models store hidden states in outputs
                hidden_states = self.last_outputs.hidden_states[-1] if self.last_outputs.hidden_states else None

        # Extract rewards if available in your inputs
        rewards = inputs.get('rewards', None) if isinstance(inputs, dict) else None

        del inputs

        if self.args.torch_empty_cache_steps is not None and self.state.global_step % self.args.torch_empty_cache_steps == 0:
            torch.cuda.empty_cache()

        kwargs = {}

        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            self.accelerator.backward(loss, **kwargs)
           
            # ============= METRICS COMPUTATION =============
            # Only compute metrics on main process to avoid duplication
            if self.accelerator.is_main_process:
                # Extract gradients after backward pass
                gradients = self._extract_global_gradients(self.accelerator, model)
                mode = "train" if self.model.training else "eval"
                # Compute metrics using our metrics computer
                if gradients:  # Only if we successfully extracted gradients
                    metrics = self.metrics_computer.compute_all_metrics(
                        gradients=gradients,
                        model=model,
                        loss=loss.item(),
                        rewards=rewards,
                        hidden_states=hidden_states,
                        training_mode=mode,
                        step=self.state.global_step
                    )

            return loss.detach()
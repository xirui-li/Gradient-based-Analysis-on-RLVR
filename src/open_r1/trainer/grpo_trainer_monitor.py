import os
import random
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
           
            gradients = self._extract_global_gradients(self.accelerator, self.model)
            mode = "train" if self.model.training else "eval"
            self._collect_gradient_stats_by_layers(gradients, mode)

            return loss.detach()

    def _collect_gradient_stats_by_layers(self, gradients, mode):
        """
        Collect gradient statistics with configurable intervals for different metrics.
        """
        if not self.accelerator.is_main_process:
            return
        step = int(getattr(self.state, "global_step", 0))

        grad_items = [(name, grad) for name, grad in gradients.items() if grad is not None]

        if not grad_items:
            print(f"Debug: No gradients found on rank {self.accelerator.process_index}. This is expected under ZeRO-3.")
            return

        step_grad_stats = {}

        # Initialize stateful containers and intervals
        if not hasattr(self, "_svd_sample_layers"):
            
            # Configurable intervals
            self._save_all_metrics_interval = 10  # save all metrics every 10 steps
            self._basic_stats_interval = 10  # mean/max/min every 5 steps
            self._nuclear_norm_interval = 10  # nuclear norm every 10 steps
            self._effective_rank_interval = 10  # effective rank every 50 steps

            self._svd_sample_size = -1
            
            # Fixed random sampling settings
            eligible_layers = [name for name, grad in grad_items if grad.ndim >= 2]
            if eligible_layers:
                if self._svd_sample_size == -1:
                    # Use all eligible layers
                    self._svd_sample_layers = set(eligible_layers)
                    print(f"[Init] Using all {len(eligible_layers)} eligible layers for SVD")
                else:
                    # Use fixed random sample
                    random.seed(42)  # Fixed seed for reproducibility
                    sample_size = min(self._svd_sample_size, len(eligible_layers))
                    self._svd_sample_layers = set(random.sample(eligible_layers, sample_size))
                    print(f"[Init] Fixed random sample of {sample_size} layers for SVD: {self._svd_sample_layers}")
            else:
                self._svd_sample_layers = set()

        # Part 1: Basic statistics (mean, max, min, frobenius_norm)
        if step % self._basic_stats_interval == 0:
            for name, grad in grad_items:
                if grad is None:
                    continue

                M_mean = grad.mean().item()
                M_max = grad.max().item()
                M_min = grad.min().item()
                frobenius_norm = torch.linalg.norm(grad).item()

                param_prefix = f"grad_stats/params/{name}"

                # Save to file
                step_grad_stats[f"{param_prefix}/M_mean"] = M_mean
                step_grad_stats[f"{param_prefix}/M_max"] = M_max
                step_grad_stats[f"{param_prefix}/M_min"] = M_min
                step_grad_stats[f"{param_prefix}/frobenius_norm"] = frobenius_norm

        # Randomly sample layers for SVD operations (when needed)
        eligible_layers = [name for name, grad in grad_items if grad.ndim >= 2]
        if eligible_layers and (step % self._nuclear_norm_interval == 0 or step % self._effective_rank_interval == 0):
            sample_size = min(self._svd_sample_size, len(eligible_layers))
            self._svd_sample_layers = set(random.sample(eligible_layers, sample_size))
            print(f"[Step {step}] Randomly sampled {sample_size} layers for SVD: {self._svd_sample_layers}")

        # Part 2: Nuclear norm calculation
        if self._svd_sample_layers and step % self._nuclear_norm_interval == 0:
            for name, grad in grad_items:
                if name not in self._svd_sample_layers or grad.ndim < 2:
                    continue

                try:
                    # Compute SVD for nuclear norm
                    S = torch.linalg.svd(grad.detach(), full_matrices=False).S
                    nuclear_norm = S.sum().item()
                    S_max = S.max().item()
                    S_min = S.min().item()

                    param_prefix = f"grad_stats/params/{name}"

                    step_grad_stats[f"{param_prefix}/nuclear_norm"] = nuclear_norm
                    step_grad_stats[f"{param_prefix}/S_max"] = S_max
                    step_grad_stats[f"{param_prefix}/S_min"] = S_min

                except Exception as e:
                    print(f"[Warning] SVD (nuclear norm) failed for {name} at step {step}: {e}")

        # Part 3: Effective rank calculation
        if self._svd_sample_layers and step % self._effective_rank_interval == 0:
            for name, grad in grad_items:
                if name not in self._svd_sample_layers or grad.ndim < 2:
                    continue

                try:
                    # Compute SVD for effective rank
                    S = torch.linalg.svd(grad.detach(), full_matrices=False).S
                    S_sum = S.sum().item()
                    p = S / S_sum
                    effective_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item()

                    param_prefix = f"grad_stats/params/{name}"
                    step_grad_stats[f"{param_prefix}/effective_rank"] = effective_rank

                except Exception as e:
                    print(f"[Warning] SVD (effective rank) failed for {name} at step {step}: {e}")

        # Save summarized data if we collected any stats this step
        if step_grad_stats and step % self._save_all_metrics_interval == 0:
            self._save_summarized_metrics(step, mode, step_grad_stats)


    def _extract_global_gradients(self, accelerator, model):
        """
        Extract full gradients under DeepSpeed ZeRO-3 using safe_get_full_grad.
        """
        unwrapped_model = accelerator.unwrap_model(model)
        gradients = {}
        for name, param in unwrapped_model.named_parameters():
            if param.requires_grad:
                full_grad = deepspeed.utils.safe_get_full_grad(param)
                if full_grad is not None:
                    gradients[name] = full_grad.clone()  # important: clone to avoid memory issues
        return gradients


    def _save_summarized_metrics(self, step: int, mode: str, step_grad_stats: dict):
        """
        Save only summarized gradient statistics data.
        """

        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Only rank 0 saves
        if rank == 0:
            save_root = os.path.join("stats", self.run_name, f"step_{step:08d}")
            os.makedirs(save_root, exist_ok=True)
            out_path = os.path.join(save_root, "grad_summary.json")

            summary = {
                "mode": mode,
                "step": int(step),
                "metrics": step_grad_stats
            }
            
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)


    def _safe_convert(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.item()
        return obj


    def _recursive_convert(self, d):
        if isinstance(d, dict):
            return {k: self._recursive_convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._safe_convert(v) for v in d]
        else:
            return self._safe_convert(d)
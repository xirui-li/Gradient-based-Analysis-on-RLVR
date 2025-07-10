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

from trl import SFTTrainer
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


class CustomSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer that inherits from TRL's SFTTrainer and adds gradient statistics collection.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize metrics storage for gradient statistics
        self._metrics = {
            "train": {},
            "eval": {}
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
        Collect gradient statistics and optionally compute SVD-derived metrics
        (nuclear norm and effective rank) for top-k layers based on average Frobenius norm.

        Runs SVD only every 50 steps after determining top-k layers during early training.
        """
        print(f"Debug: Starting gradient stats collection on rank {self.accelerator.process_index} in {mode} mode")

        grad_items = [(name, grad) for name, grad in gradients.items() if grad is not None]

        if not grad_items:
            print(f"Debug: No gradients found on rank {self.accelerator.process_index}. This is expected under ZeRO-3.")
            return

        step_grad_stats = {}

        # Initialize stateful containers
        if not hasattr(self, "_running_layer_norms"):
            from collections import defaultdict
            self._running_layer_norms = defaultdict(list)
            self._topk_svd_layers = None
            self._svd_top_k = 5  # You can make this a config arg

        for name, grad in grad_items:
            if grad is None:
                continue

            # Basic stats
            M_mean = grad.mean().item()
            M_max = grad.max().item()
            M_min = grad.min().item()
            frobenius_norm = torch.linalg.norm(grad).item()

            param_prefix = f"grad_stats/params/{name}"

            # Save to memory
            self._metrics[mode][f"{param_prefix}/M_mean"].append(M_mean)
            self._metrics[mode][f"{param_prefix}/M_max"].append(M_max)
            self._metrics[mode][f"{param_prefix}/M_min"].append(M_min)
            self._metrics[mode][f"{param_prefix}/frobenius_norm"].append(frobenius_norm)

            # Save to file
            step_grad_stats[f"{param_prefix}/M_mean"] = M_mean
            step_grad_stats[f"{param_prefix}/M_max"] = M_max
            step_grad_stats[f"{param_prefix}/M_min"] = M_min
            step_grad_stats[f"{param_prefix}/frobenius_norm"] = frobenius_norm

            # Track norm for first 50 steps
            if self.state.global_step < 50 and grad.ndim >= 2:
                self._running_layer_norms[name].append(frobenius_norm)

        # Select top-k layers at step 49
        if self.state.global_step == 49:
            avg_norms = {name: sum(vals) / len(vals) for name, vals in self._running_layer_norms.items()}
            sorted_layers = sorted(avg_norms.items(), key=lambda x: x[1], reverse=True)
            self._topk_svd_layers = {name for name, _ in sorted_layers[:self._svd_top_k]}
            print(f"[Info] Top-{self._svd_top_k} layers selected for SVD:", self._topk_svd_layers)

        # Compute SVD on selected layers every 50 steps
        if self._topk_svd_layers and self.state.global_step % 50 == 0:
            for name, grad in grad_items:
                if name not in self._topk_svd_layers or grad.ndim < 2:
                    continue

                try:
                    # GPU SVD for speed
                    S = torch.linalg.svd(grad.detach(), full_matrices=False).S
                    S_sum = S.sum().item()
                    p = S / S_sum
                    effective_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item()

                    nuclear_norm = S_sum
                    S_max = S.max().item()
                    S_min = S.min().item()

                    # Save to memory
                    param_prefix = f"grad_stats/params/{name}"
                    self._metrics[mode][f"{param_prefix}/S_sum"].append(S_sum)
                    self._metrics[mode][f"{param_prefix}/S_max"].append(S_max)
                    self._metrics[mode][f"{param_prefix}/S_min"].append(S_min)
                    self._metrics[mode][f"{param_prefix}/nuclear_norm"].append(nuclear_norm)
                    self._metrics[mode][f"{param_prefix}/effective_rank"].append(effective_rank)

                    # Save to JSONL
                    step_grad_stats[f"{param_prefix}/S_sum"] = S_sum
                    step_grad_stats[f"{param_prefix}/S_max"] = S_max
                    step_grad_stats[f"{param_prefix}/S_min"] = S_min
                    step_grad_stats[f"{param_prefix}/nuclear_norm"] = nuclear_norm
                    step_grad_stats[f"{param_prefix}/effective_rank"] = effective_rank

                except Exception as e:
                    print(f"[Warning] SVD failed for {name} at step {self.state.global_step}: {e}")
                    for stat in ["S_sum", "S_max", "S_min", "nuclear_norm", "effective_rank"]:
                        self._metrics[mode][f"{param_prefix}/{stat}"].append(None)
                        step_grad_stats[f"{param_prefix}/{stat}"] = None

        # Save stats to file
        self._save_grad_stats_to_file(step_grad_stats, step=self.state.global_step)


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

    def _append_grad_stats_to_json(self, grad_stats: dict, filepath: str):
        grad_stats_serializable = self._recursive_convert(grad_stats)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(grad_stats_serializable)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    

    def _save_grad_stats_to_file(self, data: dict, step: int):
        """
        Save gradient stats for one step as a separate JSON file in a folder.

        Args:
            data (dict): Gradient stats for this step.
            step (int): The current training step.
            save_dir (str): Directory where individual JSON files are stored.
        """
        save_dir = "grad_stats_history_sft_" + self.model_name_or_path
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"step_{step:06d}.json")

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def get_gradient_metrics(self, mode="train"):
        """
        Get the collected gradient metrics for analysis.
        
        Args:
            mode (str): Either "train" or "eval"
            
        Returns:
            dict: Dictionary containing gradient statistics
        """
        return self._metrics.get(mode, {})

    def clear_gradient_metrics(self, mode=None):
        """
        Clear stored gradient metrics.
        
        Args:
            mode (str, optional): If specified, clear only that mode's metrics.
                                If None, clear all metrics.
        """
        if mode is None:
            self._metrics = {"train": {}, "eval": {}}
        else:
            self._metrics[mode] = {}


# Usage example:
# trainer = CustomSFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset[script_args.dataset_train_split],
#     eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
#     processing_class=tokenizer,
#     peft_config=get_peft_config(model_args),
#     callbacks=get_callbacks(training_args, model_args),
# )
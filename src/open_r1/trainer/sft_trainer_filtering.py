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


class SFTTrainer_Filter(SFTTrainer):
    """
    Custom SFTTrainer that collects gradient statistics per sample for data quality filtering.
    Based on the principle: higher-quality data → lower nuclear norms + higher effective ranks
    """
    
    def __init__(self, *args, data_quality_threshold=None, save_steps=500, quality_output_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize metrics storage for gradient statistics
        self._metrics = {
            "train": {},
            "eval": {}
        }
        # Store per-sample gradient quality scores
        self._sample_quality_scores = {}
        
        # Save configuration
        self.save_steps = 500
        self.quality_output_dir = quality_output_dir or "data_quality_assessment"
        os.makedirs(self.quality_output_dir, exist_ok=True)
        
        # Optional thresholds for filtering
        self.data_quality_threshold = data_quality_threshold or {
            'nuclear_norm_percentile': 70,  # Keep samples below 70th percentile
            'effective_rank_percentile': 30  # Keep samples above 30th percentile
        }
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """
        Full training step with gradient stats collection per sample.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        # Extract sample identifiers
        sample_ids = inputs.get('sample_ids', None)
        
        if sample_ids is not None:
            # Convert tensors to list if needed
            if isinstance(sample_ids, torch.Tensor):
                sample_ids = sample_ids.tolist()
            # Ensure they're integers (dataset indices)
            sample_ids = [int(sid) for sid in sample_ids]
        else:
            # Create fallback IDs based on step
            batch_size = len(inputs['input_ids']) if 'input_ids' in inputs else 1
            base_idx = self.state.global_step * batch_size
            sample_ids = list(range(base_idx, base_idx + batch_size))

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
            # Extract gradients for quality assessment
            gradients = self._extract_global_gradients(self.accelerator, self.model)
            mode = "train" if self.model.training else "eval"
            
            # Collect stats and compute quality scores for these samples
            quality_scores = self._collect_gradient_stats_and_quality_scores(gradients, mode, sample_ids)
            
            # Store quality scores for data filtering
            for sid, score in zip(sample_ids, quality_scores):
                self._sample_quality_scores[sid] = score
            
            # Save checkpoint periodically
            if self.state.global_step > 0 and self.state.global_step % self.save_steps == 0:
                self._save_information()
            
            # Log progress
            if self.state.global_step % 100 == 0:
                self._log_progress()
            
            # Clear gradients WITHOUT updating model (for assessment only)
            self.optimizer.zero_grad()
            
            # Return zero loss since we're not training
            return torch.tensor(0.0, device=loss.device)

    def _log_progress(self):
        """Log progress during quality assessment."""
        if self._sample_quality_scores:
            # Extract the individual metrics from the dictionaries
            nuclear_norms = [score['nuclear_norm'] for score in self._sample_quality_scores.values()]
            effective_ranks = [score['effective_rank'] for score in self._sample_quality_scores.values()]
            
            mean_nuclear = np.mean(nuclear_norms) if nuclear_norms else 0
            mean_effective = np.mean(effective_ranks) if effective_ranks else 0
            
            print(f"[Step {self.state.global_step}] Processed {len(self._sample_quality_scores)} samples")
            print(f"  Nuclear Norm - Mean: {mean_nuclear:.4f}")
            print(f"  Effective Rank - Mean: {mean_effective:.4f}")
    

    def _collect_gradient_stats_and_quality_scores(self, gradients, mode, sample_ids):
        """
        Collect gradient statistics and return nuclear norm and effective rank as separate values.
        """
        print(f"Debug: Starting gradient stats collection on rank {self.accelerator.process_index} in {mode} mode")

        grad_items = [(name, grad) for name, grad in gradients.items() if grad is not None]

        if not grad_items:
            print(f"Debug: No gradients found on rank {self.accelerator.process_index}.")
            # Return dict with both metrics for each sample
            return [{'nuclear_norm': 0.0, 'effective_rank': 0.0} for _ in sample_ids]

        # Aggregate metrics across important layers
        total_nuclear_norm = 0.0
        total_effective_rank = 0.0
        layer_count = 0

        for name, grad in grad_items:
            if grad is None or grad.ndim < 2:
                continue
                
            # Focus on important layers
            if not any(key in name.lower() for key in ['attention', 'mlp', 'dense', 'linear']):
                continue

            try:
                # Compute SVD
                S = torch.linalg.svd(grad.detach(), full_matrices=False).S
                S_sum = S.sum().item()
                p = S / (S_sum + 1e-12)
                effective_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item()
                nuclear_norm = S_sum
                
                total_nuclear_norm += nuclear_norm
                total_effective_rank += effective_rank
                layer_count += 1

            except Exception as e:
                print(f"[Warning] SVD failed for {name} at step {self.state.global_step}: {e}")

        # Return both metrics separately for each sample
        sample_metrics = []
        if layer_count > 0:
            avg_nuclear_norm = total_nuclear_norm / layer_count
            avg_effective_rank = total_effective_rank / layer_count
            
            for sid in sample_ids:
                sample_metrics.append({
                    'nuclear_norm': avg_nuclear_norm,
                    'effective_rank': avg_effective_rank
                })
        else:
            for sid in sample_ids:
                sample_metrics.append({
                    'nuclear_norm': 0.0,
                    'effective_rank': 0.0
                })
        
        return sample_metrics

    def get_high_quality_sample_ids(self, top_k_percent=50, 
                                use_nuclear_norm=True,
                                use_effective_rank=True):
        """
        Get high-quality samples based on nuclear norm and/or effective rank.
        Lower nuclear norm = better, Higher effective rank = better
        """
        if not self._sample_quality_scores:
            print("No quality scores computed yet!")
            return []
        
        # Extract metrics
        sample_ids = list(self._sample_quality_scores.keys())
        nuclear_norms = [self._sample_quality_scores[sid]['nuclear_norm'] for sid in sample_ids]
        effective_ranks = [self._sample_quality_scores[sid]['effective_rank'] for sid in sample_ids]
        
        # Create combined score based on selected metrics
        scores = {}
        for i, sid in enumerate(sample_ids):
            score = 0
            if use_nuclear_norm and use_effective_rank:
                # Normalize both metrics for fair comparison
                nn_normalized = 1 - (nuclear_norms[i] - min(nuclear_norms)) / (max(nuclear_norms) - min(nuclear_norms) + 1e-6)
                er_normalized = (effective_ranks[i] - min(effective_ranks)) / (max(effective_ranks) - min(effective_ranks) + 1e-6)
                score = (nn_normalized + er_normalized) / 2
            elif use_nuclear_norm:
                # Lower is better, so invert
                score = 1 / (nuclear_norms[i] + 1e-6)
            elif use_effective_rank:
                # Higher is better
                score = effective_ranks[i]
            scores[sid] = score
        
        # Sort by score
        sorted_samples = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Keep top k%
        k = int(len(sorted_samples) * (top_k_percent / 100))
        high_quality_ids = [sid for sid, _ in sorted_samples[:k]]
        
        # Log statistics
        print(f"\nQuality score statistics:")
        print(f"  Nuclear Norm - Mean: {np.mean(nuclear_norms):.4f}, Std: {np.std(nuclear_norms):.4f}")
        print(f"  Effective Rank - Mean: {np.mean(effective_ranks):.4f}, Std: {np.std(effective_ranks):.4f}")
        print(f"  Keeping top {top_k_percent}% ({k}/{len(sorted_samples)} samples)")
        
        return high_quality_ids

    def export_quality_scores(self, output_dir="data_quality_assessment"):
        """
        Export nuclear norm and effective rank to separate files.
        """
        import json
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Separate the metrics
        nuclear_norms = {}
        effective_ranks = {}
        
        for sample_id, metrics in self._sample_quality_scores.items():
            nuclear_norms[sample_id] = metrics['nuclear_norm']
            effective_ranks[sample_id] = metrics['effective_rank']
        
        # Save nuclear norms
        nuclear_path = os.path.join(output_dir, "nuclear_norms.json")
        with open(nuclear_path, 'w') as f:
            json.dump({
                'values': nuclear_norms,
                'statistics': {
                    'mean': float(np.mean(list(nuclear_norms.values()))),
                    'std': float(np.std(list(nuclear_norms.values()))),
                    'min': float(np.min(list(nuclear_norms.values()))),
                    'max': float(np.max(list(nuclear_norms.values()))),
                    'count': len(nuclear_norms)
                }
            }, f, indent=2)
        
        # Save effective ranks
        effective_path = os.path.join(output_dir, "effective_ranks.json")
        with open(effective_path, 'w') as f:
            json.dump({
                'values': effective_ranks,
                'statistics': {
                    'mean': float(np.mean(list(effective_ranks.values()))),
                    'std': float(np.std(list(effective_ranks.values()))),
                    'min': float(np.min(list(effective_ranks.values()))),
                    'max': float(np.max(list(effective_ranks.values()))),
                    'count': len(effective_ranks)
                }
            }, f, indent=2)
        
        print(f"Metrics exported to {output_dir}/")
        print(f"  Nuclear norms saved to: {nuclear_path}")
        print(f"  Effective ranks saved to: {effective_path}")
        print(f"  Nuclear Norm - Mean: {np.mean(list(nuclear_norms.values())):.4f}")
        print(f"  Effective Rank - Mean: {np.mean(list(effective_ranks.values())):.4f}")

    def _save_information(self):
        """
        Save checkpoint with nuclear norm and effective rank in separate files.
        """
        import json
        
        # Separate the metrics
        nuclear_norms = {}
        effective_ranks = {}
        
        for sample_id, metrics in self._sample_quality_scores.items():
            nuclear_norms[sample_id] = metrics['nuclear_norm']
            effective_ranks[sample_id] = metrics['effective_rank']
        
        # Save nuclear norms checkpoint
        nuclear_path = os.path.join(
            self.quality_output_dir, 
            f"nuclear_norms_step_{self.state.global_step}.json"
        )
        with open(nuclear_path, 'w') as f:
            json.dump({
                'step': self.state.global_step,
                'values': nuclear_norms,
                'statistics': {
                    'mean': float(np.mean(list(nuclear_norms.values()))),
                    'std': float(np.std(list(nuclear_norms.values()))),
                    'min': float(np.min(list(nuclear_norms.values()))),
                    'max': float(np.max(list(nuclear_norms.values())))
                }
            }, f, indent=2)
        
        # Save effective ranks checkpoint
        effective_path = os.path.join(
            self.quality_output_dir,
            f"effective_ranks_step_{self.state.global_step}.json"
        )
        with open(effective_path, 'w') as f:
            json.dump({
                'step': self.state.global_step,
                'values': effective_ranks,
                'statistics': {
                    'mean': float(np.mean(list(effective_ranks.values()))),
                    'std': float(np.std(list(effective_ranks.values()))),
                    'min': float(np.min(list(effective_ranks.values()))),
                    'max': float(np.max(list(effective_ranks.values())))
                }
            }, f, indent=2)
        
        print(f"[Step {self.state.global_step}] Saved checkpoints:")
        print(f"  Nuclear norms: {nuclear_path}")
        print(f"  Effective ranks: {effective_path}")
        print(f"  Samples processed: {len(self._sample_quality_scores)}")

    # Keep other methods unchanged...
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
                    gradients[name] = full_grad.clone()
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
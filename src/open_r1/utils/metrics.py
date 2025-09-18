import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any, List, Callable
from pathlib import Path
import json
import os
import math

class MetricsComputer:
    """
    Configurable metrics computer with per-metric intervals and external selection.
    """
    def __init__(self, 
                 metric_config: Dict[str, Dict[str, Any]] = None,
                 svd_min_params: int = 1000,
                 output_dir: str = "metrics"):
        """
        Args:
            metric_config: Dictionary specifying which metrics to compute and their intervals
            svd_min_params: Minimum parameters for SVD computation
            output_dir: Directory to save metrics
        """
        self.svd_min_params = svd_min_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.default_config = {
            "layer_gradient_distribution": {"enabled": True, "interval": 1},
            "effective_rank": {"enabled": True, "interval": 10},
            "nuclear_norm": {"enabled": True, "interval": 10},
            "reasoning_emergence": {"enabled": True, "interval": 5},
            "global_gradient": {"enabled": True, "interval": 1},
        }
        
        # Merge provided config with defaults
        self.metric_config = self.default_config.copy()
        if metric_config:
            self.metric_config.update(metric_config)
        
        # Map metric names to their compute functions
        self.metric_functions = {
            "layer_gradient_distribution": self.compute_layer_gradient_distribution,
            "effective_rank": self.compute_effective_rank_metrics,
            "nuclear_norm": self.compute_nuclear_norm_metrics,
            "reasoning_emergence": self.compute_reasoning_emergence_metrics,
            "global_gradient": self.compute_global_gradient_metrics,
        }
        
        # Track previous states for temporal metrics
        self.prev_global_gradient = None
        self.prev_nuclear_norms = None
        self.step_count = 0
        
        # Register for custom metrics
        self.custom_metric_functions = {}
        
    def register_custom_metric(self, 
                              name: str, 
                              function: Callable,
                              interval: int = 1,
                              enabled: bool = True):
        """
        Register a custom metric function from outside the class.
        
        Args:
            name: Name of the metric
            function: Callable that takes (gradients, model, **kwargs) and returns Dict[str, float]
            interval: How often to compute this metric
            enabled: Whether to compute this metric
        """
        self.custom_metric_functions[name] = function
        self.metric_config[name] = {"enabled": enabled, "interval": interval}
        self.metric_functions[name] = function
    
    def set_metric_config(self, metric_config: Dict[str, Dict[str, Any]]):
        """
        Update metric configuration at runtime.
        
        Args:
            metric_config: New configuration dictionary
        """
        self.metric_config.update(metric_config)
    
    def enable_metrics(self, metric_names: List[str]):
        """Enable specific metrics."""
        for name in metric_names:
            if name in self.metric_config:
                self.metric_config[name]["enabled"] = True
    
    def disable_metrics(self, metric_names: List[str]):
        """Disable specific metrics."""
        for name in metric_names:
            if name in self.metric_config:
                self.metric_config[name]["enabled"] = False
    
    def set_metric_interval(self, metric_name: str, interval: int):
        """Set computation interval for a specific metric."""
        if metric_name in self.metric_config:
            self.metric_config[metric_name]["interval"] = interval
    
    def should_compute_metric(self, metric_name: str, step: int) -> bool:
        """
        Check if a metric should be computed at this step.
        
        Args:
            metric_name: Name of the metric
            step: Current training step
            
        Returns:
            True if metric should be computed
        """
        if metric_name not in self.metric_config:
            return False
        
        config = self.metric_config[metric_name]
        if not config.get("enabled", True):
            return False
        
        interval = config.get("interval", 1)
        return step % interval == 0
    
    def compute_all_metrics(self,
                           gradients: Dict[str, torch.Tensor],
                           model: nn.Module,
                           loss: float,
                           rewards: Optional[torch.Tensor] = None,
                           hidden_states: Optional[torch.Tensor] = None,
                           training_mode: str = "answer_only",
                           step: int = 0,
                           **additional_inputs) -> Dict[str, Any]:
        """
        Computes all enabled metrics based on their intervals.
        
        Args:
            gradients: Dictionary mapping parameter names to gradient tensors
            model: The model being trained
            loss: Current loss value
            rewards: Optional reward tensor for this batch
            hidden_states: Optional hidden states [batch, seq_len, hidden_dim]
            training_mode: "answer_only" or "reasoning_answer"
            step: Current training step
            **additional_inputs: Additional inputs for custom metrics
            
        Returns:
            Dictionary containing computed metrics
        """
        self.step_count = step
        metrics = {
            "step": step,
            "mode": training_mode,
            "loss": loss
        }
        print(f"Computing metrics at step {step}...")
        # Compute each metric based on its configuration
        for metric_name, metric_func in self.metric_functions.items():
            if self.should_compute_metric(metric_name, step):
                # Prepare arguments based on metric requirements
                kwargs = {
                    "gradients": gradients,
                    "model": model,
                    "rewards": rewards,
                    "hidden_states": hidden_states,
                    "step": step,
                    **additional_inputs
                }
                
                # Call metric function with appropriate arguments
                if metric_name == "layer_gradient_distribution":
                    metric_results = metric_func(gradients)
                elif metric_name == "effective_rank":
                    metric_results = metric_func(gradients)
                elif metric_name == "reasoning_emergence":
                    if hidden_states is not None:
                        metric_results = metric_func(hidden_states, rewards)
                    else:
                        metric_results = {}
                elif metric_name == "global_gradient":
                    metric_results = metric_func(gradients, rewards)
                elif metric_name == "temporal":
                    metric_results = metric_func(gradients)
                elif metric_name == "nuclear_norm":
                    metric_results = metric_func(gradients)
                else:
                    # Custom metric - pass all kwargs
                    metric_results = metric_func(**kwargs)
                
                metrics.update(metric_results)
                    
        
        # Save metrics if any were computed
        if len(metrics) > 3:  # More than just step, mode, and loss
            self.save_metrics(metrics, training_mode, step)
        
        return metrics
    
    # ============= METRIC 1: Layer-wise Gradient Distribution =============
    
    def compute_layer_gradient_distribution(self, 
                                           gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Computes gradient distribution metrics across layers.
        """
        metrics = {}
        print("Computing layer gradient distribution metrics...")
        # Group gradients by layer depth
        layer_groups = self._group_gradients_by_depth(gradients)
        
        print(f"Found {len(layer_groups)} layer groups for gradient distribution analysis.")
        # Compute statistics per layer group
        layer_norms = {}
        layer_means = {}
        layer_stds = {}
        
        for depth, grads_list in layer_groups.items():
            if not grads_list:
                continue
            
            # Concatenate all gradients at this depth
            concat_grad = torch.cat([g.flatten() for g in grads_list])
            
            # Basic statistics
            norm = concat_grad.norm().item()
            mean = concat_grad.mean().item()
            std = concat_grad.std().item()
            
            layer_norms[depth] = norm
            layer_means[depth] = mean
            layer_stds[depth] = std
            
            # Per-layer metrics
            metrics[f"layer_{depth}/grad_norm"] = norm
            metrics[f"layer_{depth}/grad_mean"] = mean
            metrics[f"layer_{depth}/grad_std"] = std
            
            # Signal-to-noise ratio
            snr = abs(mean) / (std + 1e-10)
            metrics[f"layer_{depth}/grad_snr"] = snr
        
        # Compute cross-layer statistics
        if len(layer_norms) >= 2:
            sorted_depths = sorted(layer_norms.keys())
            
            # Early vs late layer gradient ratio
            mid_point = len(sorted_depths) // 2
            early_depths = sorted_depths[:mid_point]
            late_depths = sorted_depths[mid_point:]
            
            early_norm = np.mean([layer_norms[d] for d in early_depths]) if early_depths else 0
            late_norm = np.mean([layer_norms[d] for d in late_depths]) if late_depths else 0
            
            metrics["gradient_flow_ratio"] = early_norm / (late_norm + 1e-10)
            
            # Gradient norm variance across layers
            all_norms = list(layer_norms.values())
            metrics["cross_layer_norm_variance"] = np.var(all_norms)
            metrics["cross_layer_norm_std"] = np.std(all_norms)
            
            # Gradient concentration (Gini coefficient)
            metrics["gradient_concentration_gini"] = self._compute_gini_coefficient(all_norms)
        
        return metrics
    
    # ============= METRIC 2: Effective Rank Evolution =============
    
    def compute_effective_rank_metrics(self, 
                                       gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Computes effective rank and related SVD metrics.
        """
        print("Computing effective rank metrics...")
        metrics = {}
        layer_ranks = []
        layer_conditions = []
        
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            # Skip if too small or wrong shape
            if grad.ndim < 2 or grad.numel() < self.svd_min_params:
                continue
            
            # Reshape to 2D for SVD
            grad_2d = grad.view(grad.shape[0], -1)
            
            # Compute singular values
            S = torch.linalg.svdvals(grad_2d)
            S = S[S > 1e-10]  # Filter near-zero values
            
            if len(S) == 0:
                continue
            
            # Effective rank (entropy-based)
            S_normalized = S / S.sum()
            eff_rank = torch.exp(-(S_normalized * torch.log(S_normalized + 1e-12)).sum()).item()
            
            # Store layer-specific metrics
            metrics[f"svd/{name}/effective_rank"] = eff_rank
            metrics[f"svd/{name}/rank_ratio"] = eff_rank / min(grad_2d.shape)
            
            # Condition number
            condition = (S[0] / S[-1]).item()
            metrics[f"svd/{name}/condition_number"] = condition
            
            # Spectral decay rate
            if len(S) > 1:
                indices = torch.arange(len(S), dtype=torch.float32, device=S.device)
                log_S = torch.log(S + 1e-10)
                decay_rate = torch.corrcoef(torch.stack([indices, log_S]))[0, 1].item()
                metrics[f"svd/{name}/spectral_decay_rate"] = abs(decay_rate)
            
            # Nuclear norm
            metrics[f"svd/{name}/nuclear_norm"] = S.sum().item()
            
            # Top-k energy concentration
            k = min(10, len(S))
            top_k_energy = S[:k].sum() / S.sum()
            metrics[f"svd/{name}/top_10_energy_ratio"] = top_k_energy.item()
            
            layer_ranks.append(eff_rank)
            layer_conditions.append(condition)
                
        
        # Global SVD statistics
        if layer_ranks:
            metrics["global/mean_effective_rank"] = np.mean(layer_ranks)
            metrics["global/std_effective_rank"] = np.std(layer_ranks)
            metrics["global/min_effective_rank"] = np.min(layer_ranks)
            metrics["global/max_effective_rank"] = np.max(layer_ranks)
            metrics["global/rank_diversity"] = np.std(layer_ranks) / (np.mean(layer_ranks) + 1e-10)
            metrics["global/mean_condition_number"] = np.mean(layer_conditions)
            metrics["global/max_condition_number"] = np.max(layer_conditions)
        
        return metrics
    
    # ============= METRIC 3: Reasoning Chain Emergence =============
    
    def compute_reasoning_emergence_metrics(self,
                                           hidden_states: torch.Tensor,
                                           rewards: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Analyzes hidden states to understand reasoning chain development.
        """
        metrics = {}
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Sample positions across sequence
        num_samples = min(10, seq_len)
        sample_positions = np.linspace(0, seq_len - 1, num_samples, dtype=int)
        
        position_ranks = []
        position_norms = []
        position_variances = []
        
        for pos in sample_positions:
            h_pos = hidden_states[:, pos, :]
            
            # Norm statistics
            norm_mean = h_pos.norm(dim=-1).mean().item()
            position_norms.append(norm_mean)
            
            # Variance across batch
            variance = h_pos.var(dim=0).mean().item()
            position_variances.append(variance)
            
            # Effective rank at this position
            if batch_size > 1:
                try:
                    S = torch.linalg.svdvals(h_pos)
                    S = S[S > 1e-10]
                    if len(S) > 0:
                        S_normalized = S / S.sum()
                        eff_rank = torch.exp(-(S_normalized * torch.log(S_normalized + 1e-12)).sum()).item()
                        position_ranks.append(eff_rank)
                except:
                    pass
        
        # Position-based metrics
        if position_norms:
            relative_positions = sample_positions / (seq_len - 1)
            
            norm_growth = position_norms[-1] - position_norms[0]
            metrics["hidden/norm_growth"] = norm_growth
            
            peak_idx = np.argmax(position_norms)
            metrics["hidden/norm_peak_position"] = relative_positions[peak_idx]
            
            if position_variances:
                var_growth = position_variances[-1] - position_variances[0]
                metrics["hidden/variance_growth"] = var_growth
                metrics["hidden/mean_variance"] = np.mean(position_variances)
        
        if position_ranks:
            metrics["hidden/mean_position_rank"] = np.mean(position_ranks)
            metrics["hidden/rank_growth"] = position_ranks[-1] - position_ranks[0]
            
            rank_slope, _ = np.polyfit(relative_positions[:len(position_ranks)], position_ranks, 1)
            metrics["hidden/information_accumulation_rate"] = rank_slope
        
        # Correlation with rewards
        if rewards is not None and rewards.numel() == batch_size:
            final_hidden = hidden_states[:, -1, :]
            final_norms = final_hidden.norm(dim=-1)
            
            correlation = torch.corrcoef(torch.stack([final_norms, rewards]))[0, 1].item()
            metrics["hidden/final_norm_reward_correlation"] = correlation
            
            median_reward = rewards.median()
            high_reward_hidden = final_hidden[rewards > median_reward]
            low_reward_hidden = final_hidden[rewards <= median_reward]
            
            if len(high_reward_hidden) > 0 and len(low_reward_hidden) > 0:
                high_centroid = high_reward_hidden.mean(dim=0)
                low_centroid = low_reward_hidden.mean(dim=0)
                cosine_dist = 1 - torch.nn.functional.cosine_similarity(
                    high_centroid.unsqueeze(0), 
                    low_centroid.unsqueeze(0)
                ).item()
                metrics["hidden/reward_representation_separation"] = cosine_dist
        
        return metrics
    
    # ============= METRIC 4: Global Gradient Metrics =============  
    @torch.no_grad()
    def compute_global_gradient_metrics(
        self,
        gradients: Dict[str, torch.Tensor],
        rewards: Optional[torch.Tensor] = None,
        *,
        sparsity_mode: str = "relative",   # "relative" or "absolute"
        sparsity_tau: float = 1e-3,        # relative threshold factor
        absolute_thresh: float = 1e-6,     # used if sparsity_mode == "absolute"      
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Collect grads (detached, 1D, finite)
        flat_grads = []
        total_params = 0
        n_finite = 0

        for _, g in gradients.items():
            if g is None:
                continue
            g = g.detach().reshape(-1)
            finite_mask = torch.isfinite(g)
            if finite_mask.any():
                gf = g[finite_mask]
                flat_grads.append(gf)
                total_params += g.numel()
                n_finite += gf.numel()
            else:
                total_params += g.numel()

        if not flat_grads:
            return metrics  # nothing to report

        # Concatenate once (simple, but memory heavy)
        g = torch.cat(flat_grads, dim=0)
        abs_g = g.abs()

        grad_mean = g.mean().item()
        grad_std = g.std(unbiased=False).item()
        grad_abs_mean = abs_g.mean().item()
        grad_rms = torch.sqrt((g * g).mean()).item()
        grad_l2 = g.norm(2).item()
        grad_l1 = abs_g.sum().item()

        metrics["global/total_parameters"] = float(total_params)
        metrics["global/finite_params"] = float(n_finite)
        metrics["global/grad_norm"] = grad_l2
        metrics["global/grad_mean"] = grad_mean
        metrics["global/grad_std"] = grad_std
        metrics["global/grad_abs_mean"] = grad_abs_mean
        metrics["global/grad_rms"] = grad_rms
        metrics["global/grad_l1"] = grad_l1
        metrics["global/grad_l2"] = grad_l2

        # Sparsity
        if sparsity_mode == "relative":
            tau = sparsity_tau * max(grad_rms, 1e-12)
        else:
            tau = absolute_thresh
        metrics["global/grad_sparsity"] = (abs_g < tau).float().mean().item()
        metrics["global/grad_sparsity_thresh"] = float(tau)

        # Entropy over |g|
        s = abs_g.sum()
        if s > 0:
            p = abs_g / s
            entropy = -(p * (p + 1e-12).log()).sum().item()
            max_entropy = math.log(max(int(p.numel()), 1))
            metrics["global/grad_entropy"] = entropy
            metrics["global/grad_entropy_ratio"] = (entropy / max_entropy) if max_entropy > 0 else 0.0

        # Reward stats & proper alignment scaffold
        if rewards is not None and rewards.numel() > 0:
            r = rewards.detach()
            metrics["reward/mean"] = r.mean().item()
            metrics["reward/std"] = (r.std(unbiased=False).item() if r.numel() > 1 else 0.0)

            # NOTE: True alignment requires a windowed correlation (time series) or per-sample norms.
            # Example placeholder keys to be filled elsewhere:
            # metrics["align/grad_reward_corr_window"] = self._corr_gradnorm_reward_window  # update externally
            # metrics["align/per_sample_corr"] = self._corr_per_sample if available

        return metrics

    # ============= METRIC 5: Nuclear Norm Metrics =============

    def compute_nuclear_norm_metrics(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute comprehensive nuclear norm metrics for gradient analysis.
        Nuclear norm (sum of singular values) captures the total "energy" or "capacity" of gradients.
        """
        metrics = {}
        current_nuclear = {}
        print("Computing effective nuclear norm metrics...")
        # Group gradients by layer depth for cross-layer analysis
        layer_groups = self._group_gradients_by_depth(gradients)
        layer_nuclear_norms = {}
        layer_stable_ranks = {}
        
        # Per-parameter nuclear norm analysis
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            # Handle different tensor dimensions
            if grad.ndim < 2:
                # For 1D tensors, nuclear norm is L1 norm
                nuclear = grad.abs().sum().item()
                metrics[f"nuclear/{name}/norm"] = nuclear
                metrics[f"nuclear/{name}/norm_per_param"] = nuclear / grad.numel()
                current_nuclear[name] = nuclear
                continue
            
            # Skip if too small
            if grad.numel() < self.svd_min_params:
                continue
            
            # Reshape to 2D for SVD
            original_shape = grad.shape
            if grad.ndim > 2:
                # Reshape strategy based on layer type
                if len(original_shape) == 4:  # Conv: [out_ch, in_ch, k, k]
                    grad_2d = grad.view(original_shape[0], -1)
                elif len(original_shape) == 3:  # Attention: [heads, seq, dim]
                    grad_2d = grad.view(original_shape[0], -1)
                else:
                    grad_2d = grad.view(grad.shape[0], -1)
            else:
                grad_2d = grad

            # Compute singular values
            S = torch.linalg.svdvals(grad_2d)
            S = S[S > 1e-10]  # Filter numerical zeros
            
            if len(S) == 0:
                continue
            
            # Basic nuclear norm
            nuclear_norm = S.sum().item()
            metrics[f"nuclear/{name}/norm"] = nuclear_norm
            current_nuclear[name] = nuclear_norm
            
            # Normalized versions
            frobenius_norm = grad.norm().item()
            if frobenius_norm > 0:
                # Nuclear to Frobenius ratio (indicates rank deficiency)
                metrics[f"nuclear/{name}/to_frobenius_ratio"] = nuclear_norm / frobenius_norm
            
            # Nuclear norm density (per parameter)
            metrics[f"nuclear/{name}/norm_density"] = nuclear_norm / grad.numel()
            
            # Stable rank: (nuclear_norm)² / (frobenius_norm)²
            # More stable than effective rank, provides lower bound
            stable_rank = (nuclear_norm ** 2) / (frobenius_norm ** 2 + 1e-10)
            metrics[f"nuclear/{name}/stable_rank"] = stable_rank
            
            # Spectral norm (largest singular value)
            spectral_norm = S[0].item()
            metrics[f"nuclear/{name}/spectral_norm"] = spectral_norm
            
            # Nuclear stability: nuclear_norm / spectral_norm
            # High value means many singular values contribute
            nuclear_stability = nuclear_norm / (spectral_norm + 1e-10)
            metrics[f"nuclear/{name}/stability"] = nuclear_stability
            
            # Nuclear norm concentration (how concentrated in top-k values)
            if len(S) > 1:
                cumsum_S = torch.cumsum(S, dim=0)
                
                # Top-k ratios
                for k in [1, 5, 10]:
                    if k < len(S):
                        top_k_ratio = cumsum_S[k-1].item() / nuclear_norm
                        metrics[f"nuclear/{name}/top_{k}_concentration"] = top_k_ratio
                
                # Nuclear Gini coefficient (concentration measure)
                n = len(S)
                lorenz = cumsum_S / nuclear_norm
                auc = torch.trapz(lorenz, torch.linspace(0, 1, n, device=S.device)).item()
                nuclear_gini = 1 - 2 * auc
                metrics[f"nuclear/{name}/gini_coefficient"] = nuclear_gini
            
            # Effective nuclear dimension (entropy-based)
            S_normalized = S / nuclear_norm
            nuclear_entropy = -(S_normalized * torch.log(S_normalized + 1e-12)).sum().item()
            eff_nuclear_dim = torch.exp(torch.tensor(nuclear_entropy)).item()
            metrics[f"nuclear/{name}/effective_dimension"] = eff_nuclear_dim
            
            # Store layer-wise aggregates
            depth = self._extract_layer_depth(name)
            if depth not in layer_nuclear_norms:
                layer_nuclear_norms[depth] = []
                layer_stable_ranks[depth] = []
            layer_nuclear_norms[depth].append(nuclear_norm)
            layer_stable_ranks[depth].append(stable_rank)
                
        
        # Cross-layer nuclear norm analysis
        if layer_nuclear_norms:
            # Aggregate per layer
            layer_totals = {}
            for depth, norms in layer_nuclear_norms.items():
                total = sum(norms)
                mean = np.mean(norms)
                layer_totals[depth] = total
                
                metrics[f"nuclear/layer_{depth}/total"] = total
                metrics[f"nuclear/layer_{depth}/mean"] = mean
                
                if depth in layer_stable_ranks:
                    metrics[f"nuclear/layer_{depth}/mean_stable_rank"] = np.mean(layer_stable_ranks[depth])
            
            # Global cross-layer statistics
            if len(layer_totals) > 1:
                all_totals = list(layer_totals.values())
                metrics["nuclear/global/total"] = sum(all_totals)
                metrics["nuclear/global/mean_layer_total"] = np.mean(all_totals)
                metrics["nuclear/global/std_layer_total"] = np.std(all_totals)
                
                # Early vs late layer ratio
                sorted_depths = sorted(layer_totals.keys())
                mid = len(sorted_depths) // 2
                early_depths = sorted_depths[:mid]
                late_depths = sorted_depths[mid:]
                
                early_nuclear = sum(layer_totals[d] for d in early_depths) / max(1, len(early_depths))
                late_nuclear = sum(layer_totals[d] for d in late_depths) / max(1, len(late_depths))
                
                metrics["nuclear/gradient_flow_ratio"] = early_nuclear / (late_nuclear + 1e-10)
                
                # Layer-wise concentration
                sorted_totals = sorted(all_totals)
                cumsum = np.cumsum(sorted_totals)
                total = cumsum[-1]
                if total > 0:
                    lorenz = cumsum / total
                    n = len(lorenz)
                    auc = np.trapz(lorenz, np.linspace(0, 1, n))
                    gini = 1 - 2 * auc
                    metrics["nuclear/layer_concentration_gini"] = gini
        
        # Temporal nuclear norm changes
        if self.prev_nuclear_norms is not None:
            # Compare with previous step
            common_keys = set(current_nuclear.keys()) & set(self.prev_nuclear_norms.keys())
            
            if common_keys:
                changes = []
                relative_changes = []
                
                for key in common_keys:
                    curr = current_nuclear[key]
                    prev = self.prev_nuclear_norms[key]
                    
                    change = curr - prev
                    rel_change = abs(change) / (prev + 1e-10)
                    
                    changes.append(change)
                    relative_changes.append(rel_change)
                
                if changes:
                    metrics["nuclear/temporal/mean_change"] = np.mean(changes)
                    metrics["nuclear/temporal/max_increase"] = max(changes)
                    metrics["nuclear/temporal/max_decrease"] = min(changes)
                    metrics["nuclear/temporal/mean_relative_change"] = np.mean(relative_changes)
                    
                    # Momentum (consistent direction)
                    positive = sum(1 for c in changes if c > 0)
                    momentum = (positive / len(changes)) * 2 - 1  # Range [-1, 1]
                    metrics["nuclear/temporal/momentum"] = momentum
        
        # Update history
        self.prev_nuclear_norms = current_nuclear
        
        return metrics
    
    # ============= Helper Methods (unchanged) =============
    
    def _group_gradients_by_depth(self, gradients: Dict[str, torch.Tensor]) -> Dict[int, List[torch.Tensor]]:
        """Groups gradients by layer depth."""
        layer_groups = {}
        
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            depth = self._extract_layer_depth(name)
            
            if depth not in layer_groups:
                layer_groups[depth] = []
            layer_groups[depth].append(grad)
        
        return layer_groups
    
    def _extract_layer_depth(self, param_name: str) -> int:
        """Extracts layer depth from parameter name."""
        if 'layers' in param_name or 'layer' in param_name:
            parts = param_name.split('.')
            for i, part in enumerate(parts):
                if 'layer' in part and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        pass
        
        if 'embed' in param_name.lower():
            return 0
        
        if any(x in param_name.lower() for x in ['head', 'output', 'final', 'lm_head']):
            return 999
        
        return hash(param_name.split('.')[0]) % 100
    
    def _compute_gini_coefficient(self, values: List[float]) -> float:
        """Computes Gini coefficient."""
        if not values or len(values) == 1:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
    
    def save_metrics(self, metrics: Dict[str, Any], training_mode: str, step: int):
        """Saves metrics to disk."""
        step_dir = self.output_dir / training_mode / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = step_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    clean_metrics[k] = v.item() if v.numel() == 1 else v.tolist()
                elif isinstance(v, np.ndarray):
                    clean_metrics[k] = v.item() if v.size == 1 else v.tolist()
                else:
                    clean_metrics[k] = v
            
            json.dump(clean_metrics, f, indent=2)

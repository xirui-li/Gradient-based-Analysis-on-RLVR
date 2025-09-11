import torch

from .grpo_trainer import GRPOTrainer

class GRPOTrainerMonitor(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization for monitoring can be added here

    def _collect_gradient_stats_by_layers(self, gradients, mode):
        """
        Enhanced gradient statistics collection for RLVR phenomena monitoring.
        Computes SVD and phenomena metrics continuously for eligible layers.
        """
        if not self.accelerator.is_main_process:
            return
        
        step = int(getattr(self.state, "global_step", 0))
        grad_items = [(name, grad) for name, grad in gradients.items() if grad is not None]
        
        if not grad_items:
            print(f"Debug: No gradients found on rank {self.accelerator.process_index}. This is expected under ZeRO-3.")
            return

        step_grad_stats = {}
        
        # Initialize RLVR phenomena tracking
        if not hasattr(self, "_gradient_history"):
            from collections import deque
            self._gradient_history = deque(maxlen=100)  # For aha moment detection
            self._effective_rank_history = deque(maxlen=50)
            self._gradient_diversity_history = deque(maxlen=50)
            self._entropy_history = deque(maxlen=50)
            self._last_gradients = None  # For temporal correlation
            self._min_svd_params = 1000  # Minimum params for SVD computation
            
        # Collect all gradients for global analysis
        all_grads = []
        layer_effective_ranks = []
        
        for name, grad in grad_items:
            if grad is None:
                continue
                
            flat_grad = grad.flatten()
            all_grads.append(flat_grad)
            
            # Basic stats (always computed)
            M_mean = grad.mean().item()
            M_max = grad.max().item()
            M_min = grad.min().item()
            frobenius_norm = torch.linalg.norm(grad).item()
            
            param_prefix = f"grad_stats/params/{name}"
            
            # Save basic metrics
            self._metrics[mode][f"{param_prefix}/M_mean"].append(M_mean)
            self._metrics[mode][f"{param_prefix}/M_max"].append(M_max)
            self._metrics[mode][f"{param_prefix}/M_min"].append(M_min)
            self._metrics[mode][f"{param_prefix}/frobenius_norm"].append(frobenius_norm)
            
            step_grad_stats[f"{param_prefix}/M_mean"] = M_mean
            step_grad_stats[f"{param_prefix}/M_max"] = M_max
            step_grad_stats[f"{param_prefix}/M_min"] = M_min
            step_grad_stats[f"{param_prefix}/frobenius_norm"] = frobenius_norm
            
            # Compute SVD for eligible layers (2D+ and large enough)
            if grad.ndim >= 2 and grad.numel() >= self._min_svd_params and step % 10 == 0:
                try:
                    S = torch.linalg.svd(grad.detach(), full_matrices=False).S
                    S_sum = S.sum().item()
                    p = S / (S_sum + 1e-12)
                    effective_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item()
                    layer_effective_ranks.append(effective_rank)
                    
                    # SVD-derived metrics
                    nuclear_norm = S_sum
                    S_max = S.max().item()
                    S_min = S.min().item()
                    condition_number = (S_max / (S_min + 1e-12))
                    
                    # Save SVD metrics
                    self._metrics[mode][f"{param_prefix}/nuclear_norm"].append(nuclear_norm)
                    self._metrics[mode][f"{param_prefix}/effective_rank"].append(effective_rank)
                    self._metrics[mode][f"{param_prefix}/condition_number"].append(condition_number)
                    
                    step_grad_stats[f"{param_prefix}/nuclear_norm"] = nuclear_norm
                    step_grad_stats[f"{param_prefix}/effective_rank"] = effective_rank
                    step_grad_stats[f"{param_prefix}/condition_number"] = condition_number
                    
                except Exception as e:
                    print(f"[Warning] SVD failed for {name} at step {step}: {e}")
        
        # Global gradient analysis for RLVR phenomena
        if all_grads:
            global_grad = torch.cat(all_grads)
            
            # 1. Aha Moment Detection - Gradient Alignment
            if self._last_gradients is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    global_grad.unsqueeze(0), 
                    self._last_gradients.unsqueeze(0)
                ).item()
                
                self._gradient_history.append(cos_sim)
                step_grad_stats["phenomena/gradient_alignment"] = cos_sim
                
                # Detect sudden orthogonalization
                if len(self._gradient_history) > 10:
                    recent_avg = sum(list(self._gradient_history)[-10:]) / 10
                    historical_avg = sum(list(self._gradient_history)[-50:]) / min(50, len(self._gradient_history))
                    
                    # Aha moment: sudden drop in alignment
                    if cos_sim < recent_avg - 0.3 and cos_sim < historical_avg - 0.2:
                        step_grad_stats["phenomena/aha_moment_detected"] = 1.0
                    else:
                        step_grad_stats["phenomena/aha_moment_detected"] = 0.0
            
            # 2. Overthinking - Gradient Entropy
            abs_grad = global_grad.abs()
            if abs_grad.sum() > 0:
                p = abs_grad / abs_grad.sum()
                entropy = -(p * torch.log(p + 1e-10)).sum().item()
                self._entropy_history.append(entropy)
                step_grad_stats["phenomena/gradient_entropy"] = entropy
                
                # Low entropy growth indicates overthinking
                if len(self._entropy_history) > 20:
                    recent_entropy = sum(list(self._entropy_history)[-10:]) / 10
                    old_entropy = sum(list(self._entropy_history)[-20:-10]) / 10
                    entropy_growth = (recent_entropy - old_entropy) / (old_entropy + 1e-8)
                    step_grad_stats["phenomena/overthinking_score"] = max(0, -entropy_growth) 
            
            # 3. Global Effective Rank for Capability Collapse
            if layer_effective_ranks:
                mean_rank = sum(layer_effective_ranks) / len(layer_effective_ranks)
                self._effective_rank_history.append(mean_rank)
                step_grad_stats["phenomena/mean_effective_rank"] = mean_rank
                
                # Detect decreasing trend
                if len(self._effective_rank_history) > 20:
                    initial_rank = sum(list(self._effective_rank_history)[:10]) / 10
                    recent_rank = sum(list(self._effective_rank_history)[-10:]) / 10
                    rank_decline = (initial_rank - recent_rank) / (initial_rank + 1e-8)
                    
                    step_grad_stats["phenomena/capability_collapse_risk"] = max(0, rank_decline)
            
            # 4. Gradient Diversity for Entropy Collapse
            grad_std = global_grad.std().item()
            grad_mean = global_grad.mean().item()
            cv = grad_std / (abs(grad_mean) + 1e-8)  # Coefficient of variation
            
            self._gradient_diversity_history.append(cv)
            step_grad_stats["phenomena/gradient_diversity_cv"] = cv
            
            if len(self._gradient_diversity_history) > 10:
                recent_cv = sum(list(self._gradient_diversity_history)[-10:]) / 10
                if recent_cv < 0.1:  # Very low diversity
                    step_grad_stats["phenomena/entropy_collapse_risk"] = 1.0
                else:
                    step_grad_stats["phenomena/entropy_collapse_risk"] = 0.0
            
            # 5. Reward Hacking - Gradient-Reward Decorrelation
            if 'reward' in self._metrics.get(mode, {}):
                recent_rewards = self._metrics[mode]['reward'][-10:] if len(self._metrics[mode]['reward']) > 10 else self._metrics[mode]['reward']
                if recent_rewards and len(recent_rewards) > 1:
                    reward_std = torch.tensor(recent_rewards).std().item()
                    grad_norm = global_grad.norm().item()
                    
                    # Normalize both to [0,1] range for comparison
                    if reward_std > 0:
                        reward_signal = (recent_rewards[-1] - min(recent_rewards)) / reward_std
                        grad_signal = grad_norm / (grad_norm + 1.0)  # Sigmoid-like normalization
                        
                        # Anomaly: high reward but low gradient activity
                        if reward_signal > 2.0 and grad_signal < 0.1:
                            step_grad_stats["phenomena/reward_hacking_signal"] = 1.0
                        else:
                            step_grad_stats["phenomena/reward_hacking_signal"] = 0.0
            
            self._last_gradients = global_grad.clone()
        
        # Add cross-layer statistics if we have multiple layers with SVD
        if layer_effective_ranks and len(layer_effective_ranks) > 1:
            rank_variance = torch.tensor(layer_effective_ranks).var().item()
            rank_range = max(layer_effective_ranks) - min(layer_effective_ranks)
            
            step_grad_stats["phenomena/cross_layer_rank_variance"] = rank_variance
            step_grad_stats["phenomena/cross_layer_rank_range"] = rank_range
        
        # Update metrics with all computed stats
        for key, value in step_grad_stats.items():
            if key not in self._metrics[mode]:
                self._metrics[mode][key] = []
            if not isinstance(self._metrics[mode][key], list):
                self._metrics[mode][key] = [self._metrics[mode][key]]
            self._metrics[mode][key].append(value)


    def _save_all_metrics_snapshot(self, mode: str):
        """
        Save comprehensive metrics including RLVR phenomena indicators.
        
        File structure:
        stats/{run_name}/step_{step:08d}/metrics.json
        
        Each file contains metrics from all ranks with phenomena detection summaries.
        """
        import os, json, torch, torch.distributed as dist
        import numpy as np
        from datetime import datetime
        
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        step = self.state.global_step
        
        # Compute phenomena detection flags based on thresholds
        phenomena_summary = self._detect_phenomena(mode)
        
        # Compute aggregated statistics for this step
        step_statistics = self._compute_step_statistics(mode)
        
        # Local payload (ensure JSON-serializable)
        local_entry = {
            "mode": mode,
            "step": int(step),
            "rank": int(rank),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self._recursive_convert(self._metrics.get(mode, {})),
            "phenomena_summary": phenomena_summary,
            "step_statistics": step_statistics,
            "training_state": {
                "learning_rate": self._get_learning_rate() if hasattr(self, '_get_learning_rate') else None,
                "loss": self.state.loss if hasattr(self.state, 'loss') else None,
                "epoch": self.state.epoch if hasattr(self.state, 'epoch') else None,
                "num_input_tokens_seen": self.state.num_input_tokens_seen if hasattr(self.state, 'num_input_tokens_seen') else None,
            }
        }
        
        # Gather variable-length bytes from all ranks
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        arr = np.frombuffer(json.dumps(local_entry).encode("utf-8"), dtype=np.uint8)
        local_bytes = torch.from_numpy(arr).to(dev)
        
        local_size = torch.tensor([local_bytes.numel()], dtype=torch.int64, device=dev)
        
        if world_size > 1 and dist.is_initialized():
            # Share sizes across all ranks
            all_sizes = [torch.zeros(1, dtype=torch.int64, device=dev) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)
            
            # Pad to max size for gathering
            max_size = int(max(s.item() for s in all_sizes))
            padded = torch.zeros(max_size, dtype=torch.uint8, device=dev)
            padded[:local_bytes.numel()] = local_bytes
            
            # Gather all padded data
            gathered = [torch.zeros(max_size, dtype=torch.uint8, device=dev) for _ in range(world_size)]
            dist.all_gather(gathered, padded)
        else:
            all_sizes = [local_size]
            gathered = [local_bytes]
        
        # Rank 0 writes the aggregated file
        if rank == 0:
            save_root = os.path.join("stats", self.run_name, f"step_{step:08d}")
            os.makedirs(save_root, exist_ok=True)
            out_path = os.path.join(save_root, "metrics.json")
            
            # Decode all entries
            entries = []
            for tensor, size in zip(gathered, all_sizes):
                try:
                    raw = tensor[:int(size.item())].detach().to("cpu").numpy().tobytes()
                    entries.append(json.loads(raw.decode("utf-8")))
                except Exception as e:
                    print(f"[Rank 0] Failed to decode gathered metrics: {e}")
            
            # Compute global phenomena consensus
            global_phenomena = self._compute_global_phenomena_consensus(entries)
            
            # Create the final output blob
            blob = {
                "mode": mode,
                "step": int(step),
                "world_size": int(world_size),
                "timestamp": datetime.utcnow().isoformat(),
                "global_phenomena_consensus": global_phenomena,
                "entries": entries,
                "metadata": {
                    "model_name": self.model_name_or_path if hasattr(self, 'model_name_or_path') else None,
                    "run_name": self.run_name,
                    "num_generations": self.num_generations if hasattr(self, 'num_generations') else None,
                    "batch_size": self.args.per_device_train_batch_size if hasattr(self.args, 'per_device_train_batch_size') else None,
                }
            }
            
            # Write to file
            with open(out_path, "w") as f:
                json.dump(blob, f, indent=2)
            
            # Optional: Write a lightweight phenomena-only log for quick scanning
            phenomena_log_path = os.path.join("stats", self.run_name, "phenomena_timeline.jsonl")
            phenomena_entry = {
                "step": int(step),
                "timestamp": datetime.utcnow().isoformat(),
                "phenomena": global_phenomena,
                "alert_level": self._compute_alert_level(global_phenomena)
            }
            with open(phenomena_log_path, "a") as f:
                f.write(json.dumps(phenomena_entry) + "\n")

    def _detect_phenomena(self, mode: str) -> dict:
        """
        Detect RLVR phenomena based on current metrics.
        
        Returns dict with detection flags and confidence scores.
        """
        phenomena = {
            "aha_moment": {"detected": False, "confidence": 0.0, "evidence": []},
            "overthinking": {"detected": False, "confidence": 0.0, "evidence": []},
            "capability_collapse": {"detected": False, "confidence": 0.0, "evidence": []},
            "entropy_collapse": {"detected": False, "confidence": 0.0, "evidence": []},
            "reward_hacking": {"detected": False, "confidence": 0.0, "evidence": []},
        }
        
        metrics = self._metrics.get(mode, {})
        
        # Aha Moment Detection
        if "phenomena/aha_moment_detected" in metrics:
            recent_detections = metrics["phenomena/aha_moment_detected"][-10:] if len(metrics["phenomena/aha_moment_detected"]) > 10 else metrics["phenomena/aha_moment_detected"]
            if recent_detections:
                detection_rate = sum(recent_detections) / len(recent_detections)
                if detection_rate > 0.3:
                    phenomena["aha_moment"]["detected"] = True
                    phenomena["aha_moment"]["confidence"] = detection_rate
                    phenomena["aha_moment"]["evidence"].append(f"Detection rate: {detection_rate:.2%}")
        
        # Overthinking Detection
        if "phenomena/overthinking_score" in metrics:
            recent_scores = metrics["phenomena/overthinking_score"][-10:] if len(metrics["phenomena/overthinking_score"]) > 10 else metrics["phenomena/overthinking_score"]
            if recent_scores:
                avg_score = sum(recent_scores) / len(recent_scores)
                if avg_score > 0.5:
                    phenomena["overthinking"]["detected"] = True
                    phenomena["overthinking"]["confidence"] = min(avg_score, 1.0)
                    phenomena["overthinking"]["evidence"].append(f"Average score: {avg_score:.3f}")
        
        # Capability Collapse
        if "phenomena/capability_collapse_risk" in metrics:
            recent_risks = metrics["phenomena/capability_collapse_risk"][-10:] if len(metrics["phenomena/capability_collapse_risk"]) > 10 else metrics["phenomena/capability_collapse_risk"]
            if recent_risks:
                avg_risk = sum(recent_risks) / len(recent_risks)
                if avg_risk > 0.3:
                    phenomena["capability_collapse"]["detected"] = True
                    phenomena["capability_collapse"]["confidence"] = avg_risk
                    phenomena["capability_collapse"]["evidence"].append(f"Risk level: {avg_risk:.2%}")
        
        # Entropy Collapse
        if "phenomena/entropy_collapse_risk" in metrics:
            recent_risks = metrics["phenomena/entropy_collapse_risk"][-10:] if len(metrics["phenomena/entropy_collapse_risk"]) > 10 else metrics["phenomena/entropy_collapse_risk"]
            if recent_risks:
                detection_rate = sum(recent_risks) / len(recent_risks)
                if detection_rate > 0.5:
                    phenomena["entropy_collapse"]["detected"] = True
                    phenomena["entropy_collapse"]["confidence"] = detection_rate
                    phenomena["entropy_collapse"]["evidence"].append(f"Detection rate: {detection_rate:.2%}")
        
        # Reward Hacking
        if "phenomena/reward_hacking_signal" in metrics:
            recent_signals = metrics["phenomena/reward_hacking_signal"][-10:] if len(metrics["phenomena/reward_hacking_signal"]) > 10 else metrics["phenomena/reward_hacking_signal"]
            if recent_signals:
                detection_rate = sum(recent_signals) / len(recent_signals)
                if detection_rate > 0.2:
                    phenomena["reward_hacking"]["detected"] = True
                    phenomena["reward_hacking"]["confidence"] = detection_rate
                    phenomena["reward_hacking"]["evidence"].append(f"Signal rate: {detection_rate:.2%}")
        
        return phenomena

    def _compute_step_statistics(self, mode: str) -> dict:
        """
        Compute summary statistics for the current step.
        """
        stats = {}
        metrics = self._metrics.get(mode, {})
        
        # Gradient statistics
        grad_keys = [k for k in metrics.keys() if k.startswith("grad_stats/")]
        for key in grad_keys:
            if isinstance(metrics[key], list) and metrics[key]:
                recent_values = metrics[key][-10:] if len(metrics[key]) > 10 else metrics[key]
                # Filter out None values
                recent_values = [v for v in recent_values if v is not None]
                if recent_values:
                    stats[key] = {
                        "mean": sum(recent_values) / len(recent_values),
                        "min": min(recent_values),
                        "max": max(recent_values),
                        "latest": recent_values[-1]
                    }
        
        # Phenomena indicators
        phenomena_keys = [k for k in metrics.keys() if k.startswith("phenomena/")]
        for key in phenomena_keys:
            if isinstance(metrics[key], list) and metrics[key]:
                recent_values = metrics[key][-10:] if len(metrics[key]) > 10 else metrics[key]
                recent_values = [v for v in recent_values if v is not None]
                if recent_values:
                    stats[key] = {
                        "mean": sum(recent_values) / len(recent_values),
                        "latest": recent_values[-1]
                    }
        
        return stats

    def _compute_global_phenomena_consensus(self, entries: list) -> dict:
        """
        Aggregate phenomena detection across all ranks to form consensus.
        """
        consensus = {
            "aha_moment": {"detected": False, "confidence": 0.0, "detecting_ranks": []},
            "overthinking": {"detected": False, "confidence": 0.0, "detecting_ranks": []},
            "capability_collapse": {"detected": False, "confidence": 0.0, "detecting_ranks": []},
            "entropy_collapse": {"detected": False, "confidence": 0.0, "detecting_ranks": []},
            "reward_hacking": {"detected": False, "confidence": 0.0, "detecting_ranks": []},
        }
        
        for entry in entries:
            rank = entry.get("rank", -1)
            phenomena = entry.get("phenomena_summary", {})
            
            for phenomenon_name in consensus.keys():
                if phenomenon_name in phenomena:
                    phenomenon_data = phenomena[phenomenon_name]
                    if phenomenon_data.get("detected", False):
                        consensus[phenomenon_name]["detecting_ranks"].append(rank)
                        consensus[phenomenon_name]["confidence"] += phenomenon_data.get("confidence", 0.0)
        
        # Compute consensus decision
        num_ranks = len(entries)
        for phenomenon_name in consensus.keys():
            detecting_count = len(consensus[phenomenon_name]["detecting_ranks"])
            if detecting_count > 0:
                # Majority vote with confidence weighting
                consensus[phenomenon_name]["detected"] = detecting_count > num_ranks // 2
                consensus[phenomenon_name]["confidence"] = consensus[phenomenon_name]["confidence"] / num_ranks
                consensus[phenomenon_name]["detection_rate"] = detecting_count / num_ranks
        
        return consensus

    def _compute_alert_level(self, phenomena: dict) -> str:
        """
        Compute overall alert level based on detected phenomena.
        
        Returns: "normal", "warning", "critical"
        """
        detected_count = sum(1 for p in phenomena.values() if p.get("detected", False))
        high_confidence_count = sum(1 for p in phenomena.values() if p.get("confidence", 0) > 0.7)
        
        if detected_count >= 3 or high_confidence_count >= 2:
            return "critical"
        elif detected_count >= 2 or high_confidence_count >= 1:
            return "warning"
        else:
            return "normal"
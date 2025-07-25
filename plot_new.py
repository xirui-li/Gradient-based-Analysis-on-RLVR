import os
import json
import matplotlib.pyplot as plt
import random

# Path to your JSONL file
jsonl_path = "/workspace/open-r1/stats/output/Qwen2.5-1.5B-Instruct-GRPO-Math12k-no-thinkng/train_metrics.jsonl"

# Load JSONL file
data = []
with open(jsonl_path, 'r') as f:
    for line in f:
        entry = json.loads(line)
        if entry["rank"] == 0:  # Only use rank 0
            data.append(entry)

# Extract all available Shapley and Grad stats keys
shapley_keys = set()
frobenius_norm_keys = set()
for entry in data:
    for k in entry["metrics"].keys():
        if k.startswith("shapley_stats/params/") and k.endswith("dot_product"):
            shapley_keys.add(k)
        if k.startswith("grad_stats/params/") and k.endswith("frobenius_norm"):
            frobenius_norm_keys.add(k)
shapley_keys = sorted(list(shapley_keys))
frobenius_norm_keys = sorted(list(frobenius_norm_keys))

# Randomly select 5 Shapley stats and 5 Grad stats
random.seed(42)
selected_shapley = random.sample(shapley_keys, 5)
selected_frobenius_norm = random.sample(frobenius_norm_keys, 5)

# Collect data
steps = [entry["step"] for entry in data]
accuracy_reward_mean = [entry["metrics"].get("rewards/accuracy_reward/mean", None) for entry in data]
shapley_values = {k: [entry["metrics"].get(k, None) for entry in data] for k in selected_shapley}
frobenius_norm_values = {k: [entry["metrics"].get(k, None) for entry in data] for k in selected_frobenius_norm}

# Plotting
plt.figure(figsize=(15, 10))

# Plot accuracy reward
plt.plot(steps, accuracy_reward_mean, label="rewards/accuracy_reward/mean", linewidth=2)

# Plot Shapley stats
for k in selected_shapley:
    plt.plot(steps, shapley_values[k], label=f"Shapley: {k}", linestyle="--")

# Plot Grad stats
for k in selected_frobenius_norm:
    plt.plot(steps, frobenius_norm_values[k], label=f"Frobenius: {k}", linestyle=":")

plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Training Stat Comparison on Rank 0")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)

plt_path = "training_stat_comparison.png"
plt.savefig(plt_path)
plt.close()

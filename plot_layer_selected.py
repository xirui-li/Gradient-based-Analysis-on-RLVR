import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def load_all_json_files(folder_path, max_step=1000):
    data = {}
    files = [f for f in os.listdir(folder_path) if f.startswith('step_') and f.endswith('.json')]
    files.sort()

    for filename in files:
        step_match = re.search(r'step_(\d+)\.json', filename)
        if not step_match:
            continue
        step_num = int(step_match.group(1))
        if step_num > max_step:
            continue
        with open(os.path.join(folder_path, filename), 'r') as f:
            data[step_num] = json.load(f)
    return data

def parse_metric_key(key):
    key = key.replace('grad_stats/params/', '')
    parts = key.split('/')
    layer_num = None
    component_type = None
    metric_name = parts[-1]

    for part in parts:
        if part.startswith('model.layers.'):
            match = re.search(r'model\.layers\.(\d+)', part)
            if match:
                layer_num = int(match.group(1))
                break

    if 'self_attn' in key:
        if 'q_proj' in key:
            component_type = 'self_attn.q_proj'
        elif 'k_proj' in key:
            component_type = 'self_attn.k_proj'
        elif 'v_proj' in key:
            component_type = 'self_attn.v_proj'
        elif 'o_proj' in key:
            component_type = 'self_attn.o_proj'
    elif 'mlp' in key:
        if 'gate_proj' in key:
            component_type = 'mlp.gate_proj'
        elif 'up_proj' in key:
            component_type = 'mlp.up_proj'
        elif 'down_proj' in key:
            component_type = 'mlp.down_proj'
    elif 'input_layernorm' in key:
        component_type = 'input_layernorm'
    elif 'post_attention_layernorm' in key:
        component_type = 'post_attention_layernorm'
    elif 'embed_tokens' in key:
        component_type = 'embed_tokens'
        layer_num = -1
    elif 'model.norm' in key:
        component_type = 'norm'
        layer_num = -2

    param_type = 'weight' if 'weight' in key else 'bias'
    return {
        'layer_num': layer_num,
        'component_type': component_type,
        'param_type': param_type,
        'metric_name': metric_name,
        'full_component': f"{component_type}.{param_type}" if component_type else param_type
    }

def organize_time_series(data):
    time_series = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    for step, metrics in data.items():
        for key, value in metrics.items():
            parsed = parse_metric_key(key)
            if parsed['layer_num'] is None:
                continue
            metric = parsed['metric_name']
            component = parsed['full_component']
            layer = parsed['layer_num']
            time_series[metric][component][layer][step] = value
    return time_series

def plot_metric_by_layer(time_series_data, output_dir='plots_by_layer', num_random_layers=3):
    os.makedirs(output_dir, exist_ok=True)
    for metric_name, components in time_series_data.items():
        for component, layers in components.items():
            # Compute average value per layer
            layer_averages = {
                layer: np.mean(list(step_data.values()))
                for layer, step_data in layers.items() if step_data
            }
            if not layer_averages:
                continue

            # Select layers to plot
            sorted_layers = sorted(layer_averages.items(), key=lambda x: x[1])
            lowest_layer = sorted_layers[0][0]
            highest_layer = sorted_layers[-1][0]
            remaining_layers = [l for l in layer_averages.keys() if l not in {lowest_layer, highest_layer}]
            random_layers = random.sample(remaining_layers, min(num_random_layers, len(remaining_layers)))

            selected_layers = [lowest_layer, highest_layer] + random_layers

            # Plot
            plt.figure(figsize=(12, 6))
            for layer in sorted(selected_layers):
                step_data = layers[layer]
                steps = sorted(step_data.keys())
                values = [step_data[step] for step in steps]
                plt.plot(steps, values, label=f'Layer {layer}')

            plt.xlabel("Step")
            plt.ylabel(metric_name)
            plt.title(f"{metric_name} over Steps | {component} (selected layers)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = f"{metric_name}_{component.replace('.', '_')}_time_series_selected.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=300)
            plt.close()

def main():
    folder_path = "./grad_stats_history"
    print(f"Loading data from {folder_path}...")
    raw_data = load_all_json_files(folder_path, max_step=1000)
    time_series_data = organize_time_series(raw_data)
    print("Plotting selected layers over time...")
    plot_metric_by_layer(time_series_data, output_dir="plots_by_layer_selected")
    print("Done.")

if __name__ == "__main__":
    main()

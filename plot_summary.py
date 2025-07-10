import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_all_json_files(folder_path, max_step=1000):
    """Load all JSON files from the specified folder, up to a max step."""
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
    """Parse the metric key to extract component information."""
    key = key.replace('grad_stats/params/', '')
    parts = key.split('/')
    
    layer_num = None
    component_type = None
    metric_name = parts[-1]
    
    for part in parts:
        if part.startswith('model.layers.'):
            layer_match = re.search(r'model\.layers\.(\d+)', part)
            if layer_match:
                layer_num = int(layer_match.group(1))
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

def organize_data(data):
    """Organize the data by metric type and component."""
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for step, metrics in data.items():
        for key, value in metrics.items():
            parsed = parse_metric_key(key)
            
            if parsed['layer_num'] is not None:
                metric_name = parsed['metric_name']
                component = parsed['full_component']
                layer_num = parsed['layer_num']
                
                organized[metric_name][component][layer_num].append(value)
    
    return organized

def create_grouped_summary_plots(data_dict, output_dir='plots_grouped/summary'):
    """Create grouped bar plots comparing summary stats for each run."""
    os.makedirs(output_dir, exist_ok=True)
    run_labels = list(data_dict.keys())
    
    all_metric_names = set()
    for organized_data in data_dict.values():
        all_metric_names.update(organized_data.keys())
    
    for metric_name in sorted(all_metric_names):
        all_components = set()
        for run in run_labels:
            all_components.update(data_dict[run].get(metric_name, {}).keys())
        
        for component in sorted(all_components):
            plt.figure(figsize=(12, 6))
            layer_nums_set = set()
            for run in run_labels:
                layers_data = data_dict[run].get(metric_name, {}).get(component, {})
                layer_nums_set.update(layers_data.keys())
            layer_nums = sorted(layer_nums_set)

            width = 0.35
            x = np.arange(len(layer_nums))
            for i, run in enumerate(run_labels):
                means, stds = [], []
                for layer in layer_nums:
                    values = data_dict[run].get(metric_name, {}).get(component, {}).get(layer, [])
                    means.append(np.mean(values) if values else 0)
                    stds.append(np.std(values) if values else 0)
                plt.bar(x + i * width, means, width, yerr=stds, capsize=4, label=run)

            plt.xticks(x + width / 2, layer_nums)
            plt.xlabel('Layer')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} | {component}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = f'{metric_name}_{component.replace(".", "_")}_summary.png'
            plt.savefig(os.path.join(output_dir, fname), dpi=300)
            plt.close()

def create_grouped_distribution_plots(data_dict, output_dir='plots_grouped/distribution'):
    """Create grouped box plots comparing value distributions per layer."""
    os.makedirs(output_dir, exist_ok=True)
    run_labels = list(data_dict.keys())

    all_metric_names = set()
    for organized_data in data_dict.values():
        all_metric_names.update(organized_data.keys())
    
    for metric_name in sorted(all_metric_names):
        all_components = set()
        for run in run_labels:
            all_components.update(data_dict[run].get(metric_name, {}).keys())
        
        for component in sorted(all_components):
            plt.figure(figsize=(14, 6))
            layer_nums_set = set()
            for run in run_labels:
                layers_data = data_dict[run].get(metric_name, {}).get(component, {})
                layer_nums_set.update(layers_data.keys())
            layer_nums = sorted(layer_nums_set)

            positions = []
            data = []
            ticks = []
            width = 0.3

            for i, layer in enumerate(layer_nums):
                for j, run in enumerate(run_labels):
                    values = data_dict[run].get(metric_name, {}).get(component, {}).get(layer, [])
                    data.append(values)
                    positions.append(i + j * width)
                ticks.append(i + width / 2)

            plt.boxplot(data, positions=positions, widths=width, patch_artist=True)
            plt.xticks(ticks, layer_nums)
            plt.xlabel('Layer')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} | {component}')
            plt.legend(run_labels)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = f'{metric_name}_{component.replace(".", "_")}_distribution.png'
            plt.savefig(os.path.join(output_dir, fname), dpi=300)
            plt.close()

def main():
    patterns = ["rl", "sft"]
    folder_paths = ["./grad_stats_history", "./grad_stats_history_sft"]
    
    data_dict = {}
    
    for pattern, folder_path in zip(patterns, folder_paths):
        print(f"Processing {pattern}...")
        data = load_all_json_files(folder_path, max_step=1000)
        organized_data = organize_data(data)
        data_dict[pattern] = organized_data

    print("Creating grouped summary plots...")
    create_grouped_summary_plots(data_dict, output_dir='plots_e1/summary')

    print("Creating grouped distribution plots...")
    create_grouped_distribution_plots(data_dict, output_dir='plots_e1/distribution')

    print("Done. Check plots_grouped/ for results.")

if __name__ == "__main__":
    main()


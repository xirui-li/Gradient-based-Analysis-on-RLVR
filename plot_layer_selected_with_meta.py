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

def plot_metric_by_layer(time_series_data, output_dir='plots_by_layer', num_random_layers=3, 
                        response_length_data=None, eval_accuracy_data=None):
    """
    Plot metrics by layer with optional response length and evaluation accuracy overlays.
    
    Args:
        time_series_data: Organized gradient statistics data
        output_dir: Directory to save plots
        num_random_layers: Number of random layers to include
        response_length_data: List of (step, response_length) tuples
        eval_accuracy_data: List of (step, accuracy) tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics that benefit from SVD analysis (typically 2D parameter matrices)
    svd_metrics = {'S_sum', 'S_max', 'S_min', 'nuclear_norm', 'effective_rank'}
    
    for metric_name, components in time_series_data.items():
        for component, layers in components.items():
            # Skip if no data
            if not layers:
                continue
                
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

            # Create plot
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Plot gradient statistics
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_layers)))
            for i, layer in enumerate(sorted(selected_layers)):
                step_data = layers[layer]
                steps = sorted(step_data.keys())
                values = [step_data[step] for step in steps]
                ax1.plot(steps, values, label=f'Layer {layer}', color=colors[i], linewidth=2)

            ax1.set_xlabel("Step", fontsize=12)
            ax1.set_ylabel(f"{metric_name}", fontsize=12)
            ax1.set_title(f"{metric_name} over Steps | {component} (selected layers)", fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # Always add secondary y-axis for response length and eval accuracy
            ax2 = ax1.twinx()
            
            # Plot response length as dashed line on every plot
            if response_length_data is not None:
                resp_steps, resp_lengths = zip(*response_length_data)
                ax2.plot(resp_steps, resp_lengths, '--', color='red', linewidth=2, 
                        label='Response Length', alpha=0.8)
            
            # Plot evaluation accuracy as dashed line on every plot
            if eval_accuracy_data is not None:
                eval_steps, eval_accs = zip(*eval_accuracy_data)
                ax2.plot(eval_steps, eval_accs, '--', color='green', linewidth=2, 
                        label='Eval Accuracy', alpha=0.8)
            
            # Set up secondary axis
            if response_length_data is not None or eval_accuracy_data is not None:
                ax2.set_ylabel("Response Length / Eval Accuracy", fontsize=12)
                ax2.legend(loc='upper right')
            else:
                ax2.set_yticks([])  # Hide secondary axis if no data
            
            plt.tight_layout()
            
            # Save plot with descriptive filename
            fname = f"{metric_name}_{component.replace('.', '_')}_time_series_selected.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
            plt.close()

def plot_svd_summary(time_series_data, output_dir='plots_by_layer', 
                    response_length_data=None, eval_accuracy_data=None):
    """
    Create summary plots for SVD-related metrics across all components.
    """
    svd_metrics = {'S_sum', 'S_max', 'S_min', 'nuclear_norm', 'effective_rank'}
    
    for metric_name in svd_metrics:
        if metric_name not in time_series_data:
            continue
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        components = list(time_series_data[metric_name].keys())[:6]  # Top 6 components
        
        for idx, component in enumerate(components):
            if idx >= 6:
                break
                
            ax = axes[idx]
            layers = time_series_data[metric_name][component]
            
            # Plot a few representative layers
            layer_list = list(layers.keys())
            if len(layer_list) > 3:
                selected = [min(layer_list), max(layer_list)] + random.sample(layer_list, 2)
            else:
                selected = layer_list
                
            for layer in selected:
                step_data = layers[layer]
                steps = sorted(step_data.keys())
                values = [step_data[step] for step in steps]
                ax.plot(steps, values, label=f'Layer {layer}', alpha=0.7)
            
            ax.set_title(f"{metric_name} - {component}", fontsize=10)
            ax.set_xlabel("Step")
            ax.set_ylabel(metric_name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(len(components), 6):
            axes[idx].remove()
        
        plt.suptitle(f"SVD Metric: {metric_name} Summary", fontsize=16)
        plt.tight_layout()
        
        fname = f"svd_summary_{metric_name}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

def plot_response_length_and_accuracy(response_length_data, eval_accuracy_data, 
                                    output_dir='plots_by_layer'):
    """
    Create a dedicated plot for response length and evaluation accuracy.
    """
    if response_length_data is None and eval_accuracy_data is None:
        return
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot response length
    if response_length_data is not None:
        resp_steps, resp_lengths = zip(*response_length_data)
        ax1.plot(resp_steps, resp_lengths, '-o', color='red', linewidth=2, 
                label='Response Length', markersize=4)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Response Length", color='red')
        ax1.tick_params(axis='y', labelcolor='red')
    
    # Plot evaluation accuracy on secondary axis
    if eval_accuracy_data is not None:
        ax2 = ax1.twinx()
        eval_steps, eval_accs = zip(*eval_accuracy_data)
        ax2.plot(eval_steps, eval_accs, '-s', color='green', linewidth=2, 
                label='Eval Accuracy', markersize=4)
        ax2.set_ylabel("Evaluation Accuracy", color='green')
        ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title("Model Response Length and Evaluation Accuracy Over Training", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fname = "response_length_eval_accuracy.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()

def main(response_length_data=None, eval_accuracy_data=None):
    """
    Main function with optional response length and evaluation accuracy data.
    
    Args:
        response_length_data: List of (step, response_length) tuples, measured every 50 steps
        eval_accuracy_data: List of (step, accuracy) tuples, measured every 50 steps
    """
    folder_path = "./grad_stats_history_grpo_Qwen/Qwen2.5-1.5B-Instruct-no-thinking"
    print(f"Loading data from {folder_path}...")
    raw_data = load_all_json_files(folder_path, max_step=1000)
    time_series_data = organize_time_series(raw_data)
    
    output_dir = "plots_by_layer_selected"
    print("Plotting selected layers over time...")
    plot_metric_by_layer(time_series_data, output_dir=output_dir, 
                        response_length_data=response_length_data, 
                        eval_accuracy_data=eval_accuracy_data)
    
    print("Creating SVD summary plots...")
    plot_svd_summary(time_series_data, output_dir=output_dir,
                    response_length_data=response_length_data,
                    eval_accuracy_data=eval_accuracy_data)
    
    print("Creating dedicated response length and accuracy plot...")
    plot_response_length_and_accuracy(response_length_data, eval_accuracy_data, output_dir)
    
    print("Done.")

if __name__ == "__main__":
    # Example usage with sample data for 1000 steps (every 50 steps)
    # Replace these with your actual data
    # response_length_data = [(0, 181), (100, 162), (200, 163), (300, 186), (400, 179), (500, 181), (600, 183), (700, 189), (800, 193), (900, 196), (1000, 197)]
    # eval_accuracy_data = [(0, 43.62), (100, 44.36), (200, 43.61), (300, 44.32), (400, 44.61), (500, 44.51), (600, 44.84), (700, 44.90), (800, 45.04), (900, 45.17), (1000, 45.59)]

    response_length_data = [(0, 181), (100, 141), (200, 150), (300, 169), (400, 174), (500, 178), (600, 178), (700, 185), (800, 185), (900, 195), (1000, 200)]
    eval_accuracy_data = [(0, 41.66), (100, 33.10), (200, 38.79), (300, 40.68), (400, 41.24), (500, 43.22), (600, 44.50), (700, 44.83), (800, 44.70), (900, 43.87), (1000, 44.68)]
    
    main(response_length_data=response_length_data, eval_accuracy_data=eval_accuracy_data)
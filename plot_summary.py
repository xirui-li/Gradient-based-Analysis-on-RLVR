import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class GradientStatsAnalyzer:
    def __init__(self, base_path: str, run_name: str, max_step: int = 1800, step_interval: int = 10):
        """Initialize analyzer for a single run."""
        self.base_path = Path(base_path)
        self.run_name = run_name
        self.run_path = self.base_path / run_name
        self.max_step = max_step
        self.step_interval = step_interval
        self.stats = {}
        
    def load_stats(self) -> Dict:
        """Load all gradient statistics from disk."""
        print(f"Loading stats for {self.run_name}...")
        for step in range(0, self.max_step + 1, self.step_interval):
            file_path = self.run_path / f"step_{step:08d}" / "grad_summary.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    self.stats[step] = json.load(f)
        print(f"Loaded {len(self.stats)} checkpoints")
        return self.stats
    
    def get_component_stats(self, component_type: str, metric: str) -> pd.DataFrame:
        """Get statistics for a specific component type across all layers."""
        if not self.stats:
            return pd.DataFrame()
            
        data = []
        for step, step_data in sorted(self.stats.items()):
            metrics = step_data.get('metrics', {})
            for key, value in metrics.items():
                if component_type in key and metric in key and 'weight' in key:
                    if 'layers' in key:
                        layer_num = int(key.split('.')[2])
                        data.append({
                            'step': step,
                            'layer': layer_num,
                            'value': value
                        })
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        return df.pivot(index='step', columns='layer', values='value')


class GradientComparator:
    def __init__(self, analyzer1: GradientStatsAnalyzer, 
                 analyzer2: GradientStatsAnalyzer,
                 name1: str = "Run 1", name2: str = "Run 2",
                 output_dir: str = "plots"):
        """Initialize comparator with two analyzers."""
        self.analyzer1 = analyzer1
        self.analyzer2 = analyzer2
        self.name1 = name1
        self.name2 = name2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_improved_layer_heatmap(self, metric: str = 'frobenius_norm', 
                                   component: str = 'self_attn.v_proj',
                                   layers_to_show: List[int] = None,
                                   normalize_per_layer: bool = True,
                                   figsize: Tuple[int, int] = (18, 12)):
        """Create improved heatmaps with better visibility."""
        df1 = self.analyzer1.get_component_stats(component, metric)
        df2 = self.analyzer2.get_component_stats(component, metric)
        
        if df1.empty and df2.empty:
            print(f"No data available for {component} {metric}")
            return None
        
        # Select layers to show
        if layers_to_show is None:
            all_layers = sorted(set(df1.columns.tolist() + df2.columns.tolist()))
            if len(all_layers) > 10:
                indices = np.linspace(0, len(all_layers)-1, 10, dtype=int)
                layers_to_show = [all_layers[i] for i in indices]
            else:
                layers_to_show = all_layers
        
        # Filter dataframes
        df1_filtered = df1[[col for col in layers_to_show if col in df1.columns]] if not df1.empty else pd.DataFrame()
        df2_filtered = df2[[col for col in layers_to_show if col in df2.columns]] if not df2.empty else pd.DataFrame()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        for idx, (df, name, ax_row) in enumerate([(df1_filtered, self.name1, axes[0]), 
                                                   (df2_filtered, self.name2, axes[1])]):
            if not df.empty:
                # Original scale
                im1 = ax_row[0].imshow(df.T, aspect='auto', cmap='viridis', interpolation='nearest')
                ax_row[0].set_title(f'{name} - {component} {metric} (Original)', fontsize=11)
                ax_row[0].set_xlabel('Step')
                ax_row[0].set_ylabel('Layer')
                ax_row[0].set_yticks(range(len(df.columns)))
                ax_row[0].set_yticklabels([f'L{l}' for l in df.columns])
                plt.colorbar(im1, ax=ax_row[0], label=metric)
                
                # Normalized per layer
                if normalize_per_layer:
                    df_norm = df.copy()
                    for col in df_norm.columns:
                        col_min, col_max = df_norm[col].min(), df_norm[col].max()
                        if col_max > col_min:
                            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
                    
                    im2 = ax_row[1].imshow(df_norm.T, aspect='auto', cmap='viridis', 
                                          interpolation='nearest', vmin=0, vmax=1)
                    ax_row[1].set_title(f'{name} - {component} {metric} (Normalized)', fontsize=11)
                    plt.colorbar(im2, ax=ax_row[1], label='Normalized')
                else:
                    df_log = np.log10(df + 1e-10)
                    im2 = ax_row[1].imshow(df_log.T, aspect='auto', cmap='viridis', interpolation='nearest')
                    ax_row[1].set_title(f'{name} - {component} {metric} (Log Scale)', fontsize=11)
                    plt.colorbar(im2, ax=ax_row[1], label='Log10')
                
                ax_row[1].set_xlabel('Step')
                ax_row[1].set_ylabel('Layer')
                ax_row[1].set_yticks(range(len(df.columns)))
                ax_row[1].set_yticklabels([f'L{l}' for l in df.columns])
            else:
                for ax in ax_row:
                    ax.text(0.5, 0.5, f'No data for {name}', 
                           ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle(f'{component} - {metric} Evolution', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_nuclear_norm_evolution(self, layers_to_plot: List[int] = None,
                                   figsize: Tuple[int, int] = (16, 10)):
        """Plot nuclear norm evolution across training."""
        if layers_to_plot is None:
            layers_to_plot = [0, 5, 10, 15, 20, 27]
        
        components = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 
                     'self_attn.o_proj', 'mlp.up_proj', 'mlp.down_proj']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, component in enumerate(components):
            ax = axes[idx]
            
            df1 = self.analyzer1.get_component_stats(component, 'nuclear_norm')
            df2 = self.analyzer2.get_component_stats(component, 'nuclear_norm')
            
            colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(layers_to_plot)))
            colors2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(layers_to_plot)))
            
            if not df1.empty:
                for i, layer in enumerate(layers_to_plot):
                    if layer in df1.columns:
                        ax.plot(df1.index, df1[layer], 
                               label=f'L{layer}-{self.name1[:3]}', 
                               linestyle='-', alpha=0.8, color=colors1[i], linewidth=1.5)
            
            if not df2.empty:
                for i, layer in enumerate(layers_to_plot):
                    if layer in df2.columns:
                        ax.plot(df2.index, df2[layer],
                               label=f'L{layer}-{self.name2[:3]}', 
                               linestyle='--', alpha=0.8, color=colors2[i], linewidth=1.5)
            
            ax.set_title(f'{component}', fontsize=10)
            ax.set_xlabel('Step')
            ax.set_ylabel('Nuclear Norm')
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        
        plt.suptitle(f'Nuclear Norm Evolution ({self.name1} solid vs {self.name2} dashed)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_effective_rank_evolution(self, layers_to_plot: List[int] = None,
                                     figsize: Tuple[int, int] = (16, 10)):
        """Plot effective rank evolution for selected layers."""
        if layers_to_plot is None:
            layers_to_plot = [0, 5, 10, 15, 20, 27]
        
        components = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                     'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, component in enumerate(components):
            ax = axes[idx]
            
            df1 = self.analyzer1.get_component_stats(component, 'effective_rank')
            df2 = self.analyzer2.get_component_stats(component, 'effective_rank')
            
            colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(layers_to_plot)))
            colors2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(layers_to_plot)))
            
            if not df1.empty:
                for i, layer in enumerate(layers_to_plot):
                    if layer in df1.columns:
                        ax.plot(df1.index, df1[layer], 
                               label=f'L{layer}-{self.name1[:3]}', 
                               linestyle='-', alpha=0.7, color=colors1[i])
            
            if not df2.empty:
                for i, layer in enumerate(layers_to_plot):
                    if layer in df2.columns:
                        ax.plot(df2.index, df2[layer],
                               label=f'L{layer}-{self.name2[:3]}', 
                               linestyle='--', alpha=0.7, color=colors2[i])
            
            ax.set_title(f'{component}', fontsize=10)
            ax.set_xlabel('Step')
            ax.set_ylabel('Effective Rank')
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        
        plt.suptitle(f'Effective Rank Evolution', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_nuclear_vs_frobenius_ratio(self, components: List[str] = None,
                                       layers_to_plot: List[int] = None,
                                       figsize: Tuple[int, int] = (16, 10)):
        """Plot the ratio of nuclear to Frobenius norm."""
        if components is None:
            components = ['self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.down_proj']
        if layers_to_plot is None:
            layers_to_plot = [0, 7, 14, 21, 27]
        
        n_components = len(components)
        fig, axes = plt.subplots(2, n_components, figsize=figsize)
        
        for col_idx, component in enumerate(components):
            for row_idx, (analyzer, name) in enumerate([(self.analyzer1, self.name1),
                                                        (self.analyzer2, self.name2)]):
                ax = axes[row_idx, col_idx] if n_components > 1 else axes[row_idx]
                
                nuclear_df = analyzer.get_component_stats(component, 'nuclear_norm')
                frob_df = analyzer.get_component_stats(component, 'frobenius_norm')
                
                if not nuclear_df.empty and not frob_df.empty:
                    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layers_to_plot)))
                    
                    for i, layer in enumerate(layers_to_plot):
                        if layer in nuclear_df.columns and layer in frob_df.columns:
                            ratio = nuclear_df[layer] / (frob_df[layer] + 1e-10)
                            ax.plot(ratio.index, ratio, label=f'L{layer}', 
                                   color=colors[i], linewidth=1.5, alpha=0.8)
                    
                    ax.set_title(f'{name} - {component}', fontsize=10)
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Nuclear/Frobenius')
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
                else:
                    ax.text(0.5, 0.5, f'No data', ha='center', va='center')
        
        plt.suptitle('Nuclear/Frobenius Ratio (Higher = More Low-rank)', fontsize=12)
        plt.tight_layout()
        return fig
    
    def plot_gradient_magnitude_distribution(self, step_points: List[int] = None,
                                           figsize: Tuple[int, int] = (16, 10)):
        """Plot distribution of gradient magnitudes at specific checkpoints."""
        if step_points is None:
            available_steps = sorted(set(self.analyzer1.stats.keys()) | set(self.analyzer2.stats.keys()))
            if not available_steps:
                return None
            n_points = min(4, len(available_steps))
            indices = np.linspace(0, len(available_steps)-1, n_points, dtype=int)
            step_points = [available_steps[i] for i in indices]
        
        fig, axes = plt.subplots(2, len(step_points), figsize=figsize)
        if len(step_points) == 1:
            axes = axes.reshape(2, 1)
        
        for idx, step in enumerate(step_points):
            for row_idx, (analyzer, name, color) in enumerate([(self.analyzer1, self.name1, 'blue'),
                                                               (self.analyzer2, self.name2, 'orange')]):
                norms = []
                if step in analyzer.stats:
                    metrics = analyzer.stats[step].get('metrics', {})
                    for key, value in metrics.items():
                        if 'frobenius_norm' in key and value > 0:
                            norms.append(np.log10(value + 1e-10))
                
                if norms:
                    axes[row_idx, idx].hist(norms, bins=30, alpha=0.7, color=color, edgecolor='black')
                    axes[row_idx, idx].axvline(np.mean(norms), color='red', linestyle='--', alpha=0.5)
                
                axes[row_idx, idx].set_title(f'{name} - Step {step}', fontsize=10)
                axes[row_idx, idx].set_xlabel('Log10(Gradient Norm)')
                axes[row_idx, idx].set_ylabel('Count')
                axes[row_idx, idx].grid(True, alpha=0.3)
        
        plt.suptitle('Gradient Magnitude Distribution', fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_layer_comparison_summary(self, metric: str = 'frobenius_norm',
                                     figsize: Tuple[int, int] = (16, 10)):
        """Create a summary comparison across all layers."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        components = ['self_attn.v_proj', 'mlp.down_proj']
        
        for comp_idx, component in enumerate(components):
            df1 = self.analyzer1.get_component_stats(component, metric)
            df2 = self.analyzer2.get_component_stats(component, metric)
            
            # Average gradient norm per layer
            ax = axes[0, comp_idx]
            if not df1.empty:
                mean1 = df1.mean(axis=0).sort_index()
                ax.plot(mean1.index, mean1.values, 'o-', label=self.name1, alpha=0.7)
            if not df2.empty:
                mean2 = df2.mean(axis=0).sort_index()
                ax.plot(mean2.index, mean2.values, 's--', label=self.name2, alpha=0.7)
            
            ax.set_title(f'Mean {metric} - {component}', fontsize=10)
            ax.set_xlabel('Layer')
            ax.set_ylabel(f'Mean {metric}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Variance per layer
            ax = axes[1, comp_idx]
            if not df1.empty:
                var1 = df1.var(axis=0).sort_index()
                ax.plot(var1.index, var1.values, 'o-', label=self.name1, alpha=0.7)
            if not df2.empty:
                var2 = df2.var(axis=0).sort_index()
                ax.plot(var2.index, var2.values, 's--', label=self.name2, alpha=0.7)
            
            ax.set_title(f'Variance of {metric} - {component}', fontsize=10)
            ax.set_xlabel('Layer')
            ax.set_ylabel(f'Variance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Layer-wise Gradient Statistics Summary', fontsize=14)
        plt.tight_layout()
        return fig
    
    def save_all_plots(self):
        """Generate and save all plots."""
        print("\nGenerating and saving all plots...")
        
        # 1. Frobenius norm heatmaps for different components
        components_to_plot = ['self_attn.v_proj', 'self_attn.q_proj', 'mlp.down_proj', 'mlp.up_proj']
        for comp in components_to_plot:
            print(f"Creating Frobenius norm heatmap for {comp}...")
            fig = self.plot_improved_layer_heatmap(metric='frobenius_norm', component=comp)
            if fig:
                filename = f'heatmap_frobenius_{comp.replace(".", "_")}.png'
                fig.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # 2. Nuclear norm heatmaps
        print("Creating nuclear norm heatmaps...")
        for comp in ['self_attn.v_proj', 'mlp.down_proj']:
            fig = self.plot_improved_layer_heatmap(metric='nuclear_norm', component=comp)
            if fig:
                filename = f'heatmap_nuclear_{comp.replace(".", "_")}.png'
                fig.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # 3. Nuclear norm evolution
        print("Creating nuclear norm evolution plot...")
        fig = self.plot_nuclear_norm_evolution()
        if fig:
            fig.savefig(self.output_dir / 'nuclear_norm_evolution.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # 4. Effective rank evolution
        print("Creating effective rank evolution plot...")
        fig = self.plot_effective_rank_evolution()
        if fig:
            fig.savefig(self.output_dir / 'effective_rank_evolution.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # 5. Nuclear/Frobenius ratio
        print("Creating nuclear/Frobenius ratio plots...")
        fig = self.plot_nuclear_vs_frobenius_ratio()
        if fig:
            fig.savefig(self.output_dir / 'nuclear_frobenius_ratio.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # 6. Gradient magnitude distribution
        print("Creating gradient magnitude distribution...")
        fig = self.plot_gradient_magnitude_distribution()
        if fig:
            fig.savefig(self.output_dir / 'gradient_magnitude_distribution.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # 7. Layer comparison summary
        print("Creating layer comparison summary...")
        fig = self.plot_layer_comparison_summary()
        if fig:
            fig.savefig(self.output_dir / 'layer_comparison_summary.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"\nAll plots saved to {self.output_dir}/")


def main():
    # Configuration
    base_path = "/workspace/Gradient-based-Analysis-on-RLVR/stats/output"
    output_dir = "plots-2"
    
    # Check available directories
    base_path_obj = Path(base_path)
    available_runs = [d.name for d in base_path_obj.iterdir() if d.is_dir()]
    print(f"Available runs: {available_runs}")
    
    # Find runs to compare
    run1_candidates = [r for r in available_runs if 'reasoning' in r.lower() and 'no-reasoning' not in r.lower()]
    run2_candidates = [r for r in available_runs if 'random-reward' in r.lower()]
    
    if not run1_candidates:
        run1_candidates = [r for r in available_runs if 'vanilla' in r.lower()]
    
    if not run1_candidates and not run2_candidates:
        if len(available_runs) >= 2:
            run1, run2 = available_runs[0], available_runs[1]
        else:
            print("Need at least two runs to compare")
            return
    else:
        run1 = run1_candidates[0] if run1_candidates else available_runs[0]
        run2 = run2_candidates[0] if run2_candidates else available_runs[1]
    
    print(f"\nComparing:\n  Run 1: {run1}\n  Run 2: {run2}")
    
    # Initialize analyzers
    print("\nInitializing analyzers...")
    analyzer1 = GradientStatsAnalyzer(base_path, run1)
    analyzer2 = GradientStatsAnalyzer(base_path, run2)
    
    # Load statistics
    analyzer1.load_stats()
    analyzer2.load_stats()
    
    if not analyzer1.stats and not analyzer2.stats:
        print("No data loaded for either run.")
        return
    
    # Create descriptive names
    name1 = "With Reasoning" if 'reasoning' in run1.lower() and 'no-reasoning' not in run1.lower() else run1.split('-')[-1].title()
    name2 = "No Reasoning" if 'no-reasoning' in run2.lower() else run2.split('-')[-1].title()
    
    # Create comparator and generate all plots
    comparator = GradientComparator(analyzer1, analyzer2, name1, name2, output_dir)
    comparator.save_all_plots()
    
    print("\nDone! Check the 'plots' folder for all generated visualizations.")


if __name__ == "__main__":
    main()
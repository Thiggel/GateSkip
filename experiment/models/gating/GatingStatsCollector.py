import torch
from torch import nn
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import wandb
from typing import Dict, Any, Generator, Tuple


class GatingStatsCollector:
    def __init__(self):
        self.layer_gate_values = {}  # Format: {layer_name: [gate_values_list]}
        
    def collect(self, model):
        """Collect gate values from the current forward pass"""
        if hasattr(model, "gating"):
            for name, module in model.gating.wrapped_modules.items():
                if module.current_gate_value is not None:
                    # Get mean across hidden dim for each token
                    gate_value = module.current_gate_value.mean(dim=-1).detach().cpu()
                    
                    if name not in self.layer_gate_values:
                        self.layer_gate_values[name] = []
                    
                    self.layer_gate_values[name].append(gate_value)
    
    def get_distributions(self):
        """Return concatenated gate values for each layer"""
        distributions = {}
        for name, values_list in self.layer_gate_values.items():
            # Concatenate all collected values for this layer
            all_values = torch.cat([v.flatten() for v in values_list])
            distributions[name] = all_values
        return distributions

    @contextmanager
    def visualize_gate_distributions(self, model: nn.Module) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager that visualizes gate distributions from a model with percentiles.
        
        Usage:
        ```
        with visualize_gate_distributions(model) as gate_visualizations:
            wandb.log(gate_visualizations)
        ```
        
        Args:
            model: Model with gating_stats_collector attribute
            
        Yields:
            Dictionary of visualization artifacts for wandb logging
        """
        if not hasattr(model, "gating_stats_collector"):
            print("No gating stats collector found on model")
            yield {}
            return
            
        print("Creating gate distribution visualizations...")
        
        # Get distributions from model
        distributions = self.get_distributions()
        
        # Prepare the visualization dictionary to return
        visualizations = {}

        threshold_suggestions = {
            "5%": [],
            "10%": [],
            "25%": [],
            "50%": [],
        }
        
        try:
            # Create visualizations for each distribution
            for name, values in distributions.items():
                visualization, percentile_table = self.create_gating_visualization(values, name=name)
                if visualization is None:
                    continue

                visualizations[f"gate_distributions/{name}_distribution"] = visualization
                visualizations[f"gate_distributions/{name}_percentiles"] = percentile_table

                for percent, threshold in percentile_table.data:
                    if percent in threshold_suggestions:
                        threshold_suggestions[percent].append(threshold)


            print("Threshold suggestions for skipping tokens:")
            for percent, thresholds in threshold_suggestions.items():
                print(f"{percent}: {thresholds}")
            
            overall_vis, overall_table = overall_distribution, overall_percentile_table = self.create_gating_visualization(torch.cat(list(distributions.values())), name="overall")

            if overall_vis is not None:
                visualizations["gate_distributions/overall_distribution"] = overall_vis
                visualizations["gate_distributions/overall_percentiles"] = overall_table
            
            # Yield the visualizations dictionary
            yield visualizations
        
        finally:
            # Clean up any remaining figures
            plt.close('all')

    def create_gating_visualization(self, values: torch.Tensor, name="default") -> None:
        gate_values = values.detach().cpu().numpy()
        flat_values = gate_values.flatten()
        
        # Skip if no values or all values are the same
        if len(flat_values) == 0 or np.all(flat_values == flat_values[0]):
            return None, None
            
        # Create figure with two y-axes for KDE and CDF
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        # Calculate the KDE
        density = stats.gaussian_kde(flat_values)
        x = np.linspace(flat_values.min(), flat_values.max(), 1000)
        y = density(x)
        
        # Calculate percentiles
        percentiles = np.percentile(flat_values, [5, 25, 50, 75])
        
        # Calculate CDF for secondary axis
        sorted_data = np.sort(flat_values)
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Plot the KDE on the primary axis
        ax1.plot(x, y, 'b-', label='Density')
        ax1.fill_between(x, y, alpha=0.3, color='blue')
        ax1.set_xlabel('Gate Value')
        ax1.set_ylabel('Density', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot the CDF on the secondary axis
        ax2.plot(sorted_data, yvals, 'g-', label='CDF')
        ax2.set_ylabel('Cumulative Probability', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Add grid for percentiles
        ax2.grid(True, alpha=0.3)
        
        # Mark key percentiles
        percentile_labels = ["5%", "25%", "50%", "75%"]
        for p_val, p_label in zip(percentiles, percentile_labels):
            # Find the corresponding y-value on the KDE
            idx = np.abs(x - p_val).argmin()
            kde_y = y[idx]
            
            ax1.axvline(x=p_val, color='red', linestyle='--', alpha=0.7)
            
            ax1.text(p_val, kde_y, p_label, 
                     verticalalignment='bottom', 
                     horizontalalignment='center',
                     color='red', fontweight='bold')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Add to visualizations dictionary
        distribution = wandb.Image(fig)
        
        # Also create a table of percentiles for this distribution
        percentile_table = wandb.Table(columns=["Percentile", "Value"])
        for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
            percentile_table.add_data(f"{p}%", float(np.percentile(flat_values, p)))


        
        plt.savefig(f"{name}_gate_distribution.pdf", format='pdf')
        # Close the figure to free memory
        plt.close(fig)

        return distribution, percentile_table

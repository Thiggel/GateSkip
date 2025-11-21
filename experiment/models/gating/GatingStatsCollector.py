import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import nn


class GatingStatsCollector:
    def __init__(self):
        self.layer_gate_values = {}  # Format: {layer_name: [gate_values_list]}
        # Track aggregated importance per token across the entire evaluation
        # by storing the sum of observed importances and a count of how many
        # times each token occurred.
        self.token_importance_sum: Dict[int, float] = {}
        self.token_importance_count: Dict[int, int] = {}
        # Track gate values bucketed by token type (e.g., think-tag span, POS-ish bucket)
        self.layer_gate_values_by_type: Dict[str, Dict[str, list[torch.Tensor]]] = {}

        self._prepositions = {
            "about",
            "above",
            "across",
            "after",
            "against",
            "along",
            "among",
            "around",
            "at",
            "before",
            "behind",
            "below",
            "beneath",
            "beside",
            "besides",
            "between",
            "beyond",
            "but",
            "by",
            "concerning",
            "considering",
            "despite",
            "down",
            "during",
            "except",
            "following",
            "for",
            "from",
            "in",
            "including",
            "inside",
            "into",
            "like",
            "near",
            "of",
            "off",
            "on",
            "onto",
            "opposite",
            "out",
            "outside",
            "over",
            "past",
            "regarding",
            "round",
            "since",
            "through",
            "throughout",
            "till",
            "to",
            "toward",
            "towards",
            "under",
            "underneath",
            "unlike",
            "until",
            "up",
            "upon",
            "versus",
            "via",
            "with",
            "within",
            "without",
        }

        self._common_verbs = {
            "be",
            "have",
            "do",
            "say",
            "go",
            "get",
            "make",
            "know",
            "think",
            "take",
            "see",
            "come",
            "want",
            "look",
            "use",
            "find",
            "give",
            "tell",
            "work",
            "call",
        }
        
    def _normalize_token_piece(self, token: str) -> str:
        cleaned = token.strip()
        cleaned = cleaned.lstrip("▁")  # sentencepiece spacing marker
        return cleaned.lower()

    def _word_category(self, token: str) -> str | None:
        alnum_token = re.sub(r"[^a-zA-Z0-9]", "", token)
        if alnum_token == "":
            return None

        if any(char.isdigit() for char in alnum_token):
            return "numbers"

        lowered = alnum_token.lower()
        if lowered in self._prepositions:
            return "prepositions"

        if lowered in self._common_verbs or lowered.endswith("ing") or lowered.endswith("ed"):
            return "verbs"

        if lowered.isalpha():
            return "nouns"

        return None

    def _get_token_type_masks(
        self, input_ids: torch.Tensor, tokenizer
    ) -> Dict[str, torch.Tensor]:
        if tokenizer is None:
            return {}

        flat_tokens = tokenizer.convert_ids_to_tokens(
            input_ids.view(-1).tolist(), skip_special_tokens=False
        )

        base_masks: Dict[str, torch.Tensor] = {
            "inside_think": torch.zeros_like(input_ids, dtype=torch.bool),
            "outside_think": torch.zeros_like(input_ids, dtype=torch.bool),
            "numbers": torch.zeros_like(input_ids, dtype=torch.bool),
            "prepositions": torch.zeros_like(input_ids, dtype=torch.bool),
            "verbs": torch.zeros_like(input_ids, dtype=torch.bool),
            "nouns": torch.zeros_like(input_ids, dtype=torch.bool),
        }

        inside_think = False
        for idx, token in enumerate(flat_tokens):
            normalized = self._normalize_token_piece(token)

            contains_think_start = "<think>" in normalized
            contains_think_end = "</think>" in normalized

            is_inside_now = inside_think or contains_think_start
            view = tuple(np.unravel_index(idx, input_ids.shape))

            if is_inside_now:
                base_masks["inside_think"][view] = True
            else:
                base_masks["outside_think"][view] = True

            category = self._word_category(normalized)
            if category in base_masks:
                base_masks[category][view] = True

            if contains_think_end:
                inside_think = False
            elif contains_think_start:
                inside_think = True

        return base_masks

    def _summarize_tensor_bucket(self, bucket: list[torch.Tensor]) -> Dict[str, float]:
        if not bucket:
            return {}

        values = torch.cat(bucket).float()
        quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9]))
        return {
            "count": int(values.numel()),
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "p10": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p90": float(quantiles[2]),
        }

    def _collect_token_type_gate_values(
        self,
        module_name: str,
        gate_value: torch.Tensor,
        token_type_masks: Dict[str, torch.Tensor],
        validity_mask: torch.Tensor,
    ) -> None:
        if not token_type_masks:
            return

        token_level_gate = gate_value.mean(dim=-1)

        for token_type, mask in token_type_masks.items():
            combined_mask = mask & validity_mask
            if not combined_mask.any():
                continue

            masked_values = token_level_gate.masked_select(combined_mask)
            layer_bucket = self.layer_gate_values_by_type.setdefault(module_name, {})
            layer_bucket.setdefault(token_type, []).append(masked_values.detach().cpu())

    def collect(self, model, tokenizer=None):
        """Collect gate values and token importances from the current forward pass"""
        if hasattr(model, "gating"):
            layer_importances = []
            input_ids = None
            validity_mask = None
            token_type_masks: Dict[str, torch.Tensor] = {}
            effective_tokenizer = tokenizer or getattr(model, "tokenizer", None)
            layer_gate_means: list[tuple[str, torch.Tensor]] = []

            for name, module in model.gating.wrapped_modules.items():
                if module.current_gate_value is not None:
                    gate_value = module.current_gate_value
                    gate_value_mean = gate_value.mean(dim=-1).detach().cpu()
                    if name not in self.layer_gate_values:
                        self.layer_gate_values[name] = []
                    self.layer_gate_values[name].append(gate_value_mean)
                    layer_gate_means.append((name, gate_value))

                if (
                    module.current_token_importance is not None
                    and module.current_input_ids is not None
                    and module.current_validity_mask is not None
                ):
                    layer_importances.append(module.current_token_importance.detach().cpu())
                    if input_ids is None:
                        input_ids = module.current_input_ids.detach().cpu()
                        validity_mask = module.current_validity_mask.detach().cpu()
                        token_type_masks = self._get_token_type_masks(
                            input_ids, effective_tokenizer
                        )

            if token_type_masks and validity_mask is not None:
                for name, gate_value in layer_gate_means:
                    self._collect_token_type_gate_values(
                        name, gate_value.detach().cpu(), token_type_masks, validity_mask
                    )

            if layer_importances and input_ids is not None and validity_mask is not None:
                stacked = torch.stack(layer_importances)
                avg_importance = stacked.mean(dim=0)

                for tok, imp, valid in zip(
                    input_ids.flatten(), avg_importance.flatten(), validity_mask.flatten()
                ):
                    if valid:
                        tok_id = int(tok)
                        imp_val = float(imp)
                        self.token_importance_sum[tok_id] = (
                            self.token_importance_sum.get(tok_id, 0.0) + imp_val
                        )
                        self.token_importance_count[tok_id] = (
                            self.token_importance_count.get(tok_id, 0) + 1
                        )

    def summarize_gate_values_by_type(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return summary statistics for per-layer token-type gate values."""

        summaries: Dict[str, Dict[str, Dict[str, float]]] = {}

        for layer, token_buckets in self.layer_gate_values_by_type.items():
            layer_summary: Dict[str, Dict[str, float]] = {}
            for token_type, values in token_buckets.items():
                stats = self._summarize_tensor_bucket(values)
                if stats:
                    layer_summary[token_type] = stats
            if layer_summary:
                summaries[layer] = layer_summary

        return summaries

    def save_gate_type_statistics(self, file_path: str) -> None:
        """Persist per-layer token-type gate summaries to a JSON file."""

        summaries = self.summarize_gate_values_by_type()
        if not summaries:
            return

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(summaries, f, indent=2)

    def reset_gate_type_statistics(self) -> None:
        """Clear collected gate statistics for token-type analyses."""

        self.layer_gate_values_by_type = {}
    
    def get_distributions(self):
        """Return concatenated gate values for each layer"""
        distributions = {}
        for name, values_list in self.layer_gate_values.items():
            # Concatenate all collected values for this layer
            all_values = torch.cat([v.flatten() for v in values_list])
            distributions[name] = all_values
        return distributions

    def reset_token_stats(self) -> None:
        """Clear collected token importance sums and counts"""
        self.token_importance_sum = {}
        self.token_importance_count = {}

    def save_token_importance_stats(self, tokenizer, file_path: str) -> None:
        """Save average token importance values to a JSON file"""
        import json
        from pathlib import Path

        averages: Dict[str, float] = {}
        for tok_id, total in self.token_importance_sum.items():
            count = self.token_importance_count.get(tok_id, 0)
            if count == 0:
                continue
            token = tokenizer.decode([tok_id]) if tokenizer is not None else str(tok_id)
            averages[token] = total / count

        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "w") as f:
            json.dump(averages, f, indent=2)

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

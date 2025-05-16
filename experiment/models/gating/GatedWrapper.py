from torch import nn
import torch
from typing import Any, Optional, Union

from experiment.configs.ModelConfig import ModelConfig
from experiment.configs.GatingConfig import GatingMode
from experiment.utils.threshold_finder import ThresholdFinder


class GatedWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        gate: nn.Module,
        config: ModelConfig,
        layer_idx: int,
        module_name: str,
        parent: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.module = module
        self.gate = gate
        self.current_gate_value: Optional[torch.Tensor] = None
        self.current_percent_tokens_skipped = 0.0
        self.current_token_importance: Optional[torch.Tensor] = None
        self.past_percent_skipped: list[float] = []
        self.current_input_ids = None
        self.current_validity_mask = None

        self.threshold_finder = ThresholdFinder()

        self.global_step = 1

        object.__setattr__(self, "parent", parent)

    @property
    def attn_token_importance(self) -> torch.Tensor:
        if hasattr(self.parent, "attn"):
            attn_layer = self.parent.attn
        elif hasattr(self.parent, "self_attn"):
            attn_layer = self.parent.self_attn
        else:
            raise ValueError("No attention layer found in parent module")
        attn_token_importance = attn_layer.current_token_importance

        assert attn_token_importance is not None

        return attn_token_importance

    @property
    def threshold(self) -> float:
        assert self.current_token_importance is not None
        assert self.current_validity_mask is not None

        all_token_importances = (
            self.current_token_importance[self.current_validity_mask].detach().flatten()
        )

        if self.config.increasing_threshold:
            sorted_token_importances = torch.sort(all_token_importances.float()).values

            delta_percentile = (
                self.config.end_thr_percentile - self.config.start_thr_percentile
            )

            current_fraction = (
                self.global_step / self.config.num_increasing_steps
                if self.global_step is not None
                else 1.0
            )

            current_percentile = (
                self.config.start_thr_percentile
                + min(1.0, current_fraction) * delta_percentile
            )

            kth_index = min(
                int(current_percentile * len(sorted_token_importances)),
                len(sorted_token_importances) - 1,
            )

            current_threshold = sorted_token_importances[kth_index].item()

            return current_threshold if current_threshold != 1.0 else 0.999

        # compute cumulative probability distribution over token importances
        # find the threshold that corresponds to the desired skip ratio self.config.desired_skip_ratio
        threshold = self.threshold_finder.find_threshold(all_token_importances, self.config.desired_skip_ratio) if not self.config.randomly_skip else 0.5

        return threshold

    @property
    def is_attn_layer(self) -> bool:
        return "attn" in self.module_name

    @property
    def is_skipping_enabled_on_this_module_type(self) -> bool:
        return any(
            mod_type in self.module_name for mod_type in self.config.skip_module_types
        )

    def calculate_token_importance(
        self, token_importance: torch.Tensor
    ) -> torch.Tensor:
        assert self.current_validity_mask is not None

        token_importance = torch.where(
            self.current_validity_mask,
            token_importance,
            torch.ones_like(token_importance),
        )

        assert torch.all(token_importance[~self.current_validity_mask] == 1.0)

        if self.config.only_skip_every_second_layer and (self.layer_idx % 2 == 0):
            token_importance = torch.ones_like(token_importance)

        if self.config.randomly_skip:
            token_importance = (
                torch.rand_like(token_importance) > self.config.desired_skip_ratio
            ).float()

        elif self.layer_idx in self.config.always_skip_layers:
            token_importance = torch.zeros_like(token_importance)
        elif (
            self.config.always_skip_layers is not None
            and len(self.config.always_skip_layers) > 0
        ):
            token_importance = torch.ones_like(token_importance)
        elif self.config.skip_entire_layer_based_on_attn and not self.is_attn_layer:
            token_importance = self.attn_token_importance

        if (
            not self.is_skipping_enabled_on_this_module_type
            or not self.config.skip_modules
        ):
            token_importance = torch.ones_like(token_importance)

        self.current_token_importance = token_importance

        percent_skipped = token_importance.mean().item()

        return token_importance

    def calculate_skipping_statistics(self, skip_mask: torch.Tensor) -> None:
        assert self.current_validity_mask is not None
        num_skipped = skip_mask[self.current_validity_mask].sum().item()
        total_tokens = self.current_validity_mask.sum().item()


        self.current_percent_tokens_skipped = (
            num_skipped / total_tokens if total_tokens > 0 else 0.0
        )

        print(self.layer_idx, self.current_percent_tokens_skipped)

    def update_kv_cache(
        self,
        module_output: tuple,
        past_kv,
        skip_mask: torch.Tensor,
    ):
        if past_kv is None or not skip_mask.any():
            return None

        # Get previous layer cache - method differs by model type
        if hasattr(past_kv, "get_layer_cache"):  # Gemma
            prev_layer_cache = past_kv.get_layer_cache(self.layer_idx - 1)
            current_cache = module_output[1]
            current_layer_cache = current_cache.get_layer_cache(self.layer_idx)

            # Apply the same logic as LLaMA for updating
            skip_mask = skip_mask.expand(-1, -1, current_layer_cache[0].size(-1))

            # Create updated cache
            new_k = torch.where(skip_mask, prev_layer_cache[0], current_layer_cache[0])
            new_v = torch.where(skip_mask, prev_layer_cache[1], current_layer_cache[1])

            # Update the cache - method differs by model type
            current_cache.update_layer_cache(self.layer_idx, (new_k, new_v))
            return current_cache
        else:
            prev_layer_cache = past_kv[self.layer_idx - 1]  # This returns (k, v) tuple

            if (
                isinstance(module_output, tuple)
                and len(module_output) > 1
                and module_output[1] is not None
            ):
                current_layer_cache = module_output[1][self.layer_idx]

                skip_mask = skip_mask.expand(-1, -1, current_layer_cache[0].size(-1))

                current_layer_cache[0] = torch.where(
                    skip_mask, prev_layer_cache[0], current_layer_cache[0]
                )

                current_layer_cache[1] = torch.where(
                    skip_mask, prev_layer_cache[1], current_layer_cache[1]
                )

                module_output[1][self.layer_idx] = current_layer_cache

                return module_output[1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        tuple[Union[int, torch.Tensor], Optional[torch.Tensor], Optional[Any]],
    ]:
        gate_value = self.gate(hidden_states.detach())
        self.current_gate_value = gate_value
        token_importance = self.calculate_token_importance(gate_value.mean(dim=-1))

        if (
            self.config.gating_mode == GatingMode.BEFORE_MODULE
            and self.config.actually_gate
        ):
            hidden_states = gate_value * hidden_states

        threshold = self.threshold
        skip_mask = (token_importance <= threshold).unsqueeze(-1)

        self.calculate_skipping_statistics(skip_mask)

        module_output = self.module(hidden_states, *args, **kwargs)
        main_output = (
            module_output[0] if isinstance(module_output, tuple) else module_output
        )

        try:
            updated_kv_cache = self.update_kv_cache(
                module_output, kwargs.get("past_key_value"), skip_mask
            )
        except Exception as e:
            print(f"Error in updating kv cache: {e}")
            updated_kv_cache = None

        if updated_kv_cache is not None:
            module_output = (main_output, updated_kv_cache)

        if self.config.actually_gate:
            main_output = gate_value * main_output

        zero_output = torch.zeros_like(hidden_states)
        main_output = torch.where(skip_mask, zero_output, main_output)

        if isinstance(module_output, tuple):
            return (main_output,) + module_output[1:]
        else:
            return main_output

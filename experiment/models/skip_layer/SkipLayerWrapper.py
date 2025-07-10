import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Union

from experiment.configs.ModelConfig import ModelConfig
from experiment.utils.threshold_finder import ThresholdFinder


class SkipLayerWrapper(nn.Module):
    """Wraps a module with SkipLayer gating logic."""

    def __init__(
        self,
        module: nn.Module,
        config: ModelConfig,
        layer_idx: int,
        module_name: str,
        parent: nn.Module,
        d_model: int,
    ) -> None:
        super().__init__()
        self.module = module
        self.config = config
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.gate = nn.Linear(d_model, 2)
        self.threshold_finder = ThresholdFinder()

        self.current_gate: Optional[torch.Tensor] = None
        self.current_percent_tokens_skipped: float = 0.0
        self.current_token_importance: Optional[torch.Tensor] = None
        object.__setattr__(self, "parent", parent)

    @property
    def threshold(self) -> float:
        assert self.current_token_importance is not None
        all_imp = self.current_token_importance.detach().flatten()
        return self.threshold_finder.find_threshold(
            all_imp, self.config.desired_skip_ratio
        )

    def calculate_statistics(self, skip_mask: torch.Tensor) -> None:
        num_skipped = skip_mask.sum().item()
        total = skip_mask.numel()
        self.current_percent_tokens_skipped = num_skipped / total if total > 0 else 0.0

    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        gate_logits = self.gate(hidden_states)
        if self.training:
            gate = F.gumbel_softmax(gate_logits, dim=-1, hard=True)
        else:
            gate = gate_logits.softmax(dim=-1)
        self.current_gate = gate
        compute_prob = gate[..., 1]
        self.current_token_importance = compute_prob

        if not self.training:
            thr = self.threshold
            compute_mask = compute_prob > thr
        else:
            compute_mask = gate.argmax(dim=-1).bool()

        module_output = self.module(hidden_states, *args, **kwargs)
        main_output = module_output[0] if isinstance(module_output, tuple) else module_output

        compute_out = gate[..., 1].unsqueeze(-1) * main_output
        skip_out = gate[..., 0].unsqueeze(-1) * hidden_states
        final = torch.where(compute_mask.unsqueeze(-1), compute_out, skip_out)

        self.calculate_statistics(~compute_mask)

        if isinstance(module_output, tuple):
            return (final,) + module_output[1:]
        return final

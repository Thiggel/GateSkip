from torch import nn
import torch
from typing import Dict

from experiment.configs.ModelConfig import ModelConfig

from .SkipLayerWrapper import SkipLayerWrapper


class ModelSkipLayer(nn.Module):
    """Manager for SkipLayer wrappers across transformer layers."""

    def __init__(self, config: ModelConfig, d_model: int):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.wrapped_modules: Dict[str, SkipLayerWrapper] = {}

    def wrap_module(
        self,
        name: str,
        module: nn.Module,
        parent: nn.Module,
        layer_idx: int,
    ) -> SkipLayerWrapper:
        wrapped = SkipLayerWrapper(
            module,
            self.config,
            layer_idx,
            module_name=name,
            parent=parent,
            d_model=self.d_model,
        )
        self.wrapped_modules[name] = wrapped
        return wrapped

    def compute_aux_loss(self, dtype: torch.dtype) -> torch.Tensor:
        if not self.wrapped_modules:
            return torch.tensor(0.0, dtype=dtype)

        loss = torch.tensor(0.0, dtype=dtype)
        num = 0
        target_ratio = 1.0 - self.config.desired_skip_ratio
        for module in self.wrapped_modules.values():
            if module.current_gate is not None:
                r_i = module.current_gate[..., 1].mean()
                loss += (r_i - target_ratio) ** 2
                num += 1
        if num > 0:
            loss /= num
        return loss * self.config.skip_layer_aux_loss_weight

from torch import nn
import torch
from typing import Dict, Optional

from experiment.configs.ModelConfig import ModelConfig

from .ModWrapper import ModWrapper


class ModelMod(nn.Module):
    """Handles Mixture of Depths routing for transformer models"""

    def __init__(self, config: ModelConfig, d_model: int):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.wrapped_modules: Dict[str, ModWrapper] = {}

    def wrap_module(
        self,
        name: str,
        module: nn.Module,
        parent: nn.Module,
        layer_idx: int,
    ) -> ModWrapper:
        """Wrap a module with MoD routing"""
        wrapped = ModWrapper(
            module,
            self.config,
            layer_idx,
            module_name=name,
            parent=parent,
            d_model=self.d_model,
        )
        self.wrapped_modules[name] = wrapped
        return wrapped

    def update_capacity(self, capacity: float) -> None:
        """Update capacity for all modules"""
        for module in self.wrapped_modules.values():
            module.update_capacity(capacity)

    def compute_predictor_loss(self, dtype: torch.dtype) -> torch.Tensor:
        """Compute combined predictor loss across all modules"""
        if not self.wrapped_modules:
            return torch.tensor(0.0, device=self._get_device(), dtype=dtype)

        pred_loss = torch.tensor(0.0, device=self._get_device(), dtype=dtype)
        num_modules = 0

        for module in self.wrapped_modules.values():
            pred_loss += module.compute_predictor_loss(dtype)
            num_modules += 1

        if num_modules > 0:
            pred_loss /= num_modules

        return self.config.predictor_loss_weight * pred_loss

    def _get_device(self) -> torch.device:
        """Get device from first registered module"""
        if self.wrapped_modules:
            first_module = next(iter(self.wrapped_modules.values()))
            return next(first_module.parameters()).device
        return torch.device("cpu")

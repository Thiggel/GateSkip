from torch import nn
import torch
from typing import Dict, Optional

from experiment.configs.ModelConfig import ModelConfig
from experiment.configs.GatingConfig import SparsityLossType

from .GateLayer import GateLayer
from .GatedWrapper import GatedWrapper


class ModelGating(nn.Module):
    """Handles gating for transformer models through direct module wrapping"""

    def __init__(self, config: ModelConfig, d_model: int):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.wrapped_modules: Dict[str, GatedWrapper] = {}

    def wrap_module(
        self,
        name: str,
        module: nn.Module,
        parent: nn.Module,
        layer_idx: int,
        gate: Optional[GateLayer] = None,
        frozen_gate: bool = False,
    ) -> GatedWrapper:
        """Wrap a module with gating"""
        gate = GateLayer(self.d_model, self.config) if gate is None else gate

        if frozen_gate:
            gate.requires_grad_(False)

        wrapped = GatedWrapper(
            module,
            gate,
            self.config,
            layer_idx,
            module_name=name,
            parent=parent,
        )
        self.wrapped_modules[name] = wrapped
        return wrapped

    def compute_gate_loss(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute combined loss for all gates"""
        if not self.wrapped_modules:
            return torch.tensor(0.0, device=self._get_device()), torch.tensor(
                0.0, device=self._get_device()
            )

        entropy_loss = torch.tensor(0.0, device=self._get_device())
        sparsity_loss = torch.tensor(0.0, device=self._get_device())

        num_modules = 0

        for module in self.wrapped_modules.values():
            if module.current_gate_value is not None:
                gate_value = module.current_gate_value

                if self.config.entropy_loss_weight > 0:
                    entropy_loss += self._compute_entropy_loss(gate_value)

                if self.config.sparsity_loss_weight > 0:
                    if self.config.sparsity_loss_type == SparsityLossType.L1:
                        sparsity_loss += gate_value.abs().mean()
                    elif self.config.sparsity_loss_type == SparsityLossType.L2:
                        sparsity_loss += gate_value.pow(2).mean()
                    elif self.config.sparsity_loss_type == SparsityLossType.KL:
                        target_keep_prob = 1.0 - self.config.desired_skip_ratio
                        target = torch.full_like(gate_value, target_keep_prob)
                        eps = 1e-6
                        gate_prob = gate_value.clamp(eps, 1.0 - eps)
                        target_prob = target.clamp(eps, 1.0 - eps)
                        kl_loss = gate_prob * torch.log(gate_prob / target_prob)
                        kl_loss += (1 - gate_prob) * torch.log(
                            (1 - gate_prob) / (1 - target_prob)
                        )
                        sparsity_loss += kl_loss.mean()
                    else:
                        raise ValueError(
                            f"Unsupported sparsity loss type: {self.config.sparsity_loss_type}"
                        )

                num_modules += 1

        if num_modules > 0:
            entropy_loss /= num_modules
            sparsity_loss /= num_modules

        return entropy_loss, sparsity_loss

    def _compute_entropy_loss(
        self, gate_value: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Compute entropy loss for a gate value"""
        entropy = -(
            gate_value * (gate_value + eps).log()
            + (1 - gate_value) * (1 - gate_value + eps).log()
        )
        return entropy.mean()

    def _get_device(self) -> torch.device:
        """Get device from first registered module"""
        if self.wrapped_modules:
            first_module = next(iter(self.wrapped_modules.values()))
            return next(first_module.parameters()).device
        return torch.device("cpu")

    def set_thresholds(self, thresholds: list[float]) -> None:
        """Set thresholds for all wrapped modules"""
        for i, module in enumerate(self.wrapped_modules.values()):
            module.gate.set_threshold(thresholds[i])

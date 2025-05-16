from torch import nn
import torch
from experiment.configs.ModelConfig import ModelConfig


class GateLayer(nn.Module):
    """Gate layer that applies gating to module outputs"""

    def __init__(self, d_model: int, config: ModelConfig):
        super().__init__()
        self.gate = self.init_gate(d_model, config)

    def init_gate(self, d_model: int, config: ModelConfig) -> nn.Module:
        if config.use_mlp_gate:
            gate = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.ReLU(),
                nn.Linear(
                    2 * d_model, d_model if not config.single_number_gates else 1
                ),
            )

            nn.init.normal_(
                gate[0].weight, mean=config.gate_init_mean, std=config.gate_init_std
            )
            nn.init.constant_(gate[0].bias, 0.0)
            nn.init.normal_(
                gate[2].weight, mean=config.gate_init_mean, std=config.gate_init_std
            )
            nn.init.constant_(gate[2].bias, config.gate_init_value)

            return gate

        gate = nn.Linear(d_model, d_model if not config.single_number_gates else 1)
        nn.init.normal_(
            gate.weight, mean=config.gate_init_mean, std=config.gate_init_std
        )
        nn.init.constant_(gate.bias, config.gate_init_value)

        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x))

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Union, Dict
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Union, Dict

from experiment.configs.ModelConfig import ModelConfig
from experiment.utils.threshold_finder import ThresholdFinder


class ModWrapper(nn.Module):
    """
    Wrapper that implements Mixture of Depths routing for transformer layers.
    Uses expert-choice routing to select top-k tokens for processing during training,
    and predictor-based routing during inference.
    """

    def __init__(
        self,
        module: nn.Module,
        config: ModelConfig,
        layer_idx: int,
        module_name: str,
        parent: nn.Module,
        d_model: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.module = module
        object.__setattr__(self, "parent", parent)

        self.threshold_finder = ThresholdFinder()

        # Router projects hidden states to scalar weights
        self.router = nn.Linear(d_model, 1)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.router.bias, 1.0)

        # Predictor for sampling
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Statistics tracking
        self.current_token_importance: Optional[torch.Tensor] = None
        self.current_percent_tokens_processed = 0.0
        self.past_percent_processed: list[float] = []

        self.capacity = 1.0

    def update_capacity(self, capacity: float) -> None:
        """Update capacity for routing"""
        self.capacity = capacity

    @property
    def current_percent_tokens_skipped(self) -> float:
        return 1.0 - self.current_percent_tokens_processed

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs: Any,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        if self.config.only_skip_every_second_layer and (self.layer_idx % 2 == 0):
            return self.module(
                hidden_states,
                *args,
                **kwargs,
            )

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # During generation/inference, just use the predictor output directly
        if not self.training:
            predictor_logits = self.predictor(hidden_states).squeeze(-1)

            importance = torch.sigmoid(predictor_logits)
            threshold = self.threshold_finder.find_threshold(importance, self.config.desired_skip_ratio)
            selection_mask = (importance > threshold).unsqueeze(-1)

            # Process through module
            output = self.module(
                hidden_states,
                *args,
                **kwargs,
            )

            # Track statistics
            self.current_percent_tokens_processed = selection_mask.float().mean().item()
            self.past_percent_processed.append(self.current_percent_tokens_processed)

            if isinstance(output, tuple):
                processed_states = output[0]
                other_outputs = output[1:]
                # Apply selection mask
                final_states = torch.where(
                    selection_mask, processed_states, hidden_states
                )
                return (final_states,) + other_outputs
            else:
                return torch.where(selection_mask, output, hidden_states)

        # Compute router logits
        router_logits = self.router(hidden_states).sigmoid()  # [B, S, 1]
        self.current_token_importance = router_logits

        # get true/false mask whether tokens are above the 12.5% threshold of
        # highest router logits
        # always select the same number of tokens per batch item

        k = int(seq_len * self.capacity)
        import random

        if random.random() < 0.01:
            print(k, seq_len)
            print(self.capacity)
        # Get indices of top k values per batch item
        topk_indices = torch.topk(router_logits, k, dim=1).indices.squeeze(-1)
        # Create mask of zeros
        mask = torch.zeros_like(router_logits, dtype=torch.bool)
        # For each batch item, set the top k positions to True
        batch_indices = torch.arange(mask.size(0)).unsqueeze(1).expand(-1, k)
        mask[batch_indices, topk_indices] = True

        # test
        # for i in range(topk_indices.size(0)):
        #     for j in range(topk_indices.size(1)):
        #         assert mask[i, topk_indices[i, j]] == True

        # assert mask.shape == hidden_states.shape[:2], (
        #     mask.shape,
        #     hidden_states.shape,
        # )

        # Process through module
        processed_output = self.module(
            hidden_states,
            *args,
            **kwargs,
        )

        # Handle tuple outputs
        if isinstance(processed_output, tuple):
            output = processed_output[0]
            other_outputs = processed_output[1:]
        else:
            output = processed_output
            other_outputs = ()

        output *= router_logits

        output = torch.where(
            mask.expand(-1, -1, hidden_states.size(-1)), output, hidden_states
        )

        # Store predictor targets for loss
        if self.training:
            router_targets = torch.zeros_like(router_logits)
            router_targets[mask] = 1.0

            # test
            # for i in range(topk_indices.size(0)):
            #     for j in range(topk_indices.size(1)):
            #         assert router_targets[i, topk_indices[i, j]] == 1.0

            # Get predictor outputs on detached inputs
            predictor_logits = self.predictor(hidden_states.detach())

            self.current_predictor_logits = predictor_logits
            self.current_predictor_targets = router_targets

        if other_outputs:
            return (output,) + other_outputs
        return output

    def compute_predictor_loss(self, dtype: torch.dtype) -> torch.Tensor:
        """Compute binary cross entropy loss for predictor training"""
        if hasattr(self, "current_predictor_logits") and hasattr(
            self, "current_predictor_targets"
        ):
            return F.binary_cross_entropy_with_logits(
                self.current_predictor_logits,
                self.current_predictor_targets,
                reduction="mean",
            )
        return torch.tensor(0.0, device=self._get_device(), dtype=dtype)

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    def update_cache(self, *args, **kwargs):
        """Pass through cache updates to underlying module"""
        if hasattr(self.module, "update_cache"):
            return self.module.update_cache(*args, **kwargs)
        return None

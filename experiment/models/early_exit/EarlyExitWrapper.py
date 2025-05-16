from torch import nn
import torch
import torch.nn.functional as F
from typing import Any, Optional, Union, Dict, List

from experiment.configs.ModelConfig import ModelConfig
from experiment.configs.EarlyExitConfig import ConfidenceMeasure
from experiment.utils.threshold_finder import ThresholdFinder


class EarlyExitWrapper(nn.Module):
    """Wrapper that determines when to exit early from transformer layers.

    This wrapper calculates confidence scores to decide if the current token
    has enough confidence to exit early. It does not modify the module's
    forward pass, just detects when to exit.
    """

    def __init__(
        self,
        module: nn.Module,
        config: ModelConfig,
        layer_idx: int,
        module_name: str,
        parent: nn.Module,
        controller: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.module = module
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "controller", controller)

        # For tracking statistics
        self.current_confidence: Optional[torch.Tensor] = None
        self.current_threshold: Optional[float] = None
        self.current_exit_decision: Optional[torch.Tensor] = None
        self.exit_layer_indices: List[int] = []
        self.tokens_processed = 0
        self.tokens_exited_early = 0
        self.percent_skipped = 0.0

        self.threshold_finder = ThresholdFinder()

    def compute_threshold(
        self, step_idx: int, confidence: torch.Tensor, max_steps: int = 100
    ) -> float:
        """Compute the decaying threshold based on generation step."""
        if not self.config.use_decaying_threshold:
            p = self.config.desired_skip_ratio
            current_layer_skip_ratio = 1 - (1 - p) ** (self.layer_idx + 1)
            print(f"Current layer skip ratio: {current_layer_skip_ratio:.2f}")
            threshold = self.threshold_finder.find_threshold(confidence, current_layer_skip_ratio, skip_below_threshold=False)
            return threshold

        # Following the paper's decaying threshold formula in Eq. (5)
        decay = (
            0.1
            + 0.9
            * torch.exp(
                torch.tensor(-self.config.decay_factor * step_idx / max_steps)
            ).item()
        )
        return min(1.0, max(0.0, self.config.base_threshold * decay))

    def compute_confidence(
        self, hidden_states: torch.Tensor, prev_hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute confidence score based on configured measure."""
        if self.config.confidence_measure == ConfidenceMeasure.SOFTMAX:
            # Calculate the gap between top two probabilities in softmax distribution
            # Get output embeddings from parent model
            if hasattr(self.parent, "get_output_embeddings"):
                output_embeddings = self.parent.get_output_embeddings()
            else:
                # Try common names for output embeddings
                output_embeddings = getattr(self.parent, "lm_head", None)
                if output_embeddings is None:
                    output_embeddings = getattr(self.parent, "output_projection", None)

            if output_embeddings is not None:
                logits = output_embeddings(hidden_states)
                probs = F.softmax(logits, dim=-1)
                values, _ = torch.topk(probs, k=2, dim=-1)
                confidence = values[..., 0] - values[..., 1]
            else:
                # Default to low confidence if we can't find output embeddings
                confidence = torch.zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    device=hidden_states.device,
                )

        elif self.config.confidence_measure == ConfidenceMeasure.HIDDEN_STATE:
            # Compute cosine similarity with previous layer's hidden state
            if self.controller.prev_hidden_states is None:
                confidence = torch.zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    device=hidden_states.device,
                )
            else:
                # Normalize both tensors for cosine similarity
                norm_curr = F.normalize(hidden_states, p=2, dim=-1)
                norm_prev = F.normalize(self.controller.prev_hidden_states, p=2, dim=-1)
                confidence = torch.sum(norm_curr * norm_prev, dim=-1)

        else:
            confidence = torch.zeros(
                hidden_states.shape[0],
                hidden_states.shape[1],
                device=hidden_states.device,
            )

        return confidence

    def should_exit(
        self, confidence: torch.Tensor, step_idx: int, max_steps: int = 100
    ) -> torch.Tensor:
        """Determine whether to exit early based on confidence and threshold."""
        # Fixed exit layer takes precedence
        if self.config.fixed_exit_layer > 0:
            return torch.tensor(
                self.layer_idx >= self.config.fixed_exit_layer,
                device=confidence.device,
                dtype=torch.bool,
            ).expand_as(confidence)

        # Don't exit before minimum layer
        if (self.layer_idx + 1) < self.config.min_exit_layer:
            return torch.zeros_like(confidence, dtype=torch.bool)

        if self.controller.exit_mask is not None:
            confidence = torch.where(
                self.controller.exit_mask, torch.ones_like(confidence), confidence
            )

        threshold = self.compute_threshold(step_idx, confidence, max_steps)
        self.current_threshold = threshold

        if self.controller.exit_mask is None:
            self.controller.exit_mask = torch.zeros_like(confidence, dtype=torch.bool)


        exit_mask = confidence > threshold

        self.controller.exit_mask = torch.logical_or(
            self.controller.exit_mask, exit_mask
        )

        percent_skipped = self.controller.exit_mask.float().mean().item()

        self.percent_skipped = percent_skipped


        return self.controller.exit_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        step_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Forward pass that also computes early exit confidence.

        Args:
            hidden_states: Current hidden states
            prev_hidden: Previous layer's hidden states (for hidden_state confidence)
            step_idx: Current generation step (for decaying threshold)
        """
        if self.layer_idx == 0:
            self.controller.exit_mask = None
            self.controller.prev_hidden_states = None

        # Just pass through to the wrapped module
        outputs = self.module(hidden_states, *args, **kwargs)

        if isinstance(outputs, tuple):
            main_output = outputs[0]
        else:
            main_output = outputs

        if self.controller.exit_mask is not None and self.controller.prev_hidden_states is not None:
            main_output = torch.where(
                self.controller.exit_mask.unsqueeze(-1).repeat(1, 1, self.controller.prev_hidden_states.shape[-1]), self.controller.prev_hidden_states, main_output
            )

        # Compute confidence
        self.current_confidence = self.compute_confidence(main_output, prev_hidden)


        self.controller.prev_hidden_states = main_output


        # Determine exit decision - for tracking purposes
        self.current_exit_decision = self.should_exit(self.current_confidence, step_idx)

        return outputs

    def get_exit_statistics(self) -> Dict[str, float]:
        """Get statistics about early exits"""
        exit_rate = self.tokens_exited_early / max(1, self.tokens_processed)
        print(f"Layer {self.layer_idx} exit rate: {exit_rate:.2f}")
        return {
            "exit_rate": exit_rate,
            "layer_idx": self.layer_idx,
        }

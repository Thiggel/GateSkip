from torch import nn
import torch
from typing import Dict, Optional, List, Tuple, Any
import math

from experiment.configs.ModelConfig import ModelConfig
from .EarlyExitWrapper import EarlyExitWrapper


class ModelEarlyExit(nn.Module):
    """Manages early exiting functionality for transformer models.

    This class coordinates early exit wrappers across layers and
    provides centralized access to early exit decisions and statistics.
    It also handles hidden state propagation for tokens that exit early.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wrapped_modules: Dict[str, EarlyExitWrapper] = {}

        self.prev_hidden_states: Optional[torch.Tensor] = None

        # Track which layer each token exited at
        self.exit_decisions: Dict[Tuple[int, int], int] = (
            {}
        )  # Maps (batch_idx, token_idx) to exit_layer

        # Store hidden states of early-exited tokens
        self.exit_hidden_states: Dict[Tuple[int, int], torch.Tensor] = (
            {}
        )  # Maps (batch_idx, token_idx) to hidden_state

        # Store key-value cache of early-exited tokens (needed for generation)
        self.exit_key_values: Dict[Tuple[int, int, int], torch.Tensor] = (
            {}
        )  # Maps (layer_idx, batch_idx, token_idx) to kv_state

        # Statistics tracking
        self.total_tokens = 0
        self.exit_layer_counts: Dict[int, int] = {}  # Counts per layer

        # For easier processing during generation
        self.is_generating = False
        self.current_step = 0

        self.exit_mask = None

    def compute_layer_loss_weights(self, num_layers: int) -> torch.Tensor:
        """
        Compute weights for layer losses based on configuration.

        Following the CALM paper, we set ω_i = i/∑j to favor higher layers.
        """
        # Linear increasing weights (favoring higher layers)
        weights = torch.arange(1, num_layers + 1, dtype=torch.float)

        # Normalize to sum to 1
        return weights / weights.sum()

    def compute_early_exit_loss(
        self,
        hidden_states: List[torch.Tensor],
        lm_head: torch.nn.Module,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss across layers according to the CALM paper.

        Args:
            hidden_states: List of hidden states from each layer
            lm_head: Language model head to project hidden states to vocabulary
            labels: Target labels

        Returns:
            Total loss and dictionary of individual layer losses
        """
        # Skip the first element if it's input embeddings
        if len(hidden_states) > len(self.wrapped_modules) + 1:
            hidden_states = hidden_states[1:]  # Skip input embeddings

        num_layers = len(hidden_states)
        weights = self.compute_layer_loss_weights(num_layers)
        device = hidden_states[0].device
        weights = weights.to(device)

        # Create shifted labels for causal LM loss
        shift_labels = labels[..., 1:].contiguous()
        loss_mask = shift_labels != -100

        # Compute loss for each layer
        layer_losses = []
        for i, layer_hidden in enumerate(hidden_states):
            # Project to vocabulary
            logits = lm_head(layer_hidden)

            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()

            # Compute cross entropy loss
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            )

            # Apply masking and compute mean
            loss = loss.view(shift_labels.size())
            masked_loss = loss.masked_fill(~loss_mask, 0.0)
            layer_loss = masked_loss.sum() / loss_mask.sum().clamp(min=1)
            layer_losses.append(layer_loss)

        # Combine losses with weights
        stacked_losses = torch.stack(layer_losses)
        total_loss = torch.sum(stacked_losses * weights)

        # Create dictionary of individual losses for logging
        loss_dict = {
            f"layer_{i}_loss": loss.detach() for i, loss in enumerate(layer_losses)
        }
        loss_dict["early_exit_loss"] = total_loss.detach()

        return total_loss, loss_dict

    def wrap_module(
        self,
        name: str,
        module: nn.Module,
        parent: nn.Module,
        layer_idx: int,
    ) -> EarlyExitWrapper:
        """Wrap a module with early exit functionality."""
        wrapped = EarlyExitWrapper(
            module,
            self.config,
            layer_idx,
            module_name=name,
            parent=parent,
            controller=self,
        )
        self.wrapped_modules[name] = wrapped
        return wrapped

    def record_exit_decisions(self):
        """
        Record exit decisions at current layer and store hidden states for early-exited tokens.

        Args:
            layer_idx: Current layer index
            hidden_states: Hidden states at this layer
        """
        percent_skipped = torch.tensor([wrapper.percent_skipped for _, wrapper in self.wrapped_modules.items()]).mean().item()

        return percent_skipped



    def maybe_propagate_hidden_states(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Propagate hidden states from earlier exit layers if tokens have already exited.

        Args:
            layer_idx: Current layer index
            hidden_states: Current hidden states

        Returns:
            Modified hidden states with propagated values
        """
        if not self.exit_decisions:
            return hidden_states

        # Create a copy to modify
        modified = hidden_states.clone()

        batch_size, seq_len = hidden_states.shape[:2]
        for batch_idx in range(batch_size):
            for token_idx in range(seq_len):
                key = (batch_idx, token_idx)

                # If this token exited at an earlier layer, use its stored hidden state
                if key in self.exit_decisions and self.exit_decisions[key] < layer_idx:
                    if key in self.exit_hidden_states:
                        modified[batch_idx, token_idx] = self.exit_hidden_states[key]

        return modified

    def store_kv_state(
        self, layer_idx: int, batch_idx: int, token_idx: int, kv_state: torch.Tensor
    ):
        """Store key-value state for a token that exited early."""
        self.exit_key_values[(layer_idx, batch_idx, token_idx)] = (
            kv_state.detach().clone()
        )

    def get_kv_state(
        self, layer_idx: int, batch_idx: int, token_idx: int
    ) -> Optional[torch.Tensor]:
        """Get stored key-value state for a token that exited early."""
        return self.exit_key_values.get((layer_idx, batch_idx, token_idx))

    def update_kv_cache(
        self,
        layer_idx: int,
        key_value_states: Optional[Tuple[torch.Tensor, torch.Tensor]],
        exit_decision: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Update the key-value cache for tokens that exited early.

        During generation, we need to propagate the key-value states from the
        layer where a token exited to all higher layers.

        Args:
            layer_idx: Current layer index
            key_value_states: Current key-value states (k, v)
            exit_decision: Boolean tensor indicating which tokens exited early

        Returns:
            Updated key-value states
        """
        if key_value_states is None or not self.is_generating:
            return key_value_states

        # Unpack key and value states
        k_states, v_states = key_value_states

        # Find tokens that exited at an earlier layer
        for batch_idx in range(exit_decision.shape[0]):
            for token_idx in range(exit_decision.shape[1]):
                key = (batch_idx, token_idx)

                # If this token exited at an earlier layer
                if key in self.exit_decisions and self.exit_decisions[key] < layer_idx:
                    # Look for stored key-value states
                    exit_layer = self.exit_decisions[key]
                    kv_key = (exit_layer, batch_idx, token_idx)

                    # If we have stored KV states for this token at its exit layer,
                    # propagate them to the current layer
                    if kv_key in self.exit_key_values:
                        # Here we would copy the stored KV states to the current layer's cache
                        # This is model-specific and depends on the exact KV cache format
                        pass

        return key_value_states

    def compute_early_exit_statistics(self) -> Dict[str, float]:
        """Compute statistics about early exits across all layers."""
        percent_skipped = self.record_exit_decisions()

        stats = {
            "compute_saved": percent_skipped,
        }

        return stats

    def reset_statistics(self):
        """Reset all tracking statistics."""
        self.exit_decisions = {}
        self.exit_hidden_states = {}
        self.exit_key_values = {}
        self.exit_layer_counts = {}
        self.total_tokens = 0
        self.current_step = 0

        for module in self.wrapped_modules.values():
            module.tokens_processed = 0
            module.tokens_exited_early = 0

    def _get_device(self) -> torch.device:
        """Get device from first registered module."""
        if self.wrapped_modules:
            first_module = next(iter(self.wrapped_modules.values()))
            return next(first_module.parameters()).device
        return torch.device("cpu")

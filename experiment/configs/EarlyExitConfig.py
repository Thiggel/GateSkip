from pydantic import Field
from enum import Enum


class ConfidenceMeasure(str, Enum):
    SOFTMAX = "softmax"  # Difference between top two softmax probabilities
    HIDDEN_STATE = "hidden_state"  # Cosine similarity between layers


class EarlyExitMethod(str, Enum):
    CALM = "calm"
    FREE = "free"


class EarlyExitConfig:
    """Configuration for early exiting mechanism"""

    use_early_exit: bool = Field(False, description="Whether to use early exiting")
    confidence_measure: ConfidenceMeasure = Field(
        "softmax",
        description="Method to calculate confidence for early exiting",
    )
    fixed_exit_layer: int = Field(
        -1, description="If > 0, always exit at this layer (for baseline comparison)"
    )
    base_threshold: float = Field(
        0.9, description="Base confidence threshold for early exiting"
    )
    decay_factor: float = Field(
        4.0, description="Decay factor (τ) for threshold over generation steps"
    )
    use_decaying_threshold: bool = Field(
        False, description="Whether to use decaying threshold over generation steps"
    )
    min_exit_layer: int = Field(
        1, description="Minimum layer to consider for early exiting"
    )
    early_exit_method: EarlyExitMethod = Field(
        "calm", description="Which early exit technique to use"
    )
    use_free_distillation: bool = Field(
        False,
        description="Whether to use FREE's shallow-deep distillation loss",
    )
    use_free_speculative_decoding: bool = Field(
        False,
        description="Whether to compute speculative decoding KV cache in FREE",
    )
    free_shallow_layers: list[int] = Field(
        [],
        description="Indices defining the shallow sub-network for FREE",
    )
    free_distillation_loss_weight: float = Field(
        1.0, description="Weight for the FREE distillation loss"
    )

    # --- PABEE / DeeBERT settings ---
    use_patience_exit: bool = Field(
        False,
        description="Exit based on consecutive agreement of token predictions",
    )
    patience: int = Field(
        3,
        description="Number of consecutive layers with unchanged prediction before exit",
    )
    use_entropy_exit: bool = Field(
        False,
        description="Exit based on entropy of the LM head output (DeeBERT)",
    )
    entropy_threshold: float = Field(
        1.0, description="Entropy threshold for entropy-based early exit"
    )

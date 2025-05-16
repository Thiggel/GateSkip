from pydantic import Field
from enum import Enum


class ConfidenceMeasure(str, Enum):
    SOFTMAX = "softmax"  # Difference between top two softmax probabilities
    HIDDEN_STATE = "hidden_state"  # Cosine similarity between layers


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

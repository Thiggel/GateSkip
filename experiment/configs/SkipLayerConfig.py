from pydantic import Field


class SkipLayerConfig:
    """Configuration for SkipLayer gating mechanism"""

    use_skip_layer: bool = Field(False, description="Whether to apply SkipLayer gating")
    skip_layer_aux_loss_weight: float = Field(
        0.1, description="Weight for the SkipLayer auxiliary loss"
    )

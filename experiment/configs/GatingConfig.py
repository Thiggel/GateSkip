from pydantic import Field
from enum import Enum


class GatingType(str, Enum):
    PER_LAYER = "per_layer"  # Separate gates for each layer
    SHARED = "shared"  # Shared gates across all layers


class GenerationMode(str, Enum):
    GREEDY = "greedy"
    BEAM = "beam"
    SAMPLING = "sampling"


class GatingMode(str, Enum):
    BEFORE_MODULE = "before_module"
    AFTER_MODULE = "after_module"


class SparsityLossType(str, Enum):
    L1 = "l1"
    L2 = "l2"
    KL = "kl"


class GatingConfig:
    """Configuration for gating mechanism"""

    use_gating: bool = Field(False, description="Whether to use gating")
    gating_mode: GatingMode = Field(
        "after_module", description="Whether to gate before or after the module"
    )
    frozen_gate: bool = Field(
        False, description="Whether to freeze the gate weights during training"
    )
    use_kl_div_training: bool = Field(
        False, description="Whether to use KL divergence for training"
    )
    actually_gate: bool = Field(
        True,
        description="Whether to just compute the gate value or actually gate the output",
    )
    use_mlp_gate: bool = Field(
        False, description="Whether to use MLP for gating instead of linear layer"
    )
    single_number_gates: bool = Field(
        False, description="Whether to use output single number for gating"
    )
    one_gate_per_layer: bool = Field(
        True, description="Whether to use one gate per layer"
    )
    skip_modules: bool = Field(
        False,
        description="Whether to, based ona gate's mean of values, skip attention and mlp modules if the contribute very little to the residual stream",
    )
    randomly_skip: bool = Field(
        False, description="Whether to randomly skip modules as a baseline"
    )
    desired_skip_ratio: float = Field(
        0.0,
        description="Percentage of modules to skip",
    )
    skip_threshold: list[float] = Field(
        [0.5], description="Threshold for skipping modules"
    )
    increasing_threshold: bool = Field(
        False, description="Whether to increase the threshold during training"
    )
    start_thr_percentile: float = Field(
        0.005,
    )
    end_thr_percentile: float = Field(
        0.1,
    )
    num_increasing_steps: int = Field(
        5_000, description="Number of steps to increase the threshold over"
    )
    skip_layers: bool = Field(
        False,
        description="Whether to skip layers based on gate value",
    )
    always_skip_layers: list[int] = Field(
        [],
        description="If set, always skip this layer (0-indexed) regardless of gate value",
    )
    skip_module_types: list[str] = Field(
        ["attn", "mlp"], description="Module types to skip based on gate value"
    )
    skip_entire_layer_based_on_attn: bool = Field(
        False, description="Whether to skip entire layer based on attention gate value"
    )
    only_skip_every_second_layer: bool = Field(
        False, description="Whether to only skip every second layer"
    )
    gating_type: GatingType = Field("per_layer", description="Type of gating to use")
    gate_init_value: float = Field(5.0, description="Initial gate bias value")
    gate_init_mean: float = Field(
        0.0, description="Initial gate weight mean for normal distribution"
    )
    gate_init_std: float = Field(
        0.01, description="Initial gate weight standard deviation"
    )

    entropy_loss_weight: float = Field(
        0.01, description="Weight for entropy maximization loss"
    )
    sparsity_loss_weight: float = Field(0.01, description="Weight for sparsity loss")
    sparsity_loss_type: SparsityLossType = Field(
        SparsityLossType.L2,
        description="Type of sparsity loss to apply to gate values",
    )

    gate_attention: bool = Field(True, description="Whether to gate attention outputs")
    gate_mlp: bool = Field(True, description="Whether to gate MLP outputs")
    generation_mode: GenerationMode = Field(
        "greedy", description="Generation mode for gating"
    )
    use_self_speculative_decoding: bool = Field(
        False,
        description="Whether to run self-speculative decoding for GateSkip",
    )
    self_speculative_budget: float = Field(
        0.2,
        description="Layer skip budget used during self-speculative decoding",
    )

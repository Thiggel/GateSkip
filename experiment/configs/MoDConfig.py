from pydantic import Field


class MoDConfig:
    use_mod: bool = Field(False, description="Whether to use MoD routing")
    mod_capacity_factor: float = Field(
        0.125, description="Fraction of tokens to process in each layer"
    )
    route_every_two: bool = Field(True, description="Whether to route every two layers")
    predictor_loss_weight: float = Field(
        0.01, description="Weight for the predictor loss"
    )

from pydantic import Field


class LoraConfig:
    lora_r: int = Field(8, description="The LoRA rank")
    lora_alpha: int = Field(1, description="The LoRA alpha")
    lora_target_modules: list[str] = Field(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="The target modules for LoRA",
    )
    lora_dropout: float = Field(0.0, description="The LoRA dropout rate")

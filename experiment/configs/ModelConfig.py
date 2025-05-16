from enum import Enum
from typing import Optional, Annotated
from pydantic import BaseModel, Field

from .LoraConfig import LoraConfig
from .LayerRange import LayerRange
from .GatingConfig import GatingConfig
from .EarlyExitConfig import EarlyExitConfig
from .MoDConfig import MoDConfig


class FinetuneMode(str, Enum):
    FULL = "full"
    LORA = "lora"
    FROZEN = "frozen"


class ModelConfig(BaseModel, LoraConfig, GatingConfig, MoDConfig, EarlyExitConfig):
    """Configuration for model architecture and behavior"""

    model_name: str = Field("gpt2", description="The model name to be used")
    pretrained: bool = Field(
        False, description="Whether to use a pretrained model or not"
    )
    finetune_mode: FinetuneMode = Field(
        "lora", description="The finetuning mode to use"
    )
    remove_layers: Optional[Annotated[str, LayerRange]] = Field(
        None, description="The layers to remove from the model"
    )

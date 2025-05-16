from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class TrainingProcedureType(str, Enum):
    DEFAULT = "default"
    REINFORCE = "reinforce"


class TrainingConfig(BaseModel):
    """Configuration for training parameters"""

    use_deepspeed: bool = Field(
        False, description="Whether to use DeepSpeed for training"
    )
    max_epochs: Optional[int] = Field(None, description="The maximum number of epochs")
    max_training_steps: Optional[int] = Field(
        None, description="The maximum number of epochs"
    )
    max_hours: Optional[int] = Field(
        None, description="The maximum number of hours to train"
    )
    use_early_stopping: bool = Field(
        False, description="Whether to use early stopping during training"
    )
    early_stopping_patience: int = Field(
        1, description="The number of epochs to wait before stopping"
    )
    learning_rate: float = Field(1e-3, description="The learning rate for the model")
    initial_lr: float = Field(0, description="The initial learning rate for the model")
    warmup_steps: int = Field(50, description="The number of warmup steps")
    lr_decay_steps: int = Field(100000000, description="The number of steps before decaying")
    max_grad_norm: float = Field(
        1.0, description="The maximum gradient norm for gradient clipping"
    )
    training_procedure: Optional[TrainingProcedureType] = Field(
        None, description="The training procedure to use"
    )
    gamma: float = Field(0.99, description="The discount factor")
    temperature: float = Field(1.0, description="The sampling temperature")
    baseline_decay: float = Field(0.99, description="The decay factor for the baseline")
    validate_every_n_steps: Optional[int] = Field(
        None, description="The number of steps between validation runs"
    )
    use_profiler: bool = Field(False, description="Whether to use the PyTorch profiler")

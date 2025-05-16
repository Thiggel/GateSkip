from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for dataset and data processing"""

    dataset: str = Field("gsm8k", description="The dataset to use for training")
    seq_length: int = Field(1024, description="The maximum sequence length")
    batch_size: int = Field(8, description="The batch size")
    grad_acc_steps: int = Field(1, description="Number of gradient accumulation steps")
    val_dataset_size: int = Field(
        1000, description="Number of samples in the validation dataset"
    )

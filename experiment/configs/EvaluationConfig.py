from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation and checkpointing"""

    evaluation_metrics: Optional[list[str]] = Field(
        None, description="The evaluation metrics to use"
    )
    limit: Optional[int] = Field(
        400, description="The number of examples to use for full evaluation"
    )
    load_from_checkpoint: Optional[Path] = Field(
        None, description="The path to the checkpoint to load"
    )
    save_to_checkpoint: Optional[Path] = Field(
        None, description="The path to the checkpoint to save to"
    )
    eval_batch_size: int = Field(
        300, description="The batch size to use for evaluation"
    )
    auto_adjust_batch_size: bool = Field(
        False,
        description="Automatically scale eval batch size based on skip budget",
    )
    num_fewshot: int = Field(0, description="The number of few-shot examples to use")
    use_quantization: bool = Field(
        False, description="Whether to use quantization for evaluation"
    )
    save_token_importance_histogram: bool = Field(
        False,
        description="Save histogram of token importance during evaluation",
    )

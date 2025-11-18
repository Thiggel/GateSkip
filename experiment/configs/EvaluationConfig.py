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
    num_fewshot: int = Field(0, description="The number of few-shot examples to use")
    use_quantization: bool = Field(
        False, description="Whether to use quantization for evaluation"
    )
    use_vllm_backend: bool = Field(
        False,
        description=(
            "Whether to evaluate with the optional vLLM backend. When disabled,"
            " the Hugging Face flow is preserved."
        ),
    )
    vllm_kernel_mode: str = Field(
        "torch",
        description=(
            "Kernel mode for vLLM-backed GateSkip evaluation. `torch` keeps"
            " the existing PyTorch path, `triton` attempts to use the custom"
            " Triton kernel when available."
        ),
    )
    vllm_autotune_batch: bool = Field(
        True,
        description=(
            "Whether to auto-tune batch size for vLLM evaluations based on"
            " measured throughput/utilization."
        ),
    )
    vllm_max_autotune_batch: int = Field(
        1024,
        description="Upper bound for batch-size search when auto-tuning vLLM.",
    )
    vllm_autotune_trials: int = Field(
        5,
        description="Number of batch-size candidates to probe when auto-tuning.",
    )
    vllm_profile_warmups: int = Field(
        1,
        description="Warmup iterations before profiling wall-clock throughput.",
    )
    vllm_profile_steps: int = Field(
        3,
        description="Profiling iterations per candidate batch size.",
    )
    save_token_importance_histogram: bool = Field(
        False,
        description="Save histogram of token importance during evaluation",
    )

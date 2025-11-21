from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedTokenizer

from experiment.configs import DataConfig, EvaluationConfig, ModelConfig
from experiment.utils.vllm_profiler import ProfileResult, VLLMBatchSizer
from .ModelEvaluator import ModelEvaluator

try:  # Optional dependency
    from lm_eval.models.vllm import VLLM as EvalVLLM
except Exception:  # pragma: no cover - optional import
    EvalVLLM = None


class VLLMModelEvaluator(ModelEvaluator):
    """Drop-in evaluator that prefers vLLM when available.

    Falls back to the Hugging Face `HFLM` path if `vllm` or the harness
    backend is unavailable, while still exposing profiling information.
    """

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig,
        evaluation_config: EvaluationConfig,
        data_config: DataConfig,
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            evaluation_config.eval_batch_size,
            evaluation_config.num_fewshot,
            data_config.seq_length,
        )
        self.model_config = model_config
        self.evaluation_config = evaluation_config
        self.data_config = data_config
        self.profile_result: Optional[ProfileResult] = None
        self.profile_skip_reason: Optional[str] = None

    def _maybe_autotune_batch_size(self) -> None:
        if not self.evaluation_config.vllm_autotune_batch:
            self.profile_skip_reason = "vLLM batch autotuning disabled"
            return
        if not torch.cuda.is_available():
            self.profile_skip_reason = "CUDA is unavailable for vLLM profiling"
            return

        hidden_size = getattr(self.model.model.config, "hidden_size", None)
        layers = self.model.get_decoder_layers(self.model.model)
        num_layers = len(layers) if layers is not None else 0

        if hidden_size is None or num_layers == 0:
            self.profile_skip_reason = "Model metadata missing (hidden_size or decoder layers)"
            return

        tuner = VLLMBatchSizer(
            hidden_size=hidden_size,
            num_layers=num_layers,
            seq_length=self.data_config.seq_length,
            max_batch=self.evaluation_config.vllm_max_autotune_batch,
            trials=self.evaluation_config.vllm_autotune_trials,
            warmups=self.evaluation_config.vllm_profile_warmups,
            steps=self.evaluation_config.vllm_profile_steps,
            skip_ratio=self.model_config.desired_skip_ratio,
        )

        def step_fn(batch_size: int):
            # Synthetic batch to probe throughput; uses gate-aware forward.
            device = next(self.model.parameters()).device
            dummy = torch.ones(
                (batch_size, self.data_config.seq_length),
                dtype=torch.long,
                device=device,
            )
            attn = torch.ones_like(dummy)
            with torch.no_grad():
                _ = self.model(input_ids=dummy, attention_mask=attn)

        result = tuner.autotune(step_fn)
        if result is not None:
            self.eval_batch_size = result.batch_size
            self.profile_result = result
            self.profile_skip_reason = None
        else:
            self.profile_skip_reason = "vLLM batch autotuning failed across all trials"

    def _build_model_wrapper(self, gen_kwargs: Dict[str, Any]):
        if EvalVLLM is None:
            return super().dict_to_str(gen_kwargs), None

        wrapped_model = EvalVLLM(
            pretrained=self.model_config.model_name,
            tokenizer=self.model_config.model_name,
            dtype="auto",
            max_model_len=self.data_config.seq_length,
            tensor_parallel_size=max(torch.cuda.device_count(), 1),
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
        )
        return self.dict_to_str(gen_kwargs), wrapped_model

    def evaluate(
        self,
        metrics: list[str],
        seed: int,
        experiment_name: str,
        generation_mode,
        limit: int = 10000,
    ) -> dict[str, float]:
        # Prefer vLLM backend when available, otherwise fall back gracefully
        self.profile_result = None
        self.profile_skip_reason = None
        self._maybe_autotune_batch_size()
        gen_kwargs = self.get_gen_kwargs(generation_mode)

        gen_kwargs_str, wrapped_model = self._build_model_wrapper(gen_kwargs)
        if wrapped_model is None:
            if self.profile_skip_reason is None:
                self.profile_skip_reason = "vLLM package unavailable; using HuggingFace backend"
            return super().evaluate(
                metrics=metrics,
                seed=seed,
                experiment_name=experiment_name,
                generation_mode=generation_mode,
                limit=limit,
            )

        from lm_eval.tasks import TaskManager
        from lm_eval import evaluator

        include_path = Path(__file__).resolve().parents[2] / "lm_eval" / "tasks"
        tm = TaskManager(include_path=str(include_path))

        output = evaluator.simple_evaluate(
            model=wrapped_model,
            tasks=metrics or ["commonsense_qa", "gsm8k", "piqa"],
            num_fewshot=self.num_fewshot,
            batch_size=self.eval_batch_size,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            device=self.device,
            log_samples=True,
            gen_kwargs=gen_kwargs_str,
            task_manager=tm,
            limit=limit,
        )

        results = output.get("results", {})
        if self.profile_result is not None:
            results["vllm_profile"] = {
                "batch_size": self.profile_result.batch_size,
                "wall_clock_s": self.profile_result.wall_clock_s,
                "tflops_per_s": self.profile_result.tflops_per_s,
                "tokens_per_s": self.profile_result.tokens_per_s,
                "flops_per_token": self.profile_result.flops_per_token,
                "wall_clock_per_token": self.profile_result.wall_clock_per_token,
                "tokens": self.profile_result.tokens,
                "peak_memory_bytes": self.profile_result.peak_memory_bytes,
                "peak_memory_gb": self.profile_result.peak_memory_gb,
            }
        elif self.profile_skip_reason:
            results["vllm_profile_skip_reason"] = self.profile_skip_reason

        self._save_results(results, experiment_name)
        self._save_samples(output.get("samples", {}), seed, experiment_name)

        return results

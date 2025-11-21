from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch


def _estimate_flops_per_token(hidden_size: int, num_layers: int) -> float:
    """Rough FLOP estimate for a transformer decoder layer.

    The constant captures attention + MLP projections; this is intentionally
    approximate but consistent for relative comparisons when tuning.
    """

    # 12 * hidden^2 is a common proxy for decoder per-token flops
    return 12.0 * hidden_size * hidden_size * num_layers


@dataclass
class ProfileResult:
    batch_size: int
    wall_clock_s: float
    flops: float
    tokens: int
    peak_memory_bytes: int = 0

    @property
    def tflops_per_s(self) -> float:
        if self.wall_clock_s == 0:
            return 0.0
        return (self.flops / 1e12) / self.wall_clock_s

    @property
    def tokens_per_s(self) -> float:
        if self.wall_clock_s == 0:
            return 0.0
        return self.tokens / self.wall_clock_s

    @property
    def flops_per_token(self) -> float:
        if self.tokens == 0:
            return 0.0
        return self.flops / self.tokens

    @property
    def wall_clock_per_token(self) -> float:
        if self.tokens == 0:
            return 0.0
        return self.wall_clock_s / self.tokens

    @property
    def peak_memory_gb(self) -> float:
        return self.peak_memory_bytes / (1024**3)


class VLLMBatchSizer:
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        max_batch: int,
        trials: int,
        warmups: int,
        steps: int,
        skip_ratio: float,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.max_batch = max_batch
        self.trials = trials
        self.warmups = warmups
        self.steps = steps
        self.skip_ratio = skip_ratio

    def _profile_batch(
        self, batch_size: int, step_fn: Callable[[int], None]
    ) -> Optional[ProfileResult]:
        try:
            peak_mem = 0
            is_cuda = torch.cuda.is_available()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()

            for _ in range(self.warmups):
                step_fn(batch_size)

            torch.cuda.synchronize() if is_cuda else None
            start = time.perf_counter()
            for _ in range(self.steps):
                step_fn(batch_size)
            torch.cuda.synchronize() if is_cuda else None
            wall = time.perf_counter() - start

            if is_cuda:
                allocated = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.max_memory_reserved()
                peak_mem = max(allocated, reserved)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"vLLM batch profiling failed for batch {batch_size}: {exc}")
            return None

        tokens = batch_size * self.seq_length * self.steps
        effective_layers = max(1.0 - self.skip_ratio, 0.0) * self.num_layers
        flops = _estimate_flops_per_token(self.hidden_size, effective_layers) * tokens

        return ProfileResult(
            batch_size=batch_size,
            wall_clock_s=wall,
            flops=flops,
            tokens=tokens,
            peak_memory_bytes=peak_mem,
        )

    def autotune(self, step_fn: Callable[[int], None]) -> Optional[ProfileResult]:
        if self.trials <= 0:
            return None

        candidates = [
            2*i for i in range(1, 32)
        ]

        best: Optional[ProfileResult] = None
        for batch_size in candidates:
            result = self._profile_batch(batch_size, step_fn)
            if result is None:
                continue
            if best is None or result.tokens_per_s > best.tokens_per_s:
                best = result

        assert best is not None
        return best

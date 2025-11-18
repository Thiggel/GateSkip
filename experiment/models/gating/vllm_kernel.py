from __future__ import annotations

from typing import Tuple

import torch

try:  # Optional dependency
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional import
    triton = None
    tl = None


def _maybe_flatten(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Flatten to 2D for Triton and return original shape for reshaping."""
    shape = tensor.shape
    if tensor.ndim == 3:
        return tensor.view(-1, shape[-1]), shape
    return tensor, shape


if triton is not None:

    @triton.jit
    def _fused_gate_skip(
        output_ptr,
        gate_ptr,
        mask_ptr,
        result_ptr,
        total_hidden: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_h = tl.arange(0, BLOCK_H)
        row_start = pid * total_hidden
        idx = row_start + offs_h

        mask = tl.load(mask_ptr + pid, mask=True, other=1.0)
        gate = tl.load(gate_ptr + idx, mask=offs_h < total_hidden, other=0.0)
        output = tl.load(output_ptr + idx, mask=offs_h < total_hidden, other=0.0)

        gated = gate * output
        gated = tl.where(mask > 0.0, 0.0, gated)

        tl.store(result_ptr + idx, gated, mask=offs_h < total_hidden)


def gate_skip_kernel(
    main_output: torch.Tensor,
    gate_value: torch.Tensor,
    skip_mask: torch.Tensor,
    block_size: int = 256,
    apply_gate: bool = True,
) -> torch.Tensor:
    """Fused gate + skip with an optional Triton kernel.

    Falls back to the original PyTorch path when Triton is unavailable or
    when tensors are on CPU. Inputs are expected to be shaped `[B, T, H]`.
    """

    if triton is None or not main_output.is_cuda:
        gated = gate_value * main_output if apply_gate else main_output
        return torch.where(skip_mask, torch.zeros_like(gated), gated)

    output_flat, shape = _maybe_flatten(main_output)
    gate_source = gate_value if apply_gate else torch.ones_like(main_output)
    gate_flat, _ = _maybe_flatten(gate_source)
    mask_flat, _ = _maybe_flatten(skip_mask.float())

    # mask_flat is shape [B*T, 1]; drop trailing dim for kernel convenience
    mask_flat = mask_flat.view(-1)

    total_rows = output_flat.shape[0]
    grid = (total_rows,)

    result = torch.empty_like(output_flat)
    _fused_gate_skip[grid](
        output_flat,
        gate_flat,
        mask_flat,
        result,
        output_flat.shape[-1],
        BLOCK_H=block_size,
    )

    return result.view(*shape)

"""Physical token skipping for GateSkip.

The masking path in :mod:`experiment.models.gating.vllm_kernel` runs every
module densely and zeroes the outputs of skipped tokens afterwards. It is
correct, but it saves no arithmetic: a token that is "skipped" costs exactly
as much as a token that is kept.

This module implements the alternative: skipped tokens are physically removed
from the token dimension before the module runs, so the underlying GEMMs are
launched at a smaller ``M``. The saving therefore shows up in wall-clock time
and in measured FLOPs rather than only in an analytic FLOP model.

Two paths are provided:

``compact_token_module``
    For modules that act independently on each token (the MLP branch). This is
    exact: kept tokens see bit-identical arithmetic, up to the reassociation
    a different GEMM shape may introduce.

``compact_llama_attention``
    For Llama-style attention. Attention is *not* token-independent, so only
    the query side can be compacted. Keys and values are still computed for
    every position, because kept tokens must be able to attend to skipped
    ones. Queries, attention scores, the attention-weighted sum and the output
    projection all shrink with the skip ratio; ``k_proj``/``v_proj`` do not.

The elementwise gather/scatter is available as a Triton kernel and as a
PyTorch fallback. Note that the compaction, not the kernel, is what saves the
arithmetic -- the kernel only avoids materialising an extra copy around it.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:  # Optional dependency
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional import
    triton = None
    tl = None


# ---------------------------------------------------------------------------
# Row gather / scatter
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _gather_rows_kernel(
        src_ptr,
        idx_ptr,
        dst_ptr,
        hidden: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """dst[i, :] = src[idx[i], :] -- one program per output row."""
        row = tl.program_id(0)
        src_row = tl.load(idx_ptr + row).to(tl.int64)
        src_base = src_row * hidden
        dst_base = row.to(tl.int64) * hidden

        # Loop over the hidden dimension: BLOCK_H is a tuning parameter and is
        # independent of the model width.
        for start in range(0, hidden, BLOCK_H):
            offs = start + tl.arange(0, BLOCK_H)
            m = offs < hidden
            vals = tl.load(src_ptr + src_base + offs, mask=m, other=0.0)
            tl.store(dst_ptr + dst_base + offs, vals, mask=m)

    @triton.jit
    def _scatter_rows_kernel(
        src_ptr,
        idx_ptr,
        gate_ptr,
        dst_ptr,
        hidden: tl.constexpr,
        HAS_GATE: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """dst[idx[i], :] = src[i, :] * gate[idx[i], :] -- one program per row.

        ``dst`` must be pre-zeroed: rows that are not named by ``idx`` are the
        skipped tokens and must stay zero so the residual passes through
        unchanged.
        """
        row = tl.program_id(0)
        dst_row = tl.load(idx_ptr + row).to(tl.int64)
        dst_base = dst_row * hidden
        src_base = row.to(tl.int64) * hidden

        for start in range(0, hidden, BLOCK_H):
            offs = start + tl.arange(0, BLOCK_H)
            m = offs < hidden
            vals = tl.load(src_ptr + src_base + offs, mask=m, other=0.0)
            if HAS_GATE:
                g = tl.load(gate_ptr + dst_base + offs, mask=m, other=0.0)
                vals = vals * g
            tl.store(dst_ptr + dst_base + offs, vals, mask=m)


def _use_triton(tensor: torch.Tensor, block_size: int) -> bool:
    return triton is not None and tensor.is_cuda and block_size > 0


def gather_rows(
    src: torch.Tensor, index: torch.Tensor, block_size: int = 256
) -> torch.Tensor:
    """Select ``index`` rows from a 2D ``src``.

    Falls back to ``index_select`` off CUDA. ``index_select`` is already a
    well-tuned copy kernel, so the Triton path is a small constant-factor win
    at best -- it exists so the gather can later be fused into the projection
    epilogue.
    """
    assert src.ndim == 2, f"expected [N, H], got {tuple(src.shape)}"
    if not _use_triton(src, block_size):
        return src.index_select(0, index)

    src = src.contiguous()
    dst = torch.empty(
        (index.numel(), src.shape[1]), dtype=src.dtype, device=src.device
    )
    _gather_rows_kernel[(index.numel(),)](
        src, index, dst, hidden=src.shape[1], BLOCK_H=block_size
    )
    return dst


def scatter_rows(
    values: torch.Tensor,
    index: torch.Tensor,
    out_rows: int,
    gate: Optional[torch.Tensor] = None,
    block_size: int = 256,
) -> torch.Tensor:
    """Scatter ``values`` back to their original rows, zero-filling the rest.

    When ``gate`` is given (shape ``[out_rows, H]``) it is applied to the
    scattered rows, fusing the GateSkip gate multiply into the scatter.
    """
    assert values.ndim == 2, f"expected [N, H], got {tuple(values.shape)}"
    hidden = values.shape[1]

    if not _use_triton(values, block_size):
        out = values.new_zeros((out_rows, hidden))
        if gate is not None:
            values = values * gate.index_select(0, index)
        out.index_copy_(0, index, values)
        return out

    values = values.contiguous()
    out = values.new_zeros((out_rows, hidden))
    _scatter_rows_kernel[(index.numel(),)](
        values,
        index,
        gate if gate is not None else values,
        out,
        hidden=hidden,
        HAS_GATE=gate is not None,
        BLOCK_H=block_size,
    )
    return out


# ---------------------------------------------------------------------------
# Token-independent modules (MLP)
# ---------------------------------------------------------------------------


def compact_token_module(
    module: nn.Module,
    hidden_states: torch.Tensor,
    keep_mask: torch.Tensor,
    gate_value: Optional[torch.Tensor] = None,
    block_size: int = 256,
) -> torch.Tensor:
    """Run ``module`` on kept tokens only and scatter the result back.

    Valid only for modules that map token ``t`` to output ``t`` without
    cross-token interaction -- i.e. the MLP branch, not attention.

    ``keep_mask`` is ``[B, T]`` (True = compute this token). Returns ``[B, T, H]``
    with zeros at skipped positions, matching the masked path's contract.
    """
    batch, seq, hidden = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden)
    index = keep_mask.reshape(-1).nonzero(as_tuple=True)[0]

    if index.numel() == 0:
        return torch.zeros_like(hidden_states)

    compact = gather_rows(flat, index, block_size).unsqueeze(0)
    out = module(compact)
    if isinstance(out, tuple):
        out = out[0]
    out = out.reshape(-1, out.shape[-1])

    gate_flat = (
        gate_value.reshape(-1, gate_value.shape[-1]) if gate_value is not None else None
    )
    scattered = scatter_rows(
        out, index, batch * seq, gate=gate_flat, block_size=block_size
    )
    return scattered.view(batch, seq, -1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, heads, seq, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, heads, n_rep, seq, head_dim)
    return x.reshape(batch, heads * n_rep, seq, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def build_query_index(keep_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad per-row kept positions to a common length.

    Returns ``(index, valid)`` both ``[B, N_max]``. Padding uses position 0,
    which is always visible under a causal mask -- a fully-masked query row
    would make softmax produce NaN. Padded outputs are discarded on scatter.
    """
    batch, seq = keep_mask.shape
    counts = keep_mask.sum(dim=1)
    n_max = int(counts.max().item())
    if n_max == 0:
        empty = keep_mask.new_zeros((batch, 0), dtype=torch.long)
        return empty, empty.bool()

    order = torch.argsort(keep_mask.to(torch.int8), dim=1, descending=True, stable=True)
    index = order[:, :n_max]
    valid = torch.arange(n_max, device=keep_mask.device)[None, :] < counts[:, None]
    index = torch.where(valid, index, torch.zeros_like(index))
    return index, valid


def compact_llama_attention(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    keep_mask: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    gate_value: Optional[torch.Tensor] = None,
    block_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query-side compacted attention for Llama-style modules.

    Keys and values are computed for every position -- kept tokens must still
    attend to skipped ones. Queries, scores, the weighted sum and ``o_proj``
    shrink with the skip ratio.

    Returns ``(attn_output, key_states, value_states)``; the caller owns cache
    handling. ``attn_output`` is ``[B, T, H]`` with zeros at skipped positions.
    """
    batch, seq, hidden = hidden_states.shape
    head_dim = attn.head_dim
    cos, sin = position_embeddings

    # Keys/values: dense, over all positions.
    kv_shape = (batch, seq, -1, head_dim)
    key_states = attn.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(kv_shape).transpose(1, 2)

    index, valid = build_query_index(keep_mask)
    if index.shape[1] == 0:
        zeros = torch.zeros_like(hidden_states)
        return zeros, key_states, value_states

    n_max = index.shape[1]
    flat_index = (index + torch.arange(batch, device=index.device)[:, None] * seq).reshape(-1)

    # Queries: kept positions only.
    q_in = gather_rows(hidden_states.reshape(-1, hidden), flat_index, block_size)
    q_in = q_in.view(batch, n_max, hidden)
    query_states = attn.q_proj(q_in).view(batch, n_max, -1, head_dim).transpose(1, 2)

    # RoPE must use each token's original position, so gather cos/sin too.
    q_cos = cos.gather(1, index.unsqueeze(-1).expand(-1, -1, cos.shape[-1]))
    q_sin = sin.gather(1, index.unsqueeze(-1).expand(-1, -1, sin.shape[-1]))
    k_cos, k_sin = cos.unsqueeze(1), sin.unsqueeze(1)
    key_states = key_states * k_cos + _rotate_half(key_states) * k_sin
    query_states = (
        query_states * q_cos.unsqueeze(1) + _rotate_half(query_states) * q_sin.unsqueeze(1)
    )

    keys = _repeat_kv(key_states, attn.num_key_value_groups)
    values = _repeat_kv(value_states, attn.num_key_value_groups)

    # Mask for compacted queries: gather the query rows of the dense mask, or
    # rebuild causality from original positions when no mask was supplied.
    if attention_mask is not None:
        mask = attention_mask[:, :, :seq, :seq] if attention_mask.dim() == 4 else attention_mask
        gather_idx = index[:, None, :, None].expand(
            mask.shape[0], mask.shape[1], n_max, mask.shape[-1]
        )
        q_mask = mask.gather(2, gather_idx)
    else:
        key_pos = torch.arange(seq, device=hidden_states.device)
        allowed = key_pos[None, None, None, :] <= index[:, None, :, None]
        q_mask = torch.zeros(
            (batch, 1, n_max, seq), dtype=query_states.dtype, device=query_states.device
        ).masked_fill(~allowed, torch.finfo(query_states.dtype).min)

    attn_out = F.scaled_dot_product_attention(
        query_states,
        keys,
        values,
        attn_mask=q_mask,
        dropout_p=0.0,
        scale=attn.scaling,
    )
    attn_out = attn_out.transpose(1, 2).reshape(batch, n_max, -1)
    attn_out = attn.o_proj(attn_out)

    # Drop padded query slots, then scatter kept outputs back.
    keep_flat = valid.reshape(-1)
    out = scatter_rows(
        attn_out.reshape(-1, hidden)[keep_flat],
        flat_index[keep_flat],
        batch * seq,
        gate=gate_value.reshape(-1, hidden) if gate_value is not None else None,
        block_size=block_size,
    )
    return out.view(batch, seq, hidden), key_states, value_states


def attention_is_compactable(attn: nn.Module) -> bool:
    """Whether ``attn`` exposes the Llama-style surface this path assumes."""
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "num_key_value_groups")
    return all(hasattr(attn, name) for name in required)

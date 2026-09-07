"""Measure the FLOPs GateSkip actually saves, masked vs physically compacted.

The point of the comparison is that the masking path's cost is flat in the
skip ratio -- it computes every token and throws results away -- while the
compacted path's cost falls with it. FLOPs are *counted* by PyTorch's
dispatcher, not estimated from an analytic model.

Run: ``python scripts/bench_physical_skip.py``
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import time

import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRotaryEmbedding,
)

_spec = importlib.util.spec_from_file_location(
    "physical_skip",
    pathlib.Path(__file__).resolve().parents[1]
    / "experiment/models/gating/physical_skip.py",
)
physical_skip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(physical_skip)


def causal_mask(batch: int, seq: int, dtype=torch.float32) -> torch.Tensor:
    pos = torch.arange(seq)
    allowed = pos[None, :] <= pos[:, None]
    mask = torch.zeros((seq, seq), dtype=dtype).masked_fill(
        ~allowed, torch.finfo(dtype).min
    )
    return mask[None, None].expand(batch, 1, seq, seq).contiguous()


def count_flops(fn) -> int:
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        fn()
    return counter.get_total_flops()


def time_call(fn, repeats: int = 5) -> float:
    with torch.no_grad():
        fn()  # warmup
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=args.hidden,
        intermediate_size=4 * args.hidden,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        num_hidden_layers=1,
        max_position_embeddings=args.seq,
        attn_implementation="sdpa",
    )
    mlp = LlamaMLP(config).eval()
    attn = LlamaAttention(config, layer_idx=0).eval()
    rotary = LlamaRotaryEmbedding(config)

    hidden = torch.randn(args.batch, args.seq, args.hidden)
    position_ids = torch.arange(args.seq).unsqueeze(0).expand(args.batch, -1)
    pos_emb = rotary(hidden, position_ids)
    mask = causal_mask(args.batch, args.seq)

    print(
        f"Llama-style layer: batch={args.batch} seq={args.seq} "
        f"hidden={args.hidden} heads={args.heads}/{args.kv_heads}\n"
    )
    header = (
        f"{'skip':>6} | {'masked GFLOP':>13} {'physical GFLOP':>15} {'saved':>7}"
        f" | {'masked ms':>10} {'physical ms':>12} {'speedup':>8}"
    )

    for label, dense_fn, physical_fn in (
        (
            "MLP",
            lambda keep: (lambda: torch.where(keep.unsqueeze(-1), mlp(hidden), torch.zeros(1))),
            lambda keep: (lambda: physical_skip.compact_token_module(mlp, hidden, keep)),
        ),
        (
            "Attention",
            lambda keep: (
                lambda: torch.where(
                    keep.unsqueeze(-1),
                    attn(
                        hidden_states=hidden,
                        position_embeddings=pos_emb,
                        attention_mask=mask,
                    )[0],
                    torch.zeros(1),
                )
            ),
            lambda keep: (
                lambda: physical_skip.compact_llama_attention(
                    attn, hidden, keep, pos_emb, attention_mask=mask
                )
            ),
        ),
    ):
        print(f"== {label} ==")
        print(header)
        for skip_ratio in (0.0, 0.15, 0.25, 0.35, 0.5, 0.7):
            g = torch.Generator().manual_seed(0)
            keep = torch.rand((args.batch, args.seq), generator=g) > skip_ratio

            masked_flops = count_flops(dense_fn(keep)) / 1e9
            physical_flops = count_flops(physical_fn(keep)) / 1e9
            masked_ms = time_call(dense_fn(keep)) * 1e3
            physical_ms = time_call(physical_fn(keep)) * 1e3

            saved = 1.0 - physical_flops / masked_flops if masked_flops else 0.0
            speedup = masked_ms / physical_ms if physical_ms else 0.0
            print(
                f"{skip_ratio:>6.0%} | {masked_flops:>13.2f} {physical_flops:>15.2f}"
                f" {saved:>6.1%} | {masked_ms:>10.1f} {physical_ms:>12.1f}"
                f" {speedup:>7.2f}x"
            )
        print()


if __name__ == "__main__":
    main()

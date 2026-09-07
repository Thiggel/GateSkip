# Physical token skipping

The default GateSkip path computes every module densely and zeroes the outputs
of skipped tokens afterwards. That is correct but saves no arithmetic: a
skipped token costs exactly what a kept token costs. Physical skipping removes
skipped tokens from the token dimension before the module runs, so the GEMMs
are launched at a smaller `M` and the saving is real.

Both paths produce the same tensor. The only difference is cost.

## Enabling it

Two config flags, both on `GatingConfig`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `physically_skip` | `False` | Compact the token dimension instead of masking. |
| `physical_skip_min_density` | `0.9` | Stay dense when the kept fraction is above this. Below ~10% skipping the gather/scatter costs more than it saves. |

On the command line:

```bash
python -m experiment \
  --model-name meta-llama/Llama-3.2-1B \
  --pretrained --use-gating --skip-modules \
  --mode evaluate \
  --load-from-checkpoint <your-gateskip-checkpoint> \
  --physically-skip
```

## Using it with a trained checkpoint

Physical skipping changes nothing about training or about the checkpoint
format. Train the gates however you normally would, then turn the flag on at
evaluation time. A checkpoint trained before this change works unmodified.

```python
from experiment.configs import ModelConfig

config = ModelConfig(
    model_name="meta-llama/Llama-3.2-1B",
    pretrained=True,
    use_gating=True,
    skip_modules=True,
    physically_skip=True,      # the only line that differs
    desired_skip_ratio=0.35,
)
```

The gate values, the per-layer quantile thresholds and the resulting skip
decisions are identical either way -- `physically_skip` is consulted only
after `skip_mask` has been computed.

## What actually gets skipped

| Branch | Compacted | Saving vs. skip ratio |
| --- | --- | --- |
| MLP | Yes, fully | 1:1 |
| Attention | Query side only | About 1:2 |

Attention cannot be compacted on the key/value side. A kept token must still
be able to attend to a skipped one, so keys and values are computed for every
position. What does shrink is `q_proj`, the attention scores, the
attention-weighted sum, and `o_proj`.

This is why the end-to-end saving is smaller than the token skip ratio, and
why any measurement claiming a 1:1 translation from skip ratio to wall-clock
should be treated with suspicion.

## Limitations

- **No KV cache yet.** The attention path recomputes keys and values for the
  whole sequence, so it raises `NotImplementedError` when a cache is present.
  Use it for prefill and for evaluation; single-token decode still needs the
  masked path. Eliding K/V for skipped tokens is possible under the
  copy-forward semantics GateSkip already uses, but only when a cache exists.
- **Llama-style attention only.** `attention_is_compactable` gates this; other
  architectures silently fall back to the dense path rather than guessing at a
  layout.
- **Ragged batches dilute the saving.** Queries are compacted to the longest
  kept-token count in the batch, so a row that skips little limits the whole
  batch. Per-layer quantile thresholds keep rows similar in practice.

## The Triton kernels

`physical_skip.py` provides Triton gather/scatter kernels with an
`index_select`/`index_copy_` fallback, and fuses the gate multiply into the
scatter.

Be clear about what these do and do not buy you: **the compaction saves the
arithmetic, not the kernel.** `index_select` is already a well-tuned copy, so
the Triton path is a small constant-factor win. An elementwise kernel cannot
reduce FLOPs no matter how well written -- it runs after the dense compute has
already happened. Set `vllm_kernel_block_size` to tune it; set it to 0 to force
the PyTorch path.

## Verifying

```bash
pytest tests/test_physical_skip.py          # equivalence against the masked path
python scripts/bench_physical_skip.py       # counted FLOPs, CPU
sbatch jobs/physical_skip_bench.job         # trained adapter, GPU throughput
```

The equivalence tests are the important ones: they assert the compacted path
reproduces the masked path's output, including that a skipped token still
influences later kept tokens through attention. That last test is what
separates skipping a token's *computation* from deleting the token.

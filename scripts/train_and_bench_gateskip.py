"""Train a small GateSkip adapter, then measure real throughput while skipping.

Two things this script is careful about, because the previous efficiency
measurements got both wrong:

1. Batch size is held FIXED across skip levels. Scaling it with the skip ratio
   makes throughput rise even when no compute is saved.
2. Throughput counts the tokens actually pushed through the model, not a
   nominal ``limit * seq_length`` constant.

The comparison is masked (dense compute, outputs zeroed) versus physical
(tokens removed from the token dimension). Same checkpoint, same batch, same
tokens -- the only difference is whether the arithmetic is actually skipped.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiment.configs import EvaluationConfig, ModelConfig, TrainingConfig
from experiment.models.model_adapter.ModelAdapter import ModelAdapter


def build_model(args, tokenizer, device):
    config = ModelConfig(
        model_name=args.model_name,
        pretrained=True,
        use_gating=True,
        skip_modules=True,
        finetune_mode="frozen",
        desired_skip_ratio=0.0,
        physically_skip=False,
    )
    adapter = ModelAdapter(
        config=config,
        evaluation_config=EvaluationConfig(),
        training_config=TrainingConfig(),
        tokenizer=tokenizer,
        device=device,
        seed=args.seed,
    )
    model = adapter.model

    # The repo loads every model with attn_implementation="eager". The compacted
    # attention path calls scaled_dot_product_attention, so leaving the baseline
    # on eager would compare two changes at once -- skipping AND a faster
    # attention kernel -- and overstate the saving. Put both arms on SDPA.
    model.config._attn_implementation = args.attn_impl
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            module.config._attn_implementation = args.attn_impl

    return model, config


def set_gate_context(model, input_ids, tokenizer, global_step):
    """Mirror DefaultLightningModule.give_global_step_to_gates."""
    validity_mask = (input_ids != tokenizer.pad_token_id) & (
        input_ids != tokenizer.eos_token_id
    )
    for module in model.gating.wrapped_modules.values():
        module.global_step = global_step
        module.current_input_ids = input_ids
        module.current_validity_mask = validity_mask


def get_batches(tokenizer, args):
    from datasets import load_dataset

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in raw["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]

    usable = (ids.numel() // args.seq_length) * args.seq_length
    chunks = ids[:usable].view(-1, args.seq_length)
    return DataLoader(chunks, batch_size=args.batch_size, shuffle=True, drop_last=True)


def train_gates(model, tokenizer, args, device):
    """Frozen backbone, gates only -- the paper's frozen-backbone variant."""
    for param in model.parameters():
        param.requires_grad = False
    gate_params = []
    for module in model.gating.wrapped_modules.values():
        module.gate.requires_grad_(True)
        gate_params += list(module.gate.parameters())

    trainable = sum(p.numel() for p in gate_params)
    print(f"training {trainable/1e6:.1f}M gate parameters (backbone frozen)")

    optimizer = torch.optim.AdamW(gate_params, lr=args.learning_rate)
    loader = get_batches(tokenizer, args)
    model.train()

    step = 0
    start = time.perf_counter()
    for batch in loader:
        if step >= args.steps:
            break
        input_ids = batch.to(device)
        set_gate_context(model, input_ids, tokenizer, step)

        out = model(input_ids=input_ids, labels=input_ids)
        entropy_loss, sparsity_loss = model.gating.compute_gate_loss()
        loss = out.loss + args.sparsity_loss_weight * sparsity_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 20 == 0:
            print(
                f"  step {step:4d}  lm={out.loss.item():.3f}  "
                f"sparsity={sparsity_loss.item():.4f}",
                flush=True,
            )
        step += 1

    print(f"trained {step} steps in {time.perf_counter() - start:.1f}s")
    return {
        name: module.gate.state_dict()
        for name, module in model.gating.wrapped_modules.items()
    }


@torch.no_grad()
def measure(model, tokenizer, config, args, device, skip_ratio, physical):
    """Fixed batch, fixed tokens: only the arithmetic changes."""
    config.desired_skip_ratio = skip_ratio
    config.physically_skip = physical
    for module in model.gating.wrapped_modules.values():
        module.config = config

    model.eval()
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (args.bench_batch, args.seq_length), device=device
    )

    def one_pass():
        set_gate_context(model, input_ids, tokenizer, args.steps)
        model(input_ids=input_ids)

    for _ in range(args.warmups):
        one_pass()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.bench_steps):
        one_pass()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = args.bench_batch * args.seq_length * args.bench_steps
    skipped = [
        m.current_percent_tokens_skipped for m in model.gating.wrapped_modules.values()
    ]
    return {
        "latency_s_per_pass": elapsed / args.bench_steps,
        "throughput_tokens_per_s": tokens / elapsed,
        "measured_skip_fraction": sum(skipped) / len(skipped) if skipped else 0.0,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sparsity-loss-weight", type=float, default=1.0)
    parser.add_argument("--bench-batch", type=int, default=8)
    parser.add_argument("--bench-steps", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--attn-impl",
        default="sdpa",
        help="Attention backend for BOTH arms of the benchmark.",
    )
    parser.add_argument("--checkpoint", default="gateskip_small_adapter.pt")
    parser.add_argument("--results", default="physical_skip_throughput.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, config = build_model(args, tokenizer, device)
    model.to(device)

    gate_state = train_gates(model, tokenizer, args, device)
    torch.save(gate_state, args.checkpoint)
    print(f"saved gate checkpoint -> {args.checkpoint}")

    rows = []
    for skip_ratio in (0.0, 0.15, 0.25, 0.35, 0.5, 0.7):
        for physical in (False, True):
            torch.cuda.reset_peak_memory_stats()
            row = measure(
                model, tokenizer, config, args, device, skip_ratio, physical
            )
            row.update(requested_skip=skip_ratio, mode="physical" if physical else "masked")
            rows.append(row)
            print(
                f"skip={skip_ratio:>4.0%} {row['mode']:>8}  "
                f"{row['throughput_tokens_per_s']:>9.0f} tok/s  "
                f"{row['latency_s_per_pass']*1e3:>7.1f} ms  "
                f"peak={row['peak_memory_gb']:.2f}GB",
                flush=True,
            )

    with open(args.results, "w") as handle:
        json.dump({"args": vars(args), "rows": rows}, handle, indent=2)
    print(f"wrote {args.results}")


if __name__ == "__main__":
    main()

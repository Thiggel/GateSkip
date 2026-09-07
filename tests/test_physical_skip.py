"""Physical skipping must agree with the masked path it replaces.

The masked path is the reference: run the module densely, then zero the
skipped tokens. The compacted path must produce the same tensor while
launching smaller GEMMs.
"""

import importlib.util
import pathlib

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRotaryEmbedding,
)

# Loaded by path: importing ``experiment.models`` pulls in the whole package,
# which needs a newer Python than some environments provide. The module under
# test has no intra-package imports, so a direct load is faithful.
_spec = importlib.util.spec_from_file_location(
    "physical_skip",
    pathlib.Path(__file__).resolve().parents[1]
    / "experiment/models/gating/physical_skip.py",
)
physical_skip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(physical_skip)

build_query_index = physical_skip.build_query_index
compact_llama_attention = physical_skip.compact_llama_attention
compact_token_module = physical_skip.compact_token_module
gather_rows = physical_skip.gather_rows
scatter_rows = physical_skip.scatter_rows

BATCH, SEQ, HIDDEN, HEADS, KV_HEADS = 2, 16, 32, 4, 2
TOL = dict(atol=1e-5, rtol=1e-4)


def _config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=1,
        max_position_embeddings=SEQ,
        attn_implementation="eager",
    )


def _keep_mask(skip_ratio: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((BATCH, SEQ), generator=g) > skip_ratio


# ---------------------------------------------------------------------------
# gather / scatter
# ---------------------------------------------------------------------------


def test_gather_scatter_roundtrip_preserves_kept_rows_and_zeroes_rest():
    src = torch.randn(10, 8)
    index = torch.tensor([1, 4, 7])

    gathered = gather_rows(src, index)
    assert torch.equal(gathered, src[index])

    out = scatter_rows(gathered, index, out_rows=10)
    assert torch.equal(out[index], src[index])

    dropped = [i for i in range(10) if i not in index.tolist()]
    assert torch.all(out[dropped] == 0)


def test_scatter_applies_gate_at_destination_rows():
    values = torch.randn(3, 8)
    index = torch.tensor([0, 2, 5])
    gate = torch.randn(10, 8)

    out = scatter_rows(values, index, out_rows=10, gate=gate)
    assert torch.allclose(out[index], values * gate[index], **TOL)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("skip_ratio", [0.0, 0.25, 0.5, 0.9])
def test_compacted_mlp_matches_masked_reference(skip_ratio):
    torch.manual_seed(0)
    mlp = LlamaMLP(_config()).eval()
    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    keep = _keep_mask(skip_ratio)

    with torch.no_grad():
        reference = torch.where(keep.unsqueeze(-1), mlp(hidden), torch.zeros(1))
        compacted = compact_token_module(mlp, hidden, keep)

    assert torch.allclose(compacted, reference, **TOL)


def test_compacted_mlp_applies_gate_like_masked_path():
    torch.manual_seed(0)
    mlp = LlamaMLP(_config()).eval()
    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    gate = torch.rand(BATCH, SEQ, HIDDEN)
    keep = _keep_mask(0.4)

    with torch.no_grad():
        reference = torch.where(keep.unsqueeze(-1), gate * mlp(hidden), torch.zeros(1))
        compacted = compact_token_module(mlp, hidden, keep, gate_value=gate)

    assert torch.allclose(compacted, reference, **TOL)


def test_all_tokens_skipped_returns_zeros_without_running_module():
    mlp = LlamaMLP(_config()).eval()
    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    keep = torch.zeros((BATCH, SEQ), dtype=torch.bool)

    out = compact_token_module(mlp, hidden, keep)
    assert torch.all(out == 0)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def test_query_index_is_padded_with_position_zero_and_marked_invalid():
    keep = torch.tensor([[True, False, True, False], [False, False, False, True]])
    index, valid = build_query_index(keep)

    assert index.shape == (2, 2)
    assert index[0].tolist() == [0, 2]
    assert valid[0].tolist() == [True, True]
    # Row 1 keeps one token; the pad slot points at position 0, never NaN.
    assert index[1, 0].item() == 3
    assert valid[1].tolist() == [True, False]
    assert index[1, 1].item() == 0


def _causal_mask(dtype=torch.float32) -> torch.Tensor:
    """The 4D additive causal mask a decoder passes down to attention."""
    pos = torch.arange(SEQ)
    allowed = pos[None, :] <= pos[:, None]
    mask = torch.zeros((SEQ, SEQ), dtype=dtype).masked_fill(
        ~allowed, torch.finfo(dtype).min
    )
    return mask[None, None].expand(BATCH, 1, SEQ, SEQ).contiguous()


def _masked_attention_reference(attn, hidden, keep, pos_emb, attention_mask):
    """Dense attention, then zero the skipped tokens."""
    with torch.no_grad():
        out, _ = attn(
            hidden_states=hidden,
            position_embeddings=pos_emb,
            attention_mask=attention_mask,
        )
    return torch.where(keep.unsqueeze(-1), out, torch.zeros(1))


@pytest.mark.parametrize("skip_ratio", [0.0, 0.25, 0.5])
def test_compacted_attention_matches_masked_reference(skip_ratio):
    torch.manual_seed(0)
    config = _config()
    attn = LlamaAttention(config, layer_idx=0).eval()
    rotary = LlamaRotaryEmbedding(config)

    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    position_ids = torch.arange(SEQ).unsqueeze(0).expand(BATCH, -1)
    pos_emb = rotary(hidden, position_ids)
    keep = _keep_mask(skip_ratio, seed=1)

    mask = _causal_mask()
    reference = _masked_attention_reference(attn, hidden, keep, pos_emb, mask)
    with torch.no_grad():
        compacted, _, _ = compact_llama_attention(
            attn, hidden, keep, pos_emb, attention_mask=mask
        )

    assert torch.allclose(compacted, reference, **TOL)


def test_omitted_mask_is_treated_as_causal():
    """``attention_mask=None`` must mean causal, as a decoder implies."""
    torch.manual_seed(0)
    config = _config()
    attn = LlamaAttention(config, layer_idx=0).eval()
    rotary = LlamaRotaryEmbedding(config)

    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    position_ids = torch.arange(SEQ).unsqueeze(0).expand(BATCH, -1)
    pos_emb = rotary(hidden, position_ids)
    keep = _keep_mask(0.3, seed=3)

    with torch.no_grad():
        explicit, _, _ = compact_llama_attention(
            attn, hidden, keep, pos_emb, attention_mask=_causal_mask()
        )
        implied, _, _ = compact_llama_attention(
            attn, hidden, keep, pos_emb, attention_mask=None
        )

    assert torch.allclose(explicit, implied, **TOL)


def test_compacted_attention_keeps_dense_keys_and_values():
    """Kept tokens must still attend to skipped ones: K/V stay full length."""
    torch.manual_seed(0)
    config = _config()
    attn = LlamaAttention(config, layer_idx=0).eval()
    rotary = LlamaRotaryEmbedding(config)

    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    position_ids = torch.arange(SEQ).unsqueeze(0).expand(BATCH, -1)
    keep = _keep_mask(0.5, seed=2)

    with torch.no_grad():
        _, keys, values = compact_llama_attention(
            attn, hidden, keep, rotary(hidden, position_ids), attention_mask=None
        )

    assert keys.shape == (BATCH, KV_HEADS, SEQ, config.head_dim)
    assert values.shape == (BATCH, KV_HEADS, SEQ, config.head_dim)


def test_skipped_token_still_influences_kept_tokens():
    """Guards the semantic difference from simply deleting tokens."""
    torch.manual_seed(0)
    config = _config()
    attn = LlamaAttention(config, layer_idx=0).eval()
    rotary = LlamaRotaryEmbedding(config)

    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    position_ids = torch.arange(SEQ).unsqueeze(0).expand(BATCH, -1)
    keep = torch.ones((BATCH, SEQ), dtype=torch.bool)
    keep[:, 3] = False  # skip a token early in the sequence

    perturbed = hidden.clone()
    perturbed[:, 3] += 10.0  # change only the skipped token's content

    with torch.no_grad():
        base, _, _ = compact_llama_attention(
            attn, hidden, keep, rotary(hidden, position_ids), attention_mask=None
        )
        moved, _, _ = compact_llama_attention(
            attn, perturbed, keep, rotary(perturbed, position_ids), attention_mask=None
        )

    # Later kept tokens must react, because they attend to the skipped token.
    assert not torch.allclose(base[:, 4:], moved[:, 4:], **TOL)
    # The skipped token itself contributes nothing to the residual.
    assert torch.all(base[:, 3] == 0)

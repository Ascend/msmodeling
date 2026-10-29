# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PURPOSE ONLY.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
"""Tests for the qwen4_exp (Qwen3.8-Flash-Next) meta ops, their analytic
performance formulas and the patched official forwards.

Three layers of validation:

1. Meta ops preserve shapes/dtypes on the meta device (simulation fidelity).
2. Registered performance formulas produce the exact documented breakdown
   (compute ops per dtype + read/write/readwrite bytes).
3. The patched official module forwards run on the meta device and return
   the same output structure/shapes as the unpatched official CPU forward
   (the meta ops are shape-only stubs, so patched vs official is compared
   structurally, not numerically).

Skipped automatically when transformers < 5.16 (no qwen4_exp module).
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers", minversion="5.16")
qwen4_exp_modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

# Imports gated by importorskip above must stay below it (E402 suppressed).
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo  # noqa: E402
from tensor_cast.transformers.builtin_model.qwen4_exp import (  # noqa: E402
    _patched_gated_residual_forward,
    _patched_ngram_embedding_forward,
    _patched_ple_forward,
    patch_method_for_qwen4_exp,
)


# ---------------------------------------------------------------------------
# Meta ops: shape/dtype fidelity on the meta device
# ---------------------------------------------------------------------------


def test_meta_ops_preserve_shapes_and_dtypes():
    x = torch.empty(2, 5, 64, dtype=torch.float16, device="meta")
    weight = torch.empty(64, dtype=torch.float16, device="meta")

    out = torch.ops.tensor_cast.grouped_rmsnorm.default(x, weight, 16, 1e-6)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

    # Explicit scale_mode matches the default "one_plus_weight" convention.
    out = torch.ops.tensor_cast.grouped_rmsnorm.default(x, weight, 16, 1e-6, "one_plus_weight")
    assert out.shape == x.shape
    assert out.dtype == x.dtype

    with pytest.raises(ValueError, match="scale_mode"):
        torch.ops.tensor_cast.grouped_rmsnorm.default(x, weight, 16, 1e-6, "weight")

    mix_weight = torch.empty(2, 5, 64, dtype=torch.bfloat16, device="meta")
    normed = torch.empty_like(mix_weight)
    out = torch.ops.tensor_cast.hc_gated_stream_mean.default(mix_weight, normed, 4)
    # hidden inferred as hc_hidden // hc_count = 64 // 4.
    assert out.shape == (2, 5, 16)
    assert out.dtype == normed.dtype

    input_ids = torch.empty(2, 16, dtype=torch.long, device="meta")
    multipliers = torch.empty(3, dtype=torch.long, device="meta")
    head_vocab_sizes = torch.empty(4, dtype=torch.long, device="meta")
    head_offsets = torch.empty(4, dtype=torch.long, device="meta")
    out = torch.ops.tensor_cast.qwen4_ngram_hash.default(
        input_ids, multipliers, head_vocab_sizes, head_offsets, 3, 2, 99
    )
    # (ngram_size - 1) * heads_per_ngram = 2 * 2 hash heads.
    assert out.shape == (2, 16, 4)
    assert out.dtype == torch.long

    key_normed = torch.empty(2, 5, 4, 16, dtype=torch.bfloat16, device="meta")
    query_normed = torch.empty_like(key_normed)
    value = torch.empty(2, 5, 16, dtype=torch.bfloat16, device="meta")
    out = torch.ops.tensor_cast.qwen4_ple_gate.default(key_normed, query_normed, value, 16)
    assert out.shape == (2, 5, 64)
    assert out.dtype == value.dtype

    out = torch.ops.tensor_cast.qwen4_ple_short_conv.default(x, 4, 2)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


# ---------------------------------------------------------------------------
# Performance formulas: exact breakdown assertions
# ---------------------------------------------------------------------------


def _properties(op, args, out):
    info = OpInvokeInfo(op, args, {}, out)
    return info.get_perf_properties()


def test_grouped_rmsnorm_perf_properties_exact():
    x = torch.empty(2, 5, 64, dtype=torch.float16, device="meta")
    weight = torch.empty(64, dtype=torch.float16, device="meta")
    out = torch.empty_like(x)

    p = _properties(torch.ops.tensor_cast.grouped_rmsnorm.default, (x, weight, 16, 1e-6, "one_plus_weight"), out)

    numel = x.numel()  # 640
    num_rows = numel // 64  # 10
    groups_per_row = 64 // 16  # 4
    # RMS statistics (rows*width*5) + (1 + w) elementwise scale (2 * numel).
    expected_gp = num_rows * groups_per_row * 16 * 5 + 2 * numel
    assert p.compute_ops == {
        torch.float32: OpInvokeInfo.ComputeOps(mma_ops=0, gp_ops=expected_gp),
    }
    # x read once + weight read once + out written once + one FP32
    # intermediate round-trip (8 bytes/element).
    assert p.memory_read_bytes == numel * 2 + 64 * 2
    assert p.memory_write_bytes == numel * 2
    assert p.memory_readwrite_bytes == 8 * numel
    assert p.extra_static_cost_count == 0


def test_hc_input_mix_perf_properties_exact():
    mix_weight = torch.empty(2, 5, 64, dtype=torch.bfloat16, device="meta")
    normed = torch.empty_like(mix_weight)
    out = torch.empty(2, 5, 16, dtype=torch.bfloat16, device="meta")

    p = _properties(torch.ops.tensor_cast.hc_gated_stream_mean.default, (mix_weight, normed, 4), out)

    tokens = 10
    hc_hidden = 64
    hidden_size = 16
    # sigmoid gate + gate multiply + stream mean ((hc-1) adds + 1 div).
    expected_gp = 4 * tokens * hc_hidden + tokens * hc_hidden + tokens * hidden_size * 4
    assert p.compute_ops == {
        torch.bfloat16: OpInvokeInfo.ComputeOps(mma_ops=0, gp_ops=expected_gp),
    }
    assert p.memory_read_bytes == 2 * tokens * hc_hidden * 2
    assert p.memory_write_bytes == tokens * hidden_size * 2
    assert p.memory_readwrite_bytes == 0


def test_ngram_hash_perf_properties_exact():
    input_ids = torch.empty(2, 16, dtype=torch.long, device="meta")
    multipliers = torch.empty(3, dtype=torch.long, device="meta")
    head_vocab_sizes = torch.empty(4, dtype=torch.long, device="meta")
    head_offsets = torch.empty(4, dtype=torch.long, device="meta")
    out = torch.empty(2, 16, 4, dtype=torch.long, device="meta")

    p = _properties(
        torch.ops.tensor_cast.qwen4_ngram_hash.default,
        (input_ids, multipliers, head_vocab_sizes, head_offsets, 3, 2, 99),
        out,
    )

    ngram_size = 3
    ngram_heads = 4
    tokens = 2 * 16
    # Per token: EOS-aware shifts (4 * ngram_size) + XOR hash levels 2..N
    # (2n - 1 each) + per-head modulo/offset (2 * heads).
    per_token = 4 * ngram_size + sum(2 * n - 1 for n in range(2, ngram_size + 1)) + 2 * ngram_heads
    assert per_token == 28
    # int64 vector work charged under FP32 vector throughput.
    assert p.compute_ops == {
        torch.float32: OpInvokeInfo.ComputeOps(mma_ops=0, gp_ops=tokens * per_token),
    }
    # id window read + seeded multiplier/vocab/offset tables read once
    # (excluded from the schema inputs, added explicitly) + out written.
    assert p.memory_read_bytes == tokens * 8 + (ngram_size + 2 * ngram_heads) * 8
    assert p.memory_write_bytes == tokens * ngram_heads * 8
    assert p.memory_readwrite_bytes == 0


def test_ple_gate_perf_properties_exact():
    key_normed = torch.empty(2, 5, 4, 16, dtype=torch.bfloat16, device="meta")
    query_normed = torch.empty_like(key_normed)
    value = torch.empty(2, 5, 16, dtype=torch.bfloat16, device="meta")
    out = torch.empty(2, 5, 64, dtype=torch.bfloat16, device="meta")

    p = _properties(
        torch.ops.tensor_cast.qwen4_ple_gate.default,
        (key_normed, query_normed, value, 16),
        out,
    )

    tokens = 10
    hc_count = 4
    hidden_size = 16
    # key-query dot (mul + add) + sqrt chain + sigmoid (4/el) + broadcast.
    expected_gp = (
        tokens * hc_count * hidden_size * 2 + tokens * hc_count * (1 + 5 + 4) + tokens * hc_count * hidden_size
    )
    assert p.compute_ops == {
        torch.bfloat16: OpInvokeInfo.ComputeOps(mma_ops=0, gp_ops=expected_gp),
    }
    assert p.memory_read_bytes == (tokens * hc_count * hidden_size) * 2 * 2 + tokens * hidden_size * 2
    assert p.memory_write_bytes == tokens * hc_count * hidden_size * 2
    assert p.memory_readwrite_bytes == 0


def test_ple_short_conv_perf_properties_exact():
    x = torch.empty(2, 5, 64, dtype=torch.bfloat16, device="meta")
    out = torch.empty_like(x)

    p = _properties(torch.ops.tensor_cast.qwen4_ple_short_conv.default, (x, 4, 2), out)

    tokens = 10
    channels = 64
    kernel_size = 4
    # kernel_size MACs (mul + add) per output element + SiLU (6/el).
    # The dilation extends the zero-padded window only, not the MAC count.
    expected_gp = tokens * channels * kernel_size * 2 + 6 * tokens * channels
    assert p.compute_ops == {
        torch.bfloat16: OpInvokeInfo.ComputeOps(mma_ops=0, gp_ops=expected_gp),
    }
    # x read + depthwise conv weights read once + out written.
    assert p.memory_read_bytes == tokens * channels * 2 + channels * kernel_size * 2
    assert p.memory_write_bytes == tokens * channels * 2
    assert p.memory_readwrite_bytes == 0


def test_fused_rope_partial_perf_properties_match_qwen4_exp_call_sites():
    """QSA layers call ``fused_rope`` with the official partial-RoPE geometry:
    head_dim 256 with rotary_dim 64 (partial_rotary_factor 0.25), and the
    indexer reuses rotary_dim on its 128-wide heads. Only the leading
    ``rotary_dim`` channels rotate per head; the tail channels are pure
    memory traffic.
    """
    tokens, q_heads, kv_heads, head_dim, rotary_dim = 7, 16, 2, 256, 64
    q = torch.empty(tokens, q_heads, head_dim, dtype=torch.bfloat16, device="meta")
    k = torch.empty(tokens, kv_heads, head_dim, dtype=torch.bfloat16, device="meta")
    cos_sin = torch.empty(tokens, rotary_dim * 2, dtype=torch.bfloat16, device="meta")
    q_out, k_out = torch.ops.tensor_cast.fused_rope.default(q, k, cos_sin, rotary_dim, True)

    p = _properties(torch.ops.tensor_cast.fused_rope.default, (q, k, cos_sin, rotary_dim, True), (q_out, k_out))

    q_rotated = tokens * q_heads * rotary_dim
    k_rotated = tokens * kv_heads * rotary_dim
    assert p.compute_ops == {
        torch.bfloat16: OpInvokeInfo.ComputeOps(mma_ops=0, gp_ops=3 * (q_rotated + k_rotated)),
    }
    assert p.memory_read_bytes == tokens * (q_heads + kv_heads) * head_dim * 2 + tokens * rotary_dim * 2 * 2
    assert p.memory_write_bytes == tokens * (q_heads + kv_heads) * head_dim * 2
    assert p.memory_readwrite_bytes == 0

    # Indexer call site: 128-wide heads still rotate only rotary_dim channels.
    idx_q = torch.empty(tokens, 4, 128, dtype=torch.bfloat16, device="meta")
    idx_k = torch.empty(tokens, 1, 128, dtype=torch.bfloat16, device="meta")
    idx_q_out, idx_k_out = torch.ops.tensor_cast.fused_rope.default(idx_q, idx_k, cos_sin, rotary_dim, True)
    p_idx = _properties(
        torch.ops.tensor_cast.fused_rope.default, (idx_q, idx_k, cos_sin, rotary_dim, True), (idx_q_out, idx_k_out)
    )
    assert p_idx.compute_ops == {
        torch.bfloat16: OpInvokeInfo.ComputeOps(
            mma_ops=0, gp_ops=3 * (tokens * 4 * rotary_dim + tokens * 1 * rotary_dim)
        ),
    }


# ---------------------------------------------------------------------------
# Patched official forwards: meta-safety and shape fidelity
# ---------------------------------------------------------------------------


def _official_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=4,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.5},
        vocab_size=1000,
        eos_token_id=99,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=100,
        make_ngram_vocab_size_divisible_by=8,
        seed=7,
        output_gate_type="sigmoid",
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        max_position_embeddings=256,
    )


@contextlib.contextmanager
def _class_patches_applied():
    """Install the three module-forward patches, restoring the originals."""
    modeling = qwen4_exp_modeling
    saved = {
        modeling.Qwen4ExpTextGatedResidual: modeling.Qwen4ExpTextGatedResidual.forward,
        modeling.Qwen4ExpTextNGramEmbedding: modeling.Qwen4ExpTextNGramEmbedding.forward,
        modeling.Qwen4ExpTextPLELayer: modeling.Qwen4ExpTextPLELayer.forward,
    }
    modeling.Qwen4ExpTextGatedResidual.forward = _patched_gated_residual_forward
    modeling.Qwen4ExpTextNGramEmbedding.forward = _patched_ngram_embedding_forward
    modeling.Qwen4ExpTextPLELayer.forward = _patched_ple_forward
    try:
        yield
    finally:
        for cls, original in saved.items():
            cls.forward = original


class TestPatchedForwardsMetaSafety:
    @pytest.mark.parametrize("use_combine", [True, False])
    def test_gated_residual_matches_official_structure_on_meta(self, use_combine):
        config = _official_config()
        torch.manual_seed(0)
        official = qwen4_exp_modeling.Qwen4ExpTextGatedResidual(config, use_combine=use_combine)
        official.eval()

        batch, seq_len = 2, 7
        hc_hidden = config.hc_count * config.hidden_size
        x = torch.randn(batch, seq_len, hc_hidden)
        with torch.no_grad():
            reference = official(x)

        with _class_patches_applied(), torch.no_grad():
            meta = qwen4_exp_modeling.Qwen4ExpTextGatedResidual(config, use_combine=use_combine).to("meta")
            x_meta = torch.empty(batch, seq_len, hc_hidden, device="meta")
            out = meta(x_meta)

        if use_combine:
            assert isinstance(reference, tuple) and isinstance(out, tuple)
            assert len(out) == len(reference) == 3
            for got, want in zip(out, reference):
                assert got.shape == want.shape
            # The hyper-connection streams are returned untouched.
            assert out[1] is x_meta
            assert out[0].shape == (batch, seq_len, config.hidden_size)
        else:
            assert isinstance(out, torch.Tensor)
            assert out.shape == reference.shape
            assert out.shape == (batch, seq_len, config.hidden_size)

    def test_ngram_embedding_matches_official_shape_on_meta(self):
        config = _official_config()
        official = qwen4_exp_modeling.Qwen4ExpTextNGramEmbedding(
            config, embedding_dim=config.ple_embed_dim, layer_idx=0, ple_layer_index=0
        )
        official.eval()

        torch.manual_seed(1)
        batch, seq_len = 2, 16
        tokens = torch.randint(0, config.vocab_size, (batch, seq_len))
        tokens[0, 5] = config.eos_token_id  # exercise the EOS-segment path
        with torch.no_grad():
            reference = official(tokens, None)

        with _class_patches_applied(), torch.no_grad():
            meta = qwen4_exp_modeling.Qwen4ExpTextNGramEmbedding(
                config, embedding_dim=config.ple_embed_dim, layer_idx=0, ple_layer_index=0
            ).to("meta")
            tokens_meta = torch.empty(batch, seq_len, dtype=torch.long, device="meta")
            out = meta(tokens_meta, None)

        assert out.shape == reference.shape
        assert out.dtype == reference.dtype

    @pytest.mark.parametrize("with_conv_mask", [False, True])
    def test_ple_matches_official_shape_on_meta(self, with_conv_mask):
        config = _official_config()
        official = qwen4_exp_modeling.Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=0)
        official.eval()

        torch.manual_seed(2)
        batch, seq_len = 2, 9
        streams = torch.randn(batch, seq_len, config.hc_count * config.hidden_size)
        ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        ids[0, 4] = config.eos_token_id
        conv_mask = None
        if with_conv_mask:
            conv_mask = torch.ones(batch, seq_len, dtype=torch.bool)
            conv_mask[:, 7:] = False

        with torch.no_grad():
            reference = official(streams, ids, None, conv_mask=conv_mask)

        with _class_patches_applied(), torch.no_grad():
            meta = qwen4_exp_modeling.Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=0).to("meta")
            streams_meta = torch.empty(batch, seq_len, config.hc_count * config.hidden_size, device="meta")
            ids_meta = torch.empty(batch, seq_len, dtype=torch.long, device="meta")
            mask_meta = torch.empty(batch, seq_len, dtype=torch.bool, device="meta") if with_conv_mask else None
            out = meta(streams_meta, ids_meta, None, conv_mask=mask_meta)

        assert out.shape == reference.shape
        assert out.dtype == reference.dtype

    def test_gated_residual_validates_feature_dim_on_meta(self):
        config = _official_config()
        with _class_patches_applied():
            meta = qwen4_exp_modeling.Qwen4ExpTextGatedResidual(config, use_combine=True).to("meta")
            bad = torch.empty(2, 5, config.hidden_size, device="meta")
            with pytest.raises(ValueError, match="hyper-connection features"):
                meta(bad)


# ---------------------------------------------------------------------------
# patch_method surface: all five class forwards are installed
# ---------------------------------------------------------------------------


class _StandinIndexer(torch.nn.Module):
    def __init__(self, in_features, q_heads, kv_heads, head_dim):
        super().__init__()
        self.index_qk_proj = torch.nn.Linear(in_features, q_heads * head_dim + kv_heads * head_dim, bias=False)


class _StandinQsaAttention(torch.nn.Module):
    def __init__(self, in_features, q_heads, kv_heads, head_dim):
        super().__init__()
        self.indexer = _StandinIndexer(in_features, q_heads, kv_heads, head_dim)

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class Qwen4ExpTextDecoderLayer(torch.nn.Module):
    """Name-matched stand-in: patch discovery matches on type name."""

    def __init__(self, in_features, q_heads, kv_heads, head_dim):
        super().__init__()
        self.self_attn = _StandinQsaAttention(in_features, q_heads, kv_heads, head_dim)


def _fake_text_config(**overrides):
    config = SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=24,
        head_dim=256,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        partial_rotary_factor=0.25,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_patch_method_installs_all_class_forwards():
    from tensor_cast.transformers.builtin_model import qwen4_exp as adapter

    modeling = qwen4_exp_modeling
    surface = (
        (modeling.Qwen4ExpTextGatedDeltaNet, "forward"),
        (modeling.Qwen4ExpTextModel, "reverse_embedding"),
        (modeling.Qwen4ExpTextGatedResidual, "forward"),
        (modeling.Qwen4ExpTextNGramEmbedding, "forward"),
        (modeling.Qwen4ExpTextPLELayer, "forward"),
    )
    saved = [(cls, attr, getattr(cls, attr)) for cls, attr in surface]

    inner = torch.nn.Module()
    inner.add_module("layer_0", Qwen4ExpTextDecoderLayer(2560, 4, 1, 128))
    model = SimpleNamespace(
        _inner=inner,
        parallel_group_manager=SimpleNamespace(tp_group=SimpleNamespace(world_size=1, rank_in_group=0)),
        text_config=_fake_text_config(),
    )

    try:
        assert patch_method_for_qwen4_exp(model) == 1  # one standin QSA layer
        assert modeling.Qwen4ExpTextGatedDeltaNet.forward is adapter._patched_linear_attn_forward
        assert modeling.Qwen4ExpTextModel.reverse_embedding is adapter._patched_reverse_embedding
        assert modeling.Qwen4ExpTextGatedResidual.forward is _patched_gated_residual_forward
        assert modeling.Qwen4ExpTextNGramEmbedding.forward is _patched_ngram_embedding_forward
        assert modeling.Qwen4ExpTextPLELayer.forward is _patched_ple_forward
    finally:
        for cls, attr, original in saved:
            setattr(cls, attr, original)


def test_qsa_attention_wrapper_unwraps_nested_inner_chain():
    from tensor_cast.layers.qwen4_exp_attention import Qwen4ExpQSAAttentionWrapper

    leaf = object()
    wrapper = object.__new__(Qwen4ExpQSAAttentionWrapper)
    wrapper._inner = SimpleNamespace(_inner=SimpleNamespace(_inner=leaf))
    assert wrapper._unwrap() is leaf

    direct = object.__new__(Qwen4ExpQSAAttentionWrapper)
    direct._inner = leaf
    assert direct._unwrap() is leaf

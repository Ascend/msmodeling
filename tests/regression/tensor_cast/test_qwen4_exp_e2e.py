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
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
"""End-to-end runs of the qwen4_exp simulation paths that op-level tests
cannot reach on their own: the patched GDN forward on the meta device and the
QSA attention wrapper forward (with and without paged caches).

Skipped automatically when transformers < 5.16 (no qwen4_exp module).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers", minversion="5.16")
qwen4_exp_modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

# Imports gated by importorskip above must stay below it (E402 suppressed).
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

from tensor_cast.layers.qwen4_exp_attention import Qwen4ExpQSAAttentionWrapper  # noqa: E402
from tensor_cast.transformers.builtin_model.qwen4_exp import (  # noqa: E402
    _patched_linear_attn_forward,
)

# Small architecture mirroring the real Qwen3.8-Flash-Next config at
# reduced scale (same fields as the op-level test module).
CONFIG_KWARGS = dict(
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rms_norm_eps=1e-6,
    attention_bias=False,
    attention_dropout=0.0,
    output_gate_type="sigmoid",
    rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.5},
    indexer_n_heads=4,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=4,
    vocab_size=1000,
    eos_token_id=99,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=100,
    make_ngram_vocab_size_divisible_by=8,
    seed=7,
    ple_embed_dim=32,
    ple_conv_kernel_size=4,
    hc_count=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=16,
    shared_expert_intermediate_size=16,
    max_position_embeddings=256,
)


def _official_config():
    return Qwen4ExpTextConfig(**CONFIG_KWARGS)


# ---------------------------------------------------------------------------
# Patched GDN forward (linear attention) on the meta device
# ---------------------------------------------------------------------------


class TestPatchedGatedDeltaNetForward:
    def _run_patched(self, hidden_states):
        saved = qwen4_exp_modeling.Qwen4ExpTextGatedDeltaNet.forward
        qwen4_exp_modeling.Qwen4ExpTextGatedDeltaNet.forward = _patched_linear_attn_forward
        try:
            config = _official_config()
            gdn = qwen4_exp_modeling.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).to("meta")
            return gdn(hidden_states)
        finally:
            qwen4_exp_modeling.Qwen4ExpTextGatedDeltaNet.forward = saved

    @pytest.mark.parametrize("batch,seq_len", [(2, 13), (1, 1), (2, 7)])
    def test_prefill_shapes_match_hidden_states(self, batch, seq_len):
        out = self._run_patched(torch.empty(batch, seq_len, 64, device="meta"))
        assert out.shape == (batch, seq_len, 64)


# ---------------------------------------------------------------------------
# QSA attention wrapper forward on the meta device
# ---------------------------------------------------------------------------


class TestQsaAttentionWrapperForward:
    def _make_wrapper(self):
        config = _official_config()
        official = qwen4_exp_modeling.Qwen4ExpTextAttention(config, layer_idx=0)
        wrapper = Qwen4ExpQSAAttentionWrapper(
            official,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            indexer_n_heads=config.indexer_n_heads,
            indexer_kv_heads=config.indexer_kv_heads,
            indexer_head_dim=config.indexer_head_dim,
            token_budget=config.indexer_budget,
            compress_ratio=config.indexer_compress_ratio,
            rotary_dim=int(config.head_dim * config.rope_parameters["partial_rotary_factor"]),
        )
        assert wrapper.index_qk_linear is not None  # fused indexer projection moved local
        # The official projection is rebound (not cleared), so the
        # attention_meta=None fallback keeps a callable official path.
        assert official.indexer.index_qk_proj is wrapper.index_qk_linear
        return wrapper.to("meta")

    @staticmethod
    def _attention_meta(batch, seq_len):
        return SimpleNamespace(
            seq_lens=torch.full((batch,), seq_len),
            query_lens=torch.full((batch,), seq_len),
            block_table_tensor=torch.zeros(batch, seq_len, dtype=torch.long),
            slot_mapping=torch.zeros(batch * seq_len, dtype=torch.long),
            seq_lens_values=None,
            query_lens_values=None,
            is_decode_values=None,
        )

    def test_forward_without_caches(self):
        wrapper = self._make_wrapper()
        batch, seq_len = 2, 13
        rotary_dim = 8
        cos = torch.empty(batch, seq_len, rotary_dim, device="meta")
        sin = torch.empty(batch, seq_len, rotary_dim, device="meta")
        out = wrapper(
            torch.empty(batch, seq_len, 64, device="meta"),
            (cos, sin),
            None,
            None,
            attention_meta=self._attention_meta(batch, seq_len),
        )
        assert isinstance(out, tuple)
        assert out[0].shape == (batch, seq_len, 64)
        assert out[1] is None

    def test_forward_with_packed_2d_input(self):
        """Packed [tokens, hidden] decode input: the head counts must be read
        from the LAST dim after transpose(1, 2), not head_dim. The config
        deliberately has num_attention_heads=4 / num_key_value_heads=2 with
        head_dim=16, so a wrong dim read breaks the RoPE reshape.
        """
        wrapper = self._make_wrapper()
        tokens = 2  # two packed single-token decode requests
        rotary_dim = 8
        cos = torch.empty(tokens, rotary_dim, device="meta")
        sin = torch.empty(tokens, rotary_dim, device="meta")
        out = wrapper(
            torch.empty(tokens, 64, device="meta"),
            (cos, sin),
            None,
            None,
            attention_meta=self._attention_meta(2, 1),
        )
        assert isinstance(out, tuple)
        assert out[0].shape == (tokens, 64)
        assert out[1] is None

    def test_forward_with_paged_caches(self):
        wrapper = self._make_wrapper()
        batch, seq_len = 2, 13
        rotary_dim = 8
        num_pages, page_size = 4, 4
        cos = torch.empty(batch, seq_len, rotary_dim, device="meta")
        sin = torch.empty(batch, seq_len, rotary_dim, device="meta")
        kv_cache = torch.empty(2, num_pages, page_size, 2, 16, device="meta")
        indexer_cache = torch.empty(num_pages, page_size, 1 * 16, device="meta")
        out = wrapper(
            torch.empty(batch, seq_len, 64, device="meta"),
            (cos, sin),
            None,
            None,
            attention_meta=self._attention_meta(batch, seq_len),
            kv_cache_by_layers={0: kv_cache},
            indexer_cache_by_layers={0: indexer_cache},
        )
        assert out[0].shape == (batch, seq_len, 64)

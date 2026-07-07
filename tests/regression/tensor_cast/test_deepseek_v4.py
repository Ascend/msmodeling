"""Regression test for DeepSeek-V4 model support."""

import unittest
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn

import tensor_cast.performance_model.builtin_model  # noqa: F401 — register V4 perf handlers

from tensor_cast.performance_model.builtin_model.deepseek_v4 import _safe_max_int
from tensor_cast.transformers.builtin_model.deepseek_v4 import (
    DeepseekV4Compressor,
    DeepseekV4Config,
    DeepseekV4DecoderLayer,
    DeepseekV4Indexer,
    DeepseekV4MLP,
    DeepseekV4Model,
    DeepseekV4SparseAttention as BuiltinDeepseekV4SparseAttention,
)
from tensor_cast.layers.deepseek_v4 import (
    DeepseekV4SparseAttention,
    DeepseekV4SparseAttentionIndexer,
    _is_decode_attention_batch,
    route_deepseek_v4_gate,
    has_deepseek_v4_hash_routing,
    get_window_topk_idxs,
    get_compress_topk_idxs,
)
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.layers.attention import AttentionMetadataTensorCast
from tensor_cast.layers.mla import (
    DeepseekSparseAttention,
    MultiheadLatentAttentionBase,
    _resolve_sparse_topk_limit,
)
from tensor_cast.model_config import LinearQuantConfig, MlaConfig
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.quantize_utils import LinearQuantType


def _v4_perf_props(op):
    """Resolve the performance-properties functor registered for a V4 op."""
    return OpInvokeInfo._op_properties_functors[op]


def _stub_mla_tensor_cast_init(self, mla_config, mla_module, tp_group, decode_only=False, parallel_group_manager=None):
    """Minimal MLA init for wrapper unit tests (avoids TP/kv-b setup)."""
    MultiheadLatentAttentionBase.__init__(self, mla_config, mla_module, decode_only)


def _tiny_v4_config(**overrides) -> DeepseekV4Config:
    """Small V4 config for fast builtin shell / model construction tests."""
    defaults = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 512,
        "compress_ratios": [0, 4],
        "topk_limit": 4,
        "index_n_heads": 4,
        "index_head_dim": 32,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 2,
        "qk_rope_head_dim": 16,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 48,
        "v_head_dim": 16,
        "o_groups": 2,
        "o_lora_rank": 64,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "head_dim": 32,
    }
    defaults.update(overrides)
    return DeepseekV4Config(**defaults)


class TestDeepseekV4Config(unittest.TestCase):
    """Test DeepseekV4Config initialization and validation."""

    def test_config_with_all_v4_fields(self):
        """Test config with all V4-specific fields."""
        config = DeepseekV4Config(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_hidden_layers=3,
            vocab_size=128256,
            compress_ratios=[0, 4, 128],
            topk_limit=16,
            num_hash_layers=2,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            o_groups=8,
            o_lora_rank=1024,
            score_func="sqrtsoftplus",
            route_scale=1.0,
            swiglu_limit=10.0,
        )
        assert config.model_type == "deepseek_v4"
        assert config.compress_ratios == [0, 4, 128]
        assert config.topk_limit == 16
        assert config.num_hash_layers == 2
        assert config.hc_mult == 4
        assert config.o_groups == 8
        assert config.swiglu_limit == 10.0

    def test_compress_ratios_length_validation(self):
        """Test compress_ratios must have at least num_hidden_layers entries."""
        with self.assertRaises(ValueError, msg="compress_ratios must provide at least"):
            DeepseekV4Config(
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=32,
                num_hidden_layers=5,
                vocab_size=128256,
                compress_ratios=[0, 4],  # Only 2 entries, need 5
            )

    def test_layer_types_validation(self):
        """Test layer_types must match compress_ratios."""
        config = DeepseekV4Config(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_hidden_layers=2,
            vocab_size=128256,
            compress_ratios=[0, 4],
            layer_types=["sliding_attention", "compressed_sparse_attention"],
        )
        assert config.layer_types == ["sliding_attention", "compressed_sparse_attention"]

    def test_index_topk_alias_to_topk_limit(self):
        """Test index_topk is aliased to topk_limit."""
        config = DeepseekV4Config(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_hidden_layers=2,
            vocab_size=128256,
            compress_ratios=[0, 4],
            index_topk=32,
        )
        assert config.topk_limit == 32


class TestDeepseekV4HCOperators(unittest.TestCase):
    """Test Head Compression semantic operators."""

    def test_hc_pre_inv_rms_op(self):
        """Test hc_pre_inv_rms operator returns correct shape."""
        x = torch.randn(2, 8, 4, 4096)  # [B, S, Hc, D]
        result = torch.ops.tensor_cast.hc_pre_inv_rms(x, hc_mult=4)
        # Result shape: [B, S, 1]
        assert result.shape == (2, 8, 1)

    def test_hc_pre_sinkhorn_op(self):
        """Test hc_pre_sinkhorn operator returns correct shapes."""
        x = torch.randn(2, 8, 32)  # [B, S, mix_hc] where mix_hc = (2+4)*4 = 24
        hidden_states = torch.randn(2, 8, 4, 4096)  # [B, S, Hc, D]
        hc_scale = torch.randn(3, 4)  # [3, Hc]
        hc_base = torch.randn(3, 4)  # [3, Hc]

        reduced, post, comb = torch.ops.tensor_cast.hc_pre_sinkhorn(
            x, hidden_states, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, hc_eps=1e-6
        )
        # reduced: [B, S, D]
        assert reduced.shape == (2, 8, 4096)
        # post: [B, S, Hc]
        assert post.shape == (2, 8, 4)
        # comb: [B, S, Hc, Hc]
        assert comb.shape == (2, 8, 4, 4)

    def test_hc_post_op(self):
        """Test hc_post operator combines residual correctly."""
        x = torch.randn(2, 8, 4, 4096)  # [B, S, Hc, D]
        residual = torch.randn(2, 8, 4, 4096)
        post = torch.randn(2, 8, 4)
        comb = torch.randn(2, 8, 4, 4)

        result = torch.ops.tensor_cast.hc_post(x, residual, post, comb, hc_mult=4)
        # Result shape: [B, S, Hc, D]
        assert result.shape == (2, 8, 4, 4, 4096)

    def test_hc_head_op(self):
        """Test hc_head operator reduces HC dimension."""
        x = torch.randn(2, 8, 4, 4096)  # [B, S, Hc, D]
        hc_head_fn = torch.randn(4, 4 * 4096, dtype=torch.float32)
        hc_head_scale = torch.randn(1, dtype=torch.float32)
        hc_head_base = torch.randn(4, dtype=torch.float32)

        result = torch.ops.tensor_cast.hc_head(x, hc_head_fn, hc_head_scale, hc_head_base, hc_mult=4, hc_eps=1e-6)
        # Result shape: [B, S, D]
        assert result.shape == (2, 8, 4096)


class TestDeepseekV4AttentionOperators(unittest.TestCase):
    """Test V4 attention semantic operators."""

    def test_scatter_nd_update_mla_op(self):
        """Test scatter_nd_update_mla operator."""
        kv = torch.randn(2, 8, 512)  # [B, seq, head_dim]
        kv_cache = torch.randn(2, 128, 512)
        slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)

        result = torch.ops.tensor_cast.scatter_nd_update_mla(kv, kv_cache, slot_mapping, seq_lens=None)
        assert result.shape == kv_cache.shape

    def test_compressor_op(self):
        """Test compressor operator."""
        hidden_states = torch.randn(2, 32, 4096)  # [B, S, D]
        kv_cache = torch.randn(2, 128, 512)
        seq_lens = torch.tensor([32, 32], dtype=torch.long)

        compressed_kv, kv_cache_handle = torch.ops.tensor_cast.compressor(
            hidden_states, kv_cache, compress_ratio=4, head_dim=512, rope_head_dim=64, rotate=False, seq_lens=seq_lens
        )
        # compressed_kv: [B, seq/compress_ratio, head_dim]
        assert compressed_kv.shape[0] == 2
        assert compressed_kv.shape[2] == 512
        assert kv_cache_handle.shape == kv_cache.shape

    def test_compressor_rotate_true(self):
        """Test compressor with rotate=True for indexer path."""
        hidden_states = torch.randn(2, 32, 4096)
        kv_cache = torch.randn(2, 128, 128)  # indexer cache uses index_head_dim
        seq_lens = torch.tensor([32, 32], dtype=torch.long)

        compressed_kv, kv_cache_handle = torch.ops.tensor_cast.compressor(
            hidden_states, kv_cache, compress_ratio=4, head_dim=128, rope_head_dim=64, rotate=True, seq_lens=seq_lens
        )
        assert compressed_kv.shape[2] == 128  # index_head_dim

    def test_quant_lightning_indexer_op(self):
        """Test quant_lightning_indexer operator."""
        q_states = torch.randn(2, 8, 8, 128)  # [B, S, H, D]
        weights = torch.randn(2, 8, 8)  # [B, S, H]
        indexer_cache = torch.randn(2, 16, 128)  # [B, cache_len, head_dim]
        seq_lens = torch.tensor([32, 32], dtype=torch.long)

        topk_indices = torch.ops.tensor_cast.quant_lightning_indexer(
            q_states, weights, indexer_cache, topk_limit=8, tp_world_size=1, seq_lens=seq_lens
        )
        # topk_indices: [B, S, topk]
        assert topk_indices.shape[0] == 2
        assert topk_indices.shape[1] == 8
        assert topk_indices.dtype == torch.long

    def test_sparse_attn_sharedkv_op(self):
        """Test sparse_attn_sharedkv operator."""
        q = torch.randn(2, 8, 8, 512)  # [B, S, H, D]
        kv = torch.randn(2, 16, 8, 512)  # [B, total_kv_len, H, D]
        attn_sink = torch.randn(8, dtype=torch.float32)
        topk_indices = torch.randint(0, 16, (2, 8, 4))  # [B, S, topk]

        result = torch.ops.tensor_cast.sparse_attn_sharedkv(
            q,
            kv,
            attn_sink,
            topk_indices,
            softmax_scale=0.01,
            head_dim=512,
            kv_dependency=kv,
        )
        # Result shape: [B, S, H, D]
        assert result.shape == (2, 8, 8, 512)


class TestDeepseekV4MoEOperators(unittest.TestCase):
    """Test V4 MoE routing operators."""

    def test_moe_gating_top_k_op(self):
        """Test moe_gating_top_k operator for non-hash routing."""
        scores = torch.randn(4, 128)  # [batch*seq, num_experts]
        bias = torch.randn(128) if False else None  # Optional bias

        topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k(
            scores, top_k=8, normalize_weights=True, route_scale=1.0, bias=bias
        )
        assert topk_weights.shape == (4, 8)
        assert topk_indices.shape == (4, 8)
        assert topk_indices.dtype == torch.int64

    def test_moe_gating_top_k_with_bias(self):
        """Test moe_gating_top_k with bias."""
        scores = torch.randn(4, 128)
        bias = torch.randn(128)

        topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k(
            scores, top_k=8, normalize_weights=True, route_scale=1.0, bias=bias
        )
        assert topk_weights.shape == (4, 8)
        assert topk_indices.shape == (4, 8)

    def test_moe_gating_top_k_hash_op(self):
        """Test moe_gating_top_k_hash operator for hash routing."""
        scores = torch.randn(4, 128)
        input_ids = torch.tensor([100, 200, 300, 400], dtype=torch.long)
        tid2eid = torch.randint(0, 128, (1000, 8), dtype=torch.int32)  # [vocab_size, top_k]

        topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k_hash(
            scores, top_k=8, normalize_weights=True, route_scale=1.0, input_ids=input_ids, tid2eid=tid2eid
        )
        assert topk_weights.shape == (4, 8)
        assert topk_indices.shape == (4, 8)

    def test_moe_gating_top_k_hash_requires_input_ids(self):
        """Test moe_gating_top_k_hash raises without input_ids."""
        scores = torch.randn(4, 128)
        tid2eid = torch.randint(0, 128, (1000, 8), dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "requires input_ids"):
            torch.ops.tensor_cast.moe_gating_top_k_hash(
                scores, top_k=8, normalize_weights=True, route_scale=1.0, input_ids=None, tid2eid=tid2eid
            )

    def test_v4_clamped_swiglu_op(self):
        """Test v4_clamped_swiglu operator."""
        gate = torch.randn(4, 1024)
        up = torch.randn(4, 1024)

        result = torch.ops.tensor_cast.v4_clamped_swiglu(gate, up, swiglu_limit=10.0)
        assert result.shape == up.shape
        assert result.dtype == up.dtype


class TestDeepseekV4SparseAttentionIndexer(unittest.TestCase):
    """Test DeepseekV4SparseAttentionIndexer wrapper."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 16

        # Create mock inner indexer module
        inner = nn.Module()
        inner.num_heads = 8
        inner.head_dim = 128
        inner.qk_rope_head_dim = 64
        inner.config = type("Config", (), {"topk_limit": 8})()
        inner.hidden_size = 4096
        inner.q_lora_rank = 512

        inner.wq_b = nn.Linear(inner.q_lora_rank, inner.num_heads * inner.head_dim, bias=False)
        inner.weights_proj = nn.Linear(inner.hidden_size, inner.num_heads, bias=False)

        self.inner = inner

    def test_indexer_initialization(self):
        """Test indexer wrapper initialization."""
        indexer = DeepseekV4SparseAttentionIndexer(self.inner, topk_limit=8, compress_ratio=4)
        assert indexer.topk_limit == 8
        assert indexer.compress_ratio == 4

    def test_indexer_head_dim_property(self):
        """Test indexer exposes head_dim correctly."""
        indexer = DeepseekV4SparseAttentionIndexer(self.inner)
        assert indexer.head_dim == 128

    @patch("torch.ops.tensor_cast.quant_lightning_indexer")
    @patch("torch.ops.tensor_cast.compressor")
    @patch("torch.ops.tensor_cast.apply_rope_inplace")
    def test_indexer_forward(self, mock_rope, mock_compressor, mock_indexer):
        """Test indexer forward pass."""
        mock_compressor.return_value = (torch.randn(2, 4, 128), torch.randn(2, 64, 128))
        mock_indexer.return_value = torch.randint(0, 8, (self.batch_size, self.seq_len, 4))

        indexer = DeepseekV4SparseAttentionIndexer(self.inner, topk_limit=8, compress_ratio=4)

        hidden_states = torch.randn(self.batch_size, self.seq_len, 4096)
        qa_normed = torch.randn(self.batch_size, self.seq_len, 512)
        position_embeddings = (torch.randn(self.seq_len, 64), torch.randn(self.seq_len, 64))
        indexer_cache = torch.randn(self.batch_size, 64, 128)

        result = indexer(hidden_states, qa_normed, position_embeddings, indexer_cache)

        assert result.shape[0] == self.batch_size
        assert result.shape[1] == self.seq_len
        mock_indexer.assert_called_once()


class TestDeepseekV4SparseAttention(unittest.TestCase):
    """Test DeepseekV4SparseAttention wrapper."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 16

    def _create_mock_config(self, ratio=0):
        """Create mock V4 config for attention wrapper."""
        config = MagicMock()
        config.hidden_size = 4096
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.head_dim = 512
        config.qk_rope_head_dim = 64
        config.q_lora_rank = 512
        config.qk_nope_head_dim = 448
        config.qk_head_dim = 512
        config.kv_lora_rank = 512
        config.v_head_dim = 128
        config.attention_dropout = 0.0
        config.attention_bias = False
        config.sliding_window = 128
        config.max_position_embeddings = 8192
        config.compress_ratios = [ratio]
        config.hc_mult = 4
        config.o_groups = 8
        config.o_lora_rank = 1024
        config.rms_norm_eps = 1e-6
        config.index_n_heads = 8
        config.index_head_dim = 128
        config.index_topk = 8
        config.topk_limit = 8
        return config

    def _create_mock_inner_module(self, ratio=0, use_indexer=False):
        """Create mock V4 attention inner module."""
        inner = MagicMock()
        inner.hidden_size = 4096
        inner.num_heads = 32
        inner.head_dim = 512
        inner.qk_rope_head_dim = 64
        inner.q_lora_rank = 512
        inner.qk_nope_head_dim = 448
        inner.kv_lora_rank = 512
        inner.v_head_dim = 128
        inner.attention_dropout = 0.0
        inner.config = self._create_mock_config(ratio)
        inner.compress_ratio = ratio
        inner.use_indexer = use_indexer
        inner.use_compressor = ratio > 0
        inner.window_size = 128
        inner.n_groups = 8
        inner.o_lora_rank = 1024
        inner.head_dim = 512
        inner.softmax_scale = 1.0 / (512**0.5)
        inner.scaling = 1.0 / (512**0.5)

        inner.q_a_proj = nn.Linear(4096, 512, bias=False)
        inner.q_a_layernorm = nn.LayerNorm(512)
        inner.q_b_proj = nn.Linear(512, 32 * 512, bias=False)
        inner.kv_a_proj_with_mqa = nn.Linear(4096, 512, bias=False)
        inner.kv_a_layernorm = nn.LayerNorm(512)
        inner.wo_a = nn.Linear(32 * 512 // 8, 8 * 1024, bias=False)  # per-group
        inner.o_proj = nn.Linear(8 * 1024, 4096, bias=False)
        inner.attention_sink = nn.Parameter(torch.zeros(32))

        if use_indexer:
            mock_indexer = MagicMock()
            mock_indexer.num_heads = 8
            mock_indexer.head_dim = 128
            mock_indexer.qk_rope_head_dim = 64
            mock_indexer.config = MagicMock(topk_limit=8)
            mock_indexer.hidden_size = 4096
            mock_indexer.q_lora_rank = 512
            mock_indexer.wq_b = nn.Linear(512, 8 * 128, bias=False)
            mock_indexer.weights_proj = nn.Linear(4096, 8, bias=False)
            inner.indexer = mock_indexer
        else:
            inner.indexer = None

        return inner

    def _mock_tp_group(self, world_size=1):
        tp_group = MagicMock()
        tp_group.world_size = world_size
        return tp_group

    def _create_tiny_forward_inner_module(self):
        """Create a small V4 attention module that can execute forward in CPU UT."""
        inner = MagicMock()
        inner.hidden_size = 8
        inner.num_heads = 2
        inner.head_dim = 4
        inner.qk_rope_head_dim = 2
        inner.q_lora_rank = 4
        inner.qk_nope_head_dim = 2
        inner.kv_lora_rank = 4
        inner.v_head_dim = 2
        inner.config = self._create_mock_config(ratio=0)
        inner.compress_ratio = 0
        inner.use_indexer = False
        inner.use_compressor = False
        inner.window_size = 4
        inner.n_groups = 1
        inner.o_lora_rank = 6
        inner.softmax_scale = 0.5
        inner.scaling = 0.5

        inner.q_a_proj = nn.Linear(8, 4, bias=False)
        inner.q_a_layernorm = nn.LayerNorm(4)
        inner.q_b_proj = nn.Linear(4, 2 * 4, bias=False)
        inner.kv_a_proj_with_mqa = nn.Linear(8, 4, bias=False)
        inner.kv_a_layernorm = nn.LayerNorm(4)
        inner.wo_a = nn.Linear(2 * 4, 6, bias=False)
        inner.o_proj = nn.Linear(6, 8, bias=False)
        inner.attention_sink = nn.Parameter(torch.zeros(2))
        inner.indexer = None
        return inner

    def test_v4_attention_wrapper_initialization(self):
        """Test V4 attention wrapper initialization."""
        inner = self._create_mock_inner_module(ratio=0)
        mla_config = MlaConfig(module_name="DeepseekV4SparseAttention")

        with patch(
            "tensor_cast.layers.mla.MultiheadLatentAttentionTensorCast.__init__",
            _stub_mla_tensor_cast_init,
        ):
            wrapper = DeepseekV4SparseAttention(mla_config, inner, self._mock_tp_group())

        assert wrapper.compress_ratio == 0
        assert wrapper.use_indexer is False
        assert wrapper.use_compressor is False

    def test_v4_attention_wrapper_with_compressor(self):
        """Test V4 attention wrapper with compressor (ratio=4)."""
        inner = self._create_mock_inner_module(ratio=4, use_indexer=True)
        mla_config = MlaConfig(module_name="DeepseekV4SparseAttention")

        with patch(
            "tensor_cast.layers.mla.MultiheadLatentAttentionTensorCast.__init__",
            _stub_mla_tensor_cast_init,
        ):
            wrapper = DeepseekV4SparseAttention(mla_config, inner, self._mock_tp_group())

        assert wrapper.compress_ratio == 4
        assert wrapper.use_indexer is True
        assert wrapper.use_compressor is True

    def test_short_query_length_is_decode_for_mtp_shape(self):
        """MTP decode uses one sampled token plus speculative tokens."""
        attn_meta = AttentionMetadataTensorCast(
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.long),
            seq_lens=torch.tensor([4254, 4254], dtype=torch.long),
            query_lens=torch.tensor([4, 4], dtype=torch.long),
        )

        assert _is_decode_attention_batch(4, attn_meta) is True

    def test_query_length_five_is_prefill(self):
        """The V4 decode heuristic is seq_length < 5, not <= 5."""
        attn_meta = AttentionMetadataTensorCast(
            query_start_loc=torch.tensor([0, 5, 10], dtype=torch.long),
            seq_lens=torch.tensor([5, 5], dtype=torch.long),
            query_lens=torch.tensor([5, 5], dtype=torch.long),
        )

        assert _is_decode_attention_batch(5, attn_meta) is False

    def test_flattened_short_query_length_is_decode(self):
        """Flattened input should use per-request query length."""
        attn_meta = AttentionMetadataTensorCast(
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.long),
            seq_lens=torch.tensor([1000004, 1000004], dtype=torch.long),
            query_lens=torch.tensor([4, 4], dtype=torch.long),
        )

        assert _is_decode_attention_batch(8, attn_meta, batch_size=1) is True

    def test_v4_attention_forward_window_only_prefill_with_cache(self):
        """Exercise the V4 forward cache path without full-size model tensors."""
        inner = self._create_tiny_forward_inner_module()
        mla_config = MlaConfig(module_name="DeepseekV4SparseAttention")

        with patch(
            "tensor_cast.layers.mla.MultiheadLatentAttentionTensorCast.__init__",
            _stub_mla_tensor_cast_init,
        ):
            wrapper = DeepseekV4SparseAttention(mla_config, inner, self._mock_tp_group())
        wrapper.layer_idx = 0

        hidden_states = torch.randn(1, 5, 8)
        cos = torch.ones(1, 5, 2)
        sin = torch.zeros(1, 5, 2)
        attention_meta = AttentionMetadataTensorCast(
            query_start_loc=torch.tensor([0, 5], dtype=torch.long),
            seq_lens=torch.tensor([5], dtype=torch.long),
            query_lens=torch.tensor([5], dtype=torch.long),
            slot_mapping=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        )
        kv_cache = torch.randn(1, 4, 4)

        def _dynamic_quantize_symmetric(x, *_args, **_kwargs):
            return x, torch.ones(1, dtype=torch.float32, device=x.device)

        def _sparse_attn_sharedkv(q, _kv, *_args, kv_dependency=None, **_kwargs):
            assert kv_dependency is kv_cache
            return q

        with (
            patch("torch.ops.tensor_cast.rms_norm", side_effect=lambda x, *_args, **_kwargs: x),
            patch("torch.ops.tensor_cast.apply_rope_inplace", side_effect=lambda *_args, **_kwargs: None),
            patch("torch.ops.tensor_cast.dynamic_quantize_symmetric", side_effect=_dynamic_quantize_symmetric),
            patch("torch.ops.tensor_cast.scatter_nd_update_mla", side_effect=lambda _kv, cache, *_args: cache),
            patch("torch.ops.tensor_cast.sparse_attn_sharedkv", side_effect=_sparse_attn_sharedkv),
        ):
            result, cache = wrapper(
                hidden_states,
                (cos, sin),
                attention_mask=None,
                attention_meta=attention_meta,
                kv_cache_by_layers={0: kv_cache},
            )

        assert result.shape == hidden_states.shape
        assert cache is None

    def test_v4_ratio128_topk_uses_total_seq_len(self):
        """Deterministic compressed topk should include previous context chunks."""
        inner = self._create_tiny_forward_inner_module()
        inner.compress_ratio = 128
        inner.use_compressor = True
        inner.window_size = 4
        mla_config = MlaConfig(module_name="DeepseekV4SparseAttention")

        with patch(
            "tensor_cast.layers.mla.MultiheadLatentAttentionTensorCast.__init__",
            _stub_mla_tensor_cast_init,
        ):
            wrapper = DeepseekV4SparseAttention(mla_config, inner, self._mock_tp_group())
        wrapper.layer_idx = 0

        hidden_states = torch.randn(1, 8, 8)
        cos = torch.ones(1, 8, 2)
        sin = torch.zeros(1, 8, 2)
        attention_meta = AttentionMetadataTensorCast(
            query_start_loc=torch.tensor([0, 8], dtype=torch.long),
            seq_lens=torch.tensor([1024], dtype=torch.long),
            query_lens=torch.tensor([8], dtype=torch.long),
            slot_mapping=torch.arange(8, dtype=torch.long),
            max_total_seq_len=1024,
        )
        kv_cache = torch.randn(8, 128, 4)
        expected_topk_width = 4 + 1024 // 128

        def _dynamic_quantize_symmetric(x, *_args, **_kwargs):
            return x, torch.ones(1, dtype=torch.float32, device=x.device)

        def _compressor(_hidden_states, cache, *_args, **_kwargs):
            return torch.randn(1, 8, 4), cache

        def _sparse_attn_sharedkv(q, _kv, _sink, topk_indices, *_args, kv_dependency=None, **_kwargs):
            assert topk_indices.shape == (1, 8, expected_topk_width)
            assert kv_dependency is kv_cache
            return q

        with (
            patch("torch.ops.tensor_cast.rms_norm", side_effect=lambda x, *_args, **_kwargs: x),
            patch("torch.ops.tensor_cast.apply_rope_inplace", side_effect=lambda *_args, **_kwargs: None),
            patch("torch.ops.tensor_cast.dynamic_quantize_symmetric", side_effect=_dynamic_quantize_symmetric),
            patch("torch.ops.tensor_cast.scatter_nd_update_mla", side_effect=lambda _kv, cache, *_args: cache),
            patch("torch.ops.tensor_cast.compressor", side_effect=_compressor),
            patch("torch.ops.tensor_cast.sparse_attn_sharedkv", side_effect=_sparse_attn_sharedkv),
        ):
            result, cache = wrapper(
                hidden_states,
                (cos, sin),
                attention_mask=None,
                attention_meta=attention_meta,
                kv_cache_by_layers={0: kv_cache},
            )

        assert result.shape == hidden_states.shape
        assert cache is None

    def test_w4a8_wo_a_weight_is_unpacked_before_grouped_einsum(self):
        inner = self._create_tiny_forward_inner_module()
        inner.wo_a = TensorCastQuantLinear(
            inner.wo_a,
            LinearQuantConfig(
                quant_type=LinearQuantType.W4A8,
                weight_scale=torch.tensor(1.0),
            ),
        )
        inner.wo_a.out_features = 2
        inner.wo_a.in_features = 4

        weight = DeepseekV4SparseAttention._extract_logical_linear_weight(inner.wo_a)

        assert weight.shape == (inner.wo_a.qweight.shape[0], inner.wo_a.qweight.shape[1] * 2)
        assert inner.wo_a.qweight.shape[-1] * 2 == weight.shape[-1]
        weight.reshape(inner.n_groups, inner.o_lora_rank, inner.num_heads * inner.head_dim // inner.n_groups)


class TestDeepseekV4Helpers(unittest.TestCase):
    """Test V4 helper functions."""

    def test_has_deepseek_v4_hash_routing_from_gate_attr(self):
        """Test hash routing detection from gate attribute."""
        gate = nn.Linear(4096, 128)
        gate.hash = True

        assert has_deepseek_v4_hash_routing(gate, moe_layer_idx=None) is True

    def test_has_deepseek_v4_hash_routing_from_gate_attr_false(self):
        """Test hash routing detection returns False."""
        gate = nn.Linear(4096, 128)
        gate.hash = False

        assert has_deepseek_v4_hash_routing(gate, moe_layer_idx=None) is False

    def test_has_deepseek_v4_hash_routing_from_moe_layer_idx(self):
        """Test hash routing detection from moe_layer_idx."""
        gate = nn.Linear(4096, 128)
        # No hash attribute, use moe_layer_idx
        assert has_deepseek_v4_hash_routing(gate, moe_layer_idx=2) is True  # idx < 3
        assert has_deepseek_v4_hash_routing(gate, moe_layer_idx=3) is False  # idx >= 3
        assert has_deepseek_v4_hash_routing(gate, moe_layer_idx=0) is True  # idx < 3

    def test_get_window_topk_idxs_prefill(self):
        """Test window topk indices for prefill."""
        indices = get_window_topk_idxs(window_size=128, batch_size=2, seq_length=32, device="cpu", is_decode=False)
        # width = min(32, 128) = 32
        assert indices.shape == (2, 32, 32)

    def test_get_window_topk_idxs_decode(self):
        """Test window topk indices for decode."""
        indices = get_window_topk_idxs(window_size=128, batch_size=2, seq_length=1, device="cpu", is_decode=True)
        # width = 128 (is_decode overrides)
        assert indices.shape == (2, 1, 128)

    def test_get_compress_topk_idxs(self):
        """Test compress topk indices."""
        indices = get_compress_topk_idxs(ratio=4, batch_size=2, seq_length=32, device="cpu")
        # width = max(32 // 4, 1) = 8
        assert indices.shape == (2, 32, 8)

    def test_get_compress_topk_idxs_uses_total_seq_length(self):
        """Context-prefill compressed topk width follows total seq length."""
        indices = get_compress_topk_idxs(
            ratio=128,
            batch_size=2,
            seq_length=32,
            total_seq_length=1024,
            device="cpu",
        )
        assert indices.shape == (2, 32, 8)


class TestResolveSparseTopkLimit(unittest.TestCase):
    """Test _resolve_sparse_topk_limit helper."""

    def test_resolve_from_explicit(self):
        """Test resolving from explicit topk_limit parameter."""
        inner = MagicMock()
        result = _resolve_sparse_topk_limit(inner, topk_limit=16)
        assert result == 16

    def test_resolve_from_inner_module(self):
        """Test resolving from inner module attribute."""
        inner = MagicMock()
        inner.topk_limit = 24
        result = _resolve_sparse_topk_limit(inner)
        assert result == 24

    def test_resolve_from_config(self):
        """Test resolving from config attribute."""
        inner = MagicMock()
        inner.topk_limit = None
        inner.config = MagicMock(topk_limit=32)
        result = _resolve_sparse_topk_limit(inner)
        assert result == 32

    def test_resolve_from_index_topk(self):
        """Test resolving from index_topk (GLM5 style)."""
        inner = MagicMock()
        inner.topk_limit = None
        inner.config = MagicMock(topk_limit=None, index_topk=48)
        result = _resolve_sparse_topk_limit(inner)
        assert result == 48


class TestDeepseekV4PerformanceModel(unittest.TestCase):
    """Test V4 performance model operators."""

    def test_hc_pre_inv_rms_performance(self):
        """Test hc_pre_inv_rms performance properties."""
        hc_pre_inv_rms_props = _v4_perf_props(torch.ops.tensor_cast.hc_pre_inv_rms.default)

        mock_invoke = MagicMock(spec=OpInvokeInfo)
        # x shape [B, S, hc_mult * nope_head_dim] (already flattened 2D)
        x = torch.randn(2, 8, 4 * 64)
        mock_invoke.args = [x, 4]
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = hc_pre_inv_rms_props(mock_invoke)
        assert props is not None

    def test_hc_pre_sinkhorn_performance(self):
        """Test hc_pre_sinkhorn performance properties."""
        hc_sinkhorn_props = _v4_perf_props(torch.ops.tensor_cast.hc_pre_sinkhorn.default)

        mock_invoke = MagicMock(spec=OpInvokeInfo)
        # x shape [B, S, hc] where hc = hc_mult * nope_head_dim
        x = torch.randn(2, 8, 32)
        hidden_states = torch.randn(2, 8, 4, 4096)
        hc_scale = torch.randn(3, 4)
        hc_base = torch.randn(3, 4)
        # args: (x, hidden_states, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)
        mock_invoke.args = [x, hidden_states, hc_scale, hc_base, 4, 20, 1e-6]
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = hc_sinkhorn_props(mock_invoke)
        assert props is not None

    def test_hc_post_performance(self):
        """Test hc_post performance properties."""
        hc_post_props = _v4_perf_props(torch.ops.tensor_cast.hc_post.default)

        mock_invoke = MagicMock()
        # x shape [B, S, hc, D] - flattened to 2D [B*S, hc*D]
        x = torch.randn(4, 4 * 4096)
        mock_invoke.args = [x, x, x, x, 4]
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = hc_post_props(mock_invoke)
        assert props is not None

    def test_hc_head_performance(self):
        """Test hc_head performance properties."""
        hc_head_props = _v4_perf_props(torch.ops.tensor_cast.hc_head.default)

        mock_invoke = MagicMock()
        x = torch.randn(2, 8, 4, 4096)
        hc_head_fn = torch.randn(4, 4 * 4096, dtype=torch.float32)
        hc_head_scale = torch.randn(1, dtype=torch.float32)
        hc_head_base = torch.randn(4, dtype=torch.float32)
        mock_invoke.args = [x, hc_head_fn, hc_head_scale, hc_head_base, 4, 1e-6]
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = hc_head_props(mock_invoke)
        assert props is not None

    def test_compressor_performance(self):
        """Test compressor performance properties."""
        compressor_props = _v4_perf_props(torch.ops.tensor_cast.compressor.default)

        mock_invoke = MagicMock()
        hidden_states = torch.randn(2, 32, 4096)
        kv_cache = torch.randn(2, 128, 512)
        seq_lens = torch.tensor([32, 32], dtype=torch.long)
        # args: (hidden_states, kv_cache, compress_ratio, head_dim, nope_head_dim, rotate, seq_lens)
        mock_invoke.args = [hidden_states, kv_cache, 4, 512, 64, False, seq_lens]
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = compressor_props(mock_invoke)
        assert props is not None

    def test_quant_lightning_indexer_performance(self):
        """Test quant_lightning_indexer performance properties."""
        indexer_props = _v4_perf_props(torch.ops.tensor_cast.quant_lightning_indexer.default)

        mock_invoke = MagicMock()
        q_states = torch.randn(2, 8, 8, 128)
        indexer_cache = torch.randn(2, 16, 128)
        seq_lens = torch.tensor([32, 32], dtype=torch.long)
        mock_invoke.args = [q_states, torch.randn(2, 8, 8), indexer_cache, 8, 1, seq_lens]
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = indexer_props(mock_invoke)
        assert props is not None

    def test_quant_lightning_indexer_performance_uses_safe_max_int_fallback(self):
        """Cover _safe_max_int when seq_lens is present but query_lens metadata is absent."""
        indexer_props = _v4_perf_props(torch.ops.tensor_cast.quant_lightning_indexer.default)

        mock_invoke = MagicMock()
        q_states = torch.randn(2, 8, 4, 32)
        indexer_cache = torch.randn(2, 16, 32)
        seq_lens = torch.tensor([16, 20], dtype=torch.long)
        mock_invoke.args = [q_states, torch.randn(2, 8, 4), indexer_cache, 4, 1, seq_lens]
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = indexer_props(mock_invoke)
        assert props is not None
        assert _safe_max_int(seq_lens) == 20

    def test_safe_max_int_handles_meta_and_none(self):
        assert _safe_max_int(None) is None
        assert _safe_max_int(torch.empty(2, dtype=torch.long, device="meta")) is None
        assert _safe_max_int(torch.tensor([3, 7], dtype=torch.long)) == 7

    def test_sparse_attn_sharedkv_performance(self):
        """Test sparse_attn_sharedkv performance properties."""
        attn_props = _v4_perf_props(torch.ops.tensor_cast.sparse_attn_sharedkv.default)

        mock_invoke = MagicMock()
        q = torch.randn(2, 8, 8, 512)
        kv = torch.randn(2, 16, 8, 512)
        attn_sink = torch.randn(8, dtype=torch.float32)
        topk_indices = torch.randint(0, 16, (2, 8, 4))
        mock_invoke.args = [q, kv, attn_sink]
        mock_invoke.kwargs = {"topk_indices": topk_indices, "softmax_scale": 0.01, "head_dim": 512}
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = attn_props(mock_invoke)
        assert props is not None
        mock_invoke.get_memory_access_properties.assert_called_once_with(exclude_input_ids={"kv", "topk_indices"})

    def test_sparse_attn_sharedkv_performance_reports_missing_required_arg(self):
        """Missing kwargs should report the schema name instead of leaking KeyError."""
        attn_props = _v4_perf_props(torch.ops.tensor_cast.sparse_attn_sharedkv.default)

        mock_invoke = MagicMock()
        q = torch.randn(2, 8, 8, 512)
        kv = torch.randn(2, 16, 8, 512)
        attn_sink = torch.randn(8, dtype=torch.float32)
        topk_indices = torch.randint(0, 16, (2, 8, 4))
        mock_invoke.args = [q, kv, attn_sink]
        mock_invoke.kwargs = {"topk_indices": topk_indices, "softmax_scale": 0.01}

        with self.assertRaisesRegex(ValueError, "head_dim"):
            attn_props(mock_invoke)

    def test_moe_gating_top_k_performance(self):
        """Test moe_gating_top_k performance properties."""
        gating_props = _v4_perf_props(torch.ops.tensor_cast.moe_gating_top_k.default)

        mock_invoke = MagicMock()
        scores = torch.randn(4, 128)
        mock_invoke.args = [scores, 8, True, 1.0, None, 0]
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = gating_props(mock_invoke)
        assert props is not None

    def test_moe_gating_top_k_hash_performance(self):
        """Test moe_gating_top_k_hash performance properties."""
        hash_gating_props = _v4_perf_props(torch.ops.tensor_cast.moe_gating_top_k_hash.default)

        mock_invoke = MagicMock()
        scores = torch.randn(4, 128)
        mock_invoke.args = [scores, 8, True, 1.0, None, torch.randint(0, 128, (1000, 8), dtype=torch.int32)]
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = hash_gating_props(mock_invoke)
        assert props is not None

    def test_v4_clamped_swiglu_performance(self):
        """Test v4_clamped_swiglu performance properties."""
        swiglu_props = _v4_perf_props(torch.ops.tensor_cast.v4_clamped_swiglu.default)

        mock_invoke = MagicMock()
        gate = torch.randn(4, 1024)
        up = torch.randn(4, 1024)
        # Handler unpacks (gate, up, clamp_limit).
        mock_invoke.args = [gate, up, 10.0]
        mock_invoke.get_memory_access_properties.return_value = MagicMock(
            memory_read_bytes=0, memory_write_bytes=0, memory_readwrite_bytes=0
        )

        props = swiglu_props(mock_invoke)
        assert props is not None


class TestDeepseekV4DecoderLayer(unittest.TestCase):
    """Test DeepseekV4DecoderLayer."""

    def test_decoder_layer_initialization(self):
        """Test decoder layer initialization with all components."""
        config = DeepseekV4Config(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_hidden_layers=2,
            vocab_size=128256,
            compress_ratios=[0, 4],
            topk_limit=8,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            first_k_dense_replace=0,
        )

        layer = DeepseekV4DecoderLayer(config, layer_idx=0)
        assert layer.hidden_size == 4096
        assert layer.layer_idx == 0
        assert layer.hc_mult == 4
        assert hasattr(layer, "self_attn")
        assert hasattr(layer, "mlp")

    def test_decoder_layer_dense_mlp_layer(self):
        """Test decoder layer with dense MLP (first layers)."""
        config = DeepseekV4Config(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_hidden_layers=5,
            vocab_size=128256,
            compress_ratios=[0, 0, 0, 4, 128],
            topk_limit=8,
            index_n_heads=8,
            index_head_dim=128,
            hc_mult=4,
            first_k_dense_replace=3,
        )

        layer0 = DeepseekV4DecoderLayer(config, layer_idx=0)
        assert hasattr(layer0, "mlp")

        layer3 = DeepseekV4DecoderLayer(config, layer_idx=3)
        assert hasattr(layer3, "mlp")

    def test_decoder_layer_ratio4_builds_indexer_and_compressor(self):
        config = _tiny_v4_config(num_hidden_layers=2, compress_ratios=[0, 4])
        layer = DeepseekV4DecoderLayer(config, layer_idx=1)

        assert isinstance(layer.self_attn, BuiltinDeepseekV4SparseAttention)
        assert layer.self_attn.use_indexer is True
        assert layer.self_attn.use_compressor is True
        assert isinstance(layer.self_attn.indexer, DeepseekV4Indexer)
        assert isinstance(layer.self_attn.compressor, DeepseekV4Compressor)

    def test_dense_mlp_forward_with_swiglu_limit(self):
        config = _tiny_v4_config(
            num_hidden_layers=1,
            compress_ratios=[0],
            first_k_dense_replace=1,
            swiglu_limit=5.0,
        )
        mlp = DeepseekV4MLP(config)
        x = torch.randn(2, 4, config.hidden_size)

        with patch("torch.ops.tensor_cast.v4_clamped_swiglu") as mock_swiglu:
            mock_swiglu.return_value = torch.randn(2, 4, config.intermediate_size)
            out = mlp(x)

        mock_swiglu.assert_called_once()
        assert out.shape == (2, 4, config.hidden_size)

    def test_moe_decoder_applies_swiglu_limit_patch(self):
        config = _tiny_v4_config(
            num_hidden_layers=1,
            compress_ratios=[0],
            first_k_dense_replace=0,
            swiglu_limit=8.0,
        )
        layer = DeepseekV4DecoderLayer(config, layer_idx=0)
        shared_experts = getattr(layer.mlp, "shared_experts", None)
        assert shared_experts is not None
        assert getattr(shared_experts, "_v4_swiglu_patched", False) is True
        assert getattr(shared_experts, "swiglu_limit", 0.0) == 8.0


class TestDeepseekV4Model(unittest.TestCase):
    """Test DeepseekV4Model construction."""

    def test_tiny_model_initialization(self):
        """Fast full-model shell test (covers embed/layers/norm without nightly budget)."""
        config = _tiny_v4_config(num_hidden_layers=2, compress_ratios=[0, 4])
        model = DeepseekV4Model(config)

        assert len(model.layers) == 2
        assert isinstance(model.layers[1].self_attn.indexer, DeepseekV4Indexer)
        assert hasattr(model, "embed_tokens")
        assert hasattr(model, "rotary_emb")


@pytest.mark.nightly
class TestDeepseekV4ModelNightly(unittest.TestCase):
    """Full-size V4 model init (large hidden_size; too slow for default CI)."""

    def test_model_initialization(self):
        """Test model can be initialized with V4 config."""
        config = DeepseekV4Config(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_hidden_layers=2,
            vocab_size=128256,
            compress_ratios=[0, 4],
            topk_limit=8,
            index_n_heads=8,
            index_head_dim=128,
            hc_mult=4,
        )

        model = DeepseekV4Model(config)
        assert hasattr(model, "embed_tokens")
        assert hasattr(model, "layers")
        assert hasattr(model, "norm")
        assert hasattr(model, "rotary_emb")
        assert len(model.layers) == 2


class TestDeepseekV4RouteFunctions(unittest.TestCase):
    """Test V4 routing functions."""

    def test_route_deepseek_v4_gate_non_hash(self):
        """Test routing with non-hash MoE."""
        gate = nn.Linear(4096, 128)
        gate.score_func = "sqrtsoftplus"
        gate.route_scale = 1.0
        gate.weight = nn.Parameter(torch.randn(128, 4096))
        gate.bias = None

        hidden_states = torch.randn(2, 16, 4096)

        topk_indices, topk_weights = route_deepseek_v4_gate(gate, hidden_states, top_k=8, moe_layer_idx=5)

        # [batch_size, seq_len, top_k]
        assert topk_indices.shape == (2, 16, 8)
        assert topk_weights.shape == (2, 16, 8)

    def test_route_deepseek_v4_gate_hash(self):
        """Test routing with hash MoE."""
        gate = nn.Linear(4096, 128)
        gate.score_func = "sqrtsoftplus"
        gate.route_scale = 1.0
        gate.weight = nn.Parameter(torch.randn(128, 4096))
        gate.hash = True
        gate.tid2eid = torch.randint(0, 128, (128000, 8), dtype=torch.int32)

        hidden_states = torch.randn(2, 16, 4096)
        input_ids = torch.randint(0, 128000, (2, 16), dtype=torch.long)

        topk_indices, topk_weights = route_deepseek_v4_gate(
            gate, hidden_states, top_k=8, input_ids=input_ids, moe_layer_idx=1
        )

        # [batch_size, seq_len, top_k]
        assert topk_indices.shape == (2, 16, 8)
        assert topk_weights.shape == (2, 16, 8)

    def test_route_deepseek_v4_gate_non_hash_with_tp_slice(self):
        """Test non-hash routing slices post-score logits for shared expert TP."""
        gate = nn.Linear(4096, 128)
        gate.score_func = "sqrtsoftplus"
        gate.route_scale = 1.0
        gate.weight = nn.Parameter(torch.randn(128, 4096))
        gate.bias = None

        hidden_states = torch.randn(16, 4096)

        topk_indices, topk_weights = route_deepseek_v4_gate(
            gate,
            hidden_states,
            top_k=8,
            moe_layer_idx=5,
            tp_size=4,
            tp_rank=1,
        )

        assert topk_indices.shape == (4, 8)
        assert topk_weights.shape == (4, 8)

    def test_route_deepseek_v4_gate_hash_with_tp_slice(self):
        """Test hash routing slices scores and input ids for shared expert TP."""
        gate = nn.Linear(4096, 128)
        gate.score_func = "sqrtsoftplus"
        gate.route_scale = 1.0
        gate.weight = nn.Parameter(torch.randn(128, 4096))
        gate.hash = True
        gate.tid2eid = torch.randint(0, 128, (128000, 8), dtype=torch.int32)

        hidden_states = torch.randn(16, 4096)
        input_ids = torch.randint(0, 128000, (1, 16), dtype=torch.long)

        topk_indices, topk_weights = route_deepseek_v4_gate(
            gate,
            hidden_states,
            top_k=8,
            input_ids=input_ids,
            moe_layer_idx=1,
            tp_size=4,
            tp_rank=1,
        )

        assert topk_indices.shape == (4, 8)
        assert topk_weights.shape == (4, 8)

    def test_compute_v4_gate_scores_softmax(self):
        """Test gate scores computation with softmax."""
        from tensor_cast.layers.deepseek_v4 import compute_v4_gate_scores

        gate = nn.Linear(4096, 128)
        gate.weight = nn.Parameter(torch.randn(128, 4096))
        gate.score_func = "softmax"
        gate.route_scale = 1.0

        hidden_states = torch.randn(2, 16, 4096)
        scores, route_scale, normalize_weights = compute_v4_gate_scores(gate, hidden_states)

        # [batch_size, seq_len, num_experts]
        assert scores.shape == (2, 16, 128)
        assert route_scale == 1.0
        assert normalize_weights is False

    def test_compute_v4_gate_scores_sigmoid(self):
        """Test gate scores computation with sigmoid."""
        from tensor_cast.layers.deepseek_v4 import compute_v4_gate_scores

        gate = nn.Linear(4096, 128)
        gate.weight = nn.Parameter(torch.randn(128, 4096))
        gate.score_func = "sigmoid"
        gate.route_scale = 2.0

        hidden_states = torch.randn(2, 16, 4096)
        scores, route_scale, normalize_weights = compute_v4_gate_scores(gate, hidden_states)

        # [batch_size, seq_len, num_experts]
        assert scores.shape == (2, 16, 128)
        assert normalize_weights is True


class TestDeepseekV4ConfigEdgeCases(unittest.TestCase):
    """Test V4 config edge cases."""

    def test_config_with_aliases(self):
        """Test config field name aliases."""
        config = DeepseekV4Config(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_hidden_layers=2,
            vocab_size=128256,
            compress_ratios=[0, 4],
            n_routed_experts=64,
            n_shared_experts=2,
            n_activated_experts=8,
            dim=4096,
            n_layers=2,
        )

        assert config.n_routed_experts == 64
        assert config.num_hidden_layers == 2

    def test_config_with_rope_scaling(self):
        """Test config with rope scaling parameters."""
        config = DeepseekV4Config(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_hidden_layers=2,
            vocab_size=128256,
            compress_ratios=[0, 4],
            rope_scaling={"type": "yarn", "factor": 1.0},
            rope_parameters={"type": "yarn", "factor": 1.0, "beta_fast": 32, "beta_slow": 1},
        )

        rope_params = getattr(config, "rope_parameters", None)
        assert rope_params is not None


class TestDeepseekV4MlaHooks(unittest.TestCase):
    def test_requires_indexer_cache_inherited(self):
        assert DeepseekV4SparseAttention.requires_indexer_cache()
        assert DeepseekSparseAttention.requires_indexer_cache()

    def test_kv_b_hooks_are_no_ops(self):
        wrapper = DeepseekV4SparseAttention.__new__(DeepseekV4SparseAttention)
        wrapper._setup_kv_b_decomposition(MagicMock())
        wrapper._quantize_kv_b_decomposition()

    def test_build_tp_plan_extras_registers_indexer(self):
        params = {"tp_group": MagicMock()}
        config_info = MagicMock(index_n_heads=8)
        plan = DeepseekV4SparseAttention.build_tp_plan_extras("model.layers", params, config_info)
        assert "model.layers.*.self_attn.indexer.wq_b" in plan
        assert "model.layers.*.self_attn.indexer.weights_proj" in plan

    def test_build_o_proj_tp_plan_extras_registers_v4_projections(self):
        params = {"tp_group": MagicMock(), "global_tp_group": MagicMock()}
        plan = DeepseekV4SparseAttention.build_o_proj_tp_plan_extras("model.layers", params, MagicMock())
        assert "model.layers.*.self_attn.wo_a" in plan
        assert "model.layers.*.self_attn.o_proj" not in plan

    def test_v4_attention_wrapper_skips_legacy_kv_b_setup(self):
        inner = TestDeepseekV4SparseAttention()._create_mock_inner_module(ratio=4, use_indexer=True)
        mla_config = MlaConfig(module_name="DeepseekV4SparseAttention")

        with patch(
            "tensor_cast.layers.mla.MultiheadLatentAttentionTensorCast.__init__",
            _stub_mla_tensor_cast_init,
        ):
            wrapper = DeepseekV4SparseAttention(mla_config, inner, TestDeepseekV4SparseAttention()._mock_tp_group())

        assert not hasattr(wrapper, "W_UV")


class TestDeepseekV4BuiltinModelRegistration(unittest.TestCase):
    """Test V4 builtin model registration."""

    def test_auto_model_register(self):
        """Test AutoModel registration for V4."""
        from transformers import AutoModel

        from tensor_cast.transformers.custom_model_registry import get_model_profile

        mapping = getattr(AutoModel, "_model_mapping", None)
        assert mapping is not None

        profile = get_model_profile("deepseek_v4")
        mla_config = profile.build_mla_config()
        assert mla_config is not None
        assert mla_config.field_names.kv_b_proj is None

    def test_config_normalize_methods(self):
        """Test config normalization methods."""
        DeepseekV4Config(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_hidden_layers=3,
            vocab_size=128256,
            compress_ratios=[0, 4, 128],
        )

        normalized = DeepseekV4Config._normalize_rope_params({"type": "yarn", "factor": 1.0})
        assert normalized["type"] == "yarn"
        assert normalized["rope_type"] == "yarn"

    def test_normalize_layer_policy(self):
        """Test layer policy normalization."""
        ratios, types = DeepseekV4Config._normalize_layer_policy(
            compress_ratios=[0, 4, 128],
            layer_types=None,
            num_hidden_layers=3,
            config_path=None,
        )

        assert ratios == [0, 4, 128]
        assert types == ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]


if __name__ == "__main__":
    unittest.main()

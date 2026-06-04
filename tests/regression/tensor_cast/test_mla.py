import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from tensor_cast.layers.mla import (
    DeepseekSparseAttention,
    DeepseekSparseAttentionIndexer,
    MultiheadLatentAttentionTensorCast,
)
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import LinearQuantConfig
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity, QuantScheme
from tensor_cast.transformers.builtin_model.deepseek_v32 import DeepseekV32Config


class TestMlaIndexerCacheHooks(unittest.TestCase):
    def test_base_requires_indexer_cache_is_false(self):
        self.assertFalse(MultiheadLatentAttentionTensorCast.requires_indexer_cache())

    def test_deepseek_sparse_requires_indexer_cache(self):
        self.assertTrue(DeepseekSparseAttention.requires_indexer_cache())

    def test_base_build_tp_plan_extras_empty(self):
        self.assertEqual(
            MultiheadLatentAttentionTensorCast.build_tp_plan_extras("layers", {}, SimpleNamespace()),
            {},
        )
        self.assertEqual(
            MultiheadLatentAttentionTensorCast.build_o_proj_tp_plan_extras("layers", {}, SimpleNamespace()),
            {},
        )

    def test_setup_kv_b_decomposition_splits_projection(self):
        num_heads = 4
        kv_lora_rank = 64
        qk_nope = 32
        v_head = 16
        kv_b_proj = nn.Linear(kv_lora_rank, num_heads * (qk_nope + v_head), bias=False)
        wrapper = SimpleNamespace(
            kv_b_proj=kv_b_proj,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope,
            v_head_dim=v_head,
            _num_heads_per_rank=num_heads,
        )

        tp_group = MagicMock(world_size=1, rank_in_group=0)
        MultiheadLatentAttentionTensorCast._setup_kv_b_decomposition(wrapper, tp_group)

        self.assertEqual(wrapper.W_UV.shape[0], num_heads)
        self.assertEqual(wrapper.W_UK_T.shape[0], num_heads)

    @patch("torch.ops.tensor_cast.quantize", side_effect=lambda t, *args, **kwargs: t)
    def test_quantize_kv_b_decomposition(self, _mock_quantize):
        num_heads = 2
        kv_lora_rank = 32
        qk_nope = 16
        v_head = 8
        linear_quant_config = LinearQuantConfig(
            weight_scale=torch.ones(1),
            quant_type=LinearQuantType.W8A16,
            weight_quant_granularity=QuantGranularity.PER_TENSOR,
            weight_quant_scheme=QuantScheme.SYMMETRIC,
        )
        linear = nn.Linear(kv_lora_rank, num_heads * (qk_nope + v_head), bias=False)
        setup_wrapper = SimpleNamespace(
            kv_b_proj=linear,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope,
            v_head_dim=v_head,
            _num_heads_per_rank=num_heads,
        )
        tp_group = MagicMock(world_size=1, rank_in_group=0)
        MultiheadLatentAttentionTensorCast._setup_kv_b_decomposition(setup_wrapper, tp_group)

        quant_kv_b = TensorCastQuantLinear(linear, linear_quant_config)
        wrapper = SimpleNamespace(
            kv_b_proj=quant_kv_b,
            kv_b_proj_weight_t=setup_wrapper.kv_b_proj_weight_t,
            W_UK_T=setup_wrapper.W_UK_T,
            W_UV=setup_wrapper.W_UV,
            quant_config=MagicMock(get_quant_dtype=MagicMock(return_value=torch.int8)),
        )
        MultiheadLatentAttentionTensorCast._quantize_kv_b_decomposition(wrapper)

        self.assertIs(wrapper.kv_b_proj_scale, quant_kv_b.weight_scale)


class TestDeepseekSparseAttentionIndexer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10

        inner_module = nn.Module()
        inner_module.hidden_size = 16
        inner_module.num_heads = 4
        inner_module.head_dim = 8
        inner_module.qk_rope_head_dim = 4
        inner_module.topk_limit = 2
        inner_module.q_lora_rank = 4

        inner_module.wq_b = nn.Linear(
            inner_module.q_lora_rank,
            inner_module.num_heads * inner_module.head_dim,
            bias=False,
        )
        inner_module.wk = nn.Linear(inner_module.hidden_size, inner_module.head_dim, bias=False)
        inner_module.k_norm = nn.LayerNorm(inner_module.head_dim)
        inner_module.weights_proj = nn.Linear(inner_module.hidden_size, inner_module.num_heads, bias=False)
        inner_module.softmax_scale = inner_module.head_dim**-0.5

        self.inner_module = inner_module
        self.indexer = DeepseekSparseAttentionIndexer(inner_module)

        self.hidden_states = torch.randn(self.batch_size, self.seq_len, inner_module.hidden_size)
        self.qa_normed = torch.randn(self.batch_size, self.seq_len, inner_module.q_lora_rank)
        self.position_embeddings = (
            torch.randn(self.seq_len, inner_module.qk_rope_head_dim),
            torch.randn(self.seq_len, inner_module.qk_rope_head_dim),
        )
        self.indexer_cache = torch.empty(self.batch_size, self.seq_len, inner_module.head_dim)

    def test_topk_limit_is_available_on_wrapper(self):
        self.assertEqual(self.indexer.topk_limit, 2)

    def test_topk_limit_is_cached_on_wrapper(self):
        inner_module = nn.Module()
        inner_module.config = type("Config", (), {"topk_limit": 7})()

        indexer = DeepseekSparseAttentionIndexer(inner_module)
        del inner_module.config

        self.assertEqual(indexer.topk_limit, 7)

    def test_topk_limit_can_be_passed_explicitly(self):
        inner_module = nn.Module()

        indexer = DeepseekSparseAttentionIndexer(inner_module, topk_limit=11)

        self.assertEqual(indexer.topk_limit, 11)

    def test_deepseek_config_ignores_glm5_only_field(self):
        config = DeepseekV32Config(topk_limit=33)

        self.assertEqual(config.topk_limit, 33)
        self.assertFalse(hasattr(config, "index_topk"))

    def test_glm5_index_topk_config_falls_back_when_topk_limit_is_none(self):
        inner_module = nn.Module()
        inner_module.topk_limit = None
        inner_module.config = type("GlmMoeDsaConfig", (), {"index_topk": 21})()

        indexer = DeepseekSparseAttentionIndexer(inner_module)

        self.assertEqual(indexer.topk_limit, 21)

    @patch("torch.ops.tensor_cast.dsa_indexer")
    def test_forward(self, mock_dsa_indexer):
        mock_dsa_indexer.return_value = torch.randn(
            self.batch_size,
            self.seq_len,
            min(self.indexer.topk_limit, self.seq_len),
        )

        res = self.indexer.forward(
            self.hidden_states,
            self.qa_normed,
            self.position_embeddings,
            self.indexer_cache,
        )

        self.assertEqual(
            res.shape,
            (
                self.batch_size,
                self.seq_len,
                min(self.indexer.topk_limit, self.seq_len),
            ),
        )
        mock_dsa_indexer.assert_called_once()

    def test_forward_passes_seq_lens_to_op_after_block_tables(self):
        attention_meta = SimpleNamespace(
            slot_mapping=None,
            block_table_tensor=None,
            seq_lens=torch.tensor([17, 19], dtype=torch.long),
        )

        with patch("torch.ops.tensor_cast.dsa_indexer") as mock_dsa_indexer:
            mock_dsa_indexer.return_value = torch.randn(
                self.batch_size,
                self.seq_len,
                self.indexer.topk_limit,
            )

            self.indexer.forward(
                self.hidden_states,
                self.qa_normed,
                self.position_embeddings,
                self.indexer_cache,
                attention_meta,
            )

        self.assertTrue(torch.equal(mock_dsa_indexer.call_args.args[7], attention_meta.seq_lens))

    def test_dsa_indexer_op_returns_query_major_topk_shape(self):
        batch_size = 2
        seq_len = 3
        topk_limit = 2

        out = torch.ops.tensor_cast.dsa_indexer(
            torch.randn(batch_size, seq_len, self.inner_module.hidden_size),
            torch.randn(batch_size, seq_len, self.inner_module.q_lora_rank),
            torch.randn(seq_len, self.inner_module.qk_rope_head_dim),
            torch.randn(seq_len, self.inner_module.qk_rope_head_dim),
            torch.empty(batch_size, 5, self.inner_module.head_dim),
            None,
            None,
            None,
            self.inner_module.wq_b.weight,
            self.inner_module.wk.weight,
            self.inner_module.weights_proj.weight,
            self.inner_module.k_norm.weight,
            self.inner_module.num_heads,
            self.inner_module.head_dim,
            self.inner_module.qk_rope_head_dim,
            topk_limit,
        )

        self.assertEqual(out.shape, (batch_size, seq_len, topk_limit))

    def test_dsa_indexer_op_uses_active_sequence_length_for_topk_width(self):
        batch_size = 2
        seq_len = 1
        num_heads = 1
        topk_limit = 4
        seq_lens = torch.tensor([3, 4], dtype=torch.long)

        out = torch.ops.tensor_cast.dsa_indexer(
            torch.randn(batch_size, seq_len, self.inner_module.hidden_size),
            torch.randn(batch_size, seq_len, self.inner_module.q_lora_rank),
            torch.randn(seq_len, self.inner_module.qk_rope_head_dim),
            torch.randn(seq_len, self.inner_module.qk_rope_head_dim),
            torch.empty(batch_size, 5, self.inner_module.head_dim),
            None,
            None,
            seq_lens,
            self.inner_module.wq_b.weight,
            self.inner_module.wk.weight,
            self.inner_module.weights_proj.weight,
            self.inner_module.k_norm.weight,
            num_heads,
            self.inner_module.head_dim,
            self.inner_module.qk_rope_head_dim,
            topk_limit,
        )

        self.assertEqual(out.shape, (batch_size, seq_len, topk_limit))

    def test_dsa_indexer_op_compiles_when_seq_lens_is_provided(self):
        def fn(
            hidden_states,
            qa_normed,
            cos,
            sin,
            indexer_cache,
            seq_lens,
            wq_b_weight,
            wk_weight,
            weights_proj_weight,
            k_norm_weight,
        ):
            return torch.ops.tensor_cast.dsa_indexer(
                hidden_states,
                qa_normed,
                cos,
                sin,
                indexer_cache,
                None,
                None,
                seq_lens,
                wq_b_weight,
                wk_weight,
                weights_proj_weight,
                k_norm_weight,
                1,
                self.inner_module.head_dim,
                self.inner_module.qk_rope_head_dim,
                4,
            )

        compiled = torch.compile(fn, backend="eager", fullgraph=True)

        out = compiled(
            torch.randn(2, 1, self.inner_module.hidden_size),
            torch.randn(2, 1, self.inner_module.q_lora_rank),
            torch.randn(1, self.inner_module.qk_rope_head_dim),
            torch.randn(1, self.inner_module.qk_rope_head_dim),
            torch.empty(2, 5, self.inner_module.head_dim),
            torch.tensor([3, 4], dtype=torch.long),
            self.inner_module.wq_b.weight,
            self.inner_module.wk.weight,
            self.inner_module.weights_proj.weight,
            self.inner_module.k_norm.weight,
        )

        self.assertEqual(out.shape, (2, 1, 4))

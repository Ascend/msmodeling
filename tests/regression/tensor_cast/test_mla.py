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

import operator
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from tensor_cast import config
from tensor_cast.compilation.compile_backend import CompilerBackend
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionMetadataBase
from tensor_cast.layers.mla import (
    DeepseekSparseAttention,
    DeepseekSparseAttentionIndexer,
    MultiheadLatentAttentionTensorCast,
)
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import LinearQuantConfig, MlaConfig, QuantConfig
from tensor_cast.parallel_group import ParallelGroup
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity, QuantScheme
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.transformations import quantize_linear
from tensor_cast.transformers.builtin_model.deepseek_v32 import DeepseekV32Config


class TestMlaIndexerCacheHooks(unittest.TestCase):
    def test_base_requires_indexer_cache_is_false(self):
        self.assertFalse(MultiheadLatentAttentionTensorCast.requires_indexer_cache())

    def test_deepseek_sparse_requires_indexer_cache(self):
        self.assertTrue(DeepseekSparseAttention.requires_indexer_cache())

    def test_sparse_backend_kwargs_keep_legacy_single_argument_hook(self):
        topk_indices = torch.tensor([1, 2])
        wrapper = SimpleNamespace(indexer=SimpleNamespace(topk_limit=2))

        kwargs = DeepseekSparseAttention._get_backend_kwargs(
            wrapper,
            topk_indices,
        )

        self.assertEqual(kwargs["topk_limit"], 2)
        self.assertIs(kwargs["topk_indices"], topk_indices)

    def test_sparse_backend_metadata_kwargs_propagate_phase(self):
        phase_values = [False, True]
        meta = SimpleNamespace(is_decode_values=phase_values)
        for cls in (DeepseekSparseAttention, MultiheadLatentAttentionTensorCast):
            kwargs = cls._get_backend_metadata_kwargs(SimpleNamespace(), meta)
            self.assertIs(kwargs["is_decode_values"], phase_values)

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

    def test_singleton_tp_group_uses_full_head_layout(self):
        num_heads = 4
        kv_lora_rank = 64
        qk_nope = 32
        v_head = 16
        kv_b_proj = nn.Linear(kv_lora_rank, num_heads * (qk_nope + v_head), bias=False)
        inner = nn.Module()
        inner.num_heads = num_heads
        inner.kv_lora_rank = kv_lora_rank
        inner.qk_nope_head_dim = qk_nope
        inner.v_head_dim = v_head
        inner.kv_b_proj = kv_b_proj

        wrapper = MultiheadLatentAttentionTensorCast(
            MlaConfig(module_name="FakeMla"),
            inner,
            ParallelGroup(rank=0, rank_groups=[[0]], global_world_size=2),
        )

        self.assertEqual(wrapper._num_heads_per_rank, num_heads)
        self.assertEqual(wrapper.W_UV.shape, torch.Size([num_heads, kv_lora_rank, v_head]))
        self.assertEqual(wrapper.W_UK_T.shape, torch.Size([num_heads, qk_nope, kv_lora_rank]))

    def test_dsa_cp_keeps_selected_linears_unquantized(self):
        inner = nn.Module()
        inner.self_attn = nn.Module()
        inner.self_attn.q_b_proj = nn.Linear(4, 8, bias=False)
        inner.self_attn.kv_b_proj = nn.Linear(4, 8, bias=False)
        inner.self_attn.indexer = nn.Module()
        inner.self_attn.indexer.wq_b = nn.Linear(4, 8, bias=False)
        inner.self_attn.indexer.wk = nn.Linear(4, 2, bias=False)
        inner.self_attn.indexer.weights_proj = nn.Linear(4, 1, bias=False)
        quant_config = QuantConfig(
            linear_configs={
                "*": LinearQuantConfig(
                    quant_type=LinearQuantType.W8A8,
                    weight_scale=torch.tensor(1.0),
                    activation_scale=torch.tensor(1.0),
                )
            }
        )
        model = SimpleNamespace(
            _inner=inner,
            model_config=SimpleNamespace(
                quant_linear_cls=TensorCastQuantLinear,
                quant_config=quant_config,
                mla_config=MlaConfig(module_name="FakeMla", enable_dsa_cp=True),
            ),
        )

        quantize_linear(model)

        self.assertIsInstance(model._inner.self_attn.q_b_proj, TensorCastQuantLinear)
        self.assertIsInstance(model._inner.self_attn.kv_b_proj, nn.Linear)
        self.assertIsInstance(model._inner.self_attn.indexer.wq_b, TensorCastQuantLinear)
        self.assertIsInstance(model._inner.self_attn.indexer.wk, nn.Linear)
        self.assertIsInstance(model._inner.self_attn.indexer.weights_proj, nn.Linear)

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

    def test_direct_q_forward_uses_q_proj_without_lora_intermediates(self):
        inner = nn.Module()
        inner.num_heads = 2
        inner.q_lora_rank = None
        inner.qk_nope_head_dim = 1
        inner.qk_rope_head_dim = 1
        inner.qk_head_dim = 2
        inner.kv_lora_rank = 2
        inner.v_head_dim = 2
        inner.q_proj = nn.Linear(4, 4, bias=False)
        inner.kv_a_proj_with_mqa = nn.Linear(4, 3, bias=False)
        inner.kv_a_layernorm = nn.LayerNorm(2)
        inner.kv_b_proj = nn.Linear(2, 6, bias=False)
        inner.o_proj = nn.Identity()
        wrapper = MultiheadLatentAttentionTensorCast(
            MlaConfig(module_name="FakeMla"),
            inner,
            ParallelGroup(rank=0, rank_groups=[[0]], global_world_size=1),
        )

        hidden_states = torch.empty((1, 3, 4))
        q_states = torch.empty((3, 2, 2))
        kv_c_normed = torch.empty((3, 2))
        k_rot = torch.empty((3, 1))
        unused_qa_normed = torch.empty((3, 0))

        def attention_backend(**kwargs):
            return kwargs["projected_kv"], kwargs["absorbed_q"]

        with (
            patch(
                "torch.ops.tensor_cast.mlapo",
                return_value=(q_states, kv_c_normed, k_rot, unused_qa_normed),
            ) as mock_mlapo,
            patch.object(
                MultiheadLatentAttentionTensorCast,
                "_get_attention_op",
                return_value=attention_backend,
            ),
            patch.object(
                MultiheadLatentAttentionTensorCast,
                "_pre_attention_forward",
                return_value=None,
            ) as mock_pre_attention,
            patch(
                "torch.ops.tensor_cast.mla_kv_projection",
                return_value=torch.empty((3, 2, 3)),
            ) as mock_kv_projection,
            patch(
                "torch.ops.tensor_cast.mla_merge_phase_outputs",
                return_value=torch.empty((3, 2, 2)),
            ) as mock_merge,
        ):
            output, attention_weights = wrapper(
                hidden_states,
                position_embeddings=(torch.empty((3, 1)), torch.empty((3, 1))),
                attention_mask=None,
            )

        mock_mlapo.assert_not_called()
        self.assertIsNone(mock_pre_attention.call_args.kwargs["qa_normed"])
        mock_kv_projection.assert_not_called()
        mock_merge.assert_called_once()
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertIsNone(attention_weights)


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
        # Legacy opaque-op tests explicitly disable expanded indexer preparation.
        self.indexer = DeepseekSparseAttentionIndexer(inner_module, split_bf16=False)

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

    @staticmethod
    def _make_indexer_inner(dtype):
        inner_module = nn.Module()
        inner_module.hidden_size = 16
        inner_module.num_heads = 4
        inner_module.head_dim = 8
        inner_module.qk_rope_head_dim = 4
        inner_module.topk_limit = 2
        inner_module.q_lora_rank = 4
        inner_module.wq_b = nn.Linear(4, 32, bias=False)
        inner_module.wk = nn.Linear(16, 8, bias=False)
        inner_module.k_norm = nn.LayerNorm(8).to(dtype)
        inner_module.weights_proj = nn.Linear(16, 4, bias=False)
        return inner_module

    def test_split_bf16_does_not_retype_the_shared_k_norm(self):
        """The fp32 upcast must not mutate the shared ``k_norm`` submodule.

        ``Module.float()`` retypes in place, so every other reader of the same
        submodule silently loses its original dtype (the GLM5-next indexer reads
        it and does not upcast its input).
        """
        inner_module = self._make_indexer_inner(torch.bfloat16)

        indexer = DeepseekSparseAttentionIndexer(inner_module, split_bf16=True)

        self.assertEqual(inner_module.k_norm.weight.dtype, torch.bfloat16)
        self.assertEqual(inner_module.k_norm.bias.dtype, torch.bfloat16)
        self.assertEqual(indexer.k_norm_weight.dtype, torch.float32)
        self.assertEqual(indexer.k_norm_bias.dtype, torch.float32)
        self.assertEqual(indexer.k_norm_eps, inner_module.k_norm.eps)
        self.assertTrue(torch.equal(indexer.k_norm_weight, inner_module.k_norm.weight.float()))

    def test_without_split_bf16_k_norm_stays_the_shared_submodule(self):
        inner_module = self._make_indexer_inner(torch.bfloat16)

        indexer = DeepseekSparseAttentionIndexer(inner_module, split_bf16=False)

        self.assertIs(indexer.k_norm_weight, inner_module.k_norm.weight)
        self.assertEqual(indexer.k_norm_weight.dtype, torch.bfloat16)

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

        self.assertTrue(torch.equal(mock_dsa_indexer.call_args.args[4], attention_meta.seq_lens))
        self.assertIsNone(mock_dsa_indexer.call_args.args[6])
        self.assertIsNone(mock_dsa_indexer.call_args.kwargs["is_decode_values"])

    def test_forward_caps_topk_at_materialized_sequence_length(self):
        indexer = DeepseekSparseAttentionIndexer(self.inner_module, topk_limit=8, split_bf16=False)
        attention_meta = SimpleNamespace(
            slot_mapping=None,
            block_table_tensor=None,
            seq_lens=torch.tensor([3, 4], dtype=torch.long),
            seq_lens_values=[3, 4],
        )
        with patch("torch.ops.tensor_cast.dsa_indexer") as mock_dsa_indexer:
            mock_dsa_indexer.return_value = torch.empty(self.batch_size * self.seq_len, 4)
            result = indexer.forward(
                self.hidden_states, self.qa_normed, self.position_embeddings, self.indexer_cache, attention_meta
            )
        self.assertEqual(mock_dsa_indexer.call_args.args[5], 4)
        self.assertEqual(result.shape, (self.batch_size, self.seq_len, 4))

    def test_forward_passes_decode_phase_to_op(self):
        phase_values = [False, True]
        query_lens = torch.tensor([3, 17], dtype=torch.long)
        attention_meta = SimpleNamespace(
            slot_mapping=None,
            block_table_tensor=None,
            seq_lens=torch.tensor([17, 19], dtype=torch.long),
            query_lens=query_lens,
            is_decode_values=phase_values,
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

        self.assertIs(mock_dsa_indexer.call_args.args[6], query_lens)
        self.assertIs(mock_dsa_indexer.call_args.kwargs["is_decode_values"], phase_values)

    def test_dsa_indexer_op_returns_query_major_topk_shape(self):
        batch_size = 2
        seq_len = 3
        topk_limit = 2

        out = torch.ops.tensor_cast.dsa_indexer(
            torch.randn(batch_size * seq_len, self.inner_module.num_heads, self.inner_module.head_dim),
            torch.randn(batch_size * seq_len, self.inner_module.num_heads),
            torch.empty(5, self.inner_module.head_dim),
            None,
            None,
            topk_limit,
            None,
        )

        self.assertEqual(out.shape, (batch_size * seq_len, topk_limit))

    def test_dsa_indexer_op_uses_active_sequence_length_for_topk_width(self):
        batch_size = 2
        seq_len = 1
        topk_limit = 4
        seq_lens = torch.tensor([3, 4], dtype=torch.long)

        out = torch.ops.tensor_cast.dsa_indexer(
            torch.randn(batch_size * seq_len, self.inner_module.num_heads, self.inner_module.head_dim),
            torch.randn(batch_size * seq_len, self.inner_module.num_heads),
            torch.empty(5, self.inner_module.head_dim),
            None,
            seq_lens,
            topk_limit,
            None,
        )

        self.assertEqual(out.shape, (batch_size * seq_len, topk_limit))

    def test_dsa_indexer_op_compiles_when_seq_lens_is_provided(self):
        def fn(query, weights, indexer_cache, seq_lens):
            return torch.ops.tensor_cast.dsa_indexer(
                query,
                weights,
                indexer_cache,
                None,
                seq_lens,
                4,
                None,
            )

        compiled = torch.compile(fn, backend="eager", fullgraph=True)

        out = compiled(
            torch.randn(2, self.inner_module.num_heads, self.inner_module.head_dim),
            torch.randn(2, self.inner_module.num_heads),
            torch.empty(5, self.inner_module.head_dim),
            torch.tensor([3, 4], dtype=torch.long),
        )

        self.assertEqual(out.shape, (2, 4))


class TestMlaProjectionSequenceParallelShapes(unittest.TestCase):
    def test_kv_projection_clamps_declared_tokens_to_local_activation(self):
        kv_c = torch.empty(256, 512, device="meta")
        weight = torch.empty(512, 4096, device="meta")
        out = torch.ops.tensor_cast.mla_kv_projection(kv_c, weight, 4096, 16, 0)
        self.assertEqual(tuple(out.shape), (256, 16, 256))

    def test_q_absorb_clamps_declared_tokens_to_local_activation(self):
        q = torch.empty(32, 16, 192, device="meta")
        w_uk_t = torch.empty(16, 128, 512, device="meta")
        out = torch.ops.tensor_cast.mla_q_absorb_projection(q, w_uk_t, 512, 64, 0)
        self.assertEqual(tuple(out.shape), (32, 16, 576))

    def test_merge_phase_outputs_follows_phase_tensors_not_global_token_arg(self):
        prefill = torch.empty(256, 16, 128, device="meta")
        decode = torch.empty(0, 16, 128, device="meta")
        out = torch.ops.tensor_cast.mla_merge_phase_outputs(prefill, decode, 4096, [False], [256])
        self.assertEqual(tuple(out.shape), (256, 16, 128))

    def test_mixed_sp_phase_counts_sum_to_local_activation(self):
        from tensor_cast.ops.mla import local_mla_phase_token_counts

        prefill_local, decode_local = local_mla_phase_token_counts(32, 48, 16)
        self.assertEqual((prefill_local, decode_local), (24, 8))

        kv_c = torch.empty(32, 512, device="meta")
        weight = torch.empty(512, 4096, device="meta")
        kv_out = torch.ops.tensor_cast.mla_kv_projection(kv_c, weight, 48, 16, 16)
        q = torch.empty(32, 16, 192, device="meta")
        w_uk_t = torch.empty(16, 128, 512, device="meta")
        q_out = torch.ops.tensor_cast.mla_q_absorb_projection(q, w_uk_t, 16, 64, 48)
        self.assertEqual(kv_out.shape[0], 24)
        self.assertEqual(q_out.shape[0], 8)
        merged = torch.ops.tensor_cast.mla_merge_phase_outputs(
            torch.empty(24, 16, 128, device="meta"),
            torch.empty(8, 16, 128, device="meta"),
            64,
            [False, True],
            [48, 16],
        )
        self.assertEqual(tuple(merged.shape), (32, 16, 128))


# MLA fusion and DSA boundary regression cases.


def make_layer(direct=False):
    inner = nn.Module()
    inner.num_heads = 2
    inner.q_lora_rank = None if direct else 4
    inner.qk_nope_head_dim = 2
    inner.qk_rope_head_dim = 2
    inner.qk_head_dim = 4
    inner.kv_lora_rank = 4
    inner.v_head_dim = 2
    inner.layer_idx = 0
    inner.q_proj = nn.Linear(8, 8, bias=False)
    inner.q_a_proj = nn.Linear(8, 4, bias=False)
    inner.q_a_layernorm = nn.RMSNorm(4)
    inner.q_b_proj = nn.Linear(4, 8, bias=False)
    inner.kv_a_proj_with_mqa = nn.Linear(8, 6, bias=False)
    inner.kv_a_layernorm = nn.RMSNorm(4)
    inner.kv_b_proj = nn.Linear(4, 8, bias=False)
    inner.o_proj = nn.Linear(4, 8, bias=False)
    inner.to(device="meta", dtype=torch.float16)
    return MultiheadLatentAttentionTensorCast(
        MlaConfig(module_name="TinyMla"),
        inner,
        ParallelGroup(rank=0, rank_groups=[[0]], global_world_size=1),
    )


def test_unfused_mixed_projections_use_tensor_request_lengths(monkeypatch):
    from tensor_cast.layers.mla import _get_mla_phase_token_counts

    layer = make_layer(direct=True)
    metadata = SimpleNamespace(query_lens=torch.tensor([1, 2, 1]), is_decode_values=[True, False, True])
    assert _get_mla_phase_token_counts(metadata, 4, False) == (2, 2)
    kv = torch.arange(4, dtype=torch.float16).unsqueeze(1).expand(4, 4)
    q = torch.arange(4, dtype=torch.float16).view(4, 1, 1).expand(4, 2, 4)
    selected = {}

    def project_kv(values, weights):
        selected["kv"] = values[:, 0].tolist()
        return torch.empty(values.shape[0], weights.shape[1])

    def project_q(values, weights):
        selected["q"] = values[0, :, 0].tolist()
        return torch.empty(values.shape[0], values.shape[1], weights.shape[2])

    monkeypatch.setattr(torch, "mm", project_kv)
    monkeypatch.setattr(torch, "bmm", project_q)
    layer._compute_unfused_mla_projections(kv, q, 2, 2, metadata)
    assert selected == {"kv": [1.0, 2.0], "q": [0.0, 3.0]}


def test_mixed_mla_phase_requires_materialized_request_lengths():
    from tensor_cast.layers.mla import _get_mla_phase_token_counts

    metadata = SimpleNamespace(
        query_lens=torch.empty(2, dtype=torch.long, device="meta"),
        is_decode_values=[False, True],
    )
    with pytest.raises(ValueError, match="materialized per-request query lengths"):
        _get_mla_phase_token_counts(metadata, 4, False)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("prefill,decode", [(8, 0), (0, 2), (8, 2)])
@pytest.mark.parametrize("direct", [False, True])
def test_forward_kernel_ownership(monkeypatch, enabled, prefill, decode, direct):
    monkeypatch.setattr(config.compilation.fusion_patterns, "enable_mlapo", enabled)
    layer = make_layer(direct)
    lengths = [n for n in (prefill, decode) if n]
    phases = ([False] if prefill else []) + ([True] if decode else [])
    total = prefill + decode
    metadata = AttentionMetadataBase(
        query_start_loc=torch.tensor([0] + ([prefill] if prefill else []) + ([total] if decode else [])),
        seq_lens=torch.tensor([16] * len(lengths)),
        query_lens=torch.tensor(lengths),
        slot_mapping=torch.arange(total),
        seq_lens_values=[16] * len(lengths),
        query_lens_values=lengths,
        is_decode_values=phases,
    )
    hidden = torch.empty(1, total, 8, device="meta", dtype=torch.float16)
    cache = torch.empty(1, 16, 6, device="meta", dtype=torch.float16)
    rope = torch.empty(total, 2, device="meta", dtype=torch.float16)
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
        output, _ = layer(
            hidden,
            (rope, rope),
            None,
            attention_meta=metadata,
            kv_cache_by_layers=[cache],
        )
    names = [str(event.op_invoke_info.func).split(".")[1] for event in runtime.event_list]
    assert output.shape == hidden.shape
    assert names.count("mlapo") == int(enabled)
    assert names.count("mla_kv_projection") == 0
    assert names.count("mla_q_absorb_projection") == 0
    assert names.count("mla_v_up_projection") == int(bool(decode))
    assert names.count("concat_and_cache_mla") == 0
    # The unfused cache write must ride the dedicated op: a bare index_copy_ has
    # no registered op-properties functor, so the analytic model would charge the
    # whole kv_cache instead of only the appended rows.
    assert names.count("index_copy_") == 0
    assert names.count("siso_reshape_and_cache") == int(not enabled)
    assert names.count("multihead_latent_attention") == 1
    if enabled:
        event = next(event for event in runtime.event_list if "mlapo" in str(event.op_invoke_info.func))
        assert event.perf_results["analytic"].statistics["static_cost_time_s"] == TEST_DEVICE.static_cost.mma_op_cost_s
        assert event.op_invoke_info.kwargs["is_decode_values"] == phases
        assert event.op_invoke_info.out[4].shape == (prefill, 2, 4)
        assert event.op_invoke_info.out[5].shape == (decode, 2, 6)


@pytest.mark.parametrize("module", ["text_generate", "throughput_optimizer"])
def test_cli_and_worker_preserve_fusion_selection(monkeypatch, module):
    from importlib import import_module

    from cli.registry.argparse_adapter import build_argparser
    from tensor_cast.core.user_config import UserInputConfig
    from serving_cast.parallel_runner import ParallelRunner

    parser = build_argparser(import_module("cli.registry.modules." + module).SPEC)
    required = (
        ["--num-queries", "1", "--query-length", "8"]
        if module == "text_generate"
        else ["--input-length", "8", "--output-length", "8"]
    )
    args = parser.parse_args(
        [
            "tests/assets/model_config/deepseek_v3",
            *required,
            "--compilation-config",
            "enable_mlapo",
        ]
    )
    if module == "throughput_optimizer":
        # The optimizer selects one device before constructing worker configs.
        args.device = "TEST_DEVICE"
    user_input = UserInputConfig.from_args(args)
    assert user_input.enable_mlapo
    monkeypatch.setattr(config.compilation.fusion_patterns, "enable_mlapo", False)
    ParallelRunner._apply_compilation_config(None, user_input)
    assert config.compilation.fusion_patterns.enable_mlapo
    user_input.enable_mlapo = False
    ParallelRunner._apply_compilation_config(None, user_input)
    assert not config.compilation.fusion_patterns.enable_mlapo


@pytest.mark.parametrize("direct", [False, True])
def test_compiled_unfused_prolog_allows_local_linear_merge(monkeypatch, direct):
    from tensor_cast.compilation.compile_backend import CompilerBackend

    monkeypatch.setattr(config.compilation.fusion_patterns, "enable_mlapo", False)
    layer = make_layer(direct)
    hidden = torch.empty(8, 8, dtype=torch.float16, device="meta")
    rope = torch.empty(8, 2, dtype=torch.float16, device="meta")
    compiled = torch.compile(layer._compute_mla_prolog, backend=CompilerBackend(), fullgraph=True)
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
        compiled(hidden, rope, rope)
    names = [str(e.op_invoke_info.func) for e in runtime.event_list]
    assert names.count("aten.mm.default") == (1 if direct else 2)
    merged = next(e for e in runtime.event_list if str(e.op_invoke_info.func) == "aten.mm.default")
    assert merged.op_invoke_info.out.shape == (8, 14 if direct else 10)
    assert names.count("tensor_cast.rms_norm.default") == (1 if direct else 2)
    assert names.count("tensor_cast.apply_rope_single.default") == 2
    assert not any("mlapo" in name for name in names)


@pytest.mark.parametrize("rank", [0, 1])
def test_mixed_dsa_sp_falls_back_to_full_tokens(rank):
    from tensor_cast.compilation.passes.sequence_parallel_pass import _keep_dsa_attention_local

    graph = torch.fx.Graph()
    activation = graph.placeholder("activation")
    activation.meta["val"] = torch.empty(1, 8, 4, device="meta")
    comm = graph.call_function(torch.ops.tensor_cast.all_reduce.default, (activation, 0, rank, [0, 1]))
    norm = graph.call_function(torch.ops.tensor_cast.rms_norm.default, (comm,))
    norm.meta["val"] = torch.empty(1, 8, 4, device="meta")
    attention = graph.call_function(
        torch.ops.tensor_cast.mla_sparse_attention.default,
        (norm,),
        {"is_decode_values": [False, True]},
    )
    graph.output(attention)

    _keep_dsa_attention_local(graph, comm, norm, rank, [0, 1])

    graph.lint()
    assert attention.args[0].target is torch.ops.tensor_cast.all_gather.default
    assert attention.args[0].args[0] is norm
    assert not norm.meta.get("tensor_cast_sp_local")


@pytest.mark.parametrize("rank", [0, 1])
def test_unfused_sp_metadata_follows_rank_rows(monkeypatch, rank):
    from tensor_cast.compilation.passes.sequence_parallel_pass import (
        _shard_mla_metadata,
    )

    monkeypatch.setattr(config.compilation.fusion_patterns, "enable_mlapo", False)
    graph = torch.fx.Graph()
    activation = graph.placeholder("activation")
    positions = graph.placeholder("positions")
    positions.meta["val"] = torch.arange(16).view(1, 8, 2)
    slots = graph.placeholder("slots")
    slots.meta["val"] = torch.arange(8)
    cache = graph.placeholder("cache")
    rope = graph.call_function(
        torch.ops.tensor_cast.apply_rope_single.default,
        (activation, positions, positions, True, False),
    )
    write = graph.call_function(torch.ops.aten.index_copy.default, (cache, 0, slots, rope))
    attention = graph.call_function(torch.ops.tensor_cast.mla_sparse_attention.default, (write,))
    graph.output(attention)
    _shard_mla_metadata(graph, activation, 8, 4, rank, [0, 1])
    graph.lint()
    for node in rope.args[1:3]:
        torch.testing.assert_close(node.meta["val"], positions.meta["val"][:, rank * 4 : (rank + 1) * 4])
    torch.testing.assert_close(write.args[2].meta["val"], torch.arange(rank * 4, (rank + 1) * 4))


@pytest.mark.parametrize("rank", [0, 1])
def test_sp_metadata_ignores_unrelated_rope(rank):
    from tensor_cast.compilation.passes.sequence_parallel_pass import _shard_mla_metadata, is_dsa_attention

    graph = torch.fx.Graph()
    activation = graph.placeholder("activation")
    positions = graph.placeholder("positions")
    positions.meta["val"] = torch.arange(16).view(1, 8, 2)
    other_positions = graph.placeholder("other_positions")
    other_positions.meta["val"] = torch.arange(16).view(1, 8, 2)
    relevant_rope = graph.call_function(
        torch.ops.tensor_cast.apply_rope_single.default,
        (activation, positions, positions, True, False),
    )
    unrelated_rope = graph.call_function(
        torch.ops.tensor_cast.apply_rope_single.default,
        (activation, other_positions, other_positions, True, False),
    )
    attention = graph.call_function(torch.ops.tensor_cast.mla_sparse_attention.default, (relevant_rope,))
    graph.output((attention, unrelated_rope))

    assert is_dsa_attention(activation)
    _shard_mla_metadata(graph, activation, 8, 4, rank, [0, 1])
    graph.lint()
    assert relevant_rope.args[1] is not positions
    assert unrelated_rope.args[1] is other_positions
    assert unrelated_rope.args[2] is other_positions


@pytest.mark.parametrize("rank", [0, 1])
def test_sp_metadata_stops_at_dense_mla_and_next_norm(rank):
    from tensor_cast.compilation.passes.sequence_parallel_pass import _shard_mla_metadata

    graph = torch.fx.Graph()
    activation = graph.placeholder("activation")
    positions = graph.placeholder("positions")
    positions.meta["val"] = torch.arange(16).view(1, 8, 2)
    dense_positions = graph.placeholder("dense_positions")
    dense_positions.meta["val"] = torch.arange(16).view(1, 8, 2)
    downstream_positions = graph.placeholder("downstream_positions")
    downstream_positions.meta["val"] = torch.arange(16).view(1, 8, 2)
    dsa_rope = graph.call_function(
        torch.ops.tensor_cast.apply_rope_single.default,
        (activation, positions, positions, True, False),
    )
    dense_rope = graph.call_function(
        torch.ops.tensor_cast.apply_rope_single.default,
        (activation, dense_positions, dense_positions, True, False),
    )
    dense_attention = graph.call_function(torch.ops.tensor_cast.multihead_latent_attention.default, (dense_rope,))
    downstream_rope = graph.call_function(
        torch.ops.tensor_cast.apply_rope_single.default,
        (dense_attention, downstream_positions, downstream_positions, True, False),
    )
    dsa_attention = graph.call_function(torch.ops.tensor_cast.mla_sparse_attention.default, (dsa_rope,))
    graph.output((dsa_attention, downstream_rope))

    _shard_mla_metadata(graph, activation, 8, 4, rank, [0, 1])
    graph.lint()
    assert dsa_rope.args[1] is not positions
    assert dense_rope.args[1] is dense_positions
    assert downstream_rope.args[1] is downstream_positions


def test_sp_metadata_without_dsa_sink_fails_closed():
    from tensor_cast.compilation.passes.sequence_parallel_pass import _shard_mla_metadata

    graph = torch.fx.Graph()
    activation = graph.placeholder("activation")
    positions = graph.placeholder("positions")
    graph.call_function(
        torch.ops.tensor_cast.apply_rope_single.default,
        (activation, positions, positions, True, False),
    )

    with pytest.raises(RuntimeError, match="DSA attention consumer"):
        _shard_mla_metadata(graph, activation, 8, 4, 0, [0, 1])


@pytest.mark.parametrize("rank", [0, 1])
def test_sp_metadata_shards_functionalized_cache_write(rank):
    from tensor_cast.compilation.passes.sequence_parallel_pass import _shard_mla_metadata

    graph = torch.fx.Graph()
    activation = graph.placeholder("activation")
    cache = graph.placeholder("cache")
    slots = graph.placeholder("slots")
    slots.meta["val"] = torch.arange(8)
    write = graph.call_function(
        torch.ops.higher_order.auto_functionalized_v2,
        (torch.ops.tensor_cast.siso_reshape_and_cache.default,),
        {"key": activation, "slot_mapping": slots, "_kv_cache_base_index": 0, "_all_bases": [cache]},
    )
    updated_cache = graph.call_function(operator.getitem, (write, 1))
    attention = graph.call_function(torch.ops.tensor_cast.mla_sparse_attention.default, (updated_cache,))
    graph.output(attention)

    _shard_mla_metadata(graph, activation, 8, 4, rank, [0, 1])
    graph.lint()
    torch.testing.assert_close(write.kwargs["slot_mapping"].meta["val"], torch.arange(rank * 4, (rank + 1) * 4))


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "fused_target",
    [torch.ops.tensor_cast.mlapo.default, torch.ops.tensor_cast.mlapo_quant.default],
    ids=["mlapo", "mlapo_quant"],
)
@pytest.mark.parametrize("with_slot_mapping", [True, False], ids=["slots", "no_slots"])
def test_fused_sp_metadata_follows_rank_rows(monkeypatch, rank, fused_target, with_slot_mapping):
    """The fused MLAPO prolog must shard its rotary caches and slot mapping.

    The fused node carries no ``apply_rope_single`` / ``siso_reshape_and_cache`` call
    for the unfused branches to key on, so it used to slip past the metadata
    sharding while ``_mark_local_descendants`` had already rewritten its
    activation to local tokens. The op then took a T/TP activation together with
    T rotary caches and slot mapping. The global phase counters must stay global,
    because the op splits the local activation between the phases itself.
    """
    from tensor_cast.compilation.passes.sequence_parallel_pass import (
        _shard_mla_metadata,
    )

    monkeypatch.setattr(config.compilation.fusion_patterns, "enable_mlapo", True)
    graph = torch.fx.Graph()
    activation = graph.placeholder("activation")
    activation.meta["val"] = torch.empty(8, 64)
    cos = graph.placeholder("cos")
    cos.meta["val"] = torch.arange(16).view(1, 8, 2)
    sin = graph.placeholder("sin")
    sin.meta["val"] = torch.arange(16).view(1, 8, 2)
    slots = graph.placeholder("slots")
    slots.meta["val"] = torch.arange(8)
    cache = graph.placeholder("cache")
    weight = graph.placeholder("weight")
    weight.meta["val"] = torch.empty(4, 64)
    kwargs = {"num_prefill_tokens": 8, "num_decode_tokens": 0}
    if with_slot_mapping:
        kwargs["slot_mapping"] = slots
    # Only positional indices 1/2 (cos/sin) and the slot_mapping keyword carry
    # token-indexed metadata, so the trailing prefix is shared by both variants.
    fused = graph.call_function(
        fused_target,
        (activation, cos, sin, weight, None, weight, weight, weight, 2, 4, 2, 2, 4, 4, cache),
        kwargs,
    )
    fused.meta["val"] = torch.empty(8, 2, 4)
    attention = graph.call_function(torch.ops.tensor_cast.mla_sparse_attention.default, (fused,))
    graph.output(attention)

    _shard_mla_metadata(graph, activation, 8, 4, rank, [0, 1])
    graph.lint()

    expected_rows = torch.arange(rank * 4, (rank + 1) * 4)
    for node in fused.args[1:3]:
        torch.testing.assert_close(node.meta["val"], cos.meta["val"][:, rank * 4 : (rank + 1) * 4])
    if with_slot_mapping:
        torch.testing.assert_close(fused.kwargs["slot_mapping"].meta["val"], expected_rows)
    else:
        assert "slot_mapping" not in fused.kwargs
    # Global counters stay global; only the op's own activation becomes local.
    assert fused.kwargs["num_prefill_tokens"] == 8
    assert fused.kwargs["num_decode_tokens"] == 0


@pytest.mark.parametrize("mlapo", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_indexer_projection_ownership(mlapo, compiled, cache_dtype, monkeypatch):
    monkeypatch.setattr(config.compilation.fusion_patterns, "enable_mlapo", mlapo)
    inner = nn.Module()
    inner.num_heads, inner.head_dim, inner.qk_rope_head_dim = 2, 4, 2
    inner.wk = nn.Linear(8, 4, bias=False)
    inner.weights_proj = nn.Linear(8, 2, bias=False)
    inner.wq_b = nn.Linear(4, 8, bias=False)
    inner.k_norm = nn.LayerNorm(4)
    inner.to(device="meta", dtype=torch.bfloat16)
    indexer = DeepseekSparseAttentionIndexer(inner, topk_limit=2)
    hidden = torch.empty(1, 4, 8, device="meta", dtype=torch.bfloat16)
    weight = torch.empty(8, 4, device="meta", dtype=torch.bfloat16)
    cache = torch.empty(1, 8, 4, device="meta", dtype=cache_dtype)
    rope = torch.empty(1, 4, 2, device="meta", dtype=torch.bfloat16)
    meta = SimpleNamespace(
        slot_mapping=torch.arange(4),
        block_table_tensor=torch.zeros(1, 1, dtype=torch.int32),
        seq_lens=torch.tensor([4]),
        query_lens=torch.tensor([4]),
        is_decode_values=[False],
    )

    def forward(x, cache):
        qa = torch.mm(x.view(-1, 8), weight).view(1, -1, 4)
        return indexer(x, qa, (rope, rope), cache, meta)

    if compiled:
        forward = torch.compile(forward, backend=CompilerBackend(), fullgraph=True)
    with torch.no_grad(), Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
        output = forward(hidden, cache)
    assert output.shape == (1, 4, 2)
    events = runtime.event_list
    names = [str(event.op_invoke_info.func) for event in events]
    assert names.count("tensor_cast.layer_norm.default") == 1
    assert names.count("tensor_cast.apply_rope_single.default") == 2
    assert names.count("tensor_cast.dsa_indexer.default") == 1
    if compiled:
        assert names.count("aten.mm.default") == 2
        assert names.count("aten.split_with_sizes.default") >= 1


def test_core_counts_only_scoring():
    from tensor_cast.performance_model import _dsa_indexer_properties
    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
    from tensor_cast.performance_model.profiling_database.profiling_data_source import _decompose_dsa_indexer

    query = torch.empty(4, 2, 8, dtype=torch.bfloat16, device="meta")
    weights = torch.empty(4, 2, dtype=torch.bfloat16, device="meta")
    cache = torch.empty(1, 16, 8, dtype=torch.bfloat16, device="meta")
    args = (query, weights, cache, torch.zeros(1, 1, dtype=torch.int32), torch.tensor([16]), 2, torch.tensor([4]))
    out = torch.empty(4, 2, dtype=torch.long, device="meta")
    info = OpInvokeInfo(torch.ops.tensor_cast.dsa_indexer.default, args, {"is_decode_values": [False]}, out)
    properties = _dsa_indexer_properties(info)
    assert properties.compute_ops[torch.bfloat16].mma_ops == 2 * 4 * 2 * 16 * 8
    assert properties.memory_write_bytes == out.numel() * out.element_size()
    specs = _decompose_dsa_indexer(info, {})
    assert [spec.kernel_type for spec in specs] == ["LightningIndexer"]


def test_fp8_indexer_core_does_not_charge_external_cache_append():
    from tensor_cast.performance_model import _dsa_indexer_properties
    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    query = torch.empty(1, 2, 8, dtype=torch.bfloat16, device="meta")
    weights = torch.empty(1, 2, dtype=torch.bfloat16, device="meta")
    cache = torch.empty(1, 16, 8, dtype=torch.float8_e4m3fn, device="meta")
    out = torch.empty(1, 2, dtype=torch.long, device="meta")
    info = OpInvokeInfo(
        torch.ops.tensor_cast.dsa_indexer.default,
        (query, weights, cache, None, torch.tensor([16]), 2, torch.tensor([1])),
        {"is_decode_values": [True]},
        out,
    )

    properties = _dsa_indexer_properties(info)
    assert properties.memory_write_bytes == out.numel() * out.element_size()

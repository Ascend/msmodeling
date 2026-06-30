import tempfile
import unittest
from unittest.mock import Mock

import pytest
import torch
from parameterized import parameterized
from tensor_cast.compilation import get_backend
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.parallel_linear import ColumnParallelLinear
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.performance_model import _estimate_dsa_indexer_breakdown
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel, OpBoundClassifier
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.bound_analyzer import BoundAnalyzer, StatsKey
from tensor_cast.performance_model.empirical import EmpiricalPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.performance_model.profiling_database.data_source import (
    DataSourcePerformanceModel,
    QueryResult,
    QuerySource,
)
from tensor_cast.runtime import Runtime, RuntimeEvent
from .test_common import (
    assert_close,
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    get_cached_build_model,
    has_submodule_with_cls_name,
)

# Core runtime quantization zero-size assertions were moved to the unified entry in test_dtype.py.


class PerfAnalysisTestMixin:
    @classmethod
    def setUpClass(cls):
        cls._model_cache = {}

    @classmethod
    def _get_model(cls, user_config: UserInputConfig):
        return get_cached_build_model(cls._model_cache, user_config)

    def setUp(self):
        self.data_source = Mock(spec=DataSourcePerformanceModel)
        self.fallback_model = Mock(spec=PerformanceModel)
        # Configure fallback to return a valid Result for M5 latency tracking
        fallback_result = Mock()
        fallback_result.execution_time_s = 1e-6
        self.fallback_model.process_op.return_value = fallback_result
        torch.compiler.reset()


class PerfAnalysisTestCase(PerfAnalysisTestMixin, unittest.TestCase):
    def _execute_attention_and_get_base_data(self, attention_args):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            torch.ops.tensor_cast.attention(*attention_args)
        self.assertEqual(len(runtime.event_list), 1)
        analytic_result = runtime.event_list[0].perf_results.get("analytic")
        actual_execution_time = analytic_result.execution_time_s
        return actual_execution_time

    def _execute_linear_attention_and_get_base_data(self, linear_attention_args):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            torch.ops.tensor_cast.linear_attention(*linear_attention_args)
        self.assertEqual(len(runtime.event_list), 1)
        analytic_result = runtime.event_list[0].perf_results.get("analytic")
        actual_execution_time = analytic_result.execution_time_s
        return actual_execution_time

    def _execute_multihead_latent_attention_and_get_base_data(self, mla_args):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            torch.ops.tensor_cast.multihead_latent_attention(*mla_args)
        self.assertEqual(len(runtime.event_list), 1)
        analytic_result = runtime.event_list[0].perf_results.get("analytic")
        actual_execution_time = analytic_result.execution_time_s
        return actual_execution_time

    def _execute_mlapo_and_get_base_data(self, mlapo_args):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            torch.ops.tensor_cast.mlapo(*mlapo_args)
        self.assertEqual(len(runtime.event_list), 1)
        analytic_result = runtime.event_list[0].perf_results.get("analytic")
        actual_execution_time = analytic_result.execution_time_s
        return actual_execution_time

    def _execute_mlapo_quant_and_get_base_data(self, mlapo_args):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            torch.ops.tensor_cast.mlapo_quant(*mlapo_args)
        self.assertEqual(len(runtime.event_list), 1)
        analytic_result = runtime.event_list[0].perf_results.get("analytic")
        actual_execution_time = analytic_result.execution_time_s
        return actual_execution_time

    def test_mlapo_quant_uint8_weights_use_int8_mma_modeling(self):
        hidden_states = torch.empty((4, 6144), device="meta", dtype=torch.float16)
        cos = torch.empty((1, 4, 64), device="meta", dtype=torch.float16)
        sin = torch.empty((1, 4, 64), device="meta", dtype=torch.float16)
        q_a_proj_weight = torch.empty((2048, 3072), device="meta", dtype=torch.uint8)
        q_a_layernorm_weight = torch.empty((2048,), device="meta", dtype=torch.float16)
        q_b_proj_weight = torch.empty((1024, 1024), device="meta", dtype=torch.uint8)
        kv_a_proj_weight = torch.empty((576, 3072), device="meta", dtype=torch.uint8)
        kv_a_layernorm_weight = torch.empty((512,), device="meta", dtype=torch.float16)
        scale = torch.tensor(1.0, device="meta")
        args = (
            hidden_states,
            cos,
            sin,
            q_a_proj_weight,
            q_a_layernorm_weight,
            q_b_proj_weight,
            kv_a_proj_weight,
            kv_a_layernorm_weight,
            4,
            256,
            192,
            64,
            512,
            2048,
            scale,
            None,
            scale,
            None,
            scale,
            None,
        )
        out = torch.ops.tensor_cast.mlapo_quant(*args)
        op_invoke_info = OpInvokeInfo(torch.ops.tensor_cast.mlapo_quant.default, args, None, out)

        properties = op_invoke_info.get_perf_properties()
        self.assertNotIn(torch.uint8, properties.compute_ops)
        self.assertGreater(properties.compute_ops[torch.int8].mma_ops, 0)
        self.assertEqual(properties.extra_static_cost_count, 15)

        result = AnalyticPerformanceModel(TEST_DEVICE).process_op(op_invoke_info)
        self.assertGreater(result.statistics[StatsKey.MMA_OPS], 0)

    def test_mlapo_uses_extra_static_cost_modeling(self):
        hidden_states = torch.empty((4, 16), device="meta", dtype=torch.float16)
        cos = torch.empty((1, 4, 4), device="meta", dtype=torch.float16)
        sin = torch.empty((1, 4, 4), device="meta", dtype=torch.float16)
        q_a_proj_weight = torch.empty((8, 16), device="meta", dtype=torch.float16)
        q_a_layernorm_weight = torch.empty((8,), device="meta", dtype=torch.float16)
        q_b_proj_weight = torch.empty((16, 8), device="meta", dtype=torch.float16)
        kv_a_proj_weight = torch.empty((12, 16), device="meta", dtype=torch.float16)
        kv_a_layernorm_weight = torch.empty((8,), device="meta", dtype=torch.float16)
        args = (
            hidden_states,
            cos,
            sin,
            q_a_proj_weight,
            q_a_layernorm_weight,
            q_b_proj_weight,
            kv_a_proj_weight,
            kv_a_layernorm_weight,
            2,
            8,
            4,
            4,
            8,
            8,
        )
        out = torch.ops.tensor_cast.mlapo(*args)
        properties = OpInvokeInfo(torch.ops.tensor_cast.mlapo.default, args, None, out).get_perf_properties()

        self.assertEqual(properties.extra_static_cost_count, 15)

    def test_simple_model_eager(self):
        def func(x):
            return x + x

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            x = torch.randn([100], device="meta")
            _ = func(x)
        self.assertEqual(len(runtime.event_list), 3)

    def test_simple_model_compile(self):
        @torch.compile(backend=get_backend())
        def func(x):
            return x + x

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            x = torch.randn([100], device="meta")
            _ = func(x)
        self.assertEqual(len(runtime.event_list), 3)

    def test_runtime_closes_torch_patches(self):
        from torch import _prims_common

        original_dtype_to_type = _prims_common.dtype_to_type
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)

        with self.assertRaisesRegex(RuntimeError, "stop runtime"):
            with Runtime(perf_model, device_profile):
                self.assertIsNot(_prims_common.dtype_to_type, original_dtype_to_type)
                raise RuntimeError("stop runtime")

        self.assertIs(_prims_common.dtype_to_type, original_dtype_to_type)

    def test_attention_dit_eager(self):
        B, S, num_heads, head_dim = 2, 256, 6, 64
        dtype = torch.float16

        q = torch.randn(B, S, num_heads, head_dim, device="meta", dtype=dtype)
        k = torch.randn(B, S, num_heads, head_dim, device="meta", dtype=dtype)
        v = torch.randn(B, S, num_heads, head_dim, device="meta", dtype=dtype)

        actual_execution_time = self._execute_attention_and_get_base_data((q, k, v, None, None, None, None, None))

        assert_close(self, actual_execution_time, 6.49e-6)

    def test_attention_llm_eager(self):
        B, S, num_kv_heads, head_dim = 2, 256, 8, 64
        block_size, dtype = 128, torch.float16
        hidden_size, query_len = num_kv_heads * head_dim, 1
        total_tokens = B * query_len

        q = torch.randn(total_tokens, hidden_size, device="meta", dtype=dtype)
        max_num_blocks_per_seq = (S + block_size - 1) // block_size
        num_blocks = B * max_num_blocks_per_seq
        k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device="meta", dtype=dtype)
        v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device="meta", dtype=dtype)
        block_table = torch.empty((B, max_num_blocks_per_seq), dtype=torch.long, device="meta")
        request_total_seq_lens = torch.full((B,), S, dtype=torch.long, device="cpu")
        query_lens = torch.full((B,), query_len, dtype=torch.long, device="cpu")

        actual_execution_time = self._execute_attention_and_get_base_data(
            (q, k, v, None, block_table, None, request_total_seq_lens, query_lens)
        )

        assert_close(self, actual_execution_time, 5.99e-6)

    def test_linear_attention_eager(self):
        hidden_states = torch.randn(2, 16, 4096, device="meta", dtype=torch.float16)
        actual_execution_time = self._execute_linear_attention_and_get_base_data(
            (
                hidden_states,
                None,
                None,
                16,
                64,
                128,
                128,
                4,
            )
        )
        assert_close(self, actual_execution_time, 6.78e-5)

    def test_linear_attention_chunk_gated_delta_modeling(self):
        hidden_states = torch.randn(1, 65, 256, device="meta", dtype=torch.float16)
        actual_execution_time = self._execute_linear_attention_and_get_base_data(
            (hidden_states, None, None, 2, 4, 8, 16, 4)
        )
        assert_close(self, actual_execution_time, 5.53e-6)

    def test_linear_attention_decode_uses_recurrent_modeling(self):
        hidden_states = torch.randn(1, 1, 256, device="meta", dtype=torch.float16)
        actual_execution_time = self._execute_linear_attention_and_get_base_data(
            (hidden_states, None, None, 2, 4, 8, 16, 4)
        )
        assert_close(self, actual_execution_time, 5.0e-6, rtol=0.05)

    def test_linear_attn_chunk_rule_includes_scratch_memory_and_extra_static(self):
        batch_size, seq_len, num_heads, head_dim = 1, 65, 4, 16
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device="meta", dtype=torch.float16)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device="meta", dtype=torch.float16)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device="meta", dtype=torch.float16)
        beta = torch.randn(batch_size, seq_len, num_heads, device="meta", dtype=torch.float16)
        g = torch.randn(batch_size, seq_len, num_heads, device="meta", dtype=torch.float32)

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            torch.ops.tensor_cast.linear_attn_chunk_gated_delta_rule(query, key, value, beta, g, 64, 0, 1)

        self.assertEqual(len(runtime.event_list), 1)
        properties = runtime.event_list[0].op_invoke_info.get_perf_properties()
        self.assertGreater(properties.memory_readwrite_bytes, 0)
        self.assertGreater(properties.extra_static_cost_count, 0)

    def test_linear_attn_causal_conv_eager(self):
        batch_size, conv_dim, seq_len = 1, 1536, 8
        conv_kernel_size = 4
        mixed_qkv = torch.randn(batch_size, conv_dim, seq_len, device="meta", dtype=torch.float16)

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = torch.ops.tensor_cast.linear_attn_causal_conv(mixed_qkv, conv_kernel_size)

        self.assertEqual(out.shape, (batch_size, conv_dim, seq_len))
        self.assertEqual(len(runtime.event_list), 1)
        properties = runtime.event_list[0].op_invoke_info.get_perf_properties()
        self.assertGreater(properties.memory_read_bytes, 0)
        self.assertGreater(properties.memory_write_bytes, 0)
        # linear_attn_causal_conv does NOT include state memory
        self.assertEqual(properties.memory_readwrite_bytes, 0)

    def test_linear_attn_causal_conv_update_eager(self):
        batch_size, conv_dim, seq_len = 1, 1536, 1
        conv_kernel_size = 4
        mixed_qkv = torch.randn(batch_size, conv_dim, seq_len, device="meta", dtype=torch.float16)

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = torch.ops.tensor_cast.linear_attn_causal_conv_update(mixed_qkv, conv_kernel_size)

        self.assertEqual(out.shape, (batch_size, conv_dim, seq_len))
        self.assertEqual(len(runtime.event_list), 1)
        properties = runtime.event_list[0].op_invoke_info.get_perf_properties()
        self.assertGreater(properties.memory_read_bytes, 0)
        self.assertGreater(properties.memory_write_bytes, 0)
        # linear_attn_causal_conv_update includes state memory
        self.assertGreater(properties.memory_readwrite_bytes, 0)

    def test_linear_attn_apply_padding_mask_eager(self):
        batch_size, seq_len, hidden_size = 2, 8, 256
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="meta", dtype=torch.float16)
        attention_mask = torch.ones(batch_size, 1, 1, seq_len, device="meta")

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = torch.ops.tensor_cast.linear_attn_apply_padding_mask(hidden_states, attention_mask)

        self.assertEqual(out.shape, (batch_size, seq_len, hidden_size))
        self.assertEqual(len(runtime.event_list), 1)

    def test_linear_attn_fused_gdn_gating_eager(self):
        batch_size, seq_len, num_k_heads, head_k_dim = 1, 4, 4, 16
        num_v_heads = 4
        query = torch.randn(batch_size, seq_len, num_k_heads, head_k_dim, device="meta", dtype=torch.float16)
        key = torch.randn(batch_size, seq_len, num_k_heads, head_k_dim, device="meta", dtype=torch.float16)
        b = torch.randn(batch_size, seq_len, num_v_heads, device="meta", dtype=torch.float16)
        a = torch.randn(batch_size, seq_len, num_v_heads, device="meta", dtype=torch.float16)
        a_log = torch.randn(num_v_heads, device="meta", dtype=torch.float32)
        dt_bias = torch.randn(num_v_heads, device="meta", dtype=torch.float32)

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            query_out, key_out, beta, g = torch.ops.tensor_cast.linear_attn_fused_gdn_gating(
                query, key, b, a, a_log, dt_bias, num_v_heads
            )

        expected_shape = (batch_size, seq_len, num_v_heads, head_k_dim)
        self.assertEqual(query_out.shape, expected_shape)
        self.assertEqual(key_out.shape, expected_shape)
        self.assertEqual(beta.shape, (batch_size, seq_len, num_v_heads))
        self.assertEqual(g.shape, (batch_size, seq_len, num_v_heads))
        self.assertEqual(len(runtime.event_list), 1)

    def test_linear_attn_recurrent_gated_delta_rule_eager(self):
        batch_size, seq_len, num_heads, head_dim = 1, 1, 4, 16
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device="meta", dtype=torch.float16)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device="meta", dtype=torch.float16)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device="meta", dtype=torch.float16)
        beta = torch.randn(batch_size, seq_len, num_heads, device="meta", dtype=torch.float16)
        g = torch.randn(batch_size, seq_len, num_heads, device="meta", dtype=torch.float32)

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = torch.ops.tensor_cast.linear_attn_recurrent_gated_delta_rule(query, key, value, beta, g, 1, 1)

        self.assertEqual(out.shape, (batch_size, seq_len, num_heads, head_dim))
        self.assertEqual(len(runtime.event_list), 1)
        properties = runtime.event_list[0].op_invoke_info.get_perf_properties()
        # State memory read/write should be present
        self.assertGreater(properties.memory_read_bytes, 0)
        self.assertGreater(properties.memory_write_bytes, 0)

    def test_linear_attn_gated_rmsnorm_eager(self):
        batch_size, seq_len, num_heads, head_dim = 1, 8, 4, 16
        core_attn_out = torch.randn(batch_size, seq_len, num_heads, head_dim, device="meta", dtype=torch.float16)
        z = torch.randn(batch_size, seq_len, num_heads, head_dim, device="meta", dtype=torch.float16)
        weight = torch.randn(head_dim, device="meta", dtype=torch.float32)
        eps = 1e-6

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = torch.ops.tensor_cast.linear_attn_gated_rmsnorm(core_attn_out, z, weight, eps)

        self.assertEqual(out.shape, (batch_size, seq_len, num_heads, head_dim))
        self.assertEqual(len(runtime.event_list), 1)

    def test_extra_static_cost_count_combines(self):
        p1 = OpInvokeInfo.PerformanceProperties()
        p1.extra_static_cost_count = 3
        p2 = OpInvokeInfo.PerformanceProperties()
        p2.extra_static_cost_count = 5
        p1.combine(p2)
        self.assertEqual(p1.extra_static_cost_count, 8)

    def test_qwen3_5_linear_attention_with_padding_mask(self):
        user_config = UserInputConfig(
            model_id="Qwen/Qwen3.5-397B-A17B",
            tp_size=16,
            world_size=16,
            ep_size=16,
            do_compile=False,
            num_hidden_layers_override=1,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model = build_model(user_config)
        linear_attn = model.unwrap().language_model.layers[0].linear_attn
        hidden_states = torch.randn(1, 8, model.hidden_size, device="meta")
        attention_mask = torch.ones(1, 1, 1, 8, device="meta")

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = linear_attn(hidden_states, attention_mask=attention_mask)

        self.assertEqual(out.shape, hidden_states.shape)
        op_names = {str(event.op_invoke_info.func) for event in runtime.event_list}
        self.assertIn("tensor_cast.linear_attn_apply_padding_mask.default", op_names)
        self.assertIn("tensor_cast.linear_attn_chunk_gated_delta_rule.default", op_names)

    def test_qwen3_5_linear_attention_uses_local_tp_heads(self):
        user_config = UserInputConfig(
            model_id="Qwen/Qwen3.5-397B-A17B",
            tp_size=16,
            world_size=16,
            ep_size=16,
            do_compile=False,
            num_hidden_layers_override=1,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model = build_model(user_config)
        linear_attn = model.unwrap().language_model.layers[0].linear_attn
        hidden_states = torch.randn(1, 8, model.hidden_size, device="meta")

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = linear_attn(hidden_states)

        self.assertEqual(out.shape, hidden_states.shape)
        self.assertEqual(getattr(linear_attn, "tensor_cast_tp_size", None), 16)
        op_names = {str(event.op_invoke_info.func) for event in runtime.event_list}
        self.assertIn("tensor_cast.linear_attn_fused_gdn_gating.default", op_names)
        self.assertIn("tensor_cast.linear_attn_chunk_gated_delta_rule.default", op_names)

        hidden_states = torch.randn(1, 1, model.hidden_size, device="meta")
        cache_position = torch.tensor([8], dtype=torch.long, device="cpu")
        cache_position.tensor_cast_query_lens = (1,)
        cache_position.tensor_cast_is_decode = (True,)
        cache_position.tensor_cast_has_previous_state = True

        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = linear_attn(hidden_states, cache_position=cache_position)

        self.assertEqual(out.shape, hidden_states.shape)
        op_names = {str(event.op_invoke_info.func) for event in runtime.event_list}
        self.assertIn("tensor_cast.linear_attn_causal_conv_update.default", op_names)
        self.assertIn("tensor_cast.linear_attn_recurrent_gated_delta_rule.default", op_names)
        self.assertNotIn("tensor_cast.linear_attn_chunk_gated_delta_rule.default", op_names)

        decoder_layer = model.unwrap().language_model.layers[0]
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = decoder_layer(
                hidden_states,
                position_embeddings=None,
                cache_position=cache_position,
            )

        self.assertEqual(out.shape, hidden_states.shape)
        op_names = {str(event.op_invoke_info.func) for event in runtime.event_list}
        self.assertIn("tensor_cast.linear_attn_causal_conv_update.default", op_names)
        self.assertIn("tensor_cast.linear_attn_recurrent_gated_delta_rule.default", op_names)
        self.assertNotIn("tensor_cast.linear_attn_chunk_gated_delta_rule.default", op_names)

        hidden_states = torch.randn(1, 1, model.hidden_size, device="meta")
        cache_position = torch.tensor([8], dtype=torch.long, device="cpu")
        cache_position.tensor_cast_query_lens = (1,)
        cache_position.tensor_cast_is_decode = (True,)
        cache_position.tensor_cast_has_previous_state = True
        cache_position.tensor_cast_num_mtp_tokens = 3

        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = linear_attn(hidden_states, cache_position=cache_position)

        self.assertEqual(out.shape, hidden_states.shape)
        op_names = {str(event.op_invoke_info.func) for event in runtime.event_list}
        self.assertIn("tensor_cast.linear_attn_causal_conv_update.default", op_names)
        self.assertIn("tensor_cast.linear_attn_recurrent_gated_delta_rule.default", op_names)
        self.assertNotIn("tensor_cast.linear_attn_chunk_gated_delta_rule.default", op_names)

    def test_qwen3_5_linear_attention_w8a8_reuses_quant_linear(self):
        user_config = UserInputConfig(
            model_id="Qwen/Qwen3.5-397B-A17B",
            tp_size=16,
            world_size=16,
            ep_size=16,
            do_compile=False,
            num_hidden_layers_override=1,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
        )
        model = build_model(user_config)
        linear_attn = model.unwrap().language_model.layers[0].linear_attn
        hidden_states = torch.randn(1, 8, model.hidden_size, device="meta")

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            out = linear_attn(hidden_states)

        self.assertEqual(out.shape, hidden_states.shape)
        op_names = {str(event.op_invoke_info.func) for event in runtime.event_list}
        self.assertIn("tensor_cast.quantize.default", op_names)
        self.assertIn("tensor_cast.static_quant_linear.default", op_names)
        self.assertIn("tensor_cast.linear_attn_chunk_gated_delta_rule.default", op_names)
        self.assertNotIn("tensor_cast.linear_attention.default", op_names)

    def test_qwen3_5_linear_attention_rejects_invalid_tp_size(self):
        user_config = UserInputConfig(
            model_id="Qwen/Qwen3.5-397B-A17B",
            tp_size=32,
            world_size=32,
            ep_size=16,
            do_compile=False,
            num_hidden_layers_override=1,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )

        with self.assertRaises(ValueError) as cm:
            build_model(user_config)

        self.assertIn("num_k_heads=16", str(cm.exception))
        self.assertIn("tp_size=32", str(cm.exception))

    def test_qwen3_5_vision_tp_defaults_to_unsharded(self):
        user_config = UserInputConfig(
            model_id="Qwen/Qwen3.5-27B",
            tp_size=2,
            world_size=2,
            do_compile=False,
            num_hidden_layers_override=1,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )

        model = build_model(user_config)
        vision_attn = model.unwrap().visual.blocks[0].attn

        self.assertEqual(model.parallel_group_manager.vision_tp_group.world_size, 1)
        self.assertEqual(vision_attn.num_heads, 16)
        self.assertNotIsInstance(vision_attn.qkv, ColumnParallelLinear)

    def test_qwen3_5_vision_tp_can_be_enabled_explicitly(self):
        user_config = UserInputConfig(
            model_id="Qwen/Qwen3.5-27B",
            tp_size=2,
            vision_tp_size=2,
            world_size=2,
            do_compile=False,
            num_hidden_layers_override=1,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )

        model = build_model(user_config)
        vision_attn = model.unwrap().visual.blocks[0].attn

        self.assertEqual(model.parallel_group_manager.vision_tp_group.world_size, 2)
        self.assertEqual(vision_attn.num_heads, 8)
        self.assertIsInstance(vision_attn.qkv, ColumnParallelLinear)

    def test_qwen3_5_mtp_lm_head_uses_lmhead_tp_plan(self):
        user_config = UserInputConfig(
            model_id="Qwen/Qwen3.5-27B",
            tp_size=2,
            world_size=2,
            num_mtp_tokens=1,
            do_compile=False,
            num_hidden_layers_override=1,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )

        model = build_model(user_config)

        self.assertIsInstance(model._inner.mtp.lm_head, ColumnParallelLinear)
        self.assertEqual(model._inner.mtp.lm_head.tp_group.world_size, 2)

    def test_dsa_indexer_breakdown_helper_bf16(self):
        hidden_states = torch.randn(2, 3, 16, device="meta", dtype=torch.float16)
        qa_normed = torch.randn(2, 3, 4, device="meta", dtype=torch.float16)
        indexer_cache = torch.randn(2, 5, 7, device="meta", dtype=torch.float16)
        request_total_seq_lens = torch.tensor([3, 3], dtype=torch.long)

        breakdown = _estimate_dsa_indexer_breakdown(
            hidden_states,
            qa_normed,
            indexer_cache,
            num_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            topk_limit=5,
            request_total_seq_lens=request_total_seq_lens,
        )
        self.assertEqual(breakdown["q_proj_mma"], 768)
        self.assertEqual(breakdown["k_proj_mma"], 1536)
        self.assertEqual(breakdown["weights_proj_mma"], 384)
        self.assertEqual(breakdown["rope_gp"], 216)
        self.assertEqual(breakdown["rotate_activation_gp"], 0)
        self.assertEqual(breakdown["act_quant_gp"], 0)
        self.assertEqual(breakdown["qk_index_mma"], 576)
        self.assertEqual(breakdown["head_relu_gp"], 0)
        self.assertEqual(breakdown["head_q_scale_mul_gp"], 0)
        self.assertEqual(breakdown["head_weight_mul_gp"], 36)
        self.assertEqual(breakdown["head_reduce_gp"], 36)
        self.assertEqual(breakdown["head_k_scale_mul_gp"], 0)
        self.assertEqual(breakdown["topk_gp"], 18)
        assert_close(self, breakdown["historical_effective_read_bytes"], 252 / 0.15)
        self.assertEqual(breakdown["append_cache_write_bytes"], 84)
        self.assertEqual(breakdown["append_scale_write_bytes"], 0)
        self.assertNotIn("cache_rw_bytes", breakdown)
        self.assertNotIn("scale_cache_rw_bytes", breakdown)

    def test_dsa_indexer_breakdown_helper_uses_request_total_seq_lens_for_score_length(
        self,
    ):
        hidden_states = torch.randn(2, 3, 16, device="meta", dtype=torch.float16)
        qa_normed = torch.randn(2, 3, 4, device="meta", dtype=torch.float16)
        indexer_cache = torch.randn(2, 2, 7, device="meta", dtype=torch.float16)
        request_total_seq_lens = torch.tensor([5, 5], dtype=torch.long)

        breakdown = _estimate_dsa_indexer_breakdown(
            hidden_states,
            qa_normed,
            indexer_cache,
            num_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            topk_limit=5,
            request_total_seq_lens=request_total_seq_lens,
        )

        self.assertEqual(breakdown["qk_index_mma"], 960)
        self.assertEqual(breakdown["topk_gp"], 30)

    def test_dsa_indexer_breakdown_helper_uses_request_total_seq_lens_for_cache_traffic(
        self,
    ):
        hidden_states = torch.randn(2, 3, 16, device="meta", dtype=torch.float16)
        qa_normed = torch.randn(2, 3, 4, device="meta", dtype=torch.float16)
        indexer_cache = torch.randn(2, 2, 7, device="meta", dtype=torch.float16)
        request_total_seq_lens = torch.tensor([5, 5], dtype=torch.long)

        breakdown = _estimate_dsa_indexer_breakdown(
            hidden_states,
            qa_normed,
            indexer_cache,
            num_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            topk_limit=5,
            request_total_seq_lens=request_total_seq_lens,
            fp8_mode=True,
        )

        assert_close(self, breakdown["historical_effective_read_bytes"], 540 / 0.15)
        self.assertEqual(breakdown["append_cache_write_bytes"], 84)
        self.assertEqual(breakdown["append_scale_write_bytes"], 24)
        self.assertNotIn("cache_rw_bytes", breakdown)
        self.assertNotIn("scale_cache_rw_bytes", breakdown)

    def test_dsa_indexer_breakdown_helper_fp8(self):
        hidden_states = torch.randn(2, 3, 16, device="meta", dtype=torch.float16)
        qa_normed = torch.randn(2, 3, 4, device="meta", dtype=torch.float16)
        indexer_cache = torch.empty(2, 5, 7, device="meta", dtype=torch.float8_e4m3fn)
        request_total_seq_lens = torch.tensor([3, 3], dtype=torch.long)

        breakdown = _estimate_dsa_indexer_breakdown(
            hidden_states,
            qa_normed,
            indexer_cache,
            num_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            topk_limit=5,
            request_total_seq_lens=request_total_seq_lens,
            fp8_mode=True,
        )
        self.assertEqual(breakdown["q_proj_mma"], 768)
        self.assertEqual(breakdown["k_proj_mma"], 1536)
        self.assertEqual(breakdown["weights_proj_mma"], 384)
        self.assertEqual(breakdown["rope_gp"], 216)
        self.assertEqual(breakdown["rotate_activation_gp"], 144)
        self.assertEqual(breakdown["act_quant_gp"], 144)
        self.assertEqual(breakdown["qk_index_mma"], 576)
        self.assertEqual(breakdown["head_relu_gp"], 36)
        self.assertEqual(breakdown["head_q_scale_mul_gp"], 36)
        self.assertEqual(breakdown["head_weight_mul_gp"], 36)
        self.assertEqual(breakdown["head_reduce_gp"], 36)
        self.assertEqual(breakdown["head_k_scale_mul_gp"], 18)
        self.assertEqual(breakdown["topk_gp"], 18)
        assert_close(self, breakdown["historical_effective_read_bytes"], 198 / 0.15)
        self.assertEqual(breakdown["append_cache_write_bytes"], 42)
        self.assertEqual(breakdown["append_scale_write_bytes"], 24)
        self.assertNotIn("cache_rw_bytes", breakdown)
        self.assertNotIn("scale_cache_rw_bytes", breakdown)

    def test_mlapo_eager(self):
        num_tokens = 8192
        hidden_size = 7168
        dtype = torch.float16
        num_heads = 64
        qk_head_dim = 192
        qk_rope_head_dim = 64
        qk_nope_head_dim = qk_head_dim - qk_rope_head_dim
        kv_lora_rank = 512
        q_lora_rank = 1536

        hidden_states = torch.randn(num_tokens, hidden_size, device="meta", dtype=dtype)
        cos = torch.randn(1, num_tokens, qk_rope_head_dim, device="meta", dtype=dtype)
        sin = torch.randn(1, num_tokens, qk_rope_head_dim, device="meta", dtype=dtype)
        q_a_proj_weight = torch.randn(hidden_size, q_lora_rank, device="meta", dtype=dtype)
        q_a_layernorm_weight = torch.randn(q_lora_rank, device="meta", dtype=dtype)
        q_b_proj_weight = torch.randn(q_lora_rank, num_heads * qk_head_dim, device="meta", dtype=dtype)
        kv_a_proj_weight = torch.randn(hidden_size, kv_lora_rank + qk_rope_head_dim, device="meta", dtype=dtype)
        kv_a_layernorm_weight = torch.randn(kv_lora_rank + qk_rope_head_dim, device="meta", dtype=dtype)

        actual_execution_time = self._execute_mlapo_and_get_base_data(
            (
                hidden_states,
                cos,
                sin,
                q_a_proj_weight,
                q_a_layernorm_weight,
                q_b_proj_weight,
                kv_a_proj_weight,
                kv_a_layernorm_weight,
                num_heads,
                qk_head_dim,
                qk_nope_head_dim,
                qk_rope_head_dim,
                kv_lora_rank,
                q_lora_rank,
            )
        )

        assert_close(self, actual_execution_time, 2.3537e-3)

    def test_mlapo_quant(self):
        num_tokens = 8192
        hidden_size = 7168
        dtype = torch.float16
        quant_dtype = torch.int8
        num_heads = 64
        qk_head_dim = 192
        qk_rope_head_dim = 64
        qk_nope_head_dim = qk_head_dim - qk_rope_head_dim
        kv_lora_rank = 512
        q_lora_rank = 1536

        hidden_states = torch.randn(num_tokens, hidden_size, device="meta", dtype=dtype)
        cos = torch.randn(1, num_tokens, qk_rope_head_dim, device="meta", dtype=dtype)
        sin = torch.randn(1, num_tokens, qk_rope_head_dim, device="meta", dtype=dtype)
        q_a_proj_weight = torch.empty(hidden_size, q_lora_rank, device="meta", dtype=quant_dtype)
        q_a_layernorm_weight = torch.randn(q_lora_rank, device="meta", dtype=dtype)
        q_b_proj_weight = torch.empty(q_lora_rank, num_heads * qk_head_dim, device="meta", dtype=quant_dtype)
        kv_a_proj_weight = torch.empty(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            device="meta",
            dtype=quant_dtype,
        )
        kv_a_layernorm_weight = torch.randn(kv_lora_rank + qk_rope_head_dim, device="meta", dtype=dtype)

        q_a_proj_scale = torch.ones(q_lora_rank, device="meta")
        q_b_proj_scale = torch.ones(num_heads * qk_head_dim, device="meta")
        kv_a_proj_scale = torch.ones(kv_lora_rank + qk_rope_head_dim, device="meta")

        actual_execution_time = self._execute_mlapo_quant_and_get_base_data(
            (
                hidden_states,
                cos,
                sin,
                q_a_proj_weight,
                q_a_layernorm_weight,
                q_b_proj_weight,
                kv_a_proj_weight,
                kv_a_layernorm_weight,
                num_heads,
                qk_head_dim,
                qk_nope_head_dim,
                qk_rope_head_dim,
                kv_lora_rank,
                q_lora_rank,
                q_a_proj_scale,
                None,
                q_b_proj_scale,
                None,
                kv_a_proj_scale,
                None,
            )
        )

        assert_close(self, actual_execution_time, 1.2502e-3)

    def test_moe_gating_top_k_softmax(
        self,
    ):
        """Tests the execution time of the `moe_gating_top_k_softmax` operation under AnalyticPerformanceModel.

        Given input logits and a top-k value, executes the operation and verifies that
        the analytic execution time is sufficiently close to the expected value (2.0e-6 seconds).
        """
        perf_model = AnalyticPerformanceModel(TEST_DEVICE)
        test_logits = torch.randn(1, 4, 4, device="meta", dtype=torch.float16)
        top_k = 2
        expected_shape = (*test_logits.shape[:-1], top_k)
        with (
            Runtime(perf_model, TEST_DEVICE, memory_tracker=MemoryTracker(TEST_DEVICE)) as runtime,
            torch.no_grad(),
        ):
            topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k_softmax(test_logits, top_k)
            self.assertEqual(topk_weights.shape, expected_shape)
            self.assertEqual(topk_indices.shape, expected_shape)
        self.assertEqual(len(runtime.event_list), 1)
        analytic_result = runtime.event_list[0].perf_results.get("analytic")
        actual_execution_time = analytic_result.execution_time_s
        assert_close(self, actual_execution_time, 2.0e-6)

    def test_mla_eager_prefill_without_context(self):
        B, S, num_heads, q_head_dim = 2, 3500, 8, 192
        block_size, dtype = 128, torch.float16
        kv_lora_rank, qk_rope_head_dim = 512, 64
        query_len = 3500
        qk_nope_head_dim = q_head_dim - qk_rope_head_dim
        total_tokens = B * query_len
        topk_limit = 1
        v_head_dim = 128

        q = torch.randn(total_tokens, num_heads, q_head_dim, device="meta", dtype=dtype)
        max_num_blocks_per_seq = (S + block_size - 1) // block_size
        num_blocks = B * max_num_blocks_per_seq
        kv_cache = torch.randn(
            num_blocks,
            block_size,
            kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
            device="meta",
        )
        request_total_seq_lens = torch.full((B,), S, dtype=torch.long, device="cpu")
        query_lens = torch.full((B,), query_len, dtype=torch.long, device="cpu")
        W_UK_T = torch.randn(num_heads, qk_nope_head_dim, kv_lora_rank, device="meta", dtype=dtype)
        W_UV = torch.randn(num_heads, kv_lora_rank, v_head_dim, device="meta", dtype=dtype)
        kv_b_proj = torch.randn(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            device="meta",
            dtype=dtype,
        )

        actual_execution_time = self._execute_multihead_latent_attention_and_get_base_data(
            (
                q,
                kv_cache,
                None,
                None,
                request_total_seq_lens,
                query_lens,
                W_UK_T,
                W_UV,
                kv_b_proj,
                v_head_dim,
                topk_limit,
            )
        )

        assert_close(self, actual_execution_time, 6.443208610547408e-05)

    def test_mla_eager_prefill_with_context(self):
        B, S, num_heads, q_head_dim = 2, 7008, 8, 192
        block_size, dtype = 128, torch.float16
        kv_lora_rank, qk_rope_head_dim = 512, 64
        query_len = 3500
        qk_nope_head_dim = q_head_dim - qk_rope_head_dim
        total_tokens = B * query_len
        topk_limit = 1
        v_head_dim = 128

        q = torch.randn(total_tokens, num_heads, q_head_dim, device="meta", dtype=dtype)
        max_num_blocks_per_seq = (S + block_size - 1) // block_size
        num_blocks = B * max_num_blocks_per_seq
        kv_cache = torch.randn(
            num_blocks,
            block_size,
            kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
            device="meta",
        )
        request_total_seq_lens = torch.full((B,), S, dtype=torch.long, device="cpu")
        query_lens = torch.full((B,), query_len, dtype=torch.long, device="cpu")
        W_UK_T = torch.randn(num_heads, qk_nope_head_dim, kv_lora_rank, device="meta", dtype=dtype)
        W_UV = torch.randn(num_heads, kv_lora_rank, v_head_dim, device="meta", dtype=dtype)
        kv_b_proj = torch.randn(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            device="meta",
            dtype=dtype,
        )

        actual_execution_time = self._execute_multihead_latent_attention_and_get_base_data(
            (
                q,
                kv_cache,
                None,
                None,
                request_total_seq_lens,
                query_lens,
                W_UK_T,
                W_UV,
                kv_b_proj,
                v_head_dim,
                topk_limit,
            )
        )

        assert_close(self, actual_execution_time, 6.443208610547408e-05)

    def test_mla_eager_decode(self):
        B, S, num_heads, q_head_dim = 16, 7008, 8, 192
        block_size, dtype = 128, torch.float16
        kv_lora_rank, qk_rope_head_dim = 512, 64
        query_len = 1
        qk_nope_head_dim = q_head_dim - qk_rope_head_dim
        total_tokens = B * query_len
        topk_limit = 1
        v_head_dim = 128

        q = torch.randn(total_tokens, num_heads, q_head_dim, device="meta", dtype=dtype)
        max_num_blocks_per_seq = (S + block_size - 1) // block_size
        num_blocks = B * max_num_blocks_per_seq
        kv_cache = torch.randn(
            num_blocks,
            block_size,
            kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
            device="meta",
        )
        request_total_seq_lens = torch.full((B,), S, dtype=torch.long, device="cpu")
        query_lens = torch.full((B,), query_len, dtype=torch.long, device="cpu")
        W_UK_T = torch.randn(num_heads, qk_nope_head_dim, kv_lora_rank, device="meta", dtype=dtype)
        W_UV = torch.randn(num_heads, kv_lora_rank, v_head_dim, device="meta", dtype=dtype)
        kv_b_proj = torch.randn(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            device="meta",
            dtype=dtype,
        )

        actual_execution_time = self._execute_multihead_latent_attention_and_get_base_data(
            (
                q,
                kv_cache,
                None,
                None,
                request_total_seq_lens,
                query_lens,
                W_UK_T,
                W_UV,
                kv_b_proj,
                topk_limit,
                v_head_dim,
            )
        )

        assert_close(self, actual_execution_time, 0.00015605324564501644)

    def _run_test_model(self, model_id, do_compile):
        num_tokens = 100
        user_config = UserInputConfig(model_id=model_id, do_compile=do_compile, num_hidden_layers_override=2)
        model = self._get_model(user_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(model, model.model_config)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        self.assertIn("tensor_cast.", runtime.table_averages())

    def _run_test_deepseek(self, model_id, do_compile):
        user_config = UserInputConfig(
            model_id=model_id,
            do_compile=do_compile,
            enable_dispatch_ffn_combine=False,
        )
        model = self._get_model(user_config)
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model.model_config)
        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.init_routing_v2", result)
        self.assertIn("tensor_cast.concat_and_cache_mla", result)
        self.assertIn("tensor_cast.multihead_latent_attention", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-235B-A22B", False],
            ["zai-org/GLM-4.5", False],
        ]
    )
    def test_model(self, model_id, do_compile):
        self._run_test_model(model_id, do_compile)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
        ]
    )
    def test_deepseek(self, model_id, do_compile):
        self._run_test_deepseek(model_id, do_compile)

    def test_table_averages_default(self):
        def func(x):
            return x + 2 * x + x

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            x = torch.randn([100], device="meta")
            _ = func(x)
        result = runtime.table_averages()
        self.assertIn("analytic total", result)
        self.assertIn("analytic avg", result)
        self.assertIn("aten.randn", result)
        self.assertIn("aten.add", result)
        self.assertIn("aten.mul", result)
        self.assertIn("# of Calls", result)

    def test_table_averages_group_by_shape(self):
        def func(x, y):
            return x + 2 * x + x + y

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            x = torch.randn([10, 10], device="meta")
            y = torch.randn([10, 1], device="meta")
            _ = func(x, y)
        result = runtime.table_averages(group_by_input_shapes=True)
        self.assertIn("analytic total", result)
        self.assertIn("analytic avg", result)
        self.assertIn("Input Shapes", result)
        self.assertIn("aten.randn", result)
        self.assertIn("aten.add", result)
        self.assertIn("aten.mul", result)
        self.assertIn("# of Calls", result)

    def test_table_averages_splits_same_op_by_dominant_bound(self):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        runtime = Runtime(perf_model, device_profile)
        x = torch.randn([10, 10], device="meta")
        op_info = OpInvokeInfo(torch.ops.aten.add.Tensor, (x, x), {}, x)
        runtime.event_list = [
            RuntimeEvent(
                op_invoke_info=op_info,
                perf_results={
                    "analytic": PerformanceModel.Result(
                        execution_time_s=2e-6,
                        statistics={
                            StatsKey.MEMORY_ACCESS: 2e-6,
                            StatsKey.COMPUTE: 1e-6,
                            StatsKey.MMA_OPS: 1e-6,
                            StatsKey.GP_OPS: 0.0,
                        },
                    )
                },
            ),
            RuntimeEvent(
                op_invoke_info=op_info,
                perf_results={
                    "analytic": PerformanceModel.Result(
                        execution_time_s=3e-6,
                        statistics={
                            StatsKey.MEMORY_ACCESS: 1e-6,
                            StatsKey.COMPUTE: 3e-6,
                            StatsKey.MMA_OPS: 3e-6,
                            StatsKey.GP_OPS: 0.0,
                        },
                    )
                },
            ),
        ]

        result = runtime.table_averages(dump_op_bound_results=True)

        self.assertIn("Bound (analytic)", result)
        self.assertIn("memory_bound", result)
        self.assertIn("compute_bound_mma", result)
        self.assertEqual(result.count("aten.add.Tensor"), 2)

    def test_table_averages_does_not_group_by_bound_by_default(self):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        runtime = Runtime(perf_model, device_profile)
        x = torch.randn([10, 10], device="meta")
        op_info = OpInvokeInfo(torch.ops.aten.add.Tensor, (x, x), {}, x)
        runtime.event_list = [
            RuntimeEvent(
                op_invoke_info=op_info,
                perf_results={
                    "analytic": PerformanceModel.Result(
                        execution_time_s=2e-6,
                        statistics={
                            StatsKey.MEMORY_ACCESS: 2e-6,
                            StatsKey.COMPUTE: 1e-6,
                            StatsKey.MMA_OPS: 1e-6,
                            StatsKey.GP_OPS: 0.0,
                        },
                    )
                },
            ),
            RuntimeEvent(
                op_invoke_info=op_info,
                perf_results={
                    "analytic": PerformanceModel.Result(
                        execution_time_s=3e-6,
                        statistics={
                            StatsKey.MEMORY_ACCESS: 1e-6,
                            StatsKey.COMPUTE: 3e-6,
                            StatsKey.MMA_OPS: 3e-6,
                            StatsKey.GP_OPS: 0.0,
                        },
                    )
                },
            ),
        ]

        result = runtime.table_averages()

        self.assertNotIn("Bound (analytic)", result)
        self.assertNotIn("memory_bound", result)
        self.assertNotIn("compute_bound_mma", result)
        self.assertEqual(result.count("aten.add.Tensor"), 1)

    def test_table_averages_dump_op_bound_ratios(self):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        runtime = Runtime(perf_model, device_profile)
        x = torch.randn([10, 10], device="meta")
        runtime.event_list = [
            RuntimeEvent(
                op_invoke_info=OpInvokeInfo(torch.ops.aten.mm.default, (x, x), {}, x),
                perf_results={
                    "analytic": PerformanceModel.Result(
                        execution_time_s=4e-6,
                        statistics={
                            StatsKey.MEMORY_ACCESS: 1e-6,
                            StatsKey.COMMUNICATION: 1e-6,
                            StatsKey.COMPUTE: 2e-6,
                            StatsKey.MMA_OPS: 2e-6,
                            StatsKey.GP_OPS: 0.0,
                        },
                    )
                },
            )
        ]

        result = runtime.table_averages(dump_op_bound_results=True)

        self.assertIn("analytic memory %", result)
        self.assertIn("analytic comm %", result)
        self.assertIn("analytic mma %", result)
        self.assertIn("analytic gp %", result)
        mm_lines = [line for line in result.splitlines() if "aten.mm.default" in line]
        self.assertEqual(len(mm_lines), 1)
        self.assertRegex(mm_lines[0], r"25\.00%\s+25\.00%\s+50\.00%\s+0\.00%")

    def test_table_averages_uses_compute_first_bound_semantics(self):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        runtime = Runtime(perf_model, device_profile)
        x = torch.randn([10, 10], device="meta")
        runtime.event_list = [
            RuntimeEvent(
                op_invoke_info=OpInvokeInfo(torch.ops.aten.mm.default, (x, x), {}, x),
                perf_results={
                    "analytic": PerformanceModel.Result(
                        execution_time_s=10e-6,
                        statistics={
                            StatsKey.MEMORY_ACCESS: 5e-6,
                            StatsKey.COMMUNICATION: 1e-6,
                            StatsKey.COMPUTE: 10e-6,
                            StatsKey.MMA_OPS: 1e-6,
                            StatsKey.GP_OPS: 2e-6,
                        },
                    )
                },
            )
        ]

        result = runtime.table_averages(dump_op_bound_results=True)

        self.assertIn("compute_bound_gp", result)
        self.assertNotIn("memory_bound", result)

    def test_runtime_and_op_bound_classifier_share_bound_semantics(self):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        runtime = Runtime(perf_model, device_profile)
        x = torch.randn([10, 10], device="meta")
        result = PerformanceModel.Result(
            execution_time_s=10e-6,
            statistics={
                StatsKey.MEMORY_ACCESS: 5e-6,
                StatsKey.COMMUNICATION: 1e-6,
                StatsKey.COMPUTE: 10e-6,
                StatsKey.MMA_OPS: 1e-6,
                StatsKey.GP_OPS: 2e-6,
            },
        )
        op_info = OpInvokeInfo(torch.ops.aten.mm.default, (x, x), {}, x)
        runtime.event_list = [RuntimeEvent(op_invoke_info=op_info, perf_results={"analytic": result})]

        table_result = runtime.table_averages(dump_op_bound_results=True)
        classifier_result = OpBoundClassifier().classify([(op_info, result)])

        self.assertIn("compute_bound_gp", table_result)
        self.assertEqual(classifier_result["memory_bound"], 0)
        self.assertEqual(classifier_result["communication_bound"], 0)
        self.assertEqual(classifier_result["compute_bound_mma"], 1e-6)
        self.assertEqual(classifier_result["compute_bound_gp"], 2e-6)

    def test_table_averages_bound_fallback_for_incomplete_estimator_fields(self):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        runtime = Runtime(perf_model, device_profile)
        x = torch.randn([10, 10], device="meta")
        runtime.event_list = [
            RuntimeEvent(
                op_invoke_info=OpInvokeInfo(torch.ops.tensor_cast.dispatch_ffn_combine.default, (x,), {}, x),
                perf_results={
                    "analytic": PerformanceModel.Result(
                        execution_time_s=3e-6,
                        statistics={
                            StatsKey.COMPUTE: 3e-6,
                            StatsKey.MEMORY_ACCESS: 1e-6,
                            StatsKey.COMMUNICATION: 0.0,
                        },
                    )
                },
            )
        ]

        result = runtime.table_averages(dump_op_bound_results=True)

        self.assertIn("compute_bound_mma", result)
        self.assertIn("75.00%", result)

    def test_bound_analyzer_collects_flat_prefixed_stats(self):
        result = PerformanceModel.Result(
            execution_time_s=4e-6,
            statistics={
                "matmul.mma_ops_time_s": 3e-6,
                "matmul.gp_ops_time_s": 0.0,
                "all_reduce.comm_time_s": 1e-6,
            },
        )

        components = BoundAnalyzer.components(result)

        self.assertEqual(components.mma_ops_time_s, 3e-6)
        self.assertEqual(components.gp_ops_time_s, 0.0)
        self.assertEqual(components.communication_time_s, 1e-6)
        self.assertEqual(BoundAnalyzer.dominant(result), "compute_bound_mma")

    def test_bound_analyzer_collects_nested_stats(self):
        result = PerformanceModel.Result(
            execution_time_s=4.5e-6,
            statistics={
                "matmul": {
                    "mma_ops_time_s": 3e-6,
                    "gp_ops_time_s": 0.5e-6,
                },
                "all_reduce": {
                    "comm_time_s": 1e-6,
                },
            },
        )

        components = BoundAnalyzer.components(result)

        self.assertEqual(components.mma_ops_time_s, 3e-6)
        self.assertEqual(components.gp_ops_time_s, 0.5e-6)
        self.assertEqual(components.communication_time_s, 1e-6)
        self.assertEqual(BoundAnalyzer.dominant(result), "compute_bound_mma")

    def test_bound_analyzer_falls_back_compute_time_to_mma(self):
        result = PerformanceModel.Result(
            execution_time_s=3e-6,
            statistics={
                StatsKey.COMPUTE: 3e-6,
                StatsKey.MEMORY_ACCESS: 1e-6,
            },
        )

        components = BoundAnalyzer.components(result)

        self.assertEqual(components.memory_time_s, 1e-6)
        self.assertEqual(components.mma_ops_time_s, 3e-6)
        self.assertEqual(components.gp_ops_time_s, 0.0)
        self.assertEqual(BoundAnalyzer.dominant(result), "compute_bound_mma")

    def test_table_averages_bound_fallback_for_prefixed_estimator_fields(self):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        runtime = Runtime(perf_model, device_profile)
        x = torch.randn([10, 10], device="meta")
        runtime.event_list = [
            RuntimeEvent(
                op_invoke_info=OpInvokeInfo(torch.ops.tensor_cast.matmul_all_reduce.default, (x,), {}, x),
                perf_results={
                    "analytic": PerformanceModel.Result(
                        execution_time_s=4e-6,
                        statistics={
                            "matmul.mma_ops_time_s": 3e-6,
                            "matmul.gp_ops_time_s": 0.0,
                            "all_reduce.comm_time_s": 1e-6,
                            StatsKey.MEMORY_ACCESS: 0.0,
                        },
                    )
                },
            )
        ]

        result = runtime.table_averages(dump_op_bound_results=True)

        self.assertIn("compute_bound_mma", result)
        self.assertIn("75.00%", result)

    def test_export_chrome_trace(self):
        def func(x):
            return x + 2 * x + x

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            x = torch.randn([100], device="meta")
            _ = func(x)
        with tempfile.TemporaryFile(mode="w+") as temp_file:
            runtime.export_chrome_trace(temp_file)
            temp_file.seek(0)
            content = temp_file.read()
            self.assertIn("aten.randn", content)
            self.assertIn("aten.add", content)
            self.assertIn("aten.mul", content)

    def test_model_cost_with_noop_self_copy(self):
        x = torch.randn([16], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
            torch.ops.aten.copy_.default(x, x)
        self.assertEqual(len(runtime.event_list), 1)
        self.assertEqual(runtime.total_execution_time_s()[perf_model.name], 0)
        self.assertIn("aten.copy_.default", runtime.table_averages())

    def test_model_cost_with_non_noop_copy(self):
        dst = torch.randn([16], device="meta")
        src = torch.randn([16], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
            torch.ops.aten.copy_.default(dst, src)
        self.assertEqual(len(runtime.event_list), 1)
        self.assertGreater(runtime.total_execution_time_s()[perf_model.name], 0)
        self.assertIn("aten.copy_.default", runtime.table_averages())

    def test_multistream_total_execution_time_critical_path(self):
        def func(x):
            c0 = torch.ops.tensor_cast._internal_wait_and_bind.default(x, 0, [])
            a = torch.ops.aten.relu.default(c0)
            _ = torch.ops.tensor_cast._internal_record.default(a, 0)

            c1 = torch.ops.tensor_cast._internal_wait_and_bind.default(x, 1, [])
            b = torch.ops.aten.sigmoid.default(c1)
            token_b = torch.ops.tensor_cast._internal_record.default(b, 1)

            c2 = torch.ops.tensor_cast._internal_wait_and_bind.default(a, 0, [token_b])
            out = torch.ops.aten.tanh.default(c2)
            _ = torch.ops.tensor_cast._internal_record.default(out, 0)
            return out

        durations_s = {
            torch.ops.aten.relu.default: 3.0,
            torch.ops.aten.sigmoid.default: 5.0,
            torch.ops.aten.tanh.default: 2.0,
        }
        perf_model = Mock(spec=PerformanceModel)
        perf_model.name = "fixed"
        perf_model.device_profile = TEST_DEVICE
        perf_model.get_classifiers.return_value = []

        def _fixed_duration_process_op(op_invoke_info):
            return PerformanceModel.Result(execution_time_s=durations_s.get(op_invoke_info.func, 0.0))

        perf_model.process_op.side_effect = _fixed_duration_process_op
        x = torch.randn([8, 8], device="meta")
        with Runtime(perf_model, TEST_DEVICE) as runtime, torch.no_grad():
            _ = func(x)

        # Serial sum is 10s, but critical path is 7s:
        # max(relu=3s on stream0, sigmoid=5s on stream1) + tanh=2s (depends on sigmoid).
        total_time_s = runtime.total_execution_time_s()[perf_model.name]
        assert_close(self, total_time_s, 7.0)
        tracked_events = [event for event in runtime.event_list if event.op_invoke_info.func in durations_s]
        self.assertEqual(len(tracked_events), 3)
        self.assertEqual([event.stream_id for event in tracked_events], [0, 1, 0])

    def test_multistream_anchors_do_not_inflate_memory_tracking(self):
        x = torch.randn([8, 8], device="meta")
        y = torch.ops.aten.neg.default(x)
        token = torch.empty((), dtype=torch.int64, device="meta")
        plain_runtime = Runtime([], TEST_DEVICE, memory_tracker=MemoryTracker(TEST_DEVICE))
        plain_runtime.op_info_group = [
            OpInvokeInfo(torch.ops.aten.neg.default, (x,), {}, y),
        ]
        plain_runtime.replay_op_invoke_infos()
        plain_runtime.memory_tracker.analyze()

        anchored_runtime = Runtime([], TEST_DEVICE, memory_tracker=MemoryTracker(TEST_DEVICE))
        anchored_runtime.op_info_group = [
            OpInvokeInfo(torch.ops.tensor_cast._internal_wait_and_bind.default, (x, 0, []), {}, x),
            OpInvokeInfo(torch.ops.aten.neg.default, (x,), {}, y),
            OpInvokeInfo(torch.ops.tensor_cast._internal_record.default, (y, 0), {}, token),
        ]
        anchored_runtime.replay_op_invoke_infos()
        anchored_runtime.memory_tracker.analyze()

        self.assertEqual(
            anchored_runtime.memory_tracker.peak_mem_usage(),
            plain_runtime.memory_tracker.peak_mem_usage(),
        )
        self.assertEqual(
            len(anchored_runtime.memory_tracker.get_profile()),
            len(plain_runtime.memory_tracker.get_profile()),
        )

    def test_model_cost_with_view(self):
        def func(x):
            return x.reshape(10, 10)

        x = torch.randn([100], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model,
                device_profile,
            ) as runtime,
            torch.no_grad(),
        ):
            _ = func(x)
        self.assertEqual(runtime.total_execution_time_s()[perf_model.name], 0)

    def test_model_cost_with_zero_shape_matmul(self):
        def func(x, y):
            return torch.matmul(x, y)

        x = torch.randn([0, 10], device="meta")
        y = torch.randn([10, 10], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model,
                device_profile,
            ) as runtime,
            torch.no_grad(),
        ):
            _ = func(x, y)
        self.assertEqual(runtime.total_execution_time_s()[perf_model.name], 0)

    def test_model_cost_with_zero_shape_batched_matmul(self):
        def func(x, y):
            return torch.matmul(x, y)

        x = torch.randn([0, 10, 10], device="meta")
        y = torch.randn([10, 10], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model,
                device_profile,
            ) as runtime,
            torch.no_grad(),
        ):
            _ = func(x, y)
        self.assertEqual(runtime.total_execution_time_s()[perf_model.name], 0)

    def test_model_cost_with_zero_shape_conv1d(self):
        def func(x, y):
            return torch.nn.functional.conv1d(x, y)

        x = torch.randn([0, 3, 32], device="meta")
        y = torch.randn([16, 3, 3], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model,
                device_profile,
            ) as runtime,
            torch.no_grad(),
        ):
            _ = func(x, y)
        self.assertEqual(runtime.total_execution_time_s()[perf_model.name], 0)

    def test_model_cost_with_zero_shape_conv2d(self):
        def func(x, y):
            return torch.nn.functional.conv2d(x, y)

        x = torch.randn([0, 3, 32, 32], device="meta")
        y = torch.randn([16, 3, 3, 3], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model,
                device_profile,
            ) as runtime,
            torch.no_grad(),
        ):
            _ = func(x, y)
        self.assertEqual(runtime.total_execution_time_s()[perf_model.name], 0)

    def test_model_cost_with_zero_shape_conv3d(self):
        def func(x, y):
            return torch.nn.functional.conv3d(x, y)

        x = torch.randn([0, 3, 8, 32, 32], device="meta")
        y = torch.randn([16, 3, 3, 3, 3], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model,
                device_profile,
            ) as runtime,
            torch.no_grad(),
        ):
            _ = func(x, y)
        self.assertEqual(runtime.total_execution_time_s()[perf_model.name], 0)

    def test_model_cost_with_zero_shape_addmm(self):
        def func(input_tensor, mat1, mat2):
            return torch.addmm(input_tensor, mat1, mat2)

        input_tensor = torch.randn([0, 10], device="meta")
        mat1 = torch.randn([0, 5], device="meta")
        mat2 = torch.randn([5, 10], device="meta")

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)

        with (
            Runtime(
                perf_model,
                device_profile,
            ) as runtime,
            torch.no_grad(),
        ):
            _ = func(input_tensor, mat1, mat2)
        self.assertEqual(runtime.total_execution_time_s()[perf_model.name], 0)

    # deprecated: migrated to test_dtype.py::test_zero_shape_static_quant_linear_keeps_shape
    def test_model_cost_with_zero_shape_static_quant_linear(self):
        def func(x, w, w_scale):
            return torch.ops.tensor_cast.static_quant_linear(
                x,
                w,
                w_scale,
                w_offset=None,
                x_scale=None,
                x_offset=None,
                bias=None,
                out_dtype=None,
            )

        x = torch.randn([0, 10], device="meta")
        w = torch.randint(0, 255, [10, 10], dtype=torch.uint8, device="meta")
        w_scale = torch.randn([10], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model,
                device_profile,
            ) as runtime,
            torch.no_grad(),
        ):
            _ = func(x, w, w_scale)
        self.assertEqual(runtime.total_execution_time_s()[perf_model.name], 0)

    def test_runtime_breakdown_compute_bound(self):
        def func(x, y):
            return torch.matmul(x, y)

        x = torch.randn([1000, 1000], device="meta")
        y = torch.randn([1000, 1000], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
            func(x, y)
        breakdowns = runtime.get_breakdowns()
        self.assertGreater(len(breakdowns), 0)
        self.assertTrue(any(key.endswith("OpBound") for key in breakdowns.keys()))
        for key, breakdown in breakdowns.items():
            if key.endswith("OpBound"):
                self.assertGreater(breakdown["compute_bound_mma"], 0)
                self.assertEqual(breakdown["compute_bound_gp"], 0)
                self.assertEqual(breakdown["memory_bound"], 0)
                self.assertEqual(breakdown["communication_bound"], 0)

    def test_runtime_breakdown_memory_bound(self):
        def func(x, y):
            return torch.add(x, y)

        x = torch.randn([1000, 1000], device="meta")
        y = torch.randn([1000, 1000], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
            func(x, y)
        breakdowns = runtime.get_breakdowns()
        self.assertGreater(len(breakdowns), 0)
        self.assertTrue(any(key.endswith("OpBound") for key in breakdowns.keys()))
        for key, breakdown in breakdowns.items():
            if key.endswith("OpBound"):
                self.assertEqual(breakdown["compute_bound_mma"], 0)
                self.assertEqual(breakdown["compute_bound_gp"], 0)
                self.assertGreater(breakdown["memory_bound"], 0)
                self.assertEqual(breakdown["communication_bound"], 0)

    def test_runtime_breakdown_comm_bound(self):
        def func(x):
            return torch.ops.tensor_cast.all_reduce(x, 0, [0, 1])

        x = torch.randn([1000, 1000], device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
            func(x)
        breakdowns = runtime.get_breakdowns()
        self.assertGreater(len(breakdowns), 0)
        self.assertTrue(any(key.endswith("OpBound") for key in breakdowns.keys()))
        for key, breakdown in breakdowns.items():
            if key.endswith("OpBound"):
                self.assertEqual(breakdown["compute_bound_mma"], 0)
                self.assertEqual(breakdown["compute_bound_gp"], 0)
                self.assertEqual(breakdown["memory_bound"], 0)
                self.assertGreater(breakdown["communication_bound"], 0)

    def test_empirical_model_torch_op(self):
        def func(x, y):
            return torch.matmul(x, y)

        x = torch.randn([100, 100], device="meta")
        y = torch.randn([100, 100], device="meta")
        device_profile = TEST_DEVICE

        # Configure mock data source to return a result
        query_result = Mock(spec=QueryResult)
        query_result.latency_us = 100.0
        query_result.confidence = 0.95
        query_result.source = QuerySource.MEASURED
        query_result.details = {"kernel_type": "MatMulV2"}
        query_result.shape_debug_statistics.return_value = {}
        self.data_source.lookup.return_value = query_result

        perf_model = EmpiricalPerformanceModel(device_profile, self.data_source, self.fallback_model)
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
            func(x, y)
        total_time_s = runtime.total_execution_time_s()[perf_model.name]
        self.assertGreater(total_time_s, 0)
        result = runtime.table_averages()
        self.assertIn("aten.mm.default", result)

    def test_empirical_model_torch_op_view(self):
        def func(x):
            return x.reshape(10, 10)

        x = torch.randn([100], device="meta")
        device_profile = TEST_DEVICE

        # Configure mock data source to return None (cache miss) to use fallback
        self.data_source.lookup.return_value = None

        # Configure fallback model to return a result with execution_time_s = 0
        fallback_result = Mock()
        fallback_result.execution_time_s = 0
        self.fallback_model.process_op.return_value = fallback_result

        perf_model = EmpiricalPerformanceModel(device_profile, self.data_source, self.fallback_model)
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
            func(x)
        total_time_s = runtime.total_execution_time_s()[perf_model.name]
        self.assertEqual(total_time_s, 0)
        result = runtime.table_averages()
        self.assertIn("aten.view.default", result)

    def test_empirical_model_tensorcast_op(self):
        # test tensor_cast.quantize
        def func(x, scale):
            return torch.ops.tensor_cast.quantize(x, scale, None, torch.int8)

        x = torch.randn([100, 100], device="meta")
        scale = torch.tensor(0.1, device="meta")
        device_profile = TEST_DEVICE

        # Configure mock data source to return a result
        query_result = Mock(spec=QueryResult)
        query_result.latency_us = 50.0
        query_result.confidence = 0.95
        query_result.source = QuerySource.MEASURED
        query_result.details = {"kernel_type": "AscendQuantV2"}
        query_result.shape_debug_statistics.return_value = {}
        self.data_source.lookup.return_value = query_result

        perf_model = EmpiricalPerformanceModel(device_profile, self.data_source, self.fallback_model)
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
            func(x, scale)
        total_time_s = runtime.total_execution_time_s()[perf_model.name]
        self.assertGreater(total_time_s, 0)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.quantize.default", result)


@pytest.mark.nightly
class PerfAnalysisNightlyTestCase(PerfAnalysisTestMixin, unittest.TestCase):
    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model(self, model_id):
        PerfAnalysisTestCase._run_test_model(self, model_id, True)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
            ["moonshotai/Kimi-K2-Base"],
        ]
    )
    def test_deepseek(self, model_id):
        PerfAnalysisTestCase._run_test_deepseek(self, model_id, True)

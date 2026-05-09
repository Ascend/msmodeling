import tempfile
import unittest
from unittest.mock import Mock

import torch
from parameterized import parameterized

from tensor_cast.compilation import get_backend
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model import _estimate_dsa_indexer_breakdown
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.empirical import EmpiricalPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.performance_model.profiling_database.data_source import (
    DataSourcePerformanceModel,
    QueryResult,
    QuerySource,
)
from tensor_cast.runtime import Runtime
from .test_common import (
    assert_close,
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    has_submodule_with_cls_name,
)


class PerfAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.data_source = Mock(spec=DataSourcePerformanceModel)
        self.fallback_model = Mock(spec=PerformanceModel)
        # Configure fallback to return a valid Result for M5 latency tracking
        fallback_result = Mock()
        fallback_result.execution_time_s = 1e-6
        self.fallback_model.process_op.return_value = fallback_result
        torch.compiler.reset()

    def _execute_attention_and_get_base_data(self, attention_args):
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
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
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
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
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
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
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
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
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            torch.ops.tensor_cast.mlapo_quant(*mlapo_args)
        self.assertEqual(len(runtime.event_list), 1)
        analytic_result = runtime.event_list[0].perf_results.get("analytic")
        actual_execution_time = analytic_result.execution_time_s
        return actual_execution_time

    def test_simple_model_eager(self):
        def func(x):
            return x + x

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
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
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            x = torch.randn([100], device="meta")
            _ = func(x)
        self.assertEqual(len(runtime.event_list), 3)

    def test_attention_dit_eager(self):
        B, S, num_heads, head_dim = 2, 256, 6, 64
        dtype = torch.float16

        q = torch.randn(B, S, num_heads, head_dim, device="meta", dtype=dtype)
        k = torch.randn(B, S, num_heads, head_dim, device="meta", dtype=dtype)
        v = torch.randn(B, S, num_heads, head_dim, device="meta", dtype=dtype)

        actual_execution_time = self._execute_attention_and_get_base_data(
            (q, k, v, None, None, None, None, None)
        )

        assert_close(self, actual_execution_time, 6.49e-6)

    def test_attention_llm_eager(self):
        B, S, num_kv_heads, head_dim = 2, 256, 8, 64
        block_size, dtype = 128, torch.float16
        hidden_size, query_len = num_kv_heads * head_dim, 1
        total_tokens = B * query_len

        q = torch.randn(total_tokens, hidden_size, device="meta", dtype=dtype)
        max_num_blocks_per_seq = (S + block_size - 1) // block_size
        num_blocks = B * max_num_blocks_per_seq
        k = torch.randn(
            num_blocks, block_size, num_kv_heads, head_dim, device="meta", dtype=dtype
        )
        v = torch.randn(
            num_blocks, block_size, num_kv_heads, head_dim, device="meta", dtype=dtype
        )
        block_table = torch.empty(
            (B, max_num_blocks_per_seq), dtype=torch.long, device="meta"
        )
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

    def test_qwen3_5_linear_attention_uses_local_tp_heads(self):
        user_config = UserInputConfig(
            model_id="Qwen/Qwen3.5-397B-A17B",
            tp_size=16,
            world_size=16,
            ep_size=16,
            do_compile=False,
            num_hidden_layers_override=1,
        )
        model = build_model(user_config)
        linear_attn = model.unwrap().language_model.layers[0].linear_attn
        hidden_states = torch.randn(1, 8, model.hidden_size, device="meta")

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            out = linear_attn(hidden_states)

        self.assertEqual(out.shape, hidden_states.shape)
        self.assertEqual(getattr(linear_attn, "tensor_cast_tp_size", None), 16)
        self.assertEqual(runtime.event_list[0].op_invoke_info.args[3], 1)
        self.assertEqual(runtime.event_list[0].op_invoke_info.args[4], 4)

    def test_qwen3_5_linear_attention_rejects_invalid_tp_size(self):
        user_config = UserInputConfig(
            model_id="Qwen/Qwen3.5-397B-A17B",
            tp_size=32,
            world_size=32,
            ep_size=16,
            do_compile=False,
            num_hidden_layers_override=1,
        )

        with self.assertRaises(ValueError) as cm:
            build_model(user_config)

        self.assertIn("num_k_heads=16", str(cm.exception))
        self.assertIn("tp_size=32", str(cm.exception))

    def test_dsa_indexer_breakdown_helper_bf16(self):
        hidden_states = torch.randn(2, 3, 16, device="meta", dtype=torch.float16)
        qa_normed = torch.randn(2, 3, 4, device="meta", dtype=torch.float16)
        indexer_cache = torch.randn(2, 5, 7, device="meta", dtype=torch.float16)

        breakdown = _estimate_dsa_indexer_breakdown(
            hidden_states,
            qa_normed,
            indexer_cache,
            num_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            topk_limit=5,
        )
        self.assertEqual(
            breakdown,
            {
                "q_proj_mma": 768,
                "k_proj_mma": 1536,
                "weights_proj_mma": 384,
                "rope_gp": 216,
                "rotate_activation_gp": 0,
                "act_quant_gp": 0,
                "qk_index_mma": 576,
                "head_relu_gp": 0,
                "head_q_scale_mul_gp": 0,
                "head_weight_mul_gp": 36,
                "head_reduce_gp": 36,
                "head_k_scale_mul_gp": 0,
                "topk_gp": 18,
                "cache_rw_bytes": 140,
                "scale_cache_rw_bytes": 0,
            },
        )

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

        self.assertEqual(breakdown["cache_rw_bytes"], 140)
        self.assertEqual(breakdown["scale_cache_rw_bytes"], 40)

    def test_dsa_indexer_breakdown_helper_fp8(self):
        hidden_states = torch.randn(2, 3, 16, device="meta", dtype=torch.float16)
        qa_normed = torch.randn(2, 3, 4, device="meta", dtype=torch.float16)
        indexer_cache = torch.empty(2, 5, 7, device="meta", dtype=torch.float8_e4m3fn)

        breakdown = _estimate_dsa_indexer_breakdown(
            hidden_states,
            qa_normed,
            indexer_cache,
            num_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            topk_limit=5,
            fp8_mode=True,
        )
        self.assertEqual(
            breakdown,
            {
                "q_proj_mma": 768,
                "k_proj_mma": 1536,
                "weights_proj_mma": 384,
                "rope_gp": 216,
                "rotate_activation_gp": 144,
                "act_quant_gp": 144,
                "qk_index_mma": 576,
                "head_relu_gp": 36,
                "head_q_scale_mul_gp": 36,
                "head_weight_mul_gp": 36,
                "head_reduce_gp": 36,
                "head_k_scale_mul_gp": 18,
                "topk_gp": 18,
                "cache_rw_bytes": 70,
                "scale_cache_rw_bytes": 40,
            },
        )

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
        q_a_proj_weight = torch.randn(
            hidden_size, q_lora_rank, device="meta", dtype=dtype
        )
        q_a_layernorm_weight = torch.randn(q_lora_rank, device="meta", dtype=dtype)
        q_b_proj_weight = torch.randn(
            q_lora_rank, num_heads * qk_head_dim, device="meta", dtype=dtype
        )
        kv_a_proj_weight = torch.randn(
            hidden_size, kv_lora_rank + qk_rope_head_dim, device="meta", dtype=dtype
        )
        kv_a_layernorm_weight = torch.randn(
            kv_lora_rank + qk_rope_head_dim, device="meta", dtype=dtype
        )

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

        assert_close(self, actual_execution_time, 2.28e-3)

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
        q_a_proj_weight = torch.empty(
            hidden_size, q_lora_rank, device="meta", dtype=quant_dtype
        )
        q_a_layernorm_weight = torch.randn(q_lora_rank, device="meta", dtype=dtype)
        q_b_proj_weight = torch.empty(
            q_lora_rank, num_heads * qk_head_dim, device="meta", dtype=quant_dtype
        )
        kv_a_proj_weight = torch.empty(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            device="meta",
            dtype=quant_dtype,
        )
        kv_a_layernorm_weight = torch.randn(
            kv_lora_rank + qk_rope_head_dim, device="meta", dtype=dtype
        )

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

        assert_close(self, actual_execution_time, 1.18e-3)

    def test_moe_gating_top_k_softmax(
        self,
    ):
        """
        Tests the execution time of the `moe_gating_top_k_softmax` operation under AnalyticPerformanceModel.

        Given input logits and a top-k value, executes the operation and verifies that
        the analytic execution time is sufficiently close to the expected value (2.0e-6 seconds).
        """
        perf_model = AnalyticPerformanceModel(TEST_DEVICE)
        test_logits = torch.randn(1, 4, 4, device="meta", dtype=torch.float16)
        top_k = 2
        expected_shape = (*test_logits.shape[:-1], top_k)
        with (
            Runtime(
                perf_model, TEST_DEVICE, memory_tracker=MemoryTracker(TEST_DEVICE)
            ) as runtime,
            torch.no_grad(),
        ):
            topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k_softmax(
                test_logits, top_k
            )
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
        W_UK_T = torch.randn(
            num_heads, qk_nope_head_dim, kv_lora_rank, device="meta", dtype=dtype
        )
        W_UV = torch.randn(
            num_heads, kv_lora_rank, v_head_dim, device="meta", dtype=dtype
        )
        kv_b_proj = torch.randn(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            device="meta",
            dtype=dtype,
        )

        actual_execution_time = (
            self._execute_multihead_latent_attention_and_get_base_data(
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
        W_UK_T = torch.randn(
            num_heads, qk_nope_head_dim, kv_lora_rank, device="meta", dtype=dtype
        )
        W_UV = torch.randn(
            num_heads, kv_lora_rank, v_head_dim, device="meta", dtype=dtype
        )
        kv_b_proj = torch.randn(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            device="meta",
            dtype=dtype,
        )

        actual_execution_time = (
            self._execute_multihead_latent_attention_and_get_base_data(
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
        W_UK_T = torch.randn(
            num_heads, qk_nope_head_dim, kv_lora_rank, device="meta", dtype=dtype
        )
        W_UV = torch.randn(
            num_heads, kv_lora_rank, v_head_dim, device="meta", dtype=dtype
        )
        kv_b_proj = torch.randn(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            device="meta",
            dtype=dtype,
        )

        actual_execution_time = (
            self._execute_multihead_latent_attention_and_get_base_data(
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
        )

        assert_close(self, actual_execution_time, 9.26e-6)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3-235B-A22B", False],
            ["Qwen/Qwen3-235B-A22B", True],
            ["zai-org/GLM-4.5", False],
            ["zai-org/GLM-4.5", True],
        ]
    )
    def test_model(self, model_id, do_compile):
        num_tokens = 100
        user_config = UserInputConfig(
            model_id=model_id, do_compile=do_compile, num_hidden_layers_override=2
        )
        model = build_model(user_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(
            model, model.model_config
        )
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
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

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            ["moonshotai/Kimi-K2-Base", True],
        ]
    )
    def test_deepseek(self, model_id, do_compile):
        user_config = UserInputConfig(model_id=model_id, do_compile=do_compile)
        model = build_model(user_config)
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(
            model, model.model_config
        )
        # make sure all original attention modules have been replaced
        self.assertTrue(
            has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast")
        )
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
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

    def test_table_averages_default(self):
        def func(x):
            return x + 2 * x + x

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
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
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
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

    def test_export_chrome_trace(self):
        def func(x):
            return x + 2 * x + x

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
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
            return PerformanceModel.Result(
                execution_time_s=durations_s.get(op_invoke_info.func, 0.0)
            )

        perf_model.process_op.side_effect = _fixed_duration_process_op
        x = torch.randn([8, 8], device="meta")
        with Runtime(perf_model, TEST_DEVICE) as runtime, torch.no_grad():
            _ = func(x)

        # Serial sum is 10s, but critical path is 7s:
        # max(relu=3s on stream0, sigmoid=5s on stream1) + tanh=2s (depends on sigmoid).
        total_time_s = runtime.total_execution_time_s()[perf_model.name]
        assert_close(self, total_time_s, 7.0)
        tracked_events = [
            event
            for event in runtime.event_list
            if event.op_invoke_info.func in durations_s
        ]
        self.assertEqual(len(tracked_events), 3)
        self.assertEqual([event.stream_id for event in tracked_events], [0, 1, 0])

    def test_multistream_anchors_do_not_inflate_memory_tracking(self):
        x = torch.randn([8, 8], device="meta")
        y = torch.ops.aten.neg.default(x)
        token = torch.empty((), dtype=torch.int64, device="meta")
        plain_runtime = Runtime(
            [], TEST_DEVICE, memory_tracker=MemoryTracker(TEST_DEVICE)
        )
        plain_runtime.op_info_group = [
            OpInvokeInfo(torch.ops.aten.neg.default, (x,), {}, y),
        ]
        plain_runtime.replay_op_invoke_infos()
        plain_runtime.memory_tracker.analyze()

        anchored_runtime = Runtime(
            [], TEST_DEVICE, memory_tracker=MemoryTracker(TEST_DEVICE)
        )
        anchored_runtime.op_info_group = [
            OpInvokeInfo(
                torch.ops.tensor_cast._internal_wait_and_bind.default, (x, 0, []), {}, x
            ),
            OpInvokeInfo(torch.ops.aten.neg.default, (x,), {}, y),
            OpInvokeInfo(
                torch.ops.tensor_cast._internal_record.default, (y, 0), {}, token
            ),
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

        perf_model = EmpiricalPerformanceModel(
            device_profile, self.data_source, self.fallback_model
        )
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

        perf_model = EmpiricalPerformanceModel(
            device_profile, self.data_source, self.fallback_model
        )
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

        perf_model = EmpiricalPerformanceModel(
            device_profile, self.data_source, self.fallback_model
        )
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
            func(x, scale)
        total_time_s = runtime.total_execution_time_s()[perf_model.name]
        self.assertGreater(total_time_s, 0)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.quantize.default", result)

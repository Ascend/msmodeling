import unittest
from itertools import product

import torch
from parameterized import parameterized
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import (
    TEST_DEVICE,
    CommGrid,
    DeviceProfile,
    InterconnectTopology,
)
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.model_config import (
    AttentionQuantConfig,
    ModelConfig,
    MultiheadLatentAttentionQuantConfig,
    ParallelConfig,
    QuantConfig,
)
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel, StatsKey
from tensor_cast.quantize_utils import AttentionQuantType, LinearQuantType
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.custom_model_registry import get_moe_config, get_mtp_block_module_name
from tensor_cast.transformers.model import TransformerModel
from tensor_cast.utils import DTYPE_FP8, is_fp8_dtype, performance_dtype

from .conftest import get_session_hf_config
from .test_common import (
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    get_cached_build_model,
    has_submodule_with_cls_name,
)

# Core attention quantization configuration assertions were moved to the unified entry in test_ops.py.


def get_quant_config(
    start_layer_id=-1,
    end_layer_id=-1,
    attn_quant_type: AttentionQuantType = AttentionQuantType.INT8,
):
    quant_config = QuantConfig()
    config = AttentionQuantConfig(
        quant_type=attn_quant_type,
        query_scale=torch.tensor(1.0),
        kv_scale=torch.tensor(1.0),
        attention_prob_scale=torch.tensor(1.0),
    )
    if start_layer_id == -1 or end_layer_id == -1:
        quant_config.attention_configs[-1] = config
    for i in range(start_layer_id, end_layer_id):
        quant_config.attention_configs[i] = config
    return quant_config


def get_mla_quant_config(start_layer_id=-1, end_layer_id=-1):
    from .test_common import get_quant_config as get_quant_config_common

    quant_config = get_quant_config_common(quant_type=LinearQuantType.W8A8)
    config = MultiheadLatentAttentionQuantConfig(
        quant_type=AttentionQuantType.INT8,
        query_scale=torch.tensor(1.0),
        kv_scale=torch.tensor(1.0),
        attention_prob_scale=torch.tensor(1.0),
        kv_projected_scale=torch.tensor(1.0),
        qk_scale=torch.tensor(1.0),
        v_scale=torch.tensor(1.0),
        out_scale=torch.tensor(1.0),
    )
    if start_layer_id == -1 or end_layer_id == -1:
        quant_config.attention_configs[-1] = config
    for i in range(start_layer_id, end_layer_id):
        quant_config.attention_configs[i] = config
    return quant_config


class TestQuantAttention(unittest.TestCase):
    QUANT_TYPES = [AttentionQuantType.INT8, AttentionQuantType.FP8]

    @classmethod
    def setUpClass(cls):
        cls._model_cache = {}
        cls._transformer_cache = {}

    @classmethod
    def _get_transformer_model(cls, model_id: str, model_config: ModelConfig) -> TransformerModel:
        key = (model_id, repr(model_config))
        if key not in cls._transformer_cache:
            cls._transformer_cache[key] = TransformerModel(model_id, model_config)
        return cls._transformer_cache[key]

    def test_all_torch_float8_dtypes_share_fp8_performance_dtype(self):
        fp8_dtypes = [
            dtype for name, dtype in vars(torch).items() if name.startswith("float8") and isinstance(dtype, torch.dtype)
        ]
        self.assertGreater(len(fp8_dtypes), 0)
        for dtype in fp8_dtypes:
            self.assertTrue(is_fp8_dtype(dtype))
            self.assertEqual(performance_dtype(dtype), DTYPE_FP8)

    def assert_mma_ops_time_positive(self, runtime, op_name):
        total_mma_ops_time_s = 0
        for event in runtime.event_list:
            if op_name not in str(event.op_invoke_info.func):
                continue
            for result in event.perf_results.values():
                total_mma_ops_time_s += result.statistics.get(StatsKey.MMA_OPS, 0)
        self.assertGreater(total_mma_ops_time_s, 0)

    def test_fp8_mla_quant_mma_ops_time_is_nonzero(self):
        q = torch.empty((2, 2, 16), dtype=torch.float8_e4m3fn, device="meta")
        kv_cache = torch.empty((2, 32, 12), dtype=torch.float8_e4m3fn, device="meta")
        block_table = torch.empty((2, 1), dtype=torch.int32, device="meta")
        query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
        request_total_seq_lens = torch.tensor([32, 32], dtype=torch.int32)
        query_lens = torch.tensor([1, 1], dtype=torch.int32)
        W_UK_T = torch.empty((2, 12, 8), dtype=torch.bfloat16, device="meta")
        W_UV = torch.empty((2, 8, 16), dtype=torch.bfloat16, device="meta")
        scale = torch.tensor(1.0)

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            torch.ops.tensor_cast.multihead_latent_attention_quant(
                q,
                kv_cache,
                block_table,
                query_start_loc,
                request_total_seq_lens,
                query_lens,
                W_UK_T,
                W_UV,
                None,
                16,
                None,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                torch.bfloat16,
            )

        self.assert_mma_ops_time_positive(runtime, "multihead_latent_attention_quant.default")

    def test_fp8_mla_quant_uses_custom_device_fp8_variant(self):
        device = DeviceProfile(
            name=f"TEST_DEVICE_CUSTOM_FP8_E5M2_{id(self)}",
            vendor="TEST_VENDOR",
            mma_ops={torch.float8_e5m2: 100 * 1e12},
            gp_ops={torch.float32: 10 * 1e12, torch.half: 10 * 1e12},
            memory_size_bytes=64 * (1024**3),
            memory_bandwidth_bytes_ps=1.6 * (1024**4),
            comm_grid=CommGrid(
                grid=torch.arange(2),
                topologies={0: InterconnectTopology(1e9, 1e-6)},
            ),
        )
        self.assertEqual(device.mma_ops, {DTYPE_FP8: 100 * 1e12})

        q = torch.empty((2, 2, 16), dtype=torch.float8_e4m3fn, device="meta")
        kv_cache = torch.empty((2, 32, 12), dtype=torch.float8_e4m3fn, device="meta")
        block_table = torch.empty((2, 1), dtype=torch.int32, device="meta")
        query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
        request_total_seq_lens = torch.tensor([32, 32], dtype=torch.int32)
        query_lens = torch.tensor([1, 1], dtype=torch.int32)
        W_UK_T = torch.empty((2, 12, 8), dtype=torch.bfloat16, device="meta")
        W_UV = torch.empty((2, 8, 16), dtype=torch.bfloat16, device="meta")
        scale = torch.tensor(1.0)

        perf_model = AnalyticPerformanceModel(device)
        with Runtime(perf_model, device) as runtime, torch.no_grad():
            torch.ops.tensor_cast.multihead_latent_attention_quant(
                q,
                kv_cache,
                block_table,
                query_start_loc,
                request_total_seq_lens,
                query_lens,
                W_UK_T,
                W_UV,
                None,
                16,
                None,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                torch.bfloat16,
            )

        self.assert_mma_ops_time_positive(runtime, "multihead_latent_attention_quant.default")

    def test_device_profile_rejects_mismatched_fp8_perf_values(self):
        with self.assertRaisesRegex(ValueError, "FP8 variants must share the same performance value"):
            DeviceProfile(
                name=f"TEST_DEVICE_FP8_CONFLICT_{id(self)}",
                vendor="TEST_VENDOR",
                mma_ops={
                    torch.float8_e4m3fn: 100 * 1e12,
                    torch.float8_e5m2: 120 * 1e12,
                },
                gp_ops={torch.float32: 10 * 1e12},
                memory_size_bytes=64 * (1024**3),
                memory_bandwidth_bytes_ps=1.6 * (1024**4),
                comm_grid=CommGrid(
                    grid=torch.arange(2),
                    topologies={0: InterconnectTopology(1e9, 1e-6)},
                ),
            )

    @parameterized.expand(
        list(
            product(
                ["Qwen/Qwen3-32B", "Qwen/Qwen3-235B-A22B", "zai-org/GLM-4.5"],
                QUANT_TYPES,
            )
        )
    )
    def test_standard_attention(self, model_id, attn_quant_type):
        kv_quant_start_idx = 0
        kv_quant_end_idx = 1
        hf_config = get_session_hf_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(kv_quant_start_idx, kv_quant_end_idx, attn_quant_type),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=2,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        model = self._get_transformer_model(model_id, model_config)
        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(model, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self.assertIn("quantize.default", result)
        self.assertIn("reshape_and_cache.default", result)
        self.assertIn("attention_quant.default", result)
        self.assert_mma_ops_time_positive(runtime, "attention_quant.default")

    @parameterized.expand(list(product(["deepseek-ai/DeepSeek-V3.1"], QUANT_TYPES)))
    def test_mla(self, model_id, attn_quant_type):
        num_mtp_layers = 1
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            quantize_attention_action=attn_quant_type,
        )

        model = get_cached_build_model(self._model_cache, user_config)

        mtp_block_module_name = get_mtp_block_module_name(model.model_config.hf_config.model_type)
        self.assertIsNotNone(mtp_block_module_name)
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model.model_config)
        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(),
            )
            self.assertEqual(outputs.shape, (1, num_mtp_layers + 1))
        result = runtime.table_averages()
        self.assertIn("quantize.default", result)
        self.assertIn("concat_and_cache_mla.default", result)
        self.assertIn("multihead_latent_attention_quant.default", result)
        self.assert_mma_ops_time_positive(runtime, "multihead_latent_attention_quant.default")

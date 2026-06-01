import unittest

import pytest
import torch
from parameterized import parameterized
from tensor_cast.compilation import get_backend
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import MlaConfig, ModelConfig, ParallelConfig, QuantConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity, QuantScheme
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.custom_model_registry import get_moe_config
from tensor_cast.transformers.model import TransformerModel

from .conftest import get_session_hf_config
from .test_common import create_mla_metadata_and_kv_cache
from .test_quant_linear import get_quant_config

# Core parallel-topology assertions were moved to test_layers.py::test_parallel_config_topology_flags.


def get_parallel_config(parallel_configuration: tuple):
    parallel_config = ParallelConfig(
        world_size=parallel_configuration[0],
        tensor_parallel_size=parallel_configuration[1],
        o_proj_tensor_parallel_size=parallel_configuration[2],
        mlp_tensor_parallel_size=parallel_configuration[3],
        lmhead_tensor_parallel_size=parallel_configuration[4],
    )
    if len(parallel_configuration) > 5:
        parallel_config.embedding_parallel = parallel_configuration[5]
    return parallel_config


def has_dp_transform(parallel_config: ParallelConfig):
    if parallel_config.data_parallel_size != parallel_config.mlp_data_parallel_size:
        return True
    if parallel_config.data_parallel_size != parallel_config.o_proj_data_parallel_size:
        return True
    if parallel_config.data_parallel_size != parallel_config.lmhead_data_parallel_size:
        return True
    return False


class ParallelLinearTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._model_cache = {}

    @classmethod
    def _get_hf_config(cls, model_id: str):
        return get_session_hf_config(model_id)

    @classmethod
    def _get_transformer_model(cls, model_id: str, model_config: ModelConfig) -> TransformerModel:
        key = (model_id, repr(model_config))
        if key not in cls._model_cache:
            cls._model_cache[key] = TransformerModel(model_id, model_config)
        return cls._model_cache[key]

    def setUp(self):
        num_tokens = 100
        self.input_batch_size = 2
        self.compile_backend = get_backend()
        with torch.device("meta"):
            self.inputs = torch.empty([self.input_batch_size, num_tokens], dtype=torch.long)
            self.position_ids = torch.empty([self.input_batch_size, num_tokens], dtype=torch.long)

    def _check_comm_analytic(self, trace_events, comm_op_name):
        count = 0
        for event in trace_events:
            if event["name"] == comm_op_name:
                self.assertIn("message_size_bytes", event["args"])
                count += 1
        self.assertGreater(count, 0)

    def _validate_comm_result(self, result: str, runtime: Runtime, parallel_config: ParallelConfig):
        comm_op_name = "tensor_cast.all_reduce.default"
        if parallel_config.has_attn_tp() or parallel_config.has_o_proj_tp() or parallel_config.has_mlp_tp():
            self.assertIn(comm_op_name, result)
            self._check_comm_analytic(runtime.get_trace_events(), comm_op_name)
        else:
            self.assertNotIn(comm_op_name, result)

        comm_op_name = "tensor_cast.all_gather.default"
        if parallel_config.has_lmhead_tp() or has_dp_transform(parallel_config):
            self.assertIn(comm_op_name, result)
            self._check_comm_analytic(runtime.get_trace_events(), comm_op_name)
        else:
            self.assertNotIn(comm_op_name, result)

    def _run_test_model_with_tp_and_dp(self, model_id, parallel_configuration):
        hf_config = self._get_hf_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        parallel_config = get_parallel_config(parallel_configuration)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            enable_repetition=True,
            hf_config=hf_config,
            moe_config=moe_config,
        )
        model = self._get_transformer_model(model_id, model_config)

        num_tokens = 100
        output_batch_size = self.input_batch_size
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (output_batch_size, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self._validate_comm_result(result, runtime, parallel_config)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", (16, 1, 1, 1, 1)],
            ["Qwen/Qwen3-32B", (16, 8, 8, 8, 8)],
            ["Qwen/Qwen3-32B", (16, 16, 16, 16, 16)],
            ["Qwen/Qwen3-32B", (16, 4, 2, 8, 16)],
            ["Qwen/Qwen3-32B", (16, 4, 2, 8, 16, True)],
        ]
    )
    def test_model_with_tp_and_dp(self, model_id, parallel_configuration):
        self._run_test_model_with_tp_and_dp(model_id, parallel_configuration)

    def _run_test_deepseek_with_tp_and_dp(self, model_id, parallel_configuration):
        hf_config = self._get_hf_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        parallel_config = get_parallel_config(parallel_configuration)

        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            enable_repetition=True,
            moe_config=moe_config,
            hf_config=hf_config,
            trust_remote_code=False,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        model = self._get_transformer_model(model_id, model_config)

        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with (
            Runtime(perf_model, machine_config) as runtime,
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
        self._validate_comm_result(result, runtime, parallel_config)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, 4, 2, 8, 16)],
            ["deepseek-ai/DeepSeek-V3.1", (16, 4, 2, 8, 16, True)],
        ]
    )
    def test_deepseek_with_tp_and_dp(self, model_id, parallel_configuration):
        self._run_test_deepseek_with_tp_and_dp(model_id, parallel_configuration)

    def _run_test_model_quant_with_tp_and_dp(self, model_id, parallel_configuration):
        parallel_config = get_parallel_config(parallel_configuration)
        hf_config = self._get_hf_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            enable_repetition=True,
            moe_config=moe_config,
            hf_config=hf_config,
            trust_remote_code=False,
        )
        model = self._get_transformer_model(model_id, model_config)

        model_config.quant_config = get_quant_config(model.unwrap(), quant_type=LinearQuantType.W4A8)
        model_config.quant_linear_cls = TensorCastQuantLinear
        qmodel = self._get_transformer_model(model_id, model_config)

        num_tokens = 100
        output_batch_size = self.input_batch_size
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = qmodel.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (output_batch_size, num_tokens, qmodel.vocab_size))
        result = runtime.table_averages()

        self._validate_comm_result(result, runtime, parallel_config)
        self.assertTrue("tensor_cast.dynamic_quantize_symmetric.default" in result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", (16, 1, 1, 1, 1)],
            ["Qwen/Qwen3-32B", (16, 8, 8, 8, 8)],
            ["Qwen/Qwen3-32B", (16, 16, 16, 16, 16)],
            ["Qwen/Qwen3-32B", (16, 4, 2, 8, 16)],
            ["Qwen/Qwen3-32B", (16, 4, 2, 8, 16, True)],
        ]
    )
    def test_model_quant_with_tp_and_dp(self, model_id, parallel_configuration):
        self._run_test_model_quant_with_tp_and_dp(model_id, parallel_configuration)

    def _deepseek_quant_with_tp_and_dp(self, model_id, parallel_configuration):
        hf_config = self._get_hf_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        parallel_config = get_parallel_config(parallel_configuration)

        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            enable_repetition=True,
            moe_config=moe_config,
            hf_config=hf_config,
            trust_remote_code=False,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        model = self._get_transformer_model(model_id, model_config)

        model_config.quant_config = get_quant_config(model.unwrap(), quant_type=LinearQuantType.W4A8)
        model_config.quant_linear_cls = TensorCastQuantLinear
        qmodel = self._get_transformer_model(model_id, model_config)

        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(qmodel, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with (
            Runtime(perf_model, machine_config) as runtime,
            torch.no_grad(),
        ):
            outputs = qmodel.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.vocab_size))

        result = runtime.table_averages()
        self._validate_comm_result(result, runtime, parallel_config)
        self.assertTrue("tensor_cast.dynamic_quantize_symmetric.default" in result)

    def _run_test_model_quant_mxfp4_with_tp_and_dp(self, model_id, parallel_configuration):
        hf_config = self._get_hf_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        parallel_config = get_parallel_config(parallel_configuration)
        mxfp4_quant_config = get_quant_config(
            quant_type=LinearQuantType.MXFP4,
            weight_group_size=32,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            weight_quant_scheme=QuantScheme.SYMMETRIC,
        )
        model_config_with_mxfp4 = ModelConfig(
            parallel_config,
            mxfp4_quant_config,
            attention_cls=AttentionTensorCast,
            enable_repetition=True,
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
            moe_config=moe_config,
            hf_config=hf_config,
            trust_remote_code=False,
        )
        qmodel = self._get_transformer_model(model_id, model_config_with_mxfp4)

        num_tokens = 100
        output_batch_size = self.input_batch_size
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with (
            Runtime(perf_model, machine_config) as runtime,
            torch.no_grad(),
        ):
            outputs = qmodel.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (output_batch_size, num_tokens, qmodel.vocab_size))
        result = runtime.table_averages()

        self._validate_comm_result(result, runtime, parallel_config)

        self.assertIn("tensor_cast.dynamic_quantize_mxfp4.default", result)
        self.assertIn("tensor_cast.mxfp4_linear.default", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", (16, 8, 8, 8, 8)],
            ["Qwen/Qwen3-32B", (16, 4, 2, 8, 16)],
            ["Qwen/Qwen3-32B", (16, 4, 2, 8, 16, True)],
        ]
    )
    def test_model_quant_mxfp4_with_tp_and_dp(self, model_id, parallel_configuration):
        self._run_test_model_quant_mxfp4_with_tp_and_dp(model_id, parallel_configuration)


@pytest.mark.nightly
class ParallelLinearNightlyTestCase(unittest.TestCase):
    _get_hf_config = ParallelLinearTestCase._get_hf_config
    _get_transformer_model = ParallelLinearTestCase._get_transformer_model
    _check_comm_analytic = ParallelLinearTestCase._check_comm_analytic
    _validate_comm_result = ParallelLinearTestCase._validate_comm_result
    _deepseek_quant_with_tp_and_dp = ParallelLinearTestCase._deepseek_quant_with_tp_and_dp

    @classmethod
    def setUpClass(cls):
        ParallelLinearTestCase.setUpClass()
        cls._model_cache = {}

    def setUp(self):
        ParallelLinearTestCase.setUp(self)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, 4, 2, 8, 16)],
            ["deepseek-ai/DeepSeek-V3.1", (16, 4, 2, 8, 16, True)],
            ["moonshotai/Kimi-K2-Base", (16, 4, 2, 8, 16)],
        ]
    )
    def test_deepseek_quant_with_tp_and_dp(self, model_id, parallel_configuration):
        ParallelLinearTestCase._deepseek_quant_with_tp_and_dp(self, model_id, parallel_configuration)

    @parameterized.expand(
        [
            ["zai-org/GLM-4.5", (16, 4, 2, 8, 16)],
        ]
    )
    def test_model_quant_with_tp_and_dp_glm(self, model_id, parallel_configuration):
        ParallelLinearTestCase._run_test_model_quant_with_tp_and_dp(self, model_id, parallel_configuration)

    @parameterized.expand(
        [
            ["moonshotai/Kimi-K2-Base", (16, 4, 2, 8, 16)],
        ]
    )
    def test_deepseek_with_tp_and_dp_kimi(self, model_id, parallel_configuration):
        ParallelLinearTestCase._run_test_deepseek_with_tp_and_dp(self, model_id, parallel_configuration)

    @parameterized.expand(
        [
            ["zai-org/GLM-4.5", (16, 4, 2, 8, 16)],
        ]
    )
    def test_model_with_tp_and_dp_glm(self, model_id, parallel_configuration):
        ParallelLinearTestCase._run_test_model_with_tp_and_dp(self, model_id, parallel_configuration)

    @parameterized.expand(
        [
            ["zai-org/GLM-4.5", (16, 4, 2, 8, 16)],
        ]
    )
    def test_model_quant_mxfp4_with_tp_and_dp_glm(self, model_id, parallel_configuration):
        ParallelLinearTestCase._run_test_model_quant_mxfp4_with_tp_and_dp(self, model_id, parallel_configuration)

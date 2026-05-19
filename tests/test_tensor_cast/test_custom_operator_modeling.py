import unittest

import torch

from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.model_config import ModelConfig, ParallelConfig, QuantConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.performance_model.op_estimator_registry import _op_estimator_table, register_op_estimator
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.custom_model_registry import get_moe_config
from tensor_cast.transformers.model import TransformerModel
from tensor_cast.transformers.utils import AutoModelConfigLoader
from .test_common import create_attn_metadata_and_kv_cache


def _restore_op_properties_functor(op, original):
    if original is not None:
        OpInvokeInfo._op_properties_functors[op] = original
    else:
        OpInvokeInfo._op_properties_functors.pop(op, None)


def _restore_op_estimator(device_name, op, original):
    if original is not None:
        _op_estimator_table.setdefault(device_name, {})[op] = original
    else:
        _op_estimator_table.get(device_name, {}).pop(op, None)


class CustomModelingOperatorTestCase(unittest.TestCase):
    def test_custom_operator_properties(self):
        MEMORY_READ_BYTES = 32768
        MEMORY_WRITE_BYTES = 32768
        MEMORY_READWRITE_BYTES = 0
        MMA_OPS = 100000
        GP_OPS = 5000
        NUM_TOKENS = 100
        MODEL_ID = "Qwen/Qwen3-32B"
        TARGET_OP_NAME = "reshape_and_cache"
        target_op = torch.ops.tensor_cast.reshape_and_cache.default
        original_properties_functor = OpInvokeInfo._op_properties_functors.get(target_op)
        self.addCleanup(_restore_op_properties_functor, target_op, original_properties_functor)

        @OpInvokeInfo.register_op_properties(target_op, True)
        def simple_operator_properties(
            op_invoke_info: OpInvokeInfo,
        ) -> OpInvokeInfo.PerformanceProperties:
            properties = OpInvokeInfo.PerformanceProperties()

            properties.memory_read_bytes = MEMORY_READ_BYTES
            properties.memory_write_bytes = MEMORY_WRITE_BYTES
            properties.memory_readwrite_bytes = MEMORY_READWRITE_BYTES

            compute_ops = properties.compute_ops.setdefault(torch.float16, OpInvokeInfo.ComputeOps())
            compute_ops.mma_ops = MMA_OPS
            compute_ops.gp_ops = GP_OPS

            return properties

        user_config = UserInputConfig(model_id=MODEL_ID)
        model = build_model(user_config)
        inputs = torch.empty([1, NUM_TOKENS], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, NUM_TOKENS], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(model, model.model_config)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )

        self.assertIn(
            target_op,
            OpInvokeInfo._op_properties_functors,
            "failed to register operator",
        )

        result = None
        for event in runtime.event_list:
            if (
                hasattr(event.op_invoke_info, "func")
                and event.op_invoke_info.func is not None
                and TARGET_OP_NAME in event.op_invoke_info.func._name
            ):
                result = event
                break

        self.assertIsNotNone(result, "Failed to get result")

        perf_props = result.op_invoke_info.get_perf_properties()
        self.assertEqual(perf_props.memory_read_bytes, MEMORY_READ_BYTES)
        self.assertEqual(perf_props.memory_write_bytes, MEMORY_WRITE_BYTES)
        self.assertEqual(perf_props.memory_readwrite_bytes, MEMORY_READWRITE_BYTES)

        compute_ops = perf_props.compute_ops.get(torch.float16)
        self.assertEqual(compute_ops.mma_ops, MMA_OPS)
        self.assertEqual(compute_ops.gp_ops, GP_OPS)

    def test_custom_estimate_operator_estimator(self):
        all_to_all_execution_time_s = 3.0
        target_op = torch.ops.tensor_cast.all_to_all.default
        original_estimator = _op_estimator_table.get(None, {}).get(target_op)
        self.addCleanup(_restore_op_estimator, None, target_op, original_estimator)

        @register_op_estimator(target_op, None, True)
        def _estimate_custom_comm(op_invoke_info, device_profile) -> object:
            return PerformanceModel.Result(all_to_all_execution_time_s)

        model_id = "Qwen/Qwen3-235B-A22B"
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        parallel_config = ParallelConfig(
            world_size=16,
            tensor_parallel_size=2,
            mlp_tensor_parallel_size=4,
            lmhead_tensor_parallel_size=1,
            expert_parallel_size=16,
            moe_data_parallel_size=1,
            moe_tensor_parallel_size=1,
        )
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            enable_repetition=True,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)

        NUM_TOKENS = 100
        inputs = torch.empty([1, NUM_TOKENS], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, NUM_TOKENS], dtype=torch.long, device="meta")
        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(model, model.model_config)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with (
            Runtime(perf_model, machine_config) as runtime,
            torch.no_grad(),
        ):
            model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )

        TARGET_OP_NAME = "all_to_all"
        result = None
        for event in runtime.event_list:
            if (
                hasattr(event.op_invoke_info, "func")
                and event.op_invoke_info.func is not None
                and TARGET_OP_NAME in event.op_invoke_info.func._name
            ):
                result = event
                break
        self.assertIsNotNone(result)
        self.assertEqual(
            result.perf_results.get("analytic").execution_time_s,
            all_to_all_execution_time_s,
        )


if __name__ == "__main__":
    unittest.main()

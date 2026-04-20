import unittest

import torch

from tensor_cast.core.input_generator import generate_inputs, RequestInfo
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime


class TestDeepseekV32Model(unittest.TestCase):
    def test_model_init(self):
        model_id = "deepseek-ai/DeepSeek-V3.2"
        num_queries = 3500
        user_input = UserInputConfig(
            model_id=model_id,
            num_queries=1,
            query_len=num_queries,
            context_length=num_queries,
            device="TEST_DEVICE",
            num_mtp_tokens=2,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        model = build_model(user_input)
        inputs = generate_inputs(
            model,
            [
                RequestInfo(
                    query_len=num_queries,
                    seq_len=num_queries,
                    concurrency=1,
                    is_decode=True,
                )
            ],
        )
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            model.forward(**inputs)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.multihead_latent_attention_quant.default", result)
        self.assertIn("tensor_cast.dsa_index_cache.default", result)
        total_time_s = runtime.total_execution_time_s()[perf_model.name]
        self.assertGreater(total_time_s, 0)

    def test_deepseek_v32_mla_performance(self):
        def get_mla_time(model_id: str, seq_len: int) -> float:
            user_input = UserInputConfig(
                model_id=model_id,
                num_queries=1,
                query_len=seq_len,
                context_length=seq_len,
                device="TEST_DEVICE",
                num_mtp_tokens=2,
                quantize_attention_action=QuantizeAttentionAction.INT8,
            )
            model = build_model(user_input)
            inputs = generate_inputs(
                model,
                [
                    RequestInfo(
                        query_len=seq_len,
                        seq_len=seq_len,
                        concurrency=1,
                        is_decode=True,
                    )
                ],
            )
            machine_config = TEST_DEVICE
            perf_model = AnalyticPerformanceModel(machine_config)
            with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
                model.forward(**inputs)

            total_time = 0.0
            for event in runtime.event_list:
                func_name = str(event.op_invoke_info.func)
                if "multihead_latent_attention_quant" in func_name:
                    total_time = event.perf_results.get("analytic").execution_time_s
            return total_time

        seq_len = 3500
        time_v31 = get_mla_time("deepseek-ai/DeepSeek-V3.1", seq_len)
        time_v32 = get_mla_time("deepseek-ai/DeepSeek-V3.2", seq_len)

        self.assertGreater(time_v31, 0)
        self.assertGreater(time_v32, 0)
        self.assertLess(time_v32, time_v31)

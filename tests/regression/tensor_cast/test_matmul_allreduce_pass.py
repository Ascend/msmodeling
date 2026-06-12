import unittest
from dataclasses import asdict

import pytest
import torch
from parameterized import parameterized
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig


@pytest.mark.nightly
class MatmulAllReducePassTestCase(unittest.TestCase):
    MODEL_ID = "Qwen/Qwen3-32B"
    NUM_TOKENS = 100

    def setUp(self):
        torch.compiler.reset()

    def _base_user_config(self, quant_action: QuantizeLinearAction) -> UserInputConfig:
        return UserInputConfig(
            model_id=self.MODEL_ID,
            num_queries=1,
            query_len=self.NUM_TOKENS,
            context_length=1000,
            do_compile=True,
            quantize_linear_action=quant_action,
            num_mtp_tokens=0,
            num_hidden_layers_override=1,
            world_size=2,
            tp_size=2,
        )

    def test_qwen3_fp_matmul_allreduce_fused(self):
        user_input = self._base_user_config(QuantizeLinearAction.DISABLED)
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn("tensor_cast.matmul_all_reduce.default", result["table_result"])

    @parameterized.expand(
        [
            (
                QuantizeLinearAction.W8A8_STATIC,
                "tensor_cast.static_quant_linear_all_reduce.default",
            ),
            (
                QuantizeLinearAction.W4A8_STATIC,
                "tensor_cast.static_quant_linear_int4_all_reduce.default",
            ),
            (
                QuantizeLinearAction.FP8,
                "tensor_cast.fp8_linear_all_reduce.default",
            ),
            (
                QuantizeLinearAction.MXFP4,
                "tensor_cast.mxfp4_linear_all_reduce.default",
            ),
        ]
    )
    def test_qwen3_quant_matmul_allreduce_fused(self, quant_action: QuantizeLinearAction, expected_op):
        user_input = self._base_user_config(quant_action)
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn(expected_op, result["table_result"])


if __name__ == "__main__":
    unittest.main()

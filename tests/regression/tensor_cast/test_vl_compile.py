import unittest

import pytest
import torch
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.patch_torch import patch_torch


class TestMaskedScatterMetaSafe(unittest.TestCase):
    def test_masked_scatter_on_meta_is_shape_safe(self):
        with patch_torch():
            x = torch.empty((4, 8), dtype=torch.float16, device="meta")
            mask = torch.zeros((4, 8), dtype=torch.bool, device="meta")
            src = torch.empty((0,), dtype=torch.float16, device="meta")
            out = x.masked_scatter(mask, src)
            self.assertEqual(tuple(out.shape), (4, 8))
            self.assertEqual(out.dtype, torch.float16)
            self.assertEqual(out.device.type, "meta")


@pytest.mark.nightly
class TestVLCompilePrefillNightly(unittest.TestCase):
    def test_glm45v_prefill_with_compile(self):
        user_input = UserInputConfig(
            device="TEST_DEVICE",
            model_id="zai-org/GLM-4.5V",
            num_queries=1,
            query_len=30,
            context_length=0,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        self.assertTrue(model_runner.model.is_vl_model)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            exec_time = result.execution_time_s
            if isinstance(exec_time, dict):
                exec_time = next(iter(exec_time.values()))
            self.assertGreaterEqual(exec_time, 0.0)
            self.assertIn("Total time for analytic", result.table_result)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
from parameterized import parameterized

from tensor_cast.core.input_generator import (
    generate_image_inputs,
    generate_inputs,
    generate_inputs_varlen,
    RequestInfo,
)
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.model_config import ModelConfig, ParallelConfig, QuantConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.model import TransformerModel
from tensor_cast.transformers.utils import AutoModelConfigLoader


class InputGeneratorTestCase(unittest.TestCase):
    @parameterized.expand([[True], [False]])
    def test_selected_token_indices_for_lmhead(self, is_decode):
        auto_loader = AutoModelConfigLoader()
        model_id = "Qwen/Qwen3-32B"
        hf_config = auto_loader.load_config(model_id)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            enable_repetition=True,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)

        query_len = 100
        batch_size = 2
        inputs = generate_inputs(
            model,
            [
                RequestInfo(
                    query_len=query_len,
                    seq_len=query_len,
                    concurrency=batch_size,
                    is_decode=is_decode,
                )
            ],
        )
        if is_decode:
            output_shape = (1, batch_size * query_len, model.vocab_size)
        else:
            output_shape = (1, batch_size, model.vocab_size)

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config), torch.no_grad():
            outputs = model.forward(**inputs)
            self.assertEqual(outputs.shape, output_shape)

    @parameterized.expand([[True], [False]])
    def test_varlen_selected_token_indices_for_lmhead(self, is_decode):
        auto_loader = AutoModelConfigLoader()
        model_id = "Qwen/Qwen3-32B"
        hf_config = auto_loader.load_config(model_id)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            enable_repetition=True,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)

        query_len = [90, 110]
        batch_size = len(query_len)
        request_infos = []
        for i in range(batch_size):
            request_infos.append(
                RequestInfo(
                    query_len=query_len[i],
                    seq_len=query_len[i],
                    is_decode=is_decode,
                )
            )
        inputs = generate_inputs_varlen(model, request_infos, 128)
        if is_decode:
            output_shape = (1, sum(query_len), model.vocab_size)
        else:
            output_shape = (1, batch_size, model.vocab_size)

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config), torch.no_grad():
            outputs = model.forward(**inputs)
            self.assertEqual(outputs.shape, output_shape)

    def test_dsa_indexer_cache_dtype_follows_attention_quant_config(self):
        user_input = UserInputConfig(
            model_id="deepseek-ai/DeepSeek-V3.2",
            num_queries=1,
            query_len=32,
            context_length=32,
            device="TEST_DEVICE",
            num_mtp_tokens=2,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        model = build_model(user_input)
        cache_info = generate_inputs(
            model,
            [
                RequestInfo(
                    query_len=32,
                    seq_len=32,
                    concurrency=1,
                    is_decode=True,
                )
            ],
        )

        self.assertEqual(
            cache_info["indexer_cache_by_layers"][0].dtype,
            torch.int8,
        )

    def test_dsa_indexer_cache_dtype_uses_fp8_when_attention_quant_is_fp8(self):
        user_input = UserInputConfig(
            model_id="deepseek-ai/DeepSeek-V3.2",
            num_queries=1,
            query_len=32,
            context_length=32,
            device="TEST_DEVICE",
            num_mtp_tokens=2,
            quantize_attention_action=QuantizeAttentionAction.FP8,
        )
        model = build_model(user_input)
        cache_info = generate_inputs(
            model,
            [
                RequestInfo(
                    query_len=32,
                    seq_len=32,
                    concurrency=1,
                    is_decode=True,
                )
            ],
        )

        self.assertEqual(
            cache_info["indexer_cache_by_layers"][0].dtype,
            torch.float8_e4m3fn,
        )

    def test_qwen3_vl_1080p_resize_to_1088x1920(self):
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        model_config = ModelConfig(
            parallel_config=ParallelConfig(),
            quant_config=QuantConfig(),
            dtype=torch.bfloat16,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)

        image_kwargs = generate_image_inputs(
            model=model,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
            concurrency=1,
        )

        # grid_h=68, grid_w=120 -> resized height/width = 1088x1920
        self.assertTrue(
            torch.equal(image_kwargs["image_grid_thw"], torch.tensor([[1, 68, 120]]))
        )

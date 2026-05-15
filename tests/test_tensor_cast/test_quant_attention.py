import unittest
from itertools import product

import torch
from parameterized import parameterized

from tensor_cast.core.model_builder import build_model
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.model_config import (
    AttentionQuantConfig,
    ModelConfig,
    MultiheadLatentAttentionQuantConfig,
    ParallelConfig,
    QuantConfig,
)
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.quantize_utils import AttentionQuantType, LinearQuantType
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.custom_model_registry import (
    get_moe_config,
    get_mtp_block_module_name,
)
from tensor_cast.transformers.model import TransformerModel
from tensor_cast.transformers.utils import AutoModelConfigLoader
from .test_common import (
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    has_submodule_with_cls_name,
)


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
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(kv_quant_start_idx, kv_quant_end_idx, attn_quant_type),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=2,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)
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

    @parameterized.expand(list(product(["deepseek-ai/DeepSeek-V3.1"], QUANT_TYPES)))
    def test_mla(self, model_id, attn_quant_type):
        num_mtp_layers = 1
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            quantize_attention_action=attn_quant_type,
        )

        model = build_model(user_config)

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

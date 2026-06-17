"""Unit tests for tensor_cast.core.quantization.config helpers."""

from tensor_cast.core.quantization.config import (
    _filter_action_kwargs,
    _set_linear_patterns,
    create_quant_config,
)
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.model_config import QuantConfig
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity, get_quant_config


class TestQuantizationConfigHelpers:
    def test_filter_action_kwargs_strips_mxfp4_only_fields_for_fp8(self):
        kwargs = {
            "weight_group_size": 32,
            "weight_quant_granularity": QuantGranularity.PER_GROUP,
            "extra_flag": True,
        }

        filtered = _filter_action_kwargs(QuantizeLinearAction.FP8, kwargs)

        assert filtered == {"extra_flag": True}

    def test_filter_action_kwargs_preserves_all_fields_for_mxfp4(self):
        kwargs = {
            "weight_group_size": 32,
            "weight_quant_granularity": QuantGranularity.PER_GROUP,
        }

        assert _filter_action_kwargs(QuantizeLinearAction.MXFP4, kwargs) == kwargs

    def test_set_linear_patterns_registers_config_per_pattern(self):
        quant_config = QuantConfig()

        _set_linear_patterns(quant_config, ["*.self_attn.*", "*.mlp.gate_proj"], QuantizeLinearAction.FP8)

        assert quant_config.linear_configs["*.self_attn.*"].quant_type == LinearQuantType.FP8
        assert quant_config.linear_configs["*.mlp.gate_proj"].quant_type == LinearQuantType.FP8

    def test_create_quant_config_mxfp4_experts_fp8_non_expert_override(self):
        quant_config = create_quant_config(
            quantize_linear_action=QuantizeLinearAction.MXFP4,
            quantize_non_expert_linear_action=QuantizeLinearAction.FP8,
            weight_group_size=32,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
        )

        non_expert_cfg = get_quant_config("model.layers.3.self_attn.q_a_proj", quant_config, "default_dit")
        shared_cfg = get_quant_config("model.layers.3.mlp.shared_experts.gate_proj", quant_config, "default_dit")
        expert_cfg = get_quant_config("model.layers.3.mlp.experts.5.gate_proj", quant_config, "default_dit")

        assert non_expert_cfg.quant_type == LinearQuantType.FP8
        assert shared_cfg.quant_type == LinearQuantType.FP8
        assert expert_cfg.quant_type == LinearQuantType.MXFP4
        assert non_expert_cfg.dynamic_quant_granularity == QuantGranularity.PER_TENSOR
        assert expert_cfg.weight_group_size == 32
        assert expert_cfg.weight_quant_granularity == QuantGranularity.PER_GROUP

    def test_create_quant_config_linear_only_keeps_original_behavior(self):
        quant_config = create_quant_config(quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC)

        layer_cfg = get_quant_config("model.layers.0.mlp.gate_proj", quant_config, "default_dit")

        assert layer_cfg.quant_type == LinearQuantType.W8A8

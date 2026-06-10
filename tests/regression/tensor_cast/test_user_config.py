"""Unit tests for tensor_cast.core.user_config."""

from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig


class TestUserInputConfigPrintInfo:
    def test_print_info_reports_mxfp4_and_backbone_override(self, capsys):
        user_config = UserInputConfig(
            model_id="deepseek-ai/DeepSeek-V4",
            quantize_linear_action=QuantizeLinearAction.MXFP4,
            quantize_backbone_linear_action=QuantizeLinearAction.FP8,
            mxfp4_group_size=32,
            quantize_attention_action=QuantizeAttentionAction.FP8,
        )

        user_config._print_info()
        output = capsys.readouterr().out

        assert "Quantization Linear: MXFP4" in output
        assert "MXFP4 group size: 32" in output
        assert "Quantization Backbone Linear (override): FP8" in output
        assert "Quantization Attention: FP8" in output

    def test_print_info_reports_disabled_quantization(self, capsys):
        user_config = UserInputConfig(
            model_id="test/model",
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            quantize_backbone_linear_action=QuantizeLinearAction.DISABLED,
            quantize_attention_action=QuantizeAttentionAction.DISABLED,
        )

        user_config._print_info()
        output = capsys.readouterr().out

        assert "Quantization Linear: Disabled" in output
        assert "Quantization Attention: Disabled" in output
        assert "Quantization Backbone Linear" not in output

    def test_get_quant_config_mxfp4_experts_fp8_backbone(self):
        user_config = UserInputConfig(
            quantize_linear_action=QuantizeLinearAction.MXFP4,
            quantize_backbone_linear_action=QuantizeLinearAction.FP8,
            mxfp4_group_size=32,
        )

        quant_config = user_config.get_quant_config()

        from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity, get_quant_config

        backbone_cfg = get_quant_config("model.layers.0.self_attn.q_proj", quant_config, "default_dit")
        expert_cfg = get_quant_config("model.layers.0.mlp.experts.1.up_proj", quant_config, "default_dit")

        assert backbone_cfg.quant_type == LinearQuantType.FP8
        assert expert_cfg.quant_type == LinearQuantType.MXFP4
        assert expert_cfg.weight_group_size == 32
        assert expert_cfg.weight_quant_granularity == QuantGranularity.PER_GROUP

"""Unit tests for tensor_cast.core.user_config."""

from dataclasses import fields

import pytest

from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.model_config import WordEmbeddingTPMode


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


class TestUserInputConfigWordEmbeddingTp:
    def test_word_embedding_tp_is_single_nullable_mode(self):
        user_config = UserInputConfig(word_embedding_tp="row")

        assert user_config.word_embedding_tp == WordEmbeddingTPMode.row
        removed_user_field = "word_embedding_tp" + "_mode"
        removed_parallel_field = "embedding_parallel" + "_mode"
        assert removed_user_field not in {field.name for field in fields(UserInputConfig)}

        parallel_config = user_config.get_parallel_config()
        assert parallel_config.embedding_parallel == WordEmbeddingTPMode.row
        assert removed_parallel_field not in {field.name for field in fields(type(parallel_config))}

    def test_legacy_bool_word_embedding_tp_is_still_normalized(self):
        enabled_config = UserInputConfig(word_embedding_tp=True)
        disabled_config = UserInputConfig(word_embedding_tp=False)

        assert enabled_config.word_embedding_tp == WordEmbeddingTPMode.col
        assert enabled_config.get_parallel_config().embedding_parallel == WordEmbeddingTPMode.col
        assert disabled_config.word_embedding_tp is None
        assert disabled_config.get_parallel_config().embedding_parallel is None

    def test_word_embedding_tp_invalid_value_raises(self):
        with pytest.raises(ValueError, match="word_embedding_tp must be one of"):
            UserInputConfig(word_embedding_tp="invalid")

        with pytest.raises(ValueError, match="word_embedding_tp must be one of"):
            UserInputConfig(word_embedding_tp=123)

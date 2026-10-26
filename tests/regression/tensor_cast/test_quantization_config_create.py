# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Unit tests for tensor_cast.core.quantization.config helpers."""

import pytest
import torch

from cli.inference.text_generate import arg_parse
from tensor_cast.core.quantization.config import (
    _filter_action_kwargs,
    _set_linear_patterns,
    create_quant_config,
)
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import QuantConfig
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity, get_quant_config, quantize_linear_modules


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

    @pytest.mark.parametrize("quantize_lmhead", [False, True])
    @pytest.mark.parametrize("wrapped", [False, True])
    @pytest.mark.parametrize("action", [QuantizeLinearAction.W8A8_STATIC, QuantizeLinearAction.W8A8_DYNAMIC])
    def test_user_config_lmhead_option_reaches_module_conversion(self, quantize_lmhead, wrapped, action):
        quant_config = UserInputConfig(
            quantize_linear_action=action,
            quantize_lmhead=quantize_lmhead,
        ).get_quant_config()
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(32, 64, bias=False, device="meta", dtype=torch.float16)
        model.excluded_projection = torch.nn.Linear(32, 32, bias=False, device="meta", dtype=torch.float16)
        if wrapped:
            root = torch.nn.Module()
            root._inner = model
        else:
            root = model
        # Explicit exclusions added after the factory must keep their precedence.
        quant_config.modules_to_not_convert.append("excluded_projection")
        quant_config.linear_configs["excluded_projection"] = quant_config.linear_configs["layers.*"]

        quantize_linear_modules(root, TensorCastQuantLinear, quant_config, "", None)

        assert isinstance(model.lm_head, TensorCastQuantLinear) is quantize_lmhead
        if quantize_lmhead:
            assert model.lm_head.qweight.dtype == torch.int8
        else:
            assert model.lm_head.weight.dtype == torch.float16
        assert isinstance(model.excluded_projection, torch.nn.Linear)
        assert model.excluded_projection.weight.dtype == torch.float16

    def test_lmhead_option_without_linear_quantization_keeps_default_exclusion(self):
        quant_config = create_quant_config(quantize_lmhead=True)

        assert quant_config.modules_to_not_convert == ["lm_head"]
        assert not quant_config.linear_configs

    @pytest.mark.parametrize("exclusion", ["lm_head", "*.lm_head", "re:.*lm_head$"])
    def test_explicit_lmhead_exclusion_still_wins_after_factory(self, exclusion):
        quant_config = create_quant_config(
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC, quantize_lmhead=True
        )
        quant_config.modules_to_not_convert.append(exclusion)
        model = torch.nn.Module()
        model.decoder = torch.nn.Module()
        model.decoder.lm_head = torch.nn.Linear(32, 64, bias=False, device="meta", dtype=torch.float16)

        quantize_linear_modules(model, TensorCastQuantLinear, quant_config, "", None)

        assert isinstance(model.decoder.lm_head, torch.nn.Linear)
        assert model.decoder.lm_head.weight.dtype == torch.float16

    @pytest.mark.parametrize("enable_lmhead", [False, True])
    @pytest.mark.parametrize("action", [None, "DISABLED", "W8A8_STATIC", "MXFP4"])
    def test_cli_lmhead_follows_broad_action_with_non_expert_override(self, enable_lmhead, action):
        argv = [
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-queries=1",
            "--query-length=8",
            "--quantize-non-expert-linear-action=FP8",
        ]
        if action is not None:
            argv.append(f"--quantize-linear-action={action}")
        if enable_lmhead:
            argv.append("--quantize-lmhead")
        user_config = UserInputConfig.from_args(arg_parse(argv))
        quant_config = user_config.get_quant_config()

        assert user_config.quantize_linear_action == (action or QuantizeLinearAction.W8A8_DYNAMIC)
        assert user_config.quantize_lmhead is enable_lmhead
        non_expert = get_quant_config("model.layers.0.self_attn.q_proj", quant_config, "")
        assert non_expert.quant_type == LinearQuantType.FP8
        assert non_expert.dynamic_quant_granularity == QuantGranularity.PER_TENSOR
        for name in ("lm_head", "model.lm_head"):
            head_config = get_quant_config(name, quant_config, "")
            if enable_lmhead and action != "DISABLED":
                assert "lm_head" not in quant_config.modules_to_not_convert
                expected_type = LinearQuantType.MXFP4 if action == "MXFP4" else LinearQuantType.W8A8
                assert head_config.quant_type == expected_type
                if action == "MXFP4":
                    assert head_config.weight_group_size == user_config.mxfp4_group_size
                    assert head_config.weight_quant_granularity == QuantGranularity.PER_GROUP
            else:
                assert "lm_head" in quant_config.modules_to_not_convert
                assert head_config is None

    def test_lmhead_option_does_not_mutate_later_factory_defaults(self):
        enabled = create_quant_config(quantize_linear_action=QuantizeLinearAction.W8A8_STATIC, quantize_lmhead=True)
        disabled = create_quant_config(quantize_linear_action=QuantizeLinearAction.W8A8_STATIC)

        assert enabled.modules_to_not_convert == []
        assert disabled.modules_to_not_convert == ["lm_head"]

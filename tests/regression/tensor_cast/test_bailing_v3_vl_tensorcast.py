# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the License at
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import pytest
import torch

from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.layers.rotary_embedding import CachingMRotaryEmb
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.transformers.custom_model_registry import get_model_profile
from tensor_cast.transformers.module_omission import LanguageModelOnlyModule, omit_model_parts
from tests.helpers.model_assets import vendored_model_config_path


def test_text_only_request_keeps_full_vl_weight(tmp_path, monkeypatch):
    import transformers.dynamic_module_utils as dynamic_module_utils

    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "transformers_modules"))
    runner = ModelRunner(
        UserInputConfig(
            device="TEST_DEVICE",
            model_id=vendored_model_config_path("inclusionAI/Ling-3.0-flash-VL"),
            num_queries=1,
            query_len=8,
            num_hidden_layers_override=1,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
    )
    full_vl_weight_size_gb = runner.model_weight_size_gb

    result = runner.run_inference()

    assert result.model_weight_size_gb == pytest.approx(full_vl_weight_size_gb)


def test_vl_model_rejects_invalid_visual_profile_path(tmp_path, monkeypatch):
    import transformers.dynamic_module_utils as dynamic_module_utils

    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "transformers_modules"))
    profile = get_model_profile("bailing_moe_v3_vl")
    monkeypatch.setattr(profile, "visual_module_path", "model.missing_visual")

    with pytest.raises(ValueError, match="Visual module path 'model.missing_visual'.*bailing_moe_v3_vl"):
        ModelRunner(
            UserInputConfig(
                device="TEST_DEVICE",
                model_id=vendored_model_config_path("inclusionAI/Ling-3.0-flash-VL"),
                num_queries=1,
                query_len=8,
                num_hidden_layers_override=1,
                quantize_linear_action=QuantizeLinearAction.DISABLED,
            )
        )


def test_language_model_only_omits_exact_model_paths():
    class VisualModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)

    class VLModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.visual = VisualModel()
            self.other = torch.nn.Module()
            self.other.visual = VisualModel()
            self.linear_proj = torch.nn.Linear(4, 4, bias=False)
            self.image_start = torch.nn.Parameter(torch.empty(4))
            self.language_model = torch.nn.Linear(4, 4, bias=False)

    model = VLModel()
    omit_model_parts(model, ("model.visual", "linear_proj", "image_start"))

    assert isinstance(model.model.visual, LanguageModelOnlyModule)
    assert isinstance(model.other.visual, VisualModel)
    assert isinstance(model.linear_proj, LanguageModelOnlyModule)
    assert model.model.visual.proj.in_features == 4
    assert model.image_start.numel() == 0
    assert model.image_start._language_model_only_path == "image_start"
    assert sum(parameter.numel() for parameter in model.parameters()) == 32
    with pytest.raises(RuntimeError, match="Multimodal module 'model.visual' is disabled"):
        model.model.visual(torch.empty((1, 4)))


def test_language_model_only_ignores_missing_exact_path():
    model = torch.nn.Module()
    model.visual = torch.nn.Linear(4, 4, bias=False)

    omit_model_parts(model, ("model.missing",))

    assert isinstance(model.visual, torch.nn.Linear)


def test_language_model_only_normalizes_and_deduplicates_module_paths():
    class VLModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(4, 4, bias=False)

    model = VLModel()
    omit_model_parts(model, (".visual.", "visual"))

    assert isinstance(model.visual, LanguageModelOnlyModule)
    assert model.visual.module_path == "visual"


def test_multimodal_h2d_bytes_include_only_device_payloads():
    from tensor_cast.core.input_generator import get_multimodal_h2d_num_bytes

    pixel_values = torch.empty((8, 1536), dtype=torch.bfloat16, device="meta")
    image_embeds = torch.empty((196, 2048), dtype=torch.bfloat16, device="meta")
    image_grid_thw = torch.empty((1, 3), dtype=torch.int64, device="meta")

    total = get_multimodal_h2d_num_bytes(
        {
            "pixel_values": pixel_values,
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
        }
    )

    expected = pixel_values.numel() * pixel_values.element_size()
    expected += image_embeds.numel() * image_embeds.element_size()
    assert total == expected


def test_multimodal_h2d_bytes_round_packed_payloads_to_physical_bytes():
    from tensor_cast.core.input_generator import get_multimodal_h2d_num_bytes

    pixel_values = torch.empty((3,), dtype=torch.int4, device="meta")

    assert get_multimodal_h2d_num_bytes({"pixel_values": pixel_values}) == 2


@pytest.mark.parametrize(
    ("h2d_time_s", "expected_time"),
    [
        (0.000453, "Multimodal input time: 0.453 ms"),
        (None, "Multimodal input time: unavailable (host bandwidth not configured)"),
    ],
)
def test_multimodal_h2d_metrics_print_info(capsys, h2d_time_s, expected_time):
    metrics = ModelRunnerMetrics(
        total_device_memory_gb=24.0,
        model_weight_size_gb=5.0,
        peak_memory_usage_gb=12.0,
        kv_cache_size_gb=3.0,
        kv_cache_per_token_gb=0.001,
        model_activation_size_gb=4.0,
        reserved_memory_gb=1.0,
        device_memory_available_gb=6.0,
        tps_per_model={"analytic": 200.0},
        execution_time_s={"analytic": 0.05},
        run_time_s=0.06,
        batch_size=4,
        table_result="performance_data",
        breakdowns={},
        multimodal_input_h2d_bytes=25_067_520,
        multimodal_input_h2d_time_s=h2d_time_s,
    )

    metrics.print_info()

    output = capsys.readouterr().out
    assert "Multimodal input traffic: 25,067,520 B" in output
    assert expected_time in output
    assert "Host-to-device traffic" not in output


def test_multimodal_h2d_time_is_unavailable_without_host_bandwidth():
    runner = object.__new__(ModelRunner)
    runner.device_profile = type(
        "DeviceProfileStub",
        (),
        {
            "host_bandwidth_bytes_ps": 0.0,
            "host_transfer_efficiency": 1.0,
            "host_latency_s": 0.0,
        },
    )()

    assert runner._get_host_to_device_time_s(1024) is None
    assert runner._get_host_to_device_time_s(0) == 0.0


class _FakeMRotaryEmb(torch.nn.Module):
    def forward(self, x, position_ids):
        return (
            torch.empty((*position_ids.shape, 8), device=x.device, dtype=x.dtype),
            torch.empty((*position_ids.shape, 8), device=x.device, dtype=x.dtype),
        )


def test_mrope_cache_lookup_accepts_three_axis_positions():
    cache = torch.empty((32, 24), device="meta", dtype=torch.bfloat16)
    positions = torch.empty((3, 2, 5), device="meta", dtype=torch.long)

    cos_sin = torch.ops.tensor_cast.mrope_cache_lookup(cache, positions, 12)

    assert cos_sin.shape == (2, 5, 24)
    assert cos_sin.dtype == torch.bfloat16


def test_caching_mrope_builds_cache_and_selects_runtime_positions():
    cached = CachingMRotaryEmb(_FakeMRotaryEmb(), torch.bfloat16, max_position_embeddings=32)
    x = torch.empty((2, 5), device="meta", dtype=torch.bfloat16)
    positions = torch.empty((3, 2, 5), device="meta", dtype=torch.long)

    cos, sin = cached(x, positions)

    assert cached.cos_sin_cache.shape == (32, 16)
    assert cached.rotary_dim == 8
    assert cos.shape == (2, 5, 8)
    assert sin.shape == (2, 5, 8)


def test_caching_mrope_falls_back_for_a_different_activation_dtype():
    cached = CachingMRotaryEmb(_FakeMRotaryEmb(), torch.bfloat16, max_position_embeddings=32)
    x = torch.empty((1, 5), device="meta", dtype=torch.float32)
    positions = torch.empty((1, 5), device="meta", dtype=torch.long)

    cos, sin = cached(x, positions)

    assert cos.shape == (1, 5, 8)
    assert sin.shape == (1, 5, 8)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


def test_mrope_cache_lookup_properties_charge_selected_rows_only():
    cache = torch.empty((32, 24), device="meta", dtype=torch.bfloat16)
    positions = torch.empty((3, 2, 5), device="meta", dtype=torch.long)
    op = torch.ops.tensor_cast.mrope_cache_lookup.default
    output = op(cache, positions, 12)

    properties = OpInvokeInfo(op, (cache, positions, 12), {}, output).get_perf_properties()

    assert properties.memory_read_bytes == positions.numel() * positions.element_size() + output.numel() * 2
    assert properties.memory_write_bytes == output.numel() * output.element_size()


def test_vision_rotary_uses_fp32_and_restores_input_dtypes():
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _bailing_vl_apply_vision_rotary_pos_emb,
        _bailing_vl_rotate_half,
    )

    query = torch.randn((3, 2, 8), dtype=torch.bfloat16)
    key = torch.randn((3, 2, 8), dtype=torch.float16)
    cos = torch.randn((3, 8), dtype=torch.bfloat16)
    sin = torch.randn((3, 8), dtype=torch.bfloat16)

    actual_query, actual_key = _bailing_vl_apply_vision_rotary_pos_emb(query, key, cos, sin)
    cos_fp32 = cos.unsqueeze(1).float()
    sin_fp32 = sin.unsqueeze(1).float()
    expected_query = (query.float() * cos_fp32 + _bailing_vl_rotate_half(query.float()) * sin_fp32).to(query.dtype)
    expected_key = (key.float() * cos_fp32 + _bailing_vl_rotate_half(key.float()) * sin_fp32).to(key.dtype)

    assert actual_query.dtype == query.dtype
    assert actual_key.dtype == key.dtype
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)

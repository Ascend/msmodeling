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

"""Bailing V3 VL (Ling-3.0-flash-VL) multimodal input geometry contract.

Covers PR-1 (issue #532): pixel-budget fail-closed resize, per-image
geometry, the placeholder contract measured from the real processor,
and the varlen path no longer dropping image inputs.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensor_cast.core import input_generator
from tensor_cast.core.input_generator import (
    RequestInfo,
    generate_image_inputs,
    generate_inputs_varlen,
    get_inputs_num_bytes,
    resize_image,
    _BAILING_MM2_DEFAULT_MAX_PIXELS,
    _BAILING_MM2_DEFAULT_MIN_PIXELS,
    _BAILING_MM2_IMAGE_TOKEN_ID,
    _BAILING_MM2_VISION_END_TOKEN_ID,
    _BAILING_MM2_VISION_START_TOKEN_ID,
)
from tensor_cast.core.vision_geometry import VISION_IMAGE_BOUNDARY_TOKEN_COUNT

# Ling-3.0-flash-VL vision config: qwen3_moe_vit layout.
_PATCH_SIZE = 16
_MERGE_SIZE = 2
_BF16_BYTES = 2


def _bailing_vl_model():
    return SimpleNamespace(
        is_vl_model=True,
        model_id="inclusionAI/Ling-3.0-flash-VL",
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="bailing_moe_v3_vl",
                vision_config=SimpleNamespace(
                    patch_size=_PATCH_SIZE,
                    spatial_merge_size=_MERGE_SIZE,
                    temporal_patch_size=2,
                    in_channels=3,
                ),
            ),
            dtype=torch.bfloat16,
            mtp_config=None,
            parallel_config=SimpleNamespace(data_parallel_size=1, decode_context_parallel_size=1),
        ),
    )


@pytest.fixture
def _no_preprocessor_config(monkeypatch):
    """Simulate a model repo without preprocessor_config.json."""
    monkeypatch.setattr(
        input_generator,
        "_load_preprocessor_pixel_limits",
        lambda model_id: (None, None),
    )
    input_generator._load_bailing_vl_pixel_limits.cache_clear()
    yield
    input_generator._load_bailing_vl_pixel_limits.cache_clear()


def _pixel_values_rows(image_kwargs):
    return int(image_kwargs["pixel_values"].shape[0])


@pytest.mark.parametrize(
    ("height", "width", "expected_grid_h", "expected_grid_w"),
    [
        # 224x224 = 50176 px, inside [4096, 4194304]: no rescale.
        pytest.param(224, 224, 14, 14, id="224-in-budget"),
        # 3000x4000 = 12,000,000 px > max: smart_resize to 1760x2336.
        pytest.param(3000, 4000, 110, 146, id="3000x4000-clamped"),
        # 1080x1920 = 2,073,600 px: inside budget, ceil to patch grid.
        pytest.param(1080, 1920, 68, 120, id="1080x1920"),
    ],
)
def test_bailing_resize_uses_mm2_budget(height, width, expected_grid_h, expected_grid_w, _no_preprocessor_config):
    resized = resize_image(
        "inclusionAI/Ling-3.0-flash-VL",
        "bailing_moe_v3_vl",
        height,
        width,
        patch_size=_PATCH_SIZE,
        merge_size=_MERGE_SIZE,
        temporal_patch_size=2,
    )
    assert resized == (expected_grid_h * _PATCH_SIZE, expected_grid_w * _PATCH_SIZE)


def test_bailing_resize_does_not_upsample_like_qwen(_no_preprocessor_config):
    # 160x160 = 25600 px > Bailing min (4096): no upsample -> 10x10 grid.
    # The Qwen budget (min 65536) would upsample this image.
    resized = resize_image(
        "inclusionAI/Ling-3.0-flash-VL",
        "bailing_moe_v3_vl",
        160,
        160,
        patch_size=_PATCH_SIZE,
        merge_size=_MERGE_SIZE,
        temporal_patch_size=2,
    )
    assert resized == (160, 160)


def test_bailing_pixel_budget_warns_and_stays_fail_closed(caplog, _no_preprocessor_config):
    with caplog.at_level(logging.WARNING):
        resize_image(
            "inclusionAI/Ling-3.0-flash-VL",
            "bailing_moe_v3_vl",
            224,
            224,
            patch_size=_PATCH_SIZE,
            merge_size=_MERGE_SIZE,
            temporal_patch_size=2,
        )
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "Bailing MM2 default pixel budget" in r.getMessage()
    ]
    assert warnings, "missing preprocessor_config.json must warn"
    assert (_BAILING_MM2_DEFAULT_MIN_PIXELS, _BAILING_MM2_DEFAULT_MAX_PIXELS) == (4096, 4194304)


def test_bailing_resize_prefers_processor_budget(monkeypatch):
    monkeypatch.setattr(
        input_generator,
        "_load_preprocessor_pixel_limits",
        lambda model_id: (4096, 4194304),
    )
    input_generator._load_bailing_vl_pixel_limits.cache_clear()
    try:
        min_pixels, max_pixels, source = input_generator._load_bailing_vl_pixel_limits("any")
        assert (min_pixels, max_pixels) == (4096, 4194304)
        assert source == "processor"
    finally:
        input_generator._load_bailing_vl_pixel_limits.cache_clear()


@pytest.mark.parametrize(
    ("height", "width", "grid_h", "grid_w"),
    [(224, 224, 14, 14), (3000, 4000, 110, 146), (1080, 1920, 68, 120)],
)
def test_generate_image_inputs_bailing_geometry(height, width, grid_h, grid_w, _no_preprocessor_config):
    model = _bailing_vl_model()
    image_kwargs = generate_image_inputs(model, 1, height, width, concurrency=1)

    assert image_kwargs["image_resize_source"] == "default"
    per_image = image_kwargs["per_image"]
    assert len(per_image) == 1
    item = per_image[0]
    assert item["grid_thw"] == [1, grid_h, grid_w]
    patch_rows = grid_h * grid_w
    assert item["patch_rows"] == patch_rows
    assert item["merged_rows"] == patch_rows // (_MERGE_SIZE**2)
    assert item["placeholder_len"] == patch_rows // (_MERGE_SIZE**2) + VISION_IMAGE_BOUNDARY_TOKEN_COUNT

    assert int(image_kwargs["image_grid_thw"].prod()) // (_MERGE_SIZE**2) == item["merged_rows"]
    assert image_kwargs["num_image_tokens"] == item["placeholder_len"]
    # pixel_values: one row per raw patch, channel*t*h*w columns.
    assert _pixel_values_rows(image_kwargs) == patch_rows
    assert image_kwargs["pixel_values"].shape[1] == 3 * 2 * _PATCH_SIZE * _PATCH_SIZE
    assert image_kwargs["pixel_values"].dtype == torch.bfloat16


def test_generate_image_inputs_placeholder_invariant_holds(_no_preprocessor_config):
    model = _bailing_vl_model()
    image_kwargs = generate_image_inputs(model, 1, 1080, 1920, concurrency=1)
    per_image = image_kwargs["per_image"]
    merge_length = _MERGE_SIZE**2
    assert sum(item["patch_rows"] for item in per_image) // merge_length == (
        sum(item["placeholder_len"] for item in per_image) - VISION_IMAGE_BOUNDARY_TOKEN_COUNT * len(per_image)
    )
    assert image_kwargs["num_image_tokens"] == sum(item["placeholder_len"] for item in per_image)


def test_bailing_processor_token_id_contract():
    # Measured from the real BailingMM2 processor (not the Qwen vocabulary).
    assert _BAILING_MM2_IMAGE_TOKEN_ID == 157157
    assert _BAILING_MM2_VISION_START_TOKEN_ID == 157158
    assert _BAILING_MM2_VISION_END_TOKEN_ID == 157159


def test_boundary_token_count_single_source():
    from tools.model_diagnostics.domain import constants as diag_constants

    from tensor_cast.core import vision_geometry

    assert diag_constants.VISION_IMAGE_BOUNDARY_TOKEN_COUNT is vision_geometry.VISION_IMAGE_BOUNDARY_TOKEN_COUNT


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_image_branch_preserves_image_inputs(_mock_kv, _mock_sparse, _no_preprocessor_config):
    model = _bailing_vl_model()
    requests = [
        RequestInfo(
            query_len=100,
            seq_len=100,
            is_decode=False,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        ),
        RequestInfo(
            query_len=100,
            seq_len=100,
            is_decode=False,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        ),
    ]

    inputs = generate_inputs_varlen(model, requests, 128)

    # The image tensors must reach the model forward.
    assert "pixel_values" in inputs
    assert _pixel_values_rows(inputs) == 2 * 68 * 120
    assert inputs["pixel_values"].shape[1] == 3 * 2 * _PATCH_SIZE * _PATCH_SIZE
    assert inputs["image_grid_thw"].shape == (2, 3)
    num_image_tokens = 68 * 120 // (_MERGE_SIZE**2) + VISION_IMAGE_BOUNDARY_TOKEN_COUNT
    assert inputs["attention_meta"].query_lens_values == [100 + num_image_tokens] * 2
    # Sequence grows by the language-side image tokens (merged + boundary),
    # matching the uniform generate_inputs semantics.
    assert inputs["attention_meta"].seq_lens_values == [100 + num_image_tokens] * 2
    # Input-generation metadata must not leak into forward kwargs.
    assert "per_image" not in inputs
    assert "image_resize_source" not in inputs


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_decode_only_drops_image_forward_but_counts_tokens(_mock_kv, _mock_sparse, _no_preprocessor_config):
    model = _bailing_vl_model()
    requests = [
        RequestInfo(
            query_len=1,
            seq_len=100,
            is_decode=True,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        ),
        RequestInfo(
            query_len=1,
            seq_len=100,
            is_decode=True,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        ),
    ]

    inputs = generate_inputs_varlen(model, requests, 128)

    assert "pixel_values" not in inputs
    num_image_tokens = 68 * 120 // (_MERGE_SIZE**2) + VISION_IMAGE_BOUNDARY_TOKEN_COUNT
    assert inputs["attention_meta"].query_lens_values == [1, 1]
    assert inputs["attention_meta"].seq_lens_values == [100 + num_image_tokens] * 2


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_mixed_batch_rejected(_mock_kv, _mock_sparse, _no_preprocessor_config):
    # The packed forward consumes the image kwargs once for the whole
    # batch; a mixed prefill/decode batch has no defined placeholder
    # semantics for the decode rows, so it must fail loudly.
    model = _bailing_vl_model()
    requests = [
        RequestInfo(
            query_len=100,
            seq_len=100,
            is_decode=False,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        ),
        RequestInfo(
            query_len=1,
            seq_len=100,
            is_decode=True,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        ),
    ]

    with pytest.raises(ValueError, match="mixed prefill/decode"):
        generate_inputs_varlen(model, requests, 128)


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_heterogeneous_image_params_rejected(_mock_kv, _mock_sparse, _no_preprocessor_config):
    # requests[0]'s geometry must not be silently broadcast to requests
    # with different image parameters.
    model = _bailing_vl_model()
    requests = [
        RequestInfo(
            query_len=100,
            seq_len=100,
            is_decode=False,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        ),
        RequestInfo(
            query_len=100,
            seq_len=100,
            is_decode=False,
            image_batch_size=2,
            image_height=1080,
            image_width=1920,
        ),
    ]

    with pytest.raises(ValueError, match="uniform image parameters"):
        generate_inputs_varlen(model, requests, 128)


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_missing_image_geometry_warns_and_skips(_mock_kv, _mock_sparse, _no_preprocessor_config, caplog):
    # A VL model whose varlen requests carry no image parameters keeps
    # the pre-PR-1 behaviour (no image inputs) but must not be silent.
    # Review (stormchasingg): the warning is rate-limited to once per
    # model_id so profiling loops (hundreds to thousands of text-only
    # forwards, e.g. Kimi-K3 throughput searches) do not flood the log.
    model = _bailing_vl_model()
    requests = [RequestInfo(query_len=64, seq_len=64, is_decode=False) for _ in range(2)]

    input_generator._VARLEN_TEXT_ONLY_WARNED_MODELS.clear()
    try:
        with caplog.at_level(logging.WARNING):
            inputs = generate_inputs_varlen(model, requests, 128)
        assert "pixel_values" not in inputs
        assert inputs["attention_meta"].query_lens_values == [64, 64]
        assert inputs["attention_meta"].seq_lens_values == [64, 64]
        assert any("missing image geometry" in r.getMessage() for r in caplog.records)

        # The second text-only forward for the same model must not re-warn.
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            inputs = generate_inputs_varlen(model, requests, 128)
        assert inputs["attention_meta"].query_lens_values == [64, 64]
        assert not any("missing image geometry" in r.getMessage() for r in caplog.records)
    finally:
        input_generator._VARLEN_TEXT_ONLY_WARNED_MODELS.clear()


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_get_inputs_num_bytes_includes_pixel_values(_mock_kv, _mock_sparse, _no_preprocessor_config):
    model = _bailing_vl_model()
    requests = [
        RequestInfo(
            query_len=100,
            seq_len=100,
            is_decode=False,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        )
    ]

    total = get_inputs_num_bytes(model, requests, 128)
    # 8160 patches x 1536 columns x bf16 = 25,067,520 B for one 1080p image.
    pixel_bytes = 68 * 120 * 3 * 2 * _PATCH_SIZE * _PATCH_SIZE * _BF16_BYTES
    assert pixel_bytes == 25_067_520
    assert total >= pixel_bytes


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_without_image_inputs_unchanged(_mock_kv, _mock_sparse):
    # A VL model without image parameters must behave exactly as before.
    model = _bailing_vl_model()
    requests = [RequestInfo(query_len=64, seq_len=64, is_decode=False)]

    inputs = generate_inputs_varlen(model, requests, 128)

    assert "pixel_values" not in inputs
    assert inputs["attention_meta"].query_lens_values == [64]
    assert inputs["attention_meta"].seq_lens_values == [64]


# ---------------------------------------------------------------------------
# PR-2 (issue #536): model-adapter profile, config patch,
# runtime patch and the fused MoE gate router.
# ---------------------------------------------------------------------------


def test_bailing_vl_profile_registers_visual_structure():
    from tensor_cast.transformers.custom_model_registry import get_model_profile

    profile = get_model_profile("bailing_moe_v3_vl")
    assert profile is not None
    assert profile.moe_module_name == "BailingMoeV3SparseMoeBlock"
    assert profile.mla_module_name == "BailingMoeV3MultiLatentAttention"
    assert profile.custom_expert_module_type is None
    assert profile.visual_module_path == "model.visual"
    assert profile.visual_layers_path_str == "model.visual.blocks"
    assert profile.language_layers_path_str == "model.layers"
    # Bailing merger is Norm + reshape only, but the root-level projector
    # (linear_proj.0/.2) shares the merger/projector TP mapping so it is
    # sharded under TP/VTP instead of staying fully replicated per rank
    # (review: Hudingyi).
    assert profile.visual_merger_linear_mapping == {
        "linear_proj.0": "colwise",
        "linear_proj.2": "rowwise",
    }
    assert "model.visual.blocks.*.mlp.linear_fc1" in profile.visual_mlp_linear_mapping
    assert "model.visual.blocks.*.mlp.linear_fc2" in profile.visual_mlp_linear_mapping


def test_bailing_vl_config_patch_ignores_other_model_types():
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _patch_hf_config_for_bailing_moe_v3_vl,
    )

    config = SimpleNamespace(model_type="qwen3_vl")
    assert _patch_hf_config_for_bailing_moe_v3_vl(config) is False


def test_bailing_vl_config_patch_clears_legacy_rope_and_deepstack():
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _patch_hf_config_for_bailing_moe_v3_vl,
    )
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = SimpleNamespace(
        model_type="bailing_moe_v3_vl",
        text_config=SimpleNamespace(
            rope_scaling={"rope_type": "default", "rope_theta": 6_000_000, "partial_rotary_factor": 0.5}
        ),
        vision_config=SimpleNamespace(deepstack_visual_indexes=[8, 16, 24]),
    )

    assert _patch_hf_config_for_bailing_moe_v3_vl(config) is True
    assert config.text_config.rope_scaling is None
    assert config.vision_config.deepstack_visual_indexes == []
    # The text modeling's 1D rope falls back to rope_type "default" even
    # after clearing rope_scaling, so the init function must be registered.
    assert "default" in ROPE_INIT_FUNCTIONS


def _fake_vl_model_with_kda_layer(mtp_layers=0, native_mtp_tail=True):
    kda_type = type(
        "BailingMoeV3KimiDeltaAttention",
        (torch.nn.Module,),
        {
            "layer_idx": 0,
            "head_dim": 8,
            "conv_size": 4,
            "no_kda_lora": True,
        },
    )
    attention = kda_type()
    layer = torch.nn.Module()
    layer.attention = attention
    layers = [layer]
    if mtp_layers and native_mtp_tail:
        # Mirror the official modeling: num_nextn_predict_layers tail layers
        # are BailingMoeV3MTPLayer appended to model.layers.
        native_mtp_type = type("BailingMoeV3MTPLayer", (torch.nn.Module,), {})
        layers.extend(native_mtp_type() for _ in range(mtp_layers))
    decoder = torch.nn.Module()
    decoder.layers = torch.nn.ModuleList(layers)
    decoder.num_nextn_predict_layers = mtp_layers
    unwrapped = torch.nn.Module()
    unwrapped.model = decoder

    class _Wrapper(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    wrapper = _Wrapper()
    model = SimpleNamespace(_inner=wrapper, unwrap=lambda: unwrapped)
    return model, wrapper, decoder, attention


def test_bailing_vl_model_patch_replaces_kda_and_injects_kwargs():
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _collect_bailing_v3_vl_attention_modules,
        _patch_bailing_v3_vl_model,
    )

    model, wrapper, decoder, _attention = _fake_vl_model_with_kda_layer()
    _patch_bailing_v3_vl_model(model)

    assert decoder._use_flash_attention_2 is True
    assert decoder._use_sdpa is False
    replaced = decoder.layers[0].attention
    assert type(replaced).__name__ == "BailingV3KimiDeltaAttentionTensorCast"
    assert wrapper._bailing_v3_tc_kwargs_patched is True

    # Runtime metadata flows into the attention wrapper through forward.
    metadata = object()
    wrapper(attention_meta=metadata, kv_cache_by_layers={})
    assert replaced._extra_forward_kwargs["attention_meta"] is metadata
    # Idempotent: second patch is a no-op.
    _patch_bailing_v3_vl_model(model)
    assert decoder.layers[0].attention is replaced

    collected = _collect_bailing_v3_vl_attention_modules(model, wrapper)
    assert collected == (replaced,)


def test_bailing_vl_model_patch_strips_native_mtp_tail():
    """Internal SFT configs (e.g. ling_vl_flash_sft_rc2) declare 1 native MTP
    layer while the open Ling-3.0-flash-VL config ships 0; the VL patch must
    validate and strip the tail instead of rejecting the model (PR-2 review:
    the model failed to initialize on the internal config).
    """
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import _patch_bailing_v3_vl_model

    model, _wrapper, decoder, _attention = _fake_vl_model_with_kda_layer(mtp_layers=1)
    main_layer = decoder.layers[0]

    _patch_bailing_v3_vl_model(model)

    assert list(decoder.layers) == [main_layer]
    assert decoder.num_nextn_predict_layers == 0
    # The main KDA layer still receives the TensorCast replacement.
    assert type(decoder.layers[0].attention).__name__ == "BailingV3KimiDeltaAttentionTensorCast"


def test_bailing_vl_model_patch_rejects_unexpected_native_mtp_tail():
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import _patch_bailing_v3_vl_model

    model, _wrapper, _decoder, _attention = _fake_vl_model_with_kda_layer(mtp_layers=1, native_mtp_tail=False)
    with pytest.raises(ValueError, match="Bailing V3 native MTP layers"):
        _patch_bailing_v3_vl_model(model)


def _fake_bailing_gate(num_experts=8, top_k=2):
    weight = torch.randn(num_experts, 16)
    expert_bias = torch.zeros(num_experts)

    class _Gate(torch.nn.Module):
        routed_scaling_factor = 2.5

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)
            self.register_buffer("expert_bias", expert_bias)

    gate = _Gate()
    return gate, top_k


def test_bailing_vl_route_uses_fused_sigmoid_gating_op():
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import _route_bailing_v3_vl_gate

    gate, top_k = _fake_bailing_gate()
    hidden_states = torch.randn(5, 16, dtype=torch.float16)
    indices, weights = _route_bailing_v3_vl_gate(gate, hidden_states, top_k, None, None)

    assert indices.shape == (5, top_k)
    assert indices.dtype == torch.int64
    assert weights.shape == (5, top_k)
    assert weights.dtype == hidden_states.dtype


def test_bailing_vl_route_applies_tp_padding_and_slice():
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import _route_bailing_v3_vl_gate

    gate, top_k = _fake_bailing_gate()
    hidden_states = torch.randn(5, 16, dtype=torch.float16)
    indices, weights = _route_bailing_v3_vl_gate(gate, hidden_states, top_k, None, None, tp_size=2, tp_rank=1)
    # 5 tokens pad to 6, rank 1 keeps 3 rows.
    assert indices.shape == (3, top_k)
    assert weights.shape == (3, top_k)


def test_bailing_vl_route_rejects_invalid_tp_rank():
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import _route_bailing_v3_vl_gate

    gate, top_k = _fake_bailing_gate()
    hidden_states = torch.randn(4, 16, dtype=torch.float16)
    with pytest.raises(ValueError, match="tp_rank"):
        _route_bailing_v3_vl_gate(gate, hidden_states, top_k, None, None, tp_size=2, tp_rank=2)


# ---------------------------------------------------------------------------
# Review round 2 (zhangzhenyu, 2026-10-21): DP image under-generation and
# chunked-prefill duplicate image inputs.
# ---------------------------------------------------------------------------


def _bailing_vl_model_dp(dp_size):
    return SimpleNamespace(
        is_vl_model=True,
        model_id="inclusionAI/Ling-3.0-flash-VL",
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="bailing_moe_v3_vl",
                vision_config=SimpleNamespace(
                    patch_size=_PATCH_SIZE,
                    spatial_merge_size=_MERGE_SIZE,
                    temporal_patch_size=2,
                    in_channels=3,
                ),
            ),
            dtype=torch.bfloat16,
            mtp_config=None,
            parallel_config=SimpleNamespace(data_parallel_size=dp_size, decode_context_parallel_size=1),
        ),
    )


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_dp_generates_image_per_request(_mock_kv, _mock_sparse, _no_preprocessor_config):
    # P1 fix (zhangzhenyu round 2): the varlen path passes the per-rank
    # request list, so generate_image_inputs must not divide by DP again.
    model = _bailing_vl_model_dp(dp_size=2)
    requests = [
        RequestInfo(query_len=64, seq_len=64, is_decode=False, image_batch_size=1, image_height=224, image_width=224)
        for _ in range(2)
    ]

    inputs = generate_inputs_varlen(model, requests, 128)

    # 2 requests x 1 image each: 2 * 196 = 392 patch rows, 2 grid rows.
    assert inputs["pixel_values"].shape[0] == 392
    assert inputs["image_grid_thw"].shape[0] == 2
    # Every request grows by its own 49 merged + 2 boundary tokens.
    assert inputs["attention_meta"].query_lens_values == [64 + 51, 64 + 51]
    assert inputs["attention_meta"].seq_lens_values == [64 + 51, 64 + 51]


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_text_only_rows_keep_zero_image_tokens(_mock_kv, _mock_sparse, _no_preprocessor_config):
    # Chunked prefill mixes the image-bearing first chunk with text-only
    # follow-up chunks (image_*=None): only the image row grows.
    model = _bailing_vl_model_dp(dp_size=1)
    requests = [
        RequestInfo(query_len=64, seq_len=64, is_decode=False, image_batch_size=1, image_height=224, image_width=224),
        RequestInfo(query_len=488, seq_len=552, is_decode=False),
    ]

    inputs = generate_inputs_varlen(model, requests, 128)

    # One image for the single image-bearing request.
    assert inputs["pixel_values"].shape[0] == 196
    assert inputs["image_grid_thw"].shape[0] == 1
    assert inputs["attention_meta"].query_lens_values == [64 + 51, 488]
    assert inputs["attention_meta"].seq_lens_values == [64 + 51, 552]


def test_varlen_rejects_mixed_image_geometry_among_image_rows(_no_preprocessor_config):
    model = _bailing_vl_model_dp(dp_size=1)
    requests = [
        RequestInfo(query_len=8, seq_len=8, is_decode=False, image_batch_size=1, image_height=224, image_width=224),
        RequestInfo(query_len=8, seq_len=8, is_decode=False, image_batch_size=2, image_height=224, image_width=224),
    ]
    with pytest.raises(ValueError, match="uniform image parameters"):
        generate_inputs_varlen(model, requests, 128)


@pytest.mark.parametrize(
    ("partial_request", "missing_field"),
    [
        # image_batch_size set, height and width missing: the request used to
        # enter the image-bearing list, generate no image inputs, and simulate
        # as text-only with no trace (review: ChenHuiwen round 5).
        pytest.param(
            {"image_batch_size": 1},
            "height+width",
            id="batch-size-only",
        ),
        # image_batch_size and image_height set, width missing.
        pytest.param(
            {"image_batch_size": 1, "image_height": 224},
            "width",
            id="width-missing",
        ),
    ],
)
@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_rejects_partial_image_geometry(
    _mock_kv, _mock_sparse, _no_preprocessor_config, partial_request, missing_field
):
    # Partial image geometry must fail fast instead of silently degrading
    # the request (and the whole packed batch semantics) to text-only.
    model = _bailing_vl_model_dp(dp_size=1)
    requests = [
        RequestInfo(query_len=8, seq_len=8, is_decode=False, **partial_request),
    ]

    with pytest.raises(ValueError, match="partial image geometry"):
        generate_inputs_varlen(model, requests, 128)


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_resolve_multimodal_token_budget_matches_generator(_mock_kv, _mock_sparse, _no_preprocessor_config):
    from tensor_cast.core.input_generator import resolve_multimodal_token_budget

    model = _bailing_vl_model()
    # 224x224 -> 14x14 patches -> 49 merged + 2 boundary = 51 tokens per image.
    assert resolve_multimodal_token_budget(model, 1, 224, 224) == 51
    assert resolve_multimodal_token_budget(model, 3, 224, 224) == 153
    assert resolve_multimodal_token_budget(model, None, 224, 224) == 0
    assert resolve_multimodal_token_budget(model, 1, None, 224) == 0


def test_prefill_chunk_plan_counts_image_tokens_and_marks_first_chunk():
    from serving_cast.service.utils import OptimizerData

    data = OptimizerData(
        input_length=1000,
        output_length=8,
        batch_size=1,
        max_batched_tokens=512,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )

    plan = data.get_prefill_chunk_plan(image_tokens_per_request=51)

    # The request's prefill length is 1000 + 51 = 1051 tokens, so the plan
    # honors the 512 budget including the image token budget.
    assert [chunk.query_len for chunk in plan] == [512, 512, 27]
    assert [chunk.is_first_chunk for chunk in plan] == [True, False, False]
    assert plan[-1].is_last_chunk is True
    assert all(chunk.seq_len <= 1051 for chunk in plan)
    # Without the image budget the historical plan is unchanged.
    text_only = data.get_prefill_chunk_plan()
    assert [chunk.query_len for chunk in text_only] == [512, 488]


def test_prefill_chunk_plan_distribution_marks_first_chunk_per_request():
    from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

    distribution = LengthDistribution(
        bins=[LengthBin(min_tokens=1000, max_tokens=1001, weight=1.0)],
    )
    data = OptimizerData(
        length_distribution=distribution,
        output_length=8,
        batch_size=1,
        max_batched_tokens=512,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )

    plan = data.get_prefill_chunk_plan(concurrency=2, image_tokens_per_request=51)

    # Two requests of 1051 tokens each: first chunk of each request is the
    # image-bearing chunk; the boundary wave follows the shared budget.
    first_chunks = [chunk for chunk in plan if chunk.is_first_chunk]
    assert len(first_chunks) == 2
    assert first_chunks[0].query_len <= 512
    assert sum(chunk.query_len for chunk in plan) == 2 * 1051


# ---------------------------------------------------------------------------
# Review round 3 (zhangzhenyu, 2026-10-22): the chunk planner folds the image
# token budget into the chunk lengths, so the input generators must not add it
# a second time (RequestInfo.image_tokens_included contract), and the image
# forward inputs must reach exactly one chunk forward per request.
# ---------------------------------------------------------------------------


from serving_cast.service.base_throughput_optimizer import (  # noqa: E402
    BaseThroughputOptimizer,
)


class _LightThroughputOptimizer(BaseThroughputOptimizer):
    """Concrete optimizer double for crossing the real forward entries.

    Only the ``_get_batched_forward_info``/``_get_forward_info`` plumbing is
    exercised. ``model_runner`` is assigned directly (``initialize`` needs a
    fully populated parallel config the meta model fixture does not carry).
    """

    name = "light"

    def initialize(self, model_runner):
        self.model_runner = model_runner

    def get_inference_info(self, optimizer_data):  # pragma: no cover
        raise NotImplementedError


class _CapturingRunner:
    """ModelRunner double that runs the real input generator and captures it."""

    def __init__(self, model):
        self.model = model
        self.forwards = []

    def run_inference(self, requests, generate_inputs_func=None, **kwargs):
        inputs = generate_inputs_func(self.model, requests, 128)
        self.forwards.append((requests, inputs))
        return SimpleNamespace(
            execution_time_s={"analytic": 0.001},
            device_memory_available_gb=8.0,
            breakdowns={"prefill": {"Mem": 1.0}},
        )


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_chunk_forward_counts_image_tokens_once(_mock_kv, _mock_sparse, _no_preprocessor_config):
    from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

    # The reviewer's repro: text 1000 + one 224x224 image (51 language-side
    # tokens) under a 512-token chunk budget. The planner counts the budget
    # once ([512, 512, 27]); the varlen generator must NOT add 51 again on
    # the image-bearing first chunk (was [563, 512, 27] = 1102 total against
    # a 1051 final seq_len).
    model = _bailing_vl_model_dp(dp_size=1)
    data = OptimizerData(
        length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=1000, max_tokens=1001, weight=1.0)]),
        output_length=8,
        batch_size=1,
        max_batched_tokens=512,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )
    plan = data.get_prefill_chunk_plan(concurrency=1, image_tokens_per_request=51)
    assert [chunk.query_len for chunk in plan] == [512, 512, 27]

    optimizer = _LightThroughputOptimizer()
    optimizer.model_runner = _CapturingRunner(model)
    chunk_results, _ = optimizer._get_batched_forward_info(1, data, plan)

    runner = optimizer.model_runner
    assert len(runner.forwards) == 3
    queries = [inputs["attention_meta"].query_lens_values for _, inputs in runner.forwards]
    seqs = [inputs["attention_meta"].seq_lens_values for _, inputs in runner.forwards]
    # Every wave stays within the 512 budget and the per-request query sum
    # equals the final seq_len: the image budget is counted exactly once.
    assert queries == [[512], [512], [27]]
    assert seqs == [[512], [1024], [1051]]
    assert sum(query[0] for query in queries) == 1051 == seqs[-1][0]
    # The vision input reaches exactly one chunk forward (196 patch rows for
    # one 224x224 image); the follow-up chunks are text-only.
    assert runner.forwards[0][1]["pixel_values"].shape[0] == 196
    assert "pixel_values" not in runner.forwards[1][1]
    assert "pixel_values" not in runner.forwards[2][1]


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_chunk_image_budget_larger_than_one_chunk(_mock_kv, _mock_sparse, _no_preprocessor_config):
    from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

    # The image token budget (51) exceeds one chunk (30): the image-bearing
    # first chunk cannot hold the full placeholder span, so the planner now
    # raises instead of silently straddling the image placeholders across
    # chunks (review: lutean). The historical [30, 30, 1] spread let the
    # counted length budget and the single-chunk image writeback diverge.
    data = OptimizerData(
        length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=10, max_tokens=11, weight=1.0)]),
        output_length=8,
        batch_size=1,
        max_batched_tokens=30,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )
    with pytest.raises(ValueError, match="cannot hold the per-request image token budget"):
        data.get_prefill_chunk_plan(concurrency=1, image_tokens_per_request=51)


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_uniform_forward_chunk_lengths_do_not_double_count_image_tokens(
    _mock_kv, _mock_sparse, _no_preprocessor_config
):
    from serving_cast.service.utils import OptimizerData

    # Fixed-length chunk path (aggregation/disaggregation wave loops and the
    # PP chunk pipeline all land in _get_forward_info -> generate_inputs).
    # A chunk row whose lengths already include the 51-token image budget
    # must forward with query 512 (not 512 + 51 = 563) and still carry the
    # image inputs once so the vision tower runs.
    model = _bailing_vl_model_dp(dp_size=1)
    data = OptimizerData(
        input_length=1000,
        output_length=8,
        batch_size=1,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )
    optimizer = _LightThroughputOptimizer()
    optimizer.model_runner = _CapturingRunner(model)
    optimizer._get_forward_info(
        1,
        data,
        is_decode=False,
        query_len=512,
        seq_len=512,
        forward_image=(1, 224, 224),
        image_tokens_included=True,
    )

    inputs = optimizer.model_runner.forwards[0][1]
    assert inputs["attention_meta"].query_lens_values == [512]
    assert inputs["attention_meta"].seq_lens_values == [512]
    assert inputs["pixel_values"].shape[0] == 196

    # The same call with the legacy contract (text-only lengths) still grows
    # by the image tokens: the generator stays the single place that adds
    # the budget for non-chunk callers.
    optimizer.model_runner.forwards.clear()
    optimizer._get_forward_info(1, data, is_decode=False, query_len=461, seq_len=461)
    inputs = optimizer.model_runner.forwards[0][1]
    assert inputs["attention_meta"].query_lens_values == [461 + 51]
    assert inputs["pixel_values"].shape[0] == 196


# ---------------------------------------------------------------------------
# Review round 7 (zhangzhenyu, 2026-10-24, P1): the aggregation timeline must
# consume the image-aware chunk plan. It previously re-split the text-only
# length, so a single-chunk VL wave over-packed the budget (16 x 512 planned
# against a real 16 x 2554 = 40864 forward) and multi-chunk requests re-added
# the image tokens on every chunk.
# ---------------------------------------------------------------------------


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_agg_timeline_consumes_image_aware_chunk_plan(_mock_kv, _mock_sparse, _no_preprocessor_config):
    from serving_cast.service.agg_throughput_optimizer import AggThroughputOptimizer
    from serving_cast.service.scheduler import DecodeFirstWithSlack
    from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

    model = _bailing_vl_model_dp(dp_size=1)
    data = OptimizerData(
        length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=1000, max_tokens=1001, weight=1.0)]),
        output_length=8,
        batch_size=1,
        max_batched_tokens=512,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )

    optimizer = AggThroughputOptimizer()
    optimizer.model_runner = _CapturingRunner(model)
    optimizer.model_runner.user_input = SimpleNamespace(block_size=128)
    optimizer.scheduler = DecodeFirstWithSlack()
    optimizer._get_timeline_metrics(data, concurrency=1, image_tokens_per_request=51)

    timeline = optimizer._last_aggregation_timeline
    chunks = [chunk.query_len for request in timeline.requests for chunk in request.chunks]
    # The timeline admission now shares the image-aware plan: [512, 512, 27]
    # with the 51-token budget folded in once (was the text-only [512, 488]
    # re-split that let the generator re-add 51 on every chunk).
    assert chunks == [512, 512, 27]
    assert [request.prompt_length for request in timeline.requests] == [1051]

    runner = optimizer.model_runner
    prefill_queries = [
        q
        for requests, inputs in runner.forwards
        if not requests[0].is_decode
        for q in inputs["attention_meta"].query_lens_values
    ]
    # Every prefill forward stays within the 512 budget and the total query
    # equals the final seq_len: 512 + 512 + 27 = 1051, no double-counted
    # image budget (decode forwards are excluded from the sum).
    assert all(q <= 512 for q in prefill_queries)
    assert sum(prefill_queries) == 1051
    # The vision input reaches exactly one chunk forward (196 patch rows for
    # one 224x224 image); the follow-up chunks are text-only.
    pixel_waves = [inputs for _, inputs in runner.forwards if "pixel_values" in inputs]
    assert len(pixel_waves) == 1
    assert pixel_waves[0]["pixel_values"].shape[0] == 196


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_agg_timeline_single_chunk_respects_image_budget(_mock_kv, _mock_sparse, _no_preprocessor_config):
    from serving_cast.service.agg_throughput_optimizer import AggThroughputOptimizer
    from serving_cast.service.scheduler import DecodeFirstWithSlack
    from serving_cast.service.utils import OptimizerData

    model = _bailing_vl_model_dp(dp_size=1)
    data = OptimizerData(
        input_length=512,
        output_length=8,
        batch_size=16,
        max_batched_tokens=8192,
        image_batch_size=1,
        image_height=1080,
        image_width=1920,
    )

    optimizer = AggThroughputOptimizer()
    optimizer.model_runner = _CapturingRunner(model)
    optimizer.model_runner.user_input = SimpleNamespace(block_size=128)
    optimizer.scheduler = DecodeFirstWithSlack()
    optimizer._get_timeline_metrics(data, concurrency=16, image_tokens_per_request=2042)

    timeline = optimizer._last_aggregation_timeline
    assert len(timeline.requests) == 16
    per_request = [chunk.query_len for request in timeline.requests for chunk in request.chunks]
    # Planned per-request query is 2554 (512 text + 2042 image), not the
    # text-only 512 that previously packed 16 requests into one 8192 wave
    # whose actual forward summed to 16 x 2554 = 40864 tokens.
    assert set(per_request) == {2554}
    # With image-aware admission, every simulated prefill step stays within
    # the 8192 budget (at most 3 requests of 2554 per step).
    for shapes, _ in timeline.steps:
        if any(not shape.is_decode for shape in shapes):
            assert sum(shape.query_len for shape in shapes) <= 8192
    # Every captured forward respects the budget as well.
    runner = optimizer.model_runner
    queries = [inputs["attention_meta"].query_lens_values for _, inputs in runner.forwards]
    assert all(sum(wave) <= 8192 for wave in queries)


# ---------------------------------------------------------------------------
# Review (zhenghaojie, 2026-10-28): the PP>1 aggregation timeline must carry
# the same image contract as the PP=1 path. Previously _get_pp_timeline_metrics
# built requests from text-only lengths, so heterogeneous varlen steps dropped
# the image tokens while homogeneous steps re-added them — one report under
# two incompatible contracts.
# ---------------------------------------------------------------------------


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_agg_pp_timeline_carries_image_contract(_mock_kv, _mock_sparse, _no_preprocessor_config):
    from serving_cast.service.agg_throughput_optimizer import AggThroughputOptimizer
    from serving_cast.service.scheduler import DecodeFirstWithSlack
    from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

    model = _bailing_vl_model_dp(dp_size=1)
    data = OptimizerData(
        length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=1000, max_tokens=1001, weight=1.0)]),
        output_length=8,
        batch_size=1,
        max_batched_tokens=512,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )

    stage = SimpleNamespace(
        stage_id=0,
        kv_cache_bytes=1.0,
        indexer_cache_bytes=0.0,
        runtime_peak_bytes=1.0,
        weight_bytes=1.0,
        outgoing_payload_bytes=0.0,
        compute_time_s_by_model={"analytic": 0.001},
        outgoing_comm_time_s_by_model={"analytic": 0.0},
    )
    fake_profile = SimpleNamespace(stages=(stage, stage), transfers=(), pipeline_schedule_estimate=None, pp_size=2)
    captured = []

    def fake_get_forward_info(
        concurrency,
        optimizer_data,
        is_decode,
        *,
        query_len=None,
        seq_len=None,
        forward_image=None,
        image_tokens_included=False,
    ):
        captured.append((is_decode, query_len, forward_image, image_tokens_included))
        return SimpleNamespace(pipeline_profile=fake_profile)

    optimizer = AggThroughputOptimizer()
    optimizer.model_runner = _CapturingRunner(model)
    optimizer.model_runner.total_device_memory_gb = 64.0
    optimizer.model_runner.user_input = SimpleNamespace(block_size=128, reserved_memory_gb=0.0)
    optimizer.scheduler = DecodeFirstWithSlack()
    optimizer.pp = 2
    optimizer._get_forward_info = fake_get_forward_info

    optimizer._get_pp_timeline_metrics(data, concurrency=1, image_tokens_per_request=51)

    # The PP timeline consumes the image-aware chunk plan exactly like the
    # PP=1 path: [512, 512, 27] with the 51-token budget folded in once.
    timeline = optimizer._last_aggregation_timeline
    chunks = [chunk.query_len for request in timeline.requests for chunk in request.chunks]
    assert chunks == [512, 512, 27]
    assert [request.prompt_length for request in timeline.requests] == [1051]

    prefill_calls = [call for call in captured if not call[0]]
    assert [call[1] for call in prefill_calls] == [512, 512, 27]
    # Every prefill forward declares the included contract (lengths already
    # contain the budget — no generator-side re-add).
    assert all(call[3] for call in prefill_calls)
    # Only the first chunk's wave carries the image forward inputs; the
    # vision tower runs once per request across the PP chunk pipeline.
    assert prefill_calls[0][2] == (1, 224, 224)
    assert prefill_calls[1][2] == (None, None, None)
    assert prefill_calls[2][2] == (None, None, None)
    # Decode keeps the legacy contract: the generator adds the image token
    # count for KV accounting and drops the forward image inputs.
    decode_calls = [call for call in captured if call[0]]
    assert decode_calls
    assert all(call[2] is None and not call[3] for call in decode_calls)


# ---------------------------------------------------------------------------
# Review (zhenghaojie, 2026-10-28): resolve_multimodal_token_budget must keep
# the planner budget consistent with the dedicated Step5 generator — the
# generic Qwen-style merge formula differs from ((grid + 3) // 4)^2 by up to
# 4x and silently skews the chunk-plan budget.
# ---------------------------------------------------------------------------


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_step5_token_budget_matches_dedicated_generator(_mock_kv, _mock_sparse, _no_preprocessor_config):
    from tensor_cast.core.input_generator import generate_image_inputs, resolve_multimodal_token_budget

    vision_config = SimpleNamespace(patch_size=16)
    model = SimpleNamespace(
        model_id="step5-vl-test",
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="step5_vl", vision_config=vision_config),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            dtype=torch.float32,
        ),
    )

    # 1024x1024 image on a 16-pixel grid: 64x64 patches, two stride-2
    # downsamplers -> 16x16 = 256 tokens + 2 boundary tokens per image.
    budget = resolve_multimodal_token_budget(model, 2, 1024, 1024)
    assert budget == 2 * (16 * 16 + 2) == 516

    inputs = generate_image_inputs(model, 2, 1024, 1024, concurrency=1)
    assert inputs["num_image_tokens"] == budget


# ---------------------------------------------------------------------------
# Review (AvadaKedavrua, 2026-10-29): the step5 budget must share the
# generator's divisibility pre-check (both accept or both reject), and the
# image-carrying chunk must be marked at the source (RequestShape.carries_image)
# with the image geometry folded into the timeline caches.
# ---------------------------------------------------------------------------


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_step5_token_budget_rejects_non_divisible_geometry(_mock_kv, _mock_sparse, _no_preprocessor_config):
    from tensor_cast.core.input_generator import generate_image_inputs, resolve_multimodal_token_budget

    vision_config = SimpleNamespace(patch_size=16)
    model = SimpleNamespace(
        model_id="step5-vl-test",
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="step5_vl", vision_config=vision_config),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            dtype=torch.float32,
        ),
    )

    # 1000 is not a multiple of the 16-pixel patch: the generator raises, so
    # the budget resolver must raise too instead of silently floor-dividing
    # to a budget that misses the last row/column (review: AvadaKedavrua).
    with pytest.raises(ValueError, match="multiples of the vision patch size"):
        resolve_multimodal_token_budget(model, 1, 1000, 1024)
    with pytest.raises(ValueError, match="multiples of the vision patch size"):
        generate_image_inputs(model, 1, 1024, 1000, concurrency=1)
    with pytest.raises(ValueError, match="multiples of the vision patch size"):
        resolve_multimodal_token_budget(model, 1, 1024, 1000)


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_timeline_requests_mark_first_chunk_carries_image(_mock_kv, _mock_sparse, _no_preprocessor_config):
    from serving_cast.service.aggregation_timeline import build_requests
    from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

    data = OptimizerData(
        length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=1000, max_tokens=1001, weight=1.0)]),
        output_length=8,
        batch_size=1,
        max_batched_tokens=512,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )

    # Image-bearing request: only the first chunk is marked; decode and
    # follow-up chunks stay False. The flag is set at build_requests time
    # (review: AvadaKedavrua) so a non-first chunk sharing (query_len,
    # seq_len) with another request's first chunk can never be
    # mis-classified by a reverse lookup.
    requests = build_requests(data, concurrency=2, image_tokens_per_request=51)
    for request in requests:
        assert [chunk.carries_image for chunk in request.chunks] == [True, False, False]

    # Text-only requests never carry the image flag.
    text_requests = build_requests(data, concurrency=1)
    for request in text_requests:
        assert all(not chunk.carries_image for chunk in request.chunks)


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_agg_pp_step_cache_separates_image_geometry(_mock_kv, _mock_sparse, _no_preprocessor_config):
    """The PP step cache survives across search candidates, so its key must
    include the image geometry: a second candidate with a different image
    size must not reuse the first candidate's profile (review:
    AvadaKedavrua).
    """
    from serving_cast.service.agg_throughput_optimizer import AggThroughputOptimizer
    from serving_cast.service.scheduler import DecodeFirstWithSlack
    from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

    def _make_data(image_height, image_width):
        return OptimizerData(
            length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=1000, max_tokens=1001, weight=1.0)]),
            output_length=8,
            batch_size=1,
            max_batched_tokens=512,
            image_batch_size=1,
            image_height=image_height,
            image_width=image_width,
        )

    stage = SimpleNamespace(
        stage_id=0,
        kv_cache_bytes=1.0,
        indexer_cache_bytes=0.0,
        runtime_peak_bytes=1.0,
        weight_bytes=1.0,
        outgoing_payload_bytes=0.0,
        compute_time_s_by_model={"analytic": 0.001},
        outgoing_comm_time_s_by_model={"analytic": 0.0},
    )
    fake_profile = SimpleNamespace(stages=(stage, stage), transfers=(), pipeline_schedule_estimate=None, pp_size=2)
    captured = []

    def fake_get_forward_info(
        concurrency,
        optimizer_data,
        is_decode,
        *,
        query_len=None,
        seq_len=None,
        forward_image=None,
        image_tokens_included=False,
    ):
        captured.append((is_decode, query_len, forward_image))
        return SimpleNamespace(pipeline_profile=fake_profile)

    model = _bailing_vl_model_dp(dp_size=1)
    optimizer = AggThroughputOptimizer()
    optimizer.model_runner = _CapturingRunner(model)
    optimizer.model_runner.total_device_memory_gb = 64.0
    optimizer.model_runner.user_input = SimpleNamespace(block_size=128, reserved_memory_gb=0.0)
    optimizer.scheduler = DecodeFirstWithSlack()
    optimizer.pp = 2
    optimizer._get_forward_info = fake_get_forward_info

    first = len(captured)
    optimizer._get_pp_timeline_metrics(_make_data(224, 224), concurrency=1, image_tokens_per_request=51)
    calls_first = len(captured) - first

    # Same geometry: every step hits the cache, no new forward calls.
    optimizer._get_pp_timeline_metrics(_make_data(224, 224), concurrency=1, image_tokens_per_request=51)
    assert len(captured) - first == calls_first

    # A different image height must re-profile the image-bearing step: the
    # cached profile of the 224-high candidate is not reused.
    optimizer._get_pp_timeline_metrics(_make_data(336, 224), concurrency=1, image_tokens_per_request=51)
    assert len(captured) - first > calls_first
    image_calls = [call for call in captured[first:] if call[2] is not None and call[2][0] is not None]
    assert image_calls and image_calls[-1][2] == (1, 336, 224)


# ---------------------------------------------------------------------------
# Review (lutean, 2026-10-30): (a) image_batch_size must be resolved at the
# timeline entries (the raw None desynchronized the counted budget from the
# image inputs across the homogeneous/heterogeneous/PP paths); (b) the
# image-bearing first chunk must hold the full placeholder span — move to a
# fresh wave when the leftover budget is too small, raise when the chunk cap
# itself is too small.
# ---------------------------------------------------------------------------


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_agg_pp_timeline_resolves_default_image_batch_size(_mock_kv, _mock_sparse, _no_preprocessor_config):
    """Without an explicit image_batch_size the distribution default is 1:
    the PP timeline must forward the resolved count, not the raw None the
    user left unset (review: lutean — the budget planner already resolved
    it, so the inputs and the counted length must agree on all paths).
    """
    from serving_cast.service.agg_throughput_optimizer import AggThroughputOptimizer
    from serving_cast.service.scheduler import DecodeFirstWithSlack
    from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

    model = _bailing_vl_model_dp(dp_size=1)
    data = OptimizerData(
        length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=1000, max_tokens=1001, weight=1.0)]),
        output_length=8,
        batch_size=1,
        max_batched_tokens=512,
        # image_batch_size intentionally left unset: the distribution default
        # (1 image per request) applies.
        image_height=224,
        image_width=224,
    )

    stage = SimpleNamespace(
        stage_id=0,
        kv_cache_bytes=1.0,
        indexer_cache_bytes=0.0,
        runtime_peak_bytes=1.0,
        weight_bytes=1.0,
        outgoing_payload_bytes=0.0,
        compute_time_s_by_model={"analytic": 0.001},
        outgoing_comm_time_s_by_model={"analytic": 0.0},
    )
    fake_profile = SimpleNamespace(stages=(stage, stage), transfers=(), pipeline_schedule_estimate=None, pp_size=2)
    captured = []

    def fake_get_forward_info(
        concurrency,
        optimizer_data,
        is_decode,
        *,
        query_len=None,
        seq_len=None,
        forward_image=None,
        image_tokens_included=False,
    ):
        captured.append((is_decode, query_len, forward_image))
        return SimpleNamespace(pipeline_profile=fake_profile)

    optimizer = AggThroughputOptimizer()
    optimizer.model_runner = _CapturingRunner(model)
    optimizer.model_runner.total_device_memory_gb = 64.0
    optimizer.model_runner.user_input = SimpleNamespace(block_size=128, reserved_memory_gb=0.0)
    optimizer.scheduler = DecodeFirstWithSlack()
    optimizer.pp = 2
    optimizer._get_forward_info = fake_get_forward_info

    optimizer._get_pp_timeline_metrics(data, concurrency=1, image_tokens_per_request=51)

    prefill_calls = [call for call in captured if not call[0]]
    assert prefill_calls
    # The image-bearing first chunk forwards the resolved per-request count
    # (1 for the distribution default), never the raw None.
    assert prefill_calls[0][2] == (1, 224, 224)
    # Text-only follow-up chunks carry the explicit all-None triple.
    assert all(call[2] == (None, None, None) for call in prefill_calls[1:])


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_prefill_chunk_plan_moves_image_first_chunk_to_fresh_wave(_mock_kv, _mock_sparse, _no_preprocessor_config):
    from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

    # Two requests of 10 text + 51 image tokens (61 each) under a 70-token
    # wave budget: the first request consumes 61 and leaves 9 — too small
    # for the second request's image-bearing first chunk, which must start
    # a fresh wave (index 1) instead of straddling its image budget across
    # chunks (review: lutean).
    data = OptimizerData(
        length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=10, max_tokens=11, weight=1.0)]),
        output_length=8,
        batch_size=1,
        max_batched_tokens=70,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )
    plan = data.get_prefill_chunk_plan(concurrency=2, image_tokens_per_request=51)

    assert [chunk.query_len for chunk in plan] == [61, 61]
    assert [chunk.index for chunk in plan] == [0, 1]
    assert [chunk.is_first_chunk for chunk in plan] == [True, True]
    # The moved first chunk keeps the full image span within its chunk.
    assert all(chunk.query_len >= 51 for chunk in plan)


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_timeline_build_requests_raises_when_image_budget_exceeds_chunk(
    _mock_kv, _mock_sparse, _no_preprocessor_config
):
    from serving_cast.service.aggregation_timeline import build_requests
    from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

    data = OptimizerData(
        length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=10, max_tokens=11, weight=1.0)]),
        output_length=8,
        batch_size=1,
        max_batched_tokens=30,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )
    with pytest.raises(ValueError, match="cannot hold the per-request image token budget"):
        build_requests(data, concurrency=1, image_tokens_per_request=51)

    # The fixed-length planner branch raises the same way.
    fixed = OptimizerData(
        input_length=10,
        output_length=8,
        batch_size=1,
        max_batched_tokens=30,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )
    with pytest.raises(ValueError, match="cannot hold the per-request image token budget"):
        fixed.get_prefill_chunk_plan(image_tokens_per_request=51)

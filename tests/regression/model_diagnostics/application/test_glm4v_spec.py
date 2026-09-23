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

"""GLM-4V model-diagnostics spec guards."""

# pylint: disable=protected-access

from dataclasses import replace

import pytest

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import (
    OUTPUT,
    ExecutionPhase,
    ModelRunContext,
    ParallelContext,
    SourceKind,
    TensorDirection,
)
from tools.model_diagnostics.organization.theory import build_theory_regions, flatten_theory_calls
from tools.model_diagnostics.specification.context_env import build_theory_env
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.sources.runtime_capture import _is_moe_config


class _Glm4vMoeTextConfig:
    n_routed_experts = 128
    num_experts_per_tok = 8


class _Glm4vMoeConfig:
    model_type = "glm4v_moe"

    def get_text_config(self):
        return _Glm4vMoeTextConfig()


def _glm4v_context(model_type: str) -> ModelRunContext:
    model_config = {
        "model_type": model_type,
        "hidden_size": 4096,
        "intermediate_size": 13696,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "num_hidden_layers": 46,
        "effective_num_hidden_layers": 2,
        "vocab_size": 151552,
        "head_dim": 128,
        "vision_hidden_size": 1536,
        "vision_intermediate_size": 13696,
        "vision_num_hidden_layers": 24,
        "vision_patch_size": 14,
        "vision_spatial_merge_size": 2,
        "vision_temporal_patch_size": 2,
        "vision_in_channels": 3,
        "vision_out_hidden_size": 4096,
        "image_batch_size": 1,
        "image_height": 1080,
        "image_width": 1920,
        "image_resized_height": 1092,
        "image_resized_width": 1932,
    }
    if model_type == "glm4v_moe":
        model_config.update(
            {
                "intermediate_size": 10944,
                "num_attention_heads": 96,
                "num_key_value_heads": 8,
                "vision_intermediate_size": 10944,
                "first_k_dense_replace": 1,
                "n_shared_experts": 1,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 1408,
            }
        )
    return ModelRunContext(
        model_name="zai-org/GLM-4.5V" if model_type == "glm4v_moe" else "zai-org/GLM-4.1V-9B-Thinking",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=30,
        context_length=0,
        parallel=ParallelContext(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            expert_parallel_size=1,
            moe_data_parallel_size=1,
        ),
        model_config=model_config,
        quantization_config={"quantize_linear_action": "DISABLED"},
    )


def _output_shape(call_name: str, call_by_name: dict[str, object]) -> tuple[int, ...]:
    call = call_by_name[call_name]
    for tensor in call.tensors:
        if tensor.slot.direction is TensorDirection.OUTPUT and tensor.slot.index == 0:
            assert tensor.shape is not None
            return tensor.shape
    raise AssertionError(f"{call_name} does not have OUTPUT[0]")


def test_glm4v_moe_text_config_with_n_routed_experts_is_detected_as_moe() -> None:
    assert _is_moe_config(_Glm4vMoeConfig())


def test_glm4v_dense_spec_matches_only_glm4v_model_type() -> None:
    app = create_model_diagnostics_application()
    spec = app.spec_provider.get(_glm4v_context("glm4v"))
    regions = {region.region_id: region for region in spec.regions}
    language = regions["language"]
    layer_kind = language.layer_layout[0]
    stages = tuple(stage.stage_id for stage in language.layer_specs[layer_kind].stages)

    assert spec.spec_id == "glm4v_v1"
    assert language.layer_layout == ("glm4v_dense_text_decoder", "glm4v_dense_text_decoder")
    assert "dense_ffn" in stages
    stage_specs = {stage.stage_id: stage for stage in language.layer_specs[layer_kind].stages}
    assert stage_specs["attention_qkv"].source_options[SourceKind.RUNTIME].boundary_operators == ("rms_norm",)
    assert stage_specs["dense_ffn"].source_options[SourceKind.RUNTIME].boundary_operators == ("rms_norm",)


def test_glm4v_moe_spec_uses_dense_first_layer_then_moe_layers() -> None:
    app = create_model_diagnostics_application()
    spec = app.spec_provider.get(_glm4v_context("glm4v_moe"))
    regions = {region.region_id: region for region in spec.regions}
    language = regions["language"]

    assert spec.spec_id == "glm4v_moe_v1"
    assert language.layer_layout == ("glm4v_dense_text_decoder", "glm4v_moe_text_decoder")
    assert "dense_ffn" in tuple(stage.stage_id for stage in language.layer_specs["glm4v_dense_text_decoder"].stages)
    assert tuple(stage.stage_id for stage in language.layer_specs["glm4v_moe_text_decoder"].stages) == (
        "attention_qkv",
        "attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "shared_expert",
    )


def test_glm4v_dense_and_moe_specs_share_source_accurate_vision_layout() -> None:
    app = create_model_diagnostics_application()

    for model_type in ("glm4v", "glm4v_moe"):
        spec = app.spec_provider.get(_glm4v_context(model_type))
        regions = {region.region_id: region for region in spec.regions}
        vision = regions["vision_encoder"]

        assert vision.layer_layout == ("glm4v_vision_block",) * 24
        assert tuple(stage.stage_id for stage in vision.layer_specs["glm4v_vision_block"].stages) == (
            "vision_attention_qkv",
            "vision_attention",
            "vision_o_projection",
            "vision_mlp_gate",
            "vision_mlp_activation",
            "vision_mlp_up",
            "vision_mlp_multiply",
            "vision_mlp_down",
        )
        assert tuple(stage.stage_id for stage in regions["vision_merger"].stages) == (
            "vision_post_layernorm",
            "vision_downsample",
            "vision_merger_projection",
            "vision_merger_mlp",
        )
        assert tuple(stage.stage_id for stage in regions["vision_prelude"].stages) == (
            "vision_patch_embedding",
            "vision_post_conv_layernorm",
            "vision_position_interpolation",
        )


@pytest.mark.parametrize("model_type", ("glm4v", "glm4v_moe"))
def test_glm4v_all_default_stage_regions_have_theory(model_type: str) -> None:
    app = create_model_diagnostics_application()
    context = _glm4v_context(model_type)
    spec = app.spec_provider.get(context)
    selected_stage_regions = tuple(region.region_id for region in spec.regions if region.stages)

    regions = build_theory_regions(
        context,
        spec,
        selected_layers={},
        selected_stage_regions=selected_stage_regions,
    )

    assert tuple(region.region_id for region in regions) == selected_stage_regions
    assert "vision_setup" not in selected_stage_regions


def test_glm4v_theory_requires_runtime_resize_metadata() -> None:
    context = _glm4v_context("glm4v")
    incomplete_config = dict(context.model_config)
    incomplete_config.pop("image_resized_height")
    incomplete_config.pop("image_resized_width")

    with pytest.raises(SpecificationLoadError, match="materialized by Runtime capture"):
        build_theory_env(replace(context, model_config=incomplete_config))


@pytest.mark.parametrize(
    ("model_type", "vision_intermediate_size"),
    (("glm4v", 13696), ("glm4v_moe", 10944)),
)
def test_glm4v_vision_theory_shapes_match_hf_source_structure(
    model_type: str,
    vision_intermediate_size: int,
) -> None:
    app = create_model_diagnostics_application()
    context = _glm4v_context(model_type)
    spec = app.spec_provider.get(context)

    regions = build_theory_regions(
        context,
        spec,
        selected_layers={"vision_encoder": (0,)},
        selected_stage_regions=("vision_merger",),
    )
    calls = flatten_theory_calls(regions)
    names = tuple(call.operator_name for call in calls)

    assert names == (
        "vision_qkv_projection",
        "vision_attention",
        "vision_o_projection",
        "vision_mlp_gate_projection",
        "silu",
        "vision_mlp_up_projection",
        "mul",
        "vision_mlp_down_projection",
        "vision_post_block_normalize",
        "vision_post_block_layernorm",
        "vision_downsample",
        "vision_merger_projection",
        "native_layer_norm",
        "gelu",
        "vision_merger_gate_projection",
        "silu",
        "vision_merger_up_projection",
        "mul",
        "vision_merger_down_projection",
    )
    output_shapes = tuple(tensor.shape for call in calls for tensor in call.tensors if tensor.slot == OUTPUT[0])
    assert output_shapes[:8] == (
        (10764, 4608),
        (10764, 1536),
        (10764, 1536),
        (10764, 4096),
        (10764, 4096),
        (10764, 4096),
        (10764, 4096),
        (10764, 1536),
    )
    assert output_shapes[8:] == (
        (10764, 1536),
        (10764, 1536),
        (2691, 4096, 1, 1),
        (2691, 4096),
        (2691, 4096),
        (2691, 4096),
        (2691, vision_intermediate_size),
        (2691, vision_intermediate_size),
        (2691, vision_intermediate_size),
        (2691, vision_intermediate_size),
        (2691, 4096),
    )


def test_glm4v_moe_theory_shapes_include_shared_expert() -> None:
    app = create_model_diagnostics_application()
    context = _glm4v_context("glm4v_moe")
    spec = app.spec_provider.get(context)

    regions = build_theory_regions(
        context,
        spec,
        selected_layers={"language": (1,)},
        selected_stage_regions=("input", "output"),
    )
    calls = flatten_theory_calls(regions)
    call_by_name = {call.operator_name: call for call in calls}

    assert tuple(call.operator_name for call in calls)[6:15] == (
        "moe_gate_linear",
        "moe_gating_top_k_softmax",
        "init_routing_v2",
        "grouped_matmul_swiglu",
        "grouped_matmul",
        "unpermute_tokens",
        "mul",
        "sum",
        "shared_gate_up_projection",
    )
    assert _output_shape("moe_gate_linear", call_by_name) == (2723, 128)
    assert _output_shape("grouped_matmul_swiglu", call_by_name) == (21784, 1408)
    assert _output_shape("shared_gate_up_projection", call_by_name) == (2723, 2816)
    assert _output_shape("shared_swiglu", call_by_name) == (2723, 1408)
    assert _output_shape("shared_down_projection", call_by_name) == (2723, 4096)

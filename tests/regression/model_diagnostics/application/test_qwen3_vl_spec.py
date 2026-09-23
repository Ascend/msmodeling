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

"""Qwen3-VL vision-encoder Theory layout and shape guards."""

from dataclasses import replace

import pytest

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import OUTPUT, ExecutionPhase, ModelRunContext, ParallelContext
from tools.model_diagnostics.organization.theory import build_theory_regions, flatten_theory_calls
from tools.model_diagnostics.specification.context_env import build_theory_env
from tools.model_diagnostics.specification.errors import SpecificationLoadError


def _qwen3_vl_context() -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=128,
        context_length=0,
        parallel=ParallelContext(),
        model_config={
            "model_type": "qwen3_vl",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 36,
            "effective_num_hidden_layers": 1,
            "vocab_size": 151936,
            "head_dim": 128,
            "vision_hidden_size": 1152,
            "vision_intermediate_size": 4304,
            "vision_num_hidden_layers": 27,
            "vision_layer_kinds": tuple(
                "qwen3_vl_deepstack_vision_block" if index in {8, 16, 24} else "qwen3_vl_vision_block"
                for index in range(27)
            ),
            "vision_patch_size": 16,
            "vision_spatial_merge_size": 2,
            "vision_temporal_patch_size": 2,
            "vision_in_channels": 3,
            "vision_out_hidden_size": 4096,
            "vision_patch_tokens": 256,
            "vision_projector_tokens": 64,
            "vision_text_tokens": 66,
            "image_batch_size": 1,
            "image_height": 224,
            "image_width": 224,
            "torch_dtype": "float16",
            "num_mtp_tokens": 0,
        },
        quantization_config={},
    )


def test_qwen3_vl_theory_rejects_unmaterialized_visual_geometry() -> None:
    context = _qwen3_vl_context()
    model_config = dict(context.model_config)
    for key in ("vision_patch_tokens", "vision_projector_tokens", "vision_text_tokens"):
        model_config.pop(key)

    with pytest.raises(SpecificationLoadError, match="must be materialized by Runtime capture"):
        build_theory_env(replace(context, model_config=model_config))


def test_qwen3_vl_theory_derives_tokens_from_materialized_resize_shape() -> None:
    context = _qwen3_vl_context()
    model_config = dict(context.model_config)
    for key in ("vision_patch_tokens", "vision_projector_tokens", "vision_text_tokens"):
        model_config.pop(key)
    model_config.update(
        {
            "image_resized_height": 256,
            "image_resized_width": 256,
        }
    )

    env = build_theory_env(replace(context, model_config=model_config))

    assert env["VTok"] == 256
    assert env["VProjTok"] == 64
    assert env["Q"] == 194
    assert {"ViB", "VgH", "VgW", "VIn"}.isdisjoint(env)


def test_qwen3_vl_vision_layout_marks_configured_deepstack_layers() -> None:
    app = create_model_diagnostics_application()
    context = _qwen3_vl_context()

    spec = app.spec_provider.get(context)
    regions = {region.region_id: region for region in spec.regions}
    vision = regions["vision_encoder"]

    assert spec.spec_id == "qwen3_vl_v1"
    assert len(vision.layer_layout) == 27
    assert tuple(
        index for index, layer_kind in enumerate(vision.layer_layout) if layer_kind == "qwen3_vl_deepstack_vision_block"
    ) == (8, 16, 24)
    assert tuple(stage.stage_id for stage in vision.layer_specs["qwen3_vl_vision_block"].stages) == (
        "vision_attention_qkv",
        "vision_attention",
        "vision_mlp",
    )
    assert tuple(stage.stage_id for stage in vision.layer_specs["qwen3_vl_deepstack_vision_block"].stages) == (
        "vision_attention_qkv",
        "vision_attention",
        "vision_mlp",
        "vision_deepstack_merger",
    )
    assert tuple(stage.stage_id for stage in regions["vision_merger"].stages) == ("vision_final_merger",)


def test_qwen3_vl_all_default_stage_regions_have_theory() -> None:
    app = create_model_diagnostics_application()
    context = _qwen3_vl_context()
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


def test_qwen3_vl_vision_theory_shapes_match_hf_source_structure() -> None:
    app = create_model_diagnostics_application()
    context = _qwen3_vl_context()
    spec = app.spec_provider.get(context)

    regions = build_theory_regions(
        context,
        spec,
        selected_layers={"vision_encoder": (0, 8, 26)},
        selected_stage_regions=("vision_merger",),
    )
    calls = flatten_theory_calls(regions)
    names = tuple(call.operator_name for call in calls)

    block_names = (
        "vision_qkv_projection",
        "vision_attention",
        "vision_o_projection",
        "vision_mlp_fc1",
        "gelu",
        "vision_mlp_fc2",
    )
    merger_names = ("vision_merger_fc1", "gelu", "vision_merger_fc2")
    assert names == block_names + block_names + merger_names + block_names + merger_names

    output_shapes = tuple(tensor.shape for call in calls for tensor in call.tensors if tensor.slot == OUTPUT[0])
    assert output_shapes[:6] == (
        (256, 3456),
        (256, 1152),
        (256, 1152),
        (256, 4304),
        (256, 4304),
        (256, 1152),
    )
    assert output_shapes[12:15] == (
        (64, 4608),
        (64, 4608),
        (64, 4096),
    )
    assert output_shapes[-3:] == (
        (64, 4608),
        (64, 4608),
        (64, 4096),
    )


def test_qwen3_vl_vision_layout_rejects_kind_sequence_with_wrong_length() -> None:
    context = _qwen3_vl_context()
    invalid = ModelRunContext(
        model_name=context.model_name,
        entrypoint=context.entrypoint,
        phase=context.phase,
        batch_size=context.batch_size,
        query_length=context.query_length,
        context_length=context.context_length,
        parallel=context.parallel,
        model_config={
            **context.model_config,
            "vision_layer_kinds": context.model_config["vision_layer_kinds"][:-1],
        },
        quantization_config=context.quantization_config,
    )

    with pytest.raises(SpecificationLoadError, match="list length 26 must equal count 27"):
        create_model_diagnostics_application().spec_provider.get(invalid)

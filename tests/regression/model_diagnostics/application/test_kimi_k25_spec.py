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
"""Kimi K2.5 model-diagnostics spec guards."""

from dataclasses import replace

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import OUTPUT, ExecutionPhase, ModelRunContext, ParallelContext, SourceKind
from tools.model_diagnostics.organization.theory import build_theory_regions, flatten_theory_calls
from tools.model_diagnostics.specification.context_env import build_theory_env


def _context(*, with_image: bool = True, batch_size: int = 1, image_batch_size: int = 1) -> ModelRunContext:
    local_image_batch = batch_size * image_batch_size
    image = (
        {
            "image_batch_size": image_batch_size,
            "image_height": 1080,
            "image_width": 1920,
            "image_resized_height": 1092,
            "image_resized_width": 1932,
            "vision_grid_t": 1,
            "vision_grid_h": 78,
            "vision_grid_w": 138,
            "vision_patch_tokens": local_image_batch * 10764,
            "vision_projector_tokens": local_image_batch * 2691,
            "vision_text_tokens": image_batch_size * 2693,
        }
        if with_image
        else {}
    )
    return ModelRunContext(
        model_name="moonshotai/Kimi-K2.5",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=batch_size,
        query_length=30,
        context_length=0,
        parallel=ParallelContext(tensor_parallel_size=8, expert_parallel_size=8),
        model_config={
            "model_type": "kimi_k25",
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "num_hidden_layers": 61,
            "effective_num_hidden_layers": 2,
            "vocab_size": 163840,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_experts": 384,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
            "first_k_dense_replace": 1,
            "vision_hidden_size": 1152,
            "vision_intermediate_size": 4304,
            "vision_num_hidden_layers": 27,
            "vision_patch_size": 14,
            "vision_spatial_merge_size": 2,
            "vision_temporal_patch_size": 1,
            "vision_in_channels": 3,
            "vision_out_hidden_size": 7168,
            **image,
        },
        quantization_config={"quantize_linear_action": "W4A8_DYNAMIC", "linear_input_dtype": "int8"},
    )


def test_kimi_k25_matches_unique_spec_and_dense_then_moe_layout() -> None:
    spec = create_model_diagnostics_application().spec_provider.get(_context(with_image=False))
    language = next(region for region in spec.regions if region.region_id == "language")

    assert spec.spec_id == "kimi_k25_v1"
    assert language.layer_layout == ("kimi_k25_dense_decoder", "kimi_k25_moe_decoder")


def test_kimi_k25_theory_env_exposes_moonvit_only_symbols() -> None:
    env = build_theory_env(_context())

    assert env["ViB"] == 1
    assert env["VgT"] == 1
    assert env["VgH"] == 78
    assert env["VgW"] == 138
    assert env["VProjImg"] == 2691
    assert env["VIn"] == 588


def test_kimi_k25_geometry_fallback_uses_rank_local_token_totals() -> None:
    context = _context(batch_size=2)
    model_config = dict(context.model_config)
    for key in ("vision_patch_tokens", "vision_projector_tokens", "vision_text_tokens"):
        model_config.pop(key)

    env = build_theory_env(replace(context, model_config=model_config))

    assert env["ViB"] == 2
    assert env["VTok"] == 2 * 10764
    assert env["VProjTok"] == 2 * 2691
    assert env["VProjImg"] == 2691


def test_kimi_k25_image_batch_is_independent_from_request_batch() -> None:
    env = build_theory_env(_context(batch_size=2, image_batch_size=3))

    assert env["ViB"] == 6
    assert env["VTok"] == 6 * 10764
    assert env["VProjTok"] == 6 * 2691


def test_kimi_k25_tpool_repeats_per_image_with_single_image_shapes() -> None:
    context = _context(batch_size=2)
    spec = create_model_diagnostics_application().spec_provider.get(context)
    postlude = next(region for region in spec.regions if region.region_id == "vision_postlude")
    tpool = next(stage for stage in postlude.stages if stage.stage_id == "tpool_patch_merger")
    assert tpool.source_options[SourceKind.THEORY].repeat == "ViB"
    calls = flatten_theory_calls(
        build_theory_regions(
            context,
            spec,
            selected_layers={},
            selected_stage_regions=("vision_postlude",),
        )
    )

    means = [call for call in calls if call.operator_name == "tpool_mean"]
    mergers = [call for call in calls if call.operator_name == "tpool_patch_merger"]
    assert len(means) == len(mergers) == 2
    assert means[0].tensors[0].shape == (1, 39, 69, 2, 2, 1152)
    assert mergers[0].tensors[-1].shape == (2691, 4, 1152)


def test_kimi_k25_all_default_stage_regions_have_theory() -> None:
    context = _context()
    spec = create_model_diagnostics_application().spec_provider.get(context)
    selected_stage_regions = tuple(region.region_id for region in spec.regions if region.stages)

    regions = build_theory_regions(
        context,
        spec,
        selected_layers={},
        selected_stage_regions=selected_stage_regions,
    )

    assert tuple(region.region_id for region in regions) == selected_stage_regions
    assert "vision_setup" not in selected_stage_regions


def test_kimi_k25_theory_uses_mla_moe_and_moonvit_shapes() -> None:
    context = _context()
    spec = create_model_diagnostics_application().spec_provider.get(context)
    calls = flatten_theory_calls(
        build_theory_regions(
            context,
            spec,
            selected_layers={"vision_encoder": (0, 26), "language": (0, 1)},
            selected_stage_regions=("input", "vision_prelude", "vision_postlude", "output"),
        )
    )
    by_name = {}
    for call in calls:
        by_name.setdefault(call.operator_name, []).append(call)

    assert by_name["vision_patch_embedding"][0].tensors[-1].shape == (10764, 1152)
    assert len(by_name["vision_qkv_projection"]) == 2
    assert by_name["vision_final_norm"][0].tensors[-1].shape == (10764, 1152)
    assert by_name["tpool_patch_merger"][0].tensors[-1].shape == (2691, 4, 1152)
    assert by_name["vision_projector_fc0"][0].tensors[-1].shape == (2691, 4608)
    assert by_name["vision_projector_fc1"][0].tensors[-1].shape == (2691, 7168)
    assert by_name["mlapo"][0].tensors[1].shape == (2723, 8, 192)
    assert len(by_name["mlapo"]) == 2
    assert by_name["moe_gate_linear"][0].tensors[-1].shape == (341, 384)
    assert by_name["grouped_matmul_swiglu"][0].tensors[-1].shape == (3024, 2048)
    assert by_name["grouped_matmul"][0].tensors[-1].shape == (3024, 7168)
    assert by_name["lm_head"][0].tensors[-1].shape == (2723, 20480)
    assert by_name["logits_cast"][0].tensors[-1].shape == (1, 2723, 20480)
    assert by_name["logits_select"][0].tensors[-1].shape == (1, 1, 20480)


def test_kimi_k25_theory_uses_packed_batch_layout() -> None:
    context = _context(batch_size=2)
    spec = create_model_diagnostics_application().spec_provider.get(context)
    calls = flatten_theory_calls(
        build_theory_regions(
            context,
            spec,
            selected_layers={"language": (0,)},
            selected_stage_regions=("input", "output"),
        )
    )
    outputs = {
        call.source_reference: next(tensor.shape for tensor in call.tensors if tensor.slot == OUTPUT[0])
        for call in calls
        if any(tensor.slot == OUTPUT[0] for tensor in call.tensors)
    }

    # Kimi packs local requests into one leading batch dimension. Each request
    # contains 30 text tokens plus the 2693 vision placeholders.
    assert outputs["theory:input:text_embedding:embedding"] == (1, 5446, 7168)
    assert outputs["theory:language:L0:mla_preprocess:input_layer_norm"] == (1, 5446, 7168)
    assert outputs["theory:output:lm_head:logits_cast"] == (1, 5446, 20480)
    assert outputs["theory:output:lm_head:logits_select"] == (1, 2, 20480)

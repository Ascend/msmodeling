# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3-VL MoE model-diagnostics spec guards."""

# pylint: disable=protected-access

from dataclasses import replace

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, ModelRunContext, ParallelContext, SourceKind
from tools.model_diagnostics.organization.theory import build_theory_regions, flatten_theory_calls
from tools.model_diagnostics.sources.runtime_capture import _is_moe_config, _qwen3_vl_moe_layer_kinds


class _TextMoeConfig:
    num_experts = 128
    num_experts_per_tok = 8


class _TopLevelVlMoeConfig:
    model_type = "qwen3_vl_moe"

    def get_text_config(self):
        return _TextMoeConfig()


def _qwen3_vl_moe_context() -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=4,
        query_length=16,
        context_length=200,
        parallel=ParallelContext(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            expert_parallel_size=8,
            moe_data_parallel_size=1,
        ),
        model_config={
            "model_type": "qwen3_vl_moe",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "num_hidden_layers": 94,
            "effective_num_hidden_layers": 1,
            "vocab_size": 151936,
            "head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1536,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "language_layer_kinds": ("qwen3_vl_moe_text_decoder",),
            "vision_hidden_size": 1280,
            "vision_intermediate_size": 3420,
            "vision_num_hidden_layers": 32,
            "vision_patch_size": 14,
            "vision_spatial_merge_size": 2,
            "vision_temporal_patch_size": 2,
            "vision_in_channels": 3,
            "vision_out_hidden_size": 4096,
            "image_batch_size": 1,
            "image_height": 720,
            "image_width": 1080,
            # Runtime capture materializes the official HF smart-resize result.
            "image_resized_height": 728,
            "image_resized_width": 1092,
        },
        quantization_config={"quantize_linear_action": "W8A8_DYNAMIC"},
    )


def test_qwen3_vl_moe_text_config_is_detected_as_moe() -> None:
    assert _is_moe_config(_TopLevelVlMoeConfig())


def test_qwen3_vl_moe_spec_uses_moe_decoder_fragment() -> None:
    app = create_model_diagnostics_application()
    context = _qwen3_vl_moe_context()

    spec = app.spec_provider.get(context)
    regions = {region.region_id: region for region in spec.regions}
    language = regions["language"]
    layer_kind = language.layer_layout[0]
    stages = tuple(stage.stage_id for stage in language.layer_specs[layer_kind].stages)

    assert spec.spec_id == "qwen3_vl_moe_v1"
    assert stages == (
        "attention_qkv",
        "attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
    )
    stage_specs = {stage.stage_id: stage for stage in language.layer_specs[layer_kind].stages}
    assert stage_specs["attention_qkv"].source_options[SourceKind.RUNTIME].boundary_operators == ("rms_norm",)
    assert stage_specs["moe_gate"].source_options[SourceKind.RUNTIME].boundary_operators == ("rms_norm",)
    assert regions["output"].stages[0].source_options[SourceKind.RUNTIME].boundary_operators == ("rms_norm",)


def test_qwen3_vl_moe_language_layout_follows_hf_dense_moe_rule() -> None:
    context = _qwen3_vl_moe_context()
    model_config = {
        **context.model_config,
        "effective_num_hidden_layers": 6,
        "decoder_sparse_step": 2,
        "mlp_only_layers": [3],
    }
    model_config["language_layer_kinds"] = _qwen3_vl_moe_layer_kinds(
        model_config,
        start=0,
        count=6,
    )
    context = replace(context, model_config=model_config)

    spec = create_model_diagnostics_application().spec_provider.get(context)
    language = next(region for region in spec.regions if region.region_id == "language")

    assert language.layer_layout == (
        "qwen3_vl_moe_dense_text_decoder",
        "qwen3_vl_moe_text_decoder",
        "qwen3_vl_moe_dense_text_decoder",
        "qwen3_vl_moe_dense_text_decoder",
        "qwen3_vl_moe_dense_text_decoder",
        "qwen3_vl_moe_text_decoder",
    )
    assert tuple(stage.stage_id for stage in language.layer_specs["qwen3_vl_moe_dense_text_decoder"].stages) == (
        "attention_qkv",
        "attention",
        "dense_ffn",
    )


def test_qwen3_vl_moe_prefill_reuses_fine_grained_vision_fragment() -> None:
    app = create_model_diagnostics_application()
    decode = _qwen3_vl_moe_context()
    context = replace(
        decode,
        phase=ExecutionPhase.PREFILL,
        context_length=0,
        model_config={
            **decode.model_config,
            "vision_num_hidden_layers": 27,
            "vision_layer_kinds": tuple(
                "qwen3_vl_deepstack_vision_block" if index in {8, 16, 24} else "qwen3_vl_vision_block"
                for index in range(27)
            ),
        },
    )

    spec = app.spec_provider.get(context)
    regions = {region.region_id: region for region in spec.regions}
    vision = regions["vision_encoder"]

    assert len(vision.layer_layout) == 27
    assert tuple(
        index for index, layer_kind in enumerate(vision.layer_layout) if layer_kind == "qwen3_vl_deepstack_vision_block"
    ) == (8, 16, 24)
    assert tuple(stage.stage_id for stage in regions["vision_merger"].stages) == ("vision_final_merger",)


def test_qwen3_vl_moe_all_default_stage_regions_have_theory() -> None:
    app = create_model_diagnostics_application()
    context = replace(
        _qwen3_vl_moe_context(),
        phase=ExecutionPhase.PREFILL,
        context_length=0,
        model_config={
            **_qwen3_vl_moe_context().model_config,
            "vision_layer_kinds": tuple("qwen3_vl_vision_block" for _ in range(32)),
        },
    )
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


def _moe_theory_names(context: ModelRunContext) -> tuple[str, ...]:
    app = create_model_diagnostics_application()
    spec = app.spec_provider.get(context)

    regions = build_theory_regions(
        context,
        spec,
        selected_layers={"language": (0,)},
        selected_stage_regions=("input", "output"),
    )
    calls = flatten_theory_calls(regions)
    return tuple(call.operator_name for call in calls)


def test_qwen3_vl_moe_theory_shapes_materialize_with_tp8_ep8() -> None:
    names = _moe_theory_names(_qwen3_vl_moe_context())

    assert names[:10] == (
        "embedding",
        "q_projection",
        "k_projection",
        "v_projection",
        "attention",
        "o_projection",
        "moe_gate_linear",
        "topk",
        "init_routing_v2",
        "expert_gate_projection",
    )
    # The representative rank receives 15 routed pairs.  The 16 local experts
    # therefore have 15 non-empty slices, and only those emit projections.
    assert names[10:54:3] == ("expert_up_projection",) * 15
    assert names[11:55:3] == ("expert_down_projection",) * 15
    assert names[-4:] == (
        "unpermute_tokens",
        "mul",
        "sum",
        "lm_head",
    )


def test_qwen3_vl_moe_expert_calls_follow_ep_local_expert_count() -> None:
    ep1_context = replace(
        _qwen3_vl_moe_context(),
        parallel=ParallelContext(),
    )

    names = _moe_theory_names(ep1_context)

    assert names.count("expert_gate_projection") == 128
    assert names.count("expert_up_projection") == 128
    assert names.count("expert_down_projection") == 128

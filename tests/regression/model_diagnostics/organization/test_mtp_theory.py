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
"""MTP fragment composition and Theory organization tests."""

from __future__ import annotations

import pytest

from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.domain.organization import ExecutionOrganizationRequest
from tools.model_diagnostics.domain.specification import (
    ModelDiagnosticsSpec,
    RegionSpec,
    RuntimeStageOptions,
    SpecMatchCriteria,
    StageSpec,
)
from tools.model_diagnostics.organization.theory import TheoryExecutionOrganizationStrategy
from tools.model_diagnostics.sources.theory import TheoryOperatorRecordSource
from tools.model_diagnostics.specification.builtin_activation import (
    create_builtin_operator_activation_registry,
)
from tools.model_diagnostics.specification.context_env import build_theory_env
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.loader import YamlModelDiagnosticsSpecLoader
from tools.model_diagnostics.specification.run_profile import DiagnosticsRunProfile
from tools.model_diagnostics.specification.source_options import create_builtin_source_options_parsers
from tools.model_diagnostics.specification.theory_fragments import (
    TheoryFragment,
    TheoryFragmentRegistry,
    compose_mtp_layer_stages,
    load_builtin_theory_fragment_registry,
)
from tools.model_diagnostics.domain.models import (
    INPUT,
    OUTPUT,
    ExecutionPhase,
    ModelExecutionRecord,
    ModelRunContext,
    ParallelContext,
    SourceKind,
)


def _loader() -> YamlModelDiagnosticsSpecLoader:
    fragment_registry = load_builtin_theory_fragment_registry()
    return YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(
            fragment_registry=fragment_registry,
        ),
        fragment_registry=fragment_registry,
    )


def _base_config(*, layers: int = 2, num_mtp_tokens: int = 0) -> dict[str, object]:
    return {
        "model_type": "qwen3",
        "features": ["dense"],
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 36,
        "effective_num_hidden_layers": layers,
        "vocab_size": 151936,
        "torch_dtype": "bfloat16",
        "num_mtp_tokens": num_mtp_tokens,
    }


def _context(
    *,
    phase: ExecutionPhase = ExecutionPhase.DECODE,
    batch_size: int = 1,
    query_length: int = 3,
    context_length: int | None = 128,
    layers: int = 2,
    num_mtp_tokens: int = 0,
    quantization_config: dict[str, object] | None = None,
) -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen3-0.6B",
        entrypoint="text_generate",
        phase=phase,
        batch_size=batch_size,
        query_length=query_length,
        context_length=context_length,
        parallel=ParallelContext(tensor_parallel_size=1, data_parallel_size=1),
        model_config=_base_config(layers=layers, num_mtp_tokens=num_mtp_tokens),
        quantization_config=quantization_config or {},
    )


def _region(spec, region_id: str):
    return next(region for region in spec.regions if region.region_id == region_id)


def _operator_names(organized) -> list[str]:
    names: list[str] = []
    for region in organized:
        for stage in region.stages:
            names.extend(call.operator_name for call in stage.operator_calls)
        for layer in region.layers:
            for stage in layer.stages:
                names.extend(call.operator_name for call in stage.operator_calls)
    return names


def test_mtp_zero_keeps_ordinary_theory_and_empty_mtp_layout() -> None:
    loader = _loader()
    loaded = loader.load("qwen3_dense_v1")
    prefill = loader.materialize(
        loaded,
        _context(phase=ExecutionPhase.PREFILL, query_length=4, context_length=None, num_mtp_tokens=0),
    )
    decode = loader.materialize(
        loaded,
        _context(phase=ExecutionPhase.DECODE, query_length=1, num_mtp_tokens=0),
    )

    assert _region(prefill, "mtp").layer_layout == ()
    assert _region(decode, "mtp").layer_layout == ()
    assert [stage.stage_id for stage in _region(prefill, "output").stages] == ["lm_head"]
    assert [stage.stage_id for stage in _region(decode, "output").stages] == ["lm_head"]
    assert [
        operator.operator_name
        for operator in _region(prefill, "output").stages[0].source_options[SourceKind.THEORY].operators
    ] == ["lm_head_select", "lm_head"]
    assert [
        operator.operator_name
        for operator in _region(decode, "output").stages[0].source_options[SourceKind.THEORY].operators
    ] == ["lm_head"]


def test_mtp_decode_generates_independent_layers_and_shapes_b1() -> None:
    loader = _loader()
    context = _context(batch_size=1, query_length=3, layers=1, num_mtp_tokens=2)
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)
    env = build_theory_env(context)

    assert env["T"] == 3
    assert env["Rtgt"] == 3
    assert env["Rprop"] == 1
    assert _region(spec, "mtp").layer_layout == ("dense_mtp", "dense_mtp")
    assert _region(spec, "output").stages == ()
    assert [stage.stage_id for stage in _region(spec, "mtp").stages] == [
        "target_selection",
        "target_lm_head",
        "verification_sampler",
        "mtp_output",
    ]

    selected_layers = {"language": (0,), "mtp": (0, 1)}
    selected_stage_regions = ("input", "mtp")
    organized = TheoryExecutionOrganizationStrategy().execute(
        ExecutionOrganizationRequest(
            execution=TheoryOperatorRecordSource().load_execution(
                context,
                spec,
                selected_layers,
                selected_stage_regions,
            ),
            spec=spec,
            selected_layers=selected_layers,
            selected_stage_regions=selected_stage_regions,
        )
    )

    assert [region.region_id for region in organized] == [
        "input",
        "language",
        "mtp",
    ]
    mtp = next(region for region in organized if region.region_id == "mtp")
    assert [layer.layer_index for layer in mtp.layers] == [0, 1]
    assert [stage.stage_id for stage in mtp.layers[0].stages] == [
        "input_shift",
        "embedding",
        "input_fusion",
        "attention_qkv",
        "attention",
        "dense_ffn",
        "proposal_selection",
        "proposal_lm_head",
        "proposal_sampler",
    ]
    assert [stage.stage_id for stage in mtp.stages] == [
        "target_selection",
        "target_lm_head",
        "verification_sampler",
        "mtp_output",
    ]
    select = mtp.stages[0].operator_calls[0]
    assert select.operator_name == "mtp_target_select"
    assert select.tensors[0].shape == (3, 4096)
    assert select.tensors[1].shape == (3,)
    assert select.tensors[-1].shape == (3, 4096)
    assert mtp.stages[2].operator_calls[0].tensors[-1].shape == (1, 3)
    assert mtp.stages[3].operator_calls[0].tensors[0].shape == (1, 3)
    first_shift = mtp.layers[0].stages[0].operator_calls[0]
    later_shift = mtp.layers[1].stages[0].operator_calls[0]
    assert first_shift.tensors[2].shape == (1, 3)
    assert later_shift.tensors[2].shape == (1, 1)


def test_mtp_decode_shapes_scale_with_batch() -> None:
    context = _context(batch_size=2, query_length=3, layers=1, num_mtp_tokens=2)
    env = build_theory_env(context)
    assert env["T"] == 6
    assert env["Rtgt"] == 6
    assert env["Rprop"] == 2

    loader = _loader()
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)
    selected_layers = {"mtp": (0,)}
    selected_stage_regions = ("mtp",)
    organized = TheoryExecutionOrganizationStrategy().execute(
        ExecutionOrganizationRequest(
            execution=TheoryOperatorRecordSource().load_execution(
                context,
                spec,
                selected_layers,
                selected_stage_regions,
            ),
            spec=spec,
            selected_layers=selected_layers,
            selected_stage_regions=selected_stage_regions,
        )
    )
    mtp = next(region for region in organized if region.region_id == "mtp")
    assert mtp.stages[0].operator_calls[0].tensors[-1].shape == (6, 4096)
    assert mtp.stages[3].operator_calls[0].tensors[0].shape == (2, 3)


def test_illegal_mtp_window_is_rejected() -> None:
    loader = _loader()
    with pytest.raises(SpecificationLoadError, match=r"query_length >= num_mtp_tokens \+ 1"):
        loader.materialize(loader.load("qwen3_dense_v1"), _context(query_length=2, num_mtp_tokens=2))


def test_mtp_dtype_bindings_follow_quantization_env() -> None:
    quantized = build_theory_env(
        _context(
            num_mtp_tokens=2,
            quantization_config={
                "activation_dtype": "bfloat16",
                "linear_input_dtype": "int8",
                "weight_dtype": "int8",
                "output_dtype": "bfloat16",
            },
        )
    )
    assert quantized["ACT"] == "bfloat16"
    assert quantized["LINEAR_IN"] == "int8"
    assert quantized["WEIGHT"] == "int8"
    loader = _loader()
    spec = loader.materialize(
        loader.load("qwen3_dense_v1"),
        _context(
            layers=1,
            num_mtp_tokens=2,
            quantization_config={
                "activation_dtype": "bfloat16",
                "linear_input_dtype": "int8",
                "weight_dtype": "int8",
                "output_dtype": "bfloat16",
            },
        ),
    )
    mtp_region = _region(spec, "mtp")
    target_lm_head = next(stage for stage in mtp_region.stages if stage.stage_id == "target_lm_head")
    assert target_lm_head.source_options[SourceKind.THEORY].operators[0].tensors[INPUT[0]].dtype.expression == "ACT"
    mtp_stages = mtp_region.layer_specs["dense_mtp"].stages
    fusion_stage = next(stage for stage in mtp_stages if stage.stage_id == "input_fusion")
    fusion = fusion_stage.source_options[SourceKind.THEORY].operators[0]
    assert fusion.operator_name == "mtp_fusion_projection"
    assert tuple(fusion.tensors) == (INPUT[0], OUTPUT[0])
    assert fusion.tensors[INPUT[0]].shape.expression == "[T, 2 * H]"
    assert fusion.tensors[INPUT[0]].dtype.expression == "LINEAR_IN"
    assert fusion.tensors[OUTPUT[0]].shape.expression == "[T, H]"
    assert fusion.tensors[OUTPUT[0]].dtype.expression == "OUT"
    assert fusion_stage.comparisons == {}
    qkv_stage = next(stage for stage in mtp_stages if stage.stage_id == "attention_qkv")
    q_proj = qkv_stage.source_options[SourceKind.THEORY].operators[0]
    assert q_proj.tensors[INPUT[0]].dtype.expression == "LINEAR_IN"
    proposal_lm_head = next(stage for stage in mtp_stages if stage.stage_id == "proposal_lm_head")
    assert proposal_lm_head.source_options[SourceKind.THEORY].operators[0].tensors[INPUT[0]].dtype.expression == "ACT"


def test_region_isolation_and_override_do_not_resize_mtp() -> None:
    loader = _loader()
    # Capture/Context already limited language to 4 layers; Profile override is not a
    # post-materialize truncator. MTP layout stays independent at 2.
    context = _context(layers=4, num_mtp_tokens=2)
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)
    assert len(_region(spec, "language").layer_layout) == 4
    assert len(_region(spec, "mtp").layer_layout) == 2

    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="Qwen3-0.6B",
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=3,
        context_length=128,
        num_mtp_tokens=2,
        parallel=ParallelContext(),
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
    )
    request = profile.to_request(context=context, spec=spec)
    assert request.selected_layers == {
        "language": (0, 1, 2, 3),
        "mtp": (0, 1),
    }
    assert request.selected_stage_regions == ("input", "mtp")


@pytest.mark.parametrize(
    ("num_mtp_tokens", "expected_mtp_layers"),
    (
        (0, None),
        (1, (0,)),
        (2, (0, 1)),
        (4, (0, 1)),
    ),
)
def test_profile_uses_fixed_mtp_representative_layer_policy(
    num_mtp_tokens: int,
    expected_mtp_layers: tuple[int, ...] | None,
) -> None:
    loader = _loader()
    context = _context(
        query_length=max(num_mtp_tokens + 1, 1),
        layers=3,
        num_mtp_tokens=num_mtp_tokens,
    )
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="Qwen3-0.6B",
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=context.query_length,
        context_length=128,
        num_mtp_tokens=num_mtp_tokens,
        parallel=ParallelContext(),
        selected_stage_regions=(),
        num_hidden_layers_override=3,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
        selected_language_layers=(2,),
    )

    request = profile.to_request(context=context, spec=spec)

    assert request.selected_layers["language"] == (2,)
    assert request.selected_layers.get("mtp") == expected_mtp_layers


def test_unavailable_language_layers_warn_and_do_not_affect_mtp() -> None:
    loader = _loader()
    context = _context(query_length=5, layers=3, num_mtp_tokens=4)
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="Qwen3-0.6B",
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=5,
        context_length=128,
        num_mtp_tokens=4,
        parallel=ParallelContext(),
        selected_stage_regions=(),
        num_hidden_layers_override=3,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
        selected_language_layers=(1, 3, 5),
    )

    with pytest.warns(UserWarning, match=r"unavailable indices \(3, 5\)"):
        request = profile.to_request(context=context, spec=spec)

    assert request.selected_layers == {
        "language": (1,),
        "mtp": (0, 1),
    }


def test_composed_predictor_reuses_fragment_theory_and_runtime_defaults() -> None:
    loader = _loader()
    spec = loader.materialize(loader.load("qwen3_dense_v1"), _context(layers=1, num_mtp_tokens=2))
    registry = load_builtin_theory_fragment_registry()
    decoder = registry.require_kind("qwen3_dense_decoder_v1", kind="model_decoder")
    composed_stages = _region(spec, "mtp").layer_specs["dense_mtp"].stages
    for stage in composed_stages:
        assert SourceKind.THEORY in stage.source_options
        assert SourceKind.RUNTIME in stage.source_options
    composed_by_id = {stage.stage_id: stage for stage in composed_stages}
    for fragment_stage in decoder.stages:
        composed_stage = composed_by_id[fragment_stage.stage_id]
        assert composed_stage.source_options[SourceKind.THEORY].operators == fragment_stage.operators
        assert composed_stage.source_options[SourceKind.RUNTIME] == fragment_stage.runtime_options
    comparison = composed_by_id["attention_qkv"].comparisons[(SourceKind.THEORY, SourceKind.RUNTIME)]
    assert comparison.strategy_id == "concat_shape"


def test_unknown_compose_ids_fail_at_load() -> None:
    loader = _loader()
    raw = {
        "schema_version": "1",
        "spec_id": "bad_compose",
        "spec_version": "1.0.0",
        "model_category": "test",
        "matches": {"model_types": ["test"]},
        "regions": {
            "mtp": {
                "layer_layout_rule": {
                    "strategy": "repeat",
                    "layer_kind": "dense_mtp",
                    "count_from": "model_config.effective_num_mtp_layers",
                },
                "layer_specs": {
                    "dense_mtp": {
                        "compose": {
                            "framework": "missing_framework",
                            "predictor": "qwen3_dense_decoder_v1",
                        }
                    }
                },
            }
        },
    }
    with pytest.raises(SpecificationLoadError, match="unregistered theory fragment"):
        loader.load_mapping(raw)


def test_composed_runtime_options_reject_unknown_stage_id() -> None:
    loader = _loader()
    raw = {
        "schema_version": "1",
        "spec_id": "bad_runtime_stage",
        "spec_version": "1.0.0",
        "model_category": "test",
        "matches": {"model_types": ["test"]},
        "regions": {
            "mtp": {
                "layer_layout_rule": {
                    "strategy": "repeat",
                    "layer_kind": "dense_mtp",
                    "count_from": "model_config.effective_num_mtp_layers",
                },
                "layer_specs": {
                    "dense_mtp": {
                        "compose": {
                            "framework": "mtp_framework_v1",
                            "predictor": "qwen3_dense_decoder_v1",
                        },
                        "runtime_options": {
                            "unknown_stage": {
                                "boundary_operators": ["unknown"],
                            }
                        },
                    }
                },
            }
        },
    }

    with pytest.raises(SpecificationLoadError, match="unknown included stage 'unknown_stage'"):
        loader.load_mapping(raw)


def test_selected_stage_without_theory_options_still_fails() -> None:
    spec = ModelDiagnosticsSpec(
        schema_version="1",
        spec_id="strict",
        spec_version="1.0.0",
        model_category="test",
        matches=SpecMatchCriteria(model_types=("test",)),
        regions=(
            RegionSpec(
                region_id="output",
                stages=(
                    StageSpec(
                        stage_id="lm_head",
                        source_options={
                            SourceKind.RUNTIME: RuntimeStageOptions(boundary_operators=("rms_norm",)),
                        },
                    ),
                ),
            ),
        ),
    )
    context = _context(layers=1, num_mtp_tokens=0)
    execution = ModelExecutionRecord(
        source_kind=SourceKind.THEORY,
        run_context=context,
        operator_calls=(),
    )
    with pytest.raises(SpecificationLoadError, match="missing theory source_options"):
        TheoryExecutionOrganizationStrategy().execute(
            ExecutionOrganizationRequest(
                execution=execution,
                spec=spec,
                selected_layers={},
                selected_stage_regions=("output",),
            )
        )


def test_fragment_registry_exposes_stable_framework_ids() -> None:
    registry = load_builtin_theory_fragment_registry()
    framework = registry.get("mtp_framework_v1")
    assert framework.fragment_kind == "mtp_framework"
    assert [stage.stage_id for stage in framework.stages] == [
        "target_selection",
        "target_lm_head",
        "verification_sampler",
        "mtp_output",
        "input_shift",
        "embedding",
        "input_fusion",
        "proposal_selection",
        "proposal_lm_head",
        "proposal_sampler",
    ]
    assert framework.stage("proposal_selection").operators[0].tensors[INPUT[1]].shape.expression == "[Rprop]"
    assert [stage.stage_id for stage in framework.stage_group("request")] == [
        "target_selection",
        "target_lm_head",
        "verification_sampler",
        "mtp_output",
    ]
    assert [stage.stage_id for stage in framework.stage_group("proposal_prefix")] == [
        "input_shift",
        "embedding",
        "input_fusion",
    ]
    assert registry.get("qwen3_dense_decoder_v1").fragment_kind == "model_decoder"
    decoder = registry.get("qwen3_dense_decoder_v1")
    assert [stage.stage_id for stage in decoder.stages] == [
        "attention_qkv",
        "attention",
        "dense_ffn",
    ]
    assert decoder.stage("attention").operators[0].operator_name == "attention"
    assert decoder.stage("attention").runtime_options is not None
    assert decoder.stage("attention").runtime_options.boundary_operators == ("attention",)


def test_fragment_module_group_lookup_and_error() -> None:
    operator = load_builtin_theory_fragment_registry().get("mtp_framework_v1").stage("input_fusion").operators[0]
    fragment = TheoryFragment(
        fragment_id="module_group_fragment",
        fragment_kind="model_decoder",
        module_groups={"core": (operator,)},
        stages=(),
        stage_groups={},
    )

    assert fragment.group("core") == (operator,)
    with pytest.raises(SpecificationLoadError, match="has no module group 'missing'"):
        fragment.group("missing")


def test_mtp_composition_rejects_duplicate_stage_ids_across_fragments() -> None:
    builtins = load_builtin_theory_fragment_registry()
    framework = builtins.get("mtp_framework_v1")
    predictor = builtins.get("qwen3_dense_decoder_v1")
    duplicate_adapter = TheoryFragment(
        fragment_id="duplicate_adapter",
        fragment_kind="mtp_predictor_adapter",
        module_groups={},
        stages=(framework.stage("input_fusion"),),
        stage_groups={
            "before_predictor": ("input_fusion",),
            "after_predictor": (),
        },
    )
    registry = TheoryFragmentRegistry(
        {
            framework.fragment_id: framework,
            predictor.fragment_id: predictor,
            duplicate_adapter.fragment_id: duplicate_adapter,
        }
    )

    with pytest.raises(
        SpecificationLoadError,
        match="MTP composition contains duplicate stage id 'input_fusion'",
    ):
        compose_mtp_layer_stages(
            registry,
            framework_id=framework.fragment_id,
            predictor_id=predictor.fragment_id,
            predictor_adapter_id=duplicate_adapter.fragment_id,
        )


def test_fragment_loader_rejects_duplicate_stage_ids(tmp_path) -> None:
    fragment_path = tmp_path / "dup_decoder.yaml"
    fragment_path.write_text(
        """\
fragment_id: example_decoder
fragment_kind: model_decoder
stages:
  - id: attention
    modules:
      - name: attention
        tensors:
          "OUTPUT[0]":
            shape: "[T, H]"
            dtype: ACT
  - id: attention
    modules:
      - name: o_projection
        tensors:
          "OUTPUT[0]":
            shape: "[T, H]"
            dtype: OUT
""",
        encoding="utf-8",
    )

    with pytest.raises(
        SpecificationLoadError,
        match=r"theory fragment 'example_decoder'.*duplicate stage id 'attention'.*must be unique",
    ):
        load_builtin_theory_fragment_registry(fragments_dir=tmp_path)


def test_language_and_mtp_share_decoder_theory_operators() -> None:
    loader = _loader()
    spec = loader.materialize(loader.load("qwen3_dense_v1"), _context(layers=1, num_mtp_tokens=2))
    language_qkv = _region(spec, "language").layer_specs["dense"].stages[0].source_options[SourceKind.THEORY].operators
    mtp_stages = _region(spec, "mtp").layer_specs["dense_mtp"].stages
    mtp_qkv = (
        next(stage for stage in mtp_stages if stage.stage_id == "attention_qkv")
        .source_options[SourceKind.THEORY]
        .operators
    )
    assert [operator.operator_name for operator in language_qkv] == [operator.operator_name for operator in mtp_qkv]
    assert all(language.tensors == mtp.tensors for language, mtp in zip(language_qkv, mtp_qkv, strict=True))
    language_runtime = _region(spec, "language").layer_specs["dense"].stages[0].source_options[SourceKind.RUNTIME]
    mtp_runtime = next(stage for stage in mtp_stages if stage.stage_id == "attention_qkv").source_options[
        SourceKind.RUNTIME
    ]
    assert language_runtime == mtp_runtime
    assert (
        language_runtime
        == load_builtin_theory_fragment_registry().get("qwen3_dense_decoder_v1").stage("attention_qkv").runtime_options
    )


def test_loader_and_parser_share_injected_fragment_registry() -> None:
    fragment_registry = load_builtin_theory_fragment_registry()
    parsers = create_builtin_source_options_parsers(fragment_registry=fragment_registry)
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=parsers,
        fragment_registry=fragment_registry,
    )
    assert loader.fragment_registry is fragment_registry
    assert parsers["theory"].fragment_registry is fragment_registry
    assert loader.fragment_registry is parsers["theory"].fragment_registry


def test_composition_root_shares_one_fragment_registry() -> None:
    from tools.model_diagnostics.application.composition import create_model_diagnostics_application

    app = create_model_diagnostics_application()
    loader = app.spec_provider.loader
    theory_parser = loader._source_options_parsers["theory"]  # noqa: SLF001
    assert loader.fragment_registry is theory_parser.fragment_registry


def test_ordinary_decode_does_not_generate_mtp_operators() -> None:
    loader = _loader()
    context = _context(query_length=1, layers=1, num_mtp_tokens=0)
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="Qwen3-0.6B",
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=1,
        context_length=128,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_stage_regions=(),
        num_hidden_layers_override=0,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
    )
    request = profile.to_request(context=context, spec=spec)
    assert request.selected_layers == {"language": (0,)}
    assert set(request.selected_stage_regions) == {"input", "output"}
    organized = TheoryExecutionOrganizationStrategy().execute(
        ExecutionOrganizationRequest(
            execution=TheoryOperatorRecordSource().load_execution(
                context,
                spec,
                request.selected_layers,
                request.selected_stage_regions,
            ),
            spec=spec,
            selected_layers=request.selected_layers,
            selected_stage_regions=request.selected_stage_regions,
        )
    )
    names = _operator_names(organized)
    assert all(not name.startswith("mtp_") for name in names)

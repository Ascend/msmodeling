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
"""Theory Spec YAML / schema / resolver tests without Runtime fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from tools.model_diagnostics.domain.models import (
    INPUT,
    OUTPUT,
    ExecutionPhase,
    ModelRunContext,
    ParallelContext,
    SourceKind,
)
from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.comparison import StageComparisonRegistry
from tools.model_diagnostics.domain.specification import (
    ConcatOptions,
    TensorMappingMode,
)
from tools.model_diagnostics.specification.builtin_activation import (
    create_builtin_operator_activation_registry,
)
from tools.model_diagnostics.specification.errors import (
    AmbiguousModelSpec,
    SpecificationLoadError,
    UnsupportedModelSpec,
)
from tools.model_diagnostics.specification.loader import (
    LoadedSpecDocument,
    YamlModelDiagnosticsSpecLoader,
    _LayerLayoutRule,
)
from tools.model_diagnostics.specification.provider import ResolvingSpecProvider
from tools.model_diagnostics.specification.resolver import LoadedSpecCatalogResolver, matches_context
from tools.model_diagnostics.specification.source_options import (
    RuntimeSourceOptionsParser,
    create_builtin_source_options_parsers,
    parse_theory_operator,
)
from tools.model_diagnostics.specification.theory_fragments import (
    load_builtin_theory_fragment_registry,
)


def _loader(*, registry=None) -> YamlModelDiagnosticsSpecLoader:
    fragment_registry = load_builtin_theory_fragment_registry()
    return YamlModelDiagnosticsSpecLoader(
        comparison_registry=registry or create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(
            fragment_registry=fragment_registry,
        ),
        fragment_registry=fragment_registry,
    )


def _qwen3_context(*, layers: int = 36, features: tuple[str, ...] = ("dense",)) -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen3-8B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=4,
        context_length=None,
        parallel=ParallelContext(tensor_parallel_size=1),
        model_config={
            "model_type": "qwen3",
            "features": list(features),
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 36,
            "effective_num_hidden_layers": layers,
            "vocab_size": 151936,
            "torch_dtype": "bfloat16",
            "num_mtp_tokens": 0,
        },
        quantization_config={},
    )


def _materialized(layers: int = 3) -> object:
    loader = _loader()
    context = _qwen3_context(layers=layers)
    return loader.materialize(loader.load("qwen3_dense_v1"), context), loader, context


def test_qwen3_dense_yaml_loads_and_expands_layers() -> None:
    spec, _, _ = _materialized(layers=3)
    assert spec.spec_id == "qwen3_dense_v1"
    assert [region.region_id for region in spec.regions] == [
        "input",
        "language",
        "output",
        "mtp",
    ]
    language = next(region for region in spec.regions if region.region_id == "language")
    assert language.layer_layout == ("dense", "dense", "dense")
    mtp_region = next(region for region in spec.regions if region.region_id == "mtp")
    assert mtp_region.layer_layout == ()
    assert [stage.stage_id for stage in language.layer_specs["dense"].stages] == [
        "attention_qkv",
        "attention",
        "dense_ffn",
    ]
    attention_qkv = language.layer_specs["dense"].stages[0]
    assert attention_qkv.stage_id == "attention_qkv"
    comparison = attention_qkv.comparisons[(SourceKind.THEORY, SourceKind.RUNTIME)]
    assert comparison.strategy_id == "concat_shape"
    assert isinstance(comparison.options, ConcatOptions)
    assert comparison.options.mapping.mode is TensorMappingMode.COMPOSITE
    assert comparison.options.mapping.relations == ()
    assert comparison.options.axis == -1
    runtime_options = attention_qkv.source_options[SourceKind.RUNTIME]
    assert {
        "rms_norm",
        "view",
        "index",
        "reshape_and_cache",
        "split_with_sizes",
        "slice",
        "select",
        "apply_rope",
        "permute",
        "dynamic_quantize_symmetric",
    }.issubset(runtime_options.ignored_operators)
    attention = language.layer_specs["dense"].stages[1]
    assert [operator.operator_name for operator in attention.source_options[SourceKind.THEORY].operators] == [
        "attention",
        "o_projection",
    ]
    attention_op = attention.source_options[SourceKind.THEORY].operators[0]
    assert attention_op.tensors[INPUT[0]].shape.expression == "[T, Lh * Dh]"
    assert attention_op.tensors[INPUT[0]].dtype.expression == "ACT"
    assert attention_op.tensors[INPUT[1]].shape.expression == "[Nblk, Bs, Lkv, Dh]"
    assert set(attention_op.tensors) == {INPUT[0], INPUT[1], INPUT[2], OUTPUT[0]}
    assert attention_op.tensors[OUTPUT[0]].shape.expression == "[T, Lh * Dh]"
    assert attention_op.tensors[OUTPUT[0]].dtype.expression == "OUT"
    o_projection = attention.source_options[SourceKind.THEORY].operators[1]
    assert o_projection.tensors[INPUT[0]].dtype.expression == "LINEAR_IN"
    q_projection = attention_qkv.source_options[SourceKind.THEORY].operators[0]
    assert q_projection.tensors[INPUT[0]].dtype.expression == "LINEAR_IN"
    embedding = (
        next(region for region in spec.regions if region.region_id == "input")
        .stages[0]
        .source_options[SourceKind.THEORY]
        .operators[0]
    )
    assert embedding.operator_name == "embedding"
    assert embedding.tensors[INPUT[0]].shape.expression == "[EV, EH]"
    assert embedding.tensors[INPUT[0]].dtype.expression == "WEIGHT"
    assert embedding.tensors[INPUT[1]].dtype.expression == "int64"
    assert embedding.tensors[OUTPUT[0]].shape.expression == "[B, Q, EO]"
    embedding_runtime = spec.regions[0].stages[0].source_options[SourceKind.RUNTIME]
    assert {"alias", "where", "all_gather", "all_reduce"}.issubset(embedding_runtime.ignored_operators)
    assert attention.comparisons == {}


def test_loader_expands_prefix_then_repeat_layer_layout() -> None:
    loader = _loader()
    context = _qwen3_context(layers=8)
    context = replace(
        context,
        model_config={
            **context.model_config,
            "first_k_dense_replace": 3,
        },
    )
    specs_dir = Path(__file__).resolve().parents[4] / "tools" / "model_diagnostics" / "specs"
    raw = yaml.safe_load((specs_dir / "qwen3_dense_v1.yaml").read_text(encoding="utf-8"))
    language = raw["regions"]["language"]
    language["layer_specs"]["moe"] = language["layer_specs"]["dense"]
    language["layer_layout_rule"] = {
        "strategy": "prefix_then_repeat",
        "count_from": "model_config.effective_num_hidden_layers",
        "prefix_layer_kind": "dense",
        "repeated_layer_kind": "moe",
        "prefix_count_from": "model_config.first_k_dense_replace",
    }

    spec = loader.materialize(loader.load_mapping(raw), context)
    materialized = next(region for region in spec.regions if region.region_id == "language")

    assert materialized.layer_layout == ("dense", "dense", "dense", "moe", "moe", "moe", "moe", "moe")


def test_qwen3_dense_yaml_leaves_operator_naming_to_comparison_defaults() -> None:
    spec, _, _ = _materialized(layers=1)

    assert spec.operator_aliases == {}


def test_loader_accepts_operator_alias_mapping() -> None:
    loader = _loader()
    specs_dir = Path(__file__).resolve().parents[4] / "tools" / "model_diagnostics" / "specs"
    raw = yaml.safe_load((specs_dir / "qwen3_dense_v1.yaml").read_text(encoding="utf-8"))
    raw["operator_aliases"] = {"aten.mm.default": "mm"}

    assert loader.load_mapping(raw).spec.operator_aliases == {"aten.mm.default": "mm"}


@pytest.mark.parametrize(
    ("features", "expected"),
    (("dense", True), (["dense"], True), (("dense",), True), (7, False)),
)
def test_resolver_normalizes_context_features(features, expected: bool) -> None:
    criteria = replace(_loader().load("qwen3_dense_v1").spec.matches, required_features=("dense",))
    context = _qwen3_context()
    context = replace(context, model_config={**context.model_config, "features": features})

    assert matches_context(criteria, context) is expected


def test_loader_and_runner_registry_share_every_loaded_strategy_id() -> None:
    registry = create_stage_comparison_registry()
    loader = _loader(registry=registry)
    spec = loader.load("qwen3_dense_v1").spec

    strategy_ids = {
        comparison.strategy_id
        for region in spec.regions
        for stage in (
            *region.stages,
            *(stage for layer in region.layer_specs.values() for stage in layer.stages),
        )
        for comparison in stage.comparisons.values()
    }

    assert strategy_ids == {"boundary_equal", "concat_shape"}
    assert all(registry.resolve(strategy_id) for strategy_id in strategy_ids)
    assert registry.resolve("one_to_one")


def test_loader_rejects_strategy_missing_from_runner_registry() -> None:
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=StageComparisonRegistry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
    )

    with pytest.raises(SpecificationLoadError, match="unregistered strategy_id"):
        loader.load("qwen3_dense_v1")


@pytest.mark.parametrize(
    "spec_id",
    (
        "../secrets",
        "..\\secrets",
        "sub/qwen3_dense_v1",
        "/tmp/evil",
    ),
)
def test_loader_rejects_spec_id_path_escape(tmp_path: Path, spec_id: str) -> None:
    outside = tmp_path / "secrets.yaml"
    outside.write_text("schema_version: '1'\n", encoding="utf-8")
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
        specs_dir=specs_dir,
    )

    with pytest.raises(SpecificationLoadError, match="path separators|escapes specs directory|not found"):
        loader.load(spec_id)


def test_loader_wraps_read_oserror_as_specification_load_error(tmp_path: Path, monkeypatch) -> None:
    spec_path = tmp_path / "unreadable.yaml"
    spec_path.write_text("schema_version: '1'\n", encoding="utf-8")
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
        specs_dir=tmp_path,
    )

    def _deny_read(self: Path, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", _deny_read)

    with pytest.raises(SpecificationLoadError, match="cannot read spec file"):
        loader.load("unreadable")


def test_loader_rejects_source_options_without_injected_parser() -> None:
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers={"runtime": RuntimeSourceOptionsParser()},
    )

    with pytest.raises(SpecificationLoadError, match="unregistered source options parser"):
        loader.load("qwen3_dense_v1")


def test_loader_rejects_unregistered_operator_activation() -> None:
    loader = _loader()
    raw = {
        "schema_version": "1",
        "spec_id": "bad_activation",
        "spec_version": "1.0.0",
        "model_category": "test",
        "matches": {"model_types": ["test"]},
        "regions": {
            "output": {
                "stages": [
                    {
                        "id": "output",
                        "source_options": {
                            "theory": {
                                "modules": [
                                    {
                                        "name": "selection",
                                        "activation": "missing",
                                        "tensors": {"OUTPUT[0]": {"shape": "[T, H]"}},
                                    }
                                ]
                            }
                        },
                    }
                ]
            }
        },
    }

    with pytest.raises(SpecificationLoadError, match="unregistered operator activation policy_id"):
        loader.load_mapping(raw)


def test_load_protocol_defers_layout_until_materialize() -> None:
    loader = _loader()
    loaded = loader.load("qwen3_dense_v1")
    assert loaded.spec.regions[1].layer_layout == ()
    assert "language" in loaded.layout_rules
    expanded = loader.materialize(loaded, _qwen3_context(layers=2))
    assert expanded.regions[1].layer_layout == ("dense", "dense")
    assert expanded.operator_aliases == loaded.spec.operator_aliases

    # materialize is a pure function of (loaded, context): re-materializing the
    # same immutable document with a different context (or a different loader
    # instance) yields the same result, with no dependency on prior load() calls.
    other_loader = _loader()
    assert other_loader.materialize(loaded, _qwen3_context(layers=5)).regions[1].layer_layout == ("dense",) * 5


def test_materialize_rejects_a_bare_spec() -> None:
    loader = _loader()
    loaded = loader.load("qwen3_dense_v1")

    with pytest.raises(TypeError, match="LoadedSpecDocument"):
        loader.materialize(loaded.spec, _qwen3_context(layers=2))


@pytest.mark.parametrize(
    "rule,expected_message",
    [
        (
            _LayerLayoutRule(
                strategy="repeat",
                count_from="model_config.effective_num_hidden_layers",
            ),
            "layer_kind is required for repeat strategy",
        ),
        (
            _LayerLayoutRule(
                strategy="prefix_then_repeat",
                count_from="model_config.effective_num_hidden_layers",
            ),
            "field\\(s\\) required for prefix_then_repeat strategy: "
            "prefix_layer_kind, repeated_layer_kind, prefix_count_from",
        ),
    ],
)
def test_materialize_rejects_incomplete_internal_layout_rule(
    rule: _LayerLayoutRule,
    expected_message: str,
) -> None:
    loader = _loader()
    loaded = loader.load("qwen3_dense_v1")
    invalid = LoadedSpecDocument(
        spec=loaded.spec,
        layout_rules={"language": rule},
        region_activations=loaded.region_activations,
    )

    with pytest.raises(SpecificationLoadError, match=expected_message):
        loader.materialize(invalid, _qwen3_context(layers=2))


def test_materialize_activates_lm_head_selection_from_run_context() -> None:
    loader = _loader()
    loaded = loader.load("qwen3_dense_v1")

    prefill = loader.materialize(loaded, _qwen3_context(layers=1))
    decode_context = replace(
        _qwen3_context(layers=1),
        phase=ExecutionPhase.DECODE,
        query_length=1,
        context_length=128,
    )
    decode = loader.materialize(loaded, decode_context)
    mtp_decode = loader.materialize(
        loaded,
        replace(
            decode_context,
            query_length=3,
            model_config={**decode_context.model_config, "num_mtp_tokens": 2},
        ),
    )

    def _output_theory(spec):
        output = next(region for region in spec.regions if region.region_id == "output")
        return output.stages[0].source_options[SourceKind.THEORY]

    def _region(spec, region_id: str):
        return next(region for region in spec.regions if region.region_id == region_id)

    assert [stage.stage_id for stage in _region(prefill, "output").stages] == ["lm_head"]
    assert [stage.stage_id for stage in _region(decode, "output").stages] == ["lm_head"]
    assert _region(prefill, "mtp").stages == ()
    assert _region(decode, "mtp").stages == ()
    prefill_theory = _output_theory(prefill)
    decode_theory = _output_theory(decode)
    assert [operator.operator_name for operator in prefill_theory.operators] == [
        "lm_head_select",
        "lm_head",
    ]
    assert [operator.operator_name for operator in decode_theory.operators] == ["lm_head"]

    assert _region(mtp_decode, "output").stages == ()
    mtp_stages = _region(mtp_decode, "mtp").stages
    assert [stage.stage_id for stage in mtp_stages] == [
        "target_selection",
        "target_lm_head",
        "verification_sampler",
        "mtp_output",
    ]
    assert [stage.source_options[SourceKind.THEORY].operators[0].operator_name for stage in mtp_stages] == [
        "mtp_target_select",
        "lm_head",
        "mtp_target_sampler",
        "mtp_output",
    ]
    assert _region(mtp_decode, "mtp").layer_layout == (
        "dense_mtp",
        "dense_mtp",
    )

    with pytest.raises(SpecificationLoadError, match=r"Q >= MTP \+ 1|query_length >= num_mtp_tokens \+ 1"):
        loader.materialize(
            loaded,
            replace(
                decode_context,
                query_length=2,
                model_config={**decode_context.model_config, "num_mtp_tokens": 2},
            ),
        )


def test_strict_loader_rejects_missing_theory_modules() -> None:
    loader = _loader()
    raw = {
        "schema_version": "1",
        "spec_id": "bad",
        "spec_version": "1.0.0",
        "model_category": "qwen3_dense",
        "matches": {"model_types": ["qwen3"]},
        "regions": {
            "input": {
                "stages": [
                    {
                        "id": "embedding",
                        "source_options": {
                            "theory": {"modules": []},
                            "runtime": {"boundary_operators": ["embed_tokens"]},
                        },
                    }
                ]
            }
        },
    }
    with pytest.raises(SpecificationLoadError):
        loader.load_mapping(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda raw: raw.update({"unknown": True}), "spec document fields mismatch"),
        (
            lambda raw: raw["regions"]["input"]["stages"][0].update({"unknown": True}),
            "stage fields mismatch",
        ),
        (
            lambda raw: raw["regions"]["input"]["stages"][0]["source_options"]["runtime"].update(
                {"boundary_operator": ["embed_tokens"]}
            ),
            "runtime options fields mismatch",
        ),
        (
            lambda raw: raw["regions"]["input"]["stages"][0]["source_options"]["runtime"].update(
                {"boundary_occurrence": "middle"}
            ),
            "runtime options fields mismatch",
        ),
        (
            lambda raw: raw["regions"]["input"]["stages"][0]["source_options"]["theory"]["modules"][0].update(
                {"phases": ["other"]}
            ),
            "theory.modules\\[0\\] fields mismatch",
        ),
        (
            lambda raw: raw["regions"]["input"]["stages"][0]["source_options"]["theory"]["modules"][0][
                "tensors"
            ].update({"bad-slot": {"shape": "[T, H]"}}),
            "invalid Tensor slot syntax",
        ),
        (
            lambda raw: raw["regions"]["input"]["stages"][0]["source_options"]["theory"]["modules"][0][
                "tensors"
            ].update({"INPUT[01]": {"shape": "[T, H]"}}),
            "invalid Tensor slot syntax",
        ),
        (
            lambda raw: raw["regions"]["input"]["stages"][0].update({"comparisons": {}}),
            "comparisons must be omitted when empty",
        ),
        (
            lambda raw: raw["regions"]["input"]["stages"][0].update({"default_comparison": {"strategy": "one_to_one"}}),
            "stage fields mismatch",
        ),
        (
            lambda raw: raw["regions"]["input"]["stages"][0].update({"phases": ["other"]}),
            "stage fields mismatch",
        ),
    ),
)
def test_strict_loader_rejects_unknown_fields_and_invalid_keys(
    mutation,
    message,
) -> None:
    loader = _loader()
    raw = {
        "schema_version": "1",
        "spec_id": "strict",
        "spec_version": "1.0.0",
        "model_category": "test",
        "matches": {"model_types": ["test"]},
        "regions": {
            "input": {
                "stages": [
                    {
                        "id": "embedding",
                        "source_options": {
                            "theory": {
                                "modules": [
                                    {
                                        "name": "embedding",
                                        "tensors": {"OUTPUT[0]": {"shape": "[T, H]"}},
                                    }
                                ],
                            },
                            "runtime": {
                                "boundary_operators": ["embed_tokens"],
                            },
                        },
                    }
                ]
            }
        },
    }
    mutation(raw)

    with pytest.raises(SpecificationLoadError, match=message):
        loader.load_mapping(raw)


def test_resolver_exact_match_zero_and_many() -> None:
    loader = _loader()
    spec = loader.load("qwen3_dense_v1").spec
    resolver = LoadedSpecCatalogResolver(specs=(spec,))
    assert resolver.resolve(_qwen3_context()) == "qwen3_dense_v1"

    with pytest.raises(UnsupportedModelSpec):
        resolver.resolve(replace(_qwen3_context(), entrypoint="other_entrypoint"))

    other = replace(spec, spec_id="qwen3_dense_v1_dup")
    ambiguous = LoadedSpecCatalogResolver(specs=(spec, other))
    with pytest.raises(AmbiguousModelSpec):
        ambiguous.resolve(_qwen3_context())


def test_provider_round_trip() -> None:
    loader = _loader()
    loaded = loader.load("qwen3_dense_v1")
    provider = ResolvingSpecProvider(
        resolver=LoadedSpecCatalogResolver(specs=(loaded.spec,)),
        loader=loader,
        documents={"qwen3_dense_v1": loaded},
    )
    spec = provider.get(_qwen3_context(layers=2))
    assert spec.regions[1].layer_layout == ("dense", "dense")


def test_provider_rejects_a_resolver_result_with_no_preloaded_document() -> None:
    loader = _loader()
    loaded = loader.load("qwen3_dense_v1")
    provider = ResolvingSpecProvider(
        resolver=LoadedSpecCatalogResolver(specs=(loaded.spec,)),
        loader=loader,
        documents={},
    )

    with pytest.raises(SpecificationLoadError, match="no preloaded document"):
        provider.get(_qwen3_context())


class _DuplicateSlotMapping(Mapping):
    """Mapping that can yield the same slot key twice via ``items()``."""

    def __init__(self, items: list[tuple[str, object]]) -> None:
        self._items = items

    def __getitem__(self, key: str) -> object:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def items(self):
        return iter(self._items)


def test_parse_theory_operator_rejects_duplicate_tensor_slots() -> None:
    with pytest.raises(SpecificationLoadError, match="duplicate Tensor slot"):
        parse_theory_operator(
            {
                "name": "linear",
                "tensors": _DuplicateSlotMapping(
                    [
                        ("INPUT[1]", {"shape": "[T, H]"}),
                        ("INPUT[1]", {"dtype": "ACT"}),
                    ]
                ),
            },
            0,
        )

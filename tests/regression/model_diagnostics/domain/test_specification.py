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
from types import MappingProxyType

import pytest

from tools.model_diagnostics.domain import (
    DTypeExpr,
    LayerSpec,
    ModelDiagnosticsSpec,
    RegionSpec,
    RuntimeStageOptions,
    ShapeExpr,
    SourceKind,
    SpecMatchCriteria,
    StageSpec,
    TheoryOperatorSpec,
    TheoryStageOptions,
    TheoryTensorSpec,
    OUTPUT,
)


def _attention_stage() -> StageSpec:
    return StageSpec(
        stage_id="attention",
        source_options={
            SourceKind.THEORY: TheoryStageOptions(
                operators=(
                    TheoryOperatorSpec(operator_name="attention_norm"),
                    TheoryOperatorSpec(
                        operator_name="q_projection",
                        tensors={
                            OUTPUT[0]: TheoryTensorSpec(
                                shape=ShapeExpr("B,Q,H/TP"),
                                dtype=DTypeExpr("activation_dtype"),
                            )
                        },
                    ),
                ),
            ),
            SourceKind.RUNTIME: RuntimeStageOptions(
                boundary_operators=("rms_norm", "rms_norm_add"),
            ),
        },
        comparisons={},
    )


def test_stage_spec_freezes_source_options() -> None:
    sources = {SourceKind.THEORY: TheoryStageOptions(operators=(TheoryOperatorSpec(operator_name="q_projection"),))}
    stage = StageSpec(stage_id="attention", source_options=sources, comparisons={})

    sources.clear()

    assert isinstance(stage.source_options, MappingProxyType)
    assert tuple(item.operator_name for item in stage.source_options[SourceKind.THEORY].operators) == ("q_projection",)


def test_theory_operator_freezes_tensor_mappings() -> None:
    tensors = {OUTPUT[0]: TheoryTensorSpec(shape=ShapeExpr("B,Q,H/TP"))}
    operator = TheoryOperatorSpec(operator_name="q_projection", tensors=tensors)
    options = TheoryStageOptions(operators=(operator,))

    tensors.clear()

    assert isinstance(options.operators[0].tensors, MappingProxyType)
    assert options.operators[0].tensors[OUTPUT[0]].shape.expression == "B,Q,H/TP"


def test_theory_tensor_requires_shape_or_dtype() -> None:
    with pytest.raises(ValueError, match="shape or dtype"):
        TheoryTensorSpec()


def test_region_rejects_duplicate_stage_ids() -> None:
    stage = _attention_stage()

    with pytest.raises(ValueError, match="duplicate stage_id"):
        RegionSpec(region_id="input", stages=(stage, stage))


def test_region_layer_layout_must_reference_known_layer_specs() -> None:
    dense = LayerSpec(layer_kind="dense", stages=(_attention_stage(),))

    with pytest.raises(ValueError, match="unknown layer_kind"):
        RegionSpec(
            region_id="language",
            layer_layout=("dense", "moe"),
            layer_specs={"dense": dense},
        )


def test_model_spec_freezes_operator_aliases() -> None:
    aliases = {"o_projection": "mm"}
    spec = ModelDiagnosticsSpec(
        schema_version="1",
        spec_id="qwen3_dense_v1",
        spec_version="1.0.0",
        model_category="qwen3_dense",
        matches=SpecMatchCriteria(model_types=("qwen3",)),
        regions=(RegionSpec(region_id="input", stages=(_attention_stage(),)),),
        operator_aliases=aliases,
    )

    aliases.clear()

    assert isinstance(spec.operator_aliases, MappingProxyType)
    assert spec.operator_aliases == {"o_projection": "mm"}


def test_model_spec_rejects_empty_operator_alias_key_or_value() -> None:
    values = {
        "schema_version": "1",
        "spec_id": "qwen3_dense_v1",
        "spec_version": "1.0.0",
        "model_category": "qwen3_dense",
        "matches": SpecMatchCriteria(model_types=("qwen3",)),
        "regions": (RegionSpec(region_id="input", stages=(_attention_stage(),)),),
    }

    with pytest.raises(ValueError, match="operator_aliases"):
        ModelDiagnosticsSpec(operator_aliases={"": "mm"}, **values)
    with pytest.raises(ValueError, match="operator_aliases"):
        ModelDiagnosticsSpec(operator_aliases={"o_projection": ""}, **values)


def test_model_spec_preserves_region_order_and_rejects_duplicates() -> None:
    input_region = RegionSpec(region_id="input", stages=(_attention_stage(),))
    language_region = RegionSpec(
        region_id="language",
        layer_layout=("dense",),
        layer_specs={"dense": LayerSpec("dense", (_attention_stage(),))},
    )
    values = {
        "schema_version": "1",
        "spec_id": "qwen3_dense_v1",
        "spec_version": "1.0.0",
        "model_category": "qwen3_dense",
        "matches": SpecMatchCriteria(
            entrypoints=("text_generate",),
            model_types=("qwen3",),
            required_features=("dense",),
        ),
    }

    spec = ModelDiagnosticsSpec(regions=(input_region, language_region), **values)

    assert tuple(region.region_id for region in spec.regions) == ("input", "language")
    with pytest.raises(ValueError, match="duplicate region_id"):
        ModelDiagnosticsSpec(regions=(input_region, input_region), **values)


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (ShapeExpr, ""),
        (DTypeExpr, " "),
        (SpecMatchCriteria, None),
    ],
)
def test_spec_values_reject_empty_contracts(value_type: type, value: str | None) -> None:
    if value_type is SpecMatchCriteria:
        with pytest.raises(ValueError, match="match criterion"):
            SpecMatchCriteria()
    else:
        with pytest.raises(ValueError, match="expression"):
            value_type(value)

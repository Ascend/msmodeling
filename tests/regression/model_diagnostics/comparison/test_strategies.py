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
import pytest

from tools.model_diagnostics.comparison import (
    BoundaryEqualStrategy,
    ConcatShapeStrategy,
    OneToOneEqualStrategy,
    StageComparisonRequest,
)
from tools.model_diagnostics.comparison.operator_policy import DEFAULT_OPERATOR_ALIASES
from tools.model_diagnostics.domain import (
    INPUT,
    OUTPUT,
    BoundaryEqualOptions,
    ComparisonSpec,
    ConcatOptions,
    FindingStatus,
    OneToOneOptions,
    OperatorCallRecord,
    StageExecutionRecord,
    TensorInfo,
    TensorMapping,
    TensorMappingMode,
    TensorRelation,
    TensorSlotPair,
    TensorSlotRef,
)


_OPERATOR_ALIASES = dict(DEFAULT_OPERATOR_ALIASES)


def _call(index: int, name: str, shape=(2, 4), dtype="float16") -> OperatorCallRecord:
    return OperatorCallRecord(
        call_index=index,
        operator_name=name,
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], shape, dtype),
            TensorInfo(OUTPUT[0], shape, dtype),
        ),
    )


def _request(
    left,
    right,
    options,
    strategy_id,
    *,
    operator_aliases=None,
) -> StageComparisonRequest:
    return StageComparisonRequest(
        region_id="language",
        layer_index=0,
        left_stage=StageExecutionRecord("attention", tuple(left)),
        right_stage=StageExecutionRecord("attention", tuple(right)),
        comparison=ComparisonSpec(strategy_id, options),
        operator_aliases=operator_aliases or {},
    )


def test_one_to_one_positional_reports_pass_for_equal_stage() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    request = _request([_call(10, "mm")], [_call(20, "mm")], options, "one_to_one")

    findings = OneToOneEqualStrategy().execute(request)

    assert len(findings) == 1
    assert findings[0].status is FindingStatus.PASS
    assert findings[0].expected == "(2, 4)/float16; (2, 4)/float16"
    assert findings[0].actual == findings[0].expected


@pytest.mark.parametrize(
    ("flat", "expanded"),
    [
        ((2, 8), (1, 2, 8)),
        ((6, 8), (2, 3, 8)),
        ((6, 8, 64), (2, 3, 8, 64)),
    ],
)
def test_one_to_one_accepts_product_equivalent_leading_dimensions(flat, expanded) -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    request = _request([_call(10, "mm", flat)], [_call(20, "mm", expanded)], options, "one_to_one")

    findings = OneToOneEqualStrategy().execute(request)

    assert findings
    assert all(finding.status is FindingStatus.PASS for finding in findings)
    assert all(finding.message_code == "comparison.leading_product_equivalent" for finding in findings)


@pytest.mark.parametrize(
    ("flat", "expanded"),
    [
        ((6, 8), (2, 4, 8)),
        ((6, 8), (2, 3, 9)),
        ((6, 8), (1, 2, 3, 8)),
        ((), (1,)),
    ],
)
def test_one_to_one_rejects_non_equivalent_leading_dimensions(flat, expanded) -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    request = _request([_call(10, "mm", flat)], [_call(20, "mm", expanded)], options, "one_to_one")

    findings = OneToOneEqualStrategy().execute(request)

    assert any(finding.message_code == "tensor.shape_mismatch" for finding in findings)


def test_operator_name_normalization_is_limited_to_known_namespaces() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    known = _request([_call(10, "aten.mm.default")], [_call(20, "mm")], options, "one_to_one")
    unknown = _request(
        [_call(10, "custom.mm.default")],
        [_call(20, "mm")],
        options,
        "one_to_one",
    )

    assert OneToOneEqualStrategy().execute(known)[0].status is FindingStatus.PASS
    assert any(finding.message_code == "operator.name_mismatch" for finding in OneToOneEqualStrategy().execute(unknown))


def test_explicit_strategies_reject_programmatic_empty_mapping() -> None:
    mapping = TensorMapping(TensorMappingMode.EXPLICIT)
    one_to_one = _request(
        [_call(10, "mm")],
        [_call(20, "mm")],
        OneToOneOptions(mapping),
        "one_to_one",
    )
    boundary = _request(
        [_call(10, "mm")],
        [_call(20, "mm")],
        BoundaryEqualOptions(mapping),
        "boundary_equal",
    )

    with pytest.raises(ValueError, match="at least one pair"):
        OneToOneEqualStrategy().execute(one_to_one)
    with pytest.raises(ValueError, match="at least one pair"):
        BoundaryEqualStrategy().execute(boundary)


def test_one_to_one_reports_operator_shape_and_missing_evidence() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    left = [_call(10, "mm", (2, 4)), _call(11, "relu")]
    right_call = OperatorCallRecord(
        call_index=20,
        operator_name="matmul",
        original_operator_name=None,
        tensors=(TensorInfo(INPUT[0], (2, 5), "float16"),),
    )

    findings = OneToOneEqualStrategy().execute(_request(left, [right_call], options, "one_to_one"))

    assert {finding.message_code for finding in findings if finding.status is not FindingStatus.PASS} == {
        "operator.count_mismatch",
        "operator.name_mismatch",
        "tensor.shape_mismatch",
        "tensor.missing",
    }
    assert {finding.status for finding in findings} >= {
        FindingStatus.FAIL,
        FindingStatus.INCOMPLETE,
    }


def test_one_to_one_compares_theory_slots_and_ignores_runtime_only_slots() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    theory = _call(10, "aten.mm.default", shape=(2, 4), dtype="int8")
    runtime = OperatorCallRecord(
        call_index=20,
        operator_name="tensor_cast.static_quant_linear.default",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4), "int8"),
            TensorInfo(INPUT[1], (4, 8), "int8"),
            TensorInfo(INPUT[2], (8,), "float32"),
            TensorInfo(OUTPUT[0], (2, 4), "int8"),
        ),
    )

    findings = OneToOneEqualStrategy().execute(
        _request(
            [theory],
            [runtime],
            options,
            "one_to_one",
            operator_aliases=_OPERATOR_ALIASES,
        )
    )

    assert len(findings) == 1
    assert findings[0].status is FindingStatus.PASS
    assert findings[0].expected == "(2, 4)/int8; (2, 4)/int8"
    assert "input[1]" not in str(findings[0].expected)


def test_one_to_one_quant_linear_checks_declared_shape_and_dtype() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    theory = _call(10, "aten.mm.default", shape=(2, 4), dtype="int8")
    runtime = OperatorCallRecord(
        call_index=20,
        operator_name="tensor_cast.static_quant_linear_int4.default",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 5), "int8"),
            TensorInfo(INPUT[1], (2, 8), "uint8"),
            TensorInfo(OUTPUT[0], (2, 4), "float16"),
        ),
    )

    findings = OneToOneEqualStrategy().execute(
        _request(
            [theory],
            [runtime],
            options,
            "one_to_one",
            operator_aliases=_OPERATOR_ALIASES,
        )
    )

    assert {finding.message_code for finding in findings if finding.status is not FindingStatus.PASS} == {
        "tensor.shape_mismatch",
        "tensor.dtype_mismatch",
    }


def test_one_to_one_compares_an_additional_slot_when_theory_declares_it() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    theory = OperatorCallRecord(
        call_index=10,
        operator_name="aten.mm.default",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4), "int8"),
            TensorInfo(INPUT[1], (4, 8), "int8"),
            TensorInfo(OUTPUT[0], (2, 8), "float16"),
        ),
    )
    runtime = OperatorCallRecord(
        call_index=20,
        operator_name="tensor_cast.static_quant_linear.default",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4), "int8"),
            TensorInfo(INPUT[1], (4, 7), "uint8"),
            TensorInfo(INPUT[2], (8,), "float32"),
            TensorInfo(OUTPUT[0], (2, 8), "float16"),
        ),
    )

    findings = OneToOneEqualStrategy().execute(
        _request(
            [theory],
            [runtime],
            options,
            "one_to_one",
            operator_aliases=_OPERATOR_ALIASES,
        )
    )

    mismatches = [finding for finding in findings if finding.status is not FindingStatus.PASS]
    assert {finding.message_code for finding in mismatches} == {
        "tensor.shape_mismatch",
        "tensor.dtype_mismatch",
    }
    assert all(finding.left_evidence[0].tensor_slot == INPUT[1] for finding in mismatches)


def test_one_to_one_keeps_non_linear_operator_identity_strict() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))

    findings = OneToOneEqualStrategy().execute(
        _request(
            [_call(10, "aten.mm.default")],
            [_call(20, "tensor_cast.unrelated.default")],
            options,
            "one_to_one",
        )
    )

    assert "operator.name_mismatch" in {finding.message_code for finding in findings}


def test_one_to_one_uses_theory_declared_slots_as_contract() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    theory = OperatorCallRecord(
        call_index=10,
        operator_name="attention",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4096), "float16"),
            TensorInfo(OUTPUT[0], (2, 4096), "float16"),
        ),
    )
    runtime = OperatorCallRecord(
        call_index=20,
        operator_name="tensor_cast.attention.default",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4096), "float16"),
            TensorInfo(INPUT[1], (1, 128, 8, 128), "float16"),
            TensorInfo(INPUT[2], (1, 128, 8, 128), "float16"),
            TensorInfo(OUTPUT[0], (2, 4096), "float16"),
        ),
    )

    findings = OneToOneEqualStrategy().execute(_request([theory], [runtime], options, "one_to_one"))

    assert [finding.status for finding in findings] == [FindingStatus.PASS]


def test_one_to_one_attention_checks_only_query_kv_cache_and_output_slots() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    theory = OperatorCallRecord(
        call_index=10,
        operator_name="attention",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4096), "float16"),
            TensorInfo(INPUT[1], (1, 128, 8, 128), "float16"),
            TensorInfo(INPUT[2], (1, 128, 8, 128), "float16"),
            TensorInfo(OUTPUT[0], (2, 4096), "float16"),
        ),
    )
    runtime = OperatorCallRecord(
        call_index=20,
        operator_name="tensor_cast.attention.default",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4096), "float16"),
            TensorInfo(INPUT[1], (1, 128, 8, 128), "float16"),
            TensorInfo(INPUT[2], (1, 64, 8, 128), "float16"),
            TensorInfo(INPUT[3], (99, 99), "float32"),
            TensorInfo(OUTPUT[0], (2, 4096), "float16"),
        ),
    )

    findings = OneToOneEqualStrategy().execute(_request([theory], [runtime], options, "one_to_one"))

    assert {finding.rule_id for finding in findings if finding.status is FindingStatus.FAIL} == {
        "call[0].input[2].shape"
    }


def test_one_to_one_matches_lm_head_select_to_runtime_index() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    theory = OperatorCallRecord(
        call_index=10,
        operator_name="lm_head_select",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (1, 2, 4096), "float16"),
            TensorInfo(INPUT[1], (1,), "int64"),
            TensorInfo(OUTPUT[0], (1, 1, 4096), "float16"),
        ),
    )
    runtime = OperatorCallRecord(
        call_index=20,
        operator_name="aten.index.Tensor",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (1, 2, 4096), "float16"),
            TensorInfo(INPUT[1], (1,), "int64"),
            TensorInfo(OUTPUT[0], (1, 1, 4096), "float16"),
        ),
    )

    findings = OneToOneEqualStrategy().execute(
        _request(
            [theory],
            [runtime],
            options,
            "one_to_one",
            operator_aliases={"lm_head_select": "index"},
        )
    )

    assert len(findings) == 1
    assert findings[0].status is FindingStatus.PASS
    assert "(1, 2, 4096)/float16" in str(findings[0].expected)
    assert "(1,)/int64" in str(findings[0].expected)


def test_one_to_one_compares_theory_o_projection_with_runtime_mm() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    theory = OperatorCallRecord(
        call_index=10,
        operator_name="o_projection",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4096), "float16"),
            TensorInfo(OUTPUT[0], (2, 4096), "float16"),
        ),
    )
    runtime = OperatorCallRecord(
        call_index=20,
        operator_name="aten.mm.default",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4096), "float16"),
            TensorInfo(INPUT[1], (4096, 4096), "float16"),
            TensorInfo(OUTPUT[0], (2, 4096), "float16"),
        ),
    )

    findings = OneToOneEqualStrategy().execute(
        _request(
            [theory],
            [runtime],
            options,
            "one_to_one",
            operator_aliases=_OPERATOR_ALIASES,
        )
    )

    assert len(findings) == 1
    assert findings[0].status is FindingStatus.PASS
    assert findings[0].expected == ("(2, 4096)/float16; (2, 4096)/float16")


def test_one_to_one_without_aliases_treats_projection_and_mm_as_distinct_operators() -> None:
    options = OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL))
    theory = OperatorCallRecord(
        call_index=10,
        operator_name="q_projection",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4096), "float16"),
            TensorInfo(OUTPUT[0], (2, 4096), "float16"),
        ),
    )
    runtime = OperatorCallRecord(
        call_index=20,
        operator_name="aten.mm.default",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4096), "float16"),
            TensorInfo(OUTPUT[0], (2, 4096), "float16"),
        ),
    )

    findings = OneToOneEqualStrategy().execute(_request([theory], [runtime], options, "one_to_one"))

    assert "operator.name_mismatch" in {finding.message_code for finding in findings}


def test_boundary_explicit_mapping_uses_stage_local_call_positions() -> None:
    mapping = TensorMapping(
        TensorMappingMode.EXPLICIT,
        pairs=(
            TensorSlotPair(
                left_call_index=1,
                left_slot=OUTPUT[0],
                right_call_index=0,
                right_slot=INPUT[0],
            ),
        ),
    )
    request = _request(
        [_call(100, "first"), _call(200, "boundary")],
        [_call(900, "fused")],
        BoundaryEqualOptions(mapping),
        "boundary_equal",
    )

    findings = BoundaryEqualStrategy().execute(request)

    assert len(findings) == 1
    assert findings[0].status is FindingStatus.PASS
    assert findings[0].expected == "(2, 4)/float16"
    assert findings[0].actual == "(2, 4)/float16"


def test_concat_compares_composed_shape_and_keeps_multiple_evidence_refs() -> None:
    relation = TensorRelation(
        left=(
            TensorSlotRef(0, OUTPUT[0]),
            TensorSlotRef(1, OUTPUT[0]),
        ),
        right=(TensorSlotRef(0, OUTPUT[0]),),
        operation="concat",
        axis=1,
    )
    mapping = TensorMapping(TensorMappingMode.COMPOSITE, relations=(relation,))
    request = _request(
        [_call(10, "q", (2, 2)), _call(11, "k", (2, 2))],
        [_call(20, "qk", (2, 4))],
        ConcatOptions(mapping, axis=1),
        "concat_shape",
    )

    findings = ConcatShapeStrategy().execute(request)

    assert len(findings) == 1
    assert findings[0].status is FindingStatus.PASS
    assert findings[0].expected == "(2, 4)/float16"
    assert findings[0].actual == "(2, 4)/float16"
    assert len(findings[0].left_evidence) == 2
    assert len(findings[0].right_evidence) == 1


def test_concat_reports_shape_mismatch() -> None:
    relation = TensorRelation(
        left=(TensorSlotRef(0, OUTPUT[0]), TensorSlotRef(1, OUTPUT[0])),
        right=(TensorSlotRef(0, OUTPUT[0]),),
        operation="concat",
    )
    request = _request(
        [_call(10, "q", (2, 2)), _call(11, "k", (2, 2))],
        [_call(20, "qk", (2, 5))],
        ConcatOptions(
            TensorMapping(TensorMappingMode.COMPOSITE, relations=(relation,)),
            axis=1,
        ),
        "concat_shape",
    )

    findings = ConcatShapeStrategy().execute(request)

    assert findings[0].message_code == "tensor.shape_mismatch"
    assert findings[0].expected == (2, 4)
    assert findings[0].actual == (2, 5)
    assert len(findings[0].left_evidence) == 2
    assert len(findings[0].right_evidence) == 1


def test_concat_default_uses_all_left_outputs_and_unique_right_output() -> None:
    request = _request(
        [
            _call(10, "q", (2, 4)),
            _call(11, "k", (2, 1)),
            _call(12, "v", (2, 1)),
        ],
        [_call(20, "qkv", (2, 6))],
        ConcatOptions(TensorMapping(TensorMappingMode.COMPOSITE), axis=-1),
        "concat_shape",
    )

    findings = ConcatShapeStrategy().execute(request)

    assert len(findings) == 1
    assert findings[0].status is FindingStatus.PASS
    assert findings[0].expected == "(2, 6)/float16"
    assert findings[0].actual == "(2, 6)/float16"


def test_concat_default_rejects_non_unique_runtime_call() -> None:
    request = _request(
        [_call(10, "q"), _call(11, "k"), _call(12, "v")],
        [_call(20, "qkv"), _call(21, "unexpected")],
        ConcatOptions(TensorMapping(TensorMappingMode.COMPOSITE), axis=-1),
        "concat_shape",
    )

    findings = ConcatShapeStrategy().execute(request)

    assert findings[0].status is FindingStatus.FAIL
    assert findings[0].message_code == "concat.default_cardinality_mismatch"

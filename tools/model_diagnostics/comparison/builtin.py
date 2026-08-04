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
"""Built-in pairwise Tensor comparison strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from tools.model_diagnostics.domain import (
    INPUT,
    OUTPUT,
    BoundaryEqualOptions,
    ConcatOptions,
    EvidenceRef,
    Finding,
    FindingStatus,
    OneToOneOptions,
    OperatorCallRecord,
    SourceKind,
    TensorInfo,
    TensorMapping,
    TensorMappingMode,
    TensorRelation,
    TensorSlot,
    TensorSlotPair,
    TensorSlotRef,
)

from .models import StageComparisonRequest

def _canonical_operator_name(operator_name: str, aliases: Mapping[str, str]) -> str:
    alias = aliases.get(operator_name)
    if alias is not None:
        return alias
    parts = operator_name.split(".")
    if len(parts) == 3 and parts[0] in {"aten", "tensor_cast"}:
        return parts[1]
    return operator_name

@dataclass(frozen=True)
class _LocatedTensor:
    tensor: TensorInfo
    evidence: EvidenceRef


class OneToOneEqualStrategy:
    strategy_id = "one_to_one"

    def execute(self, request: StageComparisonRequest) -> tuple[Finding, ...]:
        options = request.comparison.options
        if not isinstance(options, OneToOneOptions):
            raise TypeError("one_to_one strategy requires OneToOneOptions")
        mapping = options.mapping
        findings: list[Finding] = []
        if mapping.mode is TensorMappingMode.POSITIONAL:
            findings.extend(_compare_positional_calls(request))
        elif mapping.mode is TensorMappingMode.EXPLICIT:
            if not mapping.pairs:
                raise ValueError("one_to_one explicit mapping requires at least one pair")
            findings.extend(_compare_explicit_pairs(request, mapping))
        else:
            raise ValueError("one_to_one strategy does not support composite mapping")
        return _pass_if_empty(request, findings, self.strategy_id)


class BoundaryEqualStrategy:
    strategy_id = "boundary_equal"

    def execute(self, request: StageComparisonRequest) -> tuple[Finding, ...]:
        options = request.comparison.options
        if not isinstance(options, BoundaryEqualOptions):
            raise TypeError("boundary_equal strategy requires BoundaryEqualOptions")
        if options.mapping.mode is not TensorMappingMode.EXPLICIT:
            raise ValueError("boundary_equal strategy requires explicit mapping")
        if not options.mapping.pairs:
            raise ValueError("boundary_equal explicit mapping requires at least one pair")
        findings = _compare_explicit_pairs(request, options.mapping)
        return _pass_if_empty(request, findings, self.strategy_id)


class ConcatShapeStrategy:
    strategy_id = "concat_shape"

    def execute(self, request: StageComparisonRequest) -> tuple[Finding, ...]:
        options = request.comparison.options
        if not isinstance(options, ConcatOptions):
            raise TypeError("concat_shape strategy requires ConcatOptions")
        if options.mapping.mode is not TensorMappingMode.COMPOSITE:
            raise ValueError("concat_shape strategy requires composite mapping")
        relations = options.mapping.relations
        if not relations:
            left_count = len(request.left_stage.operator_calls)
            right_count = len(request.right_stage.operator_calls)
            if left_count == 0 or right_count != 1:
                return (
                    _finding(
                        request,
                        rule_id="default_relation",
                        comparison_kind="concat",
                        status=FindingStatus.FAIL,
                        message_code="concat.default_cardinality_mismatch",
                        message=("default concat requires at least one left call and exactly one right call"),
                        expected="left>=1,right=1",
                        actual=(left_count, right_count),
                    ),
                )
            relations = (
                TensorRelation(
                    left=tuple(TensorSlotRef(call_index=index, slot=OUTPUT[0]) for index in range(left_count)),
                    right=(TensorSlotRef(call_index=0, slot=OUTPUT[0]),),
                    operation="concat",
                    axis=-1,
                ),
            )
        findings: list[Finding] = []
        for relation_index, relation in enumerate(relations):
            findings.extend(
                _compare_concat_relation(
                    request,
                    relation,
                    relation_index=relation_index,
                    default_axis=options.axis,
                )
            )
        return _pass_if_empty(request, findings, self.strategy_id)


def _compare_positional_calls(request: StageComparisonRequest) -> list[Finding]:
    left_calls = request.left_stage.operator_calls
    right_calls = request.right_stage.operator_calls
    findings: list[Finding] = []
    if len(left_calls) != len(right_calls):
        findings.append(
            _finding(
                request,
                rule_id="operator_count",
                comparison_kind="operator_completeness",
                status=FindingStatus.FAIL,
                message_code="operator.count_mismatch",
                message="stage operator counts differ",
                expected=len(left_calls),
                actual=len(right_calls),
            )
        )
    for call_position, (left_call, right_call) in enumerate(zip(left_calls, right_calls)):
        call_findings: list[Finding] = []
        left_name = _canonical_operator_name(left_call.operator_name, request.operator_aliases)
        right_name = _canonical_operator_name(right_call.operator_name, request.operator_aliases)
        if left_name != right_name:
            call_findings.append(
                _finding(
                    request,
                    rule_id=f"operator[{call_position}]",
                    comparison_kind="operator_identity",
                    status=FindingStatus.FAIL,
                    message_code="operator.name_mismatch",
                    message="paired operator names differ",
                    expected=left_name,
                    actual=right_name,
                    left_evidence=(_call_evidence(SourceKind.THEORY, left_call, call_position),),
                    right_evidence=(_call_evidence(SourceKind.RUNTIME, right_call, call_position),),
                )
            )
        left_tensors = {tensor.slot: tensor for tensor in left_call.tensors}
        right_tensors = {tensor.slot: tensor for tensor in right_call.tensors}
        # Theory declarations are the comparison contract. Runtime may capture
        # additional implementation/control tensors without making them required.
        slots = set(left_tensors)
        matched_left: list[_LocatedTensor] = []
        matched_right: list[_LocatedTensor] = []
        for slot in sorted(slots, key=_slot_sort_key):
            left = _located(SourceKind.THEORY, left_call, call_position, left_tensors.get(slot))
            right = _located(SourceKind.RUNTIME, right_call, call_position, right_tensors.get(slot))
            slot_findings = _compare_optional_tensors(
                request,
                rule_id=f"call[{call_position}].{_slot_id(slot)}",
                left=left,
                right=right,
            )
            if slot_findings:
                call_findings.extend(slot_findings)
            elif left is not None and right is not None:
                matched_left.append(left)
                matched_right.append(right)
        if call_findings:
            findings.extend(call_findings)
            continue
        findings.append(
            _finding(
                request,
                rule_id=f"call[{call_position}]",
                comparison_kind="one_to_one",
                status=FindingStatus.PASS,
                message_code="comparison.pass",
                message="call tensors match",
                expected=_format_tensor_bundle(matched_left),
                actual=_format_tensor_bundle(matched_right),
                left_evidence=tuple(item.evidence for item in matched_left),
                right_evidence=tuple(item.evidence for item in matched_right),
            )
        )
    return findings


def _compare_explicit_pairs(
    request: StageComparisonRequest,
    mapping: TensorMapping,
) -> list[Finding]:
    findings: list[Finding] = []
    for pair_index, pair in enumerate(mapping.pairs):
        left = _resolve_pair_side(request, pair, left=True)
        right = _resolve_pair_side(request, pair, left=False)
        pair_findings = _compare_optional_tensors(
            request,
            rule_id=f"pair[{pair_index}]",
            left=left,
            right=right,
        )
        if pair_findings:
            findings.extend(pair_findings)
            continue
        if left is None or right is None:
            continue
        findings.append(
            _finding(
                request,
                rule_id=f"pair[{pair_index}]",
                comparison_kind="boundary_equal",
                status=FindingStatus.PASS,
                message_code="comparison.pass",
                message="mapped Tensor slots match",
                expected=_format_shape_dtype(left.tensor),
                actual=_format_shape_dtype(right.tensor),
                left_evidence=(left.evidence,),
                right_evidence=(right.evidence,),
            )
        )
    return findings


def _compare_concat_relation(
    request: StageComparisonRequest,
    relation: TensorRelation,
    *,
    relation_index: int,
    default_axis: int,
) -> list[Finding]:
    if relation.operation != "concat":
        raise ValueError(f"unsupported TensorRelation operation {relation.operation!r}")
    axis = relation.axis if relation.axis is not None else default_axis
    left = _resolve_refs(request, relation.left, left=True)
    right = _resolve_refs(request, relation.right, left=False)
    missing = [item for item in left + right if item is None]
    if missing:
        return [
            _finding(
                request,
                rule_id=f"relation[{relation_index}]",
                comparison_kind="concat",
                status=FindingStatus.INCOMPLETE,
                message_code="tensor.missing",
                message="concat relation references a missing call or tensor slot",
            )
        ]
    left_tensors = tuple(item for item in left if item is not None)
    right_tensors = tuple(item for item in right if item is not None)
    try:
        left_shape, left_dtype = _concat_value(left_tensors, axis)
        right_shape, right_dtype = _concat_value(right_tensors, axis)
    except ValueError as error:
        return [
            _finding(
                request,
                rule_id=f"relation[{relation_index}]",
                comparison_kind="concat",
                status=FindingStatus.INCOMPLETE,
                message_code="tensor.concat_incomplete",
                message=str(error),
                left_evidence=tuple(item.evidence for item in left_tensors),
                right_evidence=tuple(item.evidence for item in right_tensors),
            )
        ]
    evidence = {
        "left_evidence": tuple(item.evidence for item in left_tensors),
        "right_evidence": tuple(item.evidence for item in right_tensors),
    }
    findings: list[Finding] = []
    if left_shape != right_shape:
        findings.append(
            _finding(
                request,
                rule_id=f"relation[{relation_index}].shape",
                comparison_kind="concat_shape",
                status=FindingStatus.FAIL,
                message_code="tensor.shape_mismatch",
                message="concatenated Tensor shapes differ",
                expected=left_shape,
                actual=right_shape,
                **evidence,
            )
        )
    if left_dtype != right_dtype:
        findings.append(
            _finding(
                request,
                rule_id=f"relation[{relation_index}].dtype",
                comparison_kind="concat_dtype",
                status=FindingStatus.FAIL,
                message_code="tensor.dtype_mismatch",
                message="concatenated Tensor dtypes differ",
                expected=left_dtype,
                actual=right_dtype,
                **evidence,
            )
        )
    if findings:
        return findings
    return [
        _finding(
            request,
            rule_id=f"relation[{relation_index}]",
            comparison_kind="concat_shape",
            status=FindingStatus.PASS,
            message_code="comparison.pass",
            message="concatenated Tensor shape/dtype match",
            expected=_format_raw_shape_dtype(left_shape, left_dtype),
            actual=_format_raw_shape_dtype(right_shape, right_dtype),
            **evidence,
        )
    ]


def _resolve_pair_side(
    request: StageComparisonRequest,
    pair: TensorSlotPair,
    *,
    left: bool,
) -> _LocatedTensor | None:
    call_index = pair.left_call_index if left else pair.right_call_index
    slot = pair.left_slot if left else pair.right_slot
    calls = request.left_stage.operator_calls if left else request.right_stage.operator_calls
    source_kind = SourceKind.THEORY if left else SourceKind.RUNTIME
    return _resolve_tensor(calls, call_index, slot, source_kind)


def _resolve_refs(
    request: StageComparisonRequest,
    refs: tuple[TensorSlotRef, ...],
    *,
    left: bool,
) -> list[_LocatedTensor | None]:
    calls = request.left_stage.operator_calls if left else request.right_stage.operator_calls
    source_kind = SourceKind.THEORY if left else SourceKind.RUNTIME
    return [_resolve_tensor(calls, ref.call_index, ref.slot, source_kind) for ref in refs]


def _resolve_tensor(
    calls: tuple[OperatorCallRecord, ...],
    call_position: int,
    slot: TensorSlot,
    source_kind: SourceKind,
) -> _LocatedTensor | None:
    if call_position < 0 or call_position >= len(calls):
        return None
    call = calls[call_position]
    tensor = next((candidate for candidate in call.tensors if candidate.slot == slot), None)
    return _located(source_kind, call, call_position, tensor)


def _located(
    source_kind: SourceKind,
    call: OperatorCallRecord,
    call_position: int,
    tensor: TensorInfo | None,
) -> _LocatedTensor | None:
    if tensor is None:
        return None
    return _LocatedTensor(
        tensor=tensor,
        evidence=_call_evidence(source_kind, call, call_position, tensor.slot),
    )


def _compare_optional_tensors(
    request: StageComparisonRequest,
    *,
    rule_id: str,
    left: _LocatedTensor | None,
    right: _LocatedTensor | None,
) -> list[Finding]:
    if left is None or right is None:
        return [
            _finding(
                request,
                rule_id=rule_id,
                comparison_kind="tensor_equality",
                status=FindingStatus.INCOMPLETE,
                message_code="tensor.missing",
                message="required Tensor slot is missing",
                left_evidence=() if left is None else (left.evidence,),
                right_evidence=() if right is None else (right.evidence,),
            )
        ]
    findings: list[Finding] = []
    evidence = {"left_evidence": (left.evidence,), "right_evidence": (right.evidence,)}
    for field_name in ("shape", "dtype"):
        expected = getattr(left.tensor, field_name)
        actual = getattr(right.tensor, field_name)
        if expected is None or actual is None:
            findings.append(
                _finding(
                    request,
                    rule_id=f"{rule_id}.{field_name}",
                    comparison_kind=field_name,
                    status=FindingStatus.INCOMPLETE,
                    message_code=f"tensor.{field_name}_missing",
                    message=f"required Tensor {field_name} is missing",
                    expected=expected,
                    actual=actual,
                    **evidence,
                )
            )
        elif expected != actual:
            findings.append(
                _finding(
                    request,
                    rule_id=f"{rule_id}.{field_name}",
                    comparison_kind=field_name,
                    status=FindingStatus.FAIL,
                    message_code=f"tensor.{field_name}_mismatch",
                    message=f"Tensor {field_name}s differ",
                    expected=expected,
                    actual=actual,
                    **evidence,
                )
            )
    return findings


def _format_shape_dtype(tensor: TensorInfo) -> str:
    return _format_raw_shape_dtype(tensor.shape, tensor.dtype)


def _format_raw_shape_dtype(shape: tuple[int, ...] | None, dtype: str | None) -> str:
    return f"{shape}/{dtype}"


def _format_tensor_bundle(tensors: list[_LocatedTensor]) -> str:
    if not tensors:
        return ""
    return "; ".join(_format_shape_dtype(item.tensor) for item in tensors)


def _concat_value(
    tensors: tuple[_LocatedTensor, ...],
    axis: int,
) -> tuple[tuple[int, ...], str]:
    if not tensors:
        raise ValueError("concat relation must reference at least one Tensor per side")
    shapes = tuple(item.tensor.shape for item in tensors)
    dtypes = tuple(item.tensor.dtype for item in tensors)
    if any(shape is None for shape in shapes) or any(dtype is None for dtype in dtypes):
        raise ValueError("concat relation requires complete shape and dtype evidence")
    complete_shapes = tuple(shape for shape in shapes if shape is not None)
    rank = len(complete_shapes[0])
    normalized_axis = axis + rank if axis < 0 else axis
    if normalized_axis < 0 or normalized_axis >= rank:
        raise ValueError("concat axis is outside Tensor rank")
    if any(len(shape) != rank for shape in complete_shapes):
        raise ValueError("concat Tensor ranks differ")
    for dimension in range(rank):
        if dimension == normalized_axis:
            continue
        if len({shape[dimension] for shape in complete_shapes}) != 1:
            raise ValueError("concat non-axis dimensions differ")
    complete_dtypes = tuple(dtype for dtype in dtypes if dtype is not None)
    if len(set(complete_dtypes)) != 1:
        raise ValueError("concat Tensor dtypes differ within one source")
    result = list(complete_shapes[0])
    result[normalized_axis] = sum(shape[normalized_axis] for shape in complete_shapes)
    return tuple(result), complete_dtypes[0]


def _call_evidence(
    source_kind: SourceKind,
    call: OperatorCallRecord,
    call_position: int,
    tensor_slot: TensorSlot | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        source_kind=source_kind,
        call_index=call.call_index,
        stage_call_position=call_position,
        operator_name=call.operator_name,
        tensor_slot=tensor_slot,
        source_reference=call.source_reference,
    )


def _finding(
    request: StageComparisonRequest,
    *,
    rule_id: str,
    comparison_kind: str,
    status: FindingStatus,
    message_code: str,
    message: str,
    expected=None,
    actual=None,
    left_evidence: tuple[EvidenceRef, ...] = (),
    right_evidence: tuple[EvidenceRef, ...] = (),
) -> Finding:
    return Finding(
        region_id=request.region_id,
        layer_index=request.layer_index,
        stage_id=request.left_stage.stage_id,
        rule_id=rule_id,
        comparison_kind=comparison_kind,
        status=status,
        message_code=message_code,
        message=message,
        expected=expected,
        actual=actual,
        left_evidence=left_evidence,
        right_evidence=right_evidence,
    )


def _pass_if_empty(
    request: StageComparisonRequest,
    findings: list[Finding],
    strategy_id: str,
) -> tuple[Finding, ...]:
    if findings:
        return tuple(findings)
    # Empty stage with no declared Tensor checks still needs an explicit PASS row.
    return (
        _finding(
            request,
            rule_id=strategy_id,
            comparison_kind=strategy_id,
            status=FindingStatus.PASS,
            message_code="comparison.pass",
            message="all declared comparison checks passed",
            expected=(),
            actual=(),
        ),
    )


def _slot_sort_key(slot: TensorSlot) -> tuple[str, int, str]:
    return slot.direction.value, slot.index, slot.name or ""


def _slot_id(slot: TensorSlot) -> str:
    name = f".{slot.name}" if slot.name else ""
    return f"{slot.direction.value}[{slot.index}]{name}"

import dataclasses
from typing import Dict, List, Optional

from .actual import ActualSummary
from .evidence import EvidenceCase, ExpectedOp


@dataclasses.dataclass(frozen=True)
class VerificationIssue:
    category: str
    message: str
    severity: str = "error"
    expected: Optional[object] = None
    actual: Optional[object] = None
    evidence_path: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "category": self.category,
            "message": self.message,
            "severity": self.severity,
            "expected": self.expected,
            "actual": self.actual,
            "evidence_path": self.evidence_path,
        }


@dataclasses.dataclass(frozen=True)
class VerificationReport:
    case_name: str
    passed: bool
    issues: List[VerificationIssue]

    def issues_by_category(self) -> Dict[str, List[VerificationIssue]]:
        grouped: Dict[str, List[VerificationIssue]] = {}
        for issue in self.issues:
            grouped.setdefault(issue.category, []).append(issue)
        return grouped

    def to_dict(self) -> Dict[str, object]:
        return {
            "case_name": self.case_name,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _format_expected_count(expected_op: ExpectedOp) -> object:
    if expected_op.count is not None:
        return expected_op.count
    return {"min": expected_op.count_min, "max": expected_op.count_max}


def _severity_for_expected_op(expected_op: ExpectedOp) -> str:
    return "warning" if expected_op.confidence.lower() in {"low", "medium"} else "error"


def _accepted_gap_matches(evidence_case: EvidenceCase, op_name: str) -> bool:
    return any(op_name in gap or gap in op_name for gap in evidence_case.accepted_gaps)


def _is_tensor_cast_op(op_name: str) -> bool:
    return op_name.startswith("tensor_cast.")


def _is_communication_op(op_name: str) -> bool:
    lowered = op_name.lower()
    return any(
        token in lowered
        for token in (
            "allreduce",
            "all_reduce",
            "allgather",
            "all_gather",
            "alltoall",
            "all_to_all",
            "broadcast",
            "reduce_scatter",
            "hcom",
            "hccl",
            "collective",
        )
    )


def _missing_expected_category(expected_op: ExpectedOp, actual: ActualSummary) -> str:
    if expected_op.name.startswith("profiling."):
        return "FUSION_GAP_ACCEPTED_OR_NEEDS_REVIEW"
    if _is_communication_op(expected_op.name):
        return "COMMUNICATION_GAP"
    if _is_tensor_cast_op(expected_op.name) and not any(_is_tensor_cast_op(name) for name in actual.ops):
        return "PATCH_SEMANTICS_MISSING"
    return "OP_MAPPING_MISSING"


def _coverage_issues(actual: ActualSummary) -> List[VerificationIssue]:
    issues: List[VerificationIssue] = []
    for model_name, coverage in actual.coverage.items():
        if not isinstance(coverage, dict):
            continue
        m1 = coverage.get("m1") or coverage
        hit_rate = m1.get("m1_raw_op_count_hr") if isinstance(m1, dict) else None
        if hit_rate is not None and hit_rate < 1.0:
            issues.append(
                VerificationIssue(
                    category="PROFILING_SHAPE_MISSING",
                    message="Profiling coverage is incomplete for empirical performance model.",
                    severity="warning",
                    expected={"m1_raw_op_count_hr": 1.0},
                    actual={"model": model_name, "m1_raw_op_count_hr": hit_rate},
                    evidence_path="actual.coverage",
                )
            )
    return issues


def verify_evidence_case(
    evidence_case: EvidenceCase,
    actual: ActualSummary,
    extra_op_time_ratio: float = 0.05,
    extra_op_min_time_s: float = 0.0,
) -> VerificationReport:
    issues: List[VerificationIssue] = []

    if evidence_case.total_forward is not None and not evidence_case.total_forward.matches(actual.total_forward_time_s):
        issues.append(
            VerificationIssue(
                category="LATENCY_MODEL_MISMATCH",
                message="Total forward time is outside tolerance.",
                expected=evidence_case.total_forward.time_s,
                actual=actual.total_forward_time_s,
                evidence_path=f"cases[{evidence_case.name}].expected.total_forward.time_s",
            )
        )

    expected_names = {op.name for op in evidence_case.major_ops}
    for index, expected_op in enumerate(evidence_case.major_ops):
        actual_op = actual.get_op(expected_op.name)
        path = f"cases[{evidence_case.name}].expected.major_ops[{index}]"
        if actual_op is None:
            category = _missing_expected_category(expected_op, actual)
            issues.append(
                VerificationIssue(
                    category=category,
                    message=f"Expected major op {expected_op.name!r} is missing from actual summary.",
                    severity=_severity_for_expected_op(expected_op),
                    expected=expected_op.name,
                    actual=None,
                    evidence_path=f"{path}.name",
                )
            )
            continue

        if not expected_op.count_matches(actual_op.count):
            issues.append(
                VerificationIssue(
                    category="OP_COUNT_MISMATCH",
                    message=f"Op {expected_op.name!r} call count is outside expectation.",
                    severity=_severity_for_expected_op(expected_op),
                    expected=_format_expected_count(expected_op),
                    actual=actual_op.count,
                    evidence_path=f"{path}.count",
                )
            )

        if expected_op.total_time is not None and not expected_op.total_time.matches(actual_op.total_time_s):
            issues.append(
                VerificationIssue(
                    category="LATENCY_MODEL_MISMATCH",
                    message=f"Op {expected_op.name!r} total time is outside tolerance.",
                    severity=_severity_for_expected_op(expected_op),
                    expected=expected_op.total_time.time_s,
                    actual=actual_op.total_time_s,
                    evidence_path=f"{path}.total_time_s",
                )
            )

    extra_threshold = max(actual.total_forward_time_s * extra_op_time_ratio, extra_op_min_time_s)
    if extra_threshold > 0:
        for op in actual.high_time_ops(extra_threshold):
            if op.name in expected_names or _accepted_gap_matches(evidence_case, op.name):
                continue
            category = "COMMUNICATION_GAP" if _is_communication_op(op.name) else "FUSION_GAP_ACCEPTED_OR_NEEDS_REVIEW"
            issues.append(
                VerificationIssue(
                    category=category,
                    message=f"Actual high-time op {op.name!r} is not declared in evidence.",
                    severity="warning",
                    expected=None,
                    actual={"count": op.count, "total_time_s": op.total_time_s},
                    evidence_path=f"cases[{evidence_case.name}].expected.major_ops",
                )
            )

    issues.extend(_coverage_issues(actual))

    blocking = [issue for issue in issues if issue.severity == "error"]
    return VerificationReport(
        case_name=evidence_case.name,
        passed=not blocking,
        issues=issues,
    )

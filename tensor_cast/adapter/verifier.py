"""Run-through verification for streamlined model adaptation.

Verification has two guarantees, both free of measured profiling data:

1. the simulation runs the case end to end without raising;
2. key TensorCast operator call counts (attention-style ops, MoE gating ops)
   match the expectations derived from the model's public structure.

Full-coverage op-shape/dtype checks and profiling-based precision comparison
are handled by the downstream precision workflow, not here.
"""

import dataclasses
from typing import Any, Dict, List, Optional

from .actual import ActualSummary
from .expectations import (
    KEY_OP_CATEGORIES,
    MOE_GATING_CATEGORY,
    ExpectationBasis,
    classify_op,
    summarize_key_ops,
    tensor_cast_op_token,
    unclassified_tensor_cast_ops,
)


@dataclasses.dataclass(frozen=True)
class VerificationIssue:
    category: str
    message: str
    severity: str = "error"
    expected: Optional[object] = None
    actual: Optional[object] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "category": self.category,
            "message": self.message,
            "severity": self.severity,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclasses.dataclass(frozen=True)
class KeyOpCheck:
    category: str
    expected_count: int
    actual_count: int
    op_breakdown: Dict[str, int]
    basis: str
    severity: str = "error"

    @property
    def matched(self) -> bool:
        return self.expected_count == self.actual_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "matched": self.matched,
            "op_breakdown": dict(self.op_breakdown),
            "basis": self.basis,
            "severity": self.severity,
        }


@dataclasses.dataclass(frozen=True)
class SimulationVerificationReport:
    model_id: str
    model_type: Optional[str]
    case_name: str
    passed: bool
    case_input: Dict[str, Any]
    simulation: Dict[str, Any]
    expectations_basis: Dict[str, Any]
    key_op_checks: List[KeyOpCheck]
    issues: List[VerificationIssue]
    suggestions: List[Dict[str, Any]]
    actual_summary: Dict[str, Any]
    ai_tasks: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "case_name": self.case_name,
            "passed": self.passed,
            "case_input": dict(self.case_input),
            "simulation": dict(self.simulation),
            "expectations_basis": dict(self.expectations_basis),
            "key_op_checks": [check.to_dict() for check in self.key_op_checks],
            "issues": [issue.to_dict() for issue in self.issues],
            "suggestions": list(self.suggestions),
            "actual_summary": dict(self.actual_summary),
            "ai_tasks": list(self.ai_tasks),
        }
        return data


def _has_tensor_cast_ops(actual: ActualSummary) -> bool:
    return any("tensor_cast." in name for name in actual.ops)


# MoE-path TensorCast ops invoked by the MoE wrapper forward regardless of
# the gating branch (routing, dispatch/combine). Their presence proves the
# MoE blocks execute on the TensorCast path even when gating goes through
# the standard torch.topk / pre-computed route and emits no tensor_cast
# gating op. Tokens are matched exactly (dispatch_ffn_combine by prefix to
# cover its quant variants); grouped_matmul is deliberately excluded because
# its many variants are not exclusive to the MoE path.
_MOE_PATH_EXACT_TOKENS = frozenset({"init_routing_v2", "unpermute_tokens"})
_MOE_PATH_TOKEN_PREFIXES = ("dispatch_ffn_combine",)


def _is_moe_path_op(name: str) -> bool:
    token = tensor_cast_op_token(name)
    if token is None or classify_op(name) is not None:
        return False
    return token in _MOE_PATH_EXACT_TOKENS or any(token.startswith(prefix) for prefix in _MOE_PATH_TOKEN_PREFIXES)


def _moe_path_op_count(actual: ActualSummary) -> int:
    return sum(op.count for name, op in actual.ops.items() if _is_moe_path_op(name))


def _check_severity(user_input: Any) -> str:
    """Key-op equality is only asserted for basic cases.

    MTP and pipeline-parallel executions change how often key ops fire per
    layer, so their mismatches are reported as warnings instead of errors.
    """
    if getattr(user_input, "num_mtp_tokens", 0) and int(user_input.num_mtp_tokens) > 0:
        return "warning"
    if getattr(user_input, "pp_size", 1) and int(user_input.pp_size) > 1:
        return "warning"
    return "error"


def verify_key_op_counts(
    basis: ExpectationBasis,
    actual: ActualSummary,
    user_input: Any = None,
) -> List[KeyOpCheck]:
    severity = _check_severity(user_input) if user_input is not None else "error"
    actual_by_category = summarize_key_ops(actual.ops)
    checks: List[KeyOpCheck] = []
    for category in KEY_OP_CATEGORIES:
        expectation = basis.expectations[category]
        actual_entry = actual_by_category.get(category, {"count": 0, "op_breakdown": {}})
        checks.append(
            KeyOpCheck(
                category=category,
                expected_count=expectation.expected_count,
                actual_count=int(actual_entry["count"]),
                op_breakdown=dict(actual_entry["op_breakdown"]),
                basis=expectation.basis,
                severity=severity,
            )
        )
    return checks


def collect_verification_issues(
    checks: List[KeyOpCheck],
    basis: ExpectationBasis,
    actual: Optional[ActualSummary],
    user_input: Any = None,
    moe_patch_replacements: Optional[int] = None,
) -> List[VerificationIssue]:
    issues: List[VerificationIssue] = []
    if not basis.text_attention_modules:
        issues.append(
            VerificationIssue(
                category="STRUCTURE_SCAN_EMPTY",
                message=("No attention-like modules were found in the structure scan; expectations cannot be derived."),
                expected=">=1 attention modules",
                actual=0,
            )
        )
    if actual is None:
        return issues
    if not _has_tensor_cast_ops(actual):
        issues.append(
            VerificationIssue(
                category="NO_TENSOR_CAST_OPS",
                message=(
                    "The simulation recorded no tensor_cast ops. The model likely fell "
                    "back to un-adapted HF modules; check ModelProfile registration "
                    "and patch replacement."
                ),
            )
        )
    unclassified = unclassified_tensor_cast_ops(actual.ops)
    for check in checks:
        if check.matched:
            continue
        if check.category == MOE_GATING_CATEGORY and check.expected_count > 0 and check.actual_count == 0:
            if moe_patch_replacements or _moe_path_op_count(actual) > 0:
                # MoE blocks execute on the TensorCast path (a non-empty MoE
                # patch report or routing/dispatch ops prove it) and this
                # model gates through the standard non-tensor-cast path (HF
                # pre-computed top-k or torch.topk), so there is no
                # tensor_cast gating op to count. An empty patch report
                # (module name mismatch, missing fields) is not evidence and
                # falls through to MOE_NOT_ADAPTED.
                continue
            issues.append(
                VerificationIssue(
                    category="MOE_NOT_ADAPTED",
                    message=(
                        f"{check.expected_count} MoE modules were found by the structure scan "
                        "but no MoE patch was applied and no TensorCast MoE-path ops were "
                        "invoked; the MoE path fell back to un-adapted HF modules. "
                        "Register or fix the ModelProfile MoE fields."
                    ),
                    severity="error",
                    expected=check.expected_count,
                    actual=check.actual_count,
                )
            )
            continue
        message = (
            f"Key op category {check.category!r} call count is "
            f"{check.actual_count}, expected {check.expected_count} ({check.basis})."
        )
        if check.actual_count == 0:
            category = "KEY_OP_MISSING"
            message += " No matching TensorCast ops were invoked; check module replacement and op routing."
            if unclassified and check.category == "attention":
                message += (
                    " Unclassified tensor_cast ops recorded (name x count): "
                    + ", ".join(f"{name} x{count}" for name, count in unclassified)
                    + ". If one of them is a new attention computation core, register it in"
                    " _ATTENTION_CORE_OPS (tensor_cast/adapter/expectations.py)."
                )
        else:
            category = "OP_COUNT_MISMATCH"
            message += " Check layer overrides, MTP, vision input, or missing wrapper replacement."
        issues.append(
            VerificationIssue(
                category=category,
                message=message,
                severity=check.severity,
                expected=check.expected_count,
                actual=check.actual_count,
            )
        )
    if user_input is not None and _check_severity(user_input) == "warning":
        reasons = []
        if int(getattr(user_input, "num_mtp_tokens", 0) or 0) > 0:
            reasons.append("num_mtp_tokens > 0")
        if int(getattr(user_input, "pp_size", 1) or 1) > 1:
            reasons.append("pp_size > 1")
        issues.append(
            VerificationIssue(
                category="EXPECTATION_DEGRADED",
                message=(
                    "Key-op count checks are degraded to warnings for this case ("
                    + ", ".join(reasons)
                    + "); rerun verify with a basic case (no MTP, pp_size=1) for "
                    "strict equality."
                ),
                severity="warning",
            )
        )
    return issues

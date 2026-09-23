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

import contextlib
import copy
import dataclasses
import traceback as traceback_module
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.transformers.custom_model_registry import (
    get_model_profile,
    ignore_model_profiles,
)

from .advisor import AdvisorSuggestion, advise
from .context import AdaptationContext, user_input_to_case_dict
from .expectations import derive_key_op_expectations
from .inspect import ModelStructureFacts, ProfileCandidate, inspect_model_structure
from .patch_report import PatchReport
from .patch_discovery import classify_patch_failure
from .profile import profile_to_review_dict, validate_profile
from .profile_draft import render_builtin_profile_draft
from .questions import build_human_questions
from .recipes import materialization_hints_to_dict, materialize_profile_candidate
from .runner import run_simulation_case
from .verifier import (
    KeyOpCheck,
    SimulationVerificationReport,
    VerificationIssue,
    collect_verification_issues,
    verify_key_op_counts,
)


def _dataclass_to_dict(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return {key: _dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dataclass_to_dict(item) for item in value]
    return value


def structure_to_dict(structure: ModelStructureFacts) -> Dict[str, Any]:
    return _dataclass_to_dict(structure)


def candidate_to_dict(candidate: ProfileCandidate) -> Dict[str, Any]:
    return _dataclass_to_dict(candidate)


def suggestions_to_dict(
    suggestions: Iterable[AdvisorSuggestion],
) -> List[Dict[str, Any]]:
    return [_dataclass_to_dict(suggestion) for suggestion in suggestions]


def patch_reports_to_dict(reports: Iterable[PatchReport]) -> List[Dict[str, Any]]:
    return [report.to_dict() for report in reports]


@dataclasses.dataclass(frozen=True)
class DoctorReport:
    model_id: str
    model_type: Optional[str]
    adaptation_context: Optional[Dict[str, Any]]
    human_questions: List[Dict[str, Any]]
    patch_discovery: Optional[Dict[str, Any]]
    ai_tasks: List[Dict[str, Any]]
    ignored_existing_profiles: List[str]
    profile: Optional[Dict[str, Any]]
    profile_validation: Optional[Dict[str, Any]]
    structure: Dict[str, Any]
    candidate: Dict[str, Any]
    candidate_profile: Optional[Dict[str, Any]]
    candidate_profile_draft: Optional[str]
    candidate_profile_validation: Optional[Dict[str, Any]]
    materialization_hints: List[Dict[str, Any]]
    patch_reports: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def run_model_doctor(
    user_input: UserInputConfig,
    build_runtime_model: bool = True,
    adaptation_context: Optional[AdaptationContext] = None,
    ignore_existing_profiles: Optional[Sequence[str]] = None,
    patch_failure_text: Optional[str] = None,
) -> DoctorReport:
    ignored_profiles = list(ignore_existing_profiles or [])
    profile_context = ignore_model_profiles(ignored_profiles) if ignored_profiles else contextlib.nullcontext()
    # The structure scan needs the complete module tree: repeated-layer reuse
    # collapses identical layers into copy wrappers and would under-report the
    # module counts that key-op expectations are derived from. The runtime
    # verification run keeps the user's repetition setting (region replay
    # restores full op counts either way).
    scan_input = copy.copy(user_input)
    scan_input.disable_repetition = True
    with profile_context:
        model = build_model(scan_input) if build_runtime_model else None
        if model is None:
            raise ValueError(
                "build_runtime_model=False is not supported yet because structure scan needs a model instance."
            )

        structure, candidate = inspect_model_structure(model)
        candidate_profile = materialize_profile_candidate(structure, candidate)
        candidate_profile_validation = validate_profile(candidate_profile)
        profile = get_model_profile(structure.model_type) if structure.model_type else None
        profile_validation = validate_profile(profile) if profile is not None else None
        patch_reports = list(getattr(model, "patch_reports", []))
        suggestions = advise(
            structure=structure,
            candidate=candidate,
            patch_reports=patch_reports,
        )
        patch_discovery = None
        ai_tasks = []
        if patch_failure_text:
            patch_discovery = classify_patch_failure(
                patch_failure_text,
                model_type=structure.model_type,
                failed_command=(adaptation_context.raw_command if adaptation_context is not None else None),
            ).to_dict()
            ai_tasks.extend(patch_discovery.get("ai_tasks", []))
        candidate_profile_review = profile_to_review_dict(candidate_profile)
        patch_method_name = None
        if patch_discovery and patch_discovery.get("requires_patch"):
            patch_method_name = patch_discovery.get("suggested_patch_method_name")
        candidate_profile_draft = render_builtin_profile_draft(
            candidate_profile_review,
            patch_method_name=patch_method_name,
        )
        human_questions = build_human_questions(
            structure=structure,
            candidate=candidate,
        )
    return DoctorReport(
        model_id=user_input.model_id,
        model_type=structure.model_type,
        adaptation_context=adaptation_context.to_dict() if adaptation_context is not None else None,
        human_questions=human_questions,
        patch_discovery=patch_discovery,
        ai_tasks=ai_tasks,
        ignored_existing_profiles=ignored_profiles,
        profile=profile_to_review_dict(profile) if profile is not None else None,
        profile_validation=(_dataclass_to_dict(profile_validation) if profile_validation is not None else None),
        structure=structure_to_dict(structure),
        candidate=candidate_to_dict(candidate),
        candidate_profile=candidate_profile_review,
        candidate_profile_draft=candidate_profile_draft,
        candidate_profile_validation=_dataclass_to_dict(candidate_profile_validation),
        materialization_hints=materialization_hints_to_dict(structure, candidate),
        patch_reports=patch_reports_to_dict(patch_reports),
        suggestions=suggestions_to_dict(suggestions),
    )


def default_verification_case_name(user_input: UserInputConfig) -> str:
    model_name = user_input.model_id.rstrip("/").split("/")[-1].lower().replace("_", "-")
    phase = "decode" if user_input.decode else "prefill"
    return f"{model_name}-{phase}"


def _moe_patch_replacements(patch_reports: List[Dict[str, Any]]) -> Optional[int]:
    """Total MoE-block replacements recorded during the doctor build, if any."""
    total = None
    for report in patch_reports or []:
        if report.get("pass_name") != "MoE":
            continue
        count = len(report.get("replaced_modules") or [])
        total = count if total is None else total + count
    return total


def _simulation_failure_report(
    user_input: UserInputConfig,
    case_name: str,
    error: BaseException,
    doctor_report: Optional[DoctorReport] = None,
) -> SimulationVerificationReport:
    """Build a failed report for a simulation that raised.

    The failure is captured instead of propagating so that callers (the AI
    skill or the downstream precision workflow) always get a structured
    report: the error details plus patch-discovery AI tasks that guide the
    bug fix, which is the critical loop for reaching a runnable simulation.
    """
    tb_text = "".join(traceback_module.format_exception(type(error), error, error.__traceback__))
    model_type = doctor_report.model_type if doctor_report is not None else None
    patch_discovery = classify_patch_failure(
        tb_text,
        model_type=model_type,
        failed_command=None,
    ).to_dict()
    structure = doctor_report.structure if doctor_report is not None else {}
    basis = derive_key_op_expectations(structure, user_input)
    issue = VerificationIssue(
        category="SIMULATION_ERROR",
        message=(
            f"The simulation case did not complete: {type(error).__name__}: {error}. "
            "Patch-discovery findings and AI assistance tasks are attached in "
            "ai_tasks; follow them to fix the model adaptation, then rerun verify."
        ),
        severity="error",
        expected="simulation completes",
        actual=f"{type(error).__name__}: {error}",
    )
    issues = [issue]
    return SimulationVerificationReport(
        model_id=user_input.model_id,
        model_type=model_type,
        case_name=case_name,
        passed=False,
        case_input=user_input_to_case_dict(user_input),
        simulation={
            "ran_without_error": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": tb_text,
        },
        expectations_basis=basis.to_dict(),
        key_op_checks=[],
        issues=issues,
        suggestions=suggestions_to_dict(advise(verification_issues=issues)),
        actual_summary={},
        ai_tasks=list(patch_discovery.get("ai_tasks", [])),
    )


def run_simulation_verification(
    user_input: UserInputConfig,
    case_name: Optional[str] = None,
) -> SimulationVerificationReport:
    """Verify a newly adapted model end to end without measured data.

    Runs the doctor structure scan, executes one simulation case, and
    reconciles key TensorCast op call counts against structure-derived
    expectations. A simulation that raises is captured as a failed report
    with patch-discovery AI tasks attached instead of propagating.
    """
    name = case_name or default_verification_case_name(user_input)
    doctor_report: Optional[DoctorReport] = None
    try:
        doctor_report = run_model_doctor(user_input)
        result = run_simulation_case(user_input, case_name=name)
    except Exception as error:  # noqa: BLE001 - reported as SIMULATION_ERROR
        return _simulation_failure_report(user_input, name, error, doctor_report=doctor_report)
    moe_patch_replacements = _moe_patch_replacements(doctor_report.patch_reports)
    basis = derive_key_op_expectations(
        doctor_report.structure,
        user_input,
        moe_patch_replacements=moe_patch_replacements,
    )
    checks: List[KeyOpCheck] = verify_key_op_counts(basis, result.summary, user_input)
    issues: List[VerificationIssue] = collect_verification_issues(
        checks,
        basis,
        result.summary,
        user_input,
        moe_patch_replacements=moe_patch_replacements,
    )
    suggestions = advise(verification_issues=issues)
    passed = not any(issue.severity == "error" for issue in issues)
    return SimulationVerificationReport(
        model_id=user_input.model_id,
        model_type=doctor_report.model_type,
        case_name=name,
        passed=passed,
        case_input=user_input_to_case_dict(user_input),
        simulation={
            "ran_without_error": True,
            "total_forward_time_s": result.summary.total_forward_time_s,
            "perf_model_name": result.summary.perf_model_name,
        },
        expectations_basis=basis.to_dict(),
        key_op_checks=checks,
        issues=issues,
        suggestions=suggestions_to_dict(suggestions),
        actual_summary=result.summary.to_dict(),
    )

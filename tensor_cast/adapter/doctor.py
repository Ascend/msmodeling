import contextlib
import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.transformers.custom_model_registry import (
    get_model_profile,
    ignore_model_profiles,
)

from .actual import ActualSummary
from .advisor import AdvisorSuggestion, advise
from .context import AdaptationContext
from .evidence import EvidenceDocument, load_evidence
from .evidence_builder import build_evidence_draft
from .hints import HintLedger
from .inspect import ModelStructureFacts, ProfileCandidate, inspect_model_structure
from .insight import RawInsightSummary
from .patch_report import PatchReport
from .patch_discovery import classify_patch_failure
from .profile import profile_to_review_dict, validate_profile
from .profile_draft import render_builtin_profile_draft
from .questions import build_human_questions
from .recipes import materialization_hints_to_dict, materialize_profile_candidate
from .runner import run_actual_case
from .verifier import VerificationReport, verify_evidence_case


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
    raw_insight_summary: Optional[Dict[str, Any]]
    user_hints: Optional[Dict[str, Any]]
    hint_conflicts: List[Dict[str, Any]]
    evidence_draft: Optional[Dict[str, Any]]
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
    raw_insight: Optional[RawInsightSummary] = None,
    hints: Optional[HintLedger] = None,
    ignore_existing_profiles: Optional[Sequence[str]] = None,
    patch_failure_text: Optional[str] = None,
) -> DoctorReport:
    ignored_profiles = list(ignore_existing_profiles or [])
    profile_context = ignore_model_profiles(ignored_profiles) if ignored_profiles else contextlib.nullcontext()
    with profile_context:
        model = build_model(user_input) if build_runtime_model else None
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
        evidence_draft = None
        if adaptation_context is not None and raw_insight is not None:
            evidence_draft = build_evidence_draft(
                adaptation_context,
                raw_insight,
                hints=hints,
            )
        hint_conflicts = (
            [] if hints is None else [conflict.to_dict() for conflict in hints.conflicts_with_raw_insight(raw_insight)]
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
            evidence_draft=evidence_draft,
            hint_conflicts=hint_conflicts,
        )
    return DoctorReport(
        model_id=user_input.model_id,
        model_type=structure.model_type,
        adaptation_context=adaptation_context.to_dict() if adaptation_context is not None else None,
        raw_insight_summary=raw_insight.to_dict(top_n=20) if raw_insight is not None else None,
        user_hints=hints.to_dict() if hints is not None else None,
        hint_conflicts=hint_conflicts,
        evidence_draft=evidence_draft,
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


@dataclasses.dataclass(frozen=True)
class EvidenceRunReport:
    evidence_model: Dict[str, Any]
    evidence_cases: List[Dict[str, Any]]
    actual_summaries: List[Dict[str, Any]]
    verification_reports: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]

    @property
    def passed(self) -> bool:
        return all(report.get("passed", False) for report in self.verification_reports)

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["passed"] = self.passed
        return data


def verify_evidence_with_actuals(
    evidence: EvidenceDocument,
    actuals: Dict[str, ActualSummary],
) -> EvidenceRunReport:
    verification_reports: List[VerificationReport] = []
    suggestions: List[AdvisorSuggestion] = []
    for case in evidence.cases:
        actual = actuals.get(case.name)
        if actual is None:
            raise ValueError(f"No actual summary was provided for evidence case {case.name!r}.")
        verification = verify_evidence_case(case, actual)
        verification_reports.append(verification)
        suggestions.extend(advise(actual=actual, verification=verification))
    return EvidenceRunReport(
        evidence_model=evidence.model,
        evidence_cases=[_dataclass_to_dict(case) for case in evidence.cases],
        actual_summaries=[actual.to_dict() for actual in actuals.values()],
        verification_reports=[report.to_dict() for report in verification_reports],
        suggestions=suggestions_to_dict(suggestions),
    )


def run_evidence_verification(evidence_path: str, user_input: UserInputConfig) -> EvidenceRunReport:
    evidence = load_evidence(evidence_path)
    if not user_input.model_id:
        model_id = evidence.model.get("model_id")
        if model_id:
            user_input.model_id = str(model_id)
    actuals: Dict[str, ActualSummary] = {}
    for case in evidence.cases:
        result = run_actual_case(case, user_input)
        actuals[case.name] = result.summary
    return verify_evidence_with_actuals(evidence, actuals)

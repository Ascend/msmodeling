import dataclasses
from typing import List, Optional

from .actual import ActualSummary
from .inspect import ModelStructureFacts, ProfileCandidate
from .patch_report import PatchReport
from .verifier import VerificationReport


@dataclasses.dataclass(frozen=True)
class AdvisorSuggestion:
    code: str
    message: str
    confidence: str
    evidence: str


def _has_category(report: VerificationReport, category: str) -> bool:
    return any(issue.category == category for issue in report.issues)


def advise(
    structure: Optional[ModelStructureFacts] = None,
    candidate: Optional[ProfileCandidate] = None,
    patch_reports: Optional[List[PatchReport]] = None,
    actual: Optional[ActualSummary] = None,
    verification: Optional[VerificationReport] = None,
) -> List[AdvisorSuggestion]:
    suggestions: List[AdvisorSuggestion] = []
    patch_reports = [] if patch_reports is None else patch_reports

    for report in patch_reports:
        if report.target_module_name and report.replacement_count == 0:
            suggestions.append(
                AdvisorSuggestion(
                    code="PATCH_NOT_APPLIED",
                    message=(
                        f"{report.pass_name} matched {len(report.matched_modules)} "
                        f"{report.target_module_name} modules but replaced none. "
                        "Check module_name and field name overrides."
                    ),
                    confidence="high",
                    evidence=f"patch_reports[{report.pass_name}].skipped_modules",
                )
            )
        elif report.expected_replacements is not None and report.replacement_count < report.expected_replacements:
            suggestions.append(
                AdvisorSuggestion(
                    code="PATCH_COUNT_MISMATCH",
                    message=(
                        f"{report.pass_name} replaced {report.replacement_count} modules, "
                        f"expected {report.expected_replacements}. Check hybrid layers or patterns."
                    ),
                    confidence="high",
                    evidence=f"patch_reports[{report.pass_name}].replaced_modules",
                )
            )

    if verification is not None:
        if _has_category(verification, "OP_COUNT_MISMATCH"):
            suggestions.append(
                AdvisorSuggestion(
                    code="OP_COUNT_MISMATCH",
                    message=(
                        "Major op count differs from evidence. Check repetition, MTP/layer override, "
                        "or missing TensorCast wrapper replacement."
                    ),
                    confidence="medium",
                    evidence="verification.issues[OP_COUNT_MISMATCH]",
                )
            )
        if _has_category(verification, "OP_MAPPING_MISSING"):
            suggestions.append(
                AdvisorSuggestion(
                    code="OP_MAPPING_MISSING",
                    message=(
                        "Expected profiling op is absent in actual summary. Check op naming/mapping "
                        "or whether the model falls back to original HF modules."
                    ),
                    confidence="medium",
                    evidence="verification.issues[OP_MAPPING_MISSING]",
                )
            )
        if _has_category(verification, "LATENCY_MODEL_MISMATCH"):
            message = "Latency differs while deterministic counts may still match. Check profiling coverage first."
            if actual and actual.coverage:
                message += " Coverage data is available in ActualSummary.coverage."
            suggestions.append(
                AdvisorSuggestion(
                    code="LATENCY_MODEL_MISMATCH",
                    message=message,
                    confidence="medium",
                    evidence="verification.issues[LATENCY_MODEL_MISMATCH]",
                )
            )
        if _has_category(verification, "PROFILING_SHAPE_MISSING"):
            suggestions.append(
                AdvisorSuggestion(
                    code="PROFILING_SHAPE_MISSING",
                    message=(
                        "Profiling hit rate is incomplete. Add missing op mapping/shape records "
                        "before treating latency mismatch as a model-structure bug."
                    ),
                    confidence="high",
                    evidence="verification.issues[PROFILING_SHAPE_MISSING]",
                )
            )
        if _has_category(verification, "PATCH_SEMANTICS_MISSING"):
            suggestions.append(
                AdvisorSuggestion(
                    code="PATCH_SEMANTICS_MISSING",
                    message=(
                        "Expected TensorCast wrapper ops are absent. Check ModelProfile fields, "
                        "runtime patch_method, and PatchReport replacement counts."
                    ),
                    confidence="high",
                    evidence="verification.issues[PATCH_SEMANTICS_MISSING]",
                )
            )
        if _has_category(verification, "COMMUNICATION_GAP"):
            suggestions.append(
                AdvisorSuggestion(
                    code="COMMUNICATION_GAP",
                    message=(
                        "Communication operators appear missing or unexplained. Check TP/DP/EP sizes, "
                        "collective naming, and whether communication evidence should be added or accepted."
                    ),
                    confidence="medium",
                    evidence="verification.issues[COMMUNICATION_GAP]",
                )
            )

    if structure and candidate:
        if structure.moe_like_modules and candidate.moe_module_name is None:
            suggestions.append(
                AdvisorSuggestion(
                    code="PROFILE_FIELD_MISSING_OR_WRONG",
                    message="MoE-like modules were found but no moe_module_name candidate was generated.",
                    confidence="medium",
                    evidence="ModelStructureFacts.moe_like_modules",
                )
            )
        if "deepseek_like_mla" in structure.known_recipe_matches and candidate.mla_module_name is None:
            suggestions.append(
                AdvisorSuggestion(
                    code="PROFILE_FIELD_MISSING_OR_WRONG",
                    message="MLA-like attention modules were found but no mla_module_name candidate was generated.",
                    confidence="low",
                    evidence="ModelStructureFacts.known_recipe_matches[deepseek_like_mla]",
                )
            )

    return suggestions

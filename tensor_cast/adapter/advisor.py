import dataclasses
from typing import Iterable, List, Optional

from .inspect import ModelStructureFacts, ProfileCandidate
from .patch_report import PatchReport
from .verifier import VerificationIssue


@dataclasses.dataclass(frozen=True)
class AdvisorSuggestion:
    code: str
    message: str
    confidence: str
    evidence: str


def _has_category(issues: Iterable[VerificationIssue], category: str) -> bool:
    return any(issue.category == category for issue in issues)


def advise(
    structure: Optional[ModelStructureFacts] = None,
    candidate: Optional[ProfileCandidate] = None,
    patch_reports: Optional[List[PatchReport]] = None,
    verification_issues: Optional[List[VerificationIssue]] = None,
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

    if verification_issues:
        if _has_category(verification_issues, "KEY_OP_MISSING"):
            suggestions.append(
                AdvisorSuggestion(
                    code="KEY_OP_MISSING",
                    message=(
                        "A key op category was never invoked. Check ModelProfile fields, "
                        "patch replacement counts, and whether the model fell back to "
                        "original HF modules."
                    ),
                    confidence="high",
                    evidence="verification.issues[KEY_OP_MISSING]",
                )
            )
        if _has_category(verification_issues, "OP_COUNT_MISMATCH"):
            suggestions.append(
                AdvisorSuggestion(
                    code="OP_COUNT_MISMATCH",
                    message=(
                        "Key op call count differs from the structure-derived expectation. "
                        "Check layer overrides, MTP, vision input, or missing wrapper replacement."
                    ),
                    confidence="medium",
                    evidence="verification.issues[OP_COUNT_MISMATCH]",
                )
            )
        if _has_category(verification_issues, "NO_TENSOR_CAST_OPS"):
            suggestions.append(
                AdvisorSuggestion(
                    code="NO_TENSOR_CAST_OPS",
                    message=(
                        "No tensor_cast ops were recorded by the simulation. The model is not "
                        "adapted yet; register a ModelProfile and rerun doctor and verify."
                    ),
                    confidence="high",
                    evidence="verification.issues[NO_TENSOR_CAST_OPS]",
                )
            )
        if _has_category(verification_issues, "STRUCTURE_SCAN_EMPTY"):
            suggestions.append(
                AdvisorSuggestion(
                    code="STRUCTURE_SCAN_EMPTY",
                    message=(
                        "The structure scan found no attention-like modules. Check the model "
                        "build path and installed transformers source."
                    ),
                    confidence="high",
                    evidence="verification.issues[STRUCTURE_SCAN_EMPTY]",
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

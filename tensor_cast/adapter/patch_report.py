import dataclasses
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class PatchIssue:
    module_name: str
    module_type: str
    reason: str
    missing_fields: Tuple[str, ...] = ()
    candidate_aliases: Dict[str, Tuple[str, ...]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PatchReport:
    pass_name: str
    target_module_name: Optional[str] = None
    expected_replacements: Optional[int] = None
    matched_modules: List[str] = dataclasses.field(default_factory=list)
    replaced_modules: List[str] = dataclasses.field(default_factory=list)
    skipped_modules: List[PatchIssue] = dataclasses.field(default_factory=list)
    replacements: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    unmatched_patterns: List[str] = dataclasses.field(default_factory=list)

    @property
    def replacement_count(self) -> int:
        return len(self.replaced_modules)

    def add_replacement(self, name: str, old_type: str, new_type: str, fields: Optional[Dict[str, Any]] = None) -> None:
        self.replaced_modules.append(name)
        self.replacements[name] = {
            "old_type": old_type,
            "new_type": new_type,
            "fields": {} if fields is None else fields,
        }

    def add_skip(
        self,
        name: str,
        module_type: str,
        reason: str,
        missing_fields: Tuple[str, ...] = (),
        candidate_aliases: Optional[Dict[str, Tuple[str, ...]]] = None,
    ) -> None:
        self.skipped_modules.append(
            PatchIssue(
                module_name=name,
                module_type=module_type,
                reason=reason,
                missing_fields=missing_fields,
                candidate_aliases={} if candidate_aliases is None else candidate_aliases,
            )
        )

    def validate(self, strict: bool = False) -> None:
        if self.target_module_name and strict and self.replacement_count == 0:
            raise RuntimeError(
                f"{self.pass_name} did not replace any {self.target_module_name} modules. "
                "Inspect PatchReport.skipped_modules for missing fields or alias candidates."
            )
        if self.expected_replacements is not None and self.replacement_count < self.expected_replacements:
            message = (
                f"{self.pass_name} replaced {self.replacement_count} modules, "
                f"expected at least {self.expected_replacements}."
            )
            if strict:
                raise RuntimeError(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_name": self.pass_name,
            "target_module_name": self.target_module_name,
            "expected_replacements": self.expected_replacements,
            "matched_modules": list(self.matched_modules),
            "replaced_modules": list(self.replaced_modules),
            "skipped_modules": [dataclasses.asdict(issue) for issue in self.skipped_modules],
            "replacements": dict(self.replacements),
            "unmatched_patterns": list(self.unmatched_patterns),
        }


def attach_patch_report(model: Any, report: PatchReport) -> None:
    reports = getattr(model, "patch_reports", None)
    if reports is None:
        reports = []
        setattr(model, "patch_reports", reports)
    reports.append(report)

import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tensor_cast.model_config import MlaFieldNames, MoEFieldNames


@dataclasses.dataclass(frozen=True)
class ProfileValidationIssue:
    field: str
    message: str
    severity: str = "error"


@dataclasses.dataclass(frozen=True)
class ProfileValidationReport:
    model_type: str
    issues: Tuple[ProfileValidationIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.severity == "error"]
        if not errors:
            return
        details = "; ".join(f"{issue.field}: {issue.message}" for issue in errors)
        raise ValueError(f"Invalid ModelProfile for {self.model_type!r}: {details}")


def _validate_non_empty_string(
    issues: List[ProfileValidationIssue],
    field: str,
    value: Optional[str],
    required: bool = False,
) -> None:
    if value is None:
        if required:
            issues.append(ProfileValidationIssue(field, "must be set"))
        return
    if not isinstance(value, str) or not value.strip():
        issues.append(ProfileValidationIssue(field, "must be a non-empty string"))


def _field_values(field_names: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(field_names, dict):
        return field_names.items()
    if not dataclasses.is_dataclass(field_names):
        return ()
    return ((field.name, getattr(field_names, field.name)) for field in dataclasses.fields(field_names))


def _validate_field_name_values(
    issues: List[ProfileValidationIssue],
    prefix: str,
    field_names: Any,
    base_fields: Any,
) -> None:
    valid_fields = {field.name for field in dataclasses.fields(base_fields)}
    for field_name, value in _field_values(field_names):
        if field_name not in valid_fields:
            issues.append(
                ProfileValidationIssue(
                    f"{prefix}.{field_name}",
                    "must be a supported field name",
                )
            )
            continue
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            issues.append(
                ProfileValidationIssue(
                    f"{prefix}.{field_name}",
                    "must be None or a non-empty string",
                )
            )


def normalize_profile(profile: Any) -> Any:
    if profile.moe_field_names_override and not isinstance(profile.moe_field_names_override, dict):
        if not dataclasses.is_dataclass(profile.moe_field_names_override):
            raise TypeError("moe_field_names_override must be a dict")
        profile.moe_field_names_override = {
            field.name: getattr(profile.moe_field_names_override, field.name)
            for field in dataclasses.fields(profile.moe_field_names_override)
        }
    if profile.mla_field_names_override and not isinstance(profile.mla_field_names_override, dict):
        profile.mla_field_names_override = {
            field.name: value
            for field, value in (
                (field, getattr(profile.mla_field_names_override, field.name))
                for field in dataclasses.fields(profile.mla_field_names_override)
            )
        }
    return profile


def validate_profile(profile: Any) -> ProfileValidationReport:
    issues: List[ProfileValidationIssue] = []
    _validate_non_empty_string(issues, "model_type", profile.model_type, required=True)

    _validate_non_empty_string(issues, "moe_module_name", profile.moe_module_name)
    _validate_non_empty_string(issues, "mtp_block_module_name", profile.mtp_block_module_name)
    _validate_non_empty_string(issues, "mla_module_name", profile.mla_module_name)
    _validate_non_empty_string(issues, "model_family", profile.model_family)

    if profile.moe_module_name:
        if isinstance(profile.moe_num_experts_key, str):
            _validate_non_empty_string(
                issues,
                "moe_num_experts_key",
                profile.moe_num_experts_key,
                required=True,
            )
        elif isinstance(profile.moe_num_experts_key, list):
            if not profile.moe_num_experts_key:
                issues.append(
                    ProfileValidationIssue(
                        "moe_num_experts_key",
                        "list must not be empty when MoE is enabled",
                    )
                )
            for index, key in enumerate(profile.moe_num_experts_key):
                _validate_non_empty_string(
                    issues,
                    f"moe_num_experts_key[{index}]",
                    key,
                    required=True,
                )
        else:
            issues.append(
                ProfileValidationIssue(
                    "moe_num_experts_key",
                    "must be a string or a list of strings",
                )
            )
        _validate_field_name_values(
            issues,
            "moe_field_names_override",
            profile.moe_field_names_override or MoEFieldNames(),
            MoEFieldNames(),
        )

    if profile.mla_module_name:
        if profile.mla_module_class_type is None:
            issues.append(ProfileValidationIssue("mla_module_class_type", "must be set when MLA is enabled"))
        try:
            mla_config = profile.build_mla_config()
        except ValueError as exc:
            issues.append(ProfileValidationIssue("mla_field_names_override", str(exc)))
        else:
            _validate_field_name_values(
                issues,
                "mla_field_names_override",
                mla_config.field_names if mla_config is not None else MlaFieldNames(),
                MlaFieldNames(),
            )

    if profile.patch_method is not None and not callable(profile.patch_method):
        issues.append(ProfileValidationIssue("patch_method", "must be callable"))
    if profile.custom_expert_module_type is not None and not callable(profile.custom_expert_module_type):
        issues.append(ProfileValidationIssue("custom_expert_module_type", "must be callable"))

    return ProfileValidationReport(
        model_type=str(profile.model_type),
        issues=tuple(issues),
    )


def _normalize_override_for_review(value: Any, base_fields: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if not isinstance(value, dict):
        return value
    defaults = {field.name: getattr(base_fields, field.name) for field in dataclasses.fields(base_fields)}
    return {key: item for key, item in value.items() if item is not None and item != defaults.get(key)}


def profile_to_review_dict(profile: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for field in dataclasses.fields(profile):
        value = getattr(profile, field.name)
        if value is None:
            continue
        if field.name in {"moe_gate_returns_raw_logits", "moe_route_after_dp_transform"} and value is False:
            continue
        if field.name == "custom_expert_module_type" and (
            not profile.moe_module_name or _callable_name(value).endswith(".MoeExpertMLP")
        ):
            continue
        if field.name == "mla_module_class_type" and not profile.mla_module_name:
            continue
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        elif callable(value) and not isinstance(value, (str, bytes)):
            value = _callable_name(value)
        if field.name.endswith("_field_names_override"):
            base_fields = MoEFieldNames() if field.name.startswith("moe_") else MlaFieldNames()
            value = _normalize_override_for_review(value, base_fields)
            if not value:
                continue
        if field.name == "moe_num_experts_key" and value == "num_experts":
            continue
        data[field.name] = value
    return data


def _callable_name(value: Any) -> str:
    if callable(value) and not isinstance(value, (str, bytes)):
        return f"{value.__module__}.{value.__name__}"
    return str(value)

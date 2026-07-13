"""Centralised env-var configuration for all helpers.

Loaded via pydantic-settings from process environment. Shell entry scripts
(run_ci_gate.sh, run_nightly.sh, etc.) set env-vars with defaults.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field, ValidationError, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from scripts.helpers._errors import ConfigError, format_expected_got

_THRESHOLD_MAX: Final = 100.0
_BOOL_TRUE = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off"})

# Re-export for callers that historically imported from ``_config``.
__all__ = ("Config", "ConfigError", "format_expected_got")

_FIELD_ENV_KEYS: Final = {
    "test_map_path": "MSMODELING_TEST_MAP_PATH",
    "base_branch": "MSMODELING_TEST_BASE_BRANCH",
    "line_threshold": "MSMODELING_TEST_LINE_THRESHOLD",
    "branch_threshold": "MSMODELING_TEST_BRANCH_THRESHOLD",
    "benchmark_parallel": "MSMODELING_BENCHMARK_PARALLEL",
    "feishu_webhook_url": "FEISHU_WEBHOOK_URL",
    "msmodeling_cache": "MSMODELING_CACHE",
    "weights_prune": "MSMODELING_TEST_WEIGHTS_PRUNE",
    "gitcode_owner": "GITCODE_OWNER",
    "gitcode_repo": "GITCODE_REPO",
    "gitcode_pr_number": "GITCODE_PR_NUMBER",
    "gitcode_pat": "GITCODE_PAT",
}


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        field_name = str(loc[-1]) if loc else "config"
        env_key = _FIELD_ENV_KEYS.get(field_name, field_name)
        msg = err.get("msg", "invalid value")
        if isinstance(msg, str) and msg.startswith("Value error, "):
            msg = msg.removeprefix("Value error, ")
        parts.append(f"{env_key}: {msg}")
    return "\n".join(parts)


def _parse_bool_env(value: object, *, default: bool, field: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError(format_expected_got(field, "a boolean", value))
    raw = value.strip().lower()
    if not raw:
        raise ValueError(format_expected_got(field, "a boolean", value))
    if raw in _BOOL_TRUE:
        return True
    if raw in _BOOL_FALSE:
        return False
    raise ValueError(format_expected_got(field, "a boolean", value))


def _parse_float_env(value: object, *, default: float, field: str) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(format_expected_got(field, "a number", value))
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(format_expected_got(field, "a number", raw)) from exc
    raise ValueError(format_expected_got(field, "a number", value))


class Config(BaseSettings):
    """Application config read once at CLI startup and passed through helpers."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True, frozen=True)

    test_map_path: str | None = Field(default=None, validation_alias="MSMODELING_TEST_MAP_PATH")
    base_branch: str = Field(default="master", validation_alias="MSMODELING_TEST_BASE_BRANCH")
    line_threshold: float = Field(default=60.0, validation_alias="MSMODELING_TEST_LINE_THRESHOLD")
    branch_threshold: float = Field(default=40.0, validation_alias="MSMODELING_TEST_BRANCH_THRESHOLD")
    benchmark_parallel: bool = Field(default=False, validation_alias="MSMODELING_BENCHMARK_PARALLEL")
    feishu_webhook_url: str = Field(default="", validation_alias="FEISHU_WEBHOOK_URL")
    msmodeling_cache: str = Field(default=".msmodeling_cache", validation_alias="MSMODELING_CACHE")
    weights_prune: bool = Field(default=False, validation_alias="MSMODELING_TEST_WEIGHTS_PRUNE")
    gitcode_owner: str = Field(default="", validation_alias="GITCODE_OWNER")
    gitcode_repo: str = Field(default="", validation_alias="GITCODE_REPO")
    gitcode_pr_number: int | None = Field(default=None, validation_alias="GITCODE_PR_NUMBER")
    gitcode_pat: str = Field(default="", validation_alias="GITCODE_PAT")

    @field_validator("base_branch", "msmodeling_cache", mode="before")
    @classmethod
    def _strip_path_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("gitcode_owner", "gitcode_repo", "gitcode_pat", mode="before")
    @classmethod
    def _strip_gitcode_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("gitcode_pr_number", mode="before")
    @classmethod
    def _parse_gitcode_pr_number(cls, value: object) -> object:
        if value is None or value == "":
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            return int(raw)
        raise ValueError("must be an integer")

    @field_validator("test_map_path", mode="before")
    @classmethod
    def _empty_test_map_path_is_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("feishu_webhook_url", mode="before")
    @classmethod
    def _strip_feishu_webhook(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("line_threshold", mode="before")
    @classmethod
    def _parse_line_threshold(cls, value: object, info: ValidationInfo) -> float:
        field_name = info.field_name or "line_threshold"
        return _parse_float_env(value, default=60.0, field=_FIELD_ENV_KEYS[field_name])

    @field_validator("branch_threshold", mode="before")
    @classmethod
    def _parse_branch_threshold(cls, value: object, info: ValidationInfo) -> float:
        field_name = info.field_name or "branch_threshold"
        return _parse_float_env(value, default=40.0, field=_FIELD_ENV_KEYS[field_name])

    @field_validator("benchmark_parallel", mode="before")
    @classmethod
    def _parse_benchmark_parallel(cls, value: object, info: ValidationInfo) -> bool:
        field_name = info.field_name or "benchmark_parallel"
        return _parse_bool_env(value, default=False, field=_FIELD_ENV_KEYS[field_name])

    @field_validator("weights_prune", mode="before")
    @classmethod
    def _parse_weights_prune(cls, value: object, info: ValidationInfo) -> bool:
        field_name = info.field_name or "weights_prune"
        return _parse_bool_env(value, default=False, field=_FIELD_ENV_KEYS[field_name])

    @field_validator("line_threshold", "branch_threshold")
    @classmethod
    def _validate_threshold(cls, value: float, info: ValidationInfo) -> float:
        if not (0 <= value <= _THRESHOLD_MAX):
            raise ValueError(f"must be in [0, {_THRESHOLD_MAX:g}], got {value}")
        return value

    @classmethod
    def from_env(cls) -> Config:
        try:
            return cls()
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(exc)) from exc

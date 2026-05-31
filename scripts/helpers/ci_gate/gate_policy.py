"""Load and validate CI gate policy from tests/.ci/*.yaml."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal

import yaml
from scripts.helpers._config import ConfigError
from scripts.helpers.common.coverage_config import PRODUCT_SOURCE_PREFIXES

try:
    import pathspec
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError as exc:
    raise ConfigError("ci dependency group required (pydantic, pathspec). Run: uv sync --group ci") from exc

_GIT = shutil.which("git")
CI_POLICY_REL: Final = Path("tests/.ci")
GATE_POLICY_REL: Final = CI_POLICY_REL / "gate_policy.yaml"
APPROVERS_REL: Final = CI_POLICY_REL / "approvers.yaml"

_DEFAULT_INCLUDE: Final = ("**/test_*.py", "**/*_test.py")
_DEFAULT_EXCLUDE: Final = ("tests/helpers/**", "tests/assets/**")


# ---------------------------------------------------------------------------
# Runtime dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TestDiscovery:
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceExemption:
    file: str
    symbol: str
    reason: str
    applicant: str
    approver: str
    deadline: date
    ticket: str | None = None

    @property
    def symbol_key(self) -> str:
        return f"{self.file}::{self.symbol}"


@dataclass(frozen=True, slots=True)
class ExpiredExemptionReport:
    symbol_key: str
    deadline: date
    reason: str
    applicant: str
    approver: str
    ticket: str | None


@dataclass(frozen=True, slots=True)
class GatePolicy:
    discovery: TestDiscovery
    exemptions: tuple[SourceExemption, ...]
    approvers: frozenset[str]


# ---------------------------------------------------------------------------
# Pydantic boundary models
# ---------------------------------------------------------------------------


class TestDiscoveryDoc(BaseModel):
    include: list[str] = Field(default_factory=lambda: list(_DEFAULT_INCLUDE))
    exclude: list[str] = Field(default_factory=lambda: list(_DEFAULT_EXCLUDE))

    @field_validator("include", "exclude")
    @classmethod
    def non_empty_patterns(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        for pattern in value:
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(f"invalid pattern: {pattern!r}")
        return value


class ExemptionDoc(BaseModel):
    symbols: list[str]
    reason: str
    applicant: str
    approver: str
    deadline: date
    ticket: str | None = None

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        for raw in value:
            _parse_symbol_key(raw)
        return value


class GatePolicyDoc(BaseModel):
    schema_version: Literal[1]
    test_discovery: TestDiscoveryDoc = Field(default_factory=TestDiscoveryDoc)
    exemptions: list[ExemptionDoc] = Field(default_factory=list)


class ApproversDoc(BaseModel):
    schema_version: Literal[1]
    approvers: list[str]

    @field_validator("approvers")
    @classmethod
    def validate_approvers(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        seen: set[str] = set()
        for name in value:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"invalid approver name: {name!r}")
            if name in seen:
                raise ValueError(f"duplicate approver name: {name!r}")
            seen.add(name)
        return value


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def default_test_discovery() -> TestDiscovery:
    return TestDiscovery(include_patterns=_DEFAULT_INCLUDE, exclude_patterns=_DEFAULT_EXCLUDE)


def _policy_paths(repo_root: Path) -> tuple[Path, Path]:
    return repo_root / GATE_POLICY_REL, repo_root / APPROVERS_REL


def _parse_symbol_key(raw: str) -> tuple[str, str]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"expected 'path::symbol', got {raw!r}")
    if raw.count("::") != 1:
        raise ValueError(f"expected 'path::symbol', got {raw!r}")
    file_path, symbol = raw.split("::", 1)
    if not file_path or not symbol:
        raise ValueError(f"expected 'path::symbol', got {raw!r}")
    if not any(file_path.startswith(prefix) for prefix in PRODUCT_SOURCE_PREFIXES):
        prefixes = ", ".join(PRODUCT_SOURCE_PREFIXES)
        raise ValueError(f"path {file_path!r} must start with a product prefix ({prefixes})")
    return file_path, symbol


def _load_yaml(path: Path, label: str) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{label}: invalid YAML: {exc}") from exc


def _format_pydantic_error(path: Path, exc: ValidationError) -> str:
    rel = path.as_posix()
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ()))
        msg = err.get("msg", "invalid value")
        parts.append(f"{rel}: {loc}: {msg}" if loc else f"{rel}: {msg}")
    return "\n".join(parts)


def _doc_to_policy(doc: GatePolicyDoc, approvers: frozenset[str]) -> GatePolicy:
    discovery = TestDiscovery(
        include_patterns=tuple(doc.test_discovery.include),
        exclude_patterns=tuple(doc.test_discovery.exclude),
    )
    exemptions: list[SourceExemption] = []
    for entry in doc.exemptions:
        for raw in entry.symbols:
            file_path, symbol = _parse_symbol_key(raw)
            exemptions.append(
                SourceExemption(
                    file=file_path,
                    symbol=symbol,
                    reason=entry.reason,
                    applicant=entry.applicant,
                    approver=entry.approver,
                    deadline=entry.deadline,
                    ticket=entry.ticket,
                )
            )
    return GatePolicy(
        discovery=discovery,
        exemptions=tuple(exemptions),
        approvers=approvers,
    )


def _load_approvers_doc(approvers_path: Path) -> ApproversDoc:
    if not approvers_path.is_file():
        raise ConfigError(f"{APPROVERS_REL.as_posix()}: file not found")
    raw = _load_yaml(approvers_path, APPROVERS_REL.as_posix())
    try:
        return ApproversDoc.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_pydantic_error(approvers_path, exc)) from exc


def _load_gate_policy_doc(policy_path: Path) -> GatePolicyDoc:
    if not policy_path.is_file():
        raise ConfigError(f"{GATE_POLICY_REL.as_posix()}: file not found")
    raw = _load_yaml(policy_path, GATE_POLICY_REL.as_posix())
    try:
        return GatePolicyDoc.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_pydantic_error(policy_path, exc)) from exc


def _validate_approvers_in_registry(doc: GatePolicyDoc, approvers: frozenset[str]) -> None:
    errors: list[str] = []
    for index, entry in enumerate(doc.exemptions):
        if entry.approver not in approvers:
            errors.append(
                f"{GATE_POLICY_REL.as_posix()}: exemptions[{index}].approver "
                f"{entry.approver!r} not in approver registry ({APPROVERS_REL.as_posix()})"
            )
    if errors:
        raise ConfigError("\n".join(errors))


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_gate_policy(repo_root: Path) -> GatePolicy:
    """Load gate policy and approver registry from tests/.ci/."""
    policy_path, approvers_path = _policy_paths(repo_root)
    approvers_doc = _load_approvers_doc(approvers_path)
    policy_doc = _load_gate_policy_doc(policy_path)
    approver_names = frozenset(approvers_doc.approvers)
    return _doc_to_policy(policy_doc, approver_names)


def gate_policy_changed_in_diff(repo_root: Path, base_ref: str) -> bool:
    """Return True when gate_policy.yaml changed between base_ref and HEAD."""
    if _GIT is None:
        raise ConfigError("git not found")
    proc = subprocess.run(
        [_GIT, "diff", f"{base_ref}...HEAD", "--name-only", "--", GATE_POLICY_REL.as_posix()],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    if proc.returncode != 0:
        raise ConfigError(f"git diff failed: {proc.stderr.strip()}")
    target = GATE_POLICY_REL.as_posix()
    return any(line.strip() == target for line in proc.stdout.splitlines())


def validate_gate_policy_if_changed(repo_root: Path, base_ref: str) -> None:
    """Strict validation when gate_policy.yaml is in the PR diff."""
    if not gate_policy_changed_in_diff(repo_root, base_ref):
        return
    policy_path, approvers_path = _policy_paths(repo_root)
    approvers_doc = _load_approvers_doc(approvers_path)
    policy_doc = _load_gate_policy_doc(policy_path)
    _validate_approvers_in_registry(policy_doc, frozenset(approvers_doc.approvers))


# ---------------------------------------------------------------------------
# Test path matching
# ---------------------------------------------------------------------------


def is_gate_test_path(path: str, discovery: TestDiscovery) -> bool:
    """Return True when *path* is a collectible test module for the gate."""
    if not path.startswith("tests/") or not path.endswith(".py"):
        return False
    exclude_spec = pathspec.PathSpec.from_lines("gitignore", discovery.exclude_patterns)
    if exclude_spec.match_file(path):
        return False
    include_spec = pathspec.PathSpec.from_lines("gitignore", discovery.include_patterns)
    return include_spec.match_file(path)


def is_exempt(exemptions: tuple[SourceExemption, ...], file_path: str, symbol: str) -> bool:
    """Return True when (file_path, symbol) has a registered exemption."""
    return any(item.file == file_path and item.symbol == symbol for item in exemptions)


# ---------------------------------------------------------------------------
# Nightly audit
# ---------------------------------------------------------------------------


def find_expired_unmapped(
    policy: GatePolicy,
    test_map: dict[str, dict[str, list[str]]],
    *,
    today: date | None = None,
) -> tuple[ExpiredExemptionReport, ...]:
    """Return exemptions past deadline that still lack test_map coverage."""
    check_date = today or date.today()
    reports: list[ExpiredExemptionReport] = []
    for entry in policy.exemptions:
        if entry.deadline >= check_date:
            continue
        file_map = test_map.get(entry.file, {})
        if entry.symbol in file_map:
            continue
        reports.append(
            ExpiredExemptionReport(
                symbol_key=entry.symbol_key,
                deadline=entry.deadline,
                reason=entry.reason,
                applicant=entry.applicant,
                approver=entry.approver,
                ticket=entry.ticket,
            )
        )
    return tuple(reports)


def format_expired_exemptions_section(reports: tuple[ExpiredExemptionReport, ...]) -> str:
    """Build Feishu body text for expired unmapped exemptions."""
    if not reports:
        return ""
    lines = [
        f"\nExpired exemptions ({len(reports)} past deadline, still unmapped in test_map):",
    ]
    for report in reports[:10]:
        ticket = f", ticket {report.ticket}" if report.ticket else ""
        lines.append(
            f"- {report.symbol_key} (deadline {report.deadline.isoformat()}, approver {report.approver}{ticket})"
        )
    if len(reports) > 10:
        lines.append(f"- ... and {len(reports) - 10} more")
    lines.append("→ Add tests or renew the exemption in tests/.ci/gate_policy.yaml")
    return "\n".join(lines)

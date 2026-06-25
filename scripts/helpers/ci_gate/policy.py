"""Load and validate CI gate policy from tests/.ci/gate_policy.yaml."""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Final

import yaml

from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.models import (
    CiGatePolicy,
    ExpiredExemptionReport,
    PathPatterns,
    SourceExemption,
    TestDiscovery,
    TestExemption,
)
from scripts.helpers.common.ast_utils import MODULE_SYMBOL, collect_file_symbols
from scripts.helpers.common.test_map_report import find_expired_unmapped_in_map

try:
    import pathspec
    from pydantic import (
        BaseModel,
        Field,
        ValidationError,
        field_validator,
        model_validator,
    )
except ImportError as exc:
    raise ConfigError("ci dependency group required (pydantic, pathspec). Run: uv sync --frozen --group ci") from exc

_GIT = shutil.which("git")
CI_POLICY_REL: Final = Path("tests/.ci")
GATE_POLICY_REL: Final = CI_POLICY_REL / "gate_policy.yaml"
APPROVERS_REL: Final = CI_POLICY_REL / "approvers.yaml"

_CLASS_ONLY_TEST_NODE: Final = re.compile(r"^Test[A-Za-z0-9_]+$")


class ScopeDoc(BaseModel):
    include: list[str]
    exclude: list[str] = Field(default_factory=list)

    @field_validator("include")
    @classmethod
    def include_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("include must not be empty")
        return _validate_pattern_list(value)

    @field_validator("exclude")
    @classmethod
    def exclude_patterns(cls, value: list[str]) -> list[str]:
        return _validate_pattern_list(value, allow_empty=True)


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
        return value


class TestExemptionDoc(ExemptionDoc):
    pass


class ExemptionsDoc(BaseModel):
    sources: list[ExemptionDoc] = Field(default_factory=list)
    tests: list[TestExemptionDoc] = Field(default_factory=list)


class GatePolicyDoc(BaseModel):
    schema_version: int | None = None
    roots: list[str]
    tests: ScopeDoc
    configs: ScopeDoc
    exemptions: ExemptionsDoc = Field(default_factory=ExemptionsDoc)

    @field_validator("roots")
    @classmethod
    def validate_roots(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        for root in value:
            if not root.endswith("/"):
                raise ValueError(f"root must end with '/': {root!r}")
        return value

    @model_validator(mode="after")
    def validate_exemption_symbols(self) -> GatePolicyDoc:
        roots = tuple(self.roots)
        tests = _scope_to_patterns(self.tests)
        for entry in self.exemptions.sources:
            for raw in entry.symbols:
                _parse_symbol_key(raw, roots)
        for entry in self.exemptions.tests:
            for raw in entry.symbols:
                _parse_test_exemption_id(raw)
                file_part = raw.split("::", 1)[0]
                if not is_gate_test_path(file_part, _patterns_to_discovery(tests)):
                    raise ValueError(f"test exemption file {file_part!r} is not a collectible test module")
        return self


class ApproversDoc(BaseModel):
    schema_version: int | None = None
    approvers: list[str]

    @field_validator("approvers")
    @classmethod
    def validate_approvers(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        seen: set[str] = set()
        for name in value:
            if name in seen:
                raise ValueError(f"duplicate approver name: {name!r}")
            seen.add(name)
        return value


def _validate_pattern_list(value: list[str], *, allow_empty: bool = False) -> list[str]:
    if not value and not allow_empty:
        raise ValueError("must not be empty")
    for pattern in value:
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(f"invalid pattern: {pattern!r}")
    return value


def _scope_to_patterns(scope: ScopeDoc) -> PathPatterns:
    return PathPatterns(
        include_patterns=tuple(scope.include),
        exclude_patterns=tuple(scope.exclude),
    )


def _patterns_to_discovery(patterns: PathPatterns) -> TestDiscovery:
    return TestDiscovery(
        include_patterns=patterns.include_patterns,
        exclude_patterns=patterns.exclude_patterns,
    )


def _policy_paths(repo_root: Path) -> tuple[Path, Path]:
    return repo_root / GATE_POLICY_REL, repo_root / APPROVERS_REL


def _parse_symbol_key(raw: str, roots: tuple[str, ...]) -> tuple[str, str]:
    if "::" not in raw:
        raise ValueError(f"expected 'path::symbol', got {raw!r}")
    file_path, symbol = raw.split("::", 1)
    if not file_path or not symbol:
        raise ValueError(f"expected 'path::symbol', got {raw!r}")
    if not any(file_path.startswith(prefix) for prefix in roots):
        prefixes = ", ".join(roots)
        raise ValueError(f"path {file_path!r} must start with a product root ({prefixes})")
    # Lazy import: policy → coverage_omit → test_map_loader → policy.
    from scripts.helpers.common.coverage_omit import is_coverage_omitted_source

    if is_coverage_omitted_source(file_path, roots):
        raise ValueError(f"source path {file_path!r} is coverage-omitted")
    return file_path, symbol


def known_symbols_for_file(repo_root: Path, file_path: str) -> frozenset[str]:
    """Return collectible canonical symbols for a product source file."""
    file_symbols = collect_file_symbols(repo_root / file_path)
    symbols: set[str] = {MODULE_SYMBOL}
    symbols.update(span.qualified_name for span in file_symbols.definitions)
    class_suffix = f"::{MODULE_SYMBOL}"
    symbols.update(f"{cls.qualified_name}{class_suffix}" for cls in file_symbols.class_spans)
    return frozenset(symbols)


def validate_source_exemption_symbol(repo_root: Path, file_path: str, symbol: str) -> str | None:
    """Return an error message when the exemption target is missing or unknown."""
    abs_path = repo_root / file_path
    if not abs_path.is_file():
        return f"source file not found: {file_path!r}"
    if symbol not in known_symbols_for_file(repo_root, file_path):
        return f"unknown symbol {symbol!r}"
    return None


def _validate_test_exemption_collectible(
    repo_root: Path,
    raw: str,
    discovery: TestDiscovery,
    *,
    collected_nodes: frozenset[str],
) -> str | None:
    file_part = raw.split("::", 1)[0]
    if not is_gate_test_path(file_part, discovery):
        return f"test exemption file {file_part!r} is not a collectible test module"
    test_path = repo_root / file_part
    if not test_path.is_file():
        return None
    if raw not in collected_nodes and not any(node_id.startswith(f"{raw}[") for node_id in collected_nodes):
        return f"test exemption {raw!r} is not a collectible test node"
    return None


def _validate_loaded_exemptions(repo_root: Path, policy: CiGatePolicy, doc: GatePolicyDoc) -> None:
    errors: list[str] = []
    policy_label = GATE_POLICY_REL.as_posix()
    for entry in policy.source_exemptions:
        msg = validate_source_exemption_symbol(repo_root, entry.file, entry.symbol)
        if msg is not None:
            errors.append(f"{policy_label}: {entry.file}::{entry.symbol}: {msg}")
    discovery = _patterns_to_discovery(_scope_to_patterns(doc.tests))
    exemption_files: list[str] = []
    seen_files: set[str] = set()
    for entry in doc.exemptions.tests:
        for raw in entry.symbols:
            file_part = raw.split("::", 1)[0]
            if file_part in seen_files:
                continue
            if not is_gate_test_path(file_part, discovery):
                continue
            if not (repo_root / file_part).is_file():
                continue
            seen_files.add(file_part)
            exemption_files.append(file_part)
    collected_nodes: frozenset[str] = frozenset()
    if exemption_files:
        from scripts.helpers.common.pytest_runner import collect_all_test_node_ids

        collected_nodes = frozenset(collect_all_test_node_ids(exemption_files))
    for entry in doc.exemptions.tests:
        for raw in entry.symbols:
            msg = _validate_test_exemption_collectible(
                repo_root,
                raw,
                discovery,
                collected_nodes=collected_nodes,
            )
            if msg is not None:
                errors.append(f"{policy_label}: {msg}")
    if errors:
        raise ConfigError("\n".join(errors))


def _parse_test_exemption_id(raw: str) -> str:
    if "[" in raw:
        raise ValueError(f"test exemption id must not contain '[': {raw!r}")
    if not raw.startswith("tests/") or "::" not in raw:
        raise ValueError(f"invalid test exemption id: {raw!r}")
    file_part, node_part = raw.split("::", 1)
    if "::" in file_part:
        raise ValueError(f"invalid test exemption id: {raw!r}")
    if node_part.count("::") > 1:
        raise ValueError(f"invalid test exemption id: {raw!r}")
    if not file_part.endswith(".py") or not node_part:
        raise ValueError(f"invalid test exemption id: {raw!r}")
    if _CLASS_ONLY_TEST_NODE.match(node_part):
        raise ValueError("test exemption id must target a test function or method, not a class")
    return raw


def _load_yaml(path: Path, label: str) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{label}: invalid YAML: {exc}") from exc


def _format_pydantic_error(path: Path, exc: ValidationError) -> str:
    rel = path.as_posix()
    parts = [
        f"{rel}: {'.'.join(str(i) for i in err.get('loc', ()))}: {err.get('msg', 'invalid value')}"
        for err in exc.errors()
    ]
    return "\n".join(parts)


def _expand_source_exemptions(
    entries: list[ExemptionDoc],
    roots: tuple[str, ...],
) -> tuple[SourceExemption, ...]:
    return tuple(
        SourceExemption(
            file=file_path,
            symbol=symbol,
            reason=entry.reason,
            applicant=entry.applicant,
            approver=entry.approver,
            deadline=entry.deadline,
            ticket=entry.ticket,
        )
        for entry in entries
        for raw in entry.symbols
        for file_path, symbol in (_parse_symbol_key(raw, roots),)
    )


def _expand_test_exemptions(
    entries: list[TestExemptionDoc],
) -> tuple[TestExemption, ...]:
    return tuple(
        TestExemption(
            test_id=raw,
            reason=entry.reason,
            applicant=entry.applicant,
            approver=entry.approver,
            deadline=entry.deadline,
            ticket=entry.ticket,
        )
        for entry in entries
        for raw in entry.symbols
    )


def _doc_to_policy(doc: GatePolicyDoc, approvers: frozenset[str]) -> CiGatePolicy:
    roots = tuple(doc.roots)
    return CiGatePolicy(
        sources=PathPatterns(include_patterns=roots, exclude_patterns=()),
        tests=_scope_to_patterns(doc.tests),
        configs=_scope_to_patterns(doc.configs),
        source_exemptions=_expand_source_exemptions(doc.exemptions.sources, roots),
        test_exemptions=_expand_test_exemptions(doc.exemptions.tests),
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


def _load_policy_doc(policy_path: Path) -> GatePolicyDoc:
    if not policy_path.is_file():
        raise ConfigError(f"{GATE_POLICY_REL.as_posix()}: file not found")
    raw = _load_yaml(policy_path, GATE_POLICY_REL.as_posix())
    try:
        return GatePolicyDoc.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_pydantic_error(policy_path, exc)) from exc


def _validate_approvers(doc: GatePolicyDoc, approvers: frozenset[str], policy_label: str) -> None:
    errors: list[str] = []
    for section, entries in (
        ("sources", doc.exemptions.sources),
        ("tests", doc.exemptions.tests),
    ):
        for index, entry in enumerate(entries):
            if entry.approver not in approvers:
                errors.append(
                    f"{policy_label}: exemptions.{section}[{index}].approver "
                    f"{entry.approver!r} not in approver registry ({APPROVERS_REL.as_posix()})"
                )
    if errors:
        raise ConfigError("\n".join(errors))


def _policy_mtime_key(repo_root: Path) -> tuple[float, float]:
    policy_path, approvers_path = _policy_paths(repo_root)
    policy_mtime = policy_path.stat().st_mtime if policy_path.is_file() else 0.0
    approvers_mtime = approvers_path.stat().st_mtime if approvers_path.is_file() else 0.0
    return (policy_mtime, approvers_mtime)


def _load_gate_policy_uncached(repo_root: Path) -> CiGatePolicy:
    policy_path, approvers_path = _policy_paths(repo_root)
    approvers_doc = _load_approvers_doc(approvers_path)
    policy_doc = _load_policy_doc(policy_path)
    policy = _doc_to_policy(policy_doc, frozenset(approvers_doc.approvers))
    _validate_loaded_exemptions(repo_root, policy, policy_doc)
    return policy


@functools.lru_cache(maxsize=32)
def _load_gate_policy_cached(repo_root: Path, _mtime_key: tuple[float, float]) -> CiGatePolicy:
    return _load_gate_policy_uncached(repo_root)


def load_gate_policy(repo_root: Path) -> CiGatePolicy:
    """Load gate policy and approver registry from tests/.ci/."""
    return _load_gate_policy_cached(repo_root, _policy_mtime_key(repo_root))


def gate_policy_changed_in_diff(repo_root: Path, base_ref: str) -> bool:
    """Return True when gate_policy.yaml changed."""
    if _GIT is None:
        raise ConfigError("git not found")
    policy_path = GATE_POLICY_REL.as_posix()
    proc = subprocess.run(
        [_GIT, "diff", f"{base_ref}...HEAD", "--name-only", "--", policy_path],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    if proc.returncode != 0:
        raise ConfigError(f"git diff failed: {proc.stderr.strip()}")
    changed = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return policy_path in changed


def validate_gate_policy_if_changed(repo_root: Path, base_ref: str) -> None:
    """Strict validation when CI gate policy files are in the PR diff."""
    if not gate_policy_changed_in_diff(repo_root, base_ref):
        return
    policy_path, approvers_path = _policy_paths(repo_root)
    approvers_doc = _load_approvers_doc(approvers_path)
    policy_doc = _load_policy_doc(policy_path)
    _validate_approvers(policy_doc, frozenset(approvers_doc.approvers), policy_path.name)


@functools.lru_cache(maxsize=32)
def _path_specs(
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
) -> tuple[Any, Any]:
    return (
        pathspec.PathSpec.from_lines("gitignore", exclude_patterns),
        pathspec.PathSpec.from_lines("gitignore", include_patterns),
    )


def matches_path_patterns(path: str, patterns: PathPatterns) -> bool:
    """Return True when *path* matches include patterns and not exclude."""
    exclude_spec, include_spec = _path_specs(patterns.include_patterns, patterns.exclude_patterns)
    if bool(exclude_spec.match_file(path)):
        return False
    return bool(include_spec.match_file(path))


def is_gate_test_path(path: str, discovery: TestDiscovery) -> bool:
    """Return True when *path* is a collectible test module for the gate."""
    if not path.startswith("tests/") or not path.endswith(".py"):
        return False
    patterns = PathPatterns(
        include_patterns=discovery.include_patterns,
        exclude_patterns=discovery.exclude_patterns,
    )
    return matches_path_patterns(path, patterns)


def is_test_path(path: str, policy: CiGatePolicy) -> bool:
    return matches_path_patterns(path, policy.tests)


def is_policy_config_path(path: str, configs: PathPatterns) -> bool:
    """Return True when *path* is a CI config file per policy patterns."""
    return matches_path_patterns(path, configs)


def is_config_path(path: str, policy: CiGatePolicy) -> bool:
    return matches_path_patterns(path, policy.configs)


def is_source_path(path: str, policy: CiGatePolicy) -> bool:
    """Return True when *path* is a gated product source under policy sources."""
    sources = policy.sources
    if not path.endswith(".py"):
        return False
    if not any(path.startswith(prefix) for prefix in sources.include_patterns):
        return False
    if sources.exclude_patterns:
        exclude_spec, _include_spec = _path_specs(sources.include_patterns, sources.exclude_patterns)
        if exclude_spec.match_file(path):
            return False
    return True


def is_exempt(exemptions: tuple[SourceExemption, ...], file_path: str, symbol: str) -> bool:
    return any(item.file == file_path and item.symbol == symbol for item in exemptions)


def is_test_exempt(test_exemptions: tuple[TestExemption, ...], test_node_id: str) -> bool:
    if not test_exemptions:
        return False
    for entry in test_exemptions:
        exempt_id = entry.test_id
        if test_node_id == exempt_id or test_node_id.startswith(f"{exempt_id}["):
            return True
    return False


def find_expired_unmapped(
    policy: CiGatePolicy,
    test_map: dict[str, dict[str, list[str]]],
    *,
    today: date | None = None,
) -> tuple[ExpiredExemptionReport, ...]:
    return find_expired_unmapped_in_map(policy, test_map, today=today)


def find_expired_test_exemptions(
    policy: CiGatePolicy,
    *,
    today: date | None = None,
) -> tuple[ExpiredExemptionReport, ...]:
    check_date = today or date.today()
    return tuple(
        ExpiredExemptionReport(
            symbol_key=entry.test_id,
            deadline=entry.deadline,
            reason=entry.reason,
            applicant=entry.applicant,
            approver=entry.approver,
            ticket=entry.ticket,
        )
        for entry in policy.test_exemptions
        if entry.deadline < check_date
    )


def format_expired_exemptions_section(
    reports: tuple[ExpiredExemptionReport, ...],
) -> str:
    if not reports:
        return ""
    lines = [f"\nExpired exemptions ({len(reports)} past deadline, still unmapped in test_map):"]
    for report in reports[:10]:
        ticket = f", ticket {report.ticket}" if report.ticket else ""
        lines.append(
            f"- {report.symbol_key} (deadline {report.deadline.isoformat()}, approver {report.approver}{ticket})"
        )
    if len(reports) > 10:
        lines.append(f"- ... and {len(reports) - 10} more")
    lines.append("→ Add tests or renew the exemption in tests/.ci/gate_policy.yaml")
    return "\n".join(lines)


def format_expired_test_exemptions_section(
    reports: tuple[ExpiredExemptionReport, ...],
) -> str:
    if not reports:
        return ""
    lines = [f"\nExpired test exemptions ({len(reports)} past deadline):"]
    for report in reports[:10]:
        ticket = f", ticket {report.ticket}" if report.ticket else ""
        lines.append(
            f"- {report.symbol_key} (deadline {report.deadline.isoformat()}, approver {report.approver}{ticket})"
        )
    if len(reports) > 10:
        lines.append(f"- ... and {len(reports) - 10} more")
    lines.append("→ Remove the exemption or renew it in tests/.ci/gate_policy.yaml")
    return "\n".join(lines)

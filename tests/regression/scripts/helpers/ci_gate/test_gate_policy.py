"""Tests for ci_gate.gate_policy."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.gate_policy import (
    SourceExemption,
    default_test_discovery,
    find_expired_unmapped,
    format_expired_exemptions_section,
    gate_policy_changed_in_diff,
    is_exempt,
    is_gate_test_path,
    load_gate_policy,
    validate_gate_policy_if_changed,
)
from tests.helpers.fake_subprocess import FakeCompleted


def _write_ci_policy(repo: Path, *, exemptions: list | None = None) -> None:
    ci_dir = repo / "tests" / ".ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    policy = {
        "schema_version": 1,
        "test_discovery": {
            "include": ["**/test_*.py", "**/*_test.py"],
            "exclude": ["tests/helpers/**", "tests/assets/**"],
        },
        "exemptions": exemptions or [],
    }
    (ci_dir / "gate_policy.yaml").write_text(yaml.dump(policy), encoding="utf-8")
    (ci_dir / "approvers.yaml").write_text(
        yaml.dump({"schema_version": 1, "approvers": ["fangkai", "hexiaowu", "gongjiong", "liujiawang"]}),
        encoding="utf-8",
    )


def _sample_exemption(file: str, symbol: str) -> SourceExemption:
    return SourceExemption(
        file=file,
        symbol=symbol,
        reason="test",
        applicant="test",
        approver="fangkai",
        deadline=date(2099, 12, 31),
    )


def test_load_gate_policy_expands_symbols(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["tensor_cast/foo.py::fn", "cli/main.py::run"],
                "reason": "refactor",
                "applicant": "alice",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    policy = load_gate_policy(repo)
    assert len(policy.exemptions) == 2
    assert policy.exemptions[0].file == "tensor_cast/foo.py"
    assert policy.exemptions[0].symbol == "fn"


def test_load_gate_policy_invalid_symbol_raises_config_error(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["bad-format"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    with pytest.raises(ConfigError, match="expected 'path::symbol'"):
        load_gate_policy(repo)


def test_load_gate_policy_pydantic_validation_error_includes_field_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    ci_dir = repo / "tests" / ".ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    (ci_dir / "gate_policy.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "test_discovery": {"include": "not-a-list", "exclude": []},
                "exemptions": [],
            }
        ),
        encoding="utf-8",
    )
    (ci_dir / "approvers.yaml").write_text(
        yaml.dump({"schema_version": 1, "approvers": ["fangkai"]}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"tests/\.ci/gate_policy\.yaml.*test_discovery"):
        load_gate_policy(repo)


def test_load_gate_policy_unknown_approver_allowed_without_strict_validate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["cli/main.py::run"],
                "reason": "x",
                "applicant": "a",
                "approver": "unknown_person",
                "deadline": "2026-06-30",
            }
        ],
    )
    policy = load_gate_policy(repo)
    assert policy.exemptions[0].approver == "unknown_person"


def test_validate_gate_policy_if_changed_checks_approver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["cli/main.py::run"],
                "reason": "x",
                "applicant": "a",
                "approver": "unknown_person",
                "deadline": "2026-06-30",
            }
        ],
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.gate_policy.gate_policy_changed_in_diff",
        lambda *_args, **_kwargs: True,
    )
    with pytest.raises(ConfigError, match="not in approver registry"):
        validate_gate_policy_if_changed(repo, "abc123")


def test_is_gate_test_path_excludes_helpers_and_assets() -> None:
    discovery = default_test_discovery()
    assert is_gate_test_path("tests/regression/cli/test_run.py", discovery) is True
    assert is_gate_test_path("tests/helpers/assert_utils.py", discovery) is False
    assert is_gate_test_path("tests/assets/model_config/foo.py", discovery) is False


def test_is_exempt_matches_file_and_symbol() -> None:
    exemptions = (_sample_exemption("cli/main.py", "run"),)
    assert is_exempt(exemptions, "cli/main.py", "run") is True
    assert is_exempt(exemptions, "cli/main.py", "other") is False


def test_find_expired_unmapped_reports_missing_coverage() -> None:
    policy_exemption = _sample_exemption("cli/main.py", "run")
    expired = SourceExemption(
        file=policy_exemption.file,
        symbol=policy_exemption.symbol,
        reason=policy_exemption.reason,
        applicant=policy_exemption.applicant,
        approver=policy_exemption.approver,
        deadline=date(2020, 1, 1),
    )
    from scripts.helpers.ci_gate.gate_policy import GatePolicy

    policy = GatePolicy(
        discovery=default_test_discovery(),
        exemptions=(expired,),
        approvers=frozenset({"fangkai"}),
    )
    reports = find_expired_unmapped(policy, {}, today=date(2026, 1, 1))
    assert len(reports) == 1
    assert reports[0].symbol_key == "cli/main.py::run"


def test_find_expired_unmapped_skips_when_test_map_has_symbol() -> None:
    expired = SourceExemption(
        file="cli/main.py",
        symbol="run",
        reason="x",
        applicant="a",
        approver="fangkai",
        deadline=date(2020, 1, 1),
    )
    from scripts.helpers.ci_gate.gate_policy import GatePolicy

    policy = GatePolicy(
        discovery=default_test_discovery(),
        exemptions=(expired,),
        approvers=frozenset({"fangkai"}),
    )
    test_map = {"cli/main.py": {"run": ["tests/smoke/test_a.py::test_x"]}}
    assert find_expired_unmapped(policy, test_map, today=date(2026, 1, 1)) == ()


def test_format_expired_exemptions_section_empty_returns_empty_string() -> None:
    assert format_expired_exemptions_section(()) == ""


def test_gate_policy_changed_in_diff_true_when_file_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "tests/.ci/gate_policy.yaml\n", ""),
    )
    assert gate_policy_changed_in_diff(tmp_path, "abc123") is True

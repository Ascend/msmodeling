"""Tests for ci_gate.gate_policy re-exports."""

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

import pytest
import yaml

from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.gate_policy import (
    GatePolicy,
    PathPatterns,
    SourceExemption,
    TestExemption,
    _load_gate_policy_cached,
    find_expired_test_exemptions,
    find_expired_unmapped,
    format_expired_exemptions_section,
    format_expired_test_exemptions_section,
    gate_policy_changed_in_diff,
    is_exempt,
    is_gate_test_path,
    is_test_exempt,
    load_gate_policy,
    validate_gate_policy_if_changed,
)
from tests.helpers.fake_subprocess import FakeCompleted
from tests.regression.scripts.helpers.gate_policy_writer import (
    DEFAULT_CONFIG_INCLUDE,
    DEFAULT_GATE_ROOTS,
    DEFAULT_TEST_EXCLUDE,
    DEFAULT_TEST_INCLUDE,
    write_gate_policy,
    write_repo_file,
)


def _write_ci_policy(
    repo: Path,
    *,
    exemptions: list[dict[str, object]] | None = None,
    test_exemptions: list[dict[str, object]] | None = None,
    roots: list[str] | None = None,
    tests: dict[str, list[str]] | None = None,
) -> None:
    write_gate_policy(
        repo,
        roots=roots,
        tests=tests,
        source_exemptions=exemptions,
        test_exemptions=test_exemptions,
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


def _empty_policy(*, source_exemptions: tuple[SourceExemption, ...] = ()) -> GatePolicy:
    return GatePolicy(
        sources=PathPatterns(include_patterns=DEFAULT_GATE_ROOTS, exclude_patterns=()),
        tests=PathPatterns(
            include_patterns=DEFAULT_TEST_INCLUDE,
            exclude_patterns=DEFAULT_TEST_EXCLUDE,
        ),
        configs=PathPatterns(include_patterns=DEFAULT_CONFIG_INCLUDE, exclude_patterns=()),
        source_exemptions=source_exemptions,
        test_exemptions=(),
        approvers=frozenset({"fangkai"}),
    )


def test_load_gate_policy_cached_until_yaml_mtime_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_ci_policy(repo)
    _load_gate_policy_cached.cache_clear()
    first = load_gate_policy(repo)
    second = load_gate_policy(repo)
    assert first is second

    policy_path = repo / "tests" / ".ci" / "gate_policy.yaml"
    policy_path.write_text(policy_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    time.sleep(0.01)
    bumped_mtime = time.time() + 1.0
    os.utime(policy_path, (bumped_mtime, bumped_mtime))
    third = load_gate_policy(repo)
    assert third is not first


def test_load_gate_policy_expands_symbols(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_repo_file(repo, "tensor_cast/foo.py", "def fn():\n    pass\n")
    write_repo_file(repo, "cli/main.py", "def run():\n    pass\n")
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
    assert len(policy.source_exemptions) == 2
    assert policy.source_exemptions[0].file == "tensor_cast/foo.py"
    assert policy.source_exemptions[0].symbol == "fn"
    assert policy.roots == DEFAULT_GATE_ROOTS


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


def test_load_gate_policy_rejects_unknown_symbol(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_repo_file(repo, "cli/main.py", "def run():\n    pass\n")
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["cli/main.py::missing"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    with pytest.raises(ConfigError, match="unknown symbol"):
        load_gate_policy(repo)


def test_load_gate_policy_rejects_coverage_omitted_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_repo_file(
        repo,
        "tensor_cast/transformers/builtin_model/foo.py",
        "def run():\n    pass\n",
    )
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["tensor_cast/transformers/builtin_model/foo.py::run"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    with pytest.raises(ConfigError, match="coverage-omitted"):
        load_gate_policy(repo)


def test_load_gate_policy_accepts_decorator_suffix_symbol(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_repo_file(
        repo,
        "cli/main.py",
        "def _decorator(arg):\n"
        "    def wrapper(fn):\n"
        "        return fn\n"
        "    return wrapper\n\n"
        "@_decorator(torch.ops.foo.bar)\n"
        "def _():\n"
        "    pass\n",
    )
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["cli/main.py::_@_decorator(torch.ops.foo.bar)"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    policy = load_gate_policy(repo)
    assert policy.source_exemptions[0].symbol == "_@_decorator(torch.ops.foo.bar)"


def test_load_gate_policy_rejects_legacy_dot_symbol(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_repo_file(
        repo,
        "cli/main.py",
        "class Widget:\n    def run(self):\n        pass\n",
    )
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["cli/main.py::Widget.run"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    with pytest.raises(ConfigError, match="unknown symbol"):
        load_gate_policy(repo)


def test_load_gate_policy_accepts_symbol_with_internal_colons(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_repo_file(repo, "cli/main.py", "def run():\n    pass\n")
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["cli/main.py::run::extra"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    with pytest.raises(ConfigError, match="unknown symbol"):
        load_gate_policy(repo)


def test_load_gate_policy_accepts_canonical_class_method_symbol(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_repo_file(
        repo,
        "cli/main.py",
        "class Widget:\n    def run(self):\n        pass\n",
    )
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["cli/main.py::Widget::run"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    policy = load_gate_policy(repo)
    assert policy.source_exemptions[0].symbol == "Widget::run"


def test_load_gate_policy_accepts_underscore_class_method_symbol(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    write_repo_file(
        repo,
        "cli/main.py",
        "class _InternalHelper:\n    def run(self):\n        pass\n",
    )
    _write_ci_policy(
        repo,
        exemptions=[
            {
                "symbols": ["cli/main.py::_InternalHelper::run"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    policy = load_gate_policy(repo)
    assert policy.source_exemptions[0].symbol == "_InternalHelper::run"


def test_load_gate_policy_pydantic_validation_error_includes_field_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    ci_dir = repo / "tests" / ".ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    (ci_dir / "gate_policy.yaml").write_text(
        yaml.dump(
            {
                "roots": list(DEFAULT_GATE_ROOTS),
                "tests": {"include": "not-a-list", "exclude": []},
                "configs": {"include": ["pyproject.toml"], "exclude": []},
                "exemptions": {"sources": [], "tests": []},
            }
        ),
        encoding="utf-8",
    )
    (ci_dir / "approvers.yaml").write_text(
        yaml.dump({"approvers": ["fangkai"]}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"tests/\.ci/gate_policy\.yaml.*tests"):
        load_gate_policy(repo)


def test_load_gate_policy_roots_must_end_with_slash(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_ci_policy(repo, roots=["cli"])
    with pytest.raises(ConfigError, match="must end with"):
        load_gate_policy(repo)


def test_load_gate_policy_unknown_approver_allowed_without_strict_validate(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    write_repo_file(repo, "cli/main.py", "def run():\n    pass\n")
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
    assert policy.source_exemptions[0].approver == "unknown_person"


def test_validate_gate_policy_if_changed_checks_approver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    write_repo_file(repo, "cli/main.py", "def run():\n    pass\n")
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
        "scripts.helpers.ci_gate.policy.gate_policy_changed_in_diff",
        lambda *_args, **_kwargs: True,
    )
    with pytest.raises(ConfigError, match="not in approver registry"):
        validate_gate_policy_if_changed(repo, "abc123")


def test_is_gate_test_path_excludes_helpers_and_assets() -> None:
    discovery = GatePolicy(
        sources=PathPatterns(include_patterns=DEFAULT_GATE_ROOTS, exclude_patterns=()),
        tests=PathPatterns(include_patterns=DEFAULT_TEST_INCLUDE, exclude_patterns=DEFAULT_TEST_EXCLUDE),
        configs=PathPatterns(include_patterns=DEFAULT_CONFIG_INCLUDE, exclude_patterns=()),
        source_exemptions=(),
        test_exemptions=(),
        approvers=frozenset(),
    ).discovery
    assert is_gate_test_path("tests/regression/cli/test_run.py", discovery) is True
    assert is_gate_test_path("tests/helpers/assert_utils.py", discovery) is False
    assert is_gate_test_path("tests/assets/model_config/foo.py", discovery) is False


def test_is_exempt_matches_file_and_symbol() -> None:
    exemptions = (_sample_exemption("cli/main.py", "run"),)
    assert is_exempt(exemptions, "cli/main.py", "run") is True
    assert is_exempt(exemptions, "cli/main.py", "other") is False


def test_is_test_exempt_node_level_match() -> None:
    exemptions = (
        TestExemption(
            test_id="tests/regression/nightly/test_x.py::test_case",
            reason="x",
            applicant="a",
            approver="fangkai",
            deadline=date(2099, 12, 31),
        ),
    )
    assert is_test_exempt(exemptions, "tests/regression/nightly/test_x.py::test_case") is True
    assert is_test_exempt(exemptions, "tests/regression/nightly/test_x.py::test_case[param]") is True
    assert is_test_exempt(exemptions, "tests/regression/nightly/test_x.py::test_other") is False
    assert is_test_exempt(exemptions, "tests/regression/cli/test_run.py::test_case") is False
    assert is_test_exempt(exemptions, "tests/regression/nightly/test_x.py") is False


def test_load_gate_policy_rejects_class_only_test_exemption(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_ci_policy(
        repo,
        test_exemptions=[
            {
                "symbols": ["tests/regression/nightly/test_x.py::TestCase"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    with pytest.raises(ConfigError, match="must target a test function or method"):
        load_gate_policy(repo)


def test_load_gate_policy_batches_test_exemption_pytest_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    write_repo_file(repo, "tests/regression/cli/test_a.py", "def test_one():\n    pass\n")
    write_repo_file(repo, "tests/regression/cli/test_b.py", "def test_two():\n    pass\n")
    _write_ci_policy(
        repo,
        test_exemptions=[
            {
                "symbols": [
                    "tests/regression/cli/test_a.py::test_one",
                    "tests/regression/cli/test_b.py::test_two",
                ],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    calls: list[tuple[str, ...]] = []

    def _fake_collect(targets: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        calls.append(tuple(targets))
        return tuple(
            f"{target}::test_one" if target.endswith("test_a.py") else f"{target}::test_two" for target in targets
        )

    monkeypatch.setattr(
        "scripts.helpers.common.pytest_runner.collect_all_test_node_ids",
        _fake_collect,
    )
    _load_gate_policy_cached.cache_clear()
    policy = load_gate_policy(repo)
    assert len(calls) == 1
    assert set(calls[0]) == {
        "tests/regression/cli/test_a.py",
        "tests/regression/cli/test_b.py",
    }
    assert {entry.test_id for entry in policy.test_exemptions} == {
        "tests/regression/cli/test_a.py::test_one",
        "tests/regression/cli/test_b.py::test_two",
    }


def test_load_gate_policy_custom_test_discovery_validates_collectible_paths(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    custom_discovery = {
        "include": ["tests/**/test_*.py", "tests/**/*_test.py"],
        "exclude": [
            "tests/helpers/**",
            "tests/assets/**",
            "tests/regression/nightly/**",
        ],
    }
    _write_ci_policy(
        repo,
        tests=custom_discovery,
        test_exemptions=[
            {
                "symbols": ["tests/regression/nightly/test_x.py::test_case"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    with pytest.raises(ConfigError, match="not a collectible test module"):
        load_gate_policy(repo)

    _write_ci_policy(
        repo,
        tests=custom_discovery,
        test_exemptions=[
            {
                "symbols": ["tests/regression/cli/test_run.py::test_case"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    policy = load_gate_policy(repo)
    assert policy.test_exemptions[0].test_id == "tests/regression/cli/test_run.py::test_case"


def test_load_gate_policy_rejects_file_only_test_exemption(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_ci_policy(
        repo,
        test_exemptions=[
            {
                "symbols": ["tests/regression/nightly/test_x.py"],
                "reason": "x",
                "applicant": "a",
                "approver": "fangkai",
                "deadline": "2026-06-30",
            }
        ],
    )
    with pytest.raises(ConfigError, match=r"invalid test exemption id"):
        load_gate_policy(repo)


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
    policy = _empty_policy(source_exemptions=(expired,))
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
    policy = _empty_policy(source_exemptions=(expired,))
    test_map = {"tests/smoke/test_a.py::test_x": {"cli/main.py": ["run"]}}
    assert find_expired_unmapped(policy, test_map, today=date(2026, 1, 1)) == ()


def test_find_expired_test_exemptions_reports_past_deadline() -> None:
    policy = GatePolicy(
        sources=PathPatterns(include_patterns=DEFAULT_GATE_ROOTS, exclude_patterns=()),
        tests=PathPatterns(
            include_patterns=DEFAULT_TEST_INCLUDE,
            exclude_patterns=DEFAULT_TEST_EXCLUDE,
        ),
        configs=PathPatterns(include_patterns=DEFAULT_CONFIG_INCLUDE, exclude_patterns=()),
        source_exemptions=(),
        test_exemptions=(
            TestExemption(
                test_id="tests/regression/nightly/test_x.py::test_case",
                reason="x",
                applicant="a",
                approver="fangkai",
                deadline=date(2020, 1, 1),
            ),
        ),
        approvers=frozenset(),
    )
    reports = find_expired_test_exemptions(policy, today=date(2026, 1, 1))
    assert len(reports) == 1
    assert reports[0].symbol_key == "tests/regression/nightly/test_x.py::test_case"


def test_format_expired_exemptions_section_empty_returns_empty_string() -> None:
    assert format_expired_exemptions_section(()) == ""


def test_format_expired_test_exemptions_section_includes_test_id() -> None:
    report = find_expired_test_exemptions(
        GatePolicy(
            sources=PathPatterns(include_patterns=DEFAULT_GATE_ROOTS, exclude_patterns=()),
            tests=PathPatterns(
                include_patterns=DEFAULT_TEST_INCLUDE,
                exclude_patterns=DEFAULT_TEST_EXCLUDE,
            ),
            configs=PathPatterns(include_patterns=DEFAULT_CONFIG_INCLUDE, exclude_patterns=()),
            source_exemptions=(),
            test_exemptions=(
                TestExemption(
                    test_id="tests/regression/nightly/test_x.py::test_case",
                    reason="x",
                    applicant="a",
                    approver="fangkai",
                    deadline=date(2020, 1, 1),
                ),
            ),
            approvers=frozenset(),
        ),
        today=date(2026, 1, 1),
    )[0]
    section = format_expired_test_exemptions_section((report,))
    assert "Expired test exemptions" in section
    assert "tests/regression/nightly/test_x.py::test_case" in section
    assert "gate_policy.yaml" in section


def test_gate_policy_changed_in_diff_true_when_file_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "tests/.ci/gate_policy.yaml\n", ""),
    )
    assert gate_policy_changed_in_diff(tmp_path, "abc123") is True

"""Tests for ci_gate.policy — gate_policy.yaml loader and path matching."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.policy import (
    is_gate_test_path,
    is_policy_config_path,
    load_gate_policy,
    matches_path_patterns,
)
from scripts.helpers.common.test_map_loader import is_product_source
from tests.regression.scripts.helpers.gate_policy_writer import (
    DEFAULT_CONFIG_INCLUDE,
    DEFAULT_GATE_ROOTS,
    DEFAULT_TEST_EXCLUDE,
    DEFAULT_TEST_INCLUDE,
    write_gate_policy,
)


def _write_gate_policy(
    repo: Path,
    *,
    source_roots: list[str] | None = None,
    test_include: list[str] | None = None,
    test_exclude: list[str] | None = None,
    config_include: list[str] | None = None,
    exemptions: list[dict[str, object]] | None = None,
) -> None:
    write_gate_policy(
        repo,
        roots=source_roots,
        test_include=test_include,
        test_exclude=test_exclude,
        config_include=config_include,
        source_exemptions=exemptions,
        approvers=("fangkai", "hexiaowu"),
    )


def test_load_gate_policy_reads_sources_tests_configs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_gate_policy(repo)
    policy = load_gate_policy(repo)
    assert policy.roots == DEFAULT_GATE_ROOTS
    assert policy.tests.include_patterns[0].startswith("tests/")
    assert policy.configs.include_patterns == DEFAULT_CONFIG_INCLUDE


def test_load_gate_policy_requires_roots(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    ci_dir = repo / "tests" / ".ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    policy = {
        "tests": {"include": list(DEFAULT_TEST_INCLUDE), "exclude": list(DEFAULT_TEST_EXCLUDE)},
        "configs": {"include": list(DEFAULT_CONFIG_INCLUDE), "exclude": []},
        "exemptions": {"sources": [], "tests": []},
    }
    (ci_dir / "gate_policy.yaml").write_text(yaml.dump(policy), encoding="utf-8")
    (ci_dir / "approvers.yaml").write_text(yaml.dump({"approvers": ["fangkai"]}), encoding="utf-8")
    with pytest.raises(ConfigError, match="roots"):
        load_gate_policy(repo)


def test_load_gate_policy_requires_tests(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    ci_dir = repo / "tests" / ".ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    policy = {
        "roots": list(DEFAULT_GATE_ROOTS),
        "configs": {"include": list(DEFAULT_CONFIG_INCLUDE), "exclude": []},
        "exemptions": {"sources": [], "tests": []},
    }
    (ci_dir / "gate_policy.yaml").write_text(yaml.dump(policy), encoding="utf-8")
    (ci_dir / "approvers.yaml").write_text(yaml.dump({"approvers": ["fangkai"]}), encoding="utf-8")
    with pytest.raises(ConfigError, match="tests"):
        load_gate_policy(repo)


def test_load_gate_policy_requires_configs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    ci_dir = repo / "tests" / ".ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    policy = {
        "roots": list(DEFAULT_GATE_ROOTS),
        "tests": {"include": list(DEFAULT_TEST_INCLUDE), "exclude": list(DEFAULT_TEST_EXCLUDE)},
        "exemptions": {"sources": [], "tests": []},
    }
    (ci_dir / "gate_policy.yaml").write_text(yaml.dump(policy), encoding="utf-8")
    (ci_dir / "approvers.yaml").write_text(yaml.dump({"approvers": ["fangkai"]}), encoding="utf-8")
    with pytest.raises(ConfigError, match="configs"):
        load_gate_policy(repo)


def test_is_product_source_uses_policy_roots(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_gate_policy(repo)
    policy = load_gate_policy(repo)
    assert is_product_source("cli/main.py", policy.roots) is True
    assert is_product_source("web_ui/app.py", policy.roots) is True
    assert is_product_source("tests/regression/cli/test_main.py", policy.roots) is False


def test_is_policy_config_path_uses_policy_configs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_gate_policy(repo)
    policy = load_gate_policy(repo)
    assert is_policy_config_path("pyproject.toml", policy.configs) is True
    assert is_policy_config_path("tests/regression/conftest.py", policy.configs) is True
    assert is_policy_config_path("tests/.ci/gate_policy.yaml", policy.configs) is False
    assert is_policy_config_path("cli/main.py", policy.configs) is False


def test_matches_path_patterns_respects_exclude(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_gate_policy(
        repo,
        test_exclude=[
            "tests/helpers/**",
            "tests/assets/**",
            "tests/regression/nightly/**",
        ],
    )
    policy = load_gate_policy(repo)
    assert is_gate_test_path("tests/regression/cli/test_run.py", policy.discovery) is True
    assert is_gate_test_path("tests/regression/nightly/test_x.py", policy.discovery) is False
    assert matches_path_patterns("tests/regression/nightly/test_x.py", policy.tests) is False


def test_load_gate_policy_missing_file_raises(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    with pytest.raises(ConfigError, match=r"tests/\.ci/approvers\.yaml.*not found"):
        load_gate_policy(repo)

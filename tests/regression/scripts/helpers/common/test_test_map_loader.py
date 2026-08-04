"""Tests for common.test_map_loader."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.ci_gate.models import SourceExemption
from scripts.helpers.ci_gate.policy import is_exempt
from scripts.helpers.ci_gate.test_map_query import prune_deleted_sources
from scripts.helpers.common.test_map_loader import (
    assess_test_map_freshness,
    is_product_source,
    load_baseline,
    load_test_map,
    load_test_map_with_commit,
    validate_test_map_freshness,
)
from tests.regression.scripts.helpers.gate_policy_writer import write_gate_policy

_LEGACY_MAP_JSON = json.dumps(
    {
        "schema_version": 1,
        "map": {
            "tensor_cast/foo.py": {
                "bar": ["tests/smoke/test_bar.py::test_x"],
            },
            "cli/main.py": {
                "run": ["tests/regression/cli/test_run.py::test_run"],
            },
        },
    }
)

_NODE_MAP_JSON = json.dumps(
    {
        "schema_version": 1,
        "built_from_commit": "abc123",
        "map": {
            "tests/smoke/test_bar.py::test_x": {
                "tensor_cast/foo.py": ["bar"],
            },
            "tests/regression/cli/test_run.py::test_run": {
                "cli/main.py": ["run"],
            },
        },
    }
)


def test_load_test_map_rejects_source_oriented_map(tmp_path: Path) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text(_LEGACY_MAP_JSON, encoding="utf-8")
    with pytest.raises(ConfigError, match="pytest node id"):
        load_test_map(_cfg_with_path(str(map_path)))


def test_load_test_map_accepts_decorator_suffix_symbol(tmp_path: Path) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "map": {
                    "tests/regression/cli/test_run.py::test_run": {
                        "cli/main.py": ["_@_decorator(torch.ops.foo.bar)"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    result = load_test_map(_cfg_with_path(str(map_path)))
    assert result["tests/regression/cli/test_run.py::test_run"]["cli/main.py"] == ["_@_decorator(torch.ops.foo.bar)"]


def test_load_test_map_rejects_legacy_dot_symbol(tmp_path: Path) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "map": {
                    "tests/regression/cli/test_run.py::test_run": {
                        "cli/main.py": ["Widget.run"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="canonical Class::method"):
        load_test_map(_cfg_with_path(str(map_path)))


def test_load_test_map_v2_returns_node_oriented_map(tmp_path: Path) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text(_NODE_MAP_JSON, encoding="utf-8")
    result = load_test_map(_cfg_with_path(str(map_path)))
    assert result["tests/regression/cli/test_run.py::test_run"]["cli/main.py"] == ["run"]


def test_load_test_map_unsupported_schema_version_raises_config_error(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps({"schema_version": 9, "map": {}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported schema_version"):
        load_test_map(_cfg_with_path(str(map_path)))


def test_load_test_map_invalid_json_raises_config_error_with_path(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ConfigError, match=f"invalid JSON at {map_path}:"):
        load_test_map(_cfg_with_path(str(map_path)))


def test_load_test_map_invalid_test_node_raises_config_error(tmp_path: Path) -> None:
    payload = json.dumps({"schema_version": 1, "map": {"not-a-test-node": {"cli/main.py": ["run"]}}})
    map_path = tmp_path / "map.json"
    map_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ConfigError, match="pytest node id"):
        load_test_map(_cfg_with_path(str(map_path)))


def test_load_test_map_with_commit_reads_built_from_commit(tmp_path: Path) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text(_NODE_MAP_JSON, encoding="utf-8")
    mapping, commit = load_test_map_with_commit(_cfg_with_path(str(map_path)))
    assert commit == "abc123"
    assert "tests/regression/cli/test_run.py::test_run" in mapping


def test_validate_test_map_freshness_rejects_missing_commit(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="built_from_commit is required"):
        validate_test_map_freshness(tmp_path, None, "abc123")


def test_assess_test_map_freshness_warns_when_map_is_ancestor_of_merge_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []

    def _fake_ancestor(_repo: Path, ancestor: str, descendant: str) -> bool:
        calls.append((ancestor, descendant, "check"))
        return ancestor == "old" and descendant == "merge"

    monkeypatch.setattr("scripts.helpers.common.test_map_loader.is_git_ancestor", _fake_ancestor)
    result = assess_test_map_freshness(tmp_path, "old", "merge")
    assert result.block_message is None
    assert result.warn_message is not None


def test_load_baseline_assembles_test_map_and_policy(tmp_path: Path) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text(_NODE_MAP_JSON, encoding="utf-8")
    repo = tmp_path / "repo"
    write_gate_policy(repo, approvers=("fangkai",))
    baseline, commit = load_baseline(repo, _cfg_with_path(str(map_path)))
    assert commit == "abc123"
    assert baseline.test_map["tests/regression/cli/test_run.py::test_run"]["cli/main.py"] == ["run"]
    assert baseline.roots
    assert baseline.discovery.include_patterns


def test_is_exempt_matching_file_and_symbol_returns_true() -> None:
    exemptions = (
        SourceExemption(
            file="a.py",
            symbol="fn",
            reason="",
            applicant="",
            approver="fangkai",
            deadline=date(2099, 12, 31),
        ),
    )
    assert is_exempt(exemptions, "a.py", "fn") is True


def test_is_product_source_matching_prefix_returns_true() -> None:
    assert is_product_source("cli/main.py", ("cli/",)) is True


def test_prune_removes_deleted_source_paths_from_test_nodes() -> None:
    tm = {
        "tests/a.py::test_x": {"a.py": ["fn"], "b.py": ["fn"]},
    }
    result = prune_deleted_sources(tm, ("a.py",))
    assert result == {"tests/a.py::test_x": {"b.py": ["fn"]}}


def _cfg_with_path(path: str) -> Config:
    return Config(
        test_map_path=path,
        base_branch="master",
        line_threshold=70.0,
        branch_threshold=50.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )

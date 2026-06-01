"""Tests for common.test_map_loader."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from scripts.helpers._config import Config, ConfigError
from scripts.helpers.ci_gate.gate_policy import SourceExemption, is_exempt
from scripts.helpers.common.test_map_loader import (
    is_product_source,
    load_baseline,
    load_test_map,
    prune_deleted_sources,
)

# ---------------------------------------------------------------------------
# Valid test_map JSON
# ---------------------------------------------------------------------------

_VALID_MAP_JSON = json.dumps(
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


def _write_ci_files(repo: Path) -> None:
    ci_dir = repo / "tests" / ".ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    (ci_dir / "gate_policy.yaml").write_text(
        yaml.dump({"schema_version": 1, "exemptions": []}),
        encoding="utf-8",
    )
    (ci_dir / "approvers.yaml").write_text(
        yaml.dump({"schema_version": 1, "approvers": ["fangkai"]}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# load_test_map
# ---------------------------------------------------------------------------


def test_load_test_map_valid_returns_parsed_dict(tmp_path: Path) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text(_VALID_MAP_JSON, encoding="utf-8")
    cfg = _cfg_with_path(str(map_path))
    result = load_test_map(cfg)
    assert result["tensor_cast/foo.py"]["bar"] == ["tests/smoke/test_bar.py::test_x"]


def test_load_test_map_missing_schema_version_raises_config_error(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps({"map": {}}), encoding="utf-8")
    cfg = _cfg_with_path(str(map_path))
    with pytest.raises(ConfigError, match="schema_version must be 1"):
        load_test_map(cfg)


def test_load_test_map_invalid_json_raises_config_error_with_path(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text("{invalid", encoding="utf-8")
    cfg = _cfg_with_path(str(map_path))
    with pytest.raises(ConfigError, match=f"invalid JSON at {map_path}:"):
        load_test_map(cfg)


def test_load_test_map_key_not_product_prefix_raises_config_error(
    tmp_path: Path,
) -> None:
    payload = json.dumps({"schema_version": 1, "map": {"other/file.py": {}}})
    map_path = tmp_path / "map.json"
    map_path.write_text(payload, encoding="utf-8")
    cfg = _cfg_with_path(str(map_path))
    with pytest.raises(ConfigError, match="must start with a product prefix.*'other/file.py'"):
        load_test_map(cfg)


def test_load_test_map_path_traversal_key_raises_config_error(tmp_path: Path) -> None:
    payload = json.dumps({"schema_version": 1, "map": {"../escape.py": {}}})
    map_path = tmp_path / "map.json"
    map_path.write_text(payload, encoding="utf-8")
    cfg = _cfg_with_path(str(map_path))
    with pytest.raises(ConfigError, match="invalid map key.*'../escape.py'"):
        load_test_map(cfg)


# ---------------------------------------------------------------------------
# load_baseline
# ---------------------------------------------------------------------------


def test_load_baseline_assembles_test_map_and_prefixes(tmp_path: Path) -> None:
    map_path = tmp_path / "map.json"
    map_path.write_text(_VALID_MAP_JSON, encoding="utf-8")
    repo = tmp_path / "repo"
    _write_ci_files(repo)
    cfg = _cfg_with_path(str(map_path))
    baseline = load_baseline(repo, cfg)
    assert "tensor_cast/foo.py" in baseline.test_map
    assert baseline.product_prefixes is not None
    assert baseline.discovery.include_patterns


# ---------------------------------------------------------------------------
# is_exempt (from gate_policy)
# ---------------------------------------------------------------------------


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


def test_is_exempt_different_symbol_returns_false() -> None:
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
    assert is_exempt(exemptions, "a.py", "other") is False


# ---------------------------------------------------------------------------
# is_product_source
# ---------------------------------------------------------------------------


def test_is_product_source_matching_prefix_returns_true() -> None:
    assert is_product_source("cli/main.py", ("cli/",)) is True


def test_is_product_source_non_matching_prefix_returns_false() -> None:
    assert is_product_source("tests/test_x.py", ("cli/",)) is False


# ---------------------------------------------------------------------------
# prune_deleted_sources
# ---------------------------------------------------------------------------


def test_prune_removes_deleted_keys_keeps_others() -> None:
    tm = {"a.py": {}, "b.py": {}}
    result = prune_deleted_sources(tm, ("a.py",))
    assert "a.py" not in result
    assert "b.py" in result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

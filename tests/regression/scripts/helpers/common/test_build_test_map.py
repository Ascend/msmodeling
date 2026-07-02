"""Tests for common.build_test_map."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from scripts.helpers.common.ast_utils import MODULE_SYMBOL
from scripts.helpers.common.build_test_map import (
    TEST_MAP_SCHEMA_VERSION,
    _collect_allowed_node_ids,
    _normalize_pytest_context,
    _prune_missing_source_keys,
    _relative_repo_key,
    collect_from_coverage,
    collect_test_map,
    detect_redundant_cases,
    normalize_test_node_id,
    write_test_map,
)
from scripts.helpers.common.coverage_config import product_roots
from scripts.helpers.common.pytest_runner import PYTEST_IGNORE_ADDOPTS
from tests.helpers.fake_subprocess import FakeCompleted

# ---------------------------------------------------------------------------
# normalize_test_node_id
# ---------------------------------------------------------------------------


def test_normalize_test_node_id_strips_param_and_phase() -> None:
    assert normalize_test_node_id("tests/test_x.py::test_a[0]|run") == "tests/test_x.py::test_a"


# ---------------------------------------------------------------------------
# _relative_repo_key
# ---------------------------------------------------------------------------


def test_relative_repo_key_product_prefix_returns_rel_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.helpers.common import build_test_map

    (tmp_path / "cli").mkdir()
    abs_file = tmp_path / "cli" / "main.py"
    abs_file.write_text("", encoding="utf-8")

    monkeypatch.setattr(build_test_map, "REPO_ROOT", tmp_path)
    result = _relative_repo_key(str(abs_file), product_roots())
    assert result == "cli/main.py"


def test_relative_repo_key_outside_repo_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.helpers.common import build_test_map

    monkeypatch.setattr(build_test_map, "REPO_ROOT", tmp_path)
    assert _relative_repo_key("/other/path/file.py", product_roots()) is None


def test_relative_repo_key_non_product_prefix_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.helpers.common import build_test_map

    monkeypatch.setattr(build_test_map, "REPO_ROOT", tmp_path)
    (tmp_path / "other").mkdir()
    abs_file = tmp_path / "other" / "file.py"
    abs_file.write_text("", encoding="utf-8")
    result = _relative_repo_key(str(abs_file), product_roots())
    assert result is None


# ---------------------------------------------------------------------------
# _normalize_pytest_context
# ---------------------------------------------------------------------------


def test_normalize_strips_run_suffix() -> None:
    assert _normalize_pytest_context("tests/test_x.py::test_a|run") == "tests/test_x.py::test_a"


def test_normalize_strips_setup_suffix() -> None:
    assert _normalize_pytest_context("tests/test_x.py::test_a|setup") == "tests/test_x.py::test_a"


def test_normalize_strips_param_suffix() -> None:
    assert _normalize_pytest_context("tests/test_x.py::test_a[0]|run") == "tests/test_x.py::test_a"


def test_normalize_no_suffix_unchanged() -> None:
    assert _normalize_pytest_context("tests/test_x.py::test_a") == "tests/test_x.py::test_a"


def test_normalize_empty_string_returns_empty() -> None:
    assert _normalize_pytest_context("") == ""


# ---------------------------------------------------------------------------
# _collect_allowed_node_ids
# ---------------------------------------------------------------------------


def test_collect_allowed_node_ids_strips_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_stdout = (
        "tests/smoke/test_a.py::test_foo\ntests/smoke/test_a.py::test_bar[0]\ntests/smoke/test_a.py::test_bar[1]\n"
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, fake_stdout, ""))
    result = _collect_allowed_node_ids("not npu")
    assert "tests/smoke/test_a.py::test_foo" in result
    assert "tests/smoke/test_a.py::test_bar" in result
    assert "tests/smoke/test_a.py::test_bar[0]" not in result


def test_collect_allowed_node_ids_pytest_fails_raises_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(1, "", "collection error"))
    with pytest.raises(SystemExit):
        _collect_allowed_node_ids("not npu")


def test_collect_allowed_node_ids_includes_ignore_addopts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kw: object) -> FakeCompleted:
        captured.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("subprocess.run", _fake_run)
    _collect_allowed_node_ids("not npu")
    assert PYTEST_IGNORE_ADDOPTS[0] in captured[0]


def test_collect_allowed_node_ids_includes_build_helper_regression_tests() -> None:
    """tests/.../helpers/build must not be skipped by pytest norecursedirs=build."""
    from scripts.helpers.common.test_map_config import TEST_MAP_COLLECTION_MARKER

    allowed = _collect_allowed_node_ids(TEST_MAP_COLLECTION_MARKER)
    build_nodes = [node_id for node_id in allowed if "scripts/helpers/build/" in node_id]
    assert build_nodes, "build helper regression tests must be collectable for test_map sync"


def test_collect_test_map_skips_collect_when_allowed_node_ids_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.helpers.common import build_test_map

    allowed = frozenset({"tests/smoke/test_a.py::test_foo"})
    collect_calls: list[str] = []
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "main.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(build_test_map, "REPO_ROOT", tmp_path)

    def _fail_collect(_marker_expr: str, _pytest_args: list[str] | None = None) -> frozenset[str]:
        collect_calls.append(_marker_expr)
        raise AssertionError("must not call _collect_allowed_node_ids")

    monkeypatch.setattr(build_test_map, "_collect_allowed_node_ids", _fail_collect)
    monkeypatch.setattr(
        build_test_map,
        "collect_from_coverage",
        lambda node_ids, **_kwargs: (
            {
                "tests/smoke/test_a.py::test_foo": {"cli/main.py": ["run"]},
            }
            if node_ids
            else {}
        ),
    )
    result = collect_test_map(marker_expr="not npu", allowed_node_ids=allowed)
    assert collect_calls == []
    assert result == {"tests/smoke/test_a.py::test_foo": {"cli/main.py": ["run"]}}


# ---------------------------------------------------------------------------
# collect_from_coverage
# ---------------------------------------------------------------------------


def test_collect_skips_coverage_omitted_source_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.helpers.common import build_test_map

    repo = tmp_path / "repo"
    source = repo / "tensor_cast" / "builtin_model" / "foo.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    coverage_path = repo / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(build_test_map, "REPO_ROOT", repo)

    class _FakeCoverageData:
        def __init__(self, _path: str) -> None:
            self._source = str(source.resolve())

        def read(self) -> None:
            return None

        def measured_files(self) -> list[str]:
            return [self._source]

        def contexts_by_lineno(self, _path: str) -> dict[int, list[str]]:
            return {2: ["tests/regression/tensor_cast/test_a.py::test_x"]}

    monkeypatch.setattr("coverage.data.CoverageData", _FakeCoverageData)

    monkeypatch.setattr(
        "scripts.helpers.common.build_test_map.is_coverage_omitted_source",
        lambda path, _roots: "builtin_model" in path,
    )
    result = collect_from_coverage(
        frozenset({"tests/regression/tensor_cast/test_a.py::test_x"}),
        coverage_path=coverage_path,
        roots=("tensor_cast/",),
    )

    assert result == {}


def test_collect_builds_node_oriented_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.helpers.common import build_test_map

    repo = tmp_path / "repo"
    source = repo / "cli" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    coverage_path = repo / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(build_test_map, "REPO_ROOT", repo)

    class _FakeCoverageData:
        def __init__(self, _path: str) -> None:
            self._source = str(source.resolve())

        def read(self) -> None:
            return None

        def measured_files(self) -> list[str]:
            return [self._source]

        def contexts_by_lineno(self, _path: str) -> dict[int, list[str]]:
            return {2: ["tests/regression/cli/test_a.py::test_x|run"]}

    monkeypatch.setattr("coverage.data.CoverageData", _FakeCoverageData)
    result = collect_from_coverage(
        frozenset({"tests/regression/cli/test_a.py::test_x"}),
        coverage_path=coverage_path,
        roots=("cli/",),
    )
    assert result == {"tests/regression/cli/test_a.py::test_x": {"cli/main.py": ["run"]}}


def test_collect_returns_empty_when_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.helpers.common import build_test_map

    monkeypatch.setattr(build_test_map, "REPO_ROOT", tmp_path)
    result = collect_from_coverage(frozenset(), coverage_path=tmp_path / ".coverage")
    assert result == {}


def test_collect_returns_empty_on_read_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import coverage.data

    coverage_path = tmp_path / ".coverage"
    coverage_path.write_text("garbage", encoding="utf-8")
    monkeypatch.setattr(
        coverage.data.CoverageData,
        "read",
        lambda self: (_ for _ in ()).throw(OSError("bad file")),
    )
    result = collect_from_coverage(frozenset(), coverage_path=coverage_path)
    assert result == {}


# ---------------------------------------------------------------------------
# _prune_missing_source_keys
# ---------------------------------------------------------------------------


def test_prune_keeps_existing_files_drops_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.helpers.common import build_test_map

    monkeypatch.setattr(build_test_map, "REPO_ROOT", tmp_path)
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "main.py").write_text("", encoding="utf-8")
    mapping = {
        "tests/a.py::test_a": {"cli/main.py": ["run"]},
        "tests/a.py::test_b": {"cli/gone.py": ["fn"]},
    }
    result = _prune_missing_source_keys(mapping)
    assert "tests/a.py::test_a" in result
    assert "cli/gone.py" not in result.get("tests/a.py::test_b", {})
    assert "tests/a.py::test_b" not in result


# ---------------------------------------------------------------------------
# write_test_map
# ---------------------------------------------------------------------------


def test_write_test_map_creates_valid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.helpers.common.build_test_map.resolve_head_commit",
        lambda _root: "deadbeef" * 5,
    )
    output = tmp_path / "out" / "map.json"
    write_test_map(output, {"tests/a.py::test_x": {"cli/main.py": ["run"]}})
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema_version"] == TEST_MAP_SCHEMA_VERSION
    assert data["built_from_commit"] == "deadbeef" * 5
    assert data["map"]["tests/a.py::test_x"]["cli/main.py"] == ["run"]


def test_build_test_map_writes_pruned_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.helpers.common import build_test_map as build_test_map_mod

    repo = tmp_path / "repo"
    (repo / "cli").mkdir(parents=True)
    (repo / "cli" / "main.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    output = repo / "test_map.json"
    allowed = frozenset({"tests/smoke/test_a.py::test_foo"})
    monkeypatch.setattr(build_test_map_mod, "REPO_ROOT", repo)
    monkeypatch.setattr(build_test_map_mod, "_collect_allowed_node_ids", lambda _marker: allowed)
    monkeypatch.setattr(
        build_test_map_mod,
        "collect_from_coverage",
        lambda node_ids, **_kwargs: {"tests/smoke/test_a.py::test_foo": {"cli/main.py": ["run"]}} if node_ids else {},
    )
    monkeypatch.setattr(
        build_test_map_mod,
        "resolve_head_commit",
        lambda _root: "deadbeef" * 5,
    )
    build_test_map_mod.build_test_map(output, marker_expr="not npu", roots=("cli/",))
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["map"]["tests/smoke/test_a.py::test_foo"]["cli/main.py"] == ["run"]


# ---------------------------------------------------------------------------
# detect_redundant_cases
# ---------------------------------------------------------------------------


def test_detect_redundant_cases_over_covered_symbol() -> None:
    mapping = {
        "tests/regression/cli/test_a.py::test_1": {"cli/main.py": ["run"]},
        "tests/regression/cli/test_a.py::test_2": {"cli/main.py": ["run"]},
        "tests/regression/cli/test_a.py::test_3": {"cli/main.py": ["run"]},
        "tests/regression/cli/test_a.py::test_4": {"cli/main.py": ["run"]},
        "tests/regression/cli/test_a.py::test_5": {"cli/main.py": ["run"]},
        "tests/regression/cli/test_a.py::test_6": {"cli/main.py": ["run"]},
    }
    warnings = detect_redundant_cases(mapping, max_per_symbol=5)
    over_covered = [warning for warning in warnings if warning["type"] == "over_covered_symbol"]
    assert len(over_covered) == 1
    assert over_covered[0]["symbol"] == "cli/main.py::run"
    assert over_covered[0]["test_count"] == 6


def test_detect_redundant_cases_no_over_covered_when_within_limit() -> None:
    mapping = {"tests/regression/cli/test_a.py::test_1": {"cli/main.py": ["run"]}}
    warnings = detect_redundant_cases(mapping, max_per_symbol=5)
    over_covered = [warning for warning in warnings if warning["type"] == "over_covered_symbol"]
    assert len(over_covered) == 0


def test_detect_redundant_cases_redundant_pair_high_jaccard() -> None:
    mapping = {
        "tests/a.py::test_1": {"cli/main.py": ["run", "init"]},
        "tests/a.py::test_2": {"cli/main.py": ["run", "init"]},
    }
    warnings = detect_redundant_cases(mapping, jaccard_threshold=0.85)
    pairs = [warning for warning in warnings if warning["type"] == "redundant_pair"]
    assert len(pairs) == 1
    assert pairs[0]["jaccard"] == 1.0


def test_detect_redundant_cases_no_redundant_pair_low_jaccard() -> None:
    mapping = {
        "tests/a.py::test_1": {"cli/main.py": ["run"]},
        "tests/a.py::test_2": {"tensor_cast/ops.py": ["add"]},
    }
    warnings = detect_redundant_cases(mapping, jaccard_threshold=0.85)
    pairs = [warning for warning in warnings if warning["type"] == "redundant_pair"]
    assert len(pairs) == 0


def test_detect_redundant_cases_empty_mapping_returns_empty() -> None:
    warnings = detect_redundant_cases({})
    assert warnings == []


def test_detect_redundant_cases_ignores_module_symbol_for_pairs() -> None:
    mapping = {
        "tests/a.py::test_1": {"cli/main.py": [MODULE_SYMBOL]},
        "tests/a.py::test_2": {"cli/main.py": [MODULE_SYMBOL]},
    }
    warnings = detect_redundant_cases(mapping, jaccard_threshold=0.85)
    pairs = [warning for warning in warnings if warning["type"] == "redundant_pair"]
    assert pairs == []

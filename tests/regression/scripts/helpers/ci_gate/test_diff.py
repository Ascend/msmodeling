"""Tests for ci_gate.diff — resolve_base_ref, fetch_diff_line_map, classify_changes,
regression_layer_for_source, layer_of_test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.diff import (
    _classify_rename,
    classify_changes,
    fetch_diff_line_map,
    resolve_base_ref,
)
from scripts.helpers.ci_gate.gate_policy import default_test_discovery
from scripts.helpers.ci_gate.models import layer_of_test, regression_layer_for_source
from tests.helpers.fake_subprocess import FakeCompleted

# ---------------------------------------------------------------------------
# resolve_base_ref
# ---------------------------------------------------------------------------


def test_resolve_base_ref_merge_base_success_returns_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "abc123\n", ""))
    result = resolve_base_ref(tmp_path, "main")
    assert result == "abc123"


def test_resolve_base_ref_fallback_to_origin_returns_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = iter(
        [
            FakeCompleted(1, "", ""),
            FakeCompleted(0, "def456\n", ""),
        ]
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: next(calls))
    result = resolve_base_ref(tmp_path, "main")
    assert result == "def456"


def test_resolve_base_ref_both_fail_raises_config_error_with_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(1, "", "not found"))
    with pytest.raises(ConfigError, match="Cannot resolve base ref.*'nonexistent'.*not found either"):
        resolve_base_ref(tmp_path, "nonexistent")


# ---------------------------------------------------------------------------
# fetch_diff_line_map
# ---------------------------------------------------------------------------


def test_fetch_diff_line_map_parses_hunks_into_line_sets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    diff_output = "+++ b/cli/main.py\n@@ -0,0 +5,3 @@\n+++ b/tensor_cast/ops.py\n@@ -10,0 +20,2 @@\n"
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, diff_output, ""))
    result = fetch_diff_line_map(tmp_path, "abc123")
    assert result["cli/main.py"] == {5, 6, 7}
    assert result["tensor_cast/ops.py"] == {20, 21}


def test_fetch_diff_line_map_single_line_hunk_returns_single_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "+++ b/a.py\n@@ -0,0 +1 @@\n", ""),
    )
    result = fetch_diff_line_map(tmp_path, "abc123")
    assert result["a.py"] == {1}


def test_fetch_diff_line_map_empty_diff_returns_empty_dict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "", ""))
    result = fetch_diff_line_map(tmp_path, "abc123")
    assert result == {}


# ---------------------------------------------------------------------------
# classify_changes
# ---------------------------------------------------------------------------


def test_classify_changes_helpers_path_not_new_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "A\ttests/helpers/assert_utils.py\n", ""),
    )
    result = classify_changes(tmp_path, "abc123", {})
    assert result.new_test == ()


def test_classify_changes_added_test_populates_new_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "A\ttests/smoke/test_new.py\n", ""),
    )
    result = classify_changes(tmp_path, "abc123", {})
    assert "tests/smoke/test_new.py" in result.new_test


def test_classify_changes_modified_source_includes_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "M\tcli/main.py\n", ""))
    diff_map = {"cli/main.py": {10, 11}}
    result = classify_changes(tmp_path, "abc123", diff_map)
    assert len(result.modified_source) == 1
    assert result.modified_source[0][0] == "cli/main.py"
    assert result.modified_source[0][1] == frozenset({10, 11})


def test_classify_changes_deleted_test_populates_del_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "D\ttests/smoke/test_old.py\n", ""),
    )
    result = classify_changes(tmp_path, "abc123", {})
    assert "tests/smoke/test_old.py" in result.del_test


def test_classify_changes_config_triggers_full_suite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "M\tpyproject.toml\n", ""))
    result = classify_changes(tmp_path, "abc123", {})
    assert result.config == ("pyproject.toml",)


def test_classify_changes_deleted_source_populates_del_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "D\tcli/old_main.py\n", ""))
    result = classify_changes(tmp_path, "abc123", {})
    assert "cli/old_main.py" in result.del_source


# ---------------------------------------------------------------------------
# _classify_rename (PR review: old/new path boundary)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("old_path", "new_path", "score", "expected"),
    [
        (
            "tests/smoke/test_foo.py",
            "tests/regression/test_foo.py",
            100,
            {
                "del_test": ["tests/smoke/test_foo.py"],
                "new_test": ["tests/regression/test_foo.py"],
                "del_source": [],
                "renames": [],
                "modified": {},
            },
        ),
        (
            "tests/smoke/test_foo.py",
            "scripts/helpers/foo.py",
            100,
            {
                "del_test": ["tests/smoke/test_foo.py"],
                "new_test": [],
                "del_source": [],
                "renames": [],
                "modified": {},
            },
        ),
        (
            "tests/regression/test_bar.py",
            "tests/helpers/test_bar.py",
            100,
            {
                "del_test": ["tests/regression/test_bar.py"],
                "new_test": [],
                "del_source": [],
                "renames": [],
                "modified": {},
            },
        ),
        (
            "tensor_cast/old_module.py",
            "tests/regression/test_old_module.py",
            100,
            {
                "del_test": [],
                "new_test": ["tests/regression/test_old_module.py"],
                "del_source": ["tensor_cast/old_module.py"],
                "renames": [],
                "modified": {},
            },
        ),
        (
            "tensor_cast/foo.py",
            "tensor_cast/bar.py",
            100,
            {
                "del_test": [],
                "new_test": [],
                "del_source": [],
                "renames": [("tensor_cast/foo.py", "tensor_cast/bar.py", 100)],
                "modified": {},
            },
        ),
        (
            "tensor_cast/foo.py",
            "tensor_cast/bar.py",
            85,
            {
                "del_test": [],
                "new_test": [],
                "del_source": [],
                "renames": [("tensor_cast/foo.py", "tensor_cast/bar.py", 85)],
                "modified": {"tensor_cast/bar.py": frozenset({10, 11})},
            },
        ),
    ],
)
def test_classify_rename_paths(
    old_path: str,
    new_path: str,
    score: int,
    expected: dict[str, object],
) -> None:
    discovery = default_test_discovery()
    diff = {"tensor_cast/bar.py": {10, 11}}

    del_test, new_test, del_source, renames, modified = _classify_rename(
        old_path,
        new_path,
        score,
        diff,
        discovery,
    )

    assert del_test == expected["del_test"]
    assert new_test == expected["new_test"]
    assert del_source == expected["del_source"]
    assert renames == expected["renames"]
    assert modified == expected["modified"]


def test_classify_changes_rename_test_to_non_test_records_del_test_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(
            0,
            "R100\ttests/smoke/test_foo.py\tscripts/helpers/foo.py\n",
            "",
        ),
    )
    result = classify_changes(tmp_path, "abc123", {}, discovery=default_test_discovery())
    assert result.del_test == ("tests/smoke/test_foo.py",)
    assert result.new_test == ()
    assert result.renames == ()


def test_classify_changes_rename_product_to_test_records_del_source_and_new_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(
            0,
            "R100\ttensor_cast/old.py\ttests/regression/test_old.py\n",
            "",
        ),
    )
    result = classify_changes(tmp_path, "abc123", {}, discovery=default_test_discovery())
    assert result.del_source == ("tensor_cast/old.py",)
    assert result.new_test == ("tests/regression/test_old.py",)
    assert result.renames == ()


def test_classify_changes_rename_product_populates_renames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "R100\ttensor_cast/foo.py\ttensor_cast/bar.py\n", ""),
    )
    result = classify_changes(tmp_path, "abc123", {}, discovery=default_test_discovery())
    assert result.renames == (("tensor_cast/foo.py", "tensor_cast/bar.py", 100),)
    assert result.del_test == ()
    assert result.new_test == ()


# ---------------------------------------------------------------------------
# regression_layer_for_source
# ---------------------------------------------------------------------------


def test_regression_layer_tensor_cast_returns_tensor_cast_layer() -> None:
    assert regression_layer_for_source("tensor_cast/ops.py") == "tests/regression/tensor_cast/"


def test_regression_layer_serving_cast_returns_serving_cast_layer() -> None:
    assert regression_layer_for_source("serving_cast/api.py") == "tests/regression/serving_cast/"


def test_regression_layer_unknown_prefix_returns_none() -> None:
    assert regression_layer_for_source("cli/main.py") is None


# ---------------------------------------------------------------------------
# layer_of_test
# ---------------------------------------------------------------------------


def test_layer_of_test_tensor_cast_returns_tensor_cast_layer() -> None:
    assert layer_of_test("tests/regression/tensor_cast/test_ops.py::test_x") == "tests/regression/tensor_cast/"


def test_layer_of_test_cli_returns_cli_layer() -> None:
    assert layer_of_test("tests/regression/cli/test_run.py::test_y") == "tests/regression/cli/"


def test_layer_of_test_smoke_returns_none() -> None:
    assert layer_of_test("tests/smoke/test_a.py::test_z") is None

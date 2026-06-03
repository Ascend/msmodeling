"""Tests for ci_gate.rules — gate_* functions, _split_cross_layer_tests,
_merge_step_results, _product_paths.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from scripts.helpers.ci_gate.gate_policy import SourceExemption
from scripts.helpers.ci_gate.models import ChangeSet, GateError, GateStepResult
from scripts.helpers.ci_gate.rules import (
    _merge_step_results,
    _product_paths,
    _split_cross_layer_tests,
    gate_config,
    gate_deleted_source,
    gate_deleted_tests,
    gate_modified_source,
    gate_new_source,
    gate_new_tests,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def new_source_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a reusable source file under tensor_cast for gate_new_source tests."""
    src = tmp_path_factory.mktemp("gate_src") / "tensor_cast" / "new_mod.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def fn():\n    pass\n", encoding="utf-8")
    return src


# ---------------------------------------------------------------------------
# _split_cross_layer_tests
# ---------------------------------------------------------------------------


def test_split_single_layer_returns_all_immediate() -> None:
    immediate, deferred = _split_cross_layer_tests(
        "tensor_cast/ops.py",
        {"tests/regression/tensor_cast/test_a.py::test_x"},
    )
    assert immediate == {"tests/regression/tensor_cast/test_a.py::test_x"}
    assert deferred == set()


def test_split_cross_layer_defers_other_layer() -> None:
    immediate, deferred = _split_cross_layer_tests(
        "tensor_cast/ops.py",
        {
            "tests/regression/tensor_cast/test_a.py::test_x",
            "tests/regression/cli/test_b.py::test_y",
        },
    )
    assert immediate == {"tests/regression/tensor_cast/test_a.py::test_x"}
    assert deferred == {"tests/regression/cli/test_b.py::test_y"}


def test_split_no_preferred_prefix_returns_all_immediate() -> None:
    immediate, deferred = _split_cross_layer_tests(
        "other/unknown.py",
        {
            "tests/regression/tensor_cast/test_a.py::test_x",
            "tests/regression/cli/test_b.py::test_y",
        },
    )
    assert len(immediate) == 2
    assert deferred == set()


def test_split_empty_tests_returns_both_empty() -> None:
    immediate, deferred = _split_cross_layer_tests("tensor_cast/ops.py", set())
    assert immediate == set()
    assert deferred == set()


# ---------------------------------------------------------------------------
# _merge_step_results
# ---------------------------------------------------------------------------


def test_merge_combines_errors_from_all_steps() -> None:
    e1 = GateError(category="new_source", path="a.py")
    e2 = GateError(category="modified_source", path="b.py", symbol="fn")
    a = GateStepResult(errors=(e1,))
    b = GateStepResult(errors=(e2,))
    merged = _merge_step_results(a, b)
    assert merged.errors == (e1, e2)


def test_merge_unions_tests_and_deferred() -> None:
    a = GateStepResult(tests=frozenset({"t1"}), cross_layer_deferred=frozenset({"d1"}))
    b = GateStepResult(tests=frozenset({"t2"}), cross_layer_deferred=frozenset({"d2"}))
    merged = _merge_step_results(a, b)
    assert merged.tests == frozenset({"t1", "t2"})
    assert merged.cross_layer_deferred == frozenset({"d1", "d2"})


def test_merge_full_suite_true_if_any_true() -> None:
    a = GateStepResult(full_suite=False)
    b = GateStepResult(full_suite=True)
    merged = _merge_step_results(a, b)
    assert merged.full_suite is True


def test_merge_empty_returns_defaults() -> None:
    merged = _merge_step_results()
    assert merged.errors == ()
    assert merged.tests == frozenset()


# ---------------------------------------------------------------------------
# _product_paths
# ---------------------------------------------------------------------------


def test_product_paths_filters_non_product_prefixes() -> None:
    result = _product_paths(("cli/main.py", "tests/test_a.py"), ("cli/",))
    assert result == ("cli/main.py",)


def test_product_paths_empty_input_returns_empty() -> None:
    assert _product_paths((), ("cli/",)) == ()


# ---------------------------------------------------------------------------
# gate_config
# ---------------------------------------------------------------------------


def test_gate_config_triggers_full_suite() -> None:
    result = gate_config()
    assert result.full_suite is True
    assert result.errors == ()


# ---------------------------------------------------------------------------
# gate_new_tests
# ---------------------------------------------------------------------------


def test_gate_new_tests_selects_new_test_paths() -> None:
    cs = ChangeSet.build(new_test=("tests/smoke/test_a.py",))
    result = gate_new_tests(cs)
    assert result.tests == frozenset({"tests/smoke/test_a.py"})


def test_gate_new_tests_also_selects_modified_test_paths() -> None:
    cs = ChangeSet.build(
        new_test=("tests/smoke/test_a.py",),
        modified_test=("tests/regression/cli/test_b.py",),
    )
    result = gate_new_tests(cs)
    assert result.tests == frozenset({"tests/smoke/test_a.py", "tests/regression/cli/test_b.py"})


# ---------------------------------------------------------------------------
# gate_new_source
# ---------------------------------------------------------------------------


def test_gate_new_source_with_test_map_entry_returns_no_errors(
    new_source_file: Path,
) -> None:
    cs = ChangeSet.build(new_source=("tensor_cast/new_mod.py",))
    test_map = {"tensor_cast/new_mod.py": {"fn": ["test_a"]}}
    result = gate_new_source(new_source_file.parent.parent, cs, test_map, (), ("tensor_cast/",))
    assert result.errors == ()


def test_gate_new_source_missing_entry_reports_error(new_source_file: Path) -> None:
    cs = ChangeSet.build(new_source=("tensor_cast/new_mod.py",))
    result = gate_new_source(new_source_file.parent.parent, cs, {}, (), ("tensor_cast/",))
    assert len(result.errors) == 1
    assert result.errors[0].category == "new_source"
    assert result.errors[0].path == "tensor_cast/new_mod.py"


def test_gate_new_source_exempted_symbol_returns_no_errors(
    new_source_file: Path,
) -> None:
    cs = ChangeSet.build(new_source=("tensor_cast/new_mod.py",))
    exemptions = (_sample_exemption("tensor_cast/new_mod.py", "fn"),)
    result = gate_new_source(new_source_file.parent.parent, cs, {}, exemptions, ("tensor_cast/",))
    assert result.errors == ()


def test_gate_new_source_non_python_file_skipped(new_source_file: Path) -> None:
    yaml_file = new_source_file.parent / "data.yaml"
    yaml_file.write_text("key: value\n", encoding="utf-8")
    cs = ChangeSet.build(new_source=("tensor_cast/data.yaml",))
    result = gate_new_source(new_source_file.parent.parent, cs, {}, (), ("tensor_cast/",))
    assert result.errors == ()


# ---------------------------------------------------------------------------
# gate_deleted_source
# ---------------------------------------------------------------------------


def test_gate_deleted_source_selects_mapped_tests() -> None:
    cs = ChangeSet.build(del_source=("tensor_cast/old.py",))
    test_map = {
        "tensor_cast/old.py": {
            "fn": ["tests/regression/tensor_cast/test_a.py::test_x"],
        },
    }
    result = gate_deleted_source(cs, test_map, ("tensor_cast/",))
    assert "tests/regression/tensor_cast/test_a.py::test_x" in result.tests


def test_gate_deleted_source_no_map_entry_reports_error() -> None:
    cs = ChangeSet.build(del_source=("tensor_cast/old.py",))
    result = gate_deleted_source(cs, {}, ("tensor_cast/",))
    assert len(result.errors) == 1
    assert result.errors[0].category == "deleted_source"
    assert result.errors[0].path == "tensor_cast/old.py"


def test_gate_deleted_source_non_product_prefix_skipped() -> None:
    cs = ChangeSet.build(del_source=("tests/old.py",))
    result = gate_deleted_source(cs, {}, ("tensor_cast/",))
    assert result.errors == ()
    assert result.tests == frozenset()


# ---------------------------------------------------------------------------
# gate_deleted_tests
# ---------------------------------------------------------------------------


def test_gate_deleted_test_sole_coverage_reports_error() -> None:
    cs = ChangeSet.build(del_test=("tests/smoke/test_only.py::test_x",))
    test_map = {
        "cli/main.py": {
            "run": ["tests/smoke/test_only.py::test_x"],
        },
    }
    result = gate_deleted_tests(cs, test_map)
    assert len(result.errors) == 1
    assert result.errors[0].category == "deleted_test"
    assert result.errors[0].path == "tests/smoke/test_only.py::test_x"


def test_gate_deleted_test_not_sole_coverage_returns_no_errors() -> None:
    cs = ChangeSet.build(del_test=("tests/smoke/test_a.py::test_x",))
    test_map = {
        "cli/main.py": {
            "run": ["tests/smoke/test_a.py::test_x", "tests/smoke/test_b.py::test_y"],
        },
    }
    result = gate_deleted_tests(cs, test_map)
    assert result.errors == ()


# ---------------------------------------------------------------------------
# gate_modified_source
# ---------------------------------------------------------------------------


def test_gate_modified_source_mapped_symbol_selects_tests(tmp_path: Path) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n", encoding="utf-8")
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({2})})
    test_map = {"cli/main.py": {"run": ["test_a"]}}
    result = gate_modified_source(tmp_path, cs, test_map, (), ("cli/",))
    assert result.tests == frozenset({"test_a"})


def test_gate_modified_source_unmapped_symbol_reports_error(tmp_path: Path) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n", encoding="utf-8")
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({2})})
    result = gate_modified_source(tmp_path, cs, {}, (), ("cli/",))
    assert len(result.errors) == 1
    assert result.errors[0].category == "modified_source"
    assert result.errors[0].path == "cli/main.py"
    assert result.errors[0].symbol == "run"


def test_gate_modified_source_exempted_symbol_returns_no_errors(tmp_path: Path) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n", encoding="utf-8")
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({2})})
    exemptions = (_sample_exemption("cli/main.py", "run"),)
    result = gate_modified_source(tmp_path, cs, {}, exemptions, ("cli/",))
    assert result.errors == ()


def test_gate_modified_source_non_product_prefix_skipped(tmp_path: Path) -> None:
    src = tmp_path / "other" / "util.py"
    src.parent.mkdir(parents=True)
    src.write_text("def fn():\n    pass\n", encoding="utf-8")
    cs = ChangeSet.build(modified_source={"other/util.py": frozenset({1})})
    result = gate_modified_source(tmp_path, cs, {}, (), ("cli/",))
    assert result.errors == ()
    assert result.tests == frozenset()


# ---------------------------------------------------------------------------
# gate_new_source / gate_deleted_source — cross-layer deferred
# ---------------------------------------------------------------------------


def test_gate_deleted_source_defers_cross_layer_tests() -> None:
    cs = ChangeSet.build(del_source=("tensor_cast/ops.py",))
    test_map = {
        "tensor_cast/ops.py": {
            "add": [
                "tests/regression/tensor_cast/test_ops.py::test_add",
                "tests/regression/cli/test_cross.py::test_cross",
            ],
        },
    }
    result = gate_deleted_source(cs, test_map, ("tensor_cast/", "cli/"))
    assert "tests/regression/tensor_cast/test_ops.py::test_add" in result.tests
    assert "tests/regression/cli/test_cross.py::test_cross" in result.cross_layer_deferred


def test_gate_deleted_source_skips_concurrently_deleted_test_file() -> None:
    """del_test holds file paths; test_map holds full node ids — must match by prefix."""
    cs = ChangeSet.build(
        del_source=("tensor_cast/old.py",),
        del_test=("tests/regression/tensor_cast/test_old.py",),
    )
    test_map = {
        "tensor_cast/old.py": {
            "fn": ["tests/regression/tensor_cast/test_old.py::test_fn"],
        },
    }
    result = gate_deleted_source(cs, test_map, ("tensor_cast/",))
    assert result.tests == frozenset()
    assert result.errors == ()

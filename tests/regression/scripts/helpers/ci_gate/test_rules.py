"""Tests for ci_gate.rules — gate_* functions, _merge_step_results, _product_paths."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts.helpers.ci_gate.gate_policy import SourceExemption
from scripts.helpers.ci_gate.models import ChangeSet, GateError, GateStepResult
from scripts.helpers.ci_gate.rules import (
    _merge_step_results,
    _product_paths,
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
# _merge_step_results
# ---------------------------------------------------------------------------


def test_merge_combines_errors_from_all_steps() -> None:
    e1 = GateError(category="new_source", path="a.py")
    e2 = GateError(category="modified_source", path="b.py", symbol="fn")
    a = GateStepResult(errors=(e1,))
    b = GateStepResult(errors=(e2,))
    merged = _merge_step_results(a, b)
    assert merged.errors == (e1, e2)


def test_merge_unions_tests() -> None:
    a = GateStepResult(tests=frozenset({"t1"}))
    b = GateStepResult(tests=frozenset({"t2"}))
    merged = _merge_step_results(a, b)
    assert merged.tests == frozenset({"t1", "t2"})


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


def test_gate_new_tests_returns_all_new_and_modified_paths_even_when_exempted() -> None:
    cs = ChangeSet.build(
        new_test=("tests/smoke/test_a.py", "tests/regression/nightly/test_x.py"),
    )
    result = gate_new_tests(cs)
    assert result.tests == frozenset({"tests/smoke/test_a.py", "tests/regression/nightly/test_x.py"})


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
# gate_deleted_source
# ---------------------------------------------------------------------------


def test_gate_deleted_source_includes_all_guard_tests() -> None:
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
    assert result.tests == frozenset(
        {
            "tests/regression/tensor_cast/test_ops.py::test_add",
            "tests/regression/cli/test_cross.py::test_cross",
        }
    )


def test_gate_modified_source_coverage_omitted_path_returns_no_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "tensor_cast" / "builtin_model" / "foo.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n", encoding="utf-8")
    cs = ChangeSet.build(modified_source={"tensor_cast/builtin_model/foo.py": frozenset({2})})
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.is_coverage_omitted_source",
        lambda path, _roots: path.endswith("builtin_model/foo.py"),
    )
    result = gate_modified_source(
        tmp_path,
        cs,
        {},
        (),
        ("tensor_cast/",),
    )
    assert result.errors == ()


def test_gate_modified_source_coverage_fallback_skips_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n", encoding="utf-8")
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({2})})
    coverage_path = tmp_path / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.symbol_lines_covered_in_data",
        lambda *_args, **_kwargs: True,
    )

    result = gate_modified_source(
        tmp_path,
        cs,
        {},
        (),
        ("cli/",),
        coverage_path=coverage_path,
    )

    assert result.errors == ()


def test_gate_new_source_coverage_omitted_path_returns_no_errors(
    new_source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cs = ChangeSet.build(new_source=("tensor_cast/new_mod.py",))
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.is_coverage_omitted_source",
        lambda path, _roots: path == "tensor_cast/new_mod.py",
    )
    result = gate_new_source(
        new_source_file.parent.parent,
        cs,
        {},
        (),
        ("tensor_cast/",),
    )
    assert result.errors == ()


def test_gate_new_source_coverage_fallback_skips_block(
    new_source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cs = ChangeSet.build(new_source=("tensor_cast/new_mod.py",))
    coverage_path = new_source_file.parent.parent / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.symbol_lines_covered_in_data",
        lambda *_args, **_kwargs: True,
    )

    result = gate_new_source(
        new_source_file.parent.parent,
        cs,
        {},
        (),
        ("tensor_cast/",),
        coverage_path=coverage_path,
    )

    assert result.errors == ()


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

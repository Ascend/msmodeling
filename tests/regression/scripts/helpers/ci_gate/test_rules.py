"""Tests for ci_gate.rules — gate_* functions, _product_paths."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers.ci_gate.models import ChangeSet, SourceExemption
from scripts.helpers.ci_gate.rules import (
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


@pytest.fixture(autouse=True)
def _stub_collect_test_node_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_collect(targets: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return tuple(f"{path}::test_case" for path in targets)

    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.collect_all_test_node_ids",
        _fake_collect,
    )


def test_gate_new_tests_selects_collected_node_ids() -> None:
    cs = ChangeSet.build(new_test=("tests/smoke/test_a.py",))
    result = gate_new_tests(cs, (), full_suite=False)
    assert result.tests == frozenset({"tests/smoke/test_a.py::test_case"})


def test_gate_new_tests_also_selects_modified_test_nodes() -> None:
    cs = ChangeSet.build(
        new_test=("tests/smoke/test_a.py",),
        modified_test=("tests/regression/cli/test_b.py",),
    )
    result = gate_new_tests(cs, (), full_suite=False)
    assert result.tests == frozenset(
        {
            "tests/smoke/test_a.py::test_case",
            "tests/regression/cli/test_b.py::test_case",
        }
    )


def test_gate_new_tests_skips_when_full_suite() -> None:
    cs = ChangeSet.build(
        config=("pyproject.toml",),
        new_test=("tests/smoke/test_a.py",),
    )
    result = gate_new_tests(cs, (), full_suite=True)
    assert result.tests == frozenset()


def test_gate_new_tests_batch_collects_all_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def _fake_collect(targets: list[str]) -> tuple[str, ...]:
        calls.append(list(targets))
        return tuple(f"{path}::test_case" for path in targets)

    monkeypatch.setattr("scripts.helpers.ci_gate.rules.collect_all_test_node_ids", _fake_collect)
    cs = ChangeSet.build(
        new_test=("tests/smoke/test_a.py",),
        modified_test=("tests/regression/cli/test_b.py",),
    )
    result = gate_new_tests(cs, (), full_suite=False)
    assert calls == [["tests/smoke/test_a.py", "tests/regression/cli/test_b.py"]]
    assert result.tests == frozenset(
        {
            "tests/smoke/test_a.py::test_case",
            "tests/regression/cli/test_b.py::test_case",
        }
    )


def test_gate_new_tests_skips_file_when_all_nodes_exempt() -> None:
    from scripts.helpers.ci_gate.gate_policy import TestExemption

    cs = ChangeSet.build(new_test=("tests/smoke/test_a.py", "tests/regression/nightly/test_x.py"))
    exemptions = (
        TestExemption(
            test_id="tests/smoke/test_a.py::test_case",
            reason="x",
            applicant="a",
            approver="fangkai",
            deadline=date(2099, 12, 31),
        ),
        TestExemption(
            test_id="tests/regression/nightly/test_x.py::test_case",
            reason="x",
            applicant="a",
            approver="fangkai",
            deadline=date(2099, 12, 31),
        ),
    )
    result = gate_new_tests(cs, exemptions, full_suite=False)
    assert result.tests == frozenset()


# ---------------------------------------------------------------------------
# gate_new_source
# ---------------------------------------------------------------------------


def test_gate_new_source_with_test_map_entry_returns_no_errors(
    new_source_file: Path,
) -> None:
    cs = ChangeSet.build(new_source=("tensor_cast/new_mod.py",))
    test_map = {
        "tests/regression/tensor_cast/test_a.py::test_x": {
            "tensor_cast/new_mod.py": ["fn"],
        },
    }
    result = gate_new_source(new_source_file.parent.parent, cs, test_map, (), ("tensor_cast/",))
    assert result.errors == ()


def test_gate_new_source_partial_symbol_mapping_reports_missing_symbol(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    src = repo / "tensor_cast" / "multi.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n\n\ndef gamma():\n    return 3\n",
        encoding="utf-8",
    )
    cs = ChangeSet.build(new_source=("tensor_cast/multi.py",))
    test_map = {
        "tests/regression/tensor_cast/test_a.py::test_x": {
            "tensor_cast/multi.py": ["alpha", "beta"],
        },
    }
    result = gate_new_source(repo, cs, test_map, (), ("tensor_cast/",))
    assert len(result.errors) == 1
    assert result.errors[0].category == "new_source"
    assert result.errors[0].path == "tensor_cast/multi.py"
    assert result.errors[0].symbol == "gamma"


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


def test_gate_new_source_docstring_only_module_passes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_path = repo / "cli" / "__init__.py"
    init_path.parent.mkdir(parents=True)
    init_path.write_text('"""CLI package docstring only."""\n', encoding="utf-8")
    cs = ChangeSet.build(new_source=("cli/__init__.py",))
    result = gate_new_source(repo, cs, {}, (), ("cli/",))
    assert result.errors == ()


def test_gate_new_source_script_module_without_coverage_reports_file_error(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    main_path = repo / "optix" / "__main__.py"
    main_path.parent.mkdir(parents=True)
    main_path.write_text(
        '"""entry"""\n\nfrom optix import main\n\nif __name__ == "__main__":\n    main()\n',
        encoding="utf-8",
    )
    cs = ChangeSet.build(new_source=("optix/__main__.py",))
    result = gate_new_source(repo, cs, {}, (), ("optix/",))
    assert len(result.errors) == 1
    assert result.errors[0].category == "new_source"
    assert result.errors[0].path == "optix/__main__.py"
    assert result.errors[0].symbol is None


def test_gate_new_source_script_module_coverage_fallback_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    main_path = repo / "optix" / "__main__.py"
    main_path.parent.mkdir(parents=True)
    main_path.write_text(
        '"""entry"""\n\nfrom optix import main\n\nif __name__ == "__main__":\n    main()\n',
        encoding="utf-8",
    )
    coverage_path = repo / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.symbol_lines_covered_in_data",
        lambda *_args, **_kwargs: True,
    )
    cs = ChangeSet.build(new_source=("optix/__main__.py",))
    result = gate_new_source(
        repo,
        cs,
        {},
        (),
        ("optix/",),
        coverage_path=coverage_path,
    )
    assert result.errors == ()


# ---------------------------------------------------------------------------
# gate_deleted_source
# ---------------------------------------------------------------------------


def test_gate_deleted_source_selects_mapped_tests() -> None:
    cs = ChangeSet.build(del_source=("tensor_cast/old.py",))
    test_map = {
        "tests/regression/tensor_cast/test_a.py::test_x": {
            "tensor_cast/old.py": ["fn"],
        },
    }
    result = gate_deleted_source(cs, test_map, ("tensor_cast/",))
    assert "tests/regression/tensor_cast/test_a.py::test_x" in result.tests


def test_gate_deleted_source_without_watchers_returns_no_errors() -> None:
    cs = ChangeSet.build(del_source=("tensor_cast/old.py",))
    result = gate_deleted_source(cs, {}, ("tensor_cast/",))
    assert result.errors == ()
    assert result.tests == frozenset()


def test_gate_deleted_source_non_product_prefix_skipped() -> None:
    cs = ChangeSet.build(del_source=("tests/old.py",))
    result = gate_deleted_source(cs, {}, ("tensor_cast/",))
    assert result.errors == ()
    assert result.tests == frozenset()


# ---------------------------------------------------------------------------
# gate_deleted_tests
# ---------------------------------------------------------------------------


def test_gate_deleted_test_sole_coverage_reports_error() -> None:
    cs = ChangeSet.build(del_test=("tests/smoke/test_only.py",))
    test_map = {
        "tests/smoke/test_only.py::test_x": {
            "cli/main.py": ["run"],
        },
    }
    result = gate_deleted_tests(cs, test_map)
    assert len(result.errors) == 1
    assert result.errors[0].category == "deleted_test"
    assert result.errors[0].path == "tests/smoke/test_only.py"


def test_gate_deleted_test_not_sole_coverage_returns_no_errors() -> None:
    cs = ChangeSet.build(del_test=("tests/smoke/test_a.py",))
    test_map = {
        "tests/smoke/test_a.py::test_x": {
            "cli/main.py": ["run"],
        },
        "tests/smoke/test_b.py::test_y": {
            "cli/main.py": ["run"],
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
    test_map = {"tests/regression/cli/test_run.py::test_run": {"cli/main.py": ["run"]}}
    result = gate_modified_source(tmp_path, cs, test_map, (), ("cli/",))
    assert result.tests == frozenset({"tests/regression/cli/test_run.py::test_run"})


def test_gate_modified_source_canonical_map_matches_gate_symbol(tmp_path: Path) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "class Widget:\n    def run(self):\n        x = 1\n",
        encoding="utf-8",
    )
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({3})})
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run"],
        },
    }
    result = gate_modified_source(tmp_path, cs, test_map, (), ("cli/",))
    assert result.tests == frozenset({"tests/regression/cli/test_run.py::test_run"})
    assert result.errors == ()


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
        "tests/regression/tensor_cast/test_ops.py::test_add": {
            "tensor_cast/ops.py": ["add"],
        },
        "tests/regression/cli/test_cross.py::test_cross": {
            "tensor_cast/ops.py": ["add"],
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


def test_gate_modified_source_body_schedules_source_watchers(tmp_path: Path) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "class Widget:\n    x = 1\n\n    def run(self):\n        pass\n",
        encoding="utf-8",
    )
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({2})})
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run"],
        },
    }
    result = gate_modified_source(tmp_path, cs, test_map, (), ("cli/",), check_mapping=False)
    assert result.tests == frozenset({"tests/regression/cli/test_run.py::test_run"})
    assert result.errors == ()


def test_gate_modified_source_body_relaxed_coverage_skips_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "class Widget:\n    x = 1\n\n    def run(self):\n        pass\n",
        encoding="utf-8",
    )
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({2})})
    coverage_path = tmp_path / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    class _FakeCoverageData:
        def read(self) -> None:
            return None

        def measured_files(self) -> list[str]:
            return [str(src.resolve())]

        def contexts_by_lineno(self, _path: str) -> dict[int, list[str]]:
            return {2: [""]}

    monkeypatch.setattr(
        "coverage.data.CoverageData",
        lambda _path: _FakeCoverageData(),
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


def test_gate_new_source_named_symbol_strict_coverage_passes(
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


def test_gate_new_source_named_symbol_pytest_coverage_context_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    src = repo / "tensor_cast" / "new_mod.py"
    src.parent.mkdir(parents=True)
    src.write_text("def fn():\n    pass\n", encoding="utf-8")
    coverage_path = repo / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    class _FakeCoverageData:
        def read(self) -> None:
            return None

        def measured_files(self) -> list[str]:
            return [str(src.resolve())]

        def contexts_by_lineno(self, _path: str) -> dict[int, list[str]]:
            return {1: ["tests/regression/tensor_cast/test_a.py::test_x"]}

    monkeypatch.setattr(
        "coverage.data.CoverageData",
        lambda _path: _FakeCoverageData(),
    )

    cs = ChangeSet.build(new_source=("tensor_cast/new_mod.py",))
    result = gate_new_source(
        repo,
        cs,
        {},
        (),
        ("tensor_cast/",),
        coverage_path=coverage_path,
    )

    assert result.errors == ()


def test_gate_new_source_body_relaxed_coverage_skips_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    main_path = repo / "optix" / "__main__.py"
    main_path.parent.mkdir(parents=True)
    main_path.write_text(
        '"""entry"""\n\nfrom optix import main\n\nif __name__ == "__main__":\n    main()\n',
        encoding="utf-8",
    )
    coverage_path = repo / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    class _FakeCoverageData:
        def read(self) -> None:
            return None

        def measured_files(self) -> list[str]:
            return [str(main_path.resolve())]

        def contexts_by_lineno(self, _path: str) -> dict[int, list[str]]:
            return {6: [""]}

    monkeypatch.setattr(
        "coverage.data.CoverageData",
        lambda _path: _FakeCoverageData(),
    )

    cs = ChangeSet.build(new_source=("optix/__main__.py",))
    result = gate_new_source(
        repo,
        cs,
        {},
        (),
        ("optix/",),
        coverage_path=coverage_path,
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


def test_gate_deleted_source_skips_concurrently_deleted_test_file() -> None:
    """del_test holds file paths; test_map holds full node ids — must match by prefix."""
    cs = ChangeSet.build(
        del_source=("tensor_cast/old.py",),
        del_test=("tests/regression/tensor_cast/test_old.py",),
    )
    test_map = {
        "tests/regression/tensor_cast/test_old.py::test_fn": {
            "tensor_cast/old.py": ["fn"],
        },
    }
    result = gate_deleted_source(cs, test_map, ("tensor_cast/",))
    assert result.tests == frozenset()
    assert result.errors == ()


def test_gate_deleted_source_blocks_sole_coverage_mapping() -> None:
    cs = ChangeSet.build(del_source=("tensor_cast/old.py",))
    test_map = {
        "tests/regression/tensor_cast/test_a.py::test_x": {
            "tensor_cast/old.py": ["fn"],
        },
    }
    result = gate_deleted_source(cs, test_map, ("tensor_cast/",))
    assert result.errors
    assert result.errors[0].category == "deleted_source"
    assert "tensor_cast/old.py::fn" in result.errors[0].detail


def test_gate_modified_source_decorator_line_uses_mangled_symbol(
    tmp_path: Path,
) -> None:
    src = tmp_path / "tensor_cast" / "ops.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "\n".join(
            [
                "def deco(fn):",
                "    return fn",
                "",
                "@deco",
                "def run():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    cs = ChangeSet.build(modified_source={"tensor_cast/ops.py": frozenset({4})})
    result = gate_modified_source(tmp_path, cs, {}, (), ("tensor_cast/",))
    assert len(result.errors) == 1
    assert result.errors[0].symbol == "run@deco"


def test_gate_modified_source_signature_proxy_accepts_body_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({1})})
    coverage_path = tmp_path / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    captured: list[set[int]] = []

    def _fake_symbol_lines_covered(
        _repo: Path,
        _path: str,
        _symbol: str,
        lines: set[int],
        *_args: object,
        **_kwargs: object,
    ) -> bool:
        captured.append(set(lines))
        return True

    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.symbol_lines_covered_in_data",
        _fake_symbol_lines_covered,
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
    assert captured
    assert 2 in captured[0]
    assert 1 not in captured[0]


def test_gate_modified_source_decorator_only_requires_import_and_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ast

    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "\n".join(
            [
                "def deco(fn):",
                "    return fn",
                "",
                "@deco",
                "def run():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    deco_line = fn.decorator_list[0].lineno
    body_line = fn.body[0].lineno
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({deco_line})})
    coverage_path = tmp_path / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    captured: list[tuple[str, set[int]]] = []

    def _fake_symbol_lines_covered(
        _repo: Path,
        _path: str,
        symbol: str,
        lines: set[int],
        *_args: object,
        **_kwargs: object,
    ) -> bool:
        captured.append((symbol, set(lines)))
        return True

    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.symbol_lines_covered_in_data",
        _fake_symbol_lines_covered,
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
    assert ("%", {deco_line}) in captured
    assert ("run@deco", {body_line}) in captured


def test_gate_modified_source_body_only_uses_strict_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    return 1\n", encoding="utf-8")
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({2})})
    coverage_path = tmp_path / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    captured: list[tuple[str, set[int]]] = []

    def _fake_symbol_lines_covered(
        _repo: Path,
        _path: str,
        symbol: str,
        lines: set[int],
        *_args: object,
        **_kwargs: object,
    ) -> bool:
        captured.append((symbol, set(lines)))
        return True

    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.symbol_lines_covered_in_data",
        _fake_symbol_lines_covered,
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
    assert captured == [("run", {2})]


def test_gate_modified_source_decorator_and_def_header_requires_import_and_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ast

    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "\n".join(
            [
                "def deco(fn):",
                "    return fn",
                "",
                "@deco",
                "def run():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    deco_line = fn.decorator_list[0].lineno
    body_line = fn.body[0].lineno
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({deco_line, fn.lineno})})
    coverage_path = tmp_path / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    captured: list[tuple[str, set[int]]] = []

    def _fake_symbol_lines_covered(
        _repo: Path,
        _path: str,
        symbol: str,
        lines: set[int],
        *_args: object,
        **_kwargs: object,
    ) -> bool:
        captured.append((symbol, set(lines)))
        return True

    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.symbol_lines_covered_in_data",
        _fake_symbol_lines_covered,
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
    assert ("%", {deco_line}) in captured
    assert ("run@deco", {body_line}) in captured


def test_gate_modified_source_class_method_decorator_import_uses_class_percent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ast

    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "\n".join(
            [
                "class Foo:",
                "    @staticmethod",
                "    def run():",
                "        return 1",
            ]
        ),
        encoding="utf-8",
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    deco_line = method.decorator_list[0].lineno
    body_line = method.body[0].lineno
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({deco_line})})
    coverage_path = tmp_path / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    captured: list[tuple[str, set[int]]] = []

    def _fake_symbol_lines_covered(
        _repo: Path,
        _path: str,
        symbol: str,
        lines: set[int],
        *_args: object,
        **_kwargs: object,
    ) -> bool:
        captured.append((symbol, set(lines)))
        return True

    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.symbol_lines_covered_in_data",
        _fake_symbol_lines_covered,
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
    assert ("Foo::%", {deco_line}) in captured
    assert ("Foo::run@staticmethod", {body_line}) in captured

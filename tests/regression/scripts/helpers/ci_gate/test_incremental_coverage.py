# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for changed executable-line coverage."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from scripts.helpers.ci_gate.incremental_coverage import measure_incremental_coverage
from scripts.helpers.ci_gate.models import ChangeSet, SourceExemption


class _FakeCoverageData:
    def __init__(self, source: Path, ctxmap: dict[int, list[str]]) -> None:
        self._source = str(source.resolve())
        self._ctxmap = ctxmap

    def read(self) -> None:
        return None

    def measured_files(self) -> list[str]:
        return [self._source]

    def contexts_by_lineno(self, _filename: str) -> dict[int, list[str]]:
        return self._ctxmap


def _write_source(repo: Path) -> Path:
    source = repo / "cli" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def main():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n    return a + b + c + d\n", encoding="utf-8"
    )
    return source


def test_new_source_exactly_eighty_percent_is_measured(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    data = _FakeCoverageData(
        source,
        {line: ["tests/regression/cli/test_main.py::test_main"] for line in range(2, 6)},
    )

    result = measure_incremental_coverage(
        tmp_path,
        ChangeSet.build(new_source=("cli/main.py",)),
        ("cli/",),
        (),
        data,
    )

    assert result.total == 5
    assert result.covered == 4
    assert result.percent == 80.0
    assert result.missing == ("cli/main.py:6",)


def test_only_pytest_context_counts(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    data = _FakeCoverageData(source, {2: [""], 3: ["conftest"], 4: ["tests/a.py::test_a"]})

    result = measure_incremental_coverage(
        tmp_path,
        ChangeSet.build(modified_source={"cli/main.py": frozenset({2, 3, 4})}),
        ("cli/",),
        (),
        data,
    )

    assert result.covered == 1
    assert result.total == 3
    assert result.missing == ("cli/main.py:2", "cli/main.py:3")


def test_exception_branches_stay_in_denominator(tmp_path: Path) -> None:
    source = tmp_path / "cli" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def main(value):\n"
        "    try:\n"
        "        parsed = int(value)\n"
        "    except ValueError:\n"
        "        return None\n"
        "    return parsed\n",
        encoding="utf-8",
    )
    data = _FakeCoverageData(source, {3: ["tests/regression/cli/test_main.py::test_main"]})

    result = measure_incremental_coverage(
        tmp_path,
        ChangeSet.build(modified_source={"cli/main.py": frozenset({3, 4, 5})}),
        ("cli/",),
        (),
        data,
    )

    assert result.total == 3
    assert result.covered == 1
    assert result.missing == ("cli/main.py:4", "cli/main.py:5")


def test_non_executable_changed_lines_have_zero_denominator(tmp_path: Path) -> None:
    source = tmp_path / "cli" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text("# changed comment\n\n", encoding="utf-8")
    data = _FakeCoverageData(source, {})

    result = measure_incremental_coverage(
        tmp_path,
        ChangeSet.build(modified_source={"cli/main.py": frozenset({1, 2})}),
        ("cli/",),
        (),
        data,
    )

    assert result.total == 0
    assert result.percent == 100.0


def test_exempt_symbol_is_removed_from_denominator(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    exemption = SourceExemption(
        file="cli/main.py",
        symbol="main",
        reason="approved function exemption",
        applicant="alice",
        approver="bob",
        deadline=date(2099, 1, 1),
    )

    result = measure_incremental_coverage(
        tmp_path,
        ChangeSet.build(new_source=("cli/main.py",)),
        ("cli/",),
        (exemption,),
        _FakeCoverageData(source, {}),
    )

    assert result.total == 0


def test_import_time_lines_are_removed_from_new_source_denominator(tmp_path: Path) -> None:
    source = tmp_path / "cli" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import os\n\n@staticmethod\ndef current_dir():\n    return os.getcwd()\n",
        encoding="utf-8",
    )
    data = _FakeCoverageData(
        source,
        {
            1: [""],
            3: [""],
            4: [""],
            5: ["tests/cli/test_main.py::test_current_dir"],
        },
    )

    result = measure_incremental_coverage(
        tmp_path,
        ChangeSet.build(new_source=("cli/main.py",)),
        ("cli/",),
        (),
        data,
    )

    assert result.total == 1
    assert result.covered == 1
    assert result.percent == 100.0

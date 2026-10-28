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

"""Tests for ci_gate.test_map_query."""

from __future__ import annotations

from pathlib import Path

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.gate_policy import load_gate_policy
from scripts.helpers.ci_gate.test_map_query import (
    build_test_map_index,
    classify_touched_paths,
    has_unmapped_existing_source,
    is_source_file_mapped,
    is_symbol_mapped,
    prune_deleted_sources,
    select_resync_targets,
    source_watchers,
    symbol_watchers,
    symbols_mapped_for_source,
)


def test_symbol_watchers_matches_canonical_symbol() -> None:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run"],
        },
    }
    assert symbol_watchers(test_map, "cli/main.py", "Widget::run") == frozenset(
        {"tests/regression/cli/test_run.py::test_run"}
    )


def test_is_symbol_mapped_false_for_unrelated_symbol() -> None:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run"],
        },
    }
    assert is_symbol_mapped(test_map, "cli/main.py", "Widget::stop") is False


def test_is_source_file_mapped_true_when_any_symbol_present() -> None:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run"],
        },
    }
    assert is_source_file_mapped(test_map, "cli/main.py") is True
    assert is_source_file_mapped(test_map, "tensor_cast/ops.py") is False


def test_symbols_mapped_for_source_collects_unique_symbols() -> None:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run", "Widget::stop"],
        },
        "tests/regression/cli/test_other.py::test_x": {
            "cli/main.py": ["Widget::run"],
            "tensor_cast/ops.py": ["add"],
        },
    }
    assert symbols_mapped_for_source(test_map, "cli/main.py") == frozenset({"Widget::run", "Widget::stop"})
    assert symbols_mapped_for_source(test_map, "missing.py") == frozenset()


def test_build_test_map_index_matches_scan_helpers() -> None:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run", "Widget::stop"],
            "tensor_cast/ops.py": ["add"],
        },
        "tests/regression/tensor_cast/test_ops.py::test_add": {
            "tensor_cast/ops.py": ["add"],
        },
    }
    index = build_test_map_index(test_map)
    assert symbol_watchers(test_map, "cli/main.py", "Widget::run", index=index) == symbol_watchers(
        test_map,
        "cli/main.py",
        "Widget::run",
    )
    assert source_watchers(test_map, "tensor_cast/ops.py", index=index) == source_watchers(
        test_map,
        "tensor_cast/ops.py",
    )


def test_prune_deleted_sources_drops_deleted_paths() -> None:
    test_map = {
        "tests/a.py::test_x": {
            "keep.py": ["fn"],
            "drop.py": ["fn"],
        },
    }
    pruned = prune_deleted_sources(test_map, ("drop.py",))
    assert pruned == {"tests/a.py::test_x": {"keep.py": ["fn"]}}


def test_classify_touched_paths_uses_gate_policy() -> None:
    touched = classify_touched_paths(
        ("tests/smoke/test_a.py", "cli/main.py", "uv.lock", "README.md"),
        load_gate_policy(REPO_ROOT),
    )
    assert touched.tests == frozenset({"tests/smoke/test_a.py"})
    assert touched.sources == frozenset({"cli/main.py"})
    assert touched.configs == frozenset({"uv.lock"})


def test_select_resync_targets_combines_test_files_and_source_watchers(tmp_path: Path) -> None:
    test_file = "tests/smoke/test_a.py"
    source_file = "cli/main.py"
    for relative in (test_file, source_file):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {source_file: ["run"]},
    }
    touched = classify_touched_paths((test_file, source_file), load_gate_policy(REPO_ROOT))
    assert select_resync_targets(test_map, touched, repo_root=tmp_path) == (
        "tests/regression/cli/test_run.py::test_run",
        test_file,
    )


def test_unmapped_existing_source_requires_full_rebuild(tmp_path: Path) -> None:
    source_file = "cli/new.py"
    path = tmp_path / source_file
    path.parent.mkdir(parents=True)
    path.touch()
    touched = classify_touched_paths((source_file,), load_gate_policy(REPO_ROOT))
    assert has_unmapped_existing_source({}, touched, repo_root=tmp_path) is True
    path.unlink()
    assert has_unmapped_existing_source({}, touched, repo_root=tmp_path) is False

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

"""Changed executable-line coverage for the incremental CI gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.helpers.ci_gate.models import ChangeSet, SourceExemption
from scripts.helpers.ci_gate.policy import is_exempt
from scripts.helpers.common import ast_utils
from scripts.helpers.common.coverage_omit import is_coverage_omitted_source
from scripts.helpers.common.coverage_symbol_check import (
    CoverageDataReader,
    covered_test_context_lines_in_data,
)


@dataclass(frozen=True, slots=True)
class IncrementalCoverage:
    covered: int
    total: int
    missing: tuple[str, ...]

    @property
    def percent(self) -> float:
        return 100.0 if self.total == 0 else self.covered * 100.0 / self.total


def _all_lines(path: Path) -> set[int]:
    try:
        count = len(path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return set()
    return set(range(1, count + 1))


def _is_exempt_line(
    path: Path,
    rel_path: str,
    line_no: int,
    exemptions: tuple[SourceExemption, ...],
) -> bool:
    symbols = ast_utils.canonical_symbols_for_lines(path, {line_no})
    return bool(symbols) and all(is_exempt(exemptions, rel_path, symbol) for symbol in symbols)


def _candidate_lines(
    repo_root: Path,
    changes: ChangeSet,
    roots: tuple[str, ...],
    exemptions: tuple[SourceExemption, ...],
) -> dict[str, frozenset[int]]:
    raw_by_path = {path: set(lines) for path, lines in changes.modified_source}
    for path in changes.new_source:
        raw_by_path[path] = _all_lines(repo_root / path)

    candidates: dict[str, frozenset[int]] = {}
    for rel_path, raw_lines in raw_by_path.items():
        if not any(rel_path.startswith(root) for root in roots):
            continue
        if is_coverage_omitted_source(rel_path, roots):
            continue
        source_path = repo_root / rel_path
        if not source_path.is_file():
            continue
        executable = ast_utils.filter_test_context_executable_lines(source_path, raw_lines)
        gated = {line_no for line_no in executable if not _is_exempt_line(source_path, rel_path, line_no, exemptions)}
        if gated:
            candidates[rel_path] = frozenset(gated)
    return candidates


def measure_incremental_coverage(
    repo_root: Path,
    changes: ChangeSet,
    roots: tuple[str, ...],
    exemptions: tuple[SourceExemption, ...],
    coverage_data: CoverageDataReader | None,
) -> IncrementalCoverage:
    """Measure pytest-context coverage over changed executable product lines."""
    covered = 0
    total = 0
    missing: list[str] = []
    for rel_path, lines in sorted(_candidate_lines(repo_root, changes, roots, exemptions).items()):
        hit = (
            covered_test_context_lines_in_data(repo_root, rel_path, lines, coverage_data)
            if coverage_data is not None
            else frozenset()
        )
        covered += len(hit)
        total += len(lines)
        missing.extend(f"{rel_path}:{line_no}" for line_no in sorted(lines - hit))
    return IncrementalCoverage(covered=covered, total=total, missing=tuple(missing))

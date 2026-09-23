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

"""Detect gate_policy test-exemption entries broken by deleted or renamed paths.

Source exemptions are validated at policy load (``validate_source_exemption_symbol``).
Do not re-check missing/renamed source paths here — that is redundant with load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.helpers.ci_gate.models import ChangeSet, GateError

if TYPE_CHECKING:
    from scripts.helpers.ci_gate.diff import DiffEntry
    from scripts.helpers.ci_gate.models import CiGatePolicy


def iter_rename_pairs(entries: tuple[DiffEntry, ...]) -> tuple[tuple[str, str], ...]:
    """Return ``(old_path, new_path)`` for each rename entry in a git diff."""
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        if not entry.status.startswith("R"):
            continue
        if entry.old_path is None or entry.new_path is None:
            continue
        pairs.append((entry.old_path, entry.new_path))
    return tuple(pairs)


def gate_exemption_drift(
    policy: CiGatePolicy,
    changes: ChangeSet,
    rename_pairs: tuple[tuple[str, str], ...],
) -> tuple[GateError, ...]:
    """Return blocking errors when *test* exemptions reference deleted or renamed paths."""
    errors: list[GateError] = []
    deleted_tests = set(changes.del_test)
    rename_by_old = dict(rename_pairs)

    for entry in policy.test_exemptions:
        test_file = entry.test_id.split("::", 1)[0]
        if test_file in rename_by_old:
            new_file = rename_by_old[test_file]
            new_test_id = f"{new_file}{entry.test_id[len(test_file) :]}"
            errors.append(
                GateError(
                    category="exemption_drift",
                    path=test_file,
                    detail=(f"exemption {entry.test_id!r} references renamed test file; update to {new_test_id!r}"),
                )
            )
        elif test_file in deleted_tests:
            errors.append(
                GateError(
                    category="exemption_drift",
                    path=test_file,
                    detail=f"exemption {entry.test_id!r} references deleted test file",
                )
            )

    return tuple(errors)

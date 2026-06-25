"""Detect gate_policy exemption entries broken by deleted or renamed paths."""

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
    """Return blocking errors when exemptions reference deleted or renamed paths."""
    errors: list[GateError] = []
    deleted_sources = set(changes.del_source)
    deleted_tests = set(changes.del_test)
    rename_by_old = dict(rename_pairs)

    for entry in policy.source_exemptions:
        if entry.file in rename_by_old:
            new_path = rename_by_old[entry.file]
            errors.append(
                GateError(
                    category="exemption_drift",
                    path=entry.file,
                    symbol=entry.symbol,
                    detail=(
                        f"exemption {entry.symbol_key} references renamed source; update to {new_path}::{entry.symbol}"
                    ),
                )
            )
        elif entry.file in deleted_sources:
            errors.append(
                GateError(
                    category="exemption_drift",
                    path=entry.file,
                    symbol=entry.symbol,
                    detail=f"exemption {entry.symbol_key} references deleted source file",
                )
            )

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

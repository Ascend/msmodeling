"""Classify git diff entries using gate_policy.yaml path policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.diff import DiffEntry, GitDiffResult
from scripts.helpers.ci_gate.models import ChangeSet, CiGatePolicy
from scripts.helpers.ci_gate.policy import is_config_path, is_source_path, is_test_path

if TYPE_CHECKING:
    from collections.abc import Callable


def expand_rename_entries(
    entries: tuple[DiffEntry, ...],
    line_map: dict[str, set[int]],
) -> tuple[DiffEntry, ...]:
    """Treat git renames as delete + add; partial renames also modify the new path."""
    expanded: list[DiffEntry] = []
    for entry in entries:
        if not entry.status.startswith("R"):
            expanded.append(entry)
            continue
        old_path = entry.old_path
        new_path = entry.new_path
        if old_path is None or new_path is None:
            continue
        score = int(entry.status[1:]) if entry.status[1:].isdigit() else 100
        expanded.append(DiffEntry(status="D", old_path=old_path, new_path=None))
        expanded.append(DiffEntry(status="A", old_path=None, new_path=new_path))
        if score < 100:
            lines = line_map.get(new_path)
            if lines:
                expanded.append(DiffEntry(status="M", old_path=new_path, new_path=new_path))
    return tuple(expanded)


def _classify_test_path(
    status: str,
    filepath: str,
    new_test: list[str],
    del_test: list[str],
    modified_test: list[str],
) -> None:
    if status == "A":
        new_test.append(filepath)
    elif status == "D":
        del_test.append(filepath)
    elif status in ("M", "C"):
        modified_test.append(filepath)


def _classify_source_path(
    status: str,
    filepath: str,
    *,
    line_map: dict[str, set[int]],
    new_source: list[str],
    del_source: list[str],
    modified_source: dict[str, frozenset[int]],
) -> None:
    if status == "A":
        new_source.append(filepath)
    elif status == "D":
        del_source.append(filepath)
    elif status in ("M", "C"):
        modified_source[filepath] = frozenset(line_map.get(filepath, set()))


def _classify_py_path(
    status: str,
    filepath: str,
    *,
    line_map: dict[str, set[int]],
    policy: CiGatePolicy,
    new_test: list[str],
    del_test: list[str],
    modified_test: list[str],
    new_source: list[str],
    del_source: list[str],
    modified_source: dict[str, frozenset[int]],
    track_unscoped: Callable[[str], None],
) -> None:
    if is_test_path(filepath, policy):
        _classify_test_path(status, filepath, new_test, del_test, modified_test)
        return
    if is_config_path(filepath, policy):
        return
    if is_source_path(filepath, policy):
        _classify_source_path(
            status,
            filepath,
            line_map=line_map,
            new_source=new_source,
            del_source=del_source,
            modified_source=modified_source,
        )
        return
    track_unscoped(filepath)


def classify_changes(
    diff: GitDiffResult,
    policy: CiGatePolicy,
) -> ChangeSet:
    """Return a ChangeSet from parsed git diff entries and gate_policy.yaml scopes."""
    line_map = diff.line_map
    config: list[str] = []
    new_test: list[str] = []
    del_test: list[str] = []
    modified_test: list[str] = []
    new_source: list[str] = []
    del_source: list[str] = []
    modified_source: dict[str, frozenset[int]] = {}
    unscoped_python: list[str] = []

    def _track_unscoped(filepath: str) -> None:
        unscoped_python.append(filepath)

    for entry in expand_rename_entries(diff.entries, line_map):
        paths = _entry_paths(entry)
        config.extend(path for path in paths if is_config_path(path, policy))

        candidate_path = entry.new_path if entry.new_path is not None else entry.old_path
        if candidate_path is None or not candidate_path.endswith(".py"):
            continue
        _classify_py_path(
            entry.status,
            candidate_path,
            line_map=line_map,
            policy=policy,
            new_test=new_test,
            del_test=del_test,
            modified_test=modified_test,
            new_source=new_source,
            del_source=del_source,
            modified_source=modified_source,
            track_unscoped=_track_unscoped,
        )

    return ChangeSet.build(
        config=tuple(config),
        new_test=tuple(new_test),
        del_test=tuple(del_test),
        modified_test=tuple(modified_test),
        new_source=tuple(new_source),
        del_source=tuple(del_source),
        modified_source=modified_source,
        unscoped_python=tuple(sorted(set(unscoped_python))),
    )


def _entry_paths(entry: DiffEntry) -> tuple[str, ...]:
    if entry.status == "D":
        if entry.old_path is None:
            raise ConfigError("DiffEntry with status 'D' must have old_path set")
        return (entry.old_path,)
    if entry.new_path is None:
        raise ConfigError(f"DiffEntry with status {entry.status!r} must have new_path set")
    return (entry.new_path,)

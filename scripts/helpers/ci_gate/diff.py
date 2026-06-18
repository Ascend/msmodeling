"""Git diff analysis: base ref resolution, line mapping, change classification."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.gate_policy import TestDiscovery, default_test_discovery, is_gate_test_path
from scripts.helpers.ci_gate.models import ChangeSet
from scripts.helpers.common.coverage_config import product_roots
from scripts.helpers.common.test_map_config import is_config_path, is_full_suite_trigger_path, is_gate_ignored_path
from scripts.helpers.common.test_map_loader import is_product_source

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_git_path = shutil.which("git")
if _git_path is None:
    raise RuntimeError("git not found")
_GIT: str = _git_path

logger = logging.getLogger(__name__)

_HUNK_HEADER_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_GIT, *args],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )


def _parse_fetch_remote_branch(ref: str) -> tuple[str, str]:
    """Split *ref* into ``(remote, branch)`` for ``git fetch remote branch``.

    MR CI runs on the feature branch (e.g. ``mr255``); *ref* is
    ``MSMODELING_TEST_BASE_BRANCH`` (e.g. ``master`` or ``origin/master``).
    """
    if "/" in ref:
        remote, branch = ref.split("/", 1)
        return remote, branch
    return "origin", ref


def _fetch_deepen(repo_root: Path, ref: str) -> None:
    remote, branch = _parse_fetch_remote_branch(ref)
    logger.info("Deepening shallow clone with git fetch --depth=50 %s %s", remote, branch)
    proc = _run_git(repo_root, "fetch", "--depth=50", remote, branch)
    if proc.returncode != 0:
        logger.warning("git fetch failed for %s (%s %s): %s", ref, remote, branch, proc.stderr.strip())


def resolve_head_commit(repo_root: Path) -> str:
    """Return full SHA for HEAD."""
    proc = _run_git(repo_root, "rev-parse", "HEAD")
    if proc.returncode != 0 or not proc.stdout.strip():
        raise ConfigError(f"Cannot resolve HEAD commit: {proc.stderr.strip()}")
    return proc.stdout.strip()


def resolve_base_ref(repo_root: Path, branch: str) -> str:
    """Resolve merge-base between HEAD and *branch* (``MSMODELING_TEST_BASE_BRANCH``).

    CI runs on the MR branch; *branch* is the comparison target (e.g. ``master``,
    ``develop``, or ``origin/master``). If *branch* contains ``/``, it is used
    as-is (e.g. ``center/develop``). Otherwise tries bare *branch* first, then
    ``origin/<branch>``.
    """
    refs = [branch] if "/" in branch else [branch, f"origin/{branch}"]

    last_stderr = ""
    for ref in refs:
        proc = _run_git(repo_root, "merge-base", "HEAD", ref)
        if proc.returncode == 0 and proc.stdout.strip():
            logger.info("Resolved base ref using %s", ref)
            return proc.stdout.strip()
        last_stderr = proc.stderr.strip()

    for ref in refs:
        _fetch_deepen(repo_root, ref)
        proc = _run_git(repo_root, "merge-base", "HEAD", ref)
        if proc.returncode == 0 and proc.stdout.strip():
            logger.info("Resolved base ref using %s after deepen fetch", ref)
            return proc.stdout.strip()
        last_stderr = proc.stderr.strip()

    raise ConfigError(
        f"Cannot resolve base ref between HEAD and {refs[0]!r}."
        + (f" Also tried {refs[1]!r}: not found either" if len(refs) > 1 else "")
        + (f" Last error: {last_stderr}" if last_stderr else "")
    )


# ---------------------------------------------------------------------------
# Unified diff parse
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One file-level change from ``git diff --unified=0 -M --diff-filter=ACDMR``."""

    status: str
    old_path: str | None
    new_path: str | None


@dataclass(frozen=True, slots=True)
class GitDiffResult:
    line_map: dict[str, set[int]]
    entries: tuple[DiffEntry, ...]


def _flush_diff_entry(
    *,
    old_path: str | None,
    new_path: str | None,
    is_new: bool,
    is_deleted: bool,
    is_rename: bool,
    rename_similarity: int,
    is_copy: bool,
) -> DiffEntry | None:
    if old_path is None and new_path is None:
        return None
    if is_rename or (old_path and new_path and old_path != new_path):
        return DiffEntry(status=f"R{rename_similarity}", old_path=old_path, new_path=new_path)
    if is_copy:
        return DiffEntry(status="C", old_path=old_path, new_path=new_path)
    if is_new:
        return DiffEntry(status="A", old_path=None, new_path=new_path)
    if is_deleted:
        return DiffEntry(status="D", old_path=old_path, new_path=None)
    return DiffEntry(status="M", old_path=old_path, new_path=new_path)


@dataclass
class _DiffParseState:
    line_map: dict[str, set[int]]
    entries: list[DiffEntry]
    current_file: str | None = None
    old_path: str | None = None
    new_path: str | None = None
    is_new: bool = False
    is_deleted: bool = False
    is_rename: bool = False
    is_copy: bool = False
    rename_similarity: int = 100

    def commit_entry(self) -> None:
        entry = _flush_diff_entry(
            old_path=self.old_path,
            new_path=self.new_path,
            is_new=self.is_new,
            is_deleted=self.is_deleted,
            is_rename=self.is_rename,
            rename_similarity=self.rename_similarity,
            is_copy=self.is_copy,
        )
        if entry is not None:
            self.entries.append(entry)
        self.old_path = None
        self.new_path = None
        self.is_new = False
        self.is_deleted = False
        self.is_rename = False
        self.is_copy = False
        self.rename_similarity = 100
        self.current_file = None


def _apply_diff_git_header(line: str, state: _DiffParseState) -> bool:
    if not line.startswith("diff --git "):
        return False
    state.commit_entry()
    parts = line.split()
    if len(parts) >= 4:
        state.old_path = parts[2].removeprefix("a/")
        state.new_path = parts[3].removeprefix("b/")
    return True


def _apply_diff_file_meta(line: str, state: _DiffParseState) -> bool:
    if line.startswith("new file mode"):
        state.is_new = True
        return True
    if line.startswith("deleted file mode"):
        state.is_deleted = True
        return True
    if line.startswith("copy from "):
        state.is_copy = True
        state.old_path = line[len("copy from ") :]
        return True
    if line.startswith("copy to "):
        state.new_path = line[len("copy to ") :]
        return True
    if line.startswith("similarity index "):
        state.is_rename = True
        state.rename_similarity = int(line.split()[2].rstrip("%"))
        return True
    if line.startswith("rename from "):
        state.old_path = line[len("rename from ") :]
        return True
    if line.startswith("rename to "):
        state.new_path = line[len("rename to ") :]
        return True
    return False


def _apply_diff_hunk(line: str, state: _DiffParseState) -> None:
    if not line.startswith("+++ b/"):
        if not line.startswith("@@") or state.current_file is None:
            return
        match = _HUNK_HEADER_RE.search(line)
        if not match:
            return
        old_count = int(match.group(2)) if match.group(2) is not None else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) is not None else 1
        if new_count > 0:
            state.line_map[state.current_file].update(range(new_start, new_start + new_count))
        elif old_count > 0:
            state.line_map[state.current_file].add(new_start)
        return

    state.current_file = line[6:]
    if state.current_file != "/dev/null":
        state.line_map.setdefault(state.current_file, set())


def _apply_unified_diff_line(line: str, state: _DiffParseState) -> None:
    if _apply_diff_git_header(line, state):
        return
    if _apply_diff_file_meta(line, state):
        return
    _apply_diff_hunk(line, state)


def _parse_unified_diff(stdout: str) -> GitDiffResult:
    state = _DiffParseState(line_map={}, entries=[])
    for line in stdout.splitlines():
        _apply_unified_diff_line(line, state)
    state.commit_entry()
    return GitDiffResult(line_map=state.line_map, entries=tuple(state.entries))


def fetch_diff(repo_root: Path, base_ref: str) -> GitDiffResult:
    """Return added-line map and file-level status entries from one git diff subprocess."""
    diff_result = subprocess.run(
        [_GIT, "diff", f"{base_ref}...HEAD", "--unified=0", "-M", "--diff-filter=ACDMR"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    return _parse_unified_diff(diff_result.stdout)


def fetch_diff_line_map(repo_root: Path, base_ref: str) -> dict[str, set[int]]:
    """Return {file_path: set(added_line_numbers)} for Added/Copied/Modified/Renamed files."""
    return fetch_diff(repo_root, base_ref).line_map


# ---------------------------------------------------------------------------
# Change classification
# ---------------------------------------------------------------------------


def _classify_rename(
    old_path: str,
    new_path: str,
    score: int,
    diff: dict[str, set[int]],
    discovery: TestDiscovery,
    roots: tuple[str, ...],
) -> tuple[list[str], list[str], list[str], list[tuple[str, str, int]], dict[str, frozenset[int]]]:
    """Classify one git rename (R status) into gate buckets."""
    del_test: list[str] = []
    new_test: list[str] = []
    del_source: list[str] = []
    renames: list[tuple[str, str, int]] = []
    modified: dict[str, frozenset[int]] = {}

    old_is_test = is_gate_test_path(old_path, discovery)
    new_is_test = is_gate_test_path(new_path, discovery)

    if old_is_test or new_is_test:
        if old_is_test:
            del_test.append(old_path)
        if new_is_test:
            new_test.append(new_path)
        if not old_is_test and is_product_source(old_path, roots):
            del_source.append(old_path)
        return del_test, new_test, del_source, renames, modified

    if is_config_path(old_path) or is_config_path(new_path):
        return del_test, new_test, del_source, renames, modified

    if is_gate_ignored_path(old_path) or is_gate_ignored_path(new_path):
        return del_test, new_test, del_source, renames, modified

    renames.append((old_path, new_path, score))
    if score < 100:
        modified[new_path] = frozenset(diff.get(new_path, set()))
    return del_test, new_test, del_source, renames, modified


def _entry_paths(entry: DiffEntry) -> tuple[str, ...]:
    if entry.status.startswith("R"):
        assert entry.old_path is not None
        assert entry.new_path is not None
        return (entry.old_path, entry.new_path)
    if entry.status == "D":
        assert entry.old_path is not None
        return (entry.old_path,)
    assert entry.new_path is not None
    return (entry.new_path,)


def _classify_py_entry(
    status: str,
    filepath: str,
    *,
    line_map: dict[str, set[int]],
    test_discovery: TestDiscovery,
    resolved_roots: tuple[str, ...],
    new_test: list[str],
    del_test: list[str],
    modified_test: list[str],
    new_source: list[str],
    del_source: list[str],
    modified_source: dict[str, frozenset[int]],
    track_unscoped: Callable[[str], None],
) -> None:
    is_test = is_gate_test_path(filepath, test_discovery)
    is_config = is_config_path(filepath)

    if status == "A" and is_test:
        new_test.append(filepath)
    elif status == "D" and is_test:
        del_test.append(filepath)
    elif status in ("M", "C") and is_test:
        modified_test.append(filepath)
    elif is_config or is_gate_ignored_path(filepath):
        return
    elif status == "A" and not is_test:
        if is_product_source(filepath, resolved_roots):
            new_source.append(filepath)
        else:
            track_unscoped(filepath)
    elif status == "D" and not is_test:
        del_source.append(filepath)
    elif status in ("M", "C") and not is_test:
        if is_product_source(filepath, resolved_roots):
            modified_source[filepath] = frozenset(line_map.get(filepath, set()))
        else:
            track_unscoped(filepath)


def classify_changes(
    repo_root: Path,
    base_ref: str,
    diff: GitDiffResult,
    discovery: TestDiscovery | None = None,
    roots: tuple[str, ...] | None = None,
) -> ChangeSet:
    """Return a ChangeSet from parsed git diff entries."""
    del base_ref
    line_map = diff.line_map
    diff_result = diff
    resolved_roots = roots if roots is not None else product_roots(repo_root)

    config: list[str] = []
    new_test: list[str] = []
    del_test: list[str] = []
    modified_test: list[str] = []
    new_source: list[str] = []
    del_source: list[str] = []
    modified_source: dict[str, frozenset[int]] = {}
    unscoped_source: list[str] = []
    renames: list[tuple[str, str, int]] = []
    test_discovery = discovery or default_test_discovery()

    def _track_unscoped(filepath: str) -> None:
        if (
            filepath.endswith(".py")
            and not is_gate_ignored_path(filepath)
            and not is_gate_test_path(filepath, test_discovery)
            and not is_config_path(filepath)
            and not is_product_source(filepath, resolved_roots)
        ):
            unscoped_source.append(filepath)

    for entry in diff_result.entries:
        config.extend(filepath for filepath in _entry_paths(entry) if is_full_suite_trigger_path(filepath))

        status = entry.status
        if status.startswith("R"):
            old_path = entry.old_path
            new_path = entry.new_path
            if old_path is None or new_path is None:
                continue
            score = int(status[1:]) if status[1:].isdigit() else 0
            (
                rename_del_test,
                rename_new_test,
                rename_del_source,
                rename_entries,
                rename_modified,
            ) = _classify_rename(old_path, new_path, score, line_map, test_discovery, resolved_roots)
            del_test.extend(rename_del_test)
            new_test.extend(rename_new_test)
            del_source.extend(rename_del_source)
            renames.extend(rename_entries)
            modified_source.update(rename_modified)
            continue

        candidate_path = entry.new_path if entry.new_path is not None else entry.old_path
        if candidate_path is None or not candidate_path.endswith(".py"):
            continue
        _classify_py_entry(
            entry.status,
            candidate_path,
            line_map=line_map,
            test_discovery=test_discovery,
            resolved_roots=resolved_roots,
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
        renames=tuple(renames),
        unscoped_source=tuple(sorted(set(unscoped_source))),
    )

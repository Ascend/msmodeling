"""Git diff analysis: base ref resolution, line mapping, change classification."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.gate_policy import TestDiscovery, default_test_discovery, is_gate_test_path
from scripts.helpers.ci_gate.models import ChangeSet
from scripts.helpers.common.coverage_config import product_roots
from scripts.helpers.common.test_map_config import is_config_path
from scripts.helpers.common.test_map_loader import is_product_source

_git_path = shutil.which("git")
if _git_path is None:
    raise RuntimeError("git not found")
_GIT: str = _git_path

logger = logging.getLogger(__name__)

_HUNK_RE = re.compile(r"\+(\d+)(?:,(\d+))?")


# ---------------------------------------------------------------------------
# Base ref
# ---------------------------------------------------------------------------


def resolve_base_ref(repo_root: Path, branch: str) -> str:
    """Resolve merge-base between HEAD and <branch>.

    If *branch* contains '/', it is used as-is (e.g. ``center/develop``).
    Otherwise, tries bare *branch* first, then ``origin/<branch>`` as fallback.
    """
    if "/" in branch:
        refs = [branch]
    else:
        refs = [branch, f"origin/{branch}"]

    last_stderr = ""
    for ref in refs:
        proc = subprocess.run(
            [_GIT, "merge-base", "HEAD", ref],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            logger.info("Resolved base ref using %s", ref)
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


def _parse_unified_diff(stdout: str) -> GitDiffResult:
    line_map: dict[str, set[int]] = {}
    entries: list[DiffEntry] = []

    current_file: str | None = None
    old_path: str | None = None
    new_path: str | None = None
    is_new = False
    is_deleted = False
    is_rename = False
    is_copy = False
    rename_similarity = 100

    def _commit_entry() -> None:
        nonlocal current_file, old_path, new_path, is_new, is_deleted, is_rename, is_copy, rename_similarity
        entry = _flush_diff_entry(
            old_path=old_path,
            new_path=new_path,
            is_new=is_new,
            is_deleted=is_deleted,
            is_rename=is_rename,
            rename_similarity=rename_similarity,
            is_copy=is_copy,
        )
        if entry is not None:
            entries.append(entry)
        old_path = None
        new_path = None
        is_new = False
        is_deleted = False
        is_rename = False
        is_copy = False
        rename_similarity = 100
        current_file = None

    for line in stdout.splitlines():
        if line.startswith("diff --git "):
            _commit_entry()
            parts = line.split()
            if len(parts) >= 4:
                old_path = parts[2].removeprefix("a/")
                new_path = parts[3].removeprefix("b/")
            continue

        if line.startswith("new file mode"):
            is_new = True
        elif line.startswith("deleted file mode"):
            is_deleted = True
        elif line.startswith("copy from "):
            is_copy = True
            old_path = line[len("copy from ") :]
        elif line.startswith("copy to "):
            new_path = line[len("copy to ") :]
        elif line.startswith("similarity index "):
            is_rename = True
            rename_similarity = int(line.split()[2].rstrip("%"))
        elif line.startswith("rename from "):
            old_path = line[len("rename from ") :]
        elif line.startswith("rename to "):
            new_path = line[len("rename to ") :]
        elif line.startswith("+++ b/"):
            current_file = line[6:]
            if current_file != "/dev/null":
                line_map.setdefault(current_file, set())
        elif line.startswith("@@") and current_file is not None:
            match = _HUNK_RE.search(line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                line_map[current_file].update(range(start, start + count))

    _commit_entry()
    return GitDiffResult(line_map=line_map, entries=tuple(entries))


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

    renames.append((old_path, new_path, score))
    if score < 100:
        modified[new_path] = frozenset(diff.get(new_path, set()))
    return del_test, new_test, del_source, renames, modified


def _entry_paths(entry: DiffEntry) -> tuple[str, ...]:
    if entry.status.startswith("R"):
        assert entry.old_path is not None and entry.new_path is not None
        return (entry.old_path, entry.new_path)
    if entry.status == "D":
        assert entry.old_path is not None
        return (entry.old_path,)
    assert entry.new_path is not None
    return (entry.new_path,)


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
    renames: list[tuple[str, str, int]] = []
    test_discovery = discovery or default_test_discovery()

    for entry in diff_result.entries:
        for filepath in _entry_paths(entry):
            if is_config_path(filepath):
                config.append(filepath)

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
        filepath = candidate_path

        is_test = is_gate_test_path(filepath, test_discovery)
        is_config = is_config_path(filepath)

        if status == "A" and is_test:
            new_test.append(filepath)
        elif status == "D" and is_test:
            del_test.append(filepath)
        elif status in ("M", "C") and is_test:
            modified_test.append(filepath)
        elif status == "A" and not is_test:
            new_source.append(filepath)
        elif status == "D" and not is_test:
            del_source.append(filepath)
        elif status in ("M", "C") and not is_test and not is_config:
            modified_source[filepath] = frozenset(line_map.get(filepath, set()))

    return ChangeSet.build(
        config=tuple(config),
        new_test=tuple(new_test),
        del_test=tuple(del_test),
        modified_test=tuple(modified_test),
        new_source=tuple(new_source),
        del_source=tuple(del_source),
        modified_source=modified_source,
        renames=tuple(renames),
    )

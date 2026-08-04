"""Git diff analysis: base ref resolution and unified diff parsing."""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from scripts.helpers._config import ConfigError

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


def git_stdout(repo_root: Path, *args: str) -> str:
    """Run git in *repo_root* and return stripped stdout (empty on failure)."""
    return _run_git(repo_root, *args).stdout.strip()


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
        logger.warning(
            "git fetch failed for %s (%s %s): %s",
            ref,
            remote,
            branch,
            proc.stderr.strip(),
        )


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
        [
            _GIT,
            "diff",
            f"{base_ref}...HEAD",
            "--unified=0",
            "-M",
            "--diff-filter=ACDMR",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    if diff_result.returncode != 0:
        raise ConfigError(f"git diff failed: {diff_result.stderr.strip()}")
    return _parse_unified_diff(diff_result.stdout)


def fetch_diff_line_map(repo_root: Path, base_ref: str) -> dict[str, set[int]]:
    """Return {file_path: set(added_line_numbers)} for Added/Copied/Modified/Renamed files."""
    return fetch_diff(repo_root, base_ref).line_map


def resolve_ref_commit(repo_root: Path, ref: str) -> str:
    """Return full SHA for an arbitrary git ref."""
    proc = _run_git(repo_root, "rev-parse", ref)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise ConfigError(f"Cannot resolve ref {ref!r}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def is_git_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    """Return True when *ancestor* is an ancestor of *descendant*."""
    proc = _run_git(repo_root, "merge-base", "--is-ancestor", ancestor, descendant)
    return proc.returncode == 0


def fetch_ref(repo_root: Path, ref: str) -> None:
    """Fetch *ref* from its remote so local rev-parse / merge-base can resolve it."""
    _fetch_deepen(repo_root, ref)


def resolve_remote_ref(ref: str) -> str:
    """Return the git ref that points at the fetched remote tip for *ref*."""
    remote, branch = _parse_fetch_remote_branch(ref)
    return ref if "/" in ref else f"{remote}/{branch}"


def resolve_target_head(repo_root: Path, ref: str) -> str:
    """Fetch *ref* and return the full SHA of the remote tip."""
    fetch_ref(repo_root, ref)
    return resolve_ref_commit(repo_root, resolve_remote_ref(ref))


_SYNC_BRANCH_PREFIX = "msmodeling-sync/"


@dataclass(frozen=True, slots=True)
class _EphemeralCheckoutState:
    work_branch: str
    restore_ref: str


_ephemeral_checkouts: dict[str, _EphemeralCheckoutState] = {}


def _sync_work_branch_name() -> str:
    return f"{_SYNC_BRANCH_PREFIX}{os.getpid()}"


def _capture_restore_ref(repo_root: Path) -> str:
    branch = git_stdout(repo_root, "symbolic-ref", "--short", "-q", "HEAD")
    if branch:
        return branch
    return resolve_head_commit(repo_root)


def _cleanup_ephemeral_checkout(repo_root: Path) -> None:
    key = str(repo_root.resolve())
    state = _ephemeral_checkouts.pop(key, None)
    if state is None:
        return
    checkout = _run_git(repo_root, "checkout", state.restore_ref)
    if checkout.returncode != 0:
        logger.error(
            "git checkout %s failed during ephemeral cleanup (exit %d): %s",
            state.restore_ref,
            checkout.returncode,
            checkout.stderr.strip(),
        )
        force = _run_git(repo_root, "checkout", "-f", state.restore_ref)
        if force.returncode != 0:
            logger.error(
                "git checkout -f %s failed during ephemeral cleanup (exit %d): %s",
                state.restore_ref,
                force.returncode,
                force.stderr.strip(),
            )
            return
    delete = _run_git(repo_root, "branch", "-D", state.work_branch)
    if delete.returncode != 0:
        logger.warning(
            "git branch -D %s failed during ephemeral cleanup (exit %d): %s",
            state.work_branch,
            delete.returncode,
            delete.stderr.strip(),
        )


def cleanup_all_ephemeral_checkouts() -> None:
    """Restore git state for any in-flight sync checkout sessions."""
    for key in list(_ephemeral_checkouts):
        _cleanup_ephemeral_checkout(Path(key))


atexit.register(cleanup_all_ephemeral_checkouts)


@contextmanager
def ephemeral_target_checkout(repo_root: Path, ref: str) -> Iterator[str]:
    """Check out target tip on a pid-scoped branch; restore and delete on exit."""
    target_head = resolve_target_head(repo_root, ref)
    work_branch = _sync_work_branch_name()
    restore_ref = _capture_restore_ref(repo_root)
    proc = _run_git(repo_root, "checkout", "-B", work_branch, target_head)
    if proc.returncode != 0:
        raise ConfigError(f"git checkout -B {work_branch!r} {target_head[:12]} failed: {proc.stderr.strip()}")
    key = str(repo_root.resolve())
    _ephemeral_checkouts[key] = _EphemeralCheckoutState(work_branch, restore_ref)
    try:
        yield target_head
    finally:
        _cleanup_ephemeral_checkout(repo_root)


def fetch_changed_paths(repo_root: Path, base_commit: str, head_commit: str) -> frozenset[str]:
    """Return repository-relative paths changed between two commits."""
    proc = _run_git(repo_root, "diff", f"{base_commit}...{head_commit}", "--name-only")
    if proc.returncode != 0:
        raise ConfigError(f"git diff failed: {proc.stderr.strip()}")
    return frozenset(line.strip() for line in proc.stdout.splitlines() if line.strip())

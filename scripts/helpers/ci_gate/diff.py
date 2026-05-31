"""Git diff analysis: base ref resolution, line mapping, change classification."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.gate_policy import TestDiscovery, default_test_discovery, is_gate_test_path
from scripts.helpers.ci_gate.models import ChangeSet
from scripts.helpers.common.coverage_config import PRODUCT_SOURCE_PREFIXES
from scripts.helpers.common.test_map_config import is_config_path
from scripts.helpers.common.test_map_loader import is_product_source

_GIT = shutil.which("git")
if _GIT is None:
    raise RuntimeError("git not found")

logger = logging.getLogger(__name__)


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
# Diff line map
# ---------------------------------------------------------------------------


def fetch_diff_line_map(repo_root: Path, base_ref: str) -> dict[str, set[int]]:
    """Return {file_path: set(added_line_numbers)} for Added/Copied/Modified/Renamed files.

    Uses ``-M --diff-filter=ACMR``: Deleted files have no added lines so they
    are excluded; renames (R) are detected so a renamed-and-edited file's added
    lines are keyed by its new path. classify_changes uses ``ACDMR`` to also
    capture deleted file names.
    """
    diff_result = subprocess.run(
        [_GIT, "diff", f"{base_ref}...HEAD", "--unified=0", "-M", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    file_lines: dict[str, set[int]] = {}
    current_file: str | None = None
    for line in diff_result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            file_lines[current_file] = set()
        elif line.startswith("@@") and current_file is not None:
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                file_lines[current_file].update(range(start, start + count))
    return file_lines


# ---------------------------------------------------------------------------
# Change classification
# ---------------------------------------------------------------------------


def _classify_rename(
    old_path: str,
    new_path: str,
    score: int,
    diff: dict[str, set[int]],
    discovery: TestDiscovery,
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
        if not old_is_test and is_product_source(old_path, PRODUCT_SOURCE_PREFIXES):
            del_source.append(old_path)
        return del_test, new_test, del_source, renames, modified

    if is_config_path(old_path) or is_config_path(new_path):
        return del_test, new_test, del_source, renames, modified

    renames.append((old_path, new_path, score))
    if score < 100:
        modified[new_path] = frozenset(diff.get(new_path, set()))
    return del_test, new_test, del_source, renames, modified


def classify_changes(
    repo_root: Path,
    base_ref: str,
    diff: dict[str, set[int]],
    discovery: TestDiscovery | None = None,
) -> ChangeSet:
    """Return a ChangeSet from git diff --name-status.

    *diff* provides added line numbers for ACMR files (see fetch_diff_line_map).
    Deleted files are captured here via ``-M --diff-filter=ACDMR`` but have no
    line info. Renames (status ``R<score>``) are classified as follows:

    * Product-source rename: recorded in ``renames`` (so the gate can remap the
      test_map old→new). A pure rename (score 100) requires no new tests; a
      rename with edits (score < 100) also enters ``modified_source`` so only
      the changed symbols are checked against the remapped map.
    * Test rename: if either path is a gate test, treat as delete-old and/or
      add-new based on each side; a product source renamed into a test also
      records ``del_source`` for the old path.
    """
    result = subprocess.run(
        [
            _GIT,
            "diff",
            f"{base_ref}...HEAD",
            "--name-status",
            "-M",
            "--diff-filter=ACDMR",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )

    config: list[str] = []
    new_test: list[str] = []
    del_test: list[str] = []
    new_source: list[str] = []
    del_source: list[str] = []
    modified_source: dict[str, frozenset[int]] = {}
    renames: list[tuple[str, str, int]] = []
    test_discovery = discovery or default_test_discovery()

    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        filepath = parts[-1]
        is_test = is_gate_test_path(filepath, test_discovery)
        is_config = is_config_path(filepath)

        if is_config:
            config.append(filepath)

        if not filepath.endswith(".py"):
            continue

        if status.startswith("R"):
            old_path = parts[1]
            new_path = parts[-1]
            score = int(status[1:]) if status[1:].isdigit() else 0
            (
                rename_del_test,
                rename_new_test,
                rename_del_source,
                rename_entries,
                rename_modified,
            ) = _classify_rename(old_path, new_path, score, diff, test_discovery)
            del_test.extend(rename_del_test)
            new_test.extend(rename_new_test)
            del_source.extend(rename_del_source)
            renames.extend(rename_entries)
            modified_source.update(rename_modified)
            continue

        if status == "A" and is_test:
            new_test.append(filepath)
        elif status == "D" and is_test:
            del_test.append(filepath)
        elif status == "A" and not is_test:
            new_source.append(filepath)
        elif status == "D" and not is_test:
            del_source.append(filepath)
        elif status in ("M", "C") and not is_test and not is_config:
            modified_source[filepath] = frozenset(diff.get(filepath, set()))

    return ChangeSet.build(
        config=tuple(config),
        new_test=tuple(new_test),
        del_test=tuple(del_test),
        new_source=tuple(new_source),
        del_source=tuple(del_source),
        modified_source=modified_source,
        renames=tuple(renames),
    )

"""Git diff analysis: base ref resolution, line mapping, change classification."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Final

from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.models import ChangeSet
from scripts.helpers.common.test_map_config import is_config_path

# Map product source prefix → regression layer directory.
# Used by _split_cross_layer_tests (in rules.py) to decide whether a source
# change targets a specific regression layer or all layers.
_SOURCE_PREFIX_TO_LAYER: Final[dict[str, str]] = {
    "tensor_cast/": "tests/regression/tensor_cast/",
    "serving_cast/": "tests/regression/serving_cast/",
}

_REGRESSION_LAYERS: Final[tuple[str, ...]] = (
    "tests/regression/tensor_cast/",
    "tests/regression/serving_cast/",
    "tests/regression/cli/",
    "tests/regression/web_ui/",
)

_GIT = shutil.which("git")
if _GIT is None:
    raise RuntimeError("git not found")


# ---------------------------------------------------------------------------
# Base ref
# ---------------------------------------------------------------------------


def resolve_base_ref(repo_root: Path, branch: str) -> str:
    """Resolve merge-base for *branch* against HEAD.

    Falls back to origin/<branch> when local branch ref not available.
    """
    proc = subprocess.run(
        [_GIT, "merge-base", "HEAD", branch],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()

    remote_ref = f"origin/{branch}"
    fallback = subprocess.run(
        [_GIT, "rev-parse", remote_ref],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    if fallback.returncode != 0 or not fallback.stdout.strip():
        raise ConfigError(f"Cannot resolve base ref: merge-base {branch!r} failed, {remote_ref!r} not found either.")

    proc = subprocess.run(
        [_GIT, "merge-base", "HEAD", remote_ref],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise ConfigError(f"Cannot resolve merge-base between HEAD and {remote_ref!r}.")
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# Diff line map
# ---------------------------------------------------------------------------


def fetch_diff_line_map(repo_root: Path, base_ref: str) -> dict[str, set[int]]:
    """Return {file_path: set(added_line_numbers)} for Added/Copied/Modified files.

    Uses --diff-filter=ACM intentionally: Deleted files have no added lines,
    so they are excluded. classify_changes uses --diff-filter=ACDM to capture
    both added and deleted file names.
    """
    diff_result = subprocess.run(
        [_GIT, "diff", f"{base_ref}...HEAD", "--unified=0", "--diff-filter=ACM"],
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


def classify_changes(repo_root: Path, base_ref: str, diff: dict[str, set[int]]) -> ChangeSet:
    """Return a ChangeSet from git diff --name-status.

    *diff* provides added line numbers for ACM files (see fetch_diff_line_map).
    Deleted files are captured here via --diff-filter=ACDM but have no line info.
    """
    result = subprocess.run(
        [_GIT, "diff", f"{base_ref}...HEAD", "--name-status", "--diff-filter=ACDM"],
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

    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        status, filepath = parts[0], parts[-1]
        is_test = filepath.startswith("tests/")
        is_config = is_config_path(filepath)

        if is_config:
            config.append(filepath)
        elif status == "A" and is_test:
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
    )


# ---------------------------------------------------------------------------
# Layer helpers (used by rules.py)
# ---------------------------------------------------------------------------


def _regression_layer_for_source(source_path: str) -> str | None:
    """Return the regression layer directory for a source prefix, or None."""
    for prefix, layer in _SOURCE_PREFIX_TO_LAYER.items():
        if source_path.startswith(prefix):
            return layer
    return None


def _layer_of_test(test_id: str) -> str | None:
    for layer in _REGRESSION_LAYERS:
        if test_id.startswith(layer):
            return layer
    return None

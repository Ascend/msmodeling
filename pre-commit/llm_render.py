#!/usr/bin/env python3
"""Render pre-commit output as compact, LLM-friendly format.

Set PRE_COMMIT_LLM_FILTER=1 to enable.
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

# --- check enabled ---
if os.environ.get("PRE_COMMIT_LLM_FILTER") not in ("1", "true", "yes"):
    # Pass-through: copy stdin to stdout unchanged
    sys.stdout.write(sys.stdin.read())
    sys.exit(0)

# --- compiled regex (module-level, once) ---
_RE_FIXING = re.compile(r"^Fixing (.+)$")
_RE_PASSED_OR_SKIPPED = re.compile(r"(Passed|Skipped)")
_RE_FRAMEWORK_LINE = re.compile(r"^(- hook id:|.*exit code:)")
_RE_RUFF_ANNOTATION = re.compile(
    r"::error title=Ruff \((?P<rule>[^)]+)\),file=(?P<file>[^,]+),"
    r"line=(?P<line>\d+),col=(?P<col>\d+).*?::(?P<msg>.+)$"
)
_RE_RUFF_FORMAT = re.compile(r"error: Failed to parse (.+):(\d+):(\d+): (.+)$")
_RE_SPELL = re.compile(r"^(.+):(\d+): (.+)$")
_RE_PYLINT_HEADER = re.compile(r"^\*+\s+Module")
_RE_PYLINT_ISSUE = re.compile(r"^(.+?):(\d+):(\d+): (\w+): (.+)$")
_RE_BANDIT_ISSUE = re.compile(r">> Issue: \[(\w+):(\w+)\] (.+)")
_RE_BANDIT_SEVERITY = re.compile(r"Severity:\s*(.+)")
_RE_BANDIT_LOCATION = re.compile(r"Location:\s*(.+:\d+:\d+)")
_RE_BLOCK_SPLIT = re.compile(r"(?=^\S.*\.{3,})", flags=re.MULTILINE)
_RE_FAILED_MARKER = re.compile(r"Failed")
_RE_GITCODE_PREFIX = re.compile(r".*/gitcode/")

# --- display-name → hook-id mapping (framework first line) ---
_DISPLAY_TO_HOOK_ID: dict[str, str] = {
    "trim trailing whitespace": "trailing-whitespace",
    "fix end of files": "end-of-file-fixer",
    "check yaml": "check-yaml",
    "check for added large files": "check-added-large-files",
    "check for merge conflicts": "check-merge-conflict",
    "detect private key": "detect-private-key",
    "check json": "check-json",
    "ruff check": "ruff-check",
    "ruff format": "ruff-format",
    "codespell": "codespell",
    "pylint (Python code quality check)": "pylint",
    "bandit (Python security checks)": "bandit",
    "typos": "typos",
}

# --- type aliases ---
Lines = list[str]
RenderedIssue = tuple[str, int | None, str]  # (file, line_no or None, detail)
HookResult = list[RenderedIssue]
Renderer = Callable[[Lines], HookResult]


# --- renderers ---

def _parse_fixer(lines: Lines) -> HookResult:
    for line in lines:
        m = _RE_FIXING.match(line.strip())
        if m:
            return [(m.group(1), None, "(fixed)")]
    return []


def _parse_check(lines: Lines) -> HookResult:
    issues: HookResult = []
    capture = False
    for line in lines:
        if _RE_FAILED_MARKER.search(line) and not line.startswith("-"):
            capture = True
            continue
        if capture and line.strip() and not _RE_FRAMEWORK_LINE.match(line):
            issues.append(("", None, line.strip()))
    return issues


def _parse_ruff(lines: Lines) -> HookResult:
    issues: HookResult = []
    for line in lines:
        m = _RE_RUFF_ANNOTATION.search(line)
        if m:
            d = m.groupdict()
            issues.append((
                _relative_path(d["file"]),
                int(d["line"]),
                f"{d['col']} {d['rule']}: {d['msg']}",
            ))
    return issues


def _parse_ruff_format(lines: Lines) -> HookResult:
    issues: HookResult = []
    for line in lines:
        m = _RE_RUFF_FORMAT.match(line.strip())
        if m:
            file, line_no, col, msg = m.groups()
            issues.append((_relative_path(file), int(line_no), f"{col} {msg}"))
    return issues


def _parse_spell(lines: Lines) -> HookResult:
    issues: HookResult = []
    for line in lines:
        m = _RE_SPELL.match(line.strip())
        if m and "==>" in m.group(3):
            file, line_no, detail = m.groups()
            issues.append((_relative_path(file), int(line_no), detail))
    return issues


def _parse_pylint(lines: Lines) -> HookResult:
    issues: HookResult = []
    for line in lines:
        if _RE_PYLINT_HEADER.match(line):
            continue
        m = _RE_PYLINT_ISSUE.match(line.strip())
        if m:
            file, line_no, col, code, msg = m.groups()
            issues.append((_relative_path(file), int(line_no), f"{col} {code}: {msg}"))
    return issues


def _parse_bandit(lines: Lines) -> HookResult:
    issues: HookResult = []
    current_code = ""
    current_desc = ""
    current_severity = ""
    for line in lines:
        m = _RE_BANDIT_ISSUE.match(line)
        if m:
            current_code = f"{m.group(1)}:{m.group(2)}"
            current_desc = m.group(3)
            continue
        m = _RE_BANDIT_SEVERITY.match(line.strip())
        if m:
            current_severity = m.group(1)
            continue
        m = _RE_BANDIT_LOCATION.match(line.strip())
        if m:
            parts = m.group(1).rsplit(":", 2)
            if len(parts) == 3:
                file, line_no, col = parts
                issues.append((
                    _relative_path(file.strip()),
                    int(line_no),
                    f"{col} {current_code} {current_desc} [Severity: {current_severity}]",
                ))
            current_code = current_desc = current_severity = ""
    return issues


# --- helpers ---

def _relative_path(absolute: str) -> str:
    return _RE_GITCODE_PREFIX.sub("", absolute)


def _is_passed_or_skipped(lines: Lines) -> bool:
    for line in lines:
        if _RE_PASSED_OR_SKIPPED.search(line):
            return True
    return False


def _extract_hook_ids_from_config(config_path: Path) -> set[str]:
    """Parse .pre-commit-config.yaml for all hook ids (best-effort YAML)."""
    hook_ids: set[str] = set()
    try:
        text = config_path.read_text()
    except OSError:
        return hook_ids
    for m in re.finditer(r"^\s*- id:\s*(\S+)", text, flags=re.MULTILINE):
        hook_ids.add(m.group(1))
    return hook_ids


def _resolve_hook_id(first_line: str) -> str | None:
    for display_name, hook_id in _DISPLAY_TO_HOOK_ID.items():
        if first_line.startswith(display_name):
            return hook_id
    return None


# --- registry ---

HOOK_RENDERERS: dict[str, Renderer] = {
    "trailing-whitespace": _parse_fixer,
    "end-of-file-fixer": _parse_fixer,
    "check-yaml": _parse_check,
    "check-added-large-files": _parse_check,
    "check-merge-conflict": _parse_check,
    "detect-private-key": _parse_check,
    "check-json": _parse_check,
    "ruff-check": _parse_ruff,
    "ruff-format": _parse_ruff_format,
    "codespell": _parse_spell,
    "pylint": _parse_pylint,
    "bandit": _parse_bandit,
    "typos": _parse_spell,
}


# --- main ---

def main() -> None:
    # Validate: all config hook-ids have a registered renderer
    repo_root = _find_repo_root()
    config_path = repo_root / ".pre-commit-config.yaml"
    configured_ids = _extract_hook_ids_from_config(config_path)

    missing = configured_ids - set(HOOK_RENDERERS)
    if missing:
        print(
            f"ERROR: PRE_COMMIT_LLM_FILTER: no renderer registered for hook-id(s): "
            f"{', '.join(sorted(missing))}",
            file=sys.stderr,
        )
        sys.exit(1)

    raw = sys.stdin.read()
    blocks = _RE_BLOCK_SPLIT.split(raw)

    grouped: dict[str, HookResult] = defaultdict(list)

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        first_line = lines[0]
        hook_id = _resolve_hook_id(first_line)

        if hook_id is None or _is_passed_or_skipped(lines):
            continue

        # Only render hooks that are in the config (skip any unknown)
        renderer = HOOK_RENDERERS.get(hook_id)
        if renderer is None:
            continue

        issues = renderer(lines)
        if issues:
            grouped[hook_id].extend(issues)

    if not grouped:
        sys.exit(0)

    for hook_id, issues in grouped.items():
        by_file: dict[str, list[tuple[int | None, str]]] = defaultdict(list)
        for file, line_no, detail in issues:
            by_file[file].append((line_no, detail))

        print(hook_id)
        for file, details in by_file.items():
            print(f"- {file}")
            for line_no, detail in details:
                if line_no is not None:
                    print(f"> {line_no}:{detail}")
                else:
                    print(f"> {detail}")
        print()


def _find_repo_root() -> Path:
    """Locate repo root (where .pre-commit-config.yaml lives)."""
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir.parent
    while candidate != candidate.parent:
        if (candidate / ".pre-commit-config.yaml").exists():
            return candidate
        candidate = candidate.parent
    return Path.cwd()


if __name__ == "__main__":
    main()
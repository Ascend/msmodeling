#!/usr/bin/env python3
"""Validate the GitCode CLI contract required by AI-native workflows."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

MINIMUM_VERSION: Final = (0, 8, 0)
DEVELOPMENT_BUILD_MARKER: Final = "version dev"
REQUIRED_SCHEMAS: Final = {
    "issue create": {"body-file", "dry-run", "json", "repo", "title"},
    "issue list": {"assignee", "json", "repo", "state"},
    "issue comment": {"body-file", "json", "repo"},
    "pr create": {"base", "body-file", "fork", "head", "json", "repo"},
    "pr comments": {"json", "repo"},
    "pr diff": {"json", "repo"},
    "pr comment": {"body-file", "json", "path", "position", "repo"},
    "pr review": {"comment-file", "json", "repo"},
}


@dataclass(frozen=True)
class CheckResult:
    binary: str
    version_text: str
    semantic_version: str | None
    development_build: bool
    auth_ok: bool
    schemas: dict[str, bool]
    errors: list[str]
    ok: bool


def parse_semantic_version(version_text: str) -> tuple[int, int, int] | None:
    """Extract a semantic version from GitCode CLI output."""
    match = re.search(r"\bversion\s+v?(\d+)\.(\d+)\.(\d+)\b", version_text)
    if match is None:
        return None
    return tuple(int(group) for group in match.groups())


def _run(binary: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def _resolve_binary(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("GITCODE_BIN") or shutil.which("gitcode")
    if not candidate:
        raise FileNotFoundError("gitcode executable not found; set GITCODE_BIN or install GitCode CLI")
    return str(Path(candidate))


def check_cli(binary: str) -> CheckResult:
    """Run version, authentication, and schema checks."""
    errors: list[str] = []
    version_process = _run(binary, "version")
    version_text = (version_process.stdout or version_process.stderr).strip()
    semantic = parse_semantic_version(version_text)
    development_build = semantic is None and DEVELOPMENT_BUILD_MARKER in version_text.lower()

    if version_process.returncode != 0:
        errors.append("gitcode version failed")
    elif semantic is not None and semantic < MINIMUM_VERSION:
        errors.append(
            f"GitCode CLI {semantic[0]}.{semantic[1]}.{semantic[2]} is older than "
            f"{MINIMUM_VERSION[0]}.{MINIMUM_VERSION[1]}.{MINIMUM_VERSION[2]}"
        )
    elif semantic is None and not development_build:
        errors.append("unable to identify GitCode CLI version")

    auth_process = _run(binary, "auth", "status")
    auth_ok = auth_process.returncode == 0
    if not auth_ok:
        errors.append("gitcode auth status failed")

    schema_results: dict[str, bool] = {}
    for command, required_flags in REQUIRED_SCHEMAS.items():
        process = _run(binary, "schema", command)
        valid = False
        if process.returncode == 0:
            try:
                payload = json.loads(process.stdout)
                flags = {item["name"] for item in payload.get("flags", [])}
                valid = required_flags <= flags
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                valid = False
        schema_results[command] = valid
        if not valid:
            errors.append(f"schema mismatch: {command}")

    return CheckResult(
        binary=binary,
        version_text=version_text.splitlines()[0] if version_text else "",
        semantic_version=".".join(str(part) for part in semantic) if semantic else None,
        development_build=development_build,
        auth_ok=auth_ok,
        schemas=schema_results,
        errors=errors,
        ok=not errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", help="GitCode CLI executable; defaults to GITCODE_BIN or PATH")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    try:
        binary = _resolve_binary(args.binary)
        result = check_cli(binary)
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "errors": [str(exc)]}, ensure_ascii=False))
        else:
            print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(f"binary: {result.binary}")
        print(f"version: {result.version_text}")
        print(f"auth: {'ok' if result.auth_ok else 'failed'}")
        for command, valid in result.schemas.items():
            print(f"schema {command}: {'ok' if valid else 'failed'}")
        for error in result.errors:
            print(f"[ERROR] {error}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

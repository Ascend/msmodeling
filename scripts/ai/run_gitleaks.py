#!/usr/bin/env python3
"""Run the repository's pinned gitleaks binary on every supported platform."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _find_binary(repo_root: Path) -> tuple[Path | None, tuple[Path, ...]]:
    preferred = repo_root / ("gitleaks.exe" if os.name == "nt" else "gitleaks")
    fallback = repo_root / ("gitleaks" if os.name == "nt" else "gitleaks.exe")
    candidates = (preferred,) if preferred == fallback else (preferred, fallback)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate, candidates
    return None, candidates


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    binary, candidates = _find_binary(repo_root)
    if binary is None:
        checked = ", ".join(str(candidate) for candidate in candidates)
        print(f"gitleaks binary not found or not executable; checked: {checked}", file=sys.stderr)
        return 1
    try:
        return subprocess.call([str(binary), *sys.argv[1:]], cwd=repo_root)
    except OSError as exc:
        print(f"failed to run gitleaks binary {binary}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

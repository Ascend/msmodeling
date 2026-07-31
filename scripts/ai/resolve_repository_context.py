#!/usr/bin/env python3
"""Resolve canonical, source, and operation-target repository identities."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

CONTRACT_PATH = Path(".agents/repository-contract.json")
REPOSITORY_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def validate_repository_slug(slug: str) -> str:
    """Validate and return an owner/repository slug."""
    if not REPOSITORY_SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"invalid owner/repository: {slug}")
    return slug


def parse_repository_slug(remote_url: str) -> str:
    """Return owner/repository from GitCode SSH or HTTPS remote URLs."""
    value = remote_url.strip()
    scp_match = re.match(r"^[^@]+@[^:]+:(?P<path>.+)$", value)
    path = scp_match.group("path") if scp_match else urlparse(value).path
    slug = path.strip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]
    try:
        return validate_repository_slug(slug)
    except ValueError as exc:
        raise ValueError(f"cannot resolve owner/repository from remote URL: {remote_url}") from exc


def load_contract(repo_root: Path) -> dict:
    """Load the committed repository identity contract."""
    path = repo_root / CONTRACT_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def get_source_repository(repo_root: Path, remote: str) -> str:
    """Resolve the writable source repository from a local Git remote."""
    git_binary = shutil.which("git")
    if not git_binary:
        raise ValueError("git executable is unavailable")
    process = subprocess.run(
        [git_binary, "remote", "get-url", remote],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        raise ValueError(f"Git remote is unavailable: {remote}")
    return parse_repository_slug(process.stdout)


def resolve_context(
    repo_root: Path,
    *,
    operation_repository: str | None,
    write: bool,
    source_remote: str | None = None,
) -> dict:
    """Resolve repository identities and enforce explicit remote write targets."""
    contract = load_contract(repo_root)
    remote = source_remote or contract["source_repository"]["default_remote"]
    canonical = validate_repository_slug(contract["canonical_repository"])
    if write and not operation_repository:
        raise ValueError("remote writes require an explicit --repo owner/repository")
    target = validate_repository_slug(operation_repository or canonical)
    return {
        "canonical_repository": canonical,
        "default_base_branch": contract["default_base_branch"],
        "source_remote": remote,
        "source_repository": get_source_repository(repo_root, remote),
        "operation_target": target,
        "operation_kind": "write" if write else "read",
        "canonical_target": target == canonical,
        "target_pull_request_ci_required": (
            target == canonical and contract["ci"]["required_for_canonical_pull_request"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--repo", help="Explicit operation target owner/repository")
    parser.add_argument("--source-remote", help="Git remote used for branch push")
    parser.add_argument("--write", action="store_true", help="Require an explicit target")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        context = resolve_context(
            args.repo_root.resolve(),
            operation_repository=args.repo,
            write=args.write,
            source_remote=args.source_remote,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "errors": [str(exc)]}, ensure_ascii=False))
        else:
            print(f"[ERROR] {exc}")
        return 1
    if args.json:
        print(json.dumps({"ok": True, **context}, ensure_ascii=False, indent=2))
    else:
        for key, value in context.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

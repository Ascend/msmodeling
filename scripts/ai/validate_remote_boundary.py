#!/usr/bin/env python3
"""Ensure project skills do not bypass GitCode CLI."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

FORBIDDEN_PATTERNS = (
    "api.gitcode.com",
    "review_api.py",
    ".config/sig-review",
    ".config\\sig-review",
)
TEXT_SUFFIXES = {".md", ".py", ".json", ".yaml", ".yml"}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    pattern: str


def _scan_for_patterns(roots: list[Path], patterns: tuple[str, ...], repo_root: Path) -> list[Finding]:
    """Scan text files under ``roots`` for case-insensitive pattern matches."""
    findings: list[Finding] = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                lowered = line.lower()
                for pattern in patterns:
                    if pattern.lower() in lowered:
                        findings.append(
                            Finding(
                                path=path.relative_to(repo_root).as_posix(),
                                line=line_number,
                                pattern=pattern,
                            )
                        )
    return findings


def validate_boundary(repo_root: Path) -> list[Finding]:
    """Find direct GitCode API and legacy review-client references."""
    skills_root = repo_root / ".agents" / "skills"
    return _scan_for_patterns([skills_root], FORBIDDEN_PATTERNS, repo_root)


def resolve_source_repository(repo_root: Path) -> str | None:
    """Resolve owner/repository from origin without embedding a contributor identity."""
    git_binary = shutil.which("git")
    if not git_binary:
        return None
    process = subprocess.run(
        [git_binary, "remote", "get-url", "origin"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        return None
    value = process.stdout.strip()
    scp_match = re.match(r"^[^@]+@[^:]+:(?P<path>.+)$", value)
    path = scp_match.group("path") if scp_match else urlparse(value).path
    slug = path.strip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]
    return slug if len(slug.split("/")) == 2 else None


def validate_repository_identity(
    repo_root: Path,
    contributor_repository: str | None = None,
) -> list[Finding]:
    """Ensure normative assets do not hard-code a contributor's personal Fork."""
    contract_path = repo_root / ".agents" / "repository-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    canonical_repository = contract["canonical_repository"]
    source_repository = contributor_repository or resolve_source_repository(repo_root)
    if not source_repository or source_repository.lower() == canonical_repository.lower():
        return []

    roots = [
        repo_root / "AGENTS.md",
        repo_root / "README.md",
        repo_root / "CONTRIBUTING.md",
        repo_root / "spec",
        repo_root / "docs" / "ai-native",
        repo_root / ".agents" / "README.md",
        repo_root / ".agents" / "gitcode-skills.lock.json",
        repo_root / ".agents" / "repository-contract.json",
        repo_root / ".agents" / "skills",
    ]
    return _scan_for_patterns(roots, (source_repository,), repo_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    findings = validate_boundary(repo_root) + validate_repository_identity(repo_root)
    if args.json:
        print(
            json.dumps(
                {"ok": not findings, "findings": [asdict(item) for item in findings]},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for finding in findings:
            print(f"{finding.path}:{finding.line}: forbidden pattern {finding.pattern}")
        print(f"remote boundary validation: {'passed' if not findings else 'failed'}")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())

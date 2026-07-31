#!/usr/bin/env python3
"""Install the local gitleaks binary required by the pre-commit gitleaks hook.

The ``.pre-commit-config.yaml`` ``gitleaks-offline-scan`` hook uses
``language: system`` with ``entry: ./gitleaks``, so pre-commit cannot
auto-install it. This script is an idempotent, cross-platform installer with a
pinned version so AI agents and contributors can prepare the binary without
ad-hoc reasoning.

Usage::

    python scripts/ai/install_gitleaks.py            # install if missing
    python scripts/ai/install_gitleaks.py --json     # machine-readable
    python scripts/ai/install_gitleaks.py --force    # reinstall
"""

from __future__ import annotations

import argparse
import json
import platform
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

GITLEAKS_VERSION: Final = "8.21.2"
RELEASE_BASE: Final = "https://github.com/gitleaks/gitleaks/releases/download"
BINARY_NAME: Final = "gitleaks"


@dataclass(frozen=True)
class PlatformAsset:
    archive: str
    is_zip: bool
    binary_member: str


def detect_asset(version: str) -> PlatformAsset:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        arch = "x64" if machine in ("amd64", "x86_64") else "arm64"
        return PlatformAsset(f"gitleaks_{version}_windows_{arch}.zip", True, "gitleaks.exe")
    if system == "linux":
        arch = "x64" if machine in ("amd64", "x86_64") else "arm64"
        return PlatformAsset(f"gitleaks_{version}_linux_{arch}.tar.gz", False, BINARY_NAME)
    if system == "darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
        return PlatformAsset(f"gitleaks_{version}_darwin_{arch}.tar.gz", False, BINARY_NAME)
    raise RuntimeError(f"unsupported platform: {system} {machine}")


@dataclass
class InstallResult:
    binary: str
    version: str
    installed: bool
    gitignore_updated: bool
    ok: bool
    errors: list[str] = field(default_factory=list)


def _binary_version(binary: Path) -> str | None:
    if not binary.exists():
        return None
    try:
        proc = subprocess.run(
            [str(binary), "version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else None


def _ensure_gitignore(repo_root: Path) -> bool:
    gitignore = repo_root / ".gitignore"
    marker = "/gitleaks"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if any(line.strip() == marker for line in existing.splitlines()):
        return False
    prefix = "" if existing.endswith("\n") else "\n"
    addition = f"{prefix}\n# Local pre-commit binary (downloaded, not committed)\n{marker}\n"
    with gitignore.open("a", encoding="utf-8") as handle:
        handle.write(addition)
    return True


def _extract_member(archive: Path, asset: PlatformAsset, repo_root: Path) -> Path:
    target = repo_root / asset.binary_member
    if asset.is_zip:
        with zipfile.ZipFile(archive) as archive_obj:
            archive_obj.extract(asset.binary_member, repo_root)
    else:
        with tarfile.open(archive, "r:gz") as archive_obj:
            member = archive_obj.getmember(asset.binary_member)
            try:
                archive_obj.extract(member, repo_root, filter="data")
            except TypeError:  # Python < 3.12 has no filter argument
                archive_obj.extract(member, repo_root)
    if platform.system().lower() != "windows":
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def install(repo_root: Path, version: str, force: bool) -> InstallResult:
    errors: list[str] = []
    binary = repo_root / BINARY_NAME
    try:
        current = _binary_version(binary)
        if current == version and not force:
            gitignore_updated = _ensure_gitignore(repo_root)
            return InstallResult(str(binary), version, False, gitignore_updated, True, errors)
        asset = detect_asset(version)
        url = f"{RELEASE_BASE}/v{version}/{asset.archive}"
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / asset.archive
            urllib.request.urlretrieve(url, archive_path)
            _extract_member(archive_path, asset, repo_root)
        gitignore_updated = _ensure_gitignore(repo_root)
        installed = True
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        return InstallResult(str(binary), version, False, False, False, errors)
    verified = _binary_version(binary)
    if verified != version:
        errors.append(f"post-install version mismatch: got {verified!r}, expected {version!r}")
        return InstallResult(str(binary), version, installed, gitignore_updated, False, errors)
    return InstallResult(str(binary), version, installed, gitignore_updated, True, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root (default: inferred from script location)",
    )
    parser.add_argument("--version", default=GITLEAKS_VERSION, help="gitleaks version to pin")
    parser.add_argument("--force", action="store_true", help="Reinstall even if already present")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    result = install(repo_root, args.version, args.force)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        status = "installed" if result.installed else "present"
        print(f"gitleaks {result.version}: {status} at {result.binary}")
        if result.gitignore_updated:
            print("gitignore: added /gitleaks")
        for error in result.errors:
            print(f"[ERROR] {error}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

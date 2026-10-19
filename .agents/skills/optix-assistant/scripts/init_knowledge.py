# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Initialize the OptiX Wiki knowledge submodule without unrelated documents."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path


SUBMODULE_NAME = "optix-assistant-knowledge"
SUBMODULE_PATH = Path(".agents/skills/optix-assistant/knowledge")
KNOWLEDGE_DIRS = ("case-knowledge", "modeling-diagnosis", "modeling-knowledge")
EXCLUDED_DIRS = ("design", "RFC")
MINIMUM_GIT_VERSION = (2, 26)
RMTREE_SUPPORTS_ONEXC = sys.version_info >= (3, 12)


class KnowledgeInitError(RuntimeError):
    """Raised when the knowledge checkout cannot satisfy its isolation contract."""


def _git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # nosec B607 - Git is the documented runtime dependency.
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise KnowledgeInitError(f"git {' '.join(args)} failed: {detail}")
    return result


def _git_version(repo_root: Path) -> tuple[int, int]:
    output = _git("version", cwd=repo_root).stdout.strip()
    match = re.search(r"(\d+)\.(\d+)", output)
    if not match:
        raise KnowledgeInitError(f"Cannot parse Git version: {output}")
    return int(match.group(1)), int(match.group(2))


def _gitlink_commit(repo_root: Path) -> str:
    output = _git("ls-tree", "HEAD", "--", SUBMODULE_PATH.as_posix(), cwd=repo_root).stdout.strip()
    fields = output.split()
    if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
        raise KnowledgeInitError(f"Missing gitlink for {SUBMODULE_PATH.as_posix()} at HEAD")
    return fields[2]


def _submodule_url(repo_root: Path) -> str:
    result = _git(
        "config",
        "-f",
        ".gitmodules",
        "--get",
        f"submodule.{SUBMODULE_NAME}.url",
        cwd=repo_root,
    )
    url = result.stdout.strip()
    if not url:
        raise KnowledgeInitError(f"Missing URL for submodule {SUBMODULE_NAME}")
    return url


def _register_submodule(repo_root: Path, url: str) -> None:
    """Register the submodule without invoking Git's shell-based helper."""
    _git("config", f"submodule.{SUBMODULE_NAME}.active", "true", cwd=repo_root)
    _git("config", f"submodule.{SUBMODULE_NAME}.url", url, cwd=repo_root)
    _git("config", f"submodule.{SUBMODULE_NAME}.update", "none", cwd=repo_root)


def _module_git_dir(repo_root: Path) -> Path:
    git_dir = Path(_git("rev-parse", "--git-dir", cwd=repo_root).stdout.strip())
    if not git_dir.is_absolute():
        git_dir = repo_root / git_dir
    return git_dir.resolve() / "modules" / SUBMODULE_NAME


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        callback_name = "onexc" if RMTREE_SUPPORTS_ONEXC else "onerror"
        shutil.rmtree(path, **{callback_name: _remove_readonly})


def _remove_readonly(function, path: str, _exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _write_relative_gitdir(knowledge_path: Path, module_git_dir: Path) -> None:
    relative_git_dir = os.path.relpath(module_git_dir, knowledge_path).replace(os.sep, "/")
    git_file = knowledge_path / ".git"
    git_file.chmod(stat.S_IREAD | stat.S_IWRITE)
    git_file.unlink()
    git_file.write_text(f"gitdir: {relative_git_dir}\n", encoding="utf-8")


def _is_checkout(path: Path) -> bool:
    return _git("rev-parse", "--is-inside-work-tree", cwd=path, check=False).returncode == 0


def _validate_partial_clone(knowledge_path: Path) -> None:
    promisor = _git("config", "--get", "remote.origin.promisor", cwd=knowledge_path, check=False).stdout.strip()
    clone_filter = _git(
        "config",
        "--get",
        "remote.origin.partialclonefilter",
        cwd=knowledge_path,
        check=False,
    ).stdout.strip()
    if promisor.lower() != "true" or clone_filter != "blob:none":
        raise KnowledgeInitError(
            "The existing knowledge submodule is a full clone. Rerun this initializer with --force "
            "to replace it; sparse checkout alone cannot undo downloaded document blobs."
        )


def _configure_sparse_checkout(knowledge_path: Path) -> None:
    _git("sparse-checkout", "init", "--cone", cwd=knowledge_path)
    _git("sparse-checkout", "set", *KNOWLEDGE_DIRS, cwd=knowledge_path)


def validate(repo_root: Path, expected_commit: str | None = None) -> None:
    """Validate the checkout, partial-clone filter, allowlist, and pinned commit."""
    knowledge_path = repo_root / SUBMODULE_PATH
    if not knowledge_path.is_dir() or not _is_checkout(knowledge_path):
        raise KnowledgeInitError("OptiX knowledge is not initialized; run this script without --check-only")
    _validate_partial_clone(knowledge_path)

    expected = expected_commit or _gitlink_commit(repo_root)
    actual = _git("rev-parse", "HEAD", cwd=knowledge_path).stdout.strip()
    if actual != expected:
        raise KnowledgeInitError(f"Knowledge commit mismatch: expected {expected}, found {actual}")

    sparse_dirs = {
        line.strip() for line in _git("sparse-checkout", "list", cwd=knowledge_path).stdout.splitlines() if line.strip()
    }
    if sparse_dirs != set(KNOWLEDGE_DIRS):
        raise KnowledgeInitError(f"Unexpected sparse checkout allowlist: {sorted(sparse_dirs)}")

    missing = [name for name in KNOWLEDGE_DIRS if not (knowledge_path / name).is_dir()]
    excluded = [name for name in EXCLUDED_DIRS if (knowledge_path / name).exists()]
    if missing or excluded:
        raise KnowledgeInitError(f"Knowledge isolation failed: missing={missing}, excluded_present={excluded}")


def initialize(repo_root: Path, force: bool = False) -> None:
    """Create or update an isolated, pinned checkout of the Wiki knowledge directories."""
    repo_root = repo_root.resolve()
    if _git_version(repo_root) < MINIMUM_GIT_VERSION:
        version = ".".join(str(part) for part in MINIMUM_GIT_VERSION)
        raise KnowledgeInitError(f"Git {version} or newer is required for partial clone and sparse checkout")

    expected = _gitlink_commit(repo_root)
    knowledge_path = repo_root / SUBMODULE_PATH
    url = _submodule_url(repo_root)
    _register_submodule(repo_root, url)
    module_git_dir = _module_git_dir(repo_root)

    if force:
        _remove_path(knowledge_path)
        _remove_path(module_git_dir)

    if knowledge_path.exists() and any(knowledge_path.iterdir()):
        if not _is_checkout(knowledge_path):
            raise KnowledgeInitError(f"{knowledge_path} exists but is not a Git checkout")
        _validate_partial_clone(knowledge_path)
    else:
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        module_git_dir.parent.mkdir(parents=True, exist_ok=True)
        _git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--no-tags",
            f"--separate-git-dir={module_git_dir}",
            url,
            str(knowledge_path),
            cwd=repo_root,
        )
        _write_relative_gitdir(knowledge_path, module_git_dir)

    _configure_sparse_checkout(knowledge_path)
    _git(
        "fetch",
        "--filter=blob:none",
        "--no-tags",
        "origin",
        expected,
        cwd=knowledge_path,
    )
    _git("checkout", "--detach", expected, cwd=knowledge_path)
    validate(repo_root, expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an incompatible existing knowledge checkout",
    )
    args = parser.parse_args()
    if args.check_only and args.force:
        parser.error("--force cannot be combined with --check-only")
    try:
        if args.check_only:
            validate(args.repo_root.resolve())
        else:
            initialize(args.repo_root, force=args.force)
    except KnowledgeInitError as exc:
        parser.error(str(exc))
    print("OptiX knowledge checkout is isolated and up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

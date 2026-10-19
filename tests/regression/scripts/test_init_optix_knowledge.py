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

"""Tests for the isolated OptiX knowledge submodule initializer."""

import importlib.util
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3] / ".agents" / "skills" / "optix-assistant" / "scripts" / "init_knowledge.py"
)
SPEC = importlib.util.spec_from_file_location("init_optix_knowledge", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def git(cwd: Path, *args: str, check: bool = True, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
        env=env,
        encoding="utf-8",
    )
    return result.stdout.strip()


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repositories(tmp_path: Path) -> tuple[Path, Path, str]:
    wiki = tmp_path / "wiki-source"
    wiki.mkdir()
    git(wiki, "init", "-b", "main")
    git(wiki, "config", "user.name", "Test User")
    git(wiki, "config", "user.email", "test@example.com")
    for directory in (*MODULE.KNOWLEDGE_DIRS, *MODULE.EXCLUDED_DIRS):
        path = wiki / directory
        path.mkdir()
        (path / "content.md").write_text(directory, encoding="utf-8")
    wiki_commit = commit_all(wiki, "initial wiki")

    bare = tmp_path / "wiki.git"
    git(tmp_path, "clone", "--bare", str(wiki), str(bare))
    git(bare, "config", "uploadpack.allowFilter", "true")
    git(bare, "config", "uploadpack.allowAnySHA1InWant", "true")

    parent = tmp_path / "parent"
    parent.mkdir()
    git(parent, "init", "-b", "master")
    git(parent, "config", "user.name", "Test User")
    git(parent, "config", "user.email", "test@example.com")
    (parent / ".gitmodules").write_text(
        '[submodule "optix-assistant-knowledge"]\n'
        f"\tpath = {MODULE.SUBMODULE_PATH.as_posix()}\n"
        f"\turl = {bare.as_uri()}\n"
        "\tbranch = main\n"
        "\tupdate = none\n",
        encoding="utf-8",
    )
    git(
        parent,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{wiki_commit},{MODULE.SUBMODULE_PATH.as_posix()}",
    )
    git(parent, "add", ".gitmodules")
    git(parent, "commit", "-m", "add knowledge gitlink")
    return parent, wiki, wiki_commit


def test_initializer_checks_out_only_knowledge_and_keeps_document_blobs_missing(
    repositories,
):
    parent, wiki, expected = repositories
    forbidden_blob = git(wiki, "rev-parse", "HEAD:design/content.md")

    MODULE.initialize(parent)

    checkout = parent / MODULE.SUBMODULE_PATH
    assert git(checkout, "rev-parse", "HEAD") == expected
    assert {path.name for path in checkout.iterdir() if path.name != ".git"} == set(MODULE.KNOWLEDGE_DIRS)
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    result = subprocess.run(
        ["git", "cat-file", "-e", forbidden_blob],
        cwd=checkout,
        env=env,
        capture_output=True,
    )
    assert result.returncode != 0
    gitdir_file = (checkout / ".git").read_text(encoding="utf-8")
    assert gitdir_file.startswith("gitdir: ")
    assert str(parent.resolve()) not in gitdir_file
    MODULE.validate(parent)


def test_initializer_supports_multiple_worktrees(repositories, tmp_path):
    parent, _, expected = repositories
    MODULE.initialize(parent)
    primary_git_dir = MODULE._module_git_dir(parent)

    worktree = tmp_path / "secondary-worktree"
    git(parent, "worktree", "add", "--detach", str(worktree))
    MODULE.initialize(worktree)

    checkout = worktree / MODULE.SUBMODULE_PATH
    assert git(checkout, "rev-parse", "HEAD") == expected
    assert MODULE._module_git_dir(worktree) != primary_git_dir
    MODULE.validate(worktree)


def test_normal_submodule_update_is_disabled(repositories):
    parent, _, _ = repositories

    git(parent, "submodule", "update", "--init", "--recursive")

    assert (
        git(
            parent,
            "config",
            "-f",
            ".gitmodules",
            "--get",
            "submodule.optix-assistant-knowledge.update",
        )
        == "none"
    )
    assert not (parent / MODULE.SUBMODULE_PATH).exists()


def test_existing_full_clone_is_rejected(repositories):
    parent, _, _ = repositories
    checkout = parent / MODULE.SUBMODULE_PATH
    checkout.parent.mkdir(parents=True, exist_ok=True)
    git(parent, "clone", MODULE._submodule_url(parent), str(checkout))

    with pytest.raises(MODULE.KnowledgeInitError, match="full clone"):
        MODULE.initialize(parent)

    MODULE.initialize(parent, force=True)
    MODULE.validate(parent)


def test_initializer_rejects_git_before_sparse_checkout_list(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "_git_version", lambda _repo_root: (2, 25))

    with pytest.raises(MODULE.KnowledgeInitError, match="Git 2.26 or newer"):
        MODULE.initialize(tmp_path)


@pytest.mark.parametrize(("supports_onexc", "callback_name"), [(False, "onerror"), (True, "onexc")])
def test_remove_path_uses_supported_rmtree_callback(monkeypatch, tmp_path, supports_onexc, callback_name):
    directory = tmp_path / "knowledge"
    directory.mkdir()
    rmtree = Mock()
    monkeypatch.setattr(MODULE, "RMTREE_SUPPORTS_ONEXC", supports_onexc)
    monkeypatch.setattr(MODULE.shutil, "rmtree", rmtree)

    MODULE._remove_path(directory)

    rmtree.assert_called_once_with(directory, **{callback_name: MODULE._remove_readonly})

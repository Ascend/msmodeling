"""Regression coverage for non-root umask vs local model path security."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from tensor_cast.core import model_source_security as security
from tests.helpers import local_model_path_permissions as perms


def _is_group_or_world_writable(path: Path) -> bool:
    return bool(path.lstat().st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def _require_enforceable_permission_bits(path: Path) -> None:
    """Skip when FS may ignore POSIX mode bits (9p/drvfs/fuse); Windows CI rare."""

    if not security._permission_bits_are_enforceable(path):
        pytest.skip("POSIX permission bits are not enforceable on this filesystem")


def test_umask_0002_tmp_model_dir_is_rejected_until_hardened(tmp_path: Path) -> None:
    _require_enforceable_permission_bits(tmp_path)
    previous = os.umask(0o002)
    try:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        assert _is_group_or_world_writable(model_dir)

        with pytest.raises(ValueError, match="must not be group- or world-writable"):
            security.validate_local_model_path(model_dir)

        perms.harden_local_model_tree(model_dir)
        assert not _is_group_or_world_writable(model_dir)
        assert not _is_group_or_world_writable(model_dir / "config.json")
        assert security.validate_local_model_path(model_dir) == model_dir.resolve()
    finally:
        os.umask(previous)


def test_apply_test_safe_umask_prevents_group_writable_new_dirs(tmp_path: Path) -> None:
    _require_enforceable_permission_bits(tmp_path)
    previous = os.umask(0o002)
    try:
        perms.apply_test_safe_umask()
        model_dir = tmp_path / "safe_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        assert not _is_group_or_world_writable(model_dir)
        assert not _is_group_or_world_writable(model_dir / "config.json")
        assert security.validate_local_model_path(model_dir) == model_dir.resolve()
    finally:
        os.umask(previous)


def test_harden_vendored_model_config_assets(tmp_path: Path) -> None:
    _require_enforceable_permission_bits(tmp_path)
    assets = tmp_path / "assets" / "model_config" / "toy"
    assets.mkdir(parents=True)
    (assets / "config.json").write_text("{}", encoding="utf-8")
    os.chmod(assets, 0o775)
    os.chmod(assets / "config.json", 0o664)

    hardened = perms.harden_vendored_model_config_assets(tmp_path)
    assert hardened == tmp_path / "assets" / "model_config"
    assert not _is_group_or_world_writable(assets)
    assert not _is_group_or_world_writable(assets / "config.json")


def test_harden_local_model_tree_skips_root_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text("{}", encoding="utf-8")
    link = tmp_path / "model_link"
    link.symlink_to(target)

    if security._permission_bits_are_enforceable(tmp_path):
        os.chmod(target, 0o775)
        assert _is_group_or_world_writable(target)

    assert perms.harden_local_model_tree(link) == link

    if security._permission_bits_are_enforceable(tmp_path):
        # Root symlink must not follow and rewrite the target tree.
        assert _is_group_or_world_writable(target)

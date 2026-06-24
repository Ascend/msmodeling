from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

from tensor_cast.core import model_source_security as security
from tensor_cast.model_config import RemoteSource


def _make_model_dir(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o700)
    config_path = path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config_path.chmod(0o600)
    return path


def test_local_model_source_is_normalized_to_absolute_path(tmp_path) -> None:
    model_dir = _make_model_dir(tmp_path / "model")

    source_info = security.normalize_model_source(str(model_dir), RemoteSource.huggingface)

    assert source_info.is_local_path
    assert source_info.model_id == str(model_dir.resolve())


def test_remote_model_id_prints_trust_remote_code_risk_once(capsys: pytest.CaptureFixture[str]) -> None:
    model_id = f"SecurityTest/RemoteCodeWarning-{uuid4()}"

    source_info = security.normalize_model_source(model_id, RemoteSource.huggingface)
    first_warning = capsys.readouterr().err
    security.normalize_model_source(model_id, RemoteSource.huggingface)
    second_warning = capsys.readouterr().err

    assert not source_info.is_local_path
    assert "trust_remote_code=True" in first_warning
    assert "does not provide security guarantees" in first_warning
    assert second_warning == ""


def test_remote_model_id_warning_is_thread_safe(capsys: pytest.CaptureFixture[str]) -> None:
    model_id = f"SecurityTest/ThreadedRemoteCodeWarning-{uuid4()}"

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: security.warn_remote_code_risk(model_id, RemoteSource.huggingface), range(32)))

    warning = capsys.readouterr().err
    assert warning.count(model_id) == 1


def test_local_model_path_rejects_symlinked_entry(tmp_path) -> None:
    model_dir = _make_model_dir(tmp_path / "model")
    target = tmp_path / "configuration_target.py"
    target.write_text("# local test target\n", encoding="utf-8")
    target.chmod(0o600)
    (model_dir / "configuration_link.py").symlink_to(target)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        security.validate_local_model_path(model_dir)


def test_normalize_local_model_source_fails_closed_for_unsafe_path(tmp_path) -> None:
    model_dir = _make_model_dir(tmp_path / "model")
    (model_dir / "config_link.json").symlink_to(model_dir / "config.json")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        security.normalize_model_source(str(model_dir), RemoteSource.huggingface)


def test_local_model_path_rejects_group_or_world_writable_permissions(monkeypatch, tmp_path) -> None:
    model_dir = _make_model_dir(tmp_path / "model")
    original_mode = model_dir.stat().st_mode
    monkeypatch.setattr(security, "_permission_bits_are_enforceable", lambda _: True)

    os.chmod(model_dir, 0o777)
    try:
        with pytest.raises(ValueError, match="must not be group- or world-writable"):
            security.validate_local_model_path(model_dir)
    finally:
        os.chmod(model_dir, original_mode)


def test_local_model_path_rechecks_after_successful_validation(monkeypatch, tmp_path) -> None:
    model_dir = _make_model_dir(tmp_path / "model")
    original_mode = model_dir.stat().st_mode
    monkeypatch.setattr(security, "_permission_bits_are_enforceable", lambda _: True)

    assert security.validate_local_model_path(model_dir) == model_dir.resolve()

    os.chmod(model_dir, 0o777)
    try:
        with pytest.raises(ValueError, match="must not be group- or world-writable"):
            security.validate_local_model_path(model_dir)
    finally:
        os.chmod(model_dir, original_mode)


def test_fuse_filesystem_does_not_enforce_synthetic_permission_bits(monkeypatch, tmp_path) -> None:
    model_dir = _make_model_dir(tmp_path / "model")
    original_mode = model_dir.stat().st_mode
    monkeypatch.setattr(security, "_find_mount_type", lambda _: "fuse.sshfs")

    os.chmod(model_dir, 0o777)
    try:
        assert security.validate_local_model_path(model_dir) == model_dir.resolve()
    finally:
        os.chmod(model_dir, original_mode)


def test_local_model_path_fails_closed_when_max_entries_exceeded(tmp_path) -> None:
    model_dir = _make_model_dir(tmp_path / "model")
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    tokenizer_path.chmod(0o600)

    with pytest.raises(ValueError, match="maximum validation entry count"):
        security.validate_local_model_path(model_dir, max_entries=1)


def test_local_model_path_fails_closed_when_max_depth_exceeded(tmp_path) -> None:
    model_dir = _make_model_dir(tmp_path / "model")
    nested_dir = model_dir / "level1"
    nested_dir.mkdir()
    nested_dir.chmod(0o700)
    nested_config_path = nested_dir / "config.json"
    nested_config_path.write_text("{}", encoding="utf-8")
    nested_config_path.chmod(0o600)

    with pytest.raises(ValueError, match="maximum validation depth"):
        security.validate_local_model_path(model_dir, max_depth=0)


def test_local_model_path_wraps_os_walk_errors(monkeypatch, tmp_path) -> None:
    model_dir = _make_model_dir(tmp_path / "model")
    blocked_path = model_dir / "blocked"

    def fake_walk(root, followlinks=False, onerror=None):
        error = OSError("permission denied")
        error.filename = os.fspath(blocked_path)
        onerror(error)
        yield from ()

    monkeypatch.setattr(security.os, "walk", fake_walk)

    with pytest.raises(ValueError, match=f"Failed to inspect local model path entry: {blocked_path}") as exc_info:
        security.validate_local_model_path(model_dir)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert exc_info.value.__cause__.filename == os.fspath(blocked_path)

"""Tests for build wheel directory helpers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from scripts.helpers.build.run_build import _clear_wheel_output_dir, _newest_wheel

if TYPE_CHECKING:
    from pathlib import Path


def test_clear_wheel_output_dir_removes_only_msmodeling_wheels(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    (wheel_dir / "msmodeling-0.2.0-py3-none-any.whl").write_bytes(b"a")
    (wheel_dir / "msmodeling-1.0.0-py3-none-any.whl").write_bytes(b"b")
    (wheel_dir / "other.pkg").write_bytes(b"c")
    _clear_wheel_output_dir(wheel_dir)
    assert list(wheel_dir.glob("msmodeling-*.whl")) == []
    assert (wheel_dir / "other.pkg").is_file()


def test_newest_wheel_uses_mtime_not_lexicographic_order(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    older = wheel_dir / "msmodeling-1.0.0-py3-none-any.whl"
    newer = wheel_dir / "msmodeling-0.2.0-py3-none-any.whl"
    older.write_bytes(b"old")
    time.sleep(0.01)
    newer.write_bytes(b"new")
    assert _newest_wheel(wheel_dir) == newer


def test_newest_wheel_empty_dir_returns_none(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    assert _newest_wheel(wheel_dir) is None

"""Make local model trees compliant with ``model_source_security`` checks.

Git does not store directory modes. With a common non-root umask of ``0002``,
checkout and ``tmp_path``/``mkdtemp`` trees often land as group-writable
(``775``/``664``). Production validation correctly rejects those modes; tests
must create or repair paths so they stay owner-writable only.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

# Clear group/world write while preserving other bits (execute, sticky, etc.).
_INSECURE_WRITE_MASK = stat.S_IWGRP | stat.S_IWOTH
# Match a typical CI/developer umask that yields 755/644 instead of 775/664.
TEST_SAFE_UMASK = 0o022


def apply_test_safe_umask() -> int:
    """Set process umask so new files/dirs are not group- or world-writable.

    Returns the previous umask (same contract as ``os.umask``).
    """

    return os.umask(TEST_SAFE_UMASK)


def harden_local_model_tree(path: str | Path) -> Path:
    """Clear group/world-write bits on ``path`` and, if a directory, its tree.

    Symlinks are skipped, including when ``path`` itself is a symlink
    (``Path.chmod`` follows links and must not rewrite the target). Missing
    paths raise ``FileNotFoundError``.
    """

    root = Path(path)
    if not root.exists() and not root.is_symlink():
        raise FileNotFoundError(f"Local model path does not exist: {root}")
    if root.is_symlink():
        return root

    _clear_insecure_write_bits(root)
    if root.is_dir():
        for child in root.rglob("*"):
            if child.is_symlink():
                continue
            if child.is_file() or child.is_dir():
                _clear_insecure_write_bits(child)
    return root


def harden_vendored_model_config_assets(tests_root: str | Path | None = None) -> Path | None:
    """Harden ``tests/assets/model_config`` after checkout under a loose umask."""

    root = Path(tests_root) if tests_root is not None else Path(__file__).resolve().parents[1]
    assets = root / "assets" / "model_config"
    if not assets.is_dir():
        return None
    return harden_local_model_tree(assets)


def _clear_insecure_write_bits(path: Path) -> None:
    mode = path.lstat().st_mode
    if mode & _INSECURE_WRITE_MASK:
        path.chmod(stat.S_IMODE(mode) & ~_INSECURE_WRITE_MASK)

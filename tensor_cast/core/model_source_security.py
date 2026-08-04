"""Security checks and user warnings for model source inputs."""

from __future__ import annotations

import logging
import os
import stat
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_INSECURE_WRITE_MASK = stat.S_IWGRP | stat.S_IWOTH
_SYNTHETIC_PERMISSION_FS_TYPES = {"9p", "drvfs", "fuse", "fuseblk", "v9fs"}
_MAX_LOCAL_MODEL_PATH_DEPTH = 10
_MAX_LOCAL_MODEL_PATH_ENTRIES = 10000
_warned_remote_model_ids: set[tuple[str, str]] = set()
_warned_permission_filesystems: set[str] = set()
_warned_remote_model_ids_lock = threading.Lock()
_warned_permission_filesystems_lock = threading.Lock()


@dataclass(frozen=True)
class ModelSourceInfo:
    """Normalized model source information."""

    model_id: str
    is_local_path: bool


def normalize_model_source(
    model_id: str,
    remote_source: str,
    *,
    validate_local: bool = True,
    warn_remote: bool = True,
) -> ModelSourceInfo:
    """Normalize a model source and apply local-mode safety checks.

    Existing filesystem paths are treated as local safe-mode inputs and are
    normalized to absolute real paths. Non-existing inputs are treated as model
    ids for Hugging Face or ModelScope and receive an explicit remote-code risk
    warning.

    Raises:
        OSError, ValueError: If an existing local path cannot be resolved or
            fails safe-mode validation. This is intentional fail-closed
            behavior; callers must not fall back to the original path.

    Thread Safety:
        This function is thread-safe. Internal locks protect de-duplicated
        warnings from concurrent callers.
    """

    if not model_id:
        return ModelSourceInfo(model_id=model_id, is_local_path=False)

    candidate = Path(model_id).expanduser()
    if candidate.exists() or candidate.is_symlink():
        resolved = validate_local_model_path(candidate) if validate_local else candidate.resolve(strict=True)
        return ModelSourceInfo(model_id=str(resolved), is_local_path=True)

    if warn_remote:
        warn_remote_code_risk(model_id, remote_source)
    return ModelSourceInfo(model_id=model_id, is_local_path=False)


def warn_remote_code_risk(model_id: str, remote_source: str) -> None:
    """Print a once-per-model warning for remote model id mode."""

    key = (str(remote_source), model_id)
    with _warned_remote_model_ids_lock:
        if key in _warned_remote_model_ids:
            return
        _warned_remote_model_ids.add(key)
    print(
        "[msmodeling security] model_id mode is enabled for "
        f"{model_id!r} from {remote_source!s}. Loading Hugging Face or ModelScope model ids may execute "
        "remote Python code when transformers falls back to trust_remote_code=True. msmodeling does not "
        "provide security guarantees for remote model code; prefer safe local mode with a reviewed absolute "
        "model path.",
        file=sys.stderr,
    )


def validate_local_model_path(
    path: str | Path,
    *,
    max_depth: int = _MAX_LOCAL_MODEL_PATH_DEPTH,
    max_entries: int = _MAX_LOCAL_MODEL_PATH_ENTRIES,
) -> Path:
    """Validate a local model path and return its absolute resolved path.

    The local mode rejects symlinked paths, symlinked entries inside the model
    tree, unexpected owners, and group/world-writable permission bits when the
    filesystem exposes reliable POSIX mode bits. Directory validation stops
    early and fails closed when ``max_depth`` or ``max_entries`` is exceeded.

    On non-POSIX platforms without ``os.geteuid()``, owner validation is not
    available; symlink and permission-bit checks still run where the platform
    exposes those attributes.
    """

    expanded = Path(path).expanduser()
    if not expanded.exists() and not expanded.is_symlink():
        raise FileNotFoundError(f"Local model path does not exist: {expanded}")

    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    return _validate_local_model_path(os.fspath(absolute), max_depth, max_entries)


def _validate_local_model_path(absolute_path: str, max_depth: int, max_entries: int) -> Path:
    """Validate an absolute local model path."""

    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}")
    if max_entries < 1:
        raise ValueError(f"max_entries must be at least 1, got {max_entries}")

    absolute = Path(absolute_path)
    _validate_no_symlink_path_components(absolute)

    resolved = absolute.resolve(strict=True)
    enforce_permission_bits = _permission_bits_are_enforceable(resolved)
    _validate_local_entry(resolved, enforce_permission_bits=enforce_permission_bits)

    if resolved.is_dir():
        entry_count = 0
        for root, dir_names, file_names in os.walk(resolved, followlinks=False, onerror=_raise_walk_error):
            root_path = Path(root)
            relative_root = root_path.relative_to(resolved)
            current_depth = len(relative_root.parts)
            if current_depth > max_depth or (current_depth == max_depth and dir_names):
                raise ValueError(
                    "Local model path exceeds maximum validation depth "
                    f"({max_depth}). Please use a reviewed model directory with a shallower structure: {root_path}"
                )
            entry_count += len(dir_names) + len(file_names)
            if entry_count > max_entries:
                raise ValueError(
                    "Local model path exceeds maximum validation entry count "
                    f"({max_entries}). Please use a reviewed model directory with fewer files: {resolved}"
                )
            _validate_local_entry(root_path, enforce_permission_bits=enforce_permission_bits)
            for entry_name in dir_names + file_names:
                _validate_local_entry(root_path / entry_name, enforce_permission_bits=enforce_permission_bits)

    return resolved


def _validate_no_symlink_path_components(path: Path) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Local model path must not contain symlinks: {current}")


def _raise_walk_error(error: OSError) -> None:
    raise ValueError(f"Failed to inspect local model path entry: {error.filename}") from error


def _validate_local_entry(path: Path, *, enforce_permission_bits: bool) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"Failed to inspect local model path entry: {path}") from exc

    mode = metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"Local model path must not contain symlinks: {path}")
    if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
        raise ValueError(f"Local model path may only contain regular files and directories: {path}")

    _validate_owner(path, metadata.st_uid)
    if enforce_permission_bits and mode & _INSECURE_WRITE_MASK:
        raise ValueError(
            f"Local model path entries must not be group- or world-writable. Please fix permissions for: {path}"
        )


def _validate_owner(path: Path, owner_uid: int) -> None:
    if not hasattr(os, "geteuid"):
        return

    current_uid = os.geteuid()
    allowed_uids = {current_uid}
    if current_uid != 0:
        allowed_uids.add(0)
    if owner_uid not in allowed_uids:
        raise ValueError(
            "Local model path entries must be owned by the current user"
            f"{' or root' if current_uid != 0 else ''}: {path}"
        )


def _permission_bits_are_enforceable(path: Path) -> bool:
    fs_type = _find_mount_type(path)
    if not _permission_bits_may_be_synthetic(fs_type):
        return True

    should_warn = False
    with _warned_permission_filesystems_lock:
        if fs_type not in _warned_permission_filesystems:
            _warned_permission_filesystems.add(fs_type)
            should_warn = True
    if should_warn:
        logger.warning(
            "Local model path is on filesystem type %s, whose POSIX permission bits may be synthetic. "
            "Owner and symlink checks are still enforced, but group/world-writable mode bits are not rejected.",
            fs_type,
        )
    return False


def _permission_bits_may_be_synthetic(fs_type: str | None) -> bool:
    if fs_type is None:
        return False
    return fs_type in _SYNTHETIC_PERMISSION_FS_TYPES or fs_type.startswith("fuse.")


def _find_mount_type(path: Path) -> str | None:
    mounts_path = Path("/proc/mounts")
    if not mounts_path.is_file():
        return None

    target = os.path.abspath(os.fspath(path))
    best_mount = ""
    best_fs_type: str | None = None
    try:
        with mounts_path.open(encoding="utf-8") as mounts_file:
            for line in mounts_file:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = _decode_mount_field(parts[1])
                try:
                    common_path = os.path.commonpath([target, mount_point])
                except ValueError:
                    continue
                if common_path == mount_point and len(mount_point) > len(best_mount):
                    best_mount = mount_point
                    best_fs_type = parts[2]
    except OSError:
        return None
    return best_fs_type


def _decode_mount_field(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")

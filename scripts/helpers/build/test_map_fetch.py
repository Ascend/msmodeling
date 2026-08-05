"""Resolve and download the external test_map used by CI gate."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final

from scripts.helpers.defaults import build_test_map_url, default_test_map_path

logger: Final = logging.getLogger("build.test_map")

# Small JSON over OBS; hang forever is worse than failing the gate.
_DOWNLOAD_TIMEOUT_SECONDS: Final = 30


class MapFetchError(Exception):
    """Raised when test_map cannot be resolved or downloaded."""


def resolve_test_map_path(
    *,
    configured: str | None,
    base_branch: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a usable test_map file path.

    If *configured* points to an existing file with valid JSON, use it.
    Otherwise warn and use the branch-scoped cache path under
    ``{cache}/test_map/{base_branch}/test_map.json`` (download when missing
    or corrupt). Download failure raises :class:`MapFetchError`.
    """
    raw = configured.strip() if configured else ""
    if raw:
        path = Path(raw)
        if path.is_file():
            try:
                _validate_test_map_bytes(path.read_bytes(), source=str(path))
            except MapFetchError as exc:
                logger.warning("%s; re-downloading for base branch %r.", exc, base_branch)
            else:
                logger.info("Using test_map at %s", path)
                return path
        else:
            logger.warning(
                "MSMODELING_TEST_MAP_PATH=%s is not a readable file; downloading test_map for base branch %r instead.",
                raw,
                base_branch,
            )
    else:
        logger.warning(
            "MSMODELING_TEST_MAP_PATH is unset; downloading test_map for base branch %r.",
            base_branch,
        )

    dest = default_test_map_path(cache_dir=cache_dir, base_branch=base_branch)
    if not dest.is_absolute():
        from scripts.helpers._paths import REPO_ROOT

        dest = REPO_ROOT / dest

    if dest.is_file():
        try:
            _validate_test_map_bytes(dest.read_bytes(), source=str(dest))
        except MapFetchError as exc:
            logger.warning("%s; re-downloading.", exc)
        else:
            logger.info("Using cached test_map at %s", dest)
            return dest

    download_test_map(base_branch=base_branch, dest=dest)
    return dest


def download_test_map(*, base_branch: str, dest: Path) -> None:
    """Download test_map for *base_branch* to *dest*.

    Uses ``_DOWNLOAD_TIMEOUT_SECONDS`` (30s) as the socket timeout. Writes via
    ``*.tmp`` then replaces. Tmp is removed on failure; a killed process may
    leave a stale ``*.tmp`` (safe to delete manually).
    """
    url = build_test_map_url(base_branch)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    logger.info("Downloading test_map from %s → %s", url, dest)
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            body = response.read()
        _validate_test_map_bytes(body, source=url)
        tmp.write_bytes(body)
        tmp.replace(dest)
    except MapFetchError:
        _cleanup_tmp(tmp)
        raise
    except urllib.error.HTTPError as exc:
        _cleanup_tmp(tmp)
        raise MapFetchError(f"failed to download test_map from {url}: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        _cleanup_tmp(tmp)
        raise MapFetchError(f"failed to download test_map from {url}: {exc.reason}") from exc
    except OSError as exc:
        _cleanup_tmp(tmp)
        raise MapFetchError(
            f"failed to download test_map from {url} or write to {dest}: {exc}",
        ) from exc
    logger.info("test_map ready at %s (%d bytes)", dest, dest.stat().st_size)


def _validate_test_map_bytes(body: bytes, *, source: str) -> None:
    if not body:
        raise MapFetchError(f"test_map is empty: {source}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MapFetchError(f"test_map is not valid UTF-8 JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise MapFetchError(f"test_map JSON root must be an object: {source}")


def _cleanup_tmp(tmp: Path) -> None:
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        return

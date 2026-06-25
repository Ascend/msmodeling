"""Shared helpers for materializing lightweight model Hub snapshots."""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

CONFIG_JSON_ALLOW_PATTERNS = [
    "config.json",
    "**/config.json",
    "preprocessor_config.json",
]

# Weight-like artifacts to skip when a Hub client does not support a stricter
# config-only allowlist. Keep this in sync with transformer ModelScope loading.
MODELSCOPE_WEIGHT_IGNORE_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.npz",
    "*.onnx",
    "*.gguf",
    "*.zip",
    "*.tar",
    "*.tar.gz",
]

_CONFIG_ONLY_ALLOWLIST_ERROR = "ModelScope snapshot_download does not support a config-only allowlist"
_WEIGHT_IGNORELIST_ERROR = "ModelScope snapshot_download does not support a weight ignorelist"


def snapshot_huggingface_config_only(model_id: str) -> str:
    """Download only config.json files from a Hugging Face Hub model repo."""
    from huggingface_hub import snapshot_download

    return _call_snapshot_download_silently(
        snapshot_download,
        repo_id=model_id,
        allow_patterns=CONFIG_JSON_ALLOW_PATTERNS,
    )


def snapshot_modelscope_config_only(model_id: str) -> str:
    """Download only config.json files from a ModelScope model repo.

    Older ModelScope versions used ``allow_file_pattern`` while newer versions
    may support Hugging Face-style ``allow_patterns``. Refuse to call without an
    allowlist so this helper never falls back to a full repository download.
    """
    from modelscope import snapshot_download

    return _snapshot_with_patterns(
        snapshot_download,
        model_id,
        ("allow_patterns", "allow_file_pattern"),
        CONFIG_JSON_ALLOW_PATTERNS,
        _CONFIG_ONLY_ALLOWLIST_ERROR,
    )


def snapshot_modelscope_without_weights(model_id: str) -> str:
    """Download a ModelScope model repo while skipping common weight files."""
    from modelscope import snapshot_download

    return _snapshot_with_patterns(
        snapshot_download,
        model_id,
        ("ignore_patterns", "ignore_file_pattern"),
        MODELSCOPE_WEIGHT_IGNORE_PATTERNS,
        _WEIGHT_IGNORELIST_ERROR,
    )


def _call_snapshot_download_silently(snapshot_download: Callable[..., str], *args: Any, **kwargs: Any) -> str:
    with _suppress_snapshot_download_output():
        return snapshot_download(*args, **kwargs)


@contextlib.contextmanager
def _suppress_snapshot_download_output() -> Iterator[None]:
    old_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with (
            open(os.devnull, "w", encoding="utf-8") as devnull,
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):
            yield
    finally:
        logging.disable(old_disable_level)


def _snapshot_with_patterns(
    snapshot_download: Callable[..., str],
    model_id: str,
    candidate_keywords: Sequence[str],
    patterns: list[str],
    unsupported_message: str,
) -> str:
    has_signature, keyword = _select_supported_keyword(snapshot_download, candidate_keywords)
    if has_signature:
        if keyword is None:
            raise RuntimeError(unsupported_message)
        return _call_snapshot_download_silently(snapshot_download, model_id, **{keyword: patterns})

    last_type_error: TypeError | None = None
    for keyword in candidate_keywords:
        try:
            return _call_snapshot_download_silently(snapshot_download, model_id, **{keyword: patterns})
        except TypeError as exc:
            if keyword not in str(exc):
                raise
            last_type_error = exc

    raise RuntimeError(unsupported_message) from last_type_error


def _select_supported_keyword(
    func: Callable[..., Any],
    candidate_keywords: Sequence[str],
) -> tuple[bool, str | None]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False, None

    parameters = signature.parameters
    for keyword in candidate_keywords:
        if keyword in parameters:
            return True, keyword

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return True, None

    return True, None

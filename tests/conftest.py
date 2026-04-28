"""
Pytest hooks for the test suite.

After the session ends, optionally remove hub weight shards under ~/.cache while
keeping config and Python sources (insurance if a dependency still fetched weights).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_HUB_ROOTS = (
    "~/.cache/modelscope/hub",
    "~/.cache/huggingface/hub",
)

_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".onnx",
    ".gguf",
    ".npz",
    ".zip",
    ".tar",
    ".tar.gz",
)


def _is_hub_weight_file(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(".safetensors.index.json"):
        return True
    return any(lower.endswith(suf) for suf in _WEIGHT_SUFFIXES)


def _prune_hub_weight_files() -> int:
    removed = 0
    for rel in _HUB_ROOTS:
        root = os.path.expanduser(rel)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
            for fn in filenames:
                if not _is_hub_weight_file(fn):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    logger.exception("Could not remove hub weight file %s", path)
    if removed:
        logger.info(
            "Pruned %s hub weight file(s) under ~/.cache/*/hub (config and code kept).",
            removed,
        )
    return removed


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    flag = os.environ.get("TENSOR_CAST_PRUNE_HUB_WEIGHTS_AFTER_UT", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return
    _prune_hub_weight_files()

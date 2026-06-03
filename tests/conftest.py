"""
Pytest hooks for the test suite.

Hub access: default online. Set ``MSMODELING_OFFLINE=1`` to enable
``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` / ``HF_DATASETS_OFFLINE``.
See tests/README.md and docs/design/ut_refactor.md.

After the session ends, optionally remove hub weight shards under the repo-local
``.msmodeling_cache`` while keeping config and Python sources.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

pytest_plugins = (
    "tests.regression.tensor_cast.conftest",
    "tests.regression.serving_cast.conftest",
)

_REPO_CACHE = Path.cwd() / ".msmodeling_cache"


def _resolve_cache_dir() -> Path:
    raw = os.environ.get("MSMODELING_CACHE", "").strip()
    if not raw:
        return _REPO_CACHE
    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


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


def _prune_hub_weight_files(root: Path) -> int:
    removed = 0
    if not root.is_dir():
        return removed
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not _is_hub_weight_file(path.name):
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            logger.exception("Could not remove hub weight file %s", path)
    if removed:
        logger.info("Pruned %s hub weight file(s) under %s", removed, root)
    return removed


def _msmodeling_offline_enabled() -> bool:
    flag = os.environ.get("MSMODELING_OFFLINE", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def _apply_hub_offline_env() -> None:
    """Single switch for Hugging Face / Transformers / Datasets offline mode."""
    if _msmodeling_offline_enabled():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "0")


def _cache_dir_configured() -> bool:
    return bool(os.environ.get("MSMODELING_CACHE", "").strip())


def pytest_sessionstart(session) -> None:
    _apply_hub_offline_env()
    if not _cache_dir_configured():
        return
    cache_dir = _resolve_cache_dir()
    cache = str(cache_dir)
    os.environ.setdefault("TORCH_HOME", cache)
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("MODELSCOPE_CACHE", cache)


def _weights_prune_enabled() -> bool:
    raw = os.environ.get("MSMODELING_TEST_WEIGHTS_PRUNE")
    if raw is None or not raw.strip():
        raw = os.environ.get("TENSOR_CAST_PRUNE_HUB_WEIGHTS_AFTER_UT", "0")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def pytest_sessionfinish(session, exitstatus) -> None:
    if not _weights_prune_enabled():
        return
    if not _cache_dir_configured():
        return
    _prune_hub_weight_files(_resolve_cache_dir())


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed ``random`` and ``torch`` before every test for determinism."""
    import random

    import torch

    random.seed(0)
    torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot os.environ per test and restore it afterwards.

    The snapshot is taken after ``pytest_sessionstart`` set the session-level hub
    env, so session env is part of the snapshot and preserved across tests.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


@pytest.fixture(scope="session")
def cfg_registry(model_zoo) -> dict:
    """Alias-resolving session view over the shared ``model_cache`` config cache."""
    from tests.helpers.model_cache import get_hf_config

    class _CfgRegistry(dict):
        def __init__(self, alias_to_model_id: dict[str, str]):
            super().__init__()
            self._alias_to_model_id = alias_to_model_id

        def _normalize_model_id(self, key: str) -> str:
            model_id = self._alias_to_model_id.get(key, key)
            if "/" not in model_id:
                aliases = ", ".join(sorted(self._alias_to_model_id))
                raise KeyError(f"Unknown model alias '{key}'. Available aliases: {aliases}")
            return model_id

        def __getitem__(self, key):
            model_id = self._normalize_model_id(key)
            config = get_hf_config(model_id)
            dict.__setitem__(self, model_id, config)
            return config

        def __setitem__(self, key, value):
            model_id = self._normalize_model_id(key)
            dict.__setitem__(self, model_id, value)

        def get(self, key, default=None):
            try:
                return self[key]
            except KeyError:
                return default

    return _CfgRegistry(model_zoo)


@pytest.fixture(scope="session")
def device() -> str:
    """Default test device profile name."""
    return "TEST_DEVICE"


@pytest.fixture(scope="session")
def model_zoo() -> dict[str, str]:
    """Canonical model aliases used by regression fixtures."""
    return {
        "deepseek_v32": "deepseek-ai/DeepSeek-V3.2",
        "qwen3_32b": "Qwen/Qwen3-32B",
        "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    }

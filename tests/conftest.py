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
import re
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

pytest_plugins = (
    "tests.regression.tensor_cast.conftest",
    "tests.regression.serving_cast.conftest",
)

_REPO_CACHE = Path.cwd() / ".msmodeling_cache"

_MODEL_ALIASES = {
    "deepseek_v32": "deepseek-ai/DeepSeek-V3.2",
    "qwen3_32b": "Qwen/Qwen3-32B",
    "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
}


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


def _patch_config_loader_for_pr_gate() -> None:
    """Skip tests on PR CI gates whose remote model weights are not cached.

    ``ModelRunner`` / per-file helpers / session fixtures all resolve models
    through ``AutoModelConfigLoader``; raising ``pytest.skip`` at that
    chokepoint covers every runtime path (attribute-held or hardcoded model
    ids) that collection-time parameter inspection cannot see. Production
    behavior is untouched: the patch is installed only when ``CI_PR_UT=1``.
    """
    from tensor_cast.transformers.utils import AutoModelConfigLoader

    original_load_config = AutoModelConfigLoader.load_config
    original_try_to_load_model = AutoModelConfigLoader.try_to_load_model

    def _guarded_model_id(model_id) -> str | None:
        return _resolve_hub_id(str(model_id)) if isinstance(model_id, str) else None

    def load_config(self, model_id, *args, **kwargs):
        hub_id = _guarded_model_id(model_id)
        if hub_id is not None and not _hub_weights_cached(hub_id):
            pytest.skip(_PR_GATE_SKIP_REASON)
        return original_load_config(self, model_id, *args, **kwargs)

    def try_to_load_model(*args, **kwargs):
        hf_config = next((a for a in args if hasattr(a, "name_or_path")), kwargs.get("hf_config"))
        name_or_path = getattr(hf_config, "name_or_path", None)
        hub_id = _guarded_model_id(name_or_path)
        if hub_id is not None and not _hub_weights_cached(hub_id):
            pytest.skip(_PR_GATE_SKIP_REASON)
        return original_try_to_load_model(*args, **kwargs)

    AutoModelConfigLoader.load_config = load_config
    AutoModelConfigLoader.try_to_load_model = staticmethod(try_to_load_model)
    logger.info("PR CI gate active: uncached remote model tests will be skipped")


def pytest_sessionstart(session) -> None:
    if _pr_ci_gate_active():
        # PR CI runners have no live Hub access: force offline so any
        # download attempt fails fast (seconds) instead of stalling on
        # connect timeouts for hours. The makereport hook below downgrades
        # the resulting failures to skips.
        os.environ["MSMODELING_OFFLINE"] = "1"
    _apply_hub_offline_env()
    if _pr_ci_gate_active():
        _patch_config_loader_for_pr_gate()
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


def _pr_ci_gate_active() -> bool:
    return os.environ.get("CI_PR_UT", "").strip() == "1"


_PR_GATE_SKIP_REASON = (
    "PR CI gate: remote Hub model weights are not cached on this runner; "
    "nightly runs with weight prefetch still cover this case"
)

_HUB_ID_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")


def _resolve_hub_id(value: str) -> str | None:
    """Return the Hub repo id for a model-like parameter value, else ``None``."""
    resolved = _MODEL_ALIASES.get(value, value)
    if _HUB_ID_RE.match(resolved) and not os.path.exists(resolved):
        return resolved
    return None


def _hub_weights_cached(repo_id: str) -> bool:
    """True only when actual weight shards (not just configs) are cached."""
    hf_home = os.environ.get("HF_HOME")
    hf_base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    hf_snapshots = hf_base / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if _dir_contains_weight_file(hf_snapshots):
        return True
    ms_cache = os.environ.get("MODELSCOPE_CACHE")
    ms_base = Path(ms_cache) if ms_cache else Path.home() / ".cache" / "modelscope"
    org, _, name = repo_id.partition("/")
    return _dir_contains_weight_file(ms_base / org / name)


def _dir_contains_weight_file(root: Path) -> bool:
    if not root.is_dir():
        return False
    for path in root.rglob("*"):
        if path.is_file() and _is_hub_weight_file(path.name):
            return True
    return False


def _iter_param_values(item: pytest.Item):
    """Yield parameter values for both pytest parametrize and parameterized.expand."""
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        yield from callspec.params.values()
        return
    func = getattr(item, "obj", None)
    func = getattr(func, "__func__", func)
    for cell in getattr(func, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if type(value).__name__ == "param":
            for arg in getattr(value, "args", ()) or ():
                yield arg
            for kwarg in getattr(value, "kwargs", {}).values():
                yield kwarg


def pytest_collection_modifyitems(config, items):
    """Skip remote-weight model tests in PR CI gates when weights are missing.

    PR CI runners do not prefetch weights for the whole suite and have no live
    Hub access; uncached model tests would otherwise fail on downloads. Skip
    them instead in the PR gate (``CI_PR_UT=1``); nightly runs with weight
    prefetch still execute them. Behavior is unchanged elsewhere.
    """
    if not _pr_ci_gate_active():
        return
    skip_marker = pytest.mark.skip(reason=_PR_GATE_SKIP_REASON)
    for item in items:
        for value in _iter_param_values(item):
            if isinstance(value, str):
                hub_id = _resolve_hub_id(value)
                if hub_id is not None and not _hub_weights_cached(hub_id):
                    item.add_marker(skip_marker)
                    break


_HUB_UNREACHABLE_PATTERNS = (
    "OfflineModeIsEnabled",
    "offline mode is enabled",
    "HFValidationError",
    "LocalEntryNotFoundError",
    "huggingface.co",
    "hf-mirror.com",
    "HfHubHTTPError",
    "requests.exceptions.ConnectionError",
    "requests.exceptions.ConnectTimeout",
    "requests.exceptions.ReadTimeout",
    "urllib.error.URLError",
    "Connection refused",
    "Connection reset by peer",
    "Read timed out",
    "network is unreachable",
    "Temporary failure in name resolution",
    "Errno -3",
    "Errno 101",
    "Errno 13] Permission denied",  # hub cache permission failures
    ".cache/huggingface",
    ".cache/modelscope",
)


def _is_hub_unreachable_failure(report) -> bool:
    longrepr = getattr(report, "longrepr", None)
    text = str(getattr(longrepr, "reprcrash", None) and longrepr.reprcrash.message or longrepr or "")
    if not text:
        return False
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in _HUB_UNREACHABLE_PATTERNS)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Downgrade Hub-unreachable failures to skips inside PR CI gates.

    Covers download paths that bypass the guarded loader (tokenizers,
    snapshot downloads, direct AutoConfig calls): with forced offline mode
    those fail in seconds; report them as skips so the PR gate stays green,
    leaving live-Hub coverage to nightly runs.
    """
    outcome = yield
    if not _pr_ci_gate_active() or call.when != "call":
        return
    report = outcome.get_result()
    if report.failed and _is_hub_unreachable_failure(report):
        report.outcome = "skipped"
        report.longrepr = _PR_GATE_SKIP_REASON


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
    return dict(_MODEL_ALIASES)

"""Load and validate test_map JSON and gate policy baseline."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.ci_gate.gate_policy import load_gate_policy
from scripts.helpers.ci_gate.models import Baseline
from scripts.helpers.common.coverage_config import product_roots
from scripts.helpers.common.test_map_config import resolve_test_map_path

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_map_keys(
    inner: dict[str, object],
    roots: tuple[str, ...],
) -> dict[str, dict[str, list[str]]]:
    validated: dict[str, dict[str, list[str]]] = {}
    for key, symbols in inner.items():
        if not isinstance(key, str):
            raise ConfigError(f"test_map: map key must be string, got {type(key).__name__}")
        if ".." in key or key.startswith("/"):
            raise ConfigError(f"test_map: invalid map key: {key!r}")
        if not any(key.startswith(prefix) for prefix in roots):
            raise ConfigError(f"test_map: map key must start with a product root ({', '.join(roots)}): {key!r}")
        if not isinstance(symbols, dict):
            raise ConfigError(f"test_map: value for {key!r} must be object")
        sym_map: dict[str, list[str]] = {}
        for symbol, test_ids in symbols.items():
            if not isinstance(symbol, str) or not isinstance(test_ids, list):
                raise ConfigError(f"test_map: invalid symbol entry under {key!r}")
            if not all(isinstance(tid, str) for tid in test_ids):
                raise ConfigError(f"test_map: test node ids must be strings under {key!r}::{symbol}")
            sym_map[symbol] = list(test_ids)
        validated[key] = sym_map
    return validated


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_test_map(
    cfg: Config,
    *,
    roots: tuple[str, ...] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Load and validate test_map JSON from the path given in *cfg*."""
    resolved_roots = roots if roots is not None else product_roots()
    map_path = resolve_test_map_path(cfg, must_exist=True)
    try:
        data = json.loads(map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"test_map: invalid JSON at {map_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("test_map: root must be object")
    if data.get("schema_version") != 1:
        raise ConfigError("test_map: schema_version must be 1")
    inner = data.get("map")
    if not isinstance(inner, dict):
        raise ConfigError("test_map: map must be object")
    return _validate_map_keys(inner, resolved_roots)


def load_baseline(repo_root: Path, cfg: Config) -> Baseline:
    """Load full gate baseline: test_map + gate policy + product roots."""
    policy = load_gate_policy(repo_root)
    return Baseline(
        test_map=load_test_map(cfg, roots=policy.roots),
        exemptions=policy.source_exemptions,
        test_exemptions=policy.test_exemptions,
        discovery=policy.discovery,
        roots=policy.roots,
    )


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def is_product_source(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(prefix) for prefix in prefixes)


def prune_deleted_sources(
    test_map: dict[str, dict[str, list[str]]],
    deleted: tuple[str, ...],
) -> dict[str, dict[str, list[str]]]:
    deleted_set = set(deleted)
    return {key: value for key, value in test_map.items() if key not in deleted_set}

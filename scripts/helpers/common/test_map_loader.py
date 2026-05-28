"""Load and validate test_map JSON and gate_policy exemptions.

Extracted from scripts.helpers.ci_gate.py — test_map loading, exemption loading, baseline
assembly, and helper predicates (is_exempt, is_product_source, prune_deleted_sources).
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.ci_gate.models import Baseline, Exemption
from scripts.helpers.common.coverage_config import PRODUCT_SOURCE_PREFIXES
from scripts.helpers.common.test_map_config import resolve_test_map_path

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_map_keys(inner: dict[str, object]) -> dict[str, dict[str, list[str]]]:
    validated: dict[str, dict[str, list[str]]] = {}
    for key, symbols in inner.items():
        if not isinstance(key, str):
            raise ConfigError(f"test_map: map key must be string, got {type(key).__name__}")
        if ".." in key or key.startswith("/"):
            raise ConfigError(f"test_map: invalid map key: {key!r}")
        if not any(key.startswith(prefix) for prefix in PRODUCT_SOURCE_PREFIXES):
            raise ConfigError(
                f"test_map: map key must start with a product prefix ({', '.join(PRODUCT_SOURCE_PREFIXES)}): {key!r}"
            )
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


def load_test_map(cfg: Config) -> dict[str, dict[str, list[str]]]:
    """Load and validate test_map JSON from the path given in *cfg*."""
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
    return _validate_map_keys(inner)


def load_exemptions(repo_root: Path) -> tuple[Exemption, ...]:
    """Load gate_policy.json exemptions. Returns empty tuple if file absent."""
    policy_path = repo_root / "tests" / ".ci" / "gate_policy.json"
    if not policy_path.is_file():
        return ()
    try:
        data = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"gate_policy.json: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("gate_policy.json: root must be object")
    out: list[Exemption] = []
    for item in data.get("exemptions", []):
        if not isinstance(item, dict):
            continue
        file_path = item.get("file")
        symbol = item.get("symbol")
        if isinstance(file_path, str) and isinstance(symbol, str):
            reason = item.get("reason")
            out.append(
                Exemption(
                    file=file_path,
                    symbol=symbol,
                    reason=reason if isinstance(reason, str) else None,
                )
            )
    return tuple(out)


def load_baseline(repo_root: Path, cfg: Config) -> Baseline:
    """Load full gate baseline: test_map + exemptions + product prefixes."""
    return Baseline(
        test_map=load_test_map(cfg),
        exemptions=load_exemptions(repo_root),
        product_prefixes=PRODUCT_SOURCE_PREFIXES,
    )


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def is_exempt(exemptions: tuple[Exemption, ...], file_path: str, symbol: str) -> bool:
    return any(item.file == file_path and item.symbol == symbol for item in exemptions)


def is_product_source(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(prefix) for prefix in prefixes)


def prune_deleted_sources(
    test_map: dict[str, dict[str, list[str]]],
    deleted: tuple[str, ...],
) -> dict[str, dict[str, list[str]]]:
    deleted_set = set(deleted)
    return {key: value for key, value in test_map.items() if key not in deleted_set}

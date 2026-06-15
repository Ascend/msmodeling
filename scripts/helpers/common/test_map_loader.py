"""Load and validate test_map JSON and gate policy baseline."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import TYPE_CHECKING

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.ci_gate.gate_policy import load_gate_policy
from scripts.helpers.ci_gate.models import Baseline
from scripts.helpers.common.coverage_config import product_roots
from scripts.helpers.common.test_map_config import resolve_test_map_path

if TYPE_CHECKING:
    from pathlib import Path

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


def _parse_test_map_payload(
    data: dict[str, object],
    resolved_roots: tuple[str, ...],
) -> tuple[dict[str, dict[str, list[str]]], str | None]:
    if data.get("schema_version") != 1:
        raise ConfigError("test_map: schema_version must be 1")
    built_from_commit = data.get("built_from_commit")
    if built_from_commit is not None and not isinstance(built_from_commit, str):
        raise ConfigError("test_map: built_from_commit must be a string")
    inner = data.get("map")
    if not isinstance(inner, dict):
        raise ConfigError("test_map: map must be object")
    return _validate_map_keys(inner, resolved_roots), built_from_commit


def load_test_map(
    cfg: Config,
    *,
    roots: tuple[str, ...] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Load and validate test_map JSON from the path given in *cfg*."""
    mapping, _commit = load_test_map_with_commit(cfg, roots=roots)
    return mapping


def load_test_map_with_commit(
    cfg: Config,
    *,
    roots: tuple[str, ...] | None = None,
) -> tuple[dict[str, dict[str, list[str]]], str | None]:
    """Load test_map JSON and return ``(map, built_from_commit)``."""
    resolved_roots = roots if roots is not None else product_roots()
    map_path = resolve_test_map_path(cfg, must_exist=True)
    try:
        data = json.loads(map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"test_map: invalid JSON at {map_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("test_map: root must be object")
    return _parse_test_map_payload(data, resolved_roots)


def validate_test_map_freshness(
    repo_root: Path,
    built_from_commit: str | None,
    merge_base: str,
) -> None:
    """Reject stale test_map artifacts that predate the PR merge-base."""
    if not built_from_commit:
        raise ConfigError("test_map: built_from_commit is required; rebuild test_map via nightly or build_test_map")
    git_path = shutil.which("git")
    if git_path is None:
        raise ConfigError("git not found")
    proc = subprocess.run(
        [git_path, "merge-base", "--is-ancestor", merge_base, built_from_commit],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    if proc.returncode != 0:
        raise ConfigError(
            "test_map: stale built_from_commit "
            f"{built_from_commit[:12]} is not reachable from merge-base {merge_base[:12]}"
        )


def load_baseline(repo_root: Path, cfg: Config) -> tuple[Baseline, str | None]:
    """Load full gate baseline: test_map + gate policy + product roots."""
    policy = load_gate_policy(repo_root)
    test_map, built_from_commit = load_test_map_with_commit(cfg, roots=policy.roots)
    baseline = Baseline(
        test_map=test_map,
        exemptions=policy.source_exemptions,
        test_exemptions=policy.test_exemptions,
        discovery=policy.discovery,
        roots=policy.roots,
    )
    return baseline, built_from_commit


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

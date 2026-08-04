"""Load and validate test_map JSON and CI gate baseline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.helpers._config import Config, ConfigError, format_expected_got
from scripts.helpers.ci_gate.diff import is_git_ancestor
from scripts.helpers.ci_gate.models import Baseline
from scripts.helpers.ci_gate.policy import load_gate_policy
from scripts.helpers.common.coverage_config import product_roots
from scripts.helpers.common.test_map_config import resolve_test_map_path

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

TestMap = dict[str, dict[str, list[str]]]


@dataclass(frozen=True, slots=True)
class TestMapFreshness:
    block_message: str | None = None
    warn_message: str | None = None


def _validate_test_node_key(test_node: str) -> None:
    if ".." in test_node or test_node.startswith("/"):
        raise ConfigError(f"test_map: invalid test node key: {test_node!r}")
    if not test_node.startswith("tests/") or "::" not in test_node:
        raise ConfigError(f"test_map: map key must be a pytest node id under tests/: {test_node!r}")


def _validate_canonical_symbol(symbol: str, *, test_node: str, source_file: str) -> None:
    if "." in symbol and "::" not in symbol and "@" not in symbol:
        raise ConfigError(
            f"test_map: symbol under {test_node!r} -> {source_file!r}: "
            f"symbol must use canonical Class::method form, got {symbol!r}"
        )


def _validate_source_entry_types(test_node: str, source_file: object, symbols: object) -> None:
    if not isinstance(source_file, str):
        field = f"source file under {test_node!r}"
        raise ConfigError(f"test_map: {format_expected_got(field, 'a string', source_file)}")
    if not isinstance(symbols, list):
        field = f"symbols under {test_node!r} -> {source_file!r}"
        raise ConfigError(f"test_map: {format_expected_got(field, 'a list', symbols)}")


def _validate_map_payload(
    inner: dict[str, object],
    roots: tuple[str, ...],
) -> TestMap:
    validated: TestMap = {}
    for test_node, sources in inner.items():
        if not isinstance(test_node, str):
            raise ConfigError(f"test_map: {format_expected_got('map key', 'a string', test_node)}")
        _validate_test_node_key(test_node)
        if not isinstance(sources, dict):
            raise ConfigError(f"test_map: {format_expected_got(f'value for {test_node!r}', 'an object', sources)}")
        source_map: dict[str, list[str]] = {}
        for source_file, symbols in sources.items():
            _validate_source_entry_types(test_node, source_file, symbols)
            assert isinstance(source_file, str)
            assert isinstance(symbols, list)
            if not any(source_file.startswith(prefix) for prefix in roots):
                raise ConfigError(
                    f"test_map: source file must start with a product root ({', '.join(roots)}): {source_file!r}"
                )
            if not all(isinstance(symbol, str) for symbol in symbols):
                field = f"symbols under {test_node!r} -> {source_file!r}"
                raise ConfigError(f"test_map: {format_expected_got(field, 'strings', symbols)}")
            for symbol in symbols:
                _validate_canonical_symbol(symbol, test_node=test_node, source_file=source_file)
            source_map[source_file] = list(symbols)
        if source_map:
            validated[test_node] = source_map
    return validated


def parse_test_map_map_object(
    inner: object,
    *,
    roots: tuple[str, ...],
) -> TestMap:
    """Validate and return a node-oriented ``map`` object from test_map JSON."""
    if not isinstance(inner, dict):
        raise ConfigError(f"test_map: {format_expected_got('map', 'an object', inner)}")
    if inner and not any(isinstance(key, str) and key.startswith("tests/") and "::" in key for key in inner):
        sample = next(iter(inner))
        raise ConfigError(f"test_map: map must be keyed by pytest node ids (tests/...py::test_name); got {sample!r}")
    return _validate_map_payload(inner, roots)


def _parse_test_map_payload(
    data: dict[str, object],
    resolved_roots: tuple[str, ...],
) -> tuple[TestMap, str | None]:
    schema_version = data.get("schema_version")
    if schema_version not in (1, None):
        raise ConfigError(f"test_map: unsupported schema_version {schema_version!r}")
    built_from_commit = data.get("built_from_commit")
    if built_from_commit is not None and not isinstance(built_from_commit, str):
        raise ConfigError(f"test_map: {format_expected_got('built_from_commit', 'a string', built_from_commit)}")
    inner = data.get("map")
    return parse_test_map_map_object(inner, roots=resolved_roots), built_from_commit


def load_test_map(
    cfg: Config,
    *,
    roots: tuple[str, ...] | None = None,
) -> TestMap:
    mapping, _commit = load_test_map_with_commit(cfg, roots=roots)
    return mapping


def load_test_map_with_commit(
    cfg: Config,
    *,
    roots: tuple[str, ...] | None = None,
) -> tuple[TestMap, str | None]:
    resolved_roots = roots if roots is not None else product_roots()
    map_path = resolve_test_map_path(cfg, must_exist=True)
    try:
        data = json.loads(map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"test_map: invalid JSON at {map_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"test_map: {format_expected_got('root', 'an object', data)}")
    return _parse_test_map_payload(data, resolved_roots)


def assess_test_map_freshness(
    repo_root: Path,
    built_from_commit: str | None,
    merge_base: str,
) -> TestMapFreshness:
    """Return block/warn messages for stale test_map relative to merge-base."""
    if not built_from_commit:
        return TestMapFreshness(
            block_message="test_map: built_from_commit is required; rebuild test_map via nightly or build_test_map"
        )

    if is_git_ancestor(repo_root, merge_base, built_from_commit):
        return TestMapFreshness()

    if is_git_ancestor(repo_root, built_from_commit, merge_base):
        return TestMapFreshness(
            warn_message=(
                "test_map: built_from_commit "
                f"{built_from_commit[:12]} is behind merge-base {merge_base[:12]}; continuing with stale map"
            )
        )

    return TestMapFreshness(
        block_message=(
            "test_map: stale built_from_commit "
            f"{built_from_commit[:12]} is not an ancestor of merge-base {merge_base[:12]}"
        )
    )


def validate_test_map_freshness(
    repo_root: Path,
    built_from_commit: str | None,
    merge_base: str,
) -> None:
    """Raise ConfigError when test_map freshness policy blocks the gate."""
    freshness = assess_test_map_freshness(repo_root, built_from_commit, merge_base)
    if freshness.block_message:
        raise ConfigError(freshness.block_message)
    if freshness.warn_message:
        logger.warning("%s", freshness.warn_message)


def load_baseline(repo_root: Path, cfg: Config) -> tuple[Baseline, str | None]:
    """Load full gate baseline: test_map + gate policy."""
    policy = load_gate_policy(repo_root)
    test_map, built_from_commit = load_test_map_with_commit(cfg, roots=policy.roots)
    baseline = Baseline(test_map=test_map, policy=policy)
    return baseline, built_from_commit


def is_product_source(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(prefix) for prefix in prefixes)

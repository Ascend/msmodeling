"""Load pyproject.toml coverage omit patterns and match product source paths."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Final

from scripts.helpers._config import ConfigError
from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.common.test_map_loader import is_product_source

_PYPROJECT_REL: Final = Path("pyproject.toml")


@lru_cache(maxsize=1)
def load_coverage_omit_patterns() -> tuple[str, ...]:
    """Return ``[tool.coverage.run].omit`` patterns from repo pyproject.toml."""
    pyproject_path = REPO_ROOT / _PYPROJECT_REL
    if not pyproject_path.is_file():
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: file not found")

    try:
        raw = pyproject_path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: cannot read file: {exc}") from exc

    if sys.version_info >= (3, 11):
        import tomllib

        try:
            data = tomllib.loads(raw.decode("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: invalid TOML: {exc}") from exc
    else:
        try:
            import tomli
        except ImportError as exc:
            raise ConfigError(
                "tomli required to parse pyproject.toml on Python < 3.11. Run: uv sync --frozen --group ci"
            ) from exc
        try:
            data = tomli.loads(raw.decode("utf-8"))
        except tomli.TOMLDecodeError as exc:
            raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: invalid TOML: {exc}") from exc

    coverage_run = data.get("tool", {}).get("coverage", {}).get("run")
    if not isinstance(coverage_run, dict):
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: [tool.coverage.run] section missing")

    omit = coverage_run.get("omit")
    if omit is None:
        return ()
    if not isinstance(omit, list):
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: [tool.coverage.run].omit must be a list")

    patterns: list[str] = []
    for index, pattern in enumerate(omit):
        if not isinstance(pattern, str) or not pattern.strip():
            raise ConfigError(
                f"{_PYPROJECT_REL.as_posix()}: [tool.coverage.run].omit[{index}] must be a non-empty string"
            )
        patterns.append(pattern)
    return tuple(patterns)


@lru_cache(maxsize=1)
def _coverage_omit_matcher() -> object | None:
    patterns = load_coverage_omit_patterns()
    if not patterns:
        return None
    from coverage.files import GlobMatcher, prep_patterns

    return GlobMatcher(prep_patterns(list(patterns)), "omit")


def is_coverage_omitted_source(path: str, roots: tuple[str, ...]) -> bool:
    """Return True when *path* is a product source excluded by coverage omit patterns."""
    if not is_product_source(path, roots):
        return False
    matcher = _coverage_omit_matcher()
    if matcher is None:
        return False
    return bool(matcher.match(path))

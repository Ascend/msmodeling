"""Load pyproject.toml coverage omit patterns and match product source paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from scripts.helpers._config import ConfigError, format_expected_got
from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.common.pyproject_toml import read_pyproject_data
from scripts.helpers.common.test_map_loader import is_product_source

_PYPROJECT_REL: Final = Path("pyproject.toml")


@lru_cache(maxsize=1)
def load_coverage_omit_patterns() -> tuple[str, ...]:
    """Return ``[tool.coverage.run].omit`` patterns from repo pyproject.toml."""
    data = read_pyproject_data(repo_root=REPO_ROOT)
    coverage_run = data.get("tool", {}).get("coverage", {}).get("run")
    if not isinstance(coverage_run, dict):
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: [tool.coverage.run] section missing")

    omit = coverage_run.get("omit")
    if omit is None:
        return ()
    if not isinstance(omit, list):
        raise ConfigError(
            f"{_PYPROJECT_REL.as_posix()}: {format_expected_got('[tool.coverage.run].omit', 'a list', omit)}"
        )

    patterns: list[str] = []
    for index, pattern in enumerate(omit):
        if not isinstance(pattern, str) or not pattern.strip():
            raise ConfigError(
                f"{_PYPROJECT_REL.as_posix()}: "
                f"{format_expected_got(f'[tool.coverage.run].omit[{index}]', 'a non-empty string', pattern)}"
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

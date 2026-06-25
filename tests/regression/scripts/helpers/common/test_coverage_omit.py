"""Tests for common.coverage_omit."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from scripts.helpers._config import ConfigError
from scripts.helpers.common import coverage_omit


@pytest.fixture(autouse=True)
def _clear_coverage_omit_caches() -> Generator[None, None, None]:
    """Reset lru_cache state so REPO_ROOT monkeypatch in one test cannot leak."""
    coverage_omit.load_coverage_omit_patterns.cache_clear()
    coverage_omit._coverage_omit_matcher.cache_clear()
    yield
    coverage_omit.load_coverage_omit_patterns.cache_clear()
    coverage_omit._coverage_omit_matcher.cache_clear()


def test_load_coverage_omit_patterns_reads_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        """
[tool.coverage.run]
omit = ["*/builtin_model/*", "*/tests/*"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(coverage_omit, "REPO_ROOT", repo)

    patterns = coverage_omit.load_coverage_omit_patterns()
    assert patterns == ("*/builtin_model/*", "*/tests/*")


def test_load_coverage_omit_patterns_invalid_list_raises_config_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        """
[tool.coverage.run]
omit = "not-a-list"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(coverage_omit, "REPO_ROOT", repo)

    with pytest.raises(ConfigError, match=r"Expected '\[tool\.coverage\.run\]\.omit' to be a list"):
        coverage_omit.load_coverage_omit_patterns()


def test_is_coverage_omitted_source_matches_builtin_model_under_root() -> None:
    roots = ("tensor_cast/", "cli/")
    path = "tensor_cast/transformers/builtin_model/foo.py"
    assert coverage_omit.is_coverage_omitted_source(path, roots) is True
    assert coverage_omit.is_coverage_omitted_source("tensor_cast/foo.py", roots) is False


def test_is_coverage_omitted_source_false_outside_roots() -> None:
    roots = ("cli/",)
    assert coverage_omit.is_coverage_omitted_source("other/pkg/builtin_model/foo.py", roots) is False


def test_monkeypatch_repo_root_leaves_stale_cache_until_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: undoing REPO_ROOT monkeypatch alone does not refresh lru_cache."""
    fake_repo = tmp_path / "fake"
    fake_repo.mkdir()
    (fake_repo / "pyproject.toml").write_text(
        """
[tool.coverage.run]
omit = ["*/no_such_glob/*"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(coverage_omit, "REPO_ROOT", fake_repo)
    fake_patterns = coverage_omit.load_coverage_omit_patterns()
    assert fake_patterns == ("*/no_such_glob/*",)

    monkeypatch.undo()
    assert coverage_omit.load_coverage_omit_patterns() == fake_patterns

    coverage_omit.load_coverage_omit_patterns.cache_clear()
    coverage_omit._coverage_omit_matcher.cache_clear()
    real_patterns = coverage_omit.load_coverage_omit_patterns()
    assert "*/builtin_model/*" in real_patterns

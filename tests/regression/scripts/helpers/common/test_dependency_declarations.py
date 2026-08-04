"""Dependency declaration contract tests."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.common.pyproject_toml import read_pyproject_data

_REMOVED_RUNTIME_DEPENDENCIES = {
    "filelock",
    "pillow",
    "requests",
    "scikit-learn",
}
_REQUIRED_RUNTIME_DEPENDENCIES = {
    "greenlet",
    "optree",
}
_TEST_DEPENDENCIES = {
    "parameterized",
    "pytest",
    "pytest-cov",
    "pytest-xdist",
}


def _normalize_dependency_name(dependency: str) -> str:
    """Return the normalized distribution name from a dependency declaration."""
    return re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0].strip().lower()


def _dependency_names(requirements_path: Path) -> set[str]:
    names = set()
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(_normalize_dependency_name(line))
    return names


def test_legacy_requirements_files_are_ascii_for_windows_pip() -> None:
    for filename in ("requirements.txt", "requirements-ci.txt"):
        (REPO_ROOT / filename).read_bytes().decode("ascii")


def test_removed_dependencies_are_not_declared_as_direct_runtime_dependencies() -> None:
    pyproject = read_pyproject_data()
    project = pyproject["project"]
    assert isinstance(project, dict)
    runtime_names = {_normalize_dependency_name(dependency) for dependency in project["dependencies"]}

    assert runtime_names.isdisjoint(_REMOVED_RUNTIME_DEPENDENCIES)
    assert _dependency_names(REPO_ROOT / "requirements.txt").isdisjoint(
        _REMOVED_RUNTIME_DEPENDENCIES | _TEST_DEPENDENCIES
    )


def test_required_runtime_dependencies_are_declared() -> None:
    pyproject = read_pyproject_data()
    project = pyproject["project"]
    assert isinstance(project, dict)
    runtime_names = {_normalize_dependency_name(dependency) for dependency in project["dependencies"]}

    assert _REQUIRED_RUNTIME_DEPENDENCIES <= runtime_names
    assert _REQUIRED_RUNTIME_DEPENDENCIES <= _dependency_names(REPO_ROOT / "requirements.txt")


def test_test_dependencies_are_declared_in_ci_only() -> None:
    pyproject = read_pyproject_data()
    dependency_groups = pyproject["dependency-groups"]
    assert isinstance(dependency_groups, dict)
    ci_names = {_normalize_dependency_name(dependency) for dependency in dependency_groups["ci"]}

    assert _TEST_DEPENDENCIES <= ci_names
    # Keep the legacy pip CI file minimal; additions require an explicit update
    # to the audited dependency set above.
    assert _dependency_names(REPO_ROOT / "requirements-ci.txt") == _TEST_DEPENDENCIES

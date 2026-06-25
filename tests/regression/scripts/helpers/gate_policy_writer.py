"""Write minimal gate_policy.yaml fixtures for scripts/helpers regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

DEFAULT_GATE_ROOTS: Final = (
    "cli/",
    "serving_cast/",
    "tensor_cast/",
    "web_ui/",
    "scripts/",
    "tools/",
)
DEFAULT_TEST_INCLUDE: Final = ("tests/**/test_*.py", "tests/**/*_test.py")
DEFAULT_TEST_EXCLUDE: Final = ("tests/helpers/**", "tests/assets/**")
DEFAULT_CONFIG_INCLUDE: Final = (
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "tests/**/conftest.py",
)
DEFAULT_APPROVERS: Final = ("fangkai", "hexiaowu", "gongjiong", "liujiawang")


def write_repo_file(repo: Path, rel: str, content: str) -> Path:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_gate_policy(
    repo: Path,
    *,
    roots: tuple[str, ...] | list[str] | None = None,
    tests: dict[str, list[str]] | None = None,
    test_include: tuple[str, ...] | list[str] | None = None,
    test_exclude: tuple[str, ...] | list[str] | None = None,
    config_include: tuple[str, ...] | list[str] | None = None,
    source_exemptions: list[dict[str, object]] | None = None,
    test_exemptions: list[dict[str, object]] | None = None,
    approvers: tuple[str, ...] | list[str] | None = None,
) -> None:
    ci_dir = repo / "tests" / ".ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    if tests is not None:
        include_patterns = tests["include"]
        exclude_patterns = tests["exclude"]
    else:
        include_patterns = list(test_include or DEFAULT_TEST_INCLUDE)
        exclude_patterns = list(test_exclude or DEFAULT_TEST_EXCLUDE)
    policy = {
        "roots": list(roots or DEFAULT_GATE_ROOTS),
        "tests": {
            "include": include_patterns,
            "exclude": exclude_patterns,
        },
        "configs": {
            "include": list(config_include or DEFAULT_CONFIG_INCLUDE),
            "exclude": [],
        },
        "exemptions": {
            "sources": source_exemptions or [],
            "tests": test_exemptions or [],
        },
    }
    (ci_dir / "gate_policy.yaml").write_text(yaml.dump(policy), encoding="utf-8")
    (ci_dir / "approvers.yaml").write_text(
        yaml.dump({"approvers": list(approvers or DEFAULT_APPROVERS)}),
        encoding="utf-8",
    )

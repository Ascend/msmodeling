"""Shared fixtures for scripts/helpers tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from scripts.helpers._config import Config
from scripts.helpers.ci_gate.gate_policy import default_test_discovery
from scripts.helpers.ci_gate.models import Baseline
from scripts.helpers.common.coverage_gate import GateConfig


@pytest.fixture(scope="session")
def base_config() -> Config:
    """Base Config with all defaults — individual tests override env vars as needed."""
    return Config(
        test_map_path="",
        base_branch="master",
        line_threshold=70.0,
        branch_threshold=50.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )


@pytest.fixture(scope="session")
def gate_config(base_config: Config) -> GateConfig:
    """GateConfig derived from base_config — used by coverage_gate and nightly tests."""
    return GateConfig.from_config(base_config)


@pytest.fixture(scope="session")
def baseline() -> Baseline:
    """Minimal Baseline with two product source entries and cross-layer test."""
    test_map = {
        "cli/main.py": {
            "run": ["tests/regression/cli/test_run.py::test_run"],
        },
        "tensor_cast/ops.py": {
            "add": [
                "tests/regression/tensor_cast/test_ops.py::test_add",
                "tests/regression/cli/test_cross.py::test_cross",
            ],
        },
    }
    return Baseline(
        test_map=test_map,
        exemptions=(),
        test_exemptions=(),
        discovery=default_test_discovery(),
        roots=(
            "cli/",
            "tensor_cast/",
            "serving_cast/",
            "web_ui/",
            "scripts/",
            "tools/",
        ),
    )


@pytest.fixture(scope="session")
def sample_py_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a real .py file with functions, classes, docstrings, annotations.

    Used by test_ast_utils and test_rules for AST-based gate checks.
    """
    content = textwrap.dedent("""\
        \"\"\"Module docstring.\"\"\"
        __all__ = ["foo"]
        __version__ = "1.0"

        from typing import Final

        COUNT: Final = 3

        def foo() -> None:
            \"\"\"Docstring.\"\"\"
            x = 1
            return None

        class Bar:
            \"\"\"Class docstring.\"\"\"

            CLASS_VAR: int

            def method(self) -> str:
                return "hello"

        async def baz() -> None:
            pass
    """)
    path = tmp_path_factory.mktemp("sample_mod") / "test_mod.py"
    path.write_text(content, encoding="utf-8")
    return path

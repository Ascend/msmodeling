"""Shared fixtures for scripts/helpers tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.helpers._config import Config
from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.models import Baseline, CiGatePolicy
from scripts.helpers.ci_gate.policy import load_gate_policy
from scripts.helpers.common.coverage_gate import GateConfig


def default_ci_gate_policy() -> CiGatePolicy:
    return load_gate_policy(REPO_ROOT)


@pytest.fixture(scope="session")
def base_config() -> Config:
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
    return GateConfig.from_config(base_config)


@pytest.fixture(scope="session")
def baseline() -> Baseline:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["run"],
        },
        "tests/regression/tensor_cast/test_ops.py::test_add": {
            "tensor_cast/ops.py": ["add"],
        },
        "tests/regression/cli/test_cross.py::test_cross": {
            "tensor_cast/ops.py": ["add"],
        },
    }
    return Baseline(test_map=test_map, policy=default_ci_gate_policy())


@pytest.fixture(scope="session")
def sample_py_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
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

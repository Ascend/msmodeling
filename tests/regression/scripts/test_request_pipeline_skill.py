# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for the request-pipeline skill's precision-wave selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_skill_script() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / ".agents/skills/msmodeling-request-pipeline/scripts/run_change_pipeline.py"
    spec = importlib.util.spec_from_file_location("msmodeling_request_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load request-pipeline script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_skill_script()


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (["tensor_cast/model.py"], True),
        (["scripts/helpers/tool.py"], True),
        (["tests/regression/test_model.py"], True),
        (["tests/benchmark/models/cases/model.json"], True),
        (["docs/design/change.md"], False),
        ([".agents/skills/example/SKILL.md"], False),
    ],
)
def test_diff_needs_precision(changed: list[str], expected: bool) -> None:
    assert SCRIPT._diff_needs_precision(changed) is expected


def test_precision_nodes_are_removed_from_other_waves_without_reordering() -> None:
    nodes = [
        "tests/regression/test_a.py::test_a",
        "tests/benchmark/models/test_model_regression.py::test_model[case-a]",
        "tests/benchmark/ops/perf_database/test_data_source.py::test_lookup",
        "tests/regression/test_b.py::test_b",
    ]
    precision = nodes[1:3]

    assert SCRIPT._exclude_precision_nodes(nodes, precision) == [nodes[0], nodes[3]]

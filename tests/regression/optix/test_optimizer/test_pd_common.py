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

"""Regression tests for independent PD module imports and shared output helpers."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("first", ["pd_disagg", "pd_full_service"])
def test_pd_modules_can_be_imported_independently(first):
    second = "pd_full_service" if first == "pd_disagg" else "pd_disagg"
    code = """
import importlib
import sys

first, second = sys.argv[1:]
importlib.import_module("optix.optimizer." + first)
assert "optix.optimizer." + second not in sys.modules, (first, second)
importlib.import_module("optix.optimizer." + second)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, first, second],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pd_json_output_preserves_paths_and_numpy_metrics(tmp_path):
    from optix.optimizer.pd_common import write_json

    path = tmp_path / "full_service" / "phase_result.json"
    write_json(path, {"csv": tmp_path / "data.csv", "throughput": np.float64(12.5), "instances": np.int64(2)})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "csv": str(tmp_path / "data.csv"),
        "throughput": 12.5,
        "instances": 2,
    }

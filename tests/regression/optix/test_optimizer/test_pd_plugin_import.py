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

"""Verify PD plugin imports without collection-time dependency mocks."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("smoke_first", [False, True])
def test_pd_plugin_import_uses_real_invoke_in_fresh_process(pytestconfig, smoke_first):
    """Plugin imports must work without smoke mocks or collection-order effects."""
    repo_root = pytestconfig.rootpath
    script = """
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "contrib/optix/vllm_pd_simulator"))
if sys.argv[2] == "True":
    runpy.run_path(str(root / "tests/smoke/test_vllm_pd_smoke.py"))
import invoke.exceptions
from vllm_pd_simulator import pd_cluster_simulator

assert invoke.exceptions.__file__
assert pd_cluster_simulator.CommandTimedOut is invoke.exceptions.CommandTimedOut
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(repo_root), str(smoke_first)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

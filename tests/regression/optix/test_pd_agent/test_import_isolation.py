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

"""Regression tests for independent replay and OptiX script imports."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("replay_first", [True, False])
def test_script_common_modules_are_independent(replay_first):
    code = """
import importlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
replay = root / "tools/perf_data_collection/op_replay"
scripts = root / ".agents/skills/optix-assistant/scripts"
sys.path[:0] = [str(replay), str(scripts)]
if sys.argv[2] == "True":
    replay_common = importlib.import_module("common")
    writer = importlib.import_module("config_writer")
else:
    writer = importlib.import_module("config_writer")
    replay_common = importlib.import_module("common")
skill_common = importlib.import_module("optix_assistant_common")
assert Path(replay_common.__file__).resolve() == replay / "common.py"
assert Path(skill_common.__file__).resolve() == scripts / "optix_assistant_common.py"
assert writer.CONTAINER_FLAGS is skill_common.CONTAINER_FLAGS
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(replay_first)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import isolation failed:\n{result.stdout}\n{result.stderr}"

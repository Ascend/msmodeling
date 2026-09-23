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

"""Packaging boundary for the source-only model diagnostics tool."""

import tomllib
from pathlib import Path


def test_model_diagnostics_is_excluded_from_wheel_packages() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    with (repository_root / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    wheel_packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]

    assert "model_diagnostics" not in wheel_packages
    assert "tools" not in wheel_packages
    assert all(not package.startswith("tools/model_diagnostics") for package in wheel_packages)

# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

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

"""Regression tests for ``python -m optix`` (optix.__main__)."""

from __future__ import annotations

import runpy
from unittest.mock import patch

import pytest


def test_optix_main_module_delegates_to_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_optimizer_main() -> None:
        calls.append("optimizer")

    monkeypatch.setattr("optix.configure_logger", lambda: None)
    with patch("optix.optimizer.optimizer.main", side_effect=_fake_optimizer_main):
        runpy.run_module("optix", run_name="__main__", alter_sys=True)

    assert calls == ["optimizer"]

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

"""Tests for the collect-only node id plugin."""

from __future__ import annotations

from pathlib import Path

import pytest


class _Item:
    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid


class _Session:
    def __init__(self, items: list[_Item]) -> None:
        self.items = items


def test_pytest_collection_finish_writes_node_ids_and_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.helpers.common.pytest_collect_ids import pytest_collection_finish

    dest = tmp_path / "ids.txt"
    monkeypatch.setenv("MSMODELING_COLLECT_NODE_IDS", str(dest))
    session = _Session([_Item("tests/a.py::test_one"), _Item("tests/b.py::test_two")])

    pytest_collection_finish(session)

    assert dest.read_text(encoding="utf-8") == "tests/a.py::test_one\ntests/b.py::test_two\n"
    assert capsys.readouterr().out == "tests/a.py::test_one\ntests/b.py::test_two\n"


def test_pytest_collection_finish_skips_file_when_env_blank(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.helpers.common.pytest_collect_ids import pytest_collection_finish

    monkeypatch.setenv("MSMODELING_COLLECT_NODE_IDS", "   ")
    session = _Session([_Item("tests/a.py::test_plain")])

    pytest_collection_finish(session)

    assert capsys.readouterr().out == "tests/a.py::test_plain\n"

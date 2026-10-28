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

"""Portable POSIX signal simulation in subprocess test doubles."""

import signal
import subprocess

import pytest

from tests.helpers.fake_subprocess import FakePopen, FakePopenTimeoutOnFirstWait


@pytest.mark.parametrize("has_sigkill", [False, True])
def test_fake_kill_without_platform_signal(monkeypatch, has_sigkill):
    if has_sigkill:
        monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    else:
        monkeypatch.delattr(signal, "SIGKILL", raising=False)
    proc = FakePopen()
    proc.kill()
    assert proc.poll() == -9
    assert proc.kill_calls == 1
    timed = FakePopenTimeoutOnFirstWait()
    with pytest.raises(subprocess.TimeoutExpired):
        timed.wait(timeout=0.01)
    assert timed.wait() == -9

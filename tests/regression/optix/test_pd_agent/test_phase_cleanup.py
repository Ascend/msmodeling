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

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from optix.optimizer.agentic import pd_phase


def identity():
    return {
        "host": socket.gethostname(),
        "uid": os.getuid(),
        "netns": os.stat("/proc/self/ns/net").st_ino,
        "port": 8000,
        "groups": [],
        "launch_pending": False,
    }


def test_phase_shutdown_requires_process_and_port_absence(monkeypatch):
    monkeypatch.setattr(pd_phase.shutil, "which", lambda name: "/usr/bin/ss")
    record = {"cleanup_identity": identity(), "cleanup_confirmed": False}
    record["cleanup_identity"]["groups"] = [123]
    monkeypatch.setattr(os, "killpg", lambda *a: None)
    with pytest.raises(ValueError, match="still alive"):
        pd_phase.verify_phase_shutdown(record)
    assert not record["cleanup_confirmed"]

    def gone(*args):
        raise ProcessLookupError()

    monkeypatch.setattr(os, "killpg", gone)
    monkeypatch.setattr(pd_phase.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="LISTEN"))
    with pytest.raises(ValueError, match="endpoint"):
        pd_phase.verify_phase_shutdown(record)
    assert not record["cleanup_confirmed"]
    monkeypatch.setattr(pd_phase.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=""))
    pd_phase.verify_phase_shutdown(record)
    assert record["cleanup_confirmed"]


def test_phase_shutdown_fails_closed_on_inaccessible_process(monkeypatch):
    record = {"cleanup_identity": identity(), "cleanup_confirmed": False}
    record["cleanup_identity"]["groups"] = [123]

    def denied(*args):
        raise PermissionError()

    monkeypatch.setattr(os, "killpg", denied)
    with pytest.raises(ValueError, match="cannot be verified"):
        pd_phase.verify_phase_shutdown(record)
    assert not record["cleanup_confirmed"]


def test_phase_shutdown_fails_closed_without_ss(monkeypatch):
    record = {"cleanup_identity": identity(), "cleanup_confirmed": False}
    monkeypatch.setattr(pd_phase.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="ss is unavailable"):
        pd_phase.verify_phase_shutdown(record)
    assert not record["cleanup_confirmed"]


def test_recovery_closes_interrupted_phase_only_after_verified_cleanup(tmp_path, monkeypatch):
    path = tmp_path / "pd_tasks/one/attempt.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"started_at": 1, "outcome": "started", "cleanup_confirmed": False, "cleanup_identity": identity()})
    )
    monkeypatch.setattr(pd_phase, "verify_phase_shutdown", lambda record: record.update(cleanup_confirmed=True))
    pd_phase.recover_phase_attempts(tmp_path)
    record = json.loads(path.read_text())
    assert record["cleanup_confirmed"] and record["outcome"] == "aborted" and record["elapsed_unknown"]


@pytest.mark.skipif(shutil.which("ss") is None, reason="requires the ss system utility")
def test_real_local_process_group_and_listener_are_detected():
    details = identity()
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    details["port"] = server.getsockname()[1]
    record = {"cleanup_identity": details, "cleanup_confirmed": False}
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    details["groups"] = [child.pid]
    try:
        with pytest.raises(ValueError, match="still alive"):
            pd_phase.verify_phase_shutdown(record)
        child.terminate()
        child.wait(timeout=5)
        with pytest.raises(ValueError, match="endpoint"):
            pd_phase.verify_phase_shutdown(record)
        server.close()
        pd_phase.verify_phase_shutdown(record)
        assert record["cleanup_confirmed"]
    finally:
        server.close()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_launch_gate_records_group_before_service_executes(tmp_path):
    path = tmp_path / "attempt.json"
    output = tmp_path / "executed"
    record = {"cleanup_identity": identity(), "cleanup_confirmed": False}
    launch = pd_phase.PhaseLaunch(record, path)
    command = launch.prepare([sys.executable, "-c", "from pathlib import Path; Path(%r).touch()" % str(output)])
    child = subprocess.Popen(command, start_new_session=True)
    try:
        time.sleep(0.15)
        assert not output.exists()
        with pytest.raises(ValueError, match="identity was not recorded"):
            pd_phase.verify_phase_shutdown(record)
        launch.started(child)
        persisted = json.loads(path.read_text())
        assert persisted["cleanup_identity"]["groups"] == [child.pid]
        assert not persisted["cleanup_identity"]["launch_pending"]
        assert child.wait(timeout=5) == 0
        assert output.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_custom_process_uses_phase_launch_gate(tmp_path, monkeypatch):
    from optix.deploy_env import RuntimeContext
    from optix.optimizer.interfaces.custom_process import CustomProcess

    context = RuntimeContext(in_virtualenv=False, virtualenv_root=None, python_executable=Path(sys.executable))
    process = CustomProcess(runtime_ctx=context, deploy_env=dict(os.environ))
    process.command = [sys.executable, "-c", "import time; time.sleep(30)"]
    monkeypatch.setattr(process, "before_run", lambda *args: None)
    record = {"cleanup_identity": identity(), "cleanup_confirmed": False}
    process._optix_phase_launch = pd_phase.PhaseLaunch(record, tmp_path / "attempt.json")
    process.run()
    try:
        assert record["cleanup_identity"]["groups"] == [process.process.pid]
        assert os.getpgid(process.process.pid) == process.process.pid
        with pytest.raises(ValueError, match="still alive"):
            pd_phase.verify_phase_shutdown(record)
    finally:
        os.killpg(process.process.pid, 9)
        process.process.wait(timeout=5)

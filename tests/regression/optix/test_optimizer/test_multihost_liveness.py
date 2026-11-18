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

"""Unit tests for the multihost node liveness tri-state and startup guards.

No cluster or network is required: fabric is stubbed and every remote executor /
SSH call is mocked, so the ProcessLiveness branches (ALIVE/DEAD/UNKNOWN), the
fail-fast startup probe, the readiness-log tail match, and the node script upload
failure path are all covered without a real node.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[4] / "contrib/optix/multihost_infer"
_PLUGIN_PACKAGE = "multihost_inference_optimization"


def _plugin_module_names() -> list[str]:
    return [name for name in sys.modules if name == _PLUGIN_PACKAGE or name.startswith(f"{_PLUGIN_PACKAGE}.")]


def _load_plugin(monkeypatch, filename: str):
    """Load one plugin source file with a stubbed fabric, without polluting sys.modules.

    Mirrors tests/regression/optix/test_optimizer/test_multihost_adapter.py: the plugin
    binds fabric.Config/Connection at import time, so fabric is stubbed with objects and
    the real plugin package modules are dropped again afterwards.
    """
    monkeypatch.syspath_prepend(str(PLUGIN_DIR))
    fabric = types.ModuleType("fabric")
    fabric.Connection = object
    fabric.Config = object
    monkeypatch.setitem(sys.modules, "fabric", fabric)
    snapshot = {name: sys.modules[name] for name in _plugin_module_names()}
    spec = importlib.util.spec_from_file_location(
        "td_" + filename[:-3], PLUGIN_DIR / "multihost_inference_optimization" / filename
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name in _plugin_module_names():
            sys.modules.pop(name, None)
        sys.modules.update(snapshot)
    return module


# ---------------------------------------------------------------------------
# SshRemote.probe_process_alive / is_process_alive
# ---------------------------------------------------------------------------


def test_process_liveness_is_three_way(monkeypatch):
    ssh = _load_plugin(monkeypatch, "ssh_remote_tools.py")
    remote = ssh.SshRemote("worker", ssh_user="runner")
    conn = MagicMock()
    remote._conn = conn

    conn.run.return_value.ok = True
    assert remote.probe_process_alive(123) is ssh.ProcessLiveness.ALIVE
    assert remote.is_process_alive(123) is True

    conn.run.return_value.ok = False
    assert remote.probe_process_alive(123) is ssh.ProcessLiveness.DEAD
    assert remote.is_process_alive(123) is False

    conn.run.side_effect = RuntimeError("connection reset")
    assert remote.probe_process_alive(123) is ssh.ProcessLiveness.UNKNOWN
    assert remote.is_process_alive(123) is False

    conn.run.side_effect = None
    # pid None keeps is_process_alive's legacy contract (False), not UNKNOWN at this layer.
    assert remote.probe_process_alive(None) is ssh.ProcessLiveness.DEAD
    assert remote.is_process_alive(None) is False


# ---------------------------------------------------------------------------
# VLLMWorkerManager.node_liveness
# ---------------------------------------------------------------------------


def test_node_liveness_unknown_when_pid_unset(monkeypatch):
    mgr = _load_plugin(monkeypatch, "vllm_worker_manager.py")
    manager = mgr.VLLMWorkerManager.__new__(mgr.VLLMWorkerManager)
    manager._node_pid = None
    assert manager.node_liveness() is mgr.ProcessLiveness.UNKNOWN


def test_node_liveness_forwards_executor_verdict(monkeypatch):
    mgr = _load_plugin(monkeypatch, "vllm_worker_manager.py")
    manager = mgr.VLLMWorkerManager.__new__(mgr.VLLMWorkerManager)
    manager._node_pid = 123
    manager.node = None

    for verdict in (mgr.ProcessLiveness.ALIVE, mgr.ProcessLiveness.DEAD, mgr.ProcessLiveness.UNKNOWN):
        fake = MagicMock()
        fake.probe_process_alive.return_value = verdict
        manager._get_executor = lambda _node, executor=fake: executor
        assert manager.node_liveness() is verdict
    fake.probe_process_alive.assert_called_with(123)


# ---------------------------------------------------------------------------
# VLLMWorkerManager._probe_worker_alive (fail-fast startup probe)
# ---------------------------------------------------------------------------


def test_probe_worker_alive_retries_unknown_before_failing(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    mgr = _load_plugin(monkeypatch, "vllm_worker_manager.py")
    manager = mgr.VLLMWorkerManager.__new__(mgr.VLLMWorkerManager)
    fake = MagicMock()
    fake.probe_process_alive.side_effect = [mgr.ProcessLiveness.UNKNOWN, mgr.ProcessLiveness.ALIVE]
    manager._get_executor = lambda _node: fake

    assert manager._probe_worker_alive("node", "start_node", 123, "/tmp/x.log", wait_seconds=0) is True
    assert fake.probe_process_alive.call_count == 2


def test_probe_worker_alive_fails_only_on_dead(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    mgr = _load_plugin(monkeypatch, "vllm_worker_manager.py")
    manager = mgr.VLLMWorkerManager.__new__(mgr.VLLMWorkerManager)
    fake = MagicMock()
    fake.probe_process_alive.return_value = mgr.ProcessLiveness.DEAD
    fake.tail_file.return_value = "boom"
    manager._get_executor = lambda _node: fake

    assert (
        manager._probe_worker_alive(
            MagicMock(host="10.0.0.1", ssh_port=22), "start_node", 123, "/tmp/x.log", wait_seconds=0
        )
        is False
    )
    fake.tail_file.assert_called_once_with("/tmp/x.log")


# ---------------------------------------------------------------------------
# VLLMWorkerManager.upload_node_script
# ---------------------------------------------------------------------------


def test_upload_node_script_raises_when_script_missing(monkeypatch, tmp_path):
    mgr = _load_plugin(monkeypatch, "vllm_worker_manager.py")
    manager = mgr.VLLMWorkerManager.__new__(mgr.VLLMWorkerManager)
    manager.node = object()  # not None, so it tries to locate the script
    manager._build_scripts_dir = tmp_path  # empty dir -> no start_node.sh
    with pytest.raises(RuntimeError, match="node startup script not found"):
        manager.upload_node_script()


# ---------------------------------------------------------------------------
# MultiHostSimulator._cluster_ready_from_log
# ---------------------------------------------------------------------------


def _simulator(monkeypatch):
    sim = _load_plugin(monkeypatch, "simulator.py")
    instance = sim.MultiHostSimulator.__new__(sim.MultiHostSimulator)
    instance._cluster_ready = False
    instance._ready_pattern_warned = False
    instance._ready_tail_at = 0.0
    return sim, instance


def test_cluster_ready_from_log_matches_and_latches(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    sim, instance = _simulator(monkeypatch)
    tail = MagicMock(return_value="Application startup complete")
    instance.remote_executor = MagicMock(
        config=MagicMock(ready_log_pattern="Application startup complete"),
        tail_node_log=tail,
    )

    assert instance._cluster_ready_from_log() is True
    assert instance._cluster_ready is True
    tail.assert_called_once_with(lines=sim.MultiHostSimulator._READY_LOG_TAIL_LINES)

    # One-shot latch: further calls return True without reading the log again.
    tail.reset_mock()
    assert instance._cluster_ready_from_log() is True
    tail.assert_not_called()


def test_cluster_ready_from_log_throttles_tail_read(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 10.0)
    sim, instance = _simulator(monkeypatch)
    tail = MagicMock(return_value="still starting")
    instance.remote_executor = MagicMock(config=MagicMock(ready_log_pattern="ready"), tail_node_log=tail)

    # First read happens (10 - 0 >= _READY_TAIL_INTERVAL), no match -> keep waiting.
    assert instance._cluster_ready_from_log() is False
    tail.assert_called_once()

    # Same tick: the 500-line log read is skipped rather than re-shipped every poll.
    tail.reset_mock()
    assert instance._cluster_ready_from_log() is False
    tail.assert_not_called()


# ---------------------------------------------------------------------------
# MultiHostSimulator.health (only a definitive DEAD aborts)
# ---------------------------------------------------------------------------


def test_health_returns_error_only_on_cached_dead(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    sim, instance = _simulator(monkeypatch)
    # bytes inside the liveness throttle window, so health() reuses the cached verdict
    # and never touches remote_executor / requests.get.
    instance._node_liveness = sim.ProcessLiveness.DEAD
    instance._node_liveness_at = 100.0

    result = instance.health()
    assert result.stage is sim.Stage.error
    assert "master node process is not running" in result.info

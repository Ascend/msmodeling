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


import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from optix.optimizer.pd_adapter import verify_shutdown


ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "export_pd", ROOT / ".agents/skills/optix-assistant/scripts/export_best_config.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def snapshot():
    config = "a = 1\n"
    nodes = [
        {
            "role": role,
            "script": role + ".sh",
            "label": role,
            "port": 18080 + i,
            "pid": 100 + i,
            "node_config": {"ssh_ip": role},
        }
        for i, role in enumerate(["prefill", "decode", "proxy"])
    ]
    files = {node["script"]: "#!/bin/bash\necho measured\n" for node in nodes}
    files.update(
        {
            "load_balance_proxy_server_example.py": "# proxy dependency\n",
            "load_balance_proxy_layerwise_server_example.py": "# layerwise dependency\n",
        }
    )
    plan = {"files": files, "nodes": nodes, "layout": {}, "remote_dir": "/tmp/pd", "config_toml": config}
    context = {"deployment": {"type": "pd", "config_sha256": hashlib.sha256(config.encode()).hexdigest()}}
    best = {
        "source_ref": "results.round-1.json#r1",
        "metadata": {"launch_plan": plan},
        "selection_policy": "slo_compliant_best",
        "fitness": 1,
        "params": {},
    }
    return plan, context, best


def test_export_preserves_winning_config_and_reports_pending_verification(tmp_path):
    plan, context, best = snapshot()
    exporter._export_pd_snapshot(tmp_path, context, best, {})
    assert (tmp_path / "best_deployment/config.toml").read_text(encoding="utf-8") == plan["config_toml"]
    import json

    manifest = json.loads((tmp_path / "serve_command_pd.json").read_text(encoding="utf-8"))
    assert manifest["verification"]["status"] == "pending"
    assert manifest["source_ref"] == best["source_ref"]


def test_missing_proxy_dependency_cannot_be_exported(tmp_path):
    plan, context, best = snapshot()
    del plan["files"]["load_balance_proxy_layerwise_server_example.py"]
    with pytest.raises(ValueError, match="dependencies"):
        exporter._export_pd_snapshot(tmp_path, context, best, {})


def test_winning_config_drift_rejected(tmp_path):
    plan, context, best = snapshot()
    plan["config_toml"] += "b = 2\n"
    with pytest.raises(ValueError, match="match"):
        exporter._export_pd_snapshot(tmp_path, context, best, {})


def test_export_path_traversal_rejected(tmp_path):
    plan, context, best = snapshot()
    plan["files"]["../bad.sh"] = "bad"
    with pytest.raises(ValueError, match="Unsafe"):
        exporter._export_pd_snapshot(tmp_path, context, best, {})


def test_shutdown_requires_both_pid_and_endpoint_absence():
    plan, _, _ = snapshot()
    executor = MagicMock()
    executor.run.side_effect = [SimpleNamespace(ok=False, stdout=""), SimpleNamespace(ok=True, stdout="")] * 3
    verify_shutdown(plan, lambda node: executor)
    assert plan["cleanup_confirmed"] is True
    assert executor.close.call_count == 3


@pytest.mark.parametrize(
    "pid_ok, port_ok, port_output", [(True, True, ""), (False, False, ""), (False, True, "LISTEN")]
)
def test_shutdown_failure_never_claims_cleanup(pid_ok, port_ok, port_output):
    plan, _, _ = snapshot()
    executor = MagicMock()
    executor.run.side_effect = [SimpleNamespace(ok=pid_ok, stdout=""), SimpleNamespace(ok=port_ok, stdout=port_output)]
    with pytest.raises(ValueError):
        verify_shutdown(plan, lambda node: executor)
    assert not plan.get("cleanup_confirmed")


def test_cli_pd_export_excludes_diagnosis_and_repetitions_and_writes_best(tmp_path):
    import json
    import subprocess

    plan, context, best = snapshot()
    context["search_space"] = {"parameters": [], "constants": []}
    trials = []
    candidates = []
    directions = []
    for cid, kind, purpose, fitness in [
        ("search", "optimization", "search", 3),
        ("diagnosis", "diagnosis", "search", 1),
        ("repeat", "optimization", "verification_winner", 2),
    ]:
        trials.append(
            {
                "candidate_id": cid,
                "params": {},
                "fitness": fitness,
                "slo_violated": False,
                "metadata": {"launch_plan": plan, "deployment_proposal": {"purpose": purpose}},
            }
        )
        candidates.append({"candidate_id": cid, "direction_id": cid, "params": {}})
        directions.append({"direction_id": cid, "trial_type": kind})
    for name, data in [
        ("context.round-1.json", context),
        ("results.round-1.json", {"trials": trials}),
        ("candidates.round-1.json", {"round": 1, "directions": directions, "candidates": candidates}),
    ]:
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".agents/skills/optix-assistant/scripts/export_best_config.py"),
            "--run-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    chosen = json.loads((tmp_path / "best_result.json").read_text(encoding="utf-8"))
    assert chosen["candidate_id"] == "search"
    assert chosen["trial_count"] == 1 and chosen["round_count"] == 1
    assert (
        json.loads((tmp_path / "serve_command_pd.json").read_text(encoding="utf-8"))["source_ref"]
        == "results.round-1.json#trials[0]"
    )


@pytest.mark.parametrize("alive", [False, True])
def test_saved_shutdown_recreates_executor_and_requires_observed_cleanup(alive):
    from unittest.mock import patch
    from optix.optimizer.pd_adapter import verify_saved_shutdown

    plan = {
        "nodes": [
            {
                "label": "P-0",
                "pid": 123,
                "port": 18080,
                "node_config": {
                    "ssh_ip": "host",
                    "ssh_port": 2222,
                    "ssh_user": "worker",
                    "docker_container_id": "container",
                    "docker_use_sudo": True,
                },
            }
        ]
    }
    executor = MagicMock()
    executor.run.side_effect = [SimpleNamespace(ok=alive, stdout="", stderr=""), SimpleNamespace(ok=True, stdout="")]
    with patch.dict(sys.modules, {"fabric": MagicMock()}):
        remote_module = importlib.import_module("vllm_pd_simulator.tools.ssh_remote_tools")
        with patch.object(remote_module, "SshRemote", return_value=executor) as remote:
            if alive:
                with pytest.raises(ValueError, match="still alive"):
                    verify_saved_shutdown(plan)
            else:
                verify_saved_shutdown(plan)
            remote.assert_called_once_with(
                host="host", ssh_port=2222, ssh_user="worker", docker_container_id="container", docker_use_sudo=True
            )
    assert plan["cleanup_confirmed"] is (not alive)
    executor.close.assert_called_once()


def test_export_loads_matching_archive_and_rejects_mismatch(tmp_path):
    import json

    plan, context, best = snapshot()
    plan.update(
        archive_id="attempt-1", candidate_id="c1", case_id="case1", config_sha256=context["deployment"]["config_sha256"]
    )
    archive = tmp_path / "pd_deployments" / "attempt-1"
    archive.mkdir(parents=True)
    (archive / "deployment.json").write_text(json.dumps(plan), encoding="utf-8")
    plan.pop("files")
    exporter._export_pd_snapshot(tmp_path, context, best, {})
    assert (tmp_path / "best_deployment" / "prefill.sh").exists()
    assert "prefill.sh" in (tmp_path / "best_deployment" / "README.md").read_text(encoding="utf-8")
    recorded = json.loads((archive / "deployment.json").read_text(encoding="utf-8"))
    recorded["candidate_id"] = "other"
    (archive / "deployment.json").write_text(json.dumps(recorded), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_id"):
        exporter._export_pd_snapshot(tmp_path, context, best, {})


def test_workflow_status_reports_unknown_remaining_time(tmp_path):
    import json
    import subprocess

    script = ROOT / ".agents/skills/optix-assistant/scripts/pd_workflow.py"
    (tmp_path / "pd_workflow.json").write_text(
        json.dumps({"budgets": {"total": {"max_trials": 4, "time_limit_seconds": 60}}}), encoding="utf-8"
    )
    archive = tmp_path / "pd_deployments" / "one"
    archive.mkdir(parents=True)
    (archive / "deployment.json").write_text(
        json.dumps(
            {
                "purpose": "search",
                "started_at": 1,
                "elapsed_unknown": True,
                "outcome": "aborted",
                "cleanup_confirmed": True,
                "layout": {"roles": {}},
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script), "status", "--run-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    status = json.loads(result.stdout)
    assert status["budget"]["total"]["remaining_trials"] == 3
    assert status["budget"]["total"]["remaining_seconds"] is None
    assert status["coverage"]["unknown_elapsed_attempts"] == 1

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


import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from optix.optimizer.pd_adapter import prepare_pd_fields, validate_runtime_placement, _snapshot


ROOT = Path(__file__).resolve().parents[4]
# Only transport objects are mocked; script generation uses the actual plugin.
# Restore sys.modules after import so unrelated tests retain their own dependencies.
fabric = types.ModuleType("fabric")
fabric.Config = MagicMock()
fabric.Connection = MagicMock()
invoke = types.ModuleType("invoke")
invoke_exceptions = types.ModuleType("invoke.exceptions")
invoke_exceptions.CommandTimedOut = type("CommandTimedOut", (Exception,), {})
invoke.exceptions = invoke_exceptions
with patch.dict(sys.modules, {"fabric": fabric, "invoke": invoke, "invoke.exceptions": invoke_exceptions}):
    from vllm_pd_simulator.config import VLLMPDDisaggConfig
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator


def simulator(tmp_path):
    sim = PdClusterSimulator.__new__(PdClusterSimulator)
    sim.config = VLLMPDDisaggConfig(
        nodes=[
            {"role": "prefill", "ssh_ip": "p-host", "bind_ip": "p-host", "gpu_ids": [0, 1], "service_port": 18080},
            {"role": "decode", "ssh_ip": "d-host", "bind_ip": "d-host", "gpu_ids": [0, 1], "service_port": 19080},
            {"role": "proxy", "ssh_ip": "proxy-host", "bind_ip": "proxy-host", "service_port": 8000},
        ]
    )
    sim.model_path = "/models/model"
    sim.served_model_name = "model"
    sim.vllm_others = '--compilation-config \'{"mode": 3}\''
    sim._prefill_tp_size = sim._prefill_dp_size = sim._decode_tp_size = sim._decode_dp_size = 1
    sim._prefill_env_vars = {}
    sim._decode_env_vars = {}
    sim._prefill_run_vars = {}
    sim._decode_run_vars = {}
    sim._is_moe = True
    sim._round_tmp_dir = tmp_path
    sim._node_infos = None
    sim._cluster_id = 0
    sim._remote_tmp_dir = "/tmp"
    sim._particle_list = [()]
    sim.bak_path = None
    sim._executors = {}
    return sim


def field(name, value, position="run"):
    return SimpleNamespace(name=name, value=value, config_position=position)


def test_actual_plugin_renders_role_parameters_and_json(tmp_path):
    sim = simulator(tmp_path)
    fields = [
        field("MAX_NUM_SEQS", 32),
        field("MAX_NUM_SEQS_prefill", 16),
        field("enforce_eager_decode", True),
        field("cudagraph_mode_prefill", "FULL"),
        field("TENSOR_PARALLEL_SIZE_prefill", 2),
        field("CONCURRENCY", 4, "env"),
    ]
    transported, layout = prepare_pd_fields(sim, fields)
    sim.vllm_others = ""
    sim.update_config(transported)
    validate_runtime_placement(sim)
    infos = sim._generate_all_scripts()
    sim._node_infos = infos
    plan = _snapshot(sim, layout)
    p_scripts = [text for name, text in plan["files"].items() if name.startswith("P-")]
    d_scripts = [text for name, text in plan["files"].items() if name.startswith("D-")]
    assert len(p_scripts) == 1 and len(d_scripts) == 2
    assert "--max-num-seqs 16" in p_scripts[0]
    assert "--tensor-parallel-size 2" in p_scripts[0]
    assert "--enforce-eager" not in p_scripts[0]
    assert all("--max-num-seqs 32" in text and "--enforce-eager" in text for text in d_scripts)
    assert '"cudagraph_mode": "FULL"' in p_scripts[0] and '"mode": 3' in p_scripts[0]
    assert not any("CONCURRENCY" in text or "--concurrency" in text for text in p_scripts + d_scripts)
    assert any(node["role"] == "proxy" for node in plan["nodes"])


def test_plugin_owned_flags_are_rejected(tmp_path):
    sim = simulator(tmp_path)
    with pytest.raises(ValueError, match="owned"):
        prepare_pd_fields(sim, [field("port_prefill", 1234)])


def test_derived_port_collision_rejected_before_remote_io(tmp_path):
    sim = simulator(tmp_path)
    sim.config.nodes[1].ssh_ip = sim.config.nodes[0].ssh_ip
    sim.config.nodes[1].gpu_ids = [2, 3]
    sim.config.nodes[1].service_port = sim.config.nodes[0].service_port
    transported, _ = prepare_pd_fields(sim, [])
    sim.update_config(transported)
    with pytest.raises(ValueError, match="collision"):
        validate_runtime_placement(sim)


def test_dense_model_dp_cannot_silently_render_as_one(tmp_path):
    sim = simulator(tmp_path)
    sim._is_moe = False
    with pytest.raises(ValueError, match="dense-model"):
        prepare_pd_fields(sim, [field("DATA_PARALLEL_SIZE_prefill", 2)])


def test_fixed_pp_aliases_do_not_render_invalid_flags(tmp_path):
    sim = simulator(tmp_path)
    fields, _ = prepare_pd_fields(sim, [field("PP_prefill", 1), field("PCP_decode", 1)])
    assert not any("--pp" in str(f.value) or "--pcp" in str(f.value) for f in fields)


@pytest.mark.parametrize("name", ["BAD-NAME_prefill", "ASCEND_RT_VISIBLE_DEVICES_decode", "HCCL_IF_IP_prefill"])
def test_environment_cannot_override_placement_or_use_invalid_names(tmp_path, name):
    with pytest.raises(ValueError):
        prepare_pd_fields(simulator(tmp_path), [field(name, "0", "env")])


def test_planned_endpoints_survive_missing_failure_process_records(tmp_path):
    sim = simulator(tmp_path)
    transported, _ = prepare_pd_fields(sim, [])
    sim.update_config(transported)
    nodes = validate_runtime_placement(sim)
    assert {node["role"] for node in nodes} == {"prefill", "decode", "proxy"}
    assert all(node["port"] > 0 and node["node_config"]["ssh_ip"] for node in nodes)
    assert [node["devices"] for node in nodes if node["role"] == "prefill"] == [[0], [1]]


@pytest.mark.parametrize("fail_start", [False, True])
def test_adapter_uses_real_plugin_run_and_preserves_failed_scripts(tmp_path, fail_start):
    import json
    from unittest.mock import patch
    from optix.optimizer.agentic.pd import configuration_fingerprint
    from optix.optimizer.pd_adapter import run_pd_trial

    sim = simulator(tmp_path)
    sim._round_tmp_dir = None
    sim._remote_tmp_dir = str(tmp_path / "staging")
    sim.run_log_fp = None
    sim.run_log = None
    config = tmp_path / "config.toml"
    config.write_text("frozen=true\n", encoding="utf-8")
    context = {
        "source": {"config_toml": str(config)},
        "deployment": {"type": "pd", "config_sha256": configuration_fingerprint(config)},
        "pd_budget": {"max_trials": 2, "time_limit_seconds": 100},
    }
    archive = tmp_path / "run/pd_deployments/case"
    archive.parent.parent.mkdir(parents=True)
    with (
        patch.object(sim, "_clear_remote_dirs"),
        patch.object(sim, "_upload_all_scripts"),
        patch.object(sim, "_cleanup_all_nodes"),
        patch.object(sim, "_conn_for_node"),
        patch.object(sim, "_exec_remote", return_value=(None if fail_start else 123, "remote.log")),
        patch.object(sim, "_verify_all_pids"),
        patch.object(sim, "_wait_for_all_services", return_value=True),
        patch.object(sim, "_kill_all_processes"),
        patch.object(sim, "_collect_remote_logs"),
        patch.object(sim, "_probe_node", return_value={"alive": False}),
    ):
        if fail_start:
            with pytest.raises(RuntimeError, match="failed to start"):
                run_pd_trial(sim, [field("MAX_NUM_SEQS", 16)], archive, context)
        else:
            run_pd_trial(sim, [field("MAX_NUM_SEQS", 16)], archive, context)
    recorded = json.loads((archive / "deployment.json").read_text(encoding="utf-8"))
    assert recorded["purpose"] == "search"
    assert recorded["config_sha256"] == configuration_fingerprint(config)
    assert any("--max-num-seqs 16" in text for name, text in recorded["files"].items() if name.startswith("P-"))
    assert {node["role"] for node in recorded["nodes"]} == {"prefill", "decode", "proxy"}
    assert sim.vllm_others == '--compilation-config \'{"mode": 3}\''
    assert sim.bak_path is None
    assert recorded["process_ids_recorded"] is (not fail_start)

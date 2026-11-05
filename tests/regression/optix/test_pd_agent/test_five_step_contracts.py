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
import time
from copy import deepcopy
from types import SimpleNamespace
import pytest
from optix.optimizer.agentic.pd_state import check_workflow_budget
from optix.optimizer.agentic.orchestrator import AgentOptimizer
from optix.optimizer.agentic.pd_layouts import enumerate_layouts, plan_node_ports
from tests.regression.optix.test_pd_agent.test_plugin_render import simulator
from optix.optimizer.pd_adapter import validate_runtime_placement


def policy(root):
    budgets = {
        s: {"max_trials": 2, "time_limit_seconds": 100}
        for s in ("bootstrap", "baseline", "prefill_probe", "decode_probe", "search", "verification")
    }
    budgets["verification"]["max_trials"] = 4
    budgets["total"] = {"max_trials": 8, "time_limit_seconds": 600}
    data = {"version": 1, "budgets": budgets}
    (root / "pd_workflow.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def attempt(root, name, purpose, child=False):
    folder = root / ("pd_tasks" if child else "pd_deployments") / name
    folder.mkdir(parents=True)
    (folder / ("attempt.json" if child else "deployment.json")).write_text(
        json.dumps(
            {"purpose": purpose, "started_at": time.time() - 1, "finished_at": time.time(), "outcome": "failed"}
        ),
        encoding="utf-8",
    )


def test_workflow_counts_failed_children_and_reserves_verification(tmp_path):
    policy(tmp_path)
    for i, purpose in enumerate(("bootstrap", "baseline", "prefill_probe", "search")):
        attempt(tmp_path, str(i), purpose, child=purpose.endswith("probe"))
    with pytest.raises(ValueError, match="reserved"):
        check_workflow_budget(tmp_path, "decode_probe")
    assert check_workflow_budget(tmp_path, "verification_baseline")["attempts"] == 4


def test_workflow_is_frozen_and_child_budget_survives_restart(tmp_path):
    data = policy(tmp_path)
    check_workflow_budget(tmp_path, "prefill_probe")
    attempt(tmp_path, "one", "prefill_probe", True)
    attempt(tmp_path, "two", "prefill_probe", True)
    with pytest.raises(ValueError, match="prefill_probe budget"):
        check_workflow_budget(tmp_path, "prefill_probe")
    data["budgets"]["total"]["max_trials"] = 9
    (tmp_path / "pd_workflow.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        check_workflow_budget(tmp_path, "search")


def test_old_mode_does_not_require_new_policy(tmp_path):
    assert check_workflow_budget(tmp_path, "search") is None
    with pytest.raises(ValueError, match="require"):
        check_workflow_budget(tmp_path, "bootstrap")


def test_same_host_ports_planned_across_source_entries_and_roles(tmp_path):
    sim = simulator(tmp_path)
    for n in sim.config.nodes[:2]:
        n.ssh_ip = n.bind_ip = "one-host"
    sim.config.nodes[1].gpu_ids = [2, 3]
    plan_node_ports(sim.config)
    sim._apply_ep_split()
    assert len(validate_runtime_placement(sim)) == 5
    sim.config.decode_groups[0].nodes[0].ascend_base_port = sim.config.prefill_groups[0].nodes[0].ascend_base_port
    with pytest.raises(ValueError, match="collision"):
        validate_runtime_placement(sim)


def test_layouts_bounded_and_never_duplicate_cards(tmp_path):
    sim = simulator(tmp_path)
    for n in sim.config.nodes[:2]:
        n.ssh_ip = "one-host"
    sim.config.nodes[1].gpu_ids = [2, 3]
    result = enumerate_layouts(sim.config, {}, max_candidates=2)
    assert result["layouts"] and result["truncated"]
    for item in result["layouts"]:
        cards = [tuple(x) for x in item["layout"]["devices"]]
        assert len(cards) == len(set(cards)) == 4
    assert enumerate_layouts(sim.config, {}, max_combinations=1)["truncated"]


def test_invalid_baseline_never_accepted(tmp_path):
    ctx = {"model": {"name": "x"}, "benchmark_spec": {"output": 128}}
    agent = AgentOptimizer(
        pso=SimpleNamespace(scheduler=SimpleNamespace(pd_agent_context=ctx)),
        agent_config=SimpleNamespace(),
        run_dir=tmp_path,
    )
    saved = {
        "pd_context": deepcopy(ctx),
        "fitness": 1,
        "performance": {"throughput": 10, "success_rate": 1, "time_to_first_token": 0.1, "time_per_output_token": 0.01},
    }
    agent._validate_pd_baseline(saved)
    for key, value in (
        ("throughput", 0),
        ("success_rate", 0),
        ("time_per_output_token", float("nan")),
        ("throughput", "10"),
    ):
        bad = deepcopy(saved)
        bad["performance"][key] = value
        with pytest.raises(ValueError):
            agent._validate_pd_baseline(bad)
    bad = deepcopy(saved)
    bad["pd_context"]["benchmark_spec"]["output"] = 1
    with pytest.raises(ValueError, match="differs"):
        agent._validate_pd_baseline(bad)


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_phase_protocol_and_cache_binding(role):
    from optix.config.config import PdDisaggPhaseConfig
    from optix.optimizer.agentic.pd_phase import validate_protocol, capacity_key, matching_capacity

    output = 1 if role == "prefill" else 128
    phase = PdDisaggPhaseConfig(
        benchmark_run_count=1 if role == "prefill" else 2,
        simulator_command_overrides={
            "others": "--tensor-parallel-size 1 "
            + ("--no-enable-prefix-caching" if role == "prefill" else "--enable-prefix-caching")
        },
        benchmark_command_overrides={
            "others": f"--seed 1 --num-prompts 4 --random-input-len 128 --random-output-len {output}"
        },
    )
    binding = {
        **validate_protocol(role, phase, {"MAX_NUM_SEQS": 8}),
        "software": {"vllm": "known-version"},
        "hardware": {"device": "test"},
    }
    assert binding["role"] == role
    record = {"outcome": "measured", "capacity_key": capacity_key(binding), "qps": 10}
    assert matching_capacity(record, binding)
    assert not matching_capacity(record, {**binding, "seed": "different"})
    with pytest.raises(ValueError, match="protocol"):
        validate_protocol(role, phase, {"ENABLE_PREFIX_CACHING": True})
    if role == "decode":
        phase.benchmark_run_count = 1
        with pytest.raises(ValueError, match="warmup"):
            validate_protocol(role, phase, {})


def test_phase_executes_one_agent_candidate_without_pso(tmp_path, monkeypatch):
    import optix.optimizer.agentic.validation as validation
    import optix.optimizer.agentic.strategies as strategies
    import optix.optimizer.scheduler as schedulers
    import optix.optimizer.pd_disagg as builtin
    from optix.config.config import PdDisaggPhaseConfig, PerformanceIndex
    from optix.optimizer.agentic.pd_phase import run_phase_task
    from optix.optimizer.agentic.candidates import TrialResult

    policy(tmp_path)
    model = {"path": "/model", "name": "model"}
    experiment = {
        "model": model,
        "model_info": {},
        "benchmark_spec": {"input_len_avg": 128, "output_len_avg": 128},
        "deployment": {"device_inventory": [["localhost", 0]]},
    }
    (tmp_path / "pd_experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
    parent = tmp_path / "context.json"
    parent.write_text("do not overwrite", encoding="utf-8")
    (tmp_path / "baseline.json").write_text("existing business baseline", encoding="utf-8")
    (tmp_path / "pd_instance_specs.json").write_text(
        json.dumps({r: {"tp": 1, "dp": 1} for r in ("prefill", "decode")}), encoding="utf-8"
    )
    phase = PdDisaggPhaseConfig(
        simulator_command_overrides={"others": "--tensor-parallel-size 1 --no-enable-prefix-caching"},
        benchmark_command_overrides={"others": "--seed 1 --num-prompts 4 --random-input-len 128 --random-output-len 1"},
    )
    settings = SimpleNamespace(
        manage_simulator_lifecycle=True,
        use_request_rate_calibration=False,
        wait_start_time=1,
        generate_speed_target=100,
        vllm=SimpleNamespace(command=SimpleNamespace(model="/model")),
        vllm_benchmark=SimpleNamespace(command=SimpleNamespace(dataset_name="random")),
        **{
            k: getattr(phase, k)
            for k in (
                "ttft_penalty",
                "tpot_penalty",
                "success_rate_penalty",
                "ttft_slo",
                "tpot_slo",
                "success_rate_slo",
                "fine_tune_mode",
            )
        },
    )

    class Plugin:
        def __init__(self, **kwargs):
            self.env = kwargs["deploy_env"]
            self.config = SimpleNamespace(command=SimpleNamespace(port="8000"))

    class Scheduler:
        def __init__(self, *args, **kwargs):
            self.stopped = False

        def stop_target_server(self):
            self.stopped = True

    runner = SimpleNamespace(
        settings=settings,
        deploy_env={},
        runtime_ctx=None,
        _resolve_and_validate_plugins=lambda *a: (Plugin, Plugin),
        _resolve_target_fields=lambda *a: (),
        _apply_simulator_command_overrides=lambda *a: None,
        _apply_benchmark_command_overrides=lambda *a: None,
        _configure_phase_hook=lambda *a: None,
    )
    monkeypatch.setattr(
        validation,
        "validate_candidate",
        lambda *a: {"valid": True, "params": {"MAX_NUM_SEQS": 8}, "candidate_id": "p1"},
    )
    monkeypatch.setattr(schedulers, "Scheduler", Scheduler)
    from optix.optimizer.agentic import pd_phase
    from optix.optimizer.plugins import simulate

    monkeypatch.setattr(simulate, "VllmSimulator", Plugin)
    monkeypatch.setattr(pd_phase, "verify_phase_shutdown", lambda record: record.update(cleanup_confirmed=True))
    monkeypatch.setattr(builtin.OptixPhaseRunner, "run", lambda *a: pytest.fail("must not enter PSO"))
    calls = []

    def trial(self, v, scheduler, fields, context):
        calls.append(v)
        return TrialResult(
            params=v["params"], fitness=1, performance=PerformanceIndex(throughput=10), candidate_id="p1"
        )

    monkeypatch.setattr(strategies.AgentCandidateStrategy, "_run_trial", trial)
    result = run_phase_task(
        tmp_path,
        "p1",
        "prefill",
        {"ssh_ip": "localhost", "gpu_ids": [0]},
        {},
        {"model": model, "model_info": {}},
        phase,
        runner,
    )
    assert result["qps"] == 10 and result["cleanup_confirmed"]
    assert len(calls) == 1 and parent.read_text(encoding="utf-8") == "do not overwrite"
    assert (tmp_path / "baseline.json").read_text(encoding="utf-8") == "existing business baseline"
    with pytest.raises(ValueError, match="immutable"):
        run_phase_task(
            tmp_path,
            "p1",
            "prefill",
            {"ssh_ip": "localhost", "gpu_ids": [0]},
            {},
            {"model": model, "model_info": {}},
            phase,
            runner,
        )


def test_budget_timer_restored_after_expiration(tmp_path, monkeypatch):
    import signal
    import optix.optimizer.agentic.pd_state as state

    monkeypatch.setattr(state, "workflow_seconds_remaining", lambda *a: 0.01)
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(state.WorkflowBudgetExceeded):
        with state.workflow_time_limit(tmp_path, "search"):
            time.sleep(0.1)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    assert signal.getsignal(signal.SIGALRM) == previous


def test_instance_specs_frozen_only_in_new_workflow(tmp_path):
    from optix.optimizer.agentic.pd_state import freeze_instance_specs, validate_workflow_candidate

    layout = {"roles": {r: {"tp": 2, "dp": 1} for r in ("prefill", "decode")}}
    freeze_instance_specs(tmp_path, layout)
    assert not (tmp_path / "pd_instance_specs.json").exists()
    policy(tmp_path)
    freeze_instance_specs(tmp_path, layout)
    validate_workflow_candidate(tmp_path, {}, layout)
    bad = deepcopy(layout)
    bad["roles"]["prefill"]["tp"] = 1
    with pytest.raises(ValueError, match="fixed"):
        validate_workflow_candidate(tmp_path, {}, bad)


def test_invalid_checkpoint_is_preserved_and_never_used(tmp_path):
    from unittest.mock import Mock

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"fitness": float("inf"), "performance": {}}), encoding="utf-8")
    pso = SimpleNamespace(
        scheduler=SimpleNamespace(engine="vllm_pd", pd_trial_purpose="baseline", pd_agent_context={}),
        prepare_plugin=Mock(),
    )
    agent = AgentOptimizer(pso=pso, agent_config=SimpleNamespace(), run_dir=tmp_path)
    with pytest.raises(ValueError, match="Invalid PD baseline"):
        agent._ensure_baseline()
    pso.prepare_plugin.assert_not_called()
    assert path.exists()


def test_bootstrap_does_not_restore_or_replace_business_baseline(tmp_path):
    from optix.config.config import PerformanceIndex

    path = tmp_path / "baseline.json"
    path.write_text("preserve business baseline", encoding="utf-8")
    pso = SimpleNamespace(
        scheduler=SimpleNamespace(engine="vllm_pd", pd_trial_purpose="bootstrap", pd_agent_context={}),
        _baseline_from_checkpoint=True,
        default_fitness=1,
        default_res=PerformanceIndex(
            throughput=10, success_rate=1, time_to_first_token=0.1, time_per_output_token=0.01
        ),
        prepare_plugin=lambda: None,
    )
    agent = AgentOptimizer(pso=pso, agent_config=SimpleNamespace(), run_dir=tmp_path)
    agent._ensure_baseline()
    assert not pso._baseline_from_checkpoint
    assert path.read_text(encoding="utf-8") == "preserve business baseline"


def test_unconfirmed_phase_blocks_next_deployment(tmp_path):
    from optix.optimizer.agentic.pd_state import require_phase_cleanup

    attempt(tmp_path, "unfinished", "prefill_probe", True)
    with pytest.raises(ValueError, match="cleanup"):
        require_phase_cleanup(tmp_path)


def test_startup_choices_prioritize_different_specs(tmp_path):
    from optix.optimizer.agentic.pd_bootstrap import initial_layouts

    sim = simulator(tmp_path)
    result = initial_layouts(sim.config, {})
    pairs = [
        (c["params"]["TENSOR_PARALLEL_SIZE_prefill"], c["params"]["TENSOR_PARALLEL_SIZE_decode"])
        for c in result["candidates"]
    ]
    assert len(pairs) == len(set(pairs)) and (2, 2) in pairs
    fixed = initial_layouts(sim.config, {"TENSOR_PARALLEL_SIZE_prefill": 2})
    assert all(c["params"]["TENSOR_PARALLEL_SIZE_prefill"] == 2 for c in fixed["candidates"])


def test_capacity_ranking_hint_requires_matching_service_params():
    from optix.optimizer.agentic.pd_phase import current_capacity_hint

    params = {"TENSOR_PARALLEL_SIZE_prefill": 2, "MAX_NUM_SEQS_prefill": 16}
    record = {
        "outcome": "measured",
        "cleanup_confirmed": True,
        "qps": 20,
        "binding": {
            "experiment_sha256": "experiment",
            "protocol": {"role": "prefill"},
            "resource": {"gpu_ids": [0, 1]},
            "params": {"MAX_NUM_SEQS": 16},
            "phase_config": {
                "simulator_command_overrides": {"others": "--tensor-parallel-size 2 --no-enable-prefix-caching"}
            },
        },
    }
    assert current_capacity_hint(record, "experiment", "prefill", params) == 20
    assert current_capacity_hint(record, "other", "prefill", params) is None
    assert current_capacity_hint(record, "experiment", "prefill", {**params, "MAX_NUM_SEQS_prefill": 32}) is None

    assert (
        current_capacity_hint(record, "experiment", "prefill", {**params, "MAX_NUM_BATCHED_TOKENS_prefill": 1024})
        is None
    )


def test_checkpoint_without_launch_layout_is_rejected(tmp_path):
    from unittest.mock import Mock

    saved = {
        "fitness": 1,
        "performance": {"throughput": 10, "success_rate": 1, "time_to_first_token": 0.1, "time_per_output_token": 0.01},
        "pd_context": {},
    }
    (tmp_path / "baseline.json").write_text(json.dumps(saved), encoding="utf-8")
    pso = SimpleNamespace(
        scheduler=SimpleNamespace(engine="vllm_pd", pd_trial_purpose="baseline", pd_agent_context={}),
        prepare_plugin=Mock(),
    )
    agent = AgentOptimizer(pso=pso, agent_config=SimpleNamespace(), run_dir=tmp_path)
    with pytest.raises(ValueError, match="Invalid PD baseline") as error:
        agent._ensure_baseline()
    assert "launch layout" in str(error.value.__cause__)
    pso.prepare_plugin.assert_not_called()

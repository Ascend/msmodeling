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

"""Regression tests for bound PD dispatch and immutable trial identity; no device starts."""

import json
import subprocess
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import pytest
import tomlkit
import run_pd_round as dispatch
from tests.regression.optix.test_pd_agent.test_config_generation import experiment
from optix.optimizer.agentic.pd_execution import (
    validate_execution,
    validate_loaded_config,
    trial_identity,
    validate_result_identity,
)
from optix.optimizer.agentic.pd import configuration_fingerprint
from optix.optimizer.scheduler import Scheduler


ROOT = Path(__file__).resolve().parents[4]


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def setup_run(root):
    ctx = experiment(root)
    base = Path(ctx["source"]["config_toml"])
    doc = tomlkit.parse(base.read_text(encoding="utf-8"))
    doc["vllm"]["command"]["others"] = "--quantization ascend"
    doc["optimizer_strategy"] = "agent"
    doc["agent_optimizer"] = {"run_id": root.name}
    base.write_text(tomlkit.dumps(doc), encoding="utf-8")
    ctx["deployment"]["config_sha256"] = configuration_fingerprint(base)
    ctx.update(
        run_id=root.name,
        benchmark="vllm_benchmark",
        model={"name": "model", "path": "/models/model"},
        model_info={},
        benchmark_spec={
            "input_len_avg": 128,
            "output_len_avg": 32,
            "num_requests": 4,
            "concurrency": 1,
            "concurrency_range": None,
        },
        hardware={"world_size": 4},
    )
    proposal = {
        "round": 1,
        "candidates": [
            {
                "candidate_id": "first",
                "params": {},
                "deployment": {
                    "nodes": [
                        {"source_node": 0, "role": "prefill", "gpu_ids": [0, 1]},
                        {"source_node": 1, "role": "decode", "gpu_ids": [0, 1]},
                    ]
                },
            }
        ],
    }
    dump(root / "pd_experiment.json", ctx)
    dump(root / "context.json", ctx)
    dump(root / "candidates.round-1.json", proposal)
    return ctx


def prepare(root):
    setup_run(root)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[1].endswith("collect_context.py"):
            subprocess.run(
                argv,
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "contrib/optix/vllm_pd_simulator"),
                },
            )
        elif argv[1].endswith("validate_candidates.py"):
            from optix.optimizer.agentic.validation import validate_candidate

            ctx = json.loads((root / "context.json").read_text(encoding="utf-8"))
            candidate = json.loads((root / "candidates.round-1.json").read_text(encoding="utf-8"))["candidates"][0]
            result = validate_candidate(ctx, candidate)
            assert result["valid"], result["errors"]

    record = dispatch.execute_round(root, 1, runner=runner)
    return (
        json.loads((root / "context.json").read_text(encoding="utf-8")),
        json.loads(Path(record).read_text(encoding="utf-8")),
        calls,
    )


def test_dispatch_binds_generator_preflight_and_launch(tmp_path):
    ctx, record, calls = prepare(tmp_path)
    config = record["config"]["path"]
    assert [Path(c[1]).name for c in calls[:3]] == [
        "collect_context.py",
        "config_preflight.py",
        "validate_candidates.py",
    ]
    assert calls[-1][1:3] == ["-m", "optix"]
    assert calls[-1][calls[-1].index("-c") + 1] == config == ctx["source"]["config_toml"]
    assert record["round"] == 1 and record["attempt"] == 1 and record["candidate_id"] == "first"
    assert validate_execution(ctx) == record
    assert not (tmp_path / ".pd_dispatch/pd_controller.json").exists()
    doc = tomlkit.parse(Path(config).read_text(encoding="utf-8"))
    assert doc["vllm"]["command"]["others"] == "--quantization ascend"
    assert doc["agent_optimizer"]["run_dir"] == str(tmp_path)


@pytest.mark.parametrize("artifact", ["config", "context", "candidate"])
def test_prepared_input_tampering_is_rejected(tmp_path, artifact):
    ctx, record, _ = prepare(tmp_path)
    path = Path(record[artifact]["path"])
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match=artifact + " changed"):
        validate_execution(ctx)


def test_wrong_live_context_rejected(tmp_path):
    ctx, _, _ = prepare(tmp_path)
    ctx["search_space"]["parameters"][0]["default"] = 2
    with pytest.raises(ValueError, match="search_space"):
        validate_execution(ctx)


def test_preflight_failure_never_launches(tmp_path):
    setup_run(tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[1].endswith("collect_context.py"):
            subprocess.run(
                argv,
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "contrib/optix/vllm_pd_simulator"),
                },
            )
        elif argv[1].endswith("config_preflight.py"):
            raise subprocess.CalledProcessError(1, argv)

    with pytest.raises(subprocess.CalledProcessError):
        dispatch.execute_round(tmp_path, 1, runner=runner)
    assert not any("-m" in c for c in calls)
    assert not (tmp_path / "execution.round-1.attempt-1.json").exists()
    assert not (tmp_path / ".pd_dispatch/pd_controller.json").exists()


def test_actual_service_and_config_source_checked(tmp_path):
    ctx, record, _ = prepare(tmp_path)
    command = record["effective_settings"]["vllm"]["command"]
    sim = S(model_path=command["model"], served_model_name=command["served_model_name"], vllm_others=command["others"])
    settings = S(
        model_config={"toml_file": [ctx["source"]["config_toml"]]},
        model_dump=lambda **kwargs: {
            **record["effective_settings"],
            "vllm": record["effective_settings"]["vllm"],
            ctx["benchmark"]: {"command": record["effective_settings"]["benchmark"]},
        },
    )
    with patch("optix.config.config.get_settings", return_value=settings):
        validate_loaded_config(ctx, sim)
        settings.model_config["toml_file"] = [tmp_path / "wrong.toml"]
        with pytest.raises(ValueError, match="exact preflight"):
            validate_loaded_config(ctx, sim)
    sim.vllm_others = ""
    with pytest.raises(ValueError, match="others differs"):
        validate_loaded_config(ctx, sim)


def test_baseline_search_and_retry_have_distinct_archive_identity():
    ctx = {"deployment": {"round_manifest": {"round": 18, "attempt": 2, "candidate_id": "gamma2"}}}
    a = trial_identity(ctx, "baseline", "case_000001", {"candidate_id": "baseline"})
    b = trial_identity(ctx, "search", "case_000001")
    c = trial_identity(
        {"deployment": {"round_manifest": {"round": 19, "attempt": 2, "candidate_id": "next"}}}, "search", "case_000001"
    )
    assert len({i["archive_id"] for i in (a, b, c)}) == 3
    assert b["candidate_id"] == "gamma2" and b["round"] == 18 and "round-18.attempt-2" in b["archive_id"]


def test_old_plan_cannot_be_relabelled_as_next_candidate():
    plan = {"case_id": "case_1", "candidate_id": "old", "archive_id": "round-18"}
    validate_result_identity(plan, "case_1", "old", Path("round-18"))
    with pytest.raises(ValueError, match="candidate"):
        validate_result_identity(plan, "case_1", "new")
    with pytest.raises(ValueError, match="case"):
        validate_result_identity(plan, "case_2", "old")
    with pytest.raises(ValueError, match="relabeled"):
        validate_result_identity(plan, "case_1", "old", Path("round-19"))
    assert plan["candidate_id"] == "old"


def test_new_case_clears_previous_plan():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = "vllm_pd"
    scheduler.data_storage = S(next_case_id=lambda: "case_2")
    scheduler.simulator = S(_optix_pd_launch_plan={"candidate_id": "old"})
    scheduler.simulate_launch_plan = {"candidate_id": "old"}
    assert scheduler._new_case_id() == "case_2"
    assert scheduler.simulator._optix_pd_launch_plan == scheduler.simulate_launch_plan == {}


def test_runtime_archive_path_uses_current_candidate(tmp_path):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = "vllm_pd"
    scheduler.pd_agent_context = {
        "deployment": {"round_manifest": {"round": 18, "attempt": 1, "candidate_id": "gamma2"}}
    }
    scheduler.pd_candidate_identity = {"round": 18, "candidate_id": "gamma2"}
    scheduler.pd_trial_purpose = "search"
    scheduler.current_case_id = "case_000001"
    scheduler.pd_archive_root = tmp_path
    scheduler.simulate_run_info = []
    scheduler.benchmark = S()
    scheduler.simulator = S(_optix_pd_launch_plan={})
    scheduler.wait_simulate = lambda: None
    with patch("optix.optimizer.pd_adapter.run_pd_trial", return_value={}) as call:
        scheduler.run_simulate(None, ())
        identity = call.call_args.kwargs["identity"]
        assert identity["round"] == 18 and identity["candidate_id"] == "gamma2"
        assert call.call_args.args[2].name == identity["archive_id"]


@pytest.mark.parametrize("changed", ["candidate", "config", "context"])
def test_input_changed_by_preflight_is_not_bound_or_launched(tmp_path, changed):
    setup_run(tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[1].endswith("collect_context.py"):
            subprocess.run(
                argv,
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "contrib/optix/vllm_pd_simulator"),
                },
            )
        if argv[1].endswith("config_preflight.py"):
            paths = {
                "candidate": tmp_path / "candidates.round-1.json",
                "config": tmp_path / "config.round-1.toml",
                "context": tmp_path / "context.json",
            }
            paths[changed].write_text(paths[changed].read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="changed during preflight"):
        dispatch.execute_round(tmp_path, 1, runner=runner)
    assert not any("-m" in c for c in calls)
    assert not (tmp_path / "execution.round-1.attempt-1.json").exists()


def test_prepare_only_and_retry_preserve_first_record(tmp_path):
    ctx, record, _ = prepare(tmp_path)
    first_snapshot = Path(record["context"]["path"]).read_bytes()
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[1].endswith("collect_context.py"):
            subprocess.run(
                argv,
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "contrib/optix/vllm_pd_simulator"),
                },
            )

    second = dispatch.execute_round(tmp_path, 1, attempt=2, prepare_only=True, runner=runner)
    assert not any("-m" in c for c in calls)
    assert json.loads(Path(second).read_text(encoding="utf-8"))["attempt"] == 2
    assert Path(record["context"]["path"]).read_bytes() == first_snapshot
    assert validate_execution(ctx) == record


def test_unprepared_generated_round_cannot_launch(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("a=1", encoding="utf-8")
    context = {"source": {"config_toml": str(path)}, "deployment": {"round_manifest": {"round": 1}}}
    with pytest.raises(ValueError, match="run_pd_round.py"):
        validate_loaded_config(context, S())


def test_candidate_cannot_borrow_another_rounds_configuration():
    context = {"deployment": {"round_manifest": {"round": 18, "attempt": 1, "candidate_id": "gamma2"}}}
    with pytest.raises(ValueError, match="round"):
        trial_identity(context, "search", "case_1", {"round": 19, "candidate_id": "gamma2"})
    with pytest.raises(ValueError, match="candidate_id"):
        trial_identity(context, "search", "case_1", {"round": 18, "candidate_id": "other"})


def test_save_result_cannot_overwrite_previous_attempt(tmp_path):
    archive = tmp_path / "round-18"
    archive.mkdir()
    path = archive / "deployment.json"
    path.write_text('{"candidate_id":"old"}', encoding="utf-8")
    plan = {"case_id": "old-case", "candidate_id": "old", "archive_id": "round-18"}
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = "vllm_pd"
    scheduler.pd_agent_context = {}
    scheduler.current_case_id = "new-case"
    scheduler._persist_trial_log = lambda *a: None
    scheduler._finalize_report_only_time_saving = lambda *a: None
    scheduler.run_start_timestamp = None
    scheduler.performance_index = S()
    scheduler.last_outcome = None
    scheduler._error_info = None
    scheduler.simulator = S(_optix_pd_launch_plan=plan, _optix_pd_archive=archive)
    with pytest.raises(ValueError, match="case"):
        scheduler.save_result(candidate_id="next", run_dir=str(tmp_path), fitness=1)
    assert path.read_text(encoding="utf-8") == '{"candidate_id":"old"}'
    assert plan["candidate_id"] == "old"


def test_semantic_settings_change_is_rejected_after_preflight(tmp_path):
    ctx, record, _ = prepare(tmp_path)
    command = record["effective_settings"]["vllm"]["command"]
    sim = S(model_path=command["model"], served_model_name=command["served_model_name"], vllm_others=command["others"])
    loaded = deepcopy(record["effective_settings"])
    loaded[ctx["benchmark"]] = {"command": loaded.pop("benchmark")}
    loaded["ttft_slo"] = 999
    settings = S(model_config={"toml_file": [ctx["source"]["config_toml"]]}, model_dump=lambda **kwargs: loaded)
    with patch("optix.config.config.get_settings", return_value=settings):
        with pytest.raises(ValueError, match="settings differ"):
            validate_loaded_config(ctx, sim)


@pytest.mark.parametrize("change_parameter", [False, True])
def test_real_adapter_compares_preflight_parameter_plan_before_start(tmp_path, change_parameter):
    from optix.optimizer.agentic.pd_execution import load_execution_settings
    from optix.optimizer.agentic.pd import load_plugin_config
    from optix.optimizer.agentic.parameters import resolve_execution_fields
    from optix.optimizer.agentic.validation import validate_candidate
    from optix.config.config import range_to_enum, field_to_param, map_param_with_value
    from optix.optimizer.pd_adapter import run_pd_trial
    from tests.regression.optix.test_pd_agent.test_plugin_render import simulator

    ctx, record, _ = prepare(tmp_path)
    settings = load_execution_settings(record["config"]["path"])
    sim = simulator(tmp_path / "scripts")
    sim.config = load_plugin_config(record["config"]["path"])
    names = {f.name for f in sim.config.target_field}
    sim.config.target_field.extend(f for f in settings.vllm.target_field if f.name not in names)
    sim.model_path = settings.vllm.command.model
    sim.served_model_name = settings.vllm.command.served_model_name
    sim.vllm_others = settings.vllm.command.others
    range_to_enum(sim.config.target_field)
    proposal = json.loads((tmp_path / "candidates.round-1.json").read_text(encoding="utf-8"))["candidates"][0]
    validation = validate_candidate(ctx, proposal)
    fields = resolve_execution_fields(validation["params"], sim.config.target_field, ctx)
    fields = map_param_with_value(field_to_param(tuple(fields)), tuple(fields))
    if change_parameter:
        next(f for f in fields if f.name == "TENSOR_PARALLEL_SIZE_prefill").value = 2
    identity = trial_identity(ctx, "search", "case_1")
    archive = tmp_path / "pd_deployments" / identity["archive_id"]
    with (
        patch("optix.config.config.get_settings", return_value=settings),
        patch.object(sim, "update_config", side_effect=RuntimeError("STOP_BEFORE_REMOTE")) as update,
    ):
        if change_parameter:
            with pytest.raises(ValueError, match="effective launch parameters"):
                run_pd_trial(sim, fields, archive, ctx, identity=identity)
            update.assert_not_called()
        else:
            with pytest.raises(RuntimeError, match="STOP_BEFORE_REMOTE"):
                run_pd_trial(sim, fields, archive, ctx, identity=identity)
            update.assert_called_once()
    assert not archive.exists()


def test_retries_within_one_case_have_separate_archives():
    context = {"deployment": {"round_manifest": {"round": 18, "attempt": 1, "candidate_id": "gamma2"}}}
    first = trial_identity(context, "search", "case_1", launch_attempt=1)
    second = trial_identity(context, "search", "case_1", launch_attempt=2)
    assert first["archive_id"] != second["archive_id"]
    assert first["candidate_id"] == second["candidate_id"] == "gamma2"


@pytest.mark.parametrize("complete", [False, True])
def test_gamma_candidate_requires_and_preserves_method_in_generated_configuration(tmp_path, complete):
    from optix.optimizer.agentic.pd import load_plugin_config

    ctx = setup_run(tmp_path)
    base = Path(ctx["source"]["config_toml"])
    if complete:
        doc = tomlkit.parse(base.read_text(encoding="utf-8"))
        doc["vllm"]["command"]["others"] = "--quantization ascend --speculative-config '{\"method\":\"mtp\"}'"
        base.write_text(tomlkit.dumps(doc), encoding="utf-8")
        ctx["deployment"]["config_sha256"] = configuration_fingerprint(base)
        ctx["deployment"]["plugin_config"] = load_plugin_config(base).model_dump(mode="json")
    ctx["search_space"]["parameters"].append(
        {
            "name": "num_speculative_tokens_decode",
            "dtype": "int",
            "config_position": "run",
            "min": 0,
            "max": 3,
            "default": 0,
        }
    )
    proposal = json.loads((tmp_path / "candidates.round-1.json").read_text(encoding="utf-8"))
    proposal["candidates"][0]["params"]["num_speculative_tokens_decode"] = 2
    dump(tmp_path / "context.json", ctx)
    dump(tmp_path / "pd_experiment.json", ctx)
    dump(tmp_path / "candidates.round-1.json", proposal)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[1].endswith("collect_context.py"):
            subprocess.run(
                argv,
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "contrib/optix/vllm_pd_simulator"),
                },
            )

    if not complete:
        with pytest.raises(SystemExit):
            dispatch.execute_round(tmp_path, 1, runner=runner)
        assert calls == []
        assert not (tmp_path / "execution.round-1.attempt-1.json").exists()
    else:
        record_path = dispatch.execute_round(tmp_path, 1, prepare_only=True, runner=runner)
        record = json.loads(Path(record_path).read_text(encoding="utf-8"))
        spec = record["parameter_plan"]["OPTIX_SERVE_ARGS_decode"]["containers"]["--speculative-config"]
        assert spec == {"method": "mtp", "num_speculative_tokens": 2}
        config = tomlkit.parse(Path(record["config"]["path"]).read_text(encoding="utf-8"))
        assert "mtp" in config["vllm"]["command"]["others"]


def test_parameter_plan_normalizes_inline_values():
    from optix.optimizer.agentic.pd_execution import parameter_plan

    def plan(value):
        return parameter_plan([S(name="OPTIX_SERVE_ARGS_prefill", value=value)])

    assert plan("--max-model-len=8192 --flag") == plan("--max-model-len 8192 --flag")
    assert plan("--max-model-len=8192") != plan("--max-model-len=4096")


def test_retry_preserves_each_deployment_manifest(tmp_path):
    import generate_pd_config

    setup_run(tmp_path)
    proposal = tmp_path / "candidates.round-1.json"
    args = ["--run-dir", str(tmp_path), "--round", "1", "--proposal", str(proposal)]
    generate_pd_config.main(args)
    first = (tmp_path / "deployment.round-1.json").read_text(encoding="utf-8")
    generate_pd_config.main(args + ["--attempt", "2"])
    assert (tmp_path / "deployment.round-1.json").read_text(encoding="utf-8") == first
    second = json.loads((tmp_path / "deployment.round-1.attempt-2.json").read_text(encoding="utf-8"))
    assert second["attempt"] == 2
    assert Path(second["config_toml"]).name == "config.round-1.attempt-2.toml"

"""Tests for the internal optimizer workload policy."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

import torch

from tensor_cast.device import DeviceProfile
from tensor_cast.performance_model.profiling_database.query_demand import QUERY_TRACE_DIR_ENV
from tools.perf_data_collection.grid_generator.query_model import QueryModelArchitecture
from tools.perf_data_collection.grid_generator.query_workloads import (
    WorkloadScenario,
    build_workload_scenarios,
    run_query_workloads,
)


def _model_config() -> QueryModelArchitecture:
    return QueryModelArchitecture(
        num_experts=64,
        tp_sizes=(1, 2, 4),
        ep_sizes=(1, 2, 4),
        max_context_length=10000,
        num_mtp_layers=2,
    )


def _scenario(input_length: int = 128) -> WorkloadScenario:
    return WorkloadScenario(
        model_id="org/model",
        device="TEST_DEVICE",
        num_devices=1,
        input_length=input_length,
        output_length=1,
        max_batched_tokens=None,
        tp_sizes=(1,),
        ep_sizes=(1,),
        moe_dp_sizes=(1,),
        dcp_sizes=(1,),
        mtp_tokens=(),
    )


def test_policy_includes_real_non_power_of_two_topology_and_long_sequence_modes(tmp_path: Path) -> None:
    mapping = {
        "operator_mappings": {
            "tensor_cast.fused_moe.default": {
                "kernel_type": "DispatchFFNCombine",
                "query_mode": "moe_fused",
            }
        }
    }
    profile = SimpleNamespace(comm_grid=SimpleNamespace(grid=torch.empty(12)))
    with (
        mock.patch.dict(DeviceProfile.all_device_profiles, {"TEST_DEVICE": profile}),
        mock.patch(
            "tools.perf_data_collection.grid_generator.query_workloads._load_database_identity",
            return_value=("TEST_DEVICE", mapping),
        ),
    ):
        scenarios = build_workload_scenarios("org/model", _model_config(), tmp_path)

    assert {scenario.num_devices for scenario in scenarios} == {1, 4, 8, 12}
    assert {scenario.sweep_name for scenario in scenarios} >= {
        "baseline",
        "tp_axis",
        "ep_axis",
        "moe_dp_axis",
        "dcp_axis",
        "mtp_axis",
        "tp_ep_interaction",
        "topology_edge",
        "bf16_baseline",
        "int8_kv_cache",
    }
    # Regression guard: never restore the all-axes Cartesian explosion.
    optimizer_combinations = sum(
        len(scenario.tp_sizes)
        * len(scenario.ep_sizes)
        * len(scenario.moe_dp_sizes)
        * len(scenario.dcp_sizes)
        * max(1, len(scenario.mtp_tokens))
        for scenario in scenarios
    )
    assert optimizer_combinations < 300
    assert len({scenario.workload_id for scenario in scenarios}) == len(scenarios)
    assert {value for scenario in scenarios for value in scenario.tp_sizes} >= set(_model_config().tp_sizes)
    assert {value for scenario in scenarios for value in scenario.ep_sizes} >= set(_model_config().ep_sizes)
    assert {value for scenario in scenarios for value in scenario.mtp_tokens} == {0, 1, 2}
    baseline_lengths = {scenario.input_length for scenario in scenarios if scenario.sweep_name == "baseline"}
    assert {1, 4095, 4096, 4097, 9998, 9999} <= baseline_lengths
    interactions = [scenario for scenario in scenarios if scenario.sweep_name == "tp_ep_interaction"]
    assert all(len(scenario.tp_sizes) == len(scenario.ep_sizes) == 1 for scenario in interactions)
    assert {(*scenario.tp_sizes, *scenario.ep_sizes) for scenario in interactions} >= {
        (1, 4),
        (4, 1),
        (4, 4),
    }
    assert any(scenario.max_batched_tokens is not None for scenario in scenarios)
    assert all(scenario.parallel_combinations == 1 for scenario in scenarios)
    assert all(
        scenario.batch_range[0] == 1
        or scenario.batch_range[0] * scenario.input_length <= _model_config().max_context_length
        for scenario in scenarios
        if scenario.sweep_name != "baseline"
    )
    assert any(
        scenario.batch_range == (512, 512)
        for scenario in scenarios
        if scenario.sweep_name != "baseline" and scenario.input_length == 1
    )
    assert all(
        scenario.batch_range[0] == scenario.batch_range[1]
        for scenario in scenarios
        if scenario.sweep_name != "baseline"
    )
    assert all(scenario.batch_range == (1, 512) for scenario in scenarios if scenario.sweep_name == "baseline")
    assert any(scenario.compilation_config for scenario in scenarios)
    compiled_scenarios = [scenario for scenario in scenarios if scenario.compilation_config]
    compiled = compiled_scenarios[0]
    command = compiled.command(tmp_path)
    assert "--compile" in command
    assert command[command.index("--compilation-config") + 1 :] == [
        "enable_sequence_parallel",
        "enable_dispatch_ffn_combine",
    ]
    assert {value for scenario in compiled_scenarios for value in scenario.ep_sizes} == {1, 4}
    assert {value for scenario in compiled_scenarios for value in scenario.mtp_tokens} == {0, 2}
    long_compiled = [scenario for scenario in compiled_scenarios if scenario.input_length == 9999]
    assert all(scenario.ep_sizes == (1,) and scenario.mtp_tokens == (0,) for scenario in long_compiled)
    compiled_lengths = sorted({scenario.input_length for scenario in compiled_scenarios})
    interaction_length = compiled_lengths[len(compiled_lengths) // 2]
    interaction_compiled = [scenario for scenario in compiled_scenarios if scenario.input_length == interaction_length]
    assert {(*scenario.ep_sizes, *scenario.mtp_tokens) for scenario in interaction_compiled} == {
        (1, 0),
        (1, 2),
        (4, 0),
        (4, 2),
    }
    assert "--max-search-combinations" not in command
    assert any(scenario.quantize_linear_action == "DISABLED" for scenario in scenarios)
    assert any(scenario.quantize_attention_action == "INT8" for scenario in scenarios)


def test_workload_runner_passes_private_trace_environment(tmp_path: Path) -> None:
    scenario = _scenario()
    environments = []

    def command_runner(_command, **kwargs):
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(_command, 0, stdout="", stderr="")

    result = run_query_workloads(
        [scenario],
        database_path=tmp_path,
        trace_dir=tmp_path / "trace",
        repo_root=tmp_path,
        command_runner=command_runner,
    )

    assert result.attempted == result.succeeded == 1
    assert Path(environments[0][QUERY_TRACE_DIR_ENV]).is_relative_to(tmp_path / "trace")
    summary = json.loads((tmp_path / "trace" / "workload-summary.json").read_text(encoding="utf-8"))
    assert summary["attempted"] == summary["succeeded"] == 1
    assert summary["planned_parallel_combinations"] == 1


def test_workload_runner_reuses_successful_checkpoint(tmp_path: Path) -> None:
    calls = 0

    def command_runner(command, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    kwargs = {
        "database_path": tmp_path,
        "trace_dir": tmp_path / "trace",
        "repo_root": tmp_path,
        "command_runner": command_runner,
        "max_workers": 1,
    }
    first = run_query_workloads([_scenario()], **kwargs)
    second = run_query_workloads([_scenario()], **kwargs)

    assert calls == 1
    assert first.cached == 0
    assert second.cached == 1
    assert second.trace_directories == first.trace_directories


def test_workload_runner_invalidates_checkpoint_when_database_content_changes(tmp_path: Path) -> None:
    calls = 0
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text("device: TEST\n", encoding="utf-8")
    csv_path = database / "Add.csv"
    csv_path.write_text("Input Shapes\n1,4\n", encoding="utf-8")

    def command_runner(command, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    kwargs = {
        "database_path": database,
        "trace_dir": tmp_path / "trace",
        "repo_root": tmp_path,
        "command_runner": command_runner,
        "max_workers": 1,
    }
    first = run_query_workloads([_scenario()], **kwargs)
    cached = run_query_workloads([_scenario()], **kwargs)
    csv_path.write_text("Input Shapes\n2,4\n", encoding="utf-8")
    changed = run_query_workloads([_scenario()], **kwargs)

    assert calls == 2
    assert first.cached == changed.cached == 0
    assert cached.cached == 1


def test_workload_runner_retries_failed_workload(tmp_path: Path) -> None:
    calls = 0

    def command_runner(command, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1 if calls == 1 else 0, stdout="", stderr="failed")

    kwargs = {
        "database_path": tmp_path,
        "trace_dir": tmp_path / "trace",
        "repo_root": tmp_path,
        "command_runner": command_runner,
        "max_workers": 1,
    }
    first = run_query_workloads([_scenario()], **kwargs)
    second = run_query_workloads([_scenario()], **kwargs)

    assert calls == 2
    assert first.succeeded == first.cached == 0
    assert second.succeeded == 1
    assert second.cached == 0


def test_workload_runner_retries_corrupt_checkpoint(tmp_path: Path) -> None:
    calls = 0

    def command_runner(command, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    kwargs = {
        "database_path": tmp_path,
        "trace_dir": tmp_path / "trace",
        "repo_root": tmp_path,
        "command_runner": command_runner,
        "max_workers": 1,
    }
    run_query_workloads([_scenario()], **kwargs)
    checkpoint_path = next((tmp_path / "trace" / "workloads").glob("*/checkpoint.json"))
    checkpoint_path.write_text("{not-json", encoding="utf-8")
    result = run_query_workloads([_scenario()], **kwargs)

    assert calls == 2
    assert result.succeeded == 1
    assert result.cached == 0


def test_workload_runner_uses_bounded_parallelism(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def command_runner(command, **_kwargs):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_query_workloads(
        [_scenario(128), _scenario(256), _scenario(512)],
        database_path=tmp_path,
        trace_dir=tmp_path / "trace",
        repo_root=tmp_path,
        command_runner=command_runner,
        max_workers=2,
    )

    assert result.succeeded == 3
    assert peak_active == 2


def test_workload_runner_stops_idle_process_after_query_trace_converges(tmp_path: Path) -> None:
    script = tmp_path / "emit_query_then_wait.py"
    natural_exit_sentinel = tmp_path / "natural-exit-sentinel"
    script.write_text(
        """
import json
import os
from pathlib import Path
import time

trace_dir = Path(os.environ["MSMODELING_SHAPE_QUERY_TRACE_DIR"])
trace_path = trace_dir / f"query-demands-{os.getpid()}.jsonl"
trace_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "projector_version": "test/v1",
            "op_name": "Add",
            "kernel_type": "Add",
            "query_mode": "exact",
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
        },
        sort_keys=True,
    )
    + "\\n",
    encoding="utf-8",
)
os.write(1, b"\\xa9")
time.sleep(30)
Path(os.environ["MSMODELING_TEST_NATURAL_EXIT_SENTINEL"]).write_text("natural exit", encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )

    with (
        mock.patch.dict(
            os.environ,
            {"MSMODELING_TEST_NATURAL_EXIT_SENTINEL": str(natural_exit_sentinel)},
        ),
        mock.patch.object(WorkloadScenario, "command", return_value=[sys.executable, str(script)]),
        mock.patch(
            "tools.perf_data_collection.grid_generator.query_workloads.QUERY_TRACE_POLL_SECONDS",
            0.05,
        ),
    ):
        result = run_query_workloads(
            [_scenario()],
            database_path=tmp_path,
            trace_dir=tmp_path / "trace",
            repo_root=Path(__file__).parents[3],
            trace_quiet_seconds=0.1,
        )

    assert result.succeeded == 1
    assert result.failed_workloads == ()
    checkpoint_path = next((tmp_path / "trace" / "workloads").glob("*/checkpoint.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["completion_reason"] == "query_trace_converged"
    assert not natural_exit_sentinel.exists()


def test_scenario_command_forwards_actual_optimizer_flags(tmp_path: Path) -> None:
    scenario = WorkloadScenario(
        model_id="org/model",
        device="TEST_DEVICE",
        num_devices=32,
        input_length=80000,
        output_length=1024,
        max_batched_tokens=320000,
        tp_sizes=(2,),
        ep_sizes=(4,),
        moe_dp_sizes=(1,),
        dcp_sizes=(1,),
        mtp_tokens=(0, 2),
        disagg=True,
        tpot_limit=70,
        reserved_memory_gb=10,
        enable_shared_expert_tp=True,
        word_embedding_tp="row",
        mtp_acceptance_rates=(0.9, 0.6),
        compile=True,
    )
    command = scenario.command(tmp_path)
    text = " ".join(command)

    assert "--disagg" in text
    assert "--tpot-limit 70" in text
    assert "--reserved-memory-gb 10" in text
    assert "--enable-shared-expert-tp" in text
    assert "--word-embedding-tp row" in text
    assert "--mtp-acceptance-rates 0.9 0.6" in text
    assert "--compile" in command
    assert "--ttft-limit" not in text
    assert "--prefix-cache-hit-rate" not in text

    # Default-constructed scenarios keep the historical internal command.
    baseline = _scenario().command(tmp_path)
    baseline_text = " ".join(baseline)
    assert "--disagg" not in baseline_text
    assert "--reserved-memory-gb" not in baseline_text
    assert "--mtp-acceptance-rates" not in baseline_text

    # The workload identity must change with every serving-relevant field.
    identities = {
        scenario.workload_id,
        WorkloadScenario(**{**scenario.__dict__, "disagg": False}).workload_id,
        WorkloadScenario(**{**scenario.__dict__, "reserved_memory_gb": 5.0}).workload_id,
        WorkloadScenario(**{**scenario.__dict__, "mtp_acceptance_rates": (0.5,)}).workload_id,
    }
    assert len(identities) == 4

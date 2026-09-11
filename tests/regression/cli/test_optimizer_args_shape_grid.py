"""End-to-end tests for optimizer-args-driven shape generation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from tensor_cast.device import DeviceProfile
from tensor_cast.performance_model.profiling_database.query_demand import (
    KernelQueryDemand,
    QUERY_TRACE_DIR_ENV,
    QUERY_TRACE_WORKLOAD_ENV,
    QueryDemandTraceWriter,
)
from tools.perf_data_collection.generate_shape_grid import build_argparser
from tools.perf_data_collection.grid_generator import runner


ADD_HEADERS = (
    "OP State,Input Shapes,Input Data Types,Input Formats,Output Shapes,"
    "Output Data Types,Output Formats,Average Duration(us)"
)
ADD_TEMPLATE_ROW = 'enabled,"1,4",DT_BF16,ND,"1,4",DT_BF16,ND,0'

REPO_ROOT = Path(__file__).resolve().parents[3]
OP_REPLAY_DIR = REPO_ROOT / "tools" / "perf_data_collection" / "op_replay"

SPEC_YAML = """
model: org/model
device: TEST_DEVICE
num_devices: 1
input_length: 128
output_length: 1
disagg: true
ttft_limit: 10000
max_batched_tokens: 4096
batch_range: [1, 8]
tp_sizes: [1]
"""


@pytest.fixture(autouse=True)
def _test_device_profile(monkeypatch):
    profile = SimpleNamespace(comm_grid=SimpleNamespace(grid=torch.empty(1)))
    monkeypatch.setitem(DeviceProfile.all_device_profiles, "TEST_DEVICE", profile)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text("device: TEST_DEVICE\noperator_mappings: {}\n", encoding="utf-8")
    (database / "Add.csv").write_text(f"{ADD_HEADERS}\n{ADD_TEMPLATE_ROW}\n", encoding="utf-8")
    return database


def _demand(shape: tuple[int, int], workload_id: str = "spec-workload") -> KernelQueryDemand:
    return KernelQueryDemand(
        projector_version="test/v1",
        op_name="tensor_cast.add.default",
        kernel_type="Add",
        query_mode="elementwise",
        input_shapes=(shape, shape),
        output_shapes=(shape,),
        input_dtypes=("BF16", "BF16"),
        output_dtypes=("BF16",),
        tensor_parallel_size=1,
        expert_parallel_size=1,
        model_id="org/model",
        workload_id=workload_id,
    )


def _run_query_mode(
    tmp_path: Path,
    database: Path,
    demands: list[KernelQueryDemand],
    rows: int = 1,
    report_path: Path | None = None,
):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(SPEC_YAML, encoding="utf-8")
    commands: list[list[str]] = []

    def command_runner(command, **kwargs):
        commands.append(command)
        trace_dir = Path(kwargs["env"][QUERY_TRACE_DIR_ENV])
        writer = QueryDemandTraceWriter(trace_dir)
        for demand in demands:
            writer.record(replace(demand, workload_id=kwargs["env"][QUERY_TRACE_WORKLOAD_ENV]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    original = runner._query_cache_directory
    runner._query_cache_directory = lambda data_dir, model_ids, repo_root, extra_digest=None: tmp_path / "trace"
    try:
        argv = [
            "--database-path",
            str(database),
            "--rows",
            str(rows),
            "--optimizer-args-file",
            str(spec_path),
        ]
        if report_path is not None:
            argv.extend(["--report-path", str(report_path)])
        args = build_argparser().parse_args(argv)
        result = runner.run_query_mode(
            args,
            data_dir=database,
            op_replay_dir=OP_REPLAY_DIR,
            repo_root=tmp_path,
            command_runner=command_runner,
        )
    finally:
        runner._query_cache_directory = original
    return result, commands


def test_optimizer_args_scenario_captures_demand_and_writes_replay_rows(tmp_path: Path, database: Path) -> None:
    # op_replay discovery happens against the real replay directory.
    result, commands = _run_query_mode(
        tmp_path,
        database,
        [
            _demand((2, 4)),
            _demand((8, 4)),
            _demand((16, 4)),
        ],
        rows=1,
    )

    # One subprocess per expanded parallel combination; the actual CLI flags
    # from the spec must be forwarded verbatim.
    assert len(commands) == 1
    text = " ".join(commands[0])
    assert "--disagg" in text
    assert "--ttft-limit 10000" in text
    assert "--max-batched-tokens 4096" in text
    assert "--batch-range 1 8" in text

    # Exact demand rows bypass the --rows budget in optimizer-args mode; the
    # budget still bounds exactly one coverage/fallback row.
    csv_lines = (database / "Add.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 1 + 1 + 3 + 1  # header + template + 3 exact + 1 coverage
    assert '"2,4;2,4"' in (database / "Add.csv").read_text(encoding="utf-8")

    report = result.report
    assert report["mode"] == "optimizer-args"
    assert report["rows_budget_semantics"].startswith("optimizer-args exact demand rows are never budget-limited")
    assert report["optimizer_args"]["digest"]
    assert report["optimizer_args"]["workloads"][0]["name"] == "actual_1"
    assert report["captured_demands"] == 3
    assert report["unsupported_kernel_demands"] == {}
    statuses = {entry["status"] for entry in report["demand_ledger"]}
    assert statuses == {"generated"}
    operator = report["operators"][0]
    assert operator["exact_generated"] == 3
    assert operator["coverage_appended"] == 1
    assert operator["appended_rows"] == 4
    assert operator["csv_rows_before"] == 1
    assert operator["csv_rows_after"] == 5
    assert operator["pending_microbench_rows"] == 4

    report_file = result.report_path
    assert report_file is not None and report_file.is_file()
    persisted = json.loads(report_file.read_text(encoding="utf-8"))
    assert persisted["mode"] == "optimizer-args"


def test_report_path_overrides_automatic_cache_location(tmp_path: Path, database: Path) -> None:
    custom_report = tmp_path / "reports" / "custom-report.json"

    result, _ = _run_query_mode(
        tmp_path,
        database,
        [_demand((2, 4))],
        report_path=custom_report,
    )

    assert result.report_path == custom_report
    assert custom_report.is_file()
    assert not (tmp_path / "trace" / runner.QUERY_REPORT_FILE_NAME).exists()
    assert json.loads(custom_report.read_text(encoding="utf-8")) == result.report


def test_preflight_rejections_update_counts_and_demand_ledger(
    tmp_path: Path,
    database: Path,
    monkeypatch,
) -> None:
    from tools.perf_data_collection.grid_generator.preflight import RowPreflightResult

    def reject_first_exact_and_coverage(kernel_type, rows, op_replay_dir):
        assert kernel_type == "Add"
        return [
            RowPreflightResult(index, index not in {0, 2}, "rejected by test" if index in {0, 2} else "")
            for index in range(len(rows))
        ]

    monkeypatch.setattr(runner, "preflight_generated_rows", reject_first_exact_and_coverage)
    result, _ = _run_query_mode(
        tmp_path,
        database,
        [_demand((2, 4)), _demand((8, 4))],
        rows=1,
    )

    operator = result.report["operators"][0]
    assert operator["exact_generated"] == 1
    assert operator["coverage_appended"] == 0
    assert operator["preflight_rejected"] == 2
    assert operator["appended_rows"] == 1
    statuses = [entry["status"] for entry in result.report["demand_ledger"]]
    assert statuses.count("preflight_rejected") == 1
    assert statuses.count("generated") == 1


def test_spec_device_mismatch_fails_closed(tmp_path: Path, database: Path, monkeypatch) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(SPEC_YAML.replace("TEST_DEVICE", "ATLAS_800_A3_752T_128G_DIE"), encoding="utf-8")
    args = build_argparser().parse_args(
        [
            "--database-path",
            str(database),
            "--rows",
            "1",
            "--optimizer-args-file",
            str(spec_path),
        ]
    )

    # The device must be a registered profile first; register it so the runner
    # mismatch check (not spec validation) is what fails.
    profile = SimpleNamespace(comm_grid=SimpleNamespace(grid=torch.empty(1)))
    monkeypatch.setitem(DeviceProfile.all_device_profiles, "ATLAS_800_A3_752T_128G_DIE", profile)
    with pytest.raises(ValueError, match="do not match the database device"):
        runner.run_query_mode(
            args,
            data_dir=database,
            op_replay_dir=OP_REPLAY_DIR,
            repo_root=tmp_path,
        )


def test_missing_workload_source_fails(tmp_path: Path, database: Path) -> None:
    args = build_argparser().parse_args(["--database-path", str(database)])
    with pytest.raises(ValueError, match="target-models or --optimizer-args-file"):
        runner.run_query_mode(
            args,
            data_dir=database,
            op_replay_dir=OP_REPLAY_DIR,
            repo_root=tmp_path,
        )


def test_target_models_only_mode_still_accepted(tmp_path: Path, database: Path, monkeypatch) -> None:
    # Compatibility: the pre-feature invocation keeps working with the new CLI.
    args = argparse.Namespace(
        database_path=database,
        rows=10,
        target_models=["org/model"],
        ops=None,
        seed=0,
        optimizer_args_file=None,
        report_path=None,
    )

    def command_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "tools.perf_data_collection.grid_generator.runner.resolve_query_model_architecture",
        lambda model_id: SimpleNamespace(
            max_context_length=1024,
            num_experts=0,
            num_mtp_layers=0,
            tp_sizes=(1, 2),
            ep_sizes=(1,),
        ),
    )
    monkeypatch.setattr(
        "tools.perf_data_collection.grid_generator.runner._query_cache_directory",
        lambda data_dir, model_ids, repo_root, extra_digest=None: tmp_path / "trace",
    )
    with pytest.raises(RuntimeError, match="captured no profiling-database query"):
        # No demands are captured by the stub runner, which must fail loudly.
        runner.run_query_mode(
            args,
            data_dir=database,
            op_replay_dir=OP_REPLAY_DIR,
            repo_root=tmp_path,
            command_runner=command_runner,
        )

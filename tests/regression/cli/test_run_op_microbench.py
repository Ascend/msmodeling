# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.perf_data_collection import run_op_microbench
from tools.perf_data_collection.op_microbench import adapters
from tools.perf_data_collection.op_microbench import worker as op_worker


HEADERS = [
    "OP State",
    "Accelerator Core",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Average Duration(us)",
    "Profiling Average Duration(us)",
    "Profiling Median Duration(us)",
    "Profiling Std Duration(us)",
    "Profiling Average aicore_time(us)",
    "Profiling Average aic_total_cycles",
    "Profiling Average cube_utilization(%)",
]


def _request(*, include_raw_records: bool = True) -> dict:
    return {
        "schema_version": 1,
        "kernel_type": "MatMulV2",
        "replay_row": {
            "Input Shapes": "1,4;2,4",
            "Input Data Types": "DT_BF16;DT_BF16",
            "Input Formats": "ND;ND",
            "Output Shapes": "1,2",
            "Output Data Types": "DT_BF16",
            "Output Formats": "ND",
        },
        "profiling": {"include_raw_records": include_raw_records},
    }


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "OP State": "static",
        "Accelerator Core": "AI_CORE",
        # Existing database files retain one extra quote pair inside CSV cells.
        "Input Shapes": '"1,4;2,4"',
        "Input Data Types": "DT_BF16;DT_BF16",
        "Input Formats": "ND;ND",
        "Output Shapes": '"1,2"',
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        # This legacy value must never become the msprof-only primary duration.
        "Average Duration(us)": "99.000000",
        "Profiling Average Duration(us)": "10.200000",
        "Profiling Median Duration(us)": "10.000000",
        "Profiling Std Duration(us)": "0.300000",
        "Profiling Average aicore_time(us)": "8.500000",
        "Profiling Average aic_total_cycles": "12345.000000",
        "Profiling Average cube_utilization(%)": "55.500000",
    }
    row.update(overrides)
    return row


def _write_database(path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows or [_row()])
    return path


def _write_request(path: Path, request: dict | None = None) -> Path:
    path.write_text(json.dumps(request or _request()), encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_worker_metadata(request: dict) -> dict:
    return {
        "status": "succeeded",
        "operator": {
            "requested_kernel_type": request["kernel_type"],
            "resolved_adapter": "MatMulV2",
            "api_path": "torch.mm",
            "target_op_types": ["MatMulV2", "MatMulV3", "MatMulCommon"],
            "is_composite": False,
        },
        "environment": {
            "device_id": 0,
            "device_name": "ATLAS_800_A3_752T_128G_DIE",
            "cann_version": "8.5",
            "torch_version": "2.9.0",
            "torch_npu_version": "2.9.0.post1",
            "vllm_ascend_version": "0.18.0",
            "msprof_version": None,
            "simulated": False,
        },
        "inputs": [
            {
                "name": tensor["name"],
                "path": None,
                "origin": "replay_metadata",
                "logical_shape": tensor["shape"],
                "device_shape": tensor["shape"],
                "dtype": tensor["dtype"],
                "format": tensor["format"],
                "size_bytes": 8,
            }
            for tensor in request["inputs"]
        ],
        "outputs": [
            {
                "name": None,
                "path": "$",
                "origin": "operator",
                "logical_shape": [1, 2],
                "device_shape": [1, 2],
                "dtype": "DT_BF16",
                "format": "ND",
                "size_bytes": 4,
            }
        ],
        "execution": {
            "warmup_count": request["profiling"]["warmup_count"],
            "repeat_count": request["profiling"]["repeat_count"],
            "total_invocations": request["profiling"]["warmup_count"] + request["profiling"]["repeat_count"],
        },
        "error": None,
    }


def _install_fake_msprof(monkeypatch: pytest.MonkeyPatch, *, extra_records: int = 0) -> None:
    monkeypatch.setattr(run_op_microbench.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(run_op_microbench.shutil, "which", lambda name: "/usr/local/bin/msprof")

    class FakePopen:
        """Mimic subprocess.Popen for the msprof backend path.

        The real backend now uses Popen(start_new_session=True) plus
        communicate(timeout=...) so it can kill the whole process group on
        timeout. The fake honors that contract without spawning a process.
        """

        def __init__(self, cmd, **kwargs):
            del kwargs  # accept cwd/stdout/stderr/text/encoding/errors/start_new_session
            request_path = Path(cmd[cmd.index("--request") + 1])
            metadata_path = Path(cmd[cmd.index("--metadata-out") + 1])
            profile_root = Path(next(item.removeprefix("--output=") for item in cmd if item.startswith("--output=")))
            request = json.loads(request_path.read_text(encoding="utf-8"))
            metadata_path.write_text(json.dumps(_fake_worker_metadata(request)), encoding="utf-8")
            output_dir = profile_root / "PROF_000001" / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)
            summary_path = output_dir / "op_summary_001.csv"
            record_count = request["profiling"]["warmup_count"] + request["profiling"]["repeat_count"] + extra_records
            with summary_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "Op Name",
                        "OP Type",
                        "Task Type",
                        "Task Start Time(us)",
                        "Task Duration(us)",
                        "aicore_time(us)",
                        "aic_total_cycles",
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                for index in range(record_count):
                    writer.writerow(
                        {
                            "Op Name": "aclnnMm_MatMulCommon_MatMulV2",
                            "OP Type": "MatMulV2",
                            "Task Type": "AI_CORE",
                            "Task Start Time(us)": str(1000 + index),
                            "Task Duration(us)": str(
                                100.0 if index < request["profiling"]["warmup_count"] else 10 + index
                            ),
                            "aicore_time(us)": str(8 + index),
                            "aic_total_cycles": str(1000 + index),
                        }
                    )
            self.pid = 99999
            self.returncode = 0
            self._stdout = "profiling complete"
            self._stderr = ""

        def communicate(self, timeout=None):
            del timeout  # never timing out in the fake path
            return self._stdout, self._stderr

    monkeypatch.setattr(run_op_microbench.subprocess, "Popen", FakePopen)


class _FakeDType:
    def __init__(self, name: str, element_size: int, floating: bool) -> None:
        self.name = name
        self.element_size = element_size
        self.is_floating_point = floating

    def __str__(self) -> str:
        return self.name


class _FakeTensor:
    def __init__(self, shape, dtype, format_id: int = 2) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.format_id = format_id

    def npu(self):
        return self

    def to(self, dtype):
        return _FakeTensor(self.shape, dtype, self.format_id)

    def uniform_(self, low, high):
        del low, high
        return self

    def transpose(self, dim0: int, dim1: int):
        dims = list(self.shape)
        dim0 %= len(dims)
        dim1 %= len(dims)
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return _FakeTensor(dims, self.dtype, self.format_id)

    def contiguous(self):
        return _FakeTensor(self.shape, self.dtype, self.format_id)

    def numel(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return self.dtype.element_size


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch):
    dtypes = {
        "bool": _FakeDType("torch.bool", 1, False),
        "int8": _FakeDType("torch.int8", 1, False),
        "uint8": _FakeDType("torch.uint8", 1, False),
        "int16": _FakeDType("torch.int16", 2, False),
        "int32": _FakeDType("torch.int32", 4, False),
        "int64": _FakeDType("torch.int64", 8, False),
        "float16": _FakeDType("torch.float16", 2, True),
        "bfloat16": _FakeDType("torch.bfloat16", 2, True),
        "float32": _FakeDType("torch.float32", 4, True),
        "float64": _FakeDType("torch.float64", 8, True),
    }
    npu = SimpleNamespace(
        current_device=None,
        sync_count=0,
        set_device=lambda device_id: setattr(npu, "current_device", device_id),
        synchronize=lambda: setattr(npu, "sync_count", npu.sync_count + 1),
        get_device_name=lambda device_id: f"FakeNPU-{device_id}",
    )

    def tensor(shape, dtype):
        return _FakeTensor(shape, dtype)

    def mm(left, right):
        return tensor((left.shape[0], right.shape[1]), left.dtype)

    def bmm(left, right):
        return tensor((left.shape[0], left.shape[1], right.shape[2]), left.dtype)

    fake_torch = SimpleNamespace(
        __version__="2.9.0+fake",
        Tensor=_FakeTensor,
        npu=npu,
        manual_seed=lambda seed: None,
        normal=lambda mean, std, size: tensor(size, dtypes["float32"]),
        empty=lambda shape, dtype: tensor(shape, dtype),
        randint=lambda low, high, shape, dtype: tensor(shape, dtype),
        zeros=lambda shape, dtype: tensor(shape, dtype),
        ones=lambda shape, dtype: tensor(shape, dtype),
        full=lambda shape, value, dtype: tensor(shape, dtype),
        mm=mm,
        bmm=bmm,
        add=lambda left, other, alpha: tensor(left.shape, left.dtype),
        transpose=lambda value, dim0, dim1: value.transpose(dim0, dim1),
        **dtypes,
    )

    class FakeFormat:
        ND = 2
        FRACTAL_NZ = 29

    def fake_swiglu(value, dim):
        dim %= len(value.shape)
        output_shape = list(value.shape)
        output_shape[dim] //= 2
        return tensor(output_shape, value.dtype)

    fake_torch_npu = SimpleNamespace(
        __version__="2.9.0.post1+fake",
        npu=SimpleNamespace(config=SimpleNamespace(allow_internal_format=False)),
        Format=FakeFormat,
        version=SimpleNamespace(cann=None, cann_version=None),
        get_npu_format=lambda value: value.format_id,
        npu_format_cast=lambda value, format_id: _FakeTensor(value.shape, value.dtype, format_id),
        npu_rms_norm=lambda value, gamma, epsilon=None: (
            tensor(value.shape, value.dtype),
            tensor((*value.shape[:-1], 1), dtypes["float32"]),
        ),
        npu_swiglu=fake_swiglu,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    return fake_torch, fake_torch_npu


def test_database_simulation_returns_labelled_msprof_json(tmp_path: Path) -> None:
    source = _write_database(
        tmp_path
        / "database"
        / "ATLAS_800_A3_752T_128G_DIE"
        / "vllm_ascend"
        / "vllm0.18.0_torch2.9.0_cann8.5"
        / "MatMulV2.csv"
    )
    normalized = run_op_microbench.normalize_request(_request())

    result = run_op_microbench.run_database_simulation(normalized, source, "request-id")

    assert result["status"] == "succeeded"
    assert result["simulated"] is True
    assert result["result_source"] == "profiling_database_simulation"
    assert result["duration"]["value_us"] == 10.0
    assert result["duration"]["statistics"]["mean"] == 10.2
    assert result["duration"]["sample_count"] is None
    assert result["msprof"]["executed"] is False
    assert result["msprof"]["aggregated_metrics"]["aic_total_cycles"]["mean"] == 12345.0
    assert result["outputs"][0]["logical_shape"] == [1, 2]
    assert result["environment"]["device_name"] == "ATLAS_800_A3_752T_128G_DIE"
    assert result["environment"]["vllm_ascend_version"] == "0.18.0"
    assert result["artifacts"]["simulation_source"]["sha256"] == _digest(source)
    assert any("no NPU operator" in warning for warning in result["warnings"])


def test_cli_defaults_to_json_even_when_output_suffix_is_csv(tmp_path: Path) -> None:
    source = _write_database(tmp_path / "database" / "MatMulV2.csv")
    request_path = _write_request(tmp_path / "request.json")
    output = tmp_path / "result.csv"

    return_code = run_op_microbench.main(
        [
            "--request",
            str(request_path),
            "--simulate-from-database",
            str(source),
            "--output",
            str(output),
        ]
    )

    assert return_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert result["duration"]["column"] == "Profiling Median Duration(us)"


def test_cli_writes_flat_csv_from_profiling_columns(tmp_path: Path) -> None:
    source = _write_database(tmp_path / "database" / "MatMulV2.csv")
    request_path = _write_request(tmp_path / "request.json")
    output = tmp_path / "output" / "result.csv"

    return_code = run_op_microbench.main(
        [
            "--request",
            str(request_path),
            "--simulate-from-database",
            str(source),
            "--output-format",
            "csv",
            "--output",
            str(output),
        ]
    )

    assert return_code == 0
    with output.open("r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["Average Duration(us)"] == "10.200000"
    assert row["Median Duration(us)"] == "10.000000"
    assert row["Average aicore_time(us)"] == "8.500000"
    assert row["Result Source"] == "profiling_database_simulation"
    assert row["Simulated"] == "true"
    assert row["Average Duration(us)"] != "99.000000"


def test_csv_preserves_replay_slots_runtime_identity_and_separates_actual_outputs(tmp_path: Path) -> None:
    source = _write_database(tmp_path / "database" / "MatMulV2.csv")
    result = run_op_microbench.run_database_simulation(
        run_op_microbench.normalize_request(_request()), source, "request-id"
    )
    replay_row = result["configuration"]["replay_row"]
    replay_row.update(
        {
            "Input Shapes": "4,8;2,8;;2,4;;;;2;4",
            "Input Data Types": "DT_INT8;DT_INT8;;DT_BF16;;;;DT_INT64;DT_FLOAT",
            "Input Formats": "ND;FRACTAL_NZ;;ND;;;;ND;ND",
            "Runtime case_id": "grouped-case-1",
            "EP Size": "8",
        }
    )

    columns, row = run_op_microbench._result_to_csv(result)

    assert row["Input Shapes"] == "4,8;2,8;;2,4;;;;2;4"
    assert row["Input Data Types"] == "DT_INT8;DT_INT8;;DT_BF16;;;;DT_INT64;DT_FLOAT"
    assert row["Runtime case_id"] == "grouped-case-1"
    assert row["EP Size"] == "8"
    assert row["Actual Output Shapes"] == "1,2"
    assert columns.index("Runtime case_id") < columns.index("Actual Output Shapes")


def test_simulation_does_not_modify_source_database(tmp_path: Path) -> None:
    source = _write_database(tmp_path / "database" / "MatMulV2.csv")
    before_hash = _digest(source)
    before_mtime = source.stat().st_mtime_ns

    run_op_microbench.run_database_simulation(run_op_microbench.normalize_request(_request()), source, "request-id")

    assert _digest(source) == before_hash
    assert source.stat().st_mtime_ns == before_mtime


def test_output_inside_database_is_rejected_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    database_dir = tmp_path / "database"
    source = _write_database(database_dir / "MatMulV2.csv")
    request_path = _write_request(tmp_path / "request.json")
    output = database_dir / "result.json"

    return_code = run_op_microbench.main(
        [
            "--request",
            str(request_path),
            "--simulate-from-database",
            str(source),
            "--output",
            str(output),
        ]
    )

    assert return_code == 1
    assert not output.exists()
    failure = json.loads(capsys.readouterr().out)
    assert failure["error"]["code"] == "OUTPUT_INSIDE_DATABASE"


@pytest.mark.parametrize("request_kind", ["missing", "invalid_json"])
def test_request_failure_cannot_overwrite_simulation_source(
    request_kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_database(tmp_path / "database" / "MatMulV2.csv")
    request_path = tmp_path / "request.json"
    if request_kind == "invalid_json":
        request_path.write_text("{invalid-json", encoding="utf-8")
    before_hash = _digest(source)
    before_mtime = source.stat().st_mtime_ns

    return_code = run_op_microbench.main(
        [
            "--request",
            str(request_path),
            "--simulate-from-database",
            str(source),
            "--output",
            str(source),
        ]
    )

    assert return_code == 1
    assert _digest(source) == before_hash
    assert source.stat().st_mtime_ns == before_mtime
    failure = json.loads(capsys.readouterr().out)
    assert failure["error"]["code"] == "OUTPUT_INSIDE_DATABASE"


def test_duplicate_database_match_fails_closed(tmp_path: Path) -> None:
    source = _write_database(tmp_path / "database" / "MatMulV2.csv", [_row(), _row()])

    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench.run_database_simulation(run_op_microbench.normalize_request(_request()), source, "request-id")

    assert error.value.code == "AMBIGUOUS_DATABASE_RECORD"
    assert error.value.details["matching_rows"] == [2, 3]


def test_zero_profiling_duration_does_not_fall_back_to_legacy_duration(
    tmp_path: Path,
) -> None:
    source = _write_database(
        tmp_path / "database" / "MatMulV2.csv",
        [_row(**{"Profiling Median Duration(us)": "0"})],
    )

    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench.run_database_simulation(run_op_microbench.normalize_request(_request()), source, "request-id")

    assert error.value.code == "NO_VALID_PROFILING_DURATION"


def test_simulation_matches_the_complete_replay_descriptor(tmp_path: Path) -> None:
    source = _write_database(tmp_path / "database" / "MatMulV2.csv")
    request = _request()
    request["replay_row"]["Output Shapes"] = "1,3"

    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench.run_database_simulation(run_op_microbench.normalize_request(request), source, "request-id")

    assert error.value.code == "DATABASE_RECORD_NOT_FOUND"


def test_simulation_uses_runtime_columns_to_select_the_csv_row(tmp_path: Path) -> None:
    source = tmp_path / "database" / "MatMulV2.csv"
    source.parent.mkdir(parents=True)
    runtime_column = "Runtime case_id"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*HEADERS, runtime_column], lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [
                {**_row(), runtime_column: "case-a"},
                {**_row(), runtime_column: "case-b"},
            ]
        )
    request = _request()
    request["replay_row"][runtime_column] = "case-b"

    result = run_op_microbench.run_database_simulation(
        run_op_microbench.normalize_request(request), source, "request-id"
    )

    assert result["status"] == "succeeded"
    assert result["artifacts"]["simulation_source"]["row_number"] == 3


def test_no_simulation_backend_returns_structured_npu_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request_path = _write_request(tmp_path / "request.json")
    output = tmp_path / "result.json"
    monkeypatch.setattr(run_op_microbench.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(run_op_microbench.shutil, "which", lambda name: None)

    return_code = run_op_microbench.main(["--request", str(request_path), "--output", str(output)])

    assert return_code == 1
    failure = json.loads(output.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["error"]["code"] == "NPU_NOT_AVAILABLE"


@pytest.mark.parametrize(
    ("contents", "error_code"),
    [
        ("{not-json", "INVALID_REQUEST_JSON"),
        (json.dumps([]), "INVALID_REQUEST"),
    ],
)
def test_invalid_request_files_return_structured_json(tmp_path: Path, contents: str, error_code: str) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(contents, encoding="utf-8")
    output = tmp_path / "result.json"

    return_code = run_op_microbench.main(["--request", str(request_path), "--output", str(output)])

    assert return_code == 1
    failure = json.loads(output.read_text(encoding="utf-8"))
    assert failure["error"]["code"] == error_code
    assert failure["validation"]["request_valid"] is False


def test_missing_request_file_returns_structured_json(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    return_code = run_op_microbench.main(["--request", str(tmp_path / "missing.json"), "--output", str(output)])

    assert return_code == 1
    failure = json.loads(output.read_text(encoding="utf-8"))
    assert failure["error"]["code"] == "REQUEST_NOT_FOUND"


def test_request_validation_accepts_only_the_csv_row_contract() -> None:
    unknown = _request()
    unknown["unexpected"] = True
    with pytest.raises(run_op_microbench.MicrobenchError) as unknown_error:
        run_op_microbench.normalize_request(unknown)
    assert unknown_error.value.code == "INVALID_REQUEST"
    assert unknown_error.value.details["unknown_fields"] == ["unexpected"]

    for legacy_field in ("inputs", "attrs", "expected_outputs"):
        legacy = _request()
        legacy[legacy_field] = [] if legacy_field != "attrs" else {}
        with pytest.raises(run_op_microbench.MicrobenchError) as legacy_error:
            run_op_microbench.normalize_request(legacy)
        assert legacy_error.value.details["unknown_fields"] == [legacy_field]

    missing_output = _request()
    del missing_output["replay_row"]["Output Shapes"]
    with pytest.raises(run_op_microbench.MicrobenchError) as missing_error:
        run_op_microbench.normalize_request(missing_output)
    assert "Output Shapes" in missing_error.value.details["missing_columns"]


def test_database_source_must_match_operator_and_be_unambiguous(tmp_path: Path) -> None:
    wrong_file = _write_database(tmp_path / "wrong" / "OtherOperator.csv")
    with pytest.raises(run_op_microbench.MicrobenchError) as mismatch_error:
        run_op_microbench.resolve_database_csv(wrong_file, "MatMulV2")
    assert mismatch_error.value.code == "OPERATOR_DATABASE_MISMATCH"

    root = tmp_path / "ambiguous"
    _write_database(root / "version_a" / "MatMulV2.csv")
    _write_database(root / "version_b" / "MatMulV2.csv")
    with pytest.raises(run_op_microbench.MicrobenchError) as ambiguous_error:
        run_op_microbench.resolve_database_csv(root, "MatMulV2")
    assert ambiguous_error.value.code == "AMBIGUOUS_OPERATOR_DATABASE"


def test_csv_failure_writes_structured_error_only_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _write_database(tmp_path / "database" / "MatMulV2.csv")
    request = _request()
    request["replay_row"]["Input Shapes"] = "9,4;2,4"
    request_path = _write_request(tmp_path / "request.json", request)
    output = tmp_path / "result.csv"

    return_code = run_op_microbench.main(
        [
            "--request",
            str(request_path),
            "--simulate-from-database",
            str(source),
            "--output-format",
            "csv",
            "--output",
            str(output),
        ]
    )

    assert return_code == 1
    assert not output.exists()
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "DATABASE_RECORD_NOT_FOUND"


def test_available_npu_tools_run_real_msprof_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_msprof(monkeypatch)

    result = run_op_microbench.run_msprof_backend(
        run_op_microbench.normalize_request(_request()),
        "request-id",
        None,
    )

    assert result["status"] == "succeeded"
    assert result["simulated"] is False
    assert result["result_source"] == "msprof"
    assert result["duration"]["sample_count"] == 30
    assert result["duration"]["statistics"]["max"] < 100.0
    assert result["msprof"]["target_match"]["warmup_records"] == 5
    assert result["msprof"]["target_match"]["measurement_records"] == 30
    assert result["msprof"]["physical_kernels"][0]["op_type"] == "MatMulV2"
    assert result["artifacts"]["kept"] is False


def test_real_msprof_backend_fails_closed_on_extra_target_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_msprof(monkeypatch, extra_records=1)
    request = _request()
    request["profiling"]["keep_artifacts"] = "on_failure"

    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench.run_msprof_backend(
            run_op_microbench.normalize_request(request),
            "request-id",
            tmp_path,
        )

    assert error.value.code == "AMBIGUOUS_MSPROF_TARGET_RECORDS"
    assert error.value.artifacts["kept"] is True
    assert Path(error.value.artifacts["directory"]).is_dir()


def test_unregistered_operator_is_rejected_during_request_validation() -> None:
    request = _request()
    request["kernel_type"] = "ArbitraryPythonKernel"

    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench.normalize_request(request)

    assert error.value.code == "INVALID_REQUEST"
    assert "supported_operators" in error.value.details


def test_include_raw_records_false_still_allows_csv_output(tmp_path: Path) -> None:
    source = _write_database(tmp_path / "database" / "MatMulV2.csv")
    request_path = _write_request(tmp_path / "request.json", _request(include_raw_records=False))
    output = tmp_path / "output" / "result.csv"

    return_code = run_op_microbench.main(
        [
            "--request",
            str(request_path),
            "--simulate-from-database",
            str(source),
            "--output-format",
            "csv",
            "--output",
            str(output),
        ]
    )

    assert return_code == 0
    with output.open("r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["Accelerator Core"] == "AI_CORE"


def test_registered_adapters_reference_existing_op_replay_scripts() -> None:
    replay_dir = Path("tools/perf_data_collection/op_replay")
    script_names = {path.stem.removesuffix("_run") for path in replay_dir.glob("*_run.py")}

    assert set(adapters.ADAPTERS) == script_names - {"DispatchFFNCombine"}
    assert "DispatchFFNCombine" not in adapters.ADAPTERS
    assert all(spec.replay_module for spec in adapters.ADAPTERS.values())
    for spec in adapters.ADAPTERS.values():
        module_filename = f"{spec.replay_module.rsplit('.', maxsplit=1)[-1]}.py"
        assert (replay_dir / module_filename).is_file()


def test_readme_has_one_valid_example_for_every_supported_operator() -> None:
    readme = Path("tools/perf_data_collection/README_op_microbench.md").read_text(encoding="utf-8")
    examples_section = readme.split("## 逐算子使用示例", maxsplit=1)[1].split("## JSON 结果示例", maxsplit=1)[0]
    headings = re.findall(r"^### `([^`]+)`$", examples_section, flags=re.MULTILINE)
    request_blocks = re.findall(r"```json\n(.*?)\n```", examples_section, flags=re.DOTALL)
    requests = [json.loads(block) for block in request_blocks]
    documented_operators = [request["kernel_type"] for request in requests]

    assert len(headings) == len(set(headings))
    assert len(documented_operators) == len(set(documented_operators))
    assert set(headings) == set(adapters.ADAPTERS)
    assert set(documented_operators) == set(adapters.ADAPTERS)
    for request in requests:
        run_op_microbench.normalize_request(request)


def test_cli_lists_supported_and_excluded_operators(
    capsys: pytest.CaptureFixture[str],
) -> None:
    return_code = run_op_microbench.main(["--list-operators"])

    assert return_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["supported_operators"]) == 45
    assert payload["request_mode"] == "replay_row"
    assert payload["excluded_multi_card_operators"] == ["DispatchFFNCombine"]


def test_direct_script_entrypoint_lists_supported_operators() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "tools/perf_data_collection/run_op_microbench.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--list-operators"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert len(payload["supported_operators"]) == 45


def test_cli_describes_uniform_csv_row_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    return_code = run_op_microbench.main(["--describe-operator", "BatchMatMulV2"])

    assert return_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["request_mode"] == "replay_row"
    assert set(payload["required_columns"]) == run_op_microbench.REPLAY_REQUIRED_COLUMNS
    assert "CSV 行一致" in payload["description_zh"]


def test_standalone_profiler_targets_include_master_alias_metadata() -> None:
    expected_aliases = {
        "BatchMatMulV2": "BatchMatMulNd",
        "ScatterNdUpdate": "ScatterNdUpdateAiCore",
        "Slice": "SliceAiCore",
        "Transpose": "TransposeAiCore",
    }

    for kernel_type, alias in expected_aliases.items():
        assert alias in adapters.ADAPTERS[kernel_type].target_op_types


def test_replay_row_preserves_optional_slots_and_derives_input_metadata() -> None:
    replay_row = {
        "Input Shapes": "4,8;2,8;;2,4;;;;2;4",
        "Input Data Types": "DT_INT8;DT_INT8;;DT_BF16;;;;DT_INT64;DT_FLOAT",
        "Input Formats": "ND;FRACTAL_NZ;;ND;;;;ND;ND",
        "Output Shapes": "4,4",
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
    }

    request = run_op_microbench.normalize_request(
        {"schema_version": 1, "kernel_type": "GroupedMatmul", "replay_row": replay_row}
    )

    assert request["replay_row"] == replay_row
    assert [tensor["name"] for tensor in request["inputs"]] == [
        "replay_slot_0",
        "replay_slot_1",
        "replay_slot_3",
        "replay_slot_7",
        "replay_slot_8",
    ]
    assert request["inputs"][1]["format"] == "FRACTAL_NZ"


def test_every_adapter_requires_replay_row_and_rejects_legacy_fields() -> None:
    structured = {
        "schema_version": 1,
        "kernel_type": "ArgMaxV2",
        "inputs": [{"shape": [2, 4], "dtype": "DT_BF16"}],
    }
    with pytest.raises(run_op_microbench.MicrobenchError) as missing_row:
        run_op_microbench.normalize_request(structured)
    assert missing_row.value.details["unknown_fields"] == ["inputs"]

    replay_row = {
        "Input Shapes": "2,4;",
        "Input Data Types": "DT_BF16;DT_INT64",
        "Input Formats": "ND;ND",
        "Output Shapes": "2",
        "Output Data Types": "DT_INT64",
        "Output Formats": "ND",
    }
    conflicting = {**structured, "replay_row": replay_row}
    with pytest.raises(run_op_microbench.MicrobenchError):
        run_op_microbench.normalize_request(conflicting)

    with pytest.raises(run_op_microbench.MicrobenchError) as attrs_error:
        run_op_microbench.normalize_request(
            {
                "schema_version": 1,
                "kernel_type": "ArgMaxV2",
                "replay_row": replay_row,
                "attrs": {"dim": -1},
            }
        )
    assert "attrs" in attrs_error.value.message


@pytest.mark.parametrize("keep_artifacts", [["always"], {"policy": "always"}])
def test_keep_artifacts_rejects_non_string_values_with_structured_error(
    keep_artifacts,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = _request()
    request["profiling"]["keep_artifacts"] = keep_artifacts
    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench.normalize_request(request)
    assert error.value.code == "INVALID_REQUEST"

    request_path = _write_request(tmp_path / "request.json", request)
    assert run_op_microbench.main(["--request", str(request_path)]) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["error"]["code"] == "INVALID_REQUEST"


def test_replay_spec_builds_default_and_overridden_contracts() -> None:
    default = adapters._replay_spec("ExampleOp", "torch.example")
    assert default == adapters.AdapterSpec(
        name="ExampleOp",
        api_path="torch.example",
        target_op_types=("ExampleOp",),
        replay_module="tools.perf_data_collection.op_replay.ExampleOp_run",
    )

    overridden = adapters._replay_spec(
        "ExampleOp",
        "torch.example",
        target_op_types=("PhysicalExampleOp",),
        is_composite=True,
    )
    assert overridden.target_op_types == ("PhysicalExampleOp", "ExampleOp")
    assert overridden.is_composite is True


def test_replay_row_database_matching_keeps_empty_slots(tmp_path: Path) -> None:
    replay_row = {
        **_row(),
        "Input Shapes": "2,4;",
        "Input Data Types": "DT_BF16;DT_INT64",
        "Input Formats": "ND;ND",
        "Output Shapes": "2",
        "Output Data Types": "DT_INT64",
        "Output Formats": "ND",
    }
    source = _write_database(tmp_path / "ArgMaxV2.csv", [replay_row])
    request = run_op_microbench.normalize_request(
        {"schema_version": 1, "kernel_type": "ArgMaxV2", "replay_row": replay_row}
    )

    result = run_op_microbench.run_database_simulation(request, source, "request-id")

    assert result["status"] == "succeeded"
    assert result["operator"]["requested_kernel_type"] == "ArgMaxV2"


def test_generic_replay_runtime_bridge_uses_allowlisted_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch, fake_torch_npu = _install_fake_torch(monkeypatch)
    tensor = _FakeTensor((2, 4), fake_torch.bfloat16)

    class FakeReplay:
        prepared = False

        def prepare(self):
            self.prepared = True

        def build_case(self, row):
            assert row["Input Shapes"] == "2,4;"
            return {"inputs": [tensor, tensor], "nested": {"tensor": tensor}}

        def run_case(self, case):
            assert case["inputs"][0] is tensor
            return tensor

    replay = FakeReplay()
    runtime_initialized = []

    def fake_import(module_name):
        if module_name.endswith(".common"):
            return SimpleNamespace(init_runtime=lambda: runtime_initialized.append(True))
        return SimpleNamespace(op=replay)

    monkeypatch.setattr(adapters.importlib, "import_module", fake_import)
    request = run_op_microbench.normalize_request(
        {
            "schema_version": 1,
            "kernel_type": "ArgMaxV2",
            "replay_row": {
                "Input Shapes": "2,4;",
                "Input Data Types": "DT_BF16;DT_INT64",
                "Input Formats": "ND;ND",
                "Output Shapes": "2",
                "Output Data Types": "DT_INT64",
                "Output Formats": "ND",
            },
        }
    )

    case = adapters.build_runtime_case(request)

    assert replay.prepared is True
    assert runtime_initialized == [True]
    assert case["invoke"]() is tensor
    assert fake_torch_npu.npu.config.allow_internal_format is False


def test_inplace_replay_exposes_mutated_cache_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch, _ = _install_fake_torch(monkeypatch)
    tensors = [_FakeTensor((1,), fake_torch.bfloat16) for _ in range(5)]
    replay = SimpleNamespace(
        prepare=lambda: None,
        build_case=lambda row: {"inputs": tensors},
        run_case=lambda case: None,
    )

    def fake_import(module_name):
        if module_name.endswith(".common"):
            return SimpleNamespace(init_runtime=lambda: None)
        return SimpleNamespace(op=replay)

    monkeypatch.setattr(adapters.importlib, "import_module", fake_import)
    request = {
        "kernel_type": "ReshapeAndCacheNdKernel",
        "replay_row": {},
    }

    case = adapters.build_runtime_case(request)

    assert case["invoke"]() == tuple(tensors[2:4])


def test_manual_replay_bridges_issue_one_operator_call() -> None:
    token = object()
    fia_case = {
        "query": token,
        "key": token,
        "value": token,
        "atten_mask": None,
        "actual_seq_lengths": [1],
        "actual_seq_lengths_kv": [1],
        "block_table": None,
        "query_rope": None,
        "key_rope": None,
        "num_heads": 2,
        "scale": 0.5,
        "input_layout": "TND",
        "num_key_value_heads": 1,
        "sparse_mode": 0,
        "block_size": 0,
        "softmax_lse_flag": False,
    }
    fia_calls = []
    fia_module = SimpleNamespace(
        build_row_case=lambda row: fia_case,
        validate_case_for_replay=lambda case, row: None,
    )
    fake_npu = SimpleNamespace(
        npu_fused_infer_attention_score=lambda *args, **kwargs: fia_calls.append((args, kwargs)) or token
    )
    _, invoke_fia = adapters._build_fia_case(fia_module, {}, fake_npu)
    assert invoke_fia() is token
    assert len(fia_calls) == 1

    with pytest.raises(ValueError, match="not replayable"):
        adapters._build_fia_case(
            SimpleNamespace(
                build_row_case=lambda row: fia_case,
                validate_case_for_replay=lambda case, row: "not replayable",
            ),
            {},
            fake_npu,
        )

    quant_case = {
        "weight_format": "FRACTAL_NZ",
        "x_tensor": token,
        "weight_tensor": token,
        "scale_tensor": token,
        "bias_tensor": None,
        "offset_tensor": None,
        "pertoken_scale_tensor": None,
        "output_dtype_name": "DT_BF16",
    }
    quant_calls = []
    quant_module = SimpleNamespace(
        build_row_tensors=lambda row: quant_case,
        run_quant_matmul=lambda *args: quant_calls.append(args) or token,
    )
    _, invoke_quant = adapters._build_quant_batch_matmul_case(quant_module, {})
    assert invoke_quant() is token
    assert quant_calls[0][-1] is True

    mla_outputs = [object(), object()]
    mla_case = {"outputs": mla_outputs}
    mla_calls = []
    mla_module = SimpleNamespace(
        build_case=lambda row: mla_case,
        run_case=lambda case: mla_calls.append(case),
    )
    _, invoke_mla = adapters._build_mla_preprocess_case(mla_module, {})
    assert invoke_mla() == tuple(mla_outputs)
    assert mla_calls == [mla_case]


@pytest.mark.parametrize("has_prefix_state", [False, True])
def test_ring_replay_bridge_selects_a_deterministic_configuration(
    has_prefix_state: bool,
) -> None:
    token = object()
    case = {
        "q_nope": token,
        "q_rope": token,
        "k_nope": token,
        "k_rope": token,
        "value": token,
        "mask": token,
        "seqlen_candidates": [token],
        "has_prefix_state": has_prefix_state,
        "head_num": 2,
        "kv_head_num": 2,
        "pre_out": token,
        "prev_lse": token,
        "qk_scale": 0.5,
        "mask_type": "mask_type_triu",
        "calc_type": "calc_type_default",
        "output": token,
        "softmax_lse": token,
    }
    calls = []
    fake_npu = SimpleNamespace(
        atb=SimpleNamespace(npu_ring_mla=lambda **kwargs: calls.append(kwargs) or (token, token))
    )

    _, invoke = adapters._build_ring_mla_case(SimpleNamespace(build_row_case=lambda row: case), {}, fake_npu)

    assert invoke() == (token, token)
    assert calls[0]["mask_type"] == ("no_mask" if has_prefix_state else "mask_type_triu")


def test_worker_reports_replay_input_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch, _ = _install_fake_torch(monkeypatch)
    output = _FakeTensor((2,), fake_torch.int64)
    request = run_op_microbench.normalize_request(
        {
            "schema_version": 1,
            "kernel_type": "ArgMaxV2",
            "replay_row": {
                "Input Shapes": "2,4;",
                "Input Data Types": "DT_BF16;DT_INT64",
                "Input Formats": "ND;ND",
                "Output Shapes": "2",
                "Output Data Types": "DT_INT64",
                "Output Formats": "ND",
            },
            "profiling": {"warmup_count": 0, "repeat_count": 1},
        }
    )
    spec = adapters.ADAPTERS["ArgMaxV2"]
    monkeypatch.setattr(
        op_worker,
        "build_runtime_case",
        lambda normalized: {
            "spec": spec,
            "invoke": lambda: output,
        },
    )

    result = op_worker.execute(request)

    assert result["inputs"][0]["origin"] == "replay_metadata"
    assert result["inputs"][0]["device_shape"] is None
    assert result["inputs"][0]["size_bytes"] == 16


def test_replay_row_accepts_scalar_shape_and_validates_real_output_shape() -> None:
    scalar_request = _request()
    scalar_request["replay_row"]["Input Shapes"] = "[];2,4"
    normalized = run_op_microbench.normalize_request(scalar_request)
    assert normalized["inputs"][0]["shape"] == []

    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench._validate_replay_outputs(
            _request()["replay_row"],
            [{"logical_shape": [1, 3], "dtype": "DT_BF16", "format": "ND"}],
        )
    assert error.value.code == "REPLAY_OUTPUT_MISMATCH"
    assert error.value.details["mismatches"][0]["field"] == "logical_shape"


def test_scalar_and_zero_element_output_shapes_are_not_compatible() -> None:
    assert not run_op_microbench._shapes_compatible([], [0])
    assert not run_op_microbench._shapes_compatible([0], [])

    replay_row = {
        "Output Shapes": "[]",
        "Output Data Types": "DT_FLOAT",
        "Output Formats": "ND",
    }
    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench._validate_replay_outputs(
            replay_row,
            [{"logical_shape": [0], "dtype": "DT_FLOAT", "format": "ND"}],
        )
    assert error.value.code == "REPLAY_OUTPUT_MISMATCH"


def test_ring_unknown_auxiliary_output_descriptor_keeps_slot_without_asserting_fields() -> None:
    replay_row = {
        "Output Shapes": "12,16,128;",
        "Output Data Types": "DT_BF16;UNDEFINED",
        "Output Formats": "ND;UNDEFINED",
    }
    outputs = [
        {"logical_shape": [12, 16, 128], "dtype": "DT_BF16", "format": "ND"},
        {"logical_shape": [16, 12], "dtype": "DT_FLOAT", "format": "ND"},
    ]

    assert run_op_microbench._validate_replay_outputs(replay_row, outputs, kernel_type="RINGMLAPrefillBF16Kernel")


def test_ring_concrete_prefix_output_descriptor_is_still_strict() -> None:
    replay_row = {
        "Output Shapes": "12,16,128;16,12",
        "Output Data Types": "DT_BF16;DT_FLOAT",
        "Output Formats": "ND;ND",
    }
    outputs = [
        {"logical_shape": [12, 16, 128], "dtype": "DT_BF16", "format": "ND"},
        {"logical_shape": [16, 12], "dtype": "DT_FLOAT", "format": "ND"},
    ]
    assert run_op_microbench._validate_replay_outputs(replay_row, outputs, kernel_type="RINGMLAPrefillBF16Kernel")

    outputs[1]["logical_shape"] = [12, 16]
    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench._validate_replay_outputs(replay_row, outputs, kernel_type="RINGMLAPrefillBF16Kernel")
    assert error.value.code == "REPLAY_OUTPUT_MISMATCH"


@pytest.mark.parametrize(
    ("kernel_type", "replay_row", "outputs"),
    [
        (
            "ArgMaxV2",
            {"Output Shapes": "2", "Output Data Types": "DT_INT32", "Output Formats": "ND"},
            [{"logical_shape": [2], "dtype": "DT_INT64", "format": "ND"}],
        ),
        (
            "Sort",
            {
                "Output Shapes": "2,4;2,4",
                "Output Data Types": "DT_FLOAT;DT_INT32",
                "Output Formats": "ND;ND",
            },
            [
                {"logical_shape": [2, 4], "dtype": "DT_FLOAT", "format": "ND"},
                {"logical_shape": [2, 4], "dtype": "DT_INT64", "format": "ND"},
            ],
        ),
    ],
)
def test_standalone_validation_accepts_python_api_index_dtype_without_cast(kernel_type, replay_row, outputs) -> None:
    assert run_op_microbench._validate_replay_outputs(replay_row, outputs, kernel_type=kernel_type)


def test_worker_main_serializes_failure_metadata(tmp_path: Path) -> None:
    request_path = tmp_path / "bad.json"
    request_path.write_text("{bad-json", encoding="utf-8")
    metadata_path = tmp_path / "worker.json"

    return_code = op_worker.main(["--request", str(request_path), "--metadata-out", str(metadata_path)])

    assert return_code == 1
    result = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error"]["type"] == "JSONDecodeError"


def test_worker_cann_version_uses_ascend_home_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0")
    fake_torch_npu = SimpleNamespace(version=SimpleNamespace(cann=None, cann_version=None))

    assert op_worker._cann_version(fake_torch_npu) == "8.5.0"


def test_real_result_csv_uses_task_type_and_has_no_simulation_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_msprof(monkeypatch)
    request = _request(include_raw_records=True)
    result = run_op_microbench.run_msprof_backend(
        run_op_microbench.normalize_request(request),
        "request-id",
        None,
    )

    _, row = run_op_microbench._result_to_csv(result)

    assert row["Accelerator Core"] == "AI_CORE"
    assert row["Result Source"] == "msprof"
    assert row["Simulated"] == "false"
    assert row["Source CSV"] == ""
    assert row["Source Row"] == ""

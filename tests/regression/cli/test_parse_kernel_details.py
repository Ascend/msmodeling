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

"""Tests for parse_kernel_details.py — build_argparser and CLI argument parsing."""

import csv
from pathlib import Path

import pytest

from tools.perf_data_collection.parsers.parse_kernel_details import (
    DEFAULT_DEVICE,
    KernelDetailsParser,
    build_argparser,
)


class TestBuildArgparser:
    """Verify CLI argument parsing for parse_kernel_details.py."""

    def test_returns_argument_parser(self):
        parser = build_argparser()
        assert isinstance(parser, pytest.importorskip("argparse").ArgumentParser)

    def test_profiling_path_is_required(self):
        parser = build_argparser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_profiling_path_accepts_file(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.profiling_path == "kernel_details.csv"

    def test_database_path_default_none(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.database_path is None

    def test_database_path_accepts_path(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv", "--database-path", "/custom/db"])
        assert args.database_path == Path("/custom/db")

    def test_device_default(self):
        from tools.perf_data_collection.parsers.parse_kernel_details import (
            DEFAULT_DEVICE,
        )

        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.device == DEFAULT_DEVICE

    def test_device_accepts_supported(self):
        from tools.perf_data_collection.parsers.parse_kernel_details import (
            SUPPORTED_DEVICES,
        )

        parser = build_argparser()
        for device in SUPPORTED_DEVICES:
            args = parser.parse_args(["--profiling-path", "kernel_details.csv", "--device", device])
            assert args.device == device

    def test_device_rejects_unsupported(self):
        parser = build_argparser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--profiling-path", "kernel_details.csv", "--device", "UNKNOWN_DEVICE"])

    def test_vllm_version_default_none(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.vllm_version is None

    def test_vllm_version_accepts_valid(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv", "--vllm-version", "0.18.0"])
        assert args.vllm_version == "0.18.0"

    def test_vllm_version_accepts_full_dir_name(self):
        parser = build_argparser()
        args = parser.parse_args(
            [
                "--profiling-path",
                "kernel_details.csv",
                "--vllm-version",
                "vllm0.18.0_torch2.9.0_cann8.5",
            ]
        )
        assert args.vllm_version == "vllm0.18.0_torch2.9.0_cann8.5"

    def test_vllm_version_rejects_invalid(self):
        parser = build_argparser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--profiling-path", "kernel_details.csv", "--vllm-version", ""])

    def test_torch_version_default_none(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.torch_version is None

    def test_torch_version_accepts_valid(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv", "--torch-version", "2.9.0"])
        assert args.torch_version == "2.9.0"

    def test_cann_version_default_none(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.cann_version is None

    def test_cann_version_accepts_valid(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv", "--cann-version", "8.5"])
        assert args.cann_version == "8.5"

    def test_non_existent_flags_rejected(self):
        parser = build_argparser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--profiling-path", "kernel_details.csv", "--unknown-flag"])


# Columns of a msprof kernel_details CSV export (subset relevant for
# aggregation plus the extra numeric columns required by _load_rows).
KERNEL_DETAILS_FIELDNAMES = [
    "Type",
    "OP State",
    "Accelerator Core",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Duration(us)",
    "aicore_time(us)",
    "aic_total_cycles",
    "aic_mac_time(us)",
    "aic_mac_ratio",
    "aic_scalar_time(us)",
    "aic_scalar_ratio",
    "aic_mte1_time(us)",
    "aic_mte1_ratio",
    "aic_mte2_time(us)",
    "aic_mte2_ratio",
    "aic_fixpipe_time(us)",
    "aic_fixpipe_ratio",
    "aic_icache_miss_rate",
    "aiv_time(us)",
    "aiv_total_cycles",
    "aiv_vec_time(us)",
    "aiv_vec_ratio",
    "aiv_scalar_time(us)",
    "aiv_scalar_ratio",
    "aiv_mte2_time(us)",
    "aiv_mte2_ratio",
    "aiv_mte3_time(us)",
    "aiv_mte3_ratio",
    "aiv_icache_miss_rate",
    "cube_utilization(%)",
]


def _kernel_details_row(dtype: str, duration: int) -> dict:
    return {
        "Type": "Add",
        "OP State": "static",
        "Accelerator Core": "AI_VECTOR_CORE",
        "Input Shapes": "4,8;4,8",
        "Input Data Types": f"{dtype};{dtype}",
        "Input Formats": "ND;ND",
        "Output Shapes": "4,8",
        "Output Data Types": dtype,
        "Output Formats": "ND",
        "Duration(us)": str(duration),
        "aiv_time(us)": str(duration),
        "aiv_vec_time(us)": str(duration),
    }


def _write_kernel_details_csv(path: Path, specs: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KERNEL_DETAILS_FIELDNAMES)
        writer.writeheader()
        for dtype, duration in specs:
            writer.writerow(_kernel_details_row(dtype, duration))


def _parse_csv(tmp_path: Path, name: str, specs: list[tuple[str, int]]) -> list[dict]:
    csv_path = tmp_path / f"{name}.csv"
    _write_kernel_details_csv(csv_path, specs)
    parser = KernelDetailsParser(
        device=DEFAULT_DEVICE,
        kernel_details_path=str(csv_path),
        database_path=str(tmp_path / name),
    )
    output_files = parser.parse_and_export()
    assert len(output_files) == 1
    with output_files[0].open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class TestAggregationKeepsDtypeIdentity:
    """Aggregation must group rows by the full semantic identity (#499).

    Rows sharing shapes but differing in dtypes/formats are distinct perf
    cases: FP16 and FP32 samples of the same shape must not be averaged
    into one row that keeps only the first dtype.
    """

    def test_mixed_dtypes_keep_two_rows(self, tmp_path: Path):
        rows = _parse_csv(
            tmp_path,
            "mixed",
            [("DT_FLOAT16", 10), ("DT_FLOAT16", 14), ("DT_FLOAT", 30), ("DT_FLOAT", 34)],
        )
        assert len(rows) == 2
        by_dtype = {row["Input Data Types"]: row for row in rows}
        assert set(by_dtype) == {"DT_FLOAT16;DT_FLOAT16", "DT_FLOAT;DT_FLOAT"}
        fp16 = by_dtype["DT_FLOAT16;DT_FLOAT16"]
        fp32 = by_dtype["DT_FLOAT;DT_FLOAT"]
        assert fp16["Profiling Average Duration(us)"] == "12.000000"
        assert fp16["Profiling Median Duration(us)"] == "12.000000"
        assert fp16["Profiling Std Duration(us)"] == "2.000000"
        assert fp16["Output Data Types"] == "DT_FLOAT16"
        assert fp32["Profiling Average Duration(us)"] == "32.000000"
        assert fp32["Profiling Median Duration(us)"] == "32.000000"
        assert fp32["Profiling Std Duration(us)"] == "2.000000"
        assert fp32["Output Data Types"] == "DT_FLOAT"

    def test_dtype_identity_is_order_independent(self, tmp_path: Path):
        rows = _parse_csv(
            tmp_path,
            "reversed",
            [("DT_FLOAT", 30), ("DT_FLOAT", 34), ("DT_FLOAT16", 10), ("DT_FLOAT16", 14)],
        )
        assert len(rows) == 2
        by_dtype = {row["Input Data Types"]: row for row in rows}
        fp16 = by_dtype["DT_FLOAT16;DT_FLOAT16"]
        fp32 = by_dtype["DT_FLOAT;DT_FLOAT"]
        assert fp16["Profiling Average Duration(us)"] == "12.000000"
        assert fp16["Output Data Types"] == "DT_FLOAT16"
        assert fp32["Profiling Average Duration(us)"] == "32.000000"
        assert fp32["Output Data Types"] == "DT_FLOAT"

    def test_same_dtype_samples_still_aggregate(self, tmp_path: Path):
        rows = _parse_csv(tmp_path, "fp16", [("DT_FLOAT16", 10), ("DT_FLOAT16", 14)])
        assert len(rows) == 1
        row = rows[0]
        assert row["Input Data Types"] == "DT_FLOAT16;DT_FLOAT16"
        assert row["Profiling Average Duration(us)"] == "12.000000"
        assert row["Profiling Median Duration(us)"] == "12.000000"
        assert row["Profiling Std Duration(us)"] == "2.000000"

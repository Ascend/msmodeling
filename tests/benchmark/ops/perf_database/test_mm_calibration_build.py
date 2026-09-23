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

import csv

import pytest
import torch

from tensor_cast.device import DeviceProfile
from tensor_cast.performance_model.calibration.mm_build import build_mm_profile
from tensor_cast.performance_model.calibration.gmm_build import build_gmm_profile
from tensor_cast.performance_model.calibration.profile import validate_mm_profile_document


def _write_database(tmp_path, rows):
    database = tmp_path / "vllm0.18.0_torch2.9.0_cann8.5"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        "version: '0.18.0'\ndevice: TEST_DEVICE\ncann_version: '8.5'\n",
        encoding="utf-8",
    )
    with (database / "MatMulV3.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Input Shapes",
                "Input Data Types",
                "Input Formats",
                "Average Duration(us)",
                "Profiling Average Duration(us)",
            ],
        )
        writer.writeheader()
        for m, k, n, utilization in rows:
            peak = DeviceProfile.all_device_profiles["TEST_DEVICE"].mma_ops[torch.bfloat16]
            latency_us = 2 * m * k * n / (peak * utilization) * 1e6
            writer.writerow(
                {
                    "Input Shapes": f'"{m},{k};{n},{k}"',
                    "Input Data Types": "DT_BF16;DT_BF16",
                    "Input Formats": "ND;ND",
                    "Average Duration(us)": latency_us,
                    "Profiling Average Duration(us)": latency_us,
                }
            )
    return database


def test_build_mm_profile_fits_common_kn_curves_and_externalizes_audit(tmp_path):
    rows = [
        (m, 4096, 4096, 0.55 if m < 512 else 0.85)
        for m in (64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048)
    ]
    database = _write_database(tmp_path, rows)
    audit_path = tmp_path / "profile.audit.yaml"

    profile, audit = build_mm_profile(
        database,
        targets=["dense=MatMulV3"],
        profile_id="test-mm-p0",
        audit_path=audit_path,
    )

    validate_mm_profile_document(profile)
    assert profile["version"] == 3
    assert profile["source"]["audit"] == "profile.audit.yaml"
    assert profile["mm_source"] == profile["source"]
    assert profile["calibration_sources"]["mm"] == profile["source"]
    assert len(profile["mm_models"]) == 1
    model = profile["mm_models"][0]
    assert model["kind"] == "dense"
    assert model["compute_dtype"] == "bfloat16"
    assert model["calibration_mode"] == "total_latency"
    assert len(model["curves"]) == 1
    assert len(model["curves"][0]["m_points"]) >= 2
    assert "default" not in model
    assert audit["models"][0]["unique_shape_count"] == len(rows)
    assert audit["models"][0]["curve_count"] == 1
    assert audit["models"][0]["curve_knot_count"] >= 2


def test_build_mm_profile_uses_average_duration_even_when_profiling_differs(tmp_path):
    database = _write_database(tmp_path, [(64, 4096, 4096, 0.8), (128, 4096, 4096, 0.8)])
    rows = []
    path = database / "MatMulV3.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["Profiling Average Duration(us)"] = float(row["Average Duration(us)"]) * 4
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    profile, audit = build_mm_profile(database, targets=["dense=MatMulV3"], audit_path=tmp_path / "profile.audit.yaml")

    assert profile["mm_models"][0]["curves"]
    assert all(
        source["latency_column"] == "Average Duration(us)"
        for sample in audit["models"][0]["samples"]
        for source in sample["sources"]
    )


def test_build_mm_profile_rejects_rows_without_average_duration(tmp_path):
    database = _write_database(tmp_path, [(64, 4096, 4096, 0.8)])
    path = database / "MatMulV3.csv"
    path.write_text(
        "Input Shapes,Input Data Types,Input Formats,Average Duration(us),Profiling Average Duration(us)\n"
        '"64,4096;4096,4096",DT_BF16;DT_BF16,ND;ND,,1\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no_compatible_rows"):
        build_mm_profile(database, targets=["dense=MatMulV3"], audit_path=tmp_path / "profile.audit.yaml")


def test_build_mm_profile_records_missing_lfs_targets_without_fabricating_model(tmp_path):
    database = _write_database(tmp_path, [(1024, 4096, 4096, 0.8), (2048, 4096, 4096, 0.85)])
    (database / "QuantBatchMatmulV3.csv").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n",
        encoding="utf-8",
    )

    profile, audit = build_mm_profile(
        database,
        targets=["dense=MatMulV3", "weight_quant:w8a8=QuantBatchMatmulV3"],
        allow_missing_kernels=True,
        audit_path=tmp_path / "profile.audit.yaml",
    )

    assert [model["kind"] for model in profile["mm_models"]] == ["dense"]
    quant_report = next(report for report in audit["kernels"] if report["kernel"] == "QuantBatchMatmulV3")
    assert quant_report["status"] == "lfs_pointer"
    assert not any(model["kind"] == "weight_quant" for model in profile["mm_models"])


def test_build_mm_profile_rejects_missing_kernel_by_default(tmp_path):
    database = _write_database(tmp_path, [(1024, 4096, 4096, 0.8)])

    with pytest.raises(ValueError, match="kernel QuantBatchMatmulV3 is unavailable"):
        build_mm_profile(
            database,
            targets=["weight_quant:w8a8=QuantBatchMatmulV3"],
            audit_path=tmp_path / "profile.audit.yaml",
        )


def test_build_mm_profile_partitions_quantized_rows_by_input_dtype(tmp_path):
    database = _write_database(tmp_path, [(1024, 4096, 4096, 0.8)])
    peak = DeviceProfile.all_device_profiles["TEST_DEVICE"].mma_ops[torch.int8]
    with (database / "QuantBatchMatmulV3.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Input Shapes", "Input Data Types", "Input Formats", "Average Duration(us)"],
        )
        writer.writeheader()
        for m in (64, 128, 256, 512):
            k, n, utilization = 4096, 4096, 0.7
            writer.writerow(
                {
                    "Input Shapes": f'"{m},{k};{n // 32},{k // 16},16,32;{n};{n}"',
                    "Input Data Types": "INT8;INT8;FLOAT;INT32",
                    "Input Formats": "ND;FRACTAL_NZ;ND;ND",
                    "Average Duration(us)": 2 * m * k * n / (peak * utilization) * 1e6,
                }
            )

    profile, _audit = build_mm_profile(
        database,
        targets=["weight_quant:w8a8=QuantBatchMatmulV3", "weight_quant:w4a8=QuantBatchMatmulV3"],
        allow_missing_kernels=True,
        audit_path=tmp_path / "profile.audit.yaml",
    )

    assert [(model["kind"], model["quantization"], model["compute_dtype"]) for model in profile["mm_models"]] == [
        ("weight_quant", "w8a8", "int8")
    ]


def _write_gmm_database(tmp_path, *, swiglu=False, weight_orientation="kn"):
    database = tmp_path / "vllm0.18.0_torch2.9.0_cann8.5"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        "version: '0.18.0'\ndevice: TEST_DEVICE\ncann_version: '8.5'\n", encoding="utf-8"
    )
    kernel = "GroupedMatmulSwigluQuant" if swiglu else "GroupedMatmul"
    fieldnames = [
        "Input Shapes",
        "Input Data Types",
        "Input Formats",
        "Output Shapes",
        "Output Data Types",
        "Average Duration(us)",
        "Profiling Average Duration(us)",
    ]
    with (database / f"{kernel}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        peak = DeviceProfile.all_device_profiles["TEST_DEVICE"].mma_ops[torch.int8]
        for m, utilization in ((4, 0.5), (8, 0.7), (16, 0.8)):
            k, n, experts = 8, 16, 2
            gemm_n = 2 * n if swiglu else n
            weight = f"{experts},{k},{gemm_n}" if weight_orientation == "kn" else f"{experts},{gemm_n},{k}"
            if swiglu:
                input_shapes = f'"{m},{k};{weight};{experts},1;{m};{experts}"'
                input_dtypes = "INT8;INT8;FLOAT;FLOAT;INT64"
                input_formats = "ND;ND;ND;ND;ND"
                output_shapes, output_dtypes = f'"{m},{n};{m}"', "INT8;FLOAT"
            else:
                input_shapes = f'"{m},{k};{weight};;;;;;{experts};{m}"'
                input_dtypes = "INT8;INT8;INT32;DT_BF16;FLOAT;FLOAT16;FLOAT16;INT64;FLOAT"
                input_formats = "ND;ND;ND;ND;ND;ND;ND;ND;ND"
                output_shapes, output_dtypes = f'"{m},{n}"', "DT_BF16"
            latency = 2 * m * k * gemm_n / peak / utilization * 1e6
            writer.writerow(
                {
                    "Input Shapes": input_shapes,
                    "Input Data Types": input_dtypes,
                    "Input Formats": input_formats,
                    "Output Shapes": output_shapes,
                    "Output Data Types": output_dtypes,
                    "Average Duration(us)": latency,
                    "Profiling Average Duration(us)": latency,
                }
            )
    return database


def test_build_gmm_profile_aligns_csv_weight_orientation_and_keeps_variants_separate(tmp_path):
    database = _write_gmm_database(tmp_path, weight_orientation="nk")
    profile, audit = build_gmm_profile(
        database,
        targets=["plain=GroupedMatmul"],
        audit_path=tmp_path / "gmm.audit.yaml",
    )

    validate_mm_profile_document(profile)
    assert len(profile["gmm_models"]) == 1
    model = profile["gmm_models"][0]
    assert model["variant"] == "plain"
    assert model["domain"]["m_total"] == [4, 17]
    assert model["curves"][0]["num_experts"] == 2
    assert audit["models"][0]["curve_count"] == 1
    assert profile["calibration_sources"]["gmm"] == profile["gmm_source"]


def test_build_gmm_profile_fits_all_expert_counts_from_average_duration(tmp_path):
    database = _write_gmm_database(tmp_path)
    path = database / "GroupedMatmul.csv"
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["Profiling Average Duration(us)"] = float(row["Average Duration(us)"]) * 4
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    profile, audit = build_gmm_profile(
        database, targets=["plain=GroupedMatmul"], audit_path=tmp_path / "gmm.audit.yaml"
    )

    assert len(profile["gmm_models"][0]["curves"]) == 1
    assert all(
        source["latency_column"] == "Average Duration(us)"
        for sample in audit["models"][0]["samples"]
        for source in sample["sources"]
    )


def test_build_gmm_profile_rejects_missing_average_duration(tmp_path):
    database = _write_gmm_database(tmp_path)
    path = database / "GroupedMatmul.csv"
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["Average Duration(us)"] = ""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    profile, audit = build_gmm_profile(
        database, targets=["plain=GroupedMatmul"], audit_path=tmp_path / "gmm.audit.yaml"
    )
    assert audit["kernels"][0]["rejected_rows"] == 1
    assert profile["gmm_models"][0]["curves"]


def test_build_gmm_profile_handles_swiglu_quant_output_and_uses_independent_source(tmp_path):
    database = _write_gmm_database(tmp_path, swiglu=True)
    profile, _audit = build_gmm_profile(
        database,
        targets=["swiglu_quant=GroupedMatmulSwigluQuant"],
        audit_path=tmp_path / "gmm.audit.yaml",
    )
    model = profile["gmm_models"][0]
    assert model["variant"] == "swiglu_quant"
    assert model["output_dtype"] == "int8"
    assert model["curves"][0]["gemm_n"] == 32

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

"""Regression tests for the calibrated analytic foundation."""

from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.device import DeviceProfile
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.calibrated_analytic import CalibratedAnalyticPerformanceModel
from tensor_cast.performance_model.comm_analytic import CommAnalyticModel
from tensor_cast.performance_model.calibration import (
    CalibrationDataSource,
    CalibrationRule,
    EmptyCalibrationDataSource,
    ProfileCalibrationDataSource,
)
from tensor_cast.performance_model.calibration.signature import CalibrationSignature, build_calibration_signature
from tensor_cast.performance_model.calibration.sqlite_profile import write_sqlite_profile
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo


def _device() -> MagicMock:
    device = MagicMock()
    device.name = "TEST_DEVICE"
    return device


def _raw_analytic(device: MagicMock, result: PerformanceModel.Result) -> MagicMock:
    model = MagicMock(spec=PerformanceModel)
    model.name = "analytic"
    model.device_profile = device
    model.process_op.return_value = result
    model.get_classifiers.return_value = []
    return model


def _mm_info() -> OpInvokeInfo:
    lhs = torch.empty(4, 8, device="meta", dtype=torch.bfloat16)
    rhs = torch.empty(8, 16, device="meta", dtype=torch.bfloat16)
    return OpInvokeInfo(torch.ops.aten.mm.default, (lhs, rhs), {}, torch.empty(4, 16, device="meta"))


def test_calibrated_analytic_returns_raw_result_when_no_rule_matches():
    device = _device()
    raw = PerformanceModel.Result(execution_time_s=3e-6, statistics={"raw": True})
    base = _raw_analytic(device, raw)
    model = CalibratedAnalyticPerformanceModel(base, EmptyCalibrationDataSource())

    result = model.process_op(_mm_info())

    assert result.execution_time_s == 3e-6
    assert result.statistics["raw"] is True
    assert result.statistics["source"] == "ANALYTIC_RAW"
    assert result.statistics["calibration"]["fallback_reason"] == "no_matching_rule"
    base.process_op.assert_called_once()


@pytest.mark.parametrize("failure_stage", ["lookup", "apply"])
def test_calibrated_analytic_returns_raw_result_when_calibration_fails(failure_stage):
    class BrokenRule(CalibrationRule):
        def apply(self, raw_result, signature):
            del raw_result, signature
            raise ValueError("invalid calibration rule")

    class BrokenSource(CalibrationDataSource):
        def lookup(self, signature):
            del signature
            if failure_stage == "lookup":
                raise ValueError("invalid calibration lookup")
            return BrokenRule()

    raw = PerformanceModel.Result(execution_time_s=3e-6, statistics={"raw": True})
    result = CalibratedAnalyticPerformanceModel(_raw_analytic(_device(), raw), BrokenSource()).process_op(_mm_info())

    assert result.execution_time_s == raw.execution_time_s
    assert result.statistics["source"] == "ANALYTIC_RAW"
    assert result.statistics["calibration"]["fallback_reason"] == "calibration_error"


def test_calibrated_analytic_uses_distinct_model_name():
    model = CalibratedAnalyticPerformanceModel(
        _raw_analytic(_device(), PerformanceModel.Result(1.0)), EmptyCalibrationDataSource()
    )
    assert model.name == "calibrated"


def test_model_runner_requires_profile_for_calibrated_model():
    with pytest.raises(ValueError, match="analytic_calibration_profile"):
        UserInputConfig(model_id="Qwen/Qwen3-32B", performance_model="calibrated")


def test_model_runner_keeps_raw_analytic_when_profile_is_provided():
    config = UserInputConfig(
        model_id="Qwen/Qwen3-32B",
        performance_model="analytic",
        analytic_calibration_profile="missing.sqlite",
    )
    models = ModelRunner.create_performance_models(config, DeviceProfile.all_device_profiles["TEST_DEVICE"])
    assert [model.name for model in models] == ["analytic"]


def test_calibrated_analytic_applies_rule_when_signature_matches():
    class DoubleLatency(CalibrationRule):
        def apply(self, raw_result, signature):
            assert signature.op_family == "matmul"
            return PerformanceModel.Result(execution_time_s=raw_result.execution_time_s * 2)

    class MatchingSource(CalibrationDataSource):
        def lookup(self, signature):
            assert signature.tc_op == "aten.mm.default"
            return DoubleLatency()

    device = _device()
    base = _raw_analytic(device, PerformanceModel.Result(execution_time_s=3e-6))
    result = CalibratedAnalyticPerformanceModel(base, MatchingSource()).process_op(_mm_info())

    assert result.execution_time_s == 6e-6


def test_signature_uses_tensorcast_semantics_for_supported_families():
    raw = PerformanceModel.Result(execution_time_s=1e-6, statistics={"message_size_bytes": 128, "group_size": 2})
    mm_signature = build_calibration_signature(_mm_info(), raw, device_name="TEST_DEVICE")
    assert mm_signature.op_family == "matmul"
    assert mm_signature.dtype == "bfloat16"
    assert mm_signature.features == {
        "m": 4,
        "k": 8,
        "n": 16,
        "min_kn": 8,
        "max_kn": 16,
        "mn": 64,
        "aspect": 2.0,
        "mm_kind": "dense",
        "quantization": "none",
        "compute_dtype": "bfloat16",
    }

    query = torch.empty(2, 4, 8, device="meta", dtype=torch.bfloat16)
    key = torch.empty(6, 2, 8, device="meta", dtype=torch.bfloat16)
    attention_info = OpInvokeInfo(
        torch.ops.tensor_cast.attention.default,
        (query, key, key),
        {"phase": "decode"},
        query,
    )
    attention_signature = build_calibration_signature(attention_info, raw, device_name="TEST_DEVICE")
    assert attention_signature.op_family == "attention"
    assert attention_signature.features == {
        "q_tokens": 2,
        "kv_len": 6,
        "head_dim": 8,
        "heads": 4,
        "kv_heads": 2,
        "phase": "decode",
    }

    comm_info = OpInvokeInfo(
        torch.ops.tensor_cast.all_reduce.default,
        (torch.empty(16, device="meta", dtype=torch.bfloat16), 0, [0, 1]),
        {},
        torch.empty(16, device="meta", dtype=torch.bfloat16),
    )
    comm_signature = build_calibration_signature(comm_info, raw, device_name="TEST_DEVICE")
    assert comm_signature.op_family == "communication"
    assert comm_signature.features == {
        "collective": "all_reduce",
        "message_bytes": 128,
        "analytic_message_bytes": 128,
        "group_size": 2,
    }


def test_signature_rejects_non_positive_matmul_dimensions():
    lhs = torch.empty(4, 0, device="meta", dtype=torch.bfloat16)
    rhs = torch.empty(0, 16, device="meta", dtype=torch.bfloat16)
    info = OpInvokeInfo(torch.ops.aten.mm.default, (lhs, rhs), {}, torch.empty(4, 16, device="meta"))

    signature = build_calibration_signature(info, PerformanceModel.Result(1e-6), device_name="TEST_DEVICE")

    assert signature.op_family == "matmul"
    assert not {"m", "k", "n"} & signature.features.keys()


def test_sparse_mla_prefill_uses_latent_sfa_kv_semantics():
    q = torch.empty(256, 64, 256, device="meta", dtype=torch.bfloat16)
    projected_kv = torch.empty(256, 64, 448, device="meta", dtype=q.dtype)
    absorbed_q = torch.empty(0, 64, 576, device="meta", dtype=q.dtype)
    kv_cache = torch.empty(33, 128, 576, device="meta", dtype=q.dtype)
    args = (
        q,
        projected_kv,
        absorbed_q,
        kv_cache,
        torch.empty(1, 32, dtype=torch.int64, device="meta"),
        torch.tensor([0, 4096], dtype=torch.int32),
        torch.tensor([4096], dtype=torch.int64),
        torch.tensor([4096], dtype=torch.int64),
        256,
        512,
        2048,
        torch.empty(1, 256, 2048, dtype=torch.int64, device="meta"),
    )
    sparse = build_calibration_signature(
        OpInvokeInfo(
            torch.ops.tensor_cast.mla_sparse_attention.default,
            args,
            {"is_decode_values": [False]},
            q,
        ),
        PerformanceModel.Result(execution_time_s=1e-6),
        device_name="TEST_DEVICE",
    )
    assert sparse.features["heads"] == 64
    assert sparse.features["kv_heads"] == 1
    assert sparse.features["head_dim"] == 512
    assert sparse.features["effective_kv_len"] == 4096


def test_signature_covers_quantized_linear_semantics():
    x = torch.empty(32, 128, device="meta", dtype=torch.int8)
    weight = torch.empty(128, 256, device="meta", dtype=torch.int8)
    info = OpInvokeInfo(torch.ops.tensor_cast.static_quant_linear.default, (x, weight), {}, None)

    signature = build_calibration_signature(info, PerformanceModel.Result(1e-6), device_name="TEST_DEVICE")

    assert signature.op_family == "matmul"
    assert signature.dtype == "int8"
    assert signature.features == {
        "m": 32,
        "k": 128,
        "n": 256,
        "min_kn": 128,
        "max_kn": 256,
        "mn": 8192,
        "aspect": 2.0,
        "mm_kind": "weight_quant",
        "quantization": "w8a8",
        "compute_dtype": "int8",
    }


def _gmm_info(op, *, swiglu=False, skewed=False):
    m_values = (2, 2) if not skewed else (1, 3)
    k = 8
    n = 16
    gemm_n = n * 2 if swiglu else n
    x = [torch.empty(m, k, device="meta", dtype=torch.int8) for m in m_values]
    w = [torch.empty(k, gemm_n, device="meta", dtype=torch.int8) for _ in m_values]
    if swiglu:
        args = (
            x,
            w,
            [torch.empty(1, device="meta") for _ in x],
            [None for _ in x],
            [torch.empty(1, device="meta") for _ in x],
            [None for _ in x],
            [None for _ in x],
            None,
        )
        out = torch.empty(sum(m_values), n, device="meta", dtype=torch.int8)
    else:
        args = (x, w, [None for _ in x])
        out = torch.empty(sum(m_values), n, device="meta", dtype=torch.bfloat16)
    return OpInvokeInfo(op, args, {}, out)


def test_signature_covers_grouped_int8_semantics_without_flattening_experts():
    info = _gmm_info(torch.ops.tensor_cast.grouped_matmul_quant.default, skewed=True)
    signature = build_calibration_signature(info, PerformanceModel.Result(1e-6), device_name="TEST_DEVICE")

    assert signature.op_family == "matmul"
    assert signature.features["mm_kind"] == "grouped"
    assert signature.features["gmm_variant"] == "plain"
    assert signature.features["m_total"] == 4
    assert signature.features["num_experts"] == 2
    assert signature.features["k"] == 8
    assert signature.features["n"] == 16
    assert signature.features["distribution"] == "skewed"


def test_signature_separates_grouped_swiglu_quant_and_gemm_width():
    info = _gmm_info(torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default, swiglu=True)
    signature = build_calibration_signature(info, PerformanceModel.Result(1e-6), device_name="TEST_DEVICE")

    assert signature.features["gmm_variant"] == "swiglu_quant"
    assert signature.features["n"] == 16
    assert signature.features["gemm_n"] == 32
    assert signature.features["output_dtype"] == "int8"


def test_gmm_profile_matches_grouped_curve_without_affecting_dense_mm(tmp_path):
    profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-gmm-v2
device: TEST_DEVICE
software_stack: cann8.5
source: {database: test-db, database_digest: abc, audit: test.audit.yaml, audit_sha256: def}
gmm_models:
  - variant: plain
    tc_ops: [tensor_cast.grouped_matmul_quant.default]
    quantization: w8a8
    input_dtype: int8
    weight_dtype: int8
    output_dtype: bfloat16
    distribution: balanced
    calibration_mode: total_latency
    domain: {m_total: [1, 17], k: [8, 9], n: [16, 17], gemm_n: [16, 17], num_experts: [2, 3]}
    curves:
      - num_experts: 2
        k: 8
        n: 16
        gemm_n: 16
        m_points: [[1, 0.2], [16, 0.4]]
""",
    )
    info = _gmm_info(torch.ops.tensor_cast.grouped_matmul_quant.default)
    raw = PerformanceModel.Result(
        execution_time_s=25e-6,
        statistics={"mma_ops_time_s": 10e-6, "memory_access_time_s": 20e-6, "static_cost_time_s": 5e-6},
    )
    source = ProfileCalibrationDataSource(
        str(profile), device_name="TEST_DEVICE", software_stack="cann8.5", device_compute_efficiency=0.7
    )
    result = CalibratedAnalyticPerformanceModel(_raw_analytic(_device(), raw), source).process_op(info)

    assert result.execution_time_s == pytest.approx(24.748737e-6)
    assert result.statistics["calibration"]["rule_type"] == "gmm_utilization"
    assert result.statistics["calibration"]["match_level"] == "curve"

    dense = build_calibration_signature(_mm_info(), raw, device_name="TEST_DEVICE")
    assert source.lookup(dense) is None


def _write_profile(tmp_path, content: str):
    import yaml

    document = yaml.safe_load(content)
    if document.get("version") != 3:
        raise AssertionError("test profiles must use SQLite documents")
    profile = tmp_path / f"calibration-{len(list(tmp_path.glob('calibration-*.sqlite')))}.sqlite"
    write_sqlite_profile(document, profile)
    return profile


def test_mm_utilization_calibrates_only_mma_roofline_component(tmp_path):
    profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-mm-v2
device: TEST_DEVICE
software_stack: cann8.5
source:
  database: test-db
  database_digest: abc
  audit: test.audit.yaml
  audit_sha256: def
mm_models:
  - kind: dense
    quantization: none
    compute_dtype: bfloat16
    domain: {m: [1, 10], k: [1, 20], n: [1, 20]}
    regions:
      - id: dense-measured
        when: {m: [1, 10]}
        utilization: 0.5
        fixed_overhead_us: 5
        confidence: measured
""",
    )
    raw = PerformanceModel.Result(
        execution_time_s=17e-6,
        statistics={
            "mma_ops_time_s": 10e-6,
            "gp_ops_time_s": 2e-6,
            "compute_time_s": 12e-6,
            "memory_access_time_s": 8e-6,
            "static_cost_time_s": 5e-6,
        },
    )
    source = ProfileCalibrationDataSource(
        str(profile),
        device_name="TEST_DEVICE",
        software_stack="cann8.5",
        device_compute_efficiency=0.7,
    )

    result = CalibratedAnalyticPerformanceModel(_raw_analytic(_device(), raw), source).process_op(_mm_info())

    assert result.execution_time_s == pytest.approx(21e-6)
    assert result.statistics["mma_ops_time_s"] == pytest.approx(14e-6)
    assert result.statistics["compute_time_s"] == pytest.approx(16e-6)
    assert result.statistics["calibration"]["utilization"] == 0.5
    assert result.statistics["calibration"]["match_level"] == "region"


def test_mm_utilization_keeps_raw_analytic_when_no_local_curve_is_available(tmp_path):
    profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-mm-v2
device: TEST_DEVICE
software_stack: cann8.5
source: {database: test-db, database_digest: abc, audit: test.audit.yaml, audit_sha256: def}
mm_models:
  - kind: dense
    quantization: none
    compute_dtype: bfloat16
    domain: {m: [1, 17], k: [1, 129], n: [1, 129]}
    curves:
      - {k: 128, n: 128, m_points: [[8, 0.4], [16, 0.4]]}
""",
    )
    source = ProfileCalibrationDataSource(str(profile), device_name="TEST_DEVICE", device_compute_efficiency=0.7)
    signature = build_calibration_signature(_mm_info(), PerformanceModel.Result(1e-6), device_name="TEST_DEVICE")

    rule = source.lookup(signature)

    assert rule is None
    assert source.last_lookup_diagnostic["fallback_reason"] == "no_shape_aware_match"


def test_profile_rejects_removed_global_default_utilization(tmp_path):
    profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-no-global-default
device: TEST_DEVICE
software_stack: cann8.5
source: {database: test-db, database_digest: abc, audit: test.audit.yaml, audit_sha256: def}
mm_models:
  - kind: dense
    quantization: none
    compute_dtype: bfloat16
    domain: {m: [1, 5], k: [1, 9], n: [1, 17]}
    default: {utilization: 0.4}
    curves:
      - {k: 8, n: 16, m_points: [[1, 0.2], [4, 0.4]]}
""",
    )

    with pytest.raises(ValueError, match="no longer supports a global default"):
        ProfileCalibrationDataSource(str(profile), device_name="TEST_DEVICE", software_stack="cann8.5")


def _dense_signature(*, m: int, k: int, n: int) -> CalibrationSignature:
    return CalibrationSignature(
        tc_op="aten.mm.default",
        op_family="matmul",
        device="TEST_DEVICE",
        dtype="bfloat16",
        features={
            "mm_kind": "dense",
            "quantization": "none",
            "compute_dtype": "bfloat16",
            "m": m,
            "k": k,
            "n": n,
        },
    )


def test_mm_shape_aware_rules_use_kn_interpolation_knn_and_nearest_neighbor(tmp_path):
    profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-mm-shape-aware
device: TEST_DEVICE
software_stack: cann8.5
source: {database: test-db, database_digest: abc, audit: test.audit.yaml, audit_sha256: def}
mm_models:
  - kind: dense
    quantization: none
    compute_dtype: bfloat16
    calibration_mode: total_latency
    domain: {m: [1, 9], k: [1, 65], n: [1, 65]}
    curves:
      - {k: 8, n: 8, m_points: [[1, 0.2], [8, 0.2]]}
      - {k: 32, n: 8, m_points: [[1, 0.4], [8, 0.4]]}
      - {k: 8, n: 32, m_points: [[1, 0.8], [8, 0.8]]}
""",
    )
    source = ProfileCalibrationDataSource(str(profile), device_name="TEST_DEVICE", software_stack="cann8.5")

    interpolated = source.lookup(_dense_signature(m=4, k=16, n=16))
    assert interpolated is not None
    assert interpolated.details["match_level"] == "kn_interpolation"
    assert interpolated.confidence == "interpolated"

    # With no enclosing triangle, two local curves use log-shape KNN weighting.
    knn_profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-mm-knn
device: TEST_DEVICE
software_stack: cann8.5
source: {database: test-db, database_digest: abc, audit: test.audit.yaml, audit_sha256: def}
mm_models:
  - kind: dense
    quantization: none
    compute_dtype: bfloat16
    domain: {m: [1, 9], k: [1, 33], n: [1, 33]}
    curves:
      - {k: 8, n: 16, m_points: [[1, 0.2], [8, 0.2]]}
      - {k: 16, n: 8, m_points: [[1, 0.8], [8, 0.8]]}
""",
    )
    knn = ProfileCalibrationDataSource(str(knn_profile), device_name="TEST_DEVICE", software_stack="cann8.5").lookup(
        _dense_signature(m=4, k=16, n=16)
    )
    assert knn is not None
    assert knn.details["match_level"] == "knn_weighted"
    assert knn.utilization == pytest.approx(0.5)

    nearest_profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-mm-nearest
device: TEST_DEVICE
software_stack: cann8.5
source: {database: test-db, database_digest: abc, audit: test.audit.yaml, audit_sha256: def}
mm_models:
  - kind: dense
    quantization: none
    compute_dtype: bfloat16
    domain: {m: [1, 9], k: [1, 33], n: [1, 33]}
    curves:
      - {k: 8, n: 16, m_points: [[1, 0.2], [8, 0.2]]}
""",
    )
    nearest = ProfileCalibrationDataSource(
        str(nearest_profile), device_name="TEST_DEVICE", software_stack="cann8.5"
    ).lookup(_dense_signature(m=4, k=16, n=16))
    assert nearest is not None
    assert nearest.details["match_level"] == "nearest_neighbor"
    assert nearest.utilization == pytest.approx(0.2)


def test_mm_shape_aware_rules_reject_distant_nearest_neighbor(tmp_path):
    profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-mm-distant-neighbor
device: TEST_DEVICE
software_stack: cann8.5
source: {database: test-db, database_digest: abc, audit: test.audit.yaml, audit_sha256: def}
mm_models:
  - kind: dense
    quantization: none
    compute_dtype: bfloat16
    domain: {m: [1, 9], k: [1, 129], n: [1, 129]}
    curves:
      - {k: 128, n: 128, m_points: [[1, 0.2], [8, 0.2]]}
""",
    )
    source = ProfileCalibrationDataSource(str(profile), device_name="TEST_DEVICE", software_stack="cann8.5")

    rule = source.lookup(_dense_signature(m=4, k=8, n=8))

    assert rule is None
    assert source.last_lookup_diagnostic["fallback_reason"] == "no_shape_aware_match"


def test_gmm_shape_aware_matching_requires_the_same_expert_count(tmp_path):
    profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-gmm-shape-aware
device: TEST_DEVICE
software_stack: cann8.5
source: {database: test-db, database_digest: abc, audit: test.audit.yaml, audit_sha256: def}
gmm_models:
  - variant: plain
    tc_ops: [tensor_cast.grouped_matmul_quant.default]
    quantization: w8a8
    input_dtype: int8
    weight_dtype: int8
    output_dtype: bfloat16
    distribution: balanced
    calibration_mode: total_latency
    domain: {m_total: [1, 9], k: [1, 65], n: [1, 65], gemm_n: [1, 65], num_experts: [1, 9]}
    curves:
      - {num_experts: 2, k: 8, n: 32, gemm_n: 32, m_points: [[1, 0.3], [8, 0.3]]}
      - {num_experts: 4, k: 8, n: 16, gemm_n: 16, m_points: [[1, 0.9], [8, 0.9]]}
""",
    )
    signature = CalibrationSignature(
        tc_op="tensor_cast.grouped_matmul_quant.default",
        op_family="matmul",
        device="TEST_DEVICE",
        dtype="int8",
        features={
            "mm_kind": "grouped",
            "gmm_variant": "plain",
            "quantization": "w8a8",
            "compute_dtype": "int8",
            "weight_dtype": "int8",
            "distribution": "balanced",
            "m_total": 4,
            "k": 8,
            "n": 16,
            "gemm_n": 16,
            "num_experts": 2,
        },
    )
    rule = ProfileCalibrationDataSource(str(profile), device_name="TEST_DEVICE", software_stack="cann8.5").lookup(
        signature
    )

    assert rule is not None
    assert rule.details["match_level"] == "nearest_neighbor"
    assert rule.utilization == pytest.approx(0.3)
    assert rule.details["source_curves"][0]["num_experts"] == 2


def test_total_latency_curve_interpolates_m_and_ignores_raw_bound_floor(tmp_path):
    profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-mm-curve
device: TEST_DEVICE
software_stack: cann8.5
source: {database: test-db, database_digest: abc, audit: test.audit.yaml, audit_sha256: def}
mm_models:
  - kind: dense
    quantization: none
    compute_dtype: bfloat16
    calibration_mode: total_latency
    domain: {m: [1, 5], k: [8, 9], n: [16, 17]}
    curves:
      - k: 8
        n: 16
        m_points: [[1, 0.2], [4, 0.4]]
""",
    )
    info = OpInvokeInfo(
        torch.ops.aten.mm.default,
        (
            torch.empty(2, 8, device="meta", dtype=torch.bfloat16),
            torch.empty(8, 16, device="meta", dtype=torch.bfloat16),
        ),
        {},
        torch.empty(2, 16, device="meta", dtype=torch.bfloat16),
    )
    raw = PerformanceModel.Result(
        execution_time_s=25e-6,
        statistics={
            "mma_ops_time_s": 10e-6,
            "compute_time_s": 10e-6,
            "memory_access_time_s": 20e-6,
            "static_cost_time_s": 5e-6,
        },
    )
    source = ProfileCalibrationDataSource(str(profile), device_name="TEST_DEVICE", device_compute_efficiency=0.7)

    result = CalibratedAnalyticPerformanceModel(_raw_analytic(_device(), raw), source).process_op(info)

    assert result.execution_time_s == pytest.approx(24.748737e-6)
    assert result.statistics["calibration"]["calibration_mode"] == "total_latency"
    assert result.statistics["calibration"]["match_level"] == "curve"
    assert result.statistics["calibration"]["interpolated_feature"] == "m"
    assert result.statistics["calibration"]["confidence"] == "interpolated"

    knot_info = OpInvokeInfo(
        torch.ops.aten.mm.default,
        (
            torch.empty(4, 8, device="meta", dtype=torch.bfloat16),
            torch.empty(8, 16, device="meta", dtype=torch.bfloat16),
        ),
        {},
        torch.empty(4, 16, device="meta", dtype=torch.bfloat16),
    )
    knot = CalibratedAnalyticPerformanceModel(_raw_analytic(_device(), raw), source).process_op(knot_info)
    assert knot.statistics["calibration"]["confidence"] == "measured"


def test_communication_curve_uses_exact_and_adjacent_latency_buckets(tmp_path):
    profile = _write_profile(
        tmp_path,
        """
version: 3
id: test-comm-v2
device: ATLAS_800_A3_752T_128G_DIE
software_stack: cann8.5
source: {database: test-db, database_digest: abc, audit: test.audit.yaml, audit_sha256: def}
communication_source: {directory: hccl/v8.5, directory_digest: abc, audit: comm.audit.yaml, audit_sha256: def}
communication_models:
  - collective: all_reduce
    tc_op: tensor_cast.all_reduce.default
    dtype: bfloat16
    group_size: 2
    topology_tier: 2
    latency_points_us: [[128, 4.0], [256, 8.0]]
""",
    )
    device = DeviceProfile.all_device_profiles["ATLAS_800_A3_752T_128G_DIE"]
    model = CalibratedAnalyticPerformanceModel(
        CommAnalyticModel(device),
        ProfileCalibrationDataSource(str(profile), device_name=device.name, software_stack="cann8.5"),
    )

    exact = model.process_op(
        OpInvokeInfo(
            torch.ops.tensor_cast.all_reduce.default,
            (torch.empty(64, device="meta", dtype=torch.bfloat16), 0, [0, 1]),
            {},
            torch.empty(64, device="meta", dtype=torch.bfloat16),
        )
    )
    interpolated = model.process_op(
        OpInvokeInfo(
            torch.ops.tensor_cast.all_reduce.default,
            (torch.empty(96, device="meta", dtype=torch.bfloat16), 0, [0, 1]),
            {},
            torch.empty(96, device="meta", dtype=torch.bfloat16),
        )
    )
    outside = model.process_op(
        OpInvokeInfo(
            torch.ops.tensor_cast.all_reduce.default,
            (torch.empty(192, device="meta", dtype=torch.bfloat16), 0, [0, 1]),
            {},
            torch.empty(192, device="meta", dtype=torch.bfloat16),
        )
    )

    assert exact.execution_time_s == pytest.approx(4e-6)
    assert exact.statistics["calibration"]["interpolated"] is False
    assert interpolated.execution_time_s == pytest.approx(6e-6)
    assert interpolated.statistics["calibration"]["interpolated"] is True
    assert outside.statistics["source"] == "ANALYTIC_RAW"
    assert outside.statistics["calibration"]["fallback_reason"] == "shape_out_of_range:message_bytes"


def test_communication_signature_normalizes_reduce_scatter_and_all_to_all_bytes():
    device = DeviceProfile.all_device_profiles["ATLAS_800_A3_752T_128G_DIE"]
    raw_model = CommAnalyticModel(device)
    reduce_scatter = OpInvokeInfo(
        torch.ops.tensor_cast.reduce_scatter.default,
        (torch.empty(128, device="meta", dtype=torch.bfloat16), 0, 0, [0, 1]),
        {},
        torch.empty(64, device="meta", dtype=torch.bfloat16),
    )
    all_to_all = OpInvokeInfo(
        torch.ops.tensor_cast.all_to_all.default,
        (torch.empty(64, device="meta", dtype=torch.bfloat16), [32, 32], [32, 32], 0, [0, 1]),
        {},
        torch.empty(64, device="meta", dtype=torch.bfloat16),
    )

    reduce_signature = build_calibration_signature(
        reduce_scatter, raw_model.process_op(reduce_scatter), device_name=device.name
    )
    all_to_all_signature = build_calibration_signature(
        all_to_all, raw_model.process_op(all_to_all), device_name=device.name
    )

    assert reduce_signature.features["analytic_message_bytes"] == 256
    assert reduce_signature.features["message_bytes"] == 128
    assert all_to_all_signature.features["analytic_message_bytes"] == 64
    assert all_to_all_signature.features["message_bytes"] == 128

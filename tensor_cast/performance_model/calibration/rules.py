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

"""First-release analytic calibration rules.

Rules transform only total op latency.  They intentionally leave roofline
component statistics intact: PMU engine counters cannot be added as independent
latency components because their pipelines overlap.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..base import PerformanceModel
from ..bound_analyzer import BoundAnalyzer, StatsKey
from .base import CalibrationRule
from .signature import CalibrationSignature


def _calibrated_result(
    raw_result: PerformanceModel.Result,
    calibrated_latency_s: float,
    *,
    profile_id: str,
    rule_id: str,
    confidence: str,
    rule_type: str,
    details: Mapping[str, Any],
    statistics: Mapping[str, Any] | None = None,
) -> PerformanceModel.Result:
    """Copy a result and attach calibration provenance without mutating raw data."""
    raw_latency_s = raw_result.execution_time_s
    calibrated_statistics = dict(raw_result.statistics if statistics is None else statistics)
    calibrated_statistics.update(
        {
            "source": "ANALYTIC_CALIBRATED",
            "calibration": {
                "raw_latency_s": raw_latency_s,
                "calibrated_latency_s": calibrated_latency_s,
                "profile_id": profile_id,
                "rule_id": rule_id,
                "rule_type": rule_type,
                "confidence": confidence,
                "fallback_reason": None,
                **dict(details),
            },
        }
    )
    return PerformanceModel.Result(execution_time_s=calibrated_latency_s, statistics=calibrated_statistics)


@dataclass(frozen=True)
class AttentionLatencyRule(CalibrationRule):
    """Replace an attention component with its measured/interpolated latency."""

    latency_us: float
    kernel_type: str
    profile_id: str
    rule_id: str
    confidence: str = "measured"
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.latency_us <= 0:
            raise ValueError("attention latency must be positive")
        if not self.kernel_type:
            raise ValueError("attention kernel_type must be non-empty")

    def apply(
        self,
        raw_result: PerformanceModel.Result,
        signature: CalibrationSignature,
    ) -> PerformanceModel.Result:
        del signature
        return _calibrated_result(
            raw_result,
            self.latency_us * 1e-6,
            profile_id=self.profile_id,
            rule_id=self.rule_id,
            confidence=self.confidence,
            rule_type="attention_latency",
            details={
                "latency_us": self.latency_us,
                "kernel_type": self.kernel_type,
                "component_role": "attention_core",
                **dict(self.details),
            },
        )


@dataclass(frozen=True)
class MmaUtilizationRule(CalibrationRule):
    """Replace the raw MMA efficiency while preserving the analytic roofline.

    ``utilization`` is measured against the device peak throughput.  The raw
    analytic MMA time already contains ``device_compute_efficiency``, so the
    calibrated MMA component is ``raw_mma * efficiency / utilization``.
    Memory, GP, communication, and unclassified residual overhead remain
    independent instead of being scaled with the MMA component.  The generic
    device static cost is replaced by the rule's measured fixed overhead,
    because target MM kernels may launch below the device-wide default floor.
    """

    utilization: float
    device_compute_efficiency: float
    profile_id: str
    rule_id: str
    fixed_overhead_us: float = 0.0
    calibration_mode: str = "component_roofline"
    rule_type: str = "mm_utilization"
    confidence: str = "measured"
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 < self.utilization <= 1:
            raise ValueError("utilization must be in (0, 1]")
        if not 0 < self.device_compute_efficiency <= 1:
            raise ValueError("device_compute_efficiency must be in (0, 1]")
        if self.fixed_overhead_us < 0:
            raise ValueError("fixed_overhead_us must be non-negative")
        if self.calibration_mode not in {"component_roofline", "total_latency"}:
            raise ValueError("unsupported MM utilization calibration_mode")

    def apply(
        self,
        raw_result: PerformanceModel.Result,
        signature: CalibrationSignature,
    ) -> PerformanceModel.Result:
        del signature
        components = BoundAnalyzer.components(raw_result)
        raw_mma_time_s = components.mma_ops_time_s
        if raw_mma_time_s <= 0:
            raise ValueError("mm_utilization requires a positive raw MMA component")

        statistics = dict(raw_result.statistics)
        static_cost_time_s = statistics.get("static_cost_time_s", 0.0)
        if not isinstance(static_cost_time_s, (int, float)) or static_cost_time_s < 0:
            static_cost_time_s = 0.0

        raw_compute_time_s = components.mma_ops_time_s + components.gp_ops_time_s
        raw_dynamic_time_s = max(
            components.memory_time_s,
            components.communication_time_s,
            raw_compute_time_s,
        )
        accounted_raw_time_s = float(static_cost_time_s) + raw_dynamic_time_s
        residual_overhead_s = max(0.0, raw_result.execution_time_s - accounted_raw_time_s)

        calibrated_mma_time_s = raw_mma_time_s * self.device_compute_efficiency / self.utilization
        if self.calibration_mode == "total_latency":
            calibrated_latency_s = self.fixed_overhead_us * 1e-6 + calibrated_mma_time_s
            return _calibrated_result(
                raw_result,
                calibrated_latency_s,
                profile_id=self.profile_id,
                rule_id=self.rule_id,
                confidence=self.confidence,
                rule_type=self.rule_type,
                details={
                    "utilization": self.utilization,
                    "calibration_mode": self.calibration_mode,
                    "device_compute_efficiency": self.device_compute_efficiency,
                    "raw_mma_time_s": raw_mma_time_s,
                    "calibrated_mma_time_s": calibrated_mma_time_s,
                    "fixed_overhead_us": self.fixed_overhead_us,
                    **dict(self.details),
                },
            )
        calibrated_compute_time_s = calibrated_mma_time_s + components.gp_ops_time_s
        calibrated_dynamic_time_s = max(
            components.memory_time_s,
            components.communication_time_s,
            calibrated_compute_time_s,
        )
        calibrated_latency_s = residual_overhead_s + self.fixed_overhead_us * 1e-6 + calibrated_dynamic_time_s

        statistics[StatsKey.MMA_OPS] = calibrated_mma_time_s
        statistics[StatsKey.COMPUTE] = calibrated_compute_time_s
        statistics["is_compute_bound"] = calibrated_compute_time_s >= max(
            components.memory_time_s,
            components.communication_time_s,
        )
        return _calibrated_result(
            raw_result,
            calibrated_latency_s,
            profile_id=self.profile_id,
            rule_id=self.rule_id,
            confidence=self.confidence,
            rule_type=self.rule_type,
            details={
                "utilization": self.utilization,
                "calibration_mode": self.calibration_mode,
                "device_compute_efficiency": self.device_compute_efficiency,
                "raw_mma_time_s": raw_mma_time_s,
                "calibrated_mma_time_s": calibrated_mma_time_s,
                "calibrated_compute_time_s": calibrated_compute_time_s,
                "residual_overhead_s": residual_overhead_s,
                "raw_static_cost_time_s": float(static_cost_time_s),
                "fixed_overhead_us": self.fixed_overhead_us,
                **dict(self.details),
            },
            statistics=statistics,
        )


@dataclass(frozen=True)
class GmmUtilizationRule(MmaUtilizationRule):
    """Apply a total-latency utilization measured for a grouped matmul.

    The fitted utilization intentionally absorbs SwiGLU, quantization, memory,
    launch, and routing overheads.  Raw roofline component statistics remain
    available for diagnostics, but are not recombined into the calibrated
    latency.  This is the GMM counterpart of the P0 MM total-latency mode.
    """

    rule_type: str = "gmm_utilization"


@dataclass(frozen=True)
class CommunicationLatencyCurveRule(CalibrationRule):
    """Apply an in-range piecewise-linear collective latency curve."""

    points_us: tuple[tuple[int, float], ...]
    profile_id: str
    rule_id: str
    confidence: str = "measured"
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.points_us:
            raise ValueError("communication latency curve needs at least one point")
        if any(bytes_ <= 0 or latency_us < 0 for bytes_, latency_us in self.points_us):
            raise ValueError("communication latency curve points must be positive bytes and non-negative latency")
        if any(left[0] >= right[0] for left, right in zip(self.points_us, self.points_us[1:])):
            raise ValueError("communication latency curve message bytes must be strictly increasing")

    def apply(
        self,
        raw_result: PerformanceModel.Result,
        signature: CalibrationSignature,
    ) -> PerformanceModel.Result:
        message_bytes = signature.features.get("message_bytes")
        if not isinstance(message_bytes, (int, float)):
            raise ValueError("CommunicationLatencyCurveRule requires a numeric message_bytes feature")
        message_bytes = int(message_bytes)
        for point_bytes, latency_us in self.points_us:
            if message_bytes == point_bytes:
                return self._result(raw_result, message_bytes, latency_us, interpolated=False)
        for (left_bytes, left_latency), (right_bytes, right_latency) in zip(self.points_us, self.points_us[1:]):
            if left_bytes < message_bytes < right_bytes:
                ratio = (message_bytes - left_bytes) / (right_bytes - left_bytes)
                latency_us = left_latency + ratio * (right_latency - left_latency)
                return self._result(raw_result, message_bytes, latency_us, interpolated=True)
        raise ValueError("communication latency curve query is outside the declared range")

    def _result(
        self,
        raw_result: PerformanceModel.Result,
        message_bytes: int,
        latency_us: float,
        *,
        interpolated: bool,
    ) -> PerformanceModel.Result:
        return _calibrated_result(
            raw_result,
            latency_us * 1e-6,
            profile_id=self.profile_id,
            rule_id=self.rule_id,
            confidence=self.confidence,
            rule_type="communication_latency_curve",
            details={
                "message_bytes": message_bytes,
                "latency_us": latency_us,
                "interpolated": interpolated,
                **dict(self.details),
            },
        )

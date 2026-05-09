"""EmpiricalPerformanceModel: measurement-based performance model."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from overrides import override

from ..device import DeviceProfile
from .base import PerformanceModel
from .op_invoke_info import OpInvokeInfo
from .profiling_database.data_source import (
    DataSourcePerformanceModel,
    QueryResult,
    QuerySource,
)

logger = logging.getLogger(__name__)


@dataclass
class EmpiricalOpRecord:
    """Raw data captured by EmpiricalPerformanceModel for one op invocation.

    Stored in EmpiricalPerformanceModel.op_records after each process_op() call.
    MetricsCollector reads these records to compute M1-M5 metrics.
    """

    func_name: str
    lookup_result: Optional[QueryResult]  # None for full MISS
    analytic_latency_s: float
    tc_shapes: List[tuple]
    miss_reason: Optional[str] = None


class EmpiricalPerformanceModel(PerformanceModel):
    """Performance model based on measured data from a DataSourcePerformanceModel.

    Accepts DataSourcePerformanceModel instance, process_op()
    queries data source first, falls back to fallback_model on miss.

    Example::

        data_source = ProfilingDataSource(data_dir, device_profile=device_profile)
        pm = EmpiricalPerformanceModel(device_profile, data_source)
    """

    def __init__(
        self,
        device_profile: DeviceProfile,
        data_source: DataSourcePerformanceModel,
        fallback_model: Optional[PerformanceModel] = None,
    ):
        super().__init__("empirical", device_profile)
        self.data_source = data_source
        self._fallback_model = fallback_model
        # Raw op records — read by MetricsCollector to compute M1-M5 metrics
        self.op_records: List[EmpiricalOpRecord] = []

    @property
    def fallback_model(self) -> PerformanceModel:
        if self._fallback_model is None:
            from .analytic import AnalyticPerformanceModel

            self._fallback_model = AnalyticPerformanceModel(self.device_profile)
        return self._fallback_model

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        result = self.data_source.lookup(op_invoke_info)
        func_name = str(op_invoke_info.func).removeprefix("torch.ops.")

        # Analytic fallback — needed for MISS latency and as weight
        analytic_result = self.fallback_model.process_op(op_invoke_info)
        tc_shapes = [
            tuple(a.shape) for a in op_invoke_info.args if isinstance(a, torch.Tensor)
        ]

        if result is not None and result.source != QuerySource.PARTIAL:
            # Full HIT
            self.op_records.append(
                EmpiricalOpRecord(
                    func_name, result, analytic_result.execution_time_s, tc_shapes
                )
            )
            empirical_s = result.latency_us * 1e-6
            return PerformanceModel.Result(
                execution_time_s=empirical_s,
                statistics={
                    "source": result.source.name,
                    "confidence": result.confidence,
                    **result.details,
                    **result.shape_debug_statistics(),
                },
            )

        if result is not None and result.source == QuerySource.PARTIAL:
            # PARTIAL
            self.op_records.append(
                EmpiricalOpRecord(
                    func_name, result, analytic_result.execution_time_s, tc_shapes
                )
            )
            empirical_s = result.latency_us * 1e-6
            return PerformanceModel.Result(
                execution_time_s=empirical_s,
                statistics={
                    "source": result.source.name,
                    "confidence": result.confidence,
                    **result.details,
                    **result.shape_debug_statistics(),
                },
            )

        # Full MISS
        reason = getattr(self.data_source, "last_miss_reason", "unknown")
        self.op_records.append(
            EmpiricalOpRecord(
                func_name,
                None,
                analytic_result.execution_time_s,
                tc_shapes,
                miss_reason=reason,
            )
        )
        if isinstance(analytic_result.statistics, dict):
            analytic_result.statistics["shape_match_rule"] = "analytic"
        return analytic_result

    @override
    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        """
        Return classifiers from the fallback model so that breakdown reporting
        still works when an op is handled by the fallback path.
        """
        return self.fallback_model.get_classifiers()

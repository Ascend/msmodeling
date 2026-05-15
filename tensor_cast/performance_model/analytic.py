import logging

try:
    # Native in Python 3.11+
    from enum import StrEnum
except ImportError:
    # Fallback for Python 3.10
    from strenum import StrEnum
from typing import Dict, List, Tuple

from overrides import override

from ..device import DeviceProfile
from .base import PerformanceModel
from .op_estimator_registry import get_op_estimator
from .op_invoke_info import OpInvokeInfo


logger = logging.getLogger(__name__)


class StatsKey(StrEnum):
    COMPUTE = "compute_time_s"
    MMA_OPS = "mma_ops_time_s"
    GP_OPS = "gp_ops_time_s"
    MEMORY_ACCESS = "memory_access_time_s"
    COMMUNICATION = "comm_time_s"


class OpBoundClassifier(PerformanceModel.OpClassifier):
    @property
    def name(self):
        return "OpBound"

    def classify(self, event_list: List[Tuple[OpInvokeInfo, "PerformanceModel.Result"]]) -> Dict[str, float]:
        COMPUTE_BOUND_MMA = "compute_bound_mma"
        COMPUTE_BOUND_GP = "compute_bound_gp"
        MEMORY_BOUND = "memory_bound"
        COMM_BOUND = "communication_bound"
        breakdown: Dict[str, float] = {
            MEMORY_BOUND: 0,
            COMM_BOUND: 0,
            COMPUTE_BOUND_MMA: 0,
            COMPUTE_BOUND_GP: 0,
        }
        breakdown_keys = list(breakdown.keys())
        for _, result in event_list:
            time_list = [
                result.statistics.get(StatsKey.MEMORY_ACCESS, 0),
                result.statistics.get(StatsKey.COMMUNICATION, 0),
                result.statistics.get(StatsKey.COMPUTE, 0),
            ]
            max_value = max(time_list)
            max_index = time_list.index(max_value)
            if max_index < 2:
                breakdown[breakdown_keys[max_index]] += max_value
            else:
                breakdown[COMPUTE_BOUND_MMA] += result.statistics.get(StatsKey.MMA_OPS, 0)
                breakdown[COMPUTE_BOUND_GP] += result.statistics.get(StatsKey.GP_OPS, 0)
        return breakdown


class AnalyticPerformanceModel(PerformanceModel):
    """
    Analytic performance model uses simple roofline model to estimate the
    op execution time.
    TODO: add cache model to more accurately estimate the execution time.
    """

    def __init__(self, device_profile: DeviceProfile):
        super().__init__("analytic", device_profile)
        self.classifiers = [OpBoundClassifier()]

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        op_estimator = get_op_estimator(op_invoke_info.func, self.device_profile.name)
        result = op_estimator(op_invoke_info, self.device_profile)
        return result

    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self.classifiers

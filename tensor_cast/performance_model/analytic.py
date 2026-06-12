import logging
from typing import Dict, List, Tuple

from overrides import override

from ..device import DeviceProfile
from .base import PerformanceModel
from .bound_analyzer import (
    BoundComponents,
    COMMUNICATION_BOUND,
    COMPUTE_BOUND_GP,
    COMPUTE_BOUND_MMA,
    MEMORY_BOUND,
    UNKNOWN_BOUND,
    BoundAnalyzer,
)
from .op_estimator_registry import get_op_estimator
from .op_invoke_info import OpInvokeInfo


logger = logging.getLogger(__name__)


class OpBoundClassifier(PerformanceModel.OpClassifier):
    @property
    def name(self):
        return "OpBound"

    @staticmethod
    def _accumulate_dominant_components(
        breakdown: Dict[str, float],
        dominant_bound: str,
        components: BoundComponents,
    ) -> None:
        if dominant_bound == MEMORY_BOUND:
            breakdown[MEMORY_BOUND] += components.memory_time_s
        elif dominant_bound == COMMUNICATION_BOUND:
            breakdown[COMMUNICATION_BOUND] += components.communication_time_s
        elif dominant_bound in (COMPUTE_BOUND_MMA, COMPUTE_BOUND_GP):
            # Keep the compute split visible even when either compute type is dominant.
            breakdown[COMPUTE_BOUND_MMA] += components.mma_ops_time_s
            breakdown[COMPUTE_BOUND_GP] += components.gp_ops_time_s
        elif dominant_bound == UNKNOWN_BOUND:
            return
        else:
            logger.warning("Unrecognized dominant bound: %s", dominant_bound)

    def classify(self, event_list: List[Tuple[OpInvokeInfo, "PerformanceModel.Result"]]) -> Dict[str, float]:
        breakdown: Dict[str, float] = {
            MEMORY_BOUND: 0,
            COMMUNICATION_BOUND: 0,
            COMPUTE_BOUND_MMA: 0,
            COMPUTE_BOUND_GP: 0,
        }
        for _, result in event_list:
            dominant_bound = BoundAnalyzer.dominant(result)
            components = BoundAnalyzer.components(result)
            self._accumulate_dominant_components(breakdown, dominant_bound, components)
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

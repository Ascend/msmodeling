import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, Tuple

from .. import ops  # noqa: F401
from ..device import DeviceProfile
from .op_invoke_info import OpInvokeInfo


class PerformanceModel(ABC):
    """
    Performance model used to estimate the execution time of op invocations
    on a given device.
    """

    @dataclasses.dataclass
    class Result:
        execution_time_s: float
        statistics: Dict[str, Any] = dataclasses.field(default_factory=dict)
        """Misc runtime statistics produced by implementation"""

        def combine(self, other: "PerformanceModel.Result", method: str = "max"):
            if method == "max":
                self.execution_time_s = max(self.execution_time_s, other.execution_time_s)
            elif method == "sum":
                self.execution_time_s += other.execution_time_s
            else:
                raise ValueError(f"Unsupported method {method} for combining performance result")
            self.statistics.update(other.statistics)

    class OpClassifier(Protocol):
        @property
        def name(self): ...

        def classify(self, event_list: List[Tuple[OpInvokeInfo, "PerformanceModel.Result"]]) -> Dict[str, float]:
            """
            Classify an event list into a breakdown.

            [NOTE: Breakdown from Op Classifier] The semantics of the values are defined by the performance
            models but they should account for a breakdown of sum(values) so that the caller can then compute the
            percentage of each category according to the values.

            :param event_list: Event list of classify
            :return: category name -> value
            """
            ...

    def __init__(self, name, device_profile: DeviceProfile):
        self.name = name
        self.device_profile = device_profile

    @abstractmethod
    def process_op(self, op_invoke_info: OpInvokeInfo) -> "PerformanceModel.Result":
        """
        Estimate the execution time of an op invocation on the given device.
        Returns:
            op execution time in seconds and misc runtime statistics
        """

    def get_classifiers(self) -> List[OpClassifier]:
        return []


class CachingPerformanceModel(PerformanceModel):
    """
    A performance model that caches the results of another performance model.
    """

    def __init__(self, base_model: PerformanceModel):
        super().__init__(base_model.name, base_model.device_profile)
        self._base_model = base_model
        self._cache: Dict[str, PerformanceModel.Result] = {}

    def process_op(self, op_invoke_info: OpInvokeInfo) -> "PerformanceModel.Result":
        if op_invoke_info.cache_key in self._cache:
            return self._cache[op_invoke_info.cache_key]
        result = self._base_model.process_op(op_invoke_info)
        self._cache[op_invoke_info.cache_key] = result
        return result

    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self._base_model.get_classifiers()

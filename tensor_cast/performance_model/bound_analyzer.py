import dataclasses
from typing import Any, Dict

try:
    # Native in Python 3.11+
    from enum import StrEnum
except ImportError:
    # Fallback for Python 3.10
    from strenum import StrEnum

from .base import PerformanceModel


MEMORY_BOUND = "memory_bound"
COMMUNICATION_BOUND = "communication_bound"
COMPUTE_BOUND_MMA = "compute_bound_mma"
COMPUTE_BOUND_GP = "compute_bound_gp"
UNKNOWN_BOUND = "unknown_bound"


class StatsKey(StrEnum):
    COMPUTE = "compute_time_s"
    MMA_OPS = "mma_ops_time_s"
    GP_OPS = "gp_ops_time_s"
    MEMORY_ACCESS = "memory_access_time_s"
    COMMUNICATION = "comm_time_s"


@dataclasses.dataclass(frozen=True)
class BoundComponents:
    memory_time_s: float = 0.0
    communication_time_s: float = 0.0
    mma_ops_time_s: float = 0.0
    gp_ops_time_s: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            MEMORY_BOUND: self.memory_time_s,
            COMMUNICATION_BOUND: self.communication_time_s,
            COMPUTE_BOUND_MMA: self.mma_ops_time_s,
            COMPUTE_BOUND_GP: self.gp_ops_time_s,
        }


class BoundAnalyzer:
    @classmethod
    def _numeric_value(cls, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    @classmethod
    def _stat_value(cls, stats: Dict[Any, Any], key: StatsKey) -> float:
        if not isinstance(stats, dict):
            return 0.0
        return cls._numeric_value(stats.get(key, stats.get(str(key), 0.0)))

    @classmethod
    def _collect_stat_by_suffix(cls, stats: Dict[Any, Any], suffix: StatsKey) -> float:
        if not isinstance(stats, dict):
            return 0.0

        total = 0.0
        for key, value in stats.items():
            if isinstance(value, dict):
                total += cls._collect_stat_by_suffix(value, suffix)
            elif key == suffix or str(key) == str(suffix) or str(key).endswith(f".{suffix}"):
                total += cls._numeric_value(value)
        return total

    @classmethod
    def _resolved_stat(cls, stats: Dict[Any, Any], key: StatsKey) -> float:
        value = cls._stat_value(stats, key)
        if value != 0:
            return value
        return cls._collect_stat_by_suffix(stats, key)

    @classmethod
    def components(cls, result: PerformanceModel.Result) -> BoundComponents:
        stats = result.statistics if isinstance(getattr(result, "statistics", None), dict) else {}

        memory_time_s = cls._resolved_stat(stats, StatsKey.MEMORY_ACCESS)
        communication_time_s = cls._resolved_stat(stats, StatsKey.COMMUNICATION)
        compute_mma_time_s = cls._resolved_stat(stats, StatsKey.MMA_OPS)
        compute_gp_time_s = cls._resolved_stat(stats, StatsKey.GP_OPS)

        if compute_mma_time_s == 0 and compute_gp_time_s == 0:
            # Fallback for legacy/fused estimators without MMA/GP split.
            # Prefix-tagged child stats (for example "matmul.mma_ops_time_s") are
            # already resolved above. For aggregate-only GMM-heavy paths such as
            # tensor_cast.dispatch_ffn_combine*, keep compatibility by assigning
            # compute_time_s to MMA.
            compute_time_s = cls._resolved_stat(stats, StatsKey.COMPUTE)
            if compute_time_s > 0:
                compute_mma_time_s = compute_time_s

        return BoundComponents(
            memory_time_s=memory_time_s,
            communication_time_s=communication_time_s,
            mma_ops_time_s=compute_mma_time_s,
            gp_ops_time_s=compute_gp_time_s,
        )

    @classmethod
    def dominant(cls, result: PerformanceModel.Result) -> str:
        stats = result.statistics if isinstance(getattr(result, "statistics", None), dict) else {}
        memory_time_s = cls._resolved_stat(stats, StatsKey.MEMORY_ACCESS)
        communication_time_s = cls._resolved_stat(stats, StatsKey.COMMUNICATION)
        compute_time_s = cls._resolved_stat(stats, StatsKey.COMPUTE)
        components = cls.components(result)
        if compute_time_s == 0:
            compute_time_s = components.mma_ops_time_s + components.gp_ops_time_s

        top_level_times = [memory_time_s, communication_time_s, compute_time_s]
        max_value = max(top_level_times)
        if max_value <= 0:
            return UNKNOWN_BOUND

        max_index = top_level_times.index(max_value)
        if max_index == 0:
            return MEMORY_BOUND
        if max_index == 1:
            return COMMUNICATION_BOUND

        if components.mma_ops_time_s <= 0 and components.gp_ops_time_s <= 0:
            return COMPUTE_BOUND_MMA
        if components.mma_ops_time_s >= components.gp_ops_time_s:
            return COMPUTE_BOUND_MMA
        return COMPUTE_BOUND_GP

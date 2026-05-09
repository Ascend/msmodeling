import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo


class QuerySource(Enum):
    MEASURED = auto()
    INTERPOLATED = auto()
    EXTRAPOLATED = auto()
    # Forward-declared: returned by _lookup_composite_decomposed when some
    # (but not all) sub-kernels hit, enabling partial composite estimation.
    PARTIAL = auto()


@dataclass
class QueryResult:
    latency_us: float
    confidence: float
    source: QuerySource
    details: Dict[str, Any] = field(default_factory=dict)
    shape_match_info: Optional["ShapeMatchInfo"] = None
    sub_kernel_shapes: Optional[List["SubKernelShapeInfo"]] = None

    def shape_debug_statistics(self) -> dict:
        """Serialize shape debug info into statistics dict entries.

        Uses isinstance checks (not 'is not None') so that Mock objects in tests
        do not accidentally trigger iteration and raise TypeError.
        """
        out: dict = {}
        if isinstance(self.sub_kernel_shapes, list):
            out["sub_kernel_shapes"] = json.dumps(
                [
                    {
                        "kernel_type": sk.kernel_type,
                        "simulation_shapes": str(sk.simulation_shapes),
                        "kernel_shapes": str(sk.kernel_shapes),
                        "shape_match_rule": sk.shape_match_rule,
                    }
                    for sk in self.sub_kernel_shapes
                ]
            )
        elif isinstance(self.shape_match_info, ShapeMatchInfo):
            info = self.shape_match_info
            out["kernel_shapes"] = str(info.kernel_shapes) if info.kernel_shapes else ""
            out["shape_match_rule"] = info.shape_match_rule
        return out


@dataclass
class ShapeMatchInfo:
    """Shape debug info for a single profiling lookup."""

    simulation_shapes: List[List[int]]  # TC dispatch shapes
    kernel_shapes: List[List[int]]  # Matched CSV shapes ([] on MISS)
    shape_match_rule: str  # Rule name or miss reason


@dataclass
class SubKernelShapeInfo:
    """Shape debug info for one sub-kernel inside a composite op."""

    kernel_type: str
    simulation_shapes: List[List[int]]
    kernel_shapes: List[List[int]]
    shape_match_rule: str


class DataSourcePerformanceModel(ABC):
    """Abstract base class for performance data sources.
    TensorCast queries via OpInvokeInfo only, unaware of underlying data format."""

    @abstractmethod
    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """Query operator performance from OpInvokeInfo."""
        ...

    def store(self, op_invoke_info: "OpInvokeInfo", result: QueryResult) -> None:
        """Store performance data (optional). Default: read-only."""
        raise NotImplementedError("This data source is read-only")

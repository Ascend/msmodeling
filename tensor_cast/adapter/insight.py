import csv
import dataclasses
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


_RUNTIME_PREFIXES = (
    "CAPTURE_",
    "EVENT_",
    "MEM_",
    "MEMCPY_",
    "NOP",
    "NOTIFY_",
)


@dataclasses.dataclass(frozen=True)
class ObservedKernel:
    name: str
    normalized_name: str
    category: str
    wall_duration_ms: float
    self_time_ms: float
    average_wall_duration_ms: float
    max_wall_duration_ms: float
    min_wall_duration_ms: float
    occurrences: int

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "ObservedKernel":
        name = _get_required(row, "Name")
        normalized = normalize_kernel_name(name)
        return cls(
            name=name,
            normalized_name=normalized,
            category=classify_kernel(normalized),
            wall_duration_ms=_to_float(row.get("Wall Duration(ms)")),
            self_time_ms=_to_float(row.get("Self Time(ms)")),
            average_wall_duration_ms=_to_float(row.get("Average Wall Duration(ms)")),
            max_wall_duration_ms=_to_float(row.get("Max Wall Duration(ms)")),
            min_wall_duration_ms=_to_float(row.get("Min Wall Duration(ms)")),
            occurrences=_to_int(row.get("Occurrences")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class InsightTotals:
    wall_duration_ms: float
    self_time_ms: float
    average_wall_duration_ms: float
    max_wall_duration_ms: float
    min_wall_duration_ms: float
    occurrences: int

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "InsightTotals":
        return cls(
            wall_duration_ms=_to_float(row.get("Wall Duration(ms)")),
            self_time_ms=_to_float(row.get("Self Time(ms)")),
            average_wall_duration_ms=_to_float(row.get("Average Wall Duration(ms)")),
            max_wall_duration_ms=_to_float(row.get("Max Wall Duration(ms)")),
            min_wall_duration_ms=_to_float(row.get("Min Wall Duration(ms)")),
            occurrences=_to_int(row.get("Occurrences")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RawInsightSummary:
    kernels: List[ObservedKernel]
    totals: InsightTotals
    source_path: Optional[str] = None

    @property
    def total_wall_duration_ms(self) -> float:
        return self.totals.wall_duration_ms

    def top_kernels(self, limit: int = 20) -> List[ObservedKernel]:
        return sorted(self.kernels, key=lambda item: item.wall_duration_ms, reverse=True)[:limit]

    def to_dict(self, top_n: Optional[int] = None) -> Dict[str, Any]:
        kernels = self.kernels if top_n is None else self.top_kernels(top_n)
        return {
            "source_path": self.source_path,
            "totals": self.totals.to_dict(),
            "total_wall_duration_ms": self.total_wall_duration_ms,
            "kernels": [kernel.to_dict() for kernel in kernels],
        }


def _get_required(row: Dict[str, str], key: str) -> str:
    value = row.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"Raw Insight row is missing required column {key!r}.")
    return str(value).strip()


def _to_float(value: Optional[str]) -> float:
    if value is None or str(value).strip() == "":
        return 0.0
    return float(str(value).strip())


def _to_int(value: Optional[str]) -> int:
    if value is None or str(value).strip() == "":
        return 0
    return int(float(str(value).strip()))


def normalize_kernel_name(name: str) -> str:
    value = name.strip()
    if any(value.startswith(prefix) for prefix in _RUNTIME_PREFIXES):
        return value
    if value.startswith("_"):
        return re.sub(r"_[0-9]+$", "", value)
    first = value.split("_", maxsplit=1)[0]
    return first or value


def classify_kernel(normalized_name: str) -> str:
    lowered = normalized_name.lower()
    if any(normalized_name.startswith(prefix) for prefix in _RUNTIME_PREFIXES):
        return "runtime_overhead"
    if "allreduce" in lowered or "allgather" in lowered or "alltoall" in lowered or lowered.startswith("hcom"):
        return "communication"
    if "attention" in lowered or "inferattentionscore" in lowered:
        return "attention"
    if "moe" in lowered or "dispatchffncombine" in lowered or "gatingtopk" in lowered:
        return "moe"
    if "matmul" in lowered or "batchmatmul" in lowered:
        return "matmul"
    if "quant" in lowered:
        return "quant"
    if "norm" in lowered:
        return "norm"
    if lowered.startswith("cast"):
        return "cast"
    return "other"


def _sniff_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,")
    except csv.Error:
        return csv.excel_tab


def load_raw_insight(path: Union[str, Path]) -> RawInsightSummary:
    insight_path = Path(path)
    content = insight_path.read_text(encoding="utf-8-sig")
    if not content.strip():
        raise ValueError(f"Raw Insight file {insight_path} is empty.")
    dialect = _sniff_dialect(content[:4096])
    reader = csv.DictReader(content.splitlines(), dialect=dialect)
    kernels: List[ObservedKernel] = []
    totals: Optional[InsightTotals] = None
    saw_data_row = False
    for row in reader:
        if not any(row.values()):
            continue
        name = _get_required(row, "Name")
        if not saw_data_row:
            saw_data_row = True
            if name != "Totals":
                raise ValueError(
                    f"Raw Insight file {insight_path} line {reader.line_num}: "
                    "'Totals' row must immediately follow the header."
                )
            totals = InsightTotals.from_row(row)
            continue
        if name == "Totals":
            continue
        kernels.append(ObservedKernel.from_row(row))
    if totals is None:
        raise ValueError(f"Raw Insight file {insight_path} must include a 'Totals' row after the header.")
    return RawInsightSummary(kernels=kernels, totals=totals, source_path=str(insight_path))

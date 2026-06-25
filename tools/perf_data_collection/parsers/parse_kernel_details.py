import argparse
import csv
from dataclasses import dataclass
import math
import statistics
import re
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PERF_DATA_COLLECTION_DIR = CURRENT_DIR if (CURRENT_DIR / "op_replay").is_dir() else CURRENT_DIR.parent
OP_REPLAY_DIR = PERF_DATA_COLLECTION_DIR / "op_replay"
if str(OP_REPLAY_DIR) not in sys.path:
    sys.path.insert(0, str(OP_REPLAY_DIR))
if str(PERF_DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DATA_COLLECTION_DIR))

try:
    from common import (
        DEFAULT_DEVICE,
        SUPPORTED_DEVICES,
        check_version,
        get_target_data_dir,
        normalize_device_name,
    )
except ModuleNotFoundError:
    from .op_replay.common import (
        DEFAULT_DEVICE,
        SUPPORTED_DEVICES,
        check_version,
        get_target_data_dir,
        normalize_device_name,
    )

try:
    from fia_common import parse_shape_or_none, shape_to_text, split_metadata_field
except ModuleNotFoundError:
    from .fia_common import parse_shape_or_none, shape_to_text, split_metadata_field

INPUT_SHAPES = "Input Shapes"
INPUT_DTYPES = "Input Data Types"
INPUT_FORMATS = "Input Formats"
OUTPUT_SHAPES = "Output Shapes"
OUTPUT_DTYPES = "Output Data Types"
OUTPUT_FORMATS = "Output Formats"
TYPE_COL = "Type"
OP_STATE = "OP State"
ACCELERATOR_CORE = "Accelerator Core"
DURATION_US = "Duration(us)"
AVG_DURATION_US = "Profiling Average Duration(us)"
STD_DURATION_US = "Profiling Std Duration(us)"
MEDIAN_DURATION_US = "Profiling Median Duration(us)"
EXTRA_NUMERIC_COLUMNS = [
    "aicore_time(us)",
    "aic_total_cycles",
    "aic_mac_time(us)",
    "aic_mac_ratio",
    "aic_scalar_time(us)",
    "aic_scalar_ratio",
    "aic_mte1_time(us)",
    "aic_mte1_ratio",
    "aic_mte2_time(us)",
    "aic_mte2_ratio",
    "aic_fixpipe_time(us)",
    "aic_fixpipe_ratio",
    "aic_icache_miss_rate",
    "aiv_time(us)",
    "aiv_total_cycles",
    "aiv_vec_time(us)",
    "aiv_vec_ratio",
    "aiv_scalar_time(us)",
    "aiv_scalar_ratio",
    "aiv_mte2_time(us)",
    "aiv_mte2_ratio",
    "aiv_mte3_time(us)",
    "aiv_mte3_ratio",
    "aiv_icache_miss_rate",
    "cube_utilization(%)",
]
FIA_OP_TYPE = "FusedInferAttentionScore"
FIA_RUNTIME_COLUMNS = [
    "Runtime source_profile",
    "Runtime actual_seq_lengths_shape",
    "Runtime actual_seq_lengths_values",
    "Runtime actual_seq_lengths_kv_shape",
    "Runtime actual_seq_lengths_kv_values",
    "Runtime avg_seq_len",
    "Runtime block_table_shape",
    "Runtime block_table_valid_blocks",
    "Runtime num_heads",
    "Runtime num_key_value_heads",
    "Runtime sparse_mode",
    "Runtime input_layout",
    "Runtime block_size",
    "Runtime attn_state",
    "Runtime kv_cache_mode",
    "Runtime metadata_completeness",
]


@dataclass
class ProfilingBundle:
    # operator_details_files and trace_view_files are retained for FIA profiling
    # bundle inspection and future metadata extraction steps.
    root_dir: Path
    kernel_details_files: List[Path]
    operator_details_files: List[Path]
    trace_view_files: List[Path]


@dataclass
class FiaRuntimeMetadata:
    """Profile-derived FIA metadata.

    Fields without shape data are intentionally left empty here.
    """

    source_profile: str
    actual_seq_lengths_shape: str
    actual_seq_lengths_values: str
    actual_seq_lengths_kv_shape: str
    actual_seq_lengths_kv_values: str
    avg_seq_len: str
    block_table_shape: str
    block_table_valid_blocks: str
    num_heads: str
    num_key_value_heads: str
    sparse_mode: str
    input_layout: str
    block_size: str
    attn_state: str
    kv_cache_mode: str
    metadata_completeness: str


def _render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return f"[{'-' * width}]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = min(width, int(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def print_progress(*, stage: str, current: int, total: int, detail: str = "") -> None:
    bar = _render_progress(current, total)
    message = f"\r{stage} {bar} {current}/{total}"
    if detail:
        message += f" | {detail}"
    print(message, end="", flush=True)


def clear_progress() -> None:
    print("\r" + " " * 160 + "\r", end="", flush=True)


def profiling_column_name(column: str) -> str:
    return f"Profiling {column}"


def infer_avg_seq_len(actual_seq_lengths_kv_values: str) -> str:
    cleaned = (actual_seq_lengths_kv_values or "").strip()
    if not cleaned:
        return ""
    parts = [item.strip() for item in re.split(r"[;,]", cleaned) if item.strip()]
    if not parts:
        return ""
    try:
        values = [int(item) for item in parts]
    except ValueError:
        return ""
    return f"{statistics.mean(values):.6f}"


def extract_fia_profile_metadata(
    input_shapes_text: str,
    source_profile: str,
) -> FiaRuntimeMetadata:
    input_shapes = [
        parse_shape_or_none(item) for item in split_metadata_field(input_shapes_text)
    ]
    while len(input_shapes) < 31:
        input_shapes.append(None)

    actual_seq_lengths_shape = input_shapes[5]
    actual_seq_lengths_kv_shape = input_shapes[6]
    block_table_shape = input_shapes[14]

    return FiaRuntimeMetadata(
        source_profile=source_profile,
        actual_seq_lengths_shape=shape_to_text(actual_seq_lengths_shape),
        actual_seq_lengths_values="",
        actual_seq_lengths_kv_shape=shape_to_text(actual_seq_lengths_kv_shape),
        actual_seq_lengths_kv_values="",
        avg_seq_len="",
        block_table_shape=shape_to_text(block_table_shape),
        block_table_valid_blocks="",
        num_heads="",
        num_key_value_heads="",
        sparse_mode="",
        input_layout="",
        block_size="",
        attn_state="",
        kv_cache_mode="",
        metadata_completeness="profile_shapes_only",
    )


class KernelDetailsParser:
    """Parse profiling CSVs and export aggregated operator stats."""

    def __init__(
        self,
        device: str,
        kernel_details_path: str,
        vllm_ascend_version: str | None = None,
        *,
        database_path: str | Path | None = None,
        torch_version: str | None = None,
        cann_version: str | None = None,
    ):
        self.device = device
        self.kernel_details_path = Path(kernel_details_path)
        self.device_dir = normalize_device_name(device)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.output_dir = get_target_data_dir(
            device=device,
            vllm_ascend_version=vllm_ascend_version,
            database_path=database_path,
            torch_version=torch_version,
            cann_version=cann_version,
        )
        self.bundle = self._resolve_profiling_bundle()

    def _resolve_profiling_bundle(self) -> ProfilingBundle:
        if not self.kernel_details_path.exists():
            raise FileNotFoundError(
                f"kernel_details source path not found: {self.kernel_details_path}"
            )
        if self.kernel_details_path.is_file():
            return ProfilingBundle(
                root_dir=self.kernel_details_path.parent,
                kernel_details_files=[self.kernel_details_path],
                operator_details_files=[],
                trace_view_files=[],
            )

        kernel_details_files = sorted(
            path
            for path in self.kernel_details_path.rglob("*.csv")
            if "kernel_details" in path.stem.lower()
        )
        if not kernel_details_files:
            raise FileNotFoundError(
                "No CSV files with 'kernel_details' in the filename found under: "
                f"{self.kernel_details_path}"
            )
        operator_details_files = sorted(
            path for path in self.kernel_details_path.rglob("operator_details.csv")
        )
        trace_view_files = sorted(
            path for path in self.kernel_details_path.rglob("trace_view.json")
        )
        return ProfilingBundle(
            root_dir=self.kernel_details_path,
            kernel_details_files=kernel_details_files,
            operator_details_files=operator_details_files,
            trace_view_files=trace_view_files,
        )

    @staticmethod
    def _parse_duration(value: str) -> float:
        try:
            return float((value or "").strip())
        except ValueError:
            return 0.0

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name.strip())
        return sanitized or "UNKNOWN_TYPE"

    @staticmethod
    def _safe_cell(row: Dict[str, str], key: str) -> str:
        return (row.get(key, "") or "").strip()

    @staticmethod
    def _is_na_shape(value: str) -> bool:
        normalized = value.strip().strip('"').upper()
        return normalized == "N/A"

    # Kernel name normalization: map variant names to canonical kernel type.
    # - split_qkv_rmsnorm_rope_kernel_0: Triton JIT grid-config variant of
    #   split_qkv_rmsnorm_rope_kernel (same CANN kernel, different launch config
    #   for decode vs prefill). Merge into one CSV.
    _KERNEL_NAME_NORMALIZE: Dict[str, str] = {
        "split_qkv_rmsnorm_rope_kernel_0": "split_qkv_rmsnorm_rope_kernel",
        "muls_add_kernel_1": "muls_add_kernel",
    }

    @classmethod
    def _normalize_kernel_type(cls, op_type: str) -> str:
        return cls._KERNEL_NAME_NORMALIZE.get(op_type, op_type)

    def _load_rows(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        kernel_details_files = self.bundle.kernel_details_files
        required_columns = {
            TYPE_COL,
            OP_STATE,
            ACCELERATOR_CORE,
            INPUT_SHAPES,
            INPUT_DTYPES,
            INPUT_FORMATS,
            OUTPUT_SHAPES,
            OUTPUT_DTYPES,
            OUTPUT_FORMATS,
            DURATION_US,
        }
        required_columns.update(EXTRA_NUMERIC_COLUMNS)

        total_files = len(kernel_details_files)
        for file_index, kernel_details_file in enumerate(kernel_details_files, start=1):
            with kernel_details_file.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                missing = required_columns - set(reader.fieldnames or [])
                if missing:
                    missing_str = ", ".join(sorted(missing))
                    raise ValueError(
                        f"{kernel_details_file} is missing required columns: "
                        f"{missing_str}"
                    )
                rows.extend(reader)
            print_progress(
                stage="Load",
                current=file_index,
                total=total_files,
                detail=kernel_details_file.parent.name,
            )
        return rows

    @staticmethod
    def _shape_key(row: Dict[str, object]) -> Tuple[str, str]:
        return (str(row.get(INPUT_SHAPES, "")), str(row.get(OUTPUT_SHAPES, "")))

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (value or "").lower())

    @classmethod
    def _is_fia_operator_row(cls, row: Dict[str, str]) -> bool:
        candidates = [
            row.get("Type", ""),
            row.get("Name", ""),
            row.get("Op Type", ""),
            row.get("Op Name", ""),
            row.get("Operator Type", ""),
            row.get("Operator Name", ""),
        ]
        normalized = " ".join(
            cls._normalize_text(value) for value in candidates if value
        )
        return "fusedinferattentionscore" in normalized

    def _load_fia_operator_rows_by_profile(self) -> dict[Path, list[dict[str, str]]]:
        rows_by_profile: dict[Path, list[dict[str, str]]] = {}
        for csv_path in self.bundle.operator_details_files:
            fia_rows: list[dict[str, str]] = []
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if not self._is_fia_operator_row(row):
                        continue
                    input_shapes = self._safe_cell(row, INPUT_SHAPES)
                    output_shapes = self._safe_cell(row, OUTPUT_SHAPES)
                    if not input_shapes or not output_shapes:
                        continue
                    fia_rows.append(row)
            if fia_rows:
                rows_by_profile[csv_path.parent] = fia_rows
        return rows_by_profile

    def _build_fia_runtime_index(self) -> dict[tuple[str, str], FiaRuntimeMetadata]:
        runtime_index: dict[tuple[str, str], FiaRuntimeMetadata] = {}
        operator_rows_by_profile = self._load_fia_operator_rows_by_profile()
        for csv_path in self.bundle.kernel_details_files:
            kernel_rows: list[dict[str, str]] = []
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    op_type = self._safe_cell(row, TYPE_COL)
                    if op_type != FIA_OP_TYPE:
                        continue
                    kernel_rows.append(row)

            operator_rows = operator_rows_by_profile.get(csv_path.parent, [])
            paired_count = min(len(kernel_rows), len(operator_rows))
            if operator_rows and len(kernel_rows) != len(operator_rows):
                print(
                    "Warning: FIA kernel/operator row count mismatch under "
                    f"{csv_path.parent}: kernel_rows={len(kernel_rows)}, "
                    f"operator_rows={len(operator_rows)}. Pairing uses row "
                    "order for the shared prefix only."
                )

            for index, row in enumerate(kernel_rows):
                input_shapes = self._safe_cell(row, INPUT_SHAPES)
                output_shapes = self._safe_cell(row, OUTPUT_SHAPES)
                key = (input_shapes, output_shapes)
                if key in runtime_index:
                    continue

                operator_input_shapes = ""
                if index < paired_count:
                    operator_row = operator_rows[index]
                    operator_input_shapes = self._safe_cell(operator_row, INPUT_SHAPES)

                metadata_input_shapes = operator_input_shapes or input_shapes
                runtime_index[key] = extract_fia_profile_metadata(
                    input_shapes_text=metadata_input_shapes,
                    source_profile=csv_path.parent.name,
                )
        return runtime_index

    @staticmethod
    def _compute_fia_metadata_completeness(runtime_metadata: FiaRuntimeMetadata) -> str:
        if (
            runtime_metadata.actual_seq_lengths_values
            or runtime_metadata.actual_seq_lengths_kv_values
        ):
            return "runtime_values"
        return runtime_metadata.metadata_completeness or "profile_shapes_only"

    def parse_and_export(self) -> List[Path]:
        rows = self._load_rows()
        fia_runtime_index = self._build_fia_runtime_index()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        grouped: Dict[Tuple[str, str, str], Dict[str, object]] = (
            defaultdict(  # (type, input_shapes, output_shapes)
                lambda: {
                    "sum_duration": 0.0,
                    "sum_duration_sq": 0.0,
                    "durations": [],
                    "count": 0,
                    "op_state": "",
                    "accelerator_core": "",
                    "input_dtypes": "",
                    "input_formats": "",
                    "output_dtypes": "",
                    "output_formats": "",
                    "sum_extra": dict.fromkeys(EXTRA_NUMERIC_COLUMNS, 0.0),
                }
            )
        )

        total_rows = len(rows)
        progress_interval = max(1, total_rows // 100) if total_rows else 1
        for row_index, row in enumerate(rows, start=1):
            op_type = self._normalize_kernel_type(self._safe_cell(row, TYPE_COL))
            input_shapes = self._safe_cell(row, INPUT_SHAPES)
            output_shapes = self._safe_cell(row, OUTPUT_SHAPES)
            if self._is_na_shape(input_shapes) or self._is_na_shape(output_shapes):
                continue
            key = (op_type, input_shapes, output_shapes)
            item = grouped[key]

            duration = self._parse_duration(self._safe_cell(row, DURATION_US))
            item["sum_duration"] = float(item["sum_duration"]) + duration
            item["sum_duration_sq"] = (
                float(item["sum_duration_sq"]) + duration * duration
            )
            item["durations"].append(duration)
            item["count"] = int(item["count"]) + 1
            for col in EXTRA_NUMERIC_COLUMNS:
                item["sum_extra"][col] = float(item["sum_extra"][col]) + (
                    self._parse_duration(self._safe_cell(row, col))
                )

            # Keep the first non-empty meta fields for this shape pair.
            if not item["op_state"]:
                item["op_state"] = self._safe_cell(row, OP_STATE)
            if not item["accelerator_core"]:
                item["accelerator_core"] = self._safe_cell(row, ACCELERATOR_CORE)
            if not item["input_dtypes"]:
                item["input_dtypes"] = self._safe_cell(row, INPUT_DTYPES)
            if not item["input_formats"]:
                item["input_formats"] = self._safe_cell(row, INPUT_FORMATS)
            if not item["output_dtypes"]:
                item["output_dtypes"] = self._safe_cell(row, OUTPUT_DTYPES)
            if not item["output_formats"]:
                item["output_formats"] = self._safe_cell(row, OUTPUT_FORMATS)
            if row_index == total_rows or row_index % progress_interval == 0:
                print_progress(
                    stage="Aggregate",
                    current=row_index,
                    total=total_rows,
                    detail=op_type,
                )

        rows_by_type: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for (op_type, input_shapes, output_shapes), item in grouped.items():
            if not op_type:
                continue
            count = int(item["count"])
            avg_duration = float(item["sum_duration"]) / count
            avg_duration_sq = float(item["sum_duration_sq"]) / count
            variance = max(0.0, avg_duration_sq - avg_duration * avg_duration)
            std_duration = math.sqrt(variance)
            median_duration = statistics.median(item["durations"])
            avg_extra = {
                profiling_column_name(f"Average {col}"): (
                    float(item["sum_extra"][col]) / count
                )
                for col in EXTRA_NUMERIC_COLUMNS
            }
            rows_by_type[op_type].append(
                {
                    OP_STATE: item["op_state"],
                    ACCELERATOR_CORE: item["accelerator_core"],
                    INPUT_SHAPES: input_shapes,
                    INPUT_DTYPES: item["input_dtypes"],
                    INPUT_FORMATS: item["input_formats"],
                    OUTPUT_SHAPES: output_shapes,
                    OUTPUT_DTYPES: item["output_dtypes"],
                    OUTPUT_FORMATS: item["output_formats"],
                    AVG_DURATION_US: f"{avg_duration:.6f}",
                    MEDIAN_DURATION_US: f"{median_duration:.6f}",
                    STD_DURATION_US: f"{std_duration:.6f}",
                    **{k: f"{v:.6f}" for k, v in avg_extra.items()},
                }
            )
            if op_type == FIA_OP_TYPE:
                runtime_metadata = fia_runtime_index.get((input_shapes, output_shapes))
                if runtime_metadata is not None:
                    runtime_metadata.avg_seq_len = infer_avg_seq_len(
                        runtime_metadata.actual_seq_lengths_kv_values
                    )
                    runtime_metadata.metadata_completeness = (
                        self._compute_fia_metadata_completeness(runtime_metadata)
                    )
                    rows_by_type[op_type][-1].update(
                        {
                            "Runtime source_profile": runtime_metadata.source_profile,
                            "Runtime actual_seq_lengths_shape": (
                                runtime_metadata.actual_seq_lengths_shape
                            ),
                            "Runtime actual_seq_lengths_values": (
                                runtime_metadata.actual_seq_lengths_values
                            ),
                            "Runtime actual_seq_lengths_kv_shape": (
                                runtime_metadata.actual_seq_lengths_kv_shape
                            ),
                            "Runtime actual_seq_lengths_kv_values": (
                                runtime_metadata.actual_seq_lengths_kv_values
                            ),
                            "Runtime avg_seq_len": runtime_metadata.avg_seq_len,
                            "Runtime block_table_shape": (
                                runtime_metadata.block_table_shape
                            ),
                            "Runtime block_table_valid_blocks": (
                                runtime_metadata.block_table_valid_blocks
                            ),
                            "Runtime num_heads": runtime_metadata.num_heads,
                            "Runtime num_key_value_heads": (
                                runtime_metadata.num_key_value_heads
                            ),
                            "Runtime sparse_mode": runtime_metadata.sparse_mode,
                            "Runtime input_layout": runtime_metadata.input_layout,
                            "Runtime block_size": runtime_metadata.block_size,
                            "Runtime attn_state": runtime_metadata.attn_state,
                            "Runtime kv_cache_mode": runtime_metadata.kv_cache_mode,
                            "Runtime metadata_completeness": (
                                runtime_metadata.metadata_completeness
                            ),
                        }
                    )

        output_files: List[Path] = []
        base_ordered_columns = [
            OP_STATE,
            ACCELERATOR_CORE,
            INPUT_SHAPES,
            INPUT_DTYPES,
            INPUT_FORMATS,
            OUTPUT_SHAPES,
            OUTPUT_DTYPES,
            OUTPUT_FORMATS,
            AVG_DURATION_US,
            MEDIAN_DURATION_US,
            STD_DURATION_US,
        ]
        base_ordered_columns.extend(
            [profiling_column_name(f"Average {col}") for col in EXTRA_NUMERIC_COLUMNS]
        )
        total_output_files = len(rows_by_type)
        for file_index, (op_type, type_rows) in enumerate(
            rows_by_type.items(), start=1
        ):
            output_path = self.output_dir / f"{self._sanitize_filename(op_type)}.csv"
            ordered_columns = list(base_ordered_columns)
            if op_type == FIA_OP_TYPE:
                ordered_columns.extend(FIA_RUNTIME_COLUMNS)
            normalized_rows = []
            for row in type_rows:
                normalized_rows.append(
                    {col: row.get(col, "") for col in ordered_columns}
                )

            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ordered_columns)
                writer.writeheader()
                writer.writerows(normalized_rows)
            output_files.append(output_path)
            print_progress(
                stage="Write",
                current=file_index,
                total=total_output_files,
                detail=output_path.name,
            )

        clear_progress()
        return sorted(output_files)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse profiling data and split it into operator CSV files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--profiling-path",
        required=True,
        help=(
            "[Required] Path to a kernel_details*.csv file or a profiling directory. "
            "When a directory is provided, the script recursively scans all CSV files whose filename "
            "contains 'kernel_details'."
        ),
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help="[Optional] Explicit database directory to write generated CSV files into.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=SUPPORTED_DEVICES,
        help=(
            "[Optional] Device folder name used when inferring the output path. "
            f"Default: {DEFAULT_DEVICE}."
        ),
    )
    parser.add_argument(
        "--vllm-version",
        dest="vllm_version",
        type=check_version,
        help="[Optional] vLLM version, e.g. 0.9.2, or a full version-dir name.",
    )
    parser.add_argument(
        "--torch-version",
        type=check_version,
        help="[Optional] PyTorch version used to build the version-dir name, e.g. 2.9.0.",
    )
    parser.add_argument(
        "--cann-version",
        type=check_version,
        help="[Optional] CANN version used to build the version-dir name, e.g. 8.5.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    parser = KernelDetailsParser(
        device=args.device,
        kernel_details_path=args.profiling_path,
        vllm_ascend_version=args.vllm_version,
        database_path=args.database_path,
        torch_version=args.torch_version,
        cann_version=args.cann_version,
    )
    output_files = parser.parse_and_export()
    print(
        f"Generated {len(output_files)} csv file(s) under "
        f"{parser.output_dir.as_posix()} "
        f"from {args.profiling_path}"
    )


if __name__ == "__main__":
    main()

# Backward-compatible alias for external imports from older naming.
AscendProfilerParser = KernelDetailsParser

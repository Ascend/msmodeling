from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from importlib import import_module, metadata
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch_npu

FRACTAL_NZ_FORMAT_ID = 29
# common.py ?op_replay/ [0] ?perf_data_collection/ [1] ?tools/ [2] ?repo_root [3]
DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "tensor_cast"
    / "performance_model"
    / "profiling_database"
    / "data"
)
SUPPORTED_DEVICES = [
    "TEST_DEVICE",
    "ATLAS_800_A2_376T_64G",
    "ATLAS_800_A2_313T_64G",
    "ATLAS_800_A2_280T_64G",
    "ATLAS_800_A2_280T_64G_PCIE",
    "ATLAS_800_A2_280T_32G_PCIE",
    "ATLAS_800_A3_752T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE",
]
DEFAULT_DEVICE = "ATLAS_800_A3_752T_128G_DIE"
DEFAULT_REPLAY_REPEAT_COUNT = 30
REPLAY_REPEAT_COUNT_ENV = "MSMODELING_OP_REPLAY_REPEAT_COUNT"
VERSION_DIR_PATTERN = re.compile(r"^vllm.+_torch.+_cann.+$")
DEFAULT_UPDATE_MODE = "all"
SUPPORTED_UPDATE_MODES = ("all", "missing-only")
MICROBENCH_DURATION = "Average Duration(us)"
PROFILING_AVERAGE_DURATION = "Profiling Average Duration(us)"

torch = None
torch_npu = None
DTYPE_MAP = {}
INVALID_REPLAY_ROWS: list[dict[str, str]] = []


def init_runtime() -> None:
    global torch
    global torch_npu
    global DTYPE_MAP

    if torch is not None and torch_npu is not None and DTYPE_MAP:
        return

    try:
        import torch as torch_module
        import torch_npu as torch_npu_module
    except ImportError as exc:
        raise RuntimeError("NPU not found") from exc

    try:
        from vllm_ascend.utils import enable_custom_op
        enable_custom_op()
    except Exception as exc:
        print(f"Warning: custom op dependencies are unavailable ({exc}). Replay may fail for custom operators.")

    torch = torch_module
    torch_npu = torch_npu_module
    torch_npu.npu.config.allow_internal_format = True
    DTYPE_MAP = {
        "DT_FLOAT": torch.float32,
        "DT_FLOAT16": torch.float16,
        "DT_BF16": torch.bfloat16,
        "DT_DOUBLE": torch.float64,
        "DT_INT8": torch.int8,
        "DT_UINT8": torch.uint8,
        "DT_INT16": torch.int16,
        "DT_INT32": torch.int32,
        "DT_INT64": torch.int64,
        "DT_BOOL": torch.bool,
    }


def get_runtime_modules():
    init_runtime()
    return torch, torch_npu


def check_version(value: str) -> str:
    version = value.strip()
    if not re.fullmatch(r"[0-9A-Za-z]+(?:[._+-][0-9A-Za-z]+)*", version):
        raise argparse.ArgumentTypeError(
            f"Invalid version value: {value!r}. Expected something like 0.9.2, "
            "2.9.0, 8.5, or vllm0.13.0_torch2.8.0_cann8.3."
        )
    return version


def normalize_device_name(device: str) -> str:
    return device.strip()


def normalize_vllm_ascend_version(version: str) -> str:
    return version.strip()


def _normalize_stack_component(prefix: str, version: str) -> str:
    normalized = version.strip()
    lowered = normalized.lower()
    if lowered.startswith(prefix):
        normalized = normalized[len(prefix):]
    elif prefix == "vllm" and lowered.startswith("v"):
        normalized = normalized[1:]
    if prefix == "torch":
        normalized = normalized.split("+", 1)[0]
    return normalized.strip()


def is_version_dir_name(value: str) -> bool:
    return bool(VERSION_DIR_PATTERN.fullmatch((value or "").strip()))


def build_version_dir_name(
    *,
    vllm_ascend_version: str,
    torch_version: str,
    cann_version: str,
) -> str:
    return (
        f"vllm{_normalize_stack_component('vllm', vllm_ascend_version)}"
        f"_torch{_normalize_stack_component('torch', torch_version)}"
        f"_cann{_normalize_stack_component('cann', cann_version)}"
    )


def _load_distribution_version(*distribution_names: str) -> str | None:
    for distribution_name in distribution_names:
        try:
            return metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _load_module_version(*module_names: str) -> str | None:
    for module_name in module_names:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        version = getattr(module, "__version__", "")
        if version:
            return str(version)
    return None


def detect_vllm_ascend_version() -> str | None:
    version = _load_distribution_version("vllm-ascend", "vllm_ascend")
    if version:
        return version
    return _load_module_version("vllm_ascend")


def detect_torch_version() -> str | None:
    version = _load_distribution_version("torch")
    if version:
        return version
    return _load_module_version("torch")


def _extract_version_from_text(raw_text: str) -> str | None:
    patterns = [
        r"(?im)^(?:version|package_version)\s*[=:]\s*([^\s]+)\s*$",
        r"(?im)\b(?:version|package_version)\b\s*[:=]\s*([0-9A-Za-z._+-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text)
        if match:
            return match.group(1).strip()
    return None


def _iter_cann_version_files() -> list[Path]:
    home_dir = Path.home()
    roots: list[Path] = []
    for env_name in (
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_TOOLKIT_HOME_PATH",
        "ASCEND_INSTALL_PATH",
    ):
        raw_value = (os.environ.get(env_name, "") or "").strip()
        if raw_value:
            roots.append(Path(raw_value))

    roots.extend(
        [
            home_dir / "Ascend" / "cann",
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/ascend-toolkit"),
            Path("/usr/local/Ascend/cann"),
            Path("/usr/local/Ascend/latest"),
            Path("/usr/local/Ascend"),
            Path(r"C:\Program Files\Ascend\ascend-toolkit\latest"),
            Path(r"C:\Program Files\Ascend\ascend-toolkit"),
        ]
    )

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "ascend_toolkit_install.info",
                root / "version.info",
                root / "latest" / "version.info",
                root / "latest" / "ascend_toolkit_install.info",
                root / "ascend-toolkit" / "latest" / "version.info",
                root / "ascend-toolkit" / "latest" / "ascend_toolkit_install.info",
                root / "ascend-toolkit" / "version.info",
                root / "ascend-toolkit" / "ascend_toolkit_install.info",
                root / "arm64-linux" / "ascend_toolkit_install.info",
                root / "x86_64-linux" / "ascend_toolkit_install.info",
            ]
        )

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def detect_cann_version() -> str | None:
    try:
        torch_npu_module = import_module("torch_npu")
        version_module = getattr(torch_npu_module, "version", None)
        for candidate in (
            getattr(version_module, "cann", None),
            getattr(version_module, "cann_version", None),
        ):
            if candidate:
                return str(candidate)
    except Exception:
        pass

    for version_file in _iter_cann_version_files():
        if not version_file.exists():
            continue
        try:
            raw_text = version_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        version = _extract_version_from_text(raw_text)
        if version:
            return version
    return None


def detect_runtime_stack_versions() -> tuple[str | None, str | None, str | None]:
    return (
        detect_vllm_ascend_version(),
        detect_torch_version(),
        detect_cann_version(),
    )


def resolve_version_dir_name(
    *,
    vllm_ascend_version: str | None,
    torch_version: str | None,
    cann_version: str | None,
) -> str:
    if (
        vllm_ascend_version is not None
        and torch_version is None
        and cann_version is None
        and is_version_dir_name(vllm_ascend_version)
    ):
        return vllm_ascend_version.strip()

    detected_vllm, detected_torch, detected_cann = detect_runtime_stack_versions()
    resolved_vllm = vllm_ascend_version or detected_vllm
    resolved_torch = torch_version or detected_torch
    resolved_cann = cann_version or detected_cann

    missing_parts: list[str] = []
    if not resolved_vllm:
        missing_parts.append("vLLM-Ascend")
    if not resolved_torch:
        missing_parts.append("PyTorch")
    if not resolved_cann:
        missing_parts.append("CANN")
    if missing_parts:
        missing_text = ", ".join(missing_parts)
        raise RuntimeError(
            "Could not detect runtime stack version(s): "
            f"{missing_text}. Specify --database-path or pass "
            "--vllm-version/--torch-version/--cann-version explicitly."
        )

    return build_version_dir_name(
        vllm_ascend_version=resolved_vllm,
        torch_version=resolved_torch,
        cann_version=resolved_cann,
    )


def normalize_op_name(name: str) -> str:
    normalized = name.strip()
    if normalized.endswith("_run.py"):
        normalized = normalized.removesuffix("_run.py")
    elif normalized.endswith("_run"):
        normalized = normalized.removesuffix("_run")
    elif normalized.endswith(".csv"):
        normalized = normalized.removesuffix(".csv")
    return normalized


def resolve_device_type(runtime_torch) -> str:
    if hasattr(runtime_torch, "npu") and runtime_torch.npu.is_available():
        return "npu"
    if hasattr(runtime_torch, "cuda") and runtime_torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_npu_available() -> None:
    runtime_torch, _ = get_runtime_modules()
    has_npu = hasattr(runtime_torch, "npu") and runtime_torch.npu.is_available()
    if not has_npu:
        raise RuntimeError("NPU not found")


def parse_list_field(raw_value: str) -> list[str]:
    cleaned = raw_value.strip().strip('"')
    return [item.strip() for item in cleaned.split(";") if item.strip()]


def split_metadata_field(raw_value: str) -> list[str]:
    cleaned = raw_value.strip().strip('"')
    return [item.strip() for item in cleaned.split(";")]


def parse_shape(raw_shape: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw_shape.split(",") if part.strip())


def parse_shape_or_none(raw_shape: str):
    if not raw_shape.strip():
        return None
    return parse_shape(raw_shape)


def normalize_dtype_name(dtype_name: str) -> str:
    normalized = (dtype_name or "").strip()
    if not normalized:
        return "DT_UNDEFINED"
    if normalized.startswith("DT_"):
        return normalized
    return f"DT_{normalized}"


def resolve_runtime_dtype(dtype_name: str):
    normalized = normalize_dtype_name(dtype_name)
    dtype = DTYPE_MAP.get(normalized)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtype


def expand_fractal_nz_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != 4:
        raise ValueError(f"Unsupported FRACTAL_NZ shape: {shape}")
    a_dim, b_dim, c_dim, d_dim = shape
    return b_dim * c_dim, a_dim * d_dim


def normalize_shape(shape: tuple[int, ...], input_format: str) -> tuple[int, ...]:
    if input_format == "FRACTAL_NZ":
        return expand_fractal_nz_shape(shape)
    return shape


def build_host_tensor(shape: tuple[int, ...], dtype):
    runtime_torch, _ = get_runtime_modules()

    if dtype == runtime_torch.bool:
        return runtime_torch.randint(0, 2, shape, dtype=runtime_torch.int32).to(runtime_torch.bool)
    if dtype in {
        runtime_torch.float16,
        runtime_torch.bfloat16,
        runtime_torch.float32,
        runtime_torch.float64,
    }:
        return runtime_torch.randn(shape).to(dtype)
    return runtime_torch.randint(0, 8, shape, dtype=dtype)


def maybe_cast_internal_format(tensor, input_format: str):
    _, runtime_torch_npu = get_runtime_modules()
    if input_format == "FRACTAL_NZ":
        return runtime_torch_npu.npu_format_cast(tensor, FRACTAL_NZ_FORMAT_ID)
    return tensor


def build_input_tensor(
    shape: tuple[int, ...],
    input_format: str,
    dtype_name: str,
    transpose: bool = False,
):
    if shape is None:
        raise ValueError(
            f"build_input_tensor received None shape (dtype={dtype_name}, "
            f"format={input_format}). This usually means a theory-generated "
            "CSV row has an empty shape slot with a non-empty dtype inherited "
            "from the template row."
        )
    if any(dim is None for dim in shape):
        raise ValueError(
            f"build_input_tensor received shape with None elements: {shape} "
            f"(dtype={dtype_name}, format={input_format})"
        )
    dtype = resolve_runtime_dtype(dtype_name)

    normalized_shape = normalize_shape(shape, input_format)
    tensor = build_host_tensor(normalized_shape, dtype)
    tensor = tensor.npu()
    if transpose:
        tensor = tensor.t()
    return maybe_cast_internal_format(tensor, input_format)


def build_matmul_case(
    row: dict[str, str],
    *,
    kernel_type: str,
    require_exact_inputs: bool,
):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]

    enough_inputs = len(input_shapes) >= 2 and len(input_formats) >= 2 and len(input_dtypes) >= 2
    if require_exact_inputs:
        if len(input_shapes) != 2 or len(input_formats) != 2 or len(input_dtypes) != 2:
            raise ValueError(f"{kernel_type} expects exactly two inputs")
    elif not enough_inputs:
        raise ValueError(f"{kernel_type} expects at least two inputs")

    return {
        "inputs": [
            build_input_tensor(
                shape=input_shapes[0],
                input_format=input_formats[0],
                dtype_name=input_dtypes[0],
                transpose=False,
            ),
            build_input_tensor(
                shape=input_shapes[1],
                input_format=input_formats[1],
                dtype_name=input_dtypes[1],
                transpose=True,
            ),
        ],
        "kwargs": {},
    }


def get_target_data_dir(
    device: str | None = None,
    vllm_ascend_version: str | None = None,
    *,
    database_path: str | Path | None = None,
    torch_version: str | None = None,
    cann_version: str | None = None,
) -> Path:
    if database_path is not None:
        return Path(database_path)

    resolved_device = normalize_device_name(device or DEFAULT_DEVICE)
    resolved_version_dir = resolve_version_dir_name(
        vllm_ascend_version=vllm_ascend_version,
        torch_version=torch_version,
        cann_version=cann_version,
    )
    return (
        DATA_DIR
        / resolved_device
        / "vllm_ascend"
        / resolved_version_dir
    )


def build_database_cli_args(
    *,
    database_path: str | Path | None = None,
    device: str | None = None,
    vllm_ascend_version: str | None = None,
    torch_version: str | None = None,
    cann_version: str | None = None,
) -> list[str]:
    if database_path is not None:
        return ["--database-path", str(database_path)]

    cli_args: list[str] = ["--device", device or DEFAULT_DEVICE]
    if vllm_ascend_version is not None:
        cli_args += ["--vllm-version", vllm_ascend_version]
    if torch_version is not None:
        cli_args += ["--torch-version", torch_version]
    if cann_version is not None:
        cli_args += ["--cann-version", cann_version]
    return cli_args


def load_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        if fieldnames and fieldnames[0].startswith("version https://git-lfs.github.com/spec/"):
            raise RuntimeError(
                f"{csv_path} is a Git LFS pointer, not the real CSV content. "
                "Run `git lfs pull` in the repository and retry."
            )
        if not fieldnames:
            raise RuntimeError(f"{csv_path} is empty or missing a CSV header.")
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def write_csv_rows(
    csv_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reset_invalid_replay_rows() -> None:
    INVALID_REPLAY_ROWS.clear()


def get_invalid_replay_rows() -> list[dict[str, str]]:
    return list(INVALID_REPLAY_ROWS)


def register_invalid_replay_row(
    *,
    kernel_type: str,
    csv_path: Path,
    row_index: int,
    row: dict[str, str],
    exc: Exception,
) -> dict[str, str]:
    entry = {
        "kernel_type": kernel_type,
        "csv_path": str(csv_path),
        "row_index": str(row_index),
        "input_shapes": row.get("Input Shapes", ""),
        "input_formats": row.get("Input Formats", ""),
        "input_dtypes": row.get("Input Data Types", ""),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    INVALID_REPLAY_ROWS.append(entry)
    return entry


def is_fatal_replay_exception(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    return type(exc).__name__.endswith("OutOfMemoryError")


def parse_float(value: str | None) -> float:
    try:
        return float((value or "").strip())
    except (TypeError, ValueError, AttributeError):
        return 0.0


def has_real_duration(row: dict[str, str], column: str) -> bool:
    return parse_float(row.get(column, "")) > 0.0


def row_has_only_invalid_durations(
    row: dict[str, str],
    *,
    microbench_column: str = MICROBENCH_DURATION,
    profiling_column: str = PROFILING_AVERAGE_DURATION,
) -> bool:
    return not has_real_duration(row, microbench_column) and not has_real_duration(row, profiling_column)


def row_has_valid_duration(row: dict[str, str]) -> bool:
    return not row_has_only_invalid_durations(row)


def csv_has_complete_microbench(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(row_has_valid_duration(row) for row in rows)


def process_replay_csvs(
    *,
    kernel_type: str,
    csv_paths: list[Path],
    repeat_count: int,
    run_row_fn: Callable[[Path, int, dict[str, str]], None],
    update_mode: str = DEFAULT_UPDATE_MODE,
    on_row_finally: Callable[[], None] | None = None,
    can_write_cleanup: Callable[[], bool] | None = None,
    on_cleanup_written: Callable[[], None] | None = None,
) -> tuple[int, list[dict[str, str]], int, int]:
    total_rows = 0
    invalid_rows: list[dict[str, str]] = []
    source_row_count = 0
    skipped_rows = 0

    for csv_path in csv_paths:
        fieldnames, rows = load_csv_rows(csv_path)
        kept_rows: list[dict[str, str]] = []
        deleted_count = 0
        source_row_count += len(rows)

        if update_mode == "missing-only" and csv_has_complete_microbench(rows):
            skipped_rows += len(rows)
            print(
                f"[SKIP] {csv_path} all {len(rows)} row(s) already have "
                f"usable Average/Profiling durations."
            )
            continue

        for row_index, row in enumerate(rows, start=2):
            if update_mode == "missing-only" and row_has_valid_duration(row):
                kept_rows.append(row)
                skipped_rows += 1
                continue
            try:
                for _ in range(repeat_count):
                    # Each repeat gets a fresh row mapping in case run_row_fn mutates it.
                    run_row_fn(csv_path, row_index, dict(row))
                    total_rows += 1
                kept_rows.append(row)
            except Exception as exc:
                if is_fatal_replay_exception(exc):
                    raise
                deleted_count += 1
                entry = register_invalid_replay_row(
                    kernel_type=kernel_type,
                    csv_path=csv_path,
                    row_index=row_index,
                    row=row,
                    exc=exc,
                )
                invalid_rows.append(entry)
                print(
                    f"[DROP] {csv_path}:{row_index} "
                    f"shapes={row.get('Input Shapes', '')} "
                    f"error={type(exc).__name__}: {exc}"
                )
            finally:
                if on_row_finally is not None:
                    on_row_finally()

        if deleted_count and (can_write_cleanup is None or can_write_cleanup()):
            write_csv_rows(csv_path, fieldnames, kept_rows)
            print(f"[CLEANUP] {csv_path} deleted {deleted_count} invalid row(s).")

        if deleted_count and on_cleanup_written is not None:
            on_cleanup_written()

    return total_rows, invalid_rows, source_row_count, skipped_rows


def print_invalid_replay_summary(
    invalid_rows: list[dict[str, str]],
    *,
    label: str | None = None,
) -> None:
    if not invalid_rows:
        summary_label = label or "All operators"
        print(f"[SUMMARY] {summary_label}: no invalid replay rows were deleted.")
        return

    summary_label = label or "Invalid replay rows"
    print(f"[SUMMARY] {summary_label}: deleted {len(invalid_rows)} invalid row(s).")
    for entry in invalid_rows:
        print(
            f"[SUMMARY] op={entry['kernel_type']} "
            f"file={entry['csv_path']}:{entry['row_index']} "
            f"shapes={entry['input_shapes']} "
            f"error={entry['error_type']}: {entry['error']}"
        )


def get_replay_repeat_count(args_repeat_count: int | None) -> int:
    if args_repeat_count is not None:
        if args_repeat_count <= 0:
            raise ValueError(f"--repeat-count must be positive, got {args_repeat_count}")
        return args_repeat_count

    raw_env = (os.environ.get(REPLAY_REPEAT_COUNT_ENV, "") or "").strip()
    if not raw_env:
        return DEFAULT_REPLAY_REPEAT_COUNT

    try:
        repeat_count = int(raw_env)
    except ValueError as exc:
        raise ValueError(
            f"{REPLAY_REPEAT_COUNT_ENV} must be an integer, got {raw_env!r}"
        ) from exc
    if repeat_count <= 0:
        raise ValueError(f"{REPLAY_REPEAT_COUNT_ENV} must be positive, got {raw_env!r}")
    return repeat_count


def build_standard_argparser(
    *,
    description: str,
    usage_examples: list[str],
    version_help: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=description,
        epilog=(
            "Usage examples:\n"
            + "\n".join(f"  {item}" for item in usage_examples)
            + "\n\nParameter notes:\n"
            + "  --database-path         Use an explicit database directory and bypass version-dir inference.\n"
            + f"  --device                Selects the device folder under profiling_database/data. Default: {DEFAULT_DEVICE}\n"
            + "  --vllm-version          Accepts either a plain vLLM version or a full version dir name.\n"
            + "  --torch-version         Optional PyTorch version used to build the version dir name.\n"
            + "  --cann-version          Optional CANN version used to build the version dir name.\n"
            + f"  --repeat-count          Repeat each replay row this many times. Defaults to {DEFAULT_REPLAY_REPEAT_COUNT}\n"
            + f"                          or ${REPLAY_REPEAT_COUNT_ENV} when set.\n"
            + "  --update-mode          `all` replays every row; `missing-only` replays only rows whose\n"
            + "                         Average/Profiling durations are both invalid.\n"
            + "  -h, --help              Show this help message and exit."
        ),
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help=(
            "Explicit database directory to read/write, e.g. "
            "tensor_cast/performance_model/profiling_database/data/.../vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5"
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=SUPPORTED_DEVICES,
        help=(
            "Target device folder under "
            "tensor_cast/performance_model/profiling_database/data/{device}/"
        ),
    )
    parser.add_argument(
        "--vllm-version",
        dest="vllm_version",
        type=check_version,
        help=version_help,
    )
    parser.add_argument(
        "--torch-version",
        type=check_version,
        help="Optional PyTorch version, e.g. 2.9.0.",
    )
    parser.add_argument(
        "--cann-version",
        type=check_version,
        help="Optional CANN version, e.g. 8.5.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        help=(
            "Repeat each replay row this many times. Defaults to "
            f"{DEFAULT_REPLAY_REPEAT_COUNT} or ${REPLAY_REPEAT_COUNT_ENV} when set."
        ),
    )
    parser.add_argument(
        "--update-mode",
        choices=SUPPORTED_UPDATE_MODES,
        default=DEFAULT_UPDATE_MODE,
        help=(
            "Replay selection mode. "
            "`all`: replay every row. "
            "`missing-only`: replay only rows whose Average/Profiling durations are both invalid. "
            f"Default: {DEFAULT_UPDATE_MODE}."
        ),
    )
    return parser

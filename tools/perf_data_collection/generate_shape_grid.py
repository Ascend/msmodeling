"""Append shape-grid rows to perf database CSV files.

Generates a deterministic grid from theoretical dimension ranges
(config-driven, first-principles approach).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
OP_REPLAY_DIR = CURRENT_DIR / "op_replay"
if str(OP_REPLAY_DIR) not in sys.path:
    sys.path.insert(0, str(OP_REPLAY_DIR))

from common import DEFAULT_DEVICE, SUPPORTED_DEVICES, check_version, get_target_data_dir

from grid_generator.runner import load_csv_files, run_theory_mode
from grid_generator.utils import clear_progress

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "tensor_cast"
    / "performance_model"
    / "profiling_database"
    / "data"
)
DEFAULT_ROWS = 10_000


def resolve_data_dir(
    data_dir: Path | None,
    device: str | None,
    vllm_ascend_version: str | None,
    torch_version: str | None,
    cann_version: str | None,
) -> Path:
    if data_dir is not None:
        return data_dir
    if device and vllm_ascend_version:
        return get_target_data_dir(
            device=device,
            vllm_ascend_version=vllm_ascend_version,
            torch_version=torch_version,
            cann_version=cann_version,
        )
    return DEFAULT_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append shape-grid rows to perf database CSV files "
                    "using deterministic grid from theoretical dimension ranges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target-models",
        type=str,
        default=None,
        help="Comma-separated model names (e.g. 'dsv3,qwen3-32b') to prune GEMM (N,K) pairs. "
             "Only used in theory mode. If omitted, uses full NK_GRID cartesian product.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "CSV root directory. If omitted, the script uses either "
            "{repo}/tensor_cast/performance_model/profiling_database/data or "
            "{repo}/.../data/{device}/vllm_ascend/{version}/ when --device and "
            "--vllm-version are provided."
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=SUPPORTED_DEVICES,
        help=(
            "Target device name used as input folder: "
            "tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version}/"
        ),
    )
    parser.add_argument(
        "--vllm-version",
        dest="vllm_version",
        type=check_version,
        help="vLLM version, e.g. 0.9.2.",
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
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=f"Cap per CSV (randomly sampled from full grid; 0 = no cap). Default: {DEFAULT_ROWS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible output (theory mode sampling)."
    )
    parser.add_argument(
        "--max-hbm-gb",
        type=float,
        default=32.0,
        help="Maximum HBM memory budget in GiB per operator shape row (theory mode only). "
             "Shapes whose estimated input+output tensor size exceeds this limit are "
             "filtered out during generation. Set to 0 to disable. Default: 32.0",
    )
    return parser.parse_args()




def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(
        args.data_dir,
        args.device,
        args.vllm_version,
        args.torch_version,
        args.cann_version,
    )
    csv_files = load_csv_files(data_dir)
    total_appended_rows, skipped_files = run_theory_mode(args, data_dir, csv_files)

    clear_progress()
    print(f"Appended {total_appended_rows} rows across {len(csv_files)} CSV files under {data_dir}.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files (no theory generator):")
        for csv_path in skipped_files:
            print(f"  - {csv_path.name}")


if __name__ == "__main__":
    main()

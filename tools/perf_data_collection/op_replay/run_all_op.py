"""
Run all operator replay scripts in the current op_replay directory.

Purpose:
  Discover every *_run.py script next to this file and execute each one with
  the same --device and --vllm-version arguments.

Usage:
  python tools/perf_data_collection/op_replay/run_all_op.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0

Arguments:
  --device                Passed through to every operator replay script.
  --vllm-version          Passed through to every operator replay script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys

from common import (
    DEFAULT_DEVICE,
    DEFAULT_UPDATE_MODE,
    SUPPORTED_DEVICES,
    SUPPORTED_UPDATE_MODES,
    build_database_cli_args,
    check_version,
    get_invalid_replay_rows,
    get_target_data_dir,
    normalize_op_name,
    print_invalid_replay_summary,
    reset_invalid_replay_rows,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SELF_NAME = Path(__file__).name
DISPATCH_FFN_COMBINE_OP_NAME = "DispatchFFNCombine"

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Run all operator replay scripts under tools/perf_data_collection/op_replay.\n"
            "Each operator script is executed once with the same device and\n"
            "vllm_ascend version arguments.\n"
            "By default, scripts run in-process so a single outer `msprof`\n"
            "session can capture all operators into one PROF_* directory."
        ),
        epilog=(
            "Usage examples:\n"
            "  py -3 tools/perf_data_collection/op_replay/run_all_op.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0\n"
            "  py -3 tools/perf_data_collection/op_replay/run_all_op.py "
            "--database-path tensor_cast/performance_model/profiling_database/data/"
            "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5\n"
            "  python tools/perf_data_collection/op_replay/run_all_op.py "
            "--device TEST_DEVICE --vllm-version 0.9.2\n"
            "  msprof python tools/perf_data_collection/op_replay/run_all_op.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.15.0\n\n"
            "Parameter notes:\n"
            "  --database-path         Passed through to every *_run.py script when provided.\n"
            f"  --device                Passed through to every *_run.py script. Default: {DEFAULT_DEVICE}\n"
            "  --vllm-version          Accepts either a plain version or a full version dir name.\n"
            "  --torch-version         Optional PyTorch version used to build the version dir name.\n"
            "  --cann-version          Optional CANN version used to build the version dir name.\n"
            "  --repeat-count          Passed through to every *_run.py script when provided.\n"
            "  --update-mode           Passed through to every *_run.py script. Default: all.\n"
            "  --execution-mode        `inprocess` keeps all operators in one Python process;\n"
            "                          `subprocess` preserves the old per-script child-process behavior.\n"
            "  -h, --help              Show this help message and exit."
        ),
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help="Explicit database directory to read from.",
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
        "--repeat-count",
        type=int,
        default=None,
        help="Optional replay repeat count passed through to every operator script.",
    )
    parser.add_argument(
        "--update-mode",
        choices=SUPPORTED_UPDATE_MODES,
        default=DEFAULT_UPDATE_MODE,
        help=(
            "Replay selection mode passed through to every operator script. "
            f"Default: {DEFAULT_UPDATE_MODE}."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=["inprocess", "subprocess"],
        default="inprocess",
        help=(
            "How to invoke each *_run.py script. Default: inprocess."
        ),
    )
    parser.add_argument(
        "--op",
        nargs="+",
        default=None,
        help=(
            "Optional operator names to run, e.g. MatMulV2 PadV3. "
            "Names may be given as OP, OP_run, or OP_run.py."
        ),
    )
    parser.add_argument(
        "--dispatch-ffn-combine-ep-size",
        type=int,
        default=None,
        help=(
            "Optional EP size to pass through to DispatchFFNCombine_run.py. "
            "Ignored by other operators."
        ),
    )
    parser.add_argument(
        "--dispatch-ffn-combine-nproc-per-node",
        type=int,
        default=None,
        help=(
            "torchrun processes per node for DispatchFFNCombine EP replay. "
            "Default: let DispatchFFNCombine_run.py infer it."
        ),
    )
    parser.add_argument(
        "--dispatch-ffn-combine-nnodes",
        type=int,
        default=1,
        help="torchrun node count for DispatchFFNCombine EP replay. Default: 1.",
    )
    parser.add_argument(
        "--dispatch-ffn-combine-node-rank",
        type=int,
        default=0,
        help="torchrun node rank for DispatchFFNCombine EP replay. Default: 0.",
    )
    parser.add_argument(
        "--dispatch-ffn-combine-master-addr",
        default="127.0.0.1",
        help="torchrun master address for DispatchFFNCombine EP replay. Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--dispatch-ffn-combine-master-port",
        type=int,
        default=None,
        help=(
            "torchrun master port for DispatchFFNCombine EP replay. "
            "Required for multi-node; default: auto-selected for single-node."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining operator scripts even if one fails.",
    )
    return parser


def discover_run_scripts() -> list[Path]:
    scripts = []
    for script_path in sorted(SCRIPT_DIR.glob("*_run.py")):
        if script_path.name == SELF_NAME:
            continue
        scripts.append(script_path)
    return scripts


def filter_run_scripts(scripts: list[Path], selected_ops: set[str] | None) -> list[Path]:
    if not selected_ops:
        return scripts
    return [script_path for script_path in scripts if normalize_op_name(script_path.stem) in selected_ops]


def get_csv_name(script_path: Path) -> str:
    return f"{script_path.stem.removesuffix('_run')}.csv"


def has_operator_csv(target_data_dir: Path, csv_name: str) -> bool:
    return any(target_data_dir.rglob(csv_name))


def append_dispatch_ffn_combine_args(
    command: list[str],
    script_path: Path,
    *,
    dispatch_ffn_combine_ep_size: int | None,
    dispatch_ffn_combine_nproc_per_node: int | None,
    dispatch_ffn_combine_nnodes: int,
    dispatch_ffn_combine_node_rank: int,
    dispatch_ffn_combine_master_addr: str,
    dispatch_ffn_combine_master_port: int | None,
) -> None:
    if normalize_op_name(script_path.stem) != DISPATCH_FFN_COMBINE_OP_NAME:
        return
    if dispatch_ffn_combine_ep_size is not None:
        command.extend(["--ep-size", str(dispatch_ffn_combine_ep_size)])
    if dispatch_ffn_combine_nproc_per_node is not None:
        command.extend(["--nproc-per-node", str(dispatch_ffn_combine_nproc_per_node)])
    if dispatch_ffn_combine_nnodes:
        command.extend(["--nnodes", str(dispatch_ffn_combine_nnodes)])
    if dispatch_ffn_combine_node_rank:
        command.extend(["--node-rank", str(dispatch_ffn_combine_node_rank)])
    if dispatch_ffn_combine_master_addr:
        command.extend(["--master-addr", dispatch_ffn_combine_master_addr])
    if dispatch_ffn_combine_master_port is not None:
        command.extend(["--master-port", str(dispatch_ffn_combine_master_port)])


def run_script_subprocess(
    script_path: Path,
    *,
    database_path: Path | None,
    device: str,
    vllm_ascend_version: str | None,
    torch_version: str | None,
    cann_version: str | None,
    repeat_count: int | None,
    update_mode: str,
    dispatch_ffn_combine_ep_size: int | None,
    dispatch_ffn_combine_nproc_per_node: int | None,
    dispatch_ffn_combine_nnodes: int,
    dispatch_ffn_combine_node_rank: int,
    dispatch_ffn_combine_master_addr: str,
    dispatch_ffn_combine_master_port: int | None,
) -> None:
    command = [
        sys.executable,
        str(script_path),
    ]
    command.extend(
        build_database_cli_args(
            database_path=database_path,
            device=device,
            vllm_ascend_version=vllm_ascend_version,
            torch_version=torch_version,
            cann_version=cann_version,
        )
    )
    if repeat_count is not None:
        command.extend(["--repeat-count", str(repeat_count)])
    command.extend(["--update-mode", update_mode])
    append_dispatch_ffn_combine_args(
        command,
        script_path,
        dispatch_ffn_combine_ep_size=dispatch_ffn_combine_ep_size,
        dispatch_ffn_combine_nproc_per_node=dispatch_ffn_combine_nproc_per_node,
        dispatch_ffn_combine_nnodes=dispatch_ffn_combine_nnodes,
        dispatch_ffn_combine_node_rank=dispatch_ffn_combine_node_rank,
        dispatch_ffn_combine_master_addr=dispatch_ffn_combine_master_addr,
        dispatch_ffn_combine_master_port=dispatch_ffn_combine_master_port,
    )
    print(f"[RUN] {script_path.name}")
    subprocess.run(command, check=True, cwd=SCRIPT_DIR)
    print(f"[DONE] {script_path.name}")


def run_script_inprocess(
    script_path: Path,
    *,
    database_path: Path | None,
    device: str,
    vllm_ascend_version: str | None,
    torch_version: str | None,
    cann_version: str | None,
    repeat_count: int | None,
    update_mode: str,
    dispatch_ffn_combine_ep_size: int | None,
    dispatch_ffn_combine_nproc_per_node: int | None,
    dispatch_ffn_combine_nnodes: int,
    dispatch_ffn_combine_node_rank: int,
    dispatch_ffn_combine_master_addr: str,
    dispatch_ffn_combine_master_port: int | None,
) -> None:
    original_argv = sys.argv[:]
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    sys.argv = [str(script_path)]
    sys.argv.extend(
        build_database_cli_args(
            database_path=database_path,
            device=device,
            vllm_ascend_version=vllm_ascend_version,
            torch_version=torch_version,
            cann_version=cann_version,
        )
    )
    if repeat_count is not None:
        sys.argv.extend(["--repeat-count", str(repeat_count)])
    sys.argv.extend(["--update-mode", update_mode])
    append_dispatch_ffn_combine_args(
        sys.argv,
        script_path,
        dispatch_ffn_combine_ep_size=dispatch_ffn_combine_ep_size,
        dispatch_ffn_combine_nproc_per_node=dispatch_ffn_combine_nproc_per_node,
        dispatch_ffn_combine_nnodes=dispatch_ffn_combine_nnodes,
        dispatch_ffn_combine_node_rank=dispatch_ffn_combine_node_rank,
        dispatch_ffn_combine_master_addr=dispatch_ffn_combine_master_addr,
        dispatch_ffn_combine_master_port=dispatch_ffn_combine_master_port,
    )
    print(f"[RUN] {script_path.name}")
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = original_argv
    print(f"[DONE] {script_path.name}")


def run_script(
    script_path: Path,
    *,
    database_path: Path | None,
    device: str,
    vllm_ascend_version: str | None,
    torch_version: str | None,
    cann_version: str | None,
    repeat_count: int | None,
    update_mode: str,
    dispatch_ffn_combine_ep_size: int | None,
    dispatch_ffn_combine_nproc_per_node: int | None,
    dispatch_ffn_combine_nnodes: int,
    dispatch_ffn_combine_node_rank: int,
    dispatch_ffn_combine_master_addr: str,
    dispatch_ffn_combine_master_port: int | None,
    execution_mode: str,
) -> None:
    if execution_mode == "subprocess":
        run_script_subprocess(
            script_path,
            database_path=database_path,
            device=device,
            vllm_ascend_version=vllm_ascend_version,
            torch_version=torch_version,
            cann_version=cann_version,
            repeat_count=repeat_count,
            update_mode=update_mode,
            dispatch_ffn_combine_ep_size=dispatch_ffn_combine_ep_size,
            dispatch_ffn_combine_nproc_per_node=dispatch_ffn_combine_nproc_per_node,
            dispatch_ffn_combine_nnodes=dispatch_ffn_combine_nnodes,
            dispatch_ffn_combine_node_rank=dispatch_ffn_combine_node_rank,
            dispatch_ffn_combine_master_addr=dispatch_ffn_combine_master_addr,
            dispatch_ffn_combine_master_port=dispatch_ffn_combine_master_port,
        )
        return
    run_script_inprocess(
        script_path,
        database_path=database_path,
        device=device,
        vllm_ascend_version=vllm_ascend_version,
        torch_version=torch_version,
        cann_version=cann_version,
        repeat_count=repeat_count,
        update_mode=update_mode,
        dispatch_ffn_combine_ep_size=dispatch_ffn_combine_ep_size,
        dispatch_ffn_combine_nproc_per_node=dispatch_ffn_combine_nproc_per_node,
        dispatch_ffn_combine_nnodes=dispatch_ffn_combine_nnodes,
        dispatch_ffn_combine_node_rank=dispatch_ffn_combine_node_rank,
        dispatch_ffn_combine_master_addr=dispatch_ffn_combine_master_addr,
        dispatch_ffn_combine_master_port=dispatch_ffn_combine_master_port,
    )


def main() -> None:
    args = build_argparser().parse_args()
    if args.execution_mode == "inprocess":
        # Global invalid-row tracking only works when every operator runs in this process.
        reset_invalid_replay_rows()
    selected_ops = None
    if args.op:
        selected_ops = {normalize_op_name(item) for item in args.op}
    scripts = discover_run_scripts()
    scripts = filter_run_scripts(scripts, selected_ops)
    if not scripts:
        if selected_ops:
            requested = ", ".join(sorted(selected_ops))
            raise FileNotFoundError(f"No matching operator run scripts found for: {requested}")
        raise FileNotFoundError(f"No operator run scripts found under {SCRIPT_DIR}")

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_version,
        database_path=args.database_path,
        torch_version=args.torch_version,
        cann_version=args.cann_version,
    )
    executed_count = 0
    skipped_count = 0
    run_status = {"success": [], "failed": [], "skipped": []}

    for script_path in scripts:
        csv_name = get_csv_name(script_path)
        if not has_operator_csv(target_data_dir, csv_name):
            print(f"No {csv_name} operator file found in this database. Skipping.")
            skipped_count += 1
            run_status["skipped"].append(script_path.name)
            continue

        try:
            run_script(
                script_path=script_path,
                database_path=args.database_path,
                device=args.device,
                vllm_ascend_version=args.vllm_version,
                torch_version=args.torch_version,
                cann_version=args.cann_version,
                repeat_count=args.repeat_count,
                update_mode=args.update_mode,
                dispatch_ffn_combine_ep_size=args.dispatch_ffn_combine_ep_size,
                dispatch_ffn_combine_nproc_per_node=args.dispatch_ffn_combine_nproc_per_node,
                dispatch_ffn_combine_nnodes=args.dispatch_ffn_combine_nnodes,
                dispatch_ffn_combine_node_rank=args.dispatch_ffn_combine_node_rank,
                dispatch_ffn_combine_master_addr=args.dispatch_ffn_combine_master_addr,
                dispatch_ffn_combine_master_port=args.dispatch_ffn_combine_master_port,
                execution_mode=args.execution_mode,
            )
            executed_count += 1
            run_status["success"].append(script_path.name)
        except subprocess.CalledProcessError as exc:
            if not args.continue_on_error:
                raise
            print(f"[FAIL] {script_path.name} exited with code {exc.returncode}")
            run_status["failed"].append({"op": script_path.name, "reason": f"subprocess exit code {exc.returncode}"})
        except SystemExit as exc:
            if exc.code not in (0, None):
                if not args.continue_on_error:
                    raise
                print(f"[FAIL] {script_path.name} exited with code {exc.code}")
                run_status["failed"].append({"op": script_path.name, "reason": f"SystemExit code {exc.code}"})
            else:
                executed_count += 1
                run_status["success"].append(script_path.name)
        except FileNotFoundError:
            print(f"No {csv_name} operator file found in this database. Skipping.")
            skipped_count += 1
            run_status["skipped"].append(script_path.name)
        except MemoryError:
            raise
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"[FAIL] {script_path.name} raised exception: {exc}")
            run_status["failed"].append({"op": script_path.name, "reason": str(exc)})

    status_path = SCRIPT_DIR / "run_all_op_status.json"
    try:
        with status_path.open("w", encoding="utf-8") as f:
            json.dump(run_status, f, indent=2)
    except Exception as e:
        print(f"Failed to write run_all_op_status.json: {e}")

    print(
        f"Executed {executed_count} operator run script(s), skipped {skipped_count} "
        f"script(s) under {SCRIPT_DIR}."
    )
    if args.execution_mode == "inprocess":
        print_invalid_replay_summary(
            get_invalid_replay_rows(),
            label="All operators",
        )


if __name__ == "__main__":
    main()



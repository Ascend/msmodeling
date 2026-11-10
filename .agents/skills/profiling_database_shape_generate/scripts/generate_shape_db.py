# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the software at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""模式B（正式构建）：生成干净的无 mock 时长的性能数据库。

编排两个链路工具，产物可直接交给 start_microbench.py 做正式算子回放：
1. parse_kernel_details.py —— 将 SKILL 目录下 kernel_details.csv 解析为逐算子 CSV；
2. generate_shape_grid.py —— 查询驱动生成 shape 行（默认仅输入 case 的精确需求），写入目标库。

默认只生成 cases yaml 中 workload 捕获到的算子 shape 与 runtime metadata（行数量级为
输入 case 直接可见的算子需求，而非数千行模型结构预置 shape）；需要覆盖 case 之外的
常用输入输出区间（插值锚点）时，显式传 --with-model-sweep 启用模型结构 sweep。

生成的行在回放前 latency 为空/0，属正式流程的预期状态（missing-only 回放只补这些行）。

失败重试与并发控制：
- sweep workload 因系统资源不足部分失败时自动重试（工具自带缓存，已完成的
  workload 不会重跑），直到全部成功或达到 --max-attempts；
- --max-workers 通过 patch grid_generator.query_workloads.MAX_PARALLEL_QUERY_WORKLOADS
  降低内部 torch 子进程并发数（默认工具值为 8；内存有限的机器建议 1-2）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
REPO_ROOT = SKILL_DIR.parents[2]

DEFAULT_KERNEL_DETAILS = SKILL_DIR / "kernel_details.csv"
PARSE_TOOL = REPO_ROOT / "tools" / "perf_data_collection" / "parsers" / "parse_kernel_details.py"
TOOLS_DIR = REPO_ROOT / "tools" / "perf_data_collection"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a clean profiling database (parse + query-driven shape generation; case demands only by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        required=True,
        help="Target database directory; must contain op_mapping.yaml (explicitly specified by the user).",
    )
    parser.add_argument(
        "--kernel-details-csv",
        type=Path,
        default=DEFAULT_KERNEL_DETAILS,
        help="NPU profiling kernel_details*.csv to parse. Pass --skip-parse to reuse existing CSVs.",
    )
    parser.add_argument("--skip-parse", action="store_true", help="Skip the parse step.")
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help=(
            "Force the parse step even when the database already contains data rows "
            "(e.g. the profiling export changed). Default: parse only runs on an empty/new "
            "database; existing databases are appended to in place without re-parsing."
        ),
    )
    parser.add_argument(
        "--cases-yaml",
        type=Path,
        required=True,
        help="Optimizer workload spec providing exact case demands for --optimizer-args-file.",
    )
    parser.add_argument(
        "--target-model",
        default=None,
        help=(
            "HuggingFace model ID for --with-model-sweep sweeps. Default: the model field of the "
            "first workload in --cases-yaml."
        ),
    )
    parser.add_argument(
        "--with-model-sweep",
        action="store_true",
        help=(
            "Additionally run the model-structure coverage sweeps for --target-model to add "
            "interpolation anchors beyond the case shapes. Default: only the exact case demands "
            "from --cases-yaml are generated."
        ),
    )
    parser.add_argument("--rows", type=int, default=1000, help="Per-operator new-row budget per run.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic ordering seed.")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional JSON generation report path.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help=(
            "Max concurrent internal torch workload subprocesses. The chain tool default is 8; "
            "use 1-2 on memory-limited machines."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max generate attempts; failed workloads resume from cache on each retry.",
    )
    return parser


def resolve_target_model(args: argparse.Namespace) -> str:
    """--target-model 未显式提供时，从 cases yaml 第一个 workload 的 model 字段推导。"""
    if args.target_model:
        return args.target_model
    import yaml

    data = yaml.safe_load(args.cases_yaml.resolve().read_text(encoding="utf-8")) or {}
    for workload in data.get("workloads") or []:
        model = workload.get("model") if isinstance(workload, dict) else None
        if model:
            return str(model)
    raise SystemExit(
        "--with-model-sweep requires --target-model, or a 'model' field in the first workload of --cases-yaml"
    )


def build_grid_command(args: argparse.Namespace, database: Path) -> list[str]:
    command = [
        sys.executable,
        str(TOOLS_DIR / "generate_shape_grid.py"),
        "--database-path",
        str(database),
        "--optimizer-args-file",
        str(args.cases_yaml.resolve()),
        "--rows",
        str(args.rows),
        "--seed",
        str(args.seed),
    ]
    if args.with_model_sweep:
        command += ["--target-models", args.target_model]
    if args.report_path is not None:
        command += ["--report-path", str(args.report_path.resolve())]
    return command


def run_grid(args: argparse.Namespace, database: Path, attempt: int) -> bool:
    """Run one generate attempt; return True when no workload failed."""
    command = build_grid_command(args, database)
    # 工具目录经环境变量传入子进程，避免把可能含引号/反斜杠的路径内插进 python -c 源码。
    bootstrap = (
        "import os, sys\n"
        "sys.path.insert(0, os.environ['MSMODELING_PERF_TOOLS_DIR'])\n"
        "from grid_generator import query_workloads as _qw\n"
        f"_qw.MAX_PARALLEL_QUERY_WORKLOADS = {args.max_workers}\n"
        "import generate_shape_grid as _gsg\n"
        "_gsg.main()\n"
    )
    print(f"[generate_shape_db] grid attempt {attempt}: max-workers={args.max_workers}, command: {' '.join(command)}")
    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, *command[2:]],
        cwd=REPO_ROOT,
        env={**os.environ, "MSMODELING_PERF_TOOLS_DIR": str(TOOLS_DIR)},
    )
    if completed.returncode != 0:
        raise SystemExit(f"generate_shape_grid failed with exit code {completed.returncode}")
    import json

    report_file = args.report_path.resolve()
    if not report_file.is_file():
        raise SystemExit(
            f"generation report not found: {report_file}; generate_shape_grid exited 0 but did not "
            "write the report — check the tool version or cache state before retrying"
        )
    report = json.loads(report_file.read_text(encoding="utf-8"))
    workloads = report.get("workloads", {})
    attempted = int(workloads.get("attempted", 0))
    succeeded = int(workloads.get("succeeded", 0))
    cached = int(workloads.get("cached", 0))
    failed = len(workloads.get("failed_workloads", []))
    print(f"[generate_shape_db] workloads: attempted={attempted} succeeded={succeeded} cached={cached} failed={failed}")
    return failed == 0


def database_has_data_rows(database: Path) -> bool:
    """True when any operator CSV in the database already contains data rows.

    已有实测/生成行的库在追加新 case 的 shape 时无需重新 parse（parse 仅负责把
    kernel_details.csv 的实测行写入空库；重复 parse 会整文件重写，耗时且无增益）。
    """
    for csv_path in database.glob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8-sig") as handle:
                if next(handle, None) is not None:  # header
                    if next(handle, None) is not None:  # first data row
                        return True
        except OSError:
            continue
    return False


def main() -> None:
    args = build_argparser().parse_args()
    if args.skip_parse and args.force_parse:
        raise SystemExit("--skip-parse and --force-parse are mutually exclusive")
    database = args.database_path.resolve()
    cases_yaml = args.cases_yaml.resolve()
    if not cases_yaml.is_file():
        raise SystemExit(f"cases yaml not found: {cases_yaml}")
    args.cases_yaml = cases_yaml
    if args.with_model_sweep:
        args.target_model = resolve_target_model(args)
    if not (database / "op_mapping.yaml").is_file():
        raise SystemExit(
            f"missing op_mapping.yaml under {database}; copy it from an existing version directory first "
            "(e.g. ../vllm0.23.0rc1_torch2.10.0_cann9.0.1/op_mapping.yaml)"
        )
    if not (PARSE_TOOL.is_file() and (TOOLS_DIR / "generate_shape_grid.py").is_file()):
        raise SystemExit("perf_data_collection tools not found; run inside the msmodeling repository")

    if args.skip_parse:
        print("[generate_shape_db] skip-parse: reusing existing operator CSVs")
    elif not args.force_parse and database_has_data_rows(database):
        print(
            "[generate_shape_db] database already contains data rows; skipping the parse step "
            "(new case shapes will be appended to the existing CSVs). "
            "Pass --force-parse to re-parse kernel_details.csv."
        )
    else:
        kernel_details = args.kernel_details_csv.resolve()
        if not kernel_details.is_file():
            raise SystemExit(
                f"kernel_details csv not found: {kernel_details}\n"
                "place the NPU profiling export at this path (default: "
                f"{DEFAULT_KERNEL_DETAILS}) or pass --kernel-details-csv / --skip-parse"
            )
        run(
            [
                sys.executable,
                str(PARSE_TOOL),
                "--profiling-path",
                str(kernel_details),
                "--database-path",
                str(database),
            ]
        )

    if args.report_path is None:
        args.report_path = database.parent / f"{database.name}-shape-generation-report.json"

    for attempt in range(1, args.max_attempts + 1):
        if run_grid(args, database, attempt):
            break
        if attempt < args.max_attempts:
            print(f"[generate_shape_db] workloads incomplete; retrying from cache (attempt {attempt + 1})")
    else:
        raise SystemExit(
            f"workloads still incomplete after {args.max_attempts} attempts; "
            "lower --max-workers, free memory, or rerun later (cached workloads resume automatically)"
        )

    print(
        "\n[generate_shape_db done] clean database ready (no mocked durations):\n"
        f"  {database}\n"
        "Next steps:\n"
        "  - Mode B (formal): replay via tools/perf_data_collection/start_microbench.py;\n"
        "  - Mode A (verify): RE-MOCK before verifying (mock copies must be regenerated after every\n"
        "                     generate run): scripts/mock_replay_durations.py, then\n"
        "                     scripts/verify_db_coverage.py;\n"
        "  - End-of-run report: scripts/generate_run_report.py --database-path <DB>."
    )


def run(command: list[str]) -> None:
    print(f"[generate_shape_db] run: {' '.join(str(c) for c in command)}")
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"command failed with exit code {completed.returncode}: {command}")


if __name__ == "__main__":
    main()

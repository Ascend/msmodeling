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

"""模式A步骤2（本地验证）：跑常用 case 的 prefill/decode 仿真并判定数据库命中情况。

对每个 case-phase 执行 python -m cli.inference.text_generate --log-level info
（纯本地仿真，无需 NPU），捕获回显日志后做两级解析：

1. op 汇总表（``Name / empirical total / empirical avg / analytic total /
   analytic avg / # of Calls``）：MISS 回退 roofline 时 empirical 与 analytic
   完全相等，因此 ``empirical != analytic`` 即数据库命中。默认验收口径
   ``top-total``：按 analytic total 降序取总耗时 Top-N（默认 10）**计算算子**
   （通信算子 all_reduce/all_gather/all_to_all/reduce_scatter 等被排除——其时长
   由 base HCCL alpha-beta 模型给出，wrapper 插值主动跳过查库，不应参与判定），
   要求全部被数据库命中（empirical != analytic）。
2. metrics_collector 的 ``MISSes``（完全 MISS 的算子与原因分类）与
   ``MISS shapes``（未命中的算子 shape）：``--miss-scope replay``（回放支持
   口径）把 MISS 的 func_name 经 op_mapping.yaml 归到 kernel_type，仅当
   kernel_type ∈ op_replay 清单时计为失败；``all``（严格口径）任何 MISS 都
   失败。两个口径仅作补充体检，不是默认验收标准。

每个 case-phase 的完整回显日志保存到
``<database>-verify-logs/<label>_run.log`` 供审计。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import re
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_OP_REPLAY_DIR = REPO_ROOT / "tools" / "perf_data_collection" / "op_replay"

PHASES = ("prefill", "decode")

# 通信算子（HCCL 集合通信 wrapper）在 runtime 汇总表中的名称片段。这些算子的时长
# 由 base 通信模型（HCCL alpha-beta 拟合）给出，wrapper 插值主动跳过查库
# （reason=wrapper_communication_owned_by_base），不参与 top-total 计算算子判定。
COMM_OP_TOKENS = (
    "all_reduce",
    "all_gather",
    "all_to_all",
    "reduce_scatter",
    "broadcast",
)


def is_communication_op(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in COMM_OP_TOKENS)


def load_case_workloads(cases_yaml: Path) -> dict[int, dict]:
    """case id (1-based workload order) -> workload dict from the cases yaml.

    每个 workload dict 同时提供 input_length/output_length 与模型/并行/量化参数；
    text_generate 命令模板从中继承模型专属 flag（如 enable_shared_expert_tp），
    CLI 显式参数优先级更高。cases yaml 必须存在（无模型默认回退）。
    """
    if not cases_yaml.is_file():
        raise SystemExit(f"cases yaml not found: {cases_yaml} (--cases-yaml is required)")
    data = yaml.safe_load(cases_yaml.read_text(encoding="utf-8")) or {}
    workloads: dict[int, dict] = {}
    for index, workload in enumerate(data.get("workloads", []), start=1):
        workloads[index] = dict(workload) if isinstance(workload, dict) else {}
    if not workloads:
        raise SystemExit(f"cases yaml has no workloads: {cases_yaml}")
    return workloads


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify profiling-database hit/interpolation coverage for the cases-yaml workloads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--database-path", type=Path, required=True, help="Database directory to verify (usually the mock dir)."
    )
    parser.add_argument(
        "--source-database-path",
        type=Path,
        default=None,
        help=(
            "Clean source database to check mock freshness against. Default: database-path with the "
            "_mockverify suffix stripped. Freshness check compares per-CSV row counts and fails on stale mocks."
        ),
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Downgrade stale-mock detection from an error to a warning.",
    )
    parser.add_argument(
        "--cases-yaml",
        type=Path,
        required=True,
        help=(
            "Cases yaml consumed by generate_shape_grid --optimizer-args-file; workload order defines "
            "case ids. Per-case model/device/parallel/quantize flags are inherited from each workload "
            "(CLI overrides below take precedence). Required — no model-specific fallback."
        ),
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case ids (1-based indices into --cases-yaml workloads). Default: all workloads.",
    )
    parser.add_argument("--phases", default="prefill,decode", help="Comma-separated phases: prefill,decode.")
    parser.add_argument(
        "--model", default=None, help="HuggingFace model ID (default: model from the cases-yaml workload)."
    )
    parser.add_argument(
        "--device", default=None, help="DeviceProfile name (default: device from the cases-yaml workload)."
    )
    parser.add_argument(
        "--tp-size", type=int, default=None, help="Tensor-parallel size (default: first tp_sizes entry)."
    )
    parser.add_argument(
        "--ep-size", type=int, default=None, help="Expert-parallel size (default: first ep_sizes entry)."
    )
    parser.add_argument(
        "--num-devices", type=int, default=None, help="Total device count (default: workload num_devices)."
    )
    parser.add_argument(
        "--quantize-linear-action",
        default=None,
        help="Linear quant action (default: workload quantize_linear_action; text_generate CLI default W8A8_DYNAMIC applies when absent).",
    )
    parser.add_argument(
        "--word-embedding-tp",
        default=None,
        help="Word-embedding TP mode; omitted when the workload has no word_embedding_tp (model-specific).",
    )
    parser.add_argument(
        "--enable-shared-expert-tp",
        action="store_true",
        help="Enable shared-expert TP; only appended when set via CLI or the workload field (model-specific).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for captured text_generate run logs. Default: <database-path>-verify-logs next to it.",
    )
    parser.add_argument(
        "--miss-scope",
        choices=("top-total", "replay", "all"),
        default="top-total",
        help=(
            "验收口径: top-total = 按 analytic total 排序的 Top-N 耗时算子必须被数据库命中"
            " (default); replay = 仅 op_mapping 可归到 op_replay 支持算子的 MISS 计失败;"
            " all = 任何 MISS 均失败（严格口径）."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top-time ops required to hit the database under the top-total scope.",
    )
    parser.add_argument(
        "--op-replay-dir",
        type=Path,
        default=DEFAULT_OP_REPLAY_DIR,
        help="op_replay directory defining the replay-supported kernel list.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--skip-run", action="store_true", help="Skip simulation; only analyze existing run logs.")
    return parser


def _first_candidate(value) -> int:
    """Extract the first candidate from a spec parallel-dimension list (or an int)."""
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 1
    if value is None:
        return 1
    return int(value)


def _max_candidate(value) -> int:
    """Extract the upper bound from a spec batch-range list (or an int)."""
    if isinstance(value, (list, tuple)):
        return max(int(item) for item in value) if value else 1
    if value is None:
        return 1
    return int(value)


def build_command(
    args: argparse.Namespace, workload: dict, input_length: int, output_length: int, phase: str
) -> list[str]:
    """Assemble the text_generate command for one case-phase.

    模型/设备/并行/量化/模型专属 flag（word_embedding_tp、enable_shared_expert_tp、
    compile）优先从 cases yaml 的 workload 条目继承，CLI 显式参数覆盖；
    这样 dense 模型（如 Qwen3-32B）不会误带 GLM-5.2 专属的 shared-expert flag。
    workload 缺少 model/device/num_devices/tp_sizes 时直接报错（无模型默认回退）。
    """
    model = args.model or workload.get("model")
    device = args.device or workload.get("device")
    num_devices = args.num_devices or workload.get("num_devices")
    if not (model and device and num_devices and workload.get("tp_sizes")):
        raise SystemExit(
            "workload missing required fields (model/device/num_devices/tp_sizes); "
            "fix the cases yaml or pass --model/--device/--num-devices/--tp-size explicitly"
        )
    tp_size = args.tp_size or _first_candidate(workload.get("tp_sizes"))
    quantize = args.quantize_linear_action or workload.get("quantize_linear_action")
    word_embedding_tp = args.word_embedding_tp or workload.get("word_embedding_tp")
    shared_expert_tp = args.enable_shared_expert_tp or bool(workload.get("enable_shared_expert_tp"))
    # compile 缺省与 text_generate CLI（store_true，default False）及生成侧
    # build_cases（compile=bool(args.compile)）一致，yaml 未写时不带 --compile。
    compile_flag = bool(workload.get("compile", False))
    non_expert_quantize = workload.get("quantize_non_expert_linear_action")
    attention_quantize = workload.get("quantize_attention_action")
    num_mtp_tokens = _first_candidate(workload.get("num_mtp_tokens")) if workload.get("num_mtp_tokens") else 0
    # batch_range 可能是单值 int 或 [lo, hi] 区间（单值 [N] 等价 [1, N]）；仿真取最大
    # batch（最苛刻的查询需求），与生成侧 demand 的 batch 上界对齐。
    num_queries = max(_max_candidate(workload.get("batch_range", 1)), 1)

    command = [
        sys.executable,
        "-m",
        "cli.inference.text_generate",
        str(model),
        "--num-queries",
        str(num_queries),
        "--query-length",
        str(input_length if phase == "prefill" else 1),
        "--context-length",
        # decode 按实测算子约定取 input + output_length（--query-length 恒为 1，带
        # --decode，与 build_cases 的反推 input_length = context_length - output_length
        # 对齐）。
        str(0 if phase == "prefill" else input_length + output_length),
        "--device",
        str(device),
        "--tp-size",
        str(tp_size),
        "--num-devices",
        str(num_devices),
        # profiling + analytic 双模型：top-total 口径依赖汇总表的 analytic total 列
        # （MISS 回退时 empirical == analytic，据此判定命中）。
        "--performance-model",
        "profiling",
        "--performance-model",
        "analytic",
        f"--profiling-database={args.database_path.resolve()}",
        "--log-level",
        "info",
    ]
    # 仅在用户显式传参或 workload 明确给出 ep_sizes 时才带 --ep-size；
    # dense 模型不附加默认值 1 的 MoE 专属 flag（与生成侧约定一致）。
    if args.ep_size:
        command += ["--ep-size", str(args.ep_size)]
    elif workload.get("ep_sizes"):
        command += ["--ep-size", str(_first_candidate(workload["ep_sizes"]))]
    if quantize:
        command += ["--quantize-linear-action", str(quantize)]
    if non_expert_quantize and str(non_expert_quantize) != "DISABLED":
        command += ["--quantize-non-expert-linear-action", str(non_expert_quantize)]
    if attention_quantize and str(attention_quantize) != "DISABLED":
        command += ["--quantize-attention-action", str(attention_quantize)]
    if num_mtp_tokens > 0:
        command += ["--num-mtp-tokens", str(num_mtp_tokens)]
    if compile_flag:
        command.append("--compile")
    if word_embedding_tp:
        command += ["--word-embedding-tp", str(word_embedding_tp)]
    if shared_expert_tp:
        command.append("--enable-shared-expert-tp")
    if phase == "decode":
        command.append("--decode")
    else:
        command.append("--prefill")
    return command


def load_op_mapping_kernels(database: Path) -> dict[str, list[str]]:
    """op_name -> all kernel candidates (kernel_type/primary + alternate) from op_mapping.yaml."""
    path = database / "op_mapping.yaml"
    if not path.is_file():
        raise SystemExit(f"missing op_mapping.yaml under {database}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping: dict[str, list[str]] = {}
    for op_name, info in (data.get("operator_mappings") or {}).items():
        if not isinstance(info, dict):
            continue
        kernels: list[str] = []
        for key in ("kernel_type", "primary_kernel_type"):
            primary = info.get(key)
            if primary and str(primary) not in kernels:
                kernels.append(str(primary))
        for alt in info.get("alternate_kernel_types", []) or []:
            if alt and str(alt) not in kernels:
                kernels.append(str(alt))
        if kernels:
            mapping[str(op_name)] = kernels
    return mapping


def load_replay_kernels(op_replay_dir: Path) -> set[str]:
    """Kernel types with a replay entry point (<KernelType>_run.py)."""
    if not op_replay_dir.is_dir():
        raise SystemExit(f"op_replay directory not found: {op_replay_dir}")
    return {path.name[: -len("_run.py")] for path in op_replay_dir.glob("*_run.py")}


# metrics_collector INFO 回显日志格式（见 tensor_cast/performance_model/metrics_collector.py log_stats）。
# 判定只依赖 MISSes（完全 MISS 算子 + 原因分类）与 MISS shapes（未命中 shape）两段；
# M1-M5 汇总数不参与判定也不展示。
_PARTIAL_RE = re.compile(r"PARTIAL: \d+/\d+ \((.*)\)")
_MISS_NAME_RE = re.compile(r"^(.*) \(x(\d+)\)$")
_MISS_ENTRY_RE = re.compile(r"^\s+\[([^\]]+)\]")
_SHAPE_ENTRY_RE = re.compile(r"^\s+(\S+)\s(.+)$")


def parse_run_log(text: str) -> dict:
    """Parse a text_generate --log-level info run log.

    Returns a dict with:
    - ``misses``: [{"func_name", "reason"}] entries — fully-missed funcs
      (reason from the ``MISSes`` section) plus per-shape-miss-only funcs
      (reason ``per-shape`` from the ``MISS shapes`` section);
    - ``partial_line``: raw ``PARTIAL`` summary line (display only);
    - ``shapes_truncated``: count of MISS shapes lines hidden by the log's
      "... and N more" truncation (0 when complete).
    """
    full_misses: list[tuple[str, str]] = []
    shape_miss_funcs: list[str] = []
    partial_line = ""
    shapes_truncated = 0
    in_misses = False
    in_shapes = False
    for line in text.splitlines():
        if in_misses:
            entry = _MISS_ENTRY_RE.match(line)
            if entry:
                reason = entry.group(1)
                names_part = line.split("] ", 1)[1] if "] " in line else ""
                names_part = names_part.split(": ", 1)[1] if ": " in names_part else names_part
                for item in names_part.split(", "):
                    item = item.strip()
                    if not item:
                        continue
                    count_match = _MISS_NAME_RE.match(item)
                    name = count_match.group(1) if count_match else item
                    count = int(count_match.group(2)) if count_match else 1
                    full_misses.extend([(name, reason)] * count)
                continue
            in_misses = False
        if in_shapes:
            stripped = line.strip()
            more = re.match(r"^\.\.\. and (\d+) more$", stripped)
            if more:
                shapes_truncated = int(more.group(1))
                in_shapes = False
                continue
            shape_entry = _SHAPE_ENTRY_RE.match(line)
            if shape_entry:
                shape_miss_funcs.append(shape_entry.group(1))
                continue
            in_shapes = False
        if " unique reasons):" in line:
            in_misses = True
            continue
        if "MISS shapes (" in line:
            in_shapes = True
            continue
        match = _PARTIAL_RE.search(line)
        if match:
            partial_line = match.group(0).strip()

    misses = [{"func_name": func_name, "reason": reason} for func_name, reason in full_misses]
    missed_funcs = {func_name for func_name, _ in full_misses}
    for func_name in shape_miss_funcs:
        if func_name not in missed_funcs:
            misses.append({"func_name": func_name, "reason": "per-shape"})
            missed_funcs.add(func_name)
    return {
        "misses": misses,
        "partial_line": partial_line,
        "shapes_truncated": shapes_truncated,
    }


# runtime.table_averages 汇总表格式（tensor_cast/runtime.py _render_average_table）：
# 表头含 "empirical total" 与 "analytic total"，行格式为
# ` <name>  <emp total>  <emp avg>  <ana total>  <ana avg>  <calls>`，时间带 ns/us/ms/s 后缀
# （_format_time 对 <1us 的值输出 ns，缺该档会导致短时长行漏解析并提前截断表体）。
_TABLE_HEADER_RE = re.compile(r"empirical total\s+empirical avg\s+analytic total\s+analytic avg")
_TABLE_ROW_RE = re.compile(
    r"^\s*(\S+)\s+([\d.]+)(ns|us|ms|s)\s+([\d.]+)(ns|us|ms|s)\s+([\d.]+)(ns|us|ms|s)\s+([\d.]+)(ns|us|ms|s)\s+(\d+)\s*$"
)
_TIME_SCALE = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def _parse_time(value: str, unit: str) -> float:
    return float(value) * _TIME_SCALE[unit]


def _format_time(seconds: float) -> str:
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f}ms"
    return f"{seconds * 1e6:.3f}us"


def parse_op_table(text: str) -> list[dict]:
    """Parse the runtime.table_averages summary table from a run log.

    Returns entries with ``name``, ``empirical``/``analytic`` total times in
    seconds, and ``calls``. MISS 回退 roofline 时 empirical 与 analytic 完全
    相等，因此两者不等即数据库命中。
    """
    entries: list[dict] = []
    in_table = False
    for line in text.splitlines():
        if not in_table:
            if _TABLE_HEADER_RE.search(line):
                in_table = True
            continue
        row = _TABLE_ROW_RE.match(line)
        if row:
            entries.append(
                {
                    "name": row.group(1),
                    # groups: 2/3 empirical total, 4/5 empirical avg, 6/7 analytic total, 8/9 analytic avg
                    "empirical": _parse_time(row.group(2), row.group(3)),
                    "analytic": _parse_time(row.group(6), row.group(7)),
                    "calls": int(row.group(10)),
                }
            )
            continue
        # separator/footer line ends the table body
        if line.strip() and not set(line.strip()) <= {"-", " "}:
            in_table = False
    return entries


def classify_misses(
    misses: list[dict], op_kernels: dict[str, list[str]], replay_kernels: set[str]
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str], int]]:
    """Split misses into replayable failures and exempt entries.

    Returns (replayable, exempt_counter); replayable items are
    (func_name, reason, kernel) tuples, exempt_counter groups by
    (kernel_or_unmapped, reason).
    """
    replayable: list[tuple[str, str, str]] = []
    exempt: Counter = Counter()
    kernel_cache: dict[str, list[str]] = {}
    for miss in misses:
        func_name = str(miss.get("func_name", ""))
        reason = str(miss.get("reason", ""))
        if func_name not in kernel_cache:
            kernel_cache[func_name] = (
                op_kernels.get(func_name) or op_kernels.get(func_name.removesuffix(".default")) or []
            )
        kernels = kernel_cache[func_name]
        replay_hits = [kernel for kernel in kernels if kernel in replay_kernels]
        if replay_hits:
            replayable.append((func_name, reason, replay_hits[0]))
        else:
            label = kernels[0] if kernels else "unmapped"
            exempt[(label, reason)] += 1
    return replayable, dict(exempt)


def evaluate_top_total(op_table: list[dict], top_n: int) -> tuple[bool, list[str], list[str], list[dict]]:
    """Top-total 验收口径：按 analytic total 降序取**全体算子** Top-N，判定其中的计算算子。

    Top-N 从所有算子（含通信算子）中选取；命中判定只针对其中的计算算子——通信算子
    （all_reduce/all_gather 等，见 COMM_OP_TOKENS）的时长由 base HCCL alpha-beta 模型
    给出，wrapper 插值主动跳过查库，仅列出排除说明、不参与 HIT/MISS。
    MISS 回退 roofline 时 empirical 与 analytic 完全相等，因此两者不等即命中。
    已知限制：判定基于 runtime 表 3 位小数格式化文本解析出的浮点数；实际命中但两列差值
    小于当前档位显示分辨率（如 ms 档约 1us）时，可能因舍入相等被误判为 MISS（fail-closed
    方向）。对头部大耗时算子影响可忽略，不据此放宽判定口径。
    Returns (passed, problems, info_lines, ranked_entries)。
    """
    if not op_table:
        return False, ["op summary table not found in run log (need --log-level info)"], [], []
    ranked_all = sorted(op_table, key=lambda entry: entry["analytic"], reverse=True)[: max(top_n, 0)]
    comm_in_top = [entry for entry in ranked_all if is_communication_op(entry["name"])]
    judged = [entry for entry in ranked_all if not is_communication_op(entry["name"])]
    problems = [
        f"{entry['name']} analytic={_format_time(entry['analytic'])} (roofline fallback, empirical == analytic)"
        for entry in judged
        if entry["empirical"] == entry["analytic"]
    ]
    comm_desc = ", ".join(f"{entry['name']}({_format_time(entry['analytic'])})" for entry in comm_in_top) or "none"
    info = [
        f"top-{len(ranked_all)} ops by analytic total "
        f"({len(comm_in_top)} communication op(s) excluded from judgment: {comm_desc}):"
    ]
    info += [
        f"  {entry['name']}  empirical={_format_time(entry['empirical'])}  "
        f"analytic={_format_time(entry['analytic'])}  calls={entry['calls']}  "
        f"{'HIT' if entry['empirical'] != entry['analytic'] else 'MISS'}"
        for entry in judged
    ]
    ranked_view = [
        {
            "name": entry["name"],
            "empirical": entry["empirical"],
            "analytic": entry["analytic"],
            "calls": entry["calls"],
            "hit": entry["empirical"] != entry["analytic"],
            "judged": not is_communication_op(entry["name"]),
        }
        for entry in ranked_all
    ]
    if not judged:
        # Top-N 全为通信算子时没有任何计算算子被验证，fail-closed 判 FAIL，
        # 避免"零验证即通过"的误导性结论（可减小 top-n 或换 scope 复验）。
        return False, ["top-N contains no compute ops to judge"], info, ranked_view
    return not problems, problems, info, ranked_view


def evaluate(
    parsed: dict, op_kernels: dict[str, list[str]], replay_kernels: set[str], scope: str
) -> tuple[bool, list[str], list[str]]:
    """Return (passed, problems, info_lines) under the selected miss scope.

    判定标准：op_mapping ∩ op_replay 交集算子的所有 shape 均被命中 —— 即
    解析出的任何 MISS（MISSes 完全未命中 + MISS shapes 按 shape 未命中）
    都不能归到回放支持的 kernel 上。
    """
    misses = parsed.get("misses", [])
    if scope == "all":
        problems = sorted({f"{m['func_name']} reason={m['reason']}" for m in misses})
        return not misses, problems, []
    replayable, exempt = classify_misses(misses, op_kernels, replay_kernels)
    problems = sorted({f"{func_name} kernel={kernel} reason={reason}" for func_name, reason, kernel in replayable})
    info = [
        f"exempt (not replay-supported): {count} x {label} reason={reason}"
        for (label, reason), count in sorted(exempt.items())
    ]
    if parsed.get("partial_line"):
        info.append(f"partial ops (some shapes fell back): {parsed['partial_line']}")
    if parsed.get("shapes_truncated"):
        info.append(
            f"MISS shapes log truncated: ... and {parsed['shapes_truncated']} more (see the run log for details)"
        )
    return not replayable, problems, info


def csv_row_counts(database: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for csv_path in database.glob("*.csv"):
        with csv_path.open("r", encoding="utf-8-sig") as handle:
            counts[csv_path.name] = sum(1 for _ in handle) - 1  # minus header
    return {name: count for name, count in counts.items() if count > 0}


def check_mock_freshness(database: Path, source: Path) -> list[str]:
    """Return stale-mock problems: source CSVs missing from the mock, or per-CSV row counts differing."""
    problems: list[str] = []
    mock_counts = csv_row_counts(database)
    source_counts = csv_row_counts(source)
    for name, source_rows in sorted(source_counts.items()):
        mock_rows = mock_counts.get(name)
        if mock_rows is None:
            problems.append(f"{name}: missing in mock (source has {source_rows} rows)")
        elif mock_rows != source_rows:
            problems.append(f"{name}: mock has {mock_rows} rows, source has {source_rows}")
    return problems


def main() -> None:
    args = build_argparser().parse_args()
    database = args.database_path.resolve()
    if not (database / "op_mapping.yaml").is_file():
        raise SystemExit(f"not a profiling database (missing op_mapping.yaml): {database}")

    source = args.source_database_path
    if source is None and database.name.endswith("_mockverify"):
        source = database.with_name(database.name[: -len("_mockverify")])
    if source is not None and source.resolve().is_dir():
        stale = check_mock_freshness(database, source.resolve())
        if stale:
            message = (
                "mock database is STALE relative to the clean source (row counts differ); "
                "re-run scripts/mock_replay_durations.py before verifying:\n  "
                + "\n  ".join(stale[:20])
                + (f"\n  ... and {len(stale) - 20} more" if len(stale) > 20 else "")
            )
            if args.allow_stale:
                print(f"[verify_db_coverage] WARNING: {message}")
            else:
                raise SystemExit(f"[verify_db_coverage] ERROR: {message}\n(pass --allow-stale to override)")
        else:
            print("[verify_db_coverage] mock freshness check passed (row counts match source)")
    case_workloads = load_case_workloads(args.cases_yaml.resolve())
    print(
        f"[verify_db_coverage] cases-yaml: {args.cases_yaml.resolve()} "
        f"({len(case_workloads)} workload(s); case id = 1-based workload order)"
    )
    if args.cases:
        case_ids = []
        for item in args.cases.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                case_ids.append(int(item))
            except ValueError:
                raise SystemExit(
                    f"invalid case id {item!r}: --cases expects comma-separated integers (got {args.cases!r})"
                ) from None
    else:
        case_ids = sorted(case_workloads)
    phases = [item.strip() for item in args.phases.split(",") if item.strip()]
    unknown = [c for c in case_ids if c not in case_workloads]
    invalid = [p for p in phases if p not in PHASES]
    if unknown or invalid:
        raise SystemExit(
            f"unknown cases {unknown} or phases {invalid}; --cases-yaml provides case ids "
            f"{sorted(case_workloads)} (1-based workload order)"
        )

    log_dir = (args.log_dir or Path(f"{database}-verify-logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    op_kernels: dict[str, list[str]] = {}
    replay_kernels: set[str] = set()
    if args.miss_scope == "replay" and not args.dry_run:
        op_kernels = load_op_mapping_kernels(database)
        replay_kernels = load_replay_kernels(args.op_replay_dir)
        print(
            f"[verify_db_coverage] miss-scope=replay: {len(replay_kernels)} replay-supported kernels "
            f"from {args.op_replay_dir}, {len(op_kernels)} mapped ops from op_mapping.yaml"
        )

    results: list[tuple[str, bool, dict, list[str], list[str]]] = []
    summary_phases: list[dict] = []
    for case_id in case_ids:
        workload = case_workloads[case_id]
        input_length = int(workload.get("input_length", 0))
        output_length = int(workload.get("output_length", 0))
        if input_length < 1 or output_length < 1:
            raise SystemExit(
                f"case {case_id}: workload must define positive input_length/output_length "
                f"(got input_length={input_length}, output_length={output_length}); "
                "prefill uses --query-length <input_length> and decode uses "
                "--context-length <input_length + output_length>"
            )
        for phase in phases:
            label = f"case{case_id}-{phase}"
            log_file = log_dir / f"{label}_run.log"
            command = build_command(args, workload, input_length, output_length, phase)
            print(f"\n[verify_db_coverage] {label}\n  {' '.join(command)}")
            summary_entry = {
                "label": label,
                "case_id": case_id,
                "phase": phase,
                "command": command,
                "log_file": str(log_file),
            }
            if args.dry_run:
                summary_entry.update({"passed": None, "problems": [], "info": [], "top_ops": [], "misses": []})
                summary_phases.append(summary_entry)
                continue
            if not args.skip_run:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
                log_file.write_text(output, encoding="utf-8")
                summary_entry["returncode"] = completed.returncode
                if completed.returncode != 0:
                    problem = f"text_generate exited with {completed.returncode} (log: {log_file})"
                    results.append((label, False, {}, [problem], []))
                    summary_entry.update({"passed": False, "problems": [problem], "top_ops": [], "misses": []})
                    summary_phases.append(summary_entry)
                    continue
            if not log_file.is_file():
                problem = f"run log missing: {log_file}"
                results.append((label, False, {}, [problem], []))
                summary_entry.update({"passed": False, "problems": [problem], "top_ops": [], "misses": []})
                summary_phases.append(summary_entry)
                continue
            parsed = parse_run_log(log_file.read_text(encoding="utf-8"))
            if args.miss_scope == "top-total":
                op_table = parse_op_table(log_file.read_text(encoding="utf-8"))
                passed, problems, info, ranked_view = evaluate_top_total(op_table, args.top)
                summary_entry["top_ops"] = ranked_view
            else:
                passed, problems, info = evaluate(parsed, op_kernels, replay_kernels, args.miss_scope)
                summary_entry["top_ops"] = []
            summary_entry.update(
                {
                    "passed": passed,
                    "problems": problems,
                    "info": info,
                    "misses": parsed.get("misses", []),
                    "partial_line": parsed.get("partial_line", ""),
                }
            )
            summary_phases.append(summary_entry)
            results.append((label, passed, parsed, problems, info))

    if args.dry_run:
        print("\n[dry-run] commands printed only; no simulation was executed.")
        return

    summary_path = log_dir / "verify-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "database": str(database),
                "source_database": str(source) if source is not None else None,
                "cases_yaml": str(args.cases_yaml.resolve()),
                "miss_scope": args.miss_scope,
                "top": args.top if args.miss_scope == "top-total" else None,
                "phases": summary_phases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[verify_db_coverage] summary written: {summary_path}")
    print("\n===== coverage verification summary =====")
    all_passed = True
    problem_label = "FALLBACK(top-time)" if args.miss_scope == "top-total" else "MISS(replayable)"
    for label, passed, parsed, problems, info in results:
        all_passed &= passed
        print(f"  {label}: {'PASS' if passed else 'FAIL'}")
        for problem in problems:
            print(f"    {problem_label}: {problem}")
        for line in info:
            print(f"    {line}")
    scope_note = {
        "top-total": (
            "top-total scope: the Top-N ops (by analytic total) are selected from ALL ops; "
            "communication ops within the Top-N are excluded from judgment (their durations come "
            "from the base HCCL alpha-beta model) and every compute op in the Top-N must hit the "
            "database (empirical != analytic)."
        ),
        "replay": (
            "replay scope: only misses whose kernel has an op_replay entry point fail; "
            "exempt entries are listed above for visibility."
        ),
        "all": "strict scope: every miss fails.",
    }[args.miss_scope]
    verdict = {
        "top-total": (
            "ALL PASS: every top-time compute op hit the database; no roofline fallback in the top ranking."
            if all_passed
            else "FAIL: some top-time compute ops fell back to roofline/analytic (empirical == analytic). "
            "Analyze the MISS reasons and fix per the run report "
            "(scripts/generate_run_report.py), then regenerate shapes and re-verify."
        ),
        "replay": (
            "ALL PASS: every replay-supported op hit the database or was interpolated; no roofline fallback."
            if all_passed
            else "FAIL: roofline fallback detected on replay-supported kernels. Inspect MISS(replayable) reasons "
            "above, then add shapes/metadata via generate_shape_db.py and re-verify."
        ),
        "all": ("ALL PASS: no misses at all." if all_passed else "FAIL: misses detected (strict scope)."),
    }[args.miss_scope]
    print(f"\n[{scope_note}]\n{verdict}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

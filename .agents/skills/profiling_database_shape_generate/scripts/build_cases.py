# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""构建 optimizer-args cases yaml：text_generate 命令/参数组合 → workload spec。

模式1（--command）：解析一条完整 text_generate 命令行（用仓库真实 registry
解析器，获得与 CLI 完全一致的校验），按字段映射表转换为
generate_shape_grid --optimizer-args-file 消费的 workload spec yaml。

模式2（--set）：部分参数 × 未指定维度的默认网格做确定性笛卡尔组合；逐组合
复用 throughput_optimizer 的候选解析（grid_generator.optimizer_spec）做 spec
级 preflight，淘汰不可行组合并去重；--max-workloads 上限 + 均匀采样控制组合
爆炸（每个未指定维度给小默认网格而非全集）。

并行维度（tp_sizes/ep_sizes/moe_dp_sizes/dcp_sizes/pp_sizes）本身是候选列表，
由 spec 语义自动展开，不进入模式2的网格叉积。单个 workload 场景同时覆盖
prefill 与 decode 两阶段的查询需求。产物仅供 generate_shape_db.py
--cases-yaml / generate_shape_grid.py --optimizer-args-file 使用。

text_generate ↔ workload spec 字段映射表（模式1）：

  text_generate CLI                spec 字段                          映射规则
  --------------------------------------------------------------------------------------------
  <model_id>（位置参数）           model                              直接
  --device                         device                             直接
  --num-devices                    num_devices                        直接
  --tp-size N                      tp_sizes: [N]                      包裹为候选列表
  --ep-size N                      ep_sizes: [N]                      包裹为候选列表
  --moe-dp-size N                  moe_dp_sizes: [N]                  包裹为候选列表（仅 N 提供）
  --dcp-size N                     dcp_sizes: [N]                     包裹为候选列表（仅 N>1）
  --pp-size N                      pp_sizes: [N]                      包裹为候选列表（仅 N>1）
  --query-length / --context-length  input_length                     decode: context_length - output_length
                                                                     （实测算子 decode 命令 --query-length 恒为 1，
                                                                     --context-length = input_length +
                                                                     output_length，且带 --decode）；
                                                                     prefill: query_length（context>0 告警）
  （无对应参数）                   output_length                      --output-length（默认 2048）
  --num-queries N                  batch_range: [N, N]                单值精确 batch（不用 1..N sweep）
  --num-mtp-tokens N               num_mtp_tokens: [N]                仅 N>0
  --speculative-method mtp
    --num-speculative-tokens N     num_mtp_tokens: [N]                非 mtp 方法不支持，告警
  --quantize-linear-action 等 3 项 同名字段                             直接
  --compile / --compilation-config compile / compilation_config       直接（config 列表）
  --enable-shared-expert-tp        enable_shared_expert_tp: true      仅显式开启
  --word-embedding-tp              word_embedding_tp                  仅显式提供
  --prefix-cache-hit-rate          prefix_cache_hit_rate              仅 >0
  --reserved-memory-gb             reserved_memory_gb                 仅提供
  --mxfp4-group-size               mxfp4_group_size                   仅提供
  --dp-size / --fusion-plugin / --no-repetition / image-* / per-layer
  tp-dp / debug 与输出类参数        （无 spec 字段）                   丢弃并逐项告警
  --performance-model / --profiling-database 等                       丢弃（由 generate_shape_grid 管理）
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
from pathlib import Path
import shlex
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
REPO_ROOT = SKILL_DIR.parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "perf_data_collection"

for path in (str(REPO_ROOT), str(TOOLS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

DEFAULT_OUTPUT_LENGTH = 2048

# 模式2默认网格：每个未指定维度给小网格而非全集，控制组合爆炸。
GRID_DEFAULTS = {
    "input_length": [1024, 8192, 65536, 131072],
    "output_length": [1024, 2048],
    "batch_range": [1],
    "quantize_linear_action": ["W8A8_DYNAMIC"],
}
GRID_ARG_FLAGS = {
    "input_length": "--input-length-grid",
    "output_length": "--output-length-grid",
    "batch_range": "--batch-range-grid",
    "quantize_linear_action": "--quantize-grid",
}
MODE2_REQUIRED_FIELDS = ("model", "device", "num_devices")

# 模式1中丢弃并告警的 text_generate 参数：attr -> (默认值, 说明)
DROPPED_WITH_WARNING = {
    "num_queries": (None, ""),  # 单独处理（映射进 batch_range），此处仅占位说明
    "fusion_plugin": (None, "fusion 插件仅影响编译期融合，不改变查询需求"),
    "disable_repetition": (False, "层复用只影响同 shape 调用次数，不影响 per-layer 查询需求集合"),
    "num_hidden_layers_override": (0, "spec 不支持层数覆盖；PP 校验按模型真实层数"),
    "dump_input_shapes": (False, "输出格式参数"),
    "dump_op_bound_results": (False, "输出格式参数"),
    "chrome_trace": (None, "输出文件参数"),
    "graph_log_url": (None, "调试输出参数"),
    "export_empirical_metrics_file": (None, "由 verify_db_coverage.py 追加"),
    "performance_model": (None, "shape 生成恒为 profiling，由 generate_shape_grid 管理"),
    "profiling_database": (None, "由 generate_shape_grid --database-path 管理"),
    "disable_profiling_interpolation": (False, "由 generate_shape_grid 管理"),
    "analytic_calibration_profile": (None, "analytic 校准参数与 profiling 生成无关"),
    "analytic_calibration_stack": (None, "analytic 校准参数与 profiling 生成无关"),
    "remote_source": (None, "模型下载参数"),
    "image_batch_size": (None, "多模态输入 spec 暂不支持"),
    "image_height": (None, "多模态输入 spec 暂不支持"),
    "image_width": (None, "多模态输入 spec 暂不支持"),
    "o_proj_tp_size": (None, "spec 不支持 per-layer 并行覆盖"),
    "o_proj_dp_size": (None, "spec 不支持 per-layer 并行覆盖"),
    "mlp_tp_size": (None, "spec 不支持 per-layer 并行覆盖"),
    "mlp_dp_size": (None, "spec 不支持 per-layer 并行覆盖"),
    "lmhead_tp_size": (None, "spec 不支持 per-layer 并行覆盖"),
    "lmhead_dp_size": (None, "spec 不支持 per-layer 并行覆盖"),
    "moe_tp_size": (None, "spec 不支持 per-layer 并行覆盖"),
    "vision_tp_size": (1, "spec 不支持 vision 并行覆盖"),
    "enable_redundant_experts": (False, "spec 暂不支持"),
    "enable_external_shared_experts": (False, "spec 暂不支持"),
    "host_external_shared_experts": (False, "spec 暂不支持"),
    "compile_allow_graph_break": (False, "编译调试参数，不改变查询需求"),
    "quantize_lmhead": (False, "spec 暂不支持 LM Head 量化开关"),
    "dp_size": (None, "spec 无独立 DP；DP = num_devices / (TP x PP) 自动推导"),
}


def split_command(text: str) -> list[str]:
    """Split a pasted command line into argv tokens (Windows-path safe)."""
    tokens = shlex.split(text.strip(), posix=False)
    return [token.strip('"').strip("'") for token in tokens if token.strip()]


def strip_launcher_prefix(tokens: list[str]) -> list[str]:
    markers = {"cli.inference.text_generate", "text_generate", "text-generate"}
    for index, token in enumerate(tokens):
        if token in markers:
            return tokens[index + 1 :]
    return tokens


def parse_text_generate_command(argv: list[str]):
    from cli.registry.argparse_adapter import build_argparser, parse_module_args
    from cli.registry.modules import get_spec

    spec = get_spec("text_generate")
    parser = build_argparser(spec)
    return parse_module_args(spec, parser, argv)


def map_args_to_spec(args, command_warnings: list[str], output_length: int, name: str) -> dict:
    """Map parsed text_generate args onto one workload spec dict."""

    def as_spec_value(value):
        # Registry parsing yields enum instances for quantize actions; the
        # workload spec (and yaml dump) needs their plain string values.
        return value.value if hasattr(value, "value") else value

    workload: dict = {
        "name": name,
        "model": args.model_id,
        "device": args.device,
        "num_devices": args.num_devices,
        "batch_range": [args.num_queries, args.num_queries],
        "tp_sizes": [args.tp_size],
        "ep_sizes": [args.ep_size],
        "quantize_linear_action": as_spec_value(args.quantize_linear_action),
        "quantize_non_expert_linear_action": as_spec_value(args.quantize_non_expert_linear_action),
        "quantize_attention_action": as_spec_value(args.quantize_attention_action),
        "compile": bool(args.compile),
    }
    if args.decode:
        # 实测算子 decode 命令约定：--query-length 恒为 1，--context-length =
        # input_length + output_length（带 --decode）。因此 workload spec 反推为
        # input_length = context_length - output_length，与
        # verify_db_coverage.build_command 的正向映射对齐。
        input_length = args.context_length - output_length
        if input_length < 1:
            raise SystemExit(
                f"decode command needs --context-length > output_length "
                f"({output_length}) to derive input_length, got {args.context_length}"
            )
        workload["input_length"] = input_length
    else:
        workload["input_length"] = args.query_length
        if args.context_length:
            command_warnings.append(
                f"--context-length {args.context_length} on prefill: prefix context is not representable "
                "in workload specs (input_length drives both M and KV); if prefix reuse matters, set "
                "prefix_cache_hit_rate in the spec manually"
            )
    workload["output_length"] = output_length

    if args.moe_dp_size:
        workload["moe_dp_sizes"] = [args.moe_dp_size]
    if args.dcp_size and args.dcp_size > 1:
        workload["dcp_sizes"] = [args.dcp_size]
    if args.pp_size and args.pp_size > 1:
        workload["pp_sizes"] = [args.pp_size]
    if args.num_mtp_tokens:
        workload["num_mtp_tokens"] = [args.num_mtp_tokens]
    method = getattr(args, "speculative_method", None)
    if method:
        if method != "mtp":
            command_warnings.append(f"--speculative-method {method}: only mtp maps to num_mtp_tokens; dropped")
        else:
            workload["num_mtp_tokens"] = [args.num_speculative_tokens]
    if args.compilation_config:
        workload["compilation_config"] = [str(token) for token in args.compilation_config]
        workload["compile"] = True
    if args.word_embedding_tp:
        workload["word_embedding_tp"] = args.word_embedding_tp
    if args.enable_shared_expert_tp:
        workload["enable_shared_expert_tp"] = True
    if args.prefix_cache_hit_rate and args.prefix_cache_hit_rate > 0:
        workload["prefix_cache_hit_rate"] = args.prefix_cache_hit_rate
    if args.reserved_memory_gb:
        workload["reserved_memory_gb"] = args.reserved_memory_gb
    if args.mxfp4_group_size:
        workload["mxfp4_group_size"] = args.mxfp4_group_size

    for attr, (default, note) in DROPPED_WITH_WARNING.items():
        if attr == "num_queries":
            continue
        value = getattr(args, attr, None)
        if value is None or value == default or value is False:
            continue
        message = f"--{attr.replace('_', '-')}={value} dropped"
        if note:
            message += f" ({note})"
        command_warnings.append(message)
    return workload


def parse_scalar_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def coerce_set_value(raw: str):
    import yaml

    return yaml.safe_load(raw)


def mode2_grids(args: argparse.Namespace, set_fields: dict) -> dict[str, list]:
    grids: dict[str, list] = {}
    for field, default in GRID_DEFAULTS.items():
        if field in set_fields:
            continue
        if field == "batch_range":
            values = [int(item) for item in parse_scalar_list(getattr(args, "batch_range_grid"))]
            grids[field] = [[value, value] for value in values]
        elif field == "input_length":
            grids[field] = [int(item) for item in parse_scalar_list(args.input_length_grid)]
        elif field == "output_length":
            grids[field] = [int(item) for item in parse_scalar_list(args.output_length_grid)]
        else:
            grids[field] = parse_scalar_list(args.quantize_grid)
    return grids


def build_mode2_workloads(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    set_fields = {}
    for pair in args.set or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if key not in {
            "name",
            "model",
            "device",
            "num_devices",
            "input_length",
            "output_length",
            "batch_range",
            "max_batched_tokens",
            "tp_sizes",
            "ep_sizes",
            "moe_dp_sizes",
            "dcp_sizes",
            "pp_sizes",
            "num_mtp_tokens",
            "mtp_acceptance_rates",
            "quantize_linear_action",
            "quantize_non_expert_linear_action",
            "quantize_attention_action",
            "compilation_config",
            "compile",
            "enable_shared_expert_tp",
            "disagg",
            "ttft_limit",
            "tpot_limit",
            "reserved_memory_gb",
            "prefix_cache_hit_rate",
            "word_embedding_tp",
            "mxfp4_group_size",
        }:
            raise SystemExit(f"--set key '{key}' is not a workload spec field (see optimizer_spec.SPEC_FIELDS)")
        set_fields[key] = coerce_set_value(raw)

    missing = [key for key in MODE2_REQUIRED_FIELDS if key not in set_fields]
    if missing:
        raise SystemExit(f"mode 2 requires {list(MODE2_REQUIRED_FIELDS)} via --set; missing: {missing}")

    grids = mode2_grids(args, set_fields)
    combos = [dict(zip(grids, values)) for values in itertools.product(*grids.values())]
    warnings: list[str] = []
    if not combos:
        return [], warnings
    limit = args.max_workloads
    if len(combos) > limit:
        step = -(-len(combos) // limit)  # ceil division: evenly spaced subsample
        sampled = combos[::step][:limit]
        warnings.append(
            f"grid product {len(combos)} exceeds --max-workloads {limit}; "
            f"kept {len(sampled)} evenly spaced combinations (deterministic)"
        )
        combos = sampled

    prefix = args.name_prefix or f"{str(set_fields['model']).split('/')[-1]}"
    workloads: list[dict] = []
    for index, combo in enumerate(combos, start=1):
        entry = {**set_fields, **combo}
        entry["name"] = f"{entry.pop('name', None) or prefix}-combo{index}"
        workloads.append(entry)
    return workloads, warnings


_NUM_HIDDEN_LAYERS_CACHE: dict[str, int] = {}


@contextlib.contextmanager
def _memoized_model_config_lookup():
    """在 preflight/final-validation 期间把 optimizer_spec 的模型解析替换为缓存版本。

    pp_sizes 校验需要模型的 num_hidden_layers；preflight 与 final_validation 两遍展开
    会重复解析同一模型 config（冷缓存时每次都是一次 config-only 下载）。原解析函数是
    纯函数，缓存不改变最终校验结果。
    """
    from grid_generator import optimizer_spec

    original = getattr(optimizer_spec, "_resolve_num_hidden_layers", None)
    if original is None:
        raise SystemExit(
            "grid_generator.optimizer_spec._resolve_num_hidden_layers not found; "
            "build_cases memoization depends on this private helper — "
            "update this skill after upstream refactors"
        )
    cache = _NUM_HIDDEN_LAYERS_CACHE

    def cached(model_id: str) -> int:
        if model_id not in cache:
            cache[model_id] = original(model_id)
        return cache[model_id]

    optimizer_spec._resolve_num_hidden_layers = cached
    try:
        yield
    finally:
        optimizer_spec._resolve_num_hidden_layers = original


def preflight_workloads(workloads: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    """Spec-level preflight: reuse the optimizer's candidate resolution per workload.

    Mirrors what load_optimizer_spec does per entry (scenario build + parallel
    candidate expansion). Infeasible combinations and duplicate workload ids are
    dropped before writing the yaml.
    """
    try:
        from grid_generator.optimizer_spec import (
            _build_source_scenario,
            _expand_with_optimizer_candidates,
        )
    except ImportError as error:
        raise SystemExit(
            f"grid_generator.optimizer_spec private API changed; update build_cases preflight (missing: {error})"
        ) from error

    survivors: list[dict] = []
    dropped: list[str] = []
    notes: list[str] = []
    seen_ids: set[str] = set()
    for entry in workloads:
        label = str(entry.get("name", f"workload{len(survivors) + 1}"))
        try:
            source = _build_source_scenario(label, entry)
            expanded = _expand_with_optimizer_candidates(source, entry)
        except Exception as error:  # noqa: BLE001 - any spec/candidate error drops the combo
            dropped.append(f"{label}: {type(error).__name__}: {error}")
            continue
        ids = [scenario.workload_id for scenario in expanded]
        unique_ids = [wid for wid in ids if wid not in seen_ids]
        if not unique_ids:
            dropped.append(f"{label}: all expanded workload ids duplicate an earlier combo")
            continue
        seen_ids.update(unique_ids)
        survivors.append(entry)
        notes.append(f"{label}: expanded={len(expanded)} unique={len(unique_ids)}")
    return survivors, dropped, notes


def write_yaml(path: Path, workloads: list[dict], header_lines: list[str]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"workloads": workloads}
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)
    header = "\n".join(f"# {line}" for line in header_lines)
    path.write_text(header + "\n" + text, encoding="utf-8")


def final_validation(path: Path) -> tuple[int, int]:
    from grid_generator.optimizer_spec import load_optimizer_spec

    spec = load_optimizer_spec(path)
    total = sum(summary.expanded_workloads for summary in spec.workload_summaries)
    skipped = sum(summary.skipped_combinations for summary in spec.workload_summaries)
    return total, skipped


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build optimizer-args cases yaml for generate_shape_grid --optimizer-args-file. "
            "Mode 1: --command (full text_generate command). Mode 2: --set (partial params x default grids)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, required=True, help="Output workload spec yaml path.")
    parser.add_argument("--name", default=None, help="Mode 1 workload name (default derived from model/lengths).")
    parser.add_argument(
        "--command",
        default=None,
        help="Mode 1: full text_generate command line (launcher prefix optional).",
    )
    parser.add_argument("--command-file", type=Path, default=None, help="Mode 1: file containing the command line.")
    parser.add_argument(
        "--output-length",
        type=int,
        default=DEFAULT_OUTPUT_LENGTH,
        help="Mode 1: spec output_length (text_generate has no such flag).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Mode 2: fixed spec field (repeatable; value parsed as YAML).",
    )
    parser.add_argument("--input-length-grid", default="1024,8192,65536,131072", help="Mode 2: input_length grid.")
    parser.add_argument("--output-length-grid", default="1024,2048", help="Mode 2: output_length grid.")
    parser.add_argument("--batch-range-grid", default="1", help="Mode 2: decode batch grid (each value = exact batch).")
    parser.add_argument("--quantize-grid", default="W8A8_DYNAMIC", help="Mode 2: quantize_linear_action grid.")
    parser.add_argument(
        "--max-workloads",
        type=int,
        default=64,
        help="Mode 2: cap on generated combos; overflow is deterministically subsampled.",
    )
    parser.add_argument("--name-prefix", default=None, help="Mode 2: workload name prefix (default: model short name).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing the yaml.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    warnings: list[str] = []
    workloads: list[dict] = []

    if args.command or args.command_file:
        if args.set:
            raise SystemExit("--command/--command-file (mode 1) and --set (mode 2) are mutually exclusive")
        text = args.command
        if args.command_file:
            text = args.command_file.read_text(encoding="utf-8")
        if not text:
            raise SystemExit("empty command")
        argv = strip_launcher_prefix(split_command(text))
        parsed = parse_text_generate_command(argv)
        name = args.name
        workloads = [map_args_to_spec(parsed, warnings, args.output_length, name or "workload")]
        if not args.name:
            entry = workloads[0]
            workloads[0]["name"] = (
                f"{str(entry['model']).split('/')[-1]}-in{entry['input_length']}-out{entry['output_length']}"
            )
    elif args.set:
        workloads, warnings = build_mode2_workloads(args)
    else:
        raise SystemExit("provide --command/--command-file (mode 1) or --set (mode 2)")

    for message in warnings:
        print(f"[build_cases] WARNING: {message}")

    dropped: list[str] = []
    notes: list[str] = []
    if not args.dry_run and workloads:
        with _memoized_model_config_lookup():
            workloads, dropped, notes = preflight_workloads(workloads)
        for note in notes:
            print(f"[build_cases] preflight: {note}")
        for item in dropped:
            print(f"[build_cases] preflight dropped: {item}")

    if not workloads:
        raise SystemExit("no feasible workload survived; nothing written")

    if args.dry_run:
        import yaml

        print("[build_cases] dry-run payload:")
        print(yaml.safe_dump({"workloads": workloads}, sort_keys=False, allow_unicode=True))
        return

    header_lines = [
        f"generated by profiling_database_shape_generate/scripts/build_cases.py ({'command' if args.command or args.command_file else 'grid'} mode)",
        "consumed by generate_shape_grid --optimizer-args-file; each workload covers prefill + decode demands",
    ]
    write_yaml(args.output.resolve(), workloads, header_lines)
    with _memoized_model_config_lookup():
        total, skipped = final_validation(args.output.resolve())
    print(
        f"[build_cases done] wrote {len(workloads)} workload(s) to {args.output.resolve()}; "
        f"loader expansion: {total} runnable workload(s), {skipped} illegal combination(s) auto-skipped"
    )


if __name__ == "__main__":
    main()

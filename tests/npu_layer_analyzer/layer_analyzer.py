#!/usr/bin/env python3
"""Layer Analyzer — 通用的 Transformer 层结构标注 + 代表层提取工具。

合并了 layer_marker.py（全局标注）和 layer_extractor.py（层提取 + 子结构标注）。

功能：
  默认模式：全局标注 + 层提取 — 标记每行的 Layer / Marker / Stage / Is_Key，并自动选取代表层
  --no-layer：禁用层提取，只做全局标注
  --no-global：禁用全局标注，只输出层提取 CSV

通用标记规则（正则匹配 Name + Type + Full Name）：
  EMBED   — embedding / embed                 橙色
  ATT     — attention / mla / ring_mla         金黄
  NORM    — rmsnorm / layernorm / norm         浅黄
  MLP     — swiglu / silu / gelu / ffn / mlp   浅绿
  MATMUL  — matmul                             浅蓝
  LINEAR  — linear_all_reduce                  浅蓝
  COMM    — all_gather / allreduce / send       浅粉
  SAMPLE  — downsample / upsample              浅紫

Stage 标注（Stage 列）：
  Attention     → Attention 阶段
  FFN           → Dense 层的 FFN 阶段（AddRmsNormBias → MLP → RmsNorm）
  MOE           → MoE 层的 MOE 阶段（AddRmsNormBias → MoE → RmsNorm）

用法：
  # 全局标注 + 层提取（默认行为）
  python layer_analyzer.py -i kernel_details.csv --delimiter attention

  # 指定层号
  python layer_analyzer.py -i kernel_details.csv --layer-index 5

输出文件名带"仿真"前缀：
  仿真<base>_layered.csv   # 全局标注
  仿真<base>_layer.csv     # 代表层 + 子结构标注
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import layer_common as lc
from layer_common import (
    MOE_RE,
    detect_marker,
    extract_substructure,
    mark_unfused_rmsnorm,
    pick_representative_layer,
    print_structure_summary,
    refine_sub_blocks,
    write_layer_csv,
)

# ═══════════════════════════════════════════════════════════════════════════
# 常量与正则
# ═══════════════════════════════════════════════════════════════════════════

# 输出文件名前缀：仿真侧为 tensor_cast 仿真数据
OUTPUT_PREFIX = "仿真"


# ═══════════════════════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════════════════════


def compile_rules() -> list[tuple[re.Pattern, str]]:
    """从 layer_common.MARKER_RULES 编译标记规则。"""
    return [(re.compile(pattern, re.IGNORECASE), marker) for pattern, marker, _ in lc.MARKER_RULES]


def find_marker_indices(
    rows: list[dict],
    compiled_rules: list[tuple[re.Pattern, str]],
    target_marker: str,
    stream: str = "0",
) -> list[int]:
    indices = []
    for i, r in enumerate(rows):
        if r.get("Stream ID") != stream:
            continue
        # 优先使用已有的 Marker 字段（如 mark_unfused_rmsnorm 标记的）
        marker = r.get("Marker", "")
        if not marker:
            searchable = r.get("Full Name", "") + " " + r.get("Name", "") + " " + r.get("Type", "")
            marker = detect_marker(searchable, compiled_rules)
        if marker == target_marker:
            indices.append(i)
    return indices


def auto_select_stream(rows: list[dict]) -> str:
    stream_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        stream_counts[r.get("Stream ID", "")] += 1
    return max(stream_counts, key=stream_counts.get)


# ═══════════════════════════════════════════════════════════════════════════
# 层号分配（来自 layer_marker.py）
# ═══════════════════════════════════════════════════════════════════════════


def find_layer_start_norms(
    rows: list[dict],
    compiled_rules: list[tuple[re.Pattern, str]],
    stream: str = "0",
    lookahead: int = 30,
) -> list[int]:
    norm_indices = find_marker_indices(rows, compiled_rules, "NORM", stream=stream)
    att_indices = find_marker_indices(rows, compiled_rules, "ATT", stream=stream)
    att_set = set(att_indices)

    # 策略1：找 pre-attention NORM
    layer_start_norms = []
    for norm_idx in norm_indices:
        is_pre_att = False
        for j in range(norm_idx + 1, min(norm_idx + 1 + lookahead, len(rows))):
            if rows[j].get("Stream ID") != stream:
                continue
            if j in att_set:
                is_pre_att = True
                break
            searchable = rows[j].get("Full Name", "") + " " + rows[j].get("Name", "") + " " + rows[j].get("Type", "")
            if detect_marker(searchable, compiled_rules) == "NORM":
                break
        if is_pre_att:
            layer_start_norms.append(norm_idx)

    if layer_start_norms:
        return layer_start_norms

    # 策略2：按 NORM/ATT 比例计算
    if not norm_indices or not att_indices:
        return norm_indices
    norms_per_layer = max(1, round(len(norm_indices) / len(att_indices)))
    return norm_indices[::norms_per_layer]


def assign_layer_index_by_norm(
    rows: list[dict],
    compiled_rules: list[tuple[re.Pattern, str]],
    stream: str = "0",
    lookahead: int = 30,
) -> list[int]:
    n = len(rows)
    layer_indices = [-1] * n
    layer_start_norms = find_layer_start_norms(rows, compiled_rules, stream=stream, lookahead=lookahead)
    if not layer_start_norms:
        layer_start_norms = find_marker_indices(rows, compiled_rules, "NORM", stream=stream)
    if not layer_start_norms:
        return layer_indices

    for seq_idx, start_norm in enumerate(layer_start_norms):
        next_start = layer_start_norms[seq_idx + 1] if seq_idx + 1 < len(layer_start_norms) else n
        for i in range(start_norm, next_start):
            if rows[i].get("Stream ID") == stream:
                layer_indices[i] = seq_idx

    # 其它 stream 按时间窗口
    layer_time_ranges: dict[int, tuple[float, float]] = {}
    for li in range(len(layer_start_norms)):
        layer_rows_i = [rows[i] for i in range(n) if layer_indices[i] == li]
        if layer_rows_i:
            starts = [float(r.get("Start Time(us)", 0)) for r in layer_rows_i]
            ends = [float(r.get("Start Time(us)", 0)) + float(r.get("Duration(us)", 0)) for r in layer_rows_i]
            layer_time_ranges[li] = (min(starts), max(ends))

    for i in range(n):
        if layer_indices[i] >= 0:
            continue
        ts = float(rows[i].get("Start Time(us)", 0))
        for li, (t_start, t_end) in layer_time_ranges.items():
            if t_start <= ts <= t_end:
                layer_indices[i] = li
                break

    return layer_indices


def assign_layer_index_by_attention(
    rows: list[dict],
    compiled_rules: list[tuple[re.Pattern, str]],
    stream: str = "0",
) -> list[int]:
    n = len(rows)
    layer_indices = [-1] * n
    att_indices = find_marker_indices(rows, compiled_rules, "ATT", stream=stream)
    if not att_indices:
        return layer_indices

    for seq_idx, att_row in enumerate(att_indices):
        next_att = att_indices[seq_idx + 1] if seq_idx + 1 < len(att_indices) else n
        for i in range(att_row, next_att):
            if rows[i].get("Stream ID") == stream:
                layer_indices[i] = seq_idx

    # 其它 stream
    layer_time_ranges: dict[int, tuple[float, float]] = {}
    for li in range(len(att_indices)):
        layer_rows_i = [rows[i] for i in range(n) if layer_indices[i] == li]
        if layer_rows_i:
            starts = [float(r.get("Start Time(us)", 0)) for r in layer_rows_i]
            ends = [float(r.get("Start Time(us)", 0)) + float(r.get("Duration(us)", 0)) for r in layer_rows_i]
            layer_time_ranges[li] = (min(starts), max(ends))

    for i in range(n):
        if layer_indices[i] >= 0:
            continue
        ts = float(rows[i].get("Start Time(us)", 0))
        for li, (t_start, t_end) in layer_time_ranges.items():
            if t_start <= ts <= t_end:
                layer_indices[i] = li
                break

    return layer_indices


# ═══════════════════════════════════════════════════════════════════════════
# 子结构标注（refine_sub_blocks / extract_substructure）和代表层选取
# （pick_representative_layer）已抽取到 layer_common.py，两个 analyzer 共享。
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# MoE 检测
# ═══════════════════════════════════════════════════════════════════════════


def detect_moe_layers(rows: list[dict], layer_indices: list[int], stream: str) -> set[int]:
    moe_layers = set()
    for i, r in enumerate(rows):
        li = layer_indices[i]
        if li < 0:
            continue
        searchable = r.get("Full Name", "") + " " + r.get("Name", "") + " " + r.get("Type", "")
        if MOE_RE.search(searchable):
            moe_layers.add(li)
    return moe_layers


# ═══════════════════════════════════════════════════════════════════════════
# 输出：CSV
# ═══════════════════════════════════════════════════════════════════════════


def write_annotated_csv(output_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ═══════════════════════════════════════════════════════════════════════════
# 打印摘要
# ═══════════════════════════════════════════════════════════════════════════


def print_layer_summary(rows: list[dict], delimiter_name: str) -> None:
    layers: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        li = int(r.get("Layer", -1))
        if li >= 0:
            layers[li].append(r)

    print(f"\n{'=' * 60}")
    print(f"  共 {len(layers)} 层（以 {delimiter_name} 为边界）")
    print(f"{'=' * 60}")

    for li in sorted(layers.keys()):
        layer_rows = layers[li]
        markers = [r["Marker"] for r in layer_rows if r.get("Marker")]
        mc: dict[str, int] = {}
        for m in markers:
            mc[m] = mc.get(m, 0) + 1
        ts0 = layer_rows[0].get("Start Time(us)", "?")
        ts1 = layer_rows[-1].get("Start Time(us)", "?")
        marker_str = " ".join(f"{m}x{c}" for m, c in sorted(mc.items()))
        print(f"  Layer {li:3d} | {len(layer_rows):5d} ops | ts {ts0:>10s} ~ {ts1:>10s} | {marker_str}")

    outside = [r for r in rows if int(r.get("Layer", -1)) < 0]
    if outside:
        mc: dict[str, int] = {}
        for r in outside:
            m = r.get("Marker", "")
            if m:
                mc[m] = mc.get(m, 0) + 1
        marker_str = " ".join(f"{m}x{c}" for m, c in sorted(mc.items()))
        print(f"  前处理   | {len(outside):5d} ops | {marker_str}")


# ═══════════════════════════════════════════════════════════════════════════
# 层裁剪
# ═══════════════════════════════════════════════════════════════════════════


def _trim_to_target_layer(rows: list[dict], layer_index: int, layer_indices: list[int]) -> list[dict]:
    """根据 layer_index 裁剪 rows，只保留目标层的范围。

    直接使用 Layer 字段过滤，避免 attention 位置与层序号的索引空间不一致。
    """
    result = [r for r in rows if r.get("Layer") == str(layer_index)]
    return result if result else list(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Layer Analyzer — 通用 Transformer 层结构标注 + 代表层提取。")
    # 通用参数
    parser.add_argument("--input", "-i", required=True, help="输入 kernel_details.csv")
    parser.add_argument("--output", "-o", default=None, help="输出路径前缀（默认自动生成）")
    parser.add_argument("--main-stream", default=None, help="主 stream ID（默认自动选择）")
    parser.add_argument(
        "--delimiter",
        choices=["norm", "attention"],
        default="attention",
        help="层边界锚点（默认 attention，即以 Attention 为一层开头）",
    )
    parser.add_argument("--attention-pattern", default=None, help="自定义 ATT 匹配正则")
    parser.add_argument("--norm-pattern", default=None, help="自定义 NORM 匹配正则")
    parser.add_argument("--lookahead", type=int, default=30, help="判断 pre-attention NORM 前瞻行数")

    # 层提取参数
    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="指定 Dense 层号（默认自动选取前 1/3）",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"输入文件不存在: {input_path}")

    # 输出路径
    base_stem = input_path.stem.replace("_layered", "").replace("_marked", "").replace("_layer", "")
    if args.output:
        output_prefix = Path(args.output)
        # 如果提供了 -o，用它的 stem 作为 base_stem
        base_stem = output_prefix.stem
    else:
        output_prefix = input_path.parent / base_stem

    # 编译标记规则
    compiled = compile_rules()
    if args.attention_pattern:
        compiled.insert(0, (re.compile(args.attention_pattern, re.IGNORECASE), "ATT"))
    if args.norm_pattern:
        compiled.insert(0, (re.compile(args.norm_pattern, re.IGNORECASE), "NORM"))

    # 读取
    with input_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit("输入 CSV 为空。")

    for r in rows:
        r.setdefault("Full Name", r.get("Name", ""))

    # 自动选择主 stream
    main_stream = args.main_stream
    if main_stream is None:
        main_stream = auto_select_stream(rows)
        print(f"主 stream: {main_stream}")

    # ── 1. 标记每行 Marker ──
    for r in rows:
        searchable = r.get("Full Name", "") + " " + r.get("Name", "") + " " + r.get("Type", "")
        r["Marker"] = detect_marker(searchable, compiled)

    # 识别未融合 RMSNorm 序列（以 rsqrt 为锚点，标记前后相关算子为 NORM）
    rows = mark_unfused_rmsnorm(rows)

    norm_count = sum(1 for r in rows if r.get("Stream ID") == main_stream and r["Marker"] == "NORM")
    att_count = sum(1 for r in rows if r.get("Stream ID") == main_stream and r["Marker"] == "ATT")
    print(f"输入: {input_path}")
    print(f"行数: {len(rows)}")
    print(f"NORM: {norm_count} | ATT: {att_count}")

    # ── 2. 分配层号 ──
    if args.delimiter == "norm":
        if norm_count == 0:
            raise SystemExit("未找到 NORM 算子，请用 --norm-pattern 指定。")
        layer_indices = assign_layer_index_by_norm(rows, compiled, stream=main_stream, lookahead=args.lookahead)
        delimiter_name = "RMSNorm/LayerNorm"
    else:
        if att_count == 0:
            raise SystemExit("未找到 ATT 算子，请用 --attention-pattern 指定。")
        layer_indices = assign_layer_index_by_attention(rows, compiled, stream=main_stream)
        delimiter_name = "Attention"

    for i, r in enumerate(rows):
        r["Layer"] = str(layer_indices[i])

    # ── 3. 全局标注输出 ──
    output_dir = output_prefix.parent if args.output else input_path.parent
    csv_output = output_dir / f"{OUTPUT_PREFIX}{base_stem}_layered.csv"
    write_annotated_csv(csv_output, rows)
    print(f"CSV: {csv_output}")

    print_layer_summary(rows, delimiter_name)

    marker_count = sum(1 for r in rows if r.get("Marker"))
    print(f"\n标记行: {marker_count} / {len(rows)}")

    # ── 4. 层提取 + 子结构标注 ──
    # MoE 检测
    moe_layers = detect_moe_layers(rows, layer_indices, main_stream)
    has_moe = bool(moe_layers)
    if has_moe:
        print(f"\n检测到 MoE 层: {sorted(moe_layers)}")
    else:
        print("\n纯 Dense 模型")

    # 检测含 ATT 的层（与 npu_layer_analyzer.py 一致）
    att_layers = {layer_indices[i] for i in range(len(rows)) if layer_indices[i] >= 0 and lc.is_attention(rows[i])}
    picks = pick_representative_layer(layer_indices, moe_layers, args.layer_index, att_layers)
    print(f"选取代表层: Dense={picks['dense']}" + (f", MoE={picks['moe']}" if picks["moe"] is not None else ""))

    for kind, layer_idx in [("dense", picks["dense"]), ("moe", picks["moe"])]:
        if layer_idx is None:
            continue

        # 按 layer_idx 裁剪到目标层范围
        layer_rows = _trim_to_target_layer(rows, layer_idx, layer_indices)

        # 标注子结构
        layer_rows = refine_sub_blocks(layer_rows)
        # 裁剪：从上下采样到上下采样，标注 Stage
        layer_rows = extract_substructure(layer_rows, is_moe=(kind == "moe"))

        # 输出文件名：与 npu_layer_analyzer.py 一致
        if has_moe:
            fname = f"{OUTPUT_PREFIX}{base_stem}_layer{layer_idx}_{kind}.csv"
        else:
            fname = f"{OUTPUT_PREFIX}{base_stem}_layer{layer_idx}.csv"
        output_path = output_dir / fname
        write_layer_csv(output_path, layer_rows)
        print(f"\n层 CSV: {output_path}")

        print_structure_summary(layer_rows, layer_idx, kind.upper())

    # 使用提示
    print("\n--- CSV 查看提示 ---")
    print("1. 筛选 Is_Key = ★ → 只看关键边界算子（RMSNorm / Attention / MLP）")
    print("2. 按 Stage 列筛选 → 查看各阶段（Attention / FFN 或 MOE）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

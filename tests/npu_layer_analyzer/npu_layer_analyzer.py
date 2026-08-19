#!/usr/bin/env python3
"""NPU Layer Analyzer — Forward 切分 + 层内子结构标注（一体化工具）。

基于 npu_forward_segmenter.py 的 Forward 切分逻辑，
增加层提取 + 子结构标注（RMSNorm → Attention → RMSNorm → MLP）。

功能：
  1. 按 embedding 锚点切分出每次 forward
  2. 自动检测 Dense / MoE 模型
  3. 选取 1 层 Dense 代表（MoE 模型额外选 1 层 MoE 代表）
  4. 标注层内 Stage 和 Is_Key，输出 CSV

用法：
  # 完整流程：切 forward + 提取层 + 标注子结构
  python npu_layer_analyzer.py -i kernel_details.csv

  # 按 task-id 定位特定 forward
  python npu_layer_analyzer.py -i kernel_details.csv --task-id 41500

输出：
  forward_segments/
  ├── 真实summary.csv                            # Forward 汇总
  ├── 真实forward_001.csv                        # Forward 1 全部算子
  ├── 真实forward_001_prefill_layer.csv          # prefill 代表层 + 子结构标注
  ├── 真实forward_001_prefill_layer_dense.csv    # Dense 代表层（仅 MoE 模型）
  ├── 真实forward_001_prefill_layer_moe.csv      # MoE 代表层（仅 MoE 模型）
  └── ...
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import layer_common as lc
from layer_common import (
    MOE_RE,
    detect_marker,
    extract_substructure,
    is_attention,
    mark_unfused_rmsnorm,
    pick_representative_layer,
    print_structure_summary,
    refine_sub_blocks,
    write_layer_csv,
)

# ═══════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_COLUMNS = (
    "Stream ID",
    "Task ID",
    "Name",
    "Type",
    "Start Time(us)",
    "Duration(us)",
    "Input Shapes",
    "Output Shapes",
)

# Gap threshold auto-calculation constants (replacing magic numbers)
MIN_LARGE_GAP_US = 1000.0
GAP_RATIO_THRESHOLD = 5.0
MEDIAN_GAP_MULTIPLIER = 20.0
P95_GAP_MULTIPLIER = 3.0

# 输出文件名前缀：NPU 侧为真实 profiling 数据
OUTPUT_PREFIX = "真实"


# ═══════════════════════════════════════════════════════════════════════════
# 数据结构（复用自 npu_forward_segmenter.py）
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class KernelRow:
    raw: dict[str, str]
    line_no: int
    source_index: int
    stream_id: str
    task_id: str
    name: str
    op_type: str
    start_us: float
    duration_us: float
    end_us: float
    is_attention: bool
    is_embedding: bool
    full_name: str

    @property
    def searchable_text(self) -> str:
        return f"{self.name} {self.op_type}"


@dataclass
class Segment:
    index: int
    method: str
    main_stream: str
    start_us: float
    end_us: float
    main_rows: list[KernelRow]
    output_rows: list[KernelRow]
    split_reason: str
    boundary_gap_us: float = 0.0
    boundary_gap_before_task: str = ""
    boundary_gap_after_task: str = ""
    segment_kind: str = "unknown"
    is_valid: bool = False
    validity_reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Forward 切分（复用自 npu_forward_segmenter.py）
# ═══════════════════════════════════════════════════════════════════════════


def parse_float(value: str, *, default: float = 0.0) -> float:
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return default
    return float(text)


def compile_pattern(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise SystemExit(f"Invalid regex pattern {pattern!r}: {exc}") from exc


def read_rows(
    input_path: Path,
    attention_pattern: re.Pattern[str],
    embedding_pattern: re.Pattern[str],
) -> tuple[list[str], list[KernelRow]]:
    with input_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"{input_path} does not look like a CSV file.")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{input_path} is missing required columns: {', '.join(missing)}")

        rows: list[KernelRow] = []
        for source_index, raw in enumerate(reader):
            start_us = parse_float(raw["Start Time(us)"])
            duration_us = parse_float(raw["Duration(us)"])
            searchable_text = f"{raw.get('Name', '')} {raw.get('Type', '')}"
            full_name = raw.get("Full Name", raw.get("Name", ""))
            rows.append(
                KernelRow(
                    raw=raw,
                    line_no=source_index + 2,
                    source_index=source_index,
                    stream_id=raw["Stream ID"].strip(),
                    task_id=raw["Task ID"].strip(),
                    name=raw["Name"],
                    op_type=raw["Type"],
                    start_us=start_us,
                    duration_us=duration_us,
                    end_us=start_us + duration_us,
                    is_attention=bool(attention_pattern.search(searchable_text)),
                    is_embedding=bool(embedding_pattern.search(searchable_text)),
                    full_name=full_name,
                )
            )
    return reader.fieldnames, rows


def choose_main_stream(rows: Sequence[KernelRow]) -> str:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        stats[row.stream_id]["rows"] += 1
        if row.is_attention:
            stats[row.stream_id]["attention"] += 1
        if row.is_embedding:
            stats[row.stream_id]["embedding"] += 1

    if not stats:
        raise SystemExit("No rows found in input CSV.")

    def score(item: tuple[str, Counter[str]]) -> tuple[int, int, int, int]:
        stream_id, counter = item
        not_na = 0 if stream_id.upper() == "N/A" else 1
        return (
            counter["attention"],
            counter["embedding"],
            not_na,
            counter["rows"],
        )

    return max(stats.items(), key=score)[0]


def format_stream_stats(rows: Sequence[KernelRow]) -> str:
    counts = Counter(row.stream_id for row in rows)
    return ";".join(f"{stream}:{count}" for stream, count in counts.most_common())


def event_overlaps(row: KernelRow, start_us: float, end_us: float) -> bool:
    return row.start_us < end_us and row.end_us > start_us


def gap_details(rows: Sequence[KernelRow]) -> tuple[float, str, str]:
    if len(rows) < 2:
        return 0.0, "", ""
    max_gap = -1.0
    before_task = ""
    after_task = ""
    sorted_rows = sorted(rows, key=lambda row: (row.start_us, row.source_index))
    for previous, current in zip(sorted_rows, sorted_rows[1:]):
        gap = current.start_us - previous.end_us
        if gap > max_gap:
            max_gap = gap
            before_task = previous.task_id
            after_task = current.task_id
    return max(max_gap, 0.0), before_task, after_task


def trim_at_large_gap(
    rows: Sequence[KernelRow],
    threshold_us: float,
) -> tuple[list[KernelRow], float, str, str]:
    sorted_rows = sorted(rows, key=lambda row: (row.start_us, row.source_index))
    if threshold_us == float("inf"):
        return sorted_rows, 0.0, "", ""

    for index, (previous, current) in enumerate(zip(sorted_rows, sorted_rows[1:])):
        gap = current.start_us - previous.end_us
        if gap > threshold_us:
            return sorted_rows[: index + 1], gap, previous.task_id, current.task_id
    return sorted_rows, 0.0, "", ""


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def auto_gap_threshold(main_rows: Sequence[KernelRow]) -> float:
    positive_gaps: list[float] = []
    sorted_rows = sorted(main_rows, key=lambda row: (row.start_us, row.source_index))
    for previous, current in zip(sorted_rows, sorted_rows[1:]):
        gap = current.start_us - previous.end_us
        if gap > 0:
            positive_gaps.append(gap)

    if not positive_gaps:
        return float("inf")
    if len(positive_gaps) == 1:
        return positive_gaps[0] + 1.0

    descending = sorted(positive_gaps, reverse=True)
    best_ratio = 0.0
    best_threshold: float | None = None
    for high, low in zip(descending, descending[1:]):
        if low <= 0:
            continue
        ratio = high / low
        if high >= MIN_LARGE_GAP_US and ratio > best_ratio:
            best_ratio = ratio
            best_threshold = (high + low) / 2.0

    if best_threshold is not None and best_ratio >= GAP_RATIO_THRESHOLD:
        return best_threshold

    median_gap = statistics.median(positive_gaps)
    p95_gap = percentile(positive_gaps, 0.95)
    return max(
        MIN_LARGE_GAP_US,
        median_gap * MEDIAN_GAP_MULTIPLIER,
        p95_gap * P95_GAP_MULTIPLIER,
    )


def split_main_rows_by_gap(
    main_rows: Sequence[KernelRow],
    gap_threshold_us: float,
) -> list[tuple[list[KernelRow], float, str, str]]:
    sorted_main_rows = sorted(main_rows, key=lambda row: (row.start_us, row.source_index))
    if not sorted_main_rows:
        return []

    chunks: list[tuple[list[KernelRow], float, str, str]] = []
    current_chunk: list[KernelRow] = [sorted_main_rows[0]]
    for previous, current in zip(sorted_main_rows, sorted_main_rows[1:]):
        gap = current.start_us - previous.end_us
        if gap > gap_threshold_us:
            chunks.append((current_chunk, gap, previous.task_id, current.task_id))
            current_chunk = [current]
        else:
            current_chunk.append(current)
    chunks.append((current_chunk, 0.0, "", ""))
    return chunks


def split_uncovered_main_rows_by_gap(
    main_rows: Sequence[KernelRow],
    covered_source_indexes: set[int],
    gap_threshold_us: float,
) -> list[tuple[list[KernelRow], float, str, str]]:
    sorted_main_rows = sorted(main_rows, key=lambda row: (row.start_us, row.source_index))
    if not sorted_main_rows:
        return []

    chunks: list[tuple[list[KernelRow], float, str, str]] = []
    current_chunk: list[KernelRow] = []
    for row in sorted_main_rows:
        if row.source_index in covered_source_indexes:
            if current_chunk:
                chunks.append((current_chunk, 0.0, "", ""))
                current_chunk = []
            continue

        if not current_chunk:
            current_chunk = [row]
            continue

        previous = current_chunk[-1]
        gap = row.start_us - previous.end_us
        if gap > gap_threshold_us:
            chunks.append((current_chunk, gap, previous.task_id, row.task_id))
            current_chunk = [row]
        else:
            current_chunk.append(row)

    if current_chunk:
        chunks.append((current_chunk, 0.0, "", ""))
    return chunks


def segment_output_rows(
    rows: Sequence[KernelRow],
    segment_main_rows: Sequence[KernelRow],
    start_us: float,
    end_us: float,
    *,
    include_related_streams: bool,
    related_stream_policy: str,
) -> list[KernelRow]:
    if include_related_streams and related_stream_policy == "time-window":
        output_rows = [row for row in rows if event_overlaps(row, start_us, end_us)]
    else:
        output_rows = list(segment_main_rows)
    return sorted(output_rows, key=lambda row: (row.start_us, row.source_index))


def make_segment(
    rows: Sequence[KernelRow],
    segment_main_rows: Sequence[KernelRow],
    main_stream: str,
    *,
    method: str,
    split_reason: str,
    include_related_streams: bool,
    related_stream_policy: str,
    start_us: float | None = None,
    end_us: float | None = None,
    boundary_gap_us: float = 0.0,
    boundary_gap_before_task: str = "",
    boundary_gap_after_task: str = "",
) -> Segment | None:
    sorted_main_rows = sorted(segment_main_rows, key=lambda row: (row.start_us, row.source_index))
    if not sorted_main_rows:
        return None

    segment_start_us = start_us if start_us is not None else min(row.start_us for row in sorted_main_rows)
    segment_end_us = end_us if end_us is not None else max(row.end_us for row in sorted_main_rows)
    if segment_end_us <= segment_start_us:
        segment_end_us = max(row.end_us for row in sorted_main_rows)

    return Segment(
        index=0,
        method=method,
        main_stream=main_stream,
        start_us=segment_start_us,
        end_us=segment_end_us,
        main_rows=sorted_main_rows,
        output_rows=segment_output_rows(
            rows,
            sorted_main_rows,
            segment_start_us,
            segment_end_us,
            include_related_streams=include_related_streams,
            related_stream_policy=related_stream_policy,
        ),
        split_reason=split_reason,
        boundary_gap_us=boundary_gap_us,
        boundary_gap_before_task=boundary_gap_before_task,
        boundary_gap_after_task=boundary_gap_after_task,
    )


def classify_and_validate_segment(
    segment: Segment,
    *,
    expected_attention: int,
    attention_tolerance: int,
    gap_threshold_us: float,
) -> None:
    attention_main = sum(1 for row in segment.main_rows if row.is_attention)
    # embedding 可能在非主 stream 上，检查 output_rows（所有 stream）
    embedding_main = sum(1 for row in segment.output_rows if row.is_embedding)
    max_gap, _, _ = gap_details(segment.main_rows)

    attention_ok = expected_attention <= 0 or abs(attention_main - expected_attention) <= attention_tolerance
    internal_gap_checked = segment.method != "embedding-to-embedding"
    internal_gap_ok = not internal_gap_checked or gap_threshold_us == float("inf") or max_gap <= gap_threshold_us
    boundary_ok = segment.method in {
        "embedding",
        "embedding-to-embedding",
        "embedding-to-gap",
        "gap",
    } and bool(segment.main_rows)

    if embedding_main > 0:
        segment.segment_kind = "prefill"
    elif segment.method == "gap" and attention_ok:
        segment.segment_kind = "decode"
    else:
        segment.segment_kind = "unknown"

    reasons: list[str] = []
    if not attention_ok:
        reasons.append(
            f"attention mismatch: got {attention_main}, expected {expected_attention} +/- {attention_tolerance}"
        )
    if internal_gap_checked and not internal_gap_ok:
        reasons.append(f"internal gap {max_gap:.3f} us exceeds {gap_threshold_us:.3f} us")
    if not boundary_ok:
        reasons.append("boundary not trusted")

    segment.is_valid = attention_ok and internal_gap_ok and boundary_ok
    segment.validity_reason = "ok" if segment.is_valid else "; ".join(reasons)


def build_segments(
    rows: Sequence[KernelRow],
    main_rows: Sequence[KernelRow],
    main_stream: str,
    *,
    gap_threshold_us: float,
    include_related_streams: bool,
    related_stream_policy: str,
    expected_attention: int,
    attention_tolerance: int,
) -> list[Segment]:
    segments: list[Segment] = []
    sorted_main_rows = sorted(main_rows, key=lambda row: (row.start_us, row.source_index))
    embeddings = sorted(
        [row for row in sorted_main_rows if row.is_embedding],
        key=lambda row: (row.start_us, row.source_index),
    )
    covered_source_indexes: set[int] = set()

    for embedding_index, anchor in enumerate(embeddings):
        next_embedding = embeddings[embedding_index + 1] if embedding_index + 1 < len(embeddings) else None
        if next_embedding is not None:
            segment_main_rows = [
                row for row in sorted_main_rows if anchor.start_us <= row.start_us < next_embedding.start_us
            ]
            method = "embedding-to-embedding"
            split_reason = f"embedding task {anchor.task_id} to next embedding task {next_embedding.task_id}"
            boundary_gap_us = 0.0
            boundary_before_task = ""
            boundary_after_task = ""
            end_us = next_embedding.start_us
        else:
            rows_after_anchor = [row for row in sorted_main_rows if row.start_us >= anchor.start_us]
            (
                segment_main_rows,
                boundary_gap_us,
                boundary_before_task,
                boundary_after_task,
            ) = trim_at_large_gap(
                rows_after_anchor,
                gap_threshold_us,
            )
            method = "embedding-to-gap"
            split_reason = f"embedding task {anchor.task_id} to "
            if boundary_before_task and boundary_after_task:
                split_reason += (
                    f"gap task {boundary_before_task}->{boundary_after_task} "
                    f"({boundary_gap_us:.3f} us > {gap_threshold_us:.3f} us)"
                )
            else:
                split_reason += "file end"
            end_us = max(row.end_us for row in segment_main_rows) if segment_main_rows else anchor.end_us

        segment = make_segment(
            rows,
            segment_main_rows,
            main_stream,
            method=method,
            split_reason=split_reason,
            include_related_streams=include_related_streams,
            related_stream_policy=related_stream_policy,
            start_us=anchor.start_us,
            end_us=end_us,
            boundary_gap_us=boundary_gap_us,
            boundary_gap_before_task=boundary_before_task,
            boundary_gap_after_task=boundary_after_task,
        )
        if segment is None:
            continue
        segments.append(segment)
        covered_source_indexes.update(row.source_index for row in segment.main_rows)

    for (
        chunk,
        boundary_gap_us,
        boundary_before_task,
        boundary_after_task,
    ) in split_uncovered_main_rows_by_gap(
        sorted_main_rows,
        covered_source_indexes,
        gap_threshold_us,
    ):
        split_reason = f"gap threshold {gap_threshold_us:.3f} us"
        if boundary_before_task and boundary_after_task:
            split_reason += (
                f"; next boundary gap task {boundary_before_task}->{boundary_after_task} "
                f"({boundary_gap_us:.3f} us > {gap_threshold_us:.3f} us)"
            )
        segment = make_segment(
            rows,
            chunk,
            main_stream,
            method="gap",
            split_reason=split_reason,
            include_related_streams=include_related_streams,
            related_stream_policy=related_stream_policy,
            boundary_gap_us=boundary_gap_us,
            boundary_gap_before_task=boundary_before_task,
            boundary_gap_after_task=boundary_after_task,
        )
        if segment is not None:
            segments.append(segment)

    segments.sort(key=lambda segment: (segment.start_us, segment.main_rows[0].source_index))
    for index, segment in enumerate(segments):
        segment.index = index
        classify_and_validate_segment(
            segment,
            expected_attention=expected_attention,
            attention_tolerance=attention_tolerance,
            gap_threshold_us=gap_threshold_us,
        )
    return segments


# ═══════════════════════════════════════════════════════════════════════════
# 导出策略（复用自 npu_forward_segmenter.py）
# ═══════════════════════════════════════════════════════════════════════════


def attention_status(count: int, expected: int, tolerance: int) -> str:
    if expected <= 0:
        return "not checked"
    delta = abs(count - expected)
    if delta <= tolerance:
        return "ok"
    return f"mismatch expected {expected} +/- {tolerance}"


def write_segment_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[KernelRow]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.raw)


def parse_indexes(indexes: str) -> set[int]:
    selected: set[int] = set()
    if not indexes.strip():
        return selected
    for part in indexes.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise SystemExit(f"Invalid --export-indexes value {text!r}; expected integers.") from exc
        if value < 0:
            raise SystemExit(f"Invalid --export-indexes value {value}; indexes must be >= 0.")
        selected.add(value)
    return selected


def select_segments_for_export(
    segments: Sequence[Segment],
    *,
    export_policy: str,
    export_indexes: str,
    max_per_kind: int,
    task_id: int | None = None,
    forward_kind: str = "prefill",
) -> list[Segment]:
    if export_policy == "all":
        return list(segments)

    if export_policy == "indexes":
        indexes = parse_indexes(export_indexes)
        known_indexes = {segment.index for segment in segments}
        missing = sorted(indexes - known_indexes)
        if missing:
            raise SystemExit(f"--export-indexes contains missing forward indexes: {missing}")
        return [segment for segment in segments if segment.index in indexes]

    if export_policy == "task-id":
        if task_id is None:
            raise SystemExit("--export-policy task-id requires --task-id")
        for segment in segments:
            if not segment.main_rows:
                continue
            start_task = int(segment.main_rows[0].task_id)
            end_task = int(segment.main_rows[-1].task_id)
            if start_task <= task_id <= end_task:
                return [segment]
        raise SystemExit(f"Task ID {task_id} not found in any forward segment")

    if export_policy != "first-valid-by-kind":
        raise SystemExit(f"Unsupported export policy: {export_policy}")

    if max_per_kind < 1:
        raise SystemExit("--max-per-kind must be >= 1")

    selected: list[Segment] = []
    counts: Counter[str] = Counter()
    for segment in segments:
        if not segment.is_valid:
            continue
        if forward_kind != "all" and segment.segment_kind != forward_kind:
            continue
        if counts[segment.segment_kind] >= max_per_kind:
            continue
        selected.append(segment)
        counts[segment.segment_kind] += 1
    return selected


# ═══════════════════════════════════════════════════════════════════════════
# 层提取 + 子结构标注
# 子结构标注逻辑（refine_sub_blocks / extract_substructure / mark_unfused_rmsnorm /
# pick_representative_layer / write_layer_csv / print_structure_summary）已抽取到
# layer_common.py，两个 analyzer 共享，避免重复维护。
# ═══════════════════════════════════════════════════════════════════════════


def kernel_row_to_dict(row: KernelRow) -> dict:
    """将 KernelRow 转为可写的 dict，补充 Full Name。"""
    d = dict(row.raw)
    d["Full Name"] = row.full_name
    return d


def find_layer_anchors(
    segment: Segment,
    main_stream: str,
) -> tuple[list[int], list[int]]:
    """在 segment 的 main_rows 中找到 ATT 和 NORM 的行索引。"""
    att_indices = []
    norm_indices = []
    for i, row in enumerate(segment.main_rows):
        searchable = f"{row.full_name} {row.name} {row.op_type}"
        if row.is_attention or lc.ATT_RE.search(searchable) or lc.RECURRENT_ATT_RE.search(searchable):
            att_indices.append(i)
        elif lc.NORM_RE.search(searchable):
            norm_indices.append(i)
    return att_indices, norm_indices


def assign_layer_index_in_segment(
    segment: Segment,
    main_stream: str,
) -> list[int]:
    """为 segment 的 main_rows 分配层号。以 Attention 为层边界锚点。

    第 N 层 = 第 N 个 Attention 到第 N+1 个 Attention 之前。
    与 _trim_to_target_layer 的切分逻辑保持一致，避免索引空间错位。
    """
    n = len(segment.main_rows)
    layer_indices = [-1] * n

    att_indices, _ = find_layer_anchors(segment, main_stream)

    if not att_indices:
        return layer_indices

    # 直接用 ATT 作为锚点：第 N 层 = [ATT_N, ATT_N+1)
    for seq_idx, att_idx in enumerate(att_indices):
        next_att = att_indices[seq_idx + 1] if seq_idx + 1 < len(att_indices) else n
        for i in range(att_idx, next_att):
            layer_indices[i] = seq_idx

    return layer_indices


def detect_moe_in_segment(
    segment: Segment,
    layer_indices: list[int],
    all_rows: list[KernelRow] | None = None,
) -> set[int]:
    """检测 segment 内哪些层是 MoE 层。

    同时检查 main_rows 和 all_rows（全部算子），非主流算子按时间重叠归属到对应层。
    """
    moe_layers = set()

    # 1. 检查主流算子
    for i, row in enumerate(segment.main_rows):
        li = layer_indices[i]
        if li < 0:
            continue
        searchable = f"{row.full_name} {row.name} {row.op_type}"
        if MOE_RE.search(searchable):
            moe_layers.add(li)

    # 2. 检查全部算子（包括非主流，如 GroupedMatmul 可能在其他 stream）
    if all_rows:
        # 构建主流层的时间范围
        layer_time_ranges: dict[int, tuple[float, float]] = {}
        for i, row in enumerate(segment.main_rows):
            li = layer_indices[i]
            if li < 0:
                continue
            if li not in layer_time_ranges:
                layer_time_ranges[li] = (row.start_us, row.end_us)
            else:
                t_start, t_end = layer_time_ranges[li]
                layer_time_ranges[li] = (
                    min(t_start, row.start_us),
                    max(t_end, row.end_us),
                )

        if layer_time_ranges:
            main_source_indexes = {row.source_index for row in segment.main_rows}
            seg_start = segment.start_us
            seg_end = segment.end_us
            for row in all_rows:
                if row.source_index in main_source_indexes:
                    continue
                # 只检查与 segment 时间重叠的算子
                if row.start_us >= seg_end or row.end_us <= seg_start:
                    continue
                searchable = f"{row.full_name} {row.name} {row.op_type}"
                if not MOE_RE.search(searchable):
                    continue
                # 按时间重叠归属到对应层
                for li, (t_start, t_end) in layer_time_ranges.items():
                    if row.start_us < t_end and row.end_us > t_start:
                        moe_layers.add(li)
                        break

    return moe_layers


def extract_layer_rows(
    segment: Segment,
    layer_index: int,
    layer_indices: list[int],
    include_related_streams: bool,
    all_rows: Sequence[KernelRow],
    is_moe: bool = False,
) -> list[dict]:
    """从 segment 中提取指定层的行，转为 dict 列表。

    策略：先提取整个 segment 的所有行（不限于指定层），
    标注 Marker 和 RMSNorm 后，按 layer_index 定位到目标层的 Attention 范围，
    再调用 extract_substructure 做子结构标注。
    """
    # 使用整个 segment 的 main_rows（不限于指定层）
    if include_related_streams and segment.main_rows:
        start_us = min(r.start_us for r in segment.main_rows)
        end_us = max(r.end_us for r in segment.main_rows)
        layer_kernel_rows = [r for r in all_rows if event_overlaps(r, start_us, end_us)]
    else:
        layer_kernel_rows = list(segment.main_rows)

    # 转为 dict + 标注 Marker（只保留主 Stream 算子，避免关联 stream 干扰层切分）
    result = []
    for row in sorted(layer_kernel_rows, key=lambda r: (r.start_us, r.source_index)):
        if row.stream_id != segment.main_stream:
            continue
        d = kernel_row_to_dict(row)
        d["Layer"] = str(layer_index)
        # 标注 Marker（供 refine_sub_blocks 使用）
        searchable = f"{row.full_name} {row.name} {row.op_type}"
        d["Marker"] = detect_marker(searchable)
        result.append(d)

    # 识别未融合 RMSNorm 序列（以 rsqrt 为锚点，标记前后相关算子为 NORM）
    result = mark_unfused_rmsnorm(result)

    # 标注子结构（全局，用于后续定位层边界）
    result = refine_sub_blocks(result)

    # 按 layer_index 裁剪到目标层范围
    result = _trim_to_target_layer(result, layer_index, layer_indices, segment.main_stream)

    # 裁剪后重新标注 Stage
    result = extract_substructure(result, main_stream=segment.main_stream, is_moe=is_moe)
    return result


def _trim_to_target_layer(
    rows: list[dict],
    layer_index: int,
    layer_indices: list[int],
    main_stream: str,
) -> list[dict]:
    """根据 layer_index 裁剪 rows，只保留目标层的范围。

    以 Attention 为层边界锚点：第 N 层 = 第 N 个 Attention 到第 N+1 个 Attention 之前。
    layer_index 即 Attention 序号（assign_layer_index_in_segment 从 0 连续递增）。
    仅考虑主流算子，避免关联 stream 的 Attention 导致索引错位。
    """
    if layer_index not in layer_indices:
        return rows

    # 找主流中的所有 Attention 行位置（排除关联 stream 的 Attention）
    att_positions = [i for i, r in enumerate(rows) if is_attention(r) and r.get("Stream ID") == main_stream]
    if not att_positions:
        return rows

    # layer_index 直接作为 Attention 序号
    start_idx = att_positions[layer_index] if layer_index < len(att_positions) else att_positions[-1]
    end_idx = att_positions[layer_index + 1] if (layer_index + 1) < len(att_positions) else len(rows)

    return rows[start_idx:end_idx]


# ═══════════════════════════════════════════════════════════════════════════
# Summary 写入
# ═══════════════════════════════════════════════════════════════════════════


def write_summary_csv(
    path: Path,
    segments: Sequence[Segment],
    *,
    expected_attention: int,
    attention_tolerance: int,
    output_files_by_index: dict[int, Path],
    layer_files_by_index: dict[int, dict[str, Path]],
) -> None:
    fieldnames = [
        "forward_index",
        "segment_kind",
        "is_valid",
        "validity_reason",
        "method",
        "main_stream",
        "start_time_us",
        "end_time_us",
        "duration_us",
        "start_task_id",
        "end_task_id",
        "main_row_count",
        "output_row_count",
        "attention_count_main",
        "attention_count_output",
        "embedding_count_main",
        "stream_counts_output",
        "max_internal_gap_us",
        "max_internal_gap_before_task",
        "max_internal_gap_after_task",
        "boundary_gap_us",
        "boundary_gap_before_task",
        "boundary_gap_after_task",
        "attention_status",
        "split_reason",
        "output_file",
        "layer_dense_file",
        "layer_moe_file",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            main_rows = sorted(segment.main_rows, key=lambda row: (row.start_us, row.source_index))
            output_rows = segment.output_rows
            max_gap, before_task, after_task = gap_details(main_rows)
            attention_main = sum(1 for row in main_rows if row.is_attention)
            attention_output = sum(1 for row in output_rows if row.is_attention)
            lf = layer_files_by_index.get(segment.index, {})
            writer.writerow(
                {
                    "forward_index": segment.index,
                    "segment_kind": segment.segment_kind,
                    "is_valid": str(segment.is_valid).lower(),
                    "validity_reason": segment.validity_reason,
                    "method": segment.method,
                    "main_stream": segment.main_stream,
                    "start_time_us": f"{segment.start_us:.3f}",
                    "end_time_us": f"{segment.end_us:.3f}",
                    "duration_us": f"{segment.end_us - segment.start_us:.3f}",
                    "start_task_id": main_rows[0].task_id if main_rows else "",
                    "end_task_id": main_rows[-1].task_id if main_rows else "",
                    "main_row_count": len(main_rows),
                    "output_row_count": len(output_rows),
                    "attention_count_main": attention_main,
                    "attention_count_output": attention_output,
                    "embedding_count_main": sum(1 for row in main_rows if row.is_embedding),
                    "stream_counts_output": format_stream_stats(output_rows),
                    "max_internal_gap_us": f"{max_gap:.3f}",
                    "max_internal_gap_before_task": before_task,
                    "max_internal_gap_after_task": after_task,
                    "boundary_gap_us": (f"{segment.boundary_gap_us:.3f}" if segment.boundary_gap_us else ""),
                    "boundary_gap_before_task": segment.boundary_gap_before_task,
                    "boundary_gap_after_task": segment.boundary_gap_after_task,
                    "attention_status": attention_status(attention_main, expected_attention, attention_tolerance),
                    "split_reason": segment.split_reason,
                    "output_file": output_files_by_index.get(segment.index, Path("")).name,
                    "layer_dense_file": lf.get("dense", Path("")).name,
                    "layer_moe_file": lf.get("moe", Path("")).name,
                }
            )


def print_summary(
    *,
    input_path: Path,
    output_dir: Path,
    rows: Sequence[KernelRow],
    main_stream: str,
    segments: Sequence[Segment],
    expected_attention: int,
    attention_tolerance: int,
    gap_threshold: float,
) -> None:
    print(f"input: {input_path}")
    print(f"rows: {len(rows)}")
    print(f"main_stream: {main_stream}")
    if gap_threshold != float("inf"):
        print(f"gap_threshold_us: {gap_threshold:.3f}")
    print(f"segments: {len(segments)}")
    print(f"output_dir: {output_dir}")
    for segment in segments:
        attention_main = sum(1 for row in segment.main_rows if row.is_attention)
        max_gap, before_task, after_task = gap_details(segment.main_rows)
        status = attention_status(attention_main, expected_attention, attention_tolerance)
        gap_note = f", max_gap={max_gap:.3f}us"
        if before_task and after_task:
            gap_note += f" ({before_task}->{after_task})"
        if segment.boundary_gap_before_task and segment.boundary_gap_after_task:
            gap_note += (
                f", boundary_gap={segment.boundary_gap_us:.3f}us "
                f"({segment.boundary_gap_before_task}->{segment.boundary_gap_after_task})"
            )
        print(
            "  "
            f"forward_{segment.index:03d}: kind={segment.segment_kind}, "
            f"valid={str(segment.is_valid).lower()}, method={segment.method}, "
            f"main_rows={len(segment.main_rows)}, output_rows={len(segment.output_rows)}, "
            f"attention={attention_main}, status={status}, "
            f"tasks={segment.main_rows[0].task_id}->{segment.main_rows[-1].task_id}"
            f"{gap_note}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NPU Layer Analyzer — Forward 切分 + 层内子结构标注。"
            "自动切分 forward，提取 1 层 Dense（+ 1 层 MoE）代表，"
            "标注 RMSNorm → Attention → RMSNorm → MLP 子结构。"
        )
    )
    # ── Forward 切分参数 ──
    parser.add_argument("--input", "-i", default="kernel_details.csv", help="输入 kernel_details.csv")
    parser.add_argument("--output-dir", default="forward_segments", help="输出目录")
    parser.add_argument("--attention-tolerance", type=int, default=0, help="attention 数允许偏差")
    parser.add_argument(
        "--attention-pattern",
        default=r"attention|infer_attention|multihead_latent_attention|infermla|ringmla|ring_mla|grouped_attention|recurrent|attn_chunk_gated",
        help="attention 匹配正则（默认覆盖 attention/MLA/ring_mla 等）",
    )
    parser.add_argument("--embedding-pattern", default="embed", help="embedding 匹配正则")
    parser.add_argument("--main-stream", default=None, help="主 stream ID（默认自动选择）")
    parser.add_argument("--gap-us", type=float, default=None, help="手动 gap 阈值（us）")
    parser.add_argument(
        "--include-related-streams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="包含其它 stream 行",
    )
    parser.add_argument(
        "--related-stream-policy",
        choices=("time-window", "main-stream-only"),
        default="time-window",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="算子 Task ID，自动定位所在 forward segment 并导出",
    )
    parser.add_argument("--summary-name", default=f"{OUTPUT_PREFIX}summary.csv")

    # ── 层提取参数 ──
    parser.add_argument("--norm-pattern", default=None, help="自定义 NORM 匹配正则")
    parser.add_argument(
        "--no-layer-export",
        action="store_true",
        help="不导出层分析 CSV（只切 forward）",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # 覆盖正则（供 detect_marker / refine_sub_blocks / find_layer_anchors 使用）
    # 通过 layer_common 模块级变量覆盖，所有引用 lc.NORM_RE 的函数都会生效
    if args.norm_pattern:
        lc.NORM_RE = re.compile(args.norm_pattern, re.IGNORECASE)
    # 如果用户自定义了 attention pattern，同步覆盖 ATT_RE
    default_att = r"attention|infer_attention|multihead_latent_attention|infermla|ringmla|ring_mla|grouped_attention|recurrent|attn_chunk_gated"
    custom_att = args.attention_pattern
    if custom_att and custom_att != default_att:
        lc.ATT_RE = re.compile(custom_att, re.IGNORECASE)

    attention_pattern = compile_pattern(args.attention_pattern)
    embedding_pattern = compile_pattern(args.embedding_pattern)

    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # ── 1. 读取 + Forward 切分 ──
    fieldnames, rows = read_rows(input_path, attention_pattern, embedding_pattern)
    if not any(row.is_attention for row in rows):
        raise SystemExit(
            "No attention operators matched. The default pattern covers attention/mla/ring_mla. "
            "If your model uses different naming, try --attention-pattern with a custom regex."
        )

    main_stream = args.main_stream if args.main_stream is not None else choose_main_stream(rows)
    main_rows = sorted(
        [row for row in rows if row.stream_id == main_stream],
        key=lambda row: (row.start_us, row.source_index),
    )
    if not main_rows:
        raise SystemExit(f"No rows found for main stream {main_stream!r}.")

    gap_threshold = args.gap_us if args.gap_us is not None else auto_gap_threshold(main_rows)
    related_policy = args.related_stream_policy
    include_related = args.include_related_streams and related_policy == "time-window"

    segments = build_segments(
        rows,
        main_rows,
        main_stream,
        gap_threshold_us=gap_threshold,
        include_related_streams=include_related,
        related_stream_policy=related_policy,
        expected_attention=0,
        attention_tolerance=args.attention_tolerance,
    )

    if not segments:
        raise SystemExit("No forward segments found.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. 导出 Forward CSV ──
    # 默认 first-valid-by-kind 策略；指定 --task-id 时自动切换到 task-id 策略
    effective_export_policy = "task-id" if args.task_id is not None else "first-valid-by-kind"

    exported_segments = select_segments_for_export(
        segments,
        export_policy=effective_export_policy,
        export_indexes="",
        max_per_kind=1,
        task_id=args.task_id,
        forward_kind="all",
    )

    output_files_by_index: dict[int, Path] = {}
    layer_files_by_index: dict[int, dict[str, Path]] = defaultdict(dict)

    for segment in exported_segments:
        output_file = output_dir / f"{OUTPUT_PREFIX}forward_{segment.index:03d}.csv"
        write_segment_csv(output_file, fieldnames, segment.output_rows)
        output_files_by_index[segment.index] = output_file

    # ── 3. 层提取 + 子结构标注 ──
    if not args.no_layer_export:
        for segment in exported_segments:
            # 分配层号
            layer_indices = assign_layer_index_in_segment(segment, main_stream)

            # 检测 MoE
            moe_layers = detect_moe_in_segment(segment, layer_indices, all_rows=rows)
            has_moe = bool(moe_layers)

            # 检测含 ATT 的层
            att_layers = set()
            for i, row in enumerate(segment.main_rows):
                li = layer_indices[i]
                if li < 0:
                    continue
                if row.is_attention:
                    att_layers.add(li)

            # 选取代表层（优先选含 ATT 的层，自动选取前 1/3）
            picks = pick_representative_layer(layer_indices, moe_layers, None, att_layers)

            print(f"\n--- Forward {segment.index:03d} 层分析 ---")
            att_count = sum(1 for row in segment.main_rows if row.is_attention)
            norm_count_total = sum(
                1 for row in segment.main_rows if lc.NORM_RE.search(f"{row.full_name} {row.name} {row.op_type}")
            )
            print(f"  层数: {max(layer_indices) + 1 if layer_indices and max(layer_indices) >= 0 else 0}")
            print(
                f"  ATT: {att_count} | NORM: {norm_count_total} | MoE层: {sorted(moe_layers) if moe_layers else '无'}"
            )
            print(f"  选取: Dense={picks['dense']}" + (f", MoE={picks['moe']}" if picks["moe"] is not None else ""))

            for kind, layer_idx in [("dense", picks["dense"]), ("moe", picks["moe"])]:
                if layer_idx is None:
                    continue

                layer_rows = extract_layer_rows(
                    segment,
                    layer_idx,
                    layer_indices,
                    include_related_streams=include_related,
                    all_rows=rows,
                    is_moe=(kind == "moe"),
                )
                if not layer_rows:
                    continue

                # 输出
                if has_moe:
                    fname = f"{OUTPUT_PREFIX}forward_{segment.index:03d}_layer{layer_idx}_{kind}.csv"
                else:
                    fname = f"{OUTPUT_PREFIX}forward_{segment.index:03d}_layer{layer_idx}.csv"
                output_path = output_dir / fname
                write_layer_csv(output_path, layer_rows)
                layer_files_by_index[segment.index][kind] = output_path
                print(f"\n  输出: {output_path}")

                # 打印子结构摘要
                print_structure_summary(layer_rows, layer_idx, kind.upper())

    # ── 4. 写 Summary ──
    summary_path = output_dir / args.summary_name
    write_summary_csv(
        summary_path,
        segments,
        expected_attention=0,
        attention_tolerance=args.attention_tolerance,
        output_files_by_index=output_files_by_index,
        layer_files_by_index=layer_files_by_index,
    )

    # ── 5. 打印总览 ──
    print_summary(
        input_path=input_path,
        output_dir=output_dir,
        rows=rows,
        main_stream=main_stream,
        segments=segments,
        expected_attention=0,
        attention_tolerance=args.attention_tolerance,
        gap_threshold=gap_threshold,
    )
    print(f"export_policy: {effective_export_policy}")
    if args.task_id is not None:
        if exported_segments:
            print(f"task_id: {args.task_id} → forward_{exported_segments[0].index:03d}")
        else:
            print(f"task_id: {args.task_id} → not found")
    print(f"exported_forward_files: {len(output_files_by_index)}")

    if not args.no_layer_export:
        total_layer_files = sum(len(v) for v in layer_files_by_index.values())
        print(f"exported_layer_files: {total_layer_files}")
        print("\n--- CSV 查看提示 ---")
        print("1. 筛选 Is_Key = ★ → 只看关键边界算子（RMSNorm / Attention / MLP）")
        print("2. 按 Stage 列筛选 → 查看各阶段（Attention / FFN 或 MOE）")

    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

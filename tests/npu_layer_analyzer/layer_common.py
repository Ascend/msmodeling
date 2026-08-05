#!/usr/bin/env python3
"""Shared utilities for layer analysis (used by npu_layer_analyzer & layer_analyzer).

Contains common regex patterns, constants, marker detection, unfused RMSNorm
recognition, substructure annotation, layer extraction, and CSV writing logic.

Both analyzer scripts import from this module to avoid duplication and ensure
consistent Stage 划分 on both sides of the dual-tool comparison.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# Regex patterns (module-level, can be overridden by callers if needed)
# ═══════════════════════════════════════════════════════════════════════════

NORM_RE = re.compile(
    r"(?:rmsnorm|rms_norm|layernorm|layer_norm|gated_rmsnorm|[._]norm[._]|\bnorm\b)",
    re.IGNORECASE,
)
ATT_RE = re.compile(
    r"(?:attention|infer_attention|multihead_latent_attention|infermla|ringmla|ring_mla|grouped_attention|recurrent|attn_chunk_gated)",
    re.IGNORECASE,
)
RECURRENT_ATT_RE = re.compile(
    r"\b(?:delta_rule|gated_delta|recurrent|attn_chunk_gated)\b",
    re.IGNORECASE,
)
MLP_RE = re.compile(
    r"\b(?:swiglu|silu|gelu|sigmoid|ffn|mlp)\b",
    re.IGNORECASE,
)
MATMUL_RE = re.compile(r"matmul|\bmm\b", re.IGNORECASE)
LINEAR_RE = re.compile(
    r"linear_all_reduce",
    re.IGNORECASE,
)
MOE_RE = re.compile(
    r"(?:moe|router|expert|grouped[_]?matmul|dispatch[_]?ffn[_]?combine)",
    re.IGNORECASE,
)
EMBED_RE = re.compile(
    r"\bembed(?:ding)?\b",
    re.IGNORECASE,
)
SAMPLE_RE = re.compile(
    r"\b(?:downsample|upsample|interp)\b",
    re.IGNORECASE,
)
COMM_RE = re.compile(
    r"\b(?:all_gather|allreduce|all_reduce|send|recv|reduce_scatter)\b",
    re.IGNORECASE,
)

# ── Marker rules (pattern string, marker name, HTML color) ───────────────

MARKER_RULES: list[tuple[str, str, str]] = [
    (r"\bembed(?:ding)?\b", "EMBED", "#FF9800"),
    (
        r"(?:attention|infer_attention|multihead_latent_attention|infermla|ringmla|ring_mla|grouped_attention|recurrent|attn_chunk_gated)",
        "ATT",
        "#FFD700",
    ),
    (
        r"(?:rmsnorm|rms_norm|layernorm|layer_norm|gated_rmsnorm|[._]norm[._]|\bnorm\b)",
        "NORM",
        "#FFF176",
    ),
    (r"\b(?:swiglu|silu|gelu|sigmoid|ffn|mlp)\b", "MLP", "#A5D6A7"),
    (r"matmul|\bmm\b", "MATMUL", "#81D4FA"),
    (r"linear_all_reduce", "LINEAR", "#81D4FA"),
    (r"\b(?:all_gather|allreduce|all_reduce|send|recv)\b", "COMM", "#F48FB1"),
    (r"\b(?:downsample|upsample|interp)\b", "SAMPLE", "#CE93D8"),
]

MARKER_COLORS: dict[str, str] = {
    "EMBED": "#FF9800",
    "ATT": "#FFD700",
    "NORM": "#FFF176",
    "MLP": "#A5D6A7",
    "MATMUL": "#81D4FA",
    "LINEAR": "#81D4FA",
    "COMM": "#F48FB1",
    "SAMPLE": "#CE93D8",
    "OTHER": "#FFFFFF",
}

# Default marker rules built from module-level regexes (rebuilt at call time
# so callers can override NORM_RE / ATT_RE etc. via layer_common.NORM_RE = ...).
# Callers needing custom rules should pass compiled_rules to detect_marker.

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Unfused RMSNorm related operators (aten.rsqrt.default is the anchor)
UNFUSED_NORM_OPS = {
    "aten.view.default",
    "aten.add.Tensor",
    "prims.convert_element_type.default",
    "aten.pow.Tensor_Scalar",
    "aten.mean.dim",
    "aten.rsqrt.default",
    "aten.mul.Tensor",
}

# Sub-block labels
BLOCK_ATTENTION = "2_Attention"
BLOCK_POST_ATT = "3_Linear(post-att)"
BLOCK_NORM_PRE_MLP = "4_Norm(pre-MLP)"
BLOCK_MLP = "5_MLP"
KEY_STAR = "★"

# Named constants (replacing magic numbers)
UNFUSED_NORM_LOOKBACK = 15
UNFUSED_NORM_LOOKAHEAD = 10
PRE_ATT_LOOKAHEAD = 30


# ═══════════════════════════════════════════════════════════════════════════
# Marker detection
# ═══════════════════════════════════════════════════════════════════════════


def detect_marker(full_name: str, compiled_rules: list[tuple[re.Pattern, str]] | None = None) -> str:
    """Detect marker type from a searchable string.

    If compiled_rules is None, builds rules from current module-level regexes
    (so callers can override NORM_RE / ATT_RE via ``layer_common.NORM_RE = ...``).
    Callers needing custom rules should pass compiled_rules explicitly.
    """
    if compiled_rules is not None:
        rules = compiled_rules
    else:
        rules = [
            (EMBED_RE, "EMBED"),
            (ATT_RE, "ATT"),
            (NORM_RE, "NORM"),
            (MLP_RE, "MLP"),
            (MATMUL_RE, "MATMUL"),
            (LINEAR_RE, "LINEAR"),
            (COMM_RE, "COMM"),
            (SAMPLE_RE, "SAMPLE"),
        ]
    for pattern, marker in rules:
        if pattern.search(full_name):
            return marker
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Row-level predicates
# ═══════════════════════════════════════════════════════════════════════════


def match_in_row(row: dict, regex: re.Pattern) -> bool:
    """Search regex in row's Full Name + Name + Type."""
    fn = row.get("Full Name", row.get("Name", ""))
    searchable = f"{fn} {row.get('Name', '')} {row.get('Type', '')}"
    return bool(regex.search(searchable))


def is_attention(row: dict) -> bool:
    """Check if row is Attention (including recurrent delta_rule)."""
    if row.get("Marker") == "ATT":
        return True
    fn = row.get("Full Name", row.get("Name", ""))
    searchable = f"{fn} {row.get('Name', '')} {row.get('Type', '')}"
    return bool(ATT_RE.search(searchable) or RECURRENT_ATT_RE.search(searchable))


def is_norm(row: dict) -> bool:
    """Check if row is NORM (including unfused RMSNorm via Marker column)."""
    if row.get("Marker") == "NORM":
        return True
    return match_in_row(row, NORM_RE)


# ═══════════════════════════════════════════════════════════════════════════
# Unfused RMSNorm recognition
# ═══════════════════════════════════════════════════════════════════════════


def mark_unfused_rmsnorm(rows: list[dict]) -> list[dict]:
    """Identify unfused RMSNorm operator sequences and mark them as NORM.

    Uses rsqrt as anchor, expands forward/backward to cover contiguous
    related operators, marking all as Marker=NORM.
    """
    n = len(rows)
    for i, r in enumerate(rows):
        name = r.get("Name", "")
        if "rsqrt" not in name.lower():
            continue
        # Expand backward
        start = i
        for j in range(i - 1, max(i - UNFUSED_NORM_LOOKBACK, -1), -1):
            if rows[j].get("Name", "") in UNFUSED_NORM_OPS:
                start = j
            else:
                break
        # Expand forward
        end = i
        for j in range(i + 1, min(i + UNFUSED_NORM_LOOKAHEAD, n)):
            if rows[j].get("Name", "") in UNFUSED_NORM_OPS:
                end = j
            else:
                break
        # Mark as NORM
        for j in range(start, end + 1):
            rows[j]["Marker"] = "NORM"
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Substructure annotation (refine_sub_blocks)
# ═══════════════════════════════════════════════════════════════════════════


def refine_sub_blocks(layer_rows: list[dict]) -> list[dict]:
    """Annotate Structure + Is_Key columns for rows within a layer.

    State machine traversal: NORM starts a new sub-block, ATTENTION switches
    to "2_Attention", MLP (after attention) switches to "5_MLP".
    Multiple NORMs are distinguished by letter suffix: 1a_Norm, 1b_Norm ...

    Uses Marker column first (set by detect_marker / mark_unfused_rmsnorm),
    falls back to built-in regex matching.
    """
    if not layer_rows:
        return layer_rows

    def _is(row: dict, marker: str, regex: re.Pattern) -> bool:
        m = row.get("Marker", "")
        if m == marker:
            return True
        return match_in_row(row, regex)

    # Find attention position
    att_pos = None
    for idx, r in enumerate(layer_rows):
        if _is(r, "ATT", ATT_RE):
            att_pos = idx
            break

    # Find first MLP after attention
    first_mlp_after_att = None
    if att_pos is not None:
        for idx in range(att_pos + 1, len(layer_rows)):
            if _is(layer_rows[idx], "MLP", MLP_RE):
                first_mlp_after_att = idx
                break

    # Find pre-MLP NORM
    pre_mlp_norm_pos = None
    if att_pos is not None and first_mlp_after_att is not None:
        for idx in range(first_mlp_after_att - 1, att_pos, -1):
            if _is(layer_rows[idx], "NORM", NORM_RE):
                pre_mlp_norm_pos = idx
                break

    # State machine
    phase = "pre_att"
    current_block = "1a_Norm"
    norm_counter = 0

    for idx, r in enumerate(layer_rows):
        is_n = _is(r, "NORM", NORM_RE)
        is_a = _is(r, "ATT", ATT_RE)
        is_m = _is(r, "MLP", MLP_RE)

        if is_a:
            phase = "attention"
            current_block = BLOCK_ATTENTION
        elif is_n:
            norm_counter += 1
            norm_suffix = chr(ord("a") + norm_counter - 1)
            if att_pos is None or idx < att_pos:
                phase = "pre_att"
                current_block = f"1{norm_suffix}_Norm"
            elif pre_mlp_norm_pos is not None and idx >= pre_mlp_norm_pos:
                phase = "pre_mlp"
                current_block = BLOCK_NORM_PRE_MLP
            else:
                phase = "post_att"
                current_block = f"1{norm_suffix}_Norm"
        elif is_m and att_pos is not None and idx > att_pos:
            phase = "mlp"
            current_block = BLOCK_MLP
        elif att_pos is not None and idx > att_pos and phase == "attention":
            phase = "post_att"
            current_block = BLOCK_POST_ATT

        r["Structure"] = current_block
        if is_n or is_a or (is_m and phase == "mlp"):
            r["Is_Key"] = KEY_STAR
        else:
            r["Is_Key"] = ""

    return layer_rows


# ═══════════════════════════════════════════════════════════════════════════
# Stage extraction (extract_substructure)
# ═══════════════════════════════════════════════════════════════════════════


def extract_substructure(layer_rows: list[dict], main_stream: str | None = None, is_moe: bool = False) -> list[dict]:
    """Extract key substructure from layer rows.

    New 2-stage rule:
      Attention Stage = (RmsNorm 后剩余算子) + (ATT → AddRmsNormBias)
      FFN/MOE Stage = AddRmsNormBias 后 → RmsNorm

    Layer boundary: from one Attention to just before the next.
    RmsNorm = 第二个 NORM（下一层的 pre-ATT NORM）。
    RmsNorm 后面的算子拼到当前层 Attention Stage 前面，保持连续不分段。

    If main_stream is provided, ATT/NORM boundary detection only considers
    rows on that stream (avoids non-main-stream NORMs interfering).

    is_moe: True 时第二段 Stage 名为 "MOE"，False 时为 "FFN"。
    """
    second_stage_name = "MOE" if is_moe else "FFN"

    n = len(layer_rows)
    if n == 0:
        return layer_rows

    # Helper: check if row is on main stream (or all rows if no filter)
    def _on_main(r: dict) -> bool:
        return main_stream is None or r.get("Stream ID") == main_stream

    # 1. Find all Attention positions (only on main stream if specified)
    att_positions = [i for i, r in enumerate(layer_rows) if is_attention(r) and _on_main(r)]
    if not att_positions:
        return layer_rows

    # 2. Take first Attention to just before next Attention (single layer range)
    start_idx = att_positions[0]
    if len(att_positions) >= 2:
        end_idx = att_positions[1]  # exclude next Attention
    else:
        end_idx = n  # single Attention: take to end

    # 3. Trim to this layer
    trimmed = layer_rows[start_idx:end_idx]

    # 4. Initialize Stage/Is_Key
    for r in trimmed:
        r["Stage"] = ""
        r["Is_Key"] = ""

    # 5. Find first ATT, first NORM (AddRmsNormBias), second NORM (RmsNorm)
    #    Only consider main-stream rows for NORM boundary detection
    first_att = None
    first_norm = None
    second_norm = None

    for i, r in enumerate(trimmed):
        if is_attention(r) and _on_main(r) and first_att is None:
            first_att = i
            continue
        if first_att is not None:
            if is_norm(r) and _on_main(r):
                if first_norm is None:
                    first_norm = i
                elif second_norm is None:
                    second_norm = i
                    break

    # 6. Build reordered result: Attention first (continuous), then FFN/MOE
    #    AddRmsNormBias (first_norm) belongs to FFN/MOE, not Attention.
    result = []

    if first_norm is None:
        # No NORM found: everything is Attention
        for r in trimmed:
            r["Stage"] = "Attention"
        result = list(trimmed)
    elif second_norm is None:
        # Only 1 NORM: ATT → before NORM = Attention, NORM → end = FFN/MOE
        for r in trimmed[:first_norm]:
            r["Stage"] = "Attention"
        for r in trimmed[first_norm:]:
            r["Stage"] = second_stage_name
        result = list(trimmed)
    else:
        # 2 NORMs: reorder so Attention is continuous
        # Attention = after_second_norm + ATT → before first_norm
        # FFN/MOE = first_norm → second_norm (含两端)
        after_second_norm = trimmed[second_norm + 1 :]
        att_before_first_norm = trimmed[:first_norm]
        second_part = trimmed[first_norm : second_norm + 1]

        for r in after_second_norm:
            r["Stage"] = "Attention"
        for r in att_before_first_norm:
            r["Stage"] = "Attention"
        for r in second_part:
            r["Stage"] = second_stage_name

        result = after_second_norm + att_before_first_norm + second_part

    # 7. Mark key rows
    if first_att is not None and first_att < len(trimmed):
        trimmed[first_att]["Is_Key"] = KEY_STAR
    if first_norm is not None and first_norm < len(trimmed):
        trimmed[first_norm]["Is_Key"] = KEY_STAR
    if second_norm is not None and second_norm < len(trimmed):
        trimmed[second_norm]["Is_Key"] = KEY_STAR

    return result


# ═══════════════════════════════════════════════════════════════════════════
# MoE detection + representative layer selection
# ═══════════════════════════════════════════════════════════════════════════


def pick_representative_layer(
    layer_indices: list[int],
    moe_layers: set[int],
    requested_index: int | None,
    att_layers: set[int] | None = None,
) -> dict[str, int | None]:
    """Select representative layers. Prefers Dense layers containing Attention.

    Returns {"dense": layer_index, "moe": layer_index_or_None}.
    Dense pick: if requested_index is valid use it; elif att_layers given,
    pick from ATT-containing Dense layers at 1/3 position; else pick from
    all Dense layers at 1/3 position.
    """
    all_layers = sorted(set(li for li in layer_indices if li >= 0))
    if not all_layers:
        return {"dense": None, "moe": None}

    dense_layers = [li for li in all_layers if li not in moe_layers]
    att_dense = [li for li in dense_layers if li in att_layers] if att_layers else []

    if requested_index is not None:
        dense_set = set(dense_layers)
        if requested_index in dense_set:
            dense_pick = requested_index
        else:
            dense_pick = dense_layers[0] if dense_layers else None
    elif att_dense:
        dense_pick = att_dense[len(att_dense) // 3]
    else:
        dense_pick = dense_layers[len(dense_layers) // 3] if dense_layers else None

    moe_pick = None
    if moe_layers:
        moe_list = sorted(moe_layers)
        moe_pick = moe_list[len(moe_list) // 2]

    return {"dense": dense_pick, "moe": moe_pick}


# ═══════════════════════════════════════════════════════════════════════════
# CSV output
# ═══════════════════════════════════════════════════════════════════════════


def write_layer_csv(output_path: Path, rows: list[dict]) -> None:
    """Write layer CSV with substructure annotation columns."""
    if not rows:
        return

    fieldnames = [k for k in rows[0].keys() if k != "Structure"]
    for col in ("Layer", "Stage", "Is_Key"):
        if col not in fieldnames:
            fieldnames.append(col)

    # Column order: priority columns first
    priority = ["Layer", "Stage", "Is_Key"]
    ordered = [f for f in priority if f in fieldnames]
    ordered += [f for f in fieldnames if f not in ordered]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            r.setdefault("Is_Key", "")
            r.setdefault("Layer", "")
            r.setdefault("Stage", "")
            writer.writerow(r)


# ═══════════════════════════════════════════════════════════════════════════
# Print summary
# ═══════════════════════════════════════════════════════════════════════════


def print_structure_summary(rows: list[dict], layer_index: int, kind: str) -> None:
    """Print substructure summary for one layer."""
    print(f"\n  {kind} Layer {layer_index} 子结构:")
    current_block = None
    block_count = 0
    for r in rows:
        block = r.get("Structure", "")
        is_key = r.get("Is_Key", "")
        if block != current_block:
            if current_block is not None:
                print(f"    {current_block}: {block_count} ops")
            current_block = block
            block_count = 0
        block_count += 1
        if is_key == KEY_STAR:
            fn = r.get("Full Name", r.get("Name", ""))
            ts = r.get("Start Time(us)", "")
            dur = r.get("Duration(us)", "")
            print(f"      ★ {fn}  (ts={ts}, dur={dur})")
    if current_block is not None:
        print(f"    {current_block}: {block_count} ops")

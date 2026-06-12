"""
Backfill FIA runtime metadata from a JSONL dump into FusedInferAttentionScore.csv.

Uses SQL LEFT JOIN semantics: match CSV profiling rows to JSONL runtime records
by deterministic key (Q/K/V shape + num_seqs_kv + atten_mask shape + block_table shape),
then expand 1:N when multiple JSONL records share the same key.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

try:
    from fia_common import parse_shape_or_none
except ModuleNotFoundError:
    from .fia_common import parse_shape_or_none


# -- Runtime column names --

RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE = "Runtime actual_seq_lengths_shape"
RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES = "Runtime actual_seq_lengths_values"
RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE = "Runtime actual_seq_lengths_kv_shape"
RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES = "Runtime actual_seq_lengths_kv_values"
RUNTIME_AVG_SEQ_LEN = "Runtime avg_seq_len"
RUNTIME_BLOCK_TABLE_VALID_BLOCKS = "Runtime block_table_valid_blocks"
RUNTIME_NUM_HEADS = "Runtime num_heads"
RUNTIME_NUM_KEY_VALUE_HEADS = "Runtime num_key_value_heads"
RUNTIME_SPARSE_MODE = "Runtime sparse_mode"
RUNTIME_INPUT_LAYOUT = "Runtime input_layout"
RUNTIME_BLOCK_SIZE = "Runtime block_size"
RUNTIME_METADATA_COMPLETENESS = "Runtime metadata_completeness"

BACKFILL_COLUMNS = [
    RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE,
    RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES,
    RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE,
    RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES,
    RUNTIME_AVG_SEQ_LEN,
    RUNTIME_BLOCK_TABLE_VALID_BLOCKS,
    RUNTIME_NUM_HEADS,
    RUNTIME_NUM_KEY_VALUE_HEADS,
    RUNTIME_SPARSE_MODE,
    RUNTIME_INPUT_LAYOUT,
    RUNTIME_BLOCK_SIZE,
    RUNTIME_METADATA_COMPLETENESS,
]


# -- Input Shapes slot indices (FIA aclnn parameter order) --
_SLOT_QUERY = 0
_SLOT_KEY = 1
_SLOT_VALUE = 2
_SLOT_ATTEN_MASK = 4
_SLOT_ACTUAL_SEQ_LENGTHS_KV = 6
_SLOT_BLOCK_TABLE = 14

# -- Key type: 6-tuple --
# (q_shape, k_shape, v_shape, num_seqs_kv, atten_mask_shape, block_table_shape)
JoinKey = tuple


def _parse_slot(slots: list[str], index: int) -> tuple[int, ...] | None:
    """Parse a single slot from the Input Shapes 31-slot format."""
    if index >= len(slots):
        return None
    return parse_shape_or_none(slots[index].strip())


def _parse_slot_int(slots: list[str], index: int) -> int | None:
    """Parse a slot that contains a single integer (e.g., num_seqs)."""
    if index >= len(slots):
        return None
    raw = slots[index].strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def build_csv_key(row: dict[str, str]) -> JoinKey:
    """Extract JOIN key from a CSV row's Input Shapes."""
    raw = (row.get("Input Shapes", "") or "").strip().strip('"')
    slots = [s.strip() for s in raw.split(";")]
    return (
        _parse_slot(slots, _SLOT_QUERY),
        _parse_slot(slots, _SLOT_KEY),
        _parse_slot(slots, _SLOT_VALUE),
        _parse_slot_int(slots, _SLOT_ACTUAL_SEQ_LENGTHS_KV),
        _parse_slot(slots, _SLOT_ATTEN_MASK),
        _parse_slot(slots, _SLOT_BLOCK_TABLE),
    )


def _to_tuple(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(x) for x in value)


def build_jsonl_key(record: dict) -> JoinKey:
    """Extract JOIN key from a JSONL dump record."""
    seq_kv = record.get("actual_seq_lengths_kv")
    return (
        _to_tuple(record.get("query_shape")),
        _to_tuple(record.get("key_shape")),
        _to_tuple(record.get("value_shape")),
        len(seq_kv) if seq_kv is not None else None,
        _to_tuple(record.get("atten_mask_shape")),
        _to_tuple(record.get("block_table_shape")),
    )


# -- Runtime payload --

def _format_int_list(values: list[int] | None) -> str:
    if not values:
        return ""
    return ",".join(str(int(v)) for v in values)


def build_runtime_payload(record: dict) -> dict[str, str]:
    """Build the runtime columns dict from a JSONL record."""
    seq = record.get("actual_seq_lengths")
    seq_kv = record.get("actual_seq_lengths_kv")
    valid_blocks = record.get("block_table_valid_blocks")
    avg = mean(seq_kv) if seq_kv else None

    return {
        RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE: str(len(seq)) if seq else "",
        RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES: _format_int_list(seq),
        RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE: str(len(seq_kv)) if seq_kv else "",
        RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES: _format_int_list(seq_kv),
        RUNTIME_AVG_SEQ_LEN: f"{avg:.6f}" if avg is not None else "",
        RUNTIME_BLOCK_TABLE_VALID_BLOCKS: _format_int_list(valid_blocks),
        RUNTIME_NUM_HEADS: str(record["num_heads"]) if record.get("num_heads") is not None else "",
        RUNTIME_NUM_KEY_VALUE_HEADS: str(record["num_key_value_heads"]) if record.get("num_key_value_heads") is not None else "",
        RUNTIME_SPARSE_MODE: str(record["sparse_mode"]) if record.get("sparse_mode") is not None else "",
        RUNTIME_INPUT_LAYOUT: str(record["input_layout"]) if record.get("input_layout") is not None else "",
        RUNTIME_BLOCK_SIZE: str(record["block_size"]) if record.get("block_size") is not None else "",
    }


def _payload_dedup_key(payload: dict[str, str]) -> tuple:
    """Dedup key: (actual_seq_lengths_values, actual_seq_lengths_kv_values)."""
    return (
        payload[RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES],
        payload[RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES],
    )


# -- JSONL loading --

def load_jsonl(jsonl_path: Path) -> dict[JoinKey, list[dict[str, str]]]:
    """Load JSONL into a dict: JoinKey -> deduplicated list of runtime payloads."""
    grouped: dict[JoinKey, dict[tuple, dict[str, str]]] = defaultdict(dict)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = build_jsonl_key(record)
            payload = build_runtime_payload(record)
            dedup = _payload_dedup_key(payload)
            grouped[key][dedup] = payload

    return {k: list(v.values()) for k, v in grouped.items()}


# -- Backfill --

def backfill(
    rows: list[dict[str, str]],
    jsonl_index: dict[JoinKey, list[dict[str, str]]],
    metadata_tag: str,
) -> tuple[list[dict[str, str]], int, int]:
    """LEFT JOIN CSV rows with JSONL payloads. Returns (output_rows, matched, total)."""
    matched = 0
    output: list[dict[str, str]] = []

    for row in rows:
        key = build_csv_key(row)
        payloads = jsonl_index.get(key)

        if not payloads:
            output.append(row)
            continue

        matched += 1
        for payload in payloads:
            merged = dict(row)
            merged.update(payload)
            merged[RUNTIME_METADATA_COMPLETENESS] = metadata_tag
            output.append(merged)

    return output, matched, len(output)


def ensure_fieldnames(fieldnames: list[str]) -> list[str]:
    merged = list(fieldnames)
    for col in BACKFILL_COLUMNS:
        if col not in merged:
            merged.append(col)
    return merged


# -- CLI --

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backfill FIA runtime metadata from JSONL into CSV (JOIN semantics).",
    )
    p.add_argument("--csv-path", required=True, help="Path to FusedInferAttentionScore.csv")
    p.add_argument("--jsonl-path", required=True, help="Path to fia_runtime_metadata.jsonl")
    p.add_argument("--output-path", help="Output CSV path. Defaults to overwriting --csv-path.")
    p.add_argument(
        "--metadata-tag",
        default="runtime_values_dumped",
        help="Value for Runtime metadata_completeness on matched rows.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    csv_path = Path(args.csv_path)
    jsonl_path = Path(args.jsonl_path)
    output_path = Path(args.output_path) if args.output_path else csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV header is empty: {csv_path}")
        rows = list(reader)
        fieldnames = ensure_fieldnames(list(fieldnames))

    jsonl_index = load_jsonl(jsonl_path)
    output_rows, matched, total = backfill(rows, jsonl_index, args.metadata_tag)

    # Collect all keys that actually appear in output rows so that any payload
    # field not declared in BACKFILL_COLUMNS is still written rather than lost.
    all_keys: dict[str, None] = dict.fromkeys(fieldnames)
    for row in output_rows:
        all_keys.update(dict.fromkeys(row.keys()))
    fieldnames = list(all_keys)

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    expanded = total - len(rows)
    print(
        f"Backfilled {matched}/{len(rows)} CSV rows matched, "
        f"{total} output rows ({'+' + str(expanded) if expanded > 0 else expanded} from 1:N expansion) "
        f"from {jsonl_path} into {output_path}"
    )


if __name__ == "__main__":
    main()

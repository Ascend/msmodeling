from __future__ import annotations

import math
from typing import Generator

try:
    from ..model_configs import ModelConfig
    from ..model_configs import _normalize_name as _normalize_model_name
    from ..model_configs import resolve_configs
except ImportError:
    from model_configs import ModelConfig
    from model_configs import _normalize_name as _normalize_model_name
    from model_configs import resolve_configs

from .base import TheoryShapeRow

RUNTIME_SOURCE_PROFILE = "Runtime source_profile"
RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE = "Runtime actual_seq_lengths_shape"
RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES = "Runtime actual_seq_lengths_values"
RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE = "Runtime actual_seq_lengths_kv_shape"
RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES = "Runtime actual_seq_lengths_kv_values"
RUNTIME_AVG_SEQ_LEN = "Runtime avg_seq_len"
RUNTIME_BLOCK_TABLE_SHAPE = "Runtime block_table_shape"
RUNTIME_BLOCK_TABLE_VALID_BLOCKS = "Runtime block_table_valid_blocks"
RUNTIME_NUM_HEADS = "Runtime num_heads"
RUNTIME_NUM_KEY_VALUE_HEADS = "Runtime num_key_value_heads"
RUNTIME_SPARSE_MODE = "Runtime sparse_mode"
RUNTIME_INPUT_LAYOUT = "Runtime input_layout"
RUNTIME_BLOCK_SIZE = "Runtime block_size"
RUNTIME_METADATA_COMPLETENESS = "Runtime metadata_completeness"
RUNTIME_OPERATOR_INPUT_SHAPES_RAW = "Runtime operator_input_shapes_raw"

FIA_RUNTIME_COLUMNS = [
    RUNTIME_SOURCE_PROFILE,
    RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE,
    RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES,
    RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE,
    RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES,
    RUNTIME_AVG_SEQ_LEN,
    RUNTIME_BLOCK_TABLE_SHAPE,
    RUNTIME_BLOCK_TABLE_VALID_BLOCKS,
    RUNTIME_NUM_HEADS,
    RUNTIME_NUM_KEY_VALUE_HEADS,
    RUNTIME_SPARSE_MODE,
    RUNTIME_INPUT_LAYOUT,
    RUNTIME_BLOCK_SIZE,
    RUNTIME_METADATA_COMPLETENESS,
    RUNTIME_OPERATOR_INPUT_SHAPES_RAW,
]

_BLOCK_SIZE = 128
_MIN_TOTAL_BLOCKS = 16
_DEFAULT_SPARSE_MODE_3_MASK = (2048, 2048)
_FIA_INPUT_SLOT_COUNT = 31

_QWEN3_DENSE_PREFILL_BATCHES = [1, 2, 4, 8]
_QWEN3_DENSE_PREFILL_SEQS = [512, 1024, 2048, 3072, 4096, 4112, 4608, 5120, 6144]
_QWEN3_DENSE_DECODE_BATCHES = [1, 2, 4, 8, 12, 16, 24, 32]
_QWEN3_DENSE_DECODE_AVG_SEQS = [1024, 2048, 3072, 4096, 4224, 4352, 4608, 4864, 5120, 5376, 5632]

_DSV3_MLA_PREFILL_BATCHES = [1, 2, 4, 8]
_DSV3_MLA_PREFILL_SEQS = [512, 1024, 2048, 3072, 4096, 4099, 4608, 5120, 6144]
_DSV3_MLA_DECODE_BATCHES = [1, 2, 4, 8, 12, 16]
_DSV3_MLA_DECODE_AVG_SEQS = [1024, 2048, 3072, 4096, 4100, 4224, 4352, 4608, 4864, 5120, 5376, 5632]

_GENERIC_PREFILL_BATCHES = [1, 2, 4, 8]
_GENERIC_PREFILL_SEQS = [256, 512, 1024, 2048, 4096, 8192]
_GENERIC_DECODE_BATCHES = [1, 2, 4, 8, 16, 32]
_GENERIC_DECODE_AVG_SEQS = [512, 1024, 2048, 4096, 8192]


def _build_shape_cell(shapes: list[tuple[int, ...]]) -> str:
    parts = []
    for shape in shapes:
        if shape:
            parts.append(",".join(str(dim) for dim in shape))
        else:
            parts.append("")
    return ";".join(parts)


def _build_31_slots(slots: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    return (slots + [()] * (31 - len(slots)))[:31]


def _slot_text(values: list[str]) -> str:
    return ";".join(values[:_FIA_INPUT_SLOT_COUNT])


def _fia_input_metadata(input_shapes: list[tuple[int, ...]]) -> dict[str, str]:
    dtypes = ["DT_UNDEFINED"] * _FIA_INPUT_SLOT_COUNT
    formats = ["NULL"] * _FIA_INPUT_SLOT_COUNT

    for index in (0, 1, 2, 24, 25):
        if index < len(input_shapes) and input_shapes[index]:
            dtypes[index] = "DT_BF16"
            formats[index] = "ND"
    for index in (5, 6):
        if index < len(input_shapes) and input_shapes[index]:
            dtypes[index] = "INT64"
            formats[index] = "ND"
    if len(input_shapes) > 4 and input_shapes[4]:
        dtypes[4] = "INT8"
        formats[4] = "ND"
    if len(input_shapes) > 14 and input_shapes[14]:
        dtypes[14] = "INT32"
        formats[14] = "ND"

    return {
        "Input Data Types": _slot_text(dtypes),
        "Input Formats": _slot_text(formats),
    }


def _format_int_list(values: list[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _cumulative(values: list[int]) -> list[int]:
    total = 0
    result = []
    for value in values:
        total += value
        result.append(total)
    return result


def _runtime_metadata(
    *,
    avg_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    sparse_mode: int,
    input_layout: str,
    raw_input_shapes: list[tuple[int, ...]],
    actual_seq_lengths_values: list[int],
    actual_seq_lengths_kv_values: list[int],
    block_table_shape: tuple[int, ...] | None,
    block_table_valid_blocks: list[int] | None,
) -> dict[str, str]:
    values = {
        RUNTIME_SOURCE_PROFILE: "SHAPE_GRID_SCENE",
        RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE: str(len(actual_seq_lengths_values))
        if actual_seq_lengths_values
        else "",
        RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES: _format_int_list(actual_seq_lengths_values)
        if actual_seq_lengths_values
        else "",
        RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE: str(len(actual_seq_lengths_kv_values))
        if actual_seq_lengths_kv_values
        else "",
        RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES: _format_int_list(actual_seq_lengths_kv_values)
        if actual_seq_lengths_kv_values
        else "",
        RUNTIME_AVG_SEQ_LEN: f"{float(avg_seq_len):.6f}",
        RUNTIME_BLOCK_TABLE_SHAPE: ""
        if block_table_shape is None
        else ",".join(str(dim) for dim in block_table_shape),
        RUNTIME_BLOCK_TABLE_VALID_BLOCKS: ""
        if not block_table_valid_blocks
        else _format_int_list(block_table_valid_blocks),
        RUNTIME_NUM_HEADS: str(num_heads),
        RUNTIME_NUM_KEY_VALUE_HEADS: str(num_kv_heads),
        RUNTIME_SPARSE_MODE: str(sparse_mode),
        RUNTIME_INPUT_LAYOUT: input_layout,
        RUNTIME_BLOCK_SIZE: str(_BLOCK_SIZE),
        RUNTIME_METADATA_COMPLETENESS: "shape_grid_scene_generated",
        RUNTIME_OPERATOR_INPUT_SHAPES_RAW: _build_shape_cell(raw_input_shapes),
    }
    return values


def _scene_runtime_metadata(scene_name: str, values: dict[str, str]) -> dict[str, str]:
    result = dict(values)
    result[RUNTIME_SOURCE_PROFILE] = scene_name
    return result


def _profile_signature(row: TheoryShapeRow) -> tuple:
    return (
        tuple(row.input_shapes),
        row.extra_values.get("Input Data Types", ""),
        row.extra_values.get("Input Formats", ""),
        tuple(row.output_shapes),
    )


def _should_emit(row: TheoryShapeRow, seen: set[tuple]) -> bool:
    signature = _profile_signature(row)
    if signature in seen:
        return False
    seen.add(signature)
    return True


def _iter_pairs(left: list[int], right: list[int]):
    for first in left:
        for second in right:
            yield first, second


def _model_scene_key(cfg: ModelConfig) -> str:
    return cfg.model_key or _normalize_model_name(cfg.name)


def _dense_scene_grids(model_key: str) -> tuple[list[int], list[int], list[int], list[int]]:
    if model_key == "qwen332b":
        return (
            _QWEN3_DENSE_PREFILL_BATCHES,
            _QWEN3_DENSE_PREFILL_SEQS,
            _QWEN3_DENSE_DECODE_BATCHES,
            _QWEN3_DENSE_DECODE_AVG_SEQS,
        )
    return (
        _GENERIC_PREFILL_BATCHES,
        _GENERIC_PREFILL_SEQS,
        _GENERIC_DECODE_BATCHES,
        _GENERIC_DECODE_AVG_SEQS,
    )


def _mla_scene_grids(model_key: str) -> tuple[list[int], list[int], list[int], list[int]]:
    if model_key == "deepseekv3":
        return (
            _DSV3_MLA_PREFILL_BATCHES,
            _DSV3_MLA_PREFILL_SEQS,
            _DSV3_MLA_DECODE_BATCHES,
            _DSV3_MLA_DECODE_AVG_SEQS,
        )
    return (
        _GENERIC_PREFILL_BATCHES,
        _GENERIC_PREFILL_SEQS,
        _GENERIC_DECODE_BATCHES,
        _GENERIC_DECODE_AVG_SEQS,
    )


def _build_dense_prefill_row(
    *,
    scene_name: str,
    batch: int,
    seq: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> TheoryShapeRow:
    tokens = batch * seq
    query = (tokens, num_heads, head_dim)
    key_value = (tokens, num_kv_heads, head_dim)
    input_shapes = _build_31_slots(
        [query, key_value, key_value, (), _DEFAULT_SPARSE_MODE_3_MASK, (batch,), (batch,)]
    )
    output_shapes = [(tokens, num_heads, head_dim)]
    runtime_values = _scene_runtime_metadata(
        scene_name,
        _runtime_metadata(
            avg_seq_len=seq,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            sparse_mode=3,
            input_layout="TND",
            raw_input_shapes=input_shapes,
            actual_seq_lengths_values=_cumulative([seq] * batch),
            actual_seq_lengths_kv_values=_cumulative([seq] * batch),
            block_table_shape=None,
            block_table_valid_blocks=None,
        ),
    )
    return TheoryShapeRow(
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        extra_values={**runtime_values, **_fia_input_metadata(input_shapes)},
    )


def _build_dense_decode_row(
    *,
    scene_name: str,
    batch: int,
    avg_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> TheoryShapeRow:
    needed_blocks = max(1, math.ceil(avg_seq_len / _BLOCK_SIZE))
    total_blocks = max(_MIN_TOTAL_BLOCKS, batch * needed_blocks)
    query = (batch, num_heads, head_dim)
    key_value = (total_blocks, num_kv_heads, _BLOCK_SIZE, head_dim)
    block_table = (batch, needed_blocks)
    raw_input_shapes = _build_31_slots(
        [
            query,
            key_value,
            key_value,
            (),
            _DEFAULT_SPARSE_MODE_3_MASK,
            (batch,),
            (batch,),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            block_table,
        ]
    )
    output_shapes = [(batch, num_heads, head_dim)]
    runtime_values = _scene_runtime_metadata(
        scene_name,
        _runtime_metadata(
            avg_seq_len=avg_seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            sparse_mode=3,
            input_layout="TND",
            raw_input_shapes=raw_input_shapes,
            actual_seq_lengths_values=_cumulative([1] * batch),
            actual_seq_lengths_kv_values=[avg_seq_len] * batch,
            block_table_shape=block_table,
            block_table_valid_blocks=[needed_blocks] * batch,
        ),
    )
    return TheoryShapeRow(
        input_shapes=raw_input_shapes,
        output_shapes=output_shapes,
        extra_values={**runtime_values, **_fia_input_metadata(raw_input_shapes)},
    )


def _build_mla_prefill_row(
    *,
    scene_name: str,
    batch: int,
    seq: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> TheoryShapeRow:
    tokens = batch * seq
    needed_blocks = max(1, math.ceil(seq / _BLOCK_SIZE))
    total_blocks = max(_MIN_TOTAL_BLOCKS, batch * needed_blocks)
    raw_query = (batch, num_heads, seq, kv_lora_rank)
    kv = (total_blocks, 1, _BLOCK_SIZE, kv_lora_rank)
    block_table = (batch, needed_blocks)
    q_rope = (batch, num_heads, seq, qk_rope_head_dim)
    k_rope = (total_blocks, 1, _BLOCK_SIZE, qk_rope_head_dim)
    raw_input_shapes = _build_31_slots(
        [
            raw_query,
            kv,
            kv,
            (),
            _DEFAULT_SPARSE_MODE_3_MASK,
            (batch,),
            (batch,),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            block_table,
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            q_rope,
            k_rope,
        ]
    )
    output_shapes = [(num_heads, batch, seq, kv_lora_rank)]
    runtime_values = _scene_runtime_metadata(
        scene_name,
        _runtime_metadata(
            avg_seq_len=seq,
            num_heads=num_heads,
            num_kv_heads=1,
            sparse_mode=3,
            input_layout="BNSD_NBSD",
            raw_input_shapes=raw_input_shapes,
            actual_seq_lengths_values=[seq] * batch,
            actual_seq_lengths_kv_values=[seq] * batch,
            block_table_shape=block_table,
            block_table_valid_blocks=[needed_blocks] * batch,
        ),
    )
    return TheoryShapeRow(
        input_shapes=raw_input_shapes,
        output_shapes=output_shapes,
        extra_values={**runtime_values, **_fia_input_metadata(raw_input_shapes)},
    )


def _build_mla_decode_row(
    *,
    scene_name: str,
    batch: int,
    avg_seq_len: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> TheoryShapeRow:
    needed_blocks = max(1, math.ceil(avg_seq_len / _BLOCK_SIZE))
    total_blocks = max(_MIN_TOTAL_BLOCKS, batch * needed_blocks)
    raw_query = (batch, num_heads, 1, kv_lora_rank)
    kv = (total_blocks, 1, _BLOCK_SIZE, kv_lora_rank)
    block_table = (batch, needed_blocks)
    q_rope = (batch, num_heads, 1, qk_rope_head_dim)
    k_rope = (total_blocks, 1, _BLOCK_SIZE, qk_rope_head_dim)
    raw_input_shapes = _build_31_slots(
        [
            raw_query,
            kv,
            kv,
            (),
            (),
            (batch,),
            (batch,),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            block_table,
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            q_rope,
            k_rope,
        ]
    )
    output_shapes = [(num_heads, batch, 1, kv_lora_rank)]
    runtime_values = _scene_runtime_metadata(
        scene_name,
        _runtime_metadata(
            avg_seq_len=avg_seq_len,
            num_heads=num_heads,
            num_kv_heads=1,
            sparse_mode=0,
            input_layout="BNSD_NBSD",
            raw_input_shapes=raw_input_shapes,
            actual_seq_lengths_values=[1] * batch,
            actual_seq_lengths_kv_values=[avg_seq_len] * batch,
            block_table_shape=block_table,
            block_table_valid_blocks=[needed_blocks] * batch,
        ),
    )
    return TheoryShapeRow(
        input_shapes=raw_input_shapes,
        output_shapes=output_shapes,
        extra_values={**runtime_values, **_fia_input_metadata(raw_input_shapes)},
    )


def generate_fused_attention_rows(
    model_names: list[str] | None = None,
) -> Generator[TheoryShapeRow, None, None]:
    seen: set[tuple] = set()
    for cfg in resolve_configs(model_names):
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_kv_heads
        model_key = _model_scene_key(cfg)

        if cfg.is_mla():
            prefill_batches, prefill_seqs, decode_batches, decode_avg_seqs = _mla_scene_grids(model_key)
            for batch, seq in _iter_pairs(prefill_batches, prefill_seqs):
                row = _build_mla_prefill_row(
                    scene_name=f"{model_key}_mla_prefill",
                    batch=batch,
                    seq=seq,
                    num_heads=num_heads,
                    kv_lora_rank=cfg.kv_lora_rank,
                    qk_nope_head_dim=cfg.qk_nope_head_dim,
                    qk_rope_head_dim=cfg.qk_rope_head_dim,
                )
                if _should_emit(row, seen):
                    yield row
            for batch, avg_seq_len in _iter_pairs(decode_batches, decode_avg_seqs):
                row = _build_mla_decode_row(
                    scene_name=f"{model_key}_mla_decode",
                    batch=batch,
                    avg_seq_len=avg_seq_len,
                    num_heads=num_heads,
                    kv_lora_rank=cfg.kv_lora_rank,
                    qk_rope_head_dim=cfg.qk_rope_head_dim,
                )
                if _should_emit(row, seen):
                    yield row
            continue

        prefill_batches, prefill_seqs, decode_batches, decode_avg_seqs = _dense_scene_grids(model_key)
        for batch, seq in _iter_pairs(prefill_batches, prefill_seqs):
            row = _build_dense_prefill_row(
                scene_name=f"{model_key}_dense_prefill",
                batch=batch,
                seq=seq,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=cfg.head_dim,
            )
            if _should_emit(row, seen):
                yield row
        for batch, avg_seq_len in _iter_pairs(decode_batches, decode_avg_seqs):
            row = _build_dense_decode_row(
                scene_name=f"{model_key}_dense_decode",
                batch=batch,
                avg_seq_len=avg_seq_len,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=cfg.head_dim,
            )
            if _should_emit(row, seen):
                yield row

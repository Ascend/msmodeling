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

"""Unified constraint-driven fallback for shape grid generation.

When coverage interpolation (stage 2) does not fill ``--rows``, this module
synthesizes additional shapes using build_case-derived constraints. Each
operator registers a ``ConstraintSpec`` describing its fixed axes (locked
to database-confirmed single values) and variable axes (free to enumerate).

Every generated shape satisfies the algebraic constraints that the operator's
``op_replay.build_case`` relies on to reconstruct auxiliary tensors, plus
conservative runtime-safety rules that reduce microbench rejection rate.

Conservative safety rules (applied to all operators):
  - total_blocks >= batch (avoid block_tables wrap-around)
  - max_blocks_per_seq * block_size >= topk (avoid topk padding)
  - num_tokens <= 131072 (HBM OOM guard)
  - batch <= 256 (reasonable serving range)
  - HBM estimate <= 32 GiB (per-card budget)

See: tools/perf_data_collection/op_replay/{LightningIndexer,SparseFlashAttention}_run.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable

from .utils import (
    INPUT_SHAPES_COLUMN,
    OUTPUT_SHAPES_COLUMN,
    profile_dedupe_key,
)

# ── Conservative safety limits ───────────────────────────────────────────────

_MAX_NUM_TOKENS = 131072
_MAX_BATCH = 256
_MAX_HBM_BYTES = 32 * 1024 ** 3
_DTYPE_BYTES = {"DT_BF16": 2, "DT_INT32": 4, "INT32": 4}


@dataclass(frozen=True)
class AxisDomain:
    """A named axis with its enumeration domain."""
    name: str
    values: tuple[int, ...]


@dataclass(frozen=True)
class ConstraintSpec:
    """Operator-specific constraint specification for fallback generation.

    ``fixed_shapes`` is a function(num_tokens, batch, total_blocks,
    max_blocks_per_seq, **extra) returning a _FallbackShape.
    ``extra_domains`` holds operator-specific variable axes beyond the
    common four (e.g., num_heads for SparseFlashAttention).
    """
    kernel_type: str
    common_domains: tuple[AxisDomain, ...]
    extra_domains: tuple[AxisDomain, ...] = ()
    shape_builder: Callable[..., "_FallbackShape"] = field(default=None, repr=False)
    hbm_estimator: Callable[..., int] = field(default=None, repr=False)


@dataclass(frozen=True)
class _FallbackShape:
    kernel_type: str
    input_shapes: list[list[tuple[int, ...]]]
    input_dtypes: list[str]
    input_formats: list[str]
    output_shapes: list[tuple[int, ...]]
    output_dtypes: list[str]
    output_formats: list[str]


# ── Common variable-axis domains ─────────────────────────────────────────────

_NUM_TOKENS = AxisDomain("num_tokens", (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072,
))
_TOTAL_BLOCKS = AxisDomain("total_blocks", (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192,
))
_BATCH = AxisDomain("batch", (1, 2, 4, 8, 16, 32, 64, 128, 256))
_MAX_BLOCKS_PER_SEQ = AxisDomain("max_blocks_per_seq", (16, 32, 64, 128, 512, 1584, 2048, 8192))

_COMMON_DOMAINS = (_NUM_TOKENS, _TOTAL_BLOCKS, _BATCH, _MAX_BLOCKS_PER_SEQ)


# ── LightningIndexer constraints ─────────────────────────────────────────────

_LI_NUM_HEADS = 32
_LI_HEAD_DIM = 128
_LI_BLOCK_SIZE = 128
_LI_TOPK = 2048


def _li_shapes(num_tokens, batch, total_blocks, max_blocks_per_seq, **_):
    return _FallbackShape(
        kernel_type="LightningIndexer",
        input_shapes=[
            [(num_tokens, _LI_NUM_HEADS, _LI_HEAD_DIM)],
            [(total_blocks, _LI_BLOCK_SIZE, 1, _LI_HEAD_DIM)],
            [(num_tokens, _LI_NUM_HEADS)],
            [(batch,)],
            [(batch,)],
            [(batch, max_blocks_per_seq)],
        ],
        input_dtypes=["DT_BF16", "DT_BF16", "DT_BF16", "INT32", "INT32", "INT32"],
        input_formats=["ND"] * 6,
        output_shapes=[(num_tokens, 1, _LI_TOPK), (num_tokens, 1, _LI_TOPK)],
        output_dtypes=["INT32", "DT_BF16"],
        output_formats=["ND", "ND"],
    )


def _li_hbm(num_tokens, batch, total_blocks, max_blocks_per_seq, **_):
    return (
        num_tokens * _LI_NUM_HEADS * _LI_HEAD_DIM * 2
        + total_blocks * _LI_BLOCK_SIZE * _LI_HEAD_DIM * 2
        + num_tokens * _LI_NUM_HEADS * 2
        + batch * max_blocks_per_seq * 4
    )


# ── SparseFlashAttention constraints ─────────────────────────────────────────

_SFA_BLOCK_SIZE = 128
_SFA_KV_LORA_RANK = 512
_SFA_QK_ROPE_DIM = 64
_SFA_TOPK = 2048
_SFA_NUM_HEADS = AxisDomain("num_heads", (2, 4, 8, 16, 32, 64))


def _sfa_shapes(num_tokens, batch, total_blocks, max_blocks_per_seq, num_heads=64, **_):
    klr, bs, qrd = _SFA_KV_LORA_RANK, _SFA_BLOCK_SIZE, _SFA_QK_ROPE_DIM
    return _FallbackShape(
        kernel_type="SparseFlashAttention",
        input_shapes=[
            [(num_tokens, num_heads, klr)],
            [(total_blocks, bs, 1, klr)],
            [(total_blocks, bs, 1, klr)],
            [(num_tokens, 1, _SFA_TOPK)],
            [(batch, max_blocks_per_seq)],
            [(batch,)],
            [(batch,)],
            [(num_tokens, num_heads, qrd)],
            [(total_blocks, bs, 1, qrd)],
        ],
        input_dtypes=["DT_BF16"] * 3 + ["INT32"] * 3 + ["DT_BF16"] * 2 + ["DT_BF16"],
        input_formats=["ND"] * 9,
        output_shapes=[(num_tokens, num_heads, klr)],
        output_dtypes=["DT_BF16"],
        output_formats=["ND"],
    )


def _sfa_hbm(num_tokens, batch, total_blocks, max_blocks_per_seq, num_heads=64, **_):
    klr, qrd, bs = _SFA_KV_LORA_RANK, _SFA_QK_ROPE_DIM, _SFA_BLOCK_SIZE
    return (
        num_tokens * num_heads * klr * 2
        + 2 * total_blocks * bs * klr * 2
        + num_tokens * _SFA_TOPK * 4
        + num_tokens * num_heads * qrd * 2
        + total_blocks * bs * qrd * 2
    )


# ── Constraint registry ──────────────────────────────────────────────────────

_REGISTRY: dict[str, ConstraintSpec] = {
    "LightningIndexer": ConstraintSpec(
        kernel_type="LightningIndexer",
        common_domains=_COMMON_DOMAINS,
        shape_builder=_li_shapes,
        hbm_estimator=_li_hbm,
    ),
    "SparseFlashAttention": ConstraintSpec(
        kernel_type="SparseFlashAttention",
        common_domains=_COMMON_DOMAINS,
        extra_domains=(_SFA_NUM_HEADS,),
        shape_builder=_sfa_shapes,
        hbm_estimator=_sfa_hbm,
    ),
}


def _passes_safety_rules(kernel_type, num_tokens, batch, total_blocks, max_blocks_per_seq, hbm):
    """Conservative runtime-safety filter to reduce microbench rejection."""
    if num_tokens > _MAX_NUM_TOKENS:
        return False
    if batch > _MAX_BATCH:
        return False
    if total_blocks < batch:
        return False
    if kernel_type in ("LightningIndexer", "SparseFlashAttention"):
        if max_blocks_per_seq * 128 < 2048:
            return False
    if hbm > _MAX_HBM_BYTES:
        return False
    return True


def _shape_to_text(slots: list[list[tuple[int, ...]]]) -> str:
    rendered = []
    for slot in slots:
        for shape in slot:
            rendered.append(",".join(str(d) for d in shape))
    return ";".join(rendered)


def _fallback_to_row(shape, headers, template_row):
    row = dict(template_row)
    row[INPUT_SHAPES_COLUMN] = _shape_to_text(shape.input_shapes)
    row[OUTPUT_SHAPES_COLUMN] = ";".join(
        ",".join(str(d) for d in s) for s in shape.output_shapes
    )
    for key in list(row):
        if key.startswith("Runtime "):
            row[key] = ""
    for key in list(row):
        lower = key.lower()
        if (
            "duration" in lower
            or "aicore_time" in lower
            or "aic_" in lower
            or "aiv_" in lower
            or "cube_utilization" in lower
        ) and "Runtime" not in key:
            row[key] = ""
    row["OP State"] = "static"
    row["Accelerator Core"] = template_row.get("Accelerator Core", "MIX_AIC")
    if "Runtime source_profile" in headers:
        row["Runtime source_profile"] = "constraint_fallback"
    return row


def build_constraint_fallback_rows(
    *,
    kernel_type: str,
    csv_path: Path,
    headers: list[str],
    source_rows: list[dict[str, str]],
    row_limit: int,
    seed: int = 0,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Generate up to ``row_limit`` new rows via constrained Cartesian product.

    Only operators registered in _REGISTRY are handled; others return empty.
    Every candidate passes _passes_safety_rules before being emitted.
    """
    spec = _REGISTRY.get(kernel_type)
    if spec is None or not source_rows:
        return [], {"attempted": 0, "duplicates": 0, "appended": 0, "rejected_safety": 0}

    template_row = source_rows[0]
    seen = {profile_dedupe_key(csv_path, headers, row) for row in source_rows}
    generated: list[dict[str, str]] = []
    duplicate_count = 0
    attempted = 0
    rejected_safety = 0

    domains = spec.common_domains + spec.extra_domains
    for combo in product(*(d.values for d in domains)):
        attempted += 1
        kwargs = dict(zip([d.name for d in domains], combo))
        nt, tb, b, mbs = kwargs["num_tokens"], kwargs["total_blocks"], kwargs["batch"], kwargs["max_blocks_per_seq"]
        hbm = spec.hbm_estimator(**kwargs)
        if not _passes_safety_rules(kernel_type, nt, b, tb, mbs, hbm):
            rejected_safety += 1
            continue
        shape = spec.shape_builder(**kwargs)
        row = _fallback_to_row(shape, headers, template_row)
        key = profile_dedupe_key(csv_path, headers, row)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        generated.append(row)
        if len(generated) >= row_limit:
            break

    return generated, {
        "attempted": attempted,
        "duplicates": duplicate_count,
        "appended": len(generated),
        "rejected_safety": rejected_safety,
    }

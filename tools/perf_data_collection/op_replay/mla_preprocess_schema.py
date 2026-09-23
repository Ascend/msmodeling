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

"""CSV runtime-metadata contract for the MLA preprocess replay adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


MLA_PREPROCESS_KERNEL = "mla_preprocess_0_mix_aic"
MLA_PREPROCESS_CACHE_MODE = "krope_ctkv"
MLA_PREPROCESS_QUANT_MODE = "per_tensor_quant_asymm"
RUNTIME_CASE_ID = "Runtime case_id"
RUNTIME_NUM_TOKENS = "Runtime num_tokens"
RUNTIME_HIDDEN_SIZE = "Runtime hidden_size"
RUNTIME_LOCAL_NUM_HEADS = "Runtime local_num_heads"
RUNTIME_Q_LORA_RANK = "Runtime q_lora_rank"
RUNTIME_KV_LORA_RANK = "Runtime kv_lora_rank"
RUNTIME_QK_NOPE_HEAD_DIM = "Runtime qk_nope_head_dim"
RUNTIME_QK_ROPE_HEAD_DIM = "Runtime qk_rope_head_dim"
RUNTIME_BLOCK_SIZE = "Runtime block_size"
RUNTIME_CACHE_MODE = "Runtime cache_mode"
RUNTIME_QUANT_MODE = "Runtime quant_mode"
RUNTIME_ENABLE_INNER_OUT = "Runtime enable_inner_out"
RUNTIME_WEIGHT_QUANTIZED = "Runtime weight_quantized"
RUNTIME_WEIGHT_FORMAT = "Runtime weight_format"
RUNTIME_SOURCE_PROFILE = "Runtime source_profile"
RUNTIME_METADATA_COMPLETENESS = "Runtime metadata_completeness"

MLA_PREPROCESS_RUNTIME_COLUMNS = (
    RUNTIME_CASE_ID,
    RUNTIME_NUM_TOKENS,
    RUNTIME_HIDDEN_SIZE,
    RUNTIME_LOCAL_NUM_HEADS,
    RUNTIME_Q_LORA_RANK,
    RUNTIME_KV_LORA_RANK,
    RUNTIME_QK_NOPE_HEAD_DIM,
    RUNTIME_QK_ROPE_HEAD_DIM,
    RUNTIME_BLOCK_SIZE,
    RUNTIME_CACHE_MODE,
    RUNTIME_QUANT_MODE,
    RUNTIME_ENABLE_INNER_OUT,
    RUNTIME_WEIGHT_QUANTIZED,
    RUNTIME_WEIGHT_FORMAT,
    RUNTIME_SOURCE_PROFILE,
    RUNTIME_METADATA_COMPLETENESS,
)
MLA_PREPROCESS_INPUT_DTYPES = (
    "DT_BF16",
    "DT_INT8",
    "DT_FLOAT",
    "DT_BF16",
    "DT_BF16",
    "DT_INT8",
    "DT_FLOAT",
    "DT_BF16",
    "DT_BF16",
    "DT_BF16",
    "DT_BF16",
    "DT_BF16",
    "DT_BF16",
    "DT_INT32",
    "DT_BF16",
    "DT_INT8",
    "DT_INT32",
    "DT_BF16",
    "DT_INT8",
    "DT_INT32",
    "DT_BF16",
    "DT_BF16",
)
MLA_PREPROCESS_INPUT_FORMATS = (
    "ND",
    "FRACTAL_NZ",
    "ND",
    "ND",
    "ND",
    "FRACTAL_NZ",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
)
MLA_PREPROCESS_OUTPUT_DTYPES = ("DT_BF16",) * 5
MLA_PREPROCESS_OUTPUT_FORMATS = ("ND",) * 5


def _positive_int(row: Mapping[str, str], column: str) -> int:
    raw_value = (row.get(column, "") or "").strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{column} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{column} must be positive, got {value}")
    return value


def _boolean(row: Mapping[str, str], column: str) -> bool:
    normalized = (row.get(column, "") or "").strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{column} must be true or false, got {row.get(column, '')!r}")


@dataclass(frozen=True)
class MlaPreprocessRuntime:
    """Architecture-neutral parameters needed to reconstruct one replay."""

    case_id: str
    num_tokens: int
    hidden_size: int
    local_num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    block_size: int
    cache_mode: str
    quant_mode: str
    enable_inner_out: bool

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "MlaPreprocessRuntime":
        missing = [
            column
            for column in MLA_PREPROCESS_RUNTIME_COLUMNS
            if not (row.get(column, "") or "").strip()
        ]
        if missing:
            raise ValueError(
                f"Complete MLA preprocess row is missing Runtime columns: {', '.join(missing)}"
            )
        if (row[RUNTIME_METADATA_COMPLETENESS] or "").strip().lower() == "legacy":
            raise ValueError(
                "MLA preprocess replay does not accept legacy Runtime metadata"
            )

        case_id = (row[RUNTIME_CASE_ID] or "").strip()
        cache_mode = (row[RUNTIME_CACHE_MODE] or "").strip()
        quant_mode = (row[RUNTIME_QUANT_MODE] or "").strip()
        enable_inner_out = _boolean(row, RUNTIME_ENABLE_INNER_OUT)
        weight_quantized = _boolean(row, RUNTIME_WEIGHT_QUANTIZED)
        weight_format = (row[RUNTIME_WEIGHT_FORMAT] or "").strip()
        if (
            cache_mode != MLA_PREPROCESS_CACHE_MODE
            or quant_mode != MLA_PREPROCESS_QUANT_MODE
            or not enable_inner_out
            or not weight_quantized
            or weight_format != "FRACTAL_NZ"
        ):
            raise ValueError(
                "MLA preprocess row is outside the supported quantized paged-cache regime"
            )

        runtime = cls(
            case_id=case_id,
            num_tokens=_positive_int(row, RUNTIME_NUM_TOKENS),
            hidden_size=_positive_int(row, RUNTIME_HIDDEN_SIZE),
            local_num_heads=_positive_int(row, RUNTIME_LOCAL_NUM_HEADS),
            q_lora_rank=_positive_int(row, RUNTIME_Q_LORA_RANK),
            kv_lora_rank=_positive_int(row, RUNTIME_KV_LORA_RANK),
            qk_nope_head_dim=_positive_int(row, RUNTIME_QK_NOPE_HEAD_DIM),
            qk_rope_head_dim=_positive_int(row, RUNTIME_QK_ROPE_HEAD_DIM),
            block_size=_positive_int(row, RUNTIME_BLOCK_SIZE),
            cache_mode=cache_mode,
            quant_mode=quant_mode,
            enable_inner_out=enable_inner_out,
        )
        if runtime.hidden_size % 32 or runtime.q_lora_rank % 32:
            raise ValueError(
                "MLA preprocess FRACTAL_NZ dimensions must be divisible by 32"
            )
        return runtime

    def shapes(self) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Derive the physical input/output descriptors from runtime metadata."""
        fused_qkv_dim = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        q_output_dim = self.local_num_heads * (
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        cache_blocks = math.ceil(self.num_tokens / self.block_size)
        inputs = [
            (self.num_tokens, self.hidden_size),
            (1, self.hidden_size // 32, fused_qkv_dim, 32),
            (fused_qkv_dim,),
            (self.q_lora_rank,),
            (self.q_lora_rank,),
            (1, self.q_lora_rank // 32, q_output_dim, 32),
            (q_output_dim,),
            (self.kv_lora_rank,),
            (self.num_tokens, self.qk_rope_head_dim),
            (self.num_tokens, self.qk_rope_head_dim),
            (self.local_num_heads, self.qk_nope_head_dim, self.kv_lora_rank),
            (cache_blocks, self.block_size, 1, self.kv_lora_rank),
            (cache_blocks, self.block_size, 1, self.qk_rope_head_dim),
            (self.num_tokens,),
            (1,),
            (1,),
            (fused_qkv_dim,),
            (1,),
            (1,),
            (q_output_dim,),
            (1,),
            (1,),
        ]
        outputs = [
            (self.num_tokens, self.local_num_heads, self.kv_lora_rank),
            inputs[11],
            (self.num_tokens, self.local_num_heads, self.qk_rope_head_dim),
            inputs[12],
            (self.num_tokens, self.q_lora_rank),
        ]
        return inputs, outputs


__all__ = [
    name for name in globals() if name.startswith(("MLA_PREPROCESS_", "RUNTIME_"))
] + ["MlaPreprocessRuntime"]

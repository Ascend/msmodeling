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

"""Declarative operator identity used by replay and profiler aggregation.

Only metadata lives here.  Replay callables stay in their explicit adapter
modules so importing this registry never initializes an NPU runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimeSignatureMode(str, Enum):
    """How profiler rows are associated with one database row."""

    SHAPE = "shape"
    CASE_ID = "case_id"
    ATTENTION_RUNTIME = "attention_runtime"


@dataclass(frozen=True)
class OperatorMetadata:
    canonical_name: str
    aliases: tuple[str, ...] = ()
    runtime_signature_mode: RuntimeSignatureMode = RuntimeSignatureMode.SHAPE
    profiler_task_type: str | None = None
    profiler_kernel_prefix: str | None = None
    supports_case_sharding: bool = True

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.canonical_name, *self.aliases)


_OPERATORS = (
    # --- Manual / collective adapters (not case-shardable) ---
    OperatorMetadata("DispatchFFNCombine", supports_case_sharding=False),
    OperatorMetadata("FusedInferAttentionScore", supports_case_sharding=False),
    OperatorMetadata("QuantBatchMatmulV3", supports_case_sharding=False),
    OperatorMetadata("RINGMLAPrefillBF16Kernel", supports_case_sharding=False),
    # --- OpReplay-based adapters with aliases ---
    OperatorMetadata("BatchMatMulV2", ("BatchMatMulNd",)),
    OperatorMetadata("Cast", ("CastAiCore",)),
    OperatorMetadata("MatMulV2", ("MatMulV3", "MatMulCommon")),
    OperatorMetadata("Mul", ("MulAiCore",)),
    OperatorMetadata("ScatterNdUpdate", ("ScatterNdUpdateAiCore",)),
    OperatorMetadata("Slice", ("SliceAiCore",)),
    OperatorMetadata("Transpose", ("TransposeAiCore",)),
    # --- OpReplay-based adapters without aliases ---
    OperatorMetadata("Add"),
    OperatorMetadata("AddRmsNormBias"),
    OperatorMetadata("ArgMaxV2"),
    OperatorMetadata("AscendQuantV2"),
    OperatorMetadata("DynamicQuant"),
    OperatorMetadata("Fill"),
    OperatorMetadata("GatherV2"),
    OperatorMetadata("GroupedMatmul"),
    OperatorMetadata("GroupedMatmulSwigluQuant"),
    OperatorMetadata("Index"),
    OperatorMetadata("InterleaveRope"),
    OperatorMetadata("KvRmsNormRopeCache"),
    OperatorMetadata("LayerNormV3"),
    OperatorMetadata("MaskedFill"),
    OperatorMetadata("MoeGatingTopK"),
    OperatorMetadata("MoeTokenPermute"),
    OperatorMetadata("MoeTokenUnpermute"),
    OperatorMetadata("PadV3"),
    OperatorMetadata("ReshapeAndCacheNdKernel"),
    OperatorMetadata("RmsNorm"),
    OperatorMetadata("SoftmaxV2"),
    OperatorMetadata("Sort"),
    OperatorMetadata("SwiGlu"),
    OperatorMetadata("TensorMove"),
    OperatorMetadata("TransposeBatchMatMul"),
    OperatorMetadata("_triton_rope_siso"),
    OperatorMetadata("split_qkv_rmsnorm_rope_kernel"),
    OperatorMetadata(
        "LightningIndexer",
        runtime_signature_mode=RuntimeSignatureMode.ATTENTION_RUNTIME,
    ),
    OperatorMetadata(
        "SparseFlashAttention",
        runtime_signature_mode=RuntimeSignatureMode.ATTENTION_RUNTIME,
    ),
    OperatorMetadata(
        "mla_preprocess_0_mix_aic",
        aliases=("mla_preprocess",),
        runtime_signature_mode=RuntimeSignatureMode.CASE_ID,
        profiler_task_type="MIX_AIC",
        profiler_kernel_prefix="mla_preprocess_0_mix_aic",
    ),
)
_METADATA_BY_NAME = {
    name: metadata for metadata in _OPERATORS for name in metadata.all_names
}


def get_operator_metadata(operator_name: str) -> OperatorMetadata:
    """Return declared metadata or a shape-signature default."""
    normalized = (
        operator_name.removesuffix(".csv").removesuffix("_run.py").removesuffix("_run")
    )
    return _METADATA_BY_NAME.get(
        normalized,
        OperatorMetadata(normalized, supports_case_sharding=False),
    )


def profiler_aliases(operator_name: str) -> tuple[str, ...]:
    """Return equivalent profiler names, excluding the requested name."""
    metadata = get_operator_metadata(operator_name)
    normalized = (
        operator_name.removesuffix(".csv").removesuffix("_run.py").removesuffix("_run")
    )
    return tuple(name for name in metadata.all_names if name != normalized)


def profiler_alias_map() -> dict[str, tuple[str, ...]]:
    """Return the legacy symmetric alias view used by aggregation callers."""
    return {
        name: profiler_aliases(name)
        for metadata in _OPERATORS
        for name in metadata.all_names
        if metadata.aliases
    }


def runtime_aware_operator_names(*, include_aliases: bool = False) -> frozenset[str]:
    """Return operators whose write-back identity includes runtime metadata."""
    return frozenset(
        name
        for metadata in _OPERATORS
        if metadata.runtime_signature_mode is not RuntimeSignatureMode.SHAPE
        for name in (
            metadata.all_names if include_aliases else (metadata.canonical_name,)
        )
    )


def supports_case_sharding(operator_name: str) -> bool:
    """Return whether the replay adapter honors the internal shard options."""
    return get_operator_metadata(operator_name).supports_case_sharding


__all__ = [
    "OperatorMetadata",
    "RuntimeSignatureMode",
    "get_operator_metadata",
    "profiler_alias_map",
    "profiler_aliases",
    "runtime_aware_operator_names",
    "supports_case_sharding",
]

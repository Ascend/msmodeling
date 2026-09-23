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

"""Comparison-time operator naming policy.

Theory Spec YAML declares which Tensor slots to compare. This module only
normalizes semantic operator names for one-to-one alignment; Spec may override
the defaults through ``operator_aliases``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

# Theory semantic linear names and TensorCast quantized-linear kernels normalize
# to the Runtime canonical operator field (mm) so one_to_one can align pairs.
DEFAULT_OPERATOR_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "tensor_cast.static_quant_linear.default": "mm",
        "tensor_cast.static_quant_linear_int4.default": "mm",
        "tensor_cast.fp8_linear.default": "mm",
        "tensor_cast.mxfp4_linear.default": "mm",
        "o_projection": "mm",
        "down_projection": "mm",
        "gate_up_projection": "mm",
        "q_projection": "mm",
        "k_projection": "mm",
        "v_projection": "mm",
        # Category-2 MoE gate linear (HF/aten mm inside gate when raw-logits=False).
        "moe_gate_linear": "mm",
        # Quantized routed-expert kernels share Theory semantic GMM names.
        # Unqualified entries normalize theory-side/declared names; the
        # tensor_cast.*.default entries below cover Runtime-captured names.
        "grouped_matmul_quant_swiglu": "grouped_matmul_swiglu",
        "grouped_matmul_quant_int4_swiglu": "grouped_matmul_swiglu",
        "grouped_matmul_fp8_swiglu": "grouped_matmul_swiglu",
        "grouped_matmul_mxfp4_swiglu": "grouped_matmul_swiglu",
        "grouped_matmul_quant": "grouped_matmul",
        "grouped_matmul_quant_int4": "grouped_matmul",
        "grouped_matmul_fp8": "grouped_matmul",
        "grouped_matmul_mxfp4": "grouped_matmul",
        "tensor_cast.grouped_matmul_quant_swiglu.default": "grouped_matmul_swiglu",
        "tensor_cast.grouped_matmul_quant_int4_swiglu.default": "grouped_matmul_swiglu",
        "tensor_cast.grouped_matmul_fp8_swiglu.default": "grouped_matmul_swiglu",
        "tensor_cast.grouped_matmul_mxfp4_swiglu.default": "grouped_matmul_swiglu",
        "tensor_cast.grouped_matmul_quant.default": "grouped_matmul",
        "tensor_cast.grouped_matmul_quant_int4.default": "grouped_matmul",
        "tensor_cast.grouped_matmul_fp8.default": "grouped_matmul",
        "tensor_cast.grouped_matmul_mxfp4.default": "grouped_matmul",
        "lm_head": "mm",
        "lm_head_select": "index",
        "mtp_target_select": "index",
        "mtp_target_sampler": "cat",
        "mtp_output": "slice",
        "mtp_input_shift": "shift_and_update_input_ids",
        "mtp_embedding": "embedding",
        "mtp_fusion_projection": "mm",
        "mtp_proposal_select": "index",
        "mtp_proposal_lm_head": "mm",
        "mtp_proposal_sampler": "argmax",
    }
)

def resolve_operator_aliases(
    operator_aliases: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Merge Spec aliases onto comparison defaults (override keys win)."""

    aliases = {**DEFAULT_OPERATOR_ALIASES, **dict(operator_aliases or {})}
    return MappingProxyType(aliases)

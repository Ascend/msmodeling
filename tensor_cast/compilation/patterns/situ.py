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

"""SiTU (Sigmoid-Tanh-Unit) activation pattern registration.

SiTU is a custom gated activation used by Kimi K3's ``SituAndMul``.  The
decomposed aten graph (div/tanh/sigmoid/mul) is fused into the single
``tensor_cast.situ`` op via pattern matching, mirroring how ``swiglu`` is
handled in ``swiglu.py``.

Two pattern variants are registered per dtype:

1. ``linear_beta`` set (K3 default: beta=4.0, linear_beta=25.0) — the up
   branch also goes through ``linear_beta * tanh(up / linear_beta)``.
2. ``linear_beta`` is None — the up branch is used directly.
"""

import logging

import torch
import torch._prims as prims

from ... import config

logger = logging.getLogger(__name__)

_SITU_DTYPE_LIST = [torch.bfloat16, torch.float16]
_K3_BETA = 4.0
_K3_LINEAR_BETA = 25.0


class SituPattern:
    """Build SiTU pattern/replacement pairs for a given dtype."""

    @staticmethod
    def create(dtype):
        def get_inputs():
            shape = (1, 1)
            gate = torch.empty(shape, dtype=dtype, device="meta")
            up = torch.empty(shape, dtype=dtype, device="meta")
            return [gate, up]

        def _make_pattern_with_linear_beta():
            def pattern(gate, up, beta, linear_beta):
                gate_fp32 = prims.convert_element_type(gate, torch.float32)
                up_fp32 = prims.convert_element_type(up, torch.float32)
                # gate branch: beta * tanh(gate / beta) * sigmoid(gate)
                # Python `beta * tensor` traces as mul.Tensor(tensor, scalar)
                # (tensor first, scalar second) — must match actual SituAndMul graph.
                div_g = torch.ops.aten.div.Tensor(gate_fp32, beta)
                tanh_g = torch.ops.aten.tanh.default(div_g)
                mul_beta_tanh = torch.ops.aten.mul.Tensor(tanh_g, beta)
                sigmoid_g = torch.ops.aten.sigmoid.default(gate_fp32)
                situ_a = torch.ops.aten.mul.Tensor(mul_beta_tanh, sigmoid_g)
                # up branch: linear_beta * tanh(up / linear_beta)
                div_u = torch.ops.aten.div.Tensor(up_fp32, linear_beta)
                tanh_u = torch.ops.aten.tanh.default(div_u)
                up_transformed = torch.ops.aten.mul.Tensor(tanh_u, linear_beta)
                # combine
                result = torch.ops.aten.mul.Tensor(situ_a, up_transformed)
                return prims.convert_element_type(result, dtype)

            def replacement(gate, up, beta, linear_beta):
                return torch.ops.tensor_cast.situ(gate, up, beta, linear_beta)

            return pattern, replacement, get_inputs()

        def _make_pattern_no_linear_beta():
            def pattern(gate, up, beta, linear_beta):
                gate_fp32 = prims.convert_element_type(gate, torch.float32)
                up_fp32 = prims.convert_element_type(up, torch.float32)
                div_g = torch.ops.aten.div.Tensor(gate_fp32, beta)
                tanh_g = torch.ops.aten.tanh.default(div_g)
                mul_beta_tanh = torch.ops.aten.mul.Tensor(tanh_g, beta)
                sigmoid_g = torch.ops.aten.sigmoid.default(gate_fp32)
                situ_a = torch.ops.aten.mul.Tensor(mul_beta_tanh, sigmoid_g)
                # up branch: direct (no tanh transform)
                result = torch.ops.aten.mul.Tensor(situ_a, up_fp32)
                return prims.convert_element_type(result, dtype)

            def replacement(gate, up, beta, linear_beta):
                return torch.ops.tensor_cast.situ(gate, up, beta, linear_beta)

            return pattern, replacement, get_inputs()

        pat_lb, repl_lb, inputs_lb = _make_pattern_with_linear_beta()
        pat_nlb, repl_nlb, inputs_nlb = _make_pattern_no_linear_beta()

        return [
            {
                "name": f"situ_with_linear_beta_{dtype}",
                "pattern": pat_lb,
                "replacement": repl_lb,
                "inputs": inputs_lb,
                "scalar_workaround": {"beta": _K3_BETA, "linear_beta": _K3_LINEAR_BETA},
            },
            {
                "name": f"situ_no_linear_beta_{dtype}",
                "pattern": pat_nlb,
                "replacement": repl_nlb,
                "inputs": inputs_nlb,
                "scalar_workaround": {"beta": _K3_BETA, "linear_beta": None},
            },
        ]


_INSTALLED = False


def register_all_patterns():
    """Register SiTU patterns.

    Idempotent: safe to call from ``lazy_init`` (which is lru_cached) or directly
    after K3 model config sets ``enable_situ=True`` at runtime.  The latter case
    matters because ``lazy_init`` may have already run for another model before
    K3 flips the switch on.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    if not config.compilation.fusion_patterns.enable_situ:
        return

    from ... import ops as _situ_ops  # noqa: F401  ensure situ op is registered
    from . import register_pattern

    for dtype in _SITU_DTYPE_LIST:
        patterns_config = SituPattern.create(dtype)
        for pattern in patterns_config:
            register_pattern(
                pattern["name"],
                pattern["pattern"],
                pattern["replacement"],
                pattern["inputs"],
                scalar_workaround=pattern["scalar_workaround"],
            )

    _INSTALLED = True
    logger.info("Registered SiTU activation patterns (situ_with/no_linear_beta).")

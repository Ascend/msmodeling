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

"""SiTU activation op for Kimi K3.

SiTU (Sigmoid-Tanh-Unit) is Kimi K3's custom gated activation:
    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    up = linear_beta * tanh(up / linear_beta)   (when linear_beta is set)
    output = situ_a * up

Reference: ``SituAndMul`` in ``modeling_kimi_linear.py``.
The meta op only performs shape inference; the performance model lives in
``performance_model/__init__.py``.
"""

from typing import Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("situ")
def _(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    """SiTU activation meta op.

    Args:
        gate: Gate tensor, shape ``[..., d]``.
        up: Up tensor, shape ``[..., d]`` (same shape as ``gate``).
        beta: Scaling factor for the gate's tanh. Defaults to 1.0.
        linear_beta: When provided, the up branch is transformed by
                     ``linear_beta * tanh(up / linear_beta)``.

    Returns:
        Meta tensor of shape ``[..., d]`` (same as ``gate``).
    """
    del beta, linear_beta  # scalar args do not affect shape inference
    return torch.empty(gate.shape, dtype=gate.dtype, device="meta")

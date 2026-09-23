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

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("static_false_mask")
def _static_false_mask(mask: torch.Tensor) -> torch.Tensor:
    """Mark a Boolean mask whose elements are known to be false."""
    return mask.clone()


def static_false_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.ops.tensor_cast.static_false_mask.default(mask)

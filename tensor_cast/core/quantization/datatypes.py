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

# _*_coding:utf-8_*_
"""
datatypes of quantization
"""

from enum import auto, Enum


try:
    # Native in Python 3.11+
    from enum import StrEnum
except ImportError:
    # Fallback for Python 3.10
    from strenum import StrEnum


class LinearQuantType(Enum):
    W8A16 = auto()  # Weight in int8, activation in bfloat16 or half
    W8A8 = auto()  # Weight in int8, activation in int8
    W4A8 = auto()  # Weight in int4, activation in int8
    FP8 = auto()  # Weight in float8, activation in float8
    MXFP4 = auto()  # both weight and activation in MXFP4
    HIF8 = auto()  # HiFloat8 W8A8; uint8 is only its payload carrier
    MXFP8 = auto()  # Float8 payload with E8M0 block scales


MX_QUANT_TYPES = (LinearQuantType.MXFP4, LinearQuantType.MXFP8)
MXFP8_GROUP_SIZE = 32


class QuantizeLinearAction(StrEnum):
    DISABLED = "DISABLED"
    W8A16_STATIC = "W8A16_STATIC"
    W8A8_STATIC = "W8A8_STATIC"
    W4A8_STATIC = "W4A8_STATIC"
    W8A16_DYNAMIC = "W8A16_DYNAMIC"
    W8A8_DYNAMIC = "W8A8_DYNAMIC"
    W4A8_DYNAMIC = "W4A8_DYNAMIC"
    FP8 = "FP8"
    MXFP4 = "MXFP4"
    MXFP8 = "MXFP8"
    HIF8 = "HIF8"


class QuantizeAttentionAction(StrEnum):
    # TODO(jgong5): support FP8 quantization
    DISABLED = "DISABLED"
    INT8 = "INT8"
    FP8 = "FP8"
    HIF8 = "HIF8"

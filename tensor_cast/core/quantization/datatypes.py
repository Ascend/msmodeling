# _*_coding:utf-8_*_
"""
datatypes of quantization
"""

try:
    # Native in Python 3.11+
    from enum import StrEnum
except ImportError:
    # Fallback for Python 3.10
    from strenum import StrEnum


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


class QuantizeAttentionAction(StrEnum):
    # TODO(jgong5): support FP8 quantization
    DISABLED = "DISABLED"
    INT8 = "INT8"
    FP8 = "FP8"

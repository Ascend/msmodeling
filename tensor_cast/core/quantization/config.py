# _*_coding:utf-8_*_
"""
config of quantization
"""

import torch

from ...model_config import (
    LinearQuantConfig,
    MultiheadLatentAttentionQuantConfig,
    QuantConfig,
)
from ...quantize_utils import get_attention_quant_type, LinearQuantType
from .datatypes import QuantizeAttentionAction, QuantizeLinearAction


def create_linear_quant_config(quantize_linear_action: QuantizeLinearAction, **kwargs):
    # TODO: support per-channel/per-group setting
    # TODO: support asymmetric quant setting

    if quantize_linear_action in ("W8A16_STATIC", "W8A16_DYNAMIC"):
        quant_type = LinearQuantType.W8A16
    elif quantize_linear_action in ("W8A8_STATIC", "W8A8_DYNAMIC"):
        quant_type = LinearQuantType.W8A8
    elif quantize_linear_action == "FP8":
        quant_type = LinearQuantType.FP8
    elif quantize_linear_action == "MXFP4":
        quant_type = LinearQuantType.MXFP4
        if "weight_group_size" not in kwargs:
            raise ValueError("weight_group_size must be provided for MXFP4 quantization")
    elif quantize_linear_action in ("W4A8_STATIC", "W4A8_DYNAMIC"):
        quant_type = LinearQuantType.W4A8
    else:
        raise ValueError(f"Unsupported quantization action {quantize_linear_action}")

    config_args = {
        "quant_type": quant_type,
    }

    if "weight_scale" not in kwargs and quant_type != LinearQuantType.MXFP4:
        # For MXFP4, weight_scale is created from the weight tensor during model initialization
        config_args["weight_scale"] = torch.tensor(1.0)

    if quantize_linear_action in ("W8A16_STATIC", "W8A8_STATIC", "W4A8_STATIC"):
        config_args["activation_scale"] = torch.tensor(1.0)
    config_args.update(kwargs)
    return LinearQuantConfig(**config_args)


def create_attention_quant_config(quantize_attention_action: QuantizeAttentionAction):
    # default to symmetric quant with dummy scales
    # for simplicity, we use MLA quant config for both MLA and regular attention
    return MultiheadLatentAttentionQuantConfig(
        quant_type=get_attention_quant_type(quantize_attention_action),
        query_scale=torch.tensor(1.0),
        kv_scale=torch.tensor(1.0),
        attention_prob_scale=torch.tensor(1.0),
        kv_projected_scale=torch.tensor(1.0),
        qk_scale=torch.tensor(1.0),
        v_scale=torch.tensor(1.0),
        out_scale=torch.tensor(1.0),
    )


def create_quant_config(
    quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.DISABLED,
    quantize_lmhead: bool = False,
    quantize_attention_action: QuantizeAttentionAction = QuantizeAttentionAction.DISABLED,
    **kwargs,
):
    quant_config = QuantConfig()
    if quantize_linear_action != QuantizeLinearAction.DISABLED:
        quant_config.linear_configs["layers.*"] = create_linear_quant_config(quantize_linear_action, **kwargs)
        quant_config.linear_configs["*.layers.*"] = create_linear_quant_config(quantize_linear_action, **kwargs)
        quant_config.linear_configs["default_dit"] = create_linear_quant_config(quantize_linear_action, **kwargs)
        if quantize_lmhead:
            quant_config.linear_configs["lm_head"] = create_linear_quant_config(quantize_linear_action, **kwargs)
            quant_config.linear_configs["*.lm_head"] = create_linear_quant_config(quantize_linear_action, **kwargs)
    if quantize_attention_action != QuantizeAttentionAction.DISABLED:
        # default to symmetric quant with dummy scales
        quant_config.attention_configs[-1] = create_attention_quant_config(quantize_attention_action)

    return quant_config

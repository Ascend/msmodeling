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


# MXFP4-only weight-group kwargs. They must never leak into a non-MXFP4 config:
# LinearQuantConfig.__post_init__ would otherwise promote the activation
# granularity to PER_GROUP (e.g. turning an FP8 backbone into per-group dynamic).
_MXFP4_ONLY_KWARGS = ("weight_group_size", "weight_quant_granularity")

# Backbone (non-routed-expert) linear layers: attention projections, dense MLP,
# and shared experts across the model families TensorCast supports. Routed MoE
# experts (``*.experts.<id>.*``) are intentionally excluded so they keep the
# broad ``--quantize-linear-action`` quant type.
_BACKBONE_LINEAR_PATTERNS = (
    "*.self_attn.*",
    "*.attn.qkv",
    "*.attn.proj",
    "*.mlp.gate_proj",
    "*.mlp.up_proj",
    "*.mlp.down_proj",
    # DeepSeek-style shared experts: ``...mlp.shared_experts.{gate,up,down}_proj``.
    "*.mlp.shared_experts.gate_proj",
    "*.mlp.shared_experts.up_proj",
    "*.mlp.shared_experts.down_proj",
    # Other shared-expert layouts (per-expert indexed / fused MoE wrappers).
    "*.shared_expert.*.gate_proj",
    "*.shared_expert.*.up_proj",
    "*.shared_expert.*.down_proj",
    "*.mlp.fused_moe.shared_experts.gate_proj",
    "*.mlp.fused_moe.shared_experts.up_proj",
    "*.mlp.fused_moe.shared_experts.down_proj",
)

# Broad patterns cover every linear (including routed experts). ``default_dit``
# is the DiT fallback (see replace_with_quant_modules default_config_name).
_BROAD_LINEAR_PATTERNS = ("layers.*", "*.layers.*", "default_dit")
_LMHEAD_PATTERNS = ("lm_head", "*.lm_head")


def _filter_action_kwargs(action: QuantizeLinearAction, kwargs: dict) -> dict:
    """Strip MXFP4-only kwargs when the target action is not MXFP4."""
    if action == QuantizeLinearAction.MXFP4:
        return kwargs
    return {key: value for key, value in kwargs.items() if key not in _MXFP4_ONLY_KWARGS}


def _set_linear_patterns(quant_config: QuantConfig, patterns, quantize_linear_action: QuantizeLinearAction, **kwargs):
    linear_config = create_linear_quant_config(quantize_linear_action, **kwargs)
    for pattern in patterns:
        quant_config.linear_configs[pattern] = linear_config


def create_quant_config(
    quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.DISABLED,
    quantize_backbone_linear_action: QuantizeLinearAction = QuantizeLinearAction.DISABLED,
    quantize_lmhead: bool = False,
    quantize_attention_action: QuantizeAttentionAction = QuantizeAttentionAction.DISABLED,
    **kwargs,
):
    quant_config = QuantConfig()

    # Register the backbone override BEFORE the broad patterns. get_quant_config()
    # returns the first matching wildcard in insertion order, so backbone-specific
    # patterns must come first to override the broad action for those modules
    # (e.g. ``--quantize-linear-action MXFP4`` for experts + backbone FP8).
    if quantize_backbone_linear_action != QuantizeLinearAction.DISABLED:
        _set_linear_patterns(
            quant_config,
            _BACKBONE_LINEAR_PATTERNS,
            quantize_backbone_linear_action,
            **_filter_action_kwargs(quantize_backbone_linear_action, kwargs),
        )

    if quantize_linear_action != QuantizeLinearAction.DISABLED:
        broad_kwargs = _filter_action_kwargs(quantize_linear_action, kwargs)
        _set_linear_patterns(quant_config, _BROAD_LINEAR_PATTERNS, quantize_linear_action, **broad_kwargs)
        if quantize_lmhead:
            _set_linear_patterns(quant_config, _LMHEAD_PATTERNS, quantize_linear_action, **broad_kwargs)

    if quantize_attention_action != QuantizeAttentionAction.DISABLED:
        # default to symmetric quant with dummy scales
        quant_config.attention_configs[-1] = create_attention_quant_config(quantize_attention_action)

    return quant_config

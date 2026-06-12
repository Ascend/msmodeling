# _*_coding:utf-8_*_
import fnmatch
from enum import auto, Enum
from typing import Callable, Optional

import torch
import torch.nn as nn

from .core.quantization.datatypes import QuantizeAttentionAction
from .utils import DTYPE_FP4, DTYPE_FP8, pattern_match


class LinearQuantType(Enum):
    W8A16 = auto()  # Weight in int8, activation in bfloat16 or half
    W8A8 = auto()  # Weight in int8, activation in int8
    W4A8 = auto()  # Weight in int4, activation in int8
    FP8 = auto()  # Weight in float8, activation in float8
    MXFP4 = auto()  # both weight and activation in MXFP4


def quant_type_to_dynamic_quant_dtype(
    quant_type: LinearQuantType,
) -> Optional[torch.dtype]:
    if quant_type in (LinearQuantType.W8A8, LinearQuantType.W4A8):
        return torch.int8
    elif quant_type == LinearQuantType.FP8:
        return DTYPE_FP8
    elif quant_type == LinearQuantType.MXFP4:
        return DTYPE_FP4
    elif quant_type == LinearQuantType.W8A16:
        return None
    else:
        raise ValueError(f"Unsupported quant_type for dynamic quant: {quant_type}")


def quant_type_to_weight_dtype(quant_type: LinearQuantType) -> torch.dtype:
    if quant_type in (
        LinearQuantType.W8A8,
        LinearQuantType.W4A8,
        LinearQuantType.W8A16,
    ):
        return torch.int8
    elif quant_type == LinearQuantType.FP8:
        return DTYPE_FP8
    elif quant_type == LinearQuantType.MXFP4:
        return DTYPE_FP4
    else:
        raise ValueError(f"Unsupported quant_type for weight quant: {quant_type}")


class AttentionQuantType(Enum):
    INT8 = auto()
    FP8 = auto()


def get_attention_quant_type(action: QuantizeAttentionAction) -> AttentionQuantType:
    try:
        return getattr(AttentionQuantType, action.name)
    except AttributeError:
        raise ValueError(
            f"Unsupported quantization action: {action}. Ensure '{action.name}' is defined in AttentionQuantType."
        ) from None


_QUANT_TYPE_TO_TORCH_DTYPE_MAP = {
    AttentionQuantType.INT8: torch.int8,
    AttentionQuantType.FP8: torch.float8_e4m3fn,
}


def get_torch_dtype_from_quant_type(quant_type: AttentionQuantType) -> torch.dtype:
    if quant_type not in _QUANT_TYPE_TO_TORCH_DTYPE_MAP:
        raise ValueError(
            f"Unsupported attention quant type: {quant_type}. "
            f"Supported types: {list(_QUANT_TYPE_TO_TORCH_DTYPE_MAP.keys())}"
        )
    return _QUANT_TYPE_TO_TORCH_DTYPE_MAP[quant_type]


def get_torch_quant_type(action: QuantizeAttentionAction) -> AttentionQuantType:
    try:
        return getattr(AttentionQuantType, action.name)
    except AttributeError:
        raise ValueError(
            f"Unsupported quantization action: {action}. Ensure '{action.name}' is defined in AttentionQuantType."
        ) from None


class QuantGranularity(Enum):
    PER_TENSOR = auto()  # use a single quant param for the entire tensor
    PER_SAMPLE = auto()  # use quant param per sample in the batch (e.g. per-token for LLM)
    PER_GROUP = auto()  # use quant param per channel group


class QuantScheme(Enum):
    SYMMETRIC = auto()
    ASYMMETRIC = auto()


def get_quant_config(name, quant_config, default_config_name):
    if not hasattr(quant_config, "_cached_wildcard_configs"):
        quant_config._cached_wildcard_configs = {
            n: quant_config.linear_configs[n] for n in quant_config.linear_configs if "*" in n or "?" in n
        }
    wildcard_configs = quant_config._cached_wildcard_configs
    if name in quant_config.linear_configs:
        return quant_config.linear_configs[name]
    for pattern, config in wildcard_configs.items():
        if fnmatch.fnmatch(name, pattern):
            return config
    return quant_config.linear_configs.get(default_config_name)


def replace_module(name, new_module, root_module):
    if not root_module:
        return
    path = name.split(".")
    parent_name, child_name = ".".join(path[:-1]), path[-1]
    parent_module = root_module
    if parent_name:
        parent_module = parent_module.get_submodule(parent_name)
    setattr(parent_module, child_name, new_module)


def quantize_linear_modules(
    root_module: nn.Module,
    quant_linear_cls: Optional["QuantLinearBase"],  # noqa: F821
    quant_config: Optional["QuantConfig"],  # noqa: F821
    default_config_name: str,
    strip_module_fn: Optional[Callable[[str], str]],
) -> None:
    """
    Quantize Linear modules in a root module with specified quantization config and class.

    Args:
        root_module: (nn.Module) Root module containing Linear layers to be quantized
        quant_linear_cls: (QuantLinearBase) Quantized Linear class to replace original Linear modules
        quant_config: (QuantConfig) Quantization configuration object with linear config rules and exclude list
        default_config_name: (str) Fallback config name if no match found for a target Linear module
        strip_module_fn:
            (Optional[Callable[[str], str]]) Function to clean/normalize module names,
            None = use raw module name without modification
    """
    if not quant_linear_cls or not root_module:
        return
    for name, module in root_module.named_modules():
        if pattern_match(name, quant_config.modules_to_not_convert):
            continue
        if isinstance(module, torch.nn.Linear):
            module_name = strip_module_fn(name) if strip_module_fn else name
            cfg = get_quant_config(module_name, quant_config, default_config_name)
            if cfg:
                new_module = quant_linear_cls(module, cfg)
                replace_module(name, new_module, root_module)

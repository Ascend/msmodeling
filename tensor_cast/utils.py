import fnmatch
import logging
import re
from typing import List, Optional, Union

import torch
from transformers.utils.quantization_config import (
    CompressedTensorsConfig,
    FineGrainedFP8Config,
    QuantizationConfigMixin,
)

# placeholder for FP8, don't hard-code specific fp8 format
DTYPE_FP8 = torch.float8_e5m2
# use int4 placeholder for FP4
DTYPE_FP4 = torch.int4


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return str(dtype).startswith("torch.float8")


def performance_dtype(dtype: torch.dtype) -> torch.dtype:
    if is_fp8_dtype(dtype):
        return DTYPE_FP8
    return dtype


logger = logging.getLogger(__name__)


def register_tensor_cast_op(name, mutates_args=(), **kwargs):
    """
    Register tensor_cast custom op with `name` under tensor_cast namespace.
    We only support meta tensor in the tensor_cast ops so the fake implementation
    is the same as the normal implementation.
    """

    def decorator(func):
        custom_op = torch.library.custom_op(f"tensor_cast::{name}", mutates_args=mutates_args, **kwargs)(func)
        custom_op.register_fake(func)
        logger.debug("Registered Operator: tensor_cast::%s", name)
        return func

    return decorator


def exact_division(numerator, denominator):
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


def pattern_match(name: str, pattern_list: List[Optional[str]]) -> bool:
    """
    three ways to match:fnmatch/re/real_name
    example of names:
    # ['lm_head', 're:.*self_attn.*', 're:.*shared_experts.*', 're:.*mlp\\.(gate|up|gate_up|down)_proj.*']
    # ["gate","e_score_correction_bias","lm_head"]
    """
    matched = False
    if not pattern_list:
        return matched
    for pattern in pattern_list:
        if pattern.startswith("re:"):
            pattern = pattern.replace("re:", "")
            matched = bool(re.match(pattern, name))
        elif pattern in name:
            matched = True
        else:
            matched = fnmatch.fnmatch(name, pattern)
        if matched:
            break
    return matched


def get_modules_to_not_convert(
    quant_config: QuantizationConfigMixin,
) -> List[Optional[str]]:
    modules_to_not_convert = []
    if isinstance(quant_config, FineGrainedFP8Config):
        modules_to_not_convert = quant_config.modules_to_not_convert
    elif isinstance(quant_config, CompressedTensorsConfig):
        modules_to_not_convert = quant_config.quantization_config.ignore
    return modules_to_not_convert


_str_to_dtype = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def str_to_dtype(string: str) -> torch.dtype:
    res = _str_to_dtype.get(string)
    if res is None:
        raise ValueError(f"Unsupported type for model: {string}")
    return res


def get_nested_attr(obj, attr: Union[str, List[str]]):
    """Get attribute recursively from an object.

    Args:
        obj: The object to get the attribute from.
        attr: The attribute to get, can be a list of attributes.
    """
    if obj is None:
        return None
    if isinstance(attr, str):
        return getattr(obj, attr, None)
    elif isinstance(attr, list):
        if len(attr) == 0:
            return obj
        else:
            return get_nested_attr(getattr(obj, attr[0], None), attr[1:])


class EquivalentKeyManager:
    """
    Implementation of a Union-Find (Disjoint Set Union, DSU)
    data structure for managing equivalent keys,
    which groups multiple keys into the same equivalence class.

    Core functionalities:
    - Add multiple keys to the same equivalence group
    - Find the root key of the group to which a key belongs
    - Determine the root key of a group based on creation order (oldest root strategy)
    """

    def __init__(self):
        # Map each key to its parent
        self.parent = {}
        # Map each root key to its creation order (for determining oldest root)
        self.root_order = {}

    def _find(self, key):
        """Find the root of the key with path compression."""
        if key not in self.parent:
            raise KeyError(f"Key '{key}' not found in EquivalentKeyManager")
        if self.parent[key] != key:
            self.parent[key] = self._find(self.parent[key])
        return self.parent[key]

    def add_equivalent_keys(self, keys):
        """Add a list of equivalent keys to the same group."""
        if not keys:
            return

        # Ensure all keys are in the parent map
        for key in keys:
            if key not in self.parent:
                self.parent[key] = key
                self.root_order[key] = len(self.root_order)

        # Collect all unique roots of the keys
        roots = set()
        for key in keys:
            roots.add(self._find(key))

        # Find the oldest root (smallest order)
        oldest_root = min(roots, key=lambda r: self.root_order[r])

        # Union all roots to the oldest root
        for root in roots:
            if root != oldest_root:
                self.parent[root] = oldest_root
                # Remove old root from root_order as it's no longer a root
                del self.root_order[root]

    def get_group_root_key(self, key):
        """Get the root key of the group containing the given key."""
        if key not in self.parent:
            return None
        return self._find(key)

# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

import argparse
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, Optional

from tensor_cast.model_config import ParallelConfig


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}
LIMIT_COUNT = 1e6
BYTES_TO_GB = 1024**3
MAX_ITER_NUMS = 10

COMMON_COLUMNS = [
    "device_name",
    "num_devices",
    "model_id",
    "quantize_linear_action",
    "quantize_attention_action",
    "input_length",
    "output_length",
    "concurrency",
    "ttft",
    "tpot",
    "token/s",
    "token/s/device",
    "parallel",
    "batch_size",
]

AGG_COLUMNS = COMMON_COLUMNS + ["percentage_breakdowns(p)", "percentage_breakdowns(d)"]
DISAGG_COLUMNS = COMMON_COLUMNS + ["percentage_breakdowns"]


@dataclass
class OptimizerData:
    input_length: Optional[int] = None
    output_length: Optional[int] = None
    batch_size: Optional[int] = None
    image_batch_size: Optional[int] = None
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    ttft_limits: Optional[float] = None
    tpot_limits: Optional[float] = None
    max_prefill_tokens: Optional[int] = None
    num_devices: Optional[int] = None
    serving_cost: Optional[float] = None
    num_mtp_tokens: Optional[int] = None
    mtp_acceptance_rate: Optional[list] = None
    prefill_devices_per_instance: Optional[int] = None
    decode_devices_per_instance: Optional[int] = None
    prefix_cache_hit_rate: float = 0.0

    def get_effective_input_length(self, is_decode: bool = False):
        if self.input_length is None:
            return None
        effective_hit_rate = 0.0 if is_decode else self.prefix_cache_hit_rate
        cached_prefix_tokens = math.floor(self.input_length * effective_hit_rate)
        effective_input_length = self.input_length - cached_prefix_tokens
        if effective_input_length < 1:
            raise ValueError(
                "Effective input length must be at least 1 after applying prefix cache hit rate. "
                f"Got input_length={self.input_length}, prefix_cache_hit_rate={self.prefix_cache_hit_rate}."
            )
        return effective_input_length


def check_string_valid(string: str, max_len=256):
    if len(string) > max_len:
        raise argparse.ArgumentTypeError(f"String length exceeds {max_len} characters: {string!r}")
    if not re.match(r"^[a-zA-Z0-9_/.-]+$", string):
        raise argparse.ArgumentTypeError(f"String contains invalid characters: {string!r}")
    return string


def check_positive_integer(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a positive integer")
    if value > 1e6:
        raise argparse.ArgumentTypeError(f"{value!r} is too large")
    return value


def check_positive_float(value):
    if value is None:
        return None
    if value.lower() == "inf":
        return float("inf")
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a positive number")
    return value


class BatchRangeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) not in (1, 2):
            raise argparse.ArgumentTypeError(f"{option_string} expects [min max] or [max], got {values}")
        if len(values) == 2 and values[0] > values[1]:
            raise argparse.ArgumentTypeError(f"{option_string} min must be <= max, got {values}")
        if any(v <= 0 for v in values):
            raise argparse.ArgumentTypeError(f"{option_string} values must be > 0, got {values}")
        setattr(namespace, self.dest, values)


def format_breakdowns(breakdowns: Dict[str, Dict[str, float]]):
    # format the breakdowns to a string
    expected_keys = ["Mem", "Comm", "Cube", "Vec"]
    all_values = []
    for sub_dict in breakdowns.values():
        total = sum(sub_dict.values())
        if total == 0:
            continue
        for value in sub_dict.values():
            if isinstance(value, float):
                all_values.append(value / total * 100)

    formatted_parts = []
    for i, key in enumerate(expected_keys):
        if i < len(all_values):
            formatted_parts.append(f"{key} {all_values[i]:.2f}")
        else:
            formatted_parts.append(f"{key} 0.00")

    return " | ".join(formatted_parts)


def resolve_search_sizes(values: list[int] | None, target_devices: int, default_size: int) -> list[int]:
    """Resolve final candidate sizes for a search dimension.

    Args:
        values:
            - None: dimension is not searched, use fixed default_size
            - []: dimension is searched with default range (powers of 2)
            - [v1, v2, ...]: user-provided explicit candidate values
        target_devices: device count used for default range generation.
        default_size: fixed value used when values is None.

    Returns:
        A de-duplicated positive integer list preserving input order.
    """
    if values is None:
        size_list = [default_size]
    elif len(values) == 0:
        size_list = [1 << i for i in range(target_devices.bit_length())]
    else:
        size_list = values

    normalized = []
    for size in size_list:
        if size <= 0 or size in normalized:
            continue
        normalized.append(size)
    return normalized


def format_parallel_label(parallel_config: ParallelConfig, is_moe_model: bool) -> str:
    parts = [
        f"TP={parallel_config.tensor_parallel_size}",
        f"PP={parallel_config.pipeline_parallel_size}",
        f"DP={parallel_config.data_parallel_size}",
    ]
    if is_moe_model:
        parts.extend(
            [
                f"EP={parallel_config.expert_parallel_size}",
                f"MOE-TP={parallel_config.moe_tensor_parallel_size}",
                f"MOE-DP={parallel_config.moe_data_parallel_size}",
            ]
        )
    return " | ".join(parts)

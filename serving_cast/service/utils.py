# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

import argparse
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, TypedDict

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
DEFAULT_MAX_SEARCH_COMBINATIONS = 100
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
    "effective_input_length",
    "max_batched_tokens",
    "prefill_num_chunks",
    "concurrency",
    "ttft",
    "tpot",
    "token/s",
    "token/s/device",
    "parallel",
    "batch_size",
]


class MemoryInfo(TypedDict, total=False):
    total_device_memory_gb: float
    model_weight_size_gb: float
    kv_cache_size_gb: float
    model_activation_size_gb: float
    reserved_memory_gb: float
    device_memory_available_gb: float


MEMORY_KEY_TO_COLUMN = {
    "model_weight_size_gb": "weight_GB",
    "kv_cache_size_gb": "kv_cache_GB",
    "model_activation_size_gb": "activation_GB",
    "device_memory_available_gb": "avail_GB",
}
MEMORY_COLUMNS = list(MEMORY_KEY_TO_COLUMN.values())
# Note: total_device_memory_gb and reserved_memory_gb are constant across
# configurations within the same device, so they are displayed only in the
# text header (OptimizerSummary._memory_info) rather than as table columns.

AGG_COLUMNS = COMMON_COLUMNS + ["percentage_breakdowns(p)", "percentage_breakdowns(d)"] + MEMORY_COLUMNS
DISAGG_COLUMNS = COMMON_COLUMNS + ["percentage_breakdowns"] + MEMORY_COLUMNS


@dataclass
class PrefillChunk:
    index: int
    query_len: int
    seq_len: int


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
    max_batched_tokens: Optional[int] = None
    num_devices: Optional[int] = None
    serving_cost: Optional[float] = None
    num_mtp_tokens: Optional[int] = None
    mtp_acceptance_rate: Optional[list] = None
    prefill_devices_per_instance: Optional[int] = None
    decode_devices_per_instance: Optional[int] = None
    prefix_cache_hit_rate: float = 0.0
    concurrency_search_strategy: str = 'exponential'

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

    def get_prefill_chunk_plan(self) -> list[PrefillChunk]:
        """Split the effective prefill prompt into chunks bounded by max_batched_tokens."""
        effective_input_length = self.get_effective_input_length(is_decode=False)
        if effective_input_length is None:
            return []
        if self.max_batched_tokens is None or self.max_batched_tokens <= 0:
            raise ValueError(f"max_batched_tokens must be a positive integer, got {self.max_batched_tokens!r}.")

        chunks = []
        consumed = 0
        index = 0
        while consumed < effective_input_length:
            query_len = min(self.max_batched_tokens, effective_input_length - consumed)
            seq_len = consumed + query_len
            chunks.append(PrefillChunk(index=index, query_len=query_len, seq_len=seq_len))
            consumed += query_len
            index += 1

        return chunks

    def get_prefill_num_chunks(self) -> int:
        """Return the number of prefill chunks produced by the current token budget."""
        return len(self.get_prefill_chunk_plan())


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


def select_tightest_memory_info(memory_infos: Iterable[MemoryInfo | None]) -> MemoryInfo | None:
    """Select the memory info with the smallest available device memory."""
    candidates = [memory_info for memory_info in memory_infos if memory_info]
    if not candidates:
        return None

    def memory_available(memory_info: MemoryInfo) -> float:
        try:
            return float(memory_info.get("device_memory_available_gb", float("inf")))
        except (TypeError, ValueError):
            return float("inf")

    return min(candidates, key=memory_available)


def build_memory_info(batch_result) -> MemoryInfo:
    """Build memory info dict from ModelRunnerMetrics.

    Only per-row (per-configuration) fields are included in the DataFrame columns
    (see MEMORY_COLUMNS). Constant fields (total_device_memory_gb, reserved_memory_gb)
    are stored only in OptimizerSummary._memory_info for text display.
    """
    return {
        "total_device_memory_gb": getattr(batch_result, "total_device_memory_gb", float("nan")),
        "model_weight_size_gb": getattr(batch_result, "model_weight_size_gb", float("nan")),
        "kv_cache_size_gb": getattr(batch_result, "kv_cache_size_gb", float("nan")),
        "model_activation_size_gb": getattr(batch_result, "model_activation_size_gb", float("nan")),
        "reserved_memory_gb": getattr(batch_result, "reserved_memory_gb", float("nan")),
        "device_memory_available_gb": getattr(batch_result, "device_memory_available_gb", float("nan")),
    }


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


def resolve_parallel_search_candidates(
    tp_sizes: list[int] | None,
    ep_sizes: list[int] | None,
    moe_dp_sizes: list[int] | None,
    num_mtp_token_sizes: list[int] | None,
    num_mtp_tokens: int,
    target_devices: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Resolve throughput optimizer TP/EP/MOE-DP/MTP candidate lists."""
    tp_candidates = resolve_search_sizes(tp_sizes, target_devices, target_devices)
    ep_candidates = resolve_search_sizes(ep_sizes, target_devices, target_devices)
    moe_dp_candidates = resolve_search_sizes(moe_dp_sizes, target_devices, 1)
    mtp_candidates = num_mtp_token_sizes or [num_mtp_tokens]
    return tp_candidates, ep_candidates, moe_dp_candidates, mtp_candidates


def count_search_combinations(
    tp_candidates: list[int],
    ep_candidates: list[int],
    moe_dp_candidates: list[int],
    mtp_candidates: list[int],
) -> int:
    """Return Cartesian product size for parallel and MTP search dimensions."""
    return len(tp_candidates) * len(ep_candidates) * len(moe_dp_candidates) * len(mtp_candidates)


def format_parallel_label(
    parallel_config: ParallelConfig,
    is_moe_model: bool,
    num_mtp_tokens: Optional[int] = None,
) -> str:
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
    if num_mtp_tokens is not None and num_mtp_tokens > 0:
        parts.append(f"MTP={num_mtp_tokens}")
    return " | ".join(parts)

# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

import argparse
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, TypedDict

import yaml

from tensor_cast.model_config import ParallelConfig
from tensor_cast.pipeline_parallel import UnsupportedPPConfigurationError as UnsupportedPPConfigurationError

logger = logging.getLogger(__name__)


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}
LIMIT_COUNT = 1e8
DEFAULT_MAX_SEARCH_COMBINATIONS = 100
BYTES_TO_GB = 1024**3
MAX_ITER_NUMS = 10
HISTORICAL_DEFAULT_MAX_BATCHED_TOKENS = 8192
AUTO_MAX_BATCHED_TOKENS_FACTORS = (4, 2, 1)

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


# UnsupportedPPConfigurationError is imported from tensor_cast.pipeline_parallel
# and re-exported here for backward compatibility with existing callers.


@dataclass
class PrefillChunk:
    index: int
    query_len: int
    seq_len: int
    is_last_chunk: bool = False


@dataclass
class OptimizerData:
    input_length: Optional[int] = None
    length_distribution: Optional["LengthDistribution"] = None
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
    dflash_block_size: Optional[int] = None
    dflash_acceptance_length: Optional[float] = None
    dspark_block_size: Optional[int] = None
    dspark_acceptance_length: Optional[float] = None
    dspark_markov_rank: Optional[int] = None
    speculative_method: Optional[str] = None
    """Set to 'mtp' for the new MTP entry (fold uses accept+1). None for legacy MTP / baseline."""
    acceptance_length: Optional[float] = None
    """Generic acceptance_length for the new MTP entry fold (clamp to n, then accept+1)."""
    prefill_devices_per_instance: Optional[int] = None
    decode_devices_per_instance: Optional[int] = None
    prefix_cache_hit_rate: float = 0.0
    concurrency_search_strategy: str = "exponential"

    def get_representative_rows(self, strategy: str = "mid") -> list[dict]:
        """
        Get representative rows for the length distribution based on the specified strategy.

        Args:
            strategy (str): The strategy to use for determining the representative point.
                Options are "mid", "min", or "max". Default is "mid".

        Returns:
            list[dict]: A list of dictionaries representing the representative rows. Each dictionary contains:
                - "num_input_tokens": The representative input token count for the bin.
                - "query_len": The effective input token count after applying prefix cache hit rate.
                - "request_ratio": The weight of the bin in the total weight of the distribution.
        """
        representative_rows = []
        total_weight = sum(bin_.weight for bin_ in self.length_distribution.bins)
        for bin_ in self.length_distribution.bins:
            if strategy == "mid":
                representative_point = math.floor((bin_.min_tokens + bin_.max_tokens) / 2)
            elif strategy == "min":
                representative_point = bin_.min_tokens
            elif strategy == "max":
                representative_point = bin_.max_tokens
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            # prefix_cache_hit_rate is a fixed float number for now
            cached_prefix_tokens = math.floor(representative_point * self.prefix_cache_hit_rate)
            representative_rows.append(
                {
                    "num_input_tokens": representative_point,
                    "query_len": max(1, representative_point - cached_prefix_tokens),
                    "request_ratio": bin_.weight / total_weight,
                }
            )
        return representative_rows

    def get_effective_input_length(self, is_decode: bool = False):
        if self.length_distribution is not None:
            if is_decode:
                return None
            return self._weighted_representative_length("query_len")

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

    def _weighted_representative_length(self, field: str) -> int:
        """Return a weighted, positive representative length from distribution rows."""
        weighted_average = sum(row[field] * row["request_ratio"] for row in self.get_representative_rows())
        return max(1, math.floor(weighted_average))

    def get_decode_context_length(self) -> Optional[int]:
        """Return the representative full prompt length used by decode.

        Decode attends to the complete KV-cache context, so prefix-cache hits
        reduce prefill work but must not shorten this length.  Distribution mode
        has no single request length; use its weighted representative raw prompt
        length, matching the representative strategy used for prefill.
        """
        if self.length_distribution is not None:
            return self._weighted_representative_length("num_input_tokens")
        return self.input_length

    def get_prefill_cached_prefix_length(self) -> Optional[int]:
        """Return the cached prefix represented as existing Prefill KV context."""
        if self.length_distribution is not None:
            raw_length = self._weighted_representative_length("num_input_tokens")
            effective_length = self._weighted_representative_length("query_len")
            return max(0, raw_length - effective_length)
        if self.input_length is None:
            return None
        effective_length = self.get_effective_input_length()
        return max(0, self.input_length - effective_length)

    def get_prefill_chunk_plan(self, concurrency: Optional[int] = None) -> list[PrefillChunk]:
        """Split the effective prefill prompt into chunks bounded by max_batched_tokens."""
        if self.length_distribution is not None:
            if concurrency is None:
                raise ValueError("concurrency is required for variable-length prefill chunk planning.")
            if self.max_batched_tokens is None or self.max_batched_tokens <= 0:
                raise ValueError(f"max_batched_tokens must be a positive integer, got {self.max_batched_tokens!r}.")

            chunks = []
            chunk_index = 0
            remaining_token_budget = self.max_batched_tokens
            for row in self.build_concurrency_samples(concurrency):
                for _ in range(row["samples"]):
                    request_query_len = row["query_len"]
                    cached_prefix_tokens = row["num_input_tokens"] - request_query_len
                    consumed_query_len = 0
                    while consumed_query_len < request_query_len:
                        if remaining_token_budget == 0:
                            chunk_index += 1
                            remaining_token_budget = self.max_batched_tokens

                        query_len = min(
                            request_query_len - consumed_query_len,
                            remaining_token_budget,
                        )
                        consumed_query_len += query_len
                        remaining_token_budget -= query_len
                        chunks.append(
                            PrefillChunk(
                                index=chunk_index,
                                query_len=query_len,
                                seq_len=cached_prefix_tokens + consumed_query_len,
                                is_last_chunk=consumed_query_len == request_query_len,
                            )
                        )
            return chunks

        effective_input_length = self.get_effective_input_length(is_decode=False)
        if effective_input_length is None:
            return []
        if self.max_batched_tokens is None or self.max_batched_tokens <= 0:
            raise ValueError(f"max_batched_tokens must be a positive integer, got {self.max_batched_tokens!r}.")

        chunks = []
        consumed = 0
        index = 0
        cached_prefix_tokens = self.get_prefill_cached_prefix_length() or 0
        while consumed < effective_input_length:
            query_len = min(self.max_batched_tokens, effective_input_length - consumed)
            seq_len = cached_prefix_tokens + consumed + query_len
            chunks.append(
                PrefillChunk(
                    index=index,
                    query_len=query_len,
                    seq_len=seq_len,
                    is_last_chunk=consumed + query_len == effective_input_length,
                )
            )
            consumed += query_len
            index += 1

        return chunks

    def get_auto_max_batched_tokens_candidates(self) -> list[int]:
        """Return max_batched_tokens candidates for automatic Prefill OOM fallback.

        Prefer raw input_length when available so prefix-cache hit rate does not
        shrink the serving-engine token budget. Distribution mode has no single
        raw input length, so it falls back to the representative effective length.
        """
        base_length = self.input_length
        if base_length is None:
            base_length = self.get_effective_input_length(is_decode=False)
        if base_length is None:
            return []

        candidates = []
        for factor in AUTO_MAX_BATCHED_TOKENS_FACTORS:
            candidate = base_length * factor
            if candidate > 0 and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def get_prefill_num_chunks(self, chunk_plan: list[PrefillChunk]) -> int:
        """Return the number of chunks represented by a prefill chunk plan."""
        return max(chunk.index for chunk in chunk_plan) + 1 if chunk_plan else 0

    def build_concurrency_samples(self, concurrency: int):
        rows = self.get_representative_rows()
        ideal_samples = [concurrency * row["request_ratio"] for row in rows]
        base_samples = [math.floor(sample) for sample in ideal_samples]
        remaining = concurrency - sum(base_samples)

        ranked_indices = sorted(
            range(len(rows)),
            key=lambda idx: (
                ideal_samples[idx] - base_samples[idx],
                rows[idx]["query_len"],
            ),
            reverse=True,
        )

        for idx in ranked_indices[:remaining]:
            base_samples[idx] += 1

        composition_rows = []
        for row, sample in zip(rows, base_samples):
            if sample <= 0:
                continue
            composition_rows.append({**row, "samples": sample})

        return composition_rows


@dataclass(frozen=True)
class LengthBin:
    min_tokens: int
    max_tokens: int
    weight: float


@dataclass(frozen=True)
class LengthDistribution:
    bins: list[LengthBin]


def validate_length_distribution(bins: list[LengthBin]) -> None:
    if not bins:
        raise ValueError("length distribution must contain at least one bin")

    previous_max_tokens = None
    for index, bin_ in enumerate(sorted(bins, key=lambda item: item.min_tokens)):
        if bin_.min_tokens < 0:
            raise ValueError(f"bin {index} min_tokens must be >= 0")
        if bin_.max_tokens <= bin_.min_tokens:
            raise ValueError(f"bin {index} max_tokens must be > min_tokens")
        if bin_.weight <= 0:
            raise ValueError(f"bin {index} weight must be > 0")
        if previous_max_tokens is not None and bin_.min_tokens < previous_max_tokens:
            raise ValueError(f"bin {index} ranges overlap")
        previous_max_tokens = bin_.max_tokens


def load_length_distribution(
    path: str | Path = None,
) -> LengthDistribution:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "example" / "length_distribution.yaml"
    distribution_path = Path(path)
    try:
        with distribution_path.open(encoding="utf-8") as file:
            raw_distribution = yaml.safe_load(file) or {}
    except Exception as e:
        raise ValueError(f"failed to load length distribution from {path}") from e

    if not isinstance(raw_distribution, dict):
        raise ValueError("length distribution must be a mapping with a 'bins' list")
    raw_bins = raw_distribution.get("bins", [])
    if not isinstance(raw_bins, list):
        raise ValueError("length distribution 'bins' must be a list")

    bins = []
    for index, item in enumerate(raw_bins):
        if not isinstance(item, dict):
            raise ValueError(f"bin {index} must be a mapping")
        missing_keys = [key for key in ("min_tokens", "max_tokens", "weight") if key not in item]
        if missing_keys:
            raise ValueError(f"bin {index} missing required key(s): {', '.join(missing_keys)}")
        bins.append(
            LengthBin(
                min_tokens=int(item["min_tokens"]),
                max_tokens=int(item["max_tokens"]),
                weight=float(item["weight"]),
            )
        )
    validate_length_distribution(bins)
    return LengthDistribution(bins=bins)


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
    if value > LIMIT_COUNT:
        raise argparse.ArgumentTypeError(f"{value!r} is too large")
    return value


def check_positive_integer_and_string(value):
    try:
        return check_positive_integer(value)
    except argparse.ArgumentTypeError:
        input_length_path = Path(value)
        if input_length_path.is_file():
            return str(input_length_path)
        raise argparse.ArgumentTypeError(
            f"{value!r} must be a positive integer or an existing length distribution YAML file"
        ) from None


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
    pp_values = []
    for sub_dict_name, sub_dict in breakdowns.items():
        total = sum(sub_dict.values())
        if total == 0:
            continue
        if sub_dict_name.endswith("_pipeline_parallel"):
            for value in sub_dict.values():
                if isinstance(value, float):
                    pp_values.append(value / total * 100)
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

    if pp_values:
        pp_keys = ["PP Compute", "PP Comm", "PP Bubble"]
        for i, key in enumerate(pp_keys):
            if i < len(pp_values):
                formatted_parts.append(f"{key} {pp_values[i]:.2f}")

    return " | ".join(formatted_parts)


def select_tightest_memory_info(
    memory_infos: Iterable[MemoryInfo | None],
) -> MemoryInfo | None:
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


def _clamp_accept_label(accept: Optional[float], block: int) -> float:
    """Clamp acceptance to ``n = block - 1`` for display (matches fold / CLI clamp)."""
    accept = 5.0 if accept is None else float(accept)
    if accept < 0:
        accept = 0.0
    if block >= 2:
        max_accept = float(block - 1)
        if accept > max_accept:
            accept = max_accept
    return accept


def format_parallel_label(
    parallel_config: ParallelConfig,
    is_moe_model: bool,
    num_mtp_tokens: Optional[int] = None,
    dflash_block_size: Optional[int] = None,
    dflash_acceptance_length: Optional[float] = None,
    dspark_block_size: Optional[int] = None,
    dspark_acceptance_length: Optional[float] = None,
    dspark_markov_rank: Optional[int] = None,
    mtp_acceptance_length: Optional[float] = None,
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
        # Unified MTP entry may carry a scalar acceptance; legacy keeps bare MTP=N.
        if mtp_acceptance_length is not None:
            accept = _clamp_accept_label(mtp_acceptance_length, int(num_mtp_tokens) + 1)
            accept_label = int(accept) if accept == int(accept) else accept
            parts.append(f"MTP={num_mtp_tokens}/acc={accept_label}")
        else:
            parts.append(f"MTP={num_mtp_tokens}")
    # Only surface DCP when it is actually enabled, so non-DCP runs keep their label.
    if getattr(parallel_config, "decode_context_parallel_size", 1) > 1:
        parts.append(f"DCP={parallel_config.decode_context_parallel_size}")
    dspark_block = dspark_block_size or 0
    if dspark_block >= 2:
        accept = _clamp_accept_label(dspark_acceptance_length, dspark_block)
        accept_label = int(accept) if accept == int(accept) else accept
        markov = 256 if dspark_markov_rank is None else int(dspark_markov_rank)
        parts.append(f"DSpark={dspark_block}/acc={accept_label}/markov={markov}")
    else:
        dflash_block = dflash_block_size or 0
        if dflash_block >= 2:
            accept = _clamp_accept_label(dflash_acceptance_length, dflash_block)
            accept_label = int(accept) if accept == int(accept) else accept
            parts.append(f"DFlash={dflash_block}/acc={accept_label}")
    return " | ".join(parts)


@dataclass(frozen=True)
class ParallelSearchCandidate:
    """A fixed parallel-structure candidate for throughput optimization.

    Carries all resolved parallel dimensions plus the PP partition.
    ``dp_size`` and ``moe_tp_size`` are derived from the
    stage-local arithmetic (``dp = num_devices/(tp*pp)``,
    ``moe_tp = (num_devices/pp)/(ep*moe_dp)``).
    """

    tp_size: int
    pp_size: int
    ep_size: int
    moe_dp_size: int
    moe_tp_size: int
    dp_size: int
    num_mtp_tokens: int
    layer_partition: Optional[tuple[int, ...]]
    dcp_size: int = 1
    schedule: str = "forward"


def _parse_partition_list(partition_values: list[int] | tuple[int, ...] | str) -> tuple[int, ...]:
    """Validate a list of positive ints and return as tuple.

    Accepts either a list of ints (e.g. ``[40, 40]``) or a comma-separated
    string (e.g. ``"40,40"``) for flexibility across CLI and direct call sites.
    """
    if isinstance(partition_values, str):
        try:
            partition_values = [int(value.strip()) for value in partition_values.split(",")]
        except ValueError as error:
            raise ValueError(f"partition {partition_values!r} contains a non-integer entry") from error
    if not isinstance(partition_values, (list, tuple)) or not partition_values:
        raise ValueError("partition has no entries")
    if any(isinstance(count, bool) or not isinstance(count, int) for count in partition_values):
        raise ValueError(f"partition {partition_values!r} contains a non-integer entry")
    counts = tuple(partition_values)
    if any(c <= 0 for c in counts):
        raise ValueError(f"partition {partition_values!r} has non-positive entries")
    return counts


def resolve_pp_sizes(pp_sizes: list[int] | None, num_devices: int) -> list[int]:
    """Resolve ``--pp-sizes`` once for CLI and workload-spec callers."""
    if pp_sizes is None:
        return [1]
    if not isinstance(pp_sizes, list):
        raise ValueError("pp_sizes must be a list of positive integers")
    if not pp_sizes:
        return [1 << index for index in range(num_devices.bit_length())]

    resolved: list[int] = []
    for size in pp_sizes:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"pp_sizes contains non-positive integer value {size!r}")
        if size > num_devices:
            raise ValueError(f"pp_sizes contains value {size}, which is larger than 'num_devices' ({num_devices})")
        if size not in resolved:
            resolved.append(size)
    return resolved


def resolve_pp_layer_partitions(
    pp_layer_partitions: list[list[int] | str] | None,
    pp_sizes: list[int],
    num_hidden_layers: int,
) -> dict[int, list[tuple[int, ...]]]:
    """Validate and group PP layer partitions for all PP entry points."""
    if pp_layer_partitions is None:
        return {}
    if not isinstance(pp_layer_partitions, list) or not pp_layer_partitions:
        raise ValueError("pp_layer_partitions must be a non-empty list of layer-count lists")

    requested_pp = {pp for pp in pp_sizes if pp > 1}
    partitions_by_length: dict[int, list[tuple[int, ...]]] = {}
    for partition_values in pp_layer_partitions:
        partition = _parse_partition_list(partition_values)
        pp_size = len(partition)
        if pp_size not in requested_pp:
            raise ValueError(
                f"partition length {pp_size} does not match any requested pp_sizes > 1 ({sorted(requested_pp)})"
            )
        if sum(partition) != num_hidden_layers:
            raise ValueError(
                f"partition {list(partition)} sums to {sum(partition)}, expected num_hidden_layers {num_hidden_layers}"
            )
        partitions_by_length.setdefault(pp_size, []).append(partition)

    missing = sorted(requested_pp - set(partitions_by_length))
    if missing:
        raise ValueError(
            f"no partition of length {missing} for requested pp_sizes {sorted(requested_pp)}; "
            "PP>1 requires an explicit matching-length partition when pp_layer_partitions is provided"
        )
    return partitions_by_length


def build_pp_search_candidates(
    num_devices: int,
    tp_sizes: list[int] | None,
    pp_sizes: list[int] | None,
    num_hidden_layers: int,
    ep_sizes: list[int] | None = None,
    moe_dp_sizes: list[int] | None = None,
    num_mtp_token_sizes: list[int] | None = None,
    num_mtp_tokens: int = 0,
    pp_layer_partitions: list[list[int] | str] | None = None,
    dcp_sizes: list[int] | None = None,
) -> list[ParallelSearchCandidate]:
    """Enumerate PP-aware parallel search candidates with stage-local arithmetic.

    Uses ``dp = num_devices // (tp * pp)`` and
    ``moe_tp = (num_devices // pp) // (ep * moe_dp)`` so EP stays in the
    stage-local MoE sub-world. Returns candidates that satisfy all divisibility
    and partition constraints. Combinations that violate stage-local
    divisibility or exceed ``num_hidden_layers`` are filtered; malformed PP
    sizes and partitions raise ``ValueError``.

    PP=1 compatibility: when ``pp_sizes`` is None, only PP=1 is searched and
    the TP/EP/MOE-DP defaults match the legacy ``resolve_parallel_search_candidates``
    (TP/EP default to ``num_devices``, MOE-DP to 1). PP=1 forces
    ``layer_partition=None``. For PP>1, TP/EP defaults
    resolve to ``stage_devices = num_devices // pp`` so that adding
    ``--pp-sizes`` alone produces valid candidates.
    """
    pp_list = resolve_pp_sizes(pp_sizes, num_devices)
    moe_dp_list = resolve_search_sizes(moe_dp_sizes, num_devices, 1)
    mtp_list = num_mtp_token_sizes or [num_mtp_tokens]

    # Parse partition strings eagerly so format errors raise immediately. Group
    # partitions by length so each partition is paired only with matching pp_size,
    # not cartesian-producted across all PP sizes.
    if pp_layer_partitions is not None and pp_sizes is None:
        raise ValueError("pp_layer_partitions requires pp_sizes to be provided")
    partitions_by_length = resolve_pp_layer_partitions(
        pp_layer_partitions,
        pp_list,
        num_hidden_layers,
    )

    candidates: list[ParallelSearchCandidate] = []
    mtp_blocked_pp_sizes: list[int] = []
    base_failed_pp_sizes: list[int] = []
    for pp in pp_list:
        # Filter pp_size that can't divide num_devices before computing
        # stage_devices, so stage_devices is always > 0 and resolve_search_sizes
        # never gets a zero default. ``resolve_pp_sizes`` already rejects
        # pp > num_devices.
        # Record such pp as base-failed so the final MTP error attribution
        # knows not every empty pp was blocked by MTP.
        if pp > num_hidden_layers:
            base_failed_pp_sizes.append(pp)
            continue
        if num_devices % pp != 0:
            base_failed_pp_sizes.append(pp)
            continue
        stage_devices = num_devices // pp

        # Resolve TP/EP defaults per-pp using stage_devices, so PP>1 with no
        # explicit --tp-sizes/--ep-sizes still produces valid candidates.
        # PP=1: stage_devices == num_devices, matching legacy defaults.
        tp_list_pp = resolve_search_sizes(tp_sizes, stage_devices, stage_devices)
        ep_list_pp = resolve_search_sizes(ep_sizes, stage_devices, stage_devices)
        # DCP reuses TP devices (constraint: tp % dcp == 0), decoded later in
        # the inner loop.  Prefill callers pass dcp_sizes=None so this resolves
        # to [1] (DCP is decode-only).
        dcp_list_pp = resolve_search_sizes(dcp_sizes, stage_devices, 1)

        # PP>1 does not support MTP yet (model_builder rejects it). We do NOT
        # skip the pp_size here — instead we defer the MTP decision until after
        # the base (TP/EP/MoE-DP) divisibility check, so that the MTP error is
        # only attributed when the pp_size would otherwise have produced a
        # valid base candidate. This avoids misattributing an empty result to
        # MTP when the real cause was e.g. TP not dividing num_devices.
        if pp > 1:
            effective_mtp_list = [m for m in mtp_list if m == 0]
        else:
            effective_mtp_list = mtp_list

        # Partitions for this pp_size. PP=1 always uses default (None), even
        # when explicit partitions are provided for other PP sizes. PP>1
        # requires a matching-length partition when explicit partitions were
        # requested (no silent fallback to default balanced).
        if pp == 1:
            partition_options: list[tuple[int, ...] | None] = [None]
        elif pp_layer_partitions is not None:
            partition_options = list(partitions_by_length.get(pp, []))
            if not partition_options:
                raise ValueError(
                    f"no partition of length {pp} found for pp_size={pp}; "
                    f"provided partitions have lengths {sorted(partitions_by_length)}"
                )
        else:
            partition_options = [None]

        pp_had_valid_base = False
        pp_blocked_by_mtp = False
        candidates_added_this_pp = False
        for tp in tp_list_pp:
            if num_devices % (tp * pp) != 0:
                continue
            dp = num_devices // (tp * pp)
            for dcp in dcp_list_pp:
                if tp % dcp != 0:
                    continue
                for ep in ep_list_pp:
                    for moe_dp in moe_dp_list:
                        if stage_devices % (ep * moe_dp) != 0:
                            continue
                        moe_tp = stage_devices // (ep * moe_dp)
                        # Base combination (TP/EP/MoE-DP) is valid for this pp_size.
                        pp_had_valid_base = True
                        if pp > 1 and not effective_mtp_list:
                            # PP>1 requires MTP=0 but none was requested; this
                            # pp_size is blocked by MTP, not by base divisibility.
                            pp_blocked_by_mtp = True
                            continue
                        for num_mtp in effective_mtp_list:
                            for partition in partition_options:
                                candidates.append(
                                    ParallelSearchCandidate(
                                        tp_size=tp,
                                        pp_size=pp,
                                        ep_size=ep,
                                        moe_dp_size=moe_dp,
                                        moe_tp_size=moe_tp,
                                        dp_size=dp,
                                        num_mtp_tokens=num_mtp,
                                        layer_partition=partition,
                                        dcp_size=dcp,
                                    )
                                )
                                candidates_added_this_pp = True
        # Classify this pp_size's emptiness cause for final error attribution.
        if not candidates_added_this_pp:
            if pp_blocked_by_mtp and pp_had_valid_base:
                mtp_blocked_pp_sizes.append(pp)
            elif not pp_had_valid_base:
                base_failed_pp_sizes.append(pp)

    # Raise an MTP-specific error only when EVERY empty pp_size was blocked
    # solely by MTP (had a valid base but no MTP=0 fallback). If any pp_size
    # failed base divisibility, the emptiness is not purely MTP's fault, so do
    # not misattribute.
    if not candidates and mtp_blocked_pp_sizes and not base_failed_pp_sizes:
        raise ValueError(
            "PP>1 currently requires num_mtp_tokens=0; "
            f"pp sizes {mtp_blocked_pp_sizes} with mtp={mtp_list} are incompatible"
        )
    if not candidates and pp_sizes is not None and len(pp_sizes) > 0:
        logger.warning(
            "No valid PP parallel combination found for the given "
            "--tp-sizes/--pp-sizes/--ep-sizes/--moe-dp-sizes under "
            "--num-devices=%d. Check stage-local divisibility "
            "(dp = num_devices / (tp * pp), "
            "moe_tp = (num_devices / pp) / (ep * moe_dp)).",
            num_devices,
        )
    return candidates

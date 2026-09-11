"""Internal workload policy for query-driven shape collection."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Callable, Iterable

import psutil
import yaml

from tensor_cast import device_profiles  # noqa: F401
from tensor_cast.device import DeviceProfile
from tensor_cast.performance_model.profiling_database.query_demand import (
    QUERY_TRACE_DIR_ENV,
    QUERY_TRACE_MODEL_ENV,
    QUERY_TRACE_WORKLOAD_ENV,
    load_query_demand_traces,
)

from .query_model import QueryModelArchitecture


QUERY_WORKLOAD_POLICY_VERSION = "query-workloads/v4"
QUERY_WORKLOAD_CHECKPOINT_VERSION = 2
MAX_PARALLEL_QUERY_WORKLOADS = 8
QUERY_TRACE_POLL_SECONDS = 1.0
QUERY_TRACE_QUIET_SECONDS = 30.0
MAX_CONVERGENCE_WAIT_SECONDS = 300.0
QUERY_PROCESS_TERMINATE_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class WorkloadScenario:
    model_id: str
    device: str
    num_devices: int
    input_length: int
    output_length: int
    max_batched_tokens: int | None
    tp_sizes: tuple[int, ...]
    ep_sizes: tuple[int, ...]
    moe_dp_sizes: tuple[int, ...]
    dcp_sizes: tuple[int, ...]
    mtp_tokens: tuple[int, ...]
    sweep_name: str = "baseline"
    compilation_config: tuple[str, ...] = ()
    quantize_linear_action: str = "W8A8_DYNAMIC"
    quantize_non_expert_linear_action: str = "DISABLED"
    quantize_attention_action: str = "DISABLED"
    batch_range: tuple[int, int] = (1, 512)
    disagg: bool = False
    ttft_limit: float | None = None
    tpot_limit: float | None = None
    reserved_memory_gb: float | None = None
    prefix_cache_hit_rate: float = 0.0
    enable_shared_expert_tp: bool = False
    word_embedding_tp: str = ""
    mtp_acceptance_rates: tuple[float, ...] = ()
    compile: bool = False
    mxfp4_group_size: int | None = None
    pp_sizes: tuple[int, ...] = ()
    pp_layer_partitions: tuple[tuple[int, ...], ...] = ()

    @property
    def workload_id(self) -> str:
        compilation = "+".join(self.compilation_config) or "eager"
        chunk = self.max_batched_tokens or 0
        parallel = (
            f"tp={','.join(map(str, self.tp_sizes))};ep={','.join(map(str, self.ep_sizes))};"
            f"moedp={','.join(map(str, self.moe_dp_sizes))};dcp={','.join(map(str, self.dcp_sizes))};"
            f"mtp={','.join(map(str, self.mtp_tokens)) or 'default'};"
            f"pp={','.join(map(str, self.pp_sizes)) or 'default'}"
        )
        partitions = self.pp_layer_partitions
        partition_tag = (
            hashlib.sha256(
                json.dumps([list(part) for part in partitions], separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:12]
            if partitions
            else "default"
        )
        quantization = (
            f"linear={self.quantize_linear_action};nonexpert={self.quantize_non_expert_linear_action};"
            f"attention={self.quantize_attention_action}"
        )
        serving = (
            f"disagg={int(self.disagg)};ttft={self.ttft_limit or 0};tpot={self.tpot_limit or 0};"
            f"mem={self.reserved_memory_gb if self.reserved_memory_gb is not None else 'default'};"
            f"prefix={self.prefix_cache_hit_rate};sharedexperttp={int(self.enable_shared_expert_tp)};"
            f"wordembeddingtp={self.word_embedding_tp or 'off'};"
            f"mtpacc={','.join(f'{rate:g}' for rate in self.mtp_acceptance_rates) or 'default'};"
            f"torchcompile={int(self.compile)};"
            f"mxfp4group={self.mxfp4_group_size if self.mxfp4_group_size is not None else 'default'};"
            f"pppartitions={partition_tag}"
        )
        return (
            f"{QUERY_WORKLOAD_POLICY_VERSION};model={self.model_id};sweep={self.sweep_name};"
            f"devices={self.num_devices};input={self.input_length};output={self.output_length};"
            f"chunk={chunk};batch={self.batch_range[0]},{self.batch_range[1]};"
            f"mode={compilation};{parallel};{quantization};{serving}"
        )

    def command(self, database_path: Path) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "cli.inference.throughput_optimizer",
            self.model_id,
            "--device",
            self.device,
            "--num-devices",
            str(self.num_devices),
            "--input-length",
            str(self.input_length),
            "--output-length",
            str(self.output_length),
            "--performance-model",
            "profiling",
            "--profiling-database-path",
            str(database_path),
            "--batch-range",
            str(self.batch_range[0]),
            str(self.batch_range[1]),
            "--jobs",
            "1",
            "--log-level",
            "error",
            "--tp-sizes",
            *(str(value) for value in self.tp_sizes),
            "--ep-sizes",
            *(str(value) for value in self.ep_sizes),
            "--moe-dp-sizes",
            *(str(value) for value in self.moe_dp_sizes),
            "--dcp-sizes",
            *(str(value) for value in self.dcp_sizes),
            "--quantize-linear-action",
            self.quantize_linear_action,
            "--quantize-non-expert-linear-action",
            self.quantize_non_expert_linear_action,
            "--quantize-attention-action",
            self.quantize_attention_action,
        ]
        if self.mtp_tokens:
            command.extend(["--num-mtp-tokens", *(str(value) for value in self.mtp_tokens)])
        if self.mtp_acceptance_rates:
            command.extend(["--mtp-acceptance-rates", *(f"{rate:g}" for rate in self.mtp_acceptance_rates)])
        if self.max_batched_tokens is not None:
            command.extend(["--max-batched-tokens", str(self.max_batched_tokens)])
        if self.mxfp4_group_size is not None:
            command.extend(["--mxfp4-group-size", str(self.mxfp4_group_size)])
        if self.compilation_config or self.compile:
            command.append("--compile")
        if self.compilation_config:
            command.extend(["--compilation-config", *self.compilation_config])
        if self.pp_sizes:
            command.extend(["--pp-sizes", *(str(value) for value in self.pp_sizes)])
        if self.pp_layer_partitions:
            command.extend(
                [
                    "--pp-layer-partitions",
                    json.dumps([list(partition) for partition in self.pp_layer_partitions], separators=(",", ":")),
                ]
            )
        if self.disagg:
            command.append("--disagg")
        if self.ttft_limit is not None:
            command.extend(["--ttft-limit", f"{self.ttft_limit:g}"])
        if self.tpot_limit is not None:
            command.extend(["--tpot-limit", f"{self.tpot_limit:g}"])
        if self.reserved_memory_gb is not None:
            command.extend(["--reserved-memory-gb", f"{self.reserved_memory_gb:g}"])
        if self.prefix_cache_hit_rate:
            command.extend(["--prefix-cache-hit-rate", f"{self.prefix_cache_hit_rate:g}"])
        if self.enable_shared_expert_tp:
            command.append("--enable-shared-expert-tp")
        if self.word_embedding_tp:
            command.extend(["--word-embedding-tp", self.word_embedding_tp])
        return command

    @property
    def parallel_combinations(self) -> int:
        """Return the configured parallel-axis Cartesian size before validation."""
        return (
            len(self.tp_sizes)
            * max(1, len(self.pp_sizes))
            * len(self.ep_sizes)
            * len(self.moe_dp_sizes)
            * len(self.dcp_sizes)
            * max(1, len(self.mtp_tokens))
        )


@dataclass(frozen=True)
class QueryWorkloadRunResult:
    attempted: int
    succeeded: int
    failed_workloads: tuple[str, ...]
    cached: int = 0
    elapsed_seconds: float = 0.0
    trace_directories: tuple[Path, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class _WorkloadExecution:
    index: int
    scenario: WorkloadScenario
    returncode: int
    elapsed_seconds: float
    trace_directory: Path
    demand_count: int
    stderr: str = ""
    cached: bool = False
    completion_reason: str = "process_exit"


@dataclass(frozen=True)
class _CommandOutcome:
    returncode: int
    stderr: str = ""
    completion_reason: str = "process_exit"


def _powers_of_two(maximum: int) -> tuple[int, ...]:
    result = []
    value = 1
    while value <= maximum:
        result.append(value)
        value *= 2
    return tuple(result or [1])


def _device_counts(maximum: int) -> tuple[int, ...]:
    """Cover ordinary power-of-two deployments plus the real topology edge."""
    return tuple(sorted({*_powers_of_two(maximum), max(1, int(maximum))}))


def _bounded_lengths(max_context_length: int) -> tuple[int, ...]:
    # Keep one token for decode so every generated optimizer command remains
    # within the model context window.
    maximum = max(1, int(max_context_length) - 1)
    anchors = {
        1,
        2,
        8,
        32,
        128,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        maximum,
        max(1, maximum - 1),
    }
    for numerator in (1, 2, 4, 8, 12, 15):
        anchors.add(max(1, maximum * numerator // 16))
    return tuple(sorted(value for value in anchors if value <= maximum))


def _core_lengths(lengths: tuple[int, ...]) -> tuple[int, ...]:
    if len(lengths) <= 7:
        return lengths
    indices = {0, 1, len(lengths) // 4, len(lengths) // 2, 3 * len(lengths) // 4, len(lengths) - 2, len(lengths) - 1}
    return tuple(lengths[index] for index in sorted(indices))


def _coverage_lengths(max_context_length: int) -> tuple[int, ...]:
    """Keep continuous-axis representatives plus known discontinuity boundaries."""
    maximum = max(1, int(max_context_length) - 1)
    selected = set(_core_lengths(_bounded_lengths(max_context_length)))
    for boundary in (4096,):
        for value in (boundary - 1, boundary, boundary + 1):
            if 0 < value <= maximum:
                selected.add(value)
    selected.update({1, maximum, max(1, maximum - 1)})
    return tuple(sorted(selected))


def _parallel_sizes(candidates: Iterable[int], num_devices: int) -> tuple[int, ...]:
    values = sorted({value for value in candidates if 0 < value <= num_devices and num_devices % value == 0})
    return tuple(values or [1])


def _largest_candidate(candidates: Iterable[int], maximum: int) -> int:
    return max((value for value in candidates if 0 < value <= maximum), default=1)


def _representative_lengths(lengths: tuple[int, ...]) -> tuple[int, ...]:
    """Keep short/mid/long interaction points without repeating every anchor."""
    if len(lengths) <= 3:
        return lengths
    return tuple(dict.fromkeys((lengths[0], lengths[len(lengths) // 2], lengths[-1])))


def _boundary_values(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return (1,)
    return tuple(dict.fromkeys((values[0], values[-1])))


def _interaction_pairs(tp_sizes: tuple[int, ...], ep_sizes: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Cover independent-axis corners without expanding the full Cartesian product."""
    tp_mid = tp_sizes[len(tp_sizes) // 2]
    ep_mid = ep_sizes[len(ep_sizes) // 2]
    pairs = {
        (tp_sizes[0], ep_sizes[-1]),
        (tp_sizes[-1], ep_sizes[0]),
        (tp_sizes[-1], ep_sizes[-1]),
        (tp_mid, ep_mid),
    }
    return tuple(sorted(pairs))


def _split_non_baseline_batch_boundaries(
    scenarios: Iterable[WorkloadScenario],
    *,
    batch_token_budget: int,
) -> list[WorkloadScenario]:
    """Keep batch search in the baseline and query only boundaries elsewhere."""
    expanded: list[WorkloadScenario] = []
    for scenario in scenarios:
        if scenario.sweep_name == "baseline":
            expanded.append(scenario)
            continue
        maximum_batch = min(512, max(1, batch_token_budget // scenario.input_length))
        maximum_batch = 1 << (maximum_batch.bit_length() - 1)
        batch_boundaries = tuple(dict.fromkeys((1, maximum_batch)))
        expanded.extend(replace(scenario, batch_range=(batch_size, batch_size)) for batch_size in batch_boundaries)
    return expanded


def parallel_combination_is_legal(
    *,
    num_devices: int,
    tp: int,
    pp: int = 1,
    ep: int,
    moe_dp: int,
    dcp: int,
) -> bool:
    """Mirror the optimizer's PP-aware parallel divisibility constraints.

    ``dp = num_devices / (tp * pp)`` and ``moe_tp = (num_devices / pp) /
    (ep * moe_dp)`` must both be integral, ``pp`` must divide ``num_devices``,
    and DCP reuses TP devices (``tp % dcp == 0``). ``pp == 1`` reduces exactly
    to the legacy TP/EP/MOE-DP/DCP rules.
    """
    if pp > num_devices or num_devices % pp:
        return False
    stage_devices = num_devices // pp
    if num_devices % (tp * pp):
        return False
    if tp % dcp:
        return False
    return stage_devices % (ep * moe_dp) == 0


def _expand_parallel_combinations(scenarios: Iterable[WorkloadScenario]) -> list[WorkloadScenario]:
    """Make every optimizer subprocess own exactly one legal parallel configuration."""
    expanded: list[WorkloadScenario] = []
    for scenario in scenarios:
        pp_values: tuple[int | None, ...] = scenario.pp_sizes or (None,)
        mtp_values: tuple[int | None, ...] = scenario.mtp_tokens or (None,)
        for pp_size, tp_size, ep_size, moe_dp_size, dcp_size, mtp_token in product(
            pp_values,
            scenario.tp_sizes,
            scenario.ep_sizes,
            scenario.moe_dp_sizes,
            scenario.dcp_sizes,
            mtp_values,
        ):
            if pp_size is not None and pp_size > 1 and (mtp_token or 0) != 0:
                # PP>1 does not support MTP yet (model_builder rejects it).
                continue
            if not parallel_combination_is_legal(
                num_devices=scenario.num_devices,
                tp=tp_size,
                pp=1 if pp_size is None else pp_size,
                ep=ep_size,
                moe_dp=moe_dp_size,
                dcp=dcp_size,
            ):
                continue
            expanded.append(
                replace(
                    scenario,
                    pp_sizes=() if pp_size is None else (pp_size,),
                    tp_sizes=(tp_size,),
                    ep_sizes=(ep_size,),
                    moe_dp_sizes=(moe_dp_size,),
                    dcp_sizes=(dcp_size,),
                    mtp_tokens=() if mtp_token is None else (mtp_token,),
                )
            )
    return expanded


def _load_database_identity(database_path: Path) -> tuple[str, dict]:
    mapping_path = database_path / "op_mapping.yaml"
    if not mapping_path.is_file():
        raise ValueError(f"op_mapping.yaml does not exist under database path: {database_path}")
    with mapping_path.open("r", encoding="utf-8") as mapping_file:
        mapping = yaml.safe_load(mapping_file) or {}
    device = mapping.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError(f"op_mapping.yaml is missing a valid top-level device: {mapping_path}")
    if device not in DeviceProfile.all_device_profiles:
        raise ValueError(
            f"Database device {device!r} is not a registered DeviceProfile; "
            f"available profiles: {sorted(DeviceProfile.all_device_profiles)}"
        )
    return device, mapping


def build_workload_scenarios(
    model_id: str,
    model_config: QueryModelArchitecture,
    database_path: Path,
) -> list[WorkloadScenario]:
    """Build broad but bounded optimizer sweeps from model and device structure."""
    device, mapping = _load_database_identity(database_path)
    maximum_devices = int(DeviceProfile.all_device_profiles[device].comm_grid.grid.nelement())
    coverage_lengths = _coverage_lengths(model_config.max_context_length)
    representative_lengths = _representative_lengths(coverage_lengths)
    scenarios: list[WorkloadScenario] = []

    has_moe_fusion = any(
        isinstance(entry, dict) and entry.get("query_mode") == "moe_fused"
        for entry in mapping.get("operator_mappings", {}).values()
    )
    mtp_tokens = tuple(range(0, min(model_config.num_mtp_layers, 4) + 1))
    tp_max = _largest_candidate(model_config.tp_sizes, maximum_devices)
    tp_sizes = _parallel_sizes(model_config.tp_sizes, tp_max)
    ep_max = _largest_candidate(model_config.ep_sizes, maximum_devices)
    ep_sizes = _parallel_sizes(model_config.ep_sizes, ep_max)
    moe_dp_max = min(maximum_devices, max(_powers_of_two(maximum_devices)))
    moe_dp_sizes = _powers_of_two(moe_dp_max)

    def add_scenario(
        *,
        sweep_name: str,
        num_devices: int,
        input_length: int,
        tp: tuple[int, ...] = (1,),
        ep: tuple[int, ...] = (1,),
        moe_dp: tuple[int, ...] = (1,),
        dcp: tuple[int, ...] = (1,),
        mtp: tuple[int, ...] = (0,),
        output_length: int = 1,
        max_batched_tokens: int | None = None,
        compilation_config: tuple[str, ...] = (),
        quantize_linear_action: str = "W8A8_DYNAMIC",
        quantize_attention_action: str = "DISABLED",
    ) -> None:
        scenarios.append(
            WorkloadScenario(
                model_id=model_id,
                device=device,
                num_devices=num_devices,
                input_length=input_length,
                output_length=output_length,
                max_batched_tokens=max_batched_tokens,
                tp_sizes=tp,
                ep_sizes=ep,
                moe_dp_sizes=moe_dp,
                dcp_sizes=dcp,
                mtp_tokens=mtp,
                sweep_name=sweep_name,
                compilation_config=compilation_config,
                quantize_linear_action=quantize_linear_action,
                quantize_attention_action=quantize_attention_action,
            )
        )

    # Length and local-batch coverage uses a single-device baseline. Parallel
    # axes are then varied independently so the optimizer never expands their
    # full Cartesian product for every sequence length.
    for input_length in coverage_lengths:
        add_scenario(sweep_name="baseline", num_devices=1, input_length=input_length)

    for input_length in representative_lengths:
        add_scenario(
            sweep_name="tp_axis",
            num_devices=tp_max,
            input_length=input_length,
            tp=tp_sizes,
        )
        add_scenario(
            sweep_name="dcp_axis",
            num_devices=tp_max,
            input_length=input_length,
            tp=(tp_max,),
            dcp=_powers_of_two(tp_max),
        )
        # BF16 changes physical kernels/dtypes but reuses the same public API.
        add_scenario(
            sweep_name="bf16_baseline",
            num_devices=1,
            input_length=input_length,
            quantize_linear_action="DISABLED",
        )
        if model_config.num_mtp_layers:
            add_scenario(
                sweep_name="mtp_axis",
                num_devices=1,
                input_length=input_length,
                mtp=mtp_tokens,
            )
        if model_config.num_experts:
            add_scenario(
                sweep_name="ep_axis",
                num_devices=ep_max,
                input_length=input_length,
                ep=ep_sizes,
            )
            add_scenario(
                sweep_name="moe_dp_axis",
                num_devices=moe_dp_max,
                input_length=input_length,
                moe_dp=moe_dp_sizes,
            )

    # Only a few boundary workloads combine TP and EP. This captures genuine
    # cross-axis lowering without multiplying every length by the full search.
    interaction_devices = max(tp_max, ep_max)
    interaction_tp = _parallel_sizes(tp_sizes, interaction_devices)
    interaction_ep = _parallel_sizes(ep_sizes, interaction_devices)
    if model_config.num_experts:
        for input_length in representative_lengths:
            for tp_size, ep_size in _interaction_pairs(interaction_tp, interaction_ep):
                add_scenario(
                    sweep_name="tp_ep_interaction",
                    num_devices=interaction_devices,
                    input_length=input_length,
                    tp=(tp_size,),
                    ep=(ep_size,),
                )

    if maximum_devices not in _powers_of_two(maximum_devices):
        for input_length in representative_lengths:
            add_scenario(
                sweep_name="topology_edge",
                num_devices=maximum_devices,
                input_length=input_length,
                tp=(_largest_candidate(model_config.tp_sizes, maximum_devices),),
                ep=(_largest_candidate(model_config.ep_sizes, maximum_devices),),
            )

    long_candidates = tuple(length for length in coverage_lengths if length > 4096)
    long_lengths = _representative_lengths(long_candidates) if long_candidates else ()
    for input_length in long_lengths:
        add_scenario(
            sweep_name="chunked_prefill_decode",
            num_devices=1,
            input_length=input_length,
            output_length=min(512, max(1, model_config.max_context_length - input_length)),
            max_batched_tokens=min(4096, input_length),
        )

    if model_config.num_experts and has_moe_fusion:
        compiled_ep_sizes = _boundary_values(ep_sizes)
        compiled_mtp_tokens = _boundary_values(mtp_tokens)
        compiled_interaction_length = representative_lengths[len(representative_lengths) // 2]
        for input_length in representative_lengths:
            # Compile/DFC cost grows sharply for the full max-context x max-EP
            # x max-MTP corner. Query every discrete interaction at one central
            # length, while the short/long anchors isolate the length axis. The
            # coverage planner combines these axes when it densifies the grid.
            is_interaction_anchor = input_length == compiled_interaction_length
            add_scenario(
                sweep_name="compiled_moe",
                num_devices=ep_max,
                input_length=input_length,
                ep=compiled_ep_sizes if is_interaction_anchor else (compiled_ep_sizes[0],),
                mtp=compiled_mtp_tokens if is_interaction_anchor else (compiled_mtp_tokens[0],),
                max_batched_tokens=min(4096, input_length),
                compilation_config=("enable_sequence_parallel", "enable_dispatch_ffn_combine"),
            )

    # Quantized KV-cache kernels have distinct schemas. Representative points
    # are enough because runtime-rich attention rows remain exact-only later.
    for input_length in representative_lengths:
        add_scenario(
            sweep_name="int8_kv_cache",
            num_devices=1,
            input_length=input_length,
            quantize_attention_action="INT8",
        )

    batch_scenarios = _split_non_baseline_batch_boundaries(
        scenarios,
        batch_token_budget=model_config.max_context_length,
    )
    unique_scenarios = {
        scenario.workload_id: scenario for scenario in _expand_parallel_combinations(batch_scenarios)
    }
    return [unique_scenarios[key] for key in sorted(unique_scenarios)]


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _scenario_cache_directory(trace_dir: Path, scenario: WorkloadScenario) -> Path:
    digest = hashlib.sha256(scenario.workload_id.encode("utf-8")).hexdigest()[:20]
    return trace_dir / "workloads" / digest


def _command_digest(command: list[str]) -> str:
    payload = json.dumps(command, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _database_content_fingerprint(database_path: Path) -> str:
    """Return a stable fingerprint for the database files that affect lookup."""
    digest = hashlib.sha256()
    candidates = [database_path / "op_mapping.yaml", *sorted(database_path.rglob("*.csv"))]
    for path in candidates:
        if not path.is_file():
            continue
        digest.update(path.relative_to(database_path).as_posix().encode("utf-8"))
        with path.open("rb") as input_file:
            while chunk := input_file.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _cached_execution(
    *,
    index: int,
    scenario: WorkloadScenario,
    command: list[str],
    trace_dir: Path,
    database_fingerprint: str,
) -> _WorkloadExecution | None:
    scenario_dir = _scenario_cache_directory(trace_dir, scenario)
    checkpoint_path = scenario_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        checkpoint.get("checkpoint_version") != QUERY_WORKLOAD_CHECKPOINT_VERSION
        or checkpoint.get("policy_version") != QUERY_WORKLOAD_POLICY_VERSION
        or checkpoint.get("workload_id") != scenario.workload_id
        or checkpoint.get("command_digest") != _command_digest(command)
        or checkpoint.get("database_fingerprint") != database_fingerprint
        or checkpoint.get("returncode") != 0
    ):
        return None
    relative_trace = checkpoint.get("trace_directory")
    if not isinstance(relative_trace, str) or not relative_trace:
        return None
    completed_trace_dir = (trace_dir / relative_trace).resolve()
    if not completed_trace_dir.is_relative_to(trace_dir) or not completed_trace_dir.is_dir():
        return None
    try:
        demand_count = len(load_query_demand_traces(completed_trace_dir))
    except (OSError, TypeError, ValueError):
        return None
    return _WorkloadExecution(
        index=index,
        scenario=scenario,
        returncode=0,
        elapsed_seconds=0.0,
        trace_directory=completed_trace_dir,
        demand_count=demand_count,
        cached=True,
    )


def _write_checkpoint(
    execution: _WorkloadExecution,
    *,
    command: list[str],
    trace_dir: Path,
    database_fingerprint: str,
) -> None:
    scenario_dir = _scenario_cache_directory(trace_dir, execution.scenario)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = scenario_dir / "checkpoint.json"
    temporary_path = scenario_dir / f"checkpoint-{os.getpid()}-{time.time_ns()}.tmp"
    temporary_path.write_text(
        json.dumps(
            {
                "checkpoint_version": QUERY_WORKLOAD_CHECKPOINT_VERSION,
                "policy_version": QUERY_WORKLOAD_POLICY_VERSION,
                "workload_id": execution.scenario.workload_id,
                "command_digest": _command_digest(command),
                "database_fingerprint": database_fingerprint,
                "returncode": execution.returncode,
                "elapsed_seconds": round(execution.elapsed_seconds, 6),
                "demand_count": execution.demand_count,
                "trace_directory": execution.trace_directory.relative_to(trace_dir).as_posix(),
                "completion_reason": execution.completion_reason,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)


def _trace_state(trace_dir: Path) -> tuple[int, int, int]:
    trace_files = list(trace_dir.glob("query-demands-*.jsonl"))
    if not trace_files:
        return 0, 0, 0
    stats = [path.stat() for path in trace_files]
    return len(stats), sum(stat.st_size for stat in stats), max(stat.st_mtime_ns for stat in stats)


def _trace_covers_parallel_axes(trace_dir: Path, scenario: WorkloadScenario) -> bool:
    try:
        demands = load_query_demand_traces(trace_dir)
    except (OSError, TypeError, ValueError):
        return False
    if not demands:
        return False
    observed_tp = {demand.tensor_parallel_size for demand in demands}
    observed_ep = {demand.expert_parallel_size for demand in demands}
    return set(scenario.tp_sizes) <= observed_tp and set(scenario.ep_sizes) <= observed_ep


def _process_tree_cpu_seconds(process_id: int) -> float:
    try:
        root = psutil.Process(process_id)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.Error):
        return 0.0
    total = 0.0
    for process in processes:
        try:
            cpu_times = process.cpu_times()
        except (psutil.NoSuchProcess, psutil.Error):
            continue
        total += cpu_times.user + cpu_times.system
    return total


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        root = psutil.Process(process.pid)
        processes = [*root.children(recursive=True), root]
    except (psutil.NoSuchProcess, psutil.Error):
        processes = []
    for child in processes:
        try:
            child.terminate()
        except (psutil.NoSuchProcess, psutil.Error):
            continue
    _, alive = psutil.wait_procs(processes, timeout=QUERY_PROCESS_TERMINATE_GRACE_SECONDS)
    for child in alive:
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.Error):
            continue
    if alive:
        psutil.wait_procs(alive, timeout=QUERY_PROCESS_TERMINATE_GRACE_SECONDS)
    try:
        process.wait(timeout=QUERY_PROCESS_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_subprocess_until_queries_converge(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    trace_dir: Path,
    scenario: WorkloadScenario,
    stop_event: threading.Event,
    trace_quiet_seconds: float,
    max_convergence_wait_seconds: float = MAX_CONVERGENCE_WAIT_SECONDS,
) -> _CommandOutcome:
    """Stop an optimizer after its query trace and process tree are both idle."""
    log_path = trace_dir / "optimizer.log"
    with log_path.open("w+b") as log_file:
        process = subprocess.Popen(  # noqa: S603 - command is constructed internally
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        last_trace_state = (0, 0, 0)
        last_cpu_seconds = _process_tree_cpu_seconds(process.pid)
        last_activity = time.monotonic()
        convergence_deadline = last_activity + max_convergence_wait_seconds
        converged = False
        try:
            while process.poll() is None:
                if stop_event.wait(QUERY_TRACE_POLL_SECONDS):
                    raise InterruptedError("query workload collection interrupted")
                now = time.monotonic()
                trace_state = _trace_state(trace_dir)
                cpu_seconds = _process_tree_cpu_seconds(process.pid)
                if trace_state != last_trace_state or cpu_seconds > last_cpu_seconds + 0.01:
                    last_activity = now
                last_trace_state = trace_state
                last_cpu_seconds = cpu_seconds
                if (
                    trace_state[1] > 0
                    and now - last_activity >= trace_quiet_seconds
                    and (
                        _trace_covers_parallel_axes(trace_dir, scenario)
                        or now >= convergence_deadline
                    )
                ):
                    converged = True
                    _terminate_process_tree(process)
                    break
        except BaseException:
            _terminate_process_tree(process)
            raise
        returncode = process.returncode if process.returncode is not None else process.wait()
        log_file.seek(0)
        stderr = log_file.read()[-8000:].decode("utf-8", errors="replace")[-2000:]
    if converged:
        return _CommandOutcome(returncode=0, completion_reason="query_trace_converged")
    return _CommandOutcome(returncode=returncode, stderr=stderr)


def _execute_workload(
    *,
    index: int,
    total: int,
    scenario: WorkloadScenario,
    database_path: Path,
    trace_dir: Path,
    repo_root: Path,
    command_runner: CommandRunner,
    stop_event: threading.Event,
    trace_quiet_seconds: float,
    database_fingerprint: str,
) -> _WorkloadExecution:
    command = scenario.command(database_path)
    scenario_dir = _scenario_cache_directory(trace_dir, scenario)
    attempt_dir = scenario_dir / f"attempt-{time.time_ns()}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[QUERY] workload {index}/{total} started: combinations={scenario.parallel_combinations}; "
        f"{scenario.workload_id}",
        file=sys.stderr,
        flush=True,
    )
    environment = os.environ.copy()
    environment[QUERY_TRACE_DIR_ENV] = str(attempt_dir)
    environment[QUERY_TRACE_MODEL_ENV] = scenario.model_id
    environment[QUERY_TRACE_WORKLOAD_ENV] = scenario.workload_id
    started = time.monotonic()
    try:
        if command_runner is subprocess.run:
            outcome = _run_subprocess_until_queries_converge(
                command,
                cwd=repo_root,
                env=environment,
                trace_dir=attempt_dir,
                scenario=scenario,
                stop_event=stop_event,
                trace_quiet_seconds=trace_quiet_seconds,
            )
        else:
            completed = command_runner(
                command,
                cwd=repo_root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            outcome = _CommandOutcome(returncode=completed.returncode, stderr=completed.stderr or "")
    except Exception as error:  # pragma: no cover - subprocess.run normally returns a result
        outcome = _CommandOutcome(returncode=1, stderr=f"{type(error).__name__}: {error}")
    elapsed = time.monotonic() - started
    demand_count = len(load_query_demand_traces(attempt_dir)) if outcome.returncode == 0 else 0
    execution = _WorkloadExecution(
        index=index,
        scenario=scenario,
        returncode=outcome.returncode,
        elapsed_seconds=elapsed,
        trace_directory=attempt_dir,
        demand_count=demand_count,
        stderr=outcome.stderr,
        completion_reason=outcome.completion_reason,
    )
    _write_checkpoint(
        execution,
        command=command,
        trace_dir=trace_dir,
        database_fingerprint=database_fingerprint,
    )
    return execution


def _write_workload_summary(
    summary_path: Path,
    *,
    attempted: int,
    succeeded: int,
    cached: int,
    failed_workloads: list[str],
    elapsed_seconds: float,
    planned_combinations: int,
    max_workers: int,
) -> None:
    summary_path.write_text(
        json.dumps(
            {
                "policy_version": QUERY_WORKLOAD_POLICY_VERSION,
                "attempted": attempted,
                "succeeded": succeeded,
                "cached": cached,
                "failed_workloads": failed_workloads,
                "elapsed_seconds": round(elapsed_seconds, 6),
                "planned_parallel_combinations": planned_combinations,
                "max_parallel_workloads": max_workers,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run_query_workloads(
    scenarios: list[WorkloadScenario],
    *,
    database_path: Path,
    trace_dir: Path,
    repo_root: Path,
    command_runner: CommandRunner = subprocess.run,
    max_workers: int | None = None,
    trace_quiet_seconds: float = QUERY_TRACE_QUIET_SECONDS,
) -> QueryWorkloadRunResult:
    """Run isolated optimizer scenarios with bounded concurrency and checkpoints."""
    trace_dir = trace_dir.resolve()
    database_path = database_path.resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    database_fingerprint = _database_content_fingerprint(database_path)
    summary_path = trace_dir / "workload-summary.json"
    total = len(scenarios)
    planned_combinations = sum(scenario.parallel_combinations for scenario in scenarios)
    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        cpu_limit = max(1, cpu_count // 2)
        memory_limit = max(1, int(psutil.virtual_memory().available // (1024**3)))
        worker_limit = min(MAX_PARALLEL_QUERY_WORKLOADS, cpu_limit, memory_limit)
    else:
        worker_limit = int(max_workers)
    worker_limit = max(1, min(worker_limit, total or 1, cpu_count))
    started = time.monotonic()
    stop_event = threading.Event()
    executions: dict[str, _WorkloadExecution] = {}
    pending: list[tuple[int, WorkloadScenario]] = []
    for index, scenario in enumerate(scenarios, start=1):
        command = scenario.command(database_path)
        cached_execution = _cached_execution(
            index=index,
            scenario=scenario,
            command=command,
            trace_dir=trace_dir,
            database_fingerprint=database_fingerprint,
        )
        if cached_execution is None:
            pending.append((index, scenario))
            continue
        executions[scenario.workload_id] = cached_execution
        print(
            f"[QUERY] workload {index}/{total} cached: demands={cached_execution.demand_count}; "
            f"{scenario.workload_id}",
            file=sys.stderr,
            flush=True,
        )

    run_durations: list[float] = []
    if pending:
        executor = ThreadPoolExecutor(max_workers=worker_limit, thread_name_prefix="shape-query")
        try:
            futures: dict[Future[_WorkloadExecution], tuple[int, WorkloadScenario]] = {
                executor.submit(
                    _execute_workload,
                    index=index,
                    total=total,
                    scenario=scenario,
                    database_path=database_path,
                    trace_dir=trace_dir,
                    repo_root=repo_root,
                    command_runner=command_runner,
                    stop_event=stop_event,
                    trace_quiet_seconds=trace_quiet_seconds,
                    database_fingerprint=database_fingerprint,
                ): (index, scenario)
                for index, scenario in pending
            }
            for future in as_completed(futures):
                execution = future.result()
                executions[execution.scenario.workload_id] = execution
                run_durations.append(execution.elapsed_seconds)
                completed_count = len(executions)
                remaining = total - completed_count
                average_duration = sum(run_durations) / len(run_durations)
                eta_seconds = average_duration * remaining / worker_limit
                if execution.returncode == 0:
                    print(
                        f"[QUERY] workload {execution.index}/{total} finished: "
                        f"elapsed={execution.elapsed_seconds:.1f}s, demands={execution.demand_count}, "
                        f"reason={execution.completion_reason}, completed={completed_count}/{total}, "
                        f"eta={eta_seconds:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    stderr_tail = execution.stderr[-2000:]
                    print(
                        f"[QUERY] workload {execution.index}/{total} failed with exit code "
                        f"{execution.returncode}: {execution.scenario.workload_id}\n{stderr_tail}",
                        file=sys.stderr,
                        flush=True,
                    )
                failed_so_far = [
                    scenario.workload_id
                    for scenario in scenarios
                    if scenario.workload_id in executions and executions[scenario.workload_id].returncode != 0
                ]
                _write_workload_summary(
                    summary_path,
                    attempted=total,
                    succeeded=sum(execution.returncode == 0 for execution in executions.values()),
                    cached=sum(execution.cached for execution in executions.values()),
                    failed_workloads=failed_so_far,
                    elapsed_seconds=time.monotonic() - started,
                    planned_combinations=planned_combinations,
                    max_workers=worker_limit,
                )
        except BaseException:
            stop_event.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    ordered_executions = [executions[scenario.workload_id] for scenario in scenarios if scenario.workload_id in executions]
    failed_workloads = tuple(
        execution.scenario.workload_id for execution in ordered_executions if execution.returncode != 0
    )
    result = QueryWorkloadRunResult(
        attempted=total,
        succeeded=sum(execution.returncode == 0 for execution in ordered_executions),
        failed_workloads=failed_workloads,
        cached=sum(execution.cached for execution in ordered_executions),
        elapsed_seconds=time.monotonic() - started,
        trace_directories=tuple(
            execution.trace_directory for execution in ordered_executions if execution.returncode == 0
        ),
    )
    _write_workload_summary(
        summary_path,
        attempted=result.attempted,
        succeeded=result.succeeded,
        cached=result.cached,
        failed_workloads=list(result.failed_workloads),
        elapsed_seconds=result.elapsed_seconds,
        planned_combinations=planned_combinations,
        max_workers=worker_limit,
    )
    return result

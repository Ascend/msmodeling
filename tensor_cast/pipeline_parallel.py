"""Pipeline-parallel model construction helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import logging
from typing import Any, Optional

import torch

from . import ops  # noqa: F401
from .core.user_config import UserInputConfig
from .device import DeviceProfile
from .model_config import ModelConfig, ParallelConfig
from .performance_model.memory_tracker import MemoryTracker
from .performance_model.utils import bytes_of_tensor
from .runtime import Runtime


@dataclass(frozen=True)
class PipelineStageSpec:
    """Global layer range and boundary role for one PP stage."""

    stage_id: int
    pp_size: int
    layer_start: int
    layer_end: int
    is_first: bool
    is_last: bool
    layer_types: tuple[str, ...] = ()

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start


@dataclass(frozen=True)
class PipelinePlan:
    """Complete PP split plan for a decoder-only model."""

    num_hidden_layers: int
    stages: tuple[PipelineStageSpec, ...]


@dataclass(frozen=True)
class PipelineStageCacheStats:
    """KV/indexer cache memory attributed to one stage-local run."""

    kv_cache_bytes: int = 0
    kv_cache_per_token_bytes: float = 0.0
    indexer_cache_bytes: int = 0
    indexer_cache_per_token_bytes: float = 0.0


@dataclass(frozen=True)
class PipelineTransferStats:
    """Estimated hidden-state transfer cost between two adjacent stages."""

    source_stage_id: int
    target_stage_id: int
    payload_bytes: int
    time_s: float
    bandwidth_bytes_ps: float
    latency_s: float
    time_s_by_model: dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineRunResult:
    """Aggregated output and profiling data for a full PP forward pass."""

    logits: torch.Tensor
    execution_time_s: dict[str, float]
    table_result: str
    breakdowns: dict[str, dict[str, float]]
    runtime_event_list: list[dict]
    peak_memory_usage_bytes: float
    kv_cache_size_bytes: int
    kv_cache_per_token_bytes: float
    indexer_cache_size_bytes: int = 0
    indexer_cache_per_token_bytes: float = 0.0
    model_weight_size_bytes: int = 0
    model_activation_size_bytes: Optional[float] = None
    stage_latency_breakdown: list[dict] = field(default_factory=list)
    stage_memory_breakdown: list[dict] = field(default_factory=list)
    trace_events: list[dict] = field(default_factory=list)


def _exact_division(numerator: int, denominator: int, field_name: str) -> int:
    if denominator <= 0 or numerator % denominator != 0:
        raise ValueError(f"Cannot derive {field_name}: {numerator} is not divisible by {denominator}.")
    return numerator // denominator


def _derive_stage_local_rank(
    parallel_config: ParallelConfig,
    pp_size: int,
    stage_id: int,
) -> int:
    if parallel_config.rank == -1:
        return -1
    if stage_id < 0 or stage_id >= pp_size:
        raise ValueError(f"stage_id ({stage_id}) must be in [0, {pp_size}).")
    if parallel_config.rank < 0 or parallel_config.rank >= parallel_config.world_size:
        raise ValueError(f"rank ({parallel_config.rank}) must be in [0, {parallel_config.world_size}).")

    inner_parallel_size = parallel_config.tensor_parallel_size
    stage_group_size = pp_size * inner_parallel_size
    if parallel_config.world_size % stage_group_size != 0:
        raise ValueError(
            f"world_size ({parallel_config.world_size}) must be divisible by "
            f"pipeline_parallel_size ({pp_size}) * inner_parallel_size ({inner_parallel_size})."
        )

    outer_rank = parallel_config.rank // stage_group_size
    inner_rank = parallel_config.rank % inner_parallel_size
    return outer_rank * inner_parallel_size + inner_rank


def _stage_boundary_ranks(parallel_config: ParallelConfig, stage_id: int) -> tuple[int, int]:
    """Return source and target dense ranks for a PP stage boundary."""
    pp_size = parallel_config.pipeline_parallel_size
    if stage_id < 0 or stage_id >= pp_size - 1:
        raise ValueError(f"stage_id {stage_id} is not a valid non-last pipeline stage.")
    return (
        _dense_rank_for_pipeline_stage(parallel_config, stage_id),
        _dense_rank_for_pipeline_stage(parallel_config, stage_id + 1),
    )


def _dense_rank_for_pipeline_stage(parallel_config: ParallelConfig, stage_id: int) -> int:
    pp_size = parallel_config.pipeline_parallel_size
    local_world_size = _exact_division(parallel_config.world_size, pp_size, "pipeline_stage_world_size")
    stage_local_rank = _derive_stage_local_rank(parallel_config, pp_size, stage_id)
    if stage_local_rank == -1:
        stage_local_rank = 0
    if stage_local_rank < 0 or stage_local_rank >= local_world_size:
        raise ValueError(f"stage-local rank ({stage_local_rank}) must be in [0, {local_world_size}).")

    tensor_parallel_size = parallel_config.tensor_parallel_size
    data_parallel_size = parallel_config.data_parallel_size
    ranks_per_outer_group = data_parallel_size * tensor_parallel_size
    if local_world_size % ranks_per_outer_group != 0:
        raise ValueError(
            f"pipeline stage world size ({local_world_size}) must be divisible by "
            f"data_parallel_size ({data_parallel_size}) * tensor_parallel_size ({tensor_parallel_size})."
        )

    outer_rank, rank_in_outer_group = divmod(stage_local_rank, ranks_per_outer_group)
    data_parallel_rank, tensor_parallel_rank = divmod(rank_in_outer_group, tensor_parallel_size)
    return (
        (outer_rank * data_parallel_size + data_parallel_rank) * pp_size + stage_id
    ) * tensor_parallel_size + tensor_parallel_rank


_LAYER_METADATA_FIELDS = (
    "layer_types",
    "mlp_layer_types",
    "indexer_types",
    "compress_ratios",
    "hybrid_layer_pattern",
    "moe_layer_freq",
)
_EMBEDDING_BOUNDARY_PATHS = (
    "embed_tokens",
    "word_embeddings",
    "model.embed_tokens",
    "model.word_embeddings",
    "transformer.wte",
)
_FINAL_NORM_BOUNDARY_PATHS = ("norm", "model.norm", "transformer.ln_f")
_LM_HEAD_BOUNDARY_PATHS = ("lm_head", "model.lm_head", "language_model.lm_head")


def _dump_input_shapes(user_input: Optional[UserInputConfig]) -> bool:
    return bool(getattr(user_input, "dump_input_shapes", False))


def _dump_op_bound_results(user_input: Optional[UserInputConfig]) -> bool:
    return bool(getattr(user_input, "dump_op_bound_results", False))


class PPMissingLayer(torch.nn.Identity):
    """Placeholder for modules that are not owned by the current PP stage."""

    def forward(self, *args, **kwargs):
        """Return the hidden-state-like input owned by the surrounding stage wrapper."""
        if kwargs.get("inputs_embeds") is not None:
            return kwargs["inputs_embeds"]
        if kwargs.get("hidden_states") is not None:
            return kwargs["hidden_states"]
        if args and args[0] is not None:
            return args[0]
        for value in kwargs.values():
            if value is not None:
                return value
        raise ValueError("PPMissingLayer requires at least one non-None input to pass through.")


def build_pipeline_plan(
    model_config: ModelConfig,
    pp_size: int,
    layer_partition: Optional[tuple[int, ...]] = None,
) -> PipelinePlan:
    """Build a contiguous decoder-layer split for PP execution."""
    text_config = model_config.hf_config.get_text_config()
    num_hidden_layers = model_config.num_hidden_layers_override or text_config.num_hidden_layers
    if pp_size <= 1:
        if layer_partition is not None:
            raise ValueError(
                f"pp_layer_partition requires pp_size > 1; got pp_size ({pp_size}) "
                f"for num_hidden_layers ({num_hidden_layers})."
            )
        raise ValueError("Pipeline parallel plan requires pp_size > 1.")
    if num_hidden_layers < pp_size:
        raise ValueError(f"pipeline_parallel_size ({pp_size}) cannot exceed num_hidden_layers ({num_hidden_layers}).")

    if layer_partition is None:
        layers_per_partition = num_hidden_layers // pp_size
        partitions = [layers_per_partition for _ in range(pp_size)]
        if remaining_layers := num_hidden_layers % pp_size:
            # Follow vLLM's PP partitioning: avoid placing remainder layers on
            # the last stage, which usually also owns final norm / lm_head.
            for offset in range(2, remaining_layers + 2):
                partitions[-offset] += 1
        stage_layer_counts = tuple(partitions)
    else:
        stage_layer_counts = tuple(layer_partition)
        if len(stage_layer_counts) != pp_size:
            raise ValueError(
                f"pp_layer_partition length ({len(stage_layer_counts)}) must match pp_size ({pp_size}) "
                f"for num_hidden_layers ({num_hidden_layers})."
            )
        if any(stage_layers <= 0 for stage_layers in stage_layer_counts):
            raise ValueError(
                f"pp_layer_partition entries must be positive for pp_size ({pp_size}) "
                f"and num_hidden_layers ({num_hidden_layers}): {stage_layer_counts}."
            )
        partition_sum = sum(stage_layer_counts)
        if partition_sum != num_hidden_layers:
            raise ValueError(
                f"pp_layer_partition sum ({partition_sum}) must equal num_hidden_layers ({num_hidden_layers}) "
                f"for pp_size ({pp_size})."
            )
    layer_start = 0
    stages = []
    for stage_id, stage_layers in enumerate(stage_layer_counts):
        layer_end = layer_start + stage_layers
        layer_types = getattr(text_config, "layer_types", None)
        stages.append(
            PipelineStageSpec(
                stage_id=stage_id,
                pp_size=pp_size,
                layer_start=layer_start,
                layer_end=layer_end,
                is_first=stage_id == 0,
                is_last=stage_id == pp_size - 1,
                layer_types=tuple(layer_types[layer_start:layer_end]) if isinstance(layer_types, list) else (),
            )
        )
        layer_start = layer_end
    return PipelinePlan(num_hidden_layers=num_hidden_layers, stages=tuple(stages))


def build_stage_model_config(model_config: ModelConfig, stage_spec: PipelineStageSpec) -> ModelConfig:
    """Derive a stage-local ModelConfig from the global model config."""
    stage_config = copy.deepcopy(model_config)
    text_config = stage_config.hf_config.get_text_config()
    _rebase_dense_prefix_policy(text_config, stage_spec)
    _slice_hf_layer_metadata(text_config, stage_spec.layer_start, stage_spec.layer_end)
    _rebase_attention_quant_config(stage_config.quant_config, stage_spec)
    stage_config.parallel_config = build_stage_parallel_config(
        model_config.parallel_config,
        stage_spec.pp_size,
        stage_id=stage_spec.stage_id,
        has_moe=model_config.moe_config is not None,
    )
    stage_config.num_hidden_layers_override = stage_spec.num_layers
    return stage_config


def apply_stage_boundaries(stage_model: torch.nn.Module, stage_spec: PipelineStageSpec) -> None:
    """Remove boundary modules not owned by this stage."""
    boundary_roots = _get_stage_boundary_roots(stage_model)
    unwrapped_roots = [root for root in boundary_roots if root is not getattr(stage_model, "_inner", None)]
    if not stage_spec.is_first:
        _replace_boundary_paths(unwrapped_roots, _EMBEDDING_BOUNDARY_PATHS)
    if not stage_spec.is_last:
        _replace_boundary_paths(boundary_roots, _LM_HEAD_BOUNDARY_PATHS)
        _replace_boundary_paths(unwrapped_roots, _FINAL_NORM_BOUNDARY_PATHS)


def build_stage_parallel_config(
    parallel_config: ParallelConfig,
    pp_size: int,
    *,
    stage_id: int,
    has_moe: bool = False,
) -> ParallelConfig:
    """Drop the PP dimension and recompute stage-local parallel groups."""
    if parallel_config.world_size % pp_size != 0:
        raise ValueError(
            f"world_size ({parallel_config.world_size}) must be divisible by pipeline_parallel_size ({pp_size})."
        )
    local_world_size = parallel_config.world_size // pp_size
    local_rank = _derive_stage_local_rank(parallel_config, pp_size, stage_id)

    if has_moe:
        expert_parallel_size = parallel_config.expert_parallel_size
        moe_tensor_parallel_size = parallel_config.moe_tensor_parallel_size
        moe_data_parallel_size = _exact_division(
            local_world_size,
            expert_parallel_size * moe_tensor_parallel_size,
            "moe_data_parallel_size",
        )
    else:
        expert_parallel_size = 1
        moe_tensor_parallel_size = 1
        moe_data_parallel_size = local_world_size

    return ParallelConfig(
        world_size=local_world_size,
        rank=local_rank,
        tensor_parallel_size=parallel_config.tensor_parallel_size,
        data_parallel_size=_exact_division(
            local_world_size,
            parallel_config.tensor_parallel_size,
            "data_parallel_size",
        ),
        pipeline_parallel_size=1,
        source_pipeline_parallel_size=parallel_config.source_pipeline_parallel_size,
        o_proj_tensor_parallel_size=parallel_config.o_proj_tensor_parallel_size,
        o_proj_data_parallel_size=_exact_division(
            local_world_size,
            parallel_config.o_proj_tensor_parallel_size,
            "o_proj_data_parallel_size",
        ),
        mlp_tensor_parallel_size=parallel_config.mlp_tensor_parallel_size,
        mlp_data_parallel_size=_exact_division(
            local_world_size,
            parallel_config.mlp_tensor_parallel_size,
            "mlp_data_parallel_size",
        ),
        lmhead_tensor_parallel_size=parallel_config.lmhead_tensor_parallel_size,
        lmhead_data_parallel_size=_exact_division(
            local_world_size,
            parallel_config.lmhead_tensor_parallel_size,
            "lmhead_data_parallel_size",
        ),
        embedding_parallel=parallel_config.embedding_parallel,
        expert_parallel_size=expert_parallel_size,
        moe_tensor_parallel_size=moe_tensor_parallel_size,
        moe_data_parallel_size=moe_data_parallel_size,
        ulysses_size=parallel_config.ulysses_size,
    )


def _rebase_dense_prefix_policy(text_config, stage_spec: PipelineStageSpec) -> None:
    original_first_dense_replace = int(getattr(text_config, "first_k_dense_replace", 0) or 0)
    if hasattr(text_config, "first_k_dense_replace"):
        stage_first_dense_replace = max(0, original_first_dense_replace - stage_spec.layer_start)
        text_config.first_k_dense_replace = min(stage_spec.num_layers, stage_first_dense_replace)

    if hasattr(text_config, "num_hash_layers"):
        original_num_hash_layers = int(getattr(text_config, "num_hash_layers", 0) or 0)
        hash_start = original_first_dense_replace
        hash_end = hash_start + original_num_hash_layers
        stage_hash_layers = max(
            0,
            min(stage_spec.layer_end, hash_end) - max(stage_spec.layer_start, hash_start),
        )
        text_config.num_hash_layers = stage_hash_layers


def _rebase_attention_quant_config(quant_config, stage_spec: PipelineStageSpec) -> None:
    # attention_configs is keyed by global layer index; each stage-local config
    # must be rebased so its first owned layer is index 0.
    attention_configs = getattr(quant_config, "attention_configs", None)
    if not attention_configs:
        return

    rebased_configs = {}
    if -1 in attention_configs:
        rebased_configs[-1] = attention_configs[-1]
    for layer_idx, attention_config in attention_configs.items():
        if layer_idx == -1:
            continue
        if not isinstance(layer_idx, int):
            raise ValueError(f"attention_configs keys must be layer indices, got {layer_idx!r}.")
        if stage_spec.layer_start <= layer_idx < stage_spec.layer_end:
            rebased_configs[layer_idx - stage_spec.layer_start] = attention_config
    quant_config.attention_configs = rebased_configs


def _slice_hf_layer_metadata(text_config, layer_start: int, layer_end: int) -> None:
    # These fields are per-layer config metadata lists, not instantiated modules.
    # Stage-local models rebuild their module stack from the sliced config.
    for field_name in _LAYER_METADATA_FIELDS:
        if not hasattr(text_config, field_name):
            continue
        values = getattr(text_config, field_name)
        if isinstance(values, list):
            if len(values) < layer_end:
                raise ValueError(
                    f"Cannot slice hf_config.{field_name} for pipeline layer range "
                    f"{layer_start}:{layer_end}; only {len(values)} entries are available."
                )
            setattr(text_config, field_name, values[layer_start:layer_end])


def _get_stage_boundary_roots(stage_model: torch.nn.Module) -> list[torch.nn.Module]:
    roots = []
    wrapper = getattr(stage_model, "_inner", None)
    if isinstance(wrapper, torch.nn.Module):
        roots.append(wrapper)
    if hasattr(stage_model, "unwrap"):
        unwrapped = stage_model.unwrap()
        if isinstance(unwrapped, torch.nn.Module):
            roots.append(unwrapped)

    unique_roots = []
    seen_ids = set()
    for root in roots:
        if id(root) in seen_ids:
            continue
        seen_ids.add(id(root))
        unique_roots.append(root)
    return unique_roots


def _replace_boundary_paths(roots: list[torch.nn.Module], paths: tuple[str, ...]) -> None:
    for root in roots:
        for path in paths:
            _replace_module_path_if_present(root, path)


def _replace_module_path_if_present(root: torch.nn.Module, path: str) -> None:
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(parent, part):
            return
        parent = getattr(parent, part)
        if not isinstance(parent, torch.nn.Module):
            return

    child_name = parts[-1]
    if not hasattr(parent, child_name):
        return
    if isinstance(getattr(parent, child_name), torch.nn.Module):
        setattr(parent, child_name, PPMissingLayer())


_PASSTHROUGH_STAGE_KWARGS = ("attention_meta", "cache_position")


class PipelineStageModel(torch.nn.Module):
    """Wrap a stage-local model and enforce the PP stage I/O contract."""

    def __init__(self, stage_spec: PipelineStageSpec, model: torch.nn.Module):
        super().__init__()
        self.stage_spec = stage_spec
        self.model = model

    @property
    def weight_size(self) -> int:
        return getattr(self.model, "weight_size", 0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run this stage with first/middle/last stage input semantics."""
        if position_ids is None:
            raise ValueError("PipelineStageModel.forward requires position_ids.")

        if self.stage_spec.is_first:
            if input_ids is None:
                raise ValueError("The first pipeline stage requires input_ids.")
            return self._forward_to_hidden_states(input_ids=input_ids, position_ids=position_ids, **kwargs)

        if hidden_states is None:
            raise ValueError("Non-first pipeline stages require hidden_states.")
        if self.stage_spec.is_last:
            return self.model.forward(
                input_ids=None,
                position_ids=position_ids,
                inputs_embeds=hidden_states,
                **kwargs,
            )
        return self._forward_to_hidden_states(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=hidden_states,
            **kwargs,
        )

    def _forward_to_hidden_states(self, **kwargs) -> torch.Tensor:
        """Run the wrapped model and return its intermediate hidden states."""
        outputs = self.model.forward(output_intermediate_hidden_states=True, **kwargs)
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            output_length = len(outputs) if isinstance(outputs, tuple) else "N/A"
            raise ValueError(
                "Non-last pipeline stages must return (logits, hidden_states) when "
                "output_intermediate_hidden_states=True. "
                f"Got {type(outputs).__name__} with length {output_length}."
            )
        hidden_states = outputs[1]
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError(
                "Non-last pipeline stages must return hidden_states as a torch.Tensor in (logits, hidden_states). "
                f"Got {type(hidden_states).__name__}."
            )
        return hidden_states


class PipelineModel(torch.nn.Module):
    """Container for stage-local models built from a PipelinePlan."""

    def __init__(
        self,
        model_config: ModelConfig,
        plan: PipelinePlan,
        stages: list[PipelineStageModel],
    ):
        super().__init__()
        self.model_config = model_config
        self.plan = plan
        self.stages = torch.nn.ModuleList(stages)

    @property
    def weight_size(self) -> int:
        return max((stage.weight_size for stage in self.stages), default=0)

    @property
    def is_vl_model(self) -> bool:
        """Pipeline split models are text-only in this code path."""
        return False

    @property
    def hf_config(self):
        return self.model_config.hf_config

    @property
    def text_config(self):
        return self.hf_config.get_text_config()

    @property
    def total_weight_size(self) -> int:
        return sum(stage.weight_size for stage in self.stages)

    @property
    def num_hidden_layers(self) -> int:
        return self.plan.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def head_dim(self) -> int:
        return getattr(
            self.text_config,
            "head_dim",
            self.hidden_size // self.text_config.num_attention_heads,
        )

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    def forward(self, **input_kwargs) -> torch.Tensor:
        """Execute all stage-local models serially and return final logits."""
        hidden_states = None
        logits = None
        for stage in self.stages:
            stage_kwargs, _ = build_pipeline_stage_kwargs(
                stage.stage_spec,
                input_kwargs,
                hidden_states=hidden_states,
            )
            stage_output = stage(**stage_kwargs)
            if stage.stage_spec.is_last:
                logits = stage_output
            else:
                hidden_states = stage_output
        if logits is None:
            raise RuntimeError("PipelineModel.forward did not execute a last pipeline stage.")
        return logits


def build_pipeline_stage_kwargs(
    stage_spec: PipelineStageSpec,
    input_kwargs: dict[str, Any],
    *,
    hidden_states: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], PipelineStageCacheStats]:
    """Select and rebase global inputs for one pipeline stage."""
    if "position_ids" not in input_kwargs:
        raise ValueError("Pipeline stage inputs require position_ids.")

    stage_kwargs: dict[str, Any] = {"position_ids": input_kwargs["position_ids"]}
    if stage_spec.is_first:
        if "input_ids" not in input_kwargs:
            raise ValueError("The first pipeline stage requires input_ids.")
        stage_kwargs["input_ids"] = input_kwargs["input_ids"]
    else:
        if hidden_states is None:
            raise ValueError("Non-first pipeline stages require hidden_states.")
        # PipelineStageModel.forward owns the PP stage I/O contract and converts
        # this wrapper-level hidden_states input to TransformerModel inputs_embeds.
        stage_kwargs["hidden_states"] = hidden_states

    for key in _PASSTHROUGH_STAGE_KWARGS:
        if key in input_kwargs:
            stage_kwargs[key] = input_kwargs[key]

    cache_stats: dict[str, tuple[int, float]] = {}
    for cache_key, metric_key in (
        ("kv_cache_by_layers", "kv_cache_per_token"),
        ("indexer_cache_by_layers", "indexer_cache_per_token"),
    ):
        cache_by_layers, cache_bytes, cache_per_token = _slice_layer_cache(
            input_kwargs,
            stage_spec,
            cache_key=cache_key,
            metric_key=metric_key,
        )
        cache_stats[cache_key] = (cache_bytes, cache_per_token)
        if cache_key in input_kwargs:
            stage_kwargs[cache_key] = cache_by_layers
            stage_kwargs[metric_key] = cache_per_token

    if stage_spec.is_last and "sampling_metadata" in input_kwargs:
        stage_kwargs["sampling_metadata"] = input_kwargs["sampling_metadata"]

    kv_cache_bytes, kv_cache_per_token = cache_stats["kv_cache_by_layers"]
    indexer_cache_bytes, indexer_cache_per_token = cache_stats["indexer_cache_by_layers"]
    return stage_kwargs, PipelineStageCacheStats(
        kv_cache_bytes=kv_cache_bytes,
        kv_cache_per_token_bytes=kv_cache_per_token,
        indexer_cache_bytes=indexer_cache_bytes,
        indexer_cache_per_token_bytes=indexer_cache_per_token,
    )


def _slice_layer_cache(
    input_kwargs: dict[str, Any],
    stage_spec: PipelineStageSpec,
    *,
    cache_key: str,
    metric_key: str,
) -> tuple[dict[int, torch.Tensor], int, float]:
    """Slice global layer-indexed cache into stage-local layer indices."""
    if cache_key not in input_kwargs:
        return {}, 0, 0.0

    cache_by_layers = input_kwargs[cache_key]
    if not isinstance(cache_by_layers, dict):
        raise ValueError(f"{cache_key} must be a layer-indexed dict for pipeline parallel execution.")

    allow_sparse_layers = cache_key == "indexer_cache_by_layers"
    if not allow_sparse_layers:
        missing_layers = [
            layer_idx
            for layer_idx in range(stage_spec.layer_start, stage_spec.layer_end)
            if layer_idx not in cache_by_layers
        ]
        if missing_layers:
            raise ValueError(
                f"{cache_key} is missing global layer(s) {missing_layers} required by pipeline stage {stage_spec.stage_id}."
            )

    total_bytes = 0
    for layer_idx, cache in cache_by_layers.items():
        total_bytes += int(bytes_of_tensor(cache))

    local_cache = {}
    selected_bytes = 0
    for layer_idx in range(stage_spec.layer_start, stage_spec.layer_end):
        if allow_sparse_layers and layer_idx not in cache_by_layers:
            continue
        cache = cache_by_layers[layer_idx]
        local_cache[layer_idx - stage_spec.layer_start] = cache
        selected_bytes += int(bytes_of_tensor(cache))
    global_per_token_bytes = input_kwargs.get(metric_key, 0.0)
    if total_bytes <= 0 or global_per_token_bytes <= 0:
        return local_cache, selected_bytes, 0.0
    return local_cache, selected_bytes, float(global_per_token_bytes) * selected_bytes / total_bytes


logger = logging.getLogger(__name__)


@dataclass
class _PipelineStageRunResult:
    stage_spec: PipelineStageSpec
    output: Any
    execution_time_s: dict[str, float]
    table_result: Any
    breakdowns: dict[str, dict[str, float]]
    runtime_events: list[Any]
    trace_events: list[dict[str, Any]]
    cache_stats: PipelineStageCacheStats
    peak_memory_usage_bytes: float
    weight_size_bytes: int


@dataclass
class _PipelineTransferRunResult:
    stats: PipelineTransferStats
    output: torch.Tensor
    table_result: Any
    runtime_events: list[Any]
    trace_events: list[dict[str, Any]]
    peak_memory_usage_bytes: float


class StageRunner:
    """Run one PipelineStageModel under Runtime and MemoryTracker."""

    def __init__(
        self,
        stage: PipelineStageModel,
        perf_models,
        device_profile: DeviceProfile,
        *,
        user_input: Optional[UserInputConfig] = None,
    ):
        self.stage = stage
        self.perf_models = perf_models
        self.device_profile = device_profile
        self.user_input = user_input

    @property
    def stage_spec(self) -> PipelineStageSpec:
        return self.stage.stage_spec

    def run(
        self,
        stage_kwargs: dict[str, Any],
        *,
        cache_stats: PipelineStageCacheStats,
        with_sampler: bool = False,
        sampler: Optional[torch.nn.Module] = None,
        runtime_observer=None,
    ) -> _PipelineStageRunResult:
        """Execute one stage and collect runtime, trace, and memory records."""
        if with_sampler and sampler is None:
            raise ValueError("StageRunner.run requires sampler when with_sampler=True.")
        if with_sampler and "sampling_metadata" not in stage_kwargs:
            raise ValueError("StageRunner.run requires sampling_metadata on the last stage when with_sampler=True.")

        memory_tracker = MemoryTracker(self.device_profile)
        with (
            Runtime(
                self.perf_models,
                self.device_profile,
                memory_tracker=memory_tracker,
            ) as runtime,
            torch.no_grad(),
        ):
            output = self.stage(**stage_kwargs)
            if with_sampler:
                _ = sampler(output, stage_kwargs["sampling_metadata"])

        if runtime_observer is not None:
            runtime_observer(runtime)

        return _PipelineStageRunResult(
            stage_spec=self.stage_spec,
            output=output,
            execution_time_s=runtime.total_execution_time_s(),
            table_result=runtime.table_averages(
                group_by_input_shapes=_dump_input_shapes(self.user_input),
                dump_op_bound_results=_dump_op_bound_results(self.user_input),
            ),
            breakdowns=runtime.get_breakdowns(),
            runtime_events=list(runtime.event_list),
            trace_events=_tag_trace_events(
                runtime.get_trace_events(),
                name_prefix=f"pp_stage_{self.stage_spec.stage_id}",
                metadata={"pp_stage": self.stage_spec.stage_id},
            ),
            cache_stats=cache_stats,
            peak_memory_usage_bytes=_safe_peak_memory_usage(memory_tracker),
            weight_size_bytes=int(self.stage.weight_size),
        )


class PipelineCommunicator:
    """Run PP stage-boundary communication as an explicit TensorCast op."""

    def __init__(
        self,
        model_config: ModelConfig,
        perf_models,
        device_profile: DeviceProfile,
        *,
        user_input: Optional[UserInputConfig] = None,
    ):
        self.model_config = model_config
        self.perf_models = perf_models
        self.device_profile = device_profile
        self.user_input = user_input

    def transfer(
        self,
        stage_spec: PipelineStageSpec,
        hidden_states: torch.Tensor,
        *,
        runtime_observer=None,
    ) -> _PipelineTransferRunResult:
        """Transfer hidden states from this stage to the next PP stage."""
        source_rank, target_rank = _stage_boundary_ranks(self.model_config.parallel_config, stage_spec.stage_id)
        memory_tracker = MemoryTracker(self.device_profile)
        with (
            Runtime(
                self.perf_models,
                self.device_profile,
                memory_tracker=memory_tracker,
            ) as runtime,
            torch.no_grad(),
        ):
            # PP currently transfers only the residual-applied hidden states.
            output = torch.ops.tensor_cast.pipeline_send_recv(
                hidden_states,
                source_rank,
                target_rank,
                stage_spec.stage_id,
                stage_spec.stage_id + 1,
            )

        transfer_stats = self._build_transfer_stats_from_runtime(stage_spec, hidden_states, runtime)

        if runtime_observer is not None:
            runtime_observer(runtime)

        return _PipelineTransferRunResult(
            stats=transfer_stats,
            output=output,
            table_result=runtime.table_averages(
                group_by_input_shapes=_dump_input_shapes(self.user_input),
                dump_op_bound_results=_dump_op_bound_results(self.user_input),
            ),
            runtime_events=list(runtime.event_list),
            trace_events=_tag_trace_events(
                runtime.get_trace_events(),
                name_prefix=f"pp_comm_{transfer_stats.source_stage_id}_to_{transfer_stats.target_stage_id}",
                metadata={
                    "source_stage": transfer_stats.source_stage_id,
                    "target_stage": transfer_stats.target_stage_id,
                    "payload_bytes": transfer_stats.payload_bytes,
                },
            ),
            peak_memory_usage_bytes=_safe_peak_memory_usage(memory_tracker),
        )

    def _build_transfer_stats_from_runtime(
        self,
        stage_spec: PipelineStageSpec,
        hidden_states: torch.Tensor,
        runtime: Runtime,
    ) -> PipelineTransferStats:
        payload_bytes = int(bytes_of_tensor(hidden_states))
        found_transfer_event = False
        for event in runtime.event_list:
            if event.op_invoke_info.func != torch.ops.tensor_cast.pipeline_send_recv.default:
                continue
            found_transfer_event = True
            if not event.perf_results:
                logger.warning(
                    "Pipeline send/recv stage %d->%d produced no performance results for %d payload bytes; "
                    "transfer time falls back to 0.",
                    stage_spec.stage_id,
                    stage_spec.stage_id + 1,
                    payload_bytes,
                )
                break
            time_s_by_model = {
                model_name: float(perf_result.execution_time_s)
                for model_name, perf_result in event.perf_results.items()
            }
            perf_result = next(iter(event.perf_results.values()))
            statistics = perf_result.statistics
            transfer_time_s = float(perf_result.execution_time_s)
            if payload_bytes > 0 and transfer_time_s == 0.0:
                logger.warning(
                    "Pipeline send/recv stage %d->%d estimated 0 transfer time for %d payload bytes; "
                    "check performance model coverage.",
                    int(statistics.get("source_stage_id", stage_spec.stage_id)),
                    int(statistics.get("target_stage_id", stage_spec.stage_id + 1)),
                    payload_bytes,
                )
            return PipelineTransferStats(
                source_stage_id=int(statistics.get("source_stage_id", stage_spec.stage_id)),
                target_stage_id=int(statistics.get("target_stage_id", stage_spec.stage_id + 1)),
                payload_bytes=int(statistics.get("message_size_bytes", payload_bytes)),
                time_s=transfer_time_s,
                bandwidth_bytes_ps=float(statistics.get("bandwidth_bytes_ps", 0.0)),
                latency_s=float(statistics.get("latency_s", 0.0)),
                time_s_by_model=time_s_by_model,
            )
        if payload_bytes > 0 and not found_transfer_event:
            logger.warning(
                "Pipeline send/recv stage %d->%d produced no runtime event for %d payload bytes; "
                "transfer time falls back to 0.",
                stage_spec.stage_id,
                stage_spec.stage_id + 1,
                payload_bytes,
            )
        return PipelineTransferStats(
            source_stage_id=stage_spec.stage_id,
            target_stage_id=stage_spec.stage_id + 1,
            payload_bytes=payload_bytes,
            time_s=0.0,
            bandwidth_bytes_ps=0.0,
            latency_s=0.0,
        )


class PipelineRunner:
    """Execute all PP stages in order and merge compute, comm, and memory metrics."""

    def __init__(
        self,
        model: PipelineModel,
        perf_models,
        device_profile: DeviceProfile,
        *,
        user_input: Optional[UserInputConfig] = None,
    ):
        self.model = model
        self.perf_models = perf_models
        self.device_profile = device_profile
        _validate_pipeline_device_grid(model.model_config, device_profile)
        self.user_input = user_input
        self.communicator = PipelineCommunicator(
            model.model_config,
            perf_models,
            device_profile,
            user_input=user_input,
        )
        self.stage_runners = [
            StageRunner(
                stage,
                perf_models,
                device_profile,
                user_input=user_input,
            )
            for stage in model.stages
        ]

    def run(
        self,
        input_kwargs: dict[str, Any],
        *,
        with_sampler: bool = False,
        sampler: Optional[torch.nn.Module] = None,
        runtime_observer=None,
    ) -> PipelineRunResult:
        """Run stages once in order, estimate transfers, and merge PP metrics."""
        logits, stage_results, transfers = self._run_stage_chain(
            input_kwargs,
            with_sampler=with_sampler,
            sampler=sampler,
            runtime_observer=runtime_observer,
        )
        return self._build_run_result(logits, stage_results, transfers)

    def _run_stage_chain(
        self,
        input_kwargs: dict[str, Any],
        *,
        with_sampler: bool = False,
        sampler: Optional[torch.nn.Module] = None,
        runtime_observer=None,
    ) -> tuple[torch.Tensor, list[_PipelineStageRunResult], list[_PipelineTransferRunResult]]:
        """Run one token chunk through all stages and return raw stage results."""
        stage_results: list[_PipelineStageRunResult] = []
        transfers: list[_PipelineTransferRunResult] = []
        hidden_states = None
        logits = None

        for stage_runner in self.stage_runners:
            stage_kwargs, cache_stats = build_pipeline_stage_kwargs(
                stage_runner.stage_spec,
                input_kwargs,
                hidden_states=hidden_states,
            )
            stage_result = stage_runner.run(
                stage_kwargs,
                cache_stats=cache_stats,
                with_sampler=with_sampler and stage_runner.stage_spec.is_last,
                sampler=sampler,
                runtime_observer=runtime_observer,
            )
            stage_results.append(stage_result)

            if stage_runner.stage_spec.is_last:
                logits = stage_result.output
                continue

            transfer_result = self.communicator.transfer(
                stage_runner.stage_spec,
                stage_result.output,
                runtime_observer=runtime_observer,
            )
            hidden_states = transfer_result.output
            transfers.append(transfer_result)

        if logits is None:
            raise RuntimeError("PipelineRunner did not execute a last pipeline stage.")

        return logits, stage_results, transfers

    def _build_run_result(
        self,
        logits: torch.Tensor,
        stage_results: list[_PipelineStageRunResult],
        transfer_results: list[_PipelineTransferRunResult],
    ) -> PipelineRunResult:
        """Merge per-stage runtime data into the public pipeline result."""
        transfers = [result.stats for result in transfer_results]
        stage_execution_time_s: dict[str, list[float]] = {}
        for result in stage_results:
            for model_name, execution_time_s in result.execution_time_s.items():
                stage_execution_time_s.setdefault(model_name, []).append(execution_time_s)

        execution_time_s = {
            model_name: _pipeline_total_time_s(stage_times, transfers, model_name=model_name)
            for model_name, stage_times in stage_execution_time_s.items()
        }
        breakdowns = _merge_breakdowns(result.breakdowns for result in stage_results)
        breakdowns.update(_pipeline_parallel_breakdowns(stage_execution_time_s, transfers))
        # Attribute transfer workspace to the source stage so top-level peak and
        # per-stage breakdown stay rank-consistent.
        outgoing_transfer_peak = _stage_outgoing_transfer_peak(transfer_results)
        effective_peak = {
            result.stage_spec.stage_id: max(
                float(result.peak_memory_usage_bytes),
                float(outgoing_transfer_peak.get(result.stage_spec.stage_id, 0.0)),
            )
            for result in stage_results
        }
        summary_stage = max(
            stage_results,
            key=lambda result: effective_peak.get(result.stage_spec.stage_id, 0.0),
            default=None,
        )
        summary_stage_id = summary_stage.stage_spec.stage_id if summary_stage is not None else None
        summary_peak_bytes = effective_peak.get(summary_stage_id, 0.0) if summary_stage is not None else 0.0
        summary_cache_stats = summary_stage.cache_stats if summary_stage is not None else PipelineStageCacheStats()
        summary_activation_bytes = _stage_activation_memory_bytes(summary_stage)
        stage_latency_breakdown = _stage_latency_breakdown_entries(stage_results, transfers)
        stage_memory_breakdown = [
            _stage_memory_breakdown_entry(result, effective_peak.get(result.stage_spec.stage_id, 0.0))
            for result in stage_results
        ]
        perf_model_name = self.perf_models[0].name if self.perf_models else None
        return PipelineRunResult(
            logits=logits,
            execution_time_s=execution_time_s,
            table_result=_join_stage_tables(stage_results, transfer_results),
            breakdowns=breakdowns,
            runtime_event_list=(
                _aggregate_runtime_events(
                    stage_results,
                    perf_model_name,
                    lambda result: f"pp_stage_{result.stage_spec.stage_id}",
                )
                + _aggregate_runtime_events(
                    transfer_results,
                    perf_model_name,
                    lambda result: f"pp_comm_{result.stats.source_stage_id}_to_{result.stats.target_stage_id}",
                )
            ),
            peak_memory_usage_bytes=summary_peak_bytes,
            kv_cache_size_bytes=summary_cache_stats.kv_cache_bytes,
            kv_cache_per_token_bytes=summary_cache_stats.kv_cache_per_token_bytes,
            indexer_cache_size_bytes=summary_cache_stats.indexer_cache_bytes,
            indexer_cache_per_token_bytes=summary_cache_stats.indexer_cache_per_token_bytes,
            model_weight_size_bytes=summary_stage.weight_size_bytes if summary_stage is not None else 0,
            model_activation_size_bytes=summary_activation_bytes,
            stage_latency_breakdown=stage_latency_breakdown,
            stage_memory_breakdown=stage_memory_breakdown,
            trace_events=[
                trace_event for result in [*stage_results, *transfer_results] for trace_event in result.trace_events
            ],
        )

    @staticmethod
    def export_chrome_trace(path: str, trace_events: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"traceEvents": trace_events}, f)


def _stage_outgoing_comm_times(
    num_stages: int,
    transfers: list[PipelineTransferStats],
    *,
    model_name: Optional[str] = None,
) -> list[float]:
    outgoing = [0.0 for _ in range(num_stages)]
    for transfer in transfers:
        if 0 <= transfer.source_stage_id < num_stages:
            if model_name is None:
                outgoing[transfer.source_stage_id] += transfer.time_s
            else:
                outgoing[transfer.source_stage_id] += _transfer_time_s_for_stage_latency(transfer, model_name)
    return outgoing


def _stage_latency_breakdown_entries(
    stage_results: list[_PipelineStageRunResult],
    transfers: list[PipelineTransferStats],
) -> list[dict]:
    """Build per-stage compute/communication latency entries for user output."""
    if not stage_results:
        return []

    model_names: list[str] = []
    for result in stage_results:
        for model_name in result.execution_time_s:
            if model_name not in model_names:
                model_names.append(model_name)

    num_stages = max(result.stage_spec.stage_id for result in stage_results) + 1
    outgoing_by_model = {
        model_name: _stage_outgoing_comm_times(num_stages, transfers, model_name=model_name)
        for model_name in model_names
    }

    entries = []
    for result in stage_results:
        spec = result.stage_spec
        compute_time_s = {}
        outgoing_comm_time_s = {}
        total_time_s = {}
        for model_name in model_names:
            compute = float(result.execution_time_s.get(model_name, 0.0))
            outgoing = float(outgoing_by_model[model_name][spec.stage_id])
            compute_time_s[model_name] = compute
            outgoing_comm_time_s[model_name] = outgoing
            total_time_s[model_name] = compute + outgoing
        entries.append(
            {
                "stage_id": spec.stage_id,
                "layer_start": spec.layer_start,
                "layer_end": spec.layer_end,
                "compute_time_s": compute_time_s,
                "outgoing_comm_time_s": outgoing_comm_time_s,
                "total_time_s": total_time_s,
            }
        )
    return entries


def _pipeline_parallel_breakdowns(
    stage_execution_time_s: dict[str, list[float]],
    transfers: list[PipelineTransferStats],
) -> dict[str, dict[str, float]]:
    breakdowns: dict[str, dict[str, float]] = {}
    for model_name, stage_times in stage_execution_time_s.items():
        compute_s = sum(stage_times)
        communication_s = sum(_transfer_time_s(transfer, model_name) for transfer in transfers)
        total_time_s = _pipeline_total_time_s(stage_times, transfers, model_name=model_name)
        # TensorCast PP currently aggregates one model-forward simulation. It does
        # not model fill-drain, 1F1B, or overlap scheduling, so bubble is expected
        # to be 0 under this formula.
        bubble_s = max(0.0, total_time_s - compute_s - communication_s)
        breakdowns[f"{model_name}_pipeline_parallel"] = {
            "compute": compute_s,
            "communication": communication_s,
            "bubble": bubble_s,
        }
    return breakdowns


def _pipeline_total_time_s(
    stage_times: list[float],
    transfers: list[PipelineTransferStats],
    *,
    model_name: Optional[str] = None,
) -> float:
    if not stage_times:
        return 0.0
    return sum(stage_times) + sum(_transfer_time_s(transfer, model_name) for transfer in transfers)


def _transfer_time_s(transfer: PipelineTransferStats, model_name: Optional[str]) -> float:
    if model_name is not None and model_name in transfer.time_s_by_model:
        return transfer.time_s_by_model[model_name]
    return transfer.time_s


# Keep stage latency strict per performance model; aggregate transfer.time_s would look over-precise here.
def _transfer_time_s_for_stage_latency(transfer: PipelineTransferStats, model_name: str) -> float:
    if model_name in transfer.time_s_by_model:
        return transfer.time_s_by_model[model_name]
    logger.warning(
        "Pipeline transfer %d->%d has no time estimate for perf model %r; using 0.0 in stage latency breakdown.",
        transfer.source_stage_id,
        transfer.target_stage_id,
        model_name,
    )
    return 0.0


def _stage_activation_memory_bytes(result: Any) -> float:
    """Stage-local activation = peak - weight - kv - indexer, clamped to >= 0.

    All four terms come from the same stage, so the subtraction is physically
    meaningful (unlike the cross-stage ``peak - kv - weight`` in model_runner,
    which can mix different ranks and fabricate a non-existent rank).
    """
    if result is None:
        return 0.0
    cache_stats = result.cache_stats
    return max(
        0.0,
        float(result.peak_memory_usage_bytes)
        - float(result.weight_size_bytes)
        - float(cache_stats.kv_cache_bytes)
        - float(cache_stats.indexer_cache_bytes),
    )


def _stage_memory_breakdown_entry(result: Any, effective_peak_bytes: float = 0.0) -> dict:
    """Build a per-stage memory breakdown dict for the summary stage list.

    All fields come from a single stage, so the entry is a self-consistent
    per-rank memory snapshot. ``peak_bytes`` is the stage's effective peak
    (max of runtime peak and its outgoing transfer peak), so it matches the
    top-level peak when this stage is the summary. ``activation_bytes`` is
    decomposed from the runtime peak (not the transfer peak), because the
    transfer buffer is communication workspace, not stage-local activation.
    """
    if result is None:
        return {}
    spec = result.stage_spec
    cache_stats = result.cache_stats
    runtime_peak = float(result.peak_memory_usage_bytes)
    return {
        "stage_id": spec.stage_id,
        "layer_start": spec.layer_start,
        "layer_end": spec.layer_end,
        "weight_bytes": int(result.weight_size_bytes),
        "kv_cache_bytes": int(cache_stats.kv_cache_bytes),
        "indexer_cache_bytes": int(cache_stats.indexer_cache_bytes),
        "peak_bytes": int(max(runtime_peak, float(effective_peak_bytes))),
        "activation_bytes": int(_stage_activation_memory_bytes(result)),
    }


def _stage_outgoing_transfer_peak(transfer_results: list[Any]) -> dict[int, float]:
    """Map each source stage id to the max peak of transfers it originates.

    A transfer's send/recv buffer is owned by its source rank, so the transfer
    peak is attributed to the source stage for the purpose of selecting the
    summary stage and reporting its effective peak.
    """
    outgoing: dict[int, float] = {}
    for transfer in transfer_results:
        source_stage_id = transfer.stats.source_stage_id
        outgoing[source_stage_id] = max(outgoing.get(source_stage_id, 0.0), float(transfer.peak_memory_usage_bytes))
    return outgoing


def _validate_pipeline_device_grid(model_config: ModelConfig, device_profile: DeviceProfile) -> None:
    parallel_config = model_config.parallel_config
    world_size = int(parallel_config.world_size)
    device_capacity = int(device_profile.comm_grid.grid.numel())
    if world_size > device_capacity:
        raise ValueError(
            f"Pipeline parallel world_size ({world_size}) exceeds device grid capacity ({device_capacity}) "
            f"for device profile {device_profile.name}."
        )


def _merge_breakdowns(breakdowns_by_stage) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for breakdowns in breakdowns_by_stage:
        for breakdown_name, breakdown in breakdowns.items():
            entry = merged.setdefault(breakdown_name, {})
            for category, value in breakdown.items():
                entry[category] = entry.get(category, 0.0) + value
    return merged


def _join_stage_tables(
    stage_results: list[Any],
    transfer_results: list[Any],
) -> str:
    if not stage_results and not transfer_results:
        return "No events recorded."
    tables = [
        (
            f"[pipeline stage {result.stage_spec.stage_id} "
            f"layers {result.stage_spec.layer_start}:{result.stage_spec.layer_end}]\n"
            f"{result.table_result}"
        )
        for result in stage_results
    ]
    tables.extend(
        (
            f"[pipeline transfer {result.stats.source_stage_id}->{result.stats.target_stage_id} "
            f"bytes={result.stats.payload_bytes}]\n"
            f"{result.table_result}"
        )
        for result in transfer_results
    )
    return "\n\n".join(tables)


def _aggregate_runtime_events(
    runtime_results: list[Any],
    perf_model_name: Optional[str],
    name_prefix_for,
) -> list[dict]:
    aggregated: dict[str, dict[str, float]] = {}
    for runtime_result in runtime_results:
        name_prefix = name_prefix_for(runtime_result)
        for event in runtime_result.runtime_events:
            name = f"{name_prefix}:{event.op_invoke_info.func}"
            entry = aggregated.setdefault(name, {"total": 0.0, "count": 0})
            entry["count"] += 1
            if perf_model_name is None:
                continue
            result = event.perf_results.get(perf_model_name)
            if result is not None:
                entry["total"] += result.execution_time_s
    items = []
    for name, entry in aggregated.items():
        count = entry["count"]
        total = entry["total"]
        items.append(
            {
                "name": name,
                "perf_model": perf_model_name,
                "perf_total": total,
                "perf_avg": total / count if count else 0.0,
                "call_times": count,
            }
        )
    items.sort(key=lambda x: x["perf_total"], reverse=True)
    return items


def _tag_trace_events(
    trace_events: list[dict],
    *,
    name_prefix: str,
    metadata: dict[str, Any],
) -> list[dict]:
    tagged_events = []
    for trace_event in trace_events:
        tagged_event = dict(trace_event)
        tagged_event["name"] = f"{name_prefix}:{tagged_event.get('name', '')}"
        tagged_event["args"] = dict(tagged_event.get("args", {}))
        tagged_event["args"].update(metadata)
        tagged_events.append(tagged_event)
    return tagged_events


def _safe_peak_memory_usage(memory_tracker: MemoryTracker) -> float:
    try:
        return float(memory_tracker.peak_mem_usage())
    except ValueError:
        return 0.0


__all__ = [
    "PipelineCommunicator",
    "PipelineModel",
    "PPMissingLayer",
    "PipelinePlan",
    "PipelineRunResult",
    "PipelineRunner",
    "PipelineStageCacheStats",
    "PipelineStageModel",
    "PipelineStageSpec",
    "PipelineTransferStats",
    "StageRunner",
    "apply_stage_boundaries",
    "build_pipeline_plan",
    "build_pipeline_stage_kwargs",
    "build_stage_model_config",
    "build_stage_parallel_config",
]

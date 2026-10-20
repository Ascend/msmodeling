"""TensorCast wrapper entry point for DeepSeek V4.1 attention.

Phase 1 reuses the proven V4 projection/attention implementation.  V4.1
specific CSA2 state and cache ownership are introduced in Phase 2.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

from .deepseek_v4 import DeepseekV4SparseAttention


@dataclass(frozen=True)
class CSA2LayerRole:
    """Static ownership and execution role for one V4.1 backbone layer."""

    layer_id: int
    compress_ratio: int
    role: str
    compressed_kv_owner: Optional[int] = None
    index_k_owner: Optional[int] = None
    topk_owner: Optional[int] = None
    produces_candidate_pool: bool = False

    @property
    def owns_compressed_kv(self) -> bool:
        return self.compressed_kv_owner == self.layer_id

    @property
    def owns_index_k(self) -> bool:
        return self.index_k_owner == self.layer_id

    @property
    def produces_topk(self) -> bool:
        return self.topk_owner == self.layer_id


def resolve_csa2_layer_role(config, layer_id: int) -> CSA2LayerRole:
    """Resolve role from absolute layer ids and physical owner lists."""

    layer_id = int(layer_id)
    ratios = tuple(getattr(config, "compress_ratios", ()))
    if layer_id < 0 or layer_id >= len(ratios):
        raise ValueError(f"CSA2 layer id {layer_id} is outside active backbone")
    ratio = int(ratios[layer_id])
    kv_sources = tuple(getattr(config, "kv_source_layer_ids", ()))
    index_sources = tuple(getattr(config, "index_source_layer_ids", ()))
    candidate_source = getattr(config, "candidate_source_layer_id", None)
    # Keep this expression compiler-friendly.  Dynamo's builtin ``max``
    # handler does not accept the Python ``default=`` keyword during tracing.
    prior_kv_sources = tuple(x for x in kv_sources if x < layer_id)
    kv_owner = layer_id if layer_id in kv_sources else (max(prior_kv_sources) if prior_kv_sources else None)
    if ratio == 0:
        return CSA2LayerRole(layer_id, ratio, "window")
    if layer_id in kv_sources:
        role = "source"
    elif layer_id in index_sources:
        role = "reindex" if ratio == 1 and layer_id != candidate_source else "index_source"
    else:
        role = "reuse"
    prior_index_sources = tuple(x for x in index_sources if x <= layer_id)
    index_source = max(prior_index_sources) if prior_index_sources else None
    return CSA2LayerRole(
        layer_id,
        ratio,
        role,
        compressed_kv_owner=kv_owner,
        index_k_owner=kv_owner,
        # Reuse layers consume the top-k published by their physical source;
        # reindex layers own a new top-k stream at their own absolute id.
        topk_owner=index_source,
        produces_candidate_pool=layer_id == candidate_source,
    )


@dataclass
class CompressorTailState:
    pending_tokens: Optional[torch.Tensor] = None
    pending_scores: Optional[torch.Tensor] = None


@dataclass
class EngramHashState:
    """Request-local compressed-token history for 2/3/4-gram hashing."""

    history: Optional[torch.Tensor] = None

    def update(self, history: torch.Tensor) -> None:
        self.history = history


@dataclass
class DeepseekV41SharedAttentionState:
    """Per-request CSA2 state; never shared globally between forwards."""

    compressed_kv_by_source: dict[int, torch.Tensor] = field(default_factory=dict)
    compressed_kv_scale_by_source: dict[int, torch.Tensor] = field(default_factory=dict)
    index_k_by_source: dict[int, torch.Tensor] = field(default_factory=dict)
    index_k_scale_by_source: dict[int, torch.Tensor] = field(default_factory=dict)
    topk_indices_by_source: dict[int, torch.Tensor] = field(default_factory=dict)
    candidate_mask: Optional[torch.Tensor] = None
    compressor_tail_by_source: dict[object, CompressorTailState] = field(default_factory=dict)
    compressed_slot_by_source: dict[int, torch.Tensor] = field(default_factory=dict)
    draft_kv_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)

    def reset(self) -> None:
        self.compressed_kv_by_source.clear()
        self.compressed_kv_scale_by_source.clear()
        self.index_k_by_source.clear()
        self.index_k_scale_by_source.clear()
        self.topk_indices_by_source.clear()
        self.candidate_mask = None
        self.compressor_tail_by_source.clear()
        self.compressed_slot_by_source.clear()
        self.draft_kv_by_layer.clear()

    def publish_compressed_kv(self, source_layer_id: int, value: torch.Tensor) -> None:
        source_layer_id = int(source_layer_id)
        previous = self.compressed_kv_by_source.get(source_layer_id)
        if previous is None or previous is value:
            self.compressed_kv_by_source[source_layer_id] = value
            return
        if isinstance(previous, torch.Tensor) and isinstance(value, torch.Tensor):
            if previous.is_meta or value.is_meta:
                self.compressed_kv_by_source[source_layer_id] = torch.empty(
                    previous.shape[0],
                    previous.shape[1] + value.shape[1],
                    previous.shape[2],
                    dtype=value.dtype,
                    device=value.device,
                )
            elif previous.ndim == value.ndim == 3 and previous.shape[0] == value.shape[0]:
                self.compressed_kv_by_source[source_layer_id] = torch.cat((previous, value), dim=1)
            else:
                self.compressed_kv_by_source[source_layer_id] = value
        else:
            self.compressed_kv_by_source[source_layer_id] = value

    def publish_compressed_slot_mapping(self, source_layer_id: int, value: torch.Tensor) -> None:
        self.compressed_slot_by_source[int(source_layer_id)] = value

    def require_compressed_slot_mapping(self, source_layer_id: int) -> torch.Tensor:
        try:
            return self.compressed_slot_by_source[int(source_layer_id)]
        except KeyError as exc:
            raise RuntimeError(f"CSA2 compressed slot mapping for owner L{source_layer_id} is missing") from exc

    def publish_compressed_kv_scale(self, source_layer_id: int, value: torch.Tensor) -> None:
        source_layer_id = int(source_layer_id)
        previous = self.compressed_kv_scale_by_source.get(source_layer_id)
        if previous is None:
            self.compressed_kv_scale_by_source[source_layer_id] = value
        elif previous.ndim == value.ndim == 3 and previous.shape[0] == value.shape[0]:
            self.compressed_kv_scale_by_source[source_layer_id] = torch.cat((previous, value), dim=1)
        else:
            self.compressed_kv_scale_by_source[source_layer_id] = value

    def get_compressor_tail(self, stream_id: object) -> CompressorTailState:
        return self.compressor_tail_by_source.setdefault(stream_id, CompressorTailState())

    def update_compressor_tail(
        self,
        stream_id: object,
        pending_tokens: torch.Tensor,
        pending_scores: torch.Tensor,
    ) -> None:
        self.compressor_tail_by_source[stream_id] = CompressorTailState(pending_tokens, pending_scores)

    def require_compressed_kv(self, owner_layer_id: int) -> torch.Tensor:
        try:
            return self.compressed_kv_by_source[int(owner_layer_id)]
        except KeyError as exc:
            raise RuntimeError(f"CSA2 compressed KV owner L{owner_layer_id} has not published state") from exc

    def publish_topk(self, source_layer_id: int, value: torch.Tensor) -> None:
        self.topk_indices_by_source[int(source_layer_id)] = value

    def require_topk(self, owner_layer_id: int) -> torch.Tensor:
        try:
            return self.topk_indices_by_source[int(owner_layer_id)]
        except KeyError as exc:
            raise RuntimeError(f"CSA2 top-k owner L{owner_layer_id} has not published state") from exc


class DeepseekV41SparseAttention(DeepseekV4SparseAttention):
    """V4.1-named wrapper retaining the Phase 1 V4 operator path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inner = getattr(self, "_inner", None)
        if inner is None:
            return
        self.csa2_role = getattr(inner, "csa2_role", None)
        self.kv_owner_layer_id = self.csa2_role.compressed_kv_owner if self.csa2_role is not None else None
        self.produces_candidate_pool = bool(self.csa2_role is not None and self.csa2_role.produces_candidate_pool)
        # Index-source layers use the shared stream produced by their KV owner;
        # they must not emit another main compressor write.
        if self.indexer is not None and self.csa2_role is not None:
            # V4.1 index keys derive from the source layer's published
            # compressed latent. The main attention path owns the only
            # compressor invocation; indexer must not emit a second one.
            self.indexer.compress_ratio = 0

    def forward(self, *args, **kwargs):
        state = kwargs.pop("csa2_state", None)
        role = self.csa2_role
        if state is not None and role is not None and role.compressed_kv_owner not in (None, role.layer_id):
            state.require_compressed_kv(role.compressed_kv_owner)
        if state is not None:
            kwargs["csa2_state"] = state
        result = super().forward(*args, **kwargs)
        return result


def deepseek_v41_kv_owner(config, layer_id: int) -> Optional[int]:
    """Return the physical compressed-KV owner for an absolute layer id."""

    role = resolve_csa2_layer_role(config, layer_id)
    return role.compressed_kv_owner

import dataclasses
from typing import Optional, TYPE_CHECKING

import torch

from .. import ops  # noqa: F401
from ..parallel_group import _DEFAULT_PG, ParallelGroup
from ..performance_model import _PREDICTIVE_DECODING_THRESHOLD

if TYPE_CHECKING:
    from ..model_config import AttentionQuantConfig


def is_dcp_decode_batch(seq_lens: Optional[torch.Tensor], query_lens: Optional[torch.Tensor]) -> bool:
    """Return True when this batch is a genuine DCP-eligible decode step.

    Decode Context Parallel is a decode-only optimization, so it must fire on
    ordinary decode *and* MTP/speculative decode (``query_len = 1 + num_spec``)
    while staying off for prefill. We mirror the cost model's decode rule
    (``query_len < _PREDICTIVE_DECODING_THRESHOLD``) rather than the stricter
    ``query_len == 1`` proxy, and additionally require every request to already
    hold KV context (``seq_len > query_len``) so a short *prefill*
    (``seq_len == query_len``) is never mistaken for decode and never gets its
    ``seq_lens`` floored toward zero.

    Scheduling assumption -- **DCP is enabled only for whole-batch decode steps.**
    The ``.all()`` reductions mean a *mixed* prefill+decode batch (a prefill chunk
    fused with decode requests, as vllm-ascend chunked-prefill / continuous
    batching produces) fails this predicate and turns DCP off for the entire
    batch, including its decode requests. This is intentional and matches how
    MsModeling scores DCP: the Prefill and Decode paths are modeled as *separate*
    steps (the throughput optimizer emits prefill-only and decode-only configs,
    with ``is_prefill`` forcing ``dcp=1`` on the Prefill phase), so benefit is
    measured on pure-decode steps and no per-request decode mask is needed.
    Refining this to a request-granular mask is a deferred non-goal
    (rfc_context_parallel_dcp §1.3 #8).
    """
    if seq_lens is None or query_lens is None:
        return False
    return bool((query_lens < _PREDICTIVE_DECODING_THRESHOLD).all() and (seq_lens > query_lens).all())


# adapted from vLLM but trimmed to avoid redundancy
@dataclasses.dataclass
class AttentionMetadataBase:
    """Per-layer attention metadata"""

    query_start_loc: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""

    query_lens: torch.Tensor
    """(batch_size,), the actual query length of each request"""

    block_table_tensor: Optional[torch.Tensor] = None
    """(batch_size, max_blocks_per_seq)"""
    slot_mapping: Optional[torch.Tensor] = None
    """(num_tokens,) The indices of the token slots that input tokens will be
    stored into."""

    seq_lens_values: Optional[list[int]] = None
    """Materialized sequence lengths retained when tensor values become Fake/Meta."""

    query_lens_values: Optional[list[int]] = None
    """Materialized query lengths retained when tensor values become Fake/Meta."""

    is_decode_values: Optional[list[bool]] = None
    """Per-request phase retained for model-specific decode/prefill Roofline paths."""

    max_total_seq_len: Optional[int] = None
    """Python scalar equal to ``max(seq_lens)`` for this metadata batch.

    This duplicates ``seq_lens.max()`` intentionally so compile-time callers can
    size rectangular helper tensors without materializing a tensor scalar via
    ``.item()``.
    """

    is_dcp_decode: bool = dataclasses.field(init=False, default=False)
    """Whether this batch is a DCP-eligible decode step. Resolved ONCE here on
    the host (``__post_init__``) so the ``torch.compile``-traced attention
    forwards can branch on a Python bool instead of reducing ``seq_lens`` /
    ``query_lens`` to a scalar -- the latter is data-dependent control flow that
    Dynamo cannot trace."""

    def __post_init__(self):
        # Metadata is always built outside the compiled region (see
        # ``core.input_generator``), so this reduction is safe to evaluate eagerly
        # and the result rides into the graph as a traceable constant.
        self.is_dcp_decode = is_dcp_decode_batch(self.seq_lens, self.query_lens)


class AttentionBase(torch.nn.Module):
    attn_implmentation = None

    def __init__(self):
        super().__init__()
        self.quant_config: Optional[AttentionQuantConfig] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward().")


# adapted from vLLM
def flash_attention_forward(
    # Transformers args
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    **kwargs,
):
    attention_by_layers: Optional[dict[int, AttentionBase]] = kwargs.pop("attention_by_layers", None)
    is_vision_attention = False
    _tensor_cast_context = getattr(module, "_tensor_cast_context", None)
    if _tensor_cast_context is not None and not hasattr(module, "layer_idx"):
        is_vision_attention = True
        if attention_by_layers is None:
            attention_by_layers = _tensor_cast_context.get("attention_by_layers")
    elif attention_by_layers is None:
        # For VL models, the visual layer's attention_by_layers cannot be obtained from kwargs,
        # so it is retrieved from the module's _tensor_cast_context instead.
        is_vision_attention = True
        if _tensor_cast_context is not None:
            attention_by_layers = _tensor_cast_context.get("attention_by_layers")

    assert attention_by_layers is not None, "Expect attention_by_layers to be provided."
    if is_vision_attention:
        assert _tensor_cast_context is not None, "Expect _tensor_cast_context to be provided."
        depth_layer_idx = _tensor_cast_context.get("depth_layer_idx")
        self_attn = attention_by_layers[depth_layer_idx]
        kv_cache = None
        attention_meta = None
        kwargs.pop("attention_meta", None)
        kwargs.pop("attention_meta_by_layers", None)
        kwargs.pop("kv_cache_by_layers", None)
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        num_tokens = query.shape[0] * query.shape[1]
        # For subsequent time calculation, the key and value do not need to be reshaped
        query = query.reshape(num_tokens, -1)
    else:
        kv_cache_by_layers: Optional[dict[int, torch.Tensor]] = kwargs.pop("kv_cache_by_layers", None)
        attention_meta: AttentionMetadataBase = kwargs.pop("attention_meta", None)
        attention_meta_by_layers: Optional[dict[int, AttentionMetadataBase]] = kwargs.pop(
            "attention_meta_by_layers", None
        )
        assert attention_meta is None or attention_meta_by_layers is None, (
            "Only one of attention_meta and attention_meta_by_layers can be provided."
        )

        self_attn = attention_by_layers[module.layer_idx]
        kv_cache = kv_cache_by_layers[module.layer_idx] if kv_cache_by_layers else None
        attention_meta = attention_meta_by_layers[module.layer_idx] if attention_meta_by_layers else attention_meta
        # TODO: understand why we need these shape manipulation
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        num_tokens = query.shape[0] * query.shape[1]
        query, key, value = (x.reshape(num_tokens, -1) for x in (query, key, value))
    # return (attn_output, attn_weights) while we ignore attn_weights
    return self_attn.forward(
        query,
        key,
        value,
        attention_mask,
        kv_cache=kv_cache,
        attention_meta=attention_meta,
        **kwargs,
    ), None


class AttentionMetadataTensorCast(AttentionMetadataBase):
    pass


class AttentionTensorCast(AttentionBase):
    attn_implmentation = "tensor_cast"

    # Decode Context Parallel group, populated by ``patch_attention`` for text
    # attention layers. Defaults to the singleton no-op group so vision attention
    # and any unpatched layer leave the decode path untouched.
    dcp_group: ParallelGroup = _DEFAULT_PG

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> torch.Tensor:
        query_start_loc = attention_meta.query_start_loc if attention_meta else None
        seq_lens = attention_meta.seq_lens if attention_meta else None
        query_lens = attention_meta.query_lens if attention_meta else None

        # --- Decode Context Parallel (decode path only) ---
        # For GQA, DCP re-partitions KV heads across the (shared) TP communication
        # domain: each rank moves from h_kv/tp heads over the full context to
        # h_kv*dcp/tp heads over S/dcp. Per-rank attention compute AND KV read are
        # therefore INVARIANT under DCP (the head growth cancels the sequence
        # shrink), so we keep the original query / seq_lens for the local attention
        # op and let the two DCP collectives carry only their communication cost.
        # Sharding seq_lens while leaving the KV-head count unchanged (the cost
        # model reads ``key.size(-2)``, which DCP does not grow) would under-count
        # the KV read by dcp and bias the search toward DCP.
        apply_dcp = (
            getattr(self, "dcp_group", _DEFAULT_PG).world_size > 1
            and query.ndim == 2
            and attention_meta is not None
            and attention_meta.is_dcp_decode
        )
        gathered_query = None
        if apply_dcp:
            # Model the all_gather Q communication (head dim) for its cost only; the
            # gathered tensor is not consumed because the local work is DCP-invariant.
            # We still keep a handle: under torch.compile, AOT Autograd DCEs any
            # collective whose output is dead, so it is bound (value-neutrally) into
            # attn_output below to survive (mirrors deepseek_v4's KV-producer binding).
            gathered_query = self.dcp_group.all_gather(query, dim=-1)

        if attention_meta is not None:
            if self.quant_config is not None:
                kv_scale = self.quant_config.kv_scale
                kv_offset = self.quant_config.kv_offset
                key = torch.ops.tensor_cast.quantize(key, kv_scale, kv_offset, kv_cache.dtype)
                value = torch.ops.tensor_cast.quantize(value, kv_scale, kv_offset, kv_cache.dtype)
            if not (key.dtype == value.dtype == kv_cache.dtype):
                raise ValueError(
                    f"Expect key, value and kv_cache dtype match but got {key.dtype}, {value.dtype}, {kv_cache.dtype}"
                )
            torch.ops.tensor_cast.reshape_and_cache(key, value, kv_cache, attention_meta.slot_mapping)
            key = kv_cache[0]
            value = kv_cache[1]
        if self.quant_config is not None and attention_meta is not None:
            out_dtype = query.dtype
            query = torch.ops.tensor_cast.quantize(
                query,
                self.quant_config.query_scale,
                self.quant_config.query_offset,
                kv_cache.dtype,
            )
            attn_output = torch.ops.tensor_cast.attention_quant(
                query,
                key,
                value,
                attention_mask,
                attention_meta.block_table_tensor if attention_meta is not None else None,
                query_start_loc,
                seq_lens,
                query_lens,
                self.quant_config.query_scale,
                self.quant_config.query_offset,
                self.quant_config.kv_scale,
                self.quant_config.kv_offset,
                self.quant_config.attention_prob_scale,
                self.quant_config.attention_prob_offset,
                out_dtype,
            )
        else:
            attn_output = torch.ops.tensor_cast.attention(
                query,
                key,
                value,
                attention_mask,
                attention_meta.block_table_tensor if attention_meta is not None else None,
                query_start_loc,
                seq_lens,
                query_lens,
            )

        if apply_dcp:
            # head_dim is the last KV-cache dim (``key`` is ``kv_cache[0]`` here);
            # used to count the per-head lse column in the merge a2a.
            head_dim = key.size(-1)
            out_lse = self._dcp_merge_all_to_all(attn_output, head_dim)
            # Bind both DCP collectives into the returned tensor with a value-neutral
            # (multiply-by-zero) edge so torch.compile's DCE keeps them in the graph;
            # the local work is DCP-invariant, so this must not perturb attn_output.
            # Each ``[0] * 0`` term is added directly onto the full tensor (never summed
            # with another scalar first) so it stays a broadcast onto ``attn_output``
            # rather than a pure 0-dim subgraph the constant-folder would evaluate.
            attn_output = attn_output + gathered_query.reshape(-1)[0].to(attn_output.dtype) * 0
            attn_output = attn_output + out_lse.reshape(-1)[0].to(attn_output.dtype) * 0
        return attn_output

    def _dcp_merge_all_to_all(self, attn_output: torch.Tensor, head_dim: int) -> torch.Tensor:
        """Model the DCP ``output + lse`` all_to_all (communication cost only).

        ``attn_output`` is the rank-local ``(num_tokens, h_q/tp * head_dim)`` decode
        output (DCP-invariant, see ``forward``). The collective being modeled is the
        redistribution of the *gathered* output+lse: each of the ``dcp`` ranks gathered
        to ``h_q*dcp/tp`` heads, so the payload carries ``gathered_heads`` rows of
        ``num_tokens * (head_dim + 1)`` columns -- the ``+1`` is the per-head ``lse``
        column. vllm-ascend upcasts output & lse to fp32 before the a2a, so the volume
        is counted in fp32 (4 bytes/elem) via the tensor dtype, independent of the
        model/KV dtype. This mirrors the MLA merge (``mla._dcp_merge_all_to_all``) so
        the two backends model the same per-layer communication volume. The local
        online-softmax merge does not change ``attn_output``'s width; the exchanged
        ``output + lse`` payload is returned only so ``forward`` can bind it (value-
        neutrally) into ``attn_output`` and keep the a2a alive under torch.compile DCE.
        """
        dcp = self.dcp_group.world_size
        num_tokens = attn_output.shape[0]
        heads_per_rank = attn_output.shape[-1] // head_dim
        gathered_heads = heads_per_rank * dcp
        # output (head_dim) + lse (1 column per head), upcast to fp32 for the a2a.
        # The payload must carry a value edge from the placeholder-derived
        # ``attn_output``: a bare ``torch.empty`` is an all-constant input, so the meta
        # constant-folder would precompute the whole ``empty -> all_to_all`` chain and
        # erase the collective (dropping its communication cost). Adding a value-neutral
        # ``[0] * 0`` term keeps the a2a *input* non-constant so it survives folding.
        out_lse = torch.empty(
            (gathered_heads, num_tokens * (head_dim + 1)),
            dtype=torch.float32,
            device=attn_output.device,
        )
        out_lse = out_lse + attn_output.reshape(-1)[0].to(torch.float32) * 0
        split_sizes = [heads_per_rank] * dcp
        return self.dcp_group.all_to_all(out_lse, split_sizes, split_sizes)

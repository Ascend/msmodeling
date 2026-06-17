# Copyright (C) 2025 HuggingFace Inc. team.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import Optional

import torch
import torch.nn.functional as F

from ..model_config import MlaConfig
from ..parallel_group import ParallelGroup, ParallelGroupManager
from ..utils import exact_division
from .attention import AttentionMetadataBase
from . import COLWISE_LINEAR
from .mla import (
    DeepseekSparseAttention,
    MultiheadLatentAttentionTensorCast,
    _resolve_sparse_topk_limit,
)
from .mtp import MultiTokenPredictorLayer
from .quant_linear import TensorCastQuantLinear
from .utils import apply_static_quant_linear, ModelWrapperBase


# ============================================================
# V4 MoE gate routing helpers
# ============================================================


def has_deepseek_v4_hash_routing(gate: torch.nn.Module, moe_layer_idx: Optional[int]) -> bool:
    gate_hash = getattr(gate, "hash", None)
    if gate_hash is not None:
        return bool(gate_hash)
    return moe_layer_idx is not None and moe_layer_idx < 3


def compute_v4_gate_scores(
    gate: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, float, bool]:
    gate_weight = _extract_gate_weight(gate)
    score_func = str(getattr(gate, "score_func", "sqrtsoftplus"))
    route_scale = float(getattr(gate, "route_scale", 1.0))
    scores = F.linear(hidden_states.float(), gate_weight.float())
    if score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    return scores, route_scale, score_func != "softmax"


def route_deepseek_v4_gate(
    gate: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k: int,
    input_ids: Optional[torch.Tensor] = None,
    moe_layer_idx: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores, route_scale, normalize_weights = compute_v4_gate_scores(gate, hidden_states)
    use_hash_routing = has_deepseek_v4_hash_routing(gate, moe_layer_idx)
    topk_weights, topk_indices = route_v4_gate_tail(
        gate,
        top_k,
        use_hash_routing,
        scores,
        normalize_weights,
        route_scale,
        input_ids,
    )
    return topk_indices, topk_weights.to(hidden_states.dtype)


def route_v4_gate_tail(
    gate: torch.nn.Module,
    top_k: int,
    hash_routing: bool,
    scores: torch.Tensor,
    normalize_weights: bool,
    route_scale: float,
    input_ids: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if hash_routing:
        if input_ids is None:
            raise ValueError("DeepSeek V4 hash routing requires input_ids")
        tid2eid = _extract_tid2eid(gate)
        if tid2eid is None:
            raise ValueError("DeepSeek V4 hash routing requires gate.tid2eid")
        topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k_hash(
            scores,
            top_k,
            normalize_weights,
            route_scale,
            input_ids,
            tid2eid,
        )
    else:
        bias = _extract_gate_bias(gate)
        topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k(
            scores,
            top_k,
            normalize_weights,
            route_scale,
            bias,
        )
    gate_dtype = gate.weight.dtype if hasattr(gate, "weight") and hasattr(gate.weight, "dtype") else scores.dtype
    return topk_weights.to(gate_dtype), topk_indices


def _extract_gate_weight(gate: torch.nn.Module) -> torch.Tensor:
    weight = getattr(gate, "weight", None)
    if weight is not None:
        return weight.data if hasattr(weight, "data") else weight
    for param in gate.parameters(recurse=True):
        if param.ndim == 2:
            return param.data
    raise AttributeError("Hash-routing MoE gate must expose a 2D weight tensor for cost modeling")


def _extract_tid2eid(gate: torch.nn.Module) -> torch.Tensor | None:
    tid2eid = getattr(gate, "tid2eid", None)
    if tid2eid is None:
        return None
    return tid2eid.data if hasattr(tid2eid, "data") else tid2eid


def _extract_gate_bias(gate: torch.nn.Module) -> torch.Tensor | None:
    bias = getattr(gate, "bias", None)
    if bias is None:
        bias = getattr(gate, "e_score_correction_bias", None)
    if bias is None:
        return None
    return bias.data if hasattr(bias, "data") else bias


# ============================================================
# V4 MLA attention helpers
# ============================================================


def get_window_topk_idxs(
    window_size: int,
    batch_size: int,
    seq_length: int,
    device: torch.device,
    is_decode: bool = False,
) -> torch.Tensor:
    W = int(window_size)
    sl = int(seq_length)
    width = W if is_decode else min(sl, W)
    return torch.arange(width, device=device, dtype=torch.long).view(1, 1, -1).expand(int(batch_size), sl, -1)


def get_compress_topk_idxs(
    ratio: int,
    batch_size: int,
    seq_length: int,
    device: torch.device,
) -> torch.Tensor:
    R = int(ratio)
    sl = int(seq_length)
    width = max(sl // R, 1)
    return torch.arange(width, device=device, dtype=torch.long).view(1, 1, -1).expand(int(batch_size), sl, -1)


def _is_decode_attention_batch(
    seq_length: int,
    attention_meta: Optional[AttentionMetadataBase],
    batch_size: int = 1,
) -> bool:
    effective_query_length = int(seq_length)
    query_lens = attention_meta.query_lens if attention_meta is not None else None
    if query_lens is not None:
        request_count = int(query_lens.shape[0])
        if int(batch_size) == 1 and request_count > 1 and effective_query_length % request_count == 0:
            effective_query_length = effective_query_length // request_count
    # Keep aligned with performance_model's predictive-decoding rule:
    # query length < 5 is treated as decode, covering one sampled token plus MTP tokens.
    return effective_query_length < 5


# ============================================================
# V4 attention wrapper
# ============================================================


class DeepseekV4SparseAttention(DeepseekSparseAttention):
    """V4 sparse attention wrapper (covers Flash and Pro).

    The cost-modeling forward here mirrors
    `deepseek-ai/DeepSeek-V4-Flash/inference/model.py:Attention.forward` step-by-step.
    For every layer it preserves the same structural stages:

        1. ``q = q_norm(wq_a(x))``
        2. ``q = wq_b(q).view(...)`` followed by explicit per-head
           ``rsqrt(mean(q^2)+eps)`` normalization
        3. ``apply_rotary_emb(q[..., -rd:])`` on the Q branch
        4. ``kv = kv_norm(wkv(x))`` over the FULL shared `head_dim`
        5. ``apply_rotary_emb(kv[..., -rd:])`` on the shared-KV branch
        6. Window top-k index materialization (every layer): inlined
           host-side arange/clamp/where in `forward`. Prefill is op-for-op
           with `model.py:255 get_window_topk_idxs`; decode uses a single
           arange of the same shape and value range (cost-equivalent).
        7. (ratio==4 only) Indexer wrapper, which internally runs the
           indexer-local `compressor` (head_dim = `index_head_dim`) as its
           first stage and then `quant_lightning_indexer` for the learned
           compressed top-k — mirroring the reference `Indexer.forward`
        8. (ratio==128 only) Compress top-k index materialization (inlined
           in `forward`; prefill mirrors `model.py:269 get_compress_topk_idxs`
           op-for-op, decode uses an `arange + offset` matching the
           reference if-branch)
        9. ``cat(window_topk, compress_topk).int()`` (skipped on ratio==0)
       10. `scatter_nd_update_mla` writes the post-RoPE `kv_window_entry`
           into the sliding-window cache and returns a functional handle
       11. (ratio>0 only) `compressor` for the shared coarse-grain KV
       12. `sparse_attn_sharedkv` over [window | compressed] memory
       13. ``apply_rotary_emb(o[..., -rd:], inverse=True)`` on the output
       14. Group-wise ``einsum("bsgd,grd->bsgr", o, wo_a)`` followed by
           ``wo_b(o.flatten(2))``

    Compared to V3.2 (which fuses RMS+matmul+rope into `mlapo`/`mlapo_quant`
    and writes KV via `concat_and_cache_mla`), V4 (Flash/Pro) deliberately keeps
    these stages as separate ops on NPU. This wrapper therefore also
    emits them as separate ops and DOES NOT call `mlapo` or
    `concat_and_cache_mla`. The sliding-window KV write from
    `model.py:520-533` is preserved here as `scatter_nd_update_mla`: the post-
    RoPE `kv_window_entry` is written into `kv_cache`, and the op returns a
    functional handle to the updated cache that is then fed into
    `sparse_attn_sharedkv`. This builds an explicit
    `wkv -> kv_norm -> KV-RoPE -> cat -> scatter -> sparse_attn` data edge,
    matching the reference's `self.kv_cache[:bsz] = kv; o = sparse_attn_sharedkv(
    q, self.kv_cache, ...)` pattern, and ensuring the full KV branch
    (including KV-RoPE and the cache write) is accounted for in the modeled
    runtime cost.

    Layer-policy specifics:
        - ratio == 0   (layers 0/1)         : window-only, no Compressor/Indexer
        - ratio == 4   (even layers >= 2)    : window + Compressor + Indexer
        - ratio == 128 (odd layers >= 3)     : window + Compressor only

    Critically, ratio=4 layers issue TWO `compressor` ops — one for the
    indexer-local KV cache (head_dim = `index_head_dim`, e.g. 128) and one
    for the shared coarse-grain KV stream (head_dim = `head_dim`, e.g. 512).
    """

    def _setup_kv_b_decomposition(self, tp_group: ParallelGroup) -> None:
        # V4 has no kv_b_proj (shared KV path); skip the legacy decomposition.
        return None

    def _quantize_kv_b_decomposition(self) -> None:
        # V4 has no kv_b_proj decomposition tensors to quantize.
        return None

    @staticmethod
    def _local_linear_out_features(module: torch.nn.Module) -> int:
        # Read the post-shard output width directly off the wrapped projection.
        # `ColumnParallelLinear.create_weights` rewrites `_inner.out_features`
        # to the per-rank shard width; the wrapper itself also exposes the
        # same value via `out_features_per_partition`. Both paths are honored
        # so this works whether we receive the wrapper or the bare quant
        # linear.
        out_features = getattr(module, "out_features_per_partition", None)
        if out_features is None:
            out_features = getattr(module, "out_features", None)
        if out_features is None and hasattr(module, "_inner"):
            out_features = getattr(module._inner, "out_features", None)
        if out_features is None:
            raise AttributeError(f"Unable to resolve local out_features from {type(module).__name__}")
        return int(out_features)

    @staticmethod
    def _extract_logical_linear_weight(module: torch.nn.Module) -> torch.Tensor:
        target = module._inner if hasattr(module, "_inner") else module
        if isinstance(target, TensorCastQuantLinear) and target.quant_config.quant_type.name == "W4A8":
            packed_weight = target.qweight
            high_bits = (packed_weight >> 4).to(torch.int8) - 8
            low_bits = (packed_weight & 0x0F).to(torch.int8) - 8
            unpacked = torch.empty(
                packed_weight.shape[0],
                packed_weight.shape[1] * 2,
                dtype=torch.int8,
                device=packed_weight.device,
            )
            unpacked[:, ::2] = high_bits
            unpacked[:, 1::2] = low_bits
            return unpacked
        weight, _, _ = MultiheadLatentAttentionTensorCast.extract_qparams(module)
        return weight

    @classmethod
    def build_tp_plan_extras(cls, prefix: str, params: dict, config_info) -> dict[str, tuple[str, dict]]:
        from .mla import tp_plan_module_path

        return {
            tp_plan_module_path(prefix, "self_attn.indexer.wq_b"): (COLWISE_LINEAR, dict(params)),
            tp_plan_module_path(prefix, "self_attn.indexer.weights_proj"): (
                COLWISE_LINEAR,
                {
                    **dict(params),
                    "head_num": getattr(config_info, "index_n_heads"),
                },
            ),
        }

    @classmethod
    def build_o_proj_tp_plan_extras(cls, prefix: str, params: dict, config_info) -> dict[str, tuple[str, dict]]:
        from .mla import tp_plan_module_path

        return {
            tp_plan_module_path(prefix, "self_attn.wo_a"): (COLWISE_LINEAR, {**dict(params), "dim": 1}),
        }

    def __init__(
        self,
        mla_config: MlaConfig,
        mla_module: torch.nn.Module,
        tp_group: ParallelGroup,
        decode_only: bool = False,
        parallel_group_manager: Optional["ParallelGroupManager"] = None,
    ):
        MultiheadLatentAttentionTensorCast.__init__(self, mla_config, mla_module, tp_group, decode_only)
        self.compress_ratio = getattr(self._inner, "compress_ratio", 0)
        self.use_indexer = bool(getattr(self._inner, "use_indexer", False))
        self.use_compressor = bool(getattr(self._inner, "use_compressor", False))
        self.hc_mult = getattr(getattr(self._inner, "config", None), "hc_mult", 1)
        # Sliding-window size: reference model.py:452 stores it per-layer as
        # `self.window_size = args.window_size` (global, default 128). We mirror
        # that resolution order — inner attribute first, then inner.config, then
        # the V4 default of 128 — so ratio=0 layers correctly attend over
        # just the last `window_size` KV entries, and ratio>0 layers attend over
        # window + compressed-cache tails.
        inner_config = getattr(self._inner, "config", None)
        self.window_size = int(
            getattr(self._inner, "window_size", None) or getattr(inner_config, "window_size", None) or 128
        )
        # V4 distinguishes attention TP (`tp_group`) from o_proj TP
        # (`o_proj_tp_group`). When the parallel-group manager is available we
        # pick up the dedicated o_proj group; otherwise we fall back to the
        # same attention TP group (V3/V3.2 behavior).
        self.o_proj_tp_group = (
            parallel_group_manager.o_proj_tp_group
            if parallel_group_manager is not None
            and getattr(parallel_group_manager, "o_proj_tp_group", None) is not None
            else tp_group
        )
        # V4 native attributes — mirrors the reference Attention layout.
        # We pull them off the inner module (not via field_names) because
        # `head_dim` (full per-head 512), `n_groups`, and `o_lora_rank`
        # have no MLA wrapper field-name analogue.
        self._head_dim = int(getattr(self._inner, "head_dim"))
        self._qk_rope_head_dim = int(getattr(self._inner, "qk_rope_head_dim"))
        self._n_groups = int(getattr(self._inner, "n_groups", 1))
        if self.o_proj_tp_group.world_size > self._n_groups:
            raise RuntimeError(
                f"Skipped unsupported DeepSeek V4 parallel configuration: "
                f"o_proj_tp={self.o_proj_tp_group.world_size}, o_groups={self._n_groups}. "
                "Grouped O projection in the Flash/Pro model assumes o_proj_tp <= o_groups. "
                "If you have set other parallel configurations, please wait for those results."
            )
        # o_proj grouping follows o_proj_tp_group (may differ from attn TP on V4).
        self._n_local_groups = exact_division(self._n_groups, self.o_proj_tp_group.world_size)
        self._o_lora_rank = int(getattr(self._inner, "o_lora_rank", self._head_dim))
        # Local head count must agree with the actual per-rank Q-projection
        # width, not with `num_heads / tp_world_size`. The two diverge when
        # the column-parallel sharder uses `head_num` to keep whole heads on
        # each rank but the logical division leaves a different remainder
        # (e.g. 32 actual heads vs 64 logical heads after TP=2). Reading the
        # shard width off `q_b_proj` mirrors the flash-inference invariant
        # where local heads are derived from the local projection.
        self._n_local_heads = exact_division(self._local_linear_out_features(self.q_b_proj), self._head_dim)
        # Per-group input width consumed by `wo_a` after o is reshaped to
        # [B, S, n_local_groups, n_local_heads*head_dim/n_local_groups]
        self._per_group_in_dim = exact_division(self._n_local_heads * self._head_dim, self._n_local_groups)
        # Indexer's local KV cache uses `index_head_dim`, distinct from the
        # attention KV's `head_dim`. Defaults to None so ratio-0 layers do not
        # claim an indexer cache width they never use.
        self._index_head_dim: Optional[int] = None
        self.indexer = None
        if self.use_indexer and getattr(self._inner, "indexer", None) is not None:
            self.indexer = DeepseekV4SparseAttentionIndexer(
                self._inner.indexer,
                topk_limit=_resolve_sparse_topk_limit(
                    self._inner.indexer,
                    config=getattr(self._inner, "config", None),
                ),
                tp_group=tp_group,
                compress_ratio=self.compress_ratio,
            )
            self._index_head_dim = int(self.indexer.head_dim)

    @property
    def qk_rope_head_dim(self) -> int:
        return self._qk_rope_head_dim

    def _scatter_window_kv_prefill(
        self,
        kv_window_entry: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        slot_mapping: Optional[torch.Tensor],
        sl: int,
        meta_seq_lens: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write window KV entries into the sliding-window cache during prefill.

        Returns (kv_for_attn, kv_attn_handle) where ``kv_attn_handle`` is the
        post-write functional cache handle that must be wired into
        ``sparse_attn_sharedkv`` to keep the entire KV producer chain live.

        Write semantics:
            - ``sl <= W``:  single scatter of the full window entry
            - ``sl > W`` with ``cutoff == 0``: tail-``W`` fills the cache exactly
            - ``sl > W`` with ``cutoff > 0``: two-scatter split to match the
              circular-buffer semantics of the reference implementation

        The final ``kv_attn_handle + tensor * 0`` expression in the caller is
        NOT a no-op: it binds the two tensors to the same graph node so that
        ``torch.compile``'s dead-code elimination cannot prune the upstream
        KV producer chain (wkv → kv_norm → RoPE → scatter).  Both operands
        are live because their sum is consumed by ``sparse_attn_sharedkv``.
        """
        W = int(self.window_size)
        kv_for_attn = kv_window_entry
        kv_attn_handle = kv_window_entry

        if kv_cache is None:
            return kv_for_attn, kv_attn_handle

        if sl <= W:
            kv_cache = torch.ops.tensor_cast.scatter_nd_update_mla(
                kv_window_entry,
                kv_cache,
                slot_mapping,
                sl,
                meta_seq_lens,
            )
        else:
            cutoff = sl % W
            kv_window_tail = kv_window_entry[:, -W:]
            if cutoff == 0:
                kv_cache = torch.ops.tensor_cast.scatter_nd_update_mla(
                    kv_window_tail,
                    kv_cache,
                    slot_mapping,
                    W,
                    meta_seq_lens,
                )
            else:
                first, second = kv_window_tail.split([W - cutoff, cutoff], dim=1)
                kv_cache = torch.ops.tensor_cast.scatter_nd_update_mla(
                    first,
                    kv_cache,
                    slot_mapping,
                    W - cutoff,
                    meta_seq_lens,
                )
                kv_cache = torch.ops.tensor_cast.scatter_nd_update_mla(
                    second,
                    kv_cache,
                    slot_mapping,
                    cutoff,
                    meta_seq_lens,
                )
        kv_attn_handle = kv_cache

        return kv_for_attn, kv_attn_handle

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        kv_cache_unused: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        kv_cache_by_layers = kwargs.pop("kv_cache_by_layers", None)
        indexer_cache_by_layers = kwargs.pop("indexer_cache_by_layers", None)
        kv_cache = kv_cache_by_layers[self.layer_idx] if kv_cache_by_layers else None
        indexer_cache = (
            indexer_cache_by_layers[self.layer_idx]
            if indexer_cache_by_layers and self.layer_idx in indexer_cache_by_layers
            else None
        )
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings
        rd = self._qk_rope_head_dim
        head_dim = self._head_dim
        n_local_heads = self._n_local_heads

        # --- Q path (reference model.py:495-499) ---
        # Keep the decoder-layer attn_norm boundary explicit; Q path itself then
        # starts from wq_a -> q_norm -> wq_b -> per-head rsqrt -> RoPE.
        q_a = apply_static_quant_linear(hidden_states, self.q_a_proj)
        q_a_normed = torch.ops.tensor_cast.rms_norm(
            q_a,
            self.q_a_layernorm.weight.data,
            getattr(self.q_a_layernorm, "variance_epsilon", 1e-6),
        )
        q_proj = apply_static_quant_linear(q_a_normed, self.q_b_proj)
        n_local_heads = exact_division(q_proj.shape[-1], head_dim)
        q_states = q_proj.view(batch_size, seq_length, n_local_heads, head_dim)
        q_states *= torch.rsqrt(q_states.square().mean(-1, keepdim=True) + 1e-6).to(q_states.dtype)

        # Keep the op input/output at full head width (512 in V4) so memory
        # traffic matches the reference tensor shape; only the trailing `rd`
        # channels participate in RoPE math, as in model.py:499.
        torch.ops.tensor_cast.apply_rope_inplace(q_states, cos, sin, True, False, rd)

        # --- KV path (reference model.py:501-506) ---
        # The reference order is `wkv -> kv_norm -> apply_rotary_emb`; keep
        # those three stages explicit and in that order, with the wkv
        # projection between Q-RoPE and KV-RoPE so KV-RoPE is not collapsed
        # into Q-RoPE.
        #
        # Anchor `wkv`'s input to a Q-path produced tensor (`q_states`) so
        # torch.compile cannot CSE the per-input `dynamic_quantize_symmetric`
        # and concatenate `q_a_proj` (4096 -> q_lora_rank) and
        # `kv_a_proj_with_mqa` (4096 -> head_dim) weights into a single fused
        # (4096, q_lora_rank+head_dim) GEMM. Without this anchoring, inductor
        # collapses the two distinct first-linears into one wider node and
        # the KV first-linear vanishes from the trace.
        kv_input = hidden_states + q_states[..., :1, :1].reshape(batch_size, seq_length, 1).to(hidden_states.dtype) * 0
        kv = apply_static_quant_linear(kv_input, self.kv_a_proj_with_mqa).view(batch_size, seq_length, head_dim)
        kv_normed = torch.ops.tensor_cast.rms_norm(
            kv,
            self.kv_a_layernorm.weight.data,
            getattr(self.kv_a_layernorm, "variance_epsilon", 1e-6),
        )

        # Keep the op input/output at full head width so modeled bytes match
        # the reference shared-KV tensor shape; only the trailing `rd`
        # channels are rotated, as in model.py:504.
        torch.ops.tensor_cast.apply_rope_inplace(kv_normed, cos, sin, True, False, rd)

        # Reference model.py:506 `act_quant(kv[..., :-rd], 64, scale_fmt,
        # scale_dtype, True)`: QAT-style in-place FP8 act_quant over the
        # non-RoPE KV channels (RoPE dims stay bf16 for positional precision).
        # With `simulate=True`, the reference rounds values to FP8 precision
        # but leaves the tensor dtype unchanged so the downstream
        # `self.kv_cache[:bsz] = kv` write continues to consume bf16. We model
        # the per-pass cost via `dynamic_quantize_symmetric` (FP8 e4m3fn), then
        # write the quantized values back into the same slice so the graph keeps
        # a real data dependency on the KV act-quant producer chain.
        kv_nope_quant, _ = torch.ops.tensor_cast.dynamic_quantize_symmetric(
            kv_normed[..., :-rd],
            [-1],
            scale_dtype=torch.float32,
            out_dtype=torch.float8_e4m3fn,
        )
        kv_normed[..., :-rd] = kv_nope_quant.to(kv_normed.dtype)

        kv_window_entry = kv_normed

        # Reference op order (model.py:507-515 + 524-525 + 533):
        #   - Window top-k indices materialized on every layer (inlined below).
        #   - On ratio>0 layers, indexer (ratio==4) or arange-based compress
        #     topk (ratio==128) is concatenated with the window topk.
        #   - The indexer wrapper internally emits its compressor write
        #     (head_dim = `index_head_dim`) as the first stage of its forward,
        #     mirroring the reference `Indexer.forward(self.compressor(...))`.
        #   - Attention's own compressor (head_dim = `head_dim`) runs on
        #     ratio>0 layers, then `sparse_attn_sharedkv` consumes
        #     (q, kv_cache, topk_idxs).
        # `scatter_nd_update_mla` returns a functional handle so the
        # `wkv -> kv_norm -> KV-RoPE -> cat -> scatter -> sparse_attn` chain
        # is a real producer/consumer dataflow rather than relying on
        # `mutates_args` ordering.

        # Shared locals for window/compress topk construction.
        W = int(self.window_size)
        sl = int(seq_length)
        device = hidden_states.device

        meta_query_lens = attention_meta.query_lens if attention_meta is not None else None
        meta_seq_lens = attention_meta.seq_lens if attention_meta is not None else None
        is_decode = _is_decode_attention_batch(sl, attention_meta, batch_size)

        # 1. window topk (model.py:507). Keep the full construction in tensor
        # ops so torch.compile does not pull query_lens / seq_lens back through
        # Python scalars (`tolist` / `item`).
        topk_indices = get_window_topk_idxs(W, batch_size, sl, device, is_decode)

        # 2. ratio>0 layers: indexer (ratio==4) or arange-based compress topk
        # (ratio==128), then cat with window topk (model.py:508-514).
        if self.compress_ratio:
            R = int(self.compress_ratio)
            if self.use_indexer and self.indexer is not None and indexer_cache is not None:
                # Keep the indexer branch anchored to the post-KV tensor so
                # torch.compile cannot DCE or hoist the learned-indexer segment.
                indexer_hidden_states = hidden_states + kv_window_entry[..., :1].to(hidden_states.dtype) * 0
                indexer_q_a_normed = q_a_normed + kv_window_entry[..., :1].to(q_a_normed.dtype) * 0
                compress_topk_indices = self.indexer(
                    indexer_hidden_states,
                    indexer_q_a_normed,
                    position_embeddings,
                    indexer_cache,
                    attention_meta=attention_meta,
                )
            else:
                compress_topk_indices = get_compress_topk_idxs(R, batch_size, sl, device)
            topk_indices = torch.cat([topk_indices, compress_topk_indices], dim=-1)

        # 3. int-cast the merged topk indices (model.py:515).
        topk_indices = topk_indices.int()

        attn_sink = getattr(
            self._inner,
            "attention_sink",
            torch.empty(0, dtype=q_states.dtype, device=q_states.device),
        )
        softmax_scale = float(getattr(self._inner, "softmax_scale", getattr(self._inner, "scaling", head_dim**-0.5)))

        # 4-6. scatter + compressor + sparse_attn (model.py:518-533).
        slot_mapping = attention_meta.slot_mapping if attention_meta is not None else None

        if not is_decode:
            # Prefill: reference attention consumes freshly-built `kv`, but keep
            # the compiled graph anchored on the post-write cache handle so the
            # entire KV producer chain stays live. Feed sparse_attn a payload
            # whose visible data matches `[window | compressed]` while its dtype
            # / meta shape comes from the post-scatter/cache path.
            kv_for_attn, kv_attn_handle = self._scatter_window_kv_prefill(
                kv_window_entry,
                kv_cache,
                slot_mapping,
                sl,
                meta_seq_lens,
            )
            # On ratio>0 layers, chain the compressor to the post-scatter cache
            # and concat the compressed KV so sparse_attn sees [window|compressed].
            if self.use_compressor and kv_cache is not None:
                kv_compress, kv_cache = torch.ops.tensor_cast.compressor(
                    hidden_states,
                    kv_cache,
                    self.compress_ratio,
                    head_dim,
                    rd,
                    False,
                    meta_seq_lens,
                    meta_query_lens,
                )
                kv_for_attn = torch.cat([kv_for_attn, kv_compress], dim=1)
                kv_attn_handle = kv_cache
            # Bind the post-write cache handle to sparse_attn_sharedkv so
            # torch.compile cannot DCE the upstream cache-update chain
            # (wkv -> kv_norm -> RoPE -> scatter -> compressor).
            # This avoids materializing a full-cache elementwise add.
            attn_output = torch.ops.tensor_cast.sparse_attn_sharedkv(
                q_states,
                kv_for_attn,
                attn_sink,
                topk_indices,
                softmax_scale,
                head_dim,
                kv_dependency=kv_attn_handle,
            )
        else:
            # Decode (incl. packed multi-decode where sl=N>1 with all
            # query_lens==1): write `sl` rows (one per per-request decode
            # token), then route sparse_attn through the post-write handle so
            # scatter/compressor and their upstream KV producers remain in the
            # compiled graph.
            kv_for_attn = kv_window_entry
            if kv_cache is not None:
                kv_cache = torch.ops.tensor_cast.scatter_nd_update_mla(
                    kv_window_entry,
                    kv_cache,
                    slot_mapping,
                    sl,
                    meta_seq_lens,
                )
                kv_for_attn = kv_cache
            if self.use_compressor and kv_cache is not None:
                _, kv_cache = torch.ops.tensor_cast.compressor(
                    hidden_states,
                    kv_cache,
                    self.compress_ratio,
                    head_dim,
                    rd,
                    False,
                    meta_seq_lens,
                    meta_query_lens,
                )
                kv_for_attn = kv_cache
            attn_output = torch.ops.tensor_cast.sparse_attn_sharedkv(
                q_states,
                kv_for_attn,
                attn_sink,
                topk_indices,
                softmax_scale,
                head_dim,
            )
        # attn_output: [batch_size, seq_length, n_local_heads, head_dim]

        # Inverse RoPE on o[..., -rd:] (model.py:534, in-place).
        o_view = attn_output
        # Keep the op input/output at full head width so output de-rotation
        # preserves the reference tensor shape; only the trailing `rd`
        # channels are de-rotated, as in model.py:534.
        torch.ops.tensor_cast.apply_rope_inplace(o_view, cos, sin, True, True, rd)

        # --- O projection (reference model.py:537-542) ---
        # o = o.view(bsz, seqlen, n_local_groups, -1)
        # wo_a = self.wo_a.weight.view(n_local_groups, o_lora_rank, -1)
        # o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        # x = self.wo_b(o.flatten(2))
        per_group_in_dim = exact_division(n_local_heads * head_dim, self._n_local_groups)
        o_grouped = o_view.reshape(batch_size, seq_length, self._n_local_groups, per_group_in_dim)
        # `wo_a` is already sharded by the transformation pass using the
        # dedicated o-projection TP group, so the runtime only needs to reshape
        # the local shard into per-group blocks before the grouped einsum.
        wo_a_weight = self._extract_logical_linear_weight(self._inner.wo_a)
        # MXFP4 qweight uses a packed layout (strides may be non-contiguous), so
        # use reshape instead of view when grouping for the per-group einsum.
        wo_a_grouped = wo_a_weight.reshape(self._n_local_groups, self._o_lora_rank, per_group_in_dim).to(
            o_grouped.dtype
        )
        # bsgd, grd -> bsgr
        o_grouped = torch.einsum("bsgd,grd->bsgr", o_grouped, wo_a_grouped)
        # `o_proj`/`wo_b` is wrapped as RowParallel via transformations.py, so
        # its forward already performs the single output all-reduce that matches
        # the reference row-parallel path. Doing another explicit reduce here
        # would double-count communication in the cost model.
        attn_output = self.o_proj(o_grouped.flatten(2))  # wo_b
        # Cast back to hidden_states dtype: mirrors V3.2 behavior where the attention
        # output dtype is anchored to the model working precision (bf16/fp16), so
        # float8_e4m3fn from the KV/cache stream does not pollute the hidden-state
        # flow through hc_post into MTP's RMSNorm layers.
        return attn_output.to(hidden_states.dtype), None


class DeepseekV4SparseAttentionIndexer(ModelWrapperBase):
    """Wrapper for the ratio==4 learned Indexer path in V4 (Flash/Pro).

    Mirrors reference `Indexer.forward` (deepseek-ai/DeepSeek-V4-Flash/inference/model.py:402-433)
    so simulated execution cost tracks the reference's runtime. `rotate_activation`
    and `fp4_act_quant` on q have no standalone tensor_cast op; their cost is
    accounted for inside `quant_lightning_indexer` instead of as separate trace
    events.
    """

    def __init__(
        self,
        indexer,
        topk_limit: Optional[int] = None,
        tp_group: Optional[ParallelGroup] = None,
        compress_ratio: int = 0,
    ):
        super().__init__(indexer)
        self._topk_limit = _resolve_sparse_topk_limit(indexer, topk_limit=topk_limit)
        self.tp_group = tp_group
        self.compress_ratio = int(compress_ratio)

    @property
    def num_heads(self) -> int:
        return self._inner.num_heads

    @property
    def num_local_heads(self) -> int:
        # Same invariant as the main attention path: local head count must
        # come from the actual `wq_b` shard width, since the column-parallel
        # sharder may keep whole heads on each rank using `head_num` rather
        # than the logical `num_heads / world_size` split.
        out_features = getattr(self.wq_b, "out_features_per_partition", None)
        if out_features is None:
            out_features = getattr(self.wq_b, "out_features", None)
        if out_features is None and hasattr(self.wq_b, "_inner"):
            out_features = getattr(self.wq_b._inner, "out_features", None)
        if out_features is None:
            raise AttributeError(f"Unable to resolve local out_features from {type(self.wq_b).__name__}")
        return exact_division(int(out_features), self.head_dim)

    @property
    def head_dim(self) -> int:
        return self._inner.head_dim

    @property
    def topk_limit(self) -> int:
        return self._topk_limit

    def forward(
        self,
        hidden_states: torch.Tensor,
        qa_normed: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        indexer_cache: torch.Tensor,
        attention_meta: Optional[AttentionMetadataBase] = None,
    ):
        # Mirrors reference `Indexer.forward` (deepseek-ai/DeepSeek-V4-Flash/
        # inference/model.py:402-433) op-for-op so the cost trace tracks the
        # original execution. Reference order:
        #     q = wq_b(qr)
        #     q = q.unflatten(-1, (n_local_heads, head_dim))
        #     apply_rotary_emb(q[..., -rd:])           # q-side RoPE
        #     q = rotate_activation(q)                  # Hadamard-style rotation
        #     fp4_act_quant(q, fp4_block_size, True)    # FP4 simulate
        #     self.compressor(x, start_pos)             # compressor write into kv_cache
        #     weights = weights_proj(x) * scale
        #     index_score = einsum(q, kv_cache[:end_pos // ratio])
        #     index_score = (index_score.relu_() * weights[..., None]).sum(2)
        #     all_reduce(index_score)                   # when world_size > 1
        #     prefill: index_score += where(mask, -inf, 0)
        #     topk_idxs = topk(index_score, k=min(index_topk, end_pos // ratio))
        #     prefill: topk_idxs = where(mask, -1, topk_idxs + offset)
        #     decode:  topk_idxs += offset
        cos, sin = position_embeddings
        seq_lens_meta = attention_meta.seq_lens if attention_meta is not None else None
        query_lens_meta = attention_meta.query_lens if attention_meta is not None else None
        batch_size, seq_length = hidden_states.shape[:-1]
        rd = int(self.qk_rope_head_dim)

        # `wq_b` / `weights_proj` are now sharded up front by
        # transformations.py for V4 only, matching the reference
        # ColumnParallelLinear layout and keeping compile-time graphs simpler.
        q_proj = apply_static_quant_linear(qa_normed, self.wq_b)
        num_local_heads = exact_division(q_proj.shape[-1], self.head_dim)
        q_states = q_proj.view(batch_size, seq_length, num_local_heads, self.head_dim)
        # Keep the indexer q tensor at full head width for RoPE I/O shape
        # parity with the reference; only the trailing `rd` channels rotate.
        torch.ops.tensor_cast.apply_rope_inplace(q_states, cos, sin, True, False, rd)

        # `rotate_activation(q)` and `fp4_act_quant(q)` (reference model.py:414-416)
        # are pointwise, shape-preserving stages. tensor_cast has no standalone
        # semantic op for either — instead, both their FLOPs/bytes are charged
        # inside the `quant_lightning_indexer` cost model below over the full
        # (batch, seq_len, num_heads, head_dim) q tensor. We deliberately do not
        # emit a separate trace event for them so the chrome trace shape stays
        # identical to the reference, but the modeled latency does include them.

        # `self.compressor(x, start_pos)` — runs AFTER the q-side ops, matching
        # reference model.py:417. Writes the indexer-local compressed KV cache
        # used by the einsum below; rebind `indexer_cache` from the compressor
        # return so its data edge into `quant_lightning_indexer` survives
        # torch.compile DCE.
        if self.compress_ratio:
            _, indexer_cache = torch.ops.tensor_cast.compressor(
                hidden_states,
                indexer_cache,
                self.compress_ratio,
                int(self.head_dim),
                rd,
                True,
                seq_lens_meta,
                query_lens_meta,
            )

        # weights = weights_proj(x) * (softmax_scale * n_heads ** -0.5).
        weights = apply_static_quant_linear(hidden_states, self.weights_proj) * (
            float(self.head_dim) ** -0.5 * float(self.num_heads) ** -0.5
        )

        # `quant_lightning_indexer` collapses the remaining reference stages into
        # one semantic op whose cost model accounts for, in this order:
        #   - rotate_activation(q) + fp4_act_quant(q) (charged here, not above)
        #   - einsum("bshd,btd->bsht", q, kv_cache[:bsz, :end_pos // ratio])
        #   - relu_() and weighted sum across heads
        #   - all_reduce(index_score) when tp_world_size > 1
        #   - prefill index_score += where(mask, -inf, 0) mask-add
        #   - topk over min(topk_limit, end_pos // ratio)
        # See `_register_op_properties(quant_lightning_indexer)` in
        # tensor_cast/performance_model/__init__.py.
        return torch.ops.tensor_cast.quant_lightning_indexer(
            q_states,
            weights,
            indexer_cache,
            int(self.topk_limit),
            int(self.tp_group.world_size) if self.tp_group is not None else 1,
            seq_lens_meta,
            query_lens_meta,
        )


class HyperConnectedMultiTokenPredictorLayer(MultiTokenPredictorLayer):
    """MTP layer for V4 family with Hyper-Connection (HC).

    The main model output is already HC-reduced to [B,S,D] but the MTP block
    expects HC-expanded [B,S,Hc,D]. This subclass bridges the shape with HC-expand
    at entry and HC-head reduction at exit, mirroring the reference MTPBlock
    semantics (each MTPBlock owns its own hc_head params).
    """

    def __init__(self, hf_config, mtp_block: torch.nn.Module):
        super().__init__(hf_config, mtp_block)
        self.hc_mult = int(getattr(mtp_block, "hc_mult", 1) or 1)
        self.hc_eps = float(getattr(mtp_block, "hc_eps", 1e-6))
        hc_dim = self.hc_mult * hf_config.hidden_size
        self.hc_head_fn = torch.nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = torch.nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.emb_norm(inputs_embeds)
        previous_hidden_states = self.hidden_norm(previous_hidden_states)

        hidden_states = self.linear_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        # HC-expand [B,S,D] -> [B,S,Hc,D] so the decoder block's HC pre/post
        # ops trace with the correct shapes (matches main-model expansion in
        # DeepseekV4Model.forward).
        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        hidden_states = self.mtp_block(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # HC-reduce [B,S,Hc,D] -> [B,S,D] so the downstream lm_head receives
        # the standard hidden-size tensor (mirrors ParallelHead.hc_head and
        # DeepseekV4Model.forward's output reduction).
        hidden_states = torch.ops.tensor_cast.hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.hc_mult,
            self.hc_eps,
        )

        return hidden_states

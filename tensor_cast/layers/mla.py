# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, TYPE_CHECKING

import torch

from .. import config, ops  # noqa: F401
from ..model_config import MlaConfig, MultiheadLatentAttentionQuantConfig
from ..parallel_group import _DEFAULT_PG, ParallelGroup
from ..utils import exact_division

if TYPE_CHECKING:
    from ..parallel_group import ParallelGroupManager
from .attention import AttentionMetadataBase
from .quant_linear import TensorCastQuantLinear
from .utils import get_partial_sharded, ModelWrapperBase


def _get_request_phase_values(
    attention_meta: Optional[AttentionMetadataBase],
) -> Optional[list[bool]]:
    """Return per-request phase metadata (``True`` means decode), if present."""

    return getattr(attention_meta, "is_decode_values", None)


def _get_request_query_lens(
    attention_meta: Optional[AttentionMetadataBase],
) -> Optional[torch.Tensor]:
    """Return materialized per-request query lengths, if present."""

    return getattr(attention_meta, "query_lens", None)


def _get_request_query_lens_values(attention_meta: Optional[AttentionMetadataBase]) -> Optional[list[int]]:
    if attention_meta is None:
        return None
    lengths = getattr(attention_meta, "query_lens_values", None)
    if lengths is not None:
        return lengths
    query_lens = _get_request_query_lens(attention_meta)
    if query_lens is None:
        return None
    try:
        return [int(length) for length in query_lens.tolist()]
    except (RuntimeError, ValueError, TypeError):
        return None


def _get_mla_phase_token_counts(
    attention_meta: Optional[AttentionMetadataBase],
    total_tokens: int,
    decode_only: bool,
) -> tuple[int, int]:
    """Return concrete Prefill/Decode token counts without tensor data control flow."""
    phases = _get_request_phase_values(attention_meta)
    query_lens = _get_request_query_lens_values(attention_meta)
    if phases is not None and query_lens is not None and len(phases) == len(query_lens):
        prefill_tokens = sum(length for length, is_decode in zip(query_lens, phases) if not is_decode)
        decode_tokens = sum(length for length, is_decode in zip(query_lens, phases) if is_decode)
        return prefill_tokens, decode_tokens
    if phases is not None and True in phases and False in phases:
        raise ValueError("Mixed MLA phases require materialized per-request query lengths")
    if attention_meta is not None and attention_meta.is_dcp_decode:
        return 0, total_tokens
    return (0, total_tokens) if decode_only else (total_tokens, 0)


def tp_plan_module_path(prefix: str, relative_path: str) -> str:
    """Return a TP-plan glob for a module under ``prefix``.

    Decoder stacks use ``{prefix}.*.{relative_path}`` because each layer index
    sits between ``prefix`` and the submodule. MTP blocks are already addressed
    as ``mtp.layers.*.mtp_block``, so their attention/MLP weights live directly
    under that prefix without an extra layer wildcard.

    Use :func:`tp_plan_nested_module_path` instead when the target module may sit
    under extra wrapper segments such as ``self_attn._inner`` (e.g. ``o_proj``).
    """
    if ".mtp_block" in prefix:
        return f"{prefix}.{relative_path}"
    return f"{prefix}.*.{relative_path}"


def tp_plan_nested_module_path(prefix: str, relative_path: str) -> str:
    """Return a TP-plan glob that allows optional wrapper segments before ``relative_path``.

    Example: ``model.layers.*.o_proj`` matches ``model.layers.0.self_attn._inner.o_proj``,
    and ``mtp.layers.*.mtp_block.*.o_proj`` matches MTP attention outputs.
    """
    return f"{prefix}.*.{relative_path}"


class MultiheadLatentAttentionBase(torch.nn.Module, ABC):
    # Set to True in subclasses that accept parallel_group_manager in __init__
    supports_parallel_group_manager: bool = False

    def __init__(
        self,
        mla_config: MlaConfig,
        mla_module: torch.nn.Module,
        decode_only: bool = False,
    ) -> None:
        super().__init__()
        self.mla_config = mla_config
        self._inner = mla_module
        self.decode_only = decode_only
        self.quant_config: Optional[MultiheadLatentAttentionQuantConfig] = None

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        pass

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.mla_config.field_names, name):
            return getattr(self._inner, getattr(self.mla_config.field_names, name))
        return super().__getattr__(name)

    def quantize_params(self):
        """
        Called during the initialization phase after the inner module is quantized.
        This allows quantization for extra parameters in this module.
        """


# rotary embedding functions copied from DeepSeek-v3 model in Transformers
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.view(-1, cos.shape[-1])
    sin = sin.view(-1, sin.shape[-1])
    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.view(-1, cos.shape[-1])
    sin = sin.view(-1, sin.shape[-1])

    s, n, d = q.shape
    q = q.view(s, n, d // 2, 2).transpose(-1, -2).reshape(s, n, d)

    s, d = k.shape
    k = k.view(s, d // 2, 2).transpose(-1, -2).reshape(s, d)

    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiheadLatentAttentionTensorCast(MultiheadLatentAttentionBase):
    # Accept the parallel-group manager so we can pick up the DCP group; this also
    # makes ``patch_mla`` pass the manager to every MLA subclass (all of which now
    # forward ``parallel_group_manager`` to this __init__).
    supports_parallel_group_manager: bool = True

    def __init__(
        self,
        mla_config: MlaConfig,
        mla_module: torch.nn.Module,
        tp_group: ParallelGroup,
        decode_only: bool = False,
        parallel_group_manager: Optional["ParallelGroupManager"] = None,
    ):
        super().__init__(mla_config, mla_module, decode_only)
        self.tp_group = tp_group
        # Decode Context Parallel group (a contiguous sub-slice of the TP group).
        # Falls back to the singleton _DEFAULT_PG (world_size == 1) when no manager
        # is supplied or DCP is disabled, so every DCP collective is then a no-op.
        self.dcp_group = (
            parallel_group_manager.dcp_group
            if parallel_group_manager is not None and getattr(parallel_group_manager, "dcp_group", None) is not None
            else _DEFAULT_PG
        )
        self._num_heads_per_rank = exact_division(self.num_heads, tp_group.world_size)
        self._setup_kv_b_decomposition(tp_group)

    def _setup_kv_b_decomposition(self, tp_group: ParallelGroup) -> None:
        """
        Hook: shard ``kv_b_proj`` across TP ranks and split it into ``W_UK``/``W_UV``.

        Default behaviour matches the legacy MLA path used by V3 / V3.2: it requires
        ``self.kv_b_proj`` to be present on the inner module. Subclasses whose
        attention does not have a ``kv_b_proj`` (e.g. DeepSeek V4 with shared KV) can
        override this hook with a no-op.
        """
        sharded_weight = get_partial_sharded(
            self.kv_b_proj.weight.data,
            tp_group.world_size,
            tp_group.rank_in_group,
            unit_num=self.num_heads,
        )
        self.kv_b_proj_weight_t = sharded_weight.transpose(0, 1)
        kv_b_proj_view = self.kv_b_proj_weight_t.view(
            self.kv_lora_rank,
            self._num_heads_per_rank,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_view.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        self.W_UV = W_UV.transpose(0, 1)  # (num_heads_per_rank, kv_lora_rank, v_head_dim)
        self.W_UK_T = W_UK.permute(1, 2, 0)  # (num_heads_per_rank, qk_nope_head_dim, kv_lora_rank)

    @classmethod
    def requires_indexer_cache(cls) -> bool:
        """
        Hook for subclasses that require the auxiliary sparse-attention indexer
        cache allocated by input_generator.

        The default MLA path does not use an indexer cache.
        """
        return False

    @classmethod
    def build_tp_plan_extras(cls, prefix: str, params: dict, config_info) -> dict[str, tuple[str, dict]]:
        """
        Hook for subclasses to inject extra TP sharding rules into the generic MLA
        q/kv shard plan without adding model-specific branches to transformations.py.

        The default MLA path has no extra modules beyond q/kv_b projections.
        Subclasses can override this to register additional learned linears that
        belong to the attention block (e.g. V4 indexer projections).
        """
        return {}

    @classmethod
    def build_o_proj_tp_plan_extras(cls, prefix: str, params: dict, config_info) -> dict[str, tuple[str, dict]]:
        """
        Hook for subclasses to inject extra O-projection-related TP sharding rules
        into the generic MLA shard plan.
        """
        return {}

    @staticmethod
    def extract_qparams(
        module: torch.nn.Module,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        target = module
        if hasattr(module, "_inner"):
            target = module._inner
        if isinstance(target, TensorCastQuantLinear):
            return target.qweight, target.weight_scale, target.weight_offset
        weight = getattr(target, "weight", None)
        if weight is None:
            raise AttributeError(f"Module {module.__class__.__name__} does not expose a weight tensor. ")
        return weight.data, None, None

    def _pre_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_meta: Optional[AttentionMetadataBase] = None,
        qa_normed: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Pre-attention processing hook that runs before core attention computation.
        This hook is INTENDED FOR IN-PLACE CACHE PREPARATION (e.g., writing precomputed key
        features or index data into pre-allocated cache tensors such as indexer_cache).
        """
        return None

    def _use_mlapo_fusion(self) -> bool:
        if self.quant_config is not None and self.quant_config.is_hif8_kv:
            return False  # Existing MLAPO contracts do not encode HIF8 cache writes.
        return config.compilation.fusion_patterns.enable_mlapo

    @torch.no_grad()
    def _compute_mla_prolog(self, hidden_states_view, cos, sin, *, is_decode_values=None):
        """Unfused preprocessing; adapters may preserve their own prolog hook."""
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states_view)
            qa_normed = hidden_states_view.new_empty((hidden_states_view.shape[0], 0))
        else:
            qa = self.q_a_proj(hidden_states_view)
            qa_normed = torch.ops.tensor_cast.rms_norm(
                qa,
                self.q_a_layernorm.weight,
                (
                    getattr(
                        self.q_a_layernorm,
                        "variance_epsilon",
                        getattr(self.q_a_layernorm, "eps", None),
                    )
                    or torch.finfo(qa.dtype).eps
                ),
            )
            q = self.q_b_proj(qa_normed)
        q = q.view(-1, self._num_heads_per_rank, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_proj_with_mqa(hidden_states_view)
        kv_c, k_rope = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = torch.ops.tensor_cast.rms_norm(
            kv_c,
            self.kv_a_layernorm.weight,
            (
                getattr(
                    self.kv_a_layernorm,
                    "variance_epsilon",
                    getattr(self.kv_a_layernorm, "eps", None),
                )
                or torch.finfo(kv_c.dtype).eps
            ),
        )
        q_rope = torch.ops.tensor_cast.apply_rope_single(q_rope, cos, sin, True, False)
        k_rope = torch.ops.tensor_cast.apply_rope_single(k_rope, cos, sin, True, False)
        return torch.cat((q_nope, q_rope), dim=-1), kv_c, k_rope, qa_normed

    def _compute_unfused_mla_projections(self, kv, q, prefill, decode, attention_meta=None):
        # Phase counts are local; empty slices preserve the output contracts.
        prefill, decode = ops.mla.local_mla_phase_token_counts(kv.shape[0], prefill, decode)
        kv_input = kv[:prefill]
        q_decode = q[prefill : prefill + decode]
        phases = _get_request_phase_values(attention_meta)
        lengths = _get_request_query_lens_values(attention_meta)
        if prefill and decode and phases is not None and lengths is not None:
            prefill_parts, decode_parts = [], []
            offset = 0
            declared = sum(lengths)
            for length, is_decode in zip(lengths, phases):
                start = offset * kv.shape[0] // declared
                offset += length
                end = offset * kv.shape[0] // declared
                if is_decode:
                    decode_parts.append(q[start:end])
                else:
                    prefill_parts.append(kv[start:end])
            kv_input = torch.cat(prefill_parts, dim=0)
            q_decode = torch.cat(decode_parts, dim=0)
            prefill, decode = kv_input.shape[0], q_decode.shape[0]
        elif prefill and decode:
            raise ValueError("Mixed MLA projections require materialized per-request query lengths and phases")
        projected = kv.new_empty((0, self._num_heads_per_rank, self.qk_nope_head_dim + self.v_head_dim))
        absorbed = q.new_empty((0, self._num_heads_per_rank, self.kv_lora_rank + self.qk_rope_head_dim))
        if prefill:
            if self.quant_config is None or self.quant_config.is_hif8_kv:
                projected = torch.mm(kv_input, self.kv_b_proj_weight_t.to(kv_input.dtype))
            else:
                qc = self.quant_config
                quantized = torch.ops.tensor_cast.quantize(kv_input, qc.kv_projected_scale, qc.kv_projected_offset)
                projected = torch.ops.tensor_cast.static_quant_linear(
                    quantized,
                    self.kv_b_proj_weight_t,
                    self.kv_b_proj_scale,
                    self.kv_b_proj_offset,
                    qc.kv_projected_scale,
                    qc.kv_projected_offset,
                    None,
                    kv.dtype,
                )
            projected = projected.view(-1, self._num_heads_per_rank, self.qk_nope_head_dim + self.v_head_dim)
        if decode:
            q_nope = q_decode[..., : self.qk_nope_head_dim].transpose(0, 1)
            if self.quant_config is None or self.quant_config.is_hif8_kv:
                absorbed = torch.bmm(q_nope, self.W_UK_T.to(q_nope.dtype))
            else:
                qc = self.quant_config
                quantized = torch.ops.tensor_cast.quantize(q_nope, qc.qk_scale, qc.qk_offset)
                heads = self._num_heads_per_rank
                absorbed = torch.ops.tensor_cast.grouped_matmul_quant(
                    list(quantized.unbind(0)),
                    list(self.W_UK_T.unbind(0)),
                    [self.kv_b_proj_scale] * heads,
                    [self.kv_b_proj_offset] * heads,
                    [qc.qk_scale] * heads,
                    [qc.qk_offset] * heads,
                    [None] * heads,
                    q.dtype,
                ).view(heads, decode, self.kv_lora_rank)
            absorbed = torch.cat(
                (absorbed.transpose(0, 1), q_decode[..., self.qk_nope_head_dim :]),
                dim=-1,
            )
        return projected, absorbed

    def _write_unfused_mla_cache(self, kv, rope, cache, slots):
        if slots is None:
            return
        values = torch.cat((kv, rope), dim=-1)
        # Reuse the dedicated cache-write op. A bare ``aten.index_copy_`` has no
        # registered op-properties functor, so the analytic model falls back to
        # the generic bandwidth functor and charges the entire ``kv_cache`` as a
        # write plus a read/write argument on every call, instead of only the
        # appended rows. This is the default path whenever MLAPO is disabled.
        torch.ops.tensor_cast.siso_reshape_and_cache(values, cache, slots.reshape(-1))

    def _compute_fused_mla_prolog(
        self,
        hidden_states_view: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        **fusion_kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Emit fused prolog, phase projections and cache write as one op."""
        if self.q_lora_rank is None:
            self.q_a_proj_weight, self.q_a_proj_scale, self.q_a_proj_offset = self.extract_qparams(self.q_proj)
            self.q_a_layernorm_weight = None
            self.q_b_proj_weight = None
            self.q_b_proj_scale = None
            self.q_b_proj_offset = None
        else:
            self.q_a_proj_weight, self.q_a_proj_scale, self.q_a_proj_offset = self.extract_qparams(self.q_a_proj)
            self.q_b_proj_weight, self.q_b_proj_scale, self.q_b_proj_offset = self.extract_qparams(self.q_b_proj)
            self.q_a_layernorm_weight = self.q_a_layernorm.weight.data
        self.kv_a_proj_weight, self.kv_a_proj_scale, self.kv_a_proj_offset = self.extract_qparams(
            self.kv_a_proj_with_mqa
        )
        self.kv_a_layernorm_weight = self.kv_a_layernorm.weight.data
        linear_quant_enabled = (
            getattr(self, "q_a_proj_scale", None) is not None
            and (self.q_lora_rank is None or getattr(self, "q_b_proj_scale", None) is not None)
            and getattr(self, "kv_a_proj_scale", None) is not None
        )
        cache = fusion_kwargs.pop("kv_cache", None)
        if cache is None:
            cache = hidden_states_view.new_empty((0,))
        if linear_quant_enabled:
            return torch.ops.tensor_cast.mlapo_quant(
                hidden_states_view,
                cos,
                sin,
                self.q_a_proj_weight,
                self.q_a_layernorm_weight,
                self.q_b_proj_weight,
                self.kv_a_proj_weight,
                self.kv_a_layernorm_weight,
                self._num_heads_per_rank,
                self.qk_head_dim,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.kv_lora_rank,
                self.q_lora_rank,
                self.q_a_proj_scale,
                self.q_a_proj_offset,
                self.q_b_proj_scale,
                self.q_b_proj_offset,
                self.kv_a_proj_scale,
                self.kv_a_proj_offset,
                cache,
                **fusion_kwargs,
            )
        return torch.ops.tensor_cast.mlapo(
            hidden_states_view,
            cos,
            sin,
            self.q_a_proj_weight,
            self.q_a_layernorm_weight,
            self.q_b_proj_weight,
            self.kv_a_proj_weight,
            self.kv_a_layernorm_weight,
            self._num_heads_per_rank,
            self.qk_head_dim,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.kv_lora_rank,
            self.q_lora_rank,
            cache,
            **fusion_kwargs,
        )

    def _postprocess_attention_output(self, attn_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Model-specific processing after MLA phase merge and before ``o_proj``."""
        return attn_output

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
        kv_cache = kv_cache_by_layers[self.layer_idx] if kv_cache_by_layers else None
        batch_size, seq_length = hidden_states.shape[:-1]
        num_tokens = batch_size * seq_length
        hidden_states_view = hidden_states.view(num_tokens, -1)
        cos, sin = position_embeddings
        num_prefill_tokens, num_decode_tokens = _get_mla_phase_token_counts(
            attention_meta, num_tokens, self.decode_only
        )
        fuse_mlapo = self._use_mlapo_fusion()
        if fuse_mlapo:
            projection_scales = {}
            if self.quant_config is not None:
                projection_scales = {
                    "kv_projected_scale": self.quant_config.kv_projected_scale,
                    "kv_projected_offset": self.quant_config.kv_projected_offset,
                    "qk_scale": self.quant_config.qk_scale,
                    "qk_offset": self.quant_config.qk_offset,
                    "projection_weight_scale": self.kv_b_proj_scale,
                    "projection_weight_offset": self.kv_b_proj_offset,
                    "kv_scale": self.quant_config.kv_scale,
                    "kv_offset": self.quant_config.kv_offset,
                }
            q_states, kv_c_normed, k_rot, qa_normed, projected_kv, absorbed_q = self._compute_fused_mla_prolog(
                hidden_states_view,
                cos,
                sin,
                kv_b_proj=self.kv_b_proj_weight_t,
                w_uk_t=self.W_UK_T,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                kv_cache=kv_cache,
                slot_mapping=attention_meta.slot_mapping if attention_meta is not None else None,
                is_decode_values=_get_request_phase_values(attention_meta),
                **projection_scales,
            )
        else:
            q_states, kv_c_normed, k_rot, qa_normed = self._compute_mla_prolog(
                hidden_states_view, cos, sin, is_decode_values=_get_request_phase_values(attention_meta)
            )

        if self.q_lora_rank is not None:
            qa_normed = qa_normed.view(batch_size, seq_length, -1)
        else:
            qa_normed = None
        pre_attn_out = self._pre_attention_forward(
            hidden_states=hidden_states,
            qa_normed=qa_normed,
            position_embeddings=position_embeddings,
            attention_meta=attention_meta,
            **kwargs,
        )

        query_start_loc = attention_meta.query_start_loc if attention_meta else None
        seq_lens = attention_meta.seq_lens if attention_meta else None
        query_lens = attention_meta.query_lens if attention_meta else None

        if not fuse_mlapo:
            if self.quant_config is not None and self.quant_config.is_hif8_kv:
                q_states = q_states.to(torch.bfloat16)
                kv_c_normed = kv_c_normed.to(torch.bfloat16)
                k_rot = k_rot.to(torch.bfloat16)
            num_prefill_tokens, num_decode_tokens = _get_mla_phase_token_counts(
                attention_meta, int(q_states.shape[0]), self.decode_only
            )
            projected_kv, absorbed_q = self._compute_unfused_mla_projections(
                kv_c_normed, q_states, num_prefill_tokens, num_decode_tokens, attention_meta
            )

        # --- Decode Context Parallel (decode path only) ---
        # Gather Q heads across the DCP group so each rank holds h_q*dcp/tp heads,
        # then shard the KV context so each rank only attends over its local S/dcp.
        # Unlike GQA, MLA's latent KV is NOT partitioned across the TP heads, so DCP
        # genuinely seq-shards it: the gathered Q heads (cost model reads
        # ``num_heads = q.size(1)``) and the sharded ``seq_lens`` are both real
        # per-rank changes. ``is_dcp_decode_batch`` accepts ordinary decode and MTP
        # (query_len = 1 + num_spec) while rejecting prefill. all_gather Q keeps the
        # model dtype; the seq-len shard is a fresh tensor (attention_meta.seq_lens is
        # shared across layers - never mutate it).
        #
        # Only the FIA part of the op consumes the gathered head count. The two absorb
        # matmuls (``q @ W_UK_T`` and ``@ W_UV``) stay at this rank's h_q/tp heads --
        # real DCP does the former before this all_gather and the latter after the merge,
        # since a rank holds no other rank's absorb weights (``W_UK_T``/``W_UV`` here are
        # built at ``_num_heads_per_rank``). The cost model derives that from
        # ``W_UK_T.size(0)`` rather than ``q.size(1)``; see the decode branch of
        # ``_multihead_latent_attention_properties_helper``.
        apply_dcp = self.dcp_group.world_size > 1 and attention_meta is not None and attention_meta.is_dcp_decode
        if apply_dcp:
            absorbed_q = self.dcp_group.all_gather(absorbed_q, dim=1)
            dcp_size = self.dcp_group.world_size
            # Round the shard UP, and clamp to >= 1. Two reasons this is not floor:
            #
            # 1. Critical path: real DCP splits the context at block granularity, so
            #    the remainder lands on a subset of ranks that then hold
            #    ``ceil(S/dcp)`` tokens. The merge all_to_all below is a
            #    synchronization point, so the step's latency is set by the LONGEST
            #    shard -- flooring models every rank at its most optimistic length
            #    (e.g. S=2001, dcp=8 -> 250 instead of the real 251).
            # 2. Short contexts: ``is_dcp_decode_batch`` only requires
            #    ``seq_len > query_len``, so seq_len can be as low as 2 and
            #    ``seq_len < dcp`` is reachable (e.g. --context-length 4 --dcp-size 8).
            #    Flooring to 0 there makes the cost model bill the layer as "no KV to
            #    read" (both the FLOPs term ``sum(query_lens * seq_lens)`` and the
            #    KV-read term ``sum(seq_lens * ...)`` collapse), understating its time.
            seq_lens = torch.clamp(
                torch.div(seq_lens + (dcp_size - 1), dcp_size, rounding_mode="floor"),
                min=1,
            )

        hif8_kv = self.quant_config is not None and self.quant_config.is_hif8_kv
        if hif8_kv:
            if kv_cache is None or attention_meta is None:
                raise ValueError("HIF8 KV MLA requires attention metadata and a latent KV cache")
            values = torch.cat((kv_c_normed, k_rot), dim=-1)
            torch.ops.tensor_cast.hif8_kv_cache_write(
                values, None, kv_cache, attention_meta.slot_mapping.reshape(-1), self.quant_config.kv_scale
            )
        elif self.quant_config is not None:
            quant_config = self.quant_config
            out_dtype = self.quant_config.get_quant_dtype()
            q_states = torch.ops.tensor_cast.quantize(
                q_states,
                quant_config.query_scale,
                quant_config.query_offset,
                out_dtype,
            )
            if not fuse_mlapo:
                kv_c_normed = torch.ops.tensor_cast.quantize(
                    kv_c_normed,
                    quant_config.kv_scale,
                    quant_config.kv_offset,
                    out_dtype,
                )
                k_rot = torch.ops.tensor_cast.quantize(
                    k_rot,
                    quant_config.kv_scale,
                    quant_config.kv_offset,
                    out_dtype,
                )

            if attention_meta is not None and not fuse_mlapo:
                self._write_unfused_mla_cache(kv_c_normed, k_rot, kv_cache, attention_meta.slot_mapping)
        else:
            if attention_meta is not None and not fuse_mlapo:
                self._write_unfused_mla_cache(kv_c_normed, k_rot, kv_cache, attention_meta.slot_mapping)

        extra_backend_kwargs = {
            "topk_limit": None,
            "topk_indices": None,
            **self._get_backend_kwargs(pre_attn_out),
            **self._get_backend_metadata_kwargs(attention_meta),
        }
        if hif8_kv:
            attention_backend = partial(
                torch.ops.tensor_cast.multihead_latent_attention_hif8_kv,
                v_head_dim=self.v_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                kv_scale=self.quant_config.kv_scale,
                **extra_backend_kwargs,
            )
        elif self.quant_config is not None:
            attention_backend = partial(
                self._get_attention_op(quant_enabled=True),
                v_head_dim=self.v_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                query_scale=self.quant_config.query_scale,
                query_offset=self.quant_config.query_offset,
                kv_scale=self.quant_config.kv_scale,
                kv_offset=self.quant_config.kv_offset,
                v_scale=self.quant_config.v_scale,
                v_offset=self.quant_config.v_offset,
                attention_prob_scale=self.quant_config.attention_prob_scale,
                attention_prob_offset=self.quant_config.attention_prob_offset,
                out_scale=self.quant_config.out_scale,
                out_offset=self.quant_config.out_offset,
                out_dtype=hidden_states.dtype,
                **extra_backend_kwargs,
            )
        else:
            attention_backend = partial(
                self._get_attention_op(quant_enabled=False),
                v_head_dim=self.v_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                **extra_backend_kwargs,
            )

        prefill_output, decode_latent_output = attention_backend(
            q=q_states,
            projected_kv=projected_kv,
            absorbed_q=absorbed_q,
            kv_cache=kv_cache,
            block_table=attention_meta.block_table_tensor if attention_meta is not None else None,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            query_lens=query_lens,
        )

        # --- Decode Context Parallel merge ---
        # Each rank produced partial attention for h_q*dcp/tp gathered heads over its
        # local S/dcp KV. Redistribute (output + lse) across the DCP group so every
        # rank can online-softmax-merge its own h_q/tp heads from the dcp partials.
        # vllm-ascend upcasts output & lse to fp32 before this all_to_all, so the
        # communication volume must be counted in fp32 (4 bytes/elem) regardless of
        # the model/KV dtype - enforced here purely via the fp32 meta tensor.
        if apply_dcp:
            decode_latent_output = self._dcp_merge_all_to_all(decode_latent_output, batch_size, seq_length)

        v_up_projection_op = (
            torch.ops.tensor_cast.mla_v_up_projection_quant
            if self.quant_config is not None and not hif8_kv
            else torch.ops.tensor_cast.mla_v_up_projection
        )
        if num_decode_tokens > 0:
            v_weight = self.W_UV.to(decode_latent_output.dtype) if hif8_kv else self.W_UV
            decode_output = v_up_projection_op(decode_latent_output, v_weight)
        else:
            decode_output = torch.empty(
                0,
                self._num_heads_per_rank,
                self.v_head_dim,
                dtype=decode_latent_output.dtype,
                device=decode_latent_output.device,
            )
        merge_tokens = int(prefill_output.shape[0]) + int(decode_output.shape[0])
        if merge_tokens == 0:
            merge_tokens = int(q_states.shape[0])
        attn_output = torch.ops.tensor_cast.mla_merge_phase_outputs(
            prefill_output,
            decode_output,
            merge_tokens,
            _get_request_phase_values(attention_meta),
            getattr(attention_meta, "query_lens_values", None) if attention_meta is not None else None,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self._postprocess_attention_output(attn_output, hidden_states)
        attn_output = self.o_proj(attn_output.to(hidden_states.dtype) if hif8_kv else attn_output)
        return self._format_forward_output(attn_output, None, pre_attn_out)

    def _format_forward_output(self, attn_output, attn_weights, pre_attn_out) -> tuple:
        """Hook for subclasses to attach sparse-attention intermediates to forward output."""
        return attn_output, attn_weights

    def _dcp_merge_all_to_all(
        self,
        attn_output: torch.Tensor,
        batch_size: int,
        seq_length: int,
    ) -> torch.Tensor:
        """Model the DCP output+lse all_to_all and return output reduced to h_q/tp heads.

        ``attn_output`` enters as ``(num_tokens, h_q*dcp/tp, kv_lora_rank)``. We build a
        synthetic fp32 ``output + lse`` payload with the gathered heads on dim 0 (the
        dimension ``all_to_all`` splits) and exchange ``h_q/tp``-head shards across the
        ``dcp`` ranks. The exchanged tensor is only needed for its communication cost;
        the local online-softmax merge collapses the ``dcp`` partials back to the rank's
        own ``h_q/tp`` heads, so we slice ``attn_output`` accordingly for ``o_proj``.
        """
        dcp = self.dcp_group.world_size
        heads_per_rank = self._num_heads_per_rank
        gathered_heads = heads_per_rank * dcp
        num_tokens = batch_size * seq_length
        # latent output + lse (1 column per head), upcast to fp32 for the a2a.
        # The payload must carry a value edge from the placeholder-derived
        # ``attn_output``: a bare ``torch.empty`` is an all-constant input, so the meta
        # constant-folder would precompute the whole ``empty -> all_to_all`` chain and
        # erase the collective (dropping its communication cost). Adding a value-neutral
        # ``[0] * 0`` term keeps the a2a *input* non-constant so it survives folding.
        out_lse = torch.empty(
            (gathered_heads, num_tokens * (attn_output.shape[-1] + 1)),
            dtype=torch.float32,
            device=attn_output.device,
        )
        out_lse = out_lse + attn_output.reshape(-1)[0].to(torch.float32) * 0
        split_sizes = [heads_per_rank] * dcp
        exchanged = self.dcp_group.all_to_all(out_lse, split_sizes, split_sizes)
        # After the local merge each rank keeps its own h_q/tp heads. Bind the a2a
        # output back (value-neutrally) so torch.compile's DCE keeps the collective.
        merged = attn_output[:, :heads_per_rank, :].contiguous()
        merged = merged + exchanged.reshape(-1)[0].to(merged.dtype) * 0
        return merged

    def _get_backend_kwargs(self, pre_attn_out) -> dict:
        """
        Legacy hook for subclasses to inject arguments into the attention backend.

        Keep the single-argument signature for out-of-tree subclasses. Metadata-aware
        extensions belong in :meth:`_get_backend_metadata_kwargs`.
        """
        return {}

    def _get_backend_metadata_kwargs(
        self,
        attention_meta: Optional[AttentionMetadataBase] = None,
    ) -> dict:
        """Hook for subclasses to inject attention-metadata-derived arguments."""
        return {"is_decode_values": _get_request_phase_values(attention_meta)}

    def _get_attention_op(self, quant_enabled: bool):
        """Return the TC op to use for attention. Subclasses override for non-dense paths."""
        if quant_enabled:
            return torch.ops.tensor_cast.multihead_latent_attention_quant
        return torch.ops.tensor_cast.multihead_latent_attention

    def quantize_params(self):
        assert self.quant_config is not None, "quant_config must be set before quantization"
        if self.quant_config.is_hif8_kv:
            if not hasattr(self, "kv_b_proj_weight_t"):
                raise ValueError("HIF8 KV cache requires the supported MLA latent-cache layout")
            return  # KV storage changes; projection weights keep floating-point semantics.
        self._quantize_kv_b_decomposition()

    def _quantize_kv_b_decomposition(self) -> None:
        """
        Hook: quantize the ``kv_b_proj`` decomposition tensors set up by
        :meth:`_setup_kv_b_decomposition`. Subclasses that override the setup hook
        should typically override this one as well (see DeepSeek V4 wrapper).
        """
        out_dtype = self.quant_config.get_quant_dtype()
        kv_b_proj = self.kv_b_proj
        if not isinstance(kv_b_proj, TensorCastQuantLinear):
            raise ValueError("MLA quantization requires kv_b_proj to be quantized")
        self.kv_b_proj_scale = kv_b_proj.weight_scale
        self.kv_b_proj_offset = kv_b_proj.weight_offset
        self.kv_b_proj_weight_t = torch.ops.tensor_cast.quantize(
            self.kv_b_proj_weight_t,
            self.kv_b_proj_scale,
            self.kv_b_proj_offset,
            out_dtype,
        )
        self.W_UK_T = torch.ops.tensor_cast.quantize(
            self.W_UK_T,
            self.kv_b_proj_scale,
            self.kv_b_proj_offset,
            out_dtype,
        )
        self.W_UV = torch.ops.tensor_cast.quantize(
            self.W_UV,
            self.kv_b_proj_scale,
            self.kv_b_proj_offset,
            out_dtype,
        )


def _resolve_sparse_topk_limit(
    indexer: torch.nn.Module,
    config: Optional[torch.nn.Module] = None,
    topk_limit: Optional[int] = None,
) -> int:
    if topk_limit is not None:
        return topk_limit

    inner_topk_limit = getattr(indexer, "topk_limit", None)
    if inner_topk_limit is not None:
        return inner_topk_limit

    for candidate in (config, getattr(indexer, "config", None)):
        candidate_topk_limit = getattr(candidate, "topk_limit", None)
        if candidate_topk_limit is not None:
            return candidate_topk_limit

        candidate_index_topk = getattr(candidate, "index_topk", None)
        if candidate_index_topk is not None:
            return candidate_index_topk

    raise AttributeError("topk_limit")


class DeepseekSparseAttention(MultiheadLatentAttentionTensorCast):
    @classmethod
    def requires_indexer_cache(cls) -> bool:
        return True

    def __init__(
        self,
        mla_config: MlaConfig,
        mla_module: torch.nn.Module,
        tp_group: ParallelGroup,
        decode_only: bool = False,
        parallel_group_manager: Optional["ParallelGroupManager"] = None,
    ):
        super().__init__(
            mla_config,
            mla_module,
            tp_group,
            decode_only,
            parallel_group_manager=parallel_group_manager,
        )
        self.indexer = DeepseekSparseAttentionIndexer(
            self._inner.indexer,
            split_bf16=getattr(getattr(self._inner, "config", None), "model_type", None) == "glm_moe_dsa",
            topk_limit=_resolve_sparse_topk_limit(
                self._inner.indexer,
                config=getattr(self._inner, "config", None),
            ),
        )

    def _get_backend_kwargs(self, pre_attn_out) -> dict:
        """
        Sparse Attention Args:
        - topk_limit: int, number of selected sparse tokens
        - topk_indices: Tensor, precomputed sparse position indices
        """
        return {
            "topk_limit": self.indexer.topk_limit,
            "topk_indices": pre_attn_out,
        }

    def _get_attention_op(self, quant_enabled: bool):
        """Return the sparse-MLA (SFA) TC op for the DSA path.

        Overrides the dense MLA base to dispatch to mla_sparse_attention[_quant],
        which map the attention sub-kernel to NPU SparseFlashAttention instead of
        FusedInferAttentionScore.
        """
        if quant_enabled:
            return torch.ops.tensor_cast.mla_sparse_attention_quant
        return torch.ops.tensor_cast.mla_sparse_attention

    def _pre_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_meta: Optional[AttentionMetadataBase] = None,
        qa_normed: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self._run_sparse_attention_indexer(
            hidden_states, qa_normed, position_embeddings, attention_meta, **kwargs
        )

    def _run_sparse_attention_indexer(
        self,
        hidden_states: torch.Tensor,
        qa_normed: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ):
        if qa_normed is None:
            return None

        indexer_cache_by_layers = kwargs.pop("indexer_cache_by_layers", None)
        indexer_cache = indexer_cache_by_layers[self.layer_idx] if indexer_cache_by_layers else None
        return self.indexer(hidden_states, qa_normed, position_embeddings, indexer_cache, attention_meta)


class DeepseekSparseAttentionIndexer(ModelWrapperBase):
    def __init__(self, indexer, topk_limit: Optional[int] = None, split_bf16: bool = True):
        super().__init__(indexer)
        self._topk_limit = _resolve_sparse_topk_limit(indexer, topk_limit=topk_limit)
        self._split_bf16 = split_bf16
        # The fused BF16 indexer wants fp32 normalization parameters, but
        # ``indexer.k_norm`` is shared: ``Module.float()`` retypes it in place, so
        # every other reader of that submodule silently loses its original dtype
        # (the GLM5-next indexer reads the same submodule and does not upcast its
        # input). Keep wrapper-owned fp32 tensors instead, mirroring the
        # ``.weight.data`` pattern used by the MLA wrappers, so that no submodule
        # is registered and no extra ``state_dict`` key appears.
        self._k_norm_weight: Optional[torch.Tensor] = None
        self._k_norm_bias: Optional[torch.Tensor] = None
        self._k_norm_eps: Optional[float] = None
        if split_bf16 and hasattr(indexer, "k_norm"):
            k_norm = indexer.k_norm
            self._k_norm_weight = k_norm.weight.detach().float()
            bias = getattr(k_norm, "bias", None)
            self._k_norm_bias = None if bias is None else bias.detach().float()
            self._k_norm_eps = getattr(k_norm, "eps", None)

    @property
    def k_norm_weight(self) -> torch.Tensor:
        """fp32 k-norm weight when ``split_bf16`` is on, else the shared weight."""
        if self._k_norm_weight is not None:
            return self._k_norm_weight
        return self.k_norm.weight

    @property
    def k_norm_bias(self) -> Optional[torch.Tensor]:
        if self._k_norm_weight is not None:
            return self._k_norm_bias
        return getattr(self.k_norm, "bias", None)

    @property
    def k_norm_eps(self) -> Optional[float]:
        if self._k_norm_weight is not None:
            return self._k_norm_eps
        return self.k_norm.eps

    @property
    def num_heads(self) -> int:
        if hasattr(self._inner, "num_heads"):
            return self._inner.num_heads
        return self._inner.n_heads

    @property
    def head_dim(self) -> int:
        if hasattr(self._inner, "head_dim"):
            return self._inner.head_dim
        return self._inner.index_head_dim

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
        return self._forward_expanded_indexer(
            hidden_states, qa_normed, position_embeddings, indexer_cache, attention_meta
        )

    def _forward_expanded_indexer(self, hidden_states, qa_normed, position_embeddings, indexer_cache, attention_meta):
        """Expose indexer preparation; the core owns scoring and top-k only."""
        cos, sin = position_embeddings
        hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        qa = qa_normed.reshape(-1, qa_normed.shape[-1])
        k = self.wk(hidden)
        norm_input = k.float() if self._split_bf16 else k
        k = torch.ops.tensor_cast.layer_norm(
            norm_input,
            self.k_norm_weight,
            self.k_norm_bias,
            self.k_norm_eps,
        ).to(hidden_states.dtype)
        k_rope, k_nope = k.split([self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        k_rope = torch.ops.tensor_cast.apply_rope_single(k_rope, cos, sin, False, False)
        k = torch.cat((k_rope, k_nope), dim=-1)
        slots = attention_meta.slot_mapping if attention_meta is not None else None
        if slots is not None:
            torch.ops.tensor_cast.siso_reshape_and_cache(k, indexer_cache, slots.reshape(-1))
        weights = self.weights_proj(hidden)
        q = self.wq_b(qa).view(-1, self.num_heads, self.head_dim)
        q_rope, q_nope = q.split([self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_rope = torch.ops.tensor_cast.apply_rope_single(q_rope, cos, sin, False, False)
        q = torch.cat((q_rope, q_nope), dim=-1)
        topk = self.topk_limit
        if attention_meta is not None:
            active_len = getattr(attention_meta, "max_total_seq_len", None)
            if active_len is None and (lengths := getattr(attention_meta, "seq_lens_values", None)):
                active_len = max(lengths)
            if active_len is not None:
                topk = min(topk, int(active_len))
        return torch.ops.tensor_cast.dsa_indexer(
            q,
            weights,
            indexer_cache,
            attention_meta.block_table_tensor if attention_meta is not None else None,
            attention_meta.seq_lens if attention_meta is not None else None,
            topk,
            _get_request_query_lens(attention_meta),
            is_decode_values=_get_request_phase_values(attention_meta),
        ).view(hidden_states.shape[0], -1, topk)

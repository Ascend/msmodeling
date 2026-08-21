# Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
"""Dflash unified draft modeling for TensorCast.

Draft = builtin Qwen3Config + Qwen3DFlash layers + TC attention + KV injection.
Does not reuse target MLA import. See docs/RFC/rfc_dflash_unified_modeling_zh.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from .. import ops  # noqa: F401
from ..model_config import DflashConfig
from .attention import AttentionMetadataTensorCast
from .dflash_qwen3 import Qwen3DFlashDecoderLayer, _flatten_tokens
from .sampler import Sampler, SamplingMetadata
from .utils import ModelWrapperBase

_BUILTIN_DRAFT_CONFIG = (
    Path(__file__).resolve().parent.parent / "runtime_configs" / "draft_configs" / "dflash_draft_builtin.json"
)
_NON_TRANSFORMER_KEYS = frozenset(
    {
        "architectures",
        "auto_map",
        "block_size",
        "dflash_config",
        "num_target_layers",
        "mask_token_id",
        "dtype",
    }
)


def resolve_l_ctx(
    context_length: int,
    layer_type: str,
    sliding_window: Optional[int],
) -> int:
    """Per-layer context length used for KV modeling."""
    ctx = max(int(context_length), 0)
    if layer_type == "sliding_attention":
        sw = int(sliding_window) if sliding_window is not None else 2048
        return min(ctx, sw) if ctx > 0 else sw
    return ctx


def load_dflash_draft_config_dict(path: Optional[str] = None) -> dict:
    """Load draft config from path or the builtin profile."""
    if path:
        p = Path(path)
        if p.is_dir():
            p = p / "config.json"
        if not p.is_file():
            raise FileNotFoundError(f"Dflash draft config not found: {path}")
    else:
        p = _BUILTIN_DRAFT_CONFIG
        if not p.is_file():
            raise FileNotFoundError(f"Builtin Dflash draft config missing: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _sync_layer_types(layer_types: List[str], num_layers: int) -> List[str]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not layer_types:
        return ["full_attention"] * num_layers
    if len(layer_types) == num_layers:
        return list(layer_types)
    if len(layer_types) > num_layers:
        return list(layer_types[:num_layers])
    return list(layer_types) + [layer_types[-1]] * (num_layers - len(layer_types))


def build_draft_hf_config(
    dcfg: DflashConfig,
    *,
    target_hidden_size: int,
    target_vocab_size: int,
    target_max_position_embeddings: Optional[int] = None,
) -> Qwen3Config:
    """Builtin/optional path → Qwen3Config; CLI fields on ``dcfg`` already applied."""
    source = load_dflash_draft_config_dict(dcfg.draft_model_config_path)
    dflash = source.get("dflash_config", {}) or {}
    target_layer_ids = list(dcfg.aux_hidden_state_layer_ids or dflash.get("target_layer_ids") or [])
    if not target_layer_ids:
        raise ValueError("Dflash requires non-empty target_layer_ids / aux_hidden_state_layer_ids")

    transformer_dict = {k: v for k, v in source.items() if k not in _NON_TRANSFORMER_KEYS}
    transformer_dict.pop("auto_map", None)
    transformer_dict.pop("architectures", None)
    transformer_dict["hidden_size"] = int(target_hidden_size)
    transformer_dict["vocab_size"] = int(target_vocab_size)
    if target_max_position_embeddings is not None:
        transformer_dict["max_position_embeddings"] = int(target_max_position_embeddings)
    transformer_dict["num_hidden_layers"] = int(dcfg.num_draft_layers)
    layer_types = _sync_layer_types(
        list(transformer_dict.get("layer_types") or dcfg.layer_types or []),
        dcfg.num_draft_layers,
    )
    transformer_dict["layer_types"] = layer_types
    has_sw = any(t == "sliding_attention" for t in layer_types)
    if has_sw and transformer_dict.get("sliding_window") is None:
        transformer_dict["sliding_window"] = dcfg.sliding_window if dcfg.sliding_window is not None else 2048
    if has_sw:
        transformer_dict["use_sliding_window"] = True
    transformer_dict["_attn_implementation"] = "tensor_cast"

    draft_hf_config = Qwen3Config(**transformer_dict)
    # Not a HF transformer field (stripped above); keep for noise [anchor|MASK…] ids.
    mask_token_id = source.get("mask_token_id", dflash.get("mask_token_id", 0))
    draft_hf_config.mask_token_id = int(mask_token_id or 0)
    dcfg.aux_hidden_state_layer_ids = target_layer_ids
    dcfg.layer_types = layer_types
    if getattr(draft_hf_config, "sliding_window", None) is not None:
        dcfg.sliding_window = int(draft_hf_config.sliding_window)
    return draft_hf_config


def apply_cli_overrides_to_source_and_dcfg(
    dcfg: DflashConfig,
    *,
    cli_block_size: Optional[int] = None,
    cli_num_draft_layers: Optional[int] = None,
    prefer_existing: bool = False,
) -> dict:
    """Load config dict and apply CLI overrides onto ``dcfg``.

    Priority for ``block_size`` / ``num_draft_layers``:
    explicit CLI > (``prefer_existing`` keeps current ``dcfg``) > draft config file > ``dcfg`` default.

    ``prefer_existing=True`` is required for re-entry (e.g. ``build_dflash_draft_and_wrapper``)
    so a second call without CLI args cannot clobber a prior ``--dflash-block-size`` override
    with the builtin file default (8).
    """
    source = load_dflash_draft_config_dict(dcfg.draft_model_config_path)
    dflash = source.get("dflash_config", {}) or {}
    file_block = source.get("block_size") or dflash.get("block_size")
    if cli_block_size is not None and cli_block_size >= 2:
        block_size = cli_block_size
    elif prefer_existing:
        block_size = dcfg.dflash_block_size
    else:
        block_size = file_block or dcfg.dflash_block_size
    file_layers = source.get("num_hidden_layers")
    if cli_num_draft_layers is not None and cli_num_draft_layers > 0:
        num_layers = cli_num_draft_layers
    elif prefer_existing:
        num_layers = dcfg.num_draft_layers
    else:
        num_layers = int(file_layers or dcfg.num_draft_layers)
    dcfg.dflash_block_size = int(block_size)
    dcfg.num_draft_layers = int(num_layers)
    if dcfg.aux_hidden_state_layer_ids is None:
        ids = dflash.get("target_layer_ids")
        if ids:
            dcfg.aux_hidden_state_layer_ids = list(ids)
    # Re-clamp acceptance after block_size may change.
    max_accept = float(dcfg.dflash_block_size - 1)
    if dcfg.dflash_acceptance_length > max_accept:
        dcfg.dflash_acceptance_length = max_accept
    return source


def build_draft_attention_metadata(
    batch_size: int,
    ctx_len: int,
    block: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.long,
    page_block_size: int = 128,
) -> tuple[AttentionMetadataTensorCast, AttentionMetadataTensorCast]:
    """Build (context_write_meta, noise_attn_meta) for draft reshape_and_cache + attn.

    Slot layout per request: ``[0 .. ctx_len)`` context, then ``[ctx_len .. ctx_len+block)`` noise.

    Uses ``torch.arange`` (not ``torch.tensor(python_list)``) so Dynamo/FakeTensor
    tracing under ``--compile`` does not see non-fake meta tensors.
    batch_size may be a SymInt under ``dynamic_shapes``; do not call ``int(batch_size)``
    (that specializes to a wrong concrete hint and desyncs attention meta from
    noise ``[B, block, H]``, producing residual add errors like ``8`` vs ``8*s90``).

    Aligns with ``input_generator`` by filling ``seq_lens_values`` /
    ``query_lens_values`` / ``is_decode_values`` when ``batch_size`` is a concrete
    int. Draft metadata never enables target DCP.
    """
    ctx_len = max(int(ctx_len), 0)
    block = max(int(block), 1)
    total = ctx_len + block
    bases = torch.arange(batch_size, device=device, dtype=dtype) * total
    if ctx_len > 0:
        ctx_slot_mapping = (bases.unsqueeze(1) + torch.arange(ctx_len, device=device, dtype=dtype)).reshape(-1)
    else:
        # Placeholder 1-slot write when context is empty (shape residency).
        ctx_slot_mapping = bases
    noise_slot_mapping = (bases.unsqueeze(1) + (ctx_len + torch.arange(block, device=device, dtype=dtype))).reshape(-1)

    max_seq = total if ctx_len > 0 else block
    num_pages = max((max_seq + page_block_size - 1) // page_block_size, 1)
    block_table = torch.arange(num_pages, device=device, dtype=dtype).view(1, -1).expand(batch_size, -1)
    query_start_loc = torch.arange(0, (batch_size + 1) * block, block, device=device, dtype=dtype)
    # Constant per-request lens: use full (not arange*0+c) so traces stay free of
    # fake mul/add noise. batch_size may be SymInt under dynamic_shapes.
    seq_lens = torch.full((batch_size,), max_seq, device=device, dtype=dtype)
    query_lens = torch.full((batch_size,), block, device=device, dtype=dtype)

    # Match input_generator materialized fields when B is concrete. SymInt B must
    # not be int()-cast (dynamic_shapes); leave values None and rely on the
    # meta/FakeTensor guard in is_dcp_decode_batch.
    seq_lens_values = None
    query_lens_values = None
    is_decode_values = None
    if not isinstance(batch_size, torch.SymInt):
        b = int(batch_size)
        seq_lens_values = [max_seq] * b
        query_lens_values = [block] * b
        is_decode_values = [True] * b

    context_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_lens=query_lens,
        block_table_tensor=block_table,
        slot_mapping=ctx_slot_mapping,
        seq_lens_values=seq_lens_values,
        query_lens_values=query_lens_values,
        is_decode_values=is_decode_values,
        max_total_seq_len=max_seq,
    )
    noise_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_lens=query_lens,
        block_table_tensor=block_table,
        slot_mapping=noise_slot_mapping,
        seq_lens_values=seq_lens_values,
        query_lens_values=query_lens_values,
        is_decode_values=is_decode_values,
        max_total_seq_len=max_seq,
    )
    # Draft attention does not participate in target Decode Context Parallel.
    context_meta.is_dcp_decode = False
    noise_meta.is_dcp_decode = False
    return context_meta, noise_meta


def ensure_draft_kv_caches(
    kv_cache_by_layers: Optional[dict],
    *,
    layer_indices: Sequence[int],
    num_blocks: int,
    page_block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    """Ensure GQA-shaped draft caches exist (needed when target uses MLA caches).

    Prefer existing allocator/generator caches when already GQA-shaped (TP-correct
    head count). Only allocate when missing or clearly incompatible.
    """
    caches = dict(kv_cache_by_layers or {})
    shape = (2, num_blocks, page_block_size, num_kv_heads, head_dim)
    for idx in layer_indices:
        cur = caches.get(idx)
        if cur is not None and cur.ndim == 5 and int(cur.shape[0]) == 2 and int(cur.shape[-1]) == int(head_dim):
            # Keep existing (usually TP-sharded) cache from input_generator.
            continue
        caches[idx] = torch.empty(shape, dtype=dtype, device=device)
    return caches


class DflashDraftLayer(torch.nn.Module):
    def __init__(self, dflash_block: torch.nn.Module):
        super().__init__()
        self.dflash_block = dflash_block

    def forward(self, hidden_states, position_ids, position_embeddings=None, target_hidden=None, **kwargs):
        hidden_states = self.dflash_block(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states


class DflashDraftModel(torch.nn.Module):
    """Draft stack: fc/hidden_norm + one-shot context KV proj + Qwen3DFlash layers."""

    def __init__(
        self,
        draft_hf_config: Qwen3Config,
        dflash_config: DflashConfig,
        *,
        layer_idx_offset: int = 0,
    ):
        super().__init__()
        self.hf_config = draft_hf_config
        self.dflash_config = dflash_config
        self.num_draft_layers = dflash_config.num_draft_layers
        self.target_layer_ids = list(dflash_config.aux_hidden_state_layer_ids or [])
        self.num_selected_layers = dflash_config.num_selected_layers
        self.layer_types = list(dflash_config.layer_types or draft_hf_config.layer_types)
        self.sliding_window = dflash_config.sliding_window
        self.sliding_window_indices = [
            i for i, layer_type in enumerate(self.layer_types) if layer_type == "sliding_attention"
        ]
        self.context_length = int(dflash_config.context_length)

        hidden_size = draft_hf_config.hidden_size
        head_dim = getattr(draft_hf_config, "head_dim", hidden_size // draft_hf_config.num_attention_heads)
        self.head_dim = int(head_dim)
        self.num_key_value_heads = int(draft_hf_config.num_key_value_heads)
        self.kv_dim = self.num_key_value_heads * self.head_dim
        rms_norm_eps = getattr(draft_hf_config, "rms_norm_eps", 1e-5)
        self.fc = torch.nn.Linear(self.num_selected_layers * hidden_size, hidden_size, bias=False)
        # Qwen3RMSNorm (not torch.nn.RMSNorm) so --compile can fuse to tensor_cast.rms_norm.
        self.hidden_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        # One-shot context K/V: H → num_kv_heads × N_draft × 2 × head_dim (head-major).
        # Layout enables ColumnParallel head sharding so local out matches k/v_proj under TP
        # (NPU: MatMul → 1536 = local_kv_heads × N × 2 × 128; see operator_flow.md stage E).
        self.context_kv_proj = torch.nn.Linear(
            hidden_size,
            self.num_key_value_heads * self.num_draft_layers * 2 * self.head_dim,
            bias=False,
        )
        self.norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        # layer_idx = draft-local (layer_types); attention_layer_idx = global registry.
        self.layers = torch.nn.ModuleList(
            [
                DflashDraftLayer(
                    Qwen3DFlashDecoderLayer(
                        draft_hf_config,
                        layer_idx=i,
                        attention_layer_idx=layer_idx_offset + i,
                    )
                )
                for i in range(self.num_draft_layers)
            ]
        )
        # Snapshot before maybe_reuse_layers replaces non-representative layers with
        # CopyLayerWrapper (no children / no dflash_block). Plain lists keep refs
        # without double-registering modules on the nn.Module tree.
        self._draft_attn_layer_indices: list[int] = [
            int(layer.dflash_block.self_attn.layer_idx) for layer in self.layers
        ]
        self._draft_k_norms: list[torch.nn.Module] = [layer.dflash_block.self_attn.k_norm for layer in self.layers]
        self.rotary_emb = Qwen3RotaryEmbedding(draft_hf_config)
        # Placeholders until set_shared(); keeps module tree valid before attach.
        self.embed_tokens = torch.nn.Embedding(draft_hf_config.vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, draft_hf_config.vocab_size, bias=False)
        self._shared_vocab = False

    def set_shared(self, embed_tokens: torch.nn.Module, lm_head: torch.nn.Module) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self._shared_vocab = True

    def max_l_ctx(self) -> int:
        values = [
            resolve_l_ctx(self.context_length, self.layer_types[i], self.sliding_window)
            for i in range(self.num_draft_layers)
        ]
        return max(values) if values else max(int(self.context_length), 0)

    @staticmethod
    def _normalize_aux_hidden(hidden: torch.Tensor) -> torch.Tensor:
        """Normalize layer output to ``[B, S, H]``."""
        if hidden.dim() == 3:
            return hidden
        if hidden.dim() == 2:
            return hidden.unsqueeze(0)
        raise ValueError(f"Dflash aux hidden expects rank 2 or 3, got shape {tuple(hidden.shape)}")

    @staticmethod
    def _align_aux_seq(hidden: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Pad or truncate token axis to ``seq_len`` (keep rightmost tokens when truncating).

        Prefill uses ``context_length``-derived ``L_ctx``; Decode uses ``block_size`` so
        aux cat / fc / context_kv_proj stay short — long context lives in attention KV cache.

        Branch-free (left-pad ``seq`` zeros, then take the last ``seq`` tokens) so Dynamo
        does not guard on symbolic ``hidden.shape[1]`` (``Eq(u0, seq)`` / ``u0 > seq``).

        No ``hidden.sum()*0`` taint on pad: ``tensor_cast.cat([pad, hidden])`` already
        keeps a real edge to ``hidden``; the taint only injected fake sum/mul/add noise
        after target ArgMax in chrome-trace.
        """
        seq = max(int(seq_len), 1)
        # [B, seq, H] zeros + [B, cur, H] → [B, seq+cur, H] → last seq tokens.
        pad = hidden.new_zeros(hidden.shape[0], seq, hidden.shape[2])
        combined = torch.ops.tensor_cast.cat([pad, hidden], 1)
        return combined[:, -seq:, :]

    def build_context_features(
        self,
        aux_hiddens: Sequence[torch.Tensor],
        seq_len: int,
        *,
        align_seq: bool = True,
    ) -> torch.Tensor:
        """Build ``target_context`` by concatenating target-layer aux hiddens.

        Each entry in ``aux_hiddens`` is one ``[B, S, H]`` tensor (wrapper ``as_bsh``
        contract). Modeling path synthesizes ``L_aux = len(target_layer_ids)`` clones
        from last_hidden; ids set length only. Emits ``tensor_cast.cat`` over ``L_aux``
        on the hidden dim (NPU ConcatD):

        - Prefill (``align_seq=True``): pad/truncate token axis to ``L_ctx`` then cat
        - Decode (``align_seq=False``): aux is already ``[B, block, H]`` — skip pad+cat
          on dim1 (avoids fake ``(B,block)+(B,block)→(B,2*block)`` noise after ArgMax)
        """
        if len(aux_hiddens) != len(self.target_layer_ids):
            raise ValueError(
                f"Expected {len(self.target_layer_ids)} aux hiddens for "
                f"target_layer_ids={self.target_layer_ids}, got {len(aux_hiddens)}"
            )
        if not aux_hiddens:
            raise ValueError("Dflash aux_hiddens must be non-empty")
        if align_seq:
            selected = [self._align_aux_seq(self._normalize_aux_hidden(h), seq_len) for h in aux_hiddens]
        else:
            selected = [self._normalize_aux_hidden(h) for h in aux_hiddens]
        return torch.ops.tensor_cast.cat(selected, -1)

    def _split_context_kv(
        self, target_hidden: torch.Tensor
    ) -> list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]:
        """fc→hidden_norm → one fused context KV matmul → per-layer (K,V).

        Fused layout is head-major: ``[num_kv_heads, N_draft, 2, head_dim]`` so TP
        ColumnParallel(head_num=num_kv_heads) yields local width matching k/v_proj.
        """
        fused = self.context_kv_proj(target_hidden)
        batch, seqlen, out_dim = fused.shape
        per_head = self.num_draft_layers * 2 * self.head_dim
        if out_dim % per_head != 0:
            raise RuntimeError(
                f"context_kv_proj out_dim={out_dim} not divisible by "
                f"N*2*head_dim={per_head}; check TP shard of draft.context_kv_proj"
            )
        local_kv_heads = out_dim // per_head
        # [B, S, local_kv_heads, N, 2, head_dim]
        packed = fused.view(batch, seqlen, local_kv_heads, self.num_draft_layers, 2, self.head_dim)
        local_kv_dim = local_kv_heads * self.head_dim
        out: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = []
        for i in range(self.num_draft_layers):
            l_ctx = resolve_l_ctx(self.context_length, self.layer_types[i], self.sliding_window)
            k_ctx = packed[:, :, :, i, 0, :].reshape(batch, seqlen, local_kv_dim)
            v_ctx = packed[:, :, :, i, 1, :].reshape(batch, seqlen, local_kv_dim)
            # l_ctx is a Python int. Avoid comparing it to symbolic target_hidden.shape[1]:
            # ``x[:, -n:, :]`` returns the full axis when n >= cur (PyTorch semantics).
            if l_ctx <= 0:
                layer_target = target_hidden[:, :1, :]
                k_ctx = k_ctx[:, :1, :]
                v_ctx = v_ctx[:, :1, :]
            else:
                layer_target = target_hidden[:, -l_ctx:, :]
                k_ctx = k_ctx[:, -l_ctx:, :]
                v_ctx = v_ctx[:, -l_ctx:, :]
            out.append((layer_target, (k_ctx, v_ctx)))
        return out

    @staticmethod
    def _get_layer_attn(layer: torch.nn.Module):
        """Resolve Qwen3DFlashAttention through RegionMarker / CopyLayer wrappers."""
        seen: set[int] = set()
        cur: Optional[torch.nn.Module] = layer
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            block_mod = None
            # Prefer direct / forwarded attr (RegionMarkerWrapper.__getattr__).
            try:
                block_mod = cur.dflash_block  # type: ignore[attr-defined]
            except AttributeError:
                block_mod = None
            if block_mod is not None and hasattr(block_mod, "self_attn"):
                return block_mod.self_attn
            if hasattr(block_mod := getattr(cur, "self_attn", None), "k_norm"):
                return block_mod
            inner = getattr(cur, "_inner", None)
            if inner is not None and inner is not cur:
                cur = inner
                continue
            # CopyLayerWrapper: no children; use the representative real layer.
            rep = getattr(cur, "representative", None)
            if rep is not None and rep is not cur:
                cur = rep
                continue
            break
        raise RuntimeError("DflashDraftLayer missing dflash_block for context RoPE")

    def _fuse_context_rope_and_cache(
        self,
        context_kv_by_layer: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]],
        *,
        batch: int,
        block: int,
        kv_cache_by_layers: dict,
        layer_indices: Sequence[int],
        page_block_size: int,
        device: torch.device,
        ref_tensor: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer k_norm → one fused context apply_rope_single → reshape_and_cache × N.

        Matches: Tile(L_ctx×N) → _triton_rope (K only) → ReshapeAndCacheNdKernel × N.
        Returns per-layer ``(k_bshd, v_bshd)`` already RoPE'd (for no-cache fallback).
        """
        head_dim = self.head_dim
        k_bhsd_list: list[torch.Tensor] = []
        v_bshd_list: list[torch.Tensor] = []
        # Prefer init-time k_norm refs (survive CopyLayerWrapper). Fall back to unwrap.
        k_norms = getattr(self, "_draft_k_norms", None)
        for i, layer in enumerate(self.layers):
            if k_norms is not None and i < len(k_norms):
                k_norm = k_norms[i]
            else:
                k_norm = self._get_layer_attn(layer).k_norm
            _target, (k_ctx, v_ctx) = context_kv_by_layer[i]
            bsz, ctx_len, kv_dim = k_ctx.shape
            num_kv_heads = max(1, int(kv_dim) // int(head_dim))
            k_bhsd = k_norm(k_ctx.view(bsz, ctx_len, num_kv_heads, head_dim)).transpose(1, 2)
            v_bshd = v_ctx.view(bsz, ctx_len, num_kv_heads, head_dim)
            k_bhsd_list.append(k_bhsd)
            v_bshd_list.append(v_bshd)

        seqs = [int(k.shape[2]) for k in k_bhsd_list]
        k_fused = torch.cat(k_bhsd_list, dim=2)  # [B, H, sum_S, D] == NPU Tile fuse
        # Tile position ids per layer (NPU: [L_ctx] → [L_ctx×N]).
        pos_parts = [torch.arange(s, device=device, dtype=torch.long) for s in seqs]
        fused_pos = torch.cat(pos_parts, dim=0).view(1, -1)
        cos_c, sin_c = self.rotary_emb(ref_tensor, fused_pos)
        # One full-length context RoPE on K only (no Q in context stage); not repeated inside Decoder.
        # apply_rope_single(..., is_neox=True) keeps default transpose_output=True: BHSD → BSHD.
        k_roped_bshd = torch.ops.tensor_cast.apply_rope_single(k_fused, cos_c, sin_c, True)

        prepared: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for i, seq_len in enumerate(seqs):
            k_i = k_roped_bshd[:, offset : offset + seq_len, :, :]
            v_i = v_bshd_list[i]
            offset += seq_len
            prepared.append((k_i, v_i))

            layer_idx = int(layer_indices[i])
            kv_cache = kv_cache_by_layers.get(layer_idx) if kv_cache_by_layers is not None else None
            if kv_cache is None:
                continue
            ctx_meta, _noise_meta = build_draft_attention_metadata(
                batch,
                seq_len,
                block,
                device=device,
                page_block_size=page_block_size,
            )
            torch.ops.tensor_cast.reshape_and_cache(
                _flatten_tokens(k_i),
                _flatten_tokens(v_i),
                kv_cache,
                ctx_meta.slot_mapping,
            )
        return prepared

    def prepare_context_kv(
        self,
        target_context: torch.Tensor,
        *,
        batch: int,
        block: int,
        attn_use_configured_context: bool = False,
        **kwargs,
    ) -> dict:
        """Aux→fc→context KV write (after noise embedding at the wrapper).

        Returns a prepared dict consumed by ``run_noise_decoder``.
        """
        # NPU: Concat aux (caller) → noise embed (caller) → fc → hidden_norm →
        # context_kv_proj → per-layer k_norm → fused context apply_rope_single →
        # reshape_and_cache × N_draft (= len(target_layer_ids) / draft layers).
        # Noise-token reshape_and_cache happens later inside Layer0 attention
        # (after Pre-LN + qkv_proj + RoPE), matching NPU #98 before FIA.
        target_hidden = self.hidden_norm(self.fc(target_context))
        context_kv_by_layer = self._split_context_kv(target_hidden)

        max_written_ctx = target_hidden.shape[1]

        page_block_size = 128
        layer_indices = list(getattr(self, "_draft_attn_layer_indices", []))
        if len(layer_indices) != len(self.layers):
            layer_indices = []
            for i, layer in enumerate(self.layers):
                try:
                    layer_indices.append(int(self._get_layer_attn(layer).layer_idx))
                except RuntimeError:
                    layer_indices.append(i)

        local_kv_heads = max(1, int(self.kv_dim) // int(self.head_dim))
        try:
            k_proj = self._get_layer_attn(self.layers[0]).k_proj
            out_f = None
            inner_lin = getattr(k_proj, "_inner", None)
            if inner_lin is not None:
                out_f = getattr(inner_lin, "out_features", None)
            if out_f is None:
                weight = getattr(k_proj, "weight", None)
                if weight is not None:
                    out_f = int(weight.shape[0])
            if out_f is None:
                out_f = getattr(k_proj, "out_features", self.kv_dim)
            local_kv_heads = max(1, int(out_f) // int(self.head_dim))
        except RuntimeError:
            pass

        max_attn_ctx = max_written_ctx
        if attn_use_configured_context:
            max_attn_ctx = max(self.max_l_ctx(), max_written_ctx)

        num_blocks = max((max_attn_ctx + block + page_block_size - 1) // page_block_size, 1) * max(batch, 1)
        kwargs = dict(kwargs)
        kwargs["kv_cache_by_layers"] = ensure_draft_kv_caches(
            kwargs.get("kv_cache_by_layers"),
            layer_indices=layer_indices,
            num_blocks=max(num_blocks, 1),
            page_block_size=page_block_size,
            num_kv_heads=local_kv_heads,
            head_dim=self.head_dim,
            dtype=target_context.dtype,
            device=target_context.device,
        )

        context_kv_roped = self._fuse_context_rope_and_cache(
            context_kv_by_layer,
            batch=batch,
            block=block,
            kv_cache_by_layers=kwargs["kv_cache_by_layers"],
            layer_indices=layer_indices,
            page_block_size=page_block_size,
            device=target_context.device,
            ref_tensor=target_hidden,
        )
        return {
            "context_kv_by_layer": context_kv_by_layer,
            "context_kv_roped": context_kv_roped,
            "kv_cache_by_layers": kwargs["kv_cache_by_layers"],
            "attn_use_configured_context": bool(attn_use_configured_context),
            "page_block_size": page_block_size,
            "batch": batch,
            "block": int(block),
        }

    def run_noise_decoder(
        self,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        prepared: dict,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Noise embed path + draft Decoder layers (after context KV is in cache)."""
        if position_embeddings is None:
            position_embeddings = self.rotary_emb(noise_embedding, position_ids)
        # Prefer noise tensor ranks over prepared["batch"] so symbolic B stays aligned.
        batch = noise_embedding.shape[0]
        block = int(prepared["block"])
        page_block_size = int(prepared["page_block_size"])
        context_kv_by_layer = prepared["context_kv_by_layer"]
        context_kv_roped = prepared["context_kv_roped"]
        attn_use_configured_context = bool(prepared["attn_use_configured_context"])

        kwargs = dict(kwargs)
        kwargs["kv_cache_by_layers"] = prepared["kv_cache_by_layers"]

        hidden_states = noise_embedding
        for i, layer in enumerate(self.layers):
            layer_target, _raw_kv = context_kv_by_layer[i]
            written_ctx_len = layer_target.shape[1]
            if attn_use_configured_context:
                attn_ctx_len = resolve_l_ctx(self.context_length, self.layer_types[i], self.sliding_window)
            else:
                attn_ctx_len = written_ctx_len
            _ctx_meta, noise_meta = build_draft_attention_metadata(
                batch,
                attn_ctx_len,
                block,
                device=noise_embedding.device,
                page_block_size=page_block_size,
            )
            hidden_states = layer(
                hidden_states,
                position_ids,
                position_embeddings=position_embeddings,
                target_hidden=layer_target,
                context_kv=context_kv_roped[i],
                draft_noise_attention_meta=noise_meta,
                **kwargs,
            )
        return self.norm(hidden_states)

    def forward(
        self,
        noise_embedding: torch.Tensor,
        target_context: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        attn_use_configured_context: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Compatibility entry: context KV prep then noise decoder.

        Prefer ``prepare_context_kv`` → noise embed → ``run_noise_decoder`` at the
        wrapper so chrome-trace matches NPU (aux/fc/cache before noise embedding).
        """
        batch = noise_embedding.shape[0]
        block = noise_embedding.shape[1]
        prepared = self.prepare_context_kv(
            target_context,
            batch=batch,
            block=block,
            attn_use_configured_context=attn_use_configured_context,
            **kwargs,
        )
        return self.run_noise_decoder(
            noise_embedding,
            position_ids,
            prepared,
            position_embeddings=position_embeddings,
            **kwargs,
        )


class DflashWrapper(ModelWrapperBase):
    """Prefill/Decode: target forward → synthesize L_aux → draft forward.

    Draft layout contract
    ---------------------
    After the target→draft boundary, all draft tensors are ``[B, S, H]`` (or
    ``[B, block]`` for ids). Serving may pack target as ``[1, B*S, H]``; convert
    **once** via :meth:`as_bsh`. Do not re-pack or pad/slice-reshape inside the
    noise decoder / residual path.
    """

    def __init__(
        self,
        dflash_config: DflashConfig,
        hf_config,
        model: torch.nn.Module,
        draft: DflashDraftModel,
        draft_hf_config: Qwen3Config,
        target_layers: Optional[torch.nn.ModuleList] = None,
    ):
        super().__init__(model)
        self.dflash_config = dflash_config
        self.hf_config = hf_config
        self.draft_hf_config = draft_hf_config
        self.dflash_block_size = dflash_config.dflash_block_size
        self.draft = draft
        self.rotary_emb = draft.rotary_emb
        self.sampler = Sampler()
        self._aux_layer_ids = list(dflash_config.aux_hidden_state_layer_ids or [])
        # Optional: kept for callers that still pass resolved ModuleList at build time.
        self._target_layers = target_layers

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        assert sampling_metadata is not None, "No sampling metadata given for Dflash"

        is_decode = self._is_decode_step(kwargs)
        batch_size = self._resolve_batch_size(input_ids, inputs_embeds, kwargs)
        # Draft decoder seq length is always dflash_block_size (not --query-length).
        block = self.dflash_block_size
        # Target hiddens follow packed [1, B*query_len]; decode query_len == block.
        tokens_per_req = block if is_decode else self._resolve_tokens_per_request(input_ids, kwargs, batch_size)
        target_out, aux_hiddens = self._run_target_collect_aux(
            input_ids,
            position_ids,
            inputs_embeds,
            batch_size=batch_size,
            tokens_per_req=tokens_per_req,
            **kwargs,
        )

        # Sample target early so chrome-trace ArgMax sits with the target path.
        # With SpecDecodeMetadata (Dflash/MTP): Sampler emits two verify ArgMax ops —
        # bonus [1,V]→[1] then specs [S,V]→[S] — matching NPU DFlash verify.
        next_tokens = None
        if isinstance(target_out, torch.Tensor):
            next_tokens = self.sampler(target_out, sampling_metadata)

        # Construct [B, block] ids directly — do not reshape packed target positions.
        block_position_ids = self._block_position_ids(
            batch_size,
            block,
            device=position_ids.device,
            dtype=position_ids.dtype,
        )

        # NPU order (decode): ConcatD(aux) → noise embedding → fc/hidden_norm/context
        # KV write ×N_draft → Layer0 Pre-LN → qkv → RoPE → noise reshape_and_cache → Attn.
        # Prefill: align aux token axis to L_ctx before ConcatD; Decode: already [B,block,H].
        aux_seq_len = block if is_decode else self.draft.max_l_ctx()
        target_context = self.draft.build_context_features(
            aux_hiddens,
            aux_seq_len,
            align_seq=not is_decode,
        )

        # Ids must depend on live sampler output. Pure torch.zeros lets
        # fold_meta_constants erase embed → Layer0 Pre-LN → qkv (Layer1+ stay
        # because they consume attention output).
        mask_tokens = self._draft_token_ids(
            batch_size,
            block,
            device=position_ids.device,
            dtype=torch.long,
            anchor_tokens=next_tokens if isinstance(next_tokens, torch.Tensor) else None,
            mask_token_id=int(getattr(self.draft_hf_config, "mask_token_id", 0) or 0),
        )
        # Preferred Python order: ConcatD → embed → context KV → decoder.
        # No order_barrier: under --compile these subgraphs may still be reordered.
        noise_embedding = self.draft.embed_tokens(mask_tokens)
        position_embeddings = self.rotary_emb(noise_embedding, block_position_ids)

        prepared = self.draft.prepare_context_kv(
            target_context,
            batch=batch_size,
            block=block,
            attn_use_configured_context=is_decode,
            **kwargs,
        )
        # Context reshape_and_cache × num_draft_layers (= L_aux typically), then decoder:
        # input_layernorm → q/k/v_proj → RoPE → noise reshape_and_cache → attention.
        draft_hidden = self.draft.run_noise_decoder(
            noise_embedding,
            block_position_ids,
            prepared,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # NPU DFlash: Index [block,H]→[block-1,H] then lm_head (exclude anchor).
        # DSpark overrides ``_exclude_anchor_from_lm_head`` to keep full block.
        draft_logits, _lm_seq = self._apply_draft_lm_head(draft_hidden, block)

        if not is_decode:
            # Prefill/TTFT: primary output is target logits. Return draft_logits as a
            # second formal graph output so ``--compile`` cannot DCE the draft subgraph
            # (unwrap to primary outside the compiled region in model_runner).
            return target_out, draft_logits

        assert next_tokens is not None, "Dflash decode requires target sampler output"
        return self._propose_draft_tokens(draft_hidden, draft_logits, batch_size, block, next_tokens)

    def _exclude_anchor_from_lm_head(self) -> bool:
        """DFlash proposes ``block_size-1`` tokens; DSpark uses full ``block_size``."""
        return True

    def _apply_draft_lm_head(
        self,
        draft_hidden: torch.Tensor,
        block: int,
    ) -> tuple[torch.Tensor, int]:
        """Project draft hidden to vocab; DFlash excludes the anchor position.

        ``draft_hidden`` must already be ``[B, block, H]`` (layout contract).
        """
        if self._exclude_anchor_from_lm_head():
            # Match NPU Index before lm_head: drop slot0 (anchor).
            hidden = draft_hidden[:, 1:, :]
            lm_seq = block - 1
        else:
            hidden = draft_hidden
            lm_seq = block
        return self.draft.lm_head(hidden), lm_seq

    def _propose_draft_tokens(
        self,
        draft_hidden: torch.Tensor,
        draft_logits_b: torch.Tensor,
        batch_size: int,
        block: int,
        next_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return draft argmax tokens without assembling bonus|specs (overridable by subclasses).

        Primary output is ``draft_tokens`` with shape ``[B, block-1]``. ``next_tokens`` is
        returned as a formal side output so ``torch.compile`` keeps the verify sampler path.
        """
        # draft_logits_b is already [B, block-1, V] after anchor-excluding lm_head.
        draft_tokens = torch.argmax(draft_logits_b, dim=-1)
        return draft_tokens, next_tokens

    @staticmethod
    def as_bsh(tensor: torch.Tensor, batch, seq) -> torch.Tensor:
        """Single target→draft layout gate: ``[1,B*S,H]`` / ``[B*S,H]`` / ``[B,S,H]`` → ``[B,S,H]``.

        Call once at the boundary. Draft internals must not re-normalize packed layouts.
        """
        return tensor.reshape(batch, seq, tensor.shape[-1])

    # Back-compat alias used by older tests / callers.
    _unpack_target_hidden_for_draft = as_bsh

    def _synthesize_modeling_aux_hiddens(
        self,
        last_hidden: torch.Tensor,
        *,
        batch_size: int,
        tokens_per_req: int,
    ) -> list[torch.Tensor]:
        """Build ``L_aux`` buffers: ``[as_bsh(last).clone() for _ in aux_ids]``.

        - No ``output_hidden_states=True`` (compile-friendly; keeps layer reuse).
        - ``aux_hidden_state_layer_ids`` only sets length / validates config — no gather.
        - ``as_bsh`` is the single packed→``[B,S,H]`` gate; clones give independent
          storages so ``MemoryTracker`` counts multi-layer aux residency.
        """
        if not self._aux_layer_ids:
            raise RuntimeError("Dflash aux_hidden_state_layer_ids must be non-empty")
        if not isinstance(last_hidden, torch.Tensor):
            raise RuntimeError(f"Dflash expected target intermediate hidden Tensor, got {type(last_hidden)}")
        ref = self.as_bsh(last_hidden, batch_size, tokens_per_req)
        # ids participate in length only — not real per-layer gather.
        return [ref.clone() for _ in self._aux_layer_ids]

    def _run_target_collect_aux(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor],
        *,
        batch_size: int,
        tokens_per_req: int,
        **kwargs: object,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Target forward + modeling aux (MTP intermediate, not full hidden_states).

        Default path: ``output_intermediate_hidden_states=True`` then synthesize
        ``L_aux`` via :meth:`_synthesize_modeling_aux_hiddens`. Formal aux lists
        (if ever returned) are still passed through ``as_bsh`` once.
        """
        result = self._inner(
            input_ids,
            position_ids,
            inputs_embeds,
            output_intermediate_hidden_states=True,
            **kwargs,
        )
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError(
                "Dflash requires target forward to return "
                "(logits, intermediate_hidden_states); got a single tensor. "
                "Ensure CausalLmWrapper / ModelWrapper / VL supports "
                "output_intermediate_hidden_states."
            )
        target_out = result[0]
        second = result[1]

        def _normalize_aux_list(aux_list: list[torch.Tensor]) -> list[torch.Tensor]:
            if len(aux_list) != len(self._aux_layer_ids):
                raise RuntimeError(
                    f"Dflash expected {len(self._aux_layer_ids)} aux hiddens for "
                    f"layers {self._aux_layer_ids}, got {len(aux_list)}"
                )
            # One-time layout gate; values already distinct from formal path.
            return [self.as_bsh(h, batch_size, tokens_per_req) for h in aux_list]

        # Legacy formal path: (logits, aux_list) or (logits, hidden, aux_list).
        if isinstance(second, (list, tuple)) and second and isinstance(second[0], torch.Tensor):
            return target_out, _normalize_aux_list(list(second))
        if len(result) >= 3 and isinstance(result[2], (list, tuple)):
            return target_out, _normalize_aux_list(list(result[2]))
        return target_out, self._synthesize_modeling_aux_hiddens(
            second, batch_size=batch_size, tokens_per_req=tokens_per_req
        )

    @staticmethod
    def _block_position_ids(
        batch_size: int,
        block: int,
        *,
        device,
        dtype,
    ) -> torch.Tensor:
        """Build ``[B, block]`` position ids for draft noise (not from packed target pos)."""
        return torch.arange(block, device=device, dtype=dtype).view(1, block).expand(batch_size, block)

    @staticmethod
    def _draft_token_ids(
        batch_size: int,
        block: int,
        *,
        device,
        dtype=torch.long,
        anchor_tokens: Optional[torch.Tensor] = None,
        mask_token_id: int = 0,
    ) -> torch.Tensor:
        """Build ``[B, block]`` draft token ids ``[anchor | MASK*(block-1)]``.

        ``anchor_tokens`` should be the live target sampler output. Constant-only
        ids (e.g. ``torch.zeros``) are folded away under ``--compile`` together
        with the Layer0 embed / Pre-LN / qkv prologue.
        """
        if anchor_tokens is None:
            return torch.zeros(batch_size, block, device=device, dtype=dtype)
        tokens = torch.full(
            (batch_size, block),
            int(mask_token_id),
            device=device,
            dtype=dtype,
        )
        anchor = anchor_tokens.reshape(-1).to(device=device, dtype=dtype)
        if anchor.numel() <= 0:
            return tokens
        batch_index = torch.arange(batch_size, device=device) % anchor.numel()
        tokens = tokens.clone()
        tokens[:, 0] = anchor[batch_index]
        return tokens

    @staticmethod
    def _is_decode_step(kwargs: dict) -> bool:
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        return sampling_metadata is not None and sampling_metadata.selected_token_indices is None

    @staticmethod
    def _resolve_tokens_per_request(
        input_ids: Optional[torch.Tensor],
        kwargs: dict,
        batch_size: int,
    ):
        """Tokens per request in the packed target batch (prefill query_len).

        Prefer shape arithmetic over ``.item()`` so Dynamo can keep a symbolic
        ``B*Q // B`` expression instead of a data-dependent scalar guard.
        """
        if input_ids is not None:
            flat_n = input_ids.numel() if input_ids.dim() == 1 else input_ids.shape[-1]
            # When input is packed ``[1, B*Q]``, ``shape[-1] // batch`` yields Q.
            return flat_n // batch_size
        attn_meta = kwargs.get("attention_meta")
        query_lens = getattr(attn_meta, "query_lens", None) if attn_meta is not None else None
        if query_lens is not None and query_lens.numel() > 0:
            flat = query_lens.reshape(-1)
            if flat.device.type != "meta":
                return int(flat[0].item())
        raise ValueError("Unable to resolve tokens_per_req without input_ids or attention_meta")

    @staticmethod
    def _resolve_batch_size(
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        kwargs: dict,
    ) -> int:
        """Resolve request batch ``B`` for the ``[B, S, H]`` draft contract.

        Packed serving inputs are ``[1, B*Q]``; never treat dim0==1 as ``B=1``.
        Prefer ``attention_meta.query_lens`` / ``sampling_metadata.query_start_loc``.
        """
        attn_meta = kwargs.get("attention_meta")
        if attn_meta is not None and getattr(attn_meta, "query_lens", None) is not None:
            return attn_meta.query_lens.shape[0]
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        query_start_loc = getattr(sampling_metadata, "query_start_loc", None) if sampling_metadata is not None else None
        if query_start_loc is not None and query_start_loc.numel() >= 2:
            # query_start_loc is (B+1,); keep symbolic shape[0]-1 under dynamic_shapes.
            return query_start_loc.shape[0] - 1
        if input_ids is not None:
            if input_ids.dim() == 1:
                return 1
            # Unpacked [B, S] only; packed [1, B*Q] must have been handled above.
            return input_ids.size(0)
        if inputs_embeds is not None:
            if inputs_embeds.dim() >= 2 and inputs_embeds.size(0) == 1 and inputs_embeds.dim() == 2:
                # Ambiguous packed vs B=1; require meta when B>1.
                return 1
            return inputs_embeds.size(0)
        raise ValueError("Unable to resolve batch size for DflashWrapper")


def resolve_target_embed_and_lm_head(model) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Resolve target embed_tokens / lm_head for sharing (VL language tower aware)."""
    from ..layers.utils import ModelWrapperBase
    from ..transformers.custom_model_registry import get_vl_language_model

    # lm_head may live on an outer wrapper (e.g. CausalLmWrapper) while embed_tokens
    # stays on the inner backbone (e.g. Qwen3Model loaded via AutoModel.from_config).
    lm_head = None
    node = model
    while isinstance(node, ModelWrapperBase):
        wrapper_lm = getattr(node, "lm_head", None)
        if wrapper_lm is not None:
            lm_head = wrapper_lm
        node = node._inner

    unwrapped = node
    embed = getattr(unwrapped, "embed_tokens", None)
    if embed is None and hasattr(unwrapped, "get_input_embeddings"):
        embed = unwrapped.get_input_embeddings()
    if lm_head is None:
        lm_head = getattr(unwrapped, "lm_head", None)
    if lm_head is None and hasattr(unwrapped, "get_output_embeddings"):
        lm_head = unwrapped.get_output_embeddings()
    if embed is not None and lm_head is not None:
        return embed, lm_head

    language_model = get_vl_language_model(model)
    if language_model is not None:
        lm_head = getattr(language_model, "lm_head", None)
        inner = getattr(language_model, "model", language_model)
        embed = getattr(inner, "embed_tokens", None) or getattr(language_model, "embed_tokens", None)
        if embed is not None and lm_head is not None:
            return embed, lm_head

    raise ValueError(f"Unable to resolve target embed_tokens/lm_head from {type(unwrapped)}")


def build_dflash_draft_and_wrapper(
    model,
    dcfg: DflashConfig,
    hf_config,
    *,
    num_target_hidden_layers: int,
    target_hidden_size: int,
    target_vocab_size: int,
    target_max_position_embeddings: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    target_layers: Optional[torch.nn.ModuleList] = None,
) -> DflashWrapper:
    """Build draft from builtin/override config and wrap the target model."""
    # Keep CLI-resolved block_size / num_draft_layers; only fill missing aux ids etc.
    apply_cli_overrides_to_source_and_dcfg(dcfg, prefer_existing=True)
    draft_hf_config = build_draft_hf_config(
        dcfg,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        target_max_position_embeddings=target_max_position_embeddings,
    )
    max_layer = max(dcfg.aux_hidden_state_layer_ids)
    if max_layer >= num_target_hidden_layers:
        raise ValueError(
            f"target_layer_ids max={max_layer} out of range for target num_hidden_layers={num_target_hidden_layers}"
        )

    layer_idx_offset = int(num_target_hidden_layers)
    draft = DflashDraftModel(draft_hf_config, dcfg, layer_idx_offset=layer_idx_offset)
    embed, lm_head = resolve_target_embed_and_lm_head(model)
    draft.set_shared(embed, lm_head)
    if dtype is not None:
        # Keep shared modules on their existing dtype; only move draft-owned params.
        for name, module in draft.named_children():
            if name in ("embed_tokens", "lm_head"):
                continue
            module.to(dtype=dtype)
    return DflashWrapper(
        dcfg,
        hf_config,
        model._inner,
        draft,
        draft_hf_config,
        target_layers=target_layers,
    )

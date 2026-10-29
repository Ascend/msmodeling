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

"""qwen4_exp (Qwen3.8-Flash-Next) TensorCast adapter.

48-layer hybrid stack [GDN x3, QSA x1] x 12 with hyper-connections, per-layer
embedding (PLE + trigram hashing) and a 512-expert MoE. Patch surface:

- ``Qwen4ExpTextGatedDeltaNet.forward``: rerouted to the ``linear_attn_*``
  meta ops (identical projection layout to Qwen3.5: four independent
  ``in_proj_qkv/z/b/a`` projections).
- ``Qwen4ExpTextAttention``: wrapped by ``Qwen4ExpQSAAttentionWrapper`` so the
  compile-hostile indexer python loops and the HF ``Cache`` mutations are
  replaced by ``qsa_indexer`` / ``qsa_sparse_attention`` over TensorCast's
  paged caches.
- ``Qwen4ExpTextModel.reverse_embedding``: meta-safe dummy fallback (the
  official implementation uses ``nonzero``, which cannot run on meta tensors).

The official DecoderLayer/TextModel forward (hyper-connection mixing, PLE
injection, layer_type dispatch, mask mapping) is already structure-correct
and kwargs-transparent, so it runs unpatched. transformers 5.16.1 ships no
MTP classes for qwen4_exp (``_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]``),
so this profile registers no ``mtp_block_module_name``.
"""

import torch
import torch.nn.functional as F

from ...layers.qwen4_exp_attention import Qwen4ExpQSAAttentionWrapper
from ...model_config import MoEFieldNames
from ...utils import exact_division
from ..custom_model_registry import (
    ModelProfile,
    register_model_profile,
    resolve_visual_config,
)
from ..utils import has_previous_linear_attention_state, is_recurrent_linear_attention_decode_batch

QWEN4_EXP_DECODER_LAYER_NAME = "Qwen4ExpTextDecoderLayer"
QWEN4_EXP_VISUAL_CONFIG = resolve_visual_config({})


def _set_qwen4_exp_linear_attn_tp_size(model):
    tp_size = model.parallel_group_manager.tp_group.world_size
    if tp_size <= 1:
        return

    for module in model._inner.modules():
        if hasattr(module, "num_k_heads") and hasattr(module, "num_v_heads"):
            if module.num_k_heads % tp_size != 0:
                raise ValueError(
                    "Qwen4Exp linear attention requires tp_size to divide "
                    f"num_k_heads exactly, but got num_k_heads={module.num_k_heads} "
                    f"and tp_size={tp_size}."
                )
            if module.num_v_heads % tp_size != 0:
                raise ValueError(
                    "Qwen4Exp linear attention requires tp_size to divide "
                    f"num_v_heads exactly, but got num_v_heads={module.num_v_heads} "
                    f"and tp_size={tp_size}."
                )
            if module.head_k_dim != module.head_v_dim:
                raise ValueError(
                    "Qwen4Exp linear attention TP sharding requires head_k_dim to equal "
                    f"head_v_dim, but got head_k_dim={module.head_k_dim} and "
                    f"head_v_dim={module.head_v_dim}."
                )
            module.tensor_cast_tp_size = tp_size


def _get_local_linear_attn_heads(self):
    tp_size = getattr(self, "tensor_cast_tp_size", 1)
    return (
        exact_division(self.num_k_heads, tp_size),
        exact_division(self.num_v_heads, tp_size),
    )


def _patched_linear_attn_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    **kwargs,
):
    """GDN forward rerouted to linear_attn meta ops (Qwen3.5 layout).

    Qwen4Exp shares Qwen3.5's four independent projections (``in_proj_qkv`` /
    ``in_proj_z`` / ``in_proj_b`` / ``in_proj_a``) plus the FLA-style l2norm
    delta rule, so the numeric body matches ``patch_method_for_qwen3_5``.
    """
    local_num_k_heads, local_num_v_heads = _get_local_linear_attn_heads(self)
    batch_size, seq_len, _ = hidden_states.shape
    has_previous_state = has_previous_linear_attention_state(
        cache_params,
        cache_position,
        self.layer_idx,
    )
    use_recurrent = has_previous_state and is_recurrent_linear_attention_decode_batch(seq_len, cache_position)
    flatten_decode_batch = use_recurrent and seq_len != 1

    if attention_mask is not None:
        hidden_states = torch.ops.tensor_cast.linear_attn_apply_padding_mask(hidden_states, attention_mask)

    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, local_num_v_heads, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    core_batch_size = batch_size
    core_seq_len = seq_len
    if flatten_decode_batch:
        core_batch_size = batch_size * seq_len
        core_seq_len = 1
        mixed_qkv = mixed_qkv.reshape(core_batch_size, core_seq_len, -1)
        z = z.reshape(core_batch_size, core_seq_len, local_num_v_heads, self.head_v_dim)
        b = b.reshape(core_batch_size, core_seq_len, -1)
        a = a.reshape(core_batch_size, core_seq_len, -1)

    conv_op = (
        torch.ops.tensor_cast.linear_attn_causal_conv_update
        if use_recurrent
        else torch.ops.tensor_cast.linear_attn_causal_conv
    )
    mixed_qkv = mixed_qkv.transpose(1, 2)
    mixed_qkv = conv_op(
        mixed_qkv,
        self.conv_kernel_size,
    )
    mixed_qkv = mixed_qkv.transpose(1, 2)

    key_dim = local_num_k_heads * self.head_k_dim
    value_dim = local_num_v_heads * self.head_v_dim
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(core_batch_size, core_seq_len, local_num_k_heads, self.head_k_dim)
    key = key.reshape(core_batch_size, core_seq_len, local_num_k_heads, self.head_k_dim)
    value = value.reshape(core_batch_size, core_seq_len, local_num_v_heads, self.head_v_dim)

    query, key, beta, g = torch.ops.tensor_cast.linear_attn_fused_gdn_gating(
        query,
        key,
        b,
        a,
        self.A_log,
        self.dt_bias,
        local_num_v_heads,
    )

    if use_recurrent:
        core_attn_out = torch.ops.tensor_cast.linear_attn_recurrent_gated_delta_rule(query, key, value, beta, g, 1, 1)
    else:
        chunk_size = kwargs.get("chunk_size", 64)
        state_read_passes = 1 if has_previous_state else 0
        state_write_passes = 1
        core_attn_out = torch.ops.tensor_cast.linear_attn_chunk_gated_delta_rule(
            query,
            key,
            value,
            beta,
            g,
            chunk_size,
            state_read_passes,
            state_write_passes,
        )

    norm_weight = getattr(self.norm, "weight", None)
    core_attn_out = torch.ops.tensor_cast.linear_attn_gated_rmsnorm(
        core_attn_out,
        z,
        norm_weight,
        self.layer_norm_epsilon,
    )
    if flatten_decode_batch:
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, local_num_v_heads, self.head_v_dim)
    core_attn_out = core_attn_out.reshape(batch_size * seq_len, -1)
    output = self.out_proj(core_attn_out)
    return output.reshape(batch_size, seq_len, -1)


def _patched_reverse_embedding(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """Meta-safe replacement for the official ``reverse_embedding``.

    The official implementation reverses the embedding matrix via ``nonzero``
    (data-dependent, meta-unfriendly). Under TensorCast shape simulation the
    recovered ids never influence tensor shapes, so fabricate EOS-filled ids
    of the correct shape. The PLE layer saturates padding positions to EOS
    anyway, so this also matches the runtime padding semantics.
    """
    eos_token_id = getattr(self.config, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    if eos_token_id is None:
        eos_token_id = 0
    return torch.full(
        inputs_embeds.shape[:2],
        int(eos_token_id),
        dtype=torch.long,
        device=inputs_embeds.device,
    )


def _patched_gated_residual_forward(self, hyper_input: torch.Tensor):
    """HC mixer rerouted to qwen4 meta ops (official ``Qwen4ExpTextGatedResidual``).

    Boundary: the grouped RMSNorm and the sigmoid-gate -> per-stream
    multiply -> stream-mean reduction become fused ops; the lowrank
    ``input_mix_weight_down``/``up`` projections, the intermediate
    ``silu(down / hc_count)`` activation and the tiny ``2 * sigmoid``
    injection activation stay as aten ops.
    """
    hc_hidden_size = self.hc_count * self.hidden_size
    if hyper_input.shape[-1] != hc_hidden_size:
        raise ValueError(f"Expected {hc_hidden_size} hyper-connection features, got {hyper_input.shape[-1]}.")
    hyper_input_normed = torch.ops.tensor_cast.grouped_rmsnorm(
        hyper_input,
        self.hc_norm.weight,
        self.hidden_size,
        self.hc_norm.eps,
    )
    input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
    input_mix_weight = self.input_mix_weight_up(input_mix_weight)
    mixed_input = torch.ops.tensor_cast.hc_gated_stream_mean(
        input_mix_weight,
        hyper_input_normed,
        self.hc_count,
    )
    if self.block_inject_weight is None:
        return mixed_input
    inj_pre = self.block_inject_weight(hyper_input_normed)
    injection_weights = 2 * torch.sigmoid(inj_pre / self.hc_count)
    return mixed_input, hyper_input, injection_weights


def _patched_ngram_embedding_forward(
    self,
    input_ids: torch.Tensor,
    past_key_values=None,
) -> torch.Tensor:
    """N-gram hash rerouted to the ``qwen4_ngram_hash`` meta op.

    Boundary: EOS context padding, EOS-segment-aware shifts, the int64 XOR
    hash, per-head prime modulo and the table offset are fused into the op;
    the embedding table lookup stays the surrounding ``nn.Embedding``
    (aten ``embedding``). Under TensorCast the hash runs prefill-from-scratch
    semantics over the given ``[B, T]`` window (``past_key_values`` is not
    threaded).
    """
    del past_key_values
    ngram_ids = torch.ops.tensor_cast.qwen4_ngram_hash(
        input_ids.long(),
        self.layer_multipliers,
        self.ngram_heads_vocab_sizes,
        self.ngram_heads_offsets,
        self.ngram_size,
        self.heads_per_ngram,
        self.eos_token_id,
    )
    return self.ngram_embedding(ngram_ids).flatten(-2)


def _patched_ple_forward(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    past_key_values=None,
    conv_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """PLE layer rerouted to qwen4 meta ops (official ``Qwen4ExpTextPLELayer``).

    Boundary: the three grouped RMSNorms, the stream gating (dot ->
    sign-preserving sqrt -> sigmoid -> value broadcast) and the causal
    depthwise dilated conv + SiLU become fused ops; ``key_proj`` /
    ``value_proj``, the n-gram embedding and the residual add stay as aten
    ops. ``conv_mask`` is the 2D padding mask from the official
    ``create_recurrent_attention_mask``.
    """
    del past_key_values
    embeddings = self.ple_embedding(input_ids, None)

    key_normed = torch.ops.tensor_cast.grouped_rmsnorm(
        self.key_proj(embeddings),
        self.norm_key.weight,
        self.hidden_size,
        self.norm_key.eps,
    ).unflatten(-1, (self.hc_count, self.hidden_size))
    value = self.value_proj(embeddings)
    query_normed = torch.ops.tensor_cast.grouped_rmsnorm(
        hidden_states,
        self.norm_query.weight,
        self.hidden_size,
        self.norm_query.eps,
    ).unflatten(-1, (self.hc_count, self.hidden_size))

    gated_value = torch.ops.tensor_cast.qwen4_ple_gate(
        key_normed,
        query_normed,
        value,
        self.hidden_size,
    )
    gated_value_normed = torch.ops.tensor_cast.grouped_rmsnorm(
        gated_value,
        self.norm_conv.weight,
        self.hidden_size,
        self.norm_conv.eps,
    )
    if conv_mask is not None:
        gated_value = gated_value * conv_mask[:, :, None].to(gated_value.dtype)
        gated_value_normed = gated_value_normed * conv_mask[:, :, None].to(gated_value_normed.dtype)
    conv_out = torch.ops.tensor_cast.qwen4_ple_short_conv(
        gated_value_normed,
        self.conv1d.kernel_size[0],
        self.conv1d.dilation[0],
    )
    return gated_value + conv_out


def _patch_qwen4_exp_qsa_attention(model, text_config):
    hidden_size = text_config.hidden_size
    head_dim = getattr(text_config, "head_dim", hidden_size // text_config.num_attention_heads)
    indexer_n_heads = text_config.indexer_n_heads
    indexer_kv_heads = getattr(text_config, "indexer_kv_heads", 1)
    indexer_head_dim = text_config.indexer_head_dim
    token_budget = text_config.indexer_budget
    compress_ratio = text_config.indexer_compress_ratio
    rotary_dim = int(head_dim * getattr(text_config, "partial_rotary_factor", 1.0))

    tp_size = 1
    tp_rank = 0
    if model.parallel_group_manager is not None and model.parallel_group_manager.tp_group is not None:
        tp_size = model.parallel_group_manager.tp_group.world_size
        tp_rank = model.parallel_group_manager.tp_group.rank_in_group
    if indexer_n_heads % tp_size != 0:
        raise ValueError(
            "Qwen4Exp QSA indexer requires tp_size to divide indexer_n_heads "
            f"exactly, but got indexer_n_heads={indexer_n_heads} and tp_size={tp_size}."
        )
    per_rank_indexer_heads = indexer_n_heads // tp_size
    indexer_head_start = tp_rank * per_rank_indexer_heads

    patched_count = 0
    for _, module in list(model._inner.named_modules()):
        if type(module).__name__ != QWEN4_EXP_DECODER_LAYER_NAME:
            continue
        inner = module
        while hasattr(inner, "_inner"):
            inner = inner._inner
        self_attn = getattr(inner, "self_attn", None)
        if self_attn is None or isinstance(self_attn, Qwen4ExpQSAAttentionWrapper):
            continue

        inner.self_attn = Qwen4ExpQSAAttentionWrapper(
            original_module=self_attn,
            hidden_size=hidden_size,
            head_dim=head_dim,
            indexer_n_heads=per_rank_indexer_heads,
            indexer_kv_heads=indexer_kv_heads,
            indexer_head_dim=indexer_head_dim,
            token_budget=token_budget,
            compress_ratio=compress_ratio,
            rotary_dim=rotary_dim,
            indexer_head_start=indexer_head_start,
        )
        patched_count += 1
    return patched_count


def patch_method_for_qwen4_exp(model):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpModel,
        Qwen4ExpTextGatedDeltaNet,
        Qwen4ExpTextGatedResidual,
        Qwen4ExpTextModel,
        Qwen4ExpTextNGramEmbedding,
        Qwen4ExpTextPLELayer,
    )

    original_get_placeholder_mask = Qwen4ExpModel.get_placeholder_mask

    def _patched_get_placeholder_mask(self, *args, **kwargs):
        # In meta/simulation runs, HF strict image/video_features-vs-token-count
        # checks call Tensor.item() on meta tensors; skip them by forcing the
        # features to None (the placeholder masks are still computed from
        # input_ids, which is all masked_scatter needs).
        kwargs["image_features"] = None
        kwargs["video_features"] = None
        return original_get_placeholder_mask(self, *args, **kwargs)

    # Model-specific monkey patches required for TensorCast simulation:
    # - GDN op routing onto linear_attn meta ops (compile/meta stability)
    # - QSA attention wrapper (indexer loops + HF Cache -> paged meta ops)
    # - reverse_embedding meta-safe fallback
    # - HC mixer / N-gram hash / PLE fused kernels (qwen4_* meta ops)
    # - VL placeholder-mask relaxation for meta execution
    Qwen4ExpTextGatedDeltaNet.forward = _patched_linear_attn_forward
    Qwen4ExpTextModel.reverse_embedding = _patched_reverse_embedding
    Qwen4ExpModel.get_placeholder_mask = _patched_get_placeholder_mask
    Qwen4ExpTextGatedResidual.forward = _patched_gated_residual_forward
    Qwen4ExpTextNGramEmbedding.forward = _patched_ngram_embedding_forward
    Qwen4ExpTextPLELayer.forward = _patched_ple_forward
    patched_count = _patch_qwen4_exp_qsa_attention(model, model.text_config)
    _set_qwen4_exp_linear_attn_tp_size(model)
    return patched_count


register_model_profile(
    ModelProfile(
        model_type="qwen4_exp",
        moe_module_name="Qwen4ExpTextSparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key=["text_config", "num_experts"],
        moe_field_names_override=MoEFieldNames(
            shared_experts="shared_expert",
            shared_experts_gate="shared_expert_gate",
        ),
        # Reuses the Qwen3.5 linear-attention TP plan (four independent
        # in_proj_qkv/z/b/a + out_proj) and the same MoE gate 3-tuple contract.
        model_family="qwen3_5",
        patch_method=patch_method_for_qwen4_exp,
        **QWEN4_EXP_VISUAL_CONFIG,
    )
)

# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the software at
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Bailing MoE V3 VL (``bailing_moe_v3_vl``) TensorCast model profile.

Structure (checkpoint ``inclusionAI/Ling-3.0-flash-VL``, 62,563 tensors):

- ``model.visual``  27-layer Qwen3 ViT, parameter-free merger (Norm +
  2x2 reshape), no deepstack, root-level ``linear_proj`` projector.
- ``model.layers``  42 decoder layers; MLA at layers 5/11/17/23/29/35/41,
  KDA (linear attention) elsewhere; MoE from layer 2.
- Root-level ``linear_proj.0/.2`` and ``lm_head``.

The HF-side modeling is the official self-contained remote code vendored in
``tests/assets/model_config/ling_3_0_flash_vl/modeling_bailing_moe_v3_vl.py``
(byte-identical to the official copy; the earlier 411-line self-built
thin shell was replaced by it on 2026-10-23). This profile wires the
TensorCast replacements: MLA wrappers, KDA wrappers, MoE routing, the vision
TP plan, and the runtime adaptations the official modeling needs for
structural simulation (vision attention dispatch, Conv3d patch-embed,
meta-mode placeholder/M-RoPE guards - see
``_patch_bailing_v3_vl_official_runtime``).
"""

import logging
import types

import torch
import torch.nn.functional as F

from ...layers import COLWISE_LINEAR, ROWWISE_LINEAR
from ..model import _EXTRA_TC_KWARGS_KEYS
from ..custom_model_registry import ModelProfile, register_model_profile
from .bailing_v3 import (
    BailingV3MultiheadLatentAttentionTensorCast,
    _get_mtp_attention_modules,
    _replace_bailing_v3_kda_modules,
    _strip_bailing_v3_native_mtp,
)

logger = logging.getLogger(__name__)


def _route_bailing_v3_vl_gate(
    gate,
    hidden_states,
    top_k,
    input_ids,
    moe_layer_idx,
    tp_size=1,
    tp_rank=0,
):
    """Route Bailing MoE gating through the fused tensor_cast op.

    The Bailing gate scores with sigmoid, adds ``expert_bias`` and applies a
    group-limited top-k before normalizing with ``routed_scaling_factor``.
    The fused ``moe_gating_top_k_sigmoid`` op carries the sigmoid+bias+scale
    contract (mirroring ``route_minimax_m3_gate``) so the gating call count
    is visible to run-through verification; the group-limited selection
    refinement belongs to the Phase 3 measured-B2B scope.
    """
    del input_ids, moe_layer_idx
    gate_weight = gate.weight
    router_input = torch.ops.aten.alias.default(hidden_states.view(-1, hidden_states.shape[-1]))
    router_logits = F.linear(router_input.to(gate_weight.dtype), gate_weight).float()

    if tp_size > 1:
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
        num_tokens = router_logits.shape[0]
        padding_tokens = (-num_tokens) % tp_size
        if padding_tokens:
            # Keep router rows aligned with the hidden-state padding used
            # by the fused MoE layer; padded rows contribute zero.
            router_logits = F.pad(router_logits, (0, 0, 0, padding_tokens))
        router_logits = torch.tensor_split(router_logits, tp_size, dim=0)[tp_rank]

    topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k_sigmoid(
        router_logits,
        top_k,
        getattr(gate, "routed_scaling_factor", 1.0),
        getattr(gate, "expert_bias", None),
    )
    return topk_indices, topk_weights.to(hidden_states.dtype)


def _ensure_transformers_compat_symbols():
    """Backfill transformers symbols the remote modeling may rely on."""
    import transformers

    compat = {}
    cache_utils = getattr(transformers, "cache_utils", None)
    if cache_utils is not None and not hasattr(transformers, "DynamicLayer"):
        dynamic_layer = getattr(cache_utils, "DynamicLayer", None)
        if dynamic_layer is not None:
            transformers.DynamicLayer = dynamic_layer
            compat["DynamicLayer"] = True
    return compat


def _patch_hf_config_for_bailing_moe_v3_vl(config, model_id=None):
    """Pre-load patch for the top-level ``bailing_moe_v3_vl`` config.

    Runs before the remote modeling import (see
    ``AutoModelConfigLoader._apply_hf_config_patches``): bridges
    transformers compatibility symbols, fixes the legacy
    ``rope_type == "default"`` lookup and clears the vision deepstack
    default that the checkpoint does not support.
    """
    del model_id
    if getattr(config, "model_type", None) != "bailing_moe_v3_vl":
        return False

    patched = False
    if _ensure_transformers_compat_symbols():
        patched = True

    import importlib.util

    import transformers.utils.import_utils as import_utils
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if not hasattr(import_utils, "is_torch_fx_available"):
        torch_fx_available = importlib.util.find_spec("torch.fx") is not None

        def is_torch_fx_available():
            return torch_fx_available

        import_utils.is_torch_fx_available = is_torch_fx_available
        patched = True

    text_config = config.text_config
    top_mrope_section = getattr(config, "mrope_section", None)
    if top_mrope_section is not None and getattr(text_config, "mrope_section", None) is None:
        # Bridge the top-level mrope_section (the real config.json declares
        # [8, 12, 12]) into text_config: the official backbone
        # (BailingMoeV3VLModel) initializes from config.text_config and the
        # rotary reads mrope_section from it (falling back to the [12, 10,
        # 10] default), so without the bridge the T-axis frequency span
        # silently shifts 8 -> 12 (review: zhenghaojie - the sums both
        # equal 32, so structural shapes stay identical and only Phase-3
        # numeric B2B would diverge).
        text_config.mrope_section = top_mrope_section
        patched = True
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "default" and "factor" not in rope_scaling:
        text_config.rope_scaling = None
        patched = True

    if "default" not in ROPE_INIT_FUNCTIONS:

        def default_rope_parameters(rope_config, device=None):
            partial_rotary_factor = getattr(rope_config, "partial_rotary_factor", 1.0)
            dim = int(rope_config.head_dim * partial_rotary_factor)
            base = rope_config.rope_theta
            positions = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
            return 1.0 / (base ** (positions / dim)), 1.0

        ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters
        patched = True

    vision_config = config.vision_config
    if getattr(vision_config, "deepstack_visual_indexes", None):
        # The checkpoint carries no deepstack weights; the official vision
        # config class defaults to [8, 16, 24] when the key is absent.
        vision_config.deepstack_visual_indexes = []
        patched = True

    return patched


def _collect_bailing_v3_vl_attention_modules(model, wrapper):
    """Collect attention modules under the ``ForConditionalGeneration.model`` stack."""
    unwrapped = model.unwrap()
    layers = getattr(unwrapped, "layers", None)
    if layers is None:
        layers = getattr(getattr(unwrapped, "model", None), "layers", ())
    attention_modules = []
    for layer in layers:
        attention = getattr(layer, "attention", None)
        if attention is None:
            attention = getattr(layer, "self_attn", None)
        attention_modules.append(attention)
    mtp = getattr(wrapper, "mtp", None)
    if mtp is not None:
        attention_modules.extend(_get_mtp_attention_modules(mtp))

    unique_modules = []
    seen_ids = set()
    for attention in attention_modules:
        if attention is not None and id(attention) not in seen_ids:
            seen_ids.add(id(attention))
            unique_modules.append(attention)
    return tuple(unique_modules)


def _bailing_vl_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _patch_bailing_v3_vl_official_runtime(decoder):
    """Class-level runtime patches for the official self-contained modeling.

    The official remote code (a) hard-codes flash-attn/SDPA inside
    ``Qwen3VLMoeVisionAttention`` instead of the HF ``ALL_ATTENTION_FUNCTIONS``
    dispatch, (b) computes patch-embed as unfold+F.linear to match SGLang
    numerics, and (c) runs value-dependent placeholder-count checks and the
    token-driven ``get_rope_index`` inside ``BailingMoeV3VLModel.forward``.

    TensorCast structural simulation needs:

    1. the vision attention core routed through
       ``flash_attention_forward`` so each of the 27 vision blocks is counted
       as ``tensor_cast.attention`` (``_tensor_cast_context`` is attached by
       ``patch_attention`` and is read from the module, so patching the class
       forward is enough);
    2. patch-embed executed through the ``Conv3d`` module so the vision chain
       keeps the single-Conv3D anchor (PR-8 P1-06); the official unfold+
       F.linear path stays the Phase-3 numerical B2B reference;
    3. ``get_placeholder_mask`` returning a CPU-concrete mask with the exact
       expected True count on meta tensors: the official ``mask.sum().item()``
       count check passes and ``masked_scatter`` itself stays shape-only
       (meta inputs accept a concrete mask);
    4. ``compute_3d_position_ids`` short-circuited on meta tensors to the
       3-axis-shared text baseline: ``get_rope_index`` is token-value-driven
       (``.tolist()`` / ``argwhere``) and cannot run structurally; the
       official implementation is the Phase-3 numerical B2B reference.

    All patches are idempotent and class-level; the vendored asset file stays
    byte-identical to the official copy.
    """
    visual = getattr(decoder, "visual", None)
    if visual is None:
        return
    vision_attention_cls = type(visual.blocks[0].attn)
    patch_embed_cls = type(visual.patch_embed)
    vl_model_cls = type(decoder)

    # --- 1. vision attention core -> TensorCast -------------------------------
    if not getattr(vision_attention_cls, "_bailing_vl_tc_vision_patched", False):

        def _tc_vision_attention_forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
            dim = self.dim
            num_heads = self.num_heads
            head_dim = self.head_dim
            input_was_2d = hidden_states.dim() == 2
            if input_was_2d:
                hidden_states = hidden_states.unsqueeze(1)
            seq_length = hidden_states.shape[0]
            x = hidden_states.transpose(0, 1).contiguous()  # [1, S, D]
            qkv = self.qkv(x)
            query_states, key_states, value_states = qkv.split([dim, dim, dim], dim=-1)
            query_states = query_states.reshape(seq_length, num_heads, head_dim)
            key_states = key_states.reshape(seq_length, num_heads, head_dim)
            value_states = value_states.reshape(seq_length, num_heads, head_dim)

            if position_embeddings is not None:
                cos, sin = position_embeddings
                # Inline the official rotary application (cat + rotate_half)
                # without the torch.compile wrapper so meta tensors stay
                # shape-only.
                if cos.size(-1) * 2 == query_states.shape[-1]:
                    cos = torch.cat([cos, cos], dim=-1)
                    sin = torch.cat([sin, sin], dim=-1)
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
                query_states = query_states * cos + _bailing_vl_rotate_half(query_states) * sin
                key_states = key_states * cos + _bailing_vl_rotate_half(key_states) * sin

            # [S, H, hd] -> [1, H, S, hd]: the dispatch contract used by the HF
            # vision attention interface (cf. transformers Qwen3VLVisionAttention).
            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)
            from tensor_cast.layers.attention import flash_attention_forward

            attn_output, _ = flash_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                is_causal=False,
                **kwargs,
            )
            attn_output = attn_output.reshape(1, seq_length, -1).contiguous()
            attn_output = self.proj(attn_output).reshape(seq_length, 1, -1).contiguous()
            return attn_output.squeeze(1) if input_was_2d else attn_output

        vision_attention_cls.forward = _tc_vision_attention_forward
        vision_attention_cls._bailing_vl_tc_vision_patched = True

    # --- 2. patch-embed -> Conv3d module call ---------------------------------
    if not getattr(patch_embed_cls, "_bailing_vl_tc_patch_embed_patched", False):

        def _tc_patch_embed_forward(self, hidden_states):
            target_dtype = self.proj.weight.dtype
            hidden_states = hidden_states.view(
                -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
            )
            hidden_states = hidden_states.to(dtype=target_dtype)
            hidden_states = self.proj(hidden_states)
            return hidden_states.view(-1, self.embed_dim)

        patch_embed_cls.forward = _tc_patch_embed_forward
        patch_embed_cls._bailing_vl_tc_patch_embed_patched = True

    # --- 3. placeholder mask: meta-safe exact-count concrete mask -------------
    if not getattr(vl_model_cls, "_bailing_vl_tc_placeholder_patched", False):
        original_get_placeholder_mask = vl_model_cls.get_placeholder_mask

        def _tc_get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
            image_mask, video_mask = original_get_placeholder_mask(
                self, input_ids, inputs_embeds, image_features=image_features, video_features=video_features
            )
            if inputs_embeds is not None and inputs_embeds.is_meta:
                # The official forward validates ``mask.sum().item() ==
                # embeds.numel()`` right after this call - a value dependency
                # meta tensors cannot run. Return a CPU-concrete mask with the
                # exact expected True count so the check passes while
                # ``masked_scatter`` stays shape-only.
                hidden = inputs_embeds.shape[-1]

                def _concrete_count_mask(n_tokens):
                    flat = torch.zeros(inputs_embeds.numel(), dtype=torch.bool)
                    flat[: n_tokens * hidden] = True
                    return flat.view(inputs_embeds.shape)

                if image_features is not None:
                    image_mask = _concrete_count_mask(image_features.shape[0])
                if video_features is not None:
                    video_mask = _concrete_count_mask(video_features.shape[0])
            return image_mask, video_mask

        vl_model_cls.get_placeholder_mask = _tc_get_placeholder_mask
        vl_model_cls._bailing_vl_tc_placeholder_patched = True

    # --- 4. M-RoPE positions: meta text baseline ------------------------------
    if not getattr(vl_model_cls, "_bailing_vl_tc_positions_patched", False):
        original_compute_3d_position_ids = vl_model_cls.compute_3d_position_ids

        def _tc_compute_3d_position_ids(
            self,
            input_ids,
            inputs_embeds,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
            past_key_values=None,
            mm_token_type_ids=None,
        ):
            has_multimodal = image_grid_thw is not None or video_grid_thw is not None
            if input_ids is not None and input_ids.is_meta and has_multimodal:
                # ``get_rope_index`` is token-value-driven (``.tolist()`` /
                # ``argwhere``); structural simulation only needs the
                # ``[3, B, L]`` contract with the shared text baseline.
                batch_size, seq_length = input_ids.shape
                positions = torch.arange(seq_length).view(1, 1, seq_length).expand(3, batch_size, seq_length)
                self.rope_deltas = torch.zeros(batch_size, 1, dtype=positions.dtype, device=positions.device)
                return positions
            return original_compute_3d_position_ids(
                self,
                input_ids,
                inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        vl_model_cls.compute_3d_position_ids = _tc_compute_3d_position_ids
        vl_model_cls._bailing_vl_tc_positions_patched = True


def _patch_bailing_v3_vl_model(model):
    """Runtime patch for the loaded Bailing V3 VL model.

    - Force the flash-attention-2 style 2D-mask contract on the decoder.
    - Replace every ``BailingMoeV3KimiDeltaAttention`` with the TensorCast
      KDA wrapper (visual subtree has none).
    - Propagate TensorCast runtime metadata (attention_meta,
      kv_cache_by_layers, ...) into the attention wrappers.
    """
    wrapper = model._inner
    if getattr(wrapper, "_bailing_v3_tc_kwargs_patched", False):
        return

    unwrapped = model.unwrap()
    # BailingMoeV3VLForConditionalGeneration: the decoder stack lives
    # under ``.model``; fall back to the wrapper itself for text-only
    # layouts.
    decoder = getattr(unwrapped, "model", unwrapped)
    if getattr(decoder, "_bailing_v3_tc_patched", False):
        return

    decoder._use_flash_attention_2 = True
    decoder._use_sdpa = False
    stripped_mtp_layers = _strip_bailing_v3_native_mtp(decoder)
    if stripped_mtp_layers:
        logger.warning(
            "Bailing V3 VL config declares %d native MTP layer(s) "
            "(num_nextn_predict_layers > 0; e.g. the internal SFT RC2 config, "
            "while the open Ling-3.0-flash-VL config ships 0). Structural "
            "simulation never runs the MTP forward path, so the MTP tail is "
            "stripped and the decoder stays at the 42 main layers + 27 vision "
            "blocks layout.",
            stripped_mtp_layers,
        )

    _replace_bailing_v3_kda_modules(decoder)
    _patch_bailing_v3_vl_official_runtime(decoder)
    decoder._bailing_v3_tc_patched = True

    original_forward = wrapper.forward

    def patched_forward(self, *args, **kwargs):
        extra_kwargs = {key: kwargs.get(key) for key in _EXTRA_TC_KWARGS_KEYS if kwargs.get(key) is not None}
        attention_modules = getattr(self, "_bailing_v3_tc_attention_modules", None)
        if attention_modules is None:
            attention_modules = _collect_bailing_v3_vl_attention_modules(model, self)
            self._bailing_v3_tc_attention_modules = attention_modules
        for attention in attention_modules:
            if attention is not None:
                attention._extra_forward_kwargs = extra_kwargs
        return original_forward(*args, **kwargs)

    wrapper.forward = types.MethodType(patched_forward, wrapper)
    wrapper._bailing_v3_tc_kwargs_patched = True

    # The official ForCondGen chain expects ModelOutput semantics internally
    # (``BailingMoeV3VLModel.forward`` accesses ``super().forward(...).
    # last_hidden_state``), but TensorCast's ModelWrapper hard-codes
    # ``return_dict=False`` and consumes ``result[0]``. Adapt the contract at
    # the ForCondGen boundary: strip ``return_dict`` so the official chain
    # stays ModelOutput-consistent, then convert the result back to the
    # ``(logits,)`` tuple the ModelWrapper expects.
    if not getattr(unwrapped, "_bailing_v3_tc_return_dict_patched", False):
        original_fc_forward = unwrapped.forward

        def patched_fc_forward(self, *args, **kwargs):
            kwargs.pop("return_dict", None)
            result = original_fc_forward(*args, **kwargs)
            if isinstance(result, tuple):
                return result
            logits = getattr(result, "logits", None)
            if logits is None:
                return result
            return (logits,)

        unwrapped.forward = types.MethodType(patched_fc_forward, unwrapped)
        unwrapped._bailing_v3_tc_return_dict_patched = True


register_model_profile(
    ModelProfile(
        model_type="bailing_moe_v3_vl",
        moe_module_name="BailingMoeV3SparseMoeBlock",
        mla_module_name="BailingMoeV3MultiLatentAttention",
        moe_gate_router=_route_bailing_v3_vl_gate,
        # The official SparseMoeBlock ships per-expert MLPs as a
        # ModuleList (checkpoint experts.N.gate/up/down_proj); the default
        # MoeExpertMLP adapter assumes packed 3D experts, so keep the
        # official layout (MoELayer consumes it directly), mirroring the
        # text-side bailing_hybrid profile.
        custom_expert_module_type=None,
        mla_field_names_override={"o_proj": "dense"},
        mla_module_class_type=BailingV3MultiheadLatentAttentionTensorCast,
        hf_config_patch_method=_patch_hf_config_for_bailing_moe_v3_vl,
        patch_method=_patch_bailing_v3_vl_model,
        visual_module_path="model.visual",
        visual_layers_path_str="model.visual.blocks",
        language_layers_path_str="model.layers",
        # The Bailing merger itself is Norm + reshape only, but the
        # root-level projector (linear_proj.0/.2, the visual→language
        # 4608→2560→2560 chain) belongs to the same "visual feature
        # merger/projector" family this mapping covers (cf. qwen3_vl's
        # merger linear_fc1/fc2): without a pattern it stays fully
        # replicated on every rank under TP/VTP>1 and the projector's TP
        # gain is systematically under-counted (review: Hudingyi).
        visual_merger_linear_mapping={
            "linear_proj.0": COLWISE_LINEAR,
            "linear_proj.2": ROWWISE_LINEAR,
        },
        visual_mlp_linear_mapping={
            "model.visual.blocks.*.mlp.linear_fc1": COLWISE_LINEAR,
            "model.visual.blocks.*.mlp.linear_fc2": ROWWISE_LINEAR,
        },
    )
)

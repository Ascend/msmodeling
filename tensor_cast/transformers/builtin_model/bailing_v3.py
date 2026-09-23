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

import importlib.util
import types

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # pylint: disable=no-name-in-module

from ...layers import COLWISE_LINEAR, ROWWISE_LINEAR
from ...layers.mla import MultiheadLatentAttentionTensorCast
from ...layers.utils import ModelWrapperBase
from ..model import _EXTRA_TC_KWARGS_KEYS
from ..custom_model_registry import ModelProfile, register_model_profile
from ..utils import replace_module


class BailingV3MultiheadLatentAttentionTensorCast(MultiheadLatentAttentionTensorCast):
    @classmethod
    def build_tp_plan_extras(cls, prefix, params, config_info):
        del config_info
        from ...layers.mla import tp_plan_nested_module_path

        return {
            tp_plan_nested_module_path(prefix, projection): (COLWISE_LINEAR, dict(params))
            for projection in (
                "q_proj",
                "k_proj",
                "v_proj",
                "f_proj",
                "f_a_proj",
                "f_b_proj",
                "g_proj",
                "g_a_proj",
                "g_b_proj",
                "b_proj",
                "kv_b_proj",
            )
        }

    @classmethod
    def build_o_proj_tp_plan_extras(cls, prefix, params, config_info):
        del config_info
        from ...layers.mla import tp_plan_nested_module_path

        return {
            tp_plan_nested_module_path(prefix, "dense"): (ROWWISE_LINEAR, dict(params)),
        }

    def forward(self, *args, **kwargs):
        extra_kwargs = getattr(self, "_extra_forward_kwargs", {})
        for key, value in extra_kwargs.items():
            if key not in kwargs and value is not None:
                kwargs[key] = value
        attention_output, attention_weights = super().forward(*args, **kwargs)
        return attention_output, attention_weights, None


class BailingV3KimiDeltaAttentionTensorCast(ModelWrapperBase):
    def __init__(self, attention):
        super().__init__(attention)
        self.layer_idx = attention.layer_idx
        self.head_dim = attention.head_dim
        self.conv_size = attention.conv_size
        self.no_kda_lora = attention.no_kda_lora

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, **kwargs):
        del attention_mask
        extra_kwargs = getattr(self, "_extra_forward_kwargs", {})
        for key, value in extra_kwargs.items():
            if key not in kwargs and value is not None:
                kwargs[key] = value

        attention_meta = kwargs.get("attention_meta")
        kv_cache_by_layers = kwargs.get("kv_cache_by_layers")
        if attention_meta is None or kv_cache_by_layers is None:
            raise ValueError(
                "KimiDeltaAttention requires attention_meta and kv_cache_by_layers; "
                "ensure TensorCast runtime metadata was propagated to the attention layer."
            )

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        if self.no_kda_lora:
            f = self.f_proj(hidden_states)
            gate = self.g_proj(hidden_states)
        else:
            f = self.f_b_proj(self.f_a_proj(hidden_states))
            gate = self.g_b_proj(self.g_a_proj(hidden_states))
        beta = self.b_proj(hidden_states)

        local_num_heads = q.shape[-1] // self.head_dim
        q = q.view(*q.shape[:-1], local_num_heads, self.head_dim)
        k = k.view(*k.shape[:-1], local_num_heads, self.head_dim)
        v = v.view(*v.shape[:-1], local_num_heads, self.head_dim)
        f = f.view(*f.shape[:-1], local_num_heads, self.head_dim)
        gate = gate.view(*gate.shape[:-1], local_num_heads, self.head_dim)

        if self.layer_idx not in kv_cache_by_layers:
            raise KeyError(
                f"KimiDeltaAttention layer_idx={self.layer_idx} is missing from kv_cache_by_layers "
                f"(available layer indices: {sorted(kv_cache_by_layers)})"
            )
        state = kv_cache_by_layers[self.layer_idx]
        output = torch.ops.tensor_cast.kimi_delta_attention_core(
            q,
            k,
            v,
            f,
            beta,
            gate,
            state,
            attention_meta.query_lens,
            attention_meta.seq_lens,
            self.conv_size,
        )
        output = self.o_proj(output.flatten(-2))
        return output, None, past_key_value


def _slice_bailing_v3_router_output(tensor, tp_size, tp_rank):
    if tp_size > 1:
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
        padding_tokens = (-tensor.shape[0]) % tp_size
        if padding_tokens:
            # Keep router rows aligned with the hidden-state padding in
            # ParallelMoELayer._dp_transform_enter().  The zero routing weights
            # make padded rows contribute zero, and _dp_transform_exit() removes
            # those rows after the equal-sized TP collective.
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_tokens))
        tensor = torch.tensor_split(tensor, tp_size, dim=0)[tp_rank]
    return tensor


def _route_bailing_v3_gate(
    gate,
    hidden_states,
    top_k,
    input_ids,
    moe_layer_idx,
    tp_size=1,
    tp_rank=0,
):
    del top_k, input_ids, moe_layer_idx
    topk_indices, topk_weights, _ = gate(hidden_states)
    topk_indices = _slice_bailing_v3_router_output(topk_indices, tp_size, tp_rank)
    topk_weights = _slice_bailing_v3_router_output(topk_weights, tp_size, tp_rank)
    return topk_indices, topk_weights


def _get_mtp_attention_modules(mtp):
    attention_modules = []
    for layer in mtp.layers:
        mtp_block = getattr(layer, "mtp_block", None)
        if mtp_block is None:
            mtp_block = getattr(getattr(layer, "representative", None), "mtp_block", None)
        attention_modules.append(getattr(mtp_block, "attention", None))
    return attention_modules


def _replace_bailing_v3_kda_modules(root):
    # Bailing model classes come from dynamically loaded remote code, so the
    # runtime class name is the stable compatibility contract here.
    for name, module in list(root.named_modules()):
        if type(module).__name__ == "BailingMoeV3KimiDeltaAttention":
            replace_module(root, name, BailingV3KimiDeltaAttentionTensorCast(module))


def _collect_bailing_v3_attention_modules(model, wrapper):
    attention_modules = []
    for layer in getattr(model.unwrap(), "layers", ()):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            attention = getattr(layer, "attention", None)
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


def _patch_bailing_v3_model(model):
    wrapper = model._inner
    if getattr(wrapper, "_bailing_v3_tc_kwargs_patched", False):
        return

    base_model = model.unwrap()
    base_model._use_flash_attention_2 = True
    base_model._use_sdpa = False
    native_mtp_layers = int(getattr(base_model, "num_nextn_predict_layers", 0) or 0)
    if native_mtp_layers:
        layers = list(base_model.layers)
        mtp_layers = layers[-native_mtp_layers:]
        mtp_layer_types = [type(layer.unwrap() if hasattr(layer, "unwrap") else layer).__name__ for layer in mtp_layers]
        if not all(layer_type == "BailingMoeV3MTPLayer" for layer_type in mtp_layer_types):
            raise ValueError("Bailing V3 native MTP layers do not match the expected model structure")
        base_model.layers = torch.nn.ModuleList(layers[:-native_mtp_layers])
        base_model.num_nextn_predict_layers = 0

    _replace_bailing_v3_kda_modules(base_model)
    mtp = getattr(wrapper, "mtp", None)
    if mtp is not None:
        _replace_bailing_v3_kda_modules(mtp)

    original_forward = wrapper.forward

    def patched_forward(self, *args, **kwargs):
        extra_kwargs = {key: kwargs.get(key) for key in _EXTRA_TC_KWARGS_KEYS if kwargs.get(key) is not None}
        attention_modules = getattr(self, "_bailing_v3_tc_attention_modules", None)
        if attention_modules is None:
            attention_modules = _collect_bailing_v3_attention_modules(model, self)
            self._bailing_v3_tc_attention_modules = attention_modules
        for attention in attention_modules:
            if attention is not None:
                attention._extra_forward_kwargs = extra_kwargs
        return original_forward(*args, **kwargs)

    wrapper.forward = types.MethodType(patched_forward, wrapper)
    wrapper._bailing_v3_tc_kwargs_patched = True


def _patch_hf_config_for_bailing_v3(config, model_id=None):
    del model_id
    if getattr(config, "model_type", None) != "bailing_hybrid":
        return False

    import transformers.utils.import_utils as import_utils  # pylint: disable=no-name-in-module

    patched = False

    if not hasattr(import_utils, "is_torch_fx_available"):
        torch_fx_available = importlib.util.find_spec("torch.fx") is not None

        def is_torch_fx_available():
            return torch_fx_available

        import_utils.is_torch_fx_available = is_torch_fx_available
        patched = True

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "default" and "factor" not in rope_scaling:
        config.rope_scaling = None
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

    return patched


register_model_profile(
    ModelProfile(
        model_type="bailing_hybrid",
        moe_module_name="BailingMoeV3SparseMoeBlock",
        mla_module_name="BailingMoeV3MultiLatentAttention",
        mtp_block_module_name="BailingMoeV3MTPLayer",
        moe_gate_router=_route_bailing_v3_gate,
        custom_expert_module_type=None,
        mla_field_names_override={"o_proj": "dense"},
        mla_module_class_type=BailingV3MultiheadLatentAttentionTensorCast,
        hf_config_patch_method=_patch_hf_config_for_bailing_v3,
        patch_method=_patch_bailing_v3_model,
    )
)

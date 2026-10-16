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

import torch
from transformers import AutoConfig, AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from tensor_cast.layers.attention import AttentionTensorCast

from ..custom_model_registry import ModelProfile, register_model_profile
from .mimo_v2_flash_hf.configuration_mimo_v2_flash import MiMoV2FlashConfig
from .mimo_v2_flash_hf.modeling_mimo_v2_flash import MiMoV2Model


_AUTO_MODEL_ALREADY_REGISTERED_MSG = "already used by a Transformers model"


def _safe_register_mimo_v2_flash_family() -> None:
    model_type = MiMoV2FlashConfig.model_type
    if model_type not in CONFIG_MAPPING:
        AutoConfig.register(model_type, MiMoV2FlashConfig)
    AutoModel.register(MiMoV2FlashConfig, MiMoV2Model, exist_ok=True)


_safe_register_mimo_v2_flash_family()


class MiMoV2MoeExpertMLP(torch.nn.Module):
    def __init__(self, expert_module_list, expert_idx=None):
        super().__init__()
        if expert_idx is not None:
            self.expert = expert_module_list[expert_idx]
        else:
            self.expert = expert_module_list[0]

    def forward(self, hidden_states):
        return self.expert(hidden_states)


class MiMoV2AttentionTensorCast(AttentionTensorCast):
    """TensorCast attention for MiMo V2's asymmetric QK/V head widths."""

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        kv_cache=None,
        attention_meta=None,
        **kwargs,
    ):
        qk_head_dim = kwargs.pop("_tensor_cast_qk_head_dim", None)
        v_head_dim = kwargs.pop("_tensor_cast_v_head_dim", None)
        if not isinstance(qk_head_dim, int) or not isinstance(v_head_dim, int):
            raise ValueError("MiMo V2 attention requires concrete qk and value head dimensions")

        sinks = kwargs.pop("sinks", None)
        if sinks is None:
            sinks = kwargs.pop("s_aux", None)
        if sinks is not None:
            if sinks.ndim != 1:
                raise ValueError(f"MiMo V2 attention sinks must be 1D, got shape {tuple(sinks.shape)}")
            local_query_heads = query.shape[-2] if query.ndim == 4 else query.shape[-1] // qk_head_dim
            if sinks.shape[0] < local_query_heads:
                raise ValueError(
                    "MiMo V2 attention has fewer sink values than local query heads: "
                    f"{sinks.shape[0]} < {local_query_heads}"
                )
            # q_proj is column-sharded by query head, while the standalone
            # sink Parameter is not part of the linear TP plan. TensorCast
            # models one representative rank, so retain the corresponding
            # local-head slice before invoking the fused attention semantic.
            sinks = sinks[:local_query_heads]

        query_start_loc = attention_meta.query_start_loc if attention_meta else None
        seq_lens = attention_meta.seq_lens if attention_meta else None
        query_lens = attention_meta.query_lens if attention_meta else None
        block_table = attention_meta.block_table_tensor if attention_meta else None

        if attention_meta is not None:
            if kv_cache is None:
                raise ValueError("MiMo V2 attention metadata requires a packed KV cache")
            expected_width = qk_head_dim + v_head_dim
            if kv_cache.shape[0] != 1 or kv_cache.shape[-1] != expected_width:
                raise ValueError(
                    "MiMo V2 packed KV cache must have shape [1, ..., D_qk + D_v], "
                    f"got {tuple(kv_cache.shape)} for D_qk={qk_head_dim}, D_v={v_head_dim}"
                )
            if self.quant_config is not None:
                kv_scale = self.quant_config.kv_scale
                kv_offset = self.quant_config.kv_offset
                key = torch.ops.tensor_cast.quantize(key, kv_scale, kv_offset, kv_cache.dtype)
                value = torch.ops.tensor_cast.quantize(value, kv_scale, kv_offset, kv_cache.dtype)
            if not (key.dtype == value.dtype == kv_cache.dtype):
                raise ValueError(
                    f"Expect key, value and kv_cache dtype match but got {key.dtype}, {value.dtype}, {kv_cache.dtype}"
                )
            torch.ops.tensor_cast.mimo_v2_reshape_and_cache(
                key,
                value,
                kv_cache,
                attention_meta.slot_mapping,
                qk_head_dim,
                v_head_dim,
            )
            packed_cache = kv_cache[0]
            key = packed_cache[..., :qk_head_dim]
            value = packed_cache[..., qk_head_dim:expected_width]

        out_dtype = query.dtype
        if self.quant_config is not None and attention_meta is not None:
            query = torch.ops.tensor_cast.quantize(
                query,
                self.quant_config.query_scale,
                self.quant_config.query_offset,
                kv_cache.dtype,
            )

        return torch.ops.tensor_cast.mimo_v2_attention(
            query,
            key,
            value,
            attention_mask,
            block_table,
            query_start_loc,
            seq_lens,
            query_lens,
            qk_head_dim,
            v_head_dim,
            sinks,
            out_dtype,
        )


def _patch_mimo_v2_attention(model) -> None:
    model.model_config.attention_cls = MiMoV2AttentionTensorCast


register_model_profile(
    ModelProfile(
        model_type="mimo_v2_flash",
        moe_module_name="MiMoV2MoE",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key="n_routed_experts",
        mtp_block_module_name="MiMoV2DecoderLayer",
        custom_expert_module_type=MiMoV2MoeExpertMLP,
        patch_method=_patch_mimo_v2_attention,
    )
)

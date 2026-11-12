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

"""Official MiniMax-M3.1 DSpark dense draft, composed over the DFlash stack.

Source: MiniMaxAI/MiniMax-M3.1-Flash/dspark/config.json and vLLM's
models/minimax_m3/common/dspark.py. This is metadata/performance modeling,
not checkpoint loading or numerical generation.
"""

from pathlib import Path

import torch

from .dflash_qwen3 import Qwen3DFlashAttention
from .dspark import DsparkDraftModel, DsparkWrapper
from .minimax_m3_attention import GemmaRMSNormFusedWrapper
from .minimax_m31_draft_config import normalize_minimax_draft_config  # noqa: F401 (compatibility export)

MINIMAX_M31_DRAFT_CONFIG = (
    Path(__file__).resolve().parents[1] / "runtime_configs" / "draft_configs" / "minimax_m31_dspark.json"
)


def _gemma_norm(size, eps):
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLRMSNorm

    return GemmaRMSNormFusedWrapper(MiniMaxM3VLRMSNorm(size, eps=eps))


class MiniMaxM31DraftAttention(Qwen3DFlashAttention):
    """Context-injected GQA with Gemma QK norms and partial M3 RoPE."""

    def __init__(self, config, layer_idx, *, type_layer_idx):
        super().__init__(config, layer_idx, type_layer_idx=type_layer_idx)
        self.q_norm = _gemma_norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _gemma_norm(self.head_dim, config.rms_norm_eps)
        self.rotary_dim = int(config.rotary_dim)

    def _apply_noise_rope(self, query, key, cos, sin):
        batch, _, seq, dim = query.shape
        q = query.transpose(1, 2).reshape(batch * seq, -1, dim)
        k = key.transpose(1, 2).reshape(batch * seq, -1, dim)
        cos_sin = torch.cat([cos, sin], dim=-1).reshape(batch * seq, -1)
        q, k = torch.ops.tensor_cast.fused_rope(q, k, cos_sin, self.rotary_dim, True)
        return q.reshape(batch, seq, -1, dim), k.reshape(batch, seq, -1, dim)


class MiniMaxM31DsparkDraft(DsparkDraftModel):
    """Five dense sliding-attention blocks plus Markov/confidence heads."""

    def __init__(self, config, dcfg, scfg, *, layer_idx_offset=0):
        super().__init__(config, dcfg, scfg, layer_idx_offset=layer_idx_offset)
        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLDenseMLP

        from ..transformers.builtin_model.minimax_m3 import MiniMaxM3DenseMLPWrapper

        self.hidden_norm = _gemma_norm(config.hidden_size, config.rms_norm_eps)
        self.norm = _gemma_norm(config.hidden_size, config.rms_norm_eps)
        for i, layer in enumerate(self.layers):
            block = layer.dflash_block
            block.self_attn = MiniMaxM31DraftAttention(config, layer_idx_offset + i, type_layer_idx=i)
            block.input_layernorm = _gemma_norm(config.hidden_size, config.rms_norm_eps)
            block.post_attention_layernorm = _gemma_norm(config.hidden_size, config.rms_norm_eps)
            block.mlp = MiniMaxM3DenseMLPWrapper(MiniMaxM3VLDenseMLP(config))
        self._draft_k_norms = [layer.dflash_block.self_attn.k_norm for layer in self.layers]

    def _apply_context_rope(self, key, cos, sin):
        rotary_dim = int(self.hf_config.rotary_dim)
        rotated = torch.ops.tensor_cast.apply_rope_single(key[..., :rotary_dim], cos, sin, True)
        return torch.cat([rotated, key[..., rotary_dim:].transpose(1, 2)], dim=-1)


class MiniMaxM31DsparkWrapper(DsparkWrapper):
    """Verify n+1 target tokens while drafting/sampling exactly n tokens."""

    def _draft_query_length(self) -> int:
        return self.dflash_block_size - 1

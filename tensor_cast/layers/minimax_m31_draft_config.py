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

"""Dependency-free MiniMax draft config and plugin registration."""

from .draft_registry import register_draft_config_parser, register_dspark_family


def normalize_minimax_draft_config(source: dict) -> dict:
    """Translate the official nested draft config to the common build contract."""
    if not source.get("dspark_use_dense_ffn", True):
        raise ValueError("MiniMax DSpark currently supports dense FFN drafts only")
    text = dict(source["text_config"])
    dflash = dict(source.get("dflash_config") or {})
    layers = int(text["num_hidden_layers"])
    width = int(source["dspark_ffn_hidden_size"])
    if layers <= 0 or width <= 0:
        raise ValueError("MiniMax DSpark depth and FFN width must be positive")
    ids = source.get("dspark_target_layer_ids") or dflash.get("target_layer_ids")
    if not ids:
        raise ValueError("MiniMax DSpark requires target hidden-state layer ids")
    return {
        **text,
        "rotary_dim": int(text.get("rotary_dim") or text["head_dim"] * text["partial_rotary_factor"]),
        "model_type": "qwen3",
        "tensor_cast_draft_family": "minimax_m31",
        "block_size": int(source["dspark_block_size"]) + 1,
        "num_hidden_layers": layers,
        "layer_types": list(dflash.get("layer_types") or ["sliding_attention"] * layers),
        "sliding_window": int(dflash.get("sliding_window", 4096)),
        "intermediate_size": width,
        "dense_intermediate_size": width,
        # Only the temporary Qwen scaffold consumes hidden_act; replaced below
        # with M3's source-faithful clipped SwiGLU-OAI implementation.
        "hidden_act": "silu",
        "attention_bias": False,
        "mask_token_id": int(source["dspark_noise_token_id"]),
        "dflash_config": {"target_layer_ids": list(ids)},
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": text["rope_theta"],
            "partial_rotary_factor": text["partial_rotary_factor"],
        },
    }


register_draft_config_parser("DSparkMiniMaxDraftModel", normalize_minimax_draft_config)
register_dspark_family(
    "minimax_m31",
    "tensor_cast.layers.minimax_m31_dspark:MiniMaxM31DsparkDraft",
    "tensor_cast.layers.minimax_m31_dspark:MiniMaxM31DsparkWrapper",
)

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

"""Vendored model asset paths under ``tests/assets/model_config/``."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_CONFIG_ROOT = _REPO_ROOT / "assets" / "model_config"

# Hub repo id -> directory name containing a vendored ``config.json``.
# Only Wave A models that already have fixtures and actually load configs.
_VENDORED_MODEL_CONFIG_DIRS: dict[str, str] = {
    "MiniMaxAI/MiniMax-M2": "minimax_m2",
    "MiniMaxAI/MiniMax-M2.7": "minimax_m2_7",
    "inclusionAI/Ling-flash-2.0": "ling_flash_2_0",
    "moonshotai/Kimi-K2-Thinking": "kimi_k2_thinking",
    "Qwen/Qwen2.5-7B": "qwen2_5_7b",
    "Qwen/Qwen3-0.6B": "qwen3_dense_0_6b",
    "Qwen/Qwen3-8B": "qwen3_dense_8b",
    "Qwen/Qwen3-30B-A3B": "qwen3_moe_30b_a3b",
    "Qwen/Qwen3-32B": "qwen3_dense_32b",
    "Qwen/Qwen3-235B-A22B": "qwen3_moe_235b_a22b",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "qwen3_next_80b_a3b",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3_vl_8b_instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct": "qwen3_vl_moe_235b_a22b",
    "Qwen/Qwen3.5-27B": "qwen3_5_27b",
    "Qwen/Qwen3.5-397B-A17B": "qwen3_5_moe_397b_a17b",
    "XiaomiMiMo/MiMo-V2-Flash": "mimo_v2_flash",
    "deepseek-ai/DeepSeek-V3": "deepseek_v3",
    "deepseek-ai/DeepSeek-V3.1": "deepseekv3.1_remote",
    "deepseek-ai/DeepSeek-V3.2": "deepseek_v32",
    "deepseek-ai/DeepSeek-V4-Flash": "deepseek_v4_flash",
    "zai-org/GLM-4.1V-9B-Thinking": "glm4v_9b_thinking",
    "zai-org/GLM-4.5V": "glm4v_moe_4_5v",
    "zai-org/GLM-4.7": "glm4_7",
    "zai-org/GLM-5.1": "glm5_1",
}

# Hub repo id -> directory name under tests/assets/model_config/.
_VENDORED_PREPROCESSOR_DIRS: dict[str, str] = {
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3_vl_8b_instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct": "qwen3_vl_moe_235b_a22b",
}


def resolve_offline_model_id(model_id: str) -> str:
    """Rewrite a registered Hub id to its vendored config directory."""
    if model_id in _VENDORED_MODEL_CONFIG_DIRS:
        return vendored_model_config_path(model_id)
    return model_id


def vendored_model_config_path(model_id: str) -> str:
    """Return a local config directory for a model used by an offline test."""
    try:
        dir_name = _VENDORED_MODEL_CONFIG_DIRS[model_id]
    except KeyError as exc:
        raise KeyError(f"No vendored model config registered for {model_id!r}") from exc
    config_dir = _MODEL_CONFIG_ROOT / dir_name
    if not (config_dir / "config.json").is_file():
        raise FileNotFoundError(f"Vendored config is missing for {model_id!r}: {config_dir}")
    return str(config_dir)


def vendored_preprocessor_config_path(model_id: str) -> Path | None:
    """Return a vendored ``preprocessor_config.json`` path for ``model_id``, if present."""
    dir_name = _VENDORED_PREPROCESSOR_DIRS.get(model_id)
    if dir_name is None:
        return None
    config_path = _MODEL_CONFIG_ROOT / dir_name / "preprocessor_config.json"
    return config_path if config_path.is_file() else None

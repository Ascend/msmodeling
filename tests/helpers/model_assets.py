"""Vendored model asset paths under ``tests/assets/model_config/``."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_CONFIG_ROOT = _REPO_ROOT / "assets" / "model_config"

# Hub repo id -> directory name containing a vendored ``config.json``.
_VENDORED_MODEL_CONFIG_DIRS: dict[str, str] = {
    "Qwen/Qwen2.5-7B": "qwen2_5_7b",
    "Qwen/Qwen3-30B-A3B": "qwen3_moe_30b_a3b",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "qwen3_next_80b_a3b",
    "Qwen/Qwen3.5-397B-A17B": "qwen3_5_moe_397b_a17b",
    "XiaomiMiMo/MiMo-V2-Flash": "mimo_v2_flash",
}

# Hub repo id -> directory name under tests/assets/model_config/.
_VENDORED_PREPROCESSOR_DIRS: dict[str, str] = {
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3_vl_8b_instruct",
}


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

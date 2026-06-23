"""Vendored model asset paths under ``tests/assets/model_config/``."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_CONFIG_ROOT = _REPO_ROOT / "assets" / "model_config"

# Hub repo id -> directory name under tests/assets/model_config/.
_VENDORED_PREPROCESSOR_DIRS: dict[str, str] = {
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3_vl_8b_instruct",
}


def vendored_preprocessor_config_path(model_id: str) -> Path | None:
    """Return a vendored ``preprocessor_config.json`` path for ``model_id``, if present."""
    dir_name = _VENDORED_PREPROCESSOR_DIRS.get(model_id)
    if dir_name is None:
        return None
    config_path = _MODEL_CONFIG_ROOT / dir_name / "preprocessor_config.json"
    return config_path if config_path.is_file() else None

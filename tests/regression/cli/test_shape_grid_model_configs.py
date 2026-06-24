"""Tests for theory-guided shape grid model configs."""

import importlib
import sys
from pathlib import Path

import pytest


GRID_GENERATOR_DIR = Path(__file__).resolve().parents[3] / "tools" / "perf_data_collection" / "grid_generator"
if str(GRID_GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_GENERATOR_DIR))

model_configs = importlib.import_module("model_configs")
get_matmul_nk_pairs = model_configs.get_matmul_nk_pairs
resolve_configs = model_configs.resolve_configs


def fail_fetch(model_name, model_id):
    raise RuntimeError(f"offline: {model_name} {model_id}")


def test_glm51_model_id_resolves_to_static_fallback(monkeypatch):
    """GLM-5.1 model_id should work even when remote config loading is unavailable."""
    monkeypatch.setattr(model_configs, "_fetch_from_huggingface", fail_fetch)
    model_configs._RESOLVED_CONFIGS.clear()

    by_hf = resolve_configs(["zai-org/GLM-5.1"])[0]

    assert by_hf.hidden_size == 6144
    assert by_hf.q_lora_rank == 2048
    assert by_hf.kv_lora_rank == 512
    assert by_hf.head_dim == 256
    assert by_hf.expert_intermediate_size == 2048


def test_all_model_configs_dedupe_aliases():
    configs = resolve_configs(None)

    assert len(configs) == len(set(configs))
    assert [cfg.name for cfg in configs].count("GLM-5.1") == 1
    assert {cfg.model_key for cfg in configs} >= {"deepseekv3", "qwen332b", "llama70b", "glm51"}


def test_glm51_mla_pairs_include_q_and_kv_projection_dims(monkeypatch):
    """GLM-5.1 MLA matmul candidates should use 256-wide qk/v heads, not HF head_dim=64."""
    monkeypatch.setattr(model_configs, "_fetch_from_huggingface", fail_fetch)
    model_configs._RESOLVED_CONFIGS.clear()

    pairs = get_matmul_nk_pairs(["zai-org/GLM-5.1"])

    assert (2048, 6144) in pairs
    assert (576, 6144) in pairs
    assert (16384, 2048) in pairs
    assert (16384, 512) in pairs


@pytest.mark.parametrize(
    ("legacy_name", "replacement"),
    sorted(model_configs.LEGACY_MODEL_NAME_HINTS.items()),
)
def test_legacy_short_names_are_rejected(legacy_name, replacement):
    with pytest.raises(ValueError) as exc_info:
        resolve_configs([legacy_name])

    assert replacement in str(exc_info.value)

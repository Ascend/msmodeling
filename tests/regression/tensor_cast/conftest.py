"""tensor_cast regression fixtures.

Session model / hf_config caching is delegated to ``tests.helpers.model_cache``,
the single source of truth shared across unittest and pytest tests.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import tensor_cast.ops  # noqa: F401 — register custom ops for regression tests
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.model_config import ModelConfig, ParallelConfig, QuantConfig
from tensor_cast.transformers.model import TransformerModel
from tests.helpers.model_assets import vendored_preprocessor_config_path
from tests.helpers.model_cache import _BUILT_MODEL_CACHE, get_built_model, get_hf_config
from tests.helpers.op_registry import build_op_registry


@pytest.fixture(scope="session", autouse=True)
def _wire_vendored_preprocessor_configs():
    """Resolve VL preprocessor limits from tests/assets/model_config/ before Hub cache."""
    from tensor_cast.core import input_generator as input_generator_module

    original_resolve = input_generator_module._resolve_local_preprocessor_config

    def resolve_with_vendored_assets(model_id: str):
        vendored = vendored_preprocessor_config_path(model_id)
        if vendored is not None:
            return vendored
        return original_resolve(model_id)

    input_generator_module._resolve_local_preprocessor_config = resolve_with_vendored_assets
    input_generator_module._load_preprocessor_pixel_limits.cache_clear()
    yield
    input_generator_module._resolve_local_preprocessor_config = original_resolve
    input_generator_module._load_preprocessor_pixel_limits.cache_clear()


def get_session_model(user_config: UserInputConfig) -> TransformerModel:
    """Cross-file session-level build_model cache for unittest TestCase usage."""
    return get_built_model(user_config)


def get_session_hf_config(model_id: str):
    """Cross-file session-level Hugging Face config cache."""
    return get_hf_config(model_id)


@pytest.fixture(scope="session")
def session_model_cache():
    return _BUILT_MODEL_CACHE


@pytest.fixture(scope="session")
def op_registry(cfg_registry):
    """Build a lightweight op registry from shared hf config cache."""
    return build_op_registry(cfg_registry)


@pytest.fixture(scope="module")
def layer_builder(cfg_registry, op_registry) -> Callable[[str], TransformerModel]:
    """Reusable builder that creates TransformerModel with session-cached hf config."""

    def _build(model_id: str) -> TransformerModel:
        hf_config = cfg_registry.get(model_id)
        if hf_config is None:
            hf_config = get_session_hf_config(model_id)
            cfg_registry[model_id] = hf_config
            op_registry[model_id] = {
                "model_type": getattr(hf_config, "model_type", None),
                "num_hidden_layers": getattr(hf_config, "num_hidden_layers", None),
            }
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            hf_config=hf_config,
        )
        return TransformerModel(model_id, model_config)

    return _build


@pytest.fixture(scope="module")
def qwen3_32b_lmhead_attention_transformer() -> TransformerModel:
    hf_config = get_session_hf_config("Qwen/Qwen3-32B")
    model_config = ModelConfig(
        ParallelConfig(),
        QuantConfig(),
        attention_cls=AttentionTensorCast,
        enable_repetition=True,
        hf_config=hf_config,
    )
    return TransformerModel("Qwen/Qwen3-32B", model_config)


@pytest.fixture(scope="module")
def deepseek_v32_build_model_int8():
    user_input = UserInputConfig(
        model_id="deepseek-ai/DeepSeek-V3.2",
        num_queries=1,
        query_len=32,
        context_length=32,
        device="TEST_DEVICE",
        num_mtp_tokens=2,
        disable_repetition=True,
        quantize_attention_action=QuantizeAttentionAction.INT8,
    )
    return get_session_model(user_input)


@pytest.fixture(scope="module")
def deepseek_v32_build_model_fp8():
    user_input = UserInputConfig(
        model_id="deepseek-ai/DeepSeek-V3.2",
        num_queries=1,
        query_len=32,
        context_length=32,
        device="TEST_DEVICE",
        num_mtp_tokens=2,
        disable_repetition=True,
        quantize_attention_action=QuantizeAttentionAction.FP8,
    )
    return get_session_model(user_input)


@pytest.fixture(scope="module")
def qwen3_vl_8b_instruct_transformer() -> TransformerModel:
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    hf_config = get_session_hf_config(model_id)
    model_config = ModelConfig(
        parallel_config=ParallelConfig(),
        quant_config=QuantConfig(),
        dtype=torch.bfloat16,
        hf_config=hf_config,
    )
    return TransformerModel(model_id, model_config)

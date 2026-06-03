"""Construct models and dummy inputs for tests.

Concrete implementations live alongside regressions; this package collects shared builders.
"""

from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.transformers.model import TransformerModel
from tests.helpers.model_cache import user_config_build_cache_key


def make_user_input_config(
    *,
    model_id: str,
    device: str = "TEST_DEVICE",
    num_queries: int = 1,
    query_len: int = 32,
    context_length: int = 32,
) -> UserInputConfig:
    """Create a minimal reusable user config for test model builds."""
    return UserInputConfig(
        model_id=model_id,
        device=device,
        num_queries=num_queries,
        query_len=query_len,
        context_length=context_length,
    )


def build_or_get_cached_model(
    user_config: UserInputConfig,
    cache: dict[tuple, TransformerModel],
) -> TransformerModel:
    """Build model once per config key and reuse from provided cache."""
    key = user_config_build_cache_key(user_config)
    if key not in cache:
        cache[key] = build_model(user_config)
    return cache[key]

"""Single source-of-truth cache for deterministic test construction artifacts.

CACHE POLICY:
    Only deterministic, pure-construction artifacts are cached:
        * Hugging Face configs from ``AutoModelConfigLoader.load_config``.
        * Freshly ``build_model(...)`` results.
    Entries are keyed by their full determining inputs (``model_id`` for configs,
    ``user_config_build_cache_key`` for models).

    Configs are handed out as deepcopies (cheap, mutable dicts) so a caller can
    never mutate shared config state. Built models are handed out SHARED, not
    copied: freshly built models hold non-leaf / meta tensors that do not support
    ``deepcopy``, and tests treat the built model as read-only (forward runs
    produce separate runtime/event objects). Callers must NOT mutate a returned
    model.

    DO NOT cache objects that have run forward/compile, anything a test mutates,
    or per-test scratch data.

Both pytest fixtures (``cfg_registry``, ``get_session_*`` and friends) and
unittest ``TestCase`` code paths (which cannot consume fixtures) delegate here,
so the cache is loaded at most once per session regardless of entry point.
"""

from __future__ import annotations

import copy

from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.transformers.model import TransformerModel
from tensor_cast.transformers.utils import AutoModelConfigLoader

_HF_CONFIG_CACHE: dict[str, object] = {}
_BUILT_MODEL_CACHE: dict[tuple, TransformerModel] = {}


def user_config_build_cache_key(user_config: UserInputConfig) -> tuple:
    """Fields that affect ConfigResolver.resolve() / build_model()."""
    return (
        user_config.model_id,
        user_config.do_compile,
        user_config.num_mtp_tokens,
        user_config.num_hidden_layers_override,
        user_config.quantize_linear_action,
        user_config.quantize_attention_action,
        user_config.remote_source,
        user_config.allow_graph_break,
        user_config.enable_multistream,
        user_config.world_size,
        user_config.tp_size,
        user_config.mlp_tp_size,
        user_config.lmhead_tp_size,
        user_config.ep_size,
        user_config.moe_dp_size,
        user_config.moe_tp_size,
        user_config.enable_redundant_experts,
        user_config.enable_external_shared_experts,
        user_config.enable_shared_expert_tp,
        user_config.host_external_shared_experts,
        user_config.disable_repetition,
    )


def get_hf_config(model_id: str):
    """Return a deepcopy of the session-cached Hugging Face config for ``model_id``."""
    if model_id not in _HF_CONFIG_CACHE:
        _HF_CONFIG_CACHE[model_id] = AutoModelConfigLoader().load_config(model_id)
    return copy.deepcopy(_HF_CONFIG_CACHE[model_id])


def get_built_model(user_config: UserInputConfig) -> TransformerModel:
    """Return the session-cached ``build_model`` result for ``user_config``.

    Shared (not deepcopied): built models contain non-leaf / meta tensors that do
    not support ``deepcopy``, and tests treat the model as read-only. Callers must
    NOT mutate the returned model.
    """
    key = user_config_build_cache_key(user_config)
    if key not in _BUILT_MODEL_CACHE:
        _BUILT_MODEL_CACHE[key] = build_model(user_config)
    return _BUILT_MODEL_CACHE[key]

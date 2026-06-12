"""Smoke test for DeepSeek-V4 model support."""

import pytest


def test_deepseek_v4_config_registration():
    """Verify DeepSeek-V4 model can be registered with AutoConfig."""
    from transformers import AutoConfig

    # Verify deepseek_v4 is registered
    config = AutoConfig.for_model("deepseek_v4")
    assert config is not None
    assert config.model_type == "deepseek_v4"


def test_deepseek_v4_config_initialization():
    """Verify DeepseekV4Config can be initialized with required fields."""
    from tensor_cast.transformers.builtin_model.deepseek_v4 import DeepseekV4Config

    # Minimal required config
    config = DeepseekV4Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=2,
        vocab_size=128256,
        compress_ratios=[0, 4],
    )

    assert config.model_type == "deepseek_v4"
    assert config.hidden_size == 4096
    assert config.compress_ratios == [0, 4]


def test_deepseek_v4_hc_parameters():
    """Verify DeepSeek-V4 Head Compression parameters are set correctly."""
    from tensor_cast.transformers.builtin_model.deepseek_v4 import DeepseekV4Config

    config = DeepseekV4Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=2,
        vocab_size=128256,
        compress_ratios=[0, 4],
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )

    assert config.hc_mult == 4
    assert config.hc_sinkhorn_iters == 20
    assert config.hc_eps == 1e-6


def test_deepseek_v4_compress_ratio_validation():
    """Verify compress_ratios validation allows only 0, 4, 128."""
    from tensor_cast.transformers.builtin_model.deepseek_v4 import DeepseekV4Config

    # Valid ratios
    config = DeepseekV4Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=3,
        vocab_size=128256,
        compress_ratios=[0, 4, 128],
    )
    assert config.compress_ratios == [0, 4, 128]

    # Invalid ratio should raise
    with pytest.raises(ValueError, match="compress_ratios must provide at least"):
        DeepseekV4Config(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_hidden_layers=2,
            vocab_size=128256,
            compress_ratios=[3],  # Invalid
        )


def test_deepseek_v4_model_profile_registration():
    """Verify DeepSeek-V4 model profile is registered."""
    from tensor_cast.transformers.custom_model_registry import get_model_profile

    profile = get_model_profile("deepseek_v4")
    assert profile is not None
    assert profile.model_type == "deepseek_v4"
    assert profile.mla_module_class_type.__name__ == "DeepseekV4SparseAttention"

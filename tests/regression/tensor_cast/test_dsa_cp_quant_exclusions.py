"""DSA-CP must preserve the linear quantization required by quantized MLA."""

from types import SimpleNamespace

import pytest

from tensor_cast.core.quantization.config import create_attention_quant_config
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction
from tensor_cast.transformers.transformations import _exclude_unquantized_dsa_linears


@pytest.mark.parametrize("action", [None, QuantizeAttentionAction.INT8, QuantizeAttentionAction.FP8])
@pytest.mark.parametrize("existing", [None, ["custom.proj"]])
def test_dsa_cp_quantization_exclusions(action, existing):
    quant = create_attention_quant_config(action) if action is not None else None
    config = SimpleNamespace(
        quant_config=SimpleNamespace(
            modules_to_not_convert=list(existing) if existing is not None else None,
            attention_configs={-1: quant} if action is not None else {},
        )
    )
    _exclude_unquantized_dsa_linears(config)
    _exclude_unquantized_dsa_linears(config)
    patterns = config.quant_config.modules_to_not_convert
    assert patterns.count("*indexer*.wk") == 1
    assert patterns.count("*indexer*.weights_proj") == 1
    assert ("*.kv_b_proj" in patterns) == (action is None)
    if existing is not None:
        assert "custom.proj" in patterns


def test_none_attention_entry_is_not_quantized_and_user_exclusion_survives():
    config = SimpleNamespace(
        quant_config=SimpleNamespace(
            modules_to_not_convert=[],
            attention_configs={-1: None},
        )
    )
    _exclude_unquantized_dsa_linears(config)
    assert "*.kv_b_proj" in config.quant_config.modules_to_not_convert
    config.quant_config.attention_configs = {-1: create_attention_quant_config(QuantizeAttentionAction.INT8)}
    _exclude_unquantized_dsa_linears(config)
    assert "*.kv_b_proj" in config.quant_config.modules_to_not_convert

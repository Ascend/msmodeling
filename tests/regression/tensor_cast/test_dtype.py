import pytest
import torch
from tensor_cast.layers.quant_linear import QuantLinearBase
from tensor_cast.model_config import AttentionQuantConfig, LinearQuantConfig
from tensor_cast.quantize_utils import AttentionQuantType, LinearQuantType, QuantGranularity, QuantScheme
from tests.regression.tensor_cast.test_common import get_linear_quant_config


def test_attention_quant_dtype_mapping():
    cfg = AttentionQuantConfig(quant_type=AttentionQuantType.INT8)
    assert cfg.get_quant_dtype() is torch.int8


def test_mxfp4_requires_per_group_granularity():
    with pytest.raises(ValueError):
        LinearQuantConfig(
            quant_type=LinearQuantType.MXFP4,
            dynamic_quant_granularity=QuantGranularity.PER_TENSOR,
        )


def test_pack_unpack_int4_roundtrip():
    linear = torch.nn.Linear(32, 64, bias=False, dtype=torch.float32)
    config = get_linear_quant_config(LinearQuantType.W4A8, linear.weight.data)
    quant_layer = QuantLinearBase(linear, config)
    original = torch.randint(-8, 8, (64, 32), dtype=torch.int8)
    packed = quant_layer.pack_int4(original)
    unpacked = quant_layer.unpack_int4(packed)
    assert torch.equal(original, unpacked)


def test_fp8_rejects_asymmetric_scheme():
    with pytest.raises(ValueError, match="symmetric scheme"):
        LinearQuantConfig(
            quant_type=LinearQuantType.FP8,
            weight_scale=torch.tensor(1.0),
            dynamic_quant_scheme=QuantScheme.ASYMMETRIC,
        )


def test_zero_shape_static_quant_linear_keeps_shape():
    x = torch.randn([0, 10], device="meta")
    w = torch.randint(0, 255, [10, 10], dtype=torch.uint8, device="meta")
    w_scale = torch.randn([10], device="meta")
    y = torch.ops.tensor_cast.static_quant_linear(
        x,
        w,
        w_scale,
        w_offset=None,
        x_scale=None,
        x_offset=None,
        bias=None,
        out_dtype=None,
    )
    assert y.shape == (0, 10)

import torch

from tensor_cast.layers.quant_linear import QuantLinearBase
from tensor_cast.model_config import ParallelConfig
from tensor_cast.quantize_utils import LinearQuantType
from tests.regression.tensor_cast.test_common import get_linear_quant_config  # pylint: disable=no-name-in-module


def test_tensor_cast_parallel_layer_smoke():
    cfg = ParallelConfig(world_size=4, tensor_parallel_size=2, data_parallel_size=2)
    assert cfg.has_attn_tp()
    assert cfg.data_parallel_size == 2
    linear = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    quant_layer = QuantLinearBase(
        linear,
        get_linear_quant_config(LinearQuantType.W8A16, linear.weight.data),
    )
    x = torch.randn(2, 8, dtype=torch.float32)
    y = quant_layer(x)
    assert y.shape == (2, 4)

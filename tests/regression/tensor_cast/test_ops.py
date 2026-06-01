import torch
import torch.fx as fx
from tensor_cast.model_config import AttentionQuantConfig, QuantConfig
from tensor_cast.performance_model.op_estimator_registry import (
    _op_estimator_table,
    get_op_estimator,
    register_op_estimator,
)
from tensor_cast.quantize_utils import AttentionQuantType


class _NonDefaultEpsRMSNormModule(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))

    def _rms_norm(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

    def forward(self, hidden_states, residual):
        rms = self._rms_norm(hidden_states)
        add_rms = self._rms_norm(hidden_states + residual)
        added = hidden_states + residual
        add_rms2 = self._rms_norm(added)
        return rms, add_rms, add_rms2, added


def test_register_and_get_op_estimator():
    op_key = object()
    original = _op_estimator_table.get(None, {}).get(op_key)

    @register_op_estimator(op_key, None, True)
    def _estimator(op_invoke_info, device_profile):
        return "ok"

    assert get_op_estimator(op_key, None) is _estimator

    if original is None:
        _op_estimator_table[None].pop(op_key, None)
    else:
        _op_estimator_table[None][op_key] = original


def test_rms_norm_non_default_eps_path_consistency():
    module = _NonDefaultEpsRMSNormModule(eps=1e-6)
    hidden_states = torch.randn(2, 4, dtype=torch.float32)
    residual = torch.randn(2, 4, dtype=torch.float32)
    _, add_rms, add_rms2, added = module(hidden_states, residual)
    torch.testing.assert_close(add_rms, add_rms2, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(added, hidden_states + residual, rtol=0.0, atol=0.0)


def test_quant_attention_config_can_target_single_layer():
    quant_config = QuantConfig()
    attn_config = AttentionQuantConfig(
        quant_type=AttentionQuantType.INT8,
        query_scale=torch.tensor(1.0),
        kv_scale=torch.tensor(1.0),
        attention_prob_scale=torch.tensor(1.0),
    )
    quant_config.attention_configs[0] = attn_config
    assert 0 in quant_config.attention_configs
    assert quant_config.attention_configs[0].quant_type == AttentionQuantType.INT8


def test_multistream_count_nodes_helper_behavior():
    graph = fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.ops.aten.neg.default, args=(x,))
    graph.output(y)
    gm = fx.GraphModule({}, graph)
    count = sum(1 for node in gm.graph.nodes if node.target == torch.ops.aten.neg.default)
    assert count == 1

from types import SimpleNamespace

import torch

from tensor_cast.layers.glm5 import Glm5SparseAttention
from tensor_cast.layers.mla import DeepseekSparseAttention
from tensor_cast.layers.mtp import MultiTokenPredictorLayer
from tensor_cast.model_config import MtpConfig
from tensor_cast.transformers.transformations import maybe_enable_mtp


def test_glm5_sparse_attention_returns_topk_slot(monkeypatch):
    sparse_attention = object.__new__(Glm5SparseAttention)

    monkeypatch.setattr(
        DeepseekSparseAttention,
        "forward",
        lambda self, *args, **kwargs: ("hidden", None),
    )

    assert Glm5SparseAttention.forward(sparse_attention, None, None, None) == ("hidden", None, None)


def test_glm5_mtp_layer_uses_hidden_states_from_tuple_block_output():
    class TupleBlock(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return hidden_states + 1, None

    layer = MultiTokenPredictorLayer(
        SimpleNamespace(hidden_size=2, rms_norm_eps=1e-5),
        TupleBlock(),
    )
    layer.emb_norm = torch.nn.Identity()
    layer.hidden_norm = torch.nn.Identity()
    layer.linear_proj = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        layer.linear_proj.weight.zero_()
        layer.linear_proj.weight[:, 2:] = torch.eye(2)

    output = layer(
        inputs_embeds=torch.zeros(1, 1, 2),
        position_ids=torch.zeros(1, 1, dtype=torch.long),
        previous_hidden_states=torch.tensor([[[2.0, 3.0]]]),
    )

    assert torch.equal(output, torch.tensor([[[3.0, 4.0]]]))


def test_glm5_mtp_extends_indexer_types(monkeypatch):
    captured = {}

    class FakeMtpWrapper:
        def __init__(self, mtp_config, hf_config, inner):
            captured["mtp_config"] = mtp_config
            captured["hf_config"] = hf_config
            captured["inner"] = inner

    class FakeModel:
        is_vl_model = False
        text_config = None
        _inner = object()
        hf_config = SimpleNamespace(
            indexer_types=["full", "shared"],
            layer_types=["full_attention", "full_attention"],
            mlp_layer_types=["sparse", "sparse"],
        )
        model_config = SimpleNamespace(
            mtp_config=MtpConfig(num_mtp_layers=3, mtp_block_module_name="GlmMoeDsaDecoderLayer"),
            dtype=torch.float32,
        )

        def unwrap(self):
            return SimpleNamespace()

    monkeypatch.setattr("tensor_cast.layers.mtp.MtpWrapper", FakeMtpWrapper)

    maybe_enable_mtp(FakeModel())

    assert captured["hf_config"].indexer_types == ["full", "shared", "shared", "shared", "shared"]

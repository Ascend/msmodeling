from types import SimpleNamespace

import pytest
import torch

from tensor_cast.layers.glm5 import (
    Glm5SparseAttention,
    extend_glm5_indexer_types_for_mtp,
    resolve_glm5_indexer_source_layer,
)
from tensor_cast.layers.internal import CopyLayerWrapper, RegionMarkerWrapper
from tensor_cast.layers.mla import DeepseekSparseAttention
from tensor_cast.layers.mtp import MultiTokenPredictorLayer
from tensor_cast.model_config import MtpConfig
from tensor_cast.transformers.builtin_model.glm5 import (
    Glm5DecoderLayerCompat,
    Glm5ModelCompat,
    _decoder_supports_prev_topk,
    _prepare_glm5_decoder_layer,
    _resolve_glm5_mtp_block_owner,
)
from tensor_cast.transformers.transformations import maybe_enable_mtp, patch_mla


def test_glm5_sparse_attention_returns_topk_slot(monkeypatch):
    sparse_attention = object.__new__(Glm5SparseAttention)

    monkeypatch.setattr(
        DeepseekSparseAttention,
        "forward",
        lambda self, *args, **kwargs: ("hidden", None, None),
    )

    assert Glm5SparseAttention.forward(sparse_attention, None, None, None) == ("hidden", None, None)


def test_glm5_old_decoder_compat_propagates_prev_topk_indices():
    captured = {}

    class FakeAttention(torch.nn.Module):
        def forward(self, **kwargs):
            captured["prev_topk_indices"] = kwargs["prev_topk_indices"]
            return kwargs["hidden_states"], None, "next-topk"

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.Identity()
            self.post_attention_layernorm = torch.nn.Identity()
            self.self_attn = FakeAttention()
            self.mlp = torch.nn.Identity()

    layer = Glm5DecoderLayerCompat(FakeLayer())
    hidden_states, topk_indices = layer(
        torch.ones(1, 1, 2),
        prev_topk_indices="previous-topk",
    )

    assert captured["prev_topk_indices"] == "previous-topk"
    assert torch.equal(hidden_states, torch.full((1, 1, 2), 4.0))
    assert topk_indices == "next-topk"


def test_glm5_decoder_compat_rejects_missing_topk_output():
    class FakeAttention(torch.nn.Module):
        def forward(self, **_kwargs):
            return "hidden", None

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.Identity()
            self.post_attention_layernorm = torch.nn.Identity()
            self.self_attn = FakeAttention()
            self.mlp = torch.nn.Identity()

    with pytest.raises(ValueError, match="attention must return"):
        Glm5DecoderLayerCompat(FakeLayer())(torch.ones(1, 1, 2))


def test_glm5_decoder_contract_detection():
    class NewDecoder(torch.nn.Module):
        def forward(self, hidden_states, prev_topk_indices=None):
            return hidden_states, prev_topk_indices

    class OldDecoder(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    assert _decoder_supports_prev_topk(NewDecoder())
    assert not _decoder_supports_prev_topk(OldDecoder())


def test_glm5_model_compat_preserves_repetition_wrappers():
    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.Identity()

    representative = RegionMarkerWrapper(region_id=1, layer=FakeLayer())
    copy_layer = CopyLayerWrapper(region_id=1, layer=FakeLayer(), representative=representative)
    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([representative, copy_layer])
    model = Glm5ModelCompat(inner)

    assert model.layers[0] is representative
    assert isinstance(representative._inner, Glm5DecoderLayerCompat)
    assert model.layers[1] is copy_layer
    assert _prepare_glm5_decoder_layer(copy_layer) is copy_layer


def test_glm5_model_uses_configured_cache_default(monkeypatch):
    created_caches = []

    class FakeCache:
        def __init__(self, config):
            self.config = config
            created_caches.append(self)

        def get_seq_length(self):
            return 0

    class FakeRotaryEmbedding(torch.nn.Module):
        def forward(self, hidden_states, position_ids=None):
            return hidden_states

    class FakeInner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True, num_hidden_layers=0)
            self.layers = torch.nn.ModuleList()
            self.embed_tokens = torch.nn.Identity()
            self.norm = torch.nn.Identity()

    monkeypatch.setattr("tensor_cast.transformers.builtin_model.glm5.DynamicCache", FakeCache)
    monkeypatch.setattr(
        "tensor_cast.transformers.builtin_model.glm5.create_causal_mask",
        lambda **_kwargs: None,
    )

    model = Glm5ModelCompat(FakeInner())
    model.rotary_emb = FakeRotaryEmbedding()
    model(input_ids=torch.ones(1, 1, dtype=torch.long))

    assert len(created_caches) == 1
    assert created_caches[0].config.use_cache is True


def test_glm5_mtp_patch_skips_copy_layers_and_unwraps_representatives():
    class FakeMtpLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mtp_block = torch.nn.Identity()

    representative = RegionMarkerWrapper(
        region_id=1,
        layer=FakeMtpLayer(),
    )
    copy_layer = CopyLayerWrapper(
        region_id=1,
        layer=FakeMtpLayer(),
        representative=representative,
    )

    assert _resolve_glm5_mtp_block_owner(representative) is representative._inner
    assert _resolve_glm5_mtp_block_owner(copy_layer) is None


def test_glm5_returns_topk_when_next_layer_skips_topk():
    sparse_attention = object.__new__(Glm5SparseAttention)
    object.__setattr__(sparse_attention, "_inner", SimpleNamespace(next_skip_topk=True))

    assert sparse_attention._format_forward_output("hidden", None, "topk") == ("hidden", None, "topk")


def test_glm5_full_indexer_layer_executes(monkeypatch):
    sparse_attention = object.__new__(Glm5SparseAttention)
    object.__setattr__(sparse_attention, "layer_idx", 4)
    object.__setattr__(sparse_attention, "indexer_source_layer_idx", 4)

    calls = []

    def fake_run(self, *args, **kwargs):
        calls.append((args, kwargs))
        return "topk"

    monkeypatch.setattr(DeepseekSparseAttention, "_run_sparse_attention_indexer", fake_run)

    output = Glm5SparseAttention._run_sparse_attention_indexer(
        sparse_attention,
        "hidden",
        "qa",
        "position",
        None,
    )

    assert output == "topk"
    assert len(calls) == 1


def test_glm5_shared_indexer_layer_reuses_prev_topk_indices(monkeypatch):
    sparse_attention = object.__new__(Glm5SparseAttention)
    object.__setattr__(sparse_attention, "layer_idx", 5)
    object.__setattr__(sparse_attention, "indexer_source_layer_idx", 4)

    def fail_run(self, *args, **kwargs):
        raise AssertionError("shared layer should not execute indexer")

    monkeypatch.setattr(DeepseekSparseAttention, "_run_sparse_attention_indexer", fail_run)

    output = Glm5SparseAttention._run_sparse_attention_indexer(
        sparse_attention,
        "hidden",
        "qa",
        "position",
        None,
        prev_topk_indices="prev-topk",
    )

    assert output == "prev-topk"


def test_glm5_shared_indexer_layer_requires_prev_topk_indices():
    sparse_attention = object.__new__(Glm5SparseAttention)
    object.__setattr__(sparse_attention, "layer_idx", 5)
    object.__setattr__(sparse_attention, "indexer_source_layer_idx", 4)

    with pytest.raises(ValueError, match="missing prev_topk_indices from source layer 4"):
        Glm5SparseAttention._run_sparse_attention_indexer(
            sparse_attention,
            "hidden",
            "qa",
            "position",
            None,
        )


def test_glm5_indexer_source_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported GLM5 indexer type 'weird'"):
        resolve_glm5_indexer_source_layer(["weird"], 0)


def test_patch_mla_validates_glm5_indexer_types_before_indexing():
    class GlmMoeDsaAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 2

    class FakeGlm5SparseAttention(Glm5SparseAttention):
        def __init__(self, _config, module, _tp_group):
            torch.nn.Module.__init__(self)
            self.layer_idx = module.layer_idx

    model = SimpleNamespace(
        model_config=SimpleNamespace(
            mla_config=SimpleNamespace(
                module_name="GlmMoeDsaAttention",
                mla_cls=FakeGlm5SparseAttention,
                field_names=object(),
            )
        ),
        parallel_group_manager=SimpleNamespace(tp_group=None),
        _inner=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="glm_moe_dsa", indexer_types=["full", "shared"]),
            named_modules=lambda: [("layers.2.self_attn", GlmMoeDsaAttention())],
        ),
        hf_config=SimpleNamespace(model_type="glm_moe_dsa", indexer_types=["full", "shared"]),
        _replace_module=lambda *_args: None,
    )

    with pytest.raises(ValueError, match="GLM5 indexer_types has 2 entries, cannot resolve layer 2"):
        patch_mla(model)


def test_glm5_mtp_extension_ignores_empty_indexer_types():
    indexer_types = []

    extend_glm5_indexer_types_for_mtp(indexer_types, 3)

    assert indexer_types == []


def test_glm52_mtp_extension_uses_full_indexers_for_independent_blocks():
    indexer_types = ["full", "shared", "shared", "shared"] * 18 + [
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
    ]

    extend_glm5_indexer_types_for_mtp(indexer_types, 3)

    assert indexer_types[-3:] == ["full", "full", "full"]


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
            model_type="glm_moe_dsa",
            indexer_types=["full", "shared", "shared", "shared"],
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

    assert captured["hf_config"].indexer_types == [
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "full",
        "full",
    ]

# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# MindStudio is licensed under Mulan PSL v2.
# You may obtain a copy at http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
# -------------------------------------------------------------------------

"""MiniMax image geometry, decode context and compiled vision dependencies."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig, MiniMaxM3VLVisionConfig
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLVisionModel

from tensor_cast.core import input_generator as inputs
from tensor_cast.model_config import DsparkConfig
from tensor_cast.patch_torch import patch_torch
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.pipeline_parallel import PipelineStageSpec, build_pipeline_stage_kwargs
from tensor_cast.transformers.builtin_model.minimax_m3 import (
    _minimax_merge_visual_features,
    _minimax_multimodal_forward,
    _minimax_placeholder_mask,
    patch_method_for_minimax_m3,
    _minimax_vision_meta_forward,
)


@pytest.fixture(params=["MiniMaxAI/MiniMax-M3", "MiniMaxAI/MiniMax-M3.1-Flash"])
def model_id(request):
    return request.param


def _model(dp=1, model_id="MiniMaxAI/MiniMax-M3"):
    is_m31 = model_id.endswith("MiniMax-M3.1-Flash")
    config = MiniMaxM3VLConfig(vision_config={"image_size": 672 if is_m31 else 2016})
    return SimpleNamespace(
        is_vl_model=True,
        num_hidden_layers=0,
        model_id=model_id,
        model_config=SimpleNamespace(
            hf_config=config,
            dtype=torch.bfloat16,
            mtp_config=None,
            dflash_config=None,
            dspark_config=None,
            parallel_config=SimpleNamespace(data_parallel_size=dp, decode_context_parallel_size=1),
        ),
    )


@pytest.mark.parametrize("concurrency,dp,images", [(2, 1, 2), (8, 8, 1)])
def test_image_geometry_and_dp(concurrency, dp, images, model_id):
    model = _model(dp, model_id)
    result = inputs.generate_image_inputs(model, 1, 448, 448, concurrency)
    assert result["pixel_values"].shape == (images * 1024, 1176)
    assert result["image_grid_thw"].tolist() == [[1, 32, 32]] * images
    assert result["image_grid_thw"]._minimax_patch_lengths == (1024,) * images
    assert result["num_image_tokens"] == 258
    assert inputs.resolve_multimodal_token_budget(model, 1, 448, 448) == 258


def test_official_resize_caps_large_images(model_id):
    model = _model(model_id=model_id)
    result = inputs.generate_image_inputs(model, 1, 1024, 1024, 1)
    assert result["image_grid_thw"].tolist() == [[1, 48, 48]]
    assert result["num_image_tokens"] == 578


@pytest.mark.parametrize("decode,included", [(False, False), (True, False), (True, True)])
def test_image_tokens_count_once_and_decode_skips_encoder(monkeypatch, decode, included, model_id):
    monkeypatch.setattr(inputs, "_get_kv_cache_info", lambda *a, **k: ({}, 0))
    monkeypatch.setattr(inputs, "get_sparse_attention_indexer_cache_info", lambda *a, **k: {})
    model = _model(8 if decode else 1, model_id)
    query = 8 if decode else 16
    context = 131072 if decode else 0
    extra = 258 if included else 0
    request = inputs.RequestInfo(
        query_len=query,
        seq_len=context + query + extra,
        context_length=context + extra,
        concurrency=8 if decode else 2,
        is_decode=decode,
        image_batch_size=1,
        image_height=448,
        image_width=448,
        image_tokens_included=included,
    )
    result = inputs.generate_inputs(model, [request])
    meta = result["attention_meta"]
    assert meta.seq_lens_values == [context + query + 258] * (1 if decode else 2)
    assert meta.start_pos == (context + 258 if decode else 0)
    assert result["input_ids"].shape == (1, 8 if decode else 548)
    assert ("pixel_values" in result) is not decode


def test_m31_image_decode_preserves_dspark_verification_window(monkeypatch):
    monkeypatch.setattr(inputs, "_get_kv_cache_info", lambda *a, **k: ({}, 0))
    monkeypatch.setattr(inputs, "get_sparse_attention_indexer_cache_info", lambda *a, **k: {})
    model = _model(8, "MiniMaxAI/MiniMax-M3.1-Flash")
    model.model_config.dspark_config = DsparkConfig(dspark_block_size=8)
    model.model_config.draft_block_size = lambda: 8
    request = inputs.RequestInfo(
        query_len=8,
        context_length=131072,
        seq_len=131080,
        concurrency=8,
        is_decode=True,
        image_batch_size=1,
        image_height=448,
        image_width=448,
    )
    result = inputs.generate_inputs(model, [request])
    assert result["input_ids"].shape == (1, 8)
    assert result["attention_meta"].start_pos == 131330
    assert result["attention_meta"].seq_lens_values == [131338]
    assert result["sampling_metadata"].spec_decode_metadata.num_speculative_tokens == 7
    assert "pixel_values" not in result


def test_pp_only_first_stage_receives_pixels():
    kwargs = {
        "input_ids": torch.empty(1, 274, dtype=torch.long, device="meta"),
        "position_ids": torch.empty(1, 274, dtype=torch.long, device="meta"),
        "pixel_values": torch.empty(1024, 1176, device="meta"),
        "image_grid_thw": torch.tensor([[1, 32, 32]]),
    }
    for rank in range(4):
        spec = PipelineStageSpec(rank, 4, rank * 15, (rank + 1) * 15, rank == 0, rank == 3)
        stage, _ = build_pipeline_stage_kwargs(spec, kwargs, hidden_states=torch.empty(1, 274, 8, device="meta"))
        assert ("pixel_values" in stage) == (rank == 0)
        assert ("image_grid_thw" in stage) == (rank == 0)


def test_masked_scatter_keeps_vision_matmul_in_compiled_dataflow():
    def merge(text, mask, image, weight):
        return _minimax_merge_visual_features(text, mask, image @ weight)

    tensors = (
        torch.empty(1, 8, 4, device="meta"),
        torch.empty(1, 8, 4, dtype=torch.bool, device="meta"),
        torch.empty(2, 3, device="meta"),
        torch.empty(3, 4, device="meta"),
    )
    with patch_torch():
        graph = make_fx(merge)(*tensors)
    graph.graph.eliminate_dead_code()
    targets = [node.target for node in graph.graph.nodes]
    assert torch.ops.aten.mm.default in targets
    assert torch.ops.tensor_cast.masked_scatter.default in targets
    assert graph(*tensors).shape == tensors[0].shape


def test_generic_meta_scatter_keeps_existing_behavior():
    text = torch.empty(1, 8, 4, device="meta")
    mask = torch.empty_like(text, dtype=torch.bool)
    image = torch.empty(2, 4, device="meta")
    with patch_torch():
        graph = make_fx(lambda x, m, source: x.masked_scatter(m, source))(text, mask, image)
    assert torch.ops.tensor_cast.masked_scatter.default not in [node.target for node in graph.graph.nodes]


def test_minimax_forward_keeps_projector_dependency():
    def forward(text, image, weight):
        backbone = SimpleNamespace(
            get_image_features=lambda **kwargs: SimpleNamespace(pooler_output=kwargs["pixel_values"] @ weight),
            get_placeholder_mask=lambda *args, **kwargs: (torch.empty_like(text, dtype=torch.bool), None),
            language_model=lambda **kwargs: SimpleNamespace(
                last_hidden_state=kwargs["inputs_embeds"], past_key_values=None
            ),
        )
        return _minimax_multimodal_forward(backbone, inputs_embeds=text, pixel_values=image).last_hidden_state

    tensors = (
        torch.empty(1, 8, 4, device="meta"),
        torch.empty(2, 3, device="meta"),
        torch.empty(3, 4, device="meta"),
    )
    with patch_torch():
        graph = make_fx(forward)(*tensors)
    graph.graph.eliminate_dead_code()
    targets = [node.target for node in graph.graph.nodes]
    assert torch.ops.aten.mm.default in targets
    assert torch.ops.tensor_cast.masked_scatter.default in targets
    assert graph(*tensors).shape == tensors[0].shape


def test_merge_memory_properties_and_real_tensor_values():
    text = torch.empty(1, 8, 4, dtype=torch.bfloat16, device="meta")
    mask = torch.empty_like(text, dtype=torch.bool)
    source = torch.empty(2, 4, dtype=text.dtype, device="meta")
    result = torch.ops.tensor_cast.masked_scatter(text, mask, source)
    info = OpInvokeInfo(torch.ops.tensor_cast.masked_scatter.default, (text, mask, source), {}, result)
    properties = info.get_perf_properties()
    assert properties.memory_read_bytes == 32 * 2 + 32 + 8 * 2
    assert properties.memory_write_bytes == 32 * 2
    assert not properties.compute_ops
    with patch_torch():
        actual = torch.zeros(4).masked_scatter(torch.tensor([True, False, True, False]), torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(actual, torch.tensor([2.0, 0.0, 3.0, 0.0]))


def test_real_vision_tensors_keep_upstream_forward():
    config = MiniMaxM3VLVisionConfig(hidden_size=96, num_attention_heads=4, num_hidden_layers=2, intermediate_size=192)
    config._attn_implementation = "eager"
    model = MiniMaxM3VLVisionModel(config).eval()
    pixels = torch.randn(4, 1176)
    grid = torch.tensor([[1, 2, 2]])
    with torch.no_grad():
        expected = model(pixel_values=pixels, grid_thw=grid).last_hidden_state
        model._tensor_cast_original_forward = type(model).forward
        model.forward = MethodType(_minimax_vision_meta_forward, model)
        actual = model(pixel_values=pixels, grid_thw=grid).last_hidden_state
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("embeds", [None, torch.zeros(1, 2, 4)])
def test_placeholder_fallback_forwards_features_by_keyword(embeds):
    image, video = object(), object()

    def original(self, input_ids, inputs_embeds, *, image_features, video_features):
        assert inputs_embeds is embeds
        assert image_features is image and video_features is video
        return "original"

    backbone = SimpleNamespace(_tensor_cast_original_placeholder_mask=original)
    assert _minimax_placeholder_mask(backbone, None, embeds, image, video) == "original"


@pytest.mark.parametrize("quant_config", [None, SimpleNamespace(modules_to_not_convert=[])])
def test_minimax_patch_without_quant_config(monkeypatch, quant_config):
    backbone = torch.nn.Module()
    model = SimpleNamespace(
        model_config=SimpleNamespace(cache_rotary_embedding=True, quant_config=quant_config),
        unwrap=lambda: backbone,
        _inner=backbone,
    )
    from tensor_cast.transformers.builtin_model import minimax_m3

    calls = []
    names = [
        "patch_minimax_m3_attention",
        "patch_minimax_m3_layernorm",
        "patch_minimax_m3_dense_mlp",
        "patch_minimax_m3_moe_gate_attrs",
        "patch_moe",
    ]
    for name in names:

        def record(current, name=name):
            calls.append(name)
            return current

        monkeypatch.setattr(minimax_m3, name, record)
    assert patch_method_for_minimax_m3(model) is model
    assert calls == names


@pytest.mark.parametrize("grids", [((1, 32, 32), (1, 32, 32)), ((1, 16, 16), (1, 32, 32))])
@pytest.mark.parametrize("compiled", [False, True])
def test_vision_attention_preserves_image_boundaries_and_flops(grids, compiled):
    from tensor_cast.compilation import get_backend
    from tensor_cast.device import TEST_DEVICE
    from tensor_cast.layers.attention import AttentionTensorCast
    from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
    from tensor_cast.runtime import Runtime

    config = MiniMaxM3VLVisionConfig(hidden_size=96, num_attention_heads=4, num_hidden_layers=2, intermediate_size=192)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = MiniMaxM3VLVisionModel(config).eval()
    config._attn_implementation = "tensor_cast"
    model.forward = MethodType(_minimax_vision_meta_forward, model)
    attention_by_layers = {i: AttentionTensorCast() for i in range(2)}
    for i, layer in enumerate(model.layers):
        layer.self_attn._tensor_cast_context = {"depth_layer_idx": i, "attention_by_layers": attention_by_layers}
    lengths = [t * h * w for t, h, w in grids]
    pixels = torch.empty(sum(lengths), 1176, device="meta")
    grid = torch.tensor(grids)
    grid._minimax_patch_lengths = tuple(lengths)
    if compiled:
        model = torch.compile(model, backend=get_backend(), fullgraph=True)
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime, torch.no_grad():
        output = model(pixel_values=pixels, grid_thw=grid)
    assert output.last_hidden_state.shape == (1, sum(lengths), 96)
    ops = [
        e.op_invoke_info for e in runtime.event_list if str(e.op_invoke_info.func) == "tensor_cast.attention.default"
    ]
    expected_geometry = [(1, n) for n in lengths] * 2
    assert [(op.args[1].shape[0], op.args[1].shape[1]) for op in ops] == expected_geometry
    mma_ops = sum(c.mma_ops for op in ops for c in op.get_perf_properties().compute_ops.values())
    assert mma_ops == 2 * sum(n * n for n in lengths) * 96 * 4
    assert mma_ops < 2 * sum(lengths) ** 2 * 96 * 4
    if compiled and lengths[0] != lengths[1]:
        # Same tensor shapes but different image boundaries must specialize
        # again, rather than reusing a graph with stale grid values.
        reversed_grid = torch.tensor(tuple(reversed(grids)))
        reversed_grid._minimax_patch_lengths = tuple(reversed(lengths))
        with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as second_runtime, torch.no_grad():
            model(pixel_values=pixels, grid_thw=reversed_grid)
        second_ops = [
            e.op_invoke_info
            for e in second_runtime.event_list
            if str(e.op_invoke_info.func) == "tensor_cast.attention.default"
        ]
        assert [op.args[1].shape[1] for op in second_ops] == list(reversed(lengths)) * 2


def test_heterogeneous_minimax_input_retains_static_patch_lengths():
    from tensor_cast.core.vision_geometry import ImageSpec

    model = _model()
    result = inputs.generate_image_inputs_for_specs(
        model, [(ImageSpec(224, 224),), (ImageSpec(448, 448),)], concurrency=2
    )
    assert result["image_grid_thw"]._minimax_patch_lengths == (256, 1024)
    assert result["pixel_values"].shape[0] == 1280


@pytest.mark.parametrize("queries,dp,images,expected", [(2, 1, 1, 64), (8, 8, 1, 32), (1, 1, 2, 64)])
def test_minimax_verify_counts_per_image_attention(queries, dp, images, expected):
    from tensor_cast.adapter.expectations import ATTENTION_CATEGORY, derive_key_op_expectations

    structure = {
        "model_type": "minimax_m3_vl",
        "visual_module_paths": ["vision_tower"],
        "attention_like_modules": [{"path": f"vision_tower.layers.{i}.self_attn"} for i in range(32)],
    }
    user_input = SimpleNamespace(
        decode=False, image_batch_size=images, image_height=448, image_width=448, num_queries=queries, dp_size=dp
    )
    basis = derive_key_op_expectations(structure, user_input)
    assert basis.expectations[ATTENTION_CATEGORY].expected_count == expected

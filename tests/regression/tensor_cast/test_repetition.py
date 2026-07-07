import unittest
from types import SimpleNamespace

import pytest
import torch
from parameterized import parameterized
from tensor_cast.compilation import get_backend
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.internal import CopyLayerWrapper, RegionMarkerWrapper
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.model_config import ModelConfig, ParallelConfig, QuantConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.custom_model_registry import get_mtp_block_module_name
from tensor_cast.transformers.model import TransformerModel
from tensor_cast.transformers.transformations import maybe_reuse_layers

from .conftest import get_session_hf_config
from .test_common import (
    assert_close,
    create_mla_metadata_and_kv_cache,
    get_cached_build_model,
    has_submodule_with_cls_name,
)

# Core repetition layer-behavior assertions were moved to the unified entry in test_layers.py.


def test_glm5_indexer_flow_layers_are_not_reused(monkeypatch):
    class FakeAttention(torch.nn.Module):
        def __init__(self, layer_idx, skip_topk=False, next_skip_topk=False):
            super().__init__()
            self.layer_idx = layer_idx
            self.skip_topk = skip_topk
            self.next_skip_topk = next_skip_topk

    class FakeLayer(torch.nn.Module):
        def __init__(self, layer_idx, skip_topk=False, next_skip_topk=False):
            super().__init__()
            self.self_attn = FakeAttention(layer_idx, skip_topk=skip_topk, next_skip_topk=next_skip_topk)

    layers = torch.nn.ModuleList(
        [
            FakeLayer(0),
            FakeLayer(1),
            FakeLayer(2, next_skip_topk=True),
            FakeLayer(3, skip_topk=True, next_skip_topk=True),
            FakeLayer(4, skip_topk=True),
        ]
    )
    model = SimpleNamespace(
        model_config=SimpleNamespace(enable_repetition=True),
        is_vl_model=False,
        _inner=None,
        unwrap=lambda: SimpleNamespace(layers=layers),
    )
    monkeypatch.setattr("tensor_cast.transformers.transformations.get_visual_layers", lambda _model: None)

    maybe_reuse_layers(model)

    assert isinstance(layers[1], CopyLayerWrapper)
    assert not isinstance(layers[2], CopyLayerWrapper)
    assert not isinstance(layers[3], CopyLayerWrapper)
    assert not isinstance(layers[4], CopyLayerWrapper)


def test_glm5_indexer_flow_layers_are_not_reused_from_config(monkeypatch):
    class FakeAttention(torch.nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx

    class FakeLayer(torch.nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.self_attn = FakeAttention(layer_idx)

    layers = torch.nn.ModuleList([FakeLayer(i) for i in range(6)])
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        indexer_types=["full", "shared", "shared", "shared", "full", "full"],
    )
    model = SimpleNamespace(
        model_config=SimpleNamespace(enable_repetition=True),
        is_vl_model=False,
        _inner=SimpleNamespace(hf_config=hf_config),
        hf_config=hf_config,
        unwrap=lambda: SimpleNamespace(layers=layers),
    )
    monkeypatch.setattr("tensor_cast.transformers.transformations.get_visual_layers", lambda _model: None)

    maybe_reuse_layers(model)

    assert not isinstance(layers[0], CopyLayerWrapper)
    assert not isinstance(layers[1], CopyLayerWrapper)
    assert not isinstance(layers[2], CopyLayerWrapper)
    assert not isinstance(layers[3], CopyLayerWrapper)
    assert isinstance(layers[5], CopyLayerWrapper)


class RepetitionTestMixin:
    _model_cache: dict = {}

    @classmethod
    def setUpClass(cls):
        cls._transformer_cache = {}

    @classmethod
    def _get_transformer_model(cls, model_id: str, model_config: ModelConfig) -> TransformerModel:
        key = (model_id, repr(model_config))
        if key not in cls._transformer_cache:
            cls._transformer_cache[key] = TransformerModel(model_id, model_config)
        return cls._transformer_cache[key]

    def setUp(self):
        torch.compiler.reset()

    def check_num_effective_layers(self, layers, expected_num):
        count = sum(1 for layer in layers if not isinstance(layer, CopyLayerWrapper))
        self.assertEqual(count, expected_num, f"{layers}")

    def check_representative_layers(self, layers, expected_repeat_counts):
        region_layers = [layer for layer in layers if isinstance(layer, RegionMarkerWrapper)]
        self.assertEqual(len(region_layers), len(expected_repeat_counts), f"{layers}")
        for layer, expected_repeat_count in zip(region_layers, expected_repeat_counts):
            self.assertEqual(layer.repeat_count, expected_repeat_count)

    def check_copy_layers_hidden(self, layers):
        copy_layers = [layer for layer in layers if isinstance(layer, CopyLayerWrapper)]
        self.assertTrue(copy_layers, f"{layers}")
        for layer in copy_layers:
            self.assertEqual(list(layer.children()), [])
            self.assertEqual(list(layer.named_children()), [])


class RepetitionTestCase(RepetitionTestMixin, unittest.TestCase):
    def _run_test_vanilla_transformer_model(self, model_id, do_compile):
        num_tokens = 100
        # Note that specifying `AttentionTensorCast` as the `attention_cls`
        # is needed otherwise CSE would optimize out the attention mask
        # computation from the original attention implementation across layers,
        # resulting in larger op count gap between original trace and repetitive trace.
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=3,
        )
        model_config_with_repeats = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=3,
            enable_repetition=True,
        )
        model_config.hf_config = get_session_hf_config(model_id)
        model_config_with_repeats.hf_config = get_session_hf_config(model_id)
        model = self._get_transformer_model(model_id, model_config)
        model_with_repeats = self._get_transformer_model(model_id, model_config_with_repeats)
        self.check_num_effective_layers(model_with_repeats.unwrap().layers, 1)
        self.assertEqual(len(model_with_repeats.unwrap().layers), 3)
        self.check_representative_layers(model_with_repeats.unwrap().layers, [3])
        self.check_copy_layers_hidden(model_with_repeats.unwrap().layers)
        if do_compile:
            model = torch.compile(model, backend=get_backend(), dynamic=True, fullgraph=True)
            model_with_repeats = torch.compile(model_with_repeats, backend=get_backend(), dynamic=True, fullgraph=True)
        inputs = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (2, num_tokens, model.vocab_size))

        with (
            Runtime(perf_model, device_profile, MemoryTracker(device_profile)) as runtime_with_repeats,
            torch.no_grad(),
        ):
            outputs = model_with_repeats.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (2, num_tokens, model_with_repeats.vocab_size))

        # NOTE: we might miss some cross-layer fusion patterns with repetitions
        #       so we allow some errors here.
        assert_close(
            self,
            len(runtime.event_list),
            len(runtime_with_repeats.event_list),
            rtol=0.027 if do_compile else 0,
        )
        runtime_cost_s = runtime.total_execution_time_s()[perf_model.name]
        runtime_cost_with_repeats_s = runtime_with_repeats.total_execution_time_s()[perf_model.name]
        assert_close(
            self,
            runtime_cost_s,
            runtime_cost_with_repeats_s,
            rtol=0.01 if do_compile else 0,
        )
        peak_mem_usage = runtime.memory_tracker.peak_mem_usage()
        peak_mem_usage_with_repeats = runtime_with_repeats.memory_tracker.peak_mem_usage()
        assert_close(
            self,
            peak_mem_usage,
            peak_mem_usage_with_repeats,
        )

    def _run_test_deepseek_with_kvcache(self, model_id):
        num_mtp_layers = 3
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
        )

        model = get_cached_build_model(RepetitionTestMixin._model_cache, user_config)

        mtp_block_module_name = get_mtp_block_module_name(model.model_config.hf_config.model_type)
        self.assertIsNotNone(mtp_block_module_name)

        self.check_num_effective_layers(model.unwrap().layers, 2)
        self.assertEqual(len(model.unwrap().layers), model.text_config.num_hidden_layers)
        self.check_copy_layers_hidden(model.unwrap().layers)
        if model_id == "deepseek-ai/DeepSeek-V3.1":
            self.check_representative_layers(model.unwrap().layers, [3, 58])
        else:
            self.assertEqual(
                sum(layer.repeat_count for layer in model.unwrap().layers if isinstance(layer, RegionMarkerWrapper)),
                model.text_config.num_hidden_layers,
            )
        self.check_num_effective_layers(model._inner.mtp.layers, 1)
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model.model_config)
        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(
                    query_start_loc=attn_meta.query_start_loc,
                    selected_token_indices=None,
                ),
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))
        result = runtime.table_averages()
        start_str = "tensor_cast.multihead_latent_attention.default"
        end_str = "64"
        found = any(
            line.strip().startswith(start_str) and line.strip().endswith(end_str) for line in result.splitlines()
        )
        self.assertTrue(found, result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
        ]
    )
    def test_vanilla_transformer_model(self, model_id, do_compile):
        self._run_test_vanilla_transformer_model(model_id, do_compile)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
        ]
    )
    def test_deepseek_with_kvcache(self, model_id):
        self._run_test_deepseek_with_kvcache(model_id)


@pytest.mark.nightly
class RepetitionNightlyTestCase(RepetitionTestMixin, unittest.TestCase):
    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
        ]
    )
    def test_vanilla_transformer_model(self, model_id):
        RepetitionTestCase._run_test_vanilla_transformer_model(self, model_id, True)

    @parameterized.expand(
        [
            ["moonshotai/Kimi-K2-Base"],
        ]
    )
    def test_deepseek_with_kvcache(self, model_id):
        RepetitionTestCase._run_test_deepseek_with_kvcache(self, model_id)

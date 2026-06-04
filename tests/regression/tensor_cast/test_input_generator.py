from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensor_cast.core.input_generator import (
    RequestInfo,
    _layer_uses_sparse_attention_indexer,
    _resolve_decoder_layers,
    _resolve_sparse_attention_indexer_cache_width,
    _resolve_sparse_attention_kv_cache_width,
    generate_image_inputs,
    generate_inputs,
    generate_inputs_varlen,
    get_sparse_attention_indexer_cache_info,
)
from tensor_cast.layers.deepseek_v4 import DeepseekV4SparseAttention
from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.model import TransformerModel


@pytest.mark.parametrize("is_decode", [True, False])
def test_selected_token_indices_for_lmhead(qwen3_32b_lmhead_attention_transformer: TransformerModel, is_decode):
    model = qwen3_32b_lmhead_attention_transformer
    query_len = 100
    batch_size = 2
    inputs = generate_inputs(
        model,
        [
            RequestInfo(
                query_len=query_len,
                seq_len=query_len,
                concurrency=batch_size,
                is_decode=is_decode,
            )
        ],
    )
    if is_decode:
        output_shape = (1, batch_size * query_len, model.vocab_size)
    else:
        output_shape = (1, batch_size, model.vocab_size)

    machine_config = TEST_DEVICE
    perf_model = AnalyticPerformanceModel(machine_config)
    with Runtime(perf_model, machine_config), torch.no_grad():
        outputs = model.forward(**inputs)
    assert outputs.shape == output_shape


@pytest.mark.parametrize("is_decode", [True, False])
def test_varlen_selected_token_indices_for_lmhead(qwen3_32b_lmhead_attention_transformer: TransformerModel, is_decode):
    model = qwen3_32b_lmhead_attention_transformer
    query_len = [90, 110]
    batch_size = len(query_len)
    request_infos = []
    for i in range(batch_size):
        request_infos.append(
            RequestInfo(
                query_len=query_len[i],
                seq_len=query_len[i],
                is_decode=is_decode,
            )
        )
    inputs = generate_inputs_varlen(model, request_infos, 128)
    if is_decode:
        output_shape = (1, sum(query_len), model.vocab_size)
    else:
        output_shape = (1, batch_size, model.vocab_size)

    machine_config = TEST_DEVICE
    perf_model = AnalyticPerformanceModel(machine_config)
    with Runtime(perf_model, machine_config), torch.no_grad():
        outputs = model.forward(**inputs)
    assert outputs.shape == output_shape


_DSA_INDEXER_CACHE_QUERY_LEN = 32
_DSA_INDEXER_CACHE_NUM_MTP_TOKENS = 2
_DSA_INDEXER_CACHE_BLOCK_SIZE = 128
_DSA_INDEXER_CACHE_NUM_BLOCKS = (
    _DSA_INDEXER_CACHE_QUERY_LEN + _DSA_INDEXER_CACHE_NUM_MTP_TOKENS + 1 + _DSA_INDEXER_CACHE_BLOCK_SIZE - 1
) // _DSA_INDEXER_CACHE_BLOCK_SIZE


def test_dsa_indexer_cache_dtype_follows_attention_quant_config(
    deepseek_v32_build_model_int8,
):
    model = deepseek_v32_build_model_int8
    cache_info = get_sparse_attention_indexer_cache_info(
        model,
        num_blocks=_DSA_INDEXER_CACHE_NUM_BLOCKS,
        block_size=_DSA_INDEXER_CACHE_BLOCK_SIZE,
    )

    assert cache_info["indexer_cache_by_layers"][0].dtype == torch.int8


def test_dsa_indexer_cache_dtype_uses_fp8_when_attention_quant_is_fp8(
    deepseek_v32_build_model_fp8,
):
    model = deepseek_v32_build_model_fp8
    cache_info = get_sparse_attention_indexer_cache_info(
        model,
        num_blocks=_DSA_INDEXER_CACHE_NUM_BLOCKS,
        block_size=_DSA_INDEXER_CACHE_BLOCK_SIZE,
    )

    assert cache_info["indexer_cache_by_layers"][0].dtype == torch.float8_e4m3fn


def test_qwen3_vl_1080p_resize_to_1088x1920(
    qwen3_vl_8b_instruct_transformer: TransformerModel,
):
    model = qwen3_vl_8b_instruct_transformer

    image_kwargs = generate_image_inputs(
        model=model,
        image_batch_size=1,
        image_height=1080,
        image_width=1920,
        concurrency=1,
    )

    # grid_h=68, grid_w=120 -> resized height/width = 1088x1920
    assert torch.equal(image_kwargs["image_grid_thw"], torch.tensor([[1, 68, 120]]))


class TestSparseAttentionCacheHelpers:
    def test_resolve_kv_cache_width_from_attention_head_dim(self):
        model = MagicMock()
        model.text_config.kv_lora_rank = 512
        model.text_config.qk_rope_head_dim = 64
        attention = MagicMock(head_dim=480, _head_dim=None)
        assert _resolve_sparse_attention_kv_cache_width(model, attention) == 480

    def test_resolve_kv_cache_width_fallback_without_layer(self):
        model = MagicMock()
        model.text_config.kv_lora_rank = 512
        model.text_config.qk_rope_head_dim = 64
        assert _resolve_sparse_attention_kv_cache_width(model, None) == 576

    def test_resolve_indexer_cache_width_prefers_index_head_dim(self):
        model = MagicMock()
        model.text_config.index_head_dim = 999
        attention = MagicMock(_index_head_dim=128, indexer=None)
        assert _resolve_sparse_attention_indexer_cache_width(model, attention) == 128

    def test_resolve_indexer_cache_width_from_indexer_module(self):
        model = MagicMock()
        model.text_config.index_head_dim = None
        attention = MagicMock(_index_head_dim=None, indexer=MagicMock(head_dim=64))
        assert _resolve_sparse_attention_indexer_cache_width(model, attention) == 64

    def test_layer_uses_sparse_attention_indexer(self):
        assert _layer_uses_sparse_attention_indexer(MagicMock(use_indexer=True))
        assert not _layer_uses_sparse_attention_indexer(MagicMock(use_indexer=False, indexer=None))
        assert _layer_uses_sparse_attention_indexer(MagicMock(use_indexer=None, indexer=object()))

    def test_resolve_decoder_layers_direct_layout(self):
        layers = [MagicMock(), MagicMock()]
        model = MagicMock()
        model.unwrap.return_value = SimpleNamespace(layers=layers)
        assert _resolve_decoder_layers(model) is layers

    def test_resolve_decoder_layers_nested_causal_lm_layout(self):
        nested_layers = [MagicMock()]
        model = MagicMock()
        model.unwrap.return_value = SimpleNamespace(model=SimpleNamespace(layers=nested_layers))
        assert _resolve_decoder_layers(model) is nested_layers

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_get_sparse_attention_indexer_cache_info_v4_layers(self, _mock_attn_quant):
        model = MagicMock()
        model.num_hidden_layers = 2
        model.model_config.mla_config = MagicMock(mla_cls=DeepseekV4SparseAttention)
        model.model_config.dtype = torch.bfloat16
        model.unwrap.return_value = MagicMock(
            layers=[
                MagicMock(self_attn=MagicMock(use_indexer=False, indexer=None)),
                MagicMock(
                    self_attn=MagicMock(
                        use_indexer=True,
                        _index_head_dim=128,
                        indexer=None,
                    )
                ),
            ]
        )

        info = get_sparse_attention_indexer_cache_info(model, num_blocks=4, block_size=16)

        assert 1 in info["indexer_cache_by_layers"]
        assert info["indexer_cache_by_layers"][1].shape == (4, 16, 128)
        assert info["indexer_cache_per_token"] > 0

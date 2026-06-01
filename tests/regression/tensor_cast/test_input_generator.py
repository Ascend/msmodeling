import pytest
import torch
from tensor_cast.core.input_generator import RequestInfo, generate_image_inputs, generate_inputs, generate_inputs_varlen
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


def test_dsa_indexer_cache_dtype_follows_attention_quant_config(
    deepseek_v32_build_model_int8,
):
    model = deepseek_v32_build_model_int8
    cache_info = generate_inputs(
        model,
        [
            RequestInfo(
                query_len=32,
                seq_len=32,
                concurrency=1,
                is_decode=True,
            )
        ],
    )

    assert cache_info["indexer_cache_by_layers"][0].dtype == torch.int8


def test_dsa_indexer_cache_dtype_uses_fp8_when_attention_quant_is_fp8(
    deepseek_v32_build_model_fp8,
):
    model = deepseek_v32_build_model_fp8
    cache_info = generate_inputs(
        model,
        [
            RequestInfo(
                query_len=32,
                seq_len=32,
                concurrency=1,
                is_decode=True,
            )
        ],
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

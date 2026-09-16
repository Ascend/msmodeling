from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensor_cast.core.input_generator import (
    RequestInfo,
    _is_v4_model,
    _layer_uses_sparse_attention_indexer,
    _load_preprocessor_pixel_limits,
    _resolve_decoder_attention_layer,
    _resolve_decoder_layers,
    _resolve_draft_kv_cache_num_blocks,
    _resolve_dspark_kv_cache_num_blocks,
    _resolve_indexer_cache_dtype,
    _resolve_main_kv_cache_dtype,
    _resolve_sparse_attention_indexer_cache_width,
    _resolve_sparse_attention_kv_cache_width,
    _resolve_v4_kv_cache_size,
    generate_image_inputs,
    generate_inputs,
    generate_inputs_varlen,
    get_kv_cache_info,
    get_sparse_attention_indexer_cache_info,
    kv_cache_excluded_layer_indices,
    resize_image,
)
from tensor_cast.layers.deepseek_v4 import DeepseekV4SparseAttention
from tensor_cast.layers.glm5 import Glm5SparseAttention
from tensor_cast.layers.utils import ModelWrapperBase
from tensor_cast.layers.sampler import Sampler
from tensor_cast.model_config import MtpConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.utils import bytes_of_tensor
from tensor_cast.pipeline_parallel import (
    PipelineStageSpec,
    UnsupportedPPConfigurationError,
    _slice_layer_cache,
)
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.model import TransformerModel


def _fake_mtp_input_model(num_mtp_tokens=2):
    return SimpleNamespace(
        is_vl_model=False,
        num_hidden_layers=0,
        model_config=SimpleNamespace(
            mtp_config=MtpConfig(
                num_mtp_layers=num_mtp_tokens,
                mtp_block_module_name="DeepseekV3DecoderLayer",
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=1, tensor_parallel_size=1, decode_context_parallel_size=1
            ),
            mla_config=None,
            hf_config=SimpleNamespace(model_type="deepseek_v3"),
        ),
    )


def _fake_minimax_m3_indexer_cache_model():
    attention = SimpleNamespace(
        is_sparse_layer=True,
        indexer_head_dim=128,
    )
    layer = SimpleNamespace(self_attn=attention)
    unwrapped = SimpleNamespace(layers=[layer])
    return SimpleNamespace(
        unwrap=lambda: unwrapped,
        num_hidden_layers=1,
        text_config=SimpleNamespace(index_head_dim=128),
        model_config=SimpleNamespace(
            mla_config=None,
            dtype=torch.float16,
            hf_config=SimpleNamespace(model_type="minimax_m3_vl"),
        ),
    )


def test_resolve_decoder_layers_from_language_model_layout():
    layers = [SimpleNamespace()]
    model = SimpleNamespace(
        unwrap=lambda: SimpleNamespace(
            language_model=SimpleNamespace(layers=layers),
        )
    )

    assert _resolve_decoder_layers(model) is layers


def test_resolve_decoder_attention_layer_supports_attention_fallback():
    attention = torch.nn.Identity()

    assert _resolve_decoder_attention_layer(SimpleNamespace(attention=attention)) is attention


def test_resolve_decoder_attention_layer_prefers_self_attn():
    self_attn = torch.nn.Identity()
    attention = torch.nn.Linear(1, 1)

    assert _resolve_decoder_attention_layer(SimpleNamespace(self_attn=self_attn, attention=attention)) is self_attn


def test_resolve_decoder_attention_layer_unwraps_model_wrapper():
    attention = torch.nn.Identity()
    wrapper = ModelWrapperBase(attention)

    assert _resolve_decoder_attention_layer(SimpleNamespace(attention=wrapper)) is attention


@patch(
    "tensor_cast.core.input_generator._resolve_indexer_cache_dtype",
    return_value=torch.float16,
)
def test_minimax_m3_allocates_indexer_cache_without_mla(_mock_cache_dtype):
    cache_info = get_sparse_attention_indexer_cache_info(
        _fake_minimax_m3_indexer_cache_model(),
        num_blocks=34,
        block_size=128,
        batch_size=1,
        total_kv_tokens=4334,
    )

    assert cache_info["indexer_cache_by_layers"][0].shape == (34, 128, 128)
    assert cache_info["indexer_cache_per_token"] == 128 * torch.float16.itemsize


def _proposal_indices(spec_metadata):
    spec_window = spec_metadata.num_speculative_tokens + 1
    return spec_metadata.logits_indices.view(spec_metadata.num_active_requests, spec_window)[:, -1]


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


def test_generate_inputs_mtp_decode_does_not_select_all_packed_rows_for_target_lm_head():
    inputs = generate_inputs(
        _fake_mtp_input_model(),
        [RequestInfo(query_len=5, seq_len=16, concurrency=2, is_decode=True)],
    )

    spec_metadata = inputs["sampling_metadata"].spec_decode_metadata

    assert spec_metadata.logits_indices.tolist() != list(range(10))
    assert spec_metadata.logits_indices.tolist() == [2, 3, 4, 7, 8, 9]
    assert _proposal_indices(spec_metadata).tolist() == [4, 9]
    assert spec_metadata.num_active_requests == 2
    assert spec_metadata.num_speculative_tokens == 2
    assert inputs["sampling_metadata"].selected_token_indices is None


def test_generate_inputs_varlen_mtp_decode_does_not_reuse_padded_prefix_rows():
    inputs = generate_inputs_varlen(
        _fake_mtp_input_model(),
        [
            RequestInfo(query_len=5, seq_len=16, is_decode=True),
            RequestInfo(query_len=3, seq_len=12, is_decode=True),
        ],
        block_size=128,
    )

    spec_metadata = inputs["sampling_metadata"].spec_decode_metadata

    assert spec_metadata.logits_indices.tolist() != list(range(8))
    assert spec_metadata.logits_indices.tolist() == [2, 3, 4, 5, 6, 7]
    assert _proposal_indices(spec_metadata).tolist() == [4, 7]
    assert spec_metadata.num_active_requests == 2
    assert spec_metadata.num_speculative_tokens == 2
    assert inputs["sampling_metadata"].selected_token_indices is None


def test_generate_inputs_varlen_mtp_decode_uses_ordinary_selection_for_short_query_window():
    inputs = generate_inputs_varlen(
        _fake_mtp_input_model(),
        [
            RequestInfo(query_len=3, seq_len=16, is_decode=True),
            RequestInfo(query_len=2, seq_len=12, is_decode=True),
        ],
        block_size=128,
    )

    sampling_metadata = inputs["sampling_metadata"]
    next_tokens = Sampler()(torch.empty(1, 5, 8, device="meta"), sampling_metadata)

    assert sampling_metadata.spec_decode_metadata is None
    assert sampling_metadata.selected_token_indices is None
    assert next_tokens.shape == (2, 1)


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


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_qwen3_5_cache_position_starts_at_context(_mock_kv_cache, _mock_sparse_cache):
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5"),
            mtp_config=None,
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        )
    )
    requests = [
        RequestInfo(query_len=1, seq_len=2199, is_decode=True, context_length=2198),
        RequestInfo(query_len=2, seq_len=12, is_decode=True, context_length=10),
    ]

    inputs = generate_inputs_varlen(model, requests, 128)

    assert torch.equal(inputs["cache_position"], torch.tensor([2198, 10, 11], dtype=torch.long))
    assert inputs["cache_position"].tensor_cast_query_lens == (1, 2)
    assert inputs["cache_position"].tensor_cast_is_decode == (True, True)
    assert inputs["cache_position"].tensor_cast_has_previous_state


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_qwen3_5_decode_mtp_cache_position_metadata(_mock_kv_cache, _mock_sparse_cache):
    model = SimpleNamespace(
        is_vl_model=False,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5"),
            mtp_config=SimpleNamespace(num_mtp_layers=3),
            parallel_config=SimpleNamespace(data_parallel_size=1, decode_context_parallel_size=1),
        ),
    )

    inputs = generate_inputs(
        model,
        [
            RequestInfo(
                query_len=4,
                seq_len=2202,
                concurrency=21,
                is_decode=True,
                context_length=2198,
            )
        ],
    )

    cache_position = inputs["cache_position"]
    assert torch.equal(cache_position, torch.arange(2198, 2198 + 84, dtype=torch.long))
    assert cache_position.tensor_cast_query_lens == (4,) * 21
    assert cache_position.tensor_cast_is_decode == (True,) * 21
    assert cache_position.tensor_cast_has_previous_state
    assert cache_position.tensor_cast_base_decode_query_len == 1
    assert cache_position.tensor_cast_num_mtp_tokens == 3
    assert cache_position.tensor_cast_effective_decode_steps == 4


@patch(
    "tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info",
    return_value={},
)
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_varlen_qwen3_5_moe_text_cache_position_starts_at_context(_mock_kv_cache, _mock_sparse_cache):
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
            mtp_config=None,
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        )
    )
    requests = [
        RequestInfo(query_len=1, seq_len=2199, is_decode=True, context_length=2198),
        RequestInfo(query_len=2, seq_len=12, is_decode=True, context_length=10),
    ]

    inputs = generate_inputs_varlen(model, requests, 128)

    assert torch.equal(inputs["cache_position"], torch.tensor([2198, 10, 11], dtype=torch.long))
    assert inputs["cache_position"].tensor_cast_query_lens == (1, 2)
    assert inputs["cache_position"].tensor_cast_is_decode == (True, True)
    assert inputs["cache_position"].tensor_cast_has_previous_state


def test_kv_cache_excluded_layer_indices_qwen3_5():
    """Qwen3.5 returns the indices of linear_attention placeholder layers."""
    model = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5")),
        text_config=SimpleNamespace(
            layer_types=["attention", "linear_attention", "attention", "linear_attention"],
        ),
    )
    assert kv_cache_excluded_layer_indices(model) == {1, 3}


def test_kv_cache_excluded_layer_indices_qwen3_5_moe():
    """qwen3_5_moe is covered by the same exclusion rule."""
    model = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5_moe")),
        text_config=SimpleNamespace(layer_types=["linear_attention", "attention"]),
    )
    assert kv_cache_excluded_layer_indices(model) == {0}


def test_kv_cache_excluded_layer_indices_qwen3_5_moe_text():
    """qwen3_5_moe_text (Qwen3.8 text MoE) is covered by the same exclusion rule.

    Counting the linear_attention placeholder KV inflates kv_cache_per_token
    (root cause of device_memory_available_gb == 0). Regression for review
    feedback: the model_type must be in the exclusion list.
    """
    model = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5_moe_text")),
        text_config=SimpleNamespace(layer_types=["linear_attention", "attention", "linear_attention"]),
    )
    assert kv_cache_excluded_layer_indices(model) == {0, 2}


def test_kv_cache_excluded_layer_indices_non_qwen3_5_returns_empty():
    """Non-Qwen3.5 models keep the full accounting scope (empty exclusion set)."""
    model = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_next")),
        text_config=SimpleNamespace(layer_types=["attention", "linear_attention"]),
    )
    assert kv_cache_excluded_layer_indices(model) == set()


def test_kv_cache_excluded_layer_indices_missing_layer_types_returns_empty():
    """Defensive branch: missing or non-list layer_types yields an empty set."""
    model_missing = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5")),
        text_config=SimpleNamespace(),
    )
    assert kv_cache_excluded_layer_indices(model_missing) == set()
    model_non_list = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5")),
        text_config=SimpleNamespace(layer_types="attention"),
    )
    assert kv_cache_excluded_layer_indices(model_non_list) == set()


def test_slice_layer_cache_excludes_linear_attention_placeholders():
    """PP-stage KV cache accounting excludes linear_attention placeholders.

    Regression for Qwen3.5 + PP: ``selected_bytes`` / ``total_bytes`` must use
    the same scope as the (already excluded) global ``kv_cache_per_token``;
    otherwise the stage's ``kv_cache_bytes`` is inflated and
    ``device_memory_available_gb`` collapses to 0. The placeholder still stays
    in ``local_cache`` for layer-indexed forward.
    """
    real = torch.empty([2, 2], dtype=torch.float32)  # 16 bytes
    placeholder = torch.empty([2, 2], dtype=torch.float32)  # 16 bytes, excluded
    per_layer = bytes_of_tensor(real)
    input_kwargs = {
        "kv_cache_by_layers": {0: real, 1: placeholder, 2: real, 3: real},
        # global per-token already excludes the placeholder (see _get_kv_cache_info)
        "kv_cache_per_token": 6.0,
    }
    stage_spec = PipelineStageSpec(stage_id=0, pp_size=1, layer_start=0, layer_end=2, is_first=True, is_last=False)

    # With exclusion: stage [0, 2) covers layer 0 (real) + layer 1 (placeholder).
    local_cache, selected_bytes, per_token = _slice_layer_cache(
        input_kwargs,
        stage_spec,
        cache_key="kv_cache_by_layers",
        metric_key="kv_cache_per_token",
        excluded_layers=frozenset({1}),
    )
    assert set(local_cache.keys()) == {0, 1}  # placeholder kept for forward
    assert selected_bytes == per_layer  # only layer 0 counts
    assert per_token == pytest.approx(6.0 * per_layer / (3 * per_layer))

    # Without exclusion (legacy scope): both bytes and per-token are inflated.
    _, selected_bytes_full, per_token_full = _slice_layer_cache(
        input_kwargs,
        stage_spec,
        cache_key="kv_cache_by_layers",
        metric_key="kv_cache_per_token",
        excluded_layers=frozenset(),
    )
    assert selected_bytes_full == 2 * per_layer
    assert per_token_full == pytest.approx(6.0 * (2 * per_layer) / (4 * per_layer))


@patch("tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info", return_value={})
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_generate_inputs_sets_max_total_seq_len(_mock_kv_cache, _mock_sparse_cache):
    model = SimpleNamespace(
        is_vl_model=False,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v4"),
            mtp_config=None,
            parallel_config=SimpleNamespace(data_parallel_size=1, decode_context_parallel_size=1),
        ),
    )

    inputs = generate_inputs(
        model,
        [
            RequestInfo(
                query_len=102400,
                context_length=921600,
                seq_len=1024000,
                concurrency=1,
                is_decode=False,
            )
        ],
    )

    attention_meta = inputs["attention_meta"]
    assert attention_meta.max_total_seq_len == 1024000
    assert int(attention_meta.seq_lens.max().item()) == attention_meta.max_total_seq_len


@patch("tensor_cast.core.input_generator.get_sparse_attention_indexer_cache_info", return_value={})
@patch("tensor_cast.core.input_generator._get_kv_cache_info", return_value=({}, 0))
def test_generate_inputs_varlen_sets_max_total_seq_len(_mock_kv_cache, _mock_sparse_cache):
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v4"),
            mtp_config=None,
            parallel_config=SimpleNamespace(data_parallel_size=1, decode_context_parallel_size=1),
        ),
    )
    requests = [
        RequestInfo(query_len=102400, context_length=921600, seq_len=1024000, is_decode=False),
        RequestInfo(query_len=8192, context_length=253952, seq_len=262144, is_decode=False),
    ]

    inputs = generate_inputs_varlen(model, requests, 128)

    attention_meta = inputs["attention_meta"]
    assert attention_meta.max_total_seq_len == 1024000
    assert int(attention_meta.seq_lens.max().item()) == attention_meta.max_total_seq_len


def test_resize_image_uses_local_preprocessor_config(tmp_path):
    _load_preprocessor_pixel_limits.cache_clear()
    (tmp_path / "preprocessor_config.json").write_text(
        '{"size": {"shortest_edge": 65536, "longest_edge": 16777216}}',
        encoding="utf-8",
    )

    resized_height, resized_width = resize_image(
        str(tmp_path),
        "qwen3_5",
        1080,
        1920,
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
    )

    _load_preprocessor_pixel_limits.cache_clear()
    assert (resized_height, resized_width) == (1088, 1920)


def test_glm4v_dense_and_moe_use_the_same_glm_resize() -> None:
    resize_kwargs = {
        "model_id": "",
        "image_height": 32,
        "image_width": 32,
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
    }

    dense_size = resize_image(model_type="glm4v", **resize_kwargs)
    moe_size = resize_image(model_type="glm4v_moe", **resize_kwargs)
    qwen_size = resize_image(model_type="qwen3_vl", **resize_kwargs)

    assert dense_size == moe_size
    assert dense_size != qwen_size


def test_read_preprocessor_config_invalid_json_returns_none(tmp_path):
    _load_preprocessor_pixel_limits.cache_clear()
    (tmp_path / "preprocessor_config.json").write_text("not valid json", encoding="utf-8")

    with patch("tensor_cast.core.input_generator.logger") as mock_logger:
        from tensor_cast.core.input_generator import _read_preprocessor_config

        result = _read_preprocessor_config(tmp_path / "preprocessor_config.json")
        mock_logger.debug.assert_called_once()

    assert result is None


def test_read_preprocessor_config_missing_file_returns_none(tmp_path):
    _load_preprocessor_pixel_limits.cache_clear()

    from tensor_cast.core.input_generator import _read_preprocessor_config

    result = _read_preprocessor_config(tmp_path / "nonexistent.json")
    assert result is None


def test_resolve_local_preprocessor_config_non_dir_returns_none():
    _load_preprocessor_pixel_limits.cache_clear()

    from tensor_cast.core.input_generator import _resolve_local_preprocessor_config

    result = _resolve_local_preprocessor_config("not/a/real/path")
    assert result is None


def test_load_preprocessor_pixel_limits_no_config_json_returns_none(tmp_path):
    _load_preprocessor_pixel_limits.cache_clear()

    # Directory exists but has no preprocessor_config.json
    min_px, max_px = _load_preprocessor_pixel_limits(str(tmp_path))
    assert min_px is None
    assert max_px is None


def _total_kv_cache_bytes(inputs) -> int:
    return sum(bytes_of_tensor(t) for t in inputs["kv_cache_by_layers"].values())


@pytest.mark.parametrize("generate_fn", [generate_inputs, generate_inputs_varlen])
def test_dcp_does_not_shrink_gqa_kv_cache_when_kv_heads_cover_tp(qwen3_32b_lmhead_attention_transformer, generate_fn):
    """A GQA model with ``h_kv >= tp`` gets NO per-card KV footprint saving from DCP.

    DCP moves a rank from ``h_kv/tp`` heads over ``S`` to ``h_kv*dcp/tp`` heads over
    ``S/dcp``, so per-card KV bytes ``(h_kv*dcp/tp) * D * (S/dcp) == (h_kv/tp) * D * S``
    are invariant -- the head growth cancels the sequence shrink. This fixture is
    Qwen3-32B (``h_kv=8``) at ``tp=1``, so TP replicates nothing and the factor is 1.

    Regression: this used to assert ``bytes / dcp_size``, which fabricated a
    ``dcp``-fold footprint saving that no GQA deployment has. See
    ``dcp_kv_token_capacity_factor``; the MLA case (a real ``dcp`` saving) and the
    ``h_kv < tp`` de-duplication case are covered in ``test_dcp_modeling.py``.

    Both the fixed-batch (``generate_inputs``) and varlen entry points are checked,
    since ``text_generate`` uses the former and serving uses the latter.
    """
    model = qwen3_32b_lmhead_attention_transformer
    parallel_config = model.model_config.parallel_config
    block_size = 128
    request = RequestInfo(query_len=1, seq_len=8192, concurrency=2, is_decode=True)

    def _bytes_for(dcp):
        original = parallel_config.decode_context_parallel_size
        parallel_config.decode_context_parallel_size = dcp
        try:
            if generate_fn is generate_inputs:
                inputs = generate_inputs(model, [request], block_size=block_size)
            else:
                varlen_reqs = [RequestInfo(query_len=1, seq_len=8192, is_decode=True) for _ in range(2)]
                inputs = generate_inputs_varlen(model, varlen_reqs, block_size)
            return _total_kv_cache_bytes(inputs)
        finally:
            parallel_config.decode_context_parallel_size = original

    bytes_dcp1 = _bytes_for(1)
    assert bytes_dcp1 > 0
    # Exactly equal, not approximately: the factor is 1, so nothing divides num_blocks.
    for dcp in (2, 4, 8):
        assert _bytes_for(dcp) == bytes_dcp1, dcp


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


def test_qwen3_vl_1080p_resize_to_1088x1920(tmp_path):
    _load_preprocessor_pixel_limits.cache_clear()
    (tmp_path / "preprocessor_config.json").write_text(
        '{"size": {"shortest_edge": 65536, "longest_edge": 16777216}}',
        encoding="utf-8",
    )
    model = SimpleNamespace(
        model_id=str(tmp_path),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            parallel_config=SimpleNamespace(data_parallel_size=1),
            hf_config=SimpleNamespace(
                model_type="qwen3_vl",
                vision_config=SimpleNamespace(
                    patch_size=16,
                    spatial_merge_size=2,
                    temporal_patch_size=2,
                    in_channels=3,
                ),
            ),
        ),
    )

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

    def test_resolve_decoder_layers_with_pp_and_mtp(self):
        """PP+MTP: decoder layers resolved, MTP indices stay None (no AttributeError)."""
        stage0_layers = [MagicMock(), MagicMock(), MagicMock()]
        stage0_model = MagicMock()
        stage0_model.unwrap.return_value = SimpleNamespace(layers=stage0_layers)
        stage0 = SimpleNamespace(
            stage_spec=SimpleNamespace(layer_start=0),
            model=stage0_model,
        )
        stage1_layers = [MagicMock(), MagicMock()]
        stage1_model = MagicMock()
        stage1_model.unwrap.return_value = SimpleNamespace(layers=stage1_layers)
        stage1 = SimpleNamespace(
            stage_spec=SimpleNamespace(layer_start=3),
            model=stage1_model,
        )
        # num_hidden_layers=7 (5 decoder + 2 MTP), plan.num_hidden_layers=5 (decoder only)
        plan = SimpleNamespace(num_hidden_layers=5)
        inner = SimpleNamespace(
            stages=[stage0, stage1],
            num_hidden_layers=7,
            plan=plan,
        )
        model = MagicMock()
        model.unwrap.return_value = inner

        result = _resolve_decoder_layers(model)

        assert len(result) == 7
        assert result[0] is stage0_layers[0]
        assert result[2] is stage0_layers[2]
        assert result[3] is stage1_layers[0]
        assert result[4] is stage1_layers[1]
        assert result[5] is None  # MTP layer 0
        assert result[6] is None  # MTP layer 1

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


class TestBailingV3KvCacheHelpers:
    @pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16, torch.float32])
    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_kda_cache_is_fixed_per_request_state(self, _mock_attn_quant, model_dtype):
        attention_type = type("BailingMoeV3KimiDeltaAttention", (torch.nn.Module,), {})
        attention = attention_type()
        attention.num_heads = 8
        attention.head_dim = 4
        attention.conv_size = 3
        model = MagicMock()
        model.num_hidden_layers = 1
        model.model_config = SimpleNamespace(
            mla_config=SimpleNamespace(),
            dtype=model_dtype,
            hf_config=None,
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            has_draft_spec=lambda: False,
        )
        model.text_config = SimpleNamespace(model_type="bailing_hybrid")
        model.unwrap.return_value = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)],
        )

        cache_by_layers, cache_per_token = get_kv_cache_info(
            model,
            num_blocks=4,
            block_size=16,
            batch_size=3,
        )

        recurrent_state_bytes = 3 * 4 * 4 * 4 * torch.empty((), dtype=torch.float32).element_size()
        conv_state_bytes = 3 * 3 * 4 * 4 * 2 * torch.empty((), dtype=model_dtype).element_size()
        assert cache_by_layers[0].numel() == recurrent_state_bytes + conv_state_bytes
        assert cache_per_token == 0


class TestDeepseekV4KvCacheHelpers:
    @pytest.mark.parametrize(
        ("hf_model_type", "text_model_type", "expected"),
        [
            ("deepseek_v4", None, True),
            (None, "deepseek_v4", True),
            ("deepseek_v32", "deepseek_v32", False),
            (None, None, False),
        ],
    )
    def test_is_v4_model(self, hf_model_type, text_model_type, expected):
        model = MagicMock()
        model.model_config.hf_config = MagicMock(model_type=hf_model_type) if hf_model_type is not None else None
        model.text_config = MagicMock(model_type=text_model_type) if text_model_type is not None else None
        assert _is_v4_model(model) is expected

    @patch("tensor_cast.core.input_generator.get_attention_quant_config")
    def test_resolve_main_kv_cache_dtype_v4_uses_attention_quant(self, mock_get_attn_quant):
        model = MagicMock()
        model.model_config.dtype = torch.bfloat16
        model.model_config.hf_config = MagicMock(model_type="deepseek_v4")
        model.text_config = None

        for quant_dtype in (torch.float8_e4m3fn, torch.int8):
            mock_get_attn_quant.return_value = MagicMock(get_quant_dtype=lambda dtype=quant_dtype: dtype)
            assert _resolve_main_kv_cache_dtype(model, 0) == quant_dtype

    @patch("tensor_cast.core.input_generator.get_attention_quant_config")
    def test_resolve_main_kv_cache_dtype_non_v4_uses_attention_quant(self, mock_get_attn_quant):
        mock_get_attn_quant.return_value = MagicMock(get_quant_dtype=lambda: torch.float8_e4m3fn)
        model = MagicMock()
        model.model_config.dtype = torch.bfloat16
        model.model_config.hf_config = MagicMock(model_type="deepseek_v32")
        model.text_config = None

        assert _resolve_main_kv_cache_dtype(model, 0) == torch.float8_e4m3fn

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_resolve_main_kv_cache_dtype_non_v4_fallback_to_model_dtype(self, _mock_get_attn_quant):
        model = MagicMock()
        model.model_config.dtype = torch.bfloat16
        model.model_config.hf_config = MagicMock(model_type="deepseek_v32")
        model.text_config = None

        assert _resolve_main_kv_cache_dtype(model, 0) == torch.bfloat16

    @patch("tensor_cast.core.input_generator.get_attention_quant_config")
    def test_resolve_indexer_cache_dtype_uses_attention_quant(self, mock_get_attn_quant):
        mock_get_attn_quant.return_value = MagicMock(get_quant_dtype=lambda: torch.int8)
        model = MagicMock()
        model.model_config.dtype = torch.bfloat16

        assert _resolve_indexer_cache_dtype(model, 0) == torch.int8

    @patch(
        "tensor_cast.core.input_generator._resolve_sparse_attention_kv_cache_width",
        return_value=576,
    )
    def test_resolve_v4_kv_cache_size_compressed_sparse_layer(self, _mock_head_dim):
        model = MagicMock()
        model.text_config.sliding_window = 128
        model.text_config.kv_lora_rank = 512
        model.text_config.qk_rope_head_dim = 64
        attention_layer = MagicMock(compress_ratio=4, head_dim=576)

        shape = _resolve_v4_kv_cache_size(
            model,
            attention_layer=attention_layer,
            num_blocks=100,
            block_size=128,
            batch_size=2,
            total_kv_tokens=8192,
        )

        # window_slots=256, compressed_slots=2048 -> total_slots=2304 -> 18 blocks
        assert shape == [18, 128, 576]

    @patch(
        "tensor_cast.core.input_generator._resolve_sparse_attention_kv_cache_width",
        return_value=480,
    )
    def test_resolve_v4_kv_cache_size_fallback_without_batch_info(self, _mock_head_dim):
        model = MagicMock()
        model.text_config.sliding_window = 128
        attention_layer = MagicMock(compress_ratio=4)

        shape = _resolve_v4_kv_cache_size(
            model,
            attention_layer=attention_layer,
            num_blocks=42,
            block_size=128,
        )

        assert shape == [42, 128, 480]

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_v4_indexer_cache_compression_only_for_v4_model(self, _mock_attn_quant):
        model = MagicMock()
        model.num_hidden_layers = 1
        model.model_config.mla_config = MagicMock(mla_cls=DeepseekV4SparseAttention)
        model.model_config.dtype = torch.bfloat16
        model.model_config.hf_config = MagicMock(model_type="deepseek_v32")
        model.text_config = None
        model.unwrap.return_value = MagicMock(
            layers=[
                MagicMock(
                    self_attn=MagicMock(
                        use_indexer=True,
                        _index_head_dim=128,
                        indexer=None,
                        compress_ratio=4,
                    )
                )
            ]
        )

        info = get_sparse_attention_indexer_cache_info(
            model,
            num_blocks=100,
            block_size=128,
            batch_size=2,
            total_kv_tokens=8192,
        )

        # Non-V4 models must keep the full paged pool even when compress_ratio is set.
        assert info["indexer_cache_by_layers"][0].shape[0] == 100

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_glm5_indexshare_allocates_only_full_indexer_caches(self, _mock_attn_quant):
        model = MagicMock()
        model.num_hidden_layers = 6
        model.model_config.mla_config = MagicMock(mla_cls=Glm5SparseAttention)
        model.model_config.dtype = torch.bfloat16
        model.model_config.hf_config = SimpleNamespace(
            model_type="glm_moe_dsa",
            indexer_types=["full", "shared", "shared", "shared", "full", "shared"],
        )
        model.text_config = None
        model.unwrap.return_value = SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=MagicMock(use_indexer=True, _index_head_dim=128, indexer=None))
                for _ in range(6)
            ]
        )

        info = get_sparse_attention_indexer_cache_info(model, num_blocks=4, block_size=16)

        assert set(info["indexer_cache_by_layers"]) == {0, 4}
        assert info["indexer_cache_by_layers"][0].shape == (4, 16, 128)
        assert info["indexer_cache_by_layers"][4].shape == (4, 16, 128)
        assert info["indexer_cache_per_token"] == 2 * 128 * torch.bfloat16.itemsize

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_glm5_indexshare_skips_draft_layers(self, _mock_attn_quant):
        model = MagicMock()
        model.num_hidden_layers = 9
        model.model_config = SimpleNamespace(
            mla_config=SimpleNamespace(mla_cls=Glm5SparseAttention),
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                model_type="glm_moe_dsa",
                indexer_types=["full", "shared", "shared", "shared", "full", "shared"],
            ),
            has_draft_spec=lambda: True,
            draft_num_layers=lambda: 3,
        )
        model.text_config = None
        model.unwrap.return_value = SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=MagicMock(use_indexer=True, _index_head_dim=128, indexer=None))
                for _ in range(6)
            ]
        )

        info = get_sparse_attention_indexer_cache_info(model, num_blocks=4, block_size=16)

        assert set(info["indexer_cache_by_layers"]) == {0, 4}
        assert 6 not in info["indexer_cache_by_layers"]
        assert 7 not in info["indexer_cache_by_layers"]
        assert 8 not in info["indexer_cache_by_layers"]
        assert info["indexer_cache_per_token"] == 2 * 128 * torch.bfloat16.itemsize

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_glm5_without_indexshare_preserves_per_layer_cache_allocation(self, _mock_attn_quant):
        model = MagicMock()
        model.num_hidden_layers = 3
        model.model_config.mla_config = MagicMock(mla_cls=Glm5SparseAttention)
        model.model_config.dtype = torch.bfloat16
        model.model_config.hf_config = SimpleNamespace(
            model_type="glm_moe_dsa",
            indexer_types=["full", "full", "full"],
        )
        model.text_config = None
        model.unwrap.return_value = SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=MagicMock(use_indexer=True, _index_head_dim=128, indexer=None))
                for _ in range(3)
            ]
        )

        info = get_sparse_attention_indexer_cache_info(model, num_blocks=4, block_size=16)

        assert set(info["indexer_cache_by_layers"]) == {0, 1, 2}

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_glm5_shared_indexer_without_full_source_is_rejected(self, _mock_attn_quant):
        model = MagicMock()
        model.num_hidden_layers = 1
        model.model_config.mla_config = MagicMock(mla_cls=Glm5SparseAttention)
        model.model_config.dtype = torch.bfloat16
        model.model_config.hf_config = SimpleNamespace(model_type="glm_moe_dsa", indexer_types=["shared"])
        model.text_config = None
        model.unwrap.return_value = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=MagicMock(use_indexer=True, _index_head_dim=128, indexer=None))]
        )

        with pytest.raises(ValueError, match="Invalid GLM5 indexer_types for layer 0/1"):
            get_sparse_attention_indexer_cache_info(model, num_blocks=4, block_size=16)

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_glm5_indexshare_uses_mtp_extended_config(self, _mock_attn_quant):
        model = MagicMock()
        model.num_hidden_layers = 5
        model.model_config.mla_config = MagicMock(mla_cls=Glm5SparseAttention)
        model.model_config.dtype = torch.bfloat16
        model.model_config.hf_config = SimpleNamespace(
            model_type="glm_moe_dsa",
            indexer_types=["full", "shared"],
        )
        model._inner = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="glm_moe_dsa",
                indexer_types=["full", "shared", "shared", "shared", "full"],
            )
        )
        model.text_config = None
        model.unwrap.return_value = SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=MagicMock(use_indexer=True, _index_head_dim=128, indexer=None))
                for _ in range(5)
            ]
        )

        info = get_sparse_attention_indexer_cache_info(model, num_blocks=4, block_size=16)

        assert set(info["indexer_cache_by_layers"]) == {0, 4}


class TestDsparkKvCacheHelpers:
    def test_non_dspark_cache_keeps_original_block_count(self):
        model = SimpleNamespace(model_config=SimpleNamespace(dspark_config=None))
        assert _resolve_dspark_kv_cache_num_blocks(model, 32, 64, 2, 2000) == 32

    def _make_model(self, sliding_window=128):
        draft_config = SimpleNamespace(
            sliding_window=sliding_window,
            num_key_value_heads=8,
            head_dim=64,
            hidden_size=512,
            num_attention_heads=8,
        )
        model_config = SimpleNamespace(
            dtype=torch.float16,
            mla_config=None,
            hf_config=SimpleNamespace(model_type="qwen3"),
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
            dspark_config=SimpleNamespace(sliding_window=sliding_window),
            has_draft_spec=lambda: True,
            draft_num_layers=lambda: 1,
        )
        return SimpleNamespace(
            num_hidden_layers=1,
            head_dim=64,
            text_config=SimpleNamespace(num_key_value_heads=8),
            model_config=model_config,
            _inner=SimpleNamespace(draft_hf_config=draft_config),
        )

    def test_dspark_draft_cache_uses_sliding_window_blocks(self):
        model = self._make_model(sliding_window=128)

        blocks = _resolve_dspark_kv_cache_num_blocks(
            model,
            num_blocks=32,
            block_size=64,
            batch_size=2,
            total_kv_tokens=2000,
        )

        assert blocks == 4  # ceil(2 requests * 128 tokens / 64)

    def test_draft_window_blocks_round_each_request(self):
        model = self._make_model(sliding_window=130)

        blocks = _resolve_draft_kv_cache_num_blocks(
            model,
            draft_layer_idx=0,
            num_blocks=3,
            block_size=128,
            batch_size=2,
            total_kv_tokens=260,
            request_seq_lens=[130, 130],
        )

        # Each request needs ceil(130 / 128) pages; pages cannot be shared.
        assert blocks == 4

    def test_draft_window_blocks_keep_tail_after_sliding_window(self):
        model = self._make_model(sliding_window=2048)

        blocks = _resolve_draft_kv_cache_num_blocks(
            model,
            draft_layer_idx=0,
            num_blocks=32,
            block_size=128,
            batch_size=2,
            total_kv_tokens=2 * (2048 + 8),
            request_seq_lens=[2048, 2048],
            draft_tail_tokens=8,
        )

        # The ring holds 2048 context slots plus 8 draft slots per request.
        assert blocks == 34

    def test_draft_config_can_use_slots(self):
        class SlotConfig:
            __slots__ = ("dspark_config", "dflash_config")

            def __init__(self):
                self.dspark_config = SimpleNamespace(sliding_window=128)
                self.dflash_config = None

        model = SimpleNamespace(model_config=SlotConfig())

        assert _resolve_draft_kv_cache_num_blocks(model, 0, 32, 64, 2, 2000) == 4

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_dspark_cache_shape_and_per_token_cost_use_window(self, _mock_attn_quant):
        model = self._make_model(sliding_window=128)

        cache_by_layers, cache_per_token = get_kv_cache_info(
            model,
            num_blocks=32,
            block_size=64,
            batch_size=2,
            total_kv_tokens=2000,
        )

        assert tuple(cache_by_layers[0].shape) == (2, 4, 64, 1, 64)
        assert cache_per_token == 2 * 1 * 64 * 2

    def test_dflash_mixed_layer_types_use_per_layer_blocks(self):
        draft_hf = SimpleNamespace(num_key_value_heads=8, head_dim=64, hidden_size=512, num_attention_heads=8)
        model_config = SimpleNamespace(
            dtype=torch.float16,
            mla_config=None,
            hf_config=SimpleNamespace(model_type="qwen3"),
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
            dflash_config=SimpleNamespace(sliding_window=128, layer_types=["sliding_attention", "full_attention"]),
            has_draft_spec=lambda: True,
            draft_num_layers=lambda: 2,
        )
        model = SimpleNamespace(
            num_hidden_layers=2,
            head_dim=64,
            text_config=SimpleNamespace(num_key_value_heads=8),
            model_config=model_config,
            _inner=SimpleNamespace(draft_hf_config=draft_hf),
        )

        assert _resolve_draft_kv_cache_num_blocks(model, 0, 32, 64, 2, 2000) == 4
        assert _resolve_draft_kv_cache_num_blocks(model, 1, 32, 64, 2, 2000) == 32

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_dflash_mixed_layer_cache_shapes(self, _mock_attn_quant):
        draft_hf = SimpleNamespace(num_key_value_heads=8, head_dim=64, hidden_size=512, num_attention_heads=8)
        model_config = SimpleNamespace(
            dtype=torch.float16,
            mla_config=None,
            hf_config=SimpleNamespace(model_type="qwen3"),
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
            dflash_config=SimpleNamespace(sliding_window=128, layer_types=["sliding_attention", "full_attention"]),
            has_draft_spec=lambda: True,
            draft_num_layers=lambda: 2,
        )
        model = SimpleNamespace(
            num_hidden_layers=2,
            head_dim=64,
            text_config=SimpleNamespace(num_key_value_heads=8),
            model_config=model_config,
            _inner=SimpleNamespace(draft_hf_config=draft_hf),
        )

        cache_by_layers, _cache_per_token = get_kv_cache_info(
            model, num_blocks=32, block_size=64, batch_size=2, total_kv_tokens=2000
        )

        assert tuple(cache_by_layers[0].shape) == (2, 4, 64, 1, 64)
        assert tuple(cache_by_layers[1].shape) == (2, 32, 64, 1, 64)

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_dflash_mixed_layer_cache_keeps_tail_and_ignores_target_dcp_shard(self, _mock_attn_quant):
        draft_hf = SimpleNamespace(num_key_value_heads=8, head_dim=64, hidden_size=512, num_attention_heads=8)
        model_config = SimpleNamespace(
            dtype=torch.float16,
            mla_config=SimpleNamespace(),
            hf_config=SimpleNamespace(model_type="qwen3"),
            parallel_config=SimpleNamespace(tensor_parallel_size=8, decode_context_parallel_size=8),
            dflash_config=SimpleNamespace(
                sliding_window=2048,
                layer_types=["sliding_attention", "full_attention"],
            ),
            has_draft_spec=lambda: True,
            draft_num_layers=lambda: 2,
        )
        model = SimpleNamespace(
            num_hidden_layers=2,
            head_dim=64,
            text_config=SimpleNamespace(num_key_value_heads=8),
            model_config=model_config,
            _inner=SimpleNamespace(draft_hf_config=draft_hf),
        )

        cache_by_layers, _cache_per_token = get_kv_cache_info(
            model,
            num_blocks=65,
            block_size=128,
            batch_size=1,
            total_kv_tokens=65544,
            request_seq_lens=[65536],
            draft_tail_tokens=8,
            unsharded_num_blocks=513,
        )

        # The sliding layer keeps its 2048-slot window plus draft block, while
        # the full layer must not inherit the target MLA DCP=8 shard (65 pages).
        assert tuple(cache_by_layers[0].shape) == (2, 17, 128, 1, 64)
        assert tuple(cache_by_layers[1].shape) == (2, 513, 128, 1, 64)


class TestPPDraftKvCacheLayout:
    """PP draft KV cache must follow the draft (Qwen3 GQA) layout, not the target's.

    PipelineModel exposes draft_hf_config from its last stage's speculative
    wrapper; falling back to the target layout silently would mis-size draft
    KV bytes whenever target and draft head configs differ.
    """

    @staticmethod
    def _make_pp_model(draft_cfg=None):
        # 5 target layers (kv_heads=4, head_dim=4) + 2 draft layers (kv_heads=1, head_dim=3)
        model = SimpleNamespace(
            num_hidden_layers=7,
            head_dim=4,
            text_config=SimpleNamespace(num_key_value_heads=4, num_attention_heads=4, hidden_size=16),
        )
        model.model_config = SimpleNamespace(
            mla_config=None,
            dtype=torch.float16,
            hf_config=SimpleNamespace(model_type="fake_decoder"),
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            dflash_config=SimpleNamespace(),
            draft_num_layers=lambda: 2,
        )
        if draft_cfg is not None:
            # PipelineModel.draft_hf_config surface (resolved from the last stage)
            model.draft_hf_config = draft_cfg
        return model

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_draft_layers_use_draft_gqa_layout_from_last_stage(self, _mock_quant):
        draft_cfg = SimpleNamespace(
            num_key_value_heads=1,
            num_attention_heads=2,
            hidden_size=8,
            head_dim=3,
        )
        model = self._make_pp_model(draft_cfg)

        cache_by_layers, cache_per_token = get_kv_cache_info(model, num_blocks=4, block_size=16)

        # Target layers keep the target GQA layout: [2, blocks, block, 4/tp=2, head_dim=4]
        assert cache_by_layers[0].shape == (2, 4, 16, 2, 4)
        # Draft layers must use the draft config, NOT the target layout
        assert cache_by_layers[5].shape == (2, 4, 16, 1, 3)
        assert cache_by_layers[6].shape == (2, 4, 16, 1, 3)

        elem = torch.empty((), dtype=torch.float16).element_size()
        target_bytes = 5 * (2 * 4 * 16 * 2 * 4) * elem
        draft_bytes = 2 * (2 * 4 * 16 * 1 * 3) * elem
        assert cache_per_token == (target_bytes + draft_bytes) / (4 * 16)

    def test_unreachable_draft_config_refuses_target_layout_fallback(self):
        model = self._make_pp_model(draft_cfg=None)

        with pytest.raises(UnsupportedPPConfigurationError, match="refusing to allocate draft KV cache"):
            get_kv_cache_info(model, num_blocks=4, block_size=16)


@pytest.mark.parametrize("quantized,expected_bytes", [(False, 1024), (True, 576)])
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_v4_main_cache_allocation_mixed_bytes(quantized, expected_bytes, ratio):
    from tensor_cast.core.input_generator import get_kv_cache_info

    model = MagicMock()
    model.num_hidden_layers = 1
    model.model_config.mla_config = SimpleNamespace()
    model.model_config.dtype = torch.bfloat16
    model.model_config.hf_config = SimpleNamespace(model_type="deepseek_v4")
    model.model_config.has_draft_spec.return_value = False
    model.text_config = SimpleNamespace(model_type="deepseek_v4", qk_rope_head_dim=64, sliding_window=128)
    model.unwrap.return_value = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(compress_ratio=ratio))]
    )
    quant = MagicMock(get_quant_dtype=lambda: torch.float8_e4m3fn) if quantized else None
    with (
        patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=quant),
        patch("tensor_cast.core.input_generator.kv_cache_excluded_layer_indices", return_value=set()),
        patch("tensor_cast.core.input_generator._resolve_sparse_attention_kv_cache_width", return_value=512),
    ):
        caches, per_token = get_kv_cache_info(model, 256, 16, batch_size=1, total_kv_tokens=4096)
    slots = 128 + (4096 // ratio if ratio else 0)
    slots = (slots + 15) // 16 * 16
    assert caches[0].numel() * caches[0].element_size() == slots * expected_bytes
    assert per_token == slots * expected_bytes / 4096

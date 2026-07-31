import math
from types import SimpleNamespace

import pytest
import torch

from tensor_cast.layers.minimax_m3_attention import (
    GemmaRMSNormFusedWrapper,
    MiniMaxM3AttentionWrapper,
    RMSNormFusedWrapper,
    _fused_decoder_layer_forward,
    _get_eps,
)
from tensor_cast.ops.swiglu import _symmetric_quant_scale_shape as _op_symmetric_quant_scale_shape
from tensor_cast.performance_model import (
    _m3_swiglu_quant_properties_helper,
    _minimax_m3_is_prefill_request,
    _safe_tensor_int_list,
    _estimate_minimax_m3_prefill_sparse_attention_kv_read_pairs,
    _estimate_minimax_indexer_breakdown,
    _estimate_minimax_sparse_attention_breakdown,
    _symmetric_quant_scale_shape,
)
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.transformers.builtin_model.minimax_m3 import (
    _get_minimax_m3_effective_text_config,
    _resolve_minimax_m3_sparse_attention_config,
)
from tensor_cast.transformers.custom_model_registry import get_model_profile
from tensor_cast.transformers.transformations import shard_model_by_tp


class _NormWithEps(torch.nn.Module):
    def __init__(self, width: int, eps: float = 1e-5, *, gemma: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))
        if gemma:
            self.variance_epsilon = eps
        else:
            self.eps = eps


class _IdentityNorm(torch.nn.Module):
    def forward(self, x):
        return x


class _DummyIndexer(torch.nn.Module):
    def __init__(self, hidden_size: int, indexer_head_dim: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, indexer_head_dim, bias=True)
        self.k_proj = torch.nn.Linear(hidden_size, indexer_head_dim, bias=True)
        self.q_norm = _IdentityNorm()
        self.k_norm = _IdentityNorm()


class _DummyM3Attention(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 8,
        num_q_heads: int = 2,
        num_kv_heads: int = 1,
        head_dim: int = 4,
        indexer_head_dim: int = 4,
    ):
        super().__init__()
        self.layer_idx = 0
        self.q_proj = torch.nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Identity()
        self.q_norm = _IdentityNorm()
        self.k_norm = _IdentityNorm()
        self.indexer = _DummyIndexer(hidden_size, indexer_head_dim)


class _FakeParallelConfig:
    embedding_parallel = None
    pipeline_parallel_size = 1
    source_pipeline_parallel_size = 1

    @staticmethod
    def has_ep():
        return False


class _FakeMtpAttentionWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._inner = torch.nn.Module()
        self._inner.q_proj = torch.nn.Linear(8, 16, bias=False)
        self._inner.k_proj = torch.nn.Linear(8, 8, bias=False)
        self._inner.v_proj = torch.nn.Linear(8, 8, bias=False)


class _FakeMtpBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeMtpAttentionWrapper()


class _FakeMtpLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mtp_block = _FakeMtpBlock()


class _FakeMiniMaxM3ModelForTp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList([_FakeMtpBlock()])
        self.mtp = torch.nn.Module()
        self.mtp.layers = torch.nn.ModuleList([_FakeMtpLayer()])


class _FakeTransformerModelForTp:
    def __init__(self):
        group = SimpleNamespace(world_size=2, rank_in_group=0)
        self._inner = _FakeMiniMaxM3ModelForTp()
        self.parallel_group_manager = SimpleNamespace(
            tp_group=group,
            o_proj_tp_group=group,
            mlp_tp_group=group,
            vision_tp_group=group,
            moe_tp_group=group,
            lmhead_tp_group=group,
        )
        self.model_config = SimpleNamespace(
            parallel_config=_FakeParallelConfig(),
            mtp_config=SimpleNamespace(num_mtp_layers=1, mtp_block_module_name="MiniMaxM3VLDecoderLayer"),
            mla_config=None,
            moe_config=None,
        )
        self.hf_config = SimpleNamespace(
            model_type="minimax_m3_vl",
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        self.text_config = self.hf_config
        self.is_vl_model = False

    def _replace_module(self, name: str, new_module: torch.nn.Module):
        path = name.split(".")
        parent = self._inner.get_submodule(".".join(path[:-1]))
        setattr(parent, path[-1], new_module)


def _make_attention_meta(*, total_tokens: int, seq_len: int, query_len: int, block_count: int = 2):
    return SimpleNamespace(
        seq_lens=torch.tensor([seq_len], dtype=torch.long),
        query_lens=torch.tensor([query_len], dtype=torch.long),
        block_table_tensor=torch.zeros(1, block_count, dtype=torch.long),
        query_start_loc=torch.tensor([0, total_tokens], dtype=torch.long),
        slot_mapping=torch.arange(total_tokens, dtype=torch.long),
        seq_lens_values=[seq_len],
        query_lens_values=[query_len],
        is_decode_values=[query_len == 1],
    )


def test_minimax_m3_norm_wrappers_and_eps_paths():
    x = torch.empty(2, 4, dtype=torch.float16, device="meta")
    residual = torch.empty_like(x)

    gemma_norm = _NormWithEps(4, eps=1e-6, gemma=True)
    gemma_wrapper = GemmaRMSNormFusedWrapper(gemma_norm)
    out, residual_out = gemma_wrapper(x, residual)
    assert out.shape == x.shape
    assert residual_out.shape == residual.shape
    assert _get_eps(gemma_norm) == 1e-6

    rms_norm = _NormWithEps(4, eps=1e-5)
    rms_wrapper = RMSNormFusedWrapper(rms_norm)
    assert rms_wrapper(x).shape == x.shape
    assert _get_eps(rms_norm) == 1e-5

    with pytest.raises(AttributeError, match="neither 'variance_epsilon' nor 'eps'"):
        _get_eps(torch.nn.Identity())


def test_minimax_m3_attention_wrapper_dense_and_sparse_paths():
    hidden_states = torch.randn(2, 8)
    cos = torch.ones(2, 4)
    sin = torch.zeros(2, 4)
    attention_meta = _make_attention_meta(total_tokens=2, seq_len=4, query_len=2)

    dense_wrapper = MiniMaxM3AttentionWrapper(
        original_module=_DummyM3Attention(),
        is_sparse_layer=False,
        hidden_size=8,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        num_indexer_heads=1,
        indexer_head_dim=4,
        topk_blocks=2,
        block_size=4,
        local_blocks=1,
        rotary_dim=4,
    )
    dense_out, dense_aux = dense_wrapper(
        hidden_states,
        attention_meta=attention_meta,
        position_embeddings=(cos, sin),
    )
    assert dense_out.shape == hidden_states.shape
    assert dense_aux is None

    sparse_wrapper = MiniMaxM3AttentionWrapper(
        original_module=_DummyM3Attention(),
        is_sparse_layer=True,
        hidden_size=8,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        num_indexer_heads=1,
        indexer_head_dim=4,
        topk_blocks=2,
        block_size=4,
        local_blocks=1,
        rotary_dim=4,
    )
    sparse_out, sparse_aux = sparse_wrapper(
        hidden_states,
        attention_meta=attention_meta,
        position_embeddings=(cos, sin),
    )
    assert sparse_out.shape == hidden_states.shape
    assert sparse_aux is None


def test_minimax_m3_attention_wrapper_infers_actual_projection_heads():
    hidden_states = torch.randn(2, 8)
    cos = torch.ones(2, 4)
    sin = torch.zeros(2, 4)
    attention_meta = _make_attention_meta(total_tokens=2, seq_len=4, query_len=2)
    original_attention = _DummyM3Attention(
        hidden_size=8,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=4,
    )
    original_attention.o_proj = torch.nn.Linear(16, 8, bias=False)

    wrapper = MiniMaxM3AttentionWrapper(
        original_module=original_attention,
        is_sparse_layer=False,
        hidden_size=8,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        num_indexer_heads=1,
        indexer_head_dim=4,
        topk_blocks=2,
        block_size=4,
        local_blocks=1,
        rotary_dim=4,
    )

    out, aux = wrapper(
        hidden_states,
        attention_meta=attention_meta,
        position_embeddings=(cos, sin),
    )

    assert out.shape == hidden_states.shape
    assert aux is None


def test_minimax_m3_profile_declares_mtp_block():
    profile = get_model_profile("minimax_m3_vl")

    assert profile.mtp_block_module_name == "MiniMaxM3VLDecoderLayer"
    assert profile.language_module_path == "language_model"


def test_minimax_m3_nested_sparse_attention_config_is_resolved():
    text_config = SimpleNamespace(
        hidden_size=6144,
        num_attention_heads=64,
        sparse_attention_config={
            "sparse_attention_freq": [0, 0, 0, 1],
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 128,
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
            "sparse_local_block": 1,
        },
    )
    vl_config = SimpleNamespace(text_config=text_config)
    model = SimpleNamespace(text_config=vl_config, _inner=SimpleNamespace(hf_config=vl_config))

    resolved_text_config = _get_minimax_m3_effective_text_config(model)
    sparse_config = _resolve_minimax_m3_sparse_attention_config(resolved_text_config)

    assert resolved_text_config is text_config
    assert sparse_config == ([0, 0, 0, 1], 4, 128, 16, 128, 1)


def test_minimax_m3_mtp_attention_qkv_uses_tp_plan():
    model = _FakeTransformerModelForTp()

    shard_model_by_tp(model)

    mtp_attention = model._inner.mtp.layers[0].mtp_block.self_attn._inner
    assert type(mtp_attention.q_proj).__name__ == "ColumnParallelLinear"
    assert type(mtp_attention.k_proj).__name__ == "ColumnParallelLinear"
    assert type(mtp_attention.v_proj).__name__ == "ColumnParallelLinear"
    assert mtp_attention.q_proj._inner.out_features == 8
    assert mtp_attention.k_proj._inner.out_features == 4
    assert mtp_attention.v_proj._inner.out_features == 4


def test_fused_decoder_layer_forward_preserves_source_order():
    class _Decoder:
        def __init__(self):
            self.input_layernorm = torch.nn.Identity()
            self.self_attn = lambda hidden_states, **kwargs: (hidden_states + 1, None)
            self.post_attention_layernorm = lambda hidden_states, residual: (hidden_states + residual, residual)
            self.mlp = lambda hidden_states: hidden_states * 2

    hidden_states = torch.ones(2, 4)

    out = _fused_decoder_layer_forward(_Decoder(), hidden_states)

    torch.testing.assert_close(out, torch.full_like(hidden_states, 7.0))


def test_minimax_m3_sparse_attention_meta_ops_preserve_shapes_and_dtypes():
    idx_q = torch.empty(3, 1, 4, dtype=torch.float16, device="meta")
    idx_k_cache = torch.empty(4, 4, 1, 4, dtype=torch.float16, device="meta")
    seq_lens = torch.tensor([3], dtype=torch.long)
    query_lens = torch.tensor([3], dtype=torch.long)
    block_table = torch.zeros(1, 1, dtype=torch.long)

    topk_idx = torch.ops.tensor_cast.minimax_indexer.default(
        idx_q,
        idx_k_cache,
        seq_lens,
        query_lens,
        block_table,
        topk_blocks=2,
        block_size=4,
        seq_lens_values=[3],
        query_lens_values=[3],
        is_decode_values=[False],
    )
    assert topk_idx.shape == (3, 1, 2)
    assert topk_idx.dtype == torch.int32

    query = torch.empty(3, 8, dtype=torch.float16, device="meta")
    out = torch.ops.tensor_cast.minimax_sparse_attention.default(
        query,
        idx_k_cache,
        idx_k_cache,
        topk_idx,
        seq_lens,
        query_lens,
        block_table,
        hidden_size=8,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        topk_blocks=2,
        block_size=4,
        local_blocks=1,
        seq_lens_values=[3],
        query_lens_values=[3],
        is_decode_values=[False],
    )
    assert out.shape == query.shape
    assert out.dtype == query.dtype


def test_minimax_m3_related_meta_ops_preserve_contracts():
    x = torch.empty(2, 8, dtype=torch.float16, device="meta")
    gate = torch.empty(2, 4, dtype=torch.float16, device="meta")
    up = torch.empty_like(gate)

    weights, indices = torch.ops.tensor_cast.moe_gating_top_k_sigmoid.default(x, 2, 2.0, None)
    assert weights.shape == (2, 2)
    assert weights.dtype == x.dtype
    assert indices.shape == (2, 2)
    assert indices.dtype == torch.int64

    q = torch.empty(2, 1, 4, dtype=torch.float16, device="meta")
    k = torch.empty_like(q)
    cos_sin = torch.empty(2, 8, dtype=torch.float16, device="meta")
    q_out, k_out = torch.ops.tensor_cast.fused_rope.default(q, k, cos_sin, 4, True)
    assert q_out.shape == q.shape
    assert q_out.dtype == q.dtype
    assert k_out.shape == k.shape
    assert k_out.dtype == k.dtype

    assert torch.ops.tensor_cast.swiglu.default(gate, up).shape == gate.shape
    assert torch.ops.tensor_cast.m3_swiglu.default(gate, up, 1.702, 7.0).shape == gate.shape
    quant_out, scale = torch.ops.tensor_cast.m3_swiglu_quant.default(gate, up, 1.702, 7.0, 128)
    assert quant_out.shape == gate.shape
    assert quant_out.dtype == torch.int8
    assert scale.shape == (2, 1)
    assert _op_symmetric_quant_scale_shape(torch.Size([2, 4]), []) == torch.Size([])
    assert _op_symmetric_quant_scale_shape(torch.Size([2, 4]), [-1]) == torch.Size([2, 1])
    with pytest.raises(RuntimeError, match="Shape mismatch in m3_swiglu"):
        torch.ops.tensor_cast.m3_swiglu.default(gate, torch.empty(2, 5, device="meta"), 1.702, 7.0)


def test_minimax_m3_performance_model_helpers_and_registered_ops():
    assert _safe_tensor_int_list(torch.empty(3, dtype=torch.long, device="meta"), fallback_total=8) == [3, 3, 2]
    assert _safe_tensor_int_list(torch.tensor([2, 5], dtype=torch.long)) == [2, 5]
    assert _symmetric_quant_scale_shape(torch.Size([2, 4]), [-1]) == torch.Size([2, 1])
    assert _symmetric_quant_scale_shape(torch.Size([2, 4]), []) == torch.Size([])
    assert _minimax_m3_is_prefill_request(1, 0, [True]) is False
    assert _minimax_m3_is_prefill_request(4, 0, None) is True

    key = torch.empty(2, 4, dtype=torch.float16, device="meta")
    kv_cache = torch.empty(4, 4, dtype=torch.bfloat16, device="meta")
    slot_mapping = torch.empty(2, dtype=torch.long, device="meta")
    siso_info = OpInvokeInfo(
        torch.ops.tensor_cast.siso_reshape_and_cache.default,
        (key, kv_cache, slot_mapping),
        {},
        None,
    )
    siso_props = siso_info.get_perf_properties()
    assert siso_props.memory_write_bytes == key.numel() * kv_cache.element_size()

    gate = torch.empty(2, 8, dtype=torch.bfloat16, device="meta")
    up = torch.empty_like(gate)
    quant_out = (
        torch.empty_like(gate, dtype=torch.int8),
        torch.empty(2, 1, dtype=torch.float32, device="meta"),
    )
    quant_info = OpInvokeInfo(
        torch.ops.tensor_cast.m3_swiglu_quant.default,
        (gate, up, 1.702, 7.0, 128),
        {},
        quant_out,
    )
    quant_props = _m3_swiglu_quant_properties_helper(quant_info)
    assert quant_props.compute_ops[torch.bfloat16].gp_ops == gate.numel() * 7
    assert quant_info.get_perf_properties().compute_ops[torch.bfloat16].gp_ops == gate.numel() * 7

    query = torch.empty(3, 8, dtype=torch.float16, device="meta")
    key_cache = torch.empty(4, 4, 1, 4, dtype=torch.float16, device="meta")
    value_cache = torch.empty_like(key_cache)
    topk_idx = torch.empty(3, 1, 2, dtype=torch.int32, device="meta")
    seq_lens = torch.empty(1, dtype=torch.long, device="meta")
    query_lens = torch.empty(1, dtype=torch.long, device="meta")
    block_table = torch.empty(1, 1, dtype=torch.long, device="meta")
    sparse_out = torch.empty_like(query)
    sparse_info = OpInvokeInfo(
        torch.ops.tensor_cast.minimax_sparse_attention.default,
        (query, key_cache, value_cache, topk_idx, seq_lens, query_lens, block_table),
        {
            "hidden_size": 8,
            "num_q_heads": 2,
            "num_kv_heads": 1,
            "head_dim": 4,
            "topk_blocks": 2,
            "block_size": 4,
            "local_blocks": 1,
            "seq_lens_values": [3],
            "query_lens_values": [3],
            "is_decode_values": [False],
        },
        sparse_out,
    )
    sparse_props = sparse_info.get_perf_properties()
    assert sparse_props.compute_ops[torch.float16].mma_ops > 0
    assert sparse_props.memory_readwrite_bytes > 0


def test_minimax_indexer_uses_materialized_lengths_for_meta_tensors():
    batch_size = 8
    context_length = 4334
    idx_q = torch.empty(batch_size, 1, 128, dtype=torch.float16, device="meta")
    idx_k = torch.empty_like(idx_q)
    seq_lens = torch.empty(batch_size, dtype=torch.long, device="meta")
    query_lens = torch.empty(batch_size, dtype=torch.long, device="meta")

    breakdown = _estimate_minimax_indexer_breakdown(
        idx_q,
        idx_k,
        seq_lens,
        query_lens,
        topk_blocks=16,
        block_size=128,
        seq_lens_values=[context_length] * batch_size,
        query_lens_values=[1] * batch_size,
        is_decode_values=[True] * batch_size,
    )

    num_blocks = 34
    expected_mma = 2 * batch_size * context_length * 128
    expected_gp = batch_size * context_length + 4 * batch_size * num_blocks
    expected_bytes = (
        idx_q.numel() * idx_q.element_size()
        + batch_size * context_length * 128 * idx_q.element_size()
        + 8 * batch_size * num_blocks
        + 4 * batch_size * 16
    )
    assert breakdown == {
        "mma_total": expected_mma,
        "gp_total": expected_gp,
        "bytes_total": expected_bytes,
    }


def test_minimax_indexer_prefill_reads_visible_keys_once():
    query_length = 65
    context_length = 128
    idx_q = torch.empty(query_length, 1, 128, dtype=torch.float16, device="meta")
    idx_k = torch.empty_like(idx_q)
    seq_lens = torch.empty(1, dtype=torch.long, device="meta")
    query_lens = torch.empty(1, dtype=torch.long, device="meta")

    breakdown = _estimate_minimax_indexer_breakdown(
        idx_q,
        idx_k,
        seq_lens,
        query_lens,
        topk_blocks=16,
        block_size=128,
        seq_lens_values=[context_length],
        query_lens_values=[query_length],
        is_decode_values=[False],
    )

    expected_k_cache_bytes = context_length * 128 * idx_q.element_size()
    fixed_bytes = idx_q.numel() * idx_q.element_size() + 4 * query_length + 4 * query_length + 4 * query_length * 16
    assert breakdown["bytes_total"] == expected_k_cache_bytes + fixed_bytes


def test_minimax_indexer_prefill_scores_all_visible_blocks():
    query_length = 16384
    context_length = 115712
    seq_len = context_length + query_length
    block_size = 128
    idx_q = torch.empty(query_length, 1, 128, dtype=torch.float16, device="meta")
    idx_k = torch.empty_like(idx_q)
    seq_lens = torch.empty(1, dtype=torch.long, device="meta")
    query_lens = torch.empty(1, dtype=torch.long, device="meta")

    breakdown = _estimate_minimax_indexer_breakdown(
        idx_q,
        idx_k,
        seq_lens,
        query_lens,
        topk_blocks=16,
        block_size=block_size,
        seq_lens_values=[seq_len],
        query_lens_values=[query_length],
        is_decode_values=[False],
    )

    expected_score_tokens = seq_len
    expected_score_blocks = (seq_len + block_size - 1) // block_size
    expected_mma = 2 * query_length * expected_score_tokens * 128
    expected_gp = query_length * expected_score_tokens + 4 * query_length * expected_score_blocks
    assert breakdown["mma_total"] == expected_mma
    assert breakdown["gp_total"] == expected_gp


def test_minimax_indexer_k_cache_read_is_shared_across_index_heads():
    query_length = 1
    context_length = 128
    num_index_heads = 2
    idx_q = torch.empty(query_length, num_index_heads, 128, dtype=torch.float16, device="meta")
    idx_k = torch.empty(query_length, 1, 128, dtype=torch.float16, device="meta")
    seq_lens = torch.empty(1, dtype=torch.long, device="meta")
    query_lens = torch.empty(1, dtype=torch.long, device="meta")

    breakdown = _estimate_minimax_indexer_breakdown(
        idx_q,
        idx_k,
        seq_lens,
        query_lens,
        topk_blocks=16,
        block_size=128,
        seq_lens_values=[context_length],
        query_lens_values=[query_length],
        is_decode_values=[True],
    )

    expected_k_cache_bytes = context_length * 128 * idx_q.element_size()
    fixed_bytes = (
        idx_q.numel() * idx_q.element_size()
        + 4 * query_length * num_index_heads
        + 4 * query_length * num_index_heads
        + 4 * query_length * num_index_heads * 16
    )
    assert breakdown["bytes_total"] == expected_k_cache_bytes + fixed_bytes


def test_minimax_indexer_perf_properties_use_input_dtype():
    idx_q = torch.empty(1, 1, 128, dtype=torch.bfloat16, device="meta")
    idx_k = torch.empty_like(idx_q)
    seq_lens = torch.empty(1, dtype=torch.long, device="meta")
    query_lens = torch.empty(1, dtype=torch.long, device="meta")
    block_table = torch.empty(1, 1, dtype=torch.long, device="meta")
    output = torch.empty(1, 1, 16, dtype=torch.int32, device="meta")

    info = OpInvokeInfo(
        torch.ops.tensor_cast.minimax_indexer.default,
        (idx_q, idx_k, seq_lens, query_lens, block_table),
        {
            "topk_blocks": 16,
            "block_size": 128,
            "seq_lens_values": [128],
            "query_lens_values": [1],
            "is_decode_values": [True],
        },
        output,
    )

    properties = info.get_perf_properties()

    assert torch.bfloat16 in properties.compute_ops
    assert torch.float32 not in properties.compute_ops


def test_minimax_sparse_attention_uses_materialized_context_length():
    batch_size = 8
    context_length = 4334
    query = torch.empty(batch_size, 512, dtype=torch.float16, device="meta")
    seq_lens = torch.empty(batch_size, dtype=torch.long, device="meta")
    query_lens = torch.empty(batch_size, dtype=torch.long, device="meta")

    breakdown = _estimate_minimax_sparse_attention_breakdown(
        query,
        seq_lens,
        query_lens,
        num_q_heads=4,
        num_kv_heads=1,
        head_dim=128,
        topk_blocks=16,
        block_size=128,
        local_blocks=1,
        seq_lens_values=[context_length] * batch_size,
        query_lens_values=[1] * batch_size,
        is_decode_values=[True] * batch_size,
    )

    attended_tokens = 15 * 128 + context_length % 128
    expected_mma = 4 * batch_size * 4 * attended_tokens * 128
    expected_gp = 6 * batch_size * 4 * attended_tokens
    expected_kv_read_tokens = min(16 * 128, context_length)
    expected_bytes = (
        2 * query.element_size() * batch_size * 4 * 128
        + 2 * query.element_size() * batch_size * expected_kv_read_tokens * 128
        + 4 * batch_size * 16
    )
    assert breakdown == {
        "mma_total": expected_mma,
        "gp_total": expected_gp,
        "bytes_total": expected_bytes,
    }


def test_minimax_sparse_attention_prefill_reads_selected_kv_per_query():
    query_length = 16384
    context_length = 115712
    seq_len = context_length + query_length
    query = torch.empty(query_length, 1024, dtype=torch.float16, device="meta")
    seq_lens = torch.empty(1, dtype=torch.long, device="meta")
    query_lens = torch.empty(1, dtype=torch.long, device="meta")

    breakdown = _estimate_minimax_sparse_attention_breakdown(
        query,
        seq_lens,
        query_lens,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim=128,
        topk_blocks=16,
        block_size=128,
        local_blocks=1,
        seq_lens_values=[seq_len],
        query_lens_values=[query_length],
        is_decode_values=[False],
    )

    attended_pairs = 0
    for query_idx in range(query_length):
        visible_tokens = context_length + query_idx + 1
        current_block_tokens = (visible_tokens - 1) % 128 + 1
        attended_pairs += 15 * 128 + current_block_tokens
    effective_attended_pairs = _estimate_minimax_m3_prefill_sparse_attention_kv_read_pairs(
        context_length,
        query_length,
        16,
        128,
        attended_pairs,
    )
    expected_kv_bytes = 2 * query.element_size() * effective_attended_pairs * 128
    fixed_bytes = 2 * query.element_size() * query_length * 8 * 128 + 4 * query_length * 1 * 16

    assert breakdown["bytes_total"] == fixed_bytes + expected_kv_bytes


def test_minimax_sparse_attention_prefill_kv_pairs_use_selected_capacity_geometric_mean():
    query_length = 256
    context_length = 0
    block_size = 128
    topk_blocks = 16

    attended_pairs = 0
    for query_idx in range(query_length):
        visible_tokens = context_length + query_idx + 1
        visible_blocks = math.ceil(visible_tokens / block_size)
        selected_blocks = min(visible_blocks, topk_blocks)
        current_block_tokens = (visible_tokens - 1) % block_size + 1
        attended_pairs += (selected_blocks - 1) * block_size + current_block_tokens

    selected_kv_capacity = topk_blocks * block_size
    assert _estimate_minimax_m3_prefill_sparse_attention_kv_read_pairs(
        context_length,
        query_length,
        topk_blocks,
        block_size,
        attended_pairs,
    ) == max(selected_kv_capacity, math.ceil(math.sqrt(selected_kv_capacity * attended_pairs)))


def test_m3_swiglu_bf16_has_gp_roofline_properties():
    gate = torch.empty(2, 8, dtype=torch.bfloat16, device="meta")
    up = torch.empty_like(gate)
    out = torch.empty_like(gate)
    info = OpInvokeInfo(
        torch.ops.tensor_cast.m3_swiglu.default,
        (gate, up, 1.702, 7.0),
        {},
        out,
    )

    properties = info.get_perf_properties()

    assert properties.compute_ops[torch.bfloat16].gp_ops == gate.numel() * 7

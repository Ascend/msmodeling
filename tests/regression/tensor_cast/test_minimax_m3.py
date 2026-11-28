from types import SimpleNamespace

import pytest
import torch
from torch._inductor.compile_fx import fake_tensor_prop
from torch.utils._python_dispatch import TorchDispatchMode

from tensor_cast import ops  # noqa: F401
from tensor_cast.compilation.freezing_passes.grouped_matmul_swiglu_pass import GroupedMatmulSwigluPass
from tensor_cast.compilation.freezing_passes.sink_split_pass import SinkSplitPass
from tensor_cast.compilation.passes.merge_linear_pass import MergeLinearPass
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.minimax_m3_attention import MiniMaxM3AttentionWrapper
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import QuantConfig
from tensor_cast.performance_model import _estimate_minimax_indexer_breakdown
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.performance_model.utils import bytes_of_tensor
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity, QuantScheme, quantize_linear_modules
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.builtin_model.minimax_m3 import (
    MiniMaxM3DenseMLPWrapper,
    MiniMaxM3FusedMoETensorCast,
    MiniMaxM3MoeExpertMLP,
)

from .test_common import get_linear_quant_config


class _FakeDenseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.swiglu_alpha = 1.702
        self.swiglu_limit = 7.0
        self.gate_up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(4, 4, bias=False)


class _FakeExpertsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 4
        self.intermediate_dim = 4
        self.swiglu_alpha = 1.702
        self.swiglu_limit = 7.0
        self.gate_up_proj = torch.nn.Parameter(torch.randn(1, 8, 4))
        self.down_proj = torch.nn.Parameter(torch.randn(1, 4, 4))


class _FakeEpGroup:
    def __init__(self, world_size):
        self.world_size = world_size
        self.rank_in_group = 0
        self.all_reduce_calls = 0

    def all_reduce(self, input_):
        self.all_reduce_calls += 1
        return input_


class _FakeExperts:
    def call_expert(self, expert_idx, x, *args, **kwargs):
        return x


def _wrap_expert_fp8(expert: MiniMaxM3MoeExpertMLP) -> MiniMaxM3MoeExpertMLP:
    quant_config = QuantConfig()
    quant_config.linear_configs["*"] = get_linear_quant_config(
        LinearQuantType.FP8,
        dynamic_quant_granularity=QuantGranularity.PER_SAMPLE,
        dynamic_quant_scheme=QuantScheme.SYMMETRIC,
    )
    quantize_linear_modules(
        expert,
        TensorCastQuantLinear,
        quant_config,
        default_config_name="default",
        strip_module_fn=None,
    )
    return expert


def test_minimax_indexer_op_registered():
    assert hasattr(torch.ops.tensor_cast, "minimax_indexer"), "minimax_indexer op not registered"


def test_minimax_sparse_attention_op_registered():
    assert hasattr(torch.ops.tensor_cast, "minimax_sparse_attention"), "minimax_sparse_attention op not registered"


def test_siso_reshape_and_cache_op_registered():
    assert hasattr(torch.ops.tensor_cast, "siso_reshape_and_cache"), "siso_reshape_and_cache op not registered"


def test_minimax_indexer_meta_shape():
    idx_q = torch.empty(1, 4, 128, device="meta")
    idx_k = torch.empty(1, 1, 128, device="meta")
    seq_lens = torch.tensor([4096], device="meta")
    query_lens = torch.tensor([1], device="meta")
    block_table = torch.empty(1, 32, device="meta", dtype=torch.long)

    topk_idx = torch.ops.tensor_cast.minimax_indexer(
        idx_q,
        idx_k,
        seq_lens,
        query_lens,
        block_table,
        topk_blocks=16,
        block_size=128,
    )

    assert topk_idx.shape == (1, 4, 16)
    assert topk_idx.dtype == torch.int32


def test_minimax_indexer_k_cache_read_uses_block_size_q():
    """K-cache read term must be divided by BLOCK_SIZE_Q=64, not per-query-token.

    Regression: old formula counted `Q_b * N * L_b * D * s` (read K once per query
    token), which overestimated vLLM's `_index_block_score_kernel` by ~64x. The
    kernel tiles queries into BLOCK_SIZE_Q=64 and reuses each K-block across the
    tile via tl.dot(q, k).

    Note: index K cache write (2*TDs) is modeled by the standalone
    siso_reshape_and_cache op and is NOT part of this breakdown.
    """
    # Q_b=64, L_b=128, N=1, D=128, K=16, B_s=128, bf16 (s=2)
    # Naive (old): 64*1*128*128*2 = 2_097_152 bytes for K read alone
    # vLLM kernel: K read once per B_q=64 query tile -> 2_097_152 / 64 = 32_768 bytes
    idx_q = torch.empty(64, 1, 128, device="meta", dtype=torch.bfloat16)
    idx_k = torch.empty(64, 1, 128, device="meta", dtype=torch.bfloat16)
    seq_lens = torch.tensor([128])
    query_lens = torch.tensor([64])

    bd = _estimate_minimax_indexer_breakdown(
        idx_q,
        idx_k,
        seq_lens,
        query_lens,
        topk_blocks=16,
        block_size=128,
    )
    # If the formula is wrong (no /64), bytes_total would be > 2 MB.
    # Correct formula gives ~32 KB for K read plus a few KB for score writes.
    assert bd["bytes_total"] < 100_000, f"K-cache read not divided by BLOCK_SIZE_Q=64; bytes_total={bd['bytes_total']}"


def _total_memory_bytes(properties: OpInvokeInfo.PerformanceProperties) -> int:
    return properties.memory_read_bytes + properties.memory_write_bytes + properties.memory_readwrite_bytes


def test_minimax_indexer_no_double_count_input_tensors():
    """Registered minimax_indexer must not auto-count idx_q/idx_k on top of breakdown."""
    t, num_heads, head_dim, topk_blocks, block_size = 64, 1, 128, 16, 128
    idx_q = torch.empty(t, num_heads, head_dim, device="meta", dtype=torch.bfloat16)
    idx_k = torch.empty(t, 1, head_dim, device="meta", dtype=torch.bfloat16)
    seq_lens = torch.tensor([128])
    query_lens = torch.tensor([64])
    block_table = torch.empty(1, 32, device="meta", dtype=torch.long)
    topk_out = torch.empty(t, num_heads, topk_blocks, device="meta", dtype=torch.int32)

    op = OpInvokeInfo(
        torch.ops.tensor_cast.minimax_indexer.default,
        (idx_q, idx_k, seq_lens, query_lens, block_table),
        {"topk_blocks": topk_blocks, "block_size": block_size},
        topk_out,
    )
    breakdown = _estimate_minimax_indexer_breakdown(
        idx_q,
        idx_k,
        seq_lens,
        query_lens,
        topk_blocks=topk_blocks,
        block_size=block_size,
    )
    auto_excluded = op.get_memory_access_properties(exclude_input_ids={0, 1}, exclude_output_ids={0})
    auto_included = op.get_memory_access_properties()
    props = op.get_perf_properties()

    input_tensor_bytes = bytes_of_tensor(idx_q) + bytes_of_tensor(idx_k)
    assert _total_memory_bytes(auto_included) >= _total_memory_bytes(auto_excluded) + input_tensor_bytes
    assert _total_memory_bytes(props) == _total_memory_bytes(auto_excluded) + breakdown["bytes_total"]


def test_minimax_indexer_excludes_cache_write():
    """minimax_indexer breakdown must NOT depend on idx_k element size (cache write).

    Cache write is now modeled by the standalone siso_reshape_and_cache op,
    so changing idx_k dtype (which only affects cache write bytes) must leave
    the indexer breakdown unchanged.
    """
    t, num_heads, head_dim, topk_blocks, block_size = 64, 1, 128, 16, 128
    seq_lens = torch.tensor([128])
    query_lens = torch.tensor([64])

    idx_q = torch.empty(t, num_heads, head_dim, device="meta", dtype=torch.bfloat16)
    idx_k_bf16 = torch.empty(t, 1, head_dim, device="meta", dtype=torch.bfloat16)
    idx_k_fp32 = torch.empty(t, 1, head_dim, device="meta", dtype=torch.float32)

    bd_bf16 = _estimate_minimax_indexer_breakdown(
        idx_q, idx_k_bf16, seq_lens, query_lens, topk_blocks=topk_blocks, block_size=block_size
    )
    bd_fp32 = _estimate_minimax_indexer_breakdown(
        idx_q, idx_k_fp32, seq_lens, query_lens, topk_blocks=topk_blocks, block_size=block_size
    )
    assert bd_bf16["bytes_total"] == bd_fp32["bytes_total"], (
        "minimax_indexer bytes_total depends on idx_k dtype; cache write not fully removed"
    )


def test_minimax_indexer_breakdown_uses_materialized_lengths_for_meta_tensors():
    """Materialized seq/query lens values must override meta tensors in the breakdown."""
    idx_q = torch.empty(64, 1, 128, device="meta", dtype=torch.bfloat16)
    idx_k = torch.empty(64, 1, 128, device="meta", dtype=torch.bfloat16)

    meta_bd = _estimate_minimax_indexer_breakdown(
        idx_q,
        idx_k,
        torch.tensor([128], device="meta"),
        torch.tensor([64], device="meta"),
        topk_blocks=16,
        block_size=128,
        seq_lens_values=[128],
        query_lens_values=[64],
        is_decode_values=[False],
    )
    real_bd = _estimate_minimax_indexer_breakdown(
        idx_q,
        idx_k,
        torch.tensor([128]),
        torch.tensor([64]),
        topk_blocks=16,
        block_size=128,
    )
    assert meta_bd == real_bd

    with pytest.raises(ValueError, match="materialized attention lengths"):
        _estimate_minimax_indexer_breakdown(
            idx_q,
            idx_k,
            torch.tensor([128], device="meta"),
            torch.tensor([64], device="meta"),
            topk_blocks=16,
            block_size=128,
            seq_lens_values=[128, 256],
            query_lens_values=[64],
        )


class _DummyIndexer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 4, bias=False)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()


class _DummyM3Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.v_proj = torch.nn.Linear(8, 4, bias=False)
        self.o_proj = torch.nn.Linear(8, 8, bias=False)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.indexer = _DummyIndexer()


class _IndexerCaptureMode(TorchDispatchMode):
    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.tensor_cast.minimax_indexer.default:
            self.calls.append((args, kwargs))
        return func(*args, **kwargs)


def _make_m3_attention_meta(*, seq_len: int, query_len: int, total_tokens: int):
    return SimpleNamespace(
        seq_lens=torch.tensor([seq_len], dtype=torch.long),
        query_lens=torch.tensor([query_len], dtype=torch.long),
        block_table_tensor=torch.zeros(1, 2, dtype=torch.long),
        slot_mapping=torch.arange(total_tokens, dtype=torch.long),
        seq_lens_values=[seq_len],
        query_lens_values=[query_len],
        is_decode_values=[query_len == 1],
    )


def _make_sparse_wrapper() -> MiniMaxM3AttentionWrapper:
    return MiniMaxM3AttentionWrapper(
        original_module=_DummyM3Attention().to("meta"),
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
    ).to("meta")


def test_minimax_m3_attention_wrapper_passes_indexer_cache_and_lengths():
    """Sparse wrapper must feed indexer_cache (not current-step idx_k) and per-request lengths."""
    hidden_states = torch.empty(2, 8, device="meta")
    attention_meta = _make_m3_attention_meta(seq_len=4, query_len=2, total_tokens=2)
    indexer_cache = torch.empty(4, 1, 4, device="meta", dtype=torch.bfloat16)
    position_embeddings = (torch.empty(2, 4, device="meta"), torch.empty(2, 4, device="meta"))

    with _IndexerCaptureMode() as capture:
        _make_sparse_wrapper()(
            hidden_states,
            attention_meta=attention_meta,
            position_embeddings=position_embeddings,
            indexer_cache_by_layers={0: indexer_cache},
        )

    assert len(capture.calls) == 1
    args, kwargs = capture.calls[0]
    assert args[1] is indexer_cache, "minimax_indexer must receive the historical indexer_cache tensor"
    assert kwargs["seq_lens_values"] == [4]
    assert kwargs["query_lens_values"] == [2]
    assert kwargs["is_decode_values"] == [False]


def test_minimax_m3_attention_wrapper_falls_back_to_current_step_keys_without_cache():
    """Without indexer_cache_by_layers, minimax_indexer receives the current-step idx_k_flat."""
    hidden_states = torch.empty(2, 8, device="meta")
    attention_meta = _make_m3_attention_meta(seq_len=4, query_len=2, total_tokens=2)
    position_embeddings = (torch.empty(2, 4, device="meta"), torch.empty(2, 4, device="meta"))

    with _IndexerCaptureMode() as capture:
        _make_sparse_wrapper()(
            hidden_states,
            attention_meta=attention_meta,
            position_embeddings=position_embeddings,
        )

    assert len(capture.calls) == 1
    args, _ = capture.calls[0]
    assert args[1].shape == (2, 1, 4), "fallback index keys must be the current-step idx_k_flat"


def test_siso_reshape_and_cache_properties():
    """siso_reshape_and_cache must model 2*TDs traffic (read key + write cache)."""
    t, head_dim = 64, 128
    idx_k = torch.empty(t, 1, head_dim, device="meta", dtype=torch.bfloat16)
    indexer_cache = torch.empty(4, 128, head_dim, device="meta", dtype=torch.bfloat16)
    slot_mapping = torch.empty(t, device="meta", dtype=torch.long)

    op = OpInvokeInfo(
        torch.ops.tensor_cast.siso_reshape_and_cache.default,
        (idx_k, indexer_cache, slot_mapping),
        {},
        None,
    )
    props = op.get_perf_properties()

    # The cache write models: read key (TDs) + write to cache (TDs).
    # key read is auto-counted (memory_read_bytes), cache write is attributed
    # explicitly (memory_write_bytes). slot_mapping read is incidental auto-traffic.
    idx_k_bytes = bytes_of_tensor(idx_k, dtype=indexer_cache.dtype)
    assert props.memory_write_bytes == idx_k_bytes, (
        f"write bytes mismatch: got {props.memory_write_bytes}, expected {idx_k_bytes}"
    )
    assert props.memory_read_bytes >= idx_k_bytes, "key read not accounted"
    assert not props.compute_ops, "cache write must be memory-only"


def test_siso_reshape_and_cache_meta():
    """siso_reshape_and_cache op is registered and returns None."""
    idx_k = torch.empty(1, 1, 128, device="meta")
    indexer_cache = torch.empty(4, 128, 128, device="meta")
    slot_mapping = torch.empty(1, device="meta", dtype=torch.long)

    result = torch.ops.tensor_cast.siso_reshape_and_cache(idx_k, indexer_cache, slot_mapping)
    assert result is None


def test_minimax_sparse_attention_meta_shape():
    query = torch.empty(1, 64, 128, device="meta")
    key_cache = torch.empty(4096, 4, 128, device="meta")
    value_cache = torch.empty(4096, 4, 128, device="meta")
    topk_idx = torch.empty(1, 4, 16, dtype=torch.int32, device="meta")
    seq_lens = torch.tensor([4096], device="meta")
    query_lens = torch.tensor([1], device="meta")
    block_table = torch.empty(1, 32, device="meta", dtype=torch.long)

    out = torch.ops.tensor_cast.minimax_sparse_attention(
        query,
        key_cache,
        value_cache,
        topk_idx,
        seq_lens,
        query_lens,
        block_table,
        hidden_size=6144,
        num_q_heads=64,
        num_kv_heads=4,
        head_dim=128,
        topk_blocks=16,
        block_size=128,
        local_blocks=1,
    )

    assert out.shape == query.shape


def test_minimax_indexer_properties():
    assert hasattr(
        OpInvokeInfo._op_properties_functors,
        "__getitem__",
    ) or hasattr(OpInvokeInfo, "get_perf_properties"), "OpInvokeInfo missing properties registry"


def test_minimax_m3_fused_moe_does_not_ep_all_reduce_like_deepseek():
    fused_moe = MiniMaxM3FusedMoETensorCast.__new__(MiniMaxM3FusedMoETensorCast)
    torch.nn.Module.__init__(fused_moe)
    fused_moe.top_k = 1
    fused_moe.routed_scaling_factor = 1.0
    fused_moe.shared_experts = torch.nn.Identity()
    fused_moe.num_external_shared_experts = 0
    fused_moe._global_tp_size = 1
    fused_moe.ep_group = _FakeEpGroup(world_size=2)
    fused_moe.experts = _FakeExperts()

    hidden_states = torch.ones(2, 4)
    topk_indices = torch.zeros(2, 1, dtype=torch.long)
    topk_weights = torch.ones(2, 1)

    fused_moe.get_split_sizes = lambda num_tokens, top_k: ([], [], [], [])
    fused_moe.dispatch_tokens = lambda hidden, indices, *_: [hidden]
    fused_moe.combine_tokens = lambda routed, indices, *_: routed[0].unsqueeze(-2)
    fused_moe._run_shared_experts = lambda hidden: torch.full_like(hidden, 2.0)

    output = fused_moe(hidden_states, topk_indices, topk_weights)

    assert fused_moe.ep_group.all_reduce_calls == 0
    assert torch.equal(output, torch.full_like(hidden_states, 3.0))


def test_minimax_m3_dense_mlp_wrapper_uses_m3_swiglu_quant():
    wrapper = MiniMaxM3DenseMLPWrapper(_FakeDenseMLP()).to("meta")
    quantize_linear_modules(
        wrapper,
        TensorCastQuantLinear,
        QuantConfig(linear_configs={"*": get_linear_quant_config(LinearQuantType.FP8)}),
        default_config_name="default",
        strip_module_fn=None,
    )
    hidden_states = torch.empty(2, 4, device="meta")

    perf_model = AnalyticPerformanceModel(TEST_DEVICE)
    with Runtime(perf_model, TEST_DEVICE) as runtime, torch.no_grad():
        output = wrapper(hidden_states)

    result = runtime.table_averages()
    assert output.shape == hidden_states.shape
    assert "tensor_cast.m3_swiglu_quant.default" in result
    assert "aten.sigmoid.default" not in result


def test_minimax_m3_moe_expert_uses_fp8_linear_for_gate_up_and_down():
    expert = _wrap_expert_fp8(MiniMaxM3MoeExpertMLP(_FakeExpertsModule(), 0)).to("meta")
    hidden_states = torch.empty(2, 4, device="meta")

    perf_model = AnalyticPerformanceModel(TEST_DEVICE)
    with Runtime(perf_model, TEST_DEVICE) as runtime, torch.no_grad():
        output = expert(hidden_states)

    result = runtime.table_averages()
    assert output.shape == hidden_states.shape
    fp8_line = next(line for line in result.splitlines() if "tensor_cast.fp8_linear.default" in line)
    quant_line = next(line for line in result.splitlines() if "tensor_cast.dynamic_quantize_symmetric.default" in line)
    assert fp8_line.split()[-1] == "3"
    assert quant_line.split()[-1] == "2"
    assert "tensor_cast.m3_swiglu_quant.default" in result


def _run_m3_moe_fp8_freezing_passes(gm: torch.fx.GraphModule, inputs):
    fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
    MergeLinearPass()(gm)
    fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
    for _ in range(3):
        SinkSplitPass()(gm)
        fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
    GroupedMatmulSwigluPass()(gm)
    for _ in range(3):
        SinkSplitPass()(gm)


def test_minimax_m3_moe_fp8_freezing_passes_fuse_gmm_swiglu_and_down():
    """M3 FP8 expert: gate/up/down GMM fusion; m3_swiglu_quant stays separate (tuple output)."""
    dq = torch.ops.tensor_cast.dynamic_quantize_symmetric.default
    fp8 = torch.ops.tensor_cast.fp8_linear.default
    gmm = torch.ops.tensor_cast.grouped_matmul_fp8.default
    gmm_fp8_swiglu = torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default
    m3sq = torch.ops.tensor_cast.m3_swiglu_quant.default
    split = torch.ops.aten.split_with_sizes.default

    def expert_fwd(x, wg, wu, wd, wsg, wsu, wsd):
        qx, ascale = dq(x, dims=[-1], scale_dtype=torch.float32, out_dtype=torch.int8)
        gate = fp8(qx, wg, ascale, wsg, None, torch.bfloat16)
        up = fp8(qx, wu, ascale, wsu, None, torch.bfloat16)
        mid, mscale = m3sq(gate, up, 1.702, 7.0, 128)
        return fp8(mid, wd, mscale, wsd, None, torch.bfloat16)

    class _TwoExpertMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i in (0, 1):
                for name, shape in (
                    ("wg", (4, 4)),
                    ("wu", (4, 4)),
                    ("wd", (4, 4)),
                    ("wsg", (4,)),
                    ("wsu", (4,)),
                    ("wsd", (4,)),
                ):
                    param = torch.randn(shape) if len(shape) > 1 else torch.ones(shape)
                    setattr(self, f"{name}{i}", torch.nn.Parameter(param))

        def forward(self, x):
            cat = torch.cat([x[:3], x[3:]], dim=0)
            x0, x1 = split(cat, [3, 2], 0)
            o0 = expert_fwd(x0, self.wg0, self.wu0, self.wd0, self.wsg0, self.wsu0, self.wsd0)
            o1 = expert_fwd(x1, self.wg1, self.wu1, self.wd1, self.wsg1, self.wsu1, self.wsd1)
            return torch.cat([o0, o1], dim=0)

    inputs = [torch.empty(5, 4)]
    gm = torch.fx.symbolic_trace(_TwoExpertMoE())
    _run_m3_moe_fp8_freezing_passes(gm, inputs)

    def _count(target):
        return sum(1 for node in gm.graph.nodes if node.target == target)

    assert _count(fp8) == 0
    assert _count(m3sq) == 1
    assert _count(gmm_fp8_swiglu) == 0
    assert _count(gmm) == 2

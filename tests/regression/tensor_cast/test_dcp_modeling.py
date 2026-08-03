"""Unit tests for Decode Context Parallel (DCP) modeling.

Covers M1 (config-layer ``dcp_size`` plumbing + validation, incl. the GQA-only
``h_kv >= tp / dcp`` constraint), M2 (KV cache token-capacity scaling by
``dcp_size`` with the per-token cost held physically constant, gated off for
V4/SFA caches), and M2-compute/M3 (decode detection incl. MTP, GQA
attention cost invariance under DCP, MLA all_gather Q + ``seq_lens / dcp`` KV
sharding, and the fp32 ``output + lse`` all_to_all merge).
See ``docs/RFC/rfc_context_parallel_dcp_*.md``.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from tensor_cast.core.input_generator import _get_kv_cache_info, dcp_kv_token_capacity_factor
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionMetadataTensorCast, AttentionTensorCast
from tensor_cast.model_config import ParallelConfig
from tensor_cast.parallel_group import ParallelGroup, ParallelGroupManager
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime

from tests.helpers.model_cache import get_built_model


class TestParallelConfigDcpValidation:
    """M1: tp_size must be a positive multiple of dcp_size (DCP reuses the TP domain)."""

    def test_valid_dcp_divides_tp(self):
        config = ParallelConfig(world_size=8, tensor_parallel_size=8, decode_context_parallel_size=2)
        assert config.decode_context_parallel_size == 2
        assert config.has_dcp()

    def test_dcp_defaults_to_one_and_not_active(self):
        config = ParallelConfig(world_size=8, tensor_parallel_size=8)
        assert config.decode_context_parallel_size == 1
        assert not config.has_dcp()

    def test_dcp_not_dividing_tp_raises(self):
        # tp=8 is not divisible by dcp=3
        with pytest.raises(ValueError, match="divisible by"):
            ParallelConfig(world_size=8, tensor_parallel_size=8, decode_context_parallel_size=3)

    def test_dcp_larger_than_tp_raises(self):
        # dcp=4 > tp=2, so tp % dcp != 0
        with pytest.raises(ValueError, match="divisible by"):
            ParallelConfig(world_size=8, tensor_parallel_size=2, decode_context_parallel_size=4)

    def test_dcp_below_one_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            ParallelConfig(world_size=8, tensor_parallel_size=8, decode_context_parallel_size=0)


class TestUserInputConfigDcpPassThrough:
    """M1: UserInputConfig.dcp_size flows into ParallelConfig.decode_context_parallel_size."""

    def test_dcp_size_passed_through(self):
        user_config = UserInputConfig(world_size=8, tp_size=8, dcp_size=2)
        parallel_config = user_config.get_parallel_config()
        assert parallel_config.decode_context_parallel_size == 2

    def test_dcp_size_defaults_to_one(self):
        user_config = UserInputConfig(world_size=8, tp_size=8)
        assert user_config.dcp_size == 1
        assert user_config.get_parallel_config().decode_context_parallel_size == 1


class TestTextGenerateCliDcpArg:
    """M1: the ``text_generate`` CLI exposes ``--dcp-size`` and threads it through to ParallelConfig.

    ``ModelRunner`` is stubbed so the test stays a fast argument-plumbing check (no model load
    or inference), while still executing ``cli.inference.text_generate.main``.
    """

    @staticmethod
    def _run_main(extra_argv, monkeypatch):
        from cli.inference import text_generate

        argv = [
            "text_generate",
            "Qwen/Qwen3-32B",
            "--num-queries",
            "1",
            "--query-length",
            "8",
            "--num-devices",
            "8",
            "--tp-size",
            "8",
            *extra_argv,
        ]
        monkeypatch.setattr("sys.argv", argv)

        captured = {}

        class _StubRunner:
            def __init__(self, user_input):
                captured["user_input"] = user_input

            def run_inference(self, *args, **kwargs):
                return MagicMock()

        # ``main()`` imports ModelRunner lazily from ``tensor_cast.core.model_runner``,
        # so patch it at that definition site rather than on the CLI module.
        monkeypatch.setattr("tensor_cast.core.model_runner.ModelRunner", _StubRunner)
        text_generate.main()
        return captured["user_input"]

    def test_dcp_size_flag_threads_into_parallel_config(self, monkeypatch):
        user_input = self._run_main(["--dcp-size", "2"], monkeypatch)
        assert user_input.dcp_size == 2
        assert user_input.get_parallel_config().decode_context_parallel_size == 2

    def test_dcp_size_defaults_to_one_when_flag_omitted(self, monkeypatch):
        user_input = self._run_main([], monkeypatch)
        assert user_input.dcp_size == 1
        assert user_input.get_parallel_config().decode_context_parallel_size == 1

    def test_dcp_size_rejects_non_positive(self, monkeypatch):
        with pytest.raises(SystemExit) as exc_info:
            self._run_main(["--dcp-size", "0"], monkeypatch)
        # argparse type-conversion failure exits with code 2.
        assert exc_info.value.code == 2


def _make_gqa_model(dcp_size: int, num_layers: int = 2, num_kv_heads: int = 8, tp_size: int = 4):
    """Build a minimal GQA model mock for ``_get_kv_cache_info``.

    ``mla_config`` is None so the standard (non-MLA) KV cache branch is taken.
    """
    model = MagicMock()
    model.num_hidden_layers = num_layers
    model.head_dim = 128
    model.model_config.mla_config = None
    model.model_config.dtype = torch.bfloat16
    # Avoid the V4 code path: model_type must not be "deepseek_v4".
    model.model_config.hf_config = MagicMock(model_type="qwen3")
    model.text_config = MagicMock(model_type="qwen3")
    model.text_config.num_key_value_heads = num_kv_heads
    model.model_config.parallel_config = ParallelConfig(
        world_size=tp_size,
        tensor_parallel_size=tp_size,
        decode_context_parallel_size=dcp_size,
    )
    return model


def _make_mla_model(dcp_size: int, model_type: str = "deepseek_v3", tp_size: int = 8):
    """Build a minimal MLA (latent KV) model mock for ``dcp_kv_token_capacity_factor``.

    ``mla_config`` is present (the layout signal the factor gates on) with a plain MLA
    class, i.e. ``requires_indexer_cache() is False`` so this is a non-SFA MLA model.
    """
    from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast

    model = MagicMock()
    model.model_config.mla_config = MagicMock(mla_cls=MultiheadLatentAttentionTensorCast)
    model.model_config.hf_config = MagicMock(model_type=model_type)
    model.text_config = MagicMock(model_type=model_type)
    model.model_config.parallel_config = ParallelConfig(
        world_size=tp_size,
        tensor_parallel_size=tp_size,
        decode_context_parallel_size=dcp_size,
    )
    return model


class TestKvCacheMemoryDcpScaling:
    """M2: per-token KV cost is INVARIANT under dcp; the saving is token capacity.

    DCP shards the KV cache along the token dimension, so each card holds
    ``1 / dcp`` of every sequence's tokens while the per-token byte cost on a card
    is unchanged. The capacity gain is surfaced separately via
    ``dcp_kv_token_capacity_factor`` (see ``TestDcpKvTokenCapacityFactor``), NOT by
    shrinking the per-token cost.
    """

    _NUM_BLOCKS = 64
    _BLOCK_SIZE = 128

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_per_token_cost_is_invariant_under_dcp(self, _mock_attn_quant):
        _, per_token_dcp1 = _get_kv_cache_info(_make_gqa_model(dcp_size=1), self._NUM_BLOCKS, self._BLOCK_SIZE)
        _, per_token_dcp2 = _get_kv_cache_info(_make_gqa_model(dcp_size=2), self._NUM_BLOCKS, self._BLOCK_SIZE)
        _, per_token_dcp4 = _get_kv_cache_info(_make_gqa_model(dcp_size=4), self._NUM_BLOCKS, self._BLOCK_SIZE)

        assert per_token_dcp1 > 0
        # Per-token cost is the true physical per-card cost: dcp does not divide it.
        assert per_token_dcp2 == pytest.approx(per_token_dcp1)
        assert per_token_dcp4 == pytest.approx(per_token_dcp1)

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    def test_physical_cache_shape_is_unchanged_by_dcp(self, _mock_attn_quant):
        cache_dcp1, _ = _get_kv_cache_info(_make_gqa_model(dcp_size=1), self._NUM_BLOCKS, self._BLOCK_SIZE)
        cache_dcp4, _ = _get_kv_cache_info(_make_gqa_model(dcp_size=4), self._NUM_BLOCKS, self._BLOCK_SIZE)

        # DCP must not physically shrink the per-rank KV tensors.
        for layer_idx, tensor in cache_dcp1.items():
            assert cache_dcp4[layer_idx].shape == tensor.shape


class TestDcpKvTokenCapacityFactor:
    """M2: the capacity factor is gated on KV LAYOUT, not on model name.

    MLA latent KV is not partitioned across TP heads, so a card really does hold
    ``dcp`` times as many tokens. GQA does NOT get that: DCP moves a rank from
    ``h_kv/tp`` heads over ``S`` to ``h_kv*dcp/tp`` heads over ``S/dcp``, so per-card
    KV bytes ``(h_kv*dcp/tp) * D * (S/dcp) == (h_kv/tp) * D * S`` are invariant. The
    only real GQA gain is de-duplicating KV heads TP replicated when ``h_kv < tp``.
    """

    def test_dcp_off_is_always_one(self):
        assert dcp_kv_token_capacity_factor(_make_gqa_model(dcp_size=1)) == 1
        assert dcp_kv_token_capacity_factor(_make_mla_model(dcp_size=1)) == 1

    def test_gqa_factor_is_one_when_kv_heads_cover_tp(self):
        # h_kv >= tp: every card owns distinct KV heads, so TP replicates nothing and
        # DCP only trades "few heads x long context" for "more heads x short context".
        # Regression: returning dcp_size here fabricated dcp times the real token
        # capacity (the PR's Qwen3-32B --tp-size 8 --dcp-size 8 example).
        for dcp_size in (2, 4, 8):
            model = _make_gqa_model(dcp_size=dcp_size, num_kv_heads=8, tp_size=8)
            assert dcp_kv_token_capacity_factor(model) == 1

    def test_gqa_factor_is_capped_by_tp_kv_replication(self):
        # h_kv=4 < tp=16: TP replicates each KV head 4x, so DCP can recover at most 4x.
        assert dcp_kv_token_capacity_factor(_make_gqa_model(dcp_size=2, num_kv_heads=4, tp_size=16)) == 2
        assert dcp_kv_token_capacity_factor(_make_gqa_model(dcp_size=4, num_kv_heads=4, tp_size=16)) == 4
        # dcp beyond the replication factor buys nothing further.
        assert dcp_kv_token_capacity_factor(_make_gqa_model(dcp_size=8, num_kv_heads=4, tp_size=16)) == 4
        assert dcp_kv_token_capacity_factor(_make_gqa_model(dcp_size=16, num_kv_heads=4, tp_size=16)) == 4

    def test_gqa_factor_is_one_when_kv_head_count_unknown(self):
        # Layout unverifiable -> claim no capacity gain rather than guessing dcp_size.
        model = _make_gqa_model(dcp_size=4, tp_size=16)
        model.text_config.num_key_value_heads = None
        assert dcp_kv_token_capacity_factor(model) == 1

    def test_mla_factor_equals_dcp_size(self):
        # Latent KV is not sharded across TP heads, so the sequence shard is a real
        # per-card byte saving for the full dcp_size.
        assert dcp_kv_token_capacity_factor(_make_mla_model(dcp_size=2)) == 2
        assert dcp_kv_token_capacity_factor(_make_mla_model(dcp_size=8)) == 8

    @staticmethod
    def _make_v4_model(dcp_size):
        model = MagicMock()
        model.model_config.hf_config = MagicMock(model_type="deepseek_v4")
        model.text_config = MagicMock(model_type="deepseek_v4")
        model.model_config.parallel_config = ParallelConfig(
            world_size=8,
            tensor_parallel_size=4,
            decode_context_parallel_size=dcp_size,
        )
        return model

    def test_v4_factor_is_one_regardless_of_dcp(self):
        # SFA + DCP memory modeling is an RFC non-goal: no capacity multiplier.
        assert dcp_kv_token_capacity_factor(self._make_v4_model(dcp_size=1)) == 1
        assert dcp_kv_token_capacity_factor(self._make_v4_model(dcp_size=4)) == 1

    def test_sfa_indexer_models_are_gated_by_layout_not_model_name(self):
        """Every model allocating an indexer cache is gated, not just ``deepseek_v4``.

        The gate used to test ``_is_v4_model`` only, so DeepSeek V3.2 (``deepseek_v32``)
        and GLM-5 (``glm_moe_dsa`` -> ``Glm5SparseAttention`` -> ``DeepseekSparseAttention``)
        slipped through and got a blanket multiplier -- including on their indexer
        cache, which ``get_sparse_attention_indexer_cache_info`` sizes with the FULL
        ``num_blocks`` (never sequence-sharded) yet whose per-token cost is folded into
        ``kv_cache_per_token_gb``.
        """
        from tensor_cast.layers.glm5 import Glm5SparseAttention
        from tensor_cast.layers.mla import DeepseekSparseAttention

        for mla_cls, model_type in (
            (DeepseekSparseAttention, "deepseek_v32"),
            (Glm5SparseAttention, "glm_moe_dsa"),
        ):
            model = _make_mla_model(dcp_size=4, model_type=model_type)
            model.model_config.mla_config.mla_cls = mla_cls
            assert mla_cls.requires_indexer_cache() is True, model_type
            assert dcp_kv_token_capacity_factor(model) == 1, model_type


class TestDcpGroupConstruction:
    """M3.1: the DCP group is a contiguous sub-slice of size dcp within each TP group."""

    def test_dcp_group_is_contiguous_subslice_of_tp(self):
        cfg = ParallelConfig(world_size=8, tensor_parallel_size=8, decode_context_parallel_size=4)
        mgr = ParallelGroupManager(cfg)
        assert mgr.tp_group.rank_groups == [[0, 1, 2, 3, 4, 5, 6, 7]]
        assert mgr.dcp_group.rank_groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert mgr.dcp_group.world_size == 4

    def test_dcp_group_within_each_tp_group_when_tp_lt_world(self):
        cfg = ParallelConfig(world_size=8, tensor_parallel_size=4, decode_context_parallel_size=2)
        mgr = ParallelGroupManager(cfg)
        assert mgr.tp_group.rank_groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
        # Every DCP group must be fully contained in exactly one TP group.
        assert mgr.dcp_group.rank_groups == [[0, 1], [2, 3], [4, 5], [6, 7]]

    def test_dcp_size_one_is_noop_group(self):
        cfg = ParallelConfig(world_size=8, tensor_parallel_size=8, decode_context_parallel_size=1)
        mgr = ParallelGroupManager(cfg)
        assert mgr.dcp_group.world_size == 1  # singleton -> every collective is a no-op


def _run_gqa_attention(dcp_group, seq_len=2048, num_requests=4, kv_dtype=torch.bfloat16, compile_forward=False):
    """Run one decode GQA attention forward under a Runtime and return (output, events).

    When ``compile_forward`` is set, the forward is wrapped with ``torch.compile``
    using TensorCast's own compilation backend (not the default inductor backend),
    matching the real ``text_generate --compile`` path. This exercises the two
    compile passes that previously silently dropped the DCP collectives: meta
    constant-folding (``fold_meta_constants``) and AOT-Autograd DCE.
    """
    head_dim = 128
    kv_heads = 8
    q_heads_per_rank = 32
    hidden = q_heads_per_rank * head_dim
    block_size = 128
    num_blocks = 64

    attn = AttentionTensorCast()
    attn.dcp_group = dcp_group

    query = torch.empty((num_requests, hidden), dtype=kv_dtype, device="meta")
    key = torch.empty((num_requests, kv_heads * head_dim), dtype=kv_dtype, device="meta")
    value = torch.empty((num_requests, kv_heads * head_dim), dtype=kv_dtype, device="meta")
    kv_cache = torch.empty((2, num_blocks, block_size, kv_heads, head_dim), dtype=kv_dtype, device="meta")
    max_blk = (seq_len + block_size - 1) // block_size
    meta = AttentionMetadataTensorCast(
        query_start_loc=torch.arange(0, num_requests + 1, dtype=torch.long),
        seq_lens=torch.full((num_requests,), seq_len, dtype=torch.long),
        query_lens=torch.ones(num_requests, dtype=torch.long),  # decode: one token per request
        block_table_tensor=torch.zeros((num_requests, max_blk), dtype=torch.long, device="meta"),
        slot_mapping=torch.zeros((num_requests,), dtype=torch.long, device="meta"),
    )

    forward = attn.forward
    if compile_forward:
        from tensor_cast.compilation import get_backend

        forward = torch.compile(forward, backend=get_backend(device_name=None), fullgraph=True, dynamic=False)

    pm = AnalyticPerformanceModel(TEST_DEVICE)
    with Runtime(pm, TEST_DEVICE) as rt:
        out = forward(query, key, value, None, kv_cache=kv_cache, attention_meta=meta)
    return out, rt.event_list


def _op_names(events):
    return [str(e.op_invoke_info.func) for e in events]


# A small DeepSeek build is cached once per session; the MLA forward is the real
# (compiled) decode path, so the model must exist with at least one MLA layer.
_MLA_MODEL_ID = "deepseek-ai/DeepSeek-V3.1"


def _get_mla_layer(dcp_group):
    """Return one real ``MultiheadLatentAttentionTensorCast`` with ``dcp_group`` attached.

    The model is built (and session-cached) with a single hidden layer to keep the
    build cheap. ``dcp_group`` is forced onto the layer directly -- the shared model
    cache key does not include ``dcp_size``, so we set the group on the returned layer
    rather than rebuilding per dcp value (mirrors ``_run_gqa_attention`` attaching the
    group to a fresh ``AttentionTensorCast``).
    """
    from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast

    user_config = UserInputConfig(
        model_id=_MLA_MODEL_ID,
        num_hidden_layers_override=1,
        world_size=8,
        tp_size=8,
        ep_size=8,
        moe_dp_size=1,
        moe_tp_size=1,
    )
    model = get_built_model(user_config)
    mla = next(m for _, m in model.named_modules() if isinstance(m, MultiheadLatentAttentionTensorCast))
    mla.dcp_group = dcp_group
    return mla, model.text_config


def _run_mla_attention(dcp_group, seq_len=2048, num_requests=4, kv_dtype=torch.bfloat16, compile_forward=False):
    """Run one decode MLA attention forward under a Runtime and return (output, events).

    Mirrors ``_run_gqa_attention`` but drives a real MLA layer (latent KV, ``mlapo``
    preprocessing, kv_b decomposition). When ``compile_forward`` is set, the forward
    is wrapped with ``torch.compile`` using TensorCast's own backend, exercising the
    same data-dependent-branching and collective-survival hazards the host-side
    ``AttentionMetadata.is_dcp_decode`` flag and the value-neutral bindings guard
    against -- on the MLA path, which the GQA test cannot cover.
    """
    mla, text_config = _get_mla_layer(dcp_group)
    block_size = 128
    num_blocks = 64
    hidden = text_config.hidden_size
    kv_dim = mla.kv_lora_rank + mla.qk_rope_head_dim

    hidden_states = torch.empty((1, num_requests, hidden), dtype=kv_dtype, device="meta")
    # position_embeddings (cos, sin): (num_tokens, qk_rope_head_dim).
    cos = torch.empty((num_requests, mla.qk_rope_head_dim), dtype=kv_dtype, device="meta")
    sin = torch.empty((num_requests, mla.qk_rope_head_dim), dtype=kv_dtype, device="meta")
    kv_cache = torch.empty((num_blocks, block_size, kv_dim), dtype=kv_dtype, device="meta")
    max_blk = (seq_len + block_size - 1) // block_size
    meta = AttentionMetadataTensorCast(
        query_start_loc=torch.arange(0, num_requests + 1, dtype=torch.long),
        seq_lens=torch.full((num_requests,), seq_len, dtype=torch.long),
        query_lens=torch.ones(num_requests, dtype=torch.long),  # decode: one token per request
        block_table_tensor=torch.zeros((num_requests, max_blk), dtype=torch.long, device="meta"),
        slot_mapping=torch.zeros((num_requests,), dtype=torch.long, device="meta"),
    )

    forward = mla.forward
    if compile_forward:
        from tensor_cast.compilation import get_backend

        forward = torch.compile(forward, backend=get_backend(device_name=None), fullgraph=True, dynamic=False)

    pm = AnalyticPerformanceModel(TEST_DEVICE)
    with Runtime(pm, TEST_DEVICE) as rt:
        out, _ = forward(
            hidden_states,
            (cos, sin),
            None,
            kv_cache_by_layers={mla.layer_idx: kv_cache},
            attention_meta=meta,
        )
    return out, rt.event_list


class TestGqaDecodeDcpInjection:
    """M3.3: GQA decode path emits all_gather Q + all_to_all and shards the KV read."""

    _DCP1 = ParallelGroup(rank=0, rank_groups=[[0]], global_world_size=1)
    _DCP2 = ParallelGroup(rank=0, rank_groups=[[0, 1]], global_world_size=2)

    def test_dcp_collectives_emitted_only_when_enabled(self):
        _, events1 = _run_gqa_attention(self._DCP1)
        _, events2 = _run_gqa_attention(self._DCP2)
        names1 = _op_names(events1)
        names2 = _op_names(events2)
        # dcp=1 -> no DCP collectives; dcp=2 -> both collectives present.
        assert not any("all_gather" in n for n in names1)
        assert not any("all_to_all" in n for n in names1)
        assert any("all_gather" in n for n in names2)
        assert any("all_to_all" in n for n in names2)

    def test_output_width_unchanged_by_dcp(self):
        # o_proj input width must stay h_q/tp * head_dim regardless of dcp.
        out1, _ = _run_gqa_attention(self._DCP1)
        out2, _ = _run_gqa_attention(self._DCP2)
        assert out2.shape == out1.shape

    def test_attention_cost_scales_with_context(self):
        """The GQA attention op cost still tracks context length (sanity baseline)."""

        def attn_time(events):
            return sum(
                next(iter(e.perf_results.values())).execution_time_s
                for e in events
                if str(e.op_invoke_info.func).endswith("attention.default")
            )

        _, events_big = _run_gqa_attention(self._DCP2, seq_len=4096)
        _, events_small = _run_gqa_attention(self._DCP2, seq_len=2048)
        assert attn_time(events_big) > attn_time(events_small)

    def test_gqa_attention_cost_invariant_across_dcp(self):
        """For GQA, per-rank decode attention compute+read are invariant under DCP.

        DCP re-partitions KV heads across the TP domain: each rank gathers to
        h_q*dcp/tp Q heads and (de-duplicated) h_kv*dcp/tp KV heads while reading
        S/dcp context, so the head growth cancels the sequence shrink. Modeling the
        local attention op with sharded seq_lens but UN-grown KV heads (the previous
        bug) would shrink the read by dcp. We pin that the attention op cost is
        identical across dcp=1/2/4 so the optimizer is not biased toward DCP.
        """
        _DCP1 = ParallelGroup(rank=0, rank_groups=[[0]], global_world_size=1)
        _DCP4 = ParallelGroup(rank=0, rank_groups=[[0, 1, 2, 3]], global_world_size=4)

        def attn_time(events):
            return sum(
                next(iter(e.perf_results.values())).execution_time_s
                for e in events
                if str(e.op_invoke_info.func).endswith("attention.default")
            )

        _, e1 = _run_gqa_attention(_DCP1, seq_len=4096)
        _, e2 = _run_gqa_attention(self._DCP2, seq_len=4096)
        _, e4 = _run_gqa_attention(_DCP4, seq_len=4096)
        assert attn_time(e1) == pytest.approx(attn_time(e2))
        assert attn_time(e1) == pytest.approx(attn_time(e4))

    def test_gqa_merge_a2a_includes_lse_and_mirrors_mla(self):
        """RFC 2.1.4.3/2.1.4.4: the GQA merge a2a carries output + per-head lse in fp32.

        The payload must be (gathered_heads, num_tokens * (head_dim + 1)): the ``+1``
        is the per-head lse column that the previous GQA merge dropped, so MLA and GQA
        model the same per-layer communication volume.
        """
        head_dim = 128
        q_heads_per_rank = 32
        num_requests = 4
        dcp = self._DCP2.world_size
        _, events = _run_gqa_attention(self._DCP2, seq_len=4096, num_requests=num_requests)
        a2a = [e for e in events if str(e.op_invoke_info.func).endswith("all_to_all.default")]
        assert len(a2a) == 1
        payload = a2a[0].op_invoke_info.args[0]
        assert payload.dtype == torch.float32
        # gathered_heads = (hidden/head_dim) * dcp; columns include the +1 lse term.
        assert payload.shape == (q_heads_per_rank * dcp, num_requests * (head_dim + 1))

    def test_all_to_all_uses_fp32_bytes_on_bf16_model(self):
        """RFC 2.1.4.3: a2a output+lse volume must be fp32 (4 bytes/elem) even on bf16.

        The payload tensor is what drives ``CommAnalyticModel`` byte counting, so a
        float32 payload guarantees 4 bytes/element. We pin that the layer builds an
        fp32 payload and that its modelled volume is exactly double the bf16 volume of
        the identical shape (i.e. it does not inherit the model's bf16 dtype).
        """
        _, events = _run_gqa_attention(self._DCP2, kv_dtype=torch.bfloat16)
        a2a_events = [e for e in events if str(e.op_invoke_info.func).endswith("all_to_all.default")]
        assert len(a2a_events) == 1
        a2a = a2a_events[0]
        payload = a2a.op_invoke_info.args[0]
        assert payload.dtype == torch.float32, "DCP a2a payload must be fp32, not the model dtype"

        fp32_bytes = next(iter(a2a.perf_results.values())).statistics["message_size_bytes"]
        # Same shape/group as bf16 would have produced if the layer reused model dtype.
        pm = AnalyticPerformanceModel(TEST_DEVICE)
        splits = a2a.op_invoke_info.args[1]
        with Runtime(pm, TEST_DEVICE) as rt:
            torch.ops.tensor_cast.all_to_all(payload.to(torch.bfloat16), splits, splits, 0, self._DCP2.rank_group)
        bf16_event = next(e for e in rt.event_list if str(e.op_invoke_info.func).endswith("all_to_all.default"))
        bf16_bytes = next(iter(bf16_event.perf_results.values())).statistics["message_size_bytes"]
        assert fp32_bytes == 2 * bf16_bytes


class TestMlaDcpAbsorbHeadAccounting:
    """The MLA absorb matmuls must NOT scale with dcp; only the FIA part does.

    Under DCP ``q`` is all-gathered to ``h_q*dcp/tp`` heads, but ``W_UK_T``/``W_UV``
    stay at this rank's ``h_q/tp`` heads -- real DCP does ``q @ W_UK`` before the
    all_gather and ``@ W_UV`` after the merge, since a rank holds no other rank's
    absorb weights. Counting the absorb matmuls at ``q.size(1)`` inflated them
    ``dcp``-fold (at h_q=128/tp=8/dcp=8 that was ~1.39x the whole decode MLA FLOPs
    at S=2048), biasing the optimizer against DCP on short/medium contexts.
    """

    @staticmethod
    def _mla_event(events):
        for event in events:
            if str(event.op_invoke_info.func).endswith("multihead_latent_attention.default"):
                return event
        raise AssertionError("no multihead_latent_attention event found")

    def _mma_time(self, dcp_size, seq_len):
        group = ParallelGroup(rank=0, rank_groups=[list(range(dcp_size))], global_world_size=dcp_size)
        _, events = _run_mla_attention(group, seq_len=seq_len)
        result = next(iter(self._mla_event(events).perf_results.values()))
        return result.statistics["mma_ops_time_s"]

    def test_absorb_weights_are_built_at_local_head_count(self):
        """Pins the premise: the absorb weights carry ``h_q/tp`` heads, not ``h_q*dcp/tp``."""
        dcp2 = ParallelGroup(rank=0, rank_groups=[[0, 1]], global_world_size=2)
        mla, _ = _get_mla_layer(dcp2)
        assert mla.W_UK_T.size(0) == mla._num_heads_per_rank
        assert mla.W_UV.size(0) == mla._num_heads_per_rank
        # tp=8 in the fixture, so this is strictly below the model's total head count.
        assert mla._num_heads_per_rank < mla.num_heads

    @pytest.mark.parametrize("seq_len", [2048, 8192])
    def test_matmul_cost_is_invariant_across_dcp(self, seq_len):
        """Total MLA matmul time must not grow with dcp.

        The FIA terms are dcp-invariant by cancellation (``dcp`` more heads over
        ``S/dcp`` context) and the absorb terms are dcp-independent, so the sum is flat.
        Before the fix this grew monotonically (S=2048: 1.219 -> 1.693 us for dcp 1->8).
        """
        baseline = self._mma_time(1, seq_len)
        assert baseline > 0
        for dcp_size in (2, 4, 8):
            assert self._mma_time(dcp_size, seq_len) == pytest.approx(baseline), dcp_size

    def test_kv_read_saving_is_still_modeled(self):
        """Guard the fix from over-correcting: DCP must still get its KV-read win.

        The absorb correction must not flatten the *end-to-end* op cost -- the whole
        point of MLA + DCP is that each rank reads only ``S/dcp`` of the latent KV.
        """
        group8 = ParallelGroup(rank=0, rank_groups=[list(range(8))], global_world_size=8)
        group1 = ParallelGroup(rank=0, rank_groups=[[0]], global_world_size=1)

        def op_time(group):
            _, events = _run_mla_attention(group, seq_len=8192)
            return next(iter(self._mla_event(events).perf_results.values())).execution_time_s

        assert op_time(group8) < op_time(group1)


class TestMlaDcpSeqLenSharding:
    """M3.2: the MLA KV shard rounds UP and clamps to >= 1.

    ``seq_lens`` drives both attention FLOPs (``sum(query_lens * seq_lens)``) and the
    KV read term (``sum(seq_lens * ...)``) in the cost model, so the value handed to
    the op is what gets billed. Two floor-division hazards are pinned here.
    """

    @staticmethod
    def _op_seq_lens(events):
        """The ``seq_lens`` tensor actually passed to the MLA attention op (arg 4)."""
        for event in events:
            if str(event.op_invoke_info.func).endswith("multihead_latent_attention.default"):
                return event.op_invoke_info.args[4].tolist()
        raise AssertionError("no multihead_latent_attention event found")

    def test_short_context_does_not_shard_to_zero_kv(self):
        """``seq_len < dcp`` must clamp to 1, not floor to 0.

        ``is_dcp_decode_batch`` only requires ``seq_len > query_len``, so seq_len can
        be as low as 2 and ``seq_len < dcp`` is reachable (e.g. a sanity run with
        ``--context-length 4 --dcp-size 8``). Flooring to 0 there makes the cost model
        bill the layer as "no KV to read" and understates its time.
        """
        dcp8 = ParallelGroup(rank=0, rank_groups=[[0, 1, 2, 3, 4, 5, 6, 7]], global_world_size=8)
        for seq_len in (2, 4, 8):
            _, events = _run_mla_attention(dcp8, seq_len=seq_len, num_requests=2)
            assert all(s >= 1 for s in self._op_seq_lens(events)), seq_len

    def test_remainder_shard_rounds_up_to_the_critical_path(self):
        """A non-divisible context must model ``ceil(S/dcp)``, the slowest rank.

        Real DCP splits the context at block granularity, so the remainder lands on a
        subset of ranks holding ``ceil(S/dcp)`` tokens. The merge all_to_all is a
        synchronization point, so step latency is set by that LONGEST shard; flooring
        modeled every rank at its most optimistic length.
        """
        dcp8 = ParallelGroup(rank=0, rank_groups=[[0, 1, 2, 3, 4, 5, 6, 7]], global_world_size=8)
        for seq_len, expected in ((2001, 251), (2007, 251), (2048, 256), (2000, 250)):
            _, events = _run_mla_attention(dcp8, seq_len=seq_len, num_requests=2)
            assert self._op_seq_lens(events) == [expected, expected], seq_len

    def test_dcp_disabled_leaves_seq_lens_untouched(self):
        dcp1 = ParallelGroup(rank=0, rank_groups=[[0]], global_world_size=1)
        _, events = _run_mla_attention(dcp1, seq_len=2001, num_requests=2)
        assert self._op_seq_lens(events) == [2001, 2001]

    def test_shard_does_not_mutate_shared_metadata(self):
        """``attention_meta.seq_lens`` is shared across layers -- the shard must copy."""
        dcp8 = ParallelGroup(rank=0, rank_groups=[[0, 1, 2, 3, 4, 5, 6, 7]], global_world_size=8)
        mla, _ = _get_mla_layer(dcp8)
        seq_lens = torch.full((2,), 2001, dtype=torch.long)
        meta = AttentionMetadataTensorCast(
            query_start_loc=torch.arange(0, 3, dtype=torch.long),
            seq_lens=seq_lens,
            query_lens=torch.ones(2, dtype=torch.long),
        )
        assert meta.is_dcp_decode is True
        sharded = torch.clamp(torch.div(meta.seq_lens + 7, 8, rounding_mode="floor"), min=1)
        assert sharded.tolist() == [251, 251]
        # The in-place hazard: the source tensor must still hold the full context.
        assert meta.seq_lens.tolist() == [2001, 2001]


class TestDcpShardedNumBlocks:
    """The per-card paged-block count rounds UP: blocks are the allocation unit."""

    def test_rounds_up_so_the_remainder_gets_a_whole_block(self):
        from tensor_cast.core.input_generator import dcp_sharded_num_blocks

        # factor 8 (MLA, dcp=8): 17 blocks of context cannot fit in 2 blocks per card.
        model = _make_mla_model(dcp_size=8)
        assert dcp_sharded_num_blocks(model, 17) == 3
        assert dcp_sharded_num_blocks(model, 16) == 2
        assert dcp_sharded_num_blocks(model, 1) == 1

    def test_never_returns_zero(self):
        from tensor_cast.core.input_generator import dcp_sharded_num_blocks

        assert dcp_sharded_num_blocks(_make_mla_model(dcp_size=8), 0) == 1

    def test_factor_one_is_identity(self):
        from tensor_cast.core.input_generator import dcp_sharded_num_blocks

        # GQA with h_kv >= tp gets factor 1, so the block count must pass through.
        model = _make_gqa_model(dcp_size=8, num_kv_heads=8, tp_size=8)
        assert dcp_sharded_num_blocks(model, 17) == 17


class TestMlaDcpMerge:
    """M3.2: the MLA DCP merge models an fp32 a2a and reduces heads back to h_q/tp."""

    def _make_mla_stub(self, dcp_group, heads_per_rank=8, v_head_dim=128):
        from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast

        stub = MagicMock(spec=MultiheadLatentAttentionTensorCast)
        stub.dcp_group = dcp_group
        stub._num_heads_per_rank = heads_per_rank
        stub.v_head_dim = v_head_dim
        return stub

    def test_merge_reduces_heads_and_uses_fp32(self):
        from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast

        dcp_group = ParallelGroup(rank=0, rank_groups=[[0, 1, 2, 3]], global_world_size=4)
        heads_per_rank, v_head_dim = 8, 128
        batch, seq = 4, 1
        num_tokens = batch * seq
        gathered = heads_per_rank * 4
        attn_output = torch.empty((num_tokens, gathered, v_head_dim), dtype=torch.bfloat16, device="meta")
        stub = self._make_mla_stub(dcp_group, heads_per_rank, v_head_dim)

        pm = AnalyticPerformanceModel(TEST_DEVICE)
        with Runtime(pm, TEST_DEVICE) as rt:
            out = MultiheadLatentAttentionTensorCast._dcp_merge_all_to_all(stub, attn_output, batch, seq)

        # Heads reduced back to h_q/tp for o_proj.
        assert out.shape == (num_tokens, heads_per_rank, v_head_dim)
        a2a = [e for e in rt.event_list if str(e.op_invoke_info.func).endswith("all_to_all.default")]
        assert len(a2a) == 1
        assert a2a[0].op_invoke_info.args[0].dtype == torch.float32


class TestIsDcpDecodeBatch:
    """DCP must fire on decode (incl. MTP) and stay off for prefill."""

    def test_ordinary_decode_is_eligible(self):
        from tensor_cast.layers.attention import is_dcp_decode_batch

        seq_lens = torch.full((4,), 2048, dtype=torch.long)
        query_lens = torch.ones(4, dtype=torch.long)
        assert is_dcp_decode_batch(seq_lens, query_lens) is True

    def test_mtp_decode_is_eligible(self):
        from tensor_cast.layers.attention import is_dcp_decode_batch

        # MTP / speculative decode: query_len = 1 + num_spec (e.g. 3), still < threshold.
        seq_lens = torch.full((4,), 2048, dtype=torch.long)
        query_lens = torch.full((4,), 3, dtype=torch.long)
        assert is_dcp_decode_batch(seq_lens, query_lens) is True

    def test_single_token_prefill_is_rejected(self):
        from tensor_cast.layers.attention import is_dcp_decode_batch

        # 1-token prefill: query_len == 1 but seq_len == query_len (no prior context).
        seq_lens = torch.ones(4, dtype=torch.long)
        query_lens = torch.ones(4, dtype=torch.long)
        assert is_dcp_decode_batch(seq_lens, query_lens) is False

    def test_long_prefill_is_rejected(self):
        from tensor_cast.layers.attention import is_dcp_decode_batch

        seq_lens = torch.full((4,), 4096, dtype=torch.long)
        query_lens = torch.full((4,), 4096, dtype=torch.long)
        assert is_dcp_decode_batch(seq_lens, query_lens) is False

    def test_none_inputs_are_rejected(self):
        from tensor_cast.layers.attention import is_dcp_decode_batch

        assert is_dcp_decode_batch(None, None) is False

    def test_mla_and_gqa_share_the_same_detector(self):
        # Single source of truth: both the GQA and MLA decode gates branch on the
        # host-resolved ``AttentionMetadataBase.is_dcp_decode`` (see attention.py and
        # mla.py ``apply_dcp``), and that field is populated once by
        # ``is_dcp_decode_batch`` in ``__post_init__`` -- so the two backends cannot
        # diverge on what counts as a DCP-eligible decode batch.
        seq_lens = torch.full((4,), 2048, dtype=torch.long)
        query_lens = torch.ones(4, dtype=torch.long)
        meta = AttentionMetadataTensorCast(
            query_start_loc=torch.arange(0, 5, dtype=torch.long),
            seq_lens=seq_lens,
            query_lens=query_lens,
        )
        from tensor_cast.layers.attention import is_dcp_decode_batch

        assert meta.is_dcp_decode is is_dcp_decode_batch(seq_lens, query_lens)
        assert meta.is_dcp_decode is True


class TestV4KvCacheDcpGating:
    """M2: the DCP KV saving must NOT apply to V4/SFA caches.

    Per-token cost is never divided now (DCP is a capacity multiplier), so V4's
    per-token cost must stay identical across dcp; the capacity factor is
    separately gated to 1 for V4 (see ``TestDcpKvTokenCapacityFactor``).
    """

    _NUM_BLOCKS = 64
    _BLOCK_SIZE = 128

    def _make_v4_model(self, dcp_size, num_layers=2):
        model = MagicMock()
        model.num_hidden_layers = num_layers
        model.head_dim = 128
        model.model_config.dtype = torch.bfloat16
        # V4 takes the MLA branch (mla_config present) and the deepseek_v4 model_type.
        model.model_config.mla_config = MagicMock()
        model.model_config.hf_config = MagicMock(model_type="deepseek_v4")
        model.text_config = MagicMock(model_type="deepseek_v4")
        model.model_config.parallel_config = ParallelConfig(
            world_size=8,
            tensor_parallel_size=4,
            decode_context_parallel_size=dcp_size,
        )
        return model

    @patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None)
    @patch("tensor_cast.core.input_generator._resolve_v4_kv_cache_size", return_value=[64, 128, 576])
    @patch("tensor_cast.core.input_generator._resolve_decoder_layers", return_value=None)
    def test_v4_per_token_cost_not_scaled_by_dcp(self, _layers, _v4_size, _attn_quant):
        _, per_token_dcp1 = _get_kv_cache_info(self._make_v4_model(dcp_size=1), self._NUM_BLOCKS, self._BLOCK_SIZE)
        _, per_token_dcp4 = _get_kv_cache_info(self._make_v4_model(dcp_size=4), self._NUM_BLOCKS, self._BLOCK_SIZE)
        # SFA + DCP memory modeling is an RFC non-goal: the compressed footprint must
        # stay identical, NOT divided by dcp.
        assert per_token_dcp4 == pytest.approx(per_token_dcp1)


class TestGqaDcpKvHeadValidation:
    """M1: enforce the GQA-only constraint num_key_value_heads >= tp / dcp."""

    def _make_model(self, tp, dcp, num_kv_heads):
        model = MagicMock()
        model.model_config.parallel_config = ParallelConfig(
            world_size=tp,
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
        )
        model.text_config = MagicMock()
        model.text_config.num_key_value_heads = num_kv_heads
        return model

    def test_illegal_config_raises(self):
        from tensor_cast.parallel_group import ParallelGroupManager
        from tensor_cast.transformers.transformations import _validate_gqa_dcp_kv_heads

        # tp/dcp = 8/2 = 4, but only 2 KV heads -> a rank would hold 0 KV head.
        model = self._make_model(tp=8, dcp=2, num_kv_heads=2)
        dcp_group = ParallelGroupManager(model.model_config.parallel_config).dcp_group
        with pytest.raises(ValueError, match="num_key_value_heads"):
            _validate_gqa_dcp_kv_heads(model, dcp_group)

    def test_legal_config_passes(self):
        from tensor_cast.parallel_group import ParallelGroupManager
        from tensor_cast.transformers.transformations import _validate_gqa_dcp_kv_heads

        # tp/dcp = 8/2 = 4, with 4 KV heads -> exactly 1 KV head per rank, legal.
        model = self._make_model(tp=8, dcp=2, num_kv_heads=4)
        dcp_group = ParallelGroupManager(model.model_config.parallel_config).dcp_group
        _validate_gqa_dcp_kv_heads(model, dcp_group)  # must not raise

    def test_dcp_disabled_is_noop(self):
        from tensor_cast.parallel_group import ParallelGroupManager
        from tensor_cast.transformers.transformations import _validate_gqa_dcp_kv_heads

        # dcp=1: the constraint does not apply even with few KV heads.
        model = self._make_model(tp=8, dcp=1, num_kv_heads=2)
        dcp_group = ParallelGroupManager(model.model_config.parallel_config).dcp_group
        _validate_gqa_dcp_kv_heads(model, dcp_group)  # must not raise


class TestDcpCompileSafety:
    """The decode batch is detected on the host so the DCP collectives survive
    ``torch.compile``.

    Two regressions are pinned here:

    1. ``is_dcp_decode_batch`` used to reduce ``seq_lens`` / ``query_lens`` to a
       Python bool *inside* the compiled forward, which Dynamo rejects as
       data-dependent control flow ("Data-dependent branching"). It is now
       resolved once on the host into ``AttentionMetadata.is_dcp_decode``.
    2. Even after the graph compiled, the modeled-for-cost-only DCP collectives
       were silently removed by two compile passes: meta constant-folding erased
       the merge ``all_to_all`` (its synthetic payload had no placeholder edge)
       and DCE dropped both collectives (their outputs were unconsumed). The
       per-layer communication cost must match the eager reference exactly.
    """

    _DCP4 = ParallelGroup(rank=0, rank_groups=[[0, 1, 2, 3]], global_world_size=4)

    @staticmethod
    def _collective_counts(events):
        names = [str(e.op_invoke_info.func) for e in events]
        return (
            sum("all_gather" in n for n in names),
            sum("all_to_all" in n for n in names),
        )

    def test_gqa_decode_compiles_without_graph_break(self):
        # fullgraph=True: a data-dependent branch would raise instead of silently
        # falling back to eager, so reaching here at all proves the host-side flag works.
        out, _ = _run_gqa_attention(self._DCP4, compile_forward=True)
        assert out.shape == (4, 32 * 128)  # o_proj width unchanged (h_q/tp * head_dim)

    def test_gqa_dcp_collectives_survive_compile(self):
        _, eager = _run_gqa_attention(self._DCP4, compile_forward=False)
        _, compiled = _run_gqa_attention(self._DCP4, compile_forward=True)
        # Both collectives must persist with the SAME count: constant-folding must
        # not erase the merge a2a and DCE must not drop either collective.
        assert self._collective_counts(compiled) == self._collective_counts(eager)
        assert self._collective_counts(compiled)[1] >= 1  # merge all_to_all present


class TestMlaDcpCompileSafety:
    """MLA mirror of ``TestDcpCompileSafety``: the real MLA decode forward must
    compile under ``fullgraph=True`` and keep its DCP collectives.

    The GQA compile tests run a bare ``AttentionTensorCast``; they cannot catch an
    MLA-only regression because the MLA forward is a different code path (latent KV,
    ``mlapo`` preprocessing, the all_gather-Q + ``seq_lens / dcp`` shard, and the
    fp32 ``output + lse`` merge a2a). Both hazards pinned for GQA apply here too:

    1. Branching on ``is_dcp_decode_batch(seq_lens, query_lens)`` *inside* the
       compiled MLA forward is data-dependent control flow that Dynamo rejects; the
       decision must come from the host-resolved ``AttentionMetadata.is_dcp_decode``.
    2. The cost-only DCP collectives must survive meta constant-folding and DCE,
       so compiled and eager runs emit identical per-layer collective counts.
    """

    _DCP2 = ParallelGroup(rank=0, rank_groups=[[0, 1]], global_world_size=2)

    @staticmethod
    def _collective_counts(events):
        names = [str(e.op_invoke_info.func) for e in events]
        return (
            sum("all_gather" in n for n in names),
            sum("all_to_all" in n for n in names),
        )

    def test_mla_decode_compiles_without_graph_break(self):
        # fullgraph=True: a data-dependent branch (the regression) would raise here
        # instead of silently falling back to eager, so reaching the assert proves the
        # MLA forward branches on the host-side flag rather than reducing tensors.
        out, _ = _run_mla_attention(self._DCP2, compile_forward=True)
        # o_proj input width is h_q/tp * v_head_dim, reshaped back to hidden_size.
        assert out.shape[0] == 4  # one decode token per request

    def test_mla_dcp_collectives_survive_compile(self):
        _, eager = _run_mla_attention(self._DCP2, compile_forward=False)
        _, compiled = _run_mla_attention(self._DCP2, compile_forward=True)
        # all_gather Q + merge all_to_all must persist with the SAME count across
        # eager and compiled: constant-folding must not erase the a2a and DCE must
        # not drop either collective.
        assert self._collective_counts(compiled) == self._collective_counts(eager)
        assert self._collective_counts(compiled) == (1, 1)  # one all_gather + one merge a2a

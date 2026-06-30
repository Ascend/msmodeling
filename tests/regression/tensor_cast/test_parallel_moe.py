import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from parameterized import parameterized
from tensor_cast.compilation import get_backend
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.moe_layer import (
    ExpertWrapper,
    FusedMoETensorCast,
    MoELayer,
    ParallelMoELayer,
)
from tensor_cast.model_config import ModelConfig, MoEConfig, ParallelConfig, QuantConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.custom_model_registry import get_moe_config
from tensor_cast.transformers.model import TransformerModel

from .conftest import get_session_hf_config
from .test_common import create_mla_metadata_and_kv_cache, get_cached_build_model


def get_parallel_config(parallel_configuration: tuple):
    world_size = parallel_configuration[0]
    do_ep = parallel_configuration[4]
    ep_size = world_size if do_ep else 1
    moe_dp_size = 1 if do_ep else world_size
    parallel_config = ParallelConfig(
        world_size=parallel_configuration[0],
        tensor_parallel_size=parallel_configuration[1],
        mlp_tensor_parallel_size=parallel_configuration[2],
        lmhead_tensor_parallel_size=parallel_configuration[3],
        expert_parallel_size=ep_size,
        moe_data_parallel_size=moe_dp_size,
        moe_tensor_parallel_size=1,
    )
    return parallel_config


class ParallelMoETestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._model_cache = {}
        cls._transformer_cache = {}

    @classmethod
    def _get_transformer_model(cls, model_id: str, model_config: ModelConfig) -> TransformerModel:
        key = (model_id, repr(model_config))
        if key not in cls._transformer_cache:
            cls._transformer_cache[key] = TransformerModel(model_id, model_config)
        return cls._transformer_cache[key]

    def setUp(self):
        num_tokens = 100
        self.input_batch_size = 2
        self.compile_backend = get_backend()
        with torch.device("meta"):
            self.inputs = torch.empty([self.input_batch_size, num_tokens], dtype=torch.long)
            self.position_ids = torch.empty([self.input_batch_size, num_tokens], dtype=torch.long)

    def _check_comm_analytic(self, trace_events, comm_op_name):
        count = 0
        for event in trace_events:
            if event["name"] == comm_op_name:
                self.assertIn("message_size_bytes", event["args"])
                count += 1
        self.assertGreater(count, 0)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-235B-A22B", (16, 1, 1, 1, False)],
            ["Qwen/Qwen3-235B-A22B", (16, 2, 4, 1, False)],
            ["Qwen/Qwen3-235B-A22B", (16, 1, 1, 1, True)],
            ["Qwen/Qwen3-235B-A22B", (16, 2, 4, 1, True)],
        ]
    )
    def test_model_with_ep(self, model_id, parallel_configuration):
        hf_config = get_session_hf_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        parallel_config = get_parallel_config(parallel_configuration)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            enable_repetition=True,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        model = self._get_transformer_model(model_id, model_config)

        num_tokens = 100
        output_batch_size = self.input_batch_size
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with (
            Runtime(perf_model, machine_config) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (output_batch_size, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.init_routing_v2.default", result)
        self.assertIn("tensor_cast.unpermute_tokens.default", result)
        if parallel_config.has_ep():
            self.assertIn("tensor_cast.all_to_all.default", result)
            self._check_comm_analytic(runtime.get_trace_events(), "tensor_cast.all_to_all.default")

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, 2, 4, 1, True), (False, False)],
            # ["deepseek-ai/DeepSeek-V3.1", (16, 2, 4, 1, True), (True, False)],
            # ["deepseek-ai/DeepSeek-V3.1", (16, 2, 4, 1, True), (False, True)],
            # ["moonshotai/Kimi-K2-Base", (16, 2, 4, 1, True), (True, True)],
        ]
    )
    def test_deepseek_with_ep(self, model_id, parallel_configuration, moe_configuration):
        user_config = UserInputConfig(
            model_id=model_id,
            world_size=parallel_configuration[0],
            tp_size=parallel_configuration[1],
            mlp_tp_size=parallel_configuration[2],
            lmhead_tp_size=parallel_configuration[3],
            ep_size=parallel_configuration[0] if parallel_configuration[4] else 1,
            moe_dp_size=1 if parallel_configuration[4] else parallel_configuration[0],
            moe_tp_size=1,
            enable_redundant_experts=moe_configuration[0],
            enable_external_shared_experts=moe_configuration[1],
        )

        model = get_cached_build_model(self._model_cache, user_config)

        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model.model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))

        result = runtime.table_averages()
        self.assertIn("tensor_cast.init_routing_v2.default", result)
        self.assertIn("tensor_cast.unpermute_tokens.default", result)
        if model.model_config.parallel_config.has_ep():
            self.assertIn("tensor_cast.all_to_all.default", result)
            self._check_comm_analytic(runtime.get_trace_events(), "tensor_cast.all_to_all.default")

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (64, 2, 4, 1, True), (True, True), 8, 24],
            ["deepseek-ai/DeepSeek-V3.1", (64, 2, 4, 1, True), (False, False), 0, 0],
            ["deepseek-ai/DeepSeek-V3.1", (64, 2, 4, 1, True), (True, False), 0, 64],
            ["deepseek-ai/DeepSeek-V3.1", (64, 2, 4, 1, True), (False, True), 8, 24],
        ]
    )
    def test_deepseek_with_redundant_experts_and_external_shared_expert(
        self,
        model_id,
        parallel_configuration,
        moe_configuration,
        num_external_shared_experts,
        num_redundant_experts,
    ):
        user_config = UserInputConfig(
            model_id=model_id,
            world_size=parallel_configuration[0],
            tp_size=parallel_configuration[1],
            mlp_tp_size=parallel_configuration[2],
            lmhead_tp_size=parallel_configuration[3],
            ep_size=parallel_configuration[0] if parallel_configuration[4] else 1,
            moe_dp_size=1 if parallel_configuration[4] else parallel_configuration[0],
            moe_tp_size=1,
            enable_redundant_experts=moe_configuration[0],
            enable_external_shared_experts=moe_configuration[1],
        )

        model = get_cached_build_model(self._model_cache, user_config)
        self.assertEqual(
            model.num_external_shared_experts,
            num_external_shared_experts,
        )
        self.assertEqual(
            model.num_redundant_experts,
            num_redundant_experts,
        )


def _make_fake_gate(top_k, tp_size=1, tp_rank=0, shard_by_tp=False):
    """Return an nn.Identity whose output is replaced by gate-shaped zeros via hooks."""
    module = torch.nn.Identity()
    module.top_k = top_k
    module.tp_size = tp_size
    module.tp_rank = tp_rank
    module.shard_by_tp = shard_by_tp
    module.seen_shape = None

    def _pre_hook(m, args):
        m.seen_shape = tuple(args[0].shape)

    def _post_hook(m, inp, out):
        hidden_states = inp[0]
        num_tokens = hidden_states.shape[0]
        local_tokens = num_tokens
        if m.shard_by_tp and hidden_states.dim() == 2 and m.tp_size > 1:
            local_tokens = (num_tokens + m.tp_size - 1) // m.tp_size
        return torch.zeros(
            local_tokens,
            256,
            device=hidden_states.device,
            dtype=torch.float32,
        )

    module.register_forward_pre_hook(_pre_hook)
    module.register_forward_hook(_post_hook)
    return module


def _make_fake_fused_moe(
    moe_config,
    experts,
    shared_experts,
    shared_experts_gate,
    top_k,
    ep_group=None,
    num_external_shared_experts=0,
    num_global_experts=None,
    global_tp_size=1,
):
    """Return an nn.Identity that records forward inputs via hooks."""
    module = torch.nn.Identity()
    module.moe_config = moe_config
    if experts is None:
        module.experts = None
    elif isinstance(experts, ExpertWrapper):
        module.experts = experts
    else:
        module.experts = ExpertWrapper(experts)
    module.shared_experts = shared_experts
    module.shared_experts_gate = shared_experts_gate
    module.top_k = top_k
    module.ep_group = ep_group
    module.num_external_shared_experts = num_external_shared_experts
    module.num_global_experts = num_global_experts or (module.experts.num_experts if module.experts is not None else 0)
    module.forward_inputs = []

    def _run_shared_experts(hidden_states):
        assert shared_experts is not None
        output = shared_experts(hidden_states)
        if shared_experts_gate:
            output = torch.nn.functional.sigmoid(shared_experts_gate(hidden_states)) * output
        return output

    module._run_shared_experts = _run_shared_experts

    def _pre_hook(m, args, kwargs):
        skip = args[3] if len(args) > 3 else kwargs.get("skip_shared_experts", False)
        m.forward_inputs.append(
            (
                tuple(args[0].shape),
                tuple(args[1].shape),
                tuple(args[2].shape),
                skip,
            )
        )
        # Filter args to (hidden_states,) so Identity.forward works.
        return (args[0],), {}

    module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    _make_fake_fused_moe.last_instance = module
    return module


_make_fake_fused_moe.last_instance = None


def _make_spy_identity():
    """Return an nn.Identity with call_count and seen_shape tracked via forward_pre_hook."""
    module = torch.nn.Identity()
    module.call_count = 0
    module.seen_shape = None

    def _hook(m, args):
        m.call_count += 1
        m.seen_shape = tuple(args[0].shape)

    module.register_forward_pre_hook(_hook)
    return module


def _make_spy_zeros_gate():
    """Like _make_spy_identity but forward_hook replaces the output with zeros."""
    module = torch.nn.Identity()
    module.call_count = 0
    module.seen_shape = None

    def _pre_hook(m, args):
        m.call_count += 1
        m.seen_shape = tuple(args[0].shape)

    def _post_hook(m, inp, out):
        return torch.zeros_like(out)

    module.register_forward_pre_hook(_pre_hook)
    module.register_forward_hook(_post_hook)
    return module


class _FakeParallelGroup:
    def __init__(self, world_size, rank_in_group=0, name="pg"):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.name = name
        self.all_reduce_calls = 0

    def all_reduce(self, input_):
        self.all_reduce_calls += 1
        return input_

    def slice(self, input_, dim=0):
        split_size = input_.shape[dim] // self.world_size
        start = self.rank_in_group * split_size
        return torch.narrow(input_, dim=dim, start=start, length=split_size)

    def all_gather(self, input_, dim=0):
        return torch.cat([input_] * self.world_size, dim=dim)

    def all_to_all(self, input_, output_split_sizes, input_split_sizes):
        return input_


class _PadSensitiveExpert(torch.nn.Module):
    def __init__(self, divisor):
        super().__init__()
        self.divisor = divisor
        self.call_count = 0
        self.seen_shape = None

    def forward(self, x):
        self.call_count += 1
        self.seen_shape = tuple(x.shape)
        assert x.shape[0] % self.divisor == 0
        return x + 1


def test_parallel_moe_ep_route_before_tp_slice_smoke():
    gate = _make_fake_gate(top_k=2, tp_size=2, shard_by_tp=False)
    module = SimpleNamespace(
        gate=gate,
        top_k=2,
        norm_topk_prob=False,
        experts=torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)]),
        shared_experts=None,
        shared_experts_gate=None,
    )
    moe_config = MoEConfig(module_name="FakeMoE")

    with patch("tensor_cast.layers.moe_layer.FusedMoETensorCast", _make_fake_fused_moe):
        moe_layer = MoELayer(moe_config, module)
        parallel_moe = ParallelMoELayer(
            module=moe_layer,
            global_dp_group=_FakeParallelGroup(world_size=1),
            global_tp_group=_FakeParallelGroup(world_size=2, rank_in_group=0),
            mlp_tp_group=_FakeParallelGroup(world_size=2, rank_in_group=0),
            ep_group=_FakeParallelGroup(world_size=2, rank_in_group=0),
            num_external_shared_experts=0,
            num_redundant_experts=0,
        )

        hidden_states = torch.empty(1, 6, 16, device="meta", dtype=torch.float16)
        output = parallel_moe(hidden_states)

    assert output.shape == (1, 6, 16)
    # For non-shared_expert_tp path, routing happens after TP slice.
    # With tp_size=2 and seq_len=6, each TP rank routes 3 tokens.
    assert gate.seen_shape == (3, 16)

    fused_moe = _make_fake_fused_moe.last_instance
    assert fused_moe is not None
    # After TP slice (world_size=2): 6/2 = 3 tokens per rank.
    assert fused_moe.forward_inputs == [((3, 16), (3, 2), (3, 2), False)]


def test_parallel_moe_ep_route_before_tp_slice_small_seq_len():
    gate = _make_fake_gate(top_k=2, tp_size=8, shard_by_tp=False)
    module = SimpleNamespace(
        gate=gate,
        top_k=2,
        norm_topk_prob=False,
        experts=torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)]),
        shared_experts=None,
        shared_experts_gate=None,
    )
    moe_config = MoEConfig(module_name="FakeMoE")

    with patch("tensor_cast.layers.moe_layer.FusedMoETensorCast", _make_fake_fused_moe):
        moe_layer = MoELayer(moe_config, module)
        parallel_moe = ParallelMoELayer(
            module=moe_layer,
            global_dp_group=_FakeParallelGroup(world_size=1),
            global_tp_group=_FakeParallelGroup(world_size=8, rank_in_group=0),
            mlp_tp_group=_FakeParallelGroup(world_size=8, rank_in_group=0),
            ep_group=_FakeParallelGroup(world_size=2, rank_in_group=0),
            num_external_shared_experts=0,
            num_redundant_experts=0,
        )

        hidden_states = torch.empty(1, 1, 16, device="meta", dtype=torch.float16)
        output = parallel_moe(hidden_states)

    assert output.shape == (1, 1, 16)
    # Gate still sees the real token count before TP-domain alignment.
    assert gate.seen_shape == (1, 16)

    fused_moe = _make_fake_fused_moe.last_instance
    assert fused_moe is not None
    # seq_len=1 pads to tp_size=8, then TP slice keeps 1 token per rank.
    assert fused_moe.forward_inputs == [((1, 16), (1, 2), (1, 2), False)]


def test_fused_moe_per_expert_local_padding_restores_real_token_count():
    hidden_states = torch.arange(64, dtype=torch.float32).view(4, 16)
    topk_indices = torch.zeros(4, 1, dtype=torch.long)
    topk_weights = torch.ones(4, 1, dtype=torch.float32)
    expert = _PadSensitiveExpert(divisor=8)
    fused_moe = FusedMoETensorCast(
        moe_config=MoEConfig(module_name="FakeMoE"),
        experts=torch.nn.ModuleList([expert, torch.nn.Identity()]),
        shared_experts=None,
        shared_experts_gate=None,
        top_k=1,
        ep_group=_FakeParallelGroup(world_size=1),
        num_global_experts=2,
        global_tp_size=8,
    )

    output = fused_moe(hidden_states, topk_indices, topk_weights)

    assert expert.call_count == 1
    assert expert.seen_shape == (8, 16)
    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype


def test_parallel_moe_shared_expert_tp_skip_inner_shared_experts():
    gate = _make_fake_gate(top_k=2, tp_size=2, shard_by_tp=True)
    shared_experts = _make_spy_identity()
    module = SimpleNamespace(
        gate=gate,
        top_k=2,
        norm_topk_prob=False,
        experts=torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)]),
        shared_experts=shared_experts,
        shared_experts_gate=None,
    )
    moe_config = MoEConfig(
        module_name="FakeMoE",
        enable_shared_expert_tp=True,
    )

    with patch("tensor_cast.layers.moe_layer.FusedMoETensorCast", _make_fake_fused_moe):
        moe_layer = MoELayer(moe_config, module)
        global_tp_group = _FakeParallelGroup(world_size=2, rank_in_group=0, name="tp")
        mlp_tp_group = _FakeParallelGroup(world_size=4, rank_in_group=0, name="mlp_tp")
        parallel_moe = ParallelMoELayer(
            module=moe_layer,
            global_dp_group=_FakeParallelGroup(world_size=1),
            global_tp_group=global_tp_group,
            mlp_tp_group=mlp_tp_group,
            ep_group=_FakeParallelGroup(world_size=2, rank_in_group=0),
            num_external_shared_experts=0,
            num_redundant_experts=0,
        )

        hidden_states = torch.empty(1, 6, 16, device="meta", dtype=torch.float16)
        output = parallel_moe(hidden_states)

    assert output.shape == (1, 6, 16)
    assert gate.seen_shape == (6, 16)
    assert shared_experts.call_count == 1
    assert shared_experts.seen_shape == (6, 16)
    assert global_tp_group.all_reduce_calls == 0
    assert mlp_tp_group.all_reduce_calls == 1

    fused_moe = _make_fake_fused_moe.last_instance
    assert fused_moe is not None
    assert fused_moe.forward_inputs == [((3, 16), (3, 2), (3, 2), True)]


def test_parallel_moe_shared_expert_tp_with_gate():
    gate = _make_fake_gate(top_k=2, tp_size=2, shard_by_tp=True)
    shared_experts = _make_spy_identity()
    shared_experts_gate = _make_spy_zeros_gate()
    module = SimpleNamespace(
        gate=gate,
        top_k=2,
        norm_topk_prob=False,
        experts=torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)]),
        shared_experts=shared_experts,
        shared_experts_gate=shared_experts_gate,
    )
    moe_config = MoEConfig(
        module_name="FakeMoE",
        enable_shared_expert_tp=True,
    )

    with patch("tensor_cast.layers.moe_layer.FusedMoETensorCast", _make_fake_fused_moe):
        moe_layer = MoELayer(moe_config, module)
        global_tp_group = _FakeParallelGroup(world_size=2, rank_in_group=0, name="tp")
        mlp_tp_group = _FakeParallelGroup(world_size=4, rank_in_group=0, name="mlp_tp")
        parallel_moe = ParallelMoELayer(
            module=moe_layer,
            global_dp_group=_FakeParallelGroup(world_size=1),
            global_tp_group=global_tp_group,
            mlp_tp_group=mlp_tp_group,
            ep_group=_FakeParallelGroup(world_size=2, rank_in_group=0),
            num_external_shared_experts=0,
            num_redundant_experts=0,
        )

        hidden_states = torch.empty(1, 6, 16, device="meta", dtype=torch.float16)
        output = parallel_moe(hidden_states)

    assert output.shape == (1, 6, 16)
    assert gate.seen_shape == (6, 16)
    assert shared_experts.call_count == 1
    assert shared_experts.seen_shape == (6, 16)
    assert shared_experts_gate.call_count == 1
    assert shared_experts_gate.seen_shape == (6, 16)
    assert global_tp_group.all_reduce_calls == 0
    assert mlp_tp_group.all_reduce_calls == 1

    fused_moe = _make_fake_fused_moe.last_instance
    assert fused_moe is not None
    assert fused_moe.forward_inputs == [((3, 16), (3, 2), (3, 2), True)]


def test_parallel_moe_shared_expert_tp_without_dp_transform():
    gate = _make_fake_gate(top_k=2)
    shared_experts = _make_spy_identity()
    module = SimpleNamespace(
        gate=gate,
        top_k=2,
        norm_topk_prob=False,
        experts=torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)]),
        shared_experts=shared_experts,
        shared_experts_gate=None,
    )
    moe_config = MoEConfig(
        module_name="FakeMoE",
        enable_shared_expert_tp=True,
    )

    with patch("tensor_cast.layers.moe_layer.FusedMoETensorCast", _make_fake_fused_moe):
        moe_layer = MoELayer(moe_config, module)
        global_tp_group = _FakeParallelGroup(world_size=2, rank_in_group=0, name="tp")
        mlp_tp_group = _FakeParallelGroup(world_size=4, rank_in_group=0, name="mlp_tp")
        parallel_moe = ParallelMoELayer(
            module=moe_layer,
            global_dp_group=_FakeParallelGroup(world_size=2),
            global_tp_group=global_tp_group,
            mlp_tp_group=mlp_tp_group,
            ep_group=_FakeParallelGroup(world_size=2, rank_in_group=0),
            num_external_shared_experts=0,
            num_redundant_experts=0,
        )

        hidden_states = torch.empty(1, 6, 16, device="meta", dtype=torch.float16)
        output = parallel_moe(hidden_states)

    assert output.shape == (1, 6, 16)
    assert parallel_moe.transform_dp_group is False
    assert gate.seen_shape == (6, 16)
    assert shared_experts.call_count == 1
    assert shared_experts.seen_shape == (6, 16)
    assert global_tp_group.all_reduce_calls == 0
    assert mlp_tp_group.all_reduce_calls == 1

    fused_moe = _make_fake_fused_moe.last_instance
    assert fused_moe is not None
    assert fused_moe.forward_inputs == [((6, 16), (6, 2), (6, 2), True)]


# ---------------------------------------------------------------------------
# Tests for non-shared-expert-tp EP path with gate_returns_raw_logits=True
# (GLM5-style: gate on full packed-local tokens, router_logits pad+sliced
#  inside route(), hidden_states sliced by _dp_transform_enter afterwards)
# ---------------------------------------------------------------------------


def test_raw_logits_non_shared_ep_gate_sees_full_tokens():
    """Gate must receive pre-TP-slice token count (27), not post-slice (7)."""
    top_k = 2
    tp_size = 4
    seq_len = 27  # packed local tokens; pad to 28 then slice to 7 per TP rank
    gate = _make_fake_gate(top_k=top_k, tp_size=tp_size, shard_by_tp=False)
    module = SimpleNamespace(
        gate=gate,
        top_k=top_k,
        norm_topk_prob=False,
        experts=torch.nn.ModuleList([torch.nn.Identity() for _ in range(8)]),
        shared_experts=None,
        shared_experts_gate=None,
    )
    moe_config = MoEConfig(module_name="FakeMoE", gate_returns_raw_logits=True)

    with patch("tensor_cast.layers.moe_layer.FusedMoETensorCast", _make_fake_fused_moe):
        moe_layer = MoELayer(moe_config, module)
        parallel_moe = ParallelMoELayer(
            module=moe_layer,
            global_dp_group=_FakeParallelGroup(world_size=1),
            global_tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            mlp_tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            ep_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            num_external_shared_experts=0,
            num_redundant_experts=0,
        )

        hidden_states = torch.empty(seq_len, 16, device="meta", dtype=torch.float16)
        output = parallel_moe(hidden_states)

    assert output.shape == (seq_len, 16)
    # Gate must see the full packed-local token count, not the TP-sliced count.
    assert gate.seen_shape == (seq_len, 16), f"gate saw {gate.seen_shape}, want ({seq_len}, 16)"

    # fused_moe receives hidden_states after _dp_transform_enter:
    # 27 -> pad to 28 -> slice by tp_size=4 -> 7 tokens per rank.
    expected_local_tokens = (seq_len + tp_size - 1) // tp_size
    fused_moe = _make_fake_fused_moe.last_instance
    assert fused_moe is not None
    assert fused_moe.forward_inputs[0][0] == (expected_local_tokens, 16), (
        f"fused_moe saw {fused_moe.forward_inputs[0][0]}, want ({expected_local_tokens}, 16)"
    )


def test_raw_logits_non_shared_ep_small_seq_len_pad_boundary():
    """seq_len=1 with tp_size=4: pad to 4, slice to 1; gate still sees 1 token."""
    top_k = 2
    tp_size = 4
    seq_len = 1
    gate = _make_fake_gate(top_k=top_k, tp_size=tp_size, shard_by_tp=False)
    module = SimpleNamespace(
        gate=gate,
        top_k=top_k,
        norm_topk_prob=False,
        experts=torch.nn.ModuleList([torch.nn.Identity() for _ in range(8)]),
        shared_experts=None,
        shared_experts_gate=None,
    )
    moe_config = MoEConfig(module_name="FakeMoE", gate_returns_raw_logits=True)

    with patch("tensor_cast.layers.moe_layer.FusedMoETensorCast", _make_fake_fused_moe):
        moe_layer = MoELayer(moe_config, module)
        parallel_moe = ParallelMoELayer(
            module=moe_layer,
            global_dp_group=_FakeParallelGroup(world_size=1),
            global_tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            mlp_tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            ep_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            num_external_shared_experts=0,
            num_redundant_experts=0,
        )

        hidden_states = torch.empty(seq_len, 16, device="meta", dtype=torch.float16)
        output = parallel_moe(hidden_states)

    assert output.shape == (seq_len, 16)
    assert gate.seen_shape == (seq_len, 16)

    # 1 -> pad to 4 -> slice to 1 per rank (rank 0 gets index [0:1])
    fused_moe = _make_fake_fused_moe.last_instance
    assert fused_moe is not None
    assert fused_moe.forward_inputs[0][0] == (seq_len, 16)


def test_raw_logits_non_shared_ep_no_dp_transform():
    """transform_dp_group=False with gate_returns_raw_logits=True: _inner runs directly."""
    top_k = 2
    tp_size = 4
    seq_len = 6
    gate = _make_fake_gate(top_k=top_k, tp_size=tp_size, shard_by_tp=False)
    module = SimpleNamespace(
        gate=gate,
        top_k=top_k,
        norm_topk_prob=False,
        experts=torch.nn.ModuleList([torch.nn.Identity() for _ in range(8)]),
        shared_experts=None,
        shared_experts_gate=None,
    )
    moe_config = MoEConfig(module_name="FakeMoE", gate_returns_raw_logits=True)

    with patch("tensor_cast.layers.moe_layer.FusedMoETensorCast", _make_fake_fused_moe):
        moe_layer = MoELayer(moe_config, module)
        # dp_group.world_size == ep_group.world_size → transform_dp_group=False
        parallel_moe = ParallelMoELayer(
            module=moe_layer,
            global_dp_group=_FakeParallelGroup(world_size=tp_size),
            global_tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            mlp_tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            ep_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            num_external_shared_experts=0,
            num_redundant_experts=0,
        )

        hidden_states = torch.empty(seq_len, 16, device="meta", dtype=torch.float16)
        output = parallel_moe(hidden_states)

    assert parallel_moe.transform_dp_group is False
    assert output.shape == (seq_len, 16)
    # No TP slice: gate and fused_moe both see the full token count.
    assert gate.seen_shape == (seq_len, 16)
    fused_moe = _make_fake_fused_moe.last_instance
    assert fused_moe is not None
    assert fused_moe.forward_inputs[0][0] == (seq_len, 16)


def test_raw_logits_false_non_shared_ep_regression():
    """gate_returns_raw_logits=False path must be unaffected: gate sees post-slice tokens.

    Uses 3D input (1, seq_len, 16) to also exercise the flatten/unflatten
    reshape path in forward(), covering origin_shape[:2] view restore.
    """
    top_k = 2
    tp_size = 2
    seq_len = 6
    gate = _make_fake_gate(top_k=top_k, tp_size=tp_size, shard_by_tp=False)
    module = SimpleNamespace(
        gate=gate,
        top_k=top_k,
        norm_topk_prob=False,
        experts=torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)]),
        shared_experts=None,
        shared_experts_gate=None,
    )
    moe_config = MoEConfig(module_name="FakeMoE")

    with patch("tensor_cast.layers.moe_layer.FusedMoETensorCast", _make_fake_fused_moe):
        moe_layer = MoELayer(moe_config, module)
        parallel_moe = ParallelMoELayer(
            module=moe_layer,
            global_dp_group=_FakeParallelGroup(world_size=1),
            global_tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            mlp_tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            ep_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=0),
            num_external_shared_experts=0,
            num_redundant_experts=0,
        )

        hidden_states = torch.empty(1, seq_len, 16, device="meta", dtype=torch.float16)
        output = parallel_moe(hidden_states)

    assert output.shape == (1, seq_len, 16)
    # False path: _dp_transform_enter runs first, gate sees sliced token count.
    assert gate.seen_shape == ((seq_len + tp_size - 1) // tp_size, 16)

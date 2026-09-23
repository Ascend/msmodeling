# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from types import SimpleNamespace

import pytest
import torch

from tensor_cast.layers.glm5_next import glm5_next_nope_position_embeddings
from tensor_cast.transformers.builtin_model.glm5_next import (
    Glm5NextMoeExpertMLP,
)
from tensor_cast.transformers.custom_model_registry import get_model_profile


class _PackedExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 2
        self.intermediate_dim = 3
        self.swiglu_limit = 10.0
        self.gate_up_proj = torch.nn.Parameter(torch.arange(12.0).reshape(1, 6, 2))
        self.down_proj = torch.nn.Parameter(torch.arange(6.0).reshape(1, 2, 3))


def test_glm5_next_profile_registers_language_components():
    profile = get_model_profile("glm5_next")

    assert profile is not None
    assert profile.moe_module_name == "Glm5NextTextMoE"
    assert profile.mla_module_name == "Glm5NextTextAttention"
    assert profile.language_layers_path_str == "language_model.layers"
    assert profile.visual_mlp_linear_mapping["visual.blocks.*.mlp.gate_proj"] is not None
    assert profile.visual_merger_linear_mapping["visual.merger.down_proj"] is not None
    # ``merger.proj`` feeds a full-width LayerNorm and must remain replicated
    # when vision TP shards the subsequent SwiGLU projections.
    assert "visual.merger.proj" not in profile.visual_merger_linear_mapping


def test_glm5_next_packed_expert_preserves_clamped_swiglu_layout():
    packed = _PackedExperts()
    expert = Glm5NextMoeExpertMLP(packed, 0)
    hidden_states = torch.tensor([[1.0, -1.0]])

    expected_gate, expected_up = torch.nn.functional.linear(hidden_states, packed.gate_up_proj[0]).chunk(2, dim=-1)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(expected_gate.clamp(max=10.0)) * expected_up.clamp(min=-10.0, max=10.0),
        packed.down_proj[0],
    )
    torch.testing.assert_close(expert(hidden_states), expected)


def test_glm5_next_nope_embeddings_keep_sequence_and_zero_rotary_dimension():
    hidden_states = torch.empty(2, 7, 4, 16)

    cos, sin = glm5_next_nope_position_embeddings(hidden_states)

    assert cos.shape == sin.shape == (7, 0)
    assert cos.dtype == sin.dtype == hidden_states.dtype


@pytest.mark.parametrize("tp_size", [1, 2])
def test_glm5_next_kda_patch_uses_local_tp_heads(tp_size):
    modeling_glm5_next = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from tensor_cast.core.input_generator import glm5_next_kda_state_shape
    from tensor_cast.layers.glm5_next import glm5_next_kda_forward
    from tensor_cast.layers.parallel_linear import ColumnParallelLinear, RowParallelLinear
    from tensor_cast.parallel_group import ParallelGroup
    from tensor_cast.device import TEST_DEVICE
    from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
    from tensor_cast.performance_model.linear_attention import gated_delta_rule_ops
    from tensor_cast.runtime import Runtime

    cfg = Glm5NextTextConfig(hidden_size=32, linear_num_heads=4, linear_head_dim=8)
    group = ParallelGroup(0, [list(range(tp_size))], tp_size)
    with torch.device("meta"):
        module = modeling_glm5_next.Glm5NextTextLinearAttention(cfg, 0).to(torch.bfloat16)
    module.tensor_cast_tp_size = tp_size
    module.tensor_cast_tp_rank = 0
    for parent, name in [(module, n) for n in ("q_proj", "k_proj", "v_proj", "b_proj", "g_b_proj")] + [
        (module.forget_gate, "f_b_proj")
    ]:
        setattr(parent, name, ColumnParallelLinear(getattr(parent, name), group, group, head_num=4))
    module.o_proj = RowParallelLinear(module.o_proj, group, group, head_num=4)
    metadata = SimpleNamespace(
        query_lens=torch.tensor([1, 1]), seq_lens=torch.tensor([20, 100]), is_decode_values=[True, True]
    )
    state = torch.empty(glm5_next_kda_state_shape(cfg, torch.bfloat16, tp_size, 2), device="meta", dtype=torch.uint8)
    assert state.shape == (2, 1792 // tp_size)
    hidden = torch.empty((1, 2, 32), device="meta", dtype=torch.bfloat16)
    with torch.no_grad(), Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
        output = glm5_next_kda_forward(module, hidden, attention_meta=metadata, kv_cache_by_layers={0: state})
    assert output.shape == hidden.shape
    events = [e.op_invoke_info for e in runtime.event_list]
    core = next(info for info in events if info.func == torch.ops.tensor_cast.glm5_next_kda.default)
    assert core.args[0].shape == (1, 2, 4 // tp_size, 8)
    props = core.get_perf_properties()
    assert props.memory_read_bytes >= state.numel()
    assert props.memory_write_bytes >= state.numel()
    expected_mma, _, _ = gated_delta_rule_ops(1, 1, 4 // tp_size, 8, 8, use_recurrent=True)
    assert props.compute_ops[torch.float32].mma_ops == 2 * expected_mma
    assert any("all_reduce" in str(info.func) for info in events) == (tp_size > 1)


@pytest.mark.parametrize("decode", [False, True])
def test_kda_single_token_phase_and_state_io(decode):
    from tensor_cast.performance_model import OpInvokeInfo
    from tensor_cast.performance_model.linear_attention import gated_delta_rule_ops

    q = torch.empty((1, 1, 4, 8), device="meta", dtype=torch.bfloat16)
    state = torch.empty((1, 1792), device="meta", dtype=torch.uint8)
    args = (
        q,
        q,
        q,
        q,
        q.new_empty((1, 1, 4)),
        q,
        q.new_empty((3, 32, 4)),
        q.new_empty(4),
        q.new_empty(32),
        q.new_empty(8),
        state,
        torch.tensor([1]),
        torch.tensor([100 if decode else 1]),
        [decode],
        -5.0,
        1e-5,
    )
    op = torch.ops.tensor_cast.glm5_next_kda.default
    props = OpInvokeInfo(op, args, {}, op(*args)).get_perf_properties()
    expected_mma, expected_hidden_gp, expected_fp32_gp = gated_delta_rule_ops(1, 1, 4, 8, 8, use_recurrent=decode)
    assert props.compute_ops[torch.float32].mma_ops == expected_mma
    assert props.compute_ops[torch.bfloat16].gp_ops >= expected_hidden_gp
    assert props.compute_ops[torch.float32].gp_ops >= expected_fp32_gp
    assert props.memory_write_bytes == state.numel() + q.numel() * 2
    assert props.memory_readwrite_bytes > 0


def test_kda_mtp_decode_uses_chunk_path_for_multiple_query_tokens():
    """HF routes MTP decode (Q>1) through chunk KDA, not the recurrent kernel."""
    from tensor_cast.performance_model import OpInvokeInfo
    from tensor_cast.performance_model.linear_attention import gated_delta_rule_ops

    q = torch.empty((1, 2, 4, 8), device="meta", dtype=torch.bfloat16)
    state = torch.empty((1, 1792), device="meta", dtype=torch.uint8)
    args = (
        q,
        q,
        q,
        q,
        q.new_empty((1, 2, 4)),
        q,
        q.new_empty((3, 32, 4)),
        q.new_empty(4),
        q.new_empty(32),
        q.new_empty(8),
        state,
        torch.tensor([2]),
        torch.tensor([102]),
        [True],
        -5.0,
        1e-5,
    )
    op = torch.ops.tensor_cast.glm5_next_kda.default
    props = OpInvokeInfo(op, args, {}, op(*args)).get_perf_properties()

    expected_mma, _, _ = gated_delta_rule_ops(1, 2, 4, 8, 8, use_recurrent=False)
    assert props.compute_ops[torch.float32].mma_ops == expected_mma
    assert props.memory_read_bytes >= state.numel()
    assert props.memory_write_bytes >= state.numel()


def test_kpool_scores_pools_and_preserves_tail_width():
    from tensor_cast.performance_model import OpInvokeInfo

    q = torch.empty((1, 2, 4, 8), device="meta", dtype=torch.bfloat16)
    args = (
        q,
        q.new_empty((1, 2, 8)),
        q.new_empty((1, 2, 8)),
        q.new_empty((1, 2, 4)),
        q.new_empty((4, 8)),
        q.new_empty((8, 128, 17)),
        torch.tensor([1, 1]),
        torch.tensor([16, 33]),
        4,
        8,
        True,
    )
    op = torch.ops.tensor_cast.glm5_next_kpool_indexer.default
    out = op(*args)
    props = OpInvokeInfo(op, args, {}, out).get_perf_properties()
    assert out.shape == (1, 2, 11)
    assert out.dtype == torch.int32
    assert props.compute_ops[torch.float32].mma_ops == 2 * (4 + 9) * 4 * 8
    without_tail = op(*args[:-1], False)
    assert without_tail.shape[-1] == 8


def test_clamped_swiglu_compiles_as_one_activation():
    from tensor_cast.compilation import get_backend
    from tensor_cast.device import TEST_DEVICE
    from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
    from tensor_cast.runtime import Runtime

    def activation(gate, up):
        return torch.ops.tensor_cast.clamped_swiglu(gate, up, 7.0)

    compiled = torch.compile(activation, backend=get_backend(), fullgraph=True)
    x = torch.empty((2, 4), dtype=torch.bfloat16, device="meta")
    with torch.no_grad(), Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
        compiled(x, x.clone())
    names = [str(e.op_invoke_info.func) for e in runtime.event_list]
    assert names.count("tensor_cast.clamped_swiglu.default") == 1
    assert not any("clamp_" in name for name in names)
    activation_event = next(
        e.op_invoke_info
        for e in runtime.event_list
        if e.op_invoke_info.func == torch.ops.tensor_cast.clamped_swiglu.default
    )
    assert activation_event.get_perf_properties().compute_ops[torch.bfloat16].gp_ops == x.numel() * 11


@pytest.fixture
def tiny_glm5_next_path(tmp_path):
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextConfig,
        Glm5NextTextConfig,
        Glm5NextVisionConfig,
    )

    text = Glm5NextTextConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        linear_num_heads=4,
        linear_head_dim=8,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=8,
        index_kpool=4,
        vocab_size=64,
        rms_norm_eps=1e-5,
        pad_token_id=0,
        eos_token_id=1,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        mlp_layer_types=["dense"] * 3 + ["sparse"],
        indexer_types=["full"] * 4,
    )
    vision = Glm5NextVisionConfig(hidden_size=32, out_hidden_size=32, intermediate_size=64, depth=1, num_heads=4)
    cfg = Glm5NextConfig(text_config=text.to_dict(), vision_config=vision.to_dict())
    cfg.architectures = ["Glm5NextForConditionalGeneration"]
    cfg.image_token_id = 2
    cfg.video_token_id = 3
    cfg.save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize("tp,query_len,context", [(1, 1, 32), (2, 1, 32), (1, 7, 0)])
def test_glm5_next_model_compiled_semantic_coverage(tiny_glm5_next_path, tp, query_len, context):
    from tensor_cast.core.model_runner import ModelRunner
    from tensor_cast.core.input_generator import generate_inputs
    from tensor_cast.core.user_config import UserInputConfig
    from tensor_cast.device import TEST_DEVICE

    user = UserInputConfig(
        model_id=tiny_glm5_next_path,
        device=TEST_DEVICE.name,
        do_compile=True,
        world_size=tp,
        tp_size=tp,
        num_queries=2,
        query_len=query_len,
        context_length=context,
    )
    runner = ModelRunner(user)
    inputs = generate_inputs(runner.model, runner.request_info_default)
    assert inputs["kv_cache_by_layers"][0].dtype == torch.uint8
    assert inputs["kv_cache_by_layers"][0].shape[0] == 2
    assert inputs["indexer_cache_by_layers"][3].shape[-1] == 17
    result = runner.run_inference(generate_inputs_func=generate_inputs)
    event_names = {event["name"] for event in result.runtime_event_list}
    assert {
        "tensor_cast.glm5_next_kda.default",
        "tensor_cast.glm5_next_kpool_indexer.default",
        "tensor_cast.hc_pre_inv_rms.default",
        "tensor_cast.hc_pre_sinkhorn.default",
        "tensor_cast.hc_post.default",
        "tensor_cast.mhc_head.default",
        "tensor_cast.clamped_swiglu.default",
    } <= event_names
    assert "tensor_cast.linear_attention.default" not in event_names
    assert not any(name.startswith("aten.clamp") for name in event_names)
    assert "aten.mean.dim" not in event_names


def test_glm5_next_visual_tp_compile(tiny_glm5_next_path):
    from tensor_cast.core.input_generator import generate_inputs
    from tensor_cast.core.model_runner import ModelRunner
    from tensor_cast.core.user_config import UserInputConfig
    from tensor_cast.device import TEST_DEVICE

    user = UserInputConfig(
        model_id=tiny_glm5_next_path,
        device=TEST_DEVICE.name,
        do_compile=True,
        world_size=2,
        tp_size=2,
        vision_tp_size=2,
        num_queries=1,
        query_len=1,
        context_length=0,
        image_batch_size=1,
        image_height=28,
        image_width=28,
    )
    runner = ModelRunner(user)
    inputs = generate_inputs(runner.model, runner.request_info_default)
    # GLM5Next's official resize policy has a 16-token minimum.  A 28x28
    # image is therefore resized to 112x112, yielding an 8x8 patch grid.
    assert inputs["pixel_values"].shape[0] == 64
    result = runner.run_inference(generate_inputs_func=generate_inputs)
    event_names = {event["name"] for event in result.runtime_event_list}
    assert "tensor_cast.all_reduce.default" in event_names

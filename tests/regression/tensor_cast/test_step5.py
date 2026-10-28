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


"""Step5 geometry, numerical contracts and end-to-end simulation regression."""

import json
from pathlib import Path

import pytest
import torch

from tensor_cast.adapter.doctor import run_simulation_verification
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.performance_model.builtin_model.step5 import capped_causal_pairs
from tensor_cast.performance_model.builtin_model.step5 import visible_region_sum
from tensor_cast.transformers.builtin_model.step5 import Step5Config, Step5VLConfig, Step5ExpertMLP
from tests.helpers.model_assets import vendored_model_config_path


FIXTURE = Path(vendored_model_config_path("stepfun-ai/Step-5-Preview")) / "config.json"


def tiny_config():
    config = json.loads(FIXTURE.read_text(encoding="utf-8"))
    text = config["text_config"]
    text.update(
        hidden_size=32,
        intermediate_size=64,
        vocab_size=128,
        num_hidden_layers=4,
        num_nextn_predict_layers=0,
        num_attention_heads=4,
        num_sliding_attention_heads=4,
        num_attention_groups=2,
        head_dim=16,
        moe_num_experts=4,
        moe_top_k=2,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_layer_list=[3],
        moe_layers_enum="3",
        max_position_embeddings=1024,
        rope_scaling=None,
        attention_other_setting=None,
    )
    for key in ("layer_types", "rope_theta", "partial_rotary_factors", "swiglu_limits", "swiglu_limits_shared"):
        text[key] = text[key][:4]
    text["partial_rotary_factors"] = [1.0, 1.0, 1.0, 0.5]
    text["sparse_config"].update(
        proxy_dim=16, sparse_indexer_rope_dim=8, sparse_indexer_num_heads=4, num_provider_groups=2, topk=2
    )
    config["vision_config"].update(width=32, heads=4, layers=2, image_size=56, mlp_ratio=2.0)
    config["image_token_id"] = 127
    return config


def test_official_geometry_and_dense_boundaries():
    cfg = Step5VLConfig(**json.loads(FIXTURE.read_text(encoding="utf-8")))
    text = cfg.text_config
    assert text.model_type == "step5"
    assert text.num_hidden_layers == 92
    assert text.layer_types.count("full_attention") == 23
    assert text.layer_types.count("sliding_attention") == 69
    assert text.mlp_layer_types.count("sparse") == 88
    assert text.mlp_layer_types[:3] == ["dense"] * 3
    assert text.mlp_layer_types[-1] == "dense"
    assert (text.n_routed_experts, text.num_experts_per_tok, text.num_key_value_heads) == (352, 8, 4)
    assert text.sparse_config["topk"] * text.sparse_config["region_block_size"] == 4096
    assert cfg.vision_config.num_hidden_layers == 47


def test_config_round_trip_preserves_sparse_and_moe_structure():
    cfg = Step5VLConfig(**tiny_config())
    restored = Step5VLConfig.from_dict(cfg.to_dict())
    assert restored.text_config.sparse_config == cfg.text_config.sparse_config
    assert restored.text_config.mlp_layer_types == ["dense", "dense", "dense", "sparse"]


@pytest.mark.parametrize("seq,query,cap", [(8, 8, 512), (1025, 1, 512), (40, 17, 16), (16, 8, 16)])
def test_causal_pair_count_against_explicit_mask(seq, query, cap):
    expected = sum(min(position + 1, cap) for position in range(seq - query, seq))
    assert capped_causal_pairs(seq, query, cap) == expected


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 33])
def test_region_prefix_cost_counts_partial_regions(length):
    assert visible_region_sum(length, 8) == sum((p + 7) // 8 for p in range(1, length + 1))


def test_summary_cache_includes_pending_region_and_lengths():
    from tensor_cast.layers.step5 import Step5Attention

    cfg = Step5VLConfig(**tiny_config()).text_config
    with torch.device("meta"):
        attention = Step5Attention(cfg, 3)
    cache, num_blocks, block_size = attention.allocate_indexer_cache(4, 128, 2, 400)
    assert cache.dtype == torch.uint8
    assert cache.device.type == "meta"
    assert cache.numel() == (512 // 8) * 16 + 2 * (4 * 8 * 16 + 4)
    # The normalized span drives per-token accounting for arbitrary cache layouts.
    assert (num_blocks, block_size) == (4, 128)


def test_index_rope_has_independent_frequency_basis():
    from tensor_cast.layers.step5 import Step5Attention
    from transformers.models.step3p7.modeling_step3p7 import Step3p7RotaryEmbedding

    cfg = Step5VLConfig(**tiny_config()).text_config
    cfg.rope_parameters["full_attention"]["partial_rotary_factor"] = 1.0
    attention = Step5Attention(cfg, 3)
    x = torch.empty(1, 1, 32)
    positions = torch.tensor([[7]])
    main_cos, _ = Step3p7RotaryEmbedding(cfg)(x, positions, "full_attention")
    index_cos, _ = attention.index_rotary_emb(x, positions, "full_attention")
    assert index_cos.shape[-1] == 8
    assert not torch.allclose(index_cos, main_cos[..., :8])


def test_expert_clamps_after_silu_and_accumulates_in_fp32():
    from transformers.models.step3p7.modeling_step3p7 import Step3p7Experts

    cfg = Step5Config(**{k: v for k, v in tiny_config()["text_config"].items() if k != "model_type"})
    packed = Step3p7Experts(cfg, swiglu_limit=0.5)
    torch.manual_seed(5)
    with torch.no_grad():
        packed.gate_up_proj.normal_()
        packed.down_proj.normal_()
    expert = Step5ExpertMLP(packed, 0)
    x = torch.randn(2, 32)
    gate, up = torch.nn.functional.linear(x, packed.gate_up_proj[0]).chunk(2, -1)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(gate).clamp(max=0.5) * up.clamp(-0.5, 0.5), packed.down_proj[0]
    )
    torch.testing.assert_close(expert(x), expected)


@pytest.mark.parametrize("window,topk,expected_pairs", [(512, 0, 512), (0, 2, 16)])
def test_attention_cost_caps_window_or_selected_regions(window, topk, expected_pairs):
    q = torch.empty(1, 4, 16, device="meta", dtype=torch.bfloat16)
    cache = torch.empty(2, 20, 128, 2, 16, device="meta", dtype=torch.bfloat16)
    selected = torch.empty(1, 2, topk, device="meta", dtype=torch.int32) if topk else None
    args = (q, cache, selected, [1025], [1], window, topk, 8)
    op = torch.ops.tensor_cast.step5_attention.default
    result = op(*args)
    props = OpInvokeInfo(op, args, {}, result).get_perf_properties()
    assert result.shape == q.shape
    assert props.compute_ops[q.dtype].mma_ops == 4 * expected_pairs * 4 * 16
    assert props.memory_read_bytes < cache.numel() * cache.element_size()


@pytest.mark.parametrize("decode,tp", [(False, 1), (True, 1), (False, 2), (True, 2), (True, 4)])
def test_text_prefill_decode_operator_counts(tmp_path, decode, tp):
    cfg = tiny_config()
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    user = UserInputConfig(
        model_id=str(tmp_path),
        num_queries=1,
        query_len=1 if decode else 8,
        context_length=32 if decode else 0,
        decode=decode,
        world_size=tp,
        tp_size=tp,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
        disable_repetition=True,
    )
    report = run_simulation_verification(user)
    assert report.passed, report.simulation.get("traceback", report.to_dict())


@pytest.mark.parametrize("compiled", [False, True])
def test_image_prefill_and_compiled_text(tmp_path, compiled):
    (tmp_path / "config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    user = UserInputConfig(
        model_id=str(tmp_path),
        num_queries=1,
        query_len=8,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
        disable_repetition=True,
        do_compile=compiled,
        image_batch_size=None if compiled else 1,
        image_height=None if compiled else 56,
        image_width=None if compiled else 56,
    )
    report = run_simulation_verification(user)
    assert report.passed, report.simulation.get("traceback", report.to_dict())

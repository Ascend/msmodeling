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

"""Qwen3 MoE formal YAML Spec drives Runtime organization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensor_cast.layers.moe_layer import FusedMoETensorCast, _soft_critical_load, assign_experts
from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.domain import ExecutionPhase, ModelRunContext, ParallelContext, SourceKind
from tools.model_diagnostics.specification import (
    YamlModelDiagnosticsSpecLoader,
    create_builtin_operator_activation_registry,
    create_builtin_source_options_parsers,
)
from tools.model_diagnostics.specification.context_env import (
    _assign_experts,
    _simulated_ep_local_token_count,
    _soft_critical_load as _diagnostics_soft_critical_load,
    _validate_ep_token_args,
    build_theory_env,
)
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.theory_fragments import (
    load_builtin_theory_fragment_registry,
)


def _loader() -> YamlModelDiagnosticsSpecLoader:
    fragment_registry = load_builtin_theory_fragment_registry()
    return YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(
            fragment_registry=fragment_registry,
        ),
        fragment_registry=fragment_registry,
    )


@pytest.mark.parametrize(
    ("total", "parts", "num_global_experts", "num_local_experts"),
    [
        (0, 32, 384, 12),
        (48, 32, 384, 12),
        (192, 32, 384, 12),
        (7, 3, 5, 2),
        (10, 3, 6, 2),
    ],
)
def test_soft_critical_load_matches_tensor_cast(
    total: int,
    parts: int,
    num_global_experts: int,
    num_local_experts: int,
) -> None:
    """Keep the diagnostics mirror aligned with TensorCast MoE split modeling."""
    assert _diagnostics_soft_critical_load(total, parts, num_global_experts, num_local_experts) == _soft_critical_load(
        total, parts, num_global_experts, num_local_experts
    )


def _context(*, effective_layers: int = 1, ep: int = 1) -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen/Qwen3-30B-A3B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        parallel=ParallelContext(
            tensor_parallel_size=1,
            data_parallel_size=ep,
            expert_parallel_size=ep,
            moe_data_parallel_size=1,
        ),
        model_config={
            "model_type": "qwen3_moe",
            "features": ["compiled"],
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_hidden_layers": 48,
            "effective_num_hidden_layers": effective_layers,
            "vocab_size": 151936,
            "torch_dtype": "float16",
        },
        quantization_config={},
    )


def test_qwen3_moe_yaml_declares_runtime_stage_boundaries() -> None:
    loader = _loader()
    spec = loader.materialize(loader.load("qwen3_moe_v1"), _context())

    assert spec.spec_id == "qwen3_moe_v1"
    assert [region.region_id for region in spec.regions] == [
        "input",
        "language",
        "output",
        "mtp",
    ]
    language = next(region for region in spec.regions if region.region_id == "language")
    assert language.layer_layout == ("moe",)
    moe_stages = language.layer_specs["moe"].stages
    assert [stage.stage_id for stage in moe_stages] == [
        "attention_qkv",
        "attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
    ]
    assert moe_stages[2].source_options[SourceKind.RUNTIME].boundary_operators == (
        "add_rms_norm2",
        "add_rms_norm_dynamic_quant2_symmetric",
        "add_rms_norm_quant2",
        "rms_norm",
        "rms_norm_dynamic_quant_symmetric",
        "rms_norm_quant",
    )
    assert moe_stages[1].source_options[SourceKind.RUNTIME].boundary_operators == ("attention",)
    assert moe_stages[1].source_options[SourceKind.RUNTIME].ignored_operators == (
        "slice",
        "select",
        "dynamic_quantize_symmetric",
        "quantize",
        "view",
        "index",
        "reshape_and_cache",
        "split_with_sizes",
        "alias",
        "copy_",
        "apply_rope",
        "all_reduce",
    )
    assert moe_stages[3].source_options[SourceKind.RUNTIME].boundary_operators == ("init_routing_v2",)
    assert moe_stages[4].source_options[SourceKind.RUNTIME].boundary_operators == (
        "grouped_matmul_swiglu",
        "grouped_matmul_quant_swiglu",
        "grouped_matmul_quant_int4_swiglu",
        "grouped_matmul_fp8_swiglu",
        "grouped_matmul_mxfp4_swiglu",
    )
    assert moe_stages[5].source_options[SourceKind.RUNTIME].boundary_operators == ("unpermute_tokens",)
    output = next(region for region in spec.regions if region.region_id == "output")
    assert "all_gather" in output.stages[0].source_options[SourceKind.RUNTIME].ignored_operators
    mtp = next(region for region in spec.regions if region.region_id == "mtp")
    # MTP=0 => region present but inactive (no layers / emptied specs).
    assert mtp.layer_layout == ()


def test_qwen3_moe_yaml_layout_follows_effective_layer_count() -> None:
    loader = _loader()
    loaded = loader.load("qwen3_moe_v1")

    def _language_layout(layers: int):
        return next(
            region.layer_layout
            for region in loader.materialize(loaded, _context(effective_layers=layers)).regions
            if region.region_id == "language"
        )

    assert _language_layout(1) == ("moe",)
    assert _language_layout(3) == ("moe", "moe", "moe")


def test_qwen3_moe_theory_env_binds_category2_symbols() -> None:
    env = build_theory_env(_context(ep=1))
    assert env["E"] == 128
    assert env["Ktop"] == 8
    assert env["Fmoe"] == 768
    assert env["MTPt"] == 1
    assert env["MDP"] == 1
    assert env["Fe"] == 768
    assert env["T"] == 2
    assert env["Te"] == 16  # T*Ktop when EP=1


def test_qwen3_moe_theory_env_binds_simulated_te_when_ep_gt1() -> None:
    # Same contract as tensor_cast get_split_sizes (no external shared experts):
    # N=T*Ktop=16 over EP=2 uses a soft critical load of 14 on rank0;
    # Te = share*EP = 28, rather than T*Ktop/EP (=8).
    env = build_theory_env(_context(ep=2))
    assert env["EP"] == 2
    assert env["T"] == 2
    assert env["Ktop"] == 8
    assert env["E"] == 128
    assert env["Te"] == 28
    # Te contains the contributions of both EP peers.  The rank-local split
    # before concatenation has 14 non-empty expert buckets, so exactly 14
    # ModuleList experts execute their projections.
    assert env["Eactive"] == 14
    assert env["Te"] != env["T"] * env["Ktop"] // env["EP"]


def test_qwen3_moe_theory_env_binds_fe_with_fixed_moe_tp() -> None:
    # MoE tensor parallel is fixed at 1 by this module: Fe == Fmoe.
    context = ModelRunContext(
        model_name="Qwen/Qwen3-30B-A3B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        parallel=ParallelContext(),
        model_config=dict(_context().model_config),
        quantization_config={},
    )
    env = build_theory_env(context)
    assert env["MTPt"] == 1
    assert env["Fmoe"] == 768
    assert env["Fe"] == 768  # Fmoe / MTPt with MTPt fixed at 1
    assert env["Te"] == 16  # EP=1 still T*Ktop


def test_qwen3_moe_theory_env_binds_te_with_external_shared_experts() -> None:
    # Run Profile has no local-rank field, so test alternate analytic ranks
    # directly instead of injecting diagnostics-only model_config keys.
    shares = tuple(
        _simulated_ep_local_token_count(
            routed_tokens=16,
            top_k=8,
            num_global_experts=128,
            ep=4,
            ep_rank=rank,
            num_external_shared_experts=1,
        )
        for rank in range(4)
    )
    assert shares == (8, 52, 8, 4)


def test_qwen3_moe_ep_token_distribution_allows_empty_ranks() -> None:
    """A valid EP split may assign no routed tokens to later expert ranks."""

    values = tuple(
        _simulated_ep_local_token_count(
            routed_tokens=1,
            top_k=1,
            num_global_experts=8,
            ep=4,
            ep_rank=rank,
        )
        for rank in range(4)
    )

    assert values == (4, 0, 0, 0)


@pytest.mark.parametrize(
    "num_experts,world_size,rank",
    [
        (8, 4, 0),
        (10, 4, 0),
        (10, 4, 1),
        (10, 4, 2),
        (3, 4, 3),
    ],
)
def test_theory_expert_ownership_matches_tensor_cast_contract(
    num_experts: int,
    world_size: int,
    rank: int,
) -> None:
    """Detect drift while keeping the Theory implementation independent."""

    assert _assign_experts(num_experts, world_size, rank) == assign_experts(
        num_experts,
        world_size,
        rank,
    )


@pytest.mark.parametrize(
    "routed_tokens,top_k,num_experts,ep,ep_rank,num_external",
    [
        (16, 8, 128, 2, 0, 0),
        (1, 1, 8, 4, 0, 0),
        (1, 1, 8, 4, 3, 0),
        (16, 8, 128, 4, 0, 1),
        (16, 8, 128, 4, 1, 1),
    ],
)
def test_theory_ep_token_count_matches_tensor_cast_split_contract(
    routed_tokens: int,
    top_k: int,
    num_experts: int,
    ep: int,
    ep_rank: int,
    num_external: int,
) -> None:
    """Compare independent Theory math with the Runtime split implementation."""

    routing_rank = ep_rank - num_external
    if routing_rank >= 0:
        expert_start, num_local_experts = assign_experts(
            num_experts,
            ep - num_external,
            routing_rank,
        )
    else:
        expert_start, num_local_experts = 0, 0
    runtime_layer = SimpleNamespace(
        num_global_experts=num_experts,
        num_external_shared_experts=num_external,
        expert_idx_start=expert_start,
        num_local_experts=num_local_experts,
        ep_group=SimpleNamespace(world_size=ep, rank_in_group=ep_rank),
    )

    runtime_input_splits, _, _, _ = FusedMoETensorCast.get_split_sizes(
        runtime_layer,
        routed_tokens,
        top_k,
    )
    runtime_local_tokens = runtime_input_splits[ep_rank] * ep

    assert (
        _simulated_ep_local_token_count(
            routed_tokens=routed_tokens,
            top_k=top_k,
            num_global_experts=num_experts,
            ep=ep,
            ep_rank=ep_rank,
            num_external_shared_experts=num_external,
        )
        == runtime_local_tokens
    )

    if routing_rank >= 0:
        runtime_expert_splits = FusedMoETensorCast.get_split_sizes(
            runtime_layer,
            routed_tokens,
            top_k,
        )[2]
        runtime_active_experts = sum(
            split_size > 0 for split_size in runtime_expert_splits[expert_start : expert_start + num_local_experts]
        )
        assert runtime_active_experts == min(num_local_experts, runtime_local_tokens // ep)


@pytest.mark.parametrize(
    "overrides,expected_message",
    [
        ({"routed_tokens": 0}, "routed token count"),
        ({"top_k": 0}, "num_experts_per_tok"),
        ({"num_global_experts": 0}, "global expert count"),
        ({"num_external_shared_experts": True}, "must be an integer"),
        ({"num_external_shared_experts": -1}, "must be non-negative"),
        ({"num_external_shared_experts": 4}, "smaller than expert_parallel_size"),
        ({"ep_rank": 4}, "analytic EP rank"),
    ],
)
def test_ep_token_args_reject_invalid_values(
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    arguments: dict[str, object] = {
        "routed_tokens": 16,
        "top_k": 8,
        "num_global_experts": 128,
        "ep": 4,
        "ep_rank": 0,
        "num_external_shared_experts": 0,
    }
    arguments.update(overrides)

    with pytest.raises(SpecificationLoadError, match=expected_message):
        _validate_ep_token_args(**arguments)


def test_qwen3_moe_theory_env_derives_external_from_enable_flag() -> None:
    # EP=4, Ktop=8 → top_k+1 > EP ⇒ Next=1 (same as shard_model_by_ep).
    # The first routing rank uses the soft critical load.
    context = ModelRunContext(
        model_name="Qwen/Qwen3-30B-A3B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        parallel=ParallelContext(
            data_parallel_size=4,
            expert_parallel_size=4,
            moe_data_parallel_size=1,
        ),
        model_config={
            **dict(_context().model_config),
            "enable_external_shared_experts": True,
        },
        quantization_config={},
    )
    env = build_theory_env(context)
    assert env["EP"] == 4
    assert env["Te"] == 52


def test_qwen3_moe_theory_env_binds_te_with_redundant_experts() -> None:
    # Redundant experts are included in the soft critical-rank split.
    context = ModelRunContext(
        model_name="Qwen/Qwen3-30B-A3B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        parallel=ParallelContext(
            data_parallel_size=2,
            expert_parallel_size=2,
            moe_data_parallel_size=1,
        ),
        model_config={
            **dict(_context().model_config),
            "num_experts": 3,
            "num_experts_per_tok": 5,
            "enable_redundant_experts": True,
        },
        quantization_config={},
    )
    env = build_theory_env(context)
    assert env["Te"] == 10


def test_qwen3_moe_theory_env_binds_mdp_symbol() -> None:
    context = ModelRunContext(
        model_name="Qwen/Qwen3-30B-A3B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        parallel=ParallelContext(
            data_parallel_size=2,
            moe_data_parallel_size=2,
        ),
        model_config=dict(_context().model_config),
        quantization_config={},
    )
    env = build_theory_env(context)
    assert env["MDP"] == 2
    assert env["DP"] == 2
    assert env["T"] == 2  # ceil(B/DP)*Q
    assert env["Tmoe"] == 4  # T*MDP
    assert env["Te"] == 32  # Tmoe*Ktop = 4*8


@pytest.mark.parametrize(
    ("parallel", "batch_size", "query_length", "expected_tmoe"),
    (
        (
            ParallelContext(
                tensor_parallel_size=2,
                data_parallel_size=1,
                expert_parallel_size=2,
                moe_data_parallel_size=1,
            ),
            1,
            3,
            2,
        ),
        (
            ParallelContext(
                tensor_parallel_size=2,
                data_parallel_size=2,
                expert_parallel_size=2,
                moe_data_parallel_size=2,
            ),
            4,
            1,
            2,
        ),
    ),
)
def test_qwen3_moe_theory_env_mirrors_runtime_dp_transform(
    parallel: ParallelContext,
    batch_size: int,
    query_length: int,
    expected_tmoe: int,
) -> None:
    context = ModelRunContext(
        model_name="Qwen/Qwen3-30B-A3B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=batch_size,
        query_length=query_length,
        context_length=None,
        parallel=parallel,
        model_config=dict(_context().model_config),
        quantization_config={},
    )

    env = build_theory_env(context)

    assert env["Tmoe"] == expected_tmoe
    assert env["Te"] == 28


def test_qwen3_moe_dispatch_and_combine_ignore_all_to_all() -> None:
    loader = _loader()
    spec = loader.materialize(loader.load("qwen3_moe_v1"), _context())
    language = next(region for region in spec.regions if region.region_id == "language")
    stages = {stage.stage_id: stage for stage in language.layer_specs["moe"].stages}
    for stage_id in ("moe_dispatch", "moe_experts", "moe_combine"):
        ignored = stages[stage_id].source_options[SourceKind.RUNTIME].ignored_operators
        assert "all_to_all" in ignored
    assert "constant_pad_nd" in stages["moe_gate"].source_options[SourceKind.RUNTIME].ignored_operators
    assert "all_gather" in stages["moe_combine"].source_options[SourceKind.RUNTIME].ignored_operators


def test_qwen3_moe_mtp_region_activates_with_num_mtp_tokens() -> None:
    loader = _loader()
    context = ModelRunContext(
        model_name="Qwen/Qwen3-30B-A3B",
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=3,
        context_length=128,
        parallel=ParallelContext(),
        model_config={
            "model_type": "qwen3_moe",
            "features": ["compiled"],
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_hidden_layers": 48,
            "effective_num_hidden_layers": 1,
            "vocab_size": 151936,
            "torch_dtype": "float16",
            "num_mtp_tokens": 2,
        },
        quantization_config={},
    )
    spec = loader.materialize(loader.load("qwen3_moe_v1"), context)
    mtp = next(region for region in spec.regions if region.region_id == "mtp")
    assert mtp.layer_layout == ("moe_mtp", "moe_mtp")
    predictor_stages = [stage.stage_id for stage in mtp.layer_specs["moe_mtp"].stages]
    assert "moe_gate" in predictor_stages
    assert "moe_experts" in predictor_stages

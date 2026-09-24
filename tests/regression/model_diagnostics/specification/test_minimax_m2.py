# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""MiniMax-M2 formal YAML Spec and Theory environment tests."""

from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.domain import ExecutionPhase, ModelRunContext, ParallelContext
from tools.model_diagnostics.specification import (
    YamlModelDiagnosticsSpecLoader,
    create_builtin_operator_activation_registry,
    create_builtin_source_options_parsers,
)
from tools.model_diagnostics.specification.context_env import build_theory_env
from tools.model_diagnostics.specification.theory_fragments import (
    load_builtin_theory_fragment_registry,
)


def _loader() -> YamlModelDiagnosticsSpecLoader:
    fragments = load_builtin_theory_fragment_registry()
    return YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(
            fragment_registry=fragments,
        ),
        fragment_registry=fragments,
    )


def _context(
    *,
    phase: ExecutionPhase = ExecutionPhase.PREFILL,
    query_length: int = 1,
    num_mtp_tokens: int = 0,
) -> ModelRunContext:
    return ModelRunContext(
        model_name="MiniMaxAI/MiniMax-M2.7",
        entrypoint="text_generate",
        phase=phase,
        batch_size=24,
        query_length=query_length,
        context_length=3900,
        parallel=ParallelContext(
            tensor_parallel_size=8,
            data_parallel_size=2,
            expert_parallel_size=16,
            moe_data_parallel_size=1,
        ),
        model_config={
            "model_type": "minimax_m2",
            "hidden_size": 3072,
            # MiniMax-M2 uses this field as the routed-expert width.
            "intermediate_size": 1536,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 62,
            "effective_num_hidden_layers": 1,
            "vocab_size": 200064,
            "torch_dtype": "float16",
            "num_mtp_tokens": num_mtp_tokens,
        },
        quantization_config={
            "enabled": True,
            "action": "W8A8_STATIC",
            "linear_input_dtype": "int8",
        },
    )


def test_minimax_m2_theory_env_uses_routed_expert_width_alias() -> None:
    env = build_theory_env(_context())

    assert env["B"] == 12
    assert env["T"] == 12
    assert env["Lh"] == 6
    assert env["Lkv"] == 1
    assert env["E"] == 256
    assert env["Ktop"] == 8
    assert env["Fmoe"] == 1536
    assert env["Fe"] == 1536
    assert env["Tmoe"] == 2
    # Master models the first routing rank with the runtime-aligned soft
    # critical-load estimate: 96 local assignments * EP=2.
    assert env["Te"] == 192


def test_minimax_m2_yaml_materializes_gqa_and_moe_stages() -> None:
    loader = _loader()
    spec = loader.materialize(loader.load("minimax_m2_v1"), _context())

    assert spec.spec_id == "minimax_m2_v1"
    language = next(region for region in spec.regions if region.region_id == "language")
    assert language.layer_layout == ("moe",)
    assert [stage.stage_id for stage in language.layer_specs["moe"].stages] == [
        "attention_qkv",
        "q_norm",
        "k_norm",
        "attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
    ]

    mtp = next(region for region in spec.regions if region.region_id == "mtp")
    assert mtp.layer_layout == ()


def test_minimax_m2_mtp_region_uses_m2_decoder_for_each_proposal() -> None:
    loader = _loader()
    spec = loader.materialize(
        loader.load("minimax_m2_v1"),
        _context(
            phase=ExecutionPhase.DECODE,
            query_length=4,
            num_mtp_tokens=3,
        ),
    )

    mtp = next(region for region in spec.regions if region.region_id == "mtp")
    assert mtp.layer_layout == (
        "minimax_m2_mtp",
        "minimax_m2_mtp",
        "minimax_m2_mtp",
    )
    assert [stage.stage_id for stage in mtp.stages] == [
        "target_selection",
        "target_lm_head",
        "verification_sampler",
        "mtp_output",
    ]
    stage_ids = [stage.stage_id for stage in mtp.layer_specs["minimax_m2_mtp"].stages]
    assert stage_ids == [
        "input_shift",
        "embedding",
        "input_fusion",
        "attention_qkv",
        "q_norm",
        "k_norm",
        "attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "proposal_selection",
        "proposal_lm_head",
        "proposal_sampler",
    ]

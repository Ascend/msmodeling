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

"""Theory symbols and Spec layout for the Qwen3.5 hybrid category."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.domain import ExecutionPhase, ModelRunContext, ParallelContext, SourceKind
from tools.model_diagnostics.specification.builtin_activation import create_builtin_operator_activation_registry
from tools.model_diagnostics.specification.context_env import build_theory_env
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.loader import YamlModelDiagnosticsSpecLoader
from tools.model_diagnostics.specification.source_options import create_builtin_source_options_parsers
from tools.model_diagnostics.specification.theory_fragments import load_builtin_theory_fragment_registry

_CONFIG: dict[str, object] = {
    "model_type": "qwen3_5_text",
    "num_hidden_layers": 64,
    "effective_num_hidden_layers": 6,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "hidden_size": 10240,
    "intermediate_size": 12288,
    "vocab_size": 152064,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
    ],
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "torch_dtype": "bfloat16",
}

_NEXT_CONFIG: dict[str, object] = {
    **_CONFIG,
    "model_type": "qwen3_next",
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "hidden_size": 2048,
    "intermediate_size": 2048,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
}


def _context(
    *,
    parallel: ParallelContext | None = None,
    config: dict[str, object] | None = None,
) -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen/Qwen3.5-27B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        parallel=parallel or ParallelContext(),
        model_config=dict(config or _CONFIG),
        quantization_config={},
    )


def _loader() -> YamlModelDiagnosticsSpecLoader:
    fragments = load_builtin_theory_fragment_registry()
    return YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(fragment_registry=fragments),
        fragment_registry=fragments,
    )


def test_qwen3_5_theory_env_binds_linear_head_symbols() -> None:
    env = build_theory_env(_context())

    assert (env["Lk_lin"], env["Lv_lin"]) == (16, 48)
    assert (env["Klin"], env["Vlin"]) == (128, 128)
    assert env["LCONV"] == 4


def test_qwen3_5_theory_env_shards_linear_heads_by_tp() -> None:
    env = build_theory_env(_context(parallel=ParallelContext(tensor_parallel_size=2)))

    assert (env["Lk_lin"], env["Lv_lin"]) == (8, 24)


def test_qwen3_5_theory_env_rejects_tp_not_dividing_linear_heads() -> None:
    with pytest.raises(SpecificationLoadError, match="tensor_parallel_size must divide"):
        build_theory_env(_context(parallel=ParallelContext(tensor_parallel_size=3)))


def test_qwen3_5_theory_env_rejects_mismatched_linear_head_dims() -> None:
    with pytest.raises(SpecificationLoadError, match="linear_key_head_dim must equal"):
        build_theory_env(_context(config={**_CONFIG, "linear_value_head_dim": 64}))


def test_qwen3_5_dense_spec_materializes_hybrid_sequence() -> None:
    loader = _loader()
    spec = loader.materialize(loader.load("qwen3_5_next_v1"), _context())
    language = next(region for region in spec.regions if region.region_id == "language")

    assert language.layer_layout == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
    )
    full_stages = {stage.stage_id for stage in language.layer_specs["full_attention"].stages}
    assert full_stages == {"attention_qkv", "attention", "dense_ffn"}
    linear_stages = {stage.stage_id for stage in language.layer_specs["linear_attention"].stages}
    assert linear_stages == {
        "linear_projection",
        "linear_delta_rule",
        "linear_output",
        "dense_ffn",
    }


def test_qwen3_5_moe_spec_materializes_hybrid_sequence() -> None:
    loader = _loader()
    context = _context(config={**_CONFIG, "model_type": "qwen3_5_moe_text"})
    spec = loader.materialize(loader.load("qwen3_5_next_v1"), context)
    language = next(region for region in spec.regions if region.region_id == "language")

    assert language.layer_layout == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
    )
    full_stages = {stage.stage_id for stage in language.layer_specs["full_attention"].stages}
    assert full_stages == {
        "attention_qkv",
        "attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "shared_ffn",
    }
    linear_stages = {stage.stage_id for stage in language.layer_specs["linear_attention"].stages}
    assert linear_stages == {
        "linear_projection",
        "linear_delta_rule",
        "linear_output",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "shared_ffn",
    }


def test_qwen3_next_spec_materializes_decomposed_linear_attention() -> None:
    loader = _loader()
    spec = loader.materialize(loader.load("qwen3_5_next_v1"), _context(config=_NEXT_CONFIG))
    language = next(region for region in spec.regions if region.region_id == "language")

    assert language.layer_layout == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
    )
    full_stages = {stage.stage_id for stage in language.layer_specs["full_attention"].stages}
    assert full_stages == {
        "attention_qkv",
        "attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "shared_ffn",
    }
    linear_stages = {stage.stage_id for stage in language.layer_specs["linear_attention"].stages}
    assert linear_stages == {
        "linear_projection",
        "linear_delta_rule",
        "linear_output",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "shared_ffn",
    }
    linear_delta_rule = next(
        stage for stage in language.layer_specs["linear_attention"].stages if stage.stage_id == "linear_delta_rule"
    )
    operators = linear_delta_rule.source_options[SourceKind.THEORY].operators
    assert [operator.operator_name for operator in operators] == ["linear_attn_chunk_gated_delta_rule"]


def test_qwen3_5_theory_env_doubles_query_head_dim() -> None:
    env = build_theory_env(_context())

    assert env["QH"] == env["Dh"] * 2


def test_qwen3_5_theory_env_keeps_query_head_dim_for_other_families() -> None:
    env = build_theory_env(_context(config={**_CONFIG, "model_type": "qwen3", "head_dim": 128}))

    assert env["QH"] == env["Dh"] == 128


def test_qwen3_5_moe_tp_keeps_full_expert_domain() -> None:
    config = {
        **_CONFIG,
        "model_type": "qwen3_5_moe_text",
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
    }
    env = build_theory_env(_context(config=config, parallel=ParallelContext(tensor_parallel_size=2)))

    # MoE tensor parallel is fixed at 1: dispatch and experts stay on the full
    # routed domain even when TP>1 (verified against the compiled graph on
    # transformers 5.13.x).
    assert env["Tmoe"] == env["T"] == 2
    assert env["Fe"] == 1024
    assert env["Te"] == 20


def test_qwen3_5_moe_tp1_keeps_full_expert_domain() -> None:
    config = {
        **_CONFIG,
        "model_type": "qwen3_5_moe_text",
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
    }
    env = build_theory_env(_context(config=config))

    assert env["Tmoe"] == env["T"] == 2
    assert env["Fe"] == 1024
    assert env["Te"] == 20


def test_qwen3_5_mtp_compose_predictors_follow_trimmed_layer_types() -> None:
    loader = _loader()
    loaded = loader.load("qwen3_5_next_v1")
    mtp = next(region for region in loaded.spec.regions if region.region_id == "mtp")
    assert set(mtp.layer_specs) == {"full_attention", "linear_attention"}
    full_ids = [stage.stage_id for stage in mtp.layer_specs["full_attention"].stages]
    linear_ids = [stage.stage_id for stage in mtp.layer_specs["linear_attention"].stages]
    assert full_ids[:3] == ["input_shift", "embedding", "input_fusion"]
    assert "attention_qkv" in full_ids
    assert "linear_projection" in linear_ids

    context = replace(
        _context(),
        phase=ExecutionPhase.DECODE,
        query_length=3,
        model_config={
            **_CONFIG,
            "num_mtp_tokens": 2,
        },
    )
    spec = loader.materialize(loaded, context)
    materialized = next(region for region in spec.regions if region.region_id == "mtp")
    assert materialized.layer_layout == ("linear_attention", "linear_attention")

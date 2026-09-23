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

"""Qwen3 Dense formal YAML Spec drives Runtime organization."""

from __future__ import annotations

from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.domain import ExecutionPhase, ModelRunContext, ParallelContext, SourceKind
from tools.model_diagnostics.specification import (
    YamlModelDiagnosticsSpecLoader,
    create_builtin_operator_activation_registry,
    create_builtin_source_options_parsers,
)
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


def _context(*, effective_layers: int = 1) -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen/Qwen3-8B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        parallel=ParallelContext(tensor_parallel_size=1),
        model_config={
            "model_type": "qwen3",
            "features": ["dense", "compiled"],
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 36,
            "effective_num_hidden_layers": effective_layers,
            "vocab_size": 151936,
            "torch_dtype": "float16",
        },
        quantization_config={},
    )


def test_qwen3_yaml_declares_runtime_stage_boundaries() -> None:
    loader = _loader()
    spec = loader.materialize(loader.load("qwen3_dense_v1"), _context())

    assert spec.spec_id == "qwen3_dense_v1"
    assert [region.region_id for region in spec.regions] == [
        "input",
        "language",
        "output",
        "mtp",
    ]
    language = next(region for region in spec.regions if region.region_id == "language")
    assert language.layer_layout == ("dense",)
    dense_stages = language.layer_specs["dense"].stages
    assert [stage.stage_id for stage in dense_stages] == [
        "attention_qkv",
        "attention",
        "dense_ffn",
    ]
    assert dense_stages[0].source_options[SourceKind.RUNTIME].boundary_operators == (
        "rms_norm",
        "rms_norm_dynamic_quant_symmetric",
        "rms_norm_quant",
    )
    assert dense_stages[1].source_options[SourceKind.RUNTIME].boundary_operators == ("attention",)
    assert dense_stages[2].source_options[SourceKind.RUNTIME].boundary_operators == (
        "add_rms_norm2",
        "add_rms_norm_dynamic_quant2_symmetric",
        "add_rms_norm_quant2",
        "rms_norm",
        "rms_norm_dynamic_quant_symmetric",
        "rms_norm_quant",
    )
    output = next(region for region in spec.regions if region.region_id == "output")
    assert [stage.stage_id for stage in output.stages] == ["lm_head"]
    lm_head = output.stages[0]
    assert lm_head.source_options[SourceKind.RUNTIME].boundary_operators == ("rms_norm",)
    assert "slice" in lm_head.source_options[SourceKind.RUNTIME].ignored_operators
    assert [operator.operator_name for operator in lm_head.source_options[SourceKind.THEORY].operators] == [
        "lm_head_select",
        "lm_head",
    ]
    assert lm_head.source_options[SourceKind.THEORY].operators[0].activation == "lm_head_token_selection"
    assert lm_head.source_options[SourceKind.THEORY].operators[1].activation == "non_mtp_lm_head"


def test_qwen3_yaml_layout_follows_effective_layer_count() -> None:
    loader = _loader()
    loaded = loader.load("qwen3_dense_v1")

    def _language_layout(layers: int):
        return next(
            region.layer_layout
            for region in loader.materialize(loaded, _context(effective_layers=layers)).regions
            if region.region_id == "language"
        )

    assert _language_layout(1) == ("dense",)
    assert _language_layout(3) == ("dense", "dense", "dense")

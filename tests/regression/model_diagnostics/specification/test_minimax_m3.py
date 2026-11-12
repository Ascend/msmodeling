# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Official MiniMax-M3.1 dimensions, layer layouts and sparse Theory checks."""

from dataclasses import replace

import pytest

from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.domain import ExecutionPhase, ModelRunContext, ParallelContext
from tools.model_diagnostics.specification import (
    YamlModelDiagnosticsSpecLoader,
    create_builtin_operator_activation_registry,
    create_builtin_source_options_parsers,
)
from tools.model_diagnostics.specification.context_env import build_theory_env
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.theory_fragments import load_builtin_theory_fragment_registry


def _context():
    return ModelRunContext(
        model_name="MiniMaxAI/MiniMax-M3.1-Flash",
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=8,
        query_length=1,
        context_length=131072,
        parallel=ParallelContext(
            tensor_parallel_size=4, data_parallel_size=8, expert_parallel_size=32, moe_data_parallel_size=1
        ),
        model_config={
            "model_type": "minimax_m3_vl_text",
            "hidden_size": 6144,
            "intermediate_size": 3072,
            "dense_intermediate_size": 12288,
            "shared_intermediate_size": 3072,
            "moe_intermediate_size": 3072,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "num_hidden_layers": 60,
            "effective_num_hidden_layers": 60,
            "vocab_size": 200064,
            "torch_dtype": "bfloat16",
            "index_n_heads": 4,
            "index_head_dim": 128,
            "index_topk_blocks": 16,
            "layer_types": ["minimax_m3_sparse"] * 60,
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 57,
        },
        quantization_config={"enabled": True, "action": "W8A8_DYNAMIC", "linear_input_dtype": "int8"},
    )


def _loader():
    fragments = load_builtin_theory_fragment_registry()
    return YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(fragment_registry=fragments),
        fragment_registry=fragments,
    )


def test_official_m31_dimensions_and_hybrid_layer_layout():
    context = _context()
    env = build_theory_env(context)
    assert (env["B"], env["T"], env["Lh"], env["Lkv"]) == (1, 1, 16, 1)
    assert (env["M3Ih"], env["M3Id"], env["M3Topk"]) == (4, 128, 16)
    assert env["M3LocalIh"] == 1
    assert (env["M3Dense"], env["M3Shared"], env["Fe"]) == (3072, 3072, 3072)
    assert env["int32"] == "int32"
    loader = _loader()
    spec = loader.materialize(loader.load("minimax_m3_v1"), context)
    language = next(r for r in spec.regions if r.region_id == "language")
    assert language.layer_layout == ("dense",) * 3 + ("sparse",) * 57
    assert "indexer" in [s.stage_id for s in language.layer_specs["dense"].stages]
    assert "moe_gate" in [s.stage_id for s in language.layer_specs["sparse"].stages]


def test_m31_prefill_cache_effective_request_dimensions():
    context = replace(
        _context(),
        phase=ExecutionPhase.PREFILL,
        batch_size=2,
        query_length=13108,
        context_length=117964,
        parallel=ParallelContext(
            tensor_parallel_size=4, data_parallel_size=4, pipeline_parallel_size=4, expert_parallel_size=16
        ),
    )
    env = build_theory_env(context)
    assert (env["B"], env["Q"], env["C"], env["S"], env["T"]) == (1, 13108, 117964, 131072, 13108)


def test_m31_rejects_unmodeled_attention_variant():
    context = _context()
    context.model_config["layer_types"][0] = "full_attention"
    with pytest.raises(SpecificationLoadError, match="all-sparse"):
        build_theory_env(context)


def test_m31_capture_metadata_uses_source_fields(tmp_path):
    from types import SimpleNamespace
    from tools.model_diagnostics.sources.runtime_capture import _run_context_after_model_load
    from tools.model_diagnostics.specification.run_profile import load_diagnostics_run_profile

    config = dict(_context().model_config)
    config["num_local_experts"] = config.pop("num_experts")
    config.pop("moe_intermediate_size")
    text = SimpleNamespace(**config)
    vision = SimpleNamespace(
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_channels=3,
        projection_dim=6144,
    )
    root = SimpleNamespace(model_type="minimax_m3_vl", vision_config=vision, get_text_config=lambda: text)
    model = SimpleNamespace(hf_config=root, text_config=text)
    runner = SimpleNamespace(model=model)
    path = tmp_path / "profile.yaml"
    path.write_text(
        "model_name: MiniMaxAI/MiniMax-M3.1-Flash\nphase: decode\n"
        "batch_size: 8\nquery_length: 1\ncontext_length: 131072\n",
        encoding="utf-8",
    )
    context = _run_context_after_model_load(load_diagnostics_run_profile(path), runner)
    assert context.model_config["num_experts"] == 128
    assert context.model_config["moe_intermediate_size"] == 3072
    assert context.model_config["vision_in_channels"] == 3
    assert context.model_config["vision_out_hidden_size"] == 6144
    assert context.model_config["mlp_layer_types"] == ["dense"] * 3 + ["sparse"] * 57

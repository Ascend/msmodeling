# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Theory symbol bindings for the DeepSeek-V3 model category."""

from __future__ import annotations

import pytest

from tools.model_diagnostics.domain import INPUT, OUTPUT, ExecutionPhase, ModelRunContext, ParallelContext, SourceKind
from tools.model_diagnostics.organization.theory import build_theory_regions
from tools.model_diagnostics.specification.context_env import build_theory_env

_DEEPSEEK_CONFIG: dict[str, object] = {
    "model_type": "deepseek_v32",
    "features": ["compiled"],
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "first_k_dense_replace": 3,
    "num_experts_per_tok": 8,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "num_hidden_layers": 61,
    "effective_num_hidden_layers": 1,
    "vocab_size": 129280,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 2048,
    "torch_dtype": "float16",
}


def _context(
    *,
    phase: ExecutionPhase = ExecutionPhase.PREFILL,
    config: dict[str, object] | None = None,
    query_length: int = 2,
    context_length: int | None = None,
    batch_size: int = 1,
    parallel: ParallelContext | None = None,
) -> ModelRunContext:
    return ModelRunContext(
        model_name="deepseek-ai/DeepSeek-V3.2",
        entrypoint="text_generate",
        phase=phase,
        batch_size=batch_size,
        query_length=query_length,
        context_length=context_length,
        parallel=parallel or ParallelContext(),
        model_config=dict(config) if config is not None else dict(_DEEPSEEK_CONFIG),
        quantization_config={},
    )


def test_deepseek_v3_theory_env_binds_attention_symbols() -> None:
    env = build_theory_env(_context())

    assert (env["Qlora"], env["KVlora"]) == (1536, 512)
    assert (env["QKnope"], env["QKrope"], env["Vh"], env["Hmla"]) == (128, 64, 128, 192)
    assert env["Dsa_k"] == 2
    assert env["MOE_COMBINE_DTYPE"] == "float32"


@pytest.mark.parametrize("model_type", ("deepseek_v3", "glm_moe_dsa", "kimi_k2"))
def test_deepseek_v3_theory_env_uses_activation_dtype_for_legacy_moe_combine(model_type: str) -> None:
    env = build_theory_env(_context(config={**_DEEPSEEK_CONFIG, "model_type": model_type}))

    assert env["MOE_COMBINE_DTYPE"] == "float16"


@pytest.mark.parametrize(
    ("model_type", "ep", "gate_domain"),
    (
        ("deepseek_v32", 1, "tmoe"),
        ("deepseek_v32", 2, "tmoe"),
        ("deepseek_v3", 1, "tmoe"),
        ("deepseek_v3", 2, "t"),
        ("glm_moe_dsa", 1, "tmoe"),
        ("glm_moe_dsa", 2, "t"),
        ("kimi_k2", 2, "tmoe"),
    ),
)
def test_deepseek_v3_theory_env_binds_moe_gate_token_domain(
    model_type: str,
    ep: int,
    gate_domain: str,
) -> None:
    env = build_theory_env(
        _context(
            config={**_DEEPSEEK_CONFIG, "model_type": model_type},
            query_length=3,
            parallel=ParallelContext(expert_parallel_size=ep),
        )
    )

    assert env["MOE_GATE_TOKENS"] == (env["T"] if gate_domain == "t" else env["Tmoe"])
    assert env["T"] == 3


def test_deepseek_v3_theory_env_binds_decode_dsa_width() -> None:
    env = build_theory_env(_context(phase=ExecutionPhase.DECODE, query_length=1, context_length=128))

    assert env["S"] == 129
    assert env["Dsa_k"] == 129


def test_deepseek_v3_theory_env_defaults_lm_head_tp_to_tensor_parallel() -> None:
    env = build_theory_env(_context(parallel=ParallelContext(tensor_parallel_size=2)))

    assert env["LMTP"] == 2
    assert env["Vtp"] == env["V"] // 2


def test_deepseek_v3_theory_env_binds_routed_and_shared_experts() -> None:
    env = build_theory_env(_context())

    assert (env["E"], env["Ktop"], env["Fmoe"], env["Fe"]) == (256, 8, 2048, 2048)
    assert (env["T"], env["Tmoe"], env["Te"]) == (2, 2, 16)
    assert env["Nshared"] == 1
    assert env["Fshared"] == 2048


def test_deepseek_v3_theory_env_scales_shared_expert_width() -> None:
    env = build_theory_env(_context(config={**_DEEPSEEK_CONFIG, "n_shared_experts": 2}))

    assert env["Nshared"] == 2
    assert env["Fshared"] == 4096


def test_deepseek_v3_theory_env_zero_shared_binds_nothing() -> None:
    env = build_theory_env(_context(config={**_DEEPSEEK_CONFIG, "n_shared_experts": 0}))

    assert "Nshared" not in env
    assert "Fshared" not in env
    assert env["E"] == 256


def test_qwen3_moe_config_binds_no_deepseek_symbols() -> None:
    env = build_theory_env(
        _context(
            config={
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
            }
        )
    )

    assert env["E"] == 128
    assert "Nshared" not in env
    assert "Qlora" not in env
    assert "Dsa_k" not in env


def test_deepseek_v3_spec_expands_dense_prefix_and_moe_layers() -> None:
    from tools.model_diagnostics.builtin import create_stage_comparison_registry
    from tools.model_diagnostics.specification.builtin_activation import create_builtin_operator_activation_registry
    from tools.model_diagnostics.specification.loader import YamlModelDiagnosticsSpecLoader
    from tools.model_diagnostics.specification.source_options import create_builtin_source_options_parsers
    from tools.model_diagnostics.specification.theory_fragments import load_builtin_theory_fragment_registry

    fragments = load_builtin_theory_fragment_registry()
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(fragment_registry=fragments),
        fragment_registry=fragments,
    )
    spec = loader.materialize(
        loader.load("deepseek_v3_v1"), _context(config={**_DEEPSEEK_CONFIG, "effective_num_hidden_layers": 5})
    )
    language = next(region for region in spec.regions if region.region_id == "language")

    assert language.layer_layout == ("dense", "dense", "dense", "moe", "moe")
    assert [stage.stage_id for stage in language.layer_specs["dense"].stages] == [
        "mla_projection",
        "dsa_indexer",
        "sparse_attention",
        "dense_ffn",
    ]
    assert [stage.stage_id for stage in language.layer_specs["moe"].stages] == [
        "mla_projection",
        "dsa_indexer",
        "sparse_attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "shared_ffn",
    ]
    moe_combine = next(stage for stage in language.layer_specs["moe"].stages if stage.stage_id == "moe_combine")
    theory = moe_combine.source_options[SourceKind.THEORY]
    assert [operator.operator_name for operator in theory.operators] == ["unpermute_tokens", "mul", "sum"]


def test_deepseek_v3_spec_omits_dsa_stage_without_dsa_config() -> None:
    from tools.model_diagnostics.builtin import create_stage_comparison_registry
    from tools.model_diagnostics.specification.builtin_activation import create_builtin_operator_activation_registry
    from tools.model_diagnostics.specification.loader import YamlModelDiagnosticsSpecLoader
    from tools.model_diagnostics.specification.source_options import create_builtin_source_options_parsers
    from tools.model_diagnostics.specification.theory_fragments import load_builtin_theory_fragment_registry

    config = {**_DEEPSEEK_CONFIG, "model_type": "deepseek_v3", "effective_num_hidden_layers": 4}
    config.pop("index_topk")
    fragments = load_builtin_theory_fragment_registry()
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(fragment_registry=fragments),
        fragment_registry=fragments,
    )

    spec = loader.materialize(loader.load("deepseek_v3_v1"), _context(config=config))
    language = next(region for region in spec.regions if region.region_id == "language")

    for layer_spec in language.layer_specs.values():
        assert "dsa_indexer" not in {stage.stage_id for stage in layer_spec.stages}
        attention = next(stage for stage in layer_spec.stages if stage.stage_id == "sparse_attention")
        theory = attention.source_options[SourceKind.THEORY]
        assert all("INPUT[9]" not in {str(slot) for slot in operator.tensors} for operator in theory.operators)


def test_deepseek_v3_sparse_attention_declares_stable_runtime_inputs() -> None:
    from tools.model_diagnostics.builtin import create_stage_comparison_registry
    from tools.model_diagnostics.specification.builtin_activation import create_builtin_operator_activation_registry
    from tools.model_diagnostics.specification.loader import YamlModelDiagnosticsSpecLoader
    from tools.model_diagnostics.specification.source_options import create_builtin_source_options_parsers
    from tools.model_diagnostics.specification.theory_fragments import load_builtin_theory_fragment_registry

    fragments = load_builtin_theory_fragment_registry()
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(fragment_registry=fragments),
        fragment_registry=fragments,
    )
    spec = loader.materialize(loader.load("deepseek_v3_v1"), _context())
    language = next(region for region in spec.regions if region.region_id == "language")

    for layer_spec in language.layer_specs.values():
        attention = next(stage for stage in layer_spec.stages if stage.stage_id == "sparse_attention")
        theory = attention.source_options[SourceKind.THEORY]
        sparse_attention = next(
            operator for operator in theory.operators if operator.operator_name == "mla_sparse_attention"
        )

        assert set(sparse_attention.tensors) == {
            INPUT[0],
            INPUT[1],
            INPUT[2],
            INPUT[3],
            INPUT[4],
            INPUT[5],
            OUTPUT[0],
        }
        assert sparse_attention.tensors[INPUT[1]].shape.expression == "[Nblk, Bs, KVlora + QKrope]"
        assert sparse_attention.tensors[INPUT[2]].shape.expression == "[B, Mb]"


@pytest.mark.parametrize(
    ("parallel", "batch_size", "query_length", "operator_name", "expected_shape"),
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
            "mlapo",
            (3, 64, 192),
        ),
        (
            ParallelContext(
                tensor_parallel_size=1,
                data_parallel_size=2,
                expert_parallel_size=1,
                moe_data_parallel_size=2,
            ),
            1,
            2,
            "swiglu",
            (4, 2048),
        ),
    ),
    ids=("tp_local_mla_heads", "dp_moe_shared_tokens"),
)
def test_deepseek_v3_theory_uses_rank_local_parallel_shapes(
    parallel: ParallelContext,
    batch_size: int,
    query_length: int,
    operator_name: str,
    expected_shape: tuple[int, ...],
) -> None:
    from tools.model_diagnostics.builtin import create_stage_comparison_registry
    from tools.model_diagnostics.specification.builtin_activation import create_builtin_operator_activation_registry
    from tools.model_diagnostics.specification.loader import YamlModelDiagnosticsSpecLoader
    from tools.model_diagnostics.specification.source_options import create_builtin_source_options_parsers
    from tools.model_diagnostics.specification.theory_fragments import load_builtin_theory_fragment_registry

    context = _context(
        config={**_DEEPSEEK_CONFIG, "effective_num_hidden_layers": 4},
        batch_size=batch_size,
        query_length=query_length,
        parallel=parallel,
    )
    fragments = load_builtin_theory_fragment_registry()
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(fragment_registry=fragments),
        fragment_registry=fragments,
    )
    spec = loader.materialize(loader.load("deepseek_v3_v1"), context)
    regions = build_theory_regions(context, spec, {"language": (3,)}, ())
    calls = [call for stage in regions[0].layers[0].stages for call in stage.operator_calls]
    matching_calls = [call for call in calls if call.operator_name == operator_name]
    call = matching_calls[0] if operator_name == "mlapo" else matching_calls[-1]

    tensor = next(tensor for tensor in call.tensors if tensor.slot == OUTPUT[0])
    assert tensor.shape == expected_shape

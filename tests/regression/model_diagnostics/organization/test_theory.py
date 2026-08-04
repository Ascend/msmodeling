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
"""Theory source and organization tests without Runtime artifacts."""

from __future__ import annotations

from dataclasses import replace

from tools.model_diagnostics.domain.models import (
    ExecutionPhase,
    ModelExecutionRecord,
    ModelRunContext,
    OperatorCallRecord,
    ParallelContext,
    SourceKind,
)
from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.specification import create_builtin_source_options_parsers
from tools.model_diagnostics.specification.context_env import build_theory_env
from tools.model_diagnostics.specification.builtin_activation import (
    create_builtin_operator_activation_registry,
)
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.domain.organization import ExecutionOrganizationRequest
from tools.model_diagnostics.errors import SourceLoadError
from tools.model_diagnostics.organization.theory import TheoryExecutionOrganizationStrategy
from tools.model_diagnostics.sources.theory import TheoryOperatorRecordSource
from tools.model_diagnostics.specification.loader import YamlModelDiagnosticsSpecLoader

import pytest


def _context(layers: int = 2) -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen3-8B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=2,
        query_length=4,
        context_length=None,
        parallel=ParallelContext(tensor_parallel_size=1, data_parallel_size=1),
        model_config={
            "model_type": "qwen3",
            "features": ["dense"],
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 36,
            "effective_num_hidden_layers": layers,
            "vocab_size": 151936,
            "torch_dtype": "bfloat16",
        },
        quantization_config={},
    )


def test_theory_source_and_organizer_selected_global_slice() -> None:
    context = _context(layers=4)
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
    )
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)
    selected_layers = {"language": (0, 2)}
    selected_stage_regions = ("input", "output")

    execution = TheoryOperatorRecordSource().load_execution(
        context,
        spec,
        selected_layers,
        selected_stage_regions,
    )
    # embedding(1) + 2 layers * (3 qkv + 2 attention + 3 ffn) + select + lm_head
    assert len(execution.operator_calls) == 1 + 2 * 8 + 2

    embed = execution.operator_calls[0]
    assert embed.operator_name == "embedding"
    assert [tensor.slot.direction.value for tensor in embed.tensors] == [
        "input",
        "input",
        "output",
    ]
    assert embed.tensors[0].shape == (151936, 4096)  # [Vtp, H]
    assert embed.tensors[1].shape == (2, 4)  # [B, Q]
    assert embed.tensors[1].dtype == "int64"
    out = next(tensor for tensor in embed.tensors if tensor.slot.direction.value == "output")
    assert out.shape == (2, 4, 4096)
    assert out.dtype == "bfloat16"

    organized = TheoryExecutionOrganizationStrategy().execute(
        ExecutionOrganizationRequest(
            execution=execution,
            spec=spec,
            selected_layers=selected_layers,
            selected_stage_regions=selected_stage_regions,
        )
    )
    assert [region.region_id for region in organized] == ["input", "language", "output"]
    language = organized[1]
    assert [layer.layer_index for layer in language.layers] == [0, 2]
    for layer in language.layers:
        assert [stage.stage_id for stage in layer.stages] == [
            "attention_qkv",
            "attention",
            "dense_ffn",
        ]
        assert [call.operator_name for call in layer.stages[0].operator_calls] == [
            "q_projection",
            "k_projection",
            "v_projection",
        ]
        assert [call.operator_name for call in layer.stages[1].operator_calls] == [
            "attention",
            "o_projection",
        ]
        assert [call.operator_name for call in layer.stages[2].operator_calls] == [
            "gate_up_projection",
            "swiglu",
            "down_projection",
        ]
        attention = next(call for call in layer.stages[1].operator_calls if call.operator_name == "attention")
        assert [tensor.slot.index for tensor in attention.tensors if tensor.slot.direction.value == "input"] == [
            0,
            1,
            2,
        ]
        assert attention.tensors[0].shape == (8, 4096)  # query [T, Lh*Dh]
        assert attention.tensors[1].shape == (1, 128, 8, 128)  # key cache [Nblk, Bs, Lkv, Dh]
        assert attention.tensors[2].shape == (1, 128, 8, 128)  # value cache [Nblk, Bs, Lkv, Dh]
        attention_out = next(tensor for tensor in attention.tensors if tensor.slot.direction.value == "output")
        assert attention_out.shape == (8, 4096)  # [T, Lh*Dh], Lh=32, Dh=128
        assert attention_out.dtype == "bfloat16"
    assert [stage.stage_id for stage in organized[2].stages] == ["lm_head"]
    assert [call.operator_name for call in organized[2].stages[0].operator_calls] == [
        "lm_head_select",
        "lm_head",
    ]
    lm_head = organized[2].stages[0].operator_calls[1]
    lm_out = next(tensor for tensor in lm_head.tensors if tensor.slot.direction.value == "output")
    # Prefill Rout=B=2, Vtp=V=151936
    assert lm_out.shape == (2, 151936)
    lm_head_select = organized[2].stages[0].operator_calls[0]
    assert lm_head_select.operator_name == "lm_head_select"
    select_input = next(tensor for tensor in lm_head_select.tensors if tensor.slot.direction.value == "input")
    select_output = next(tensor for tensor in lm_head_select.tensors if tensor.slot.direction.value == "output")
    assert select_input.shape == (2, 4, 4096)
    assert select_output.shape == (2, 1, 4096)


@pytest.mark.parametrize(
    ("mode", "expected_weight", "expected_output"),
    (
        (None, (151936, 4096), (2, 4, 4096)),
        ("col", (151936, 2048), (2, 4, 2048)),
        ("row", (75968, 4096), (2, 4, 4096)),
    ),
)
def test_theory_embedding_shapes_follow_msmodeling_tp_mode(
    mode: str | None,
    expected_weight: tuple[int, ...],
    expected_output: tuple[int, ...],
) -> None:
    base = _context(layers=1)
    model_config = dict(base.model_config)
    if mode is not None:
        model_config["word_embedding_tp"] = mode
    context = replace(
        base,
        parallel=ParallelContext(tensor_parallel_size=2, data_parallel_size=1),
        model_config=model_config,
    )
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
    )
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)
    execution = TheoryOperatorRecordSource().load_execution(
        context,
        spec,
        selected_layers={},
        selected_stage_regions=("input",),
    )

    embedding = execution.operator_calls[0]
    assert embedding.tensors[0].shape == expected_weight
    assert embedding.tensors[-1].shape == expected_output


def test_theory_rejects_unknown_runtime_embedding_tp_mode() -> None:
    base = _context(layers=1)
    context = replace(
        base,
        model_config={**base.model_config, "word_embedding_tp": "diagonal"},
    )

    with pytest.raises(SpecificationLoadError, match="word_embedding_tp"):
        build_theory_env(context)


def test_theory_organizer_rejects_execution_stream_that_diverges_from_spec() -> None:
    context = _context(layers=1)
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
    )
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)
    selected_layers = {"language": (0,)}
    selected_stage_regions = ("input",)
    execution = TheoryOperatorRecordSource().load_execution(
        context,
        spec,
        selected_layers,
        selected_stage_regions,
    )
    tampered = ModelExecutionRecord(
        source_kind=SourceKind.THEORY,
        run_context=context,
        operator_calls=(
            OperatorCallRecord(
                call_index=0,
                operator_name="not_embedding",
                original_operator_name=None,
                tensors=(),
                source_reference="tampered",
            ),
            *execution.operator_calls[1:],
        ),
    )

    with pytest.raises(SourceLoadError, match="diverges from Spec at call\\[0\\]"):
        TheoryExecutionOrganizationStrategy().execute(
            ExecutionOrganizationRequest(
                execution=tampered,
                spec=spec,
                selected_layers=selected_layers,
                selected_stage_regions=selected_stage_regions,
            )
        )


def test_theory_organizer_rejects_execution_stream_with_wrong_call_count() -> None:
    context = _context(layers=1)
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
    )
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)
    selected_layers = {"language": (0,)}
    selected_stage_regions = ("input",)
    execution = TheoryOperatorRecordSource().load_execution(
        context,
        spec,
        selected_layers,
        selected_stage_regions,
    )
    truncated = ModelExecutionRecord(
        source_kind=SourceKind.THEORY,
        run_context=context,
        operator_calls=execution.operator_calls[:-1],
    )

    with pytest.raises(SourceLoadError, match="call count diverges"):
        TheoryExecutionOrganizationStrategy().execute(
            ExecutionOrganizationRequest(
                execution=truncated,
                spec=spec,
                selected_layers=selected_layers,
                selected_stage_regions=selected_stage_regions,
            )
        )


def test_theory_dtype_bindings_follow_explicit_quantization_context() -> None:
    base = _context(layers=1)
    context = ModelRunContext(
        model_name=base.model_name,
        entrypoint=base.entrypoint,
        phase=base.phase,
        batch_size=base.batch_size,
        query_length=base.query_length,
        context_length=base.context_length,
        parallel=base.parallel,
        model_config=base.model_config,
        quantization_config={
            "activation_dtype": "int8",
            "weight_dtype": "int8",
            "scale_dtype": "float32",
            "accumulation_dtype": "int32",
            "output_dtype": "bfloat16",
        },
    )

    env = build_theory_env(context)

    assert env["D"] == "bfloat16"
    assert env["ACT"] == "int8"
    assert env["LINEAR_IN"] == "int8"
    assert env["WEIGHT"] == "int8"
    assert env["SCALE"] == "float32"
    assert env["ACC"] == "int32"
    assert env["OUT"] == "bfloat16"


def test_qwen3_yaml_resolves_quantized_activation_and_output_dtypes() -> None:
    base = _context(layers=1)
    context = ModelRunContext(
        model_name=base.model_name,
        entrypoint=base.entrypoint,
        phase=base.phase,
        batch_size=base.batch_size,
        query_length=base.query_length,
        context_length=base.context_length,
        parallel=base.parallel,
        model_config=base.model_config,
        quantization_config={
            "activation_dtype": "int8",
            "output_dtype": "bfloat16",
        },
    )
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
    )
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)

    execution = TheoryOperatorRecordSource().load_execution(
        context,
        spec,
        {"language": (0,)},
        ("input",),
    )

    embedding = execution.operator_calls[0]
    embedding_output = next(tensor for tensor in embedding.tensors if tensor.slot.direction.value == "output")
    q_projection = next(call for call in execution.operator_calls if call.operator_name == "q_projection")
    q_input = next(tensor for tensor in q_projection.tensors if tensor.slot.direction.value == "input")
    q_output = next(tensor for tensor in q_projection.tensors if tensor.slot.direction.value == "output")

    assert embedding_output.dtype == "int8"
    assert q_input.dtype == "int8"
    assert q_output.dtype == "bfloat16"


def test_theory_dtype_bindings_default_to_model_dtype_without_quantization() -> None:
    env = build_theory_env(_context(layers=1))

    assert env["D"] == "bfloat16"
    assert env["ACT"] == "bfloat16"
    assert env["LINEAR_IN"] == "bfloat16"
    assert env["WEIGHT"] == "bfloat16"
    assert env["SCALE"] == "float32"
    assert env["ACC"] == "bfloat16"
    assert env["OUT"] == "bfloat16"


def test_theory_dynamic_quantization_only_changes_linear_input_dtype() -> None:
    base = _context(layers=1)
    context = replace(
        base,
        quantization_config={
            "enabled": True,
            "action": "W8A8_DYNAMIC",
            "linear_input_dtype": "int8",
        },
    )

    env = build_theory_env(context)

    assert env["ACT"] == "bfloat16"
    assert env["LINEAR_IN"] == "int8"
    assert env["OUT"] == "bfloat16"


def test_theory_dtype_bindings_reject_invalid_quantization_dtype() -> None:
    base = _context(layers=1)
    context = ModelRunContext(
        model_name=base.model_name,
        entrypoint=base.entrypoint,
        phase=base.phase,
        batch_size=base.batch_size,
        query_length=base.query_length,
        context_length=base.context_length,
        parallel=base.parallel,
        model_config=base.model_config,
        quantization_config={"activation_dtype": ""},
    )

    import pytest

    with pytest.raises(
        SpecificationLoadError,
        match="quantization_config.activation_dtype",
    ):
        build_theory_env(context)


def test_theory_modules_do_not_import_runtime_or_tensor_cast() -> None:
    import ast
    from pathlib import Path

    roots = [
        Path(__file__).parents[4] / "tools" / "model_diagnostics" / "specification",
        Path(__file__).parents[4] / "tools" / "model_diagnostics" / "sources" / "theory.py",
        Path(__file__).parents[4] / "tools" / "model_diagnostics" / "organization" / "theory.py",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(root.rglob("*.py"))

    banned = (
        "tensor_cast",
        "sources.runtime_capture",
        "organization.runtime",
        "specs.qwen3_dense",
    )
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: tuple[str, ...]
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = (node.module or "",)
            else:
                continue
            for name in names:
                assert not any(name == item or name.startswith(item + ".") for item in banned), f"{path} imports {name}"

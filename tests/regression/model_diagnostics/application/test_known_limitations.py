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
"""Tests for Runner-known Limitations and their renderer surface."""

from __future__ import annotations

from tools.model_diagnostics.application.runner import _known_limitations
from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.domain import (
    DiagnosticsResult,
    ExecutionPhase,
    Finding,
    FindingStatus,
    Limitation,
    ModelRunContext,
    ModelRunContextSummary,
    ParallelContext,
    SourceDescription,
    SourceKind,
    summarize_findings,
)
from tools.model_diagnostics.rendering import ComparisonHtmlRenderer
from tools.model_diagnostics.specification import (
    create_builtin_operator_activation_registry,
    create_builtin_source_options_parsers,
)
from tools.model_diagnostics.specification.loader import YamlModelDiagnosticsSpecLoader


def _context(
    *,
    declared_torch_dtype: str | None = "bfloat16",
    torch_dtype: str = "bfloat16",
) -> ModelRunContext:
    model_config: dict[str, object] = {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 36,
        "effective_num_hidden_layers": 1,
        "vocab_size": 151936,
        "torch_dtype": torch_dtype,
    }
    if declared_torch_dtype is not None:
        model_config["declared_torch_dtype"] = declared_torch_dtype
    return ModelRunContext(
        model_name="Qwen/Qwen3-8B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        parallel=ParallelContext(),
        model_config=model_config,
        quantization_config={},
    )


def _qwen3_spec():
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
    )
    return loader.materialize(loader.load("qwen3_dense_v1"), _context())


def test_known_limitations_include_runtime_dtype_mismatch_and_ignored_ops() -> None:
    limitations = _known_limitations(
        _context(declared_torch_dtype="bfloat16", torch_dtype="float16"),
        _qwen3_spec(),
    )
    codes = {item.code for item in limitations}

    assert "theory.dtype.runtime_dtype_mismatch" in codes
    assert "runtime.mechanical_ops_ignored" in codes
    binding = next(item for item in limitations if item.code == "theory.dtype.runtime_dtype_mismatch")
    assert "bfloat16" in binding.message
    assert "float16" in binding.message
    ignored = next(item for item in limitations if item.code == "runtime.mechanical_ops_ignored")
    assert "view" in ignored.message or "index" in ignored.message


def test_known_limitations_omit_runtime_dtype_mismatch_when_dtypes_match() -> None:
    limitations = _known_limitations(_context(), _qwen3_spec())
    codes = {item.code for item in limitations}

    assert "theory.dtype.runtime_dtype_mismatch" not in codes
    assert codes == {"runtime.mechanical_ops_ignored"}


def _result_with_limitations() -> DiagnosticsResult:
    findings = (
        Finding(
            region_id="input",
            layer_index=None,
            stage_id="embedding",
            rule_id="call[0]",
            comparison_kind="one_to_one",
            status=FindingStatus.PASS,
            message_code="call.match",
            message="ok",
            expected="float16",
            actual="float16",
        ),
    )
    return DiagnosticsResult(
        schema_version="1",
        spec_id="qwen3_dense_v1",
        spec_version="1.0.0",
        context=ModelRunContextSummary(
            model_name="Qwen/Qwen3-8B",
            entrypoint="text_generate",
            phase=ExecutionPhase.PREFILL,
            batch_size=1,
            query_length=2,
            context_length=None,
        ),
        left_source=SourceDescription(SourceKind.THEORY),
        right_source=SourceDescription(SourceKind.RUNTIME),
        selected_layers={"language": (0,)},
        selected_stage_regions=("input", "output"),
        findings=findings,
        summary=summarize_findings(findings),
        limitations=(
            Limitation(
                code="theory.dtype.runtime_dtype_mismatch",
                message="Declared BF16 differs from Runtime float16",
            ),
        ),
    )


def test_comparison_html_renderer_emits_limitations_section() -> None:
    rendered = ComparisonHtmlRenderer().render(_result_with_limitations())

    assert "Limitations" in rendered
    assert "theory.dtype.runtime_dtype_mismatch" in rendered
    assert "Declared BF16 differs from Runtime float16" in rendered

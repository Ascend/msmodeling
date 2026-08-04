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
"""Offline ``python -m tools.model_diagnostics <run.yaml>`` entry."""

from __future__ import annotations

import io
import importlib
import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.cli import (
    _artifact_context_line,
    _build_parser,
    _print_user_actionable_warnings,
    _replay_captured_runtime_output,
    main,
)
from tools.model_diagnostics.domain import ExecutionPhase, ModelRunContext, ParallelContext
from tools.model_diagnostics.errors import InvalidDiagnosticsRequest
from tools.model_diagnostics.specification import (
    SpecificationLoadError,
    YamlModelDiagnosticsSpecLoader,
    create_builtin_operator_activation_registry,
    create_builtin_source_options_parsers,
    load_diagnostics_run_profile,
)

_PROFILES_DIR = Path(__file__).resolve().parents[4] / "tools" / "model_diagnostics" / "profiles"


def test_help_text_is_printable_on_windows_default_console() -> None:
    _build_parser().format_help().encode("gbk")


def test_module_entrypoint_delegates_to_cli(monkeypatch) -> None:
    monkeypatch.setattr("tools.model_diagnostics.cli.main", lambda: 7)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("tools.model_diagnostics.__main__", run_name="__main__")

    assert error.value.code == 7


def test_builtin_specs_package_is_importable() -> None:
    package = importlib.reload(importlib.import_module("tools.model_diagnostics.specs"))

    assert package.__all__ == []


def test_runtime_context_and_captured_output_helpers(capsys) -> None:
    context = ModelRunContext(
        model_name="synthetic/model",
        entrypoint="test",
        phase=None,
        batch_size=2,
        query_length=3,
        context_length=None,
        parallel=ParallelContext(tensor_parallel_size=4),
        model_config={},
        quantization_config={},
    )
    artifact = type("Artifact", (), {"run_context": context})()

    assert _artifact_context_line(artifact) == ("synthetic/model | ? | batch=2 query=3 context=0 | TP=4")
    _replay_captured_runtime_output(io.StringIO("runtime out\n"), io.StringIO("runtime err\n"))
    assert capsys.readouterr().err == "runtime out\nruntime err\n"


def test_comparison_report_requires_theory_compare(tmp_path: Path, capsys) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([str(profile), "--comparison-report"])
    assert error.value.code == 2
    assert "requires --theory-compare" in capsys.readouterr().err


def test_default_runtime_output_keeps_selection_warning(capsys) -> None:
    captured_stderr = io.StringIO(
        "profile.py:1: DiagnosticsSelectionWarning: selected language layer 5 is unavailable\n"
    )

    _print_user_actionable_warnings(io.StringIO(), captured_stderr)

    captured = capsys.readouterr()
    assert captured.err == ("[model_diagnostics warning] selected language layer 5 is unavailable\n")


@pytest.mark.parametrize(
    ("filename", "phase", "query_length", "context_length"),
    (
        ("prefill_example.yaml", ExecutionPhase.PREFILL, 2, None),
        ("decode_example.yaml", ExecutionPhase.DECODE, 1, 128),
    ),
)
def test_load_category_1_run_profile(
    filename: str,
    phase: ExecutionPhase,
    query_length: int,
    context_length: int | None,
) -> None:
    profile = load_diagnostics_run_profile(_PROFILES_DIR / filename)

    assert profile.model_name == "Qwen/Qwen3-8B"
    assert profile.entrypoint == "text_generate"
    # Omitted in the profile YAML: derived from the Spec at to_request() time,
    # not hardcoded to a particular model's region ids.
    assert profile.selected_language_layers is None
    assert profile.selected_stage_regions == ()
    assert profile.num_hidden_layers_override == 1
    assert profile.num_mtp_tokens == 0
    assert profile.do_compile is True
    assert profile.quantize_linear_action == "W8A8_DYNAMIC"
    assert profile.word_embedding_tp is None

    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
    )
    context = ModelRunContext(
        model_name=profile.model_name,
        entrypoint=profile.entrypoint,
        phase=profile.phase,
        batch_size=profile.batch_size,
        query_length=profile.query_length,
        context_length=profile.context_length,
        parallel=profile.parallel,
        model_config={"model_type": "qwen3", "effective_num_hidden_layers": 1},
        quantization_config={},
    )
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)

    request = profile.to_request(context=context, spec=spec)

    assert request.context.phase is phase
    assert request.context.query_length == query_length
    assert request.context.context_length == context_length
    assert request.selected_layers == {"language": (0,)}
    assert request.selected_stage_regions == ("input", "output")


def test_run_profile_accepts_mtp_decode(tmp_path: Path) -> None:
    path = tmp_path / "mtp_decode.yaml"
    path.write_text(
        "\n".join(
            (
                "model_name: Qwen/Qwen3-0.6B",
                "phase: decode",
                "batch_size: 1",
                "query_length: 3",
                "context_length: 128",
                "num_mtp_tokens: 2",
                "num_hidden_layers_override: 1",
            )
        ),
        encoding="utf-8",
    )

    profile = load_diagnostics_run_profile(path)

    assert profile.num_mtp_tokens == 2


def test_run_profile_can_explicitly_disable_compile(tmp_path: Path) -> None:
    path = tmp_path / "eager.yaml"
    path.write_text(
        "\n".join(
            [
                "model_name: Qwen/Qwen3-8B",
                "phase: prefill",
                "batch_size: 1",
                "query_length: 2",
                "do_compile: false",
            ]
        ),
        encoding="utf-8",
    )

    assert load_diagnostics_run_profile(path).do_compile is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", "false", "schema_version"),
        ("selected_stage_regions", "output", "must be a sequence"),
        ("device", "[]", "device must be a non-empty string"),
        ("entrypoint", '""', "entrypoint must be a non-empty string"),
        ("entrypoint", "1", "entrypoint must be a non-empty string"),
        ("quantize_linear_action", "UNKNOWN", "must be one of"),
        ("quantize_linear_action", '""', "must be one of"),
    ),
)
def test_run_profile_rejects_invalid_scalar_fields(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    path = tmp_path / "invalid_scalar.yaml"
    path.write_text(
        "\n".join(
            (
                "model_name: Qwen/Qwen3-8B",
                "phase: prefill",
                "batch_size: 1",
                "query_length: 2",
                f"{field}: {value}",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpecificationLoadError, match=message):
        load_diagnostics_run_profile(path)


@pytest.mark.parametrize("value", ("true", '"2"', "0"))
def test_run_profile_rejects_invalid_parallel_degrees(tmp_path: Path, value: str) -> None:
    path = tmp_path / "invalid_parallel.yaml"
    path.write_text(
        "\n".join(
            (
                "model_name: Qwen/Qwen3-8B",
                "phase: prefill",
                "batch_size: 1",
                "query_length: 2",
                "parallel:",
                f"  tensor_parallel_size: {value}",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpecificationLoadError, match="parallel.tensor_parallel_size"):
        load_diagnostics_run_profile(path)


def test_run_profile_accepts_zero_context_length(tmp_path: Path) -> None:
    path = tmp_path / "zero_context.yaml"
    path.write_text(
        "\n".join(
            (
                "model_name: Qwen/Qwen3-8B",
                "phase: prefill",
                "batch_size: 1",
                "query_length: 2",
                "context_length: 0",
            )
        ),
        encoding="utf-8",
    )

    assert load_diagnostics_run_profile(path).context_length == 0


@pytest.mark.parametrize("mode", ("col", "row"))
def test_run_profile_accepts_word_embedding_tp_modes(
    tmp_path: Path,
    mode: str,
) -> None:
    path = tmp_path / f"embedding_{mode}.yaml"
    path.write_text(
        "\n".join(
            [
                "model_name: Qwen/Qwen3-8B",
                "phase: prefill",
                "batch_size: 1",
                "query_length: 2",
                f"word_embedding_tp: {mode}",
            ]
        ),
        encoding="utf-8",
    )

    assert load_diagnostics_run_profile(path).word_embedding_tp == mode


def test_run_profile_rejects_unknown_word_embedding_tp_mode(tmp_path: Path) -> None:
    path = tmp_path / "embedding_invalid.yaml"
    path.write_text(
        "\n".join(
            [
                "model_name: Qwen/Qwen3-8B",
                "phase: prefill",
                "batch_size: 1",
                "query_length: 2",
                "word_embedding_tp: diagonal",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpecificationLoadError, match="word_embedding_tp must be one of"):
        load_diagnostics_run_profile(path)


def test_run_profile_selects_only_explicit_language_layers(tmp_path: Path) -> None:
    path = tmp_path / "selected.yaml"
    path.write_text(
        "\n".join(
            [
                "model_name: Qwen/Qwen3-8B",
                "phase: prefill",
                "batch_size: 1",
                "query_length: 2",
                "num_hidden_layers_override: 6",
                "selected_language_layers: [5, 0, 3]",
            ]
        ),
        encoding="utf-8",
    )
    profile = load_diagnostics_run_profile(path)
    assert profile.selected_language_layers == (0, 3, 5)

    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=create_stage_comparison_registry(),
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=create_builtin_source_options_parsers(),
    )
    context = ModelRunContext(
        model_name=profile.model_name,
        entrypoint=profile.entrypoint,
        phase=profile.phase,
        batch_size=profile.batch_size,
        query_length=profile.query_length,
        context_length=profile.context_length,
        parallel=profile.parallel,
        model_config={"model_type": "qwen3", "effective_num_hidden_layers": 6},
        quantization_config={},
    )
    spec = loader.materialize(loader.load("qwen3_dense_v1"), context)

    request = profile.to_request(context=context, spec=spec)

    assert request.selected_layers == {"language": (0, 3, 5)}
    assert request.selected_stage_regions == ("input", "output")

    with pytest.warns(UserWarning, match="unavailable indices"):
        skipped = replace(profile, selected_language_layers=(3, 6)).to_request(
            context=replace(
                context,
                model_config={"model_type": "qwen3", "effective_num_hidden_layers": 4},
            ),
            spec=loader.materialize(
                loader.load("qwen3_dense_v1"),
                replace(
                    context,
                    model_config={"model_type": "qwen3", "effective_num_hidden_layers": 4},
                ),
            ),
        )
    assert skipped.selected_layers == {"language": (3,)}

    narrow_context = replace(
        context,
        model_config={"model_type": "qwen3", "effective_num_hidden_layers": 1},
    )
    with pytest.raises(InvalidDiagnosticsRequest, match="no indices within captured language layers"):
        replace(profile, selected_language_layers=(5,)).to_request(
            context=narrow_context,
            spec=loader.materialize(loader.load("qwen3_dense_v1"), narrow_context),
        )


@pytest.mark.parametrize(
    ("selected_language_layers_yaml", "message"),
    (
        ("selected_language_layers: []", "non-empty sequence"),
        ("selected_language_layers: [0, true]", "indices must be integers"),
        ("selected_language_layers: [-1]", "must be non-negative"),
    ),
)
def test_run_profile_rejects_invalid_selected_language_layers(
    tmp_path: Path,
    selected_language_layers_yaml: str,
    message: str,
) -> None:
    path = tmp_path / "bad_selected.yaml"
    path.write_text(
        "\n".join(
            [
                "model_name: Qwen/Qwen3-8B",
                "phase: prefill",
                "batch_size: 1",
                "query_length: 2",
                "num_hidden_layers_override: 6",
                selected_language_layers_yaml,
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpecificationLoadError, match=message):
        load_diagnostics_run_profile(path)


def test_run_profile_rejects_nested_capture_block(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "\n".join(
            [
                "model_name: Qwen/Qwen3-8B",
                "phase: prefill",
                "batch_size: 1",
                "query_length: 2",
                "num_hidden_layers_override: 1",
                "capture:",
                "  model_id: Qwen/Qwen3-8B",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpecificationLoadError, match="unsupported run-profile field"):
        load_diagnostics_run_profile(path)


def test_run_profile_rejects_user_model_config(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "\n".join(
            [
                "model_name: Qwen/Qwen3-8B",
                "phase: prefill",
                "batch_size: 1",
                "query_length: 2",
                "num_hidden_layers_override: 1",
                "model_config:",
                "  hidden_size: 4096",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpecificationLoadError, match="unsupported run-profile field"):
        load_diagnostics_run_profile(path)


def test_module_entry_rejects_missing_profile(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.yaml"

    code = main([str(missing)])

    captured = capsys.readouterr()
    assert code == 2
    assert "run profile not found" in captured.err

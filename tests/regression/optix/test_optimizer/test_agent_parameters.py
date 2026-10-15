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

"""Context definitions drive candidate validation and runtime field assembly."""

import contextlib
import json
from unittest.mock import Mock

import pytest
from loguru import logger

from optix.config.config import OptimizerConfigField, PerformanceIndex, field_to_param, map_param_with_value
from optix.config.custom_command import VllmCommand, VllmCommandConfig
from optix.optimizer.agentic.strategies import AgentCandidateStrategy
from optix.optimizer.agentic.validation import validate_candidate


def entry(name, default, dtype="int", minimum=0, maximum=128, **kwargs):
    return {
        "name": name,
        "default": default,
        "dtype": dtype,
        "min": minimum,
        "max": maximum,
        "config_position": "run",
        **kwargs,
    }


def context(*fields):
    return {"engine": "vllm", "search_space": {"parameters": list(fields), "constants": []}, "constraints": {}}


def candidate(**params):
    return {"candidate_id": "r1-a", "params": params}


def test_context_defaults_and_constant_constraints():
    ctx = context(entry("tp", 8, minimum=8, maximum=8, constant=8), entry("MAX_NUM_SEQS", 16))
    result = validate_candidate(ctx, candidate(MAX_NUM_SEQS=32))
    assert result["valid"], result["errors"]
    assert result["params"] == {"tp": 8, "MAX_NUM_SEQS": 32}
    invalid = validate_candidate(ctx, candidate(tp=4))
    assert not invalid["valid"]
    assert invalid["params"]["tp"] == 8
    assert any("conflicts" in error for error in invalid["errors"])
    assert not validate_candidate(ctx, candidate(unregistered=1))["valid"]


def test_next_context_can_change_domain_and_remove_extra_parameter():
    ctx = context(entry("batch_test", 16, maximum=32), entry("extra_test", 1))
    assert not validate_candidate(ctx, candidate(batch_test=64))["valid"]
    ctx["search_space"]["parameters"] = [entry("batch_test", 16, maximum=64)]
    result = validate_candidate(ctx, candidate(batch_test=64))
    assert result["valid"]
    assert result["params"] == {"batch_test": 64}
    assert not validate_candidate(ctx, candidate(extra_test=1))["valid"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numeric_values_are_rejected(value):
    result = validate_candidate(context(entry("x", 1, "float")), candidate(x=value))
    assert not result["valid"]
    assert any("must be finite" in error for error in result["errors"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_ratio_values_are_rejected(value):
    result = validate_candidate(
        context(entry("ratio_param", 0.9, "ratio", minimum=0.0, maximum=1.0)), candidate(ratio_param=value)
    )
    assert not result["valid"]


def test_nan_ratio_reports_finite_error():
    result = validate_candidate(
        context(entry("ratio_param", 0.9, "ratio", minimum=0.0, maximum=1.0)), candidate(ratio_param=float("nan"))
    )
    assert any("must be finite" in error for error in result["errors"])


def test_pinned_min_max_value_wins_over_divergent_default():
    ctx = context(entry("x", 4, minimum=8, maximum=8))
    result = validate_candidate(ctx, candidate(x=8))
    assert result["valid"], result["errors"]
    assert result["params"]["x"] == 8
    invalid = validate_candidate(ctx, candidate(x=4))
    assert not invalid["valid"]
    assert invalid["params"]["x"] == 8
    assert any("conflicts" in error for error in invalid["errors"])


@pytest.mark.parametrize("value", [0.5, 2])
def test_non_boolean_is_rejected(value):
    assert not validate_candidate(context(entry("flag", False, "bool", maximum=1)), candidate(flag=value))["valid"]


def test_constant_conflict_is_rejected_even_inside_numeric_range():
    result = validate_candidate(context(entry("x", 8, minimum=1, maximum=16, constant=8)), candidate(x=4))
    assert not result["valid"]
    assert result["params"]["x"] == 8
    assert any("conflicts" in error for error in result["errors"])


def test_toml_fields_keep_their_execution_position():
    base = OptimizerConfigField(name="x", dtype="int", min=0, max=10, value=3, config_position="run")
    ctx = context(entry("x", 3, maximum=10, config_position="env"))
    result = validate_candidate(ctx, candidate(x=4))
    fields = AgentCandidateStrategy("unused")._params_to_target_field(result["params"], [base], ctx)
    assert fields[0].config_position == "run"


def test_candidate_file_round_mismatch_is_rejected(tmp_path):
    (tmp_path / "candidates.round-1.json").write_text(json.dumps({"round": 2, "candidates": [candidate()]}))
    with pytest.raises(ValueError, match="round"):
        AgentCandidateStrategy(tmp_path)._load_candidates()


def test_round_with_all_candidates_rejected_logs_warning(tmp_path):
    (tmp_path / "candidates.round-1.json").write_text(
        json.dumps({"round": 1, "candidates": [candidate(unknown_param=1)]})
    )
    records = []
    handler_id = logger.add(records.append, level="WARNING")
    try:
        scheduler = Mock(error_info="", last_outcome=None)
        result = AgentCandidateStrategy(tmp_path).run(scheduler, [], context(entry("tp", 8, minimum=8, maximum=8)))
    finally:
        logger.remove(handler_id)
    scheduler.run.assert_not_called()
    assert result.trials == []
    assert any("runs 0 trials" in record and "unknown_param" in record for record in records)


def test_scalar_candidate_with_context_extras_runs_without_toml_edits(tmp_path):
    ctx = context(
        entry("tp", 8, minimum=8, maximum=8),
        entry("enforce_eager", True, "bool", maximum=1),
        entry("cudagraph_mode", "FULL_DECODE_ONLY", "enum", maximum=1, choices=["FULL_DECODE_ONLY"]),
        entry("cudagraph_capture_sizes", [1, 4], "enum", maximum=1, choices=[[1, 4], [4, 8]]),
        entry("CUSTOM_ENV", "on", "enum", maximum=1, choices=["on"], config_position="env"),
    )
    raw = candidate(cudagraph_capture_sizes=[4, 8])
    (tmp_path / "candidates.round-1.json").write_text(json.dumps({"round": 1, "candidates": [raw]}))
    fields = [OptimizerConfigField(name="tp", dtype="int", min=8, max=8, value=8, config_position="run")]
    scheduler = Mock(error_info="", last_outcome=None)
    scheduler.disable_early_exit.return_value = contextlib.nullcontext()
    scheduler.run.return_value = PerformanceIndex(generate_speed=100, success_rate=1)
    strategy = AgentCandidateStrategy(tmp_path)
    result = strategy.run(scheduler, fields, ctx)
    assert len(result.trials) == 1 and result.trials[0].error is None
    position, active = scheduler.run.call_args.args
    resolved = map_param_with_value(position, active)
    argv = VllmCommand(VllmCommandConfig(model="model", served_model_name="served"), resolved).command
    assert "--enforce-eager" in argv
    graph = json.loads(argv[argv.index("--compilation-config") + 1])
    assert graph["cudagraph_capture_sizes"] == [4, 8]
    assert graph["cudagraph_mode"] == "FULL_DECODE_ONLY"
    env = next(f for f in resolved if f.name == "CUSTOM_ENV")
    assert (env.value, env.config_position) == ("on", "env")
    assert result.trials[0].metadata == {"round_id": 1}
    assert result.trials[0].params["enforce_eager"] is True


def test_configured_container_and_boolean_values_survive_assembly():
    base = [
        OptimizerConfigField(
            name="speculative_enforce_eager", dtype="bool", min=0, max=1, value=False, config_position="run"
        ),
        OptimizerConfigField(
            name="cudagraph_capture_sizes",
            dtype="enum",
            min=0,
            max=1,
            dtype_param=[[1, 4], [4, 8]],
            value="[1, 4]",
            config_position="run",
        ),
    ]
    fields = AgentCandidateStrategy("unused")._params_to_target_field(
        {"speculative_enforce_eager": True, "cudagraph_capture_sizes": [4, 8]}, base, {}
    )
    resolved = map_param_with_value(field_to_param(tuple(fields)), tuple(fields))
    config = VllmCommandConfig(
        model="model",
        served_model_name="served",
        others='--speculative-config \'{"method": "mtp", "num_speculative_tokens": 2}\'',
    )
    argv = VllmCommand(config, resolved).command
    spec = json.loads(argv[argv.index("--speculative-config") + 1])
    graph = json.loads(argv[argv.index("--compilation-config") + 1])
    assert spec["enforce_eager"] is True
    assert graph["cudagraph_capture_sizes"] == [4, 8]

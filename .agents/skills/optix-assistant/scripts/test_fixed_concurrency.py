"""Confirmed benchmark concurrency must survive TOML normalization and candidate assembly."""

import copy
import hashlib
import json

import pytest
import tomlkit

import config_preflight
from check_search_space import _schema_from_fields, check_parameters
from validate_candidates import validate_round


@pytest.fixture
def fixed_load(monkeypatch):
    # Only external package discovery is stubbed; all configuration and candidate
    # checks use their real implementations.
    monkeypatch.setattr(config_preflight, "_ais_bench_pkg_root", lambda: None)
    field = dict(name="CONCURRENCY", dtype="int", config_position="env", value=64, min=64, max=64)
    config = {"vllm": {"target_field": [field]}, "ais_bench": {"command": {"others": "--num-prompts 100"}}}
    context = {
        "engine": "vllm",
        "benchmark": "ais_bench",
        "benchmark_spec_confirmed": True,
        "benchmark_spec": dict(input_len_avg=128, output_len_avg=64, num_requests=100, concurrency=64),
        "model_info": {"is_moe": False},
    }
    return config, context


def preflight(tmp_path, config, context):
    path = tmp_path / "context.json"
    path.write_text(json.dumps(context), encoding="utf-8")
    return config_preflight._check_benchmark_consistency(config, "vllm", str(path), "ais_bench")[0]


@pytest.mark.parametrize("key", ["value", "min", "max"])
@pytest.mark.parametrize("value", [None, "64", True, 32, 64.9, float("nan"), float("inf")])
def test_preflight_rejects_incomplete_or_nonmatching_fixed_load(tmp_path, fixed_load, key, value):
    config, context = fixed_load
    field = config["vllm"]["target_field"][0]
    if value is None:
        field.pop(key)
    else:
        field[key] = value
    assert any("value/min/max" in message for message in preflight(tmp_path, config, context))
    section = config["vllm"]
    issues = check_parameters(context, section=section, fields=_schema_from_fields(section["target_field"]))
    assert any("value/min/max" in issue["msg"] for issue in issues)


@pytest.mark.parametrize(
    "key,value",
    [
        ("constant", 32),
        ("constant", "64"),
        ("constant", True),
        ("constant", 64.9),
        ("config_position", "run"),
        ("config_position", None),
        ("dtype", "bool"),
        ("name", "concurrency"),
    ],
)
def test_preflight_rejects_incompatible_runtime_definition(tmp_path, fixed_load, key, value):
    config, context = fixed_load
    config["vllm"]["target_field"][0][key] = value
    assert preflight(tmp_path, config, context)


@pytest.mark.parametrize("count", [0, 2])
def test_preflight_requires_one_concurrency_field(tmp_path, fixed_load, count):
    config, context = fixed_load
    config["vllm"]["target_field"] *= count
    assert preflight(tmp_path, config, context)


@pytest.mark.parametrize("constant", [None, 64])
def test_fixed_load_passes_without_mutating_inputs(tmp_path, fixed_load, constant):
    config, context = fixed_load
    config["vllm"]["target_field"][0]["constant"] = constant
    section = config["vllm"]
    fields = _schema_from_fields(section["target_field"])
    before = copy.deepcopy((config, context, fields))
    assert not preflight(tmp_path, config, context)
    assert not check_parameters(context, section=section, fields=fields, params={"CONCURRENCY": 64})
    assert (config, context, fields) == before


@pytest.mark.parametrize("params", [{"CONCURRENCY": 32}, {"CONCURRENCY": "64"}])
def test_parameter_check_rejects_candidate_load_drift(fixed_load, params):
    config, context = fixed_load
    section = config["vllm"]
    issues = check_parameters(
        context, section=section, fields=_schema_from_fields(section["target_field"]), params=params
    )
    assert any("候选 CONCURRENCY" in issue["msg"] for issue in issues)


def test_parameter_check_rejects_widened_active_domain(fixed_load):
    config, context = fixed_load
    section = config["vllm"]
    fields = _schema_from_fields(section["target_field"])
    fields[0].update(min=0, max=100)
    issues = check_parameters(context, section=section, fields=fields)
    assert any("default/min/max" in issue["msg"] for issue in issues)


def prepare_round(tmp_path, config, context, concurrency):
    path = tmp_path / "config.toml"
    path.write_text(tomlkit.dumps(config), encoding="utf-8")
    context["source"] = {"config_toml": str(path), "config_toml_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    context["search_space"] = {"parameters": _schema_from_fields(config["vllm"]["target_field"]), "constants": []}
    (tmp_path / "context.json").write_text(json.dumps(context), encoding="utf-8")
    data = {
        "round": 1,
        "directions": [{"direction_id": "d", "trial_type": "optimization"}],
        "candidates": [{"candidate_id": "c", "direction_id": "d", "params": {"CONCURRENCY": concurrency}}],
    }
    (tmp_path / "candidates.round-1.json").write_text(json.dumps(data), encoding="utf-8")
    return data


@pytest.mark.parametrize("change", ["missing_bounds", "missing_value", "wrong_fixed_value"])
def test_round_rejects_invalid_toml_even_without_preflight(tmp_path, fixed_load, change):
    config, context = fixed_load
    field = config["vllm"]["target_field"][0]
    if change == "missing_bounds":
        field.pop("min")
        field.pop("max")
    elif change == "missing_value":
        field.pop("value")  # Normalization would silently replace the default with min.
    else:
        field.update(value=32, min=32, max=32)
    data = prepare_round(tmp_path, config, context, 32)
    with pytest.raises(ValueError, match="benchmark_spec.concurrency=64"):
        validate_round(data, tmp_path, 1)


def test_round_accepts_fixed_load_and_rejects_changed_candidate(tmp_path, fixed_load):
    config, context = fixed_load
    validate_round(prepare_round(tmp_path, config, context, 64), tmp_path, 1)
    data = prepare_round(tmp_path, config, context, 64)
    data["candidates"][0]["params"] = {}
    validate_round(data, tmp_path, 1)  # An omitted constant is resolved to its fixed default.
    with pytest.raises(ValueError, match="CONCURRENCY"):
        validate_round(prepare_round(tmp_path, config, context, 32), tmp_path, 1)


@pytest.mark.parametrize("mode", ["range", "unconfirmed"])
def test_adjustable_or_unconfirmed_load_keeps_existing_behavior(tmp_path, fixed_load, mode):
    config, context = fixed_load
    config["vllm"]["target_field"][0].update(min=1, max=100)
    if mode == "range":
        context["benchmark_spec"].update(concurrency=None, concurrency_range=[1, 100])
    else:
        context["benchmark_spec_confirmed"] = False
    assert not preflight(tmp_path, config, context)
    validate_round(prepare_round(tmp_path, config, context, 32), tmp_path, 1)

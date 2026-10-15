"""Regression: a round cannot unfreeze a TOML constant."""

import copy
import json

import pytest

import validate_candidates as gate
import check_search_space


def field(constant=2):
    return dict(name="x", dtype="int", config_position="run", min=0, max=8, default=constant, constant=constant)


def space(item):
    return {"parameters": [item], "constants": []}


@pytest.mark.parametrize("constant", [0, 2])
@pytest.mark.parametrize("change", ["omit", "change", "widen", "default"])
def test_space_cannot_unfreeze_toml(constant, change):
    base = field(constant)
    item = dict(base, min=constant, max=constant)
    if change == "omit":
        item.pop("constant")
    elif change == "change":
        item["constant"] = item["default"] = 4
    elif change == "widen":
        item["max"] = 8
    else:
        item["default"] = 4
    with pytest.raises(ValueError, match="constant"):
        gate.validate_search_space(space(item), [base])


@pytest.mark.parametrize("constant", [0, 2])
def test_preserved_freeze_passes_without_mutating_input(constant):
    item = dict(field(constant), min=constant, max=constant)
    original = copy.deepcopy(item)
    assert gate.validate_search_space(space(item), [field(constant)]) == [item]
    assert item == original


def prepare(tmp_path, monkeypatch):
    base = field()
    item = dict(base, min=2, max=2)
    section = {"target_field": [dict(name="x", dtype="int", min=0, max=8, value=2, constant=2, config_position="run")]}
    (tmp_path / "context.json").write_text(json.dumps({"search_space": space(item)}))
    monkeypatch.setattr(check_search_space, "load_configured_parameters", lambda c: (section, [base]))
    monkeypatch.setattr(check_search_space, "check_parameters", lambda *a, **k: [])
    monkeypatch.setattr(gate, "check_run_entry_flags", lambda *a: [])
    data = {
        "round": 1,
        "directions": [{"direction_id": "d", "trial_type": "optimization"}],
        "candidates": [{"candidate_id": "c", "direction_id": "d", "params": {}}],
    }
    return item, data


def test_round_rechecks_space_when_space_check_is_bypassed(tmp_path, monkeypatch):
    item, data = prepare(tmp_path, monkeypatch)
    item.pop("constant")
    monkeypatch.setattr(gate, "validate_search_space", lambda *a: [item])
    with pytest.raises(ValueError, match="execution search_space must retain"):
        gate.validate_round(data, tmp_path, 1)


def test_round_rechecks_resolved_values_before_assembly(tmp_path, monkeypatch):
    from optix.optimizer.agentic import validation
    from optix.optimizer.agentic.strategies import AgentCandidateStrategy

    _, data = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(validation, "validate_candidate", lambda *a: {"valid": True, "params": {"x": 4}})

    def must_not_assemble(*a, **k):
        pytest.fail("Conflicting frozen value reached execution assembly")

    monkeypatch.setattr(AgentCandidateStrategy, "_params_to_target_field", must_not_assemble)
    with pytest.raises(ValueError, match="resolved candidate conflicts with TOML constant"):
        gate.validate_round(data, tmp_path, 1)


def test_round_accepts_frozen_value_and_rejects_override(tmp_path, monkeypatch):
    _, data = prepare(tmp_path, monkeypatch)
    gate.validate_round(data, tmp_path, 1)
    data["candidates"][0]["params"] = {"x": 4}
    with pytest.raises(ValueError):
        gate.validate_round(data, tmp_path, 1)


def enum_field(constant=2):
    return dict(
        name="e",
        dtype="enum",
        config_position="run",
        min=0,
        max=1,
        default=constant,
        constant=constant,
        choices=[constant],
    )


@pytest.mark.parametrize("constant", [0, 2])
@pytest.mark.parametrize("change", ["omit", "change", "widen", "default"])
def test_enum_space_cannot_unfreeze_toml(constant, change):
    base = enum_field(constant)
    item = dict(base)
    if change == "omit":
        item.pop("constant")
    elif change == "change":
        item["constant"] = item["default"] = 4
        item["choices"] = [4]
    elif change == "widen":
        item["choices"] = [constant, 8]
    else:
        item["default"] = 4
    with pytest.raises(ValueError, match="constant"):
        gate.validate_search_space(space(item), [base])


def test_enum_freeze_passes_and_allows_numeric_domain_growth():
    base = enum_field(2)
    item = dict(enum_field(2))
    assert gate.validate_search_space(space(item), [base]) == [item]
    numeric = dict(name="n", dtype="int", config_position="run", min=0, max=4, default=2)
    grown = dict(numeric, max=8)
    assert gate.validate_search_space(space([grown][0]), [numeric]) == [grown]


def test_round_rejects_toml_modified_after_round_zero(tmp_path, monkeypatch):
    import hashlib

    base = field()
    item = dict(base, min=2, max=2)
    section = {"target_field": [dict(name="x", dtype="int", min=0, max=8, value=2, constant=2, config_position="run")]}
    toml = tmp_path / "config.toml"
    toml.write_text("[vllm]\n", encoding="utf-8")
    context = {
        "search_space": space(item),
        "source": {"config_toml": str(toml), "config_toml_sha256": hashlib.sha256(toml.read_bytes()).hexdigest()},
    }
    (tmp_path / "context.json").write_text(json.dumps(context))
    monkeypatch.setattr(check_search_space, "load_configured_parameters", lambda c: (section, [base]))
    monkeypatch.setattr(check_search_space, "check_parameters", lambda *a, **k: [])
    monkeypatch.setattr(gate, "check_run_entry_flags", lambda *a: [])
    data = {
        "round": 1,
        "directions": [{"direction_id": "d", "trial_type": "optimization"}],
        "candidates": [{"candidate_id": "c", "direction_id": "d", "params": {}}],
    }
    gate.validate_round(data, tmp_path, 1)
    toml.write_text("[vllm]\nmodified = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        gate.validate_round(data, tmp_path, 1)

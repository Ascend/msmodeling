"""The skill gate validates direction types even with the original OptiX engine."""

import builtins
import copy

import pytest

import validate_candidates as gate


def payload(kind="optimization"):
    return {
        "directions": [{"direction_id": "d1", "trial_type": kind}],
        "candidates": [{"candidate_id": "c1", "direction_id": "d1", "params": {}}],
    }


@pytest.mark.parametrize("kind", ["optimization", "diagnosis"])
@pytest.mark.parametrize("repeat_type", [False, True])
def test_classification_check_is_local_and_read_only(monkeypatch, kind, repeat_type):
    data = payload(kind)
    if repeat_type:
        data["candidates"][0]["trial_type"] = kind
    original = copy.deepcopy(data)
    original_import = builtins.__import__

    def no_engine_import(name, *args, **kwargs):
        if name == "optix" or name.startswith("optix."):
            pytest.fail("Classification must not import OptiX")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_engine_import)
    gate.validate_trial_types(data)
    assert data == original


@pytest.mark.parametrize("kind", [None, "", "typo", True, [], {}])
def test_invalid_types_are_rejected_before_loading_execution_context(tmp_path, kind):
    data = payload(kind)
    # No context or engine dependencies are needed to reject the classification.
    with pytest.raises(ValueError, match="trial_type"):
        gate.validate_round(data, tmp_path, 1)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_type",
        "missing_directions",
        "bad_directions",
        "bad_direction",
        "missing_id",
        "blank_id",
        "bad_id",
        "duplicate_id",
        "dangling_id",
        "missing_reference",
        "bad_reference",
        "conflict",
        "old_constraint",
        "old_flag",
        "old_candidate_flag",
        "bad_candidates",
        "bad_candidate",
    ],
)
def test_invalid_direction_contract_is_rejected(tmp_path, problem):
    data = payload()
    direction = data["directions"][0]
    candidate = data["candidates"][0]
    if problem == "missing_type":
        direction.pop("trial_type")
    elif problem == "missing_directions":
        data.pop("directions")
    elif problem == "bad_directions":
        data["directions"] = {}
    elif problem == "bad_direction":
        data["directions"] = [None]
    elif problem == "missing_id":
        direction.pop("direction_id")
    elif problem == "blank_id":
        direction["direction_id"] = " "
    elif problem == "bad_id":
        direction["direction_id"] = []
    elif problem == "duplicate_id":
        data["directions"].append(copy.deepcopy(direction))
    elif problem == "dangling_id":
        candidate["direction_id"] = "missing"
    elif problem == "missing_reference":
        candidate.pop("direction_id")
    elif problem == "bad_reference":
        candidate["direction_id"] = []
    elif problem == "conflict":
        candidate["trial_type"] = "diagnosis"
    elif problem == "old_constraint":
        direction["constraints"] = ["diagnostic_only"]
    elif problem == "old_flag":
        direction["diagnostic_only"] = True
    elif problem == "old_candidate_flag":
        candidate["diagnostic_only"] = True
    elif problem == "bad_candidates":
        data["candidates"] = {}
    elif problem == "bad_candidate":
        data["candidates"] = [None]
    with pytest.raises(ValueError):
        gate.validate_round(data, tmp_path, 1)

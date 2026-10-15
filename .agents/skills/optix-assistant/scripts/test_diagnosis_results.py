"""Export-only diagnosis isolation with the unchanged engine."""

import contextlib
import copy
import json
import sys
from types import SimpleNamespace

import pytest

import export_best_config as export


def save(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def prepare(tmp_path, round_no=1):
    config = tmp_path / "config.toml"
    config.write_text('[vllm.command]' + chr(10) + 'model="model"' + chr(10) + 'served_model_name="served"')
    save(
        tmp_path / f"context.round-{round_no}.json",
        {
            "engine": "vllm",
            "source": {"config_toml": str(config)},
            "search_space": {"parameters": [], "constants": []},
        },
    )
    return {
        "round": round_no,
        "directions": [
            {"direction_id": "d", "type": "local_search", "trial_type": "diagnosis"},
            {"direction_id": "o", "type": "local_search", "trial_type": "optimization"},
        ],
        "candidates": [
            {"candidate_id": "probe", "direction_id": "d", "params": {}},
            {"candidate_id": "normal", "direction_id": "o", "params": {}},
        ],
    }


def trial(cid, fitness, **kwargs):
    return {"candidate_id": cid, "fitness": fitness, "params": {}, "slo_violated": False, **kwargs}


def run_export(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["export_best_config", "--run-dir", str(tmp_path)])
    export.main()
    return json.loads((tmp_path / "best_result.json").read_text())


def test_unchanged_engine_best_diagnosis_is_excluded_from_export(tmp_path, monkeypatch):
    from optix.config.config import PerformanceIndex
    from optix.optimizer.agentic.strategies import AgentCandidateStrategy

    save(tmp_path / "candidates.round-1.json", prepare(tmp_path))
    speeds = iter([1.0, 2.0])
    scheduler = SimpleNamespace(
        run=lambda *a: PerformanceIndex(generate_speed=next(speeds)),
        error_info="",
        last_outcome=None,
        save_result=lambda **kw: None,
        disable_early_exit=contextlib.nullcontext,
    )
    result = AgentCandidateStrategy(tmp_path).run(scheduler, [], {})
    assert result.best_candidate.candidate_id == "probe"
    assert all("trial_type" not in t.metadata for t in result.trials)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    best = run_export(tmp_path, monkeypatch)
    assert best["candidate_id"] == "normal" and best["fitness"] == 2
    assert best["trial_count"] == best["round_count"] == 1
    assert json.loads((tmp_path / "optimizer_config_handoff.json").read_text())["fitness"] == 2
    assert (tmp_path / "serve_command.json").exists()
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())


def test_same_candidate_id_in_different_rounds_uses_own_direction(tmp_path, monkeypatch):
    for r, kind, fitness in [(1, "optimization", 2), (2, "diagnosis", 1)]:
        data = prepare(tmp_path, r)
        data["directions"][0]["trial_type"] = kind
        save(tmp_path / f"candidates.round-{r}.json", data)
        save(tmp_path / f"results.round-{r}.json", {"trials": [trial("probe", fitness)]})
    best = run_export(tmp_path, monkeypatch)
    assert best["source_ref"] == "results.round-1.json#trials[0]"
    assert best["trial_count"] == best["round_count"] == 1


def test_diagnosis_only_run_has_no_export(tmp_path, monkeypatch):
    save(tmp_path / "candidates.round-1.json", prepare(tmp_path))
    save(tmp_path / "results.round-1.json", {"trials": [trial("probe", 1)]})
    with pytest.raises(SystemExit, match="optimization trial"):
        run_export(tmp_path, monkeypatch)
    assert not (tmp_path / "best_result.json").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "missing_archive",
        "malformed_archive",
        "wrong_round",
        "unknown_candidate",
        "duplicate_candidate",
        "duplicate_direction",
        "unknown_direction",
        "missing_direction",
        "invalid_type",
        "missing_type",
        "candidate_conflict",
        "result_conflict",
        "old_marker",
    ],
)
def test_ambiguous_classification_fails_before_export(tmp_path, monkeypatch, capsys, problem):
    data = prepare(tmp_path)
    result = trial("probe", 1)
    if problem == "wrong_round":
        data["round"] = 2
    elif problem == "unknown_candidate":
        result["candidate_id"] = "missing"
    elif problem == "duplicate_candidate":
        data["candidates"].append(copy.deepcopy(data["candidates"][0]))
    elif problem == "duplicate_direction":
        data["directions"].append(copy.deepcopy(data["directions"][0]))
    elif problem == "unknown_direction":
        data["candidates"][0]["direction_id"] = "missing"
    elif problem == "missing_direction":
        data["candidates"][0].pop("direction_id")
    elif problem == "invalid_type":
        data["directions"][0]["trial_type"] = "typo"
    elif problem == "missing_type":
        data["directions"][0].pop("trial_type")
    elif problem == "candidate_conflict":
        data["candidates"][0]["trial_type"] = "optimization"
    elif problem == "result_conflict":
        result["metadata"] = {"trial_type": "optimization"}
    elif problem == "old_marker":
        data["directions"][0]["constraints"] = ["diagnostic_only"]
    path = tmp_path / "candidates.round-1.json"
    if problem != "missing_archive":
        save(path, data)
        if problem == "malformed_archive":
            path.write_text("{")
    save(tmp_path / "results.round-1.json", {"trials": [result]})
    with pytest.raises(SystemExit) as exc:
        run_export(tmp_path, monkeypatch)
    assert exc.value.code == 1
    assert "Cannot classify trial results" in capsys.readouterr().err
    assert not (tmp_path / "best_result.json").exists()


def test_metadata_only_historical_results_still_filter_diagnosis(tmp_path, monkeypatch):
    prepare(tmp_path)
    save(
        tmp_path / "results.round-1.json",
        {
            "trials": [
                trial("probe", 1, metadata={"trial_type": "diagnosis"}),
                trial("normal", 2, metadata={"trial_type": "optimization"}),
            ]
        },
    )
    assert run_export(tmp_path, monkeypatch)["candidate_id"] == "normal"


@pytest.mark.parametrize("with_directions", [False, True])
def test_legacy_candidate_without_type_is_optimization(tmp_path, monkeypatch, with_directions):
    data = prepare(tmp_path)
    if with_directions:
        for d in data["directions"]:
            d.pop("trial_type")
    else:
        data = {"round": 1, "candidates": [{"candidate_id": "normal", "params": {}}]}
    save(tmp_path / "candidates.round-1.json", data)
    save(tmp_path / "results.round-1.json", {"trials": [trial("normal", 2)]})
    assert run_export(tmp_path, monkeypatch)["candidate_id"] == "normal"


def test_slo_preference_and_early_exit_are_preserved(tmp_path, monkeypatch):
    data = prepare(tmp_path)
    for cid in ["noncompliant", "exit"]:
        data["candidates"].append({"candidate_id": cid, "direction_id": "o", "params": {}})
    save(tmp_path / "candidates.round-1.json", data)
    save(
        tmp_path / "results.round-1.json",
        {
            "trials": [
                trial("probe", 0.1),
                trial("exit", 0.2, early_exit=True),
                trial("noncompliant", 0.5, slo_violated=True),
                trial("normal", 2),
            ]
        },
    )
    best = run_export(tmp_path, monkeypatch)
    assert best["candidate_id"] == "normal" and best["selection_policy"] == "slo_compliant_best"

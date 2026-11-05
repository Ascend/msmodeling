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


import json
import time
import pytest
from optix.optimizer.agentic.pd_state import check_search_budget, search_summary


def record(root, name, **values):
    path = root / "pd_deployments" / name
    path.mkdir(parents=True)
    data = {
        "purpose": "search",
        "started_at": time.time() - 5,
        "finished_at": time.time(),
        "outcome": "measured",
        "layout": {"roles": {"prefill": {"devices": 2, "tp": 1, "dp": 1}, "decode": {"devices": 2, "tp": 1, "dp": 1}}},
    }
    data.update(values)
    (path / "deployment.json").write_text(json.dumps(data), encoding="utf-8")


def test_budget_survives_restart_and_counts_failures(tmp_path):
    limits = {"max_trials": 2, "time_limit_seconds": None}
    check_search_budget(tmp_path, limits)
    record(tmp_path, "first", outcome="failed")
    assert check_search_budget(tmp_path, limits)["attempts"] == 1
    record(tmp_path, "second")
    with pytest.raises(ValueError, match="exhausted"):
        check_search_budget(tmp_path, limits)


def test_budget_cannot_change_after_restart(tmp_path):
    check_search_budget(tmp_path, {"max_trials": 2})
    with pytest.raises(ValueError, match="changed"):
        check_search_budget(tmp_path, {"max_trials": 3})


def test_only_measured_layouts_cover_plan(tmp_path):
    target = {"prefill": {"devices": 2}, "decode": {"devices": 2}}
    (tmp_path / "pd_search_plan.json").write_text(json.dumps({"layouts": [target]}), encoding="utf-8")
    record(tmp_path, "failed", outcome="failed")
    assert search_summary(tmp_path)["uncovered_layouts"] == [target]
    record(tmp_path, "passed")
    summary = search_summary(tmp_path)
    assert summary["coverage_complete"]
    assert summary["duplicate_layout_attempts"] == 1


def test_baseline_does_not_consume_search_budget(tmp_path):
    record(tmp_path, "baseline", purpose="baseline")
    assert check_search_budget(tmp_path, {"max_trials": 1})["attempts"] == 0


def test_no_declared_plan_never_claims_complete_coverage(tmp_path):
    record(tmp_path, "passed")
    assert not search_summary(tmp_path)["coverage_complete"]


@pytest.mark.parametrize("target", [{}, {"prefill": {}}, {"prefill": {}, "decode": {}}])
def test_empty_coverage_target_cannot_claim_completion(tmp_path, target):
    (tmp_path / "pd_search_plan.json").write_text(json.dumps({"layouts": [target]}), encoding="utf-8")
    with pytest.raises(ValueError):
        search_summary(tmp_path)


def test_verification_budget_is_separate_and_frozen(tmp_path):
    from optix.optimizer.agentic.pd_state import check_verification_budget

    policy = {"required_repetitions": 2, "aggregation": "median_fitness", "max_trials": 4, "time_limit_seconds": 3600}
    path = tmp_path / "pd_verification_budget.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    record(tmp_path, "search")
    assert check_verification_budget(tmp_path) == policy
    for i in range(4):
        record(tmp_path, f"repeat-{i}", purpose="verification_winner")
    with pytest.raises(ValueError, match="exhausted"):
        check_verification_budget(tmp_path)
    policy["max_trials"] = 6
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        check_verification_budget(tmp_path)


def test_search_budget_excludes_bootstrap_and_verification(tmp_path):
    for purpose in ("bootstrap", "baseline", "verification_baseline", "verification_winner"):
        record(tmp_path, purpose, purpose=purpose)
    assert check_search_budget(tmp_path, {"max_trials": 1})["attempts"] == 0


def test_interrupted_attempt_stops_charging_downtime_but_counts_attempt(tmp_path, monkeypatch):
    from optix.optimizer.agentic import pd_state

    monkeypatch.setattr(pd_state.time, "time", lambda: 10000)
    record(tmp_path, "interrupted", started_at=1, outcome="started", cleanup_confirmed=True)
    path = tmp_path / "pd_deployments/interrupted/deployment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("finished_at")
    pd_state.finish_interrupted_attempt(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    assert data["outcome"] == "aborted" and data["elapsed_unknown"]
    monkeypatch.setattr(pd_state.time, "time", lambda: 20000)
    summary = pd_state.check_search_budget(tmp_path, {"max_trials": 2, "time_limit_seconds": 10})
    assert summary["attempts"] == 1 and summary["elapsed_seconds"] == 0
    with pytest.raises(ValueError, match="cleanup"):
        pd_state.finish_interrupted_attempt({"started_at": 1, "cleanup_confirmed": False})


def test_active_attempt_still_consumes_workflow_time(tmp_path, monkeypatch):
    from optix.optimizer.agentic import pd_state
    from tests.regression.optix.test_pd_agent.test_five_step_contracts import policy

    policy(tmp_path)
    pd_state.check_workflow_budget(tmp_path, "search")
    record(tmp_path, "running", started_at=90)
    path = tmp_path / "pd_deployments/running/deployment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("finished_at")
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(pd_state.time, "time", lambda: 100)
    assert pd_state.workflow_seconds_remaining(tmp_path, "search") == 90


def test_budget_cache_only_for_one_active_matching_attempt(tmp_path, monkeypatch):
    from optix.optimizer.agentic import pd_state
    from tests.regression.optix.test_pd_agent.test_five_step_contracts import policy

    policy(tmp_path)
    pd_state.check_workflow_budget(tmp_path, "search")
    record(tmp_path, "one", started_at=90)
    path = tmp_path / "pd_deployments/one/deployment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("finished_at")
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(pd_state.time, "time", lambda: 100)
    assert pd_state.workflow_seconds_remaining(tmp_path, "search", cacheable=True) == (90, True)
    record(tmp_path, "two", started_at=95)
    second = tmp_path / "pd_deployments/two/deployment.json"
    data = json.loads(second.read_text(encoding="utf-8"))
    data.pop("finished_at")
    second.write_text(json.dumps(data), encoding="utf-8")
    assert pd_state.workflow_seconds_remaining(tmp_path, "search", cacheable=True)[1] is False


def test_scheduler_uses_deadline_then_refreshes_on_new_case(monkeypatch, tmp_path):
    from optix.optimizer import scheduler as module
    from optix.optimizer.agentic import pd_state
    from unittest.mock import MagicMock

    scheduler = module.Scheduler.__new__(module.Scheduler)
    scheduler.pd_workflow_root = tmp_path
    scheduler.pd_trial_purpose = "search"
    scheduler._pd_budget_deadline = None
    scheduler.engine = "vllm"
    scheduler.data_storage = MagicMock()
    scheduler.data_storage.next_case_id.return_value = "case-1"
    remaining = MagicMock(return_value=(10, True))
    monkeypatch.setattr(pd_state, "workflow_seconds_remaining", remaining)
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    scheduler._check_workflow_time()
    clock[0] = 105.0
    scheduler._check_workflow_time()
    assert remaining.call_count == 1
    scheduler._new_case_id()
    scheduler._check_workflow_time()
    assert remaining.call_count == 2
    clock[0] = 116.0
    with pytest.raises(pd_state.WorkflowBudgetExceeded):
        scheduler._check_workflow_time()

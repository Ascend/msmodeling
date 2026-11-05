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


from types import SimpleNamespace
from unittest.mock import patch
import pytest
from optix.optimizer.agentic.orchestrator import AgentOptimizer


def optimizer(tmp_path):
    optimizer = AgentOptimizer.__new__(AgentOptimizer)
    optimizer.run_dir = tmp_path
    optimizer.config = SimpleNamespace(max_trials=2, time_limit_minutes=1, max_rounds=1)
    optimizer.pso = SimpleNamespace(scheduler=SimpleNamespace(engine="vllm_pd"))
    return optimizer


def test_verification_uses_own_round_beyond_search_budget(tmp_path):
    opt = optimizer(tmp_path)
    (tmp_path / "candidates.round-1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "candidates.round-5.json").write_text("{}", encoding="utf-8")
    context = {"deployment": {"round_manifest": {"round": 5, "purpose": "verification_winner"}}}
    with patch("optix.optimizer.agentic.orchestrator.AgentCandidateStrategy") as strategy:
        strategy.return_value.run.return_value = SimpleNamespace(best_fitness=1)
        opt._run_rounds(set(), float("inf"), [], context)
        assert strategy.call_args.kwargs["round_index"] == 5
        assert strategy.call_count == 1


def test_search_cannot_bypass_round_budget(tmp_path):
    opt = optimizer(tmp_path)
    with pytest.raises(ValueError, match="round budget"):
        opt._run_rounds(set(), float("inf"), [], {"deployment": {"round_manifest": {"round": 2, "purpose": "search"}}})


@pytest.mark.parametrize("purpose", ["bootstrap", "search", "verification_winner"])
def test_pd_main_flow_preserves_purpose_and_freezes_baseline(tmp_path, purpose):
    import json
    from unittest.mock import Mock

    opt = optimizer(tmp_path)
    opt.pso.scheduler.simulator = object()
    opt.pso.target_field = []
    opt.pso.use_request_rate_calibration = False
    context = {"deployment": {"round_manifest": {"round": 7, "purpose": purpose, "candidate_id": "candidate-7"}}}
    opt._load_multihost_context = Mock(return_value=None)
    opt._load_search_space_context = Mock(return_value=context)
    opt._run_rounds = Mock()
    (tmp_path / "pd_workflow.json").write_text("{}", encoding="utf-8")
    layout = {"roles": {"prefill": {"tp": 2, "dp": 1}, "decode": {"tp": 1, "dp": 2}}}
    identities = []

    def baseline():
        identities.append(dict(opt.pso.scheduler.pd_candidate_identity))
        (tmp_path / "baseline.json").write_text(json.dumps({"pd_launch_plan": {"layout": layout}}), encoding="utf-8")

    opt._ensure_baseline = Mock(side_effect=baseline)
    with patch("optix.optimizer.agentic.pd.validate_simulator") as validate:
        opt._run()
    validate.assert_called_once_with(context, opt.pso.scheduler.simulator)
    assert context["pd_budget"] == {"max_trials": 2, "time_limit_seconds": 60}
    assert opt.pso.scheduler.pd_archive_root == tmp_path / "pd_deployments"
    assert identities == [{"round": 7, "candidate_id": "candidate-7" if purpose == "bootstrap" else "baseline"}]
    if purpose == "bootstrap":
        opt._run_rounds.assert_not_called()
        assert not (tmp_path / "pd_instance_specs.json").exists()
    else:
        opt._run_rounds.assert_called_once_with(set(), float("inf"), [], context)
        assert opt.pso.scheduler.pd_trial_purpose == purpose
        assert json.loads((tmp_path / "pd_instance_specs.json").read_text(encoding="utf-8")) == layout["roles"]


def test_resume_scan_only_skips_successful_finite_trials(tmp_path):
    import json

    opt = optimizer(tmp_path)
    opt.config.max_rounds = 5
    for number, trials in [
        (1, [{"fitness": 3}, {"fitness": 2}]),
        (2, [{"fitness": 1, "error": "failed"}]),
        (3, [{"fitness": float("inf")}]),
    ]:
        (tmp_path / f"results.round-{number}.json").write_text(json.dumps({"trials": trials}), encoding="utf-8")
    (tmp_path / "results.round-4.json").write_text("broken", encoding="utf-8")
    completed, best = opt._scan_completed_rounds()
    assert completed == {1}
    assert best == 2


@pytest.mark.parametrize("calibration", [False, True])
def test_pinning_restores_original_search_fields(tmp_path, calibration):
    from optix.config.config import OptimizerConfigField

    opt = optimizer(tmp_path)
    original = [
        OptimizerConfigField(name="CONCURRENCY", min=1, max=8, value=2, dtype="int"),
        OptimizerConfigField(name="REQUESTRATE", min=0, max=100, value=3, dtype="int"),
    ]
    opt.pso.target_field = original
    opt.pso.use_request_rate_calibration = calibration
    fixed = opt._pin_concurrency()
    assert opt.pso.target_field is original
    assert [field.value for field in original] == [2, 3]
    assert fixed[0].value == (8 if calibration else 2)
    assert fixed[0].constant == (8 if calibration else None)
    assert fixed[1].value == fixed[1].constant == 0


def test_non_pd_main_flow_keeps_search_context(tmp_path):
    from unittest.mock import Mock

    opt = optimizer(tmp_path)
    opt.pso.scheduler.engine = "vllm"
    opt.pso.target_field = []
    opt.pso.use_request_rate_calibration = False
    context = {"search_space": {"parameters": []}}
    opt._load_multihost_context = Mock(return_value=None)
    opt._load_search_space_context = Mock(return_value=context)
    opt._ensure_baseline = Mock()
    opt._run_rounds = Mock()
    opt._run()
    opt._ensure_baseline.assert_called_once()
    opt._run_rounds.assert_called_once_with(set(), float("inf"), [], context)


def test_bootstrap_followed_by_business_baseline_creates_checkpoint(tmp_path):
    import json
    from unittest.mock import Mock
    from optix.config.config import PerformanceIndex

    opt = optimizer(tmp_path)
    scheduler = opt.pso.scheduler
    scheduler.simulator = object()
    scheduler.simulate_launch_plan = {
        "layout": {"roles": {"prefill": {"tp": 1, "dp": 1}, "decode": {"tp": 1, "dp": 1}}}
    }
    scheduler.simulate_run_info = []
    opt.pso.default_res = PerformanceIndex(
        throughput=10, success_rate=1, time_to_first_token=0.1, time_per_output_token=0.01
    )
    opt.pso.default_fitness = 1
    opt.pso.prepare_plugin = Mock()
    opt._load_multihost_context = Mock(return_value=None)
    opt._run_rounds = Mock()
    opt._pin_concurrency = Mock(return_value=[])
    (tmp_path / "pd_workflow.json").write_text("{}", encoding="utf-8")
    context = {"deployment": {"round_manifest": {"round": 1, "purpose": "bootstrap", "candidate_id": "c"}}}
    opt._load_search_space_context = Mock(return_value=context)
    with patch("optix.optimizer.agentic.pd.validate_simulator"):
        opt._run()
        assert not (tmp_path / "baseline.json").exists()
        context["deployment"]["round_manifest"] = {"round": 2, "purpose": "search", "candidate_id": "c2"}
        opt._run()
    saved = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert saved["pd_launch_plan"]["layout"] == scheduler.simulate_launch_plan["layout"]
    assert scheduler.pd_trial_purpose == "search"


def test_pd_manifest_missing_round_is_rejected(tmp_path):
    opt = optimizer(tmp_path)
    with pytest.raises(ValueError, match="round_manifest is missing required round"):
        opt._run_rounds(set(), float("inf"), [], {"deployment": {"round_manifest": {"purpose": "search"}}})


@pytest.mark.parametrize("baseline", [None, "{}", "{broken"])
def test_workflow_checkpoint_missing_or_invalid_after_baseline(tmp_path, baseline):
    from unittest.mock import Mock

    opt = optimizer(tmp_path)
    opt.pso.scheduler.simulator = object()
    opt._load_multihost_context = Mock(return_value=None)
    opt._load_search_space_context = Mock(return_value={"deployment": {}})
    opt._ensure_baseline = Mock()
    (tmp_path / "pd_workflow.json").write_text("{}", encoding="utf-8")
    if baseline is not None:
        (tmp_path / "baseline.json").write_text(baseline, encoding="utf-8")
    with patch("optix.optimizer.agentic.pd.validate_simulator"):
        with pytest.raises(ValueError, match="PD workflow baseline checkpoint .* missing or invalid"):
            opt._run()

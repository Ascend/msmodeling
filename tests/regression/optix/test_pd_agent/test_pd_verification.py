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


import importlib.util
from pathlib import Path
from copy import deepcopy
import pytest


ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "pd_summary", ROOT / ".agents/skills/optix-assistant/scripts/summarize_pd_search.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def facts():
    expected = {
        "config_sha256": "hash",
        "params": {"x": 1},
        "benchmark_spec": {"requests": 100},
        "purpose": "verification_winner",
    }
    record = {
        "reference": "results.round-3.json#trials[0]",
        "context": {"benchmark_spec": {"requests": 100}},
        "trial": {
            "params": {"x": 1},
            "fitness": 2.0,
            "slo_violated": False,
            "metadata": {
                "launch_plan": {"config_sha256": "hash"},
                "deployment_proposal": {"purpose": "verification_winner"},
            },
            "performance": {"throughput": 10},
        },
    }
    return expected, record


def test_repetitions_report_median_and_range():
    expected, first = facts()
    second = deepcopy(first)
    second["reference"] = "results.round-4.json#trials[0]"
    second["trial"]["fitness"] = 4.0
    result = module.repetition_summary([first, second], expected, 2)
    assert result["status"] == "verified"
    assert result["median_fitness"] == 3.0
    assert result["fitness_range"] == [2.0, 4.0]


def test_insufficient_repetitions_stay_pending():
    expected, first = facts()
    assert module.repetition_summary([first], expected, 2)["status"] == "pending"


@pytest.mark.parametrize("change", ["config", "params", "load", "purpose", "slo", "early"])
def test_incomparable_or_search_measurements_rejected(change):
    expected, first = facts()
    if change == "config":
        first["trial"]["metadata"]["launch_plan"]["config_sha256"] = "other"
    elif change == "params":
        first["trial"]["params"] = {"x": 2}
    elif change == "load":
        first["context"]["benchmark_spec"] = {"requests": 200}
    elif change == "purpose":
        first["trial"]["metadata"]["deployment_proposal"]["purpose"] = "search"
    elif change == "slo":
        first["trial"]["slo_violated"] = True
    else:
        first["trial"]["early_exit"] = True
    with pytest.raises(ValueError):
        module.repetition_summary([first], expected, 2)


def test_duplicate_evidence_rejected():
    expected, first = facts()
    with pytest.raises(ValueError, match="distinct"):
        module.repetition_summary([first, first], expected, 2)


def test_diagnosis_does_not_count_as_verification():
    expected, record = facts()
    record["trial"]["metadata"]["trial_type"] = "diagnosis"
    with pytest.raises(ValueError, match="Diagnosis"):
        module.repetition_summary([record], expected, 2)

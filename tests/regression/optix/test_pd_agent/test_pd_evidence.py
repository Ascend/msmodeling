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


import hashlib
import json
import pytest
from optix.optimizer.agentic.pd_evidence import generate_replay, verification_anchor


def archive(root):
    text = "optimizer_strategy=\"agent\"\r\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    context = {
        "deployment": {"config_sha256": digest},
        "benchmark_spec": {"requests": 100},
        "search_space": {"parameters": [], "constants": []},
    }
    launch = {"config_toml": text, "config_sha256": digest, "layout": {"roles": {}}}
    params = {"MAX_NUM_SEQS_prefill": 16}
    (root / "baseline.json").write_text(
        json.dumps({"pd_context": context, "pd_launch_plan": launch, "pd_parameters": params}), encoding="utf-8"
    )
    trial = {
        "candidate_id": "winner",
        "params": params,
        "metadata": {"launch_plan": launch, "deployment_proposal": {"purpose": "search"}},
    }
    (root / "results.round-1.json").write_text(json.dumps({"trials": [trial]}), encoding="utf-8")
    (root / "context.round-1.json").write_text(json.dumps(context), encoding="utf-8")
    (root / "best_result.json").write_text(
        json.dumps({**trial, "source_ref": "results.round-1.json#trials[0]"}), encoding="utf-8"
    )
    return text, params


@pytest.mark.parametrize("side", ["baseline", "winner"])
def test_replay_preserves_original_bytes_and_evidence(side, tmp_path):
    text, params = archive(tmp_path)
    target = tmp_path / "replay.toml"
    result = generate_replay(tmp_path, side, {"params": params}, target)
    assert target.read_bytes() == text.encode()
    assert result["config_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert result["replay_search_space"] == {"parameters": [], "constants": []}


def test_changed_parameters_rejected_without_creating_config(tmp_path):
    archive(tmp_path)
    target = tmp_path / "bad.toml"
    with pytest.raises(ValueError, match="complete effective"):
        generate_replay(tmp_path, "baseline", {"params": {}}, target)
    assert not target.exists()


def test_selected_winner_must_match_original_trial(tmp_path):
    archive(tmp_path)
    path = tmp_path / "best_result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["params"] = {"other": 2}
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="archived trial"):
        verification_anchor(tmp_path, "winner")


def test_repetition_cannot_become_search_winner(tmp_path):
    archive(tmp_path)
    path = tmp_path / "results.round-1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["trials"][0]["metadata"]["deployment_proposal"]["purpose"] = "verification_winner"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="repetition"):
        verification_anchor(tmp_path, "winner")


def test_corrupt_baseline_archive_rejected(tmp_path):
    archive(tmp_path)
    path = tmp_path / "baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pd_launch_plan"]["config_toml"] = "different"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        verification_anchor(tmp_path, "baseline")

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


"""Bind PD verification to archived baseline and selected search winner."""

import hashlib
import json
from pathlib import Path


def verification_anchor(run_dir, side):
    root = Path(run_dir)
    if side == "baseline":
        saved = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
        context = saved["pd_context"]
        launch = saved["pd_launch_plan"]
        params = saved["pd_parameters"]
        reference = "baseline.json"
    elif side == "winner":
        saved = json.loads((root / "best_result.json").read_text(encoding="utf-8"))
        reference = saved["source_ref"]
        filename, index = reference.split("#trials[")
        if Path(filename).name != filename or not filename.startswith("results.round-"):
            raise ValueError("Winner source must be an archived search result")
        index = int(index.rstrip("]"))
        if index < 0:
            raise ValueError("Winner trial index must be nonnegative")
        trial = json.loads((root / filename).read_text(encoding="utf-8"))["trials"][index]
        if trial["params"] != saved["params"] or trial["candidate_id"] != saved["candidate_id"]:
            raise ValueError("Selected PD winner no longer matches its archived trial")
        binding = trial.get("metadata", {}).get("deployment_proposal") or {}
        if binding.get("purpose", "search") != "search":
            raise ValueError("Verification cannot select a repetition as the search winner")
        context = json.loads(
            (root / filename.replace("results.round-", "context.round-", 1)).read_text(encoding="utf-8")
        )
        launch = trial["metadata"]["launch_plan"]
        params = trial["params"]
    else:
        raise ValueError("PD verification side must be baseline or winner")
    text = launch["config_toml"]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != launch["config_sha256"] or digest != context["deployment"]["config_sha256"]:
        raise ValueError("Archived verification configuration is inconsistent")
    return {
        "reference": reference,
        "context": context,
        "launch_plan": launch,
        "expected": {
            "purpose": "verification_" + side,
            "config_sha256": digest,
            "params": params,
            "benchmark_spec": context["benchmark_spec"],
        },
    }


def generate_replay(run_dir, side, candidate, output):
    """Replay the exact measured configuration, rather than normalizing it again."""
    anchor = verification_anchor(run_dir, side)
    if candidate.get("params") != anchor["expected"]["params"]:
        raise ValueError("PD replay candidate must supply the original complete effective parameters")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = anchor["launch_plan"]["config_toml"]
    with output.open("xb") as stream:
        stream.write(text.encode("utf-8"))
    return {
        "config_toml": str(output.resolve()),
        "config_sha256": anchor["expected"]["config_sha256"],
        "layout": anchor["launch_plan"]["layout"],
        "params": anchor["expected"]["params"],
        "replay_source": anchor["reference"],
        "replay_search_space": anchor["context"].get("search_space"),
    }

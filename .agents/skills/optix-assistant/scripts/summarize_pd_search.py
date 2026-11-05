#!/usr/bin/env python3

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

"""Ownership: agent. Summarize PD coverage and independently repeated measurements."""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from optix.optimizer.agentic.pd_state import search_summary
from optix.optimizer.agentic.pd_evidence import verification_anchor


def repetition_summary(records, expected, required):
    """Accept only matching deployment, parameters and workload, with distinct evidence."""
    accepted = []
    seen = set()
    for record in records:
        reference = record["reference"]
        if reference in seen:
            raise ValueError("Repeated measurement references must be distinct")
        seen.add(reference)
        trial, context = record["trial"], record["context"]
        plan = (trial.get("metadata") or {}).get("launch_plan") or {}
        if (trial.get("metadata") or {}).get("trial_type") == "diagnosis":
            raise ValueError("Diagnosis trials cannot count as PD verification repetitions")
        binding = (trial.get("metadata") or {}).get("deployment_proposal") or {}
        if binding.get("purpose") != expected.get("purpose") or expected.get("purpose") not in {
            "verification_baseline",
            "verification_winner",
        }:
            raise ValueError("Independent repetitions require an explicit verification purpose")
        if trial.get("error") or trial.get("slo_violated") is not False:
            raise ValueError("PD repetitions must be successful and SLO-compliant")
        if trial.get("early_exit") or (trial.get("performance") or {}).get("early_exit"):
            raise ValueError("Early-exit measurements cannot verify a deployment")
        if plan.get("config_sha256") != expected["config_sha256"]:
            raise ValueError("Repeated PD deployment configuration does not match")
        if trial.get("params") != expected["params"]:
            raise ValueError("Repeated PD effective parameters do not match")
        if context.get("benchmark_spec") != expected["benchmark_spec"]:
            raise ValueError("Repeated PD workload does not match")
        fitness = trial.get("fitness")
        if isinstance(fitness, bool) or not isinstance(fitness, (int, float)) or not isfinite(fitness):
            raise ValueError("Repeated PD fitness must be finite")
        accepted.append(
            {
                "reference": reference,
                "fitness": fitness,
                "performance": trial.get("performance"),
            }
        )
    values = [r["fitness"] for r in accepted]
    return {
        "status": "verified" if len(values) >= required else "pending",
        "required_repetitions": required,
        "observed_repetitions": len(values),
        "aggregation": "median_fitness",
        "median_fitness": median(values) if values else None,
        "fitness_range": [min(values), max(values)] if values else None,
        "measurements": accepted,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--verification-plan",
        help="JSON with baseline/winner expected facts and measurement references",
    )
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    summary = search_summary(root)
    verification = {
        "status": "pending",
        "reason": "No independent repetition evidence supplied",
    }
    if args.verification_plan:
        plan = json.loads(Path(args.verification_plan).read_text(encoding="utf-8"))
        declared = json.loads((root / "pd_verification_budget.frozen.json").read_text(encoding="utf-8"))
        required = plan["required_repetitions"]
        if required != declared["required_repetitions"] or declared["aggregation"] != "median_fitness":
            raise ValueError("Verification summary must use the declared repetition policy")
        if isinstance(required, bool) or not isinstance(required, int) or required < 2:
            raise ValueError("Declare at least two independent repetitions per deployment")
        sides = {}
        all_refs = set()
        for side in ("baseline", "winner"):
            item = plan[side]
            anchor = verification_anchor(root, side)
            if item["expected"] != anchor["expected"]:
                raise ValueError("PD verification expectations differ from the archived baseline or winner")
            records = []
            for reference in item["references"]:
                if reference in all_refs:
                    raise ValueError("Baseline and winner require distinct repetition records")
                all_refs.add(reference)
                filename, index = reference.split("#trials[")
                if Path(filename).name != filename or not filename.startswith("results.round-"):
                    raise ValueError("Verification references must name archived round results")
                index = int(index.rstrip("]"))
                if index < 0:
                    raise ValueError("Trial index must be nonnegative")
                trial = json.loads((root / filename).read_text(encoding="utf-8"))["trials"][index]
                from export_best_config import _candidate_trial_types, _is_optimization_trial

                candidates_path = root / filename.replace("results.round-", "candidates.round-", 1)
                if not _is_optimization_trial(trial, _candidate_trial_types(candidates_path), candidates_path):
                    raise ValueError("Diagnosis trials cannot count as PD verification repetitions")
                context = json.loads(
                    (root / filename.replace("results.round-", "context.round-", 1)).read_text(encoding="utf-8")
                )
                records.append({"reference": reference, "trial": trial, "context": context})
            sides[side] = repetition_summary(records, item["expected"], required)
        if plan["baseline"]["expected"]["benchmark_spec"] != plan["winner"]["expected"]["benchmark_spec"]:
            raise ValueError("Baseline and winner workloads must match")
        verification = {
            "status": "verified" if all(s["status"] == "verified" for s in sides.values()) else "pending",
            "sides": sides,
            "winner_config_sha256": plan["winner"]["expected"]["config_sha256"],
            "winner_params": plan["winner"]["expected"]["params"],
            "benchmark_spec": plan["winner"]["expected"]["benchmark_spec"],
        }
    summary["verification"] = verification
    (root / "pd_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

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


"""Persist PD search attempts across controller restarts."""

from contextlib import contextmanager
import json
import time
from pathlib import Path

from .pd import ROLES, layout_fingerprint, positive_integer


def read_attempts(run_dir):
    records = []
    for path in sorted((Path(run_dir) / "pd_deployments").glob("*/deployment.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["archive"] = str(path.parent)
        records.append(record)
    return records


def finish_interrupted_attempt(record):
    """Close an abandoned, verified-clean attempt without charging controller downtime."""
    if "finished_at" not in record:
        if not record.get("cleanup_confirmed"):
            raise ValueError("Cannot finalize an attempt before cleanup is confirmed")
        record.update(finished_at=time.time(), outcome="aborted", elapsed_unknown=True)
    return record


def _elapsed(records, now):
    return sum(
        max(0, record.get("finished_at", now) - record["started_at"])
        for record in records
        if not record.get("elapsed_unknown")
    )


def search_summary(run_dir):
    records = [r for r in read_attempts(run_dir) if r.get("purpose") == "search"]
    now = time.time()
    seconds = _elapsed(records, now)
    fingerprints = [layout_fingerprint(r["layout"]) for r in records]
    declared_path = Path(run_dir) / "pd_search_plan.json"
    targets = []
    if declared_path.exists():
        plan = json.loads(declared_path.read_text(encoding="utf-8"))
        targets = plan.get("layouts", [])
        if not isinstance(targets, list) or any(not isinstance(t, dict) for t in targets):
            raise ValueError("PD search plan layouts must be a list of role layouts")
        for target in targets:
            if set(target) != set(ROLES):
                raise ValueError("Each PD coverage target must declare both roles")
            for role, values in target.items():
                if (
                    not isinstance(values, dict)
                    or not values
                    or not set(values) <= {"devices", "tp", "dp", "instances"}
                ):
                    raise ValueError("PD coverage targets require explicit resource dimensions")
                for key, value in values.items():
                    positive_integer(value, f"{role} coverage {key}")
    covered = []
    for target in targets:
        if any(
            all(
                all(record["layout"]["roles"].get(role, {}).get(k) == v for k, v in values.items())
                for role, values in target.items()
            )
            for record in records
            if record.get("outcome") == "measured"
        ):
            covered.append(target)
    return {
        "attempts": len(records),
        "elapsed_seconds": seconds,
        "unknown_elapsed_attempts": sum(bool(r.get("elapsed_unknown")) for r in records),
        "distinct_layouts": len(set(fingerprints)),
        "duplicate_layout_attempts": len(fingerprints) - len(set(fingerprints)),
        "failed_or_unfinished": sum(r.get("outcome") != "measured" for r in records),
        "declared_layouts": targets,
        "uncovered_layouts": [t for t in targets if t not in covered],
        "coverage_complete": bool(targets) and len(covered) == len(targets),
    }


def check_search_budget(run_dir, limits):
    """Freeze configured limits once, count failed launches as actual attempts."""
    path = Path(run_dir) / "pd_budget.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != limits:
            raise ValueError("PD experiment budget changed across rounds")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(limits, stream, indent=2)
    summary = search_summary(run_dir)
    maximum = limits.get("max_trials")
    seconds = limits.get("time_limit_seconds")
    if maximum is not None and summary["attempts"] >= maximum:
        raise ValueError("PD experiment search trial budget exhausted")
    if seconds is not None and summary["elapsed_seconds"] >= seconds:
        raise ValueError("PD experiment search time budget exhausted")
    return summary


def check_verification_budget(run_dir):
    """Verification has a separately declared budget, never resets search history."""
    root = Path(run_dir)
    limits = json.loads((root / "pd_verification_budget.json").read_text(encoding="utf-8"))
    required = positive_integer(limits["required_repetitions"], "verification repetitions")
    if required < 2 or limits.get("aggregation") != "median_fitness":
        raise ValueError("PD verification requires at least two repeats and median_fitness")
    maximum = positive_integer(limits["max_trials"], "verification trial budget")
    if maximum < required * 2:
        raise ValueError("Verification budget must cover both baseline and winner repeats")
    seconds = positive_integer(limits["time_limit_seconds"], "verification time budget")
    checkpoint = root / "pd_verification_budget.frozen.json"
    if checkpoint.exists():
        if json.loads(checkpoint.read_text(encoding="utf-8")) != limits:
            raise ValueError("PD verification budget changed after execution began")
    else:
        with checkpoint.open("x", encoding="utf-8") as stream:
            json.dump(limits, stream, indent=2)
    records = [r for r in read_attempts(root) if r.get("purpose") in {"verification_baseline", "verification_winner"}]
    elapsed = _elapsed(records, time.time())
    if len(records) >= maximum or elapsed >= seconds:
        raise ValueError("PD independent verification budget exhausted")
    return limits


WORKFLOW_PURPOSES = {
    "bootstrap",
    "baseline",
    "prefill_probe",
    "decode_probe",
    "search",
    "verification_baseline",
    "verification_winner",
}


def check_workflow_budget(run_dir, purpose):
    """Opt-in five-step policy; count every started attempt across child tasks."""
    root = Path(run_dir)
    policy_path = root / "pd_workflow.json"
    if not policy_path.exists():
        if purpose not in {
            "baseline",
            "search",
            "verification_baseline",
            "verification_winner",
        }:
            raise ValueError("PD startup/phase tasks require pd_workflow.json")
        return None
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if type(policy.get("version")) is not int or policy["version"] != 1:
        raise ValueError("Unsupported PD workflow version")
    if purpose not in WORKFLOW_PURPOSES:
        raise ValueError("Unknown PD workflow task purpose")
    budgets = policy["budgets"]
    total = budgets["total"]
    verification = budgets["verification"]
    for scope in (
        "total",
        "bootstrap",
        "baseline",
        "prefill_probe",
        "decode_probe",
        "search",
        "verification",
    ):
        limits = budgets[scope]
        for name in ("max_trials", "time_limit_seconds"):
            if type(limits[name]) is not int:
                raise ValueError(f"PD {scope} {name} must be an integer")
            positive_integer(limits[name], scope + " " + name)
    if verification["max_trials"] < 4:
        raise ValueError("Verification budget must cover at least two baseline and two winner repeats")
    if (
        total["max_trials"] <= verification["max_trials"]
        or total["time_limit_seconds"] <= verification["time_limit_seconds"]
    ):
        raise ValueError("Total budget must leave room for search and reserved verification")
    frozen = root / "pd_workflow.frozen.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf-8")) != policy:
            raise ValueError("PD workflow policy changed after execution began")
    else:
        with frozen.open("x", encoding="utf-8") as stream:
            json.dump(policy, stream, indent=2)
    records = read_attempts(root)
    for path in (root / "pd_tasks").glob("*/attempt.json"):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    now = time.time()

    def elapsed(rows):
        return _elapsed(rows, now)

    verifying = purpose.startswith("verification_")
    scope = "verification" if verifying else purpose
    scoped = [
        r for r in records if (r["purpose"].startswith("verification_") if verifying else r["purpose"] == purpose)
    ]
    if len(scoped) >= budgets[scope]["max_trials"] or elapsed(scoped) >= budgets[scope]["time_limit_seconds"]:
        raise ValueError(f"PD {scope} budget exhausted")
    if len(records) >= total["max_trials"] or elapsed(records) >= total["time_limit_seconds"]:
        raise ValueError("PD total workflow budget exhausted")
    if not verifying:
        exploratory = [r for r in records if not r["purpose"].startswith("verification_")]
        if (
            len(exploratory) >= total["max_trials"] - verification["max_trials"]
            or elapsed(exploratory) >= total["time_limit_seconds"] - verification["time_limit_seconds"]
        ):
            raise ValueError("PD remaining budget is reserved for verification")
    return {
        "purpose": purpose,
        "attempts": len(records),
        "elapsed_seconds": elapsed(records),
    }


def freeze_instance_specs(run_dir, layout):
    """Freeze the first measured business baseline's per-instance resource specs."""
    root = Path(run_dir)
    if not (root / "pd_workflow.json").exists():
        return
    specs = {r: {k: layout["roles"][r][k] for k in ("tp", "dp")} for r in ROLES}
    path = root / "pd_instance_specs.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != specs:
            raise ValueError("PD instance specifications differ from the frozen business baseline")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(specs, stream, indent=2)


def validate_workflow_candidate(run_dir, params, layout, *, bootstrap=False):
    root = Path(run_dir)
    policy_path = root / "pd_workflow.json"
    if not policy_path.exists():
        return
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    for name, value in policy.get("fixed_params", {}).items():
        if params.get(name) != value:
            raise ValueError(f"PD candidate changed user-fixed parameter {name}")
    path = root / "pd_instance_specs.json"
    if path.exists():
        if bootstrap:
            raise ValueError("Startup exploration cannot restart after the business baseline is frozen")
        specs = json.loads(path.read_text(encoding="utf-8"))
        for role in ROLES:
            if any(layout["roles"][role][key] != specs[role][key] for key in ("tp", "dp")):
                raise ValueError("PD five-step search keeps per-instance TP/DP fixed")


class WorkflowBudgetExceeded(TimeoutError):
    """Stop new work while allowing the caller's normal cleanup to run."""


def workflow_seconds_remaining(run_dir, purpose, *, cacheable=False):
    root = Path(run_dir)
    path = root / "pd_workflow.frozen.json"
    if not path.exists():
        return (None, False) if cacheable else None
    budgets = json.loads(path.read_text(encoding="utf-8"))["budgets"]
    records = read_attempts(root)
    records.extend(json.loads(p.read_text(encoding="utf-8")) for p in (root / "pd_tasks").glob("*/attempt.json"))
    now = time.time()

    def elapsed(rows):
        return _elapsed(rows, now)

    verifying = purpose.startswith("verification_")
    scope = "verification" if verifying else purpose
    scoped = [
        r for r in records if (r["purpose"].startswith("verification_") if verifying else r["purpose"] == purpose)
    ]
    remaining = [
        budgets["total"]["time_limit_seconds"] - elapsed(records),
        budgets[scope]["time_limit_seconds"] - elapsed(scoped),
    ]
    if not verifying:
        exploratory = [r for r in records if not r["purpose"].startswith("verification_")]
        remaining.append(
            budgets["total"]["time_limit_seconds"]
            - budgets["verification"]["time_limit_seconds"]
            - elapsed(exploratory)
        )
    seconds = min(remaining)
    # Cache only while exactly one recorded attempt is active. The controller
    # lock serializes work within a run; ambiguous interrupted records fall back
    # to the original live calculation.
    active = [r for r in records if "finished_at" not in r and not r.get("elapsed_unknown")]
    safe = len(active) == 1 and active[0].get("purpose") == purpose
    return (seconds, safe) if cacheable else seconds


@contextmanager
def workflow_time_limit(run_dir, purpose):
    """Linux main-thread deadline; always restore the signal before outer cleanup."""
    seconds = workflow_seconds_remaining(run_dir, purpose)
    if seconds is None:
        yield
        return
    if seconds <= 0:
        raise WorkflowBudgetExceeded("PD workflow time budget exhausted")
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        raise ValueError("PD workflow tasks must execute on the controller main thread")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("Cannot replace an existing controller timer")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise WorkflowBudgetExceeded("PD workflow time budget exhausted; cleanup required")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def require_phase_cleanup(run_dir):
    """An interrupted single-side task must not be hidden by a new PD round."""
    for path in (Path(run_dir) / "pd_tasks").glob("*/attempt.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("cleanup_confirmed") or not record.get("cleanup_identity"):
            raise ValueError(f"Phase cleanup is unconfirmed: {path.parent.name}; inspect resources before resuming")

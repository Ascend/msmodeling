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

"""Prepare startup/layout choices, inspect workflow state, or execute one agent phase probe."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["bootstrap", "layouts", "probe", "status", "recover-phase"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--params", help="Path to JSON file of supplied/fixed parameter values")
    parser.add_argument("--prefill-capacity", help="Current experiment's measured P attempt.json")
    parser.add_argument("--decode-capacity", help="Current experiment's measured D attempt.json")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--allow-unused-devices", action="store_true")
    parser.add_argument("--phase", choices=["prefill", "decode"])
    parser.add_argument("--task-id")
    parser.add_argument(
        "--config",
        help="Phase-only TOML; pd_disagg.<phase> supplies measurement overrides",
    )
    parser.add_argument("--context", help="Path to separate phase context JSON file")
    parser.add_argument("--candidate", help="Path to JSON file containing one agent candidate")
    parser.add_argument("--resource", help="Path to JSON file with ssh_ip and gpu_ids")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    try:
        if args.action == "recover-phase":
            from optix.optimizer.agentic.pd import controller_session
            from optix.optimizer.agentic.pd_phase import recover_phase_attempts

            with controller_session(root):
                recover_phase_attempts(root)
            output = {"phase_cleanup": "confirmed"}
        elif args.action == "status":
            from optix.optimizer.agentic.pd_state import read_attempts, search_summary

            attempts = read_attempts(root)
            phases = [read(p) for p in (root / "pd_tasks").glob("*/attempt.json")]
            policy_path = root / "pd_workflow.frozen.json"
            if not policy_path.exists():
                policy_path = root / "pd_workflow.json"
            policy = read(policy_path) if policy_path.exists() else {}
            records = attempts + phases
            import time

            now = time.time()
            budget = {}
            for scope, limits in policy.get("budgets", {}).items():
                selected = [
                    record
                    for record in records
                    if scope == "total"
                    or (scope == "verification" and record.get("purpose", "").startswith("verification_"))
                    or record.get("purpose") == scope
                ]
                unknown = sum(bool(record.get("elapsed_unknown")) for record in selected)
                elapsed = sum(
                    max(0, record.get("finished_at", now) - record["started_at"])
                    for record in selected
                    if not record.get("elapsed_unknown")
                )
                budget[scope] = {
                    "used_trials": len(selected),
                    "max_trials": limits.get("max_trials"),
                    "remaining_trials": max(0, limits["max_trials"] - len(selected)),
                    "elapsed_seconds": elapsed,
                    "remaining_seconds": None if unknown else max(0, limits["time_limit_seconds"] - elapsed),
                    "unknown_elapsed_attempts": unknown,
                }
            best_path = root / "best_result.json"
            best = read(best_path) if best_path.exists() else None
            output = {
                "best": {key: best.get(key) for key in ("candidate_id", "fitness", "source_ref", "params")}
                if best
                else None,
                "budget": budget,
                "coverage": search_summary(root),
                "cleanup_unconfirmed": sum(not record.get("cleanup_confirmed") for record in records),
                "baseline_exists": (root / "baseline.json").exists(),
                "pd_attempts": [{k: r.get(k) for k in ("purpose", "outcome", "cleanup_confirmed")} for r in attempts],
                "phase_attempts": [
                    {k: r.get(k) for k in ("purpose", "outcome", "qps", "capacity_key")} for r in phases
                ],
            }
        elif args.action in {"bootstrap", "layouts"}:
            from optix.optimizer.agentic.pd import load_plugin_config
            from optix.optimizer.agentic.pd_bootstrap import initial_layouts
            from optix.optimizer.agentic.pd_layouts import enumerate_layouts

            experiment = read(root / "pd_experiment.json")
            config = load_plugin_config(experiment["source"]["config_toml"])
            params = read(args.params) if args.params else {}
            if (root / "pd_workflow.json").exists():
                fixed = read(root / "pd_workflow.json").get("fixed_params", {})
                if any(name in params and params[name] != value for name, value in fixed.items()):
                    raise ValueError("Supplied parameters conflict with user-fixed constraints")
                params = {**params, **fixed}
            if args.action == "bootstrap":
                output = initial_layouts(config, params, max_candidates=args.limit)
            else:
                if not args.params:
                    raise ValueError("Layout search requires frozen per-role TP/DP in --params")
                for role in ("prefill", "decode"):
                    if "TENSOR_PARALLEL_SIZE_" + role not in params:
                        raise ValueError("Explicit per-role TP required after startup")
                from optix.optimizer.agentic.pd_phase import current_capacity_hint

                signature = hashlib.sha256((root / "pd_experiment.json").read_bytes()).hexdigest()
                hints = {}
                for role, path in (("prefill", args.prefill_capacity), ("decode", args.decode_capacity)):
                    hints[role] = (
                        current_capacity_hint(read(path), signature, role, params, config.vllm_others) if path else None
                    )
                use_hints = all(value is not None for value in hints.values())
                output = enumerate_layouts(
                    config,
                    params,
                    max_candidates=args.limit,
                    use_full_device=not args.allow_unused_devices,
                    prefill_qps=hints["prefill"] if use_hints else None,
                    decode_qps=hints["decode"] if use_hints else None,
                )
                output["capacity_hints_used"] = use_hints
        else:
            if not all(
                (
                    args.phase,
                    args.task_id,
                    args.config,
                    args.context,
                    args.candidate,
                    args.resource,
                )
            ):
                raise ValueError("Probe requires phase, task-id, config, context, candidate and resource")
            import optix.config.config as config_module
            from optix.config.config import Settings
            from optix.deploy_env import resolve_deploy_context

            # Import built-in plugin registrations, without entering the PSO CLI.
            import optix.optimizer.plugins.simulate  # noqa: F401
            import optix.optimizer.plugins.benchmark  # noqa: F401
            from optix.optimizer.pd_disagg import OptixPhaseRunner
            from optix.optimizer.agentic.pd_phase import run_phase_task

            class PhaseSettings(Settings):
                model_config = {
                    **Settings.model_config,
                    "toml_file": [
                        *Settings.model_config.get("toml_file", []),
                        Path(args.config).resolve(),
                    ],
                    "extra": "allow",
                }

            settings = PhaseSettings()
            config_module.settings = settings
            runtime, env = resolve_deploy_context()
            runner = OptixPhaseRunner(
                settings,
                runtime_ctx=runtime,
                deploy_env=env,
                bak_path=root,
                load_breakpoint=False,
                top_k=0,
            )
            output = run_phase_task(
                root,
                args.task_id,
                args.phase,
                read(args.resource),
                read(args.candidate),
                read(args.context),
                getattr(settings.pd_disagg, args.phase),
                runner,
            )
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        parser.exit(1, f"PD workflow failed: {error}\n")


if __name__ == "__main__":
    main()

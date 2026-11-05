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

"""Prepare, validate and execute exactly one PD candidate with a bound configuration."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def child_environment():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def collect_arguments(root, config, experiment):
    """Reuse frozen facts; callers cannot independently select another configuration."""
    model = experiment["model"]
    spec = experiment["benchmark_spec"]
    args = [
        "--run-dir",
        str(root),
        "--run-id",
        experiment["run_id"],
        "--engine",
        "vllm",
        "--deployment",
        "pd",
        "--config-toml",
        str(config),
        "--benchmark",
        experiment["benchmark"],
        "--model-name",
        model["name"],
        "--model-path",
        model["path"],
    ]
    for flag, value in (
        ("--input-len", spec["input_len_avg"]),
        ("--output-len", spec["output_len_avg"]),
        ("--num-requests", spec["num_requests"]),
        ("--memory-gb", experiment.get("hardware", {}).get("single_card_memory_gb")),
    ):
        if value is not None:
            args += [flag, str(value)]
    if spec.get("concurrency") is not None:
        args += ["--concurrency", str(spec["concurrency"])]
    elif spec.get("concurrency_range"):
        bounds = spec["concurrency_range"]
        args += ["--concurrency-min", str(bounds[0]), "--concurrency-max", str(bounds[1])]
    return args


def prepared_parameter_plan(context, candidate, others):
    from types import SimpleNamespace
    from optix.config.config import range_to_enum
    from optix.optimizer.agentic.parameters import resolve_execution_fields
    from optix.optimizer.agentic.validation import validate_candidate
    from optix.optimizer.agentic.pd import load_plugin_config
    from optix.optimizer.agentic.pd_execution import parameter_plan, load_execution_settings
    from optix.optimizer.pd_adapter import prepare_pd_fields

    checked = validate_candidate(context, candidate)
    if not checked["valid"]:
        raise ValueError("; ".join(checked["errors"]))
    plugin = load_plugin_config(context["source"]["config_toml"])
    settings = load_execution_settings(context["source"]["config_toml"])
    fields = list(plugin.target_field)
    names = {field.name for field in fields}
    fields.extend(field for field in settings.vllm.target_field if field.name not in names)
    range_to_enum(fields)
    fields = resolve_execution_fields(checked["params"], fields, context)
    simulator = SimpleNamespace(config=plugin, vllm_others=others, _is_moe=context.get("model_info", {}).get("is_moe"))
    transported, _ = prepare_pd_fields(simulator, tuple(fields))
    return parameter_plan(transported)


def check_base_review_gate(root, allow_unreviewed=False, require=False):
    """R0 案例底座裁决门禁（口径与 scripts/check_base_review.py 完全一致）。

    背景：④ 审计把 D 侧独有项（如 --additional-config:recompute_scheduler_enable）判成
    needs_review 后，"人工裁决"从未闭环，缺口一路继承到每一轮（实测 13 份启动脚本里 0 次）。
    R0 装配流水线的第 5 步已经硬阻断；这里是开跑前的第二道网：
      - run 目录有 R0 产物且存在未闭环项 → ValueError，禁止开跑；
      - run 目录无 R0 产物 → 默认只告警（非案例装配的 run 仍可跑），--require-base-review 强制失败；
      - allow_unreviewed=True → 显式放行（用于确认过缺口的 run）。
    """
    gate = SCRIPTS / "check_base_review.py"
    if allow_unreviewed:
        print("[warn] --allow-unreviewed-base：跳过 R0 案例底座裁决门禁", file=sys.stderr)
        return
    if not gate.exists():
        print(f"[warn] 未找到 {gate}，跳过 R0 案例底座裁决门禁", file=sys.stderr)
        return
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import check_base_review as gate_module

    coverage_path, decisions_path = gate_module.run_dir_artifacts(root)
    result = gate_module.evaluate(coverage_path, decisions_path)
    if not result["coverage_present"]:
        message = (
            f"run 目录缺少 R0 案例底座产物 {coverage_path}：无法确认 needs_review 已闭环。"
            "先跑 R0 装配流水线并把 base_coverage.after.json / base_review_decisions.json 复制进 run 目录"
        )
        if require:
            raise ValueError(message)
        if decisions_path.exists():
            print(f"[warn] {message}（本次仅告警；--require-base-review 可强制要求）", file=sys.stderr)
        return
    if result["rows"] or result["problems"]:
        report = gate_module.write_report(result, Path(root), allowed=False)
        raise ValueError(
            f"R0 案例底座裁决未闭环（{len(result['rows'])} 项，另有 {len(result['problems'])} 条无效裁决记录）："
            f"禁止开跑。清单见 {report}；按报告两条路径之一闭环后重跑，"
            "或确认缺口后加 --allow-unreviewed-base 显式放行"
        )


def prepare_round(
    root,
    number,
    attempt=1,
    purpose="search",
    runner=subprocess.run,
    allow_unreviewed_base=False,
    require_base_review=False,
):
    """All validation consumes the generator's selected configuration, never a caller path."""
    import generate_pd_config
    from optix.optimizer.agentic.pd_execution import file_sha256, validate_execution, prepare_settings

    root = Path(root).resolve()
    check_base_review_gate(root, allow_unreviewed_base, require_base_review)
    candidate = root / f"candidates.round-{number}.json"
    if not candidate.exists():
        raise ValueError(
            f"未找到第 {number} 轮候选文件 {candidate}；请先按 schemas/candidates.schema.json "
            "生成本轮候选，PD 部署示例见 references/pd.md"
        )
    proposal = read(candidate)
    candidate_sha256 = file_sha256(candidate)
    if proposal.get("round") != number or len(proposal.get("candidates", [])) != 1:
        raise ValueError("Exactly one candidate for this round is required")
    context = read(root / "context.json")
    context.get("deployment", {}).pop("execution_record", None)
    write(root / "context.json", context)
    generate_pd_config.main(
        [
            "--run-dir",
            str(root),
            "--round",
            str(number),
            "--attempt",
            str(attempt),
            "--purpose",
            purpose,
            "--proposal",
            str(candidate),
        ]
    )
    manifest_name = f"deployment.round-{number}" + (f".attempt-{attempt}" if attempt > 1 else "") + ".json"
    manifest = read(root / manifest_name)
    config = Path(manifest["config_toml"]).resolve()
    experiment = read(root / "pd_experiment.json")

    def script(name, args):
        runner([sys.executable, str(SCRIPTS / name), *args], check=True, env=child_environment())

    script("collect_context.py", collect_arguments(root, config, experiment))
    checked_inputs = {
        "candidate": (candidate, candidate_sha256),
        "config": (config, manifest["config_sha256"]),
        "context": (root / "context.json", file_sha256(root / "context.json")),
    }
    effective_settings = prepare_settings(config, experiment["benchmark"])
    for name, (path, expected_sha) in checked_inputs.items():
        if file_sha256(path) != expected_sha:
            raise ValueError(f"PD {name} changed during preparation")
    script(
        "config_preflight.py",
        [
            "--check-only",
            "--config",
            str(config),
            "--engine",
            "vllm",
            "--benchmark",
            experiment["benchmark"],
            "--context",
            str(root / "context.json"),
        ],
    )
    script("validate_candidates.py", ["--run-dir", str(root), "--round", str(number)])
    for name, (path, expected_sha) in checked_inputs.items():
        if file_sha256(path) != expected_sha:
            raise ValueError(f"PD {name} changed during preflight")
    if prepare_settings(config, experiment["benchmark"]) != effective_settings:
        raise ValueError("Effective settings changed during preflight")
    context = read(root / "context.json")
    if (
        Path(context["source"]["config_toml"]).resolve() != config
        or context["deployment"]["round_manifest"] != manifest
    ):
        raise ValueError("PD preflight context is not bound to the generated configuration")
    record_path = root / f"execution.round-{number}.attempt-{attempt}.json"
    if record_path.exists():
        raise ValueError("Execution record already exists; preserve it and select a new attempt")
    planned_parameters = prepared_parameter_plan(
        context, proposal["candidates"][0], effective_settings["vllm"]["command"]["others"]
    )
    context["deployment"]["execution_record"] = str(record_path)
    snapshot = root / f"context.round-{number}.attempt-{attempt}.json"
    if snapshot.exists():
        raise ValueError("Execution context snapshot already exists")
    write(snapshot, context)
    write(root / "context.json", context)
    record = {key: manifest[key] for key in ("round", "attempt", "candidate_id", "purpose")}
    record["effective_settings"] = effective_settings
    record["parameter_plan"] = planned_parameters
    for name, path in (("config", config), ("context", snapshot), ("candidate", candidate)):
        record[name] = {"path": str(path), "sha256": file_sha256(path)}
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    validate_execution(context)
    return context


def execute_round(
    root,
    number,
    attempt=1,
    purpose="search",
    prepare_only=False,
    runner=subprocess.run,
    allow_unreviewed_base=False,
    require_base_review=False,
):
    from optix.optimizer.agentic.pd import controller_session
    from optix.optimizer.agentic.pd_execution import validate_execution

    root = Path(root).resolve()
    with controller_session(root / ".pd_dispatch"):
        context = prepare_round(root, number, attempt, purpose, runner, allow_unreviewed_base, require_base_review)
        if not prepare_only:
            validate_execution(context)
            runner(
                [
                    sys.executable,
                    "-m",
                    "optix",
                    "-e",
                    "vllm_pd",
                    "-b",
                    context["benchmark"],
                    "-c",
                    context["source"]["config_toml"],
                    "--backup",
                ],
                check=True,
                env=child_environment(),
            )
        return context["deployment"]["execution_record"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--purpose", choices=["bootstrap", "search", "verification_baseline", "verification_winner"], default="search"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Run all preparation gates without starting services"
    )
    parser.add_argument(
        "--allow-unreviewed-base",
        action="store_true",
        help="显式跳过 R0 案例底座裁决门禁（确认过缺口时才用；缺口会被带进本轮候选）",
    )
    parser.add_argument(
        "--require-base-review",
        action="store_true",
        help="要求 run 目录必须存在 R0 底座裁决产物（缺产物即失败，默认只告警）",
    )
    args = parser.parse_args()
    try:
        record = execute_round(
            args.run_dir,
            args.round,
            args.attempt,
            args.purpose,
            args.prepare_only,
            allow_unreviewed_base=args.allow_unreviewed_base,
            require_base_review=args.require_base_review,
        )
        print(json.dumps({"execution_record": record, "prepared_only": args.prepare_only}))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"PD round stopped: {exc}\n")


if __name__ == "__main__":
    main()

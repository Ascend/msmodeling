"""Ownership: shared-precheck. Search space self-check: validate context.json and produce checklist_report.md."""

import argparse
from pathlib import Path

from common import read_json


# Parameters considered "batch-related" for coverage checks
BATCH_PARAMS = {"max_num_batched_tokens", "max_num_seqs", "max_batch_size", "max_prefill_batch_size", "max_num_tokens"}
PREPILL_PARAMS = {"prefill_time_ms_per_req", "max_prefill_batch_size", "max_num_batched_tokens"}
CONCURRENCY_PARAMS = {"concurrency", "requestrate"}
PARALLEL_EP_PARAMS = {"enable_expert_parallel", "expert_parallel_size", "ep_size", "ep"}
PARALLEL_DP_PARAMS = {"dp", "data_parallel_size", "dp_size"}
PARALLEL_TP_PARAMS = {"tp", "tensor_parallel_size", "tp_size"}


def _check_model_info(context: dict) -> list[dict]:
    """Check model_info completeness."""
    issues = []
    model_info = context.get("model_info", {}) or {}
    if not model_info:
        issues.append(
            {"level": "error", "msg": "model_info 为空，无法判断模型架构（MoE/Dense），将缺失模型族级经验匹配"}
        )
        return issues
    if "is_moe" not in model_info:
        issues.append({"level": "warning", "msg": "model_info 缺少 is_moe 字段，无法判断 MoE 架构"})
    return issues


def _check_parallel_coverage(context: dict) -> list[dict]:
    """Check parallel dimension coverage, especially for MoE models."""
    issues = []
    model_info = context.get("model_info", {}) or {}
    is_moe = model_info.get("is_moe", False)
    search_space = context.get("search_space", {})
    param_names = {p["name"].lower() for p in search_space.get("parameters", [])}

    has_ep = bool(PARALLEL_EP_PARAMS & param_names)
    has_dp = bool(PARALLEL_DP_PARAMS & param_names)
    tp_params = param_names & PARALLEL_TP_PARAMS
    has_tp = bool(tp_params)
    # Check if TP is the only tunable parallel dimension
    tp_only = has_tp and not has_ep and not has_dp
    tp_tunable = any(
        p["name"].lower() in tp_params for p in search_space.get("parameters", []) if p.get("min") != p.get("max")
    )

    if is_moe and tp_only:
        issues.append(
            {
                "level": "warning",
                "msg": "MoE 模型仅有 TP 维度可调（无 EP/DP），并行策略搜索空间不足。"
                "建议启用 enable_expert_parallel 或追加 DP/EP 参数",
            }
        )
    elif is_moe and not has_ep:
        issues.append({"level": "warning", "msg": "MoE 模型未检测到 EP 相关参数，known_patterns 可能未注入"})

    if not tp_tunable and not has_ep and not has_dp:
        issues.append({"level": "warning", "msg": "所有并行维度均为固定值，将无法搜索并行策略"})

    return issues


def _check_missing_params(context: dict) -> list[dict]:
    """Check for missing critical parameter categories."""
    issues = []
    search_space = context.get("search_space", {})
    param_names = {p["name"].lower() for p in search_space.get("parameters", [])}
    all_names = param_names | {c["name"].lower() for c in search_space.get("constants", [])}

    if not (CONCURRENCY_PARAMS & all_names):
        issues.append({"level": "warning", "msg": "缺少 CONCURRENCY/REQUESTRATE 参数，无法探索并发影响"})
    if not (BATCH_PARAMS & all_names):
        issues.append({"level": "warning", "msg": "缺少 batch 相关参数（MAX_NUM_BATCHED_TOKENS 等）"})
    if not (PREPILL_PARAMS & all_names):
        issues.append({"level": "warning", "msg": "缺少 prefill 相关参数，TTFT 优化能力受限"})

    return issues


def _check_knowledge_injection(context: dict) -> list[dict]:
    """Check known_patterns knowledge injection coverage."""
    issues = []
    knowledge = context.get("knowledge", {}) or {}
    patterns = knowledge.get("known_patterns", [])
    if not patterns:
        return issues

    search_space = context.get("search_space", {})
    param_names = {p["name"].lower() for p in search_space.get("parameters", [])}
    all_names = param_names | {c["name"].lower() for c in search_space.get("constants", [])}

    for idx, pattern in enumerate(patterns):
        hint = pattern if isinstance(pattern, dict) else {}
        params = hint.get("params", {})
        for pname, pvalue in params.items():
            if isinstance(pvalue, dict) and "explore" in pvalue:
                # explore directive — param should exist in search_space
                if pname.lower() not in all_names:
                    issues.append(
                        {
                            "level": "warning",
                            "msg": f"known_patterns[{idx}] 建议的 '{pname}' 未在 search_space 中找到，"
                            f"可能未被 collect_context.py 注入",
                        }
                    )
            elif pname.lower() not in all_names:
                scope = hint.get("scope", "")
                if scope in ("do_not_tune", "strong_suggest"):
                    issues.append(
                        {
                            "level": "warning",
                            "msg": f"known_patterns[{idx}] scope={scope} 参数 '{pname}' 未在 search_space 中找到",
                        }
                    )
    return issues


def _is_scalar(value) -> bool:
    """True if a param value is renderable as a scalar CLI value (int/float/bool/str)."""
    return isinstance(value, (int, float, bool, str)) or value is None


def _check_runtime_renderability(context: dict) -> list[dict]:
    """Check that search-space params are renderable by optix at runtime.

    known_patterns can inject params whose choices are *lists* (e.g.
    cudagraph_capture_sizes = [[1,8,16], ...]). optix's OptimizerConfigField
    historically only accepted scalar values and crashed;
    after the fix it accepts list/dict, but the command layer only renders them
    into JSON-container flags (compilation-config). Flag any non-scalar param so
    the agent knows it has special rendering requirements and verifies optix
    supports it — never let a silently-unrenderable param pass.
    """
    issues = []
    search_space = context.get("search_space", {})
    params = search_space.get("parameters", [])
    for p in params:
        name = p.get("name", "")
        choices = p.get("choices") or []
        default = p.get("default")
        container_ok = name in (
            "cudagraph_capture_sizes",
            "cudagraph_mode",
            "num_speculative_tokens",
            "speculative_method",
        )
        for val in list(choices) + ([default] if default is not None else []):
            if not _is_scalar(val):
                level = "pass" if container_ok else "warning"
                issues.append(
                    {
                        "level": level,
                        "msg": (
                            f"参数 '{name}' 含非标量值（list/dict），optix 仅在 JSON 容器参数"
                            f"{'(已支持: cudagraph_capture_sizes 等)' if container_ok else ''}下可渲染；"
                            f"请确认对应 optix 渲染支持，否则该参数运行时会崩溃"
                        ),
                    }
                )
                break
    return issues


def _check_constraint_satisfiability(context: dict) -> list[dict]:
    """Check if constraints have at least one feasible solution."""
    issues = []
    constraints = context.get("constraints", {})
    world_size = constraints.get("world_size")
    expressions = constraints.get("expressions", [])
    if not expressions or world_size is None:
        return issues

    search_space = context.get("search_space", {})
    params = {p["name"].lower(): p for p in search_space.get("parameters", [])}

    # Quick check: TP*DP*PP expression with world_size
    tp_info = params.get("tp", {})
    dp_info = params.get("dp", {})
    pp_info = params.get("pp", {})
    if tp_info and dp_info and pp_info:
        tp_choices = tp_info.get("choices") if tp_info.get("dtype") == "enum" else None
        dp_choices = dp_info.get("choices") if dp_info.get("dtype") == "enum" else None
        pp_choices = pp_info.get("choices") if pp_info.get("dtype") == "enum" else None

        tp_min = int(min(tp_choices)) if tp_choices else int(tp_info.get("min", 1))
        tp_max = int(max(tp_choices)) if tp_choices else int(tp_info.get("max", world_size))
        dp_min = int(min(dp_choices)) if dp_choices else int(dp_info.get("min", 1))
        dp_max = int(max(dp_choices)) if dp_choices else int(dp_info.get("max", world_size))
        pp_min = int(min(pp_choices)) if pp_choices else int(pp_info.get("min", 1))
        pp_max = int(max(pp_choices)) if pp_choices else int(pp_info.get("max", world_size))
        found = any(
            t * d * p == world_size for t in (tp_min, tp_max) for d in (dp_min, dp_max) for p in (pp_min, pp_max)
        )
        if not found:
            issues.append(
                {
                    "level": "error",
                    "msg": f"TP({tp_min}-{tp_max})×DP({dp_min}-{dp_max})×PP({pp_min}-{pp_max}) "
                    f"无可行解满足 WORLD_SIZE={world_size}",
                }
            )

    return issues


def _check_engine_fixed_params(context: dict) -> list[dict]:
    """Warn for unverified engine-level fixed params (e.g. preset enforce_eager).

    Issue 1.1: vendor preset 的 additional_config（如 speculative-config 里的
    enforce_eager=true）只写进 config.toml `others`，不进搜索空间、不生成候选。
    collect_context 现在把它们登记为 engine_fixed_params；本检查在它们被
    单变量反转验证（verified=True）前持续告警，防止 agent 把预设默认值当
    "物理极限"。
    """
    issues = []
    fixed = context.get("engine_fixed_params") or []
    for entry in fixed:
        if entry.get("flip_required") and not entry.get("verified"):
            search_hint = f"（{entry.get('search_name')} 取相反值）" if entry.get("search_name") else ""
            issues.append(
                {
                    "level": "warning",
                    "msg": (
                        f"引擎级固定参数 {entry.get('path')}={entry.get('value')!r}"
                        f"（来源 {entry.get('source', '?')}）未验证 —— "
                        f"Round 1 应加入单变量反转候选 {search_hint}"
                    ),
                }
            )
    return issues


def main():
    parser = argparse.ArgumentParser(description="Check search space for agent optimizer.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    context_path = run_dir / "context.json"
    if not context_path.exists():
        print(f"context.json not found at {context_path}")
        return

    context = read_json(context_path)

    all_issues = []
    all_issues.extend(_check_model_info(context))
    all_issues.extend(_check_parallel_coverage(context))
    all_issues.extend(_check_missing_params(context))
    all_issues.extend(_check_knowledge_injection(context))
    all_issues.extend(_check_runtime_renderability(context))
    all_issues.extend(_check_constraint_satisfiability(context))
    all_issues.extend(_check_engine_fixed_params(context))

    warnings = [i for i in all_issues if i["level"] == "warning"]
    errors = [i for i in all_issues if i["level"] == "error"]

    # Build report
    lines = ["# 搜索空间自检报告", ""]

    # Collect "passes" from context properties that were verified
    model_info = context.get("model_info", {}) or {}
    knowledge = context.get("knowledge", {}) or {}
    constraints = context.get("constraints", {})

    passed_items = []
    if model_info:
        passed_items.append(
            f"模型信息完整：{model_info.get('architectures', ['unknown'])[0]}, is_moe={model_info.get('is_moe', 'unknown')}"
        )
    param_names = {p["name"].lower() for p in context.get("search_space", {}).get("parameters", [])}
    parallel_parts = []
    if param_names & PARALLEL_TP_PARAMS:
        tp_p = next(p for p in context["search_space"]["parameters"] if p["name"].lower() in PARALLEL_TP_PARAMS)
        parallel_parts.append(f"TP({tp_p.get('min')}-{tp_p.get('max')})")
    if param_names & PARALLEL_DP_PARAMS:
        dp_p = next(p for p in context["search_space"]["parameters"] if p["name"].lower() in PARALLEL_DP_PARAMS)
        parallel_parts.append(f"DP({dp_p.get('min')}-{dp_p.get('max')})")
    if param_names & PARALLEL_EP_PARAMS:
        parallel_parts.append("EP=on")
    if parallel_parts:
        passed_items.append(f"并行维度覆盖：{', '.join(parallel_parts)}")

    patterns = knowledge.get("known_patterns", [])
    if patterns:
        passed_items.append(f"known_patterns 生效：{len(patterns)} 条匹配")

    if constraints.get("expressions"):
        passed_items.append(f"约束已检查：{len(constraints['expressions'])} 条表达式")

    verified_fixed = [
        f"{f.get('path')}={f.get('value')!r}"
        for f in (context.get("engine_fixed_params") or [])
        if f.get("flip_required") and f.get("verified")
    ]
    if verified_fixed:
        passed_items.append(f"引擎级固定参数已验证：{', '.join(verified_fixed)}")

    lines.append("## ✅ 通过")
    for item in passed_items:
        lines.append(f"- {item}")
    if not passed_items:
        lines.append("- （无）")

    lines.extend(["", "## ⚠️ 警告"])
    for w in warnings:
        lines.append(f"- {w['msg']}")
    if not warnings:
        lines.append("- （无）")

    lines.extend(["", "## ❌ 错误"])
    for e in errors:
        lines.append(f"- {e['msg']}")
    if not errors:
        lines.append("- （无）")

    lines.append("")

    report_path = run_dir / "checklist_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Checklist written to {report_path}")
    if errors:
        print(f"ERROR: {len(errors)} issue(s) found")
        raise SystemExit(1)
    elif warnings:
        print(f"WARNING: {len(warnings)} issue(s) found (non-blocking)")
    else:
        print("All checks passed")


if __name__ == "__main__":
    main()

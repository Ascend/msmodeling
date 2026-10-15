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

"""Ownership: shared-precheck. Collect environment facts and confirmed workload into context.json."""

import argparse
import hashlib
import tomllib
from pathlib import Path

from common import (
    read_json,
    write_json,
)


def coalesce(*values, default=None):
    for value in values:
        if value not in (None, ""):
            return value
    return default


def recommendation_field(recommendation, *paths):
    for path in paths:
        cur = recommendation
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if cur not in (None, ""):
            return cur
    return None


def collect_constraints(world_size):
    # 只采集事实，不生成约束表达式：表达式是领域知识，由 agent 在读取知识库后
    # 按需自行编写（见 workflows/data-analysis.md 建空间规则）。脚本内含表达式
    # 模板会让 agent 把未配置的维度误登记为空间条目。
    return {"world_size": world_size, "expressions": []}


def _infer_is_moe(config: dict) -> bool:
    """Infer MoE from HF config.json without relying on a standard ``is_moe`` field."""
    # Explicit expert count > 1
    for key in ("n_routed_experts", "num_experts", "num_local_experts"):
        if config.get(key, 0) > 1:
            return True
    # MoE-specific fields — presence implies expert routing
    for key in ("n_shared_experts", "moe_layer_freq", "moe_intermediate_size"):
        if key in config:
            return True
    return False


def _read_model_info(model_path):
    """Read architectures and infer is_moe from model config.json.

    ``model_path`` may be a directory (e.g. /weights/Llama-3-8B) or the
    full path to config.json (e.g. /weights/DeepSeek-V3/config.json).

    Handles multimodal/nested HF configs: fields like ``num_hidden_layers`` /
    ``mtp_num_hidden_layers`` often live under ``text_config`` (or
    ``language_config`` / ``llm_config``) rather than at the top level. We read
    from the top level first, then fall back to the first nested sub-config that
    provides the field (top-level wins).
    """
    if not model_path:
        return {}
    model_p = Path(model_path)
    if model_p.is_file() and model_p.name == "config.json":
        config_path = model_p
    else:
        config_path = model_p / "config.json"
    if not config_path.exists():
        return {}
    try:
        config = read_json(config_path)
        # Nested-sub-config lookup: fields are read top-level first, then from
        # the first sub-config that has them (text_config is the HF convention for
        # multimodal models, but cover the common aliases too).
        SUB_KEYS = ("text_config", "language_config", "llm_config", "text_llm_config")
        sub_configs = [config.get(k, {}) for k in SUB_KEYS if isinstance(config.get(k), dict)]
        # non-empty sub-configs first
        sub_configs = [c for c in sub_configs if c] or [{}]

        def _get(key, default=0):
            if key in config and config.get(key) not in (None, ""):
                return config.get(key, default)
            for sub in sub_configs:
                if sub.get(key) not in (None, ""):
                    return sub.get(key, default)
            return default

        result = {}
        arch = _get("architectures", []) or config.get("architectures", [])
        if arch:
            result["architectures"] = list(arch) if isinstance(arch, list) else [arch]
        result["is_moe"] = _infer_is_moe(config) or _infer_is_moe(sub_configs[0] if sub_configs else {})
        result["num_hidden_layers"] = int(_get("num_hidden_layers", 0))
        result["num_layers"] = int(_get("num_layers", 0) or 0)
        result["use_mtp"] = bool(
            _get("use_mtp", False)
            or int(_get("mtp_num_hidden_layers", 0)) > 0
            or int(_get("num_nextn_predict_layers", 0)) > 0
        )
        result["num_mtp_modules"] = int(_get("num_mtp_modules", 0) or 0)
        # --- 增强 MTP/投机解码信息采集 ---
        result["num_speculative_tokens"] = (
            _get("num_speculative_tokens", 0)
            or _get("speculative_tokens", 0)
            or _get("speculative_config", {}).get("num_speculative_tokens", 0)
            or 0
        )
        result["mtp_num_hidden_layers"] = int(
            _get("mtp_num_hidden_layers", 0) or _get("num_nextn_predict_layers", 0) or result["num_mtp_modules"] or 0
        )
        # dspark_block_size: DSpark block drafter 的 block 大小
        result["dspark_block_size"] = int(_get("dspark_block_size", 0) or 0)
        # mtp_acceptance_rate: 如果有实测背书
        result["mtp_acceptance_rate"] = _get("mtp_acceptance_rate", None)
        # eagle_head_path / speculative_model_path: 从 model_config 段读取
        model_cfg = config.get("model_config", {}) or config.get("speculative_config", {}) or {}
        result["eagle_head_path"] = model_cfg.get("eagle_head_path") or ""
        result["dflash_draft_model_path"] = model_cfg.get("dflash_draft_model_path", "")
        result["speculative_model_path"] = model_cfg.get("speculative_model", "")
        # model_type: 架构类型
        result["model_type"] = _get("model_type", "")
        return result
    except Exception:
        return {}


def _get_skill_dir() -> Path:
    """Return the skill root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def _find_config_toml() -> Path | None:
    """Auto-discover config.toml by walking up from CWD (then the skill dir).

    CWD is the primary anchor; walk upward so running from a repo subdirectory
    (or the repo root) still finds ``<project_root>/optix/config.toml``. Returns
    None only if truly not found — callers must treat that as a hard error, not
    silently produce a search space without target_fields.
    """
    for origin in (Path.cwd(), _get_skill_dir()):
        candidate = origin
        for _ in range(8):
            config_path = candidate / "optix" / "config.toml"
            if config_path.exists():
                return config_path
            parent = candidate.parent
            if parent == candidate:
                break
            candidate = parent
    return None


def _parse_config_toml(path: Path, engine: str) -> dict:
    """Parse target_fields from a config.toml file for the given engine."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return {"target_fields": []}

    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
    except Exception:
        return {"target_fields": []}

    engine_lower = engine.lower()
    section = None
    for key, value in config.items():
        if key.lower() == engine_lower:
            section = value
            break
    if section is None:
        return {"target_fields": []}

    target_fields = []
    for field in section.get("target_field") or []:
        if isinstance(field, dict) and field.get("name"):
            field.setdefault("search", True)
            target_fields.append(field)

    return {"target_fields": target_fields}


def main():
    parser = argparse.ArgumentParser(description="Collect an agent optimizer context file.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--engine", default=None)
    parser.add_argument(
        "--benchmark",
        default=None,
        help="记录进 context.json 的 benchmark 工具（默认 ais_bench，与 config_preflight --benchmark 一致）",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--num-per-node", type=int, default=None)
    parser.add_argument("--memory-gb", type=float, default=None)
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="业务输入长度（用户确认），写入 context.workload 的 input_len_avg/max，供负载分析",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="业务输出长度（用户确认），写入 context.workload 的 output_len_avg/max，供负载分析",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="压测并发（用户确认），写入 context.benchmark_spec；固定形态：硬约束、不可搜索。"
        "与 --concurrency-min/--concurrency-max 互斥",
    )
    parser.add_argument(
        "--concurrency-min",
        type=int,
        default=None,
        help="压测并发范围下限（吞吐优先场景：用户不关心具体并发时使用）；"
        "与 --concurrency-max 成对出现，并发进搜索空间",
    )
    parser.add_argument(
        "--concurrency-max",
        type=int,
        default=None,
        help="压测并发范围上限，与 --concurrency-min 成对出现",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="压测总请求数（用户确认），写入 context.benchmark_spec；硬约束、不可搜索",
    )
    parser.add_argument("--recommend-result", default="")
    parser.add_argument(
        "--config-toml",
        default="",
        help="Path to config.toml for auto-extracting target_fields from the engine section (vllm or mindie). Auto-discovers optix/config.toml from project root if not specified.",
    )
    # 预算旗标仅作为信息记录写入 context.json；运行时终止以 config.toml
    # [agent_optimizer] 为准（用 config_preflight --set-max-* 回填），默认值与其对齐
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--time-limit-minutes", type=int, default=1440)
    args = parser.parse_args()

    # 并发两形态互斥校验：固定（--concurrency）或可调范围（--concurrency-min/max）
    if args.concurrency is not None and (args.concurrency_min is not None or args.concurrency_max is not None):
        parser.error("--concurrency 与 --concurrency-min/--concurrency-max 互斥（固定/可调二选一）")
    if (args.concurrency_min is None) != (args.concurrency_max is None):
        parser.error("--concurrency-min 与 --concurrency-max 必须成对出现")
    if args.concurrency_min is not None and args.concurrency_min >= args.concurrency_max:
        parser.error("--concurrency-min 必须小于 --concurrency-max")
    if args.concurrency_max is not None and args.num_requests is not None and args.concurrency_max > args.num_requests:
        parser.error("--concurrency-max 不能大于 --num-requests（并发上限超过总请求数没有意义）")

    # --recommend-result 是 PSO→agent 的可选衔接：只读取 recommend_params 输出里
    # 实际存在的顶层字段（engine / benchmark_policy），其余输入一律走 CLI 旗标。
    recommendation = read_json(Path(args.recommend_result)) if args.recommend_result else {}
    world_size = int(args.world_size or 0)
    if world_size <= 0:
        parser.error("--world-size is required")

    engine = coalesce(args.engine, recommendation_field(recommendation, "engine"), default="")

    config_toml_path = args.config_toml or _find_config_toml()
    if config_toml_path is None:
        parser.error("config.toml not found; pass --config-toml")
    cfg_path = Path(config_toml_path).resolve()
    try:
        config = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read config.toml: {exc}")
    engine = engine or str(config.get("framework", "vllm")).lower()
    parsed = _parse_config_toml(cfg_path, engine)
    if not parsed.get("target_fields"):
        parser.error(f"No target_field definitions for {engine} in {cfg_path}")
    source = {
        "config_toml": str(cfg_path),
        "config_toml_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "optimizer_strategy": config.get("optimizer_strategy", "pso"),
    }

    num_nodes = int(args.num_nodes or 1)
    num_per_node = int(args.num_per_node or world_size // num_nodes)
    memory_gb = float(args.memory_gb or 0)
    benchmark = coalesce(
        args.benchmark,
        recommendation_field(recommendation, "benchmark", "benchmark_policy"),
        default="ais_bench",
    )
    workload = {}
    if args.input_len is not None:
        workload["input_len_avg"] = args.input_len
        workload["input_len_max"] = args.input_len
    if args.output_len is not None:
        workload["output_len_avg"] = args.output_len
        workload["output_len_max"] = args.output_len
    workload_confirmed = bool(workload.get("input_len_avg") and workload.get("output_len_avg"))

    # benchmark 硬约束（用户确认的压测负载：输入/输出长度、请求数 + 并发）登记为
    # benchmark_spec —— 它是 config_preflight 一致性门禁比对 config.toml 与
    # ais_bench 实际配置的唯一事实源。并发有两种形态：
    #   固定：concurrency=N（延迟敏感场景，用户关心并发）—— 硬约束、不可搜索；
    #   可调：concurrency_range=[lo,hi]（吞吐优先场景，用户不关心具体并发）——
    #         并发作为搜索维度进搜索空间，optix 每 trial 改写。
    # headless 模式任务消息已给全参数 = 已确认，必须带全负载旗标
    # （confirmed=true 时门禁才执行比对）。
    concurrency_range = None
    if args.concurrency_min is not None:
        concurrency_range = [args.concurrency_min, args.concurrency_max]
    benchmark_spec = {
        "input_len_avg": args.input_len,
        "output_len_avg": args.output_len,
        "concurrency": args.concurrency,
        "concurrency_range": concurrency_range,
        "num_requests": args.num_requests,
    }
    base_confirmed = all(v is not None for v in (args.input_len, args.output_len, args.num_requests))
    concurrency_confirmed = args.concurrency is not None or concurrency_range is not None
    benchmark_spec_confirmed = base_confirmed and concurrency_confirmed

    context = {
        "run_id": args.run_id or Path(args.run_dir).name,
        "engine": engine,
        "benchmark": benchmark,
        "hardware": {
            "world_size": world_size,
            "num_nodes": num_nodes,
            "num_per_node": num_per_node,
            "single_card_memory_gb": memory_gb,
        },
        "model": {
            "name": args.model_name or "",
            "path": args.model_path or "",
        },
        "model_info": _read_model_info(args.model_path),
        "workload": workload,
        "workload_confirmed": workload_confirmed,
        "benchmark_spec": benchmark_spec,
        "benchmark_spec_confirmed": benchmark_spec_confirmed,
        "target": "",
        "constraints": collect_constraints(world_size),
    }
    # PSO preset 通道：命中 presets 目录时写入 knowledge.vendor_preset，
    # 供 apply_launch_config 应用 env/additional_config（Round 0 不建搜索空间，
    # preset params 不进 search_space）。agent 模式不注入：vendor_preset 的唯一
    # 消费者 apply_launch_config 是 PSO 专用，agent 模式没有消费方。
    if source["optimizer_strategy"] != "agent":
        context = _inject_vendor_preset(context)
    # --- Append remaining fields ------------------------------------------
    # Budgets: align with config.toml [agent_optimizer] (the real runtime
    # termination source) when available; CLI flags act as overrides.
    # Fallback defaults (32/12/1440) must not diverge from the actual config.
    cfg_budgets = {}
    if cfg_path and Path(cfg_path).exists():
        try:
            parsed_ao = tomllib.loads(Path(cfg_path).read_text(encoding="utf-8")).get("agent_optimizer", {}) or {}
            cfg_budgets = {
                "max_rounds": parsed_ao.get("max_rounds"),
                "max_trials": parsed_ao.get("max_trials"),
                "time_limit_minutes": parsed_ao.get("time_limit_minutes"),
            }
        except Exception:
            cfg_budgets = {}
    context["budgets"] = {
        "max_rounds": args.max_rounds if args.max_rounds != 32 else (cfg_budgets.get("max_rounds") or args.max_rounds),
        "max_trials": args.max_trials if args.max_trials != 12 else (cfg_budgets.get("max_trials") or args.max_trials),
        "time_limit_minutes": (
            args.time_limit_minutes
            if args.time_limit_minutes != 1440
            else (cfg_budgets.get("time_limit_minutes") or args.time_limit_minutes)
        ),
    }
    context["source"] = source
    # -----------------------------------------------------------------------
    write_json(Path(args.run_dir) / "context.json", context)


# ---------------------------------------------------------------------------
# Vendor preset injection (vLLM Ascend 官方推荐配置目录)
# ---------------------------------------------------------------------------


def _vendor_presets_path() -> Path:
    """Path to presets/ascend_vllm_presets.json next to this skill."""
    return Path(__file__).resolve().parent.parent / "presets" / "ascend_vllm_presets.json"


def _load_vendor_presets() -> dict:
    """Load the preset catalog; {} if missing/corrupt (fallback = no injection)."""
    path = _vendor_presets_path()
    if not path.exists():
        return {}
    try:
        data = read_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _find_vendor_preset(context: dict, presets: dict) -> dict | None:
    """Match context.model.name against catalog name/aliases (case-insensitive)."""
    model_name = ((context.get("model") or {}).get("name") or "").strip().lower()
    if not model_name:
        return None
    for preset in presets.get("presets", []):
        names = [preset.get("model", {}).get("name", "")] + preset.get("model", {}).get("aliases", [])
        if any(name and model_name == name.lower() for name in names):
            return preset
    return None


def _hardware_matches(preset: dict, context: dict) -> bool:
    """Conservative hardware gate: if the preset declares hardware, at least one
    entry must match the user's world_size (and memory, when both known).
    """
    hardware = preset.get("hardware") or []
    if not hardware:
        return True
    world_size = (context.get("constraints") or {}).get("world_size")
    if not world_size:
        world_size = (context.get("hardware") or {}).get("world_size")
    mem_gb = (context.get("hardware") or {}).get("single_card_memory_gb")
    for hw in hardware:
        if hw.get("cards") != world_size:
            continue
        if mem_gb and hw.get("mem_per_card_gb") and abs(hw["mem_per_card_gb"] - mem_gb) > 8:
            continue
        return True
    return False


def _select_scenario(preset: dict, context: dict) -> dict:
    """Pick the scenario whose workload input length is nearest the user's."""
    scenarios = preset.get("scenarios", [])
    if not scenarios:
        return {}
    workload = context.get("workload") or {}
    input_len = workload.get("input_len_avg") or workload.get("input_len_max")
    if not input_len:
        return scenarios[0]
    best, best_delta = scenarios[0], None
    for scenario in scenarios:
        avg = (scenario.get("workload") or {}).get("input_len_avg")
        if avg is None:
            continue
        delta = abs(avg - input_len)
        if best_delta is None or delta < best_delta:
            best, best_delta = scenario, delta
    return best


def _inject_vendor_preset(context: dict) -> dict:
    """Record the matched preset scenario in ``knowledge.vendor_preset``.

    PSO 分支的 preset 通道（见 references/pso-mode.md「preset 分流」）：
    apply_launch_config 消费 env/additional_config 落到 config.toml others 与
    launch_env.sh。Round 0 不建搜索空间，preset params 不注入 search_space。
    ``hardware`` 是参考信息而非门禁：记录官方推荐硬件 vs 实际硬件，matched=False
    时读取方应意识到并行参数是按推荐硬件给的，其余推荐值仍可用作种子。
    无匹配 / 目录缺失 / 场景为空 → context 不变。
    """
    presets = _load_vendor_presets()
    if not presets:
        return context
    preset = _find_vendor_preset(context, presets)
    if not preset:
        return context
    scenario = _select_scenario(preset, context)
    if not scenario:
        return context

    context.setdefault("knowledge", {})["vendor_preset"] = {
        "model": preset.get("model", {}).get("name", ""),
        "scenario_id": scenario.get("id", ""),
        "description": scenario.get("description", ""),
        "params": scenario.get("params", {}),
        "env": scenario.get("env", {}),
        "additional_config": scenario.get("additional_config", {}),
        "source_url": scenario.get("source_url", ""),
        "hardware": {
            "recommended": preset.get("hardware") or [],
            "actual_world_size": (context.get("constraints") or {}).get("world_size"),
            "matched": _hardware_matches(preset, context),
        },
    }
    return context


if __name__ == "__main__":
    main()

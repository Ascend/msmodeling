"""Ownership: shared-precheck. Collect run context into context.json (search space assembly)."""

import argparse
import json
import tomllib
from pathlib import Path

from common import (
    match_known_patterns,
    read_json,
    search_space_from_recommendation,
    search_space_from_target_fields,
    write_json,
)


def default_parallel_search_space(world_size: int):
    choices = [value for value in (1, 2, 4, 8, 16, 32, 64) if value <= max(world_size, 1) and world_size % value == 0]
    if not choices:
        choices = [1]
    pp_choices = [value for value in choices if value <= min(world_size, 8)]
    return {
        "parameters": [
            {"name": "tp", "dtype": "enum", "min": 0, "max": 1, "choices": choices, "default": 1},
            {"name": "dp", "dtype": "enum", "min": 0, "max": 1, "choices": choices, "default": 1},
            {"name": "pp", "dtype": "enum", "min": 0, "max": 1, "choices": pp_choices, "default": 1},
            {
                "name": "GPU_MEMORY_UTILIZATION",
                "dtype": "ratio",
                "default": 0.9,
                "min": 0.0,
                "max": 1.0,
                "reason": "Common vLLM memory-reservation ratio; higher values leave less headroom for KV cache fragmentation.",
                "source": "default",
            },
        ],
        "constants": [],
    }


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


def recommendation_hardware(recommendation):
    handoff = recommendation.get("agent_optimizer_handoff") or {}
    return handoff.get("hardware") or recommendation.get("hardware") or {}


def recommendation_constraints(recommendation, world_size, engine=""):
    handoff = recommendation.get("agent_optimizer_handoff") or {}
    constraints = dict(handoff.get("constraints") or {})
    if constraints:
        constraints["world_size"] = world_size
        constraints.setdefault("expressions", [])
        return constraints
    resolved_engine = str(
        coalesce(
            engine,
            recommendation_field(recommendation, "engine", "agent_optimizer_handoff.source.engine"),
            default="",
        )
    ).lower()
    if resolved_engine == "vllm":
        # 用户显式指定 N 卡时，要求并行维度恰好用满 N 卡（tp×dp×pp == N）。
        # `<=` 会允许少用卡的组合（如 2 卡只用 1 卡），产生次优解。
        expressions = ["$TP * $DP * $PP == $WORLD_SIZE"]
    elif resolved_engine == "mindie":
        expressions = ["$DP * $TP == $WORLD_SIZE"]
    elif not resolved_engine:
        expressions = ["$DP * $TP * $PP == $WORLD_SIZE"]
    else:
        expressions = []
    return {
        "world_size": world_size,
        "expressions": expressions,
    }


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
        result["eagle_head_path"] = model_cfg.get("eagle_head_path") or model_cfg.get("speculative_model") or ""
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


def _inject_model_derived_params(search_space: dict, model_info: dict) -> dict:
    """Add search-space entries derived from model architecture analysis.

    - MoE models: ensure enable_expert_parallel, enable_shared_expert_dp exist
    - MTP-capable models: ensure num_speculative_tokens exists
    - Deep models (>40 layers): suggest cudagraph_mode
    """
    existing_names = {p["name"].lower() for p in search_space.get("parameters", [])}
    existing_names |= {c["name"].lower() for c in search_space.get("constants", [])}
    injected = []

    is_moe = model_info.get("is_moe", False)
    num_layers = model_info.get("num_hidden_layers") or model_info.get("num_layers") or 0

    if is_moe:
        if "enable_expert_parallel" not in existing_names:
            injected.append(
                {
                    "name": "enable_expert_parallel",
                    "config_position": "env",
                    "dtype": "enum",
                    "default": True,
                    "choices": [False, True],
                    "reason": "MoE model: expert parallel reduces allreduce overhead",
                    "source": "model_derived",
                }
            )
            existing_names.add("enable_expert_parallel")
        if "enable_shared_expert_dp" not in existing_names:
            injected.append(
                {
                    "name": "enable_shared_expert_dp",
                    "config_position": "env",
                    "dtype": "enum",
                    "default": True,
                    "choices": [False, True],
                    "reason": "MoE model: shared expert data parallel",
                    "source": "model_derived",
                }
            )
            existing_names.add("enable_shared_expert_dp")

    use_mtp = model_info.get("use_mtp", False) or model_info.get("num_mtp_modules", 0) > 0
    if use_mtp and "num_speculative_tokens" not in existing_names:
        injected.append(
            {
                "name": "num_speculative_tokens",
                "config_position": "env",
                "dtype": "enum",
                # 0 = 关闭投机解码（O2b 关闭档，恒合法）；渲染端 num_spec==0 时
                # 整个 speculative-config 容器不渲染，等效"无投机基线"。
                "default": 3,
                "choices": [0, 2, 3, 4, 5],
                "reason": "MTP-capable model: speculative decoding token count (0 = off)",
                "source": "model_derived",
            }
        )
        existing_names.add("num_speculative_tokens")

    if int(num_layers) > 40 and "cudagraph_mode" not in existing_names:
        injected.append(
            {
                "name": "cudagraph_mode",
                "config_position": "env",
                "dtype": "enum",
                "default": "FULL_DECODE_ONLY",
                "choices": ["FULL_DECODE_ONLY"],
                "reason": f"Deep model ({num_layers} layers): CUDA Graph reduces recapture overhead",
                "source": "model_derived",
            }
        )
        existing_names.add("cudagraph_mode")

    if injected:
        import copy

        result = copy.deepcopy(search_space)
        result.setdefault("parameters", [])
        result["parameters"].extend(injected)
        return result
    return search_space


def _legal_num_spec_values(model_info: dict, explore=(2, 3, 4, 5)) -> list:
    """vLLM spec rule: num_spec <= n_predict, or num_spec % n_predict == 0.

    n_predict = mtp_num_hidden_layers; <=1 (single-layer MTP, e.g.
    Qwen3.5/3.6) makes any positive int legal. ``0`` (speculation OFF /
    O2b 关闭档) is always legal and is kept as the first choice.
    """
    n_predict = int(model_info.get("mtp_num_hidden_layers") or 0)
    if n_predict <= 1:
        return [0] + list(explore)  # common single-layer MTP: keep status quo
    legal = [v for v in explore if v <= n_predict or v % n_predict == 0]
    return [0] + (legal or [1])


def _enforce_num_spec_legality(search_space: dict, model_info: dict) -> dict:
    """Narrow num_speculative_tokens choices/default to MTP-legal values.

    Architecture fact (mtp_num_hidden_layers) > generic experience, so this
    runs LAST in the injection chain and applies regardless of which source
    (known_patterns / model_derived / vendor preset) injected the parameter.
    Illegal values are not caught by validate (only "in choices"), and vLLM
    crashes at serve time instead — keep them out of the search space.
    """
    for _p in search_space.get("parameters", []):
        if str(_p.get("name", "")).lower() != "num_speculative_tokens":
            continue
        legal = _legal_num_spec_values(model_info)
        choices = list(_p.get("choices") or [])
        narrowed = [c for c in choices if c in legal]
        if not narrowed:  # source choices entirely illegal — replace wholesale
            narrowed = legal
        if narrowed != choices:
            _p["choices"] = narrowed
            reason = _p.get("reason", "")
            tail = f" | narrowed to MTP-legal values {narrowed} by mtp_num_hidden_layers"
            if tail not in reason:
                _p["reason"] = (reason + tail).strip()
        default = _p.get("default")
        if default is not None and default not in narrowed:
            nearest = min(narrowed, key=lambda v: abs(v - default))
            _p["default"] = nearest
            reason = _p.get("reason", "")
            note = f" | default moved to {nearest} (was {default})"
            if note not in reason:
                _p["reason"] = (reason + note).strip()
        break
    return search_space


def _enforce_num_spec_base(search_space: dict, context: dict) -> dict:
    """num_speculative_tokens 可搜的前提：speculative-config 有 method/model 底座。

    method 是架构常量而非搜索变量，其字符串（mtp / qwen3_5_mtp /
    deepseek_mtp / eagle3 / dflash ...）由 vLLM 按模型架构注册，只能来自实机
    背书的 preset 场景——apply_launch_config 会把该场景 additional_config 的
    speculative-config（method 底座）写进 config.toml [engine.command].others，
    渲染端（custom_command 容器合并）据此与候选 num_spec 子键拼出完整容器。

    无底座时注入 num_spec 会渲染出孤儿容器
    ``--speculative-config '{"num_speculative_tokens": N}'``，vLLM 拒绝
    （vLLM 报错：provided but without speculative model）。宁缺勿错：
    底座不可得 → 从搜索空间移除该参数（自动复刻缺陷现场"剔除参数"的
    规避语义，但仅对无底座模型生效）。
    """
    params = search_space.get("parameters", [])
    present = any(str(p.get("name", "")).lower() == "num_speculative_tokens" for p in params)
    if not present:
        return search_space
    vp = (context.get("knowledge") or {}).get("vendor_preset") or {}
    spec = (vp.get("additional_config") or {}).get("speculative-config") or {}
    if not isinstance(spec, dict):
        spec = {}
    has_base = bool((spec.get("method") or "").strip()) or bool((spec.get("model") or "").strip())
    if has_base:
        return search_space
    kept = [p for p in params if str(p.get("name", "")).lower() != "num_speculative_tokens"]
    if len(kept) != len(params):
        result = dict(search_space)
        result["parameters"] = kept
        print(
            "WARNING: 模型无投机解码 method 底座（未命中含 speculative-config 的 "
            "preset 场景），从搜索空间移除 num_speculative_tokens —— 孤儿容器 "
            "--speculative-config '{...}' 会被 vLLM 拒绝，宁缺勿错。"
        )
        return result
    return search_space


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
        help="业务输入长度（用户确认），写入 context.workload 的 input_len_avg/max，供 Roofline/KV 推导",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="业务输出长度（用户确认），写入 context.workload 的 output_len_avg/max，供 Roofline/KV 推导",
    )
    parser.add_argument("--recommend-result", default="")
    parser.add_argument("--search-space-file", default="")
    parser.add_argument(
        "--config-toml",
        default="",
        help="Path to config.toml for auto-extracting target_fields from the engine section (vllm or mindie). Auto-discovers optix/config.toml from project root if not specified.",
    )
    # 预算旗标仅作为信息记录写入 context.json；运行时终止以 config.toml
    # [agent_optimizer] 为准（用 config_preflight --set-max-* 回填），默认值与其对齐
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--candidates-per-round", type=int, default=4)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--time-limit-minutes", type=int, default=1440)
    args = parser.parse_args()

    recommendation = read_json(Path(args.recommend_result)) if args.recommend_result else {}
    hardware = recommendation_hardware(recommendation)
    world_size = int(coalesce(args.world_size, hardware.get("world_size"), default=0))
    if world_size <= 0:
        parser.error("--world-size is required unless --recommend-result provides hardware.world_size")

    engine = coalesce(
        args.engine, recommendation_field(recommendation, "engine", "agent_optimizer_handoff.source.engine"), default=""
    )

    # --- Search space decision (3 sources, in priority order) ---
    search_space = default_parallel_search_space(world_size)
    source = {}
    config_handoff = {}
    parallel_aliases = {}

    if args.recommend_result:
        search_space = search_space_from_recommendation(recommendation)
        source["param_recommend_result"] = str(Path(args.recommend_result))
        config_handoff = (
            recommendation.get("config_skill_handoff")
            or (recommendation.get("agent_optimizer_handoff") or {}).get("config_handoff")
            or {}
        )
        parallel_aliases = (recommendation.get("agent_optimizer_handoff") or {}).get("parallel_aliases", {})
    elif args.search_space_file:
        search_space = read_json(Path(args.search_space_file))
        source["search_space_file"] = str(Path(args.search_space_file))
    elif engine:
        config_toml_path = args.config_toml or _find_config_toml()
        if config_toml_path is None:
            # Silently producing a search space without the engine target_fields
            # corrupts the run — hard fail instead.
            parser.error(
                "config.toml not found (no --config-toml, and none discovered from CWD upward). "
                "Pass --config-toml explicitly or run from inside the project."
            )
        parsed = _parse_config_toml(Path(config_toml_path), engine)
        if parsed.get("target_fields"):
            search_space = search_space_from_target_fields(parsed["target_fields"])
            # Merge tp/dp/pp from default search space if not already present
            default_ss = default_parallel_search_space(world_size)
            existing_names = {p["name"].lower() for p in search_space.get("parameters", [])}
            for param in default_ss.get("parameters", []):
                if param["name"].lower() not in existing_names:
                    search_space.setdefault("parameters", []).append(param)
            source["config_toml"] = str(config_toml_path)
            config_handoff = {
                "target_fields": parsed["target_fields"],
                "source": "config.toml",
            }

    num_nodes = int(coalesce(args.num_nodes, hardware.get("num_nodes"), default=1))
    num_per_node = int(
        coalesce(args.num_per_node, hardware.get("num_per_node"), hardware.get("num_per_nodes"), default=1)
    )
    memory_gb = float(
        coalesce(
            args.memory_gb,
            hardware.get("single_card_memory_gb"),
            hardware.get("single_card_mem_gb"),
            default=0,
        )
    )
    benchmark = coalesce(
        args.benchmark,
        recommendation_field(
            recommendation, "benchmark", "benchmark_policy", "agent_optimizer_handoff.source.benchmark_policy"
        ),
        default="ais_bench",
    )
    # 工作负载：优先 recommend-result；用户确认的输入/输出长度（--input-len/--output-len）
    # 覆盖并标记为已确认。两者皆缺时 workload_confirmed=false，experience_injector 据此
    # 在报告中声明数值推导置信度低，agent 须先向用户确认再参考。
    workload = recommendation_field(recommendation, "workload", "agent_optimizer_handoff.workload") or {}
    if args.input_len is not None:
        workload["input_len_avg"] = args.input_len
        workload["input_len_max"] = args.input_len
    if args.output_len is not None:
        workload["output_len_avg"] = args.output_len
        workload["output_len_max"] = args.output_len
    workload_confirmed = bool(workload.get("input_len_avg") and workload.get("output_len_avg"))

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
            "name": coalesce(
                args.model_name,
                recommendation_field(recommendation, "model.name", "agent_optimizer_handoff.model.name"),
                default="",
            ),
            "path": coalesce(
                args.model_path,
                recommendation_field(
                    recommendation,
                    "model.path",
                    "model.config_path",
                    "agent_optimizer_handoff.model.path",
                    "agent_optimizer_handoff.model.config_path",
                ),
                default="",
            ),
        },
        "model_info": _read_model_info(
            coalesce(
                args.model_path,
                recommendation_field(
                    recommendation,
                    "model.path",
                    "model.config_path",
                    "agent_optimizer_handoff.model.path",
                    "agent_optimizer_handoff.model.config_path",
                ),
                default=None,
            )
        ),
        "workload": workload,
        "workload_confirmed": workload_confirmed,
        "target": recommendation_field(recommendation, "target", "agent_optimizer_handoff.target") or "",
        "search_space": search_space,
        "constraints": recommendation_constraints(recommendation, world_size, engine),
    }
    # --- Inject knowledge (patterns) right after constraints ---------------
    known = _collect_known_patterns(context)
    if known:
        context["search_space"] = _inject_knowledge_params(context["search_space"], known, engine)

    # NEW: model-derived injection
    model_info = context.get("model_info", {}) or {}
    context["search_space"] = _inject_model_derived_params(context["search_space"], model_info)

    knowledge = {"known_patterns": known} if known else {}
    if knowledge:
        context["knowledge"] = knowledge
    # --- Vendor preset injection (vLLM Ascend 官方推荐配置) ----------------
    # Must run AFTER the knowledge block above, which reassigns
    # context["knowledge"] and would otherwise clobber vendor_preset.
    context = _inject_vendor_preset(context)
    # --- Architecture legality enforcement (LAST in injection chain) -------
    # Narrow num_speculative_tokens choices/default to values this model's MTP
    # stack accepts (num_spec <= n_predict or num_spec % n_predict == 0),
    # regardless of which source injected the param. Illegal values escape
    # validate ("value in choices") and only crash vLLM at serve time.
    context["search_space"] = _enforce_num_spec_legality(context["search_space"], model_info)
    # --- Speculative method base enforcement (LAST, after vendor preset) ----
    # num_spec is only searchable when a real (harvested) speculative-config
    # method base exists in the matched preset — otherwise the candidate-only
    # param renders an orphan container that vLLM rejects.
    context["search_space"] = _enforce_num_spec_base(context["search_space"], context)
    # --- Append remaining fields ------------------------------------------
    # Budgets: align with config.toml [agent_optimizer] (the real runtime
    # termination source) when available; CLI flags act as overrides.
    # Fallback defaults (32/12/1440) must not diverge from the actual config.
    cfg_budgets = {}
    cfg_path = args.config_toml or _find_config_toml()
    if cfg_path and Path(cfg_path).exists():
        try:
            parsed_ao = tomllib.loads(Path(cfg_path).read_text(encoding="utf-8")).get("agent_optimizer", {}) or {}
            cfg_budgets = {
                "max_rounds": parsed_ao.get("max_rounds"),
                "candidates_per_round": parsed_ao.get("candidates_per_round"),
                "max_trials": parsed_ao.get("max_trials"),
                "time_limit_minutes": parsed_ao.get("time_limit_minutes"),
            }
        except Exception:
            cfg_budgets = {}
    context["budgets"] = {
        "max_rounds": args.max_rounds if args.max_rounds != 32 else (cfg_budgets.get("max_rounds") or args.max_rounds),
        "candidates_per_round": (
            args.candidates_per_round
            if args.candidates_per_round != 4
            else (cfg_budgets.get("candidates_per_round") or args.candidates_per_round)
        ),
        "max_trials": args.max_trials if args.max_trials != 12 else (cfg_budgets.get("max_trials") or args.max_trials),
        "time_limit_minutes": (
            args.time_limit_minutes
            if args.time_limit_minutes != 1440
            else (cfg_budgets.get("time_limit_minutes") or args.time_limit_minutes)
        ),
    }
    context["source"] = source
    context["parallel_aliases"] = parallel_aliases
    context["config_handoff"] = config_handoff
    # -----------------------------------------------------------------------
    write_json(Path(args.run_dir) / "context.json", context)


def _inject_knowledge_params(search_space: dict, known_hints: list, engine: str) -> dict:
    """Inject known_patterns suggested parameters that are missing from search_space.

    Returns a (possibly modified) copy of ``search_space`` with new parameter
    entries derived from ``known_hints``.  Existing parameters (matched by
    case-insensitive name) are never overwritten.
    """
    existing_names = {p["name"].lower() for p in search_space.get("parameters", [])}
    existing_names |= {c["name"].lower() for c in search_space.get("constants", [])}

    section = engine.lower() if engine else "vllm"
    injected: list[dict] = []

    for hint in known_hints:
        hint_params = hint.get("params") or {}
        for key, value in hint_params.items():
            if key.lower() in existing_names:
                continue  # already present, don't override

            entry = _param_from_knowledge(key, value, section)
            if entry is None:
                continue

            existing_names.add(key.lower())
            injected.append(entry)

    if injected:
        import copy

        result = copy.deepcopy(search_space)
        result.setdefault("parameters", [])
        result["parameters"].extend(injected)
        return result
    return search_space


def _param_from_knowledge(name: str, value, section: str) -> dict | None:
    """Convert a known_patterns param value into a search_space parameter entry."""
    if isinstance(value, bool):
        return {
            "name": name,
            "section": section,
            "config_position": "env",
            "dtype": "enum",
            "default": value,
            "min": 0,
            "max": 1,
            "choices": [False, True],
            "reason": "Injected from known_patterns (human-curated rule).",
            "source": "known_patterns",
        }
    if isinstance(value, dict) and "explore" in value:
        explore = value["explore"]
        if isinstance(explore, list) and explore:
            return {
                "name": name,
                "section": section,
                "config_position": "env",
                "dtype": "enum",
                "default": explore[0],
                "min": 0,
                "max": 1,
                "choices": explore,
                "reason": f"Injected from known_patterns: explore {explore!r}.",
                "source": "known_patterns",
            }
    if isinstance(value, str) and value:
        return {
            "name": name,
            "section": section,
            "config_position": "env",
            "dtype": "enum",
            "default": value,
            "min": 0,
            "max": 1,
            "choices": [value],
            "reason": "Injected from known_patterns (human-curated rule).",
            "source": "known_patterns",
        }
    # Numeric scalar (int/float) — becomes a searchable int/float param.
    # Used e.g. api-server-count, max-num-batched-tokens, gpu-memory-utilization.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        is_int = isinstance(value, int) and not isinstance(value, bool)
        return {
            "name": name,
            "section": section,
            "config_position": "env",
            "dtype": "int" if is_int else "float",
            "default": value,
            "min": 1 if is_int else 0.0,
            "max": max(value * 2, 2) if is_int else max(float(value) * 2, 1.0),
            "reason": "Injected from known_patterns (human-curated rule).",
            "source": "known_patterns",
        }
    # JSON-able container (list/dict) — a fixed value rendered as a JSON flag,
    # NOT a searchable range (no clean numeric range for it). dtype=str keeps it
    # as a single-choice enum so optix renders the value verbatim.
    if isinstance(value, (list, dict)):
        if isinstance(value, dict) and ({"direction", "max_multiplier"} & set(value)):
            # Hint-only metadata (e.g. {"direction":"increase","max_multiplier":4})
            # carries no concrete value/base — skip injection instead of
            # materializing the hint as a bogus dtype=str search param.
            return None
        return {
            "name": name,
            "section": section,
            "config_position": "env",
            "dtype": "str",
            "default": json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            "min": 0,
            "max": 1,
            "choices": [json.dumps(value, ensure_ascii=False, separators=(",", ":"))],
            "reason": "Injected from known_patterns (human-curated rule).",
            "source": "known_patterns",
        }
    return None


def _collect_known_patterns(context: dict) -> list:
    """Load known_patterns.json and return matched hints, or []."""
    patterns_path = Path(__file__).resolve().parent.parent / "knowledge" / "known_patterns.json"
    if not patterns_path.exists():
        return []
    try:
        return match_known_patterns(read_json(patterns_path), context)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Vendor preset injection (vLLM Ascend 官方推荐配置目录)
# See docs/design/vllm-ascend-preset-catalog-design.md
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


def _param_from_vendor_preset(name: str, value, preset: dict, scenario: dict) -> dict | None:
    """Convert a vendor preset param into a search-space entry (default = official
    value, with a searchable band around it). Only injects missing params.
    """
    reason = f"vLLM Ascend 官方推荐 ({preset.get('model', {}).get('name', '?')}): {scenario.get('source_url', '')}"
    if isinstance(value, bool):
        return {
            "name": name,
            "config_position": "env",
            "dtype": "enum",
            "default": value,
            "min": 0,
            "max": 1,
            "choices": [False, True],
            "source": "vendor_preset",
            "reason": reason,
        }
    if isinstance(value, int):
        return {
            "name": name,
            "config_position": "env",
            "dtype": "int",
            "default": value,
            "min": max(1, value // 2),
            "max": value * 2,
            "source": "vendor_preset",
            "reason": reason,
        }
    if isinstance(value, float):
        return {
            "name": name,
            "config_position": "env",
            "dtype": "ratio",
            "default": value,
            "min": max(0.0, value - 0.1),
            "max": min(1.0, value + 0.1),
            "source": "vendor_preset",
            "reason": reason,
        }
    if isinstance(value, str):
        return {
            "name": name,
            "config_position": "env",
            "dtype": "enum",
            "default": value,
            "min": 0,
            "max": 1,
            "choices": [value],
            "source": "vendor_preset",
            "reason": reason,
        }
    return None


#: additional_config 中"值得单变量反转验证"的引擎级固定参数 -> 搜索空间参数名。
#: 这类参数由 vendor preset 写死进 config.toml `others`，但可能并非目标硬件上的
#: 最优值（例如 speculative-config.enforce_eager=true 禁用了 CUDA Graph）。
#: 登记后 Round 1 必须至少测一次相反值，收敛判定前必须声明已验证/未验证。
#: 注入名用 speculative_enforce_eager 而非裸名 enforce_eager：裸名 enforce_eager
#: 是主模型顶层 --enforce-eager（见 optix custom_command FLAG_NAME_MAP），与容器内
#: 投机 eager 语义不同——一义一名，避免渲染端把容器意图错路由到顶层。
_FLIPPABLE_ENGINE_FIXED_PARAMS = {
    "speculative-config.enforce_eager": "speculative_enforce_eager",
}

#: additional_config 中命令骨架字段（config.toml command 有专用槽位），不视为固定参数
_COMMAND_SKELETON_KEYS = {"host", "port", "model", "served_model_name", "served-model-name"}


def _flatten_additional_config(additional_config: dict, prefix: str = "") -> list[dict]:
    """Flatten additional_config (excluding command-skeleton keys) into leaf entries.

    e.g. {"speculative-config": {"method": "qwen3_5_mtp", "enforce_eager": true}}
    -> [{"path": "speculative-config.method", "value": "qwen3_5_mtp"},
        {"path": "speculative-config.enforce_eager", "value": True}]
    """
    entries: list[dict] = []
    for key, value in (additional_config or {}).items():
        if key in _COMMAND_SKELETON_KEYS:
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            entries.extend(_flatten_additional_config(value, path))
        else:
            entries.append({"path": path, "value": value})
    return entries


def _register_engine_fixed_params(
    search_space: dict, additional_config: dict, existing: set, preset: dict, scenario: dict
) -> list[dict]:
    """Register preset additional_config keys as engine-level fixed params.

    Every leaf key becomes a ``context.engine_fixed_params`` entry recording its
    value, source, and whether a single-variable flip verification is required.
    Flippable params also get a search-space entry (rendered via JSON_SUBKEY_MAP
    in optix/config/custom_command.py) so candidates can carry the flipped value.
    """
    reason = (
        f"vLLM Ascend 官方推荐固定参数 ({preset.get('model', {}).get('name', '?')}): {scenario.get('source_url', '')}"
    )
    entries: list[dict] = []
    for leaf in _flatten_additional_config(additional_config):
        path = leaf["path"]
        search_name = _FLIPPABLE_ENGINE_FIXED_PARAMS.get(path)
        # 容器原子性：speculative-config 子键可 flip 的前提是容器
        # 已有 method/model 底座；无底座宁缺勿错——不注入搜索空间（渲染端对孤儿
        # 容器同样整容器丢弃，装配/渲染双端一致）。
        container_ok = True
        if path.startswith("speculative-config."):
            _spec_cfg = additional_config.get("speculative-config") or {}
            container_ok = bool(_spec_cfg.get("method") or _spec_cfg.get("model"))
        flip_ok = bool(search_name) and container_ok
        entries.append(
            {
                "path": path,
                "value": leaf["value"],
                "source": "vendor_preset",
                "flip_required": flip_ok,
                "search_name": search_name if flip_ok else None,
                "verified": False,
            }
        )
        if flip_ok and search_name.lower() not in existing and isinstance(leaf["value"], bool):
            search_space.setdefault("parameters", []).append(
                {
                    "name": search_name,
                    "config_position": "env",
                    "dtype": "enum",
                    "default": leaf["value"],
                    "min": 0,
                    "max": 1,
                    "choices": [False, True],
                    "source": "vendor_preset_engine_fixed",
                    "reason": reason + "（单变量反转验证用）",
                }
            )
            existing.add(search_name.lower())
    return entries


def _inject_vendor_preset(context: dict) -> dict:
    """Inject official recommended params from the preset catalog into context.

    Only fills params the search space does NOT already define (never overrides
    user configuration / existing entries), and records the matched scenario in
    ``knowledge.vendor_preset`` so the agent can use it as the Round-1 baseline.

    ``hardware`` is advisory, not a gate: parallel sizes (tp/dp/pp) are already
    in the search space and never overridden, and the other recommended values
    (MAX_NUM_SEQS, cudagraph_mode, GPU_MEMORY_UTILIZATION, …) transfer across
    card counts. So a card-count mismatch never blocks injection — it is recorded
    for the agent instead. No match / missing catalog → context unchanged.
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

    import copy

    result = copy.deepcopy(context)
    search_space = result["search_space"]
    existing = {p["name"].lower() for p in search_space.get("parameters", [])}
    existing |= {c["name"].lower() for c in search_space.get("constants", [])}

    for pname, pvalue in (scenario.get("params") or {}).items():
        key = pname.lower()
        if key not in existing:
            entry = _param_from_vendor_preset(pname, pvalue, preset, scenario)
            if entry:
                search_space.setdefault("parameters", []).append(entry)
                existing.add(key)
            continue
        # Param already in the search space (e.g. injected earlier by
        # known_patterns or default_parallel_search_space) — the official
        # vendor_preset value should still become the DEFAULT (Round-1 baseline
        # seed, per agent-mode contract "官方值作基线 / strong_suggest"), NOT be
        # dropped by `existing`. Override default only when the existing entry is
        # generic (not user/config-sourced), and only touch default — choices /
        # min / max stay as the agent's search range. Otherwise num_spec from
        # known_patterns (default=2) silently wins over the model's official 3.
        for _p in search_space.get("parameters", []):
            if _p.get("name", "").lower() == key and _p.get("source") not in ("config",):
                old_default = _p.get("default")
                if old_default != pvalue and pvalue not in (None, ""):
                    _p["default"] = pvalue
                    _p["reason"] = f"官方推荐默认（vendor_preset {scenario.get('id', '')}，覆盖通用默认 {old_default}）"
                break

    result.setdefault("knowledge", {})
    result["knowledge"]["vendor_preset"] = {
        "model": preset.get("model", {}).get("name", ""),
        "scenario_id": scenario.get("id", ""),
        "description": scenario.get("description", ""),
        "params": scenario.get("params", {}),
        "env": scenario.get("env", {}),
        "additional_config": scenario.get("additional_config", {}),
        "source_url": scenario.get("source_url", ""),
        # hardware 是参考信息：官方推荐硬件 vs 实际运行硬件，matched=False 时
        # agent 应意识到并行参数是按推荐硬件给的，其余推荐值仍可用作种子
        "hardware": {
            "recommended": preset.get("hardware") or [],
            "actual_world_size": (context.get("constraints") or {}).get("world_size"),
            "matched": _hardware_matches(preset, context),
        },
    }
    # 登记引擎级固定参数：additional_config 的每个 key 都进入待验证清单，
    # 可反转的（如 speculative-config.enforce_eager）同时注入搜索空间。
    engine_fixed_params = _register_engine_fixed_params(
        search_space,
        scenario.get("additional_config", {}),
        existing,
        preset,
        scenario,
    )
    if engine_fixed_params:
        result["engine_fixed_params"] = engine_fixed_params
    return result


if __name__ == "__main__":
    main()

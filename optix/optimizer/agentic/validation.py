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

# optix/optimizer/agentic/validation.py
"""Single source of truth for candidate parameter validation and failure classification."""

import ast
import operator as op
import re
import subprocess
import shutil
from math import isclose
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

# 启动期配置类错误置于表首、先于 oom 命中：speculative-config / ValidationError 等
# 启动失败若按旧逻辑全文匹配，会被服务日志中的 memory 字样误判为 oom/weight →
# suggested_action=increase_tp（TP=全部卡数时不可执行，方向性误导）。
FAILURE_CLASSIFICATION_TABLE = {
    "spec_config": {
        "validation": {
            "patterns": [
                r"SpeculativeConfig",
                r"create_speculative_config",
                r"num_speculative_tokens",
                r"speculative (model|method|draft)",
                r"without speculative",
                r"ValidationError",
            ],
            "suggested_action": "fix_spec_config",
        },
    },
    "unsupported_flag": {
        "argparse": {
            "patterns": [
                r"unrecognized arguments",
                r"unknown arguments",
                r"invalid choice",
                r"no such option",
                r"expected one argument",
                r"unsupported.*(flag|option|argument)",
            ],
            "suggested_action": "remove_unsupported_flag",
        },
    },
    "oom": {
        "kv_cache": {
            "patterns": [r"kv.?cache", r"KV.?cache", r"kvcache", r"cache.*memory", r"out of.*cache"],
            "suggested_action": "reduce_batch",
        },
        "weight": {
            "patterns": [r"weight.*memory", r"model.*load.*fail", r"cuda.?malloc.*weight"],
            "suggested_action": "increase_tp",
        },
        "activation": {
            "patterns": [r"activation.*memory", r"hidden.*states", r"intermediate.*buffer"],
            "suggested_action": "reduce_batch",
        },
        "_default": {
            "patterns": [
                r"out of memory",
                r"OOM",
                r"CUDA_OUT_OF_MEMORY",
                r"MemoryError",
                r"NPU.*OOM",
                r"memory.*exceed",
            ],
            "suggested_action": "reduce_batch",
        },
    },
    "constraint": {
        "tp_dp_pp": {
            "patterns": [r"TP.*DP.*PP", r"tensor.*parallel.*data.*parallel", r"parallel.*config.*invalid"],
            "suggested_action": "fix_parallel",
        },
        "ratio": {
            "patterns": [r"ratio.*out", r"dtype_param.*invalid", r"must be less than"],
            "suggested_action": "fix_ratio",
        },
        "_default": {
            "patterns": [r"constraint.*fail", r"invalid.*config", r"parameter.*out.*range"],
            "suggested_action": "fix_ratio",
        },
    },
    "service": {
        "startup_timeout": {
            "patterns": [r"start.*timeout", r"wait.*timeout", r"server.*start.*fail"],
            "suggested_action": "retry",
        },
        "crash": {
            "patterns": [r"segfault", r"SIGSEGV", r"SIGABRT", r"core.?dump", r"process.*exit", r"unexpected.*exit"],
            "suggested_action": "retry_or_skip",
        },
        "_default": {
            "patterns": [r"service.*error", r"server.*error", r"connection.*refused"],
            "suggested_action": "retry",
        },
    },
    "benchmark": {
        "timeout": {
            "patterns": [r"bench.*timeout", r"eval.*timeout", r"timed out"],
            "suggested_action": "retry",
        },
        "connection_refused": {
            "patterns": [r"connection refused", r"ECONNREFUSED", r"cannot connect"],
            "suggested_action": "retry",
        },
        "_default": {
            "patterns": [r"benchmark.*fail", r"eval.*fail"],
            "suggested_action": "retry",
        },
    },
    "slo": {
        "ttft": {
            "patterns": [r"TTFT.*exceed", r"time.to.first.*SLO", r"first.token.*latency.*SLO"],
            "suggested_action": "reduce_concurrency",
        },
        "tpot": {
            "patterns": [r"TPOT.*exceed", r"time.per.output.*SLO", r"inter.token.*latency.*SLO"],
            "suggested_action": "reduce_concurrency",
        },
        "_default": {
            "patterns": [r"SLO.*violat", r"latency.*SLO", r"throughput.*target"],
            "suggested_action": "reduce_concurrency",
        },
    },
}


FAILURE_MESSAGE_MAX = 500


def _clip_message(text: str, limit: int = FAILURE_MESSAGE_MAX) -> str:
    """Cap message length while keeping the final (actionable) line intact.

    启动/评测日志的有意义异常信息位于尾部（traceback 最后一行）；旧的
    text[:500] 从头截断会把 log tail 截没（实测 O2 场景 message 仅剩 "l"）。
    分级截断：head 截短 + "..." 标记 + 最后一行强制保留。
    """
    if len(text) <= limit:
        return text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text[:limit]
    last = lines[-1]
    # 尾行过长时仍需压缩，但保证保留其结尾片段
    tail = last if len(last) <= limit - 3 else "..." + last[-(limit - 3) :]
    head_budget = limit - len(tail) - 3
    head = text[:head_budget].rstrip() if head_budget > 0 else ""
    return f"{head}...{tail}"


def _scan_candidates(candidates) -> Optional[Tuple[Dict[str, Any], str]]:
    """Run classification table against an iterable of candidate text regions.

    Returns (result_dict, matched_region) or None. 每个 region 内部复刻旧的两阶段
    语义：先全表 named 子类别（含新置于表首的 spec_config/unsupported_flag），
    全部未命中后再全表 _default 兜底——保证 named 永远优先于 _default、且
    新类别（无 _default）不会抢占后续类别的 named 命中。
    """
    for region in candidates:
        if not region:
            continue
        # 阶段 1: named 子类别（按表顺序）
        for category, sub_cats in FAILURE_CLASSIFICATION_TABLE.items():
            for sub_key, sub_info in sub_cats.items():
                if sub_key == "_default":
                    continue
                for pattern in sub_info.get("patterns", []):
                    if re.search(pattern, region, re.IGNORECASE):
                        return {
                            "category": category,
                            "sub_category": sub_key,
                            "suggested_action": sub_info["suggested_action"],
                        }, region
        # 阶段 2: _default 兜底
        for category, sub_cats in FAILURE_CLASSIFICATION_TABLE.items():
            default_info = sub_cats.get("_default")
            if not default_info:
                continue
            for pattern in default_info.get("patterns", []):
                if re.search(pattern, region, re.IGNORECASE):
                    return {
                        "category": category,
                        "sub_category": "other",
                        "suggested_action": default_info["suggested_action"],
                    }, region
    return None


def classify_failure(error_info: str, outcome: Any = None) -> Optional[Dict[str, Any]]:
    """Classify a failure from error string and outcome object into structured format.

    Returns None if the error string is empty or represents a success.
    Returns a dict with category, sub_category, message, evidence_line,
    suggested_action fields.

    匹配按"尾部异常行优先"：traceback/log tail 的最后几行承载真正的异常，
    因此先对末尾最多 5 个非空行倒序匹配；未命中再对全文逐行正扫兜底，
    保持与旧"全文 search"等价（所有 pattern 均不跨行）。
    """
    text = (str(error_info) if error_info else "").strip()
    if not text:
        return None

    # Check outcome for structured SLO violations first
    if outcome and hasattr(outcome, "status") and getattr(outcome, "status", None) == "failed":
        text = f"{text} {getattr(outcome, 'message', '')}"

    lines = [ln for ln in text.splitlines() if ln.strip()]

    if lines:
        # 1) 尾部异常行优先：末尾至多 5 个非空行，倒序（最后一行最先匹配）
        hit = _scan_candidates(list(reversed(lines[-5:])))
        # 2) 兜底：全文逐行正扫（与旧全文 search 语义等价）
        if hit is None:
            hit = _scan_candidates(lines)
    else:
        hit = _scan_candidates([text])

    if hit is None:
        return None
    result, matched_line = hit
    result["message"] = _clip_message(text)
    result["evidence_line"] = matched_line.strip()
    return result


# ---------------------------------------------------------------------------
# Expression evaluation (safe, whitelist-based AST)
# ---------------------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
    ast.Not: op.not_,
}


def _replace_variables(expr: str, values: Dict[str, Any]) -> str:
    """Replace $VAR references with their values from the values dict."""

    def _repl(match: re.Match) -> str:
        name = match.group(1).upper()
        if name not in values:
            raise KeyError(name)
        return str(values[name])

    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", _repl, expr)


def _eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST expression node using the whitelist."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BoolOp):
        result = _eval_node(node.values[0])
        for value in node.values[1:]:
            result = _ALLOWED_OPS[type(node.op)](result, _eval_node(value))
        return result
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left)
        for cmp_op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator)
            if not _ALLOWED_OPS[type(cmp_op)](left, right):
                return False
            left = right
        return True
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def eval_expression(expression: str) -> bool:
    """Safely evaluate a constraint expression. Returns True if satisfied."""
    try:
        return bool(_eval_node(ast.parse(expression, mode="eval")))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _values_equal(left: Any, right: Any) -> bool:
    """Compare two values, handling numeric tolerance."""
    try:
        return isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-8)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _is_constant_or_derived(item: Dict[str, Any]) -> bool:
    """Check if a parameter is effectively constant (non-tunable)."""
    if item.get("dtype") in {"factories", "times"}:
        return True
    if "constant" in item and item.get("constant") is not None:
        return True
    try:
        return isclose(float(item.get("min")), float(item.get("max")), rel_tol=1e-5)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Derived-parameter contract — mirrors the engine's DERIVED_FIELD_HANDLERS
# (optix.config.config) only where a candidate-set value has meaning. The rule:
# mirror what a candidate may set, reject what the engine computes.
# ---------------------------------------------------------------------------

#: Derived dtypes the validation layer understands. Candidates MAY set these:
#: factories/times are checked against the engine derivation (see
#: ``_apply_derived_params``); ratio is range-checked (candidate value is the
#: RATIO — the engine multiplies by the target at render time).
DERIVED_CANDIDATE_SETTABLE = frozenset({"factories", "times", "ratio"})

#: Derived dtypes the ENGINE computes from dependency fields. A candidate-set
#: value is meaningless — the engine overwrites it (share = min+max-value) or
#: can crash on a missing dtype_param (le_enum needs ``values``/``target_name``,
#: which dynamic field creation does not carry). Candidates MUST NOT set these;
#: validation rejects them explicitly instead of silently passing a value that
#: diverges from — or breaks — the engine run.
DERIVED_ENGINE_ONLY = frozenset({"share", "ternary_factories", "ternary_times", "le_enum"})


def _apply_derived_params(params: Dict[str, Any], schema_items: List[Dict[str, Any]]) -> None:
    """Compute derived parameter values from their dependencies.

    Mirrors the engine only for factories/times (single-target dict dtype_param
    with ``target_name`` + ``product``) — the two derived dtypes a candidate is
    allowed to set, where a conflict with the engine derivation must be caught.
    The other engine derived dtypes are deliberately NOT mirrored here:
    share / ternary_factories / ternary_times / le_enum are computed by the
    engine from dependencies and candidates are rejected upstream (see
    DERIVED_ENGINE_ONLY), so they never flow into this function; a ternary-style
    dtype_param (list ``target_names``) falls through the ``target_name`` lookup
    and is skipped, which is the intended no-op for an unsupported shape.
    """
    for item in schema_items:
        dtype = item.get("dtype")
        dtype_param = item.get("dtype_param")
        if not isinstance(dtype_param, dict):
            continue
        target_name = dtype_param.get("target_name")
        if target_name not in params or params.get(target_name) in (None, 0):
            continue
        try:
            target_value = float(params[target_name])
            product = float(dtype_param.get("product", 1))
            if dtype == "factories":
                value = product / target_value
            elif dtype == "times":
                value = product * target_value
            else:
                continue
            value_dtype = dtype_param.get("dtype", "float")
            if value_dtype == "int":
                params[item["name"]] = int(value)
            elif value_dtype == "bool":
                params[item["name"]] = bool(value)
            else:
                params[item["name"]] = value
        except (TypeError, ValueError, ZeroDivisionError):
            continue


def _validate_ratio_param(
    name: str, value: Any, item: Dict[str, Any], params: Dict[str, Any], errors: List[str]
) -> None:
    """Validate a ratio-type parameter (candidate value is the RATIO itself).

    Two shapes share this dtype:
    - plain ratio (no dtype_param), e.g. GPU_MEMORY_UTILIZATION: the value IS the
      final ratio and is consumed directly by the engine — range check only;
    - target-backed ratio (dtype_param names a target field): the engine computes
      ``int(ratio x target.value)`` at render time (update_optimizer_value). This
      layer must NOT pre-multiply and write the absolute value back into params —
      the engine would multiply a second time, silently squaring the magnitude.
      Only the ratio range is checked here; target presence is verified by the
      caller (validate_candidate) before this function runs.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name}={value!r} is not numeric")
        return
    min_value = float(item.get("min", numeric))
    max_value = float(item.get("max", numeric))
    if numeric < min_value or numeric > max_value:
        errors.append(f"{name}={value!r} is outside ratio range [{item.get('min')}, {item.get('max')}]")


def _get_schema_items(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract schema items from context search_space."""
    search_space = context.get("search_space", {})
    return list(search_space.get("parameters", [])) + list(search_space.get("constants", []))


# ---------------------------------------------------------------------------
# Combo safety checks — catches param combinations that are individually valid
# but collectively likely to cause OOM or service startup failure.
# ---------------------------------------------------------------------------


def check_combo_safety(params: Dict[str, Any], context: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Check param combinations for safety issues that individual range checks miss.

    Args:
        params: Resolved full params dict (after defaults + derived).
        context: Full context dict with search_space, hardware, model_info.

    Returns:
        (errors, warnings) — errors are hard-blocks, warnings are advisory.
    """
    errors: List[str] = []
    warnings: List[str] = []

    def _get(name: str, default: Any = None) -> Any:
        """Look up a param case-insensitively."""
        for key, value in params.items():
            if key.upper() == name.upper():
                return value
        return default

    # 1. GPU memory utilization hard boundary
    gpu_mem = _get("GPU_MEMORY_UTILIZATION")
    if gpu_mem is not None:
        try:
            if float(gpu_mem) > 0.95:
                errors.append(f"GPU_MEMORY_UTILIZATION={gpu_mem} > 0.95: almost certain OOM")
        except (TypeError, ValueError):
            pass

    # 2. KV cache rough estimation
    hardware = context.get("hardware", {}) or {}
    single_card_memory_gb = float(hardware.get("single_card_memory_gb", 0) or 0)
    if single_card_memory_gb > 0:
        batch_tokens = _get("MAX_NUM_BATCHED_TOKENS", 0) or 0
        num_seqs = _get("MAX_NUM_SEQS", 0) or 0
        block_size = _get("BLOCK_SIZE", 16) or 16
        # Rough KV cache estimate: 2 layers of hidden_dim per token per seq
        # Use conservative 8192 hidden_dim if model_info unavailable
        hidden_size = 8192
        try:
            batch_tokens_f = float(batch_tokens)
            num_seqs_f = float(num_seqs)
            block_size_f = float(block_size)
            kv_bytes = batch_tokens_f * num_seqs_f * block_size_f * hidden_size * 2
            kv_gb = kv_bytes / (1024**3)
            mem_80pct = single_card_memory_gb * 0.8
            if kv_gb > mem_80pct:
                warnings.append(
                    f"KV cache estimate {kv_gb:.1f}GB > 80% single-card memory "
                    f"({mem_80pct:.1f}GB): OOM risk. "
                    f"Consider reducing BATCH_TOKENS ({batch_tokens}), SEQS ({num_seqs}), "
                    f"or BLOCK_SIZE ({block_size})"
                )
        except (TypeError, ValueError):
            pass

    # 3. Concurrency-batch conflict
    concurrency = _get("CONCURRENCY")
    batch_tokens = _get("MAX_NUM_BATCHED_TOKENS")
    if concurrency is not None and batch_tokens is not None:
        try:
            conc_f = float(concurrency)
            batch_f = float(batch_tokens)
            # Default batch from search space is the baseline
            search_space = context.get("search_space", {})
            default_batch = 65536
            for p in search_space.get("parameters", []):
                if p.get("name", "").upper() == "MAX_NUM_BATCHED_TOKENS":
                    default_batch = float(p.get("default", 65536))
                    break
            if conc_f > 128 and batch_f > default_batch * 2:
                warnings.append(
                    f"CONCURRENCY={concurrency} > 128 combined with "
                    f"BATCH_TOKENS={batch_tokens} > {default_batch * 2:.0f} (2× default): "
                    f"high TTFT/TPOT risk"
                )
        except (TypeError, ValueError):
            pass

    return errors, warnings


# ---------------------------------------------------------------------------
# CLI support check — runtime dynamic, no hardcoded flag lists
# ---------------------------------------------------------------------------

# Cache per process: CLI help output doesn't change during a run
_cli_support_cache: Dict[str, Set[str]] = {}


def get_supported_cli_flags(engine: str = "vllm") -> Set[str]:
    """Query current runtime for all supported CLI flags.

    Runs ``<engine> serve --help`` and ``<engine> bench serve --help``,
    parses out every ``--flag-name`` pattern, and returns the union.

    Results are cached per engine — subsequent calls return instantly.
    Returns an empty set if the binary is not found.
    """
    cache_key = engine.lower()
    if cache_key in _cli_support_cache:
        return _cli_support_cache[cache_key]

    binary = shutil.which(engine)
    if binary is None:
        _cli_support_cache[cache_key] = set()
        return set()

    flags: Set[str] = set()
    help_commands = [
        [binary, "serve", "--help=all"],
        [binary, "bench", "serve", "--help"],
    ]

    for cmd in help_commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            output = result.stdout + result.stderr
            # Extract --flag-name patterns from help text
            for match in re.finditer(r"--([a-z][a-z0-9-]*)", output):
                flags.add(match.group(1))
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue

    _cli_support_cache[cache_key] = flags
    return flags


def _param_to_cli_flag(name: str) -> str:
    """Map a search-space param name to the real vLLM CLI flag name (no '--').

    Single source of truth lives in ``optix.config.custom_command`` so command
    generation and runtime validation can never drift.
    """
    from ...config.custom_command import param_to_cli_flag as _map_flag

    return _map_flag(name)


# ---------------------------------------------------------------------------
# Main validation function — SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------


def validate_candidate(context: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single candidate against the context's search space and constraints.

    Args:
        context: The full context dict with search_space and constraints.
        candidate: A dict with at least candidate_id and params keys.

    Returns:
        A dict with keys: candidate_id, valid, params (resolved), errors, warnings, candidate.
    """
    errors: List[str] = []
    warnings: List[str] = []
    params = dict(candidate.get("params", {}))
    schema_items = _get_schema_items(context)
    schema = {item["name"]: item for item in schema_items}

    # 1. Check for unknown parameter names
    for name in params:
        if name not in schema:
            errors.append(f"Unknown parameter: {name}")

    # 2. Resolve: defaults + candidate overrides + derived
    full_params = {name: item.get("default") for name, item in schema.items()}
    full_params.update(params)
    _apply_derived_params(full_params, schema_items)

    # 3. Validate each candidate-specified parameter
    for name, value in params.items():
        item = schema.get(name)
        if not item:
            continue
        dtype = item.get("dtype")
        if dtype == "enum" and value not in item.get("choices", []):
            errors.append(f"{name}={value!r} is not in choices {item.get('choices', [])!r}")
        elif dtype == "ratio":
            # Target-backed ratio: verify the declared target exists in the field
            # model and has a resolved value BEFORE range-checking. A missing
            # target would otherwise let the engine keep the raw ratio
            # unmultiplied (wrong magnitude) or crash — reject at validation.
            target_name = item.get("dtype_param")
            if target_name is not None and not isinstance(target_name, str):
                errors.append(f"{name}: ratio dtype_param must be a target field name string, got {target_name!r}")
            elif isinstance(target_name, str) and target_name:
                if target_name not in schema:
                    errors.append(
                        f"{name}: ratio target '{target_name}' is not a field in the search space — "
                        f"check dtype_param spelling/case or whether the engine field whitelist "
                        f"dropped it; candidate rejected"
                    )
                elif full_params.get(target_name) is None:
                    errors.append(f"{name}: ratio target '{target_name}' has no resolved value")
            _validate_ratio_param(name, value, item, full_params, errors)
        elif dtype in DERIVED_ENGINE_ONLY:
            # Engine-computed derived fields (share/ternary_*/le_enum): the value
            # is derived from dependency fields at run time — a candidate-set
            # value is meaningless (overwritten) or can crash the engine (missing
            # dtype_param after dynamic field creation). Reject explicitly rather
            # than pass a value that diverges from the engine computation.
            errors.append(
                f"{name}={value!r}: derived field type '{dtype}' is computed by the engine from its "
                f"dependency fields (share/ternary_*/le_enum); do not set it explicitly"
            )
        elif dtype in {"int", "float", "bool", "range"}:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"{name}={value!r} is not numeric")
                continue
            if dtype == "int" and not numeric.is_integer():
                errors.append(f"{name}={value!r} must be an integer")
            elif dtype == "int":
                full_params[name] = int(numeric)
            if numeric < float(item.get("min", numeric)) or numeric > float(item.get("max", numeric)):
                errors.append(f"{name}={value!r} is outside range [{item.get('min')}, {item.get('max')}]")

        # 4. Check constant/derived conflict
        if _is_constant_or_derived(item):
            warnings.append(f"Candidate explicitly sets constant parameter: {name}")
            if not _values_equal(value, full_params.get(name)):
                errors.append(f"{name}={value!r} conflicts with derived/constant value {full_params.get(name)!r}")

    # 5. Evaluate constraint expressions
    constraints = context.get("constraints", {})
    eval_values = {key.upper(): value for key, value in full_params.items()}
    world_size = constraints.get("world_size")
    if world_size is not None:
        eval_values["WORLD_SIZE"] = world_size
        eval_values["NPU_COUNT"] = world_size
    for expr in constraints.get("expressions", []):
        try:
            resolved_expr = _replace_variables(expr, eval_values)
            if not eval_expression(resolved_expr):
                errors.append(f"Constraint failed: {expr.replace('$', '')}")
        except Exception as exc:
            errors.append(f"Constraint failed: {expr.replace('$', '')} ({exc})")

    # 6. Combo safety checks
    combo_errors, combo_warnings = check_combo_safety(full_params, context)
    errors.extend(combo_errors)
    warnings.extend(combo_warnings)

    # 7. Runtime CLI support check — only candidate-explicit params
    supported_flags = context.get("_supported_cli_flags")
    if supported_flags is not None and supported_flags:
        for name in params:
            value = params.get(name)
            if value is None or value == "":
                continue
            flag = _param_to_cli_flag(name)
            if flag not in supported_flags:
                errors.append(
                    f"{name}: --{flag} is not supported by the current {context.get('engine', 'vllm')} runtime"
                )

    return {
        "candidate_id": candidate.get("candidate_id") or candidate.get("id"),
        "valid": not errors,
        "params": full_params,
        "errors": errors,
        "warnings": warnings,
        "candidate": candidate,
    }

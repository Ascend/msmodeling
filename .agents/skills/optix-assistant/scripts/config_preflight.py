#!/usr/bin/env python3
"""Ownership: shared-precheck. Self-contained config.toml preflight for optix-assistant.

Validates a config.toml for the pieces required to run agent-mode optimization
(model path, served model name, benchmark dataset, optimizer_strategy=agent) and
back-fills missing pieces from values the user provides. Lives entirely inside
optix-assistant — deliberately self-contained, no cross-skill dependency.

TOML back-fill is implemented with ``tomlkit`` (round-trip preserving: comments
and untouched lines survive byte-for-byte; only the touched key is normalized).
Requires the repo uv environment (``uv sync``) so ``tomlkit`` is importable.

Run in Round-0 preflight and before every optix launch:
    # check only (exit 1 if incomplete)
    python skills/optix-assistant/scripts/config_preflight.py --check-only --config <path>
    # back-fill one missing piece at a time (each exits 0 if complete after fill)
    python skills/optix-assistant/scripts/config_preflight.py --config <path> --set-model /weights/qwen
    python skills/optix-assistant/scripts/config_preflight.py --config <path> --set-agent-mode --ttft-slo 3.0
    # write run-time budgets into config.toml [agent_optimizer] — the actual
    # termination source of truth for optix agent mode
    python skills/optix-assistant/scripts/config_preflight.py --config <path> \
        --set-max-rounds 12 --set-max-trials 24 --set-time-limit-minutes 720
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import tomlkit

from common import bad_json_tokens, table_or_create

#: placeholder values in the default config.toml that must be replaced by the user
_PLACEHOLDER_VALUES = {"", "model_path", "model_name", "dataset_name", "models", "port", "host"}

#: vllm bench 输入/输出长度 flag（random dataset 是 vllm bench 默认数据集）
_INPUT_OUTPUT_FLAGS = {"--input-len", "--random-input-len", "--output-len", "--random-output-len"}

#: 24 小时全局预算上限（分钟）—— 与 collect_context --time-limit-minutes 默认值一致
MAX_TIME_LIMIT_MINUTES = 1440

#: 显式指定的卡号环境变量（可写在 launch_env.sh 或 shell 环境）
_VISIBLE_DEVICES_ENV = "ASCEND_RT_VISIBLE_DEVICES"


def load_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


_IGNORED_DIRS = {
    ".git",
    ".claude",
    ".agent_optimizer",
    ".workbuddy",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "tests",
    "output",
}


def discover_configs(root: Path | None = None) -> List[Path]:
    """Find candidate config.toml files in the project (excluding ignored dirs).

    Note: discovery is an aid only — the validated file must be the exact one
    optix consumes (pass it explicitly via ``--config`` to both preflight and
    the optix launch; see SKILL.md 共享前置流程 step 1).
    """
    root = root or Path.cwd()
    found: List[Path] = []
    for p in sorted(root.rglob("config.toml")):
        if any(part in _IGNORED_DIRS for part in p.parts):
            continue
        found.append(p)
    return found


def _parse_io_from_others(others: str) -> Dict[str, str]:
    """Parse input/output length flags from vllm bench ``others``.

    e.g. ``--random-input-len 3500 --random-output-len 512`` ->
    ``{"--random-input-len": "3500", "--random-output-len": "512"}``.
    """
    tokens = shlex.split(others or "")
    io: Dict[str, str] = {}
    for i, token in enumerate(tokens):
        if token in _INPUT_OUTPUT_FLAGS and i + 1 < len(tokens):
            io[token] = tokens[i + 1]
    return io


def _looks_like_local_path(value: str) -> bool:
    """Heuristic: is this value a local filesystem path vs a remote model id?"""
    if not value:
        return False
    if value.startswith((".", "/", "~", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
        return True
    return "/" in value or "\\" in value


def _visible_devices() -> str:
    return os.environ.get(_VISIBLE_DEVICES_ENV, "").strip()


def _visible_device_count() -> int:
    raw = _visible_devices()
    if not raw:
        return 0
    return len([d for d in raw.replace(" ", "").split(",") if d])


def _tp_factor_choices(world_size: int) -> List[int]:
    """TP choices for a world size: all its divisors (capped at 64).

    Same algorithm as collect_context.default_parallel_search_space, so the
    guided config_writer command yields the identical search domain.
    """
    return [v for v in (1, 2, 4, 8, 16, 32, 64) if v <= world_size and world_size % v == 0] or [1]


def check_completeness(
    config: Dict[str, Any],
    engine: str = "vllm",
    expected_run_id: str | None = None,
    benchmark: str = "ais_bench",
    world_size: int | None = None,
) -> List[str]:
    """Return blocking errors for run-required config that is missing/placeholder.

    ``expected_run_id`` is the optix-assistant run-id; when given and the strategy
    is agent, the ``[agent_optimizer].run_id`` must match so optix resolves the
    same run dir (otherwise it looks for candidates in the wrong place).

    ``benchmark`` selects which command section must be complete
    (``ais_bench`` default per SKILL.md 0a; ``vllm_benchmark`` alternative).
    ``world_size`` (optional) is cross-checked against the number of explicitly
    visible devices (``ASCEND_RT_VISIBLE_DEVICES``) when both are known.
    """
    errors: List[str] = []
    cmd = (config.get(engine) or {}).get("command") or {}

    model = str(cmd.get("model") or "").strip()
    if not model or model in _PLACEHOLDER_VALUES:
        errors.append(f"[{engine}.command].model 未设置或为占位符 —— 需用户提供模型路径")

    served = str(cmd.get("served_model_name") or "").strip()
    if not served or served in _PLACEHOLDER_VALUES:
        errors.append(f"[{engine}.command].served_model_name 未设置或为占位符 —— 需用户提供服务模型名")

    # host/port 占位符检测：模板值是 "port"/"host" 等，若不检查会以 `--port port`
    # 启动服务导致必然失败。
    for key in ("host", "port"):
        value = str(cmd.get(key) or "").strip()
        if not value or value in _PLACEHOLDER_VALUES:
            errors.append(f"[{engine}.command].{key} 未设置或为占位符 —— 需用户提供服务 {key}")

    # --- benchmark 工具分支（不再写死 ais_bench） --------------------------------
    if benchmark == "ais_bench":
        ais = (config.get("ais_bench") or {}).get("command") or {}
        ais_models = str(ais.get("models") or "").strip()
        if not ais_models or ais_models in _PLACEHOLDER_VALUES:
            errors.append("ais_bench.command.models 未配置或为占位符 —— 需用户提供 benchmark 数据集路径")
    else:
        bench = (config.get("vllm_benchmark") or {}).get("command") or {}
        for key in ("host", "port", "model", "served_model_name"):
            value = str(bench.get(key) or "").strip()
            if not value or value in _PLACEHOLDER_VALUES:
                errors.append(f"vllm_benchmark.command.{key} 未设置或为占位符")
        io = _parse_io_from_others(str(bench.get("others") or ""))
        dataset = str(bench.get("dataset_name") or "").strip()
        dataset_path = str(bench.get("dataset_path") or "").strip()
        has_input = bool(io.get("--input-len") or io.get("--random-input-len"))
        # random 是 vllm bench 默认数据集：仅声明 dataset_name=random 不算"已确认输入输出"，
        # 仍需显式 --random-input-len/--input-len（或提供真实数据集）
        has_dataset = (dataset and dataset not in _PLACEHOLDER_VALUES and dataset != "random") or (
            dataset_path and dataset_path not in _PLACEHOLDER_VALUES
        )
        if not has_input and not has_dataset:
            errors.append(
                "vllm_benchmark 未配置输入输出：请在 [vllm_benchmark.command].others 中给出 "
                "--random-input-len/--input-len（vllm bench 默认 random dataset，长度需显式确认），"
                "或配置 dataset_name/dataset_path"
            )

    # --- 寻优目标（TPOT/TTFT 硬约束）必填 ---------------------------------------
    for slo_key in ("tpot_slo", "ttft_slo"):
        slo = config.get(slo_key)
        if slo is None or slo == "" or str(slo).strip() in _PLACEHOLDER_VALUES:
            errors.append(f"{slo_key} 未设置 —— 需用户确认寻优目标（TPOT/TTFT SLO，单位秒）")
        else:
            try:
                if float(slo) <= 0:
                    errors.append(f"{slo_key}={slo} 必须大于 0（单位秒）")
            except (TypeError, ValueError):
                errors.append(f"{slo_key}={slo!r} 不是有效数值（单位秒）")

    if config.get("optimizer_strategy", "pso") != "agent":
        errors.append('optimizer_strategy 未设为 "agent" —— agent 寻优不会启用，需确认切换到 agent 模式')
    else:
        # agent 模式必须与 optix-assistant 的 run 目录对齐；--set-agent-mode 会生成新 run_id，
        # 若已从 --check-only 捕获 run-id，务必用 --run-id 显式传入保持一致
        agent_cfg = config.get("agent_optimizer") or {}
        run_id = str(agent_cfg.get("run_id") or "").strip()
        run_dir = str(agent_cfg.get("run_dir") or "").strip()
        if not run_id or run_id == "default":
            errors.append(
                '[agent_optimizer].run_id 未设置或为模板占位 "default" —— 先运行 '
                'config_preflight --set-agent-mode（自动生成新 run-id，勿沿用 default）'
            )
        elif expected_run_id and run_id != expected_run_id:
            errors.append(
                f'[agent_optimizer].run_id 为 "{run_id}"，与 agent run-id "{expected_run_id}" 不一致 —— '
                "optix 会在错误的目录找候选"
            )
        elif run_dir and not run_dir.endswith(f".agent_optimizer/runs/{run_id}"):
            errors.append(
                f"[agent_optimizer].run_dir 指向 {run_dir!r}，与 agent run-dir 不一致 —— 建议留空让 optix 自动解析"
            )
        # 24h 全局时间预算校验
        time_limit = agent_cfg.get("time_limit_minutes")
        if time_limit is not None:
            try:
                if int(time_limit) <= 0:
                    errors.append(f"[agent_optimizer].time_limit_minutes={time_limit} 必须大于 0")
                elif int(time_limit) > MAX_TIME_LIMIT_MINUTES:
                    errors.append(
                        f"[agent_optimizer].time_limit_minutes={time_limit} 超过 24h 上限 "
                        f"({MAX_TIME_LIMIT_MINUTES} 分钟) —— 需确认"
                    )
            except (TypeError, ValueError):
                errors.append(f"[agent_optimizer].time_limit_minutes={time_limit!r} 不是有效整数")

    # --- agent 模式专用：use_request_rate_calibration 必须为 false ------------------
    # true 时 optix 会把候选的 CONCURRENCY 钉死为 max（optimizer.py fixed_target 逻辑），
    # agent 模式的并发维度静默失效，闭环测不出并发-SLO 拐点。--set-agent-mode 会自动写入
    # false；此处缺失/为 true 一律拦截（config.py 代码默认 True，缺失即运行时 true）。
    if config.get("optimizer_strategy", "pso") == "agent":
        if config.get("use_request_rate_calibration", True) is not False:
            errors.append(
                "use_request_rate_calibration 需为 false（agent 模式专用）：true 时 optix 会把候选 "
                "CONCURRENCY 钉死为 max，并发维度静默失效 —— 重跑 config_preflight --set-agent-mode "
                "自动修正，无需手改 TOML"
            )

    # --- 使用卡数与可见卡号一致性（可选交叉校验） --------------------------------
    visible_count = _visible_device_count()
    if world_size is not None and visible_count and world_size != visible_count:
        errors.append(
            f"使用卡数 world_size={world_size} 与 {_VISIBLE_DEVICES_ENV} 可见卡数 "
            f"{visible_count}（{_visible_devices()}）不一致 —— 需确认"
        )

    # --- 多卡并行拆分：vllm target_field 必须含 tp（baseline 唯一读源） -----------
    # baseline 用 config target_field 起服务（scheduler.update_data_field 只注入
    # config_position=="run" 的字段）。多卡且无 tp 时 baseline 以 tp=1 起服务必 OOM；
    # trial 路径不崩只是靠 collect_context 兜底合成——并行维度必须声明在 optix 认的
    # 字段源里。只拦多卡、只报错引导（config_writer 是写 target_field 的唯一入口）。
    if engine == "vllm" and world_size is not None and world_size > 1:
        tf = (config.get(engine) or {}).get("target_field") or []
        has_tp = any(str(t.get("name") or "").strip().lower() == "tp" for t in tf)
        if not has_tp:
            factors = _tp_factor_choices(world_size)
            errors.append(
                f"[{engine}.target_field] 缺少并行拆分参数 tp（world_size={world_size}）—— "
                f"baseline 将以 tp=1 起服务，多卡必 OOM。运行（保持单一写入入口，勿手改 TOML）：\n"
                f"  config_writer --config <config 路径> --target-field engine={engine} "
                f"name=tp config_position=run dtype=enum value={world_size} "
                f"dtype_param='{factors}'\n"
                f"（因数集={factors}；dp/pp 可留空，collect_context 会兜底补齐）"
            )

    # --- others 伪 JSON 兜底（最后一道门禁） ---------------------------------------
    # 结构化入口（config_writer --set …others.<flag>={json}）从根上不产生畸形；此处
    # 兜底拦截绕过配置期的手写/历史畸形，避免拖到 vllm serve 才崩（畸形不猜测：
    # 只 fail-fast，提示用结构化入口重写该容器）。
    for section_key in (engine, benchmark):
        others = str(((config.get(section_key) or {}).get("command") or {}).get("others") or "")
        bad = bad_json_tokens(others)
        if bad:
            errors.append(
                f"[{section_key}.command].others 含畸形 JSON 容器 token：{bad} —— "
                f"请用结构化入口重写：config_writer --set {section_key}.command.others.<flag>={{json}}"
            )

    return errors


def _run_ais_bench_search_gate(config: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """First-principles guard for ais_bench short names.

    A config.toml ``[ais_bench.command].models`` (and the ``--datasets`` token in
    ``others``) is valid only if the *installed* ais_bench resolves it. The
    matching rule (basename vs relative subpath) varies by ais_bench version and
    can drift from any doc, so ask the tool itself via ``--search`` instead of
    trusting prose. Fail-closed (returns an ERROR) whenever ais_bench is runnable
    but the short name does not resolve; degrade to a WARNING only when ais_bench
    is not on PATH (e.g. config authoring on a machine without it).

    Returns ``(errors, warnings)``. Skips silently when models is unset/placeholder
    (completeness already reports that).
    """
    errors: List[str] = []
    warnings: List[str] = []
    ais = (config.get("ais_bench") or {}).get("command") or {}
    models = str(ais.get("models") or "").strip()
    if not models or models in _PLACEHOLDER_VALUES:
        return errors, warnings
    tokens = shlex.split(str(ais.get("others") or ""))
    datasets = None
    for i, tok in enumerate(tokens):
        if tok == "--datasets" and i + 1 < len(tokens):
            datasets = tokens[i + 1]
            break
    exe = shutil.which("ais_bench")
    if not exe:
        warnings.append(
            "ais_bench 不在 PATH，跳过短名 --search 校验 —— 启动前必须在运行环境补做 "
            "`ais_bench --models <短名> [--datasets <短名>] --search` 预检"
        )
        return errors, warnings
    cmd = [exe, "--models", models]
    if datasets:
        cmd += ["--datasets", datasets]
    cmd.append("--search")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001 - tool unavailable/crash must not hard-fail authoring
        warnings.append(f"ais_bench --search 无法执行（{exc}），跳过短名校验")
        return errors, warnings
    if res.returncode != 0:
        # ais_bench 输出带 ANSI 色码，先剥离；kind 在完整文本上归因（避免被截断前缀影响）
        plain = re.sub(r"\x1b\[[0-9;]*m", "", res.stderr or res.stdout or "")
        kind = "UTILS-MATCH-001" if "UTILS-MATCH-001" in plain else f"ais_bench rc={res.returncode}"
        tail = plain.strip().splitlines()
        snippet = " | ".join(t.strip() for t in tail if t.strip())[:300]
        parts = [f"ais_bench 短名无法解析（{kind}）", f"models={models!r}"]
        if datasets:
            parts.append(f"datasets={datasets!r}")
        parts.append(
            "匹配规则随 ais_bench 版本而变：只接受能被 `ais_bench --search` 唯一解析的短名"
            "（当前实测按 config 文件 basename 匹配，不带目录前缀、不接受绝对路径）"
        )
        if snippet:
            parts.append(f"ais_bench 输出: {snippet}")
        errors.append(" —— ".join(parts))
    return errors, warnings


def _model_path_warning(config: Dict[str, Any], engine: str = "vllm") -> List[str]:
    """Non-blocking warnings for locally-looking model paths that do not exist."""
    warnings: List[str] = []
    model = str(((config.get(engine) or {}).get("command") or {}).get("model") or "").strip()
    if model and model not in _PLACEHOLDER_VALUES and _looks_like_local_path(model):
        path = Path(model)
        if not path.exists() and not (path / "config.json").exists():
            warnings.append(f"[{engine}.command].model 看起来是本地路径但不存在：{model}")
    return warnings


def _print_confirmation_table(
    config: Dict[str, Any],
    engine: str,
    benchmark: str,
    run_id: str,
    world_size: int | None,
) -> None:
    """Print the effective-config confirmation table for Round 0."""
    cmd = (config.get(engine) or {}).get("command") or {}
    bench = (config.get("vllm_benchmark") or {}).get("command") or {}
    io = _parse_io_from_others(str(bench.get("others") or ""))
    io_desc = ", ".join(f"{k} {v}" for k, v in io.items()) if io else ""
    dataset = str(bench.get("dataset_name") or "").strip()
    if not io_desc and dataset and dataset not in _PLACEHOLDER_VALUES:
        io_desc = f"dataset={dataset}"
    elif not io_desc:
        io_desc = "(vllm bench 默认 random dataset)"

    visible = _visible_devices()
    if not visible:
        launch_env = Path.cwd() / ".agent_optimizer" / "runs" / run_id / "launch_env.sh"
        if launch_env.exists():
            for line in launch_env.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith(f"export {_VISIBLE_DEVICES_ENV}="):
                    visible = line.split("=", 1)[1].strip().strip('"')
                    break
    cards_desc = f"{visible} [{_VISIBLE_DEVICES_ENV}]" if visible else "未指定（需向用户确认）"
    world_desc = f"{world_size}" if world_size is not None else "未提供（需向用户确认）"

    time_limit = (config.get("agent_optimizer") or {}).get("time_limit_minutes")
    time_desc = f"{time_limit} min" if time_limit is not None else "未设置（默认 1440 min）"

    calib = config.get("use_request_rate_calibration")
    if config.get("optimizer_strategy") == "agent":
        calib_desc = "false（agent 模式专用，候选 CONCURRENCY 逐 trial 生效）"
    elif calib is None:
        calib_desc = "未设置（默认 true）"
    else:
        calib_desc = str(calib)

    print("=== 生效配置确认表 ===")
    print(f"模型路径        : {cmd.get('model') or '-'}              [{engine}.command.model]")
    print(f"服务模型名      : {cmd.get('served_model_name') or '-'}  [{engine}.command.served_model_name]")
    print(f"benchmark 工具  : {benchmark}                            [--benchmark]")
    print(f"输入/输出       : {io_desc or '-'}                       [vllm_benchmark.command.others / dataset]")
    print(f"TPOT SLO        : {config.get('tpot_slo', '-')}s         [tpot_slo]")
    print(f"TTFT SLO        : {config.get('ttft_slo', '-')}s         [ttft_slo]")
    print(f"校准开关        : {calib_desc}                            [use_request_rate_calibration]")
    print(f"使用卡数        : {world_desc}                           [--world-size]")
    print(f"指定卡号        : {cards_desc}")
    print(f"时间预算        : {time_desc}                            [agent_optimizer.time_limit_minutes]")
    print(f"run-id          : {run_id}                               [config_preflight 自动生成]")
    print("===========================")


def _header_to_path(section_header: str) -> List[str]:
    """Convert ``[a.b.c]`` -> ``["a", "b", "c"]`` (supports nested sections)."""
    return [part.strip() for part in section_header.strip().strip("[]").split(".") if part.strip()]


def set_top_level_value(content: str, key: str, value: Any) -> str:
    """Set/replace a top-level ``key = value`` (tomlkit round-trip; comments and
    untouched lines are preserved byte-for-byte).
    """
    doc = tomlkit.parse(content)
    doc[key] = value
    return tomlkit.dumps(doc)


def _set_section_value(content: str, section_header: str, key: str, value: Any) -> str:
    """Set ``<key>`` inside a ``[section]`` (creates the section if missing)."""
    doc = tomlkit.parse(content)
    table = doc
    for part in _header_to_path(section_header):
        table = table_or_create(table, part)
    table[key] = value
    return tomlkit.dumps(doc)


def _set_command_value(content: str, engine: str, key: str, value: Any) -> str:
    """Set ``[<engine>.command].<key>`` (model / served_model_name)."""
    return _set_section_value(content, f"[{engine}.command]", key, value)


def effective_run_id(config: Dict[str, Any], explicit: str = "") -> str:
    """Resolve the run-id: explicit > existing config.toml > auto-generated.

    The run-id is an internal identifier the user never provides — it keeps the
    agent's run dir and ``[agent_optimizer].run_id`` aligned. Auto-generation is
    the fallback so a fresh default config.toml still gets a stable id.
    """
    if explicit.strip():
        return explicit.strip()
    existing = str(((config.get("agent_optimizer") or {}).get("run_id") or "")).strip()
    # "default" 是 config.toml 模板占位（无实义 run-id），视为未设置 → 自动生成新 run-id；
    # 已有真实 run-id（run-<时间戳>）则沿用（同一 config 续跑同一实验，candidates 连续累积）。
    if existing and existing != "default" and existing not in _PLACEHOLDER_VALUES:
        return existing
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}"


def set_agent_run_id(content: str, run_id: str) -> str:
    """Align ``[agent_optimizer].run_id`` with the agent's run-id and clear run_dir.

    optix resolves its run dir as ``<run_dir>`` if set, else ``cwd/.agent_optimizer/runs/<run_id>``.
    Clearing run_dir makes optix auto-resolve to the optix-assistant run dir.
    """
    doc = tomlkit.parse(content)
    agent = table_or_create(doc, "agent_optimizer")
    agent["run_id"] = run_id
    agent["run_dir"] = ""
    return tomlkit.dumps(doc)


def apply_backfill(content: str, args: argparse.Namespace, engine: str) -> str:
    """Apply the requested back-fill to content; returns updated content."""
    if args.set_model is not None:
        content = _set_command_value(content, engine, "model", args.set_model)
    if args.set_served_name is not None:
        content = _set_command_value(content, engine, "served_model_name", args.set_served_name)
    if args.set_bench is not None:
        content = _set_bench_dataset(content, args.set_bench, getattr(args, "benchmark", "ais_bench"))
    if args.ttft_slo is not None:
        content = set_top_level_value(content, "ttft_slo", args.ttft_slo)
    if args.tpot_slo is not None:
        content = set_top_level_value(content, "tpot_slo", args.tpot_slo)
    # 运行时预算写入 config.toml [agent_optimizer] —— optix 的真实终止依据；
    # 只写 context.json 的预算不会被 optix 消费
    for key in ("max_rounds", "max_trials", "time_limit_minutes"):
        value = getattr(args, f"set_{key}", None)
        if value is not None:
            content = _set_section_value(content, "[agent_optimizer]", key, value)
    if args.set_agent_mode:
        content = set_top_level_value(content, "optimizer_strategy", "agent")
        # agent 模式专用：候选 CONCURRENCY 必须逐 trial 生效，REQUESTRATE 固定 max。
        # false 时 optix 走 scheduler.run，候选 CONCURRENCY 不被钉死为 max（optimizer.py
        # fixed_target 逻辑），闭环才能测出并发-SLO 拐点。PSO 全局默认不受影响。
        content = set_top_level_value(content, "use_request_rate_calibration", False)
    return content


def _set_bench_dataset(content: str, value: str, benchmark: str) -> str:
    """Back-fill the dataset for the selected benchmark tool.

    vllm_benchmark writes ``[vllm_benchmark.command].dataset_path`` (pinning
    ``dataset_name = "custom"`` so a real path passes the completeness check);
    ais_bench writes ``[ais_bench.command].models``.
    """
    if benchmark == "vllm_benchmark":
        content = _set_section_value(content, "[vllm_benchmark.command]", "dataset_name", "custom")
        return _set_section_value(content, "[vllm_benchmark.command]", "dataset_path", value)
    return _set_command_value(content, "ais_bench", "models", value)


def _write_with_validate(path: Path, content: str) -> int:
    try:
        load_toml_from_str(content)
    except ValueError as exc:
        print(f"ERROR: 更新后 config.toml 无法解析: {exc}")
        return 1
    backup = path.with_suffix(".toml.bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    return 0


def load_toml_from_str(content: str) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    return tomllib.loads(content)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="optix-assistant config.toml preflight (self-contained).")
    parser.add_argument("--config", default="optix/config.toml", help="Path to config.toml")
    parser.add_argument("--discover", action="store_true", help="列出项目中的 config.toml 候选（供 agent 向用户确认）")
    parser.add_argument("--engine", default="vllm", choices=["vllm", "mindie"])
    parser.add_argument(
        "--benchmark",
        default="ais_bench",
        choices=["vllm_benchmark", "ais_bench"],
        help="benchmark 工具（默认 ais_bench，昇腾生态最常用；流程上仍须询问用户确认，vllm_benchmark 需显式输入输出）",
    )
    parser.add_argument("--world-size", type=int, default=None, help="使用卡数（可选，与可见卡号交叉校验）")
    parser.add_argument("--check-only", action="store_true", help="Only validate completeness; exit 1 if incomplete.")
    parser.add_argument("--set-model", help="Back-fill model path")
    parser.add_argument("--set-served-name", help="Back-fill served model name")
    parser.add_argument(
        "--set-bench",
        help=(
            "Back-fill benchmark dataset（按 --benchmark 写入：vllm_benchmark → "
            "[vllm_benchmark.command].dataset_path；ais_bench → [ais_bench.command].models）"
        ),
    )
    parser.add_argument(
        "--set-max-rounds", type=int, help="Back-fill [agent_optimizer].max_rounds（轮数上限，optix 运行终止依据）"
    )
    parser.add_argument("--set-max-trials", type=int, help="Back-fill [agent_optimizer].max_trials（trial 总数上限）")
    parser.add_argument(
        "--set-time-limit-minutes",
        type=int,
        help="Back-fill [agent_optimizer].time_limit_minutes（全局时间预算，分钟，≤1440）",
    )
    parser.add_argument("--ttft-slo", type=float, help="Back-fill ttft_slo")
    parser.add_argument("--tpot-slo", type=float, help="Back-fill tpot_slo")
    parser.add_argument("--set-agent-mode", action="store_true", help="Set optimizer_strategy=agent")
    parser.add_argument(
        "--run-id",
        default="",
        help=(
            "optix-assistant run-id（与 collect_context --run-id 一致）。仅用于校验 "
            "[agent_optimizer].run_id 是否与它对齐，不写入 config.toml；写入需通过 "
            "--set-agent-mode（自动对齐 run_id + 清空 run_dir）。"
        ),
    )
    args = parser.parse_args(argv)

    if args.discover:
        candidates = discover_configs()
        for i, p in enumerate(candidates, start=1):
            print(f"{i}. {p}")
        if not candidates:
            print("未找到任何 config.toml —— 需向用户确认路径")
            return 1
        return 0

    path = Path(args.config)
    if not path.exists():
        print(f"ERROR: config 不存在: {path}")
        return 2
    try:
        config = load_toml(path)
    except ValueError as exc:
        print(f"ERROR: {path} 解析失败: {exc}")
        return 1

    has_backfill = any(
        [
            args.set_model is not None,
            args.set_served_name is not None,
            args.set_bench is not None,
            args.ttft_slo is not None,
            args.tpot_slo is not None,
            getattr(args, "set_max_rounds", None) is not None,
            getattr(args, "set_max_trials", None) is not None,
            getattr(args, "set_time_limit_minutes", None) is not None,
            args.set_agent_mode,
        ]
    )
    # run-id 自动解析（显式 > 已有 > 自动生成），只在非校验模式下写入
    run_id = effective_run_id(config, args.run_id)
    if has_backfill and not args.check_only:
        content = path.read_text(encoding="utf-8")
        content = set_agent_run_id(content, run_id)
        content = apply_backfill(content, args, args.engine)
        rc = _write_with_validate(path, content)
        if rc:
            return rc
        print(f"✓ 已回填 {path}")
        config = load_toml(path)
    if args.check_only or has_backfill:
        # agent 捕获此 run-id，用于 collect_context --run-id 保持一致
        print(f"run-id: {run_id}")

    expected_run_id = run_id if args.check_only else None
    errors = check_completeness(
        config,
        args.engine,
        expected_run_id=expected_run_id,
        benchmark=args.benchmark,
        world_size=args.world_size,
    )
    gate_errors: List[str] = []
    gate_warnings: List[str] = []
    if args.benchmark == "ais_bench":
        # 短名合法性由安装的 ais_bench 判定（--search fail-closed），不依赖文档经验
        gate_errors, gate_warnings = _run_ais_bench_search_gate(config)
    errors = [*errors, *gate_errors]
    for e in errors:
        print(f"ERROR: {e}")
    if errors:
        print(f"FAIL: {len(errors)} 项缺失，需补齐后才能运行寻优")
        return 1
    if args.check_only or has_backfill:
        warnings = [*_model_path_warning(config, args.engine), *gate_warnings]
        for w in warnings:
            print(f"WARNING: {w}")
        _print_confirmation_table(config, args.engine, args.benchmark, run_id, args.world_size)
        print(f"OK: {path} 配置完整")
    return 0


if __name__ == "__main__":
    sys.exit(main())

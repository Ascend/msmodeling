#!/usr/bin/env python3
"""validate_candidates.py — candidates.round-N.json 分析与引用门禁（B 步骤，执行前强制）

检查每个 direction 的 evidence_analysis / refs_analysis / candidates_analysis
均为非空字符串，不限制正文格式、无需独立分析文件；同时校验顶层 routing_review
（瓶颈假设及 L1 机制层匹配记录）、空间定义、TOML 一致性和候选装配。
保留 evidence_refs / hypothesis / rationale / risk 的路径检查，并检查三个分析
字段中"长得像路径"的引用能否解析到真实存在的文件。设计要点：

1. 前缀分类：skill 相对路径（knowledge/ workflows/ references/ schemas/
   scripts/ presets/）按 skill 根解析；run 产物
   （results.round-*.json / summary.md / trial_logs/ ...）按 run 目录解析；
   不带路径前缀的自由文本（vendor preset 描述、数据摘要等）放行不校验。
2. 目录引用算违规：如 "knowledge/modeling-knowledge" 存在但是目录——
   目录不可溯源，必须指到具体卡/文件。这是拦截伪引用的关键规则。
3. 锚点与中英文边界：路径 token 在空格 / 全角冒号 / § / # / 中文标点处终止，
   "fc1.md §2：互斥" 提取为 "fc1.md"。

退出码：0 通过；1 存在违规（禁止进入本轮执行）；2 用法/文件错误。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from math import isfinite

SKILL_ROOT = Path(__file__).resolve().parent.parent

# Retain the removed prefix so stale references are rejected as missing files.
SKILL_PATH_RE = re.compile(
    r"(?:modeling-skill|workflows|references|schemas|scripts|presets|knowledge)"
    r"/[A-Za-z0-9_\-./]+"
)
RUN_PATH_RE = re.compile(
    r"(?:results\.round-[\w.-]+\.json|candidates\.round-\d+\.json|progress\.round-\d+\.json"
    r"|summary\.md|baseline\.json|context(?:\.round-\d+)?\.json|best_result\.json|serve_command\.(?:json|sh)"
    r"|optimizer_config_handoff\.json|checklist_report\.md"
    r"|launch_env\.sh|unset_env\.sh|trial_logs/[\w.-]+)"
)

ANALYSIS_FIELDS = ("evidence_analysis", "refs_analysis", "candidates_analysis")

# L1 机制层名 → 方法卡目录（knowledge/modeling-knowledge/methods/ 下）。
# 分层标题以 method-index.md 的 "## <层名>" 为准，_check_layer_mapping 负责核对。
LAYER_DIRS = {
    "调度层": "scheduling",
    "并行层": "parallelism",
    "负载均衡层": "load-balance",
    "缓存层": "caching",
    "通信层": "communication",
    "执行层": "execution",
    "投机层": "speculative",
    "通用层": "generic",
}
METHODS_ROOT = SKILL_ROOT / "knowledge" / "modeling-knowledge" / "methods"
METHOD_INDEX = SKILL_ROOT / "knowledge" / "modeling-knowledge" / "method-index.md"
METHODS_PATH_RE = re.compile(r"(?:knowledge/modeling-knowledge/)?methods/([A-Za-z0-9_\-]+)/")
_CARD_TOKEN_RE: dict[str, re.Pattern] = {}

SCAN_FIELDS = (
    ("directions", "evidence_refs"),
    ("directions", "hypothesis"),
    *(("directions", field) for field in ANALYSIS_FIELDS),
    ("candidates", "rationale"),
    ("candidates", "risk"),
)


def check_ref(text: str, run_dir: Path) -> list[str]:
    errs: list[str] = []
    skill_root = SKILL_ROOT.resolve()
    for m in SKILL_PATH_RE.finditer(text):
        rel = m.group(0).rstrip("/.")
        target = (SKILL_ROOT / rel).resolve()
        if not target.is_relative_to(skill_root):
            errs.append(f"skill 路径越出 skill 根目录: {rel}（路径穿越或指向外部的符号链接，一律拒绝）")
        elif not target.exists():
            errs.append(f"skill 路径不存在: {rel}（SKILL_ROOT={SKILL_ROOT.name}）")
        elif target.is_dir():
            errs.append(f"引用的是目录而非具体文件: {rel}（目录不可溯源，须指到具体卡/文件）")
    for m in RUN_PATH_RE.finditer(text):
        rel = m.group(0).rstrip("/.")
        if not (run_dir / rel).exists():
            errs.append(f"run 产物不存在: {rel}")
    return errs


def _check_layer_mapping() -> list[str]:
    """核对层名映射与 method-index.md 分层标题、methods/ 目录的一致性。"""
    errs: list[str] = []
    try:
        headings = set(re.findall(r"^## (\S+)$", METHOD_INDEX.read_text(encoding="utf-8"), re.MULTILINE))
    except OSError:
        return [f"无法读取 {METHOD_INDEX}，无法核对机制层映射"]
    for heading in headings:
        if heading.endswith("层") and heading not in LAYER_DIRS:
            errs.append(f"method-index.md 分层标题“{heading}”未登记在 LAYER_DIRS，请先补全映射再校验")
    for layer, directory in LAYER_DIRS.items():
        if not (METHODS_ROOT / directory).is_dir():
            errs.append(f"LAYER_DIRS 映射错误：{layer} → methods/{directory}/ 不存在")
    return errs


def _extract_method_dirs(texts: list[str]) -> set[str]:
    """提取分析文本中引用的方法卡所属目录（methods/<dir>/）。"""
    dirs: set[str] = set()
    for text in texts:
        for m in METHODS_PATH_RE.finditer(text):
            dirs.add(m.group(1))
    return dirs


def _layer_card_index() -> dict[str, set[str]]:
    """按层目录扫描方法卡文件名（不含扩展名），形成 层目录 → 卡名词干集合。"""
    index: dict[str, set[str]] = {}
    if METHODS_ROOT.is_dir():
        for directory in LAYER_DIRS.values():
            d = METHODS_ROOT / directory
            if d.is_dir():
                index[directory] = {p.stem for p in d.glob("*.md")}
    return index


def _card_token_pattern(stem: str) -> re.Pattern:
    pattern = _CARD_TOKEN_RE.get(stem)
    if pattern is None:
        pattern = re.compile(rf"(?<![A-Za-z0-9_\-]){re.escape(stem)}(?![A-Za-z0-9_\-])")
        _CARD_TOKEN_RE[stem] = pattern
    return pattern


def _layer_cited(layer: str, texts: list[str], path_dirs: set[str], card_index: dict[str, set[str]]) -> bool:
    """判断某层的方法卡是否被引用：目录路径前缀或层内任一卡名（独立词）命中均可。"""
    directory = LAYER_DIRS[layer]
    if directory in path_dirs:
        return True
    joined = "\n".join(texts)
    return any(_card_token_pattern(stem).search(joined) for stem in card_index.get(directory, ()))


def validate_routing_review(data: dict, refs_texts: list[str]) -> tuple[list[str], list[str]]:
    """校验顶层 routing_review，返回 (violations, warnings)。"""
    violations: list[str] = []
    warnings: list[str] = []
    rr = data.get("routing_review")
    if not isinstance(rr, dict):
        violations.append(
            "缺少 routing_review 对象：每轮候选必须记录瓶颈假设及其命中的机制层（结构见 schemas/candidates.schema.json）。"
        )
        return violations, warnings
    hypothesis = rr.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        violations.append("routing_review.hypothesis 必填，且必须是非空字符串。")
    layers = rr.get("l1_layers_matched")
    if not isinstance(layers, list) or not layers or not all(isinstance(x, str) and x.strip() for x in layers):
        violations.append("routing_review.l1_layers_matched 必填，且至少包含一个机制层名。")
        layers = []
    deferred = rr.get("layers_deferred")
    if not isinstance(deferred, dict):
        violations.append("routing_review.layers_deferred 必须是对象（无暂缓层时写空对象）。")
        deferred = {}
    for layer in layers:
        if layer not in LAYER_DIRS:
            violations.append(
                f"routing_review.l1_layers_matched 含未知机制层“{layer}”：层名须与 method-index.md 分层标题一致。"
            )
    for layer, reason in deferred.items():
        if layer not in layers:
            violations.append(f"layers_deferred 含未命中层“{layer}”：暂缓层必须来自 l1_layers_matched。")
        if not isinstance(reason, str) or not reason.strip():
            violations.append(f"layers_deferred.{layer} 理由必填，且必须是非空字符串。")
    cited_dirs = _extract_method_dirs(refs_texts)
    card_index = _layer_card_index()
    for layer in layers:
        if layer in LAYER_DIRS and layer not in deferred:
            if not _layer_cited(layer, refs_texts, cited_dirs, card_index):
                violations.append(
                    f"命中层“{layer}”本轮行动但未给出依据：refs_analysis 须引用 methods/{LAYER_DIRS[layer]}/ 下的方法卡"
                    f"（写路径或卡名均可），或在 layers_deferred 中说明暂缓理由。"
                )
    if isinstance(hypothesis, str):
        for layer in LAYER_DIRS:
            stem = layer[:-1]
            if stem and stem in hypothesis and layer not in layers:
                warnings.append(f"假设文本包含“{stem}”但 l1_layers_matched 未列出“{layer}”，请核对路由匹配是否完整。")
    return violations, warnings


def find_candidates_file(round_no: int, run_dir: Path | None) -> tuple[Path, Path]:
    if run_dir:
        cf = run_dir / f"candidates.round-{round_no}.json"
        if not cf.exists():
            raise FileNotFoundError(cf)
        return cf, run_dir
    runs_root = Path.cwd() / ".agent_optimizer" / "runs"
    matches = sorted(runs_root.glob(f"*/candidates.round-{round_no}.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"未找到 candidates.round-{round_no}.json（搜索 {runs_root}；可用 --run-dir 显式指定）")
    latest = matches[-1]
    return latest, latest.parent


def _space_items(space: dict) -> list[dict]:
    if not isinstance(space, dict):
        raise ValueError("search_space must be an object")
    items = []
    names = set()
    for group in ("parameters", "constants"):
        entries = space.get(group)
        if not isinstance(entries, list):
            raise ValueError(f"search_space.{group} must be an array")
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError(f"search_space.{group} entries must be objects")
            name = item.get("name")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError(f"Missing or duplicate search_space parameter: {name!r}")
            names.add(name)
            for key in ("dtype", "config_position"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"{name}: {key} is required")
            if item.get("default") is None:
                raise ValueError(f"{name}: default is required")
            for key in ("min", "max"):
                value = item.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                    raise ValueError(f"{name}: finite numeric {key} is required")
            if item["min"] > item["max"]:
                raise ValueError(f"{name}: min exceeds max")
            if item.get("constant") is not None and item["default"] != item["constant"]:
                raise ValueError(f"{name}: default conflicts with constant")
            if item["dtype"] == "enum":
                choices = item.get("choices")
                if not isinstance(choices, list) or not choices or item["default"] not in choices:
                    raise ValueError(f"{name}: default must be in nonempty choices")
                if len(choices) > 1 and item["min"] == item["max"]:
                    raise ValueError(f"{name}: enum requires distinct min/max for runtime coordinates")
            if group == "constants" and item.get("constant") is None and item["min"] != item["max"]:
                raise ValueError(
                    f"{name}: constants 分组只放真正的常量（设 constant 字段或 min==max）；"
                    "仅想给默认取值的可调参数请放 parameters 分组"
                )
            items.append(item)
    return items


_flags_cache: dict[str, set[str]] = {}


def _flags_cache_path(engine: str, binary: str) -> Path:
    import hashlib
    import tempfile

    try:
        mtime = Path(binary).stat().st_mtime
    except OSError:
        mtime = 0
    digest = hashlib.sha256(f"{engine}|{binary}|{mtime}".encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"optix-assistant-flags-{digest}.json"


def _probe_supported_flags(engine: str) -> set[str]:
    """探测引擎支持的 CLI flag 集合。

    optix 自带的 get_supported_cli_flags 超时只有 15s，vllm 冷启动 --help=all
    实测约 26s，必然超时返回空集导致检查静默跳过，故在此自行探测。
    结果按引擎二进制路径+mtime 缓存到临时目录（不写 run 目录，校验不新增产物）；
    探测不到二进制或超时时返回空集（跳过检查）。
    """
    cached = _flags_cache.get(engine)
    if cached is not None:
        return cached
    import shutil
    import subprocess

    binary = shutil.which(engine)
    if binary is None:
        return set()
    cache = _flags_cache_path(engine, binary)
    if cache.is_file():
        try:
            flags = set(json.loads(cache.read_text(encoding="utf-8")))
            _flags_cache[engine] = flags
            return flags
        except (OSError, json.JSONDecodeError):
            pass
    flags: set[str] = set()
    try:
        result = subprocess.run([binary, "serve", "--help=all"], capture_output=True, text=True, timeout=120)
        flags.update(re.findall(r"--([a-z][a-z0-9-]*)", result.stdout + result.stderr))
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    if not flags:
        # serve 探测失败时宁可跳过也不用残缺集合误拦（bench 子集不含 serve flag）
        return set()
    try:
        result = subprocess.run([binary, "bench", "serve", "--help"], capture_output=True, text=True, timeout=120)
        flags.update(re.findall(r"--([a-z][a-z0-9-]*)", result.stdout + result.stderr))
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    _flags_cache[engine] = flags
    try:
        cache.write_text(json.dumps(sorted(flags)), encoding="utf-8")
    except OSError:
        pass
    return flags


def check_run_entry_flags(fields: list[dict], engine: str) -> list[str]:
    """config_position=run 的空间条目（变量与常量都可能进最终命令）映射出的
    --flag 必须在引擎运行时支持的 CLI 列表里；探测不到引擎二进制时跳过。
    """
    sys.path.insert(0, str(SKILL_ROOT.parents[2]))
    from optix.config.custom_command import param_to_cli_flag

    supported = _probe_supported_flags(engine)
    if not supported:
        print(f"[warn] 未能探测 {engine} 支持的 CLI flag 列表，跳过 flag 存在性校验", file=sys.stderr)
        return []
    errs = []
    for item in fields:
        if item.get("config_position") != "run":
            continue
        flag = param_to_cli_flag(item["name"])
        if flag not in supported:
            errs.append(f"{item['name']}: --{flag} 不在当前 {engine} 运行时支持的 CLI 列表里")
    return errs


def validate_search_space(space: dict, configured_fields: list[dict]) -> list[dict]:
    """Validate the active space against the fields the engine will execute."""
    fields = _space_items(space)
    from optix.optimizer.agentic.topology import TOPOLOGY_FACT_NAMES

    if any(item["name"].upper() in TOPOLOGY_FACT_NAMES for item in fields):
        raise ValueError("Topology facts belong in context.hardware, not search_space or service parameters")
    configured = {f["name"]: f for f in configured_fields}
    active = {f["name"]: f for f in fields}
    missing = configured.keys() - active.keys()
    if missing:
        raise ValueError(f"search_space must retain TOML fields: {sorted(missing)}")
    for name, item in active.items():
        base = configured.get(name)
        if base is None:
            if item["dtype"] not in {"int", "float", "bool", "enum"}:
                raise ValueError(f"{name}: additional fields support int/float/bool/enum; use enum choices for strings")
            continue
        for key in ("dtype", "config_position"):
            if item[key] != base[key]:
                raise ValueError(
                    f"{name}: {key} conflicts with config.toml; TOML is frozen after Round 0 assembly"
                    " — align the search_space entry with TOML, or terminate this run and reassemble"
                )
        if item["dtype"] != "enum" and item.get("dtype_param") != base.get("dtype_param"):
            raise ValueError(f"{name}: dtype_param conflicts with config.toml")
        if base.get("constant") is not None and item["default"] != base["constant"]:
            raise ValueError(f"{name}: default conflicts with TOML constant")
        if base.get("constant") is not None:
            constant = base["constant"]
            if item.get("constant") != constant:
                raise ValueError(f"{name}: must retain TOML constant")
            if item["dtype"] == "enum":
                if item.get("choices") != [constant]:
                    raise ValueError(f"{name}: choices must pin TOML constant")
            elif item["min"] != constant or item["max"] != constant:
                raise ValueError(f"{name}: min/max must retain TOML constant")
    return fields


def validate_trial_types(data: dict) -> None:
    """Validate new candidate classifications locally, without engine support."""
    directions = data.get("directions")
    if not isinstance(directions, list) or not directions:
        raise ValueError("directions must be a non-empty array")
    by_id = {}
    for direction in directions:
        if not isinstance(direction, dict):
            raise ValueError("Each direction must be an object")
        did = direction.get("direction_id")
        if not isinstance(did, str) or not did.strip() or did in by_id:
            raise ValueError(f"Missing or duplicate direction_id: {did!r}")
        kind = direction.get("trial_type")
        if kind not in ("optimization", "diagnosis"):
            raise ValueError(f"Direction {did}: trial_type must be optimization or diagnosis")
        if direction.get("diagnostic_only") or "diagnostic_only" in (direction.get("constraints") or []):
            raise ValueError(f"Direction {did}: replace diagnostic_only with trial_type: diagnosis")
        by_id[did] = kind
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be an array")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be an object")
        did = candidate.get("direction_id")
        if not isinstance(did, str) or did not in by_id:
            raise ValueError(f"Each candidate must reference an existing direction_id: {did!r}")
        if "trial_type" in candidate and candidate["trial_type"] != by_id[did]:
            raise ValueError(f"Candidate trial_type conflicts with direction {did}")
        if candidate.get("diagnostic_only") or "diagnostic_only" in (candidate.get("constraints") or []):
            raise ValueError("Replace candidate diagnostic_only with direction trial_type: diagnosis")


def validate_round(data: dict, run_dir: Path, round_no: int) -> None:
    """Check the exact space consumed by the next optix process."""
    validate_trial_types(data)

    from check_search_space import check_parameters, load_configured_parameters

    sys.path.insert(0, str(SKILL_ROOT.parents[2]))
    from optix.config.config import OptimizerConfigField, field_to_param, range_to_enum, map_param_with_value
    from optix.optimizer.agentic.strategies import AgentCandidateStrategy
    from optix.optimizer.agentic.validation import validate_candidate

    context = json.loads((run_dir / "context.json").read_text(encoding="utf-8"))
    if "search_space" not in context:
        raise ValueError("Missing context.search_space; prepare this round's definitions before validating candidates")
    source = context.get("source") or {}
    toml_path = source.get("config_toml")
    expected_sha = source.get("config_toml_sha256")
    if toml_path and expected_sha:
        try:
            actual_sha = hashlib.sha256(Path(toml_path).read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"config.toml unreadable at {toml_path}: {exc}") from exc
        if actual_sha != expected_sha:
            raise ValueError(
                "config.toml modified after Round 0 assembly (sha256 mismatch) — terminate this run and reassemble"
            )
    section, configured_fields = load_configured_parameters(context)
    fields = validate_search_space(context["search_space"], configured_fields)
    # Recheck frozen TOML fields at the execution boundary.  This is
    # intentionally independent of candidate validation: a round must not be
    # able to widen or remove a constant between space validation and engine
    # assembly.
    active_by_name = {item["name"]: item for item in fields}
    for base in configured_fields:
        if base.get("constant") is None:
            continue
        item = active_by_name.get(base["name"])
        constant = base["constant"]
        if item is None or item.get("constant") != constant:
            raise ValueError(f"{base['name']}: execution search_space must retain TOML constant")
        if item.get("dtype") == "enum":
            if item.get("choices") != [constant] or item.get("default") != constant:
                raise ValueError(f"{base['name']}: execution search_space must pin choices/default to TOML constant")
        elif item.get("default") != constant or item.get("min") != constant or item.get("max") != constant:
            raise ValueError(f"{base['name']}: execution search_space must pin min/max/default to TOML constant")
    flag_errors = check_run_entry_flags(fields, str(context.get("engine") or "vllm"))
    if flag_errors:
        raise ValueError("; ".join(flag_errors))
    errors = [
        item["msg"] for item in check_parameters(context, fields=fields, section=section) if item["level"] == "error"
    ]
    if errors:
        raise ValueError("; ".join(errors))
    if round_no < 1 or data.get("round") != round_no:
        raise ValueError("Candidate round must match --round and filename")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("Agent workflow requires exactly one candidate per round")
    for number in range(1, round_no):
        results_path = run_dir / f"results.round-{number}.json"
        results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
        if not any(
            t.get("error") is None and t.get("fitness", float("inf")) < float("inf") for t in results.get("trials", [])
        ):
            raise ValueError(f"Round {number} is unfinished; complete it before round {round_no}")
        archive_path = run_dir / f"context.round-{number}.json"
        if not archive_path.is_file():
            raise ValueError(f"Round {number} is missing {archive_path.name}; archive its context before advancing")
        archived = json.loads(archive_path.read_text(encoding="utf-8"))
        if not isinstance(archived, dict):
            raise ValueError(f"Invalid archived context: {archive_path.name}")
        _space_items(archived.get("search_space"))
    # A single process reads context once and may resume failed earlier rounds.
    for path in sorted(run_dir.glob("candidates.round-*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1])):
        number = int(path.stem.rsplit("-", 1)[1])
        results_path = run_dir / f"results.round-{number}.json"
        results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
        succeeded = any(
            t.get("error") is None and t.get("fitness", float("inf")) < float("inf") for t in results.get("trials", [])
        )
        if number == round_no and succeeded:
            raise ValueError(f"Round {number} already succeeded; preserve its candidate and archived context")
        if number < round_no and not succeeded:
            raise ValueError(f"Round {number} is unfinished; repair and retry it before round {round_no}")
        if number > round_no and not succeeded:
            raise ValueError(f"Future candidate round {number} exists; generate it only after this process exits")

    target_fields = [OptimizerConfigField(**f) for f in section["target_field"]]
    range_to_enum(target_fields)
    strategy = AgentCandidateStrategy(run_dir=run_dir, round_index=round_no)
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("params"), dict):
            raise ValueError("Each candidate must contain a params object")
        validation = validate_candidate(context, candidate)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        for base in configured_fields:
            if base.get("constant") is not None and validation["params"].get(base["name"]) != base["constant"]:
                raise ValueError(f"{base['name']}: resolved candidate conflicts with TOML constant")
        errors = [
            item["msg"]
            for item in check_parameters(context, fields=fields, section=section, params=validation["params"])
            if item["level"] == "error"
        ]
        if errors:
            raise ValueError("; ".join(errors))
        resolved = strategy._params_to_target_field(validation["params"], target_fields, context)
        position = field_to_param(tuple(resolved))
        if (context.get("deployment") or {}).get("type") == "multihost":
            from types import SimpleNamespace
            from common import load_cluster_topology
            from optix.optimizer.multihost_adapter import prepare_multihost_fields

            topology = load_cluster_topology(context["source"]["cluster_config"])
            if not topology["ok"]:
                raise ValueError(topology["error"])
            plugin_config = SimpleNamespace(
                env=topology["env"], workers=topology["workers"], chips_per_node=topology["chips_per_node"]
            )
            simulator = SimpleNamespace(
                remote_executor=SimpleNamespace(config=plugin_config),
                config=SimpleNamespace(
                    command=SimpleNamespace(others=(section.get("command") or {}).get("others", ""))
                ),
            )
            active_fields = map_param_with_value(position, tuple(resolved))
            prepare_multihost_fields(simulator, active_fields)


def main() -> int:
    ap = argparse.ArgumentParser(description="candidates 分析字段与引用可解析性门禁")
    ap.add_argument("--round", type=int, required=True, help="轮次 N（candidates.round-N.json）")
    ap.add_argument(
        "--run-dir", type=Path, default=None, help="run 目录；缺省取 .agent_optimizer/runs 下含该轮候选的最新 run"
    )
    args = ap.parse_args()

    try:
        cf, run_dir = find_candidates_file(args.round, args.run_dir)
        data = json.loads(cf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[usage] {exc}", file=sys.stderr)
        return 2

    violations: list[tuple[str, str, list[str]]] = []
    if not isinstance(data, dict):
        print(f"✗ validate_candidates: 顶层必须是 JSON 对象（{cf}）", file=sys.stderr)
        return 1

    directions = data.get("directions")
    if not isinstance(directions, list) or not directions:
        violations.append(("directions", repr(directions), ["必须提供非空 directions 数组及各方向的分析字段"]))
    for section in ("directions", "candidates"):
        items = data.get(section, [])
        if not isinstance(items, list):
            if section != "directions":
                violations.append((section, repr(items), ["必须是数组"]))
            continue
        for index, item in enumerate(items):
            where = f"{section}[{index}]"
            if not isinstance(item, dict):
                violations.append((where, repr(item), ["必须是对象"]))
                continue
            if section == "directions":
                for field in ANALYSIS_FIELDS:
                    value = item.get(field)
                    if not isinstance(value, str) or not value.strip():
                        violations.append((f"{where}.{field}", repr(value), ["必填，且必须是非空字符串"]))

    for section, field in SCAN_FIELDS:
        items = data.get(section, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            ident = item.get("candidate_id") or item.get("direction_id") or "?"
            value = item.get(field)
            texts = value if isinstance(value, list) else [value]
            for text in texts:
                if isinstance(text, str) and text.strip():
                    errs = check_ref(text, run_dir)
                    if errs:
                        violations.append((f"{section}.{ident} {field}", text, errs))

    mapping_errors = _check_layer_mapping()
    for err in mapping_errors:
        violations.append(("routing_review 映射自检", METHOD_INDEX.name, [err]))
    refs_texts = [
        item.get(field)
        for section in ("directions", "candidates")
        for item in (data.get(section) or [])
        if isinstance(item, dict)
        for field in ANALYSIS_FIELDS
        if isinstance(item.get(field), str)
    ]
    for item in data.get("directions") or []:
        if isinstance(item, dict):
            refs_texts.extend(str(x) for x in item.get("evidence_refs") or [] if isinstance(x, str))
    rr_violations, rr_warnings = validate_routing_review(data, refs_texts)
    if rr_violations:
        violations.append(("routing_review", data.get("routing_review"), rr_violations))

    if violations:
        print(f"✗ validate_candidates: {len(violations)} 处分析字段或引用违规（{cf}）\n", file=sys.stderr)
        for where, text, errs in violations:
            print(f"[{where}]\n  内容: {str(text)[:400]}", file=sys.stderr)
            for e in errs:
                print(f"  -> {e}", file=sys.stderr)
        print(
            "\n修正要求：每个 direction 须填写 evidence_analysis / refs_analysis / candidates_analysis 非空字符串；"
            "顶层 routing_review 须记录瓶颈假设、命中的机制层及暂缓理由；"
            "命中的机制层在 refs_analysis 中须引用该层目录下的方法卡，或在 layers_deferred 中说明暂缓理由；"
            "引用须解析到真实存在的文件；目录引用改为具体卡文件；"
            "引用实际读过的方法卡及运行结果，注明采用的参数建议和适用条件。",
            file=sys.stderr,
        )
        return 1
    for warning in rr_warnings:
        print(f"⚠ validate_candidates: {warning}", file=sys.stderr)

    try:
        validate_round(data, run_dir, args.round)
    except ImportError as exc:
        print(
            f"✗ validate_candidates: optix 依赖不可用（{exc}），请在 msmodeling 仓库的 uv 环境中运行",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"✗ validate_candidates: {exc}", file=sys.stderr)
        return 1
    print(f"✓ validate_candidates: 分析、引用、搜索空间及候选装配校验通过（{cf}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

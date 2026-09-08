#!/usr/bin/env python3
"""config_writer.py — optix config.toml 薄层写入器。

第一性原理：把 config.toml 当数据结构（dict）操作，而不是正则文本手术。
  parse (tomlkit) → mutate (plain dict ops) → serialize (tomlkit) → check

写入职责分工：
  - agent/用户   决定"写什么值"（模型路径/SLO/卡数/搜索参数/引擎开关）
  - 本脚本       决定"怎么进 TOML"（只做机械 dict 操作，不含 schema/业务逻辑）
  - 校验器       决定"写对没有"（$VAR 协议 + JSON 段 + toml 回环；pydantic Settings 可选）

自包含：标准库 + tomlkit + 同目录 common.py（容器序列化/合并与 apply_launch_config
共享同一实现，保证两个写入入口产出完全一致的 canonical 格式）；不 import optix。
optix 的 pydantic Settings 校验在依赖可用时自动生效（见 --check 的 best-effort 段）。

本脚本取代 auto_config.py 的 regex 写入路径（引擎命令 / benchmark / 搜索与固定参数，
2026-09-02 迁移）。写入职责分工（`[agent_optimizer]` 专用节走 config_preflight --set-*）
见 SKILL.md「共享前置流程」共享护栏与 scripts/README.md 归属矩阵。

用法：
  # 引擎命令（--set-vllm-command 的替代）
  python config_writer.py --config <path> \
    --set vllm.command.model=/home/.../Kimi-K2.6-w4a8 \
    --set vllm.command.served_model_name=kimi_k26 --set vllm.command.port=8089 \
    --set-others vllm="--quantization ascend ... --speculative-config '{\"method\": \"dflash\", \
      \"num_speculative_tokens\": $NUM_SPECULATIVE_TOKENS}'"

  # benchmark（--set-vllm-benchmark / --set-ais-bench 的替代）
  python config_writer.py --config <path> \
    --set vllm_benchmark.command.dataset_name=random \
    --set-others vllm_benchmark="--random-input-len 2048 --random-output-len 512"
  python config_writer.py --config <path> \
    --set ais_bench.command.models=my_models --set ais_bench.command.mode=perf \
    --append-others ais_bench="--datasets my_datasets --num-prompts 3000"

  # 搜索/固定参数（--add-search-param / --add-fixed-param 的替代；可重复）
  python config_writer.py --config <path> \
    --target-field engine=vllm name=num_speculative_tokens min=3 max=12 dtype=int config_position=env

  # JSON-container 结构化写入（推荐，取代 --set-others 手拼 JSON 字符串）：
  #   --set <engine>.command.others.<flag>='{...JSON dict...}'
  # dict 由脚本经 common 序列化（单引号包裹）并 key 级合并进既有 others，永不畸形。
  python config_writer.py --config <path> \
    --set 'vllm.command.others.speculative-config={"method": "dflash", "num_speculative_tokens": 3}' \
    --set 'vllm.command.others.additional-config={"enable_cpu_binding": true}'
  # （旧 --set-others 字符串入口仍可用，JSON 值需自带单引号，用于无 JSON 的简单 others 更合适）

  # 校验 / 预览 / 打印
  python config_writer.py --config <path> --check-only
  python config_writer.py --config <path> --set ... --dry-run
  python config_writer.py --config <path> --print
"""

import argparse
import json
import re
import sys

import tomlkit

#: JSON-container 序列化/合并与 apply_launch_config 共享 common.py 实现，保证两个
#: 写入入口产出完全一致的 canonical 格式（单引号包裹 JSON）。自包含声明相应放开：
#: 容器序列化是跨脚本的机械能力，不是单脚本业务逻辑。
from common import (
    CONTAINER_FLAGS,
    bad_json_tokens,
    build_others,
    merge_others,
    table_or_create,
)

# optix 内建 $VAR（出现在 base 命令/特殊字段，不要求出现在 target_field）
KNOWN_BUILTIN_VARS = {
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "REQUESTRATE",
    "CONCURRENCY",
    "MAXCONCURRENCY",
}
#: scheduler/运行时内建消费的 env 参数：无需在 .others 里显式 $VAR（unwired 检查豁免）
SCHEDULER_CONSUMED_VARS = KNOWN_BUILTIN_VARS
#: 含 command.others 的段（引擎 + benchmark）
COMMAND_SECTIONS = ("vllm", "mindie", "vllm_benchmark", "ais_bench")
#: 可带 target_field 的引擎
ENGINES = ("vllm", "mindie")
#: --target-field 允许的字段（键名即 TOML 字段）
_TF_FIELDS = ("name", "config_position", "dtype", "min", "max", "value", "dtype_param")
#: command 段中约定为字符串的字段（optix custom_command.py: port: str = ""；config.toml 占位
#: port = "port"）。coerce 的数字推断不得作用其上，否则纯数字值会被写成 int 导致类型不符。
_STRING_COMMAND_FIELDS = {"port"}


# ---------------------------------------------------------------- IO
def parse_toml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return tomlkit.parse(f.read())


def dump_toml(path: str, doc) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))


# ---------------------------------------------------------------- mutate（纯 dict 操作）
def coerce(value: str, *, force_str: bool = False):
    """CLI 字符串 → TOML 值：'3'→3、'0.5'→0.5、'true'→True，否则保留字符串。

    force_str=True 时不做任何转换（字符串型 command 字段用，见 _STRING_COMMAND_FIELDS）。
    """
    if force_str:
        return value
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def coerce_deep(value: str):
    """target_field 值：数值/布尔原样；'[..]'/'{..}' → json.loads（enum 的 dtype_param 列表 /
    factories/times 的 dtype_param inline table）；其余保留字符串。
    """
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    s = value.strip()
    if s.startswith(("[", "{")):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return value


def _aot_or_new(tbl: tomlkit.items.Table, key: str):
    """返回 tbl[key] 的 array-of-tables；缺失时创建（保证序列化为 [[..]] 块而非内联 []）。"""
    val = tbl.get(key)
    if val is None:
        aot = tomlkit.aot()
        tbl[key] = aot
        return aot
    return val


def set_value(doc, key_path: str, value) -> None:
    """'vllm.command.model' → doc['vllm']['command']['model']（按需建中间表）。"""
    parts = key_path.split(".")
    cur = doc
    for p in parts[:-1]:
        cur = table_or_create(cur, p)
    cur[parts[-1]] = value


def set_others(doc, section: str, text: str) -> None:
    """整体覆盖 [section.command].others。

    经 common.merge_others 规范化一次（容器统一单引号 JSON），与 apply 同源。
    全量覆盖语义：用空 text 可清空；用目标 others 全串可重写。
    """
    cmd = table_or_create(table_or_create(doc, section), "command")
    # merge into '' == normalize to canonical, not append
    cmd["others"] = merge_others("", text)


def append_others(doc, section: str, text: str) -> None:
    """token 级并入 [section.command].others（ais_bench datasets 追加 / 容器 key 合并）。"""
    cmd = table_or_create(table_or_create(doc, section), "command")
    existing = str(cmd.get("others") or "").strip()
    cmd["others"] = merge_others(existing, text)


def set_container(doc, section: str, flag: str, value: dict) -> None:
    """结构化写入一个 JSON-container flag：把 ``value`` dict 合并进
    ``[section.command].others`` 的 ``--flag`` 容器（key 级，new 覆盖旧）。

    与 apply_launch_config 同源（common.build_others + merge_others），所以
    这里永不产生畸形 JSON——dict 经 json.dumps + 单引号序列化。这是取代
    "--set-others 手拼 JSON 字符串"的结构化入口。
    """
    full_flag = flag if flag.startswith("--") else f"--{flag}"
    if full_flag not in CONTAINER_FLAGS:
        raise ValueError(f"--set 容器路径需指向 JSON-container flag ({'/'.join(CONTAINER_FLAGS)})，收到 {full_flag!r}")
    cmd = table_or_create(table_or_create(doc, section), "command")
    existing = str(cmd.get("others") or "").strip()
    cmd["others"] = merge_others(existing, build_others({full_flag[2:]: value}))


def upsert_target_field(doc, engine: str, updates: dict) -> str:
    """target_field 列表按 name upsert；不存在则 append（保持 [[..]] 块格式）。返回 'added'/'updated'。"""
    table = table_or_create(doc, engine)
    fields = _aot_or_new(table, "target_field")
    for item in fields:
        if item.get("name") == updates.get("name"):
            item.update(updates)
            return "updated"
    fields.append(updates)
    return "added"


# ---------------------------------------------------------------- check（确定性门禁）
def _iter_others(doc):
    for section in COMMAND_SECTIONS:
        cmd = (doc.get(section) or {}).get("command") or {}
        if isinstance(cmd.get("others"), str):
            yield section, cmd["others"]


def _target_field_names(doc) -> set:
    names = set()
    for engine in ENGINES:
        for f in (doc.get(engine) or {}).get("target_field") or []:
            if f.get("name"):
                names.add(str(f["name"]))
    return names


def check_protocol(doc) -> list:
    """$VAR ↔ target_field 大写协议 + others 里 JSON 段合法性。返回错误列表（空 = 通过）。"""
    errors = []
    known = {n.upper() for n in _target_field_names(doc)} | KNOWN_BUILTIN_VARS
    for section, others in _iter_others(doc):
        # ① 每个 $VAR 必须能在 target_field name.upper() 或内建变量里找到
        for var in re.findall(r"\$[A-Z0-9_]+", others):
            varname = var[1:]
            if varname not in known:
                errors.append(
                    f"[{section}.command].others 里的 {var} 无对应 target_field name "
                    f"（大写协议 {varname.lower()} → {var}）或内建变量"
                )
        # ② 看起来像 JSON 的 token：$VAR 掩码 + json.loads 核心在 common.bad_json_tokens，
        #    此处只负责带 section 上下文拼装错误（掩码/解析规则单一宿主，勿在此重复实现）
        for tok in bad_json_tokens(others):
            errors.append(
                f"[{section}.command].others 的 JSON 段非法: {tok[:60]}…（请用结构化入口重写："
                f"config_writer --set {section}.command.others.<flag>={{json}}）"
            )
    return errors


def check_unwired_params(doc) -> list:
    """env-position target_field 的 $NAME 未被任何 others 引用 → 警告（可能写了搜索参数但没接进命令）。"""
    refs = set()
    for _, others in _iter_others(doc):
        refs.update(re.findall(r"\$([A-Z0-9_]+)", others))
    warnings = []
    for engine in ENGINES:
        for f in (doc.get(engine) or {}).get("target_field") or []:
            name = str(f.get("name") or "")
            if (
                name
                and f.get("config_position") == "env"
                and name not in SCHEDULER_CONSUMED_VARS
                and name.upper() not in refs
            ):
                warnings.append(
                    f"⚠ [{engine}.target_field] {name} config_position=env 但未在任一 .others 出现 $NAME —— "
                    "参数已写入但可能未接进服务命令，请确认是否需在 others 里加 --<flag> $NAME"
                )
    return warnings


def validate_settings(content: str) -> tuple:
    """best-effort：用 optix 的 pydantic Settings 做真·加载校验。

    返回 (errors, notes)：notes 为信息性说明（如依赖不可用），不计入失败。
    注意：SettingsConfigDict 未设 extra='forbid'，未知键会被 pydantic 静默忽略——
    未知键 allowlist 需要完整环境里的 schema，属于完整环境门禁的补充项。
    """
    try:
        import tomllib

        from optix.config.config import Settings
    except Exception as e:  # noqa: BLE001
        return [], [f"ℹ 跳过 pydantic Settings 校验（optix 依赖不可用: {type(e).__name__}: {e}），在完整环境自动生效"]
    try:
        Settings.model_validate(tomllib.loads(content))
        return [], []
    except Exception as e:  # noqa: BLE001
        return [f"pydantic Settings 校验失败: {e}"], []


def run_checks(content: str, doc) -> bool:
    """确定性门禁：$VAR 协议 + JSON 段 + TOML 回环 +（可选）pydantic Settings。"""
    errors = check_protocol(doc)
    warnings = check_unwired_params(doc)
    set_errors, notes = validate_settings(content)
    errors += set_errors
    for n in notes:
        print(n)
    for w in warnings:
        print(w)
    if not errors:
        parts = ["$VAR 协议", "JSON 段", "TOML 回环"]
        if not notes:
            parts.append("pydantic Settings")
        print("✓ 校验通过：" + " + ".join(parts))
        return True
    for e in errors:
        print(f"✗ {e}")
    return False


# ---------------------------------------------------------------- print
def _print_sections(doc, engines=ENGINES) -> None:
    print("== 写入后相关段 ==")
    for section in COMMAND_SECTIONS:
        blk = doc.get(section)
        if not blk:
            continue
        cmd = blk.get("command") or {}
        interesting = {
            k: cmd.get(k)
            for k in ("model", "served_model_name", "host", "port", "dataset_name", "models", "mode")
            if cmd.get(k) is not None
        }
        if interesting:
            print(f"[{section}.command] " + " ".join(f"{k}={v!r}" for k, v in interesting.items()))
        others = cmd.get("others")
        if others:
            print(f"[{section}.command].others = {others}")
        if section in engines:
            for f in blk.get("target_field") or []:
                name = f.get("name")
                if name and (f.get("config_position") == "env" or name == "num_speculative_tokens"):
                    extra = [
                        f"{k}={f.get(k)}"
                        for k in ("min", "max", "dtype", "value", "dtype_param")
                        if f.get(k) is not None
                    ]
                    print(
                        f"  [[{section}.target_field]] {name} config_position={f.get('config_position')} "
                        + " ".join(extra)
                    )


# ---------------------------------------------------------------- CLI
def _parse_kv_list(pairs) -> dict:
    result = {}
    for pv in pairs:
        key, _, value = pv.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"参数需 k=v: {pv!r}")
        result[key] = coerce(value)
    return result


def _parse_target_field(pairs) -> tuple:
    """'--target-field engine=vllm name=X min=3 ...' → (engine, name, spec)。"""
    spec = {}
    for pv in pairs:
        key, _, value = pv.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--target-field 需 k=v: {pv!r}")
        if key not in _TF_FIELDS and key not in ("engine",):
            raise ValueError(f"--target-field 字段应为 engine|{'|'.join(_TF_FIELDS)}，收到 {key!r}")
        spec[key] = coerce_deep(value)
    engine = spec.pop("engine", None)
    name = spec.pop("name", None)
    if engine not in ENGINES:
        raise ValueError(f"--target-field engine 应为 {'/'.join(ENGINES)}，收到 {engine!r}")
    if not name:
        raise ValueError("--target-field 必须含 name=")
    return engine, str(name), spec


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        description="optix config.toml 薄层写入器（parse→mutate→serialize→check）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例见文件头 docstring。",
    )
    ap.add_argument("--config", required=True, help="config.toml 绝对路径")
    ap.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="dot 路径写值，可重复（如 vllm.command.port=8089）",
    )
    ap.add_argument(
        "--set-others",
        action="append",
        metavar="ENGINE=TEXT",
        default=[],
        help="整体覆盖 [ENGINE.command].others，可重复（ENGINE ∈ vllm/mindie/vllm_benchmark/ais_bench）",
    )
    ap.add_argument(
        "--append-others",
        action="append",
        metavar="ENGINE=TEXT",
        default=[],
        help="token 级并入 [ENGINE.command].others（ais_bench datasets 追加用）",
    )
    ap.add_argument(
        "--target-field",
        action="append",
        nargs="+",
        metavar="k=v",
        default=[],
        help="按 (engine,name) upsert 一个 target_field 块，可重复；"
        "必填 engine/name，可选 min/max/dtype/value/config_position/dtype_param",
    )
    ap.add_argument("--dry-run", action="store_true", help="只预览不写文件")
    ap.add_argument("--check-only", action="store_true", help="只校验不写，错误退出 1")
    ap.add_argument("--check", action="store_true", help="写完后运行校验，失败退出 1")
    ap.add_argument("--print", action="store_true", help="写完后打印相关段")
    args = ap.parse_args(argv)

    # --- 静态校验 CLI 参数 ---
    for ev in args.set_others + args.append_others:
        engine, _, _ = ev.partition("=")
        if engine not in COMMAND_SECTIONS:
            print(f"✗ 段 {engine!r} 应为 {'/'.join(COMMAND_SECTIONS)}")
            return 2

    try:
        doc = parse_toml(args.config)
    except Exception as e:  # noqa: BLE001
        print(f"✗ TOML 解析失败: {e}")
        return 1

    if args.check_only:
        content = tomlkit.dumps(doc)
        return 0 if run_checks(content, doc) else 1

    try:
        for kv in args.set:
            key, _, value = kv.partition("=")
            if not key:
                raise ValueError(f"--set 需 KEY=VALUE: {kv!r}")
            # 结构化容器写入: <engine>.command.others.<flag>=<JSON dict>
            # (如 vllm.command.others.speculative-config='{"method":"dflash"}')
            # → dict 经 common 序列化+key级合并进 others，永不产生畸形 JSON。
            parts = key.split(".")
            if len(parts) == 4 and parts[1] == "command" and parts[2] == "others" and parts[0] in ENGINES:
                flag = parts[3]
                dict_value = json.loads(value)  # 必须整体是一个 JSON object
                if not isinstance(dict_value, dict):
                    raise ValueError(f"--set 容器路径 {key} 的值必须是 JSON dict")
                set_container(doc, parts[0], flag, dict_value)
                print(f"✓ 容器合并: [{parts[0]}] --{flag} {dict_value}")
                continue
            leaf = key.rsplit(".", 1)[-1]
            set_value(doc, key, coerce(value, force_str=leaf in _STRING_COMMAND_FIELDS))
        for ev in args.set_others:
            engine, _, text = ev.partition("=")
            set_others(doc, engine, text)
        for ev in args.append_others:
            engine, _, text = ev.partition("=")
            append_others(doc, engine, text)
        for pairs in args.target_field:
            engine, name, spec = _parse_target_field(pairs)
            action = upsert_target_field(doc, engine, {"name": name, **spec})
            print(f"{'✓ 更新' if action == 'updated' else '✓ 新增'} target_field: [{engine}] {name}")
    except ValueError as e:
        print(f"✗ {e}")
        return 2

    content = tomlkit.dumps(doc)

    # 回环校验：序列化产物必须能被重新解析（防止写入畸形 TOML）
    try:
        tomlkit.parse(content)
    except Exception as e:  # noqa: BLE001
        print(f"✗ TOML 回环解析失败（写入的产物非法）: {e}")
        return 2

    if args.dry_run:
        print("[DRY RUN] 预览（未写文件）：")
        print("-" * 50)
        print(content)
        if args.check:
            print("-" * 50)
            ok = run_checks(content, doc)
            if args.print:
                _print_sections(doc)
            return 0 if ok else 1
        return 0

    with open(args.config, "r", encoding="utf-8") as f:
        original = f.read()
    backup = args.config + ".bak"
    with open(backup, "w", encoding="utf-8") as f:
        f.write(original)
    print(f"✓ 原文件已备份: {backup}")
    dump_toml(args.config, doc)
    print(f"✓ 已写入 {args.config}")

    if args.check:
        ok = run_checks(content, doc)
        if args.print:
            _print_sections(doc)
        return 0 if ok else 1
    if args.print:
        _print_sections(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())

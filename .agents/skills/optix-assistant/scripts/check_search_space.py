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

"""Ownership: shared-precheck. Check configured parameter domains and resource feasibility.

读取 context.json 及 source.config_toml 指向的基础配置，检查参数定义、并行资源与
投机配置，输出 checklist_report.md；有错误时退出 1。候选门禁复用其中的检查函数。
只检查，不改写 TOML 或 context，不初始化搜索空间或注入推荐值。
"""

import argparse
import json
import shlex
from itertools import product
from pathlib import Path

from common import read_json

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _schema_from_fields(fields: list) -> list[dict]:
    """Normalize configured fields without adding recommendation defaults."""
    result = []
    names = set()
    for field in fields:
        item = dict(field) if isinstance(field, dict) else field.model_dump()
        name = item.get("name")
        if not name or name in names:
            raise ValueError(f"Missing or duplicate target_field name: {name!r}")
        names.add(name)
        item.setdefault("dtype", "float")
        item.setdefault("min", 0.0)
        item.setdefault("max", 100.0)
        if item["dtype"] == "range" and item.get("dtype_param"):
            start, end, step = int(item["min"]), int(item["max"]), int(item["dtype_param"])
            if step <= 0:
                raise ValueError(f"{name}: range step must be positive")
            item.update(dtype="enum", min=0, max=1, dtype_param=list(range(start, end + step, step)))
        item["default"] = item.get("value", 0.0)
        if item.get("constant") is not None:
            item["default"] = item["constant"]
        elif item["min"] == item["max"] and item["dtype"] in {"int", "float", "bool"}:
            item["constant"] = item["default"] = item["min"]
        if item["dtype"] == "enum":
            item["choices"] = item.get("dtype_param") or []
            if isinstance(item["default"], str):
                try:
                    decoded = json.loads(item["default"])
                    if decoded in item["choices"]:
                        item["default"] = decoded
                except ValueError:
                    pass
        result.append(item)
    return result


def _read_engine_config(path: str | Path, engine: str) -> dict:
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    section = next((v for k, v in config.items() if k.lower() == engine.lower()), None)
    if not isinstance(section, dict):
        raise ValueError(f"Missing engine section [{engine}] in {path}")
    return section


def load_configured_parameters(context: dict) -> tuple[dict, list[dict]]:
    """Read baseline definitions for checks without modifying the run context."""
    path = (context.get("source") or {}).get("config_toml")
    if not path:
        raise ValueError("context.source.config_toml is required")
    section = _read_engine_config(path, context.get("engine", ""))
    fields = section.get("target_field")
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"No target_field definitions in {path}")
    return section, _schema_from_fields(fields)


def check_fixed_concurrency(
    fields: list[dict], expected: int, *, value_key: str = "value", params: dict | None = None
) -> list[str]:
    """Require an explicit fixed load in TOML, active definitions and resolved candidates."""
    matches = [f for f in fields if str(f.get("name") or "").strip().upper() == "CONCURRENCY"]
    if len(matches) != 1:
        return [f"固定并发要求恰好一个 CONCURRENCY 字段（benchmark_spec.concurrency={expected}）"]
    field = matches[0]
    errors = []

    def matches_expected(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value == expected

    # Do not truncate floats or discard missing/non-numeric values. Missing
    # TOML bounds otherwise become the runtime's tunable 0..100 domain.
    if any(not matches_expected(field.get(key)) for key in (value_key, "min", "max")):
        errors.append(f"CONCURRENCY 固定并发要求 {value_key}/min/max 均明确等于 benchmark_spec.concurrency={expected}")
    if field.get("name") != "CONCURRENCY" or field.get("dtype") != "int" or field.get("config_position") != "env":
        errors.append("固定并发要求 name=CONCURRENCY、dtype=int、config_position=env")
    if field.get("constant") is not None and not matches_expected(field["constant"]):
        errors.append(f"CONCURRENCY constant 与 benchmark_spec.concurrency={expected} 不一致")
    if params is not None and "CONCURRENCY" in params and not matches_expected(params.get("CONCURRENCY")):
        errors.append(f"候选 CONCURRENCY 必须等于 benchmark_spec.concurrency={expected}")
    return errors


def check_parameters(
    context: dict, *, fields: list[dict] | None = None, section: dict | None = None, params: dict | None = None
) -> list[dict]:
    issues = []

    def error(message):
        issues.append({"level": "error", "msg": message})

    multihost = (context.get("deployment") or {}).get("type") == "multihost"
    try:
        if multihost:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from optix.optimizer.agentic.topology import validate_topology_snapshot

            validate_topology_snapshot(context)
        if section is None or fields is None:
            section, fields = load_configured_parameters(context)
    except (OSError, ValueError) as exc:
        error(str(exc))
        return issues
    fixed_concurrency = (context.get("benchmark_spec") or {}).get("concurrency")
    if (
        context.get("benchmark_spec_confirmed")
        and isinstance(fixed_concurrency, int)
        and not isinstance(fixed_concurrency, bool)
    ):
        # Check raw TOML as well as active definitions: normalization can hide
        # a missing value, and a round can otherwise widen an unfrozen domain.
        for message in check_fixed_concurrency(section.get("target_field") or [], fixed_concurrency):
            error(f"config.toml: {message}")
        for message in check_fixed_concurrency(fields, fixed_concurrency, value_key="default", params=params):
            error(f"参数定义/候选: {message}")
        if issues:
            return issues
    if not context.get("model_info"):
        issues.append({"level": "warning", "msg": "模型架构信息缺失，选择方法前需补充模型配置。"})
    for field in fields:
        name, dtype, value = field["name"], field["dtype"], field["default"]
        if not field.get("config_position"):
            error(f"{name}: missing config_position")
        if field["min"] > field["max"]:
            error(f"{name}: min exceeds max")
        if dtype == "enum":
            choices = field.get("choices")
            if not isinstance(choices, list) or not choices or value not in choices:
                error(f"{name}: value must be in nonempty dtype_param choices")
        elif dtype in {"int", "float", "bool", "range", "ratio"}:
            try:
                number = float(value)
                if not field["min"] <= number <= field["max"]:
                    error(f"{name}: value is outside configured range")
                if dtype == "int" and not number.is_integer():
                    error(f"{name}: value must be an integer")
                if dtype == "bool" and value not in (0, 1, False, True):
                    error(f"{name}: value must be a boolean")
            except (TypeError, ValueError):
                error(f"{name}: value must be numeric")
    try:
        others = (section.get("command") or {}).get("others", "")
        if others is None:
            others = ""
        if not isinstance(others, str):
            raise ValueError("command.others must be a command-line string")
        tokens = shlex.split(others)
        flags = {}
        for i, token in enumerate(tokens):
            if not token.startswith("--"):
                continue
            name, separator, value = token[2:].partition("=")
            if not separator and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value = tokens[i + 1]
            flags[name.replace("-", "_").lower()] = value
        spec = json.loads(flags.get("speculative_config", "{}"))
        if not isinstance(spec, dict):
            raise ValueError("speculative-config must be a JSON object")
        values = [spec.get("num_speculative_tokens", 0)]
        for field in fields:
            if field["name"].lower() == "num_speculative_tokens":
                values = [params[field["name"]]] if params is not None else field.get("choices", [field["max"]])
                break
        has_spec_base = any(isinstance(spec.get(key), str) and spec[key].strip() for key in ("method", "model"))
        if any(float(v) > 0 for v in values) and not has_spec_base:
            error("num_speculative_tokens requires method/model in command.others speculative-config")
        world = (context.get("constraints") or {}).get("world_size") or (context.get("hardware") or {}).get(
            "world_size"
        )
        if world and context["engine"] in {"vllm", "mindie"}:
            domains = []
            by_name = {f["name"].lower(): f for f in fields}
            aliases = [("tp", "tensor_parallel_size"), ("dp", "data_parallel_size")]
            if context["engine"] == "vllm":
                aliases += [("pp", "pipeline_parallel_size"), ("pcp", "prefill_context_parallel_size")]
            divisors = [v for v in range(1, world + 1) if world % v == 0]
            for short, long in aliases:
                field = by_name.get(short) or by_name.get(long)
                if field is None:
                    domain = [
                        int(
                            flags.get(
                                long,
                                flags.get(
                                    short,
                                    (context.get("hardware") or {}).get("num_nodes", 1)
                                    if multihost and short == "dp"
                                    else 1,
                                ),
                            )
                        )
                    ]
                elif params is not None:
                    domain = [params[field["name"]]]
                elif field.get("constant") is not None:
                    domain = [field["constant"]]
                elif field["dtype"] == "enum":
                    domain = field["choices"]
                else:
                    domain = [v for v in divisors if field["min"] <= v <= field["max"]]
                domains.append(domain)
            from math import prod

            def feasible(values):
                if prod(values) != world:
                    return False
                if multihost:
                    from optix.optimizer.agentic.topology import validate_parallel_placement

                    hardware = context.get("hardware") or {}
                    try:
                        validate_parallel_placement(*values, hardware["num_nodes"], hardware["num_per_node"])
                    except (TypeError, ValueError):
                        return False
                return True

            if not any(feasible(values) for values in product(*domains)):
                subject = "Candidate parallel values" if params is not None else "Configured parallel domains"
                error(f"{subject} cannot use WORLD_SIZE={world}")
    except (TypeError, ValueError) as exc:
        error(f"Invalid service configuration: {exc}")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Check config.toml parameter definitions and resource feasibility.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    try:
        context = read_json(run_dir / "context.json")
        issues = check_parameters(context)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Configuration check failed: {exc}\n")
    lines = ["# 配置参数检查", "", f"配置文件：`{context['source']['config_toml']}`", ""]
    lines.extend(f"- {item['level']}: {item['msg']}" for item in issues)
    if not issues:
        lines.append("- 参数定义与并行资源检查通过。")
    (run_dir / "checklist_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    errors = sum(item["level"] == "error" for item in issues)
    print(f"Configuration check: {errors} error(s), {len(issues) - errors} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Validate the vLLM Ascend preset catalog JSON against its schema.

The preset catalog (``presets/ascend_vllm_presets.json``) is an offline-harvested,
statically committed data asset consumed by ``collect_context.py`` / ``recommend_params.py`` (optix-assistant).
This module is the single source of truth for the catalog schema:

- Pure ``validate_presets()`` used by unit tests.
- CLI entry: exit non-zero if any ``error``-level issue is found.

See ``docs/design/vllm-ascend-preset-catalog-design.md`` for the data model.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL = ("schema_version", "meta", "presets")
REQUIRED_META = ("source_domain", "source_section", "harvested_at")
REQUIRED_MODEL = ("name",)
REQUIRED_SCENARIO = ("id", "params", "source_url")
SCALAR_TYPES = (int, float, bool, str)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file; raise on missing file or malformed JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def is_scalar(value: Any) -> bool:
    """Preset param values must be JSON scalars (int/float/bool/str)."""
    return type(value) in SCALAR_TYPES


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return type(value).__name__


def _error(issues: list[dict[str, str]], msg: str) -> None:
    issues.append({"level": "error", "msg": msg})


def _warn(issues: list[dict[str, str]], msg: str) -> None:
    issues.append({"level": "warning", "msg": msg})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_presets(presets: dict[str, Any]) -> list[dict[str, str]]:
    """Validate a loaded preset catalog. Returns a list of issues.

    Each issue is ``{"level": "error"|"warning", "msg": str}``. ``error`` blocks
    commit/consumption; ``warning`` suggests provenance/quality gaps.
    """
    issues: list[dict[str, str]] = []
    if not isinstance(presets, dict):
        _error(issues, "顶层必须是 JSON 对象")
        return issues

    for key in REQUIRED_TOP_LEVEL:
        if key not in presets:
            _error(issues, f"缺少顶层字段 '{key}'")

    meta = presets.get("meta")
    if isinstance(meta, dict):
        for key in REQUIRED_META:
            if key not in meta:
                _warn(issues, f"meta 缺少字段 '{key}'（溯源信息）")
    elif "meta" in presets:
        _error(issues, "'meta' 必须是对象")

    if "presets" not in presets:
        return issues
    preset_list = presets["presets"]
    if not isinstance(preset_list, list):
        _error(issues, "'presets' 必须是数组")
        return issues
    if not preset_list:
        _error(issues, "'presets' 为空数组，目录至少包含一个模型")

    for idx, preset in enumerate(preset_list):
        prefix = f"presets[{idx}]"
        if not isinstance(preset, dict):
            _error(issues, f"{prefix} 必须是对象")
            continue
        _validate_preset(preset, prefix, issues)

    return issues


def _validate_preset(preset: dict[str, Any], prefix: str, issues: list[dict[str, str]]) -> None:
    model = preset.get("model")
    if not isinstance(model, dict):
        _error(issues, f"{prefix}.model 必须是对象")
    else:
        for key in REQUIRED_MODEL:
            if not model.get(key):
                _error(issues, f"{prefix}.model 缺少必填字段 '{key}'")
        if isinstance(model.get("name"), str) and not model["name"].strip():
            _error(issues, f"{prefix}.model.name 不能为空字符串")
        if "aliases" in model and not isinstance(model["aliases"], list):
            _error(issues, f"{prefix}.model.aliases 必须是数组")
        if "is_moe" in model and not isinstance(model["is_moe"], bool):
            _error(issues, f"{prefix}.model.is_moe 必须是布尔值")

    scenarios = preset.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        _error(issues, f"{prefix}.scenarios 必须是非空数组")
        return

    seen_ids: set[str] = set()
    for sidx, scenario in enumerate(scenarios):
        s_prefix = f"{prefix}.scenarios[{sidx}]"
        if not isinstance(scenario, dict):
            _error(issues, f"{s_prefix} 必须是对象")
            continue
        _validate_scenario(scenario, s_prefix, issues, seen_ids)


def _validate_scenario(scenario: dict[str, Any], prefix: str, issues: list[dict[str, str]], seen_ids: set[str]) -> None:
    for key in REQUIRED_SCENARIO:
        if key not in scenario:
            _error(issues, f"{prefix} 缺少必填字段 '{key}'")

    scenario_id = scenario.get("id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        _error(issues, f"{prefix}.id 必须是非空字符串")
    elif scenario_id in seen_ids:
        _error(issues, f"{prefix}.id '{scenario_id}' 重复")
    else:
        seen_ids.add(scenario_id)

    params = scenario.get("params")
    if not isinstance(params, dict):
        _error(issues, f"{prefix}.params 必须是对象")
    elif not params:
        _warn(issues, f"{prefix}.params 为空（官方页面未给出推荐值？）")
    else:
        for pname, pvalue in params.items():
            if not is_scalar(pvalue):
                _error(issues, f"{prefix}.params['{pname}'] 值类型必须是标量，得到 {_type_name(pvalue)}")

    env = scenario.get("env")
    if env is not None and not isinstance(env, dict):
        _error(issues, f"{prefix}.env 必须是对象")
    if isinstance(env, dict):
        for k, v in env.items():
            if not isinstance(v, str):
                _error(issues, f"{prefix}.env['{k}'] 值必须是字符串")

    additional = scenario.get("additional_config")
    if additional is not None and not isinstance(additional, dict):
        _error(issues, f"{prefix}.additional_config 必须是对象")

    url = scenario.get("source_url")
    if isinstance(url, str) and not url.startswith("http"):
        _warn(issues, f"{prefix}.source_url 应以 http 开头，当前 '{url}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate vLLM Ascend preset catalog JSON.")
    parser.add_argument("--presets", required=True, help="Path to presets JSON")
    parser.add_argument("--quiet", action="store_true", help="只输出错误，不输出通过/警告")
    args = parser.parse_args(argv)

    path = Path(args.presets)
    if not path.exists():
        print(f"ERROR: 文件不存在: {path}")
        return 2
    try:
        presets = read_json(path)
    except json.JSONDecodeError as exc:
        print(f"ERROR: JSON 解析失败: {exc}")
        return 2

    issues = validate_presets(presets)
    errors = [i for i in issues if i["level"] == "error"]
    warnings = [i for i in issues if i["level"] == "warning"]

    for issue in errors:
        print(f"ERROR: {issue['msg']}")
    if not args.quiet:
        for issue in warnings:
            print(f"WARNING: {issue['msg']}")
        if not issues:
            print("OK: schema 校验通过")

    if errors:
        print(f"FAIL: {len(errors)} 个错误")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

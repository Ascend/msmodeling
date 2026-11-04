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
"""One self-contained, human-readable result for the complete PD pipeline."""

import json
from pathlib import Path

from ..io_utils import open_file


def _read_result(path):
    if not path.is_file():
        return {}
    with open_file(path, "r") as stream:
        return json.load(stream)


def _cell(value):
    if value is None:
        return "未提供"
    if isinstance(value, (dict, list, bool)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\n", "<br>").replace("\r", "")


def _table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows),
        "",
    ]


def _json_block(value):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    # A model path or command may itself contain Markdown backticks.
    fence = "```"
    while fence in text:
        fence += "`"
    return [fence + "json", text, fence, ""]


def _metric_rows(performance, phase):
    rows = []
    for title, key, target, upper in (
        ("TTFT（秒）", "time_to_first_token", phase.ttft_slo, True),
        ("TPOT（秒）", "time_per_output_token", phase.tpot_slo, True),
        ("成功率", "success_rate", phase.success_rate_slo, False),
    ):
        actual = performance.get(key)
        met = actual is not None and (actual <= target if upper else actual >= target)
        verdict = "未测量" if actual is None else ("达标" if met else "未达标")
        rows.append((title, actual, ("≤ " if upper else "≥ ") + str(target), verdict))
    rows.extend(
        (title, performance.get(key), "—", "—")
        for title, key in (("请求吞吐（请求/秒）", "throughput"), ("输出吞吐（token/秒）", "generate_speed"))
    )
    return rows


def _deployment_section(deployment):
    if not deployment:
        return ["## 最终部署", "", "尚未生成完整服务部署配置。", ""]
    allocation = deployment.get("instances") or {}
    lines = ["## 最终部署", ""]
    lines += _table(
        ["P 实例", "D 实例", "P 卡数", "D 卡数"],
        [[allocation.get(key) for key in ("prefill", "decode", "prefill_devices", "decode_devices")]],
    )
    usage = deployment.get("resource_usage") or {}
    lines += _table(
        ["设备池总卡数", "预算卡数", "已分配", "预算内余量", "池内闲置"],
        [
            [
                usage.get("pool_total_devices"),
                usage.get("budget_devices"),
                usage.get("allocated_devices", allocation.get("used_devices")),
                usage.get("budget_remaining_devices", allocation.get("remaining_devices")),
                usage.get("pool_idle_devices"),
            ]
        ],
    )
    if usage.get("pool_total_devices") is None:
        lines += ["此部署记录未保存原始设备池总量，无法确定池内闲置卡数。", ""]
    config = deployment.get("config") or {}
    rows = []
    for role in ("prefill", "decode"):
        plan = deployment.get("plans", {}).get(role, {})
        for index, group in enumerate(config.get(role + "_groups", [])):
            for node in group.get("nodes", []):
                rows.append(
                    (
                        role,
                        index,
                        node.get("ssh_ip"),
                        node.get("docker_container_id"),
                        node.get("gpu_ids"),
                        f"{node.get('bind_ip')}:{node.get('service_port')}",
                        f"{plan.get('tp')}/{plan.get('dp')}/{plan.get('pp')}",
                    )
                )
    lines += _table(["角色", "组", "机器", "容器", "物理卡", "服务地址", "TP/DP/PP"], rows)
    proxy = config.get("proxy") or {}
    lines += [f"Proxy：{_cell(proxy.get('bind_ip'))}:{_cell(proxy.get('service_port'))}", ""]
    lines += ["### 完整部署配置", "", "包含两侧模型、启动参数、环境和节点配置，可直接在本文件中核对和提取。", ""]
    lines += _json_block(deployment)
    return lines


def write_final_report(run_dir: Path, summary: dict, config) -> Path:
    """Combine selected results, deployed parameters and SLO verdict in one file."""
    phase = config.full_service
    full = summary.get("full_service") or {}
    completed = summary.get("status") == "completed"
    verdict = "达标" if full.get("slo_met") else "未达标"
    if not completed:
        verdict = "未完成，不能判定完整流程达标"
    lines = [
        "# OptiX PD 完整流程结果",
        "",
        f"运行 ID：{_cell(summary.get('run_id'))}",
        "",
        "执行状态：" + ("已完成" if completed else "失败 / 未完成"),
        "",
        f"完整服务 SLO：{verdict}",
        "",
        "本文件集中保存最终结果，无需打开各阶段 JSON 或 CSV 拼接配置。",
        "",
        "**分享提示：本报告可能包含机器地址、容器标识、模型与输出路径、启动参数和环境配置，"
        "均未自动脱敏。对外分享前请检查并脱敏。**",
        "",
    ]
    if summary.get("error"):
        error = summary["error"]
        lines += [
            f"失败阶段：{_cell(summary.get('failed_phase'))}",
            "",
            f"错误：{_cell(error.get('type'))}: {_cell(error.get('message'))}",
            "",
            "以下参数和测量仅代表失败前已保存的结果。",
            "",
        ]
    cleanup_errors = full.get("cleanup_errors") or []
    if cleanup_errors:
        lines += ["## 清理异常（附加信息）", ""]
        lines += _table(
            ["清理操作", "异常类型", "错误"],
            ((error.get("operation"), error.get("type"), error.get("message")) for error in cleanup_errors),
        )
    lines += ["## 完整服务实测与负载", ""]
    runs = phase.benchmark_run_count
    lines += [
        f"完整服务每个负载候选压测轮数：{runs}；"
        + (f"前 {runs - 1} 轮预热，仅第 {runs} 轮计分。" if runs > 1 else "单轮直接计分，未额外安排压测预热轮次。"),
        "",
    ]
    if runs > 1:
        lines += ["各轮复用同一套 PD 服务，候选负载参数保持不变。", ""]
    ttft_source = "显式配置" if "ttft_slo" in phase.model_fields_set else "默认值"
    lines += [f"TTFT 目标来源：pd_disagg.full_service.ttft_slo（{ttft_source}：{phase.ttft_slo} 秒）。", ""]
    if "ttft_slo" not in phase.model_fields_set:
        lines += ["请按模型、输入输出长度和业务要求显式配置完整链路 TTFT 目标。", ""]
    lines += _table(["指标", "实测值", "目标", "结论"], _metric_rows(full.get("performance_index") or {}, phase))
    lines += _table(["最终负载参数", "值"], (full.get("benchmark_params") or {}).items())
    lines += [
        "负载微调策略：pd_mixed，在各字段配置范围内联合调整并发和请求率，固定字段保持固定。",
        "",
        "完整服务阶段固定 use_request_rate_calibration=false；顶层请求率标定开关不作用于本阶段。",
        "",
        "负载范围来自 pd_disagg.full_service.target_field；最终值从本阶段已实测的基线和微调候选中择优。",
        "",
    ]
    concurrency = next((field for field in phase.target_field if field.name == "CONCURRENCY"), None)
    if concurrency is not None:
        actual = (full.get("benchmark_params") or {}).get("CONCURRENCY")
        lines += _table(
            ["并发参数", "配置下限", "配置上限", "配置初始值", "最终实测值", "取值来源"],
            [
                (
                    "CONCURRENCY",
                    concurrency.min,
                    concurrency.max,
                    concurrency.value,
                    actual,
                    "完整服务实测候选择优（含基线）" if actual is not None else "未生成完整服务选参结果",
                )
            ],
        )
    lines += [f"压测插件：{_cell(phase.benchmark_policy)}", "", "压测命令覆盖（输入输出长度、请求数等）：", ""]
    lines += _json_block(phase.benchmark_command_overrides)
    lines += [
        "## P/D 单侧搜索结果",
        "",
        "单侧指标用于选参和计算配比，与完整服务实测分别展示。",
        "",
        "硬件一致性：未采集单侧搜索与部署节点的卡型号，未自动核对硬件一致性。"
        "单侧搜索在当前控制端配置的设备上执行；部署节点型号、显存或带宽不同时，需另行验证参数适用性。",
        "",
    ]
    lines += _table(
        ["阶段", "状态", "QPS（请求/秒）", "TTFT（秒）", "TPOT（秒）", "成功率"],
        [
            [role, *(summary.get(role, {}).get(key) for key in ("status", "qps", "ttft_s", "tpot_s", "success_rate"))]
            for role in ("prefill", "decode")
        ],
    )
    for role in ("prefill", "decode"):
        result = _read_result(run_dir / role / "phase_result.json")
        lines += [f"### {role} 最终选中参数", ""]
        lines += _table(["服务参数", "值"], (result.get("service_params") or {}).items())
        lines += _table(["单侧压测参数", "值"], (result.get("benchmark_params") or {}).items())
        role_config = getattr(config, role)
        lines += [
            f"每个候选压测轮数：{role_config.benchmark_run_count}；"
            + ("前面各轮预热，仅最后一轮计分。" if role_config.benchmark_run_count > 1 else "单轮计分。"),
            "",
        ]
    lines += ["## 配比推荐", "", f"理论 P:D = {_cell(summary.get('pd_ratio'))}:1", ""]
    if summary.get("allocation_warning"):
        lines += [_cell(summary["allocation_warning"]), ""]
    lines += _deployment_section(_read_result(run_dir / "full_service" / "deployment.json"))
    lines += ["## 追溯文件（可选）", "", "排查单个候选时再查看以下原始产物：", ""]
    lines += _table(
        ["产物", "路径"], ((key, value) for key, value in summary.get("artifacts", {}).items() if key != "final_report")
    )
    path = run_dir / "final_result.md"
    with open_file(path, "w") as stream:
        stream.write("\n".join(lines))
    return path

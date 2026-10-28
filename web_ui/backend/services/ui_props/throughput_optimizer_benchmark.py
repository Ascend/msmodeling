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

"""throughput_optimizer_benchmark UI field properties for Web UI form generation.

This module defines UI-specific properties for the Benchmark mode of throughput_optimizer.
Benchmark mode exposes only 14 core fields (vs ~60 in full mode) and uses convention
fields (pd-mode, quantize-preset) that are expanded by the runner before execution.
"""

from web_ui.backend.services.ui_props import I18nText, UIFieldProps, MODEL_ID_OPTIONS
from web_ui.backend.services.ui_props.throughput_optimizer import (
    GROUP_LABELS,
    OPTION_SOURCE_REGISTRY,
    UI as FULL_UI,
)
from cli.registry.modules import get_spec
from web_ui.backend.services.json_adapter import FormConfig, export_form_json


VERSION = "1.0.10"
TITLE = I18nText("吞吐优化 (Benchmark)", "Throughput Optimizer (Benchmark)")
RUNNER = "ParallelRunner"

# Reuse group labels from full mode
GROUPS = [
    {"group_key": "General", "defaultCollapsed": False},
    {"group_key": "Request", "defaultCollapsed": False},
    {"group_key": "SLO", "defaultCollapsed": False},
    {"group_key": "Search", "defaultCollapsed": False},
    {"group_key": "Quantization", "defaultCollapsed": False},
]

# Benchmark-specific validators
VALIDATOR_UI = {
    "effectiveLenGe1": {
        "fields": ["input-length", "prefix-cache-hit-rate"],
        "message": I18nText(
            "有效输入长度(扣除前缀缓存后)必须 ≥ 1",
            "Effective input length (after prefix cache) must be ≥ 1",
        ),
    },
}


# ── Visible fields (13 core fields) ──────────────────────────────────────────

UI: dict[str, UIFieldProps] = {
    # 1. model-id (general)
    "model-id": UIFieldProps(
        label=I18nText("模型名称", "Model Name"),
        tooltip=I18nText(
            "待仿真模型的标准 HuggingFace 名称（组织/模型，如 Qwen/Qwen3-32B）或本地路径。",
            "Standard HuggingFace model name (org/model, e.g. Qwen/Qwen3-32B) or local path.",
        ),
        placeholder=I18nText("如 Qwen/Qwen3-32B", "e.g. Qwen/Qwen3-32B"),
        default="Qwen/Qwen3-32B",
        required=True,
        group={"zh": "通用", "en": "General"},
        control="combobox",
        option_source={"type": "inline", "values": MODEL_ID_OPTIONS},
    ),
    # 2. device (general)
    "device": UIFieldProps(
        label=I18nText("硬件", "Device"),
        tooltip=I18nText(
            "选择用于仿真的设备 Profile（可多选；每个取值单独仿真，结果区生成多用例对比）。",
            "Device profile(s) to simulate on (multi-select; each value runs independently and yields a multi-case comparison).",
        ),
        placeholder=I18nText("请选择设备型号", "Select device models"),
        option_source={"type": "dynamic", "name": "devices"},
        multi_values=True,
        data_type="string[]",
        default=["ATLAS_350_425T_112G"],
        required=True,
        group={"zh": "通用", "en": "General"},
        control="multi-select",
    ),
    # 3. num-devices (general)
    "num-devices": UIFieldProps(
        label=I18nText("设备数量", "Number of Devices"),
        tooltip=I18nText(
            "参与仿真的 Die 数量（≥1）。",
            "Number of devices used in the simulation (≥1).",
        ),
        default=4,
        required=True,
        group={"zh": "通用", "en": "General"},
        control="number",
    ),
    # 4. pd-mode (general) - convention field
    "pd-mode": UIFieldProps(
        label=I18nText("部署模式", "Deployment Mode"),
        tooltip=I18nText(
            "选择 Prefill/Decode 部署模式。pd_mixed（聚合模式）为默认，表示 prefill 和 decode 在同一实例内并发执行。",
            "Select Prefill/Decode deployment mode. pd_mixed (aggregation) is default, meaning prefill and decode run concurrently in the same instance.",
        ),
        default="pd_mixed",
        required=True,
        group={"zh": "通用", "en": "General"},
        control="select",
        option_source={
            "type": "inline",
            "values": [
                {
                    "value": "pure_prefill",
                    "label": {"zh": "仅 Prefill", "en": "Prefill Only"},
                },
                {
                    "value": "pure_decode",
                    "label": {"zh": "仅 Decode", "en": "Decode Only"},
                },
                {
                    "value": "pd_separation",
                    "label": {"zh": "PD 分离", "en": "PD Separation"},
                },
                {"value": "pd_mixed", "label": {"zh": "聚合模式", "en": "Aggregation"}},
            ],
        },
    ),
    # 5. input-length (request)
    "input-length": UIFieldProps(
        label=I18nText("输入长度", "Input Length"),
        tooltip=I18nText(
            "输入 prompt 的 token 长度（≥1，<1e6）。",
            "Input prompt token length (≥1, <1e6).",
        ),
        default=3500,
        required=True,
        group={"zh": "请求", "en": "Request"},
        control="number",
    ),
    # 6. output-length (request)
    "output-length": UIFieldProps(
        label=I18nText("输出长度", "Output Length"),
        tooltip=I18nText("输出 token 长度（≥1，<1e6）。", "Output token length (≥1, <1e6)."),
        default=1500,
        required=True,
        group={"zh": "请求", "en": "Request"},
        control="number",
    ),
    # 7. prefix-cache-hit-rate (request)
    "prefix-cache-hit-rate": UIFieldProps(
        label=I18nText("PrefixCache 命中率", "Prefix Cache Hit Rate"),
        tooltip=I18nText(
            "输入 prompt 中可被 PrefixCache 命中的比例（[0, 1)）。",
            "Ratio of input prompt that can be hit by PrefixCache ([0, 1)).",
        ),
        default=0.0,
        group={"zh": "请求", "en": "Request"},
        control="number",
    ),
    # 8. ttft-limit (SLO) - auto-disabled when pd-mode = pure_decode (no prefill phase)
    "ttft-limit": UIFieldProps(
        label=I18nText("TTFT SLO (ms)", "TTFT SLO (ms)"),
        tooltip=I18nText(
            "首 token 延迟约束（ms）。仅 Decode 模式下自动禁用（无 prefill 阶段）。",
            "Time to first token constraint (ms). Auto-disabled in Decode-only mode (no prefill phase).",
        ),
        group={"zh": "SLO", "en": "SLO"},
        control="number",
        conditions={
            "enabled": {"not": {"field": "pd-mode", "op": "eq", "value": "pure_decode"}},
        },
    ),
    # 9. tpot-limit (SLO) - auto-disabled when pd-mode = pure_prefill (no decode phase)
    "tpot-limit": UIFieldProps(
        label=I18nText("TPOT SLO (ms)", "TPOT SLO (ms)"),
        tooltip=I18nText(
            "每 token 输出延迟约束（ms）。仅 Prefill 模式下自动禁用（无 decode 阶段）。",
            "Time per output token constraint (ms). Auto-disabled in Prefill-only mode (no decode phase).",
        ),
        group={"zh": "SLO", "en": "SLO"},
        control="number",
        conditions={
            "enabled": {"not": {"field": "pd-mode", "op": "eq", "value": "pure_prefill"}},
        },
    ),
    # 10. max-concurrency (constraints)
    "max-concurrency": UIFieldProps(
        label=I18nText("最大并发", "Maximum Concurrency"),
        tooltip=I18nText(
            "每个 Prefill 或 Decode 实例的请求并发上限。",
            "Maximum simultaneous requests for each optimizer instance, including each Prefill or Decode phase.",
        ),
        group={"zh": "约束", "en": "Constraints"},
        control="number",
    ),
    # 11. tp-sizes (search)
    "tp-sizes": UIFieldProps(
        label=I18nText("TP 搜索尺寸", "TP Search Sizes"),
        tooltip=I18nText(
            "逗号分隔的张量并行搜索尺寸（如 1,2,4），每个 ≤ num_devices；每个尺寸都会评估并在结果扫描表中对比各组合；留空表示自动搜索 2 的幂次。",
            "Comma-separated TP search sizes (e.g. 1,2,4), each ≤ num_devices; every size is evaluated and compared in the result sweep table; empty = auto-search powers of 2.",
        ),
        group={"zh": "搜索", "en": "Search"},
        control="text",
    ),
    # 12. pp-sizes (search)
    "pp-sizes": UIFieldProps(
        label=I18nText("PP 搜索尺寸", "PP Search Sizes"),
        tooltip=I18nText(
            "逗号分隔的流水线并行搜索尺寸（如 1,2,4）；留空表示不搜索，仅使用 PP=1；指定值后会在结果扫描表中对比各组合。",
            "Comma-separated pipeline-parallel search sizes (e.g. 1,2,4); empty = no search, use PP=1 only; specified values are evaluated and compared in the result sweep table.",
        ),
        placeholder=I18nText("如 1,2,4", "e.g. 1,2,4"),
        group={"zh": "搜索", "en": "Search"},
        control="text",
    ),
    # 13. pp-layer-partitions (search)
    "pp-layer-partitions": UIFieldProps(
        label=I18nText("PP 层分配", "PP Layer Partitions"),
        tooltip=I18nText(
            "显式指定 PP 层分配，JSON 格式的列表的列表。例如：[[31,30],[16,15,15,15]]。每个内部列表的长度必须等于对应的 pp_size，且总和等于 num_hidden_layers。留空表示使用默认的均衡分配。",
            "Explicit PP layer partitions as a JSON list of lists. Example: [[31,30],[16,15,15,15]]. Each inner list's length must equal its pp_size and sum to num_hidden_layers. Empty = use default balanced partition.",
        ),
        placeholder=I18nText("如 [[31,30],[16,15,15,15]]", "e.g. [[31,30],[16,15,15,15]]"),
        group={"zh": "搜索", "en": "Search"},
        control="text",
    ),
    # 14. quantize-preset (quantization) - convention field
    "quantize-preset": UIFieldProps(
        label=I18nText("量化精度", "Quantization Preset"),
        tooltip=I18nText(
            "量化精度预设。W{N}A{M}C{K} 三段式：W=权重精度，A=激活精度，C=KV Cache 量化。custom 表示沿用全量模式设置。",
            "Quantization preset. W{N}A{M}C{K} format: W=weight precision, A=activation precision, C=KV cache quantization. custom means using full mode settings.",
        ),
        default="W8A8C8",
        group={"zh": "量化", "en": "Quantization"},
        control="select",
        option_source={
            "type": "inline",
            "values": [
                {"value": "none", "label": {"zh": "不量化", "en": "None"}},
                {"value": "W8A8C8", "label": "W8A8C8"},
                {"value": "W8A8C16", "label": "W8A8C16"},
                {"value": "W8A16C8", "label": "W8A16C8"},
                {"value": "W8A16C16", "label": "W8A16C16"},
                {"value": "W4A8C8", "label": "W4A8C8"},
                {"value": "W4A8C16", "label": "W4A8C16"},
                {
                    "value": "custom",
                    "label": {"zh": "沿用 full 设置", "en": "Use full settings"},
                },
            ],
        },
    ),
}


# ── Hidden fields (~48 fields from full mode) ────────────────────────────────
# These fields are managed by the benchmark schema but not visible to the user.
# They are populated with defaults or expanded from convention fields by the runner.
# FULL_UI (imported above) supplies the hidden-field definitions.

# Benchmark visible fields (14 core fields)
_BENCHMARK_VISIBLE_FIELDS = {
    "model-id",
    "device",
    "num-devices",
    "pd-mode",
    "input-length",
    "output-length",
    "prefix-cache-hit-rate",
    "ttft-limit",
    "tpot-limit",
    "max-concurrency",
    "tp-sizes",
    "pp-sizes",
    "pp-layer-partitions",
    "quantize-preset",
}

# Copy ALL fields from full UI and mark them as hidden (except the 13 visible ones)
for field_id, field_props in FULL_UI.items():
    if field_id in _BENCHMARK_VISIBLE_FIELDS:
        continue  # Skip visible fields (already defined above)

    # Create a copy with hidden=True
    UI[field_id] = UIFieldProps(
        label=field_props.label,
        tooltip=field_props.tooltip,
        placeholder=field_props.placeholder,
        option_source=field_props.option_source,
        disabled=field_props.disabled,
        conditions=field_props.conditions,
        hidden=True,  # Key difference: hidden in benchmark mode
        multi_values=field_props.multi_values,
        data_type=field_props.data_type,
        choices=field_props.choices,
        default=field_props.default,
        required=field_props.required,
        group=field_props.group,
        control=field_props.control,
        clearable=field_props.clearable,
    )


def get_form_config() -> tuple:
    """Return (ModuleSpec, FormConfig) for benchmark mode."""
    spec = get_spec("throughput_optimizer")  # Reuse the same ModuleSpec
    form_config = FormConfig(
        version=VERSION,
        title=TITLE,
        runner=RUNNER,
        group_labels=GROUP_LABELS,
        groups=GROUPS,
        option_source_registry=OPTION_SOURCE_REGISTRY,
        validator_ui=VALIDATOR_UI,
        ui=UI,
    )
    return spec, form_config


def generate_form_json() -> dict:
    """Generate the form JSON envelope for benchmark mode."""
    spec, form_config = get_form_config()
    result = export_form_json(spec, form_config)
    # Override moduleId to distinguish from full mode schema
    result["moduleId"] = "throughput_optimizer_benchmark"
    return result

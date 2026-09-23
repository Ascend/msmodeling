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

"""image_generate UI field properties for Web UI form generation.

This module defines UI-specific properties (labels, tooltips, validation, grouping)
for image_generate parameters. Combined with the ModuleSpec from
cli/registry/modules/image_generate.py, this generates the form JSON.
"""

from web_ui.backend.services.ui_props import I18nText, UIFieldProps, LOG_LEVEL_UI_FULL
from cli.registry.modules import get_spec
from web_ui.backend.services.json_adapter import FormConfig, export_form_json


VERSION = "2.3.3"
TITLE = I18nText("图像生成", "Image Generate")
RUNNER = "ImageGenerateRunner"

# Group labels (zh/en)
GROUP_LABELS = {
    "General Options": I18nText("通用选项", "General Options"),
    "Request": I18nText("请求", "Request"),
    "Data Type": I18nText("数据类型", "Data Type"),
    "Model Source": I18nText("模型源", "Model Source"),
    "Model & Quantization Options": I18nText("模型与量化", "Model & Quantization"),
    "Quantization Options": I18nText("量化", "Quantization"),
    "Optimization Options": I18nText("优化", "Optimization"),
    "Performance Model": I18nText("性能模型", "Performance Model"),
    "Parallelism": I18nText("并行", "Parallelism"),
    "CFG": I18nText("CFG", "CFG"),
    "Cache": I18nText("缓存", "Cache"),
    "Debug": I18nText("调试", "Debug"),
}


# Option source registry (dynamic options from backend)
OPTION_SOURCE_REGISTRY = {
    "devices": {"endpoint": "/api/options/devices", "cache": "session"},
}

# Cross-field validator UI mappings
VALIDATOR_UI = {
    "cfgParallelRequiresUseCfg": {
        "fields": ["cfg-parallel", "use-cfg"],
        "message": I18nText(
            "启用 cfg-parallel 时必须同时启用 use-cfg",
            "cfg-parallel requires use-cfg to be enabled",
        ),
    },
    "numDevicesMatchesUlyssesSize": {
        "fields": ["num-devices", "ulysses-size", "cfg-parallel"],
        "message": I18nText(
            "num-devices 必须等于 ulysses-size，启用 cfg-parallel 时须等于 2 × ulysses-size",
            "num-devices must equal ulysses-size, or 2 × ulysses-size with cfg-parallel",
        ),
    },
    "cfgParallelRequiresWorldSize2": {
        "fields": ["cfg-parallel", "num-devices"],
        "message": I18nText(
            "启用 cfg-parallel 时 num-devices 必须 ≥ 2",
            "cfg-parallel requires num-devices ≥ 2",
        ),
    },
}

# UI field properties (labels, tooltips, validation, etc.)
UI: dict[str, UIFieldProps] = {
    # Request
    "model-id": UIFieldProps(
        label=I18nText("模型 ID", "Model ID"),
        tooltip=I18nText(
            "待仿真模型的标准 HuggingFace 名称（组织/模型，如 black-forest-labs/FLUX.1-dev）或本地路径。",
            "Standard HuggingFace model name (org/model, e.g. black-forest-labs/FLUX.1-dev) or local path.",
        ),
        placeholder=I18nText("如 black-forest-labs/FLUX.1-dev", "e.g. black-forest-labs/FLUX.1-dev"),
        default="black-forest-labs/FLUX.1-dev",
        required=True,
    ),
    "device": UIFieldProps(
        label=I18nText("设备类型", "Target Device"),
        tooltip=I18nText(
            "选择用于仿真的设备 Profile。",
            "Device profile to simulate on.",
        ),
        placeholder=I18nText("请选择设备型号", "Select device model"),
        default="TEST_DEVICE",
        required=True,
        control="select",
        option_source={"type": "dynamic", "name": "devices"},
    ),
    "batch-size": UIFieldProps(
        label=I18nText("批大小", "Batch Size"),
        tooltip=I18nText(
            "基础工作负载批大小（≥1），非提示词或源图像数量。",
            "Base workload batch size (≥1), not prompt or source-image count.",
        ),
        default=1,
        required=True,
    ),
    "output-image-size": UIFieldProps(
        label=I18nText("输出图像尺寸", "Output Image Size"),
        tooltip=I18nText(
            "输出图像尺寸（高度 宽度）。仅提供一次，用于推导形状。",
            "Output image size (height width). Provide exactly once. Used only to derive shapes.",
        ),
        placeholder=I18nText("如 512 512", "e.g. 512 512"),
        default=[512, 512],
        required=True,
    ),
    "text-seq-len": UIFieldProps(
        label=I18nText("文本序列长度", "Text Sequence Length"),
        tooltip=I18nText(
            "进入 Transformer 的文本条件长度（≥1）。不执行编码。",
            "Text condition length that enters the Transformer (≥1). Encoding is not executed.",
        ),
        default=512,
        required=True,
    ),
    "source-image-size": UIFieldProps(
        label=I18nText("源图像尺寸", "Source Image Size"),
        tooltip=I18nText(
            "源图像尺寸（高度 宽度）。可重复。仅编辑类型使用。",
            "Source image size (height width). Repeatable. Editing kinds only.",
        ),
        placeholder=I18nText("如 512 512", "e.g. 512 512"),
        default=[],
    ),
    "sample-step": UIFieldProps(
        label=I18nText("采样步数", "Sample Steps"),
        tooltip=I18nText(
            "相同 Transformer 工作负载迭代次数（≥1）。",
            "Number of identical Transformer workload iterations (≥1).",
        ),
        default=1,
        required=True,
    ),
    # Data Type
    "dtype": UIFieldProps(
        label=I18nText("数据类型", "Data Type"),
        tooltip=I18nText(
            "计算数据类型。",
            "Computation data type.",
        ),
        default="float16",
        control="select",
        required=True,
        choices=[],  # Suppress ModuleSpec choices; use option_source only
        option_source={
            "type": "inline",
            "values": [
                {"value": "float16", "label": {"zh": "float16", "en": "float16"}},
                {"value": "bfloat16", "label": {"zh": "bfloat16", "en": "bfloat16"}},
                {"value": "float32", "label": {"zh": "float32", "en": "float32"}},
            ],
        },
    ),
    # Model Source
    "remote-source": UIFieldProps(
        label=I18nText("远程源", "Remote Source"),
        tooltip=I18nText(
            "非本地 Diffusers 仓库 ID 的远程源。",
            "Remote source for non-local Diffusers repo ids.",
        ),
        default="huggingface",
        control="select",
        choices=[],  # Suppress ModuleSpec choices; use option_source only
        option_source={
            "type": "inline",
            "values": [
                {"value": "huggingface", "label": {"zh": "HuggingFace", "en": "HuggingFace"}},
                {"value": "modelscope", "label": {"zh": "ModelScope", "en": "ModelScope"}},
            ],
        },
    ),
    # Quantization
    "quantize-linear-action": UIFieldProps(
        label=I18nText("线性层量化", "Linear-Layer Quantization"),
        tooltip=I18nText(
            "线性层的量化策略。",
            "Quantization action for linear layers.",
        ),
        default="DISABLED",
        control="select",
        required=True,
        choices=[],  # Suppress ModuleSpec choices; use option_source only
        option_source={
            "type": "inline",
            "values": [
                {"value": "DISABLED", "label": "DISABLED"},
                {"value": "W8A16_STATIC", "label": "W8A16_STATIC"},
                {"value": "W8A8_STATIC", "label": "W8A8_STATIC"},
                {"value": "W4A8_STATIC", "label": "W4A8_STATIC"},
                {"value": "W8A16_DYNAMIC", "label": "W8A16_DYNAMIC"},
                {"value": "W8A8_DYNAMIC", "label": "W8A8_DYNAMIC"},
                {"value": "W4A8_DYNAMIC", "label": "W4A8_DYNAMIC"},
                {"value": "FP8", "label": "FP8"},
                {"value": "MXFP4", "label": "MXFP4"},
            ],
        },
    ),
    "mxfp4-group-size": UIFieldProps(
        label=I18nText("MXFP4 组大小", "MXFP4 Group Size"),
        tooltip=I18nText(
            "MXFP4 量化的组大小（≥1）。",
            "Group size for MXFP4 quantization (≥1).",
        ),
        default=32,
        required=True,
    ),
    "quantize-attention-action": UIFieldProps(
        label=I18nText("注意力量化", "Attention Quantization"),
        tooltip=I18nText(
            "注意力计算的量化策略。",
            "Quantization action for attention computation.",
        ),
        default="DISABLED",
        control="select",
        required=True,
        choices=[],  # Suppress ModuleSpec choices; use option_source only
        option_source={
            "type": "inline",
            "values": [
                {"value": "DISABLED", "label": "DISABLED"},
                {"value": "FP8", "label": "FP8"},
            ],
        },
    ),
    # Optimization
    "compile": UIFieldProps(
        label=I18nText("编译", "Compile"),
        tooltip=I18nText(
            "在仿真前编译 Transformer。",
            "Compile the transformer before simulation.",
        ),
        default=False,
    ),
    "compile-allow-graph-break": UIFieldProps(
        label=I18nText("允许图中断", "Allow Graph Break"),
        tooltip=I18nText(
            "允许 torch.compile() 期间的图中断。",
            "Allow graph breaks during torch.compile().",
        ),
        default=False,
    ),
    # Parallelism
    "num-devices": UIFieldProps(
        label=I18nText("设备数量", "Number of Devices"),
        tooltip=I18nText(
            "设备数量（≥1）。必须等于 --ulysses-size，或在启用 --cfg-parallel 时等于 2 * --ulysses-size。",
            "Number of devices (≥1). Must equal --ulysses-size, or 2 * --ulysses-size with --cfg-parallel.",
        ),
        default=1,
        required=True,
    ),
    "ulysses-size": UIFieldProps(
        label=I18nText("Ulysses 并行", "Ulysses Parallel Size"),
        tooltip=I18nText(
            "Ulysses 序列并行度数（≥1）。",
            "Ulysses sequence-parallel size (≥1).",
        ),
        default=1,
        required=True,
    ),
    "cfg-parallel": UIFieldProps(
        label=I18nText("CFG 并行", "CFG Parallel"),
        tooltip=I18nText(
            "启用 CFG 并行。需要启用 --use-cfg。",
            "Enable CFG parallelism. Requires --use-cfg.",
        ),
        default=False,
    ),
    # CFG
    "use-cfg": UIFieldProps(
        label=I18nText("启用 CFG", "Enable CFG"),
        tooltip=I18nText(
            "启用分类器自由引导工作负载近似。",
            "Enable classifier-free guidance workload approximation.",
        ),
        default=False,
    ),
    # Cache
    "dit-cache": UIFieldProps(
        label=I18nText("启用 DiT 块缓存", "Enable DiT Block Cache"),
        tooltip=I18nText(
            "启用 DiT 块缓存以加速推理。",
            "Enable DiT block cache to speed up inference.",
        ),
        default=False,
    ),
    "cache-step-range": UIFieldProps(
        label=I18nText("缓存步区间", "Cache Step Range"),
        tooltip=I18nText(
            "缓存步范围（格式：start,end，包含两端）。当 interval > 1 时与 --dit-cache 一起使用必填。",
            "Cache step range 'start,end' (inclusive). Required with --dit-cache when interval > 1.",
        ),
        default=None,
        required=True,
        conditions={"enabled": {"field": "dit-cache", "op": "isTrue"}},
    ),
    "cache-step-interval": UIFieldProps(
        label=I18nText("缓存更新间隔", "Cache Update Interval"),
        tooltip=I18nText(
            "每 N 步更新缓存（1 禁用重用）。",
            "Update cache every N steps (1 disables reuse).",
        ),
        default=1,
        required=True,
        conditions={"enabled": {"field": "dit-cache", "op": "isTrue"}},
    ),
    "cache-block-range": UIFieldProps(
        label=I18nText("缓存块范围", "Cache Block Range"),
        tooltip=I18nText(
            "缓存块范围（格式：start,end，start 包含，end 排除）。",
            "Cache block range 'start,end' (start inclusive, end exclusive).",
        ),
        default=None,
        conditions={"enabled": {"field": "dit-cache", "op": "isTrue"}},
    ),
    # Debug
    "chrome-trace-file": UIFieldProps(
        label=I18nText("Chrome trace 导出", "Chrome Trace Export"),
        tooltip=I18nText(
            "导出 Chrome trace JSON 文件。",
            "Write chrome trace JSON file.",
        ),
        default=None,
    ),
    "log-level": LOG_LEVEL_UI_FULL.override(
        tooltip=I18nText("日志输出级别。", "Log output level."),
        choices=[],  # Suppress ModuleSpec choices; use option_source only
    ),
}


# Assemble FormConfig
spec = get_spec("image_generate")
config = FormConfig(
    version=VERSION,
    title=TITLE,
    runner=RUNNER,
    ui=UI,
    group_labels=GROUP_LABELS,
    option_source_registry=OPTION_SOURCE_REGISTRY,
    validator_ui=VALIDATOR_UI,
)


def get_form_config() -> tuple:
    """Get the ModuleSpec and FormConfig for image_generate module."""
    return spec, config


def generate_form_json() -> dict:
    """Generate form JSON for image_generate."""
    return export_form_json(spec, config)


if __name__ == "__main__":
    import json

    form_json = generate_form_json()
    print(json.dumps(form_json, indent=2, ensure_ascii=False))

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

"""video_generate UI field properties for Web UI form generation.

This module defines UI-specific properties (labels, tooltips, validation, grouping)
for video_generate parameters. Combined with the ModuleSpec from
cli/registry/modules/video_generate.py, this generates the form JSON.
"""

from web_ui.backend.services.ui_props import I18nText, UIFieldProps, LOG_LEVEL_UI, VIDEO_MODEL_ID_OPTIONS
from cli.registry.modules import get_spec
from web_ui.backend.services.json_adapter import FormConfig, export_form_json


VERSION = "2.6.9"
TITLE = I18nText("视频生成", "Video Generate")
RUNNER = "VideoGenerateRunner"

# Group labels (zh/en)
GROUP_LABELS = {
    "General": I18nText("通用", "General"),
    "General Options": I18nText("通用", "General Options"),  # CLI canonical alias
    "Input": I18nText("输入", "Input"),
    "Video": I18nText("视频", "Video"),
    "Image": I18nText("图像", "Image"),
    "CFG": I18nText("CFG", "CFG"),
    "Cache": I18nText("缓存", "Cache"),
    "Debug": I18nText("调试", "Debug"),
    "Parallelism": I18nText("并行", "Parallelism"),
    "Quantization": I18nText("量化", "Quantization"),
    "Quantization Options": I18nText("量化", "Quantization Options"),  # CLI canonical alias
    "Model Source": I18nText("模型来源", "Model Source"),  # CLI canonical
    "Attention Options": I18nText("注意力", "Attention Options"),  # CLI canonical
    "Model & Quantization Options": I18nText("模型与量化", "Model & Quantization"),  # CLI canonical
    "Optimization Options": I18nText("优化", "Optimization Options"),  # CLI canonical alias
    "Request": I18nText("请求", "Request"),
}

# ── Group collapse metadata ────────────────────────────────────────
GROUPS = [
    {"group_key": "Image", "defaultCollapsed": True},
    {"group_key": "CFG", "defaultCollapsed": True},
    {"group_key": "Cache", "defaultCollapsed": True},
    {"group_key": "Debug", "defaultCollapsed": True},
]


# Option source registry (dynamic options from backend)
OPTION_SOURCE_REGISTRY = {
    "devices": {"endpoint": "/api/options/devices", "cache": "session"},
}

# UI field properties (labels, tooltips, validation, etc.)
VALIDATOR_UI = {
    "cfgParallelRequiresWorldSize2": {
        "fields": ["use-cfg", "cfg-parallel", "num-devices"],
        "message": I18nText(
            "启用 CFG 并行时 world_size 必须 ≥ 2",
            "cfg_parallel requires world_size ≥ 2",
        ),
    },
}

UI: dict[str, UIFieldProps] = {
    "model-id": UIFieldProps(
        label=I18nText('模型 ID', 'Model ID'),
        tooltip=I18nText(
            '待仿真模型的标准 HuggingFace 名称（组织/模型，如 Wan-AI/Wan2.1-T2V-14B-Diffusers）或本地路径。',
            'Standard HuggingFace model name (org/model, e.g. Wan-AI/Wan2.1-T2V-14B-Diffusers) or local path.',
        ),
        placeholder=I18nText('如 Wan-AI/Wan2.1-T2V-14B-Diffusers', 'e.g. Wan-AI/Wan2.1-T2V-14B-Diffusers'),
        default='Wan-AI/Wan2.1-T2V-14B-Diffusers',
        required=True,
        group={'zh': '通用', 'en': 'General'},
        control='combobox',
        option_source={'type': 'inline', 'values': VIDEO_MODEL_ID_OPTIONS},
    ),
    "device": UIFieldProps(
        label=I18nText('设备类型', 'Target Device'),
        tooltip=I18nText(
            '选择用于仿真的设备 Profile（可多选；每个取值单独仿真，结果区生成多用例对比）。',
            'Device profile(s) to simulate on (multi-select; each value runs independently and yields a multi-case comparison).',
        ),
        placeholder=I18nText('请选择设备型号', 'Select device model(s)'),
        option_source={'type': 'dynamic', 'name': 'devices'},
        multi_values=True,
        data_type='string[]',
        default=['ATLAS_350_425T_112G'],
        required=True,
        group={'zh': '通用', 'en': 'General'},
        control='multi-select',
    ),
    "num-devices": UIFieldProps(
        label=I18nText('设备数', 'World Size'),
        tooltip=I18nText('总设备数量（≥1）。', 'Total device count (≥1).'),
        default=8,
        required=True,
        group={'zh': '并行', 'en': 'Parallelism'},
        control='number',
    ),
    "reserved-memory-gb": UIFieldProps(
        label=I18nText('预留显存(GB)', 'Reserved Memory (GB)'),
        tooltip=I18nText('单卡为其他进程预留的显存(GB)。', 'Per-device memory reserved for other processes (GB).'),
        hidden=True,
        default=0.0,
        required=True,
    ),
    "log-level": LOG_LEVEL_UI.override(
        tooltip=I18nText('日志输出级别。', 'Log output level.'),
        default='info',
        group={'zh': '调试', 'en': 'Debug'},
    ),
    "batch-size": UIFieldProps(
        label=I18nText('批大小', 'Batch Size'),
        tooltip=I18nText('批大小（≥1）。', 'Batch size (≥1).'),
        default=1,
        required=True,
        group={'zh': '输入', 'en': 'Input'},
        control='number',
    ),
    "seq-len": UIFieldProps(
        label=I18nText('文本序列长度', 'Text Sequence Length'),
        tooltip=I18nText('文本序列长度（≥1）。', 'Text sequence length (≥1).'),
        default=128,
        required=True,
        group={'zh': '输入', 'en': 'Input'},
        control='number',
    ),
    "height": UIFieldProps(
        label=I18nText('图像高度', 'Image Height'),
        tooltip=I18nText('生成图像的高度（像素）。', 'Generated image height (pixels).'),
        default=1280,
        group={'zh': '图像', 'en': 'Image'},
        control='number',
    ),
    "width": UIFieldProps(
        label=I18nText('图像宽度', 'Image Width'),
        tooltip=I18nText('生成图像的宽度（像素）。', 'Generated image width (pixels).'),
        default=720,
        group={'zh': '图像', 'en': 'Image'},
        control='number',
    ),
    "frame-num": UIFieldProps(
        label=I18nText('视频帧数', 'Frame Count'),
        tooltip=I18nText('生成的视频帧数。', 'Number of video frames to generate.'),
        default=129,
        group={'zh': '视频', 'en': 'Video'},
        control='number',
    ),
    "sample-step": UIFieldProps(
        label=I18nText('采样步数', 'Sample Steps'),
        tooltip=I18nText('DDPM 采样步数。', 'DDPM sampling steps.'),
        default=50,
        group={'zh': '视频', 'en': 'Video'},
        control='number',
    ),
    "dtype": UIFieldProps(
        label=I18nText('数据类型', 'Data Type'),
        tooltip=I18nText('计算数据类型。', 'Computation data type.'),
        option_source={
            'type': 'inline',
            'values': [
                {'value': 'float16', 'label': 'Float16'},
                {'value': 'bfloat16', 'label': 'BFloat16'},
                {'value': 'float32', 'label': 'Float32'},
            ],
        },
        default='float16',
        group={'zh': '通用', 'en': 'General'},
        control='select',
    ),
    "quantize-linear-action": UIFieldProps(
        label=I18nText('线性层量化', 'Linear-Layer Quantization'),
        tooltip=I18nText(
            '线性层的量化策略（可多选；每个取值单独仿真，结果区生成多用例对比）。',
            'Quantization action(s) for linear layers (multi-select; each value runs independently and yields a multi-case comparison).',
        ),
        option_source={
            'type': 'inline',
            'values': [
                {'value': 'DISABLED', 'label': 'DISABLED'},
                {'value': 'W8A16_STATIC', 'label': 'W8A16_STATIC'},
                {'value': 'W8A8_STATIC', 'label': 'W8A8_STATIC'},
                {'value': 'W4A8_STATIC', 'label': 'W4A8_STATIC'},
                {'value': 'W8A16_DYNAMIC', 'label': 'W8A16_DYNAMIC'},
                {'value': 'W8A8_DYNAMIC', 'label': 'W8A8_DYNAMIC'},
                {'value': 'W4A8_DYNAMIC', 'label': 'W4A8_DYNAMIC'},
                {'value': 'FP8', 'label': 'FP8'},
                {'value': 'MXFP4', 'label': 'MXFP4'},
            ],
        },
        multi_values=True,
        data_type='string[]',
        default=['W8A8_DYNAMIC'],
        group={'zh': '量化', 'en': 'Quantization'},
        control='multi-select',
    ),
    "ulysses-size": UIFieldProps(
        label=I18nText('Ulysses 并行', 'Ulysses Parallel Size'),
        tooltip=I18nText(
            'Ulysses 序列并行度数；逗号分隔多值时每个取值单独仿真，结果区生成多用例对比（如 1,2,4，须整除 world_size）。',
            'Ulysses SP degree; with a comma-list each value runs independently and yields a multi-case comparison (e.g. 1,2,4; must divide world_size).',
        ),
        placeholder=I18nText('如 1,2,4', 'e.g. 1,2,4'),
        multi_values=True,
        default=4,
        group={'zh': '并行', 'en': 'Parallelism'},
        control='text',
    ),
    "dit-cache": UIFieldProps(
        label=I18nText('启用 DiT 块缓存', 'Enable DiT Block Cache'),
        tooltip=I18nText('启用 DiT 块缓存以加速推理。', 'Enable DiT block cache to speed up inference.'),
        default=True,
        group={'zh': '缓存', 'en': 'Cache'},
        control='switch',
    ),
    "cache-step-range": UIFieldProps(
        label=I18nText('缓存步区间', 'Cache Step Range'),
        tooltip=I18nText(
            '缓存步范围（格式：start,end，end≥start）。此字段仅在 dit_cache 启用时有效且必填',
            'Cache step range (format: start,end, end≥start). This field is only effective and required when dit_cache is enabled',
        ),
        default='20,30',
        group={'zh': '缓存', 'en': 'Cache'},
        control='text',
        conditions={"enabled": {"field": "dit-cache", "op": "isTrue"}},
    ),
    "cache-step-interval": UIFieldProps(
        label=I18nText('缓存更新间隔', 'Cache Update Interval'),
        tooltip=I18nText(
            '缓存更新间隔（1 表示禁用）。此字段仅在 dit_cache 启用时有效',
            'Cache update interval (1 disables caching). This field is only effective when dit_cache is enabled',
        ),
        default=5,
        group={'zh': '缓存', 'en': 'Cache'},
        control='number',
        conditions={"enabled": {"field": "dit-cache", "op": "isTrue"}},
    ),
    "cache-block-range": UIFieldProps(
        label=I18nText('缓存块范围', 'Cache Block Range'),
        tooltip=I18nText(
            '缓存块范围（格式：start,end）。此字段仅在 dit_cache 启用时有效',
            'Cache block range (format: start,end). This field is only effective when dit_cache is enabled',
        ),
        group={'zh': '缓存', 'en': 'Cache'},
        control='text',
    ),
    "attention-backend": UIFieldProps(
        label=I18nText('注意力后端', 'Attention Backend'),
        tooltip=I18nText('注意力计算后端。', 'Attention computation backend.'),
        option_source={
            'type': 'inline',
            'values': [
                {'value': 'default', 'label': 'Default'},
                {'value': 'block_sparse_attention', 'label': 'Block Sparse Attention'},
            ],
        },
        hidden=True,
        control='select',
    ),
    "attention-block-size": UIFieldProps(
        label=I18nText('注意力块大小', 'Attention Block Size'),
        tooltip=I18nText('Block sparse attention 的块大小。', 'Block size for block sparse attention.'),
        hidden=True,
    ),
    "attention-sparsity": UIFieldProps(
        label=I18nText('注意力稀疏度', 'Attention Sparsity'),
        tooltip=I18nText('Block sparse attention 的稀疏度(0-1)。', 'Sparsity for block sparse attention (0-1).'),
        hidden=True,
    ),
    "remote-source": UIFieldProps(hidden=True),
    "compile": UIFieldProps(hidden=True),
    "compile-allow-graph-break": UIFieldProps(hidden=True),
    "prefix-cache-hit-rate": UIFieldProps(hidden=True),
    "quantize-attention-action": UIFieldProps(hidden=True),
    "use-cfg": UIFieldProps(
        label=I18nText('启用 CFG', 'Enable CFG'),
        tooltip=I18nText('启用 Classifier-Free Guidance。', 'Enable Classifier-Free Guidance.'),
        default=True,
        group={'zh': 'CFG', 'en': 'CFG'},
        control='switch',
    ),
    "cfg-parallel": UIFieldProps(
        label=I18nText('CFG 并行', 'CFG Parallel'),
        tooltip=I18nText(
            'CFG 使用并行计算（隐式要求 world_size ≥ 2）。此字段仅在 use_cfg 启用时有效',
            'Use parallel computation for CFG (implicitly requires world_size ≥ 2). This field is only effective when use_cfg is enabled',
        ),
        default=True,
        group={'zh': 'CFG', 'en': 'CFG'},
        control='switch',
    ),
    "chrome-trace-file": UIFieldProps(
        label=I18nText('Chrome trace 导出', 'Chrome Trace Export'),
        tooltip=I18nText(
            '开启后每个用例导出 Chrome trace，完成后可在结果页下载。',
            'Export a Chrome trace per case; downloadable from the result page.',
        ),
        data_type='boolean',
        default=False,
        group={'zh': '调试', 'en': 'Debug'},
        control='switch',
    ),
    "dump-input-shapes": UIFieldProps(hidden=True),
    "performance-model": UIFieldProps(hidden=True),
    "profiling-database-path": UIFieldProps(hidden=True),
}


def get_form_config() -> tuple:
    """Get the ModuleSpec and FormConfig for video_generate module."""
    spec = get_spec("video_generate")
    config = FormConfig(
        version=VERSION,
        title=TITLE,
        runner=RUNNER,
        ui=UI,
        group_labels=GROUP_LABELS,
        groups=GROUPS,
        option_source_registry=OPTION_SOURCE_REGISTRY,
        validator_ui=VALIDATOR_UI,
    )
    return spec, config


def generate_form_json() -> dict:
    """Generate form JSON for video_generate."""
    spec, config = get_form_config()
    return export_form_json(spec, config)


if __name__ == "__main__":
    import json

    form_json = generate_form_json()
    print(json.dumps(form_json, indent=2, ensure_ascii=False))

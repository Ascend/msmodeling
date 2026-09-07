"""throughput_optimizer UI field properties for Web UI form generation.

This module defines UI-specific properties (labels, tooltips, validation, grouping)
for throughput_optimizer parameters. Combined with the ModuleSpec from
cli/registry/modules/throughput_optimizer.py, this generates the form JSON.
"""

from web_ui.backend.services.ui_props import I18nText, UIFieldProps, LOG_LEVEL_UI, MODEL_ID_OPTIONS
from cli.registry.modules import get_spec
from web_ui.backend.services.json_adapter import FormConfig, export_form_json


VERSION = "2.9.12"
TITLE = I18nText("吞吐优化", "Throughput Optimizer")
RUNNER = "ParallelRunner"

# Group labels (zh/en)
GROUP_LABELS = {
    "General": I18nText("通用", "General"),
    "Input": I18nText("输入", "Input"),
    "Output": I18nText("输出", "Output"),
    "Model": I18nText("模型", "Model"),
    "Model & Quantization Options": I18nText("模型与量化", "Model & Quantization"),  # CLI canonical
    "MTP": I18nText("MTP", "MTP"),
    "Cache": I18nText("缓存", "Cache"),
    "Cost": I18nText("成本", "Cost"),
    "Mode": I18nText("模式", "Mode"),
    "Execution": I18nText("执行", "Execution"),
    "PD Ratio": I18nText("PD 配比", "PD Ratio"),
    "Multimodal": I18nText("多模态", "Multimodal"),
    "Constraints": I18nText("约束", "Constraints"),
    "Debug": I18nText("调试", "Debug"),
    "Optimization": I18nText("优化", "Optimization"),
    "Optimization Options": I18nText("优化", "Optimization Options"),  # CLI canonical alias
    "Quantization": I18nText("量化", "Quantization"),
    "Quantization Options": I18nText("量化", "Quantization Options"),  # CLI canonical alias
    "Search": I18nText("搜索", "Search"),
    "Performance Model": I18nText("性能模型", "Performance Model"),
    "Performance Model Options": I18nText("性能模型", "Performance Model Options"),  # CLI canonical
    "Model Source": I18nText("模型来源", "Model Source"),  # CLI canonical
    "Request": I18nText("请求", "Request"),
}

# ── Group collapse metadata ────────────────────────────────────────
GROUPS = [
    {"group_key": "MTP", "defaultCollapsed": True},
    {"group_key": "Cache", "defaultCollapsed": True},
    {"group_key": "Cost", "defaultCollapsed": True},
    {"group_key": "Mode", "defaultCollapsed": True},
    {"group_key": "Execution", "defaultCollapsed": True},
    {"group_key": "Output", "defaultCollapsed": True},
    {"group_key": "Model", "defaultCollapsed": True},
    {"group_key": "PD Ratio", "defaultCollapsed": True},
    {"group_key": "Multimodal", "defaultCollapsed": True},
]


# Option source registry (dynamic options from backend)
OPTION_SOURCE_REGISTRY = {
    "devices": {"endpoint": "/api/options/devices", "cache": "session"},
}

# Cross-field validators (UI metadata)
VALIDATOR_UI = {
    "validParallelCombo": {
        "fields": ["tp-sizes", "ep-sizes", "moe-dp-sizes", "num-devices"],
        "message": I18nText(
            "在当前 num_devices 下,不存在有效的 (tp, ep, moe_dp) 组合",
            "No valid (tp, ep, moe_dp) combination exists under current num_devices",
        ),
    },
    "effectiveLenGe1": {
        "fields": ["input-length", "prefix-cache-hit-rate"],
        "message": I18nText(
            "有效输入长度(扣除前缀缓存后)必须 ≥ 1",
            "Effective input length (after prefix cache) must be ≥ 1",
        ),
    },
    "pdRatioMutexDisagg": {
        "fields": ["enable-optimize-prefill-decode-ratio", "disagg"],
        "message": I18nText(
            "PD 配比优化不能与分离部署模式(disagg)同时使用",
            "PD-ratio optimization cannot be used together with disagg",
        ),
    },
    "pdRatioRequiresDevices": {
        "fields": [
            "enable-optimize-prefill-decode-ratio",
            "prefill-devices-per-instance",
            "decode-devices-per-instance",
        ],
        "message": I18nText(
            "启用 PD 配比优化时，必须指定 prefill-devices-per-instance 和 decode-devices-per-instance",
            "Both prefill-devices-per-instance and decode-devices-per-instance are required when PD ratio optimization is enabled",
        ),
    },
    "mtpTokensVsAcceptanceRate": {
        "fields": ["num-mtp-tokens", "mtp-acceptance-rates"],
        "message": I18nText(
            "num_mtp_tokens 不得超过 mtp_acceptance_rate 列表长度 + 1",
            "num_mtp_tokens must be ≤ mtp_acceptance_rate list length + 1",
        ),
    },
    "draftDependentsRequireMethod": {
        "fields": ["speculative-method"],
        "message": I18nText(
            "草稿依赖参数需要先指定 speculative-method（dflash/dspark）",
            "Draft-dependent options require --speculative-method (dflash or dspark) first",
        ),
    },
    "draftMtpMutex": {
        "fields": ["speculative-method", "num-mtp-tokens"],
        "message": I18nText(
            "DFlash/DSpark 不能与 MTP 同时启用（两者是互斥的投机解码方法）",
            "DFlash/DSpark and MTP are mutually exclusive speculative decoding methods",
        ),
    },
    "draftLegacyMtpMutex": {
        "fields": ["speculative-method", "num-mtp-tokens"],
        "message": I18nText(
            "使用统一的 --speculative-method 时，不能再传旧版 --num-mtp-tokens；请改用 --num-speculative-tokens",
            "--num-mtp-tokens is the legacy entry; use --num-speculative-tokens together with --speculative-method instead",
        ),
    },
    "mtpRequiresNumSpeculativeTokens": {
        "fields": ["speculative-method", "num-speculative-tokens"],
        "message": I18nText(
            "--speculative-method=mtp 必须配合 --num-speculative-tokens 使用",
            "--speculative-method=mtp requires --num-speculative-tokens",
        ),
    },
    "mtpNoDraftLayers": {
        "fields": ["speculative-method", "num-draft-layers"],
        "message": I18nText(
            "--speculative-method=mtp 不能使用 --num-draft-layers（这是 draft 专用选项）",
            "--speculative-method=mtp cannot use --num-draft-layers (draft-specific option)",
        ),
    },
    "speculativeTokensPositive": {
        "fields": ["speculative-method", "num-speculative-tokens"],
        "message": I18nText(
            "使用 --speculative-method 时，--num-speculative-tokens 必须 > 0",
            "--num-speculative-tokens must be > 0 when --speculative-method is set",
        ),
    },
    "mtpNoLegacyAcceptanceRate": {
        "fields": ["speculative-method", "mtp-acceptance-rates"],
        "message": I18nText(
            "--speculative-method=mtp 不能使用旧版 --mtp-acceptance-rate(s)",
            "--speculative-method=mtp cannot use legacy --mtp-acceptance-rate(s)",
        ),
    },
    "legacyMtpNoAcceptanceLength": {
        "fields": ["num-mtp-tokens", "acceptance-length"],
        "message": I18nText(
            "旧版 --num-mtp-tokens 不能配合 --acceptance-length 使用",
            "Legacy --num-mtp-tokens cannot be used together with --acceptance-length",
        ),
    },
}


# UI field properties (labels, tooltips, validation, etc.)
UI: dict[str, UIFieldProps] = {
    "model-id": UIFieldProps(
        label=I18nText('模型 ID', 'Model ID'),
        tooltip=I18nText(
            '待仿真模型的标准 HuggingFace 名称（组织/模型，如 Qwen/Qwen3-32B）或本地路径。',
            'Standard HuggingFace model name (org/model, e.g. Qwen/Qwen3-32B) or local path.',
        ),
        placeholder=I18nText('如 Qwen/Qwen3-32B', 'e.g. Qwen/Qwen3-32B'),
        default='Qwen/Qwen3-32B',
        required=True,
        group={'zh': '通用', 'en': 'General'},
        control='combobox',
        option_source={'type': 'inline', 'values': MODEL_ID_OPTIONS},
    ),
    "device": UIFieldProps(
        label=I18nText('设备类型', 'Target Device'),
        tooltip=I18nText(
            '选择用于仿真的设备 Profile（可多选；每个取值单独仿真，结果区生成多用例对比）。',
            'Device profile(s) to simulate on (multi-select; each value runs independently and yields a multi-case comparison).',
        ),
        placeholder=I18nText('请选择设备型号', 'Select device models'),
        option_source={'type': 'dynamic', 'name': 'devices'},
        multi_values=True,
        data_type='string[]',
        default=['ATLAS_350_425T_112G'],
        required=True,
        group={'zh': '通用', 'en': 'General'},
        control='multi-select',
    ),
    "num-devices": UIFieldProps(
        label=I18nText('设备数量', 'Number of Devices'),
        tooltip=I18nText('参与仿真的 Die 数量（≥1）。', 'Number of devices used in the simulation (≥1).'),
        default=4,
        required=True,
        group={'zh': '通用', 'en': 'General'},
        control='number',
    ),
    "reserved-memory-gb": UIFieldProps(
        label=I18nText('预留显存(GB)', 'Reserved Memory (GB)'),
        tooltip=I18nText('单卡为其他进程预留的显存（GB）。', 'Per-device memory reserved for other processes (GB).'),
        default=10.0,
        required=True,
        group={'zh': '通用', 'en': 'General'},
        control='number',
    ),
    "log-level": LOG_LEVEL_UI.override(
        tooltip=I18nText('日志输出级别。', 'Log output level.'),
        group={'zh': '通用', 'en': 'General'},
    ),
    "input-length": UIFieldProps(
        label=I18nText('输入(prompt)长度', 'Input Prompt Length'),
        tooltip=I18nText('输入 prompt 的 token 长度（≥1，<1e6）。', 'Input prompt token length (≥1, <1e6).'),
        default='3500',
        required=True,
        group={'zh': '输入', 'en': 'Input'},
        control='number',
    ),
    "output-length": UIFieldProps(
        label=I18nText('输出长度', 'Output Length'),
        tooltip=I18nText('输出 token 长度（≥1，<1e6）。', 'Output token length (≥1, <1e6).'),
        default=1500,
        required=True,
        group={'zh': '输入', 'en': 'Input'},
        control='number',
    ),
    "tpot-limit": UIFieldProps(
        label=I18nText('TPOT 约束(ms)', 'TPOT Limit (ms)'),
        tooltip=I18nText(
            'TPOT 约束（毫秒）；逗号分隔多值时每个取值单独仿真，结果区生成多用例对比；>0 或 inf 表示无约束。',
            'TPOT limit in milliseconds; with a comma-list each value runs independently and yields a multi-case comparison; >0 or inf for unlimited.',
        ),
        placeholder=I18nText('如 20,50 或留空', 'e.g. 20,50 or empty'),
        multi_values=True,
        default='',
        group={'zh': '约束', 'en': 'Constraints'},
        control='text',
    ),
    "ttft-limit": UIFieldProps(
        label=I18nText('TTFT 约束(ms)', 'TTFT Limit (ms)'),
        tooltip=I18nText(
            'TTFT 约束（毫秒）；逗号分隔多值时每个取值单独仿真，结果区生成多用例对比；>0 或 inf 表示无约束。',
            'TTFT limit in milliseconds; with a comma-list each value runs independently and yields a multi-case comparison; >0 or inf for unlimited.',
        ),
        placeholder=I18nText('如 200,500 或留空', 'e.g. 200,500 or empty'),
        multi_values=True,
        default='',
        group={'zh': '约束', 'en': 'Constraints'},
        control='text',
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
    "quantize-attention-action": UIFieldProps(
        label=I18nText('KV cache 量化', 'Attention KV Cache Quantization'),
        tooltip=I18nText(
            'Attention KV Cache 的量化策略（可多选；每个取值单独仿真，结果区生成多用例对比）。',
            'Quantization for attention KV cache (multi-select; each value runs independently and yields a multi-case comparison).',
        ),
        option_source={
            'type': 'inline',
            'values': [
                {'value': 'DISABLED', 'label': 'DISABLED'},
                {'value': 'INT8', 'label': 'INT8'},
                {'value': 'FP8', 'label': 'FP8'},
            ],
        },
        multi_values=True,
        data_type='string[]',
        default=['INT8'],
        group={'zh': '量化', 'en': 'Quantization'},
        control='multi-select',
    ),
    "compile": UIFieldProps(
        label=I18nText('启用 torch.compile', 'Enable torch.compile'),
        tooltip=I18nText(
            '默认启用 torch.compile 加速（已锁定，不可修改）。',
            'torch.compile acceleration is enabled by default (locked).',
        ),
        default=True,
        group={'zh': '优化', 'en': 'Optimization'},
        control='switch',
    ),
    "compile-allow-graph-break": UIFieldProps(hidden=True),
    "num-mtp-tokens": UIFieldProps(
        label=I18nText('MTP token 数', 'MTP Token Count'),
        tooltip=I18nText(
            'MTP token 数量（argparse 允许 0-9；运行时受 mtp_acceptance_rate 长度限制）。推荐使用"投机解码方法=mtp + 投机 token 数"统一入口。选定 speculative-method 后此字段自动禁用。',
            'MTP token count (argparse allows 0-9; runtime bounded by mtp_acceptance_rate length). Prefer the unified entry (speculative-method=mtp + num-speculative-tokens). Auto-disabled when speculative-method is set.',
        ),
        data_type='integer',
        default=0,
        option_source={
            'type': 'inline',
            'values': [{'value': i, 'label': str(i)} for i in range(10)],
        },
        group={'zh': 'MTP', 'en': 'MTP'},
        control='select',
        conditions={"enabled": {"field": "speculative-method", "op": "empty"}},
    ),
    "mtp-acceptance-rates": UIFieldProps(
        label=I18nText('MTP 接收率列表', 'MTP Acceptance Rate List'),
        tooltip=I18nText(
            'MTP 接收率列表。仅在 num_mtp_tokens > 0 时有效。选定 speculative-method 后此字段自动禁用。',
            'MTP acceptance rate list. Only effective when num_mtp_tokens > 0. Auto-disabled when speculative-method is set.',
        ),
        default='0.8, 0.6, 0.4, 0.2',
        group={'zh': 'MTP', 'en': 'MTP'},
        control='text',
        conditions={"enabled": {"field": "speculative-method", "op": "empty"}},
    ),
    "prefix-cache-hit-rate": UIFieldProps(
        label=I18nText('前缀缓存命中率', 'Prefix Cache Hit Rate'),
        tooltip=I18nText('模拟的前缀缓存命中率 [0, 1)。', 'Simulated prefix cache hit rate [0, 1).'),
        default=0.0,
        group={'zh': '缓存', 'en': 'Cache'},
        control='number',
    ),
    "quantize-non-expert-linear-action": UIFieldProps(
        label=I18nText('非专家线性层', 'Non-Expert Linear Quantization'),
        tooltip=I18nText(
            '为非专家线性层（如注意力投影、稠密MLP层）设置的单独量化类型',
            'Separate quantization type for non-expert linear layers (e.g. attention projection, dense MLP layers).',
        ),
        default='DISABLED',
        group={'zh': '量化', 'en': 'Quantization'},
        control='select',
    ),
    "mxfp4-group-size": UIFieldProps(
        label=I18nText('MXFP4 分组大小', 'MXFP4 Group Size'),
        tooltip=I18nText(
            'MXFP4 量化的分组大小（>0，≤1e6）。此字段仅在 quantize_linear_action 或 quantize_non_expert_linear_action 包含 MXFP4 时有效',
            'MXFP4 quantization group size (>0, ≤1e6). This field is only effective when quantize_linear_action or quantize_non_expert_linear_action contains MXFP4',
        ),
        default=32,
        group={'zh': '量化', 'en': 'Quantization'},
        control='number',
    ),
    "tp-sizes": UIFieldProps(
        label=I18nText('TP 搜索尺寸', 'TP Search Sizes'),
        tooltip=I18nText(
            '逗号分隔的张量并行搜索尺寸（如 1,2,4），每个 ≤ num_devices；每个尺寸都会评估并在结果扫描表中对比各组合；留空表示不搜索。',
            'Comma-separated TP search sizes (e.g. 1,2,4), each ≤ num_devices; every size is evaluated and compared in the result sweep table; empty = no search.',
        ),
        placeholder=I18nText('如 1,2,4', 'e.g. 1,2,4'),
        group={'zh': '搜索', 'en': 'Search'},
        control='text',
    ),
    "ep-sizes": UIFieldProps(
        label=I18nText('EP 搜索尺寸', 'EP Search Sizes'),
        tooltip=I18nText(
            '逗号分隔的专家并行搜索尺寸（如 1,2,4），每个 ≤ num_devices；仅对 MoE 模型生效，每个尺寸都会评估并在结果扫描表中对比各组合；留空表示不搜索。',
            'Comma-separated EP search sizes (e.g. 1,2,4), each ≤ num_devices; only effective for MoE models; every size is evaluated and compared in the result sweep table; empty = no search.',
        ),
        placeholder=I18nText('如 1,2,4', 'e.g. 1,2,4'),
        group={'zh': '搜索', 'en': 'Search'},
        control='text',
    ),
    "moe-dp-sizes": UIFieldProps(
        label=I18nText('MOE-DP 搜索尺寸', 'MOE-DP Search Sizes'),
        tooltip=I18nText(
            '逗号分隔的 MoE 数据并行搜索尺寸（如 1,2），每个 ≤ num_devices；仅对 MoE 模型生效，每个尺寸都会评估并在结果扫描表中对比各组合；留空表示不搜索。',
            'Comma-separated MoE-DP search sizes (e.g. 1,2), each ≤ num_devices; only effective for MoE models; every size is evaluated and compared in the result sweep table; empty = no search.',
        ),
        placeholder=I18nText('如 1,2', 'e.g. 1,2'),
        group={'zh': '搜索', 'en': 'Search'},
        control='text',
    ),
    "pp-sizes": UIFieldProps(
        label=I18nText('PP 搜索尺寸', 'PP Search Sizes'),
        tooltip=I18nText(
            '逗号分隔的流水线并行搜索尺寸（如 1,2,4）；留空表示默认搜索 2 的幂次直到 num_devices；仅 PP=1 时不启用流水线并行。',
            'Comma-separated pipeline-parallel search sizes (e.g. 1,2,4); empty = default to powers of 2 up to num_devices; PP=1 only means no pipeline parallelism.',
        ),
        placeholder=I18nText('如 1,2,4', 'e.g. 1,2,4'),
        group={'zh': '搜索', 'en': 'Search'},
        control='text',
    ),
    "pp-layer-partitions": UIFieldProps(
        label=I18nText('PP 层分配', 'PP Layer Partitions'),
        tooltip=I18nText(
            'JSON 格式的显式层分配（如 [[31,30],[16,15,15,15]]）；每个内部列表长度必须等于对应的 pp_size 且总和为 num_hidden_layers；留空使用默认均衡分配。',
            'Explicit layer partitions as JSON (e.g. [[31,30],[16,15,15,15]]); each inner list length must equal its pp_size and sum to num_hidden_layers; empty = balanced partition.',
        ),
        placeholder=I18nText('如 [[31,30],[16,15,15,15]]', 'e.g. [[31,30],[16,15,15,15]]'),
        group={'zh': '搜索', 'en': 'Search'},
        control='text',
    ),
    "dcp-sizes": UIFieldProps(hidden=True),
    "enable-shared-expert-tp": UIFieldProps(hidden=True),
    "compilation-config": UIFieldProps(
        label=I18nText('编译优化选项', 'Compilation Config'),
        tooltip=I18nText(
            '选择要启用的编译期优化（多选）。', 'Select compilation optimizations to enable (multi-select).'
        ),
        placeholder=I18nText('请选择编译优化选项', 'Select compilation options'),
        default=[],
        group={'zh': '优化', 'en': 'Optimization'},
        control='multi-select',
    ),
    "word-embedding-tp": UIFieldProps(hidden=True),
    "performance-model": UIFieldProps(hidden=True),
    "profiling-database-path": UIFieldProps(hidden=True),
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
    "max-batched-tokens": UIFieldProps(
        label=I18nText('单步最大 token 数', 'Max Batched Tokens'),
        tooltip=I18nText('单步最大 token 数（>0，≤1e6）。', 'Maximum tokens per step (>0, ≤1e6).'),
        default=None,
        group={'zh': '约束', 'en': 'Constraints'},
        control='number',
    ),
    "batch-range": UIFieldProps(
        label=I18nText('批大小范围', 'Batch Size Range'),
        tooltip=I18nText(
            '批大小范围，格式 start,end 或单值（min≤max）。',
            'Batch size range, format start,end or a single value (min≤max).',
        ),
        placeholder=I18nText('如 1,8', 'e.g. 1,8'),
        group={'zh': '约束', 'en': 'Constraints'},
        control='text',
    ),
    "serving-cost": UIFieldProps(
        label=I18nText('服务成本', 'Serving Cost'),
        tooltip=I18nText('服务成本（≥0）。', 'Serving cost (≥0).'),
        default=0.0,
        group={'zh': '成本', 'en': 'Cost'},
        control='number',
    ),
    "disagg": UIFieldProps(
        label=I18nText('分离部署模式', 'Disaggregated Mode'),
        tooltip=I18nText(
            '启用分离部署模式（Prefill+Decode 分离，与 PD 配比优化互斥）。',
            'Enable disaggregated mode (Prefill+Decode separation, mutually exclusive with PD-ratio optimization).',
        ),
        default=False,
        group={'zh': '模式', 'en': 'Mode'},
        control='switch',
        conditions={"enabled": {"not": {"field": "enable-optimize-prefill-decode-ratio", "op": "isTrue"}}},
    ),
    "jobs": UIFieldProps(
        label=I18nText('并行作业数', 'Parallel Jobs'),
        tooltip=I18nText('并行作业数（>0，≤1e6）。', 'Number of parallel jobs (>0, ≤1e6).'),
        default=8,
        group={'zh': '执行', 'en': 'Execution'},
        control='number',
    ),
    "max-search-combinations": UIFieldProps(
        label=I18nText('最大搜索组合数', 'Max Search Combinations'),
        tooltip=I18nText(
            '当 TP/EP/MOE-DP/MTP 搜索组合数超过该值时输出警告（≥0；设为 0 关闭警告）。默认 100。',
            'Warn when TP/EP/MOE-DP/MTP search combinations exceed this value (≥0; set 0 to disable). Default 100.',
        ),
        default=100,
        group={'zh': '执行', 'en': 'Execution'},
        control='number',
    ),
    "concurrency-search-strategy": UIFieldProps(
        label=I18nText('并发搜索策略', 'Concurrency Search Strategy'),
        tooltip=I18nText('并发度搜索策略。', 'Concurrency search strategy.'),
        default='exponential',
        group={'zh': '搜索', 'en': 'Search'},
        control='select',
    ),
    "dump-original-results": UIFieldProps(
        label=I18nText('导出原始结果', 'Dump Original Results'),
        tooltip=I18nText('导出原始优化结果。', 'Export original optimization results.'),
        default=False,
        group={'zh': '输出', 'en': 'Output'},
        control='switch',
    ),
    "image-batch-size": UIFieldProps(
        label=I18nText('图像批大小', 'Image Batch Size'),
        tooltip=I18nText(
            '多模态图像批大小（≥1，≤1e6；省略且指定图像高度时运行时回退为 batch_size）。',
            'Multimodal image batch size (≥1, ≤1e6; falls back to batch_size at runtime if omitted while image height is set).',
        ),
        group={'zh': '多模态', 'en': 'Multimodal'},
        control='number',
    ),
    "image-height": UIFieldProps(
        label=I18nText('图像高度', 'Image Height'),
        tooltip=I18nText('多模态图像高度（≥1，≤1e6）。', 'Multimodal image height (≥1, ≤1e6).'),
        group={'zh': '多模态', 'en': 'Multimodal'},
        control='number',
    ),
    "image-width": UIFieldProps(
        label=I18nText('图像宽度', 'Image Width'),
        tooltip=I18nText('多模态图像宽度（≥1，≤1e6）。', 'Multimodal image width (≥1, ≤1e6).'),
        group={'zh': '多模态', 'en': 'Multimodal'},
        control='number',
    ),
    "prefill-devices-per-instance": UIFieldProps(
        label=I18nText('Prefill 实例设备数', 'Prefill Devices Per Instance'),
        tooltip=I18nText(
            'Prefill 实例设备数（>0）。此字段仅在 enable_optimize_prefill_decode_ratio 启用时有效',
            'Prefill devices per instance (>0). This field is only effective when enable_optimize_prefill_decode_ratio is enabled',
        ),
        group={'zh': 'PD 配比', 'en': 'PD Ratio'},
        control='number',
    ),
    "decode-devices-per-instance": UIFieldProps(
        label=I18nText('Decode 实例设备数', 'Decode Devices Per Instance'),
        tooltip=I18nText(
            'Decode 实例设备数（>0）。此字段仅在 enable_optimize_prefill_decode_ratio 启用时有效',
            'Decode devices per instance (>0). This field is only effective when enable_optimize_prefill_decode_ratio is enabled',
        ),
        group={'zh': 'PD 配比', 'en': 'PD Ratio'},
        control='number',
    ),
    "enable-optimize-prefill-decode-ratio": UIFieldProps(
        label=I18nText('启用 PD 配比优化', 'Enable Prefill-Decode Ratio Optimization'),
        tooltip=I18nText(
            '启用 Prefill-Decode 配比优化（与 disagg 互斥）。仅当分离部署模式关闭时可编辑。',
            'Enable prefill-decode ratio optimization (mutually exclusive with disagg). Only editable when disaggregated mode is off.',
        ),
        default=False,
        group={'zh': '模式', 'en': 'Mode'},
        control='switch',
        conditions={"enabled": {"not": {"field": "disagg", "op": "isTrue"}}},
    ),
    "speculative-method": UIFieldProps(
        label=I18nText('投机解码方法', 'Speculative Method'),
        tooltip=I18nText(
            '启用投机解码：mtp、dflash 或 dspark。与旧版 MTP 入口（num-mtp-tokens）互斥。需要先指定才能使用其他投机相关参数。',
            'Enable speculative decoding: mtp, dflash, or dspark. Mutually exclusive with the legacy MTP entry (num-mtp-tokens). Required before speculative-dependent options.',
        ),
        choices=[
            {'value': 'mtp', 'label': 'MTP'},
            {'value': 'dflash', 'label': 'DFlash'},
            {'value': 'dspark', 'label': 'DSpark'},
        ],
        default=None,
        group={'zh': 'MTP', 'en': 'MTP'},
        control='select',
        clearable=True,
    ),
    "num-speculative-tokens": UIFieldProps(
        label=I18nText('投机 token 数', 'Number of Speculative Tokens'),
        tooltip=I18nText(
            '投机 token 数量（不含锚点/奖励 token）。需要 --speculative-method。>= 1 时内部 block_size = n + 1。可传多个值做搜索。省略时 dflash/dspark 使用内置 block_size；mtp 必须显式传入。显式 0 会被拒绝。',
            'Number of speculative tokens excluding anchor/bonus tokens. Requires --speculative-method. When >= 1, internal block_size = n + 1. Multiple values sweep during optimization. Omitting keeps builtin block_size for dflash/dspark; mtp always requires an explicit value. Explicit 0 is rejected.',
        ),
        default=None,
        group={'zh': 'MTP', 'en': 'MTP'},
        control='number',
    ),
    "num-draft-layers": UIFieldProps(
        label=I18nText('草稿层数', 'Number of Draft Layers'),
        tooltip=I18nText(
            '覆盖草稿模型的隐藏层数。需要 --speculative-method dflash 或 dspark（mtp 不允许）。0 = 使用配置默认值。',
            'Override draft model num_hidden_layers. Requires --speculative-method dflash or dspark (not allowed with mtp). 0 = use config default.',
        ),
        default=None,
        group={'zh': 'MTP', 'en': 'MTP'},
        control='number',
        conditions={"enabled": {"field": "speculative-method", "op": "include", "value": ["dflash", "dspark"]}},
    ),
    "draft-model-config-path": UIFieldProps(
        hidden=True,
    ),
    "dspark-markov-rank": UIFieldProps(
        label=I18nText('DSpark Markov 秩', 'DSpark Markov Rank'),
        tooltip=I18nText(
            'Markov 嵌入秩。需要 --speculative-method dspark。0 禁用 MarkovHead。默认 256。',
            'Markov embedding rank. Requires --speculative-method dspark. 0 disables MarkovHead. Default: 256.',
        ),
        default=256,
        group={'zh': 'MTP', 'en': 'MTP'},
        control='number',
        conditions={"enabled": {"field": "speculative-method", "op": "eq", "value": "dspark"}},
    ),
    "dspark-markov-head": UIFieldProps(
        label=I18nText('DSpark Markov Head 类型', 'DSpark Markov Head Type'),
        tooltip=I18nText(
            'Markov head 类型。需要 --speculative-method dspark。可选 vanilla（默认）、gated 或 rnn。',
            'Markov head type. Requires --speculative-method dspark. Options: vanilla (default), gated, or rnn.',
        ),
        choices=[
            {'value': 'vanilla', 'label': 'Vanilla'},
            {'value': 'gated', 'label': 'Gated'},
            {'value': 'rnn', 'label': 'RNN'},
        ],
        default="vanilla",
        group={'zh': 'MTP', 'en': 'MTP'},
        control='select',
        conditions={"enabled": {"field": "speculative-method", "op": "eq", "value": "dspark"}},
    ),
    "acceptance-length": UIFieldProps(
        label=I18nText('接受长度', 'Acceptance Length'),
        tooltip=I18nText(
            '解码折叠标量。需要 --speculative-method。所有方法统一限制为 num_speculative_tokens (n)。',
            'Decode fold scalar. Requires --speculative-method. Clamped to num_speculative_tokens (n) for all methods.',
        ),
        default=None,
        group={'zh': 'MTP', 'en': 'MTP'},
        control='number',
    ),
}


def get_form_config() -> tuple:
    """Get the ModuleSpec and FormConfig for throughput_optimizer module."""
    spec = get_spec("throughput_optimizer")
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
    """Generate form JSON for throughput_optimizer."""
    spec, config = get_form_config()
    return export_form_json(spec, config)


if __name__ == "__main__":
    import json

    form_json = generate_form_json()
    print(json.dumps(form_json, indent=2, ensure_ascii=False))

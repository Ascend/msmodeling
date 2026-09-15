"""Web UI field properties for text_generate module.

Defines UIFieldProps for all text_generate parameters including i18n labels,
tooltips, validation rules, option sources, and conditional visibility.
"""

from cli.registry.datatypes import ModuleSpec
from web_ui.backend.services.ui_props import I18nText, UIFieldProps, LOG_LEVEL_UI, MODEL_ID_OPTIONS
from web_ui.backend.services.json_adapter import FormConfig
from cli.registry.modules import get_spec


# ── Module metadata ─────────────────────────────────────────────

MODULE_ID = "text_generate"
VERSION = "2.9.14"
TITLE = I18nText("文本生成", "Text Generation")
RUNNER = "ModelRunner"

# ── Group labels ──────────────────────────────────────────────────

GROUP_LABELS = {
    "General": I18nText("通用选项", "General"),
    "General Options": I18nText("通用选项", "General Options"),  # CLI canonical alias
    "Request": I18nText("请求参数", "Request"),
    "Optimization": I18nText("优化选项", "Optimization"),
    "Optimization Options": I18nText("优化选项", "Optimization Options"),  # CLI canonical alias
    "Quantization": I18nText("量化选项", "Quantization"),
    "Quantization Options": I18nText("量化选项", "Quantization Options"),  # CLI canonical alias
    "Debug": I18nText("调试选项", "Debug"),
    "Parallelism": I18nText("并行配置", "Parallelism"),
    "Expert": I18nText("专家并行", "Expert"),
    "Advanced Parallelism": I18nText("高级并行", "Advanced Parallelism"),
    "Multimodal": I18nText("多模态", "Multimodal"),
    "Performance Model": I18nText("性能模型", "Performance Model"),
    "Model": I18nText("模型", "Model"),
    "Model Source": I18nText("模型来源", "Model Source"),  # CLI canonical
}

# ── Group collapse metadata ────────────────────────────────────────
# Groups collapsed by default in the frontend form.
GROUPS = [
    {"group_key": "Expert", "defaultCollapsed": True},
    {"group_key": "Advanced Parallelism", "defaultCollapsed": True},
    {"group_key": "Multimodal", "defaultCollapsed": True},
    {"group_key": "Debug", "defaultCollapsed": True},
    {"group_key": "Model", "defaultCollapsed": True},
]

# ── Option sources ─────────────────────────────────────────────────

OPTION_SOURCE_REGISTRY = {
    "devices": {
        "type": "endpoint",
        "endpoint": "/api/options/devices",
        "cache": "session",
    },
}

# ── Validator UI mappings ─────────────────────────────────────────

VALIDATOR_UI = {
    "productEqNumDevices": {
        "fields": ["tp-size", "dp-size", "pp-size", "num-devices"],
        "message": I18nText(
            "tp_size × dp_size × pp_size 必须等于 num_devices",
            "tp_size × dp_size × pp_size must equal num_devices",
        ),
    },
    "moeProductEqNumDevices": {
        "fields": ["moe-tp-size", "moe-dp-size", "ep-size", "num-devices"],
        "message": I18nText(
            "moe_tp_size × moe_dp_size × ep_size 必须等于 num_devices",
            "moe_tp_size × moe_dp_size × ep_size must equal num_devices",
        ),
    },
    "perLayerProductEqNumDevices": {
        "fields": [
            "o-proj-tp-size",
            "o-proj-dp-size",
            "mlp-tp-size",
            "mlp-dp-size",
            "lmhead-tp-size",
            "lmhead-dp-size",
            "num-devices",
        ],
        "message": I18nText(
            "每层 TP × DP 必须等于 num_devices",
            "Per-layer TP × DP must equal num_devices",
        ),
    },
    "sharedExpertMutex": {
        "fields": ["enable-shared-expert-tp", "host-external-shared-experts", "ep-size"],
        "message": I18nText(
            "共享专家相关参数互斥或需 ep_size > 1",
            "Shared-expert options are mutually exclusive or require ep_size > 1",
        ),
    },
    "effectiveLenGe1": {
        "fields": ["query-length", "prefix-cache-hit-rate"],
        "message": I18nText(
            "有效请求长度必须 ≥ 1",
            "Effective query length must be ≥ 1",
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
}


# ── UI field properties ───────────────────────────────────────────

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
        label=I18nText('目标设备', 'Target Device'),
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
        label=I18nText('设备数量', 'Number of Devices'),
        tooltip=I18nText('参与仿真的 Die 数量（≥1）。', 'Number of devices used in the simulation (≥1).'),
        default=1,
        required=True,
        group={'zh': '通用', 'en': 'General'},
        control='number',
    ),
    "reserved-memory-gb": UIFieldProps(
        label=I18nText('预留显存(GB)', 'Reserved Memory (GB)'),
        tooltip=I18nText('单卡为其他进程预留的显存（GB）。', 'Per-device memory reserved for other processes (GB).'),
        default=0.0,
        required=True,
        group={'zh': '通用', 'en': 'General'},
        control='number',
    ),
    "num-queries": UIFieldProps(
        label=I18nText('并行请求数', 'Number of Queries'),
        tooltip=I18nText(
            '并发推理的请求数量；逗号分隔多值时每个取值单独仿真，结果区生成多用例对比（如 1,4,8）。',
            'Concurrently inferred query count; with a comma-list each value runs independently and yields a multi-case comparison (e.g. 1,4,8).',
        ),
        placeholder=I18nText('如 1,4,8', 'e.g. 1,4,8'),
        multi_values=True,
        default=1,
        required=True,
        group={'zh': '请求', 'en': 'Request'},
        control='text',
    ),
    "query-length": UIFieldProps(
        label=I18nText('输入序列长度', 'Input Sequence Length'),
        tooltip=I18nText(
            '单条请求的输入 token 长度 + mtp token 数（≥1）。', 'Input token length per query + mtp token count (≥1).'
        ),
        default=1,
        required=True,
        group={'zh': '请求', 'en': 'Request'},
        control='number',
    ),
    "context-length": UIFieldProps(
        label=I18nText('上下文长度', 'Context Length'),
        tooltip=I18nText('上下文窗口大小（0 表示不限制）。', 'Context window size (0 = unlimited).'),
        default=4500,
        group={'zh': '请求', 'en': 'Request'},
        control='number',
    ),
    "decode": UIFieldProps(
        label=I18nText('自回归解码', 'Autoregressive Decode'),
        tooltip=I18nText('启用自回归解码阶段。', 'Enable the autoregressive decode stage.'),
        default=True,
        group={'zh': '请求', 'en': 'Request'},
        control='switch',
    ),
    "prefix-cache-hit-rate": UIFieldProps(
        label=I18nText('前缀缓存命中率', 'Prefix Cache Hit Rate'),
        tooltip=I18nText('模拟的前缀缓存命中率 [0, 1)。', 'Simulated prefix cache hit rate [0, 1).'),
        default=0.0,
        group={'zh': '请求', 'en': 'Request'},
        control='number',
    ),
    "num-mtp-tokens": UIFieldProps(
        label=I18nText('MTP token 数', 'MTP Token Count'),
        tooltip=I18nText(
            'MTP token 数量。推荐使用上方的"投机解码方法=mtp + 投机 token 数"统一入口。选定 speculative-method 后此字段自动禁用。',
            'MTP token count. Prefer the unified entry above (speculative-method=mtp + num-speculative-tokens). Auto-disabled when speculative-method is set.',
        ),
        default=0,
        group={'zh': '请求', 'en': 'Request'},
        control='number',
        conditions={"enabled": {"field": "speculative-method", "op": "empty"}},
    ),
    "no-repetition": UIFieldProps(
        label=I18nText('禁用重复层优化', 'Disable Repetition Reuse'),
        tooltip=I18nText(
            '关闭后不再自动检测并利用模型中结构相同的重复层进行计算共享，每层独立建模，保留 Transformer 原始行为；仿真耗时会增加。',
            'Disable automatic detection and reuse of structurally identical transformer layers. Each layer is modeled independently, preserving the original transformer behavior at increased runtime cost.',
        ),
        default=False,
        group={'zh': '调试', 'en': 'Debug'},
        control='switch',
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
    "compile-allow-graph-break": UIFieldProps(
        label=I18nText('允许图断裂', 'Allow Graph Break'),
        tooltip=I18nText(
            '允许在 torch.compile() 期间出现图断裂，适用于具有动态控制流的模型',
            'Allow graph breaks during torch.compile() for models with dynamic control flow',
        ),
        hidden=True,
        default=False,
        control='switch',
    ),
    "compilation-config": UIFieldProps(
        label=I18nText('编译优化选项', 'Compilation Config'),
        tooltip=I18nText(
            '选择要启用的编译期优化（多选）。', 'Select compilation optimizations to enable (multi-select).'
        ),
        placeholder=I18nText('请选择编译优化选项', 'Select compilation options'),
        choices=[
            {'value': 'enable_multistream', 'label': 'Enable Multistream'},
            {'value': 'enable_sequence_parallel', 'label': 'Enable Sequence Parallel'},
            {'value': 'enable_matmul_allreduce', 'label': 'Enable Matmul Allreduce'},
            {'value': 'enable_dispatch_ffn_combine', 'label': 'Enable Dispatch FFN Combine'},
        ],
        default=[],
        group={'zh': '优化', 'en': 'Optimization'},
        control='multi-select',
    ),
    "quantize-linear-action": UIFieldProps(
        label=I18nText('线性层量化', 'Linear-Layer Quantization'),
        tooltip=I18nText(
            '线性层的量化策略（可多选；每个取值单独仿真，结果区生成多用例对比）。',
            'Quantization action(s) for linear layers (multi-select; each value runs independently and yields a multi-case comparison).',
        ),
        multi_values=True,
        choices=[
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
        default=['W8A8_DYNAMIC'],
        group={'zh': '量化', 'en': 'Quantization'},
        control='multi-select',
    ),
    "quantize-non-expert-linear-action": UIFieldProps(
        label=I18nText('非专家线性层', 'Non-Expert Linear Quantization'),
        tooltip=I18nText(
            '为非专家线性层（如注意力投影、稠密MLP层）设置的单独量化类型',
            'Separate quantization type for non-expert linear layers (e.g. attention projection, dense MLP layers).',
        ),
        choices=[
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
        default='DISABLED',
        group={'zh': '量化', 'en': 'Quantization'},
        control='select',
    ),
    "quantize-lmhead": UIFieldProps(
        label=I18nText('量化 LM Head', 'Quantize LM Head'),
        tooltip=I18nText('对 LM Head 层应用量化。', 'Apply quantization to LM Head layer.'),
        default=False,
        group={'zh': '量化', 'en': 'Quantization'},
        control='switch',
    ),
    "mxfp4-group-size": UIFieldProps(
        label=I18nText('MXFP4 分组大小', 'MXFP4 Group Size'),
        tooltip=I18nText(
            'MXFP4 量化的分组大小（>0）。此字段仅在 quantize_linear_action 或 quantize_non_expert_linear_action 包含 MXFP4 时有效',
            'MXFP4 quantization group size (>0). This field is only effective when quantize_linear_action or quantize_non_expert_linear_action contains MXFP4',
        ),
        default=32,
        group={'zh': '量化', 'en': 'Quantization'},
        control='number',
    ),
    "quantize-attention-action": UIFieldProps(
        label=I18nText('KV cache 量化', 'Attention KV Cache Quantization'),
        tooltip=I18nText(
            'Attention KV Cache 的量化策略（可多选；每个取值单独仿真，结果区生成多用例对比）。',
            'Quantization for attention KV cache (multi-select; each value runs independently and yields a multi-case comparison).',
        ),
        multi_values=True,
        choices=[
            {'value': 'DISABLED', 'label': 'DISABLED'},
            {'value': 'INT8', 'label': 'INT8'},
            {'value': 'FP8', 'label': 'FP8'},
        ],
        default=['DISABLED'],
        group={'zh': '量化', 'en': 'Quantization'},
        control='multi-select',
    ),
    "graph-log-path": UIFieldProps(
        label=I18nText('图日志 URL', 'Graph Log URL'),
        tooltip=I18nText('编译图转储路径', 'Path for dumping compiled graphs'),
        placeholder=I18nText('如 /path/to/graph.json', 'e.g., /path/to/graph.json'),
        hidden=True,
    ),
    "dump-input-shapes": UIFieldProps(
        label=I18nText('按输入形状分组', 'Group by Input Shapes'),
        tooltip=I18nText(
            '开启后算子平均表按输入 tensor 形状分组统计，显示 Input Shapes 列。',
            'When enabled, the operator average table groups statistics by input tensor shapes and shows an Input Shapes column.',
        ),
        default=False,
        group={'zh': '调试', 'en': 'Debug'},
        control='switch',
    ),
    "dump-op-bound-results": UIFieldProps(
        label=I18nText('显示算子瓶颈分析', 'Show Operator Bound Analysis'),
        tooltip=I18nText(
            '开启后算子平均表显示每个算子的性能瓶颈（Memory/Communication/MMA/GP）及百分比。',
            'When enabled, the operator average table shows per-operator performance bound (Memory/Communication/MMA/GP) and percentages.',
        ),
        default=False,
        group={'zh': '调试', 'en': 'Debug'},
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
    "num-hidden-layers-override": UIFieldProps(
        label=I18nText('覆盖隐藏层数', 'Num Hidden Layers Override'),
        tooltip=I18nText(
            '覆盖模型的隐藏层数量（0 = 不覆盖）。', 'Override the number of hidden layers (0 = no override).'
        ),
        default=0,
        group={'zh': '调试', 'en': 'Debug'},
        control='number',
    ),
    "export-empirical-metrics-file": UIFieldProps(
        label=I18nText('导出经验指标', 'Export Empirical Metrics'),
        tooltip=I18nText(
            '（仅开发者）将 M1-M5 指标报告作为 JSON 导出到指定路径',
            '(developer only) Export M1-M5 metrics report as JSON to the specified path',
        ),
        placeholder=I18nText('如 /path/to/metrics.json', 'e.g., /path/to/metrics.json'),
        hidden=True,
    ),
    "tp-size": UIFieldProps(
        label=I18nText('张量并行', 'Tensor Parallel Size'),
        tooltip=I18nText(
            '张量并行度；逗号分隔多值时每个取值单独仿真，结果区生成多用例对比（如 1,2,4）。',
            'Tensor-parallel degree; with a comma-list each value runs independently and yields a multi-case comparison (e.g. 1,2,4).',
        ),
        placeholder=I18nText('如 1,2,4', 'e.g. 1,2,4'),
        multi_values=True,
        default=1,
        required=True,
        group={'zh': '并行', 'en': 'Parallelism'},
        control='text',
    ),
    "pp-size": UIFieldProps(
        label=I18nText('流水线并行', 'Pipeline Parallel Size'),
        tooltip=I18nText(
            '流水线并行度（≥1）。tp_size × dp_size × pp_size 必须等于 num_devices。',
            'Pipeline-parallel degree (≥1). tp_size × dp_size × pp_size must equal num_devices.',
        ),
        default=1,
        required=True,
        group={'zh': '并行', 'en': 'Parallelism'},
        control='number',
    ),
    "dcp-size": UIFieldProps(
        label=I18nText('解码上下文并行大小', 'Decode Context Parallel Size'),
        tooltip=I18nText('解码上下文并行大小', 'Decode Context Parallel size'),
        hidden=True,
        default=1,
    ),
    "dp-size": UIFieldProps(
        label=I18nText('数据并行', 'Data Parallel Size'),
        tooltip=I18nText(
            '数据并行度，留空时自动按 num_devices // (tp_size × pp_size) 推导。',
            'Data-parallel degree; auto-derived as num_devices // (tp_size × pp_size) when left empty.',
        ),
        group={'zh': '并行', 'en': 'Parallelism'},
        control='number',
    ),
    "word-embedding-tp": UIFieldProps(
        label=I18nText('词嵌入 TP 模式', 'Word Embedding TP Mode'),
        tooltip=I18nText('词嵌入张量并行模式。', 'Word embedding tensor parallelism mode.'),
        choices=[{'value': 'col', 'label': 'Column Parallel'}, {'value': 'row', 'label': 'Row Parallel'}],
        group={'zh': '并行', 'en': 'Parallelism'},
        control='select',
    ),
    "ep-size": UIFieldProps(
        label=I18nText('专家并行', 'Expert Parallel Size'),
        tooltip=I18nText('专家并行度（MoE 模型，≥1）。', 'Expert-parallel degree (MoE models, ≥1).'),
        default=1,
        required=True,
        group={'zh': '专家并行', 'en': 'Expert'},
        control='number',
    ),
    "moe-tp-size": UIFieldProps(
        label=I18nText('MoE 张量并行', 'MoE Tensor Parallel Size'),
        tooltip=I18nText(
            'MoE 专家张量并行度，留空时自动计算。',
            'MoE expert tensor-parallel degree; auto-calculated when empty.',
        ),
        group={'zh': '专家并行', 'en': 'Expert'},
        control='number',
    ),
    "moe-dp-size": UIFieldProps(
        label=I18nText('MoE 数据并行', 'MoE Data Parallel Size'),
        tooltip=I18nText('MoE 专家数据并行度（≥1）。', 'MoE expert data-parallel degree (≥1).'),
        default=1,
        required=True,
        group={'zh': '专家并行', 'en': 'Expert'},
        control='number',
    ),
    "enable-redundant-experts": UIFieldProps(
        label=I18nText('启用冗余专家', 'Enable Redundant Experts'),
        tooltip=I18nText('启用冗余专家（仅 MoE 模型）。', 'Enable redundant experts (MoE models only).'),
        default=False,
        group={'zh': '专家并行', 'en': 'Expert'},
        control='switch',
    ),
    "enable-shared-expert-tp": UIFieldProps(
        label=I18nText('启用共享专家 TP', 'Enable Shared Expert TP'),
        tooltip=I18nText(
            '对共享专家启用张量并行。此字段要求 ep_size > 1，且与 host_external_shared_experts 互斥（不能同时启用）',
            'Tensor-parallelize the shared expert. This field requires ep_size > 1 and is mutually exclusive with host_external_shared_experts (cannot be enabled simultaneously)',
        ),
        default=False,
        group={'zh': '专家并行', 'en': 'Expert'},
        control='switch',
    ),
    "enable-external-shared-experts": UIFieldProps(
        label=I18nText('启用外部共享专家', 'Enable External Shared Experts'),
        tooltip=I18nText('启用外部共享专家（仅 MoE 模型）。', 'Enable external shared experts (MoE models only).'),
        default=False,
        group={'zh': '专家并行', 'en': 'Expert'},
        control='switch',
    ),
    "host-external-shared-experts": UIFieldProps(
        label=I18nText('宿主外部共享专家', 'Host External Shared Experts'),
        tooltip=I18nText(
            '将外部共享专家宿主于计算流。此字段与 enable_shared_expert_tp 互斥（不能同时启用）',
            'Host external shared experts on the compute stream. This field is mutually exclusive with enable_shared_expert_tp (cannot be enabled simultaneously)',
        ),
        default=False,
        group={'zh': '专家并行', 'en': 'Expert'},
        control='switch',
    ),
    "o-proj-tp-size": UIFieldProps(
        label=I18nText('o_proj 张量并行', 'O-proj Tensor Parallel Size'),
        tooltip=I18nText(
            'O-proj 层张量并行度，留空时继承 tp_size。',
            'O-proj layer tensor-parallel degree; inherits tp_size when empty.',
        ),
        group={'zh': '高级并行', 'en': 'Advanced Parallelism'},
        control='number',
    ),
    "o-proj-dp-size": UIFieldProps(
        label=I18nText('o_proj 数据并行', 'O-proj Data Parallel Size'),
        tooltip=I18nText(
            'O-proj 层数据并行度，留空时自动计算。',
            'O-proj layer data-parallel degree; auto-calculated when empty.',
        ),
        group={'zh': '高级并行', 'en': 'Advanced Parallelism'},
        control='number',
    ),
    "mlp-tp-size": UIFieldProps(
        label=I18nText('MLP 张量并行', 'MLP Tensor Parallel Size'),
        tooltip=I18nText(
            'MLP 层张量并行度，留空时继承 tp_size。',
            'MLP layer tensor-parallel degree; inherits tp_size when empty.',
        ),
        group={'zh': '高级并行', 'en': 'Advanced Parallelism'},
        control='number',
    ),
    "mlp-dp-size": UIFieldProps(
        label=I18nText('MLP 数据并行', 'MLP Data Parallel Size'),
        tooltip=I18nText(
            'MLP 层数据并行度，留空时自动计算。', 'MLP layer data-parallel degree; auto-calculated when empty.'
        ),
        group={'zh': '高级并行', 'en': 'Advanced Parallelism'},
        control='number',
    ),
    "lmhead-tp-size": UIFieldProps(
        label=I18nText('LM Head 张量并行', 'LM Head Tensor Parallel Size'),
        tooltip=I18nText(
            'LM Head 层张量并行度，留空时继承 tp_size。',
            'LM Head layer tensor-parallel degree; inherits tp_size when empty.',
        ),
        group={'zh': '高级并行', 'en': 'Advanced Parallelism'},
        control='number',
    ),
    "lmhead-dp-size": UIFieldProps(
        label=I18nText('LM Head 数据并行', 'LM Head Data Parallel Size'),
        tooltip=I18nText(
            'LM Head 层数据并行度，留空时自动计算。',
            'LM Head layer data-parallel degree; auto-calculated when empty.',
        ),
        group={'zh': '高级并行', 'en': 'Advanced Parallelism'},
        control='number',
    ),
    "vision-tp-size": UIFieldProps(
        label=I18nText('Vision 张量并行', 'Vision Tensor Parallel Size'),
        tooltip=I18nText(
            '视觉编码张量并行度，不得超过 num_devices 且 num_devices 必须能被其整除。',
            'Vision tensor-parallel degree; must be ≤ num_devices and num_devices must be divisible by it.',
        ),
        default=1,
        group={'zh': '高级并行', 'en': 'Advanced Parallelism'},
        control='number',
    ),
    "image-batch-size": UIFieldProps(
        label=I18nText('图像批大小', 'Image Batch Size'),
        tooltip=I18nText(
            '每次处理的图像数量（多模态模型）。', 'Number of images processed per batch (multimodal models).'
        ),
        group={'zh': '多模态', 'en': 'Multimodal'},
        control='number',
    ),
    "image-height": UIFieldProps(
        label=I18nText('图像高度', 'Image Height'),
        tooltip=I18nText('输入图像高度（多模态模型）。', 'Input image height (multimodal models).'),
        group={'zh': '多模态', 'en': 'Multimodal'},
        control='number',
    ),
    "image-width": UIFieldProps(
        label=I18nText('图像宽度', 'Image Width'),
        tooltip=I18nText('输入图像宽度（多模态模型）。', 'Input image width (multimodal models).'),
        group={'zh': '多模态', 'en': 'Multimodal'},
        control='number',
    ),
    "performance-model": UIFieldProps(
        label=I18nText('性能模型', 'Performance Model'),
        tooltip=I18nText('性能模型类型。可以多次指定', 'Performance model type(s). Can be specified multiple times'),
        hidden=True,
        data_type='string[]',
        choices=[
            {'value': 'analytic', 'label': 'Analytic'},
            {'value': 'calibrated', 'label': 'Calibrated'},
            {'value': 'profiling', 'label': 'Profiling'},
        ],
        default=['analytic'],
        control='multi-select',
    ),
    "profiling-database-path": UIFieldProps(
        label=I18nText('性能分析数据库', 'Profiling Database'),
        tooltip=I18nText('性能分析数据库目录', 'Directory of the profiling database'),
        placeholder=I18nText('如 /path/to/db', 'e.g., /path/to/db'),
        hidden=True,
    ),
    "disable-profiling-interpolation": UIFieldProps(
        label=I18nText('禁用性能分析插值', 'Disable Profiling Interpolation'),
        tooltip=I18nText(
            '仅使用精确和部分性能分析匹配（无插值）',
            'Use exact and partial profiling matches only (no interpolation)',
        ),
        hidden=True,
        default=False,
        control='switch',
    ),
    "analytic-calibration-profile": UIFieldProps(hidden=True),
    "analytic-calibration-stack": UIFieldProps(hidden=True),
    "remote-source": UIFieldProps(
        label=I18nText('模型远程来源', 'Model Remote Source'),
        tooltip=I18nText('模型加载的远程来源。', 'Remote source for model loading.'),
        choices=[{'value': 'huggingface', 'label': 'Hugging Face'}, {'value': 'modelscope', 'label': 'ModelScope'}],
        default='huggingface',
        group={'zh': '模型', 'en': 'Model'},
        control='select',
    ),
    "fusion-plugin": UIFieldProps(hidden=True),
    "log-level": LOG_LEVEL_UI.override(
        tooltip=I18nText('日志输出级别。', 'Log output level.'),
        group={'zh': '通用', 'en': 'General'},
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
        group={'zh': '优化', 'en': 'Optimization'},
        control='select',
        clearable=True,
    ),
    "num-speculative-tokens": UIFieldProps(
        label=I18nText('投机 token 数', 'Number of Speculative Tokens'),
        tooltip=I18nText(
            '投机 token 数量（不含锚点/奖励 token）。需要 --speculative-method。>= 1 时内部 block_size = n + 1。省略时 dflash/dspark 使用内置 block_size；mtp 必须显式传入。显式 0 会被拒绝。',
            'Number of speculative tokens excluding anchor/bonus tokens. Requires --speculative-method. When >= 1, internal block_size = n + 1. Omitting keeps builtin block_size for dflash/dspark; mtp always requires an explicit value. Explicit 0 is rejected.',
        ),
        default=None,
        group={'zh': '优化', 'en': 'Optimization'},
        control='number',
    ),
    "num-draft-layers": UIFieldProps(
        label=I18nText('草稿层数', 'Number of Draft Layers'),
        tooltip=I18nText(
            '覆盖草稿模型的隐藏层数。需要 --speculative-method dflash 或 dspark（mtp 不允许）。0 = 使用配置默认值。',
            'Override draft model num_hidden_layers. Requires --speculative-method dflash or dspark (not allowed with mtp). 0 = use config default.',
        ),
        default=None,
        group={'zh': '优化', 'en': 'Optimization'},
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
        group={'zh': '优化', 'en': 'Optimization'},
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
        group={'zh': '优化', 'en': 'Optimization'},
        control='select',
        conditions={"enabled": {"field": "speculative-method", "op": "eq", "value": "dspark"}},
    ),
}


# ── FormConfig assembly ───────────────────────────────────────────


def get_form_config() -> tuple[ModuleSpec, FormConfig]:
    """Get the ModuleSpec and FormConfig for text_generate module."""
    spec = get_spec(MODULE_ID)
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


# ── JSON export entry point ───────────────────────────────────────


def export_form_json() -> dict:
    """Generate the form JSON for text_generate module."""
    from web_ui.backend.services.json_adapter import export_form_json as _export

    spec, config = get_form_config()
    return _export(spec, config)


if __name__ == "__main__":
    import json

    form_json = export_form_json()
    print(json.dumps(form_json, indent=2, ensure_ascii=False))

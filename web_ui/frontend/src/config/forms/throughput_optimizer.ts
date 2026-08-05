/* eslint-disable */
// AUTO-CONVERTED from JSON. This .ts is the source of truth (data + inlined
// validators). A build step regenerates the data-only .json for the backend
// (validators are stripped by JSON.stringify). Do not edit the .json by hand.
import { stringValid, prefixCacheRate, lteNumDevices, batchRange, validParallelCombo, positiveOrInf, effectiveLenGe1, pdRatioMutexDisagg, mtpAcceptanceRatesPositive, mtpTokensVsAcceptanceRate } from "./_validators"
import { QUANTIZE_LINEAR_OPTIONS, QUANTIZE_ATTENTION_OPTIONS, SERVING_LOG_LEVEL_OPTIONS } from "./_validators"

export default {
"$schema": "form-schema/v1",
  "moduleId": "throughput_optimizer",
  "title": { "zh": "吞吐优化", "en": "Throughput Optimizer" },
  "runner": "ParallelRunner",
  "version": "1.10.0",
  "optionSourceRegistry": {
    "devices": { "endpoint": "/api/options/devices", "cache": "session" }
  },
  "formValidation": [
    {
      "rule": "validator",
      "value": "validParallelCombo",
      "message": {
        "zh": "在当前 num_devices 下，不存在有效的 (tp, ep, moe_dp) 组合",
        "en": "No valid (tp, ep, moe_dp) combination exists under current num_devices"
      },
      "dependsOn": ["tp_sizes", "ep_sizes", "moe_dp_sizes", "num_devices"]
    },
    {
      "rule": "validator",
      "value": "effectiveLenGe1",
      "message": {
        "zh": "有效输入长度（扣除前缀缓存后）必须 ≥ 1",
        "en": "Effective input length (after prefix cache) must be ≥ 1"
      },
      "dependsOn": ["input_length", "prefix_cache_hit_rate"]
    },
    {
      "rule": "validator",
      "value": "pdRatioMutexDisagg",
      "message": {
        "zh": "PD 配比优化不能与分离部署模式（disagg）同时使用",
        "en": "PD-ratio optimization cannot be used together with disagg"
      },
      "dependsOn": ["enable_optimize_prefill_decode_ratio", "disagg"]
    },
    {
      "rule": "validator",
      "value": "mtpTokensVsAcceptanceRate",
      "message": {
        "zh": "num_mtp_tokens 不得超过 mtp_acceptance_rate 列表长度 + 1",
        "en": "num_mtp_tokens must be ≤ mtp_acceptance_rate list length + 1"
      },
      "dependsOn": ["num_mtp_tokens", "mtp_acceptance_rate"]
    }
  ],  "groups": [
    { "label": { "zh": "MTP", "en": "MTP" }, "defaultCollapsed": true },
    { "label": { "zh": "缓存", "en": "Cache" }, "defaultCollapsed": true },
    { "label": { "zh": "成本", "en": "Cost" }, "defaultCollapsed": true },
    { "label": { "zh": "模式", "en": "Mode" }, "defaultCollapsed": true, "description": { "zh": "以下两个开关均关闭时，将使用默认的 PD 混部（聚合）部署模式。", "en": "With both switches off, the default PD-co-located (aggregated) deployment mode is used." } },
    { "label": { "zh": "执行", "en": "Execution" }, "defaultCollapsed": true },
    { "label": { "zh": "输出", "en": "Output" }, "defaultCollapsed": true },
    { "label": { "zh": "模型", "en": "Model" }, "defaultCollapsed": true },
    { "label": { "zh": "PD 配比", "en": "PD Ratio" }, "defaultCollapsed": true },
    { "label": { "zh": "多模态", "en": "Multimodal" }, "defaultCollapsed": true }
  ],
  "fields": [
    {
      "id": "model_id",
      "label": { "zh": "模型 ID", "en": "Model ID" },
      "control": "text",
      "dataType": "string",
      "default": "Qwen/Qwen3-32B",
      "group": { "zh": "通用", "en": "General" },
      "tooltip": { "zh": "待仿真模型的标准 HuggingFace 名称（组织/模型，如 Qwen/Qwen3-32B）或本地路径。", "en": "Standard HuggingFace model name (org/model, e.g. Qwen/Qwen3-32B) or local path." },
      "placeholder": { "zh": "如 Qwen/Qwen3-32B", "en": "e.g. Qwen/Qwen3-32B" },
      "validation": [
        { "rule": "required", "message": { "zh": "模型 ID 为必填项", "en": "Model ID is required" }, "trigger": ["change", "blur"] },
        { "rule": "validator", "value": "stringValid", "message": { "zh": "模型 ID 含非法字符或过长", "en": "Model ID has invalid characters or is too long" }, "trigger": ["blur"] }
      ]
    },
    {
      "id": "device",
      "label": { "zh": "设备类型", "en": "Target Device" },
      "control": "multi-select",
      "dataType": "string[]",
      "default": ["ATLAS_350_425T_112G"],
      "group": { "zh": "通用", "en": "General" },
      "tooltip": { "zh": "选择用于仿真的设备 Profile（可多选；每个取值单独仿真，结果区生成多用例对比）。", "en": "Device profile(s) to simulate on (multi-select; each value runs independently and yields a multi-case comparison)." },
      "placeholder": { "zh": "请选择设备型号", "en": "Select device models" },
      "optionSource": { "type": "dynamic", "name": "devices" },
      "validation": [
        { "rule": "required", "message": { "zh": "设备为必选项", "en": "Device is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "num_devices",
      "label": { "zh": "设备数量", "en": "Number of Devices" },
      "control": "number",
      "dataType": "integer",
      "default": 4,
      "group": { "zh": "通用", "en": "General" },
      "tooltip": { "zh": "参与仿真的 Die 数量（≥1）。", "en": "Number of devices used in the simulation (≥1)." },
      "validation": [
        { "rule": "required", "message": { "zh": "设备数量为必填项", "en": "Number of Devices is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "reserved_memory_gb",
      "label": { "zh": "预留显存(GB)", "en": "Reserved Memory (GB)" },
      "control": "number",
      "dataType": "number",
      "default": 10.0,
      "group": { "zh": "通用", "en": "General" },
      "tooltip": { "zh": "单卡为其他进程预留的显存（GB）。", "en": "Per-device memory reserved for other processes (GB)." },
      "validation": [
        { "rule": "required", "message": { "zh": "预留显存为必填项", "en": "Reserved Memory is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 0, "type": "number", "message": { "zh": "必须 ≥ 0", "en": "Must be ≥ 0" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "log_level",
      "label": { "zh": "日志级别", "en": "Log Level" },
      "control": "select",
      "dataType": "string",
      "default": "error",
      "group": { "zh": "通用", "en": "General" },
      "tooltip": { "zh": "日志输出级别。", "en": "Log output level." },
      "optionSource": { "type": "inline", "values": SERVING_LOG_LEVEL_OPTIONS },
      "validation": [
        { "rule": "required", "message": { "zh": "日志级别为必填项", "en": "Log Level is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "input_length",
      "label": { "zh": "输入(prompt)长度", "en": "Input Prompt Length" },
      "control": "number",
      "dataType": "integer",
      "default": 3500,
      "group": { "zh": "输入", "en": "Input" },
      "tooltip": { "zh": "输入 prompt 的 token 长度（≥1，<1e6）。", "en": "Input prompt token length (≥1, <1e6)." },
      "validation": [
        { "rule": "required", "message": { "zh": "输入长度为必填项", "en": "Input length is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] },
        { "rule": "max", "value": 1000000, "type": "integer", "message": { "zh": "必须 < 1e6", "en": "Must be < 1e6" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "output_length",
      "label": { "zh": "输出长度", "en": "Output Length" },
      "control": "number",
      "dataType": "integer",
      "default": 1500,
      "group": { "zh": "输入", "en": "Input" },
      "tooltip": { "zh": "输出 token 长度（≥1，<1e6）。", "en": "Output token length (≥1, <1e6)." },
      "validation": [
        { "rule": "required", "message": { "zh": "输出长度为必填项", "en": "Output length is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] },
        { "rule": "max", "value": 1000000, "type": "integer", "message": { "zh": "必须 < 1e6", "en": "Must be < 1e6" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "compile",
      "label": { "zh": "启用 torch.compile", "en": "Enable torch.compile" },
      "control": "switch",
      "dataType": "boolean",
      "default": true,
      "disabled": true,
      "group": { "zh": "优化", "en": "Optimization" },
      "tooltip": { "zh": "默认启用 torch.compile 加速（已锁定，不可修改）。", "en": "torch.compile acceleration is enabled by default (locked)." }
    },
    {
      "id": "compilation_config",
      "label": { "zh": "编译优化选项", "en": "Compilation Config" },
      "control": "multi-select",
      "dataType": "string[]",
      "default": [],
      "group": { "zh": "优化", "en": "Optimization" },
      "tooltip": { "zh": "选择要启用的编译期优化（多选）。", "en": "Select compilation optimizations to enable (multi-select)." },
      "placeholder": { "zh": "请选择编译优化选项", "en": "Select compilation options" },
      "optionSource": {
        "type": "inline",
        "values": [
          { "label": { "zh": "多流并行", "en": "Multistream" }, "value": "enable_multistream" },
          { "label": { "zh": "序列并行", "en": "Sequence Parallel" }, "value": "enable_sequence_parallel" },
          { "label": { "zh": "MatMul-AllReduce 融合", "en": "MatMul-AllReduce Fusion" }, "value": "enable_matmul_allreduce" },
          { "label": { "zh": "Dispatch FFN 合并", "en": "Dispatch FFN Combine" }, "value": "enable_dispatch_ffn_combine" }
        ]
      }
    },
    {
      "id": "num_mtp_tokens",
      "label": { "zh": "MTP token 数", "en": "MTP Token Count" },
      "control": "select",
      "dataType": "integer",
      "default": 0,
      "group": { "zh": "MTP", "en": "MTP" },
      "tooltip": { "zh": "Multi-Text Prediction token 数量（argparse 允许 0-9；运行时实际受 mtp_acceptance_rate 长度限制：num_mtp_tokens ≤ len(mtp_acceptance_rate)+1）。", "en": "Multi-Text Prediction token count (argparse allows 0-9; runtime is bounded by mtp_acceptance_rate length: num_mtp_tokens ≤ len(mtp_acceptance_rate)+1)." },
      "validation": [
        { "rule": "required", "message": { "zh": "MTP token 数为必填项", "en": "MTP Token Count is required" }, "trigger": ["change", "blur"] }
      ],
      "optionSource": {
        "type": "inline",
        "values": [
          { "value": 0, "label": "0" },
          { "value": 1, "label": "1" },
          { "value": 2, "label": "2" },
          { "value": 3, "label": "3" },
          { "value": 4, "label": "4" },
          { "value": 5, "label": "5" },
          { "value": 6, "label": "6" },
          { "value": 7, "label": "7" },
          { "value": 8, "label": "8" },
          { "value": 9, "label": "9" }
        ]
      }
    },
    {
      "id": "mtp_acceptance_rate",
      "label": { "zh": "MTP 接收率列表", "en": "MTP Acceptance Rate List" },
      "control": "text",
      "dataType": "string",
      "default": "0.8, 0.6, 0.4, 0.2",
      "group": { "zh": "MTP", "en": "MTP" },
      "tooltip": { "zh": "MTP 接收率列表（逗号或空格分隔的浮点数，每项 >0）。运行时约束：num_mtp_tokens ≤ 列表长度 + 1。此字段仅在 num_mtp_tokens > 0 时有效", "en": "MTP acceptance rate list (comma/space separated floats, each >0). Runtime constraint: num_mtp_tokens ≤ list length + 1. This field is only effective when num_mtp_tokens > 0" },
      "validation": [
        { "rule": "required", "message": { "zh": "MTP 接收率列表为必填项", "en": "MTP Acceptance Rate List is required" }, "trigger": ["change", "blur"] },
        { "rule": "validator", "value": "mtpAcceptanceRatesPositive", "message": { "zh": "每项必须为正浮点数（逗号或空格分隔）", "en": "Each item must be a positive float (comma/space separated)" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "prefix_cache_hit_rate",
      "label": { "zh": "前缀缓存命中率", "en": "Prefix Cache Hit Rate" },
      "control": "number",
      "dataType": "number",
      "default": 0.0,
      "group": { "zh": "缓存", "en": "Cache" },
      "tooltip": { "zh": "模拟的前缀缓存命中率 [0, 1)。", "en": "Simulated prefix cache hit rate [0, 1)." },
      "validation": [
        { "rule": "required", "message": { "zh": "前缀缓存命中率为必填项", "en": "Prefix Cache Hit Rate is required" }, "trigger": ["change", "blur"] },
        { "rule": "validator", "value": "prefixCacheRate", "message": { "zh": "必须在 [0, 1) 区间", "en": "Must be in [0, 1)" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "quantize_linear_action",
      "label": { "zh": "线性层量化", "en": "Linear-Layer Quantization" },
      "control": "multi-select",
      "dataType": "string[]",
      "default": ["W8A8_DYNAMIC"],
      "group": { "zh": "量化", "en": "Quantization" },
      "tooltip": { "zh": "线性层的量化策略（可多选；每个取值单独仿真，结果区生成多用例对比）。", "en": "Quantization action(s) for linear layers (multi-select; each value runs independently and yields a multi-case comparison)." },
      "optionSource": { "type": "inline", "values": QUANTIZE_LINEAR_OPTIONS },
      "validation": [
        { "rule": "required", "message": { "zh": "线性层量化为必填项", "en": "Linear-Layer Quantization is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "quantize_non_expert_linear_action",
      "label": { "zh": "非专家线性层", "en": "Non-Expert Linear Quantization" },
      "control": "select",
      "dataType": "string",
      "default": "DISABLED",
      "group": { "zh": "量化", "en": "Quantization" },
      "tooltip": { "zh": "为非专家线性层（如注意力投影、稠密MLP层）设置的单独量化类型", "en": "Separate quantization type for non-expert linear layers (e.g. attention projection, dense MLP layers)." },
      "optionSource": { "type": "inline", "values": QUANTIZE_LINEAR_OPTIONS },
      "validation": [
        { "rule": "required", "message": { "zh": "非专家线性层量化为必填项", "en": "Non-Expert Linear Quantization is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "mxfp4_group_size",
      "label": { "zh": "MXFP4 分组大小", "en": "MXFP4 Group Size" },
      "control": "number",
      "dataType": "integer",
      "default": 32,
      "group": { "zh": "量化", "en": "Quantization" },
      "tooltip": { "zh": "MXFP4 量化的分组大小（>0，≤1e6）。此字段仅在 quantize_linear_action 或 quantize_non_expert_linear_action 包含 MXFP4 时有效", "en": "MXFP4 quantization group size (>0, ≤1e6). This field is only effective when quantize_linear_action or quantize_non_expert_linear_action contains MXFP4" },
      "validation": [
        { "rule": "required", "message": { "zh": "MXFP4 分组大小为必填项", "en": "MXFP4 Group Size is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] },
        { "rule": "max", "value": 1000000, "type": "integer", "message": { "zh": "必须 ≤ 1e6", "en": "Must be ≤ 1e6" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "quantize_attention_action",
      "label": { "zh": "KV cache 量化", "en": "Attention KV Cache Quantization" },
      "control": "multi-select",
      "dataType": "string[]",
      "default": ["INT8"],
      "group": { "zh": "量化", "en": "Quantization" },
      "tooltip": { "zh": "Attention KV Cache 的量化策略（可多选；每个取值单独仿真，结果区生成多用例对比）。", "en": "Quantization for attention KV cache (multi-select; each value runs independently and yields a multi-case comparison)." },
      "optionSource": { "type": "inline", "values": QUANTIZE_ATTENTION_OPTIONS },
      "validation": [
        { "rule": "required", "message": { "zh": "KV cache 量化为必填项", "en": "Attention KV Cache Quantization is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "tp_sizes",
      "label": { "zh": "TP 搜索尺寸", "en": "TP Search Sizes" },
      "control": "text",
      "dataType": "string",
      "default": null,
      "group": { "zh": "搜索", "en": "Search" },
      "tooltip": { "zh": "逗号分隔的张量并行搜索尺寸（如 1,2,4），每个 ≤ num_devices；每个尺寸都会评估并在结果扫描表中对比各组合；留空表示不搜索。", "en": "Comma-separated TP search sizes (e.g. 1,2,4), each ≤ num_devices; every size is evaluated and compared in the result sweep table; empty = no search." },
      "placeholder": { "zh": "如 1,2,4", "en": "e.g. 1,2,4" },
      "validation": [
        { "rule": "validator", "value": "lteNumDevices", "message": { "zh": "所有值不得超过 num_devices", "en": "All values must not exceed num_devices" }, "dependsOn": ["num_devices", "tp_sizes"], "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "ep_sizes",
      "label": { "zh": "EP 搜索尺寸", "en": "EP Search Sizes" },
      "control": "text",
      "dataType": "string",
      "default": null,
      "group": { "zh": "搜索", "en": "Search" },
      "tooltip": { "zh": "逗号分隔的专家并行搜索尺寸（如 1,2,4），每个 ≤ num_devices；仅对 MoE 模型生效，每个尺寸都会评估并在结果扫描表中对比各组合；留空表示不搜索。", "en": "Comma-separated EP search sizes (e.g. 1,2,4), each ≤ num_devices; only effective for MoE models; every size is evaluated and compared in the result sweep table; empty = no search." },
      "placeholder": { "zh": "如 1,2,4", "en": "e.g. 1,2,4" },
      "validation": [
        { "rule": "validator", "value": "lteNumDevices", "message": { "zh": "所有值不得超过 num_devices", "en": "All values must not exceed num_devices" }, "dependsOn": ["num_devices", "ep_sizes"], "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "moe_dp_sizes",
      "label": { "zh": "MOE-DP 搜索尺寸", "en": "MOE-DP Search Sizes" },
      "control": "text",
      "dataType": "string",
      "default": null,
      "group": { "zh": "搜索", "en": "Search" },
      "tooltip": { "zh": "逗号分隔的 MoE 数据并行搜索尺寸（如 1,2），每个 ≤ num_devices；仅对 MoE 模型生效，每个尺寸都会评估并在结果扫描表中对比各组合；留空表示不搜索。", "en": "Comma-separated MoE-DP search sizes (e.g. 1,2), each ≤ num_devices; only effective for MoE models; every size is evaluated and compared in the result sweep table; empty = no search." },
      "placeholder": { "zh": "如 1,2", "en": "e.g. 1,2" },
      "validation": [
        { "rule": "validator", "value": "lteNumDevices", "message": { "zh": "所有值不得超过 num_devices", "en": "All values must not exceed num_devices" }, "dependsOn": ["num_devices", "moe_dp_sizes"], "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "ttft_limits",
      "label": { "zh": "TTFT 约束(ms)", "en": "TTFT Limit (ms)" },
      "control": "text",
      "dataType": "string",
      "default": "",
      "group": { "zh": "约束", "en": "Constraints" },
      "tooltip": { "zh": "TTFT 约束（毫秒）；逗号分隔多值时每个取值单独仿真，结果区生成多用例对比；>0 或 inf 表示无约束。", "en": "TTFT limit in milliseconds; with a comma-list each value runs independently and yields a multi-case comparison; >0 or inf for unlimited." },
      "placeholder": { "zh": "如 200,500 或留空", "en": "e.g. 200,500 or empty" },
      "validation": [
        { "rule": "validator", "value": "positiveOrInf", "message": { "zh": "每项必须为正数或 inf", "en": "Each value must be positive or inf" }, "trigger": ["blur"] }
      ]
    },
    {
      "id": "tpot_limits",
      "label": { "zh": "TPOT 约束(ms)", "en": "TPOT Limit (ms)" },
      "control": "text",
      "dataType": "string",
      "default": "",
      "group": { "zh": "约束", "en": "Constraints" },
      "tooltip": { "zh": "TPOT 约束（毫秒）；逗号分隔多值时每个取值单独仿真，结果区生成多用例对比；>0 或 inf 表示无约束。", "en": "TPOT limit in milliseconds; with a comma-list each value runs independently and yields a multi-case comparison; >0 or inf for unlimited." },
      "placeholder": { "zh": "如 20,50 或留空", "en": "e.g. 20,50 or empty" },
      "validation": [
        { "rule": "validator", "value": "positiveOrInf", "message": { "zh": "每项必须为正数或 inf", "en": "Each value must be positive or inf" }, "trigger": ["blur"] }
      ]
    },
    {
      "id": "max_batched_tokens",
      "label": { "zh": "单步最大 token 数", "en": "Max Batched Tokens" },
      "control": "number",
      "dataType": "integer",
      "default": 8192,
      "group": { "zh": "约束", "en": "Constraints" },
      "tooltip": { "zh": "单步最大 token 数（>0，≤1e6）。", "en": "Maximum tokens per step (>0, ≤1e6)." },
      "validation": [
        { "rule": "required", "message": { "zh": "单步最大 token 数为必填项", "en": "Max Batched Tokens is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] },
        { "rule": "max", "value": 1000000, "type": "integer", "message": { "zh": "必须 ≤ 1e6", "en": "Must be ≤ 1e6" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "batch_range",
      "label": { "zh": "批大小范围", "en": "Batch Size Range" },
      "control": "text",
      "dataType": "string",
      "default": null,
      "group": { "zh": "约束", "en": "Constraints" },
      "tooltip": { "zh": "批大小范围，格式 start,end 或单值（min≤max）。", "en": "Batch size range, format start,end or a single value (min≤max)." },
      "placeholder": { "zh": "如 1,8", "en": "e.g. 1,8" },
      "validation": [
        { "rule": "validator", "value": "batchRange", "message": { "zh": "格式应为 1-2 个值且 min≤max", "en": "Should be 1-2 values with min≤max" }, "trigger": ["blur"] }
      ]
    },
    {
      "id": "serving_cost",
      "label": { "zh": "服务成本", "en": "Serving Cost" },
      "control": "number",
      "dataType": "number",
      "default": 0,
      "group": { "zh": "成本", "en": "Cost" },
      "tooltip": { "zh": "服务成本（≥0）。", "en": "Serving cost (≥0)." },
      "validation": [
        { "rule": "required", "message": { "zh": "服务成本为必填项", "en": "Serving Cost is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 0, "type": "number", "message": { "zh": "必须 ≥ 0", "en": "Must be ≥ 0" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "disagg",
      "label": { "zh": "分离部署模式", "en": "Disaggregated Mode" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "模式", "en": "Mode" },
      "tooltip": { "zh": "启用分离部署模式（Prefill+Decode 分离，与 PD 配比优化互斥）。", "en": "Enable disaggregated mode (Prefill+Decode separation, mutually exclusive with PD-ratio optimization)." },
      "conditions": {
        "enabled": { "not": { "field": "enable_optimize_prefill_decode_ratio", "op": "isTrue" } }
      }
    },
    {
      "id": "jobs",
      "label": { "zh": "并行作业数", "en": "Parallel Jobs" },
      "control": "number",
      "dataType": "integer",
      "default": 8,
      "group": { "zh": "执行", "en": "Execution" },
      "tooltip": { "zh": "并行作业数（>0，≤1e6）。", "en": "Number of parallel jobs (>0, ≤1e6)." },
      "validation": [
        { "rule": "required", "message": { "zh": "并行作业数为必填项", "en": "Parallel Jobs is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] },
        { "rule": "max", "value": 1000000, "type": "integer", "message": { "zh": "必须 ≤ 1e6", "en": "Must be ≤ 1e6" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "max_search_combinations",
      "label": { "zh": "最大搜索组合数", "en": "Max Search Combinations" },
      "control": "number",
      "dataType": "integer",
      "default": 100,
      "group": { "zh": "执行", "en": "Execution" },
      "tooltip": { "zh": "当 TP/EP/MOE-DP/MTP 搜索组合数超过该值时输出警告（≥0；设为 0 关闭警告）。默认 100。", "en": "Warn when TP/EP/MOE-DP/MTP search combinations exceed this value (≥0; set 0 to disable). Default 100." },
      "validation": [
        { "rule": "required", "message": { "zh": "最大搜索组合数为必填项", "en": "Max Search Combinations is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 0, "type": "integer", "message": { "zh": "必须为非负整数", "en": "Must be a non-negative integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "concurrency_search_strategy",
      "label": { "zh": "并发搜索策略", "en": "Concurrency Search Strategy" },
      "control": "select",
      "dataType": "string",
      "default": "exponential",
      "group": { "zh": "搜索", "en": "Search" },
      "tooltip": { "zh": "并发度搜索策略。", "en": "Concurrency search strategy." },
      "validation": [
        { "rule": "required", "message": { "zh": "并发搜索策略为必填项", "en": "Concurrency Search Strategy is required" }, "trigger": ["change", "blur"] }
      ],
      "optionSource": {
        "type": "inline",
        "values": [
          { "value": "exponential", "label": { "zh": "exponential", "en": "exponential" } },
          { "value": "linear_exponential", "label": { "zh": "linear_exponential", "en": "linear_exponential" } }
        ]
      }
    },
    {
      "id": "dump_original_results",
      "label": { "zh": "导出原始结果", "en": "Dump Original Results" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "输出", "en": "Output" },
      "tooltip": { "zh": "导出原始优化结果。", "en": "Export original optimization results." }
    },
    {
      "id": "chrome_trace",
      "label": { "zh": "Chrome trace 导出", "en": "Chrome Trace Export" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "调试", "en": "Debug" },
      "tooltip": { "zh": "开启后每个用例导出 Chrome trace，完成后可在结果页下载。", "en": "Export a Chrome trace per case; downloadable from the result page." }
    },
    {
      "id": "enable_optimize_prefill_decode_ratio",
      "label": { "zh": "启用 PD 配比优化", "en": "Enable Prefill-Decode Ratio Optimization" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "模式", "en": "Mode" },
      "tooltip": { "zh": "启用 Prefill-Decode 配比优化（与 disagg 互斥）。仅当分离部署模式关闭时可编辑。", "en": "Enable prefill-decode ratio optimization (mutually exclusive with disagg). Only editable when disaggregated mode is off." },
      "conditions": {
        "enabled": { "not": { "field": "disagg", "op": "isTrue" } }
      }
    },
    {
      "id": "prefill_devices_per_instance",
      "label": { "zh": "Prefill 实例设备数", "en": "Prefill Devices Per Instance" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "PD 配比", "en": "PD Ratio" },
      "tooltip": { "zh": "Prefill 实例设备数（>0）。此字段仅在 enable_optimize_prefill_decode_ratio 启用时有效", "en": "Prefill devices per instance (>0). This field is only effective when enable_optimize_prefill_decode_ratio is enabled" },
      "validation": [
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "decode_devices_per_instance",
      "label": { "zh": "Decode 实例设备数", "en": "Decode Devices Per Instance" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "PD 配比", "en": "PD Ratio" },
      "tooltip": { "zh": "Decode 实例设备数（>0）。此字段仅在 enable_optimize_prefill_decode_ratio 启用时有效", "en": "Decode devices per instance (>0). This field is only effective when enable_optimize_prefill_decode_ratio is enabled" },
      "validation": [
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "image_batch_size",
      "label": { "zh": "图像批大小", "en": "Image Batch Size" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "多模态", "en": "Multimodal" },
      "tooltip": { "zh": "多模态图像批大小（≥1，≤1e6；省略且指定图像高度时运行时回退为 batch_size）。", "en": "Multimodal image batch size (≥1, ≤1e6; falls back to batch_size at runtime if omitted while image height is set)." },
      "validation": [
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] },
        { "rule": "max", "value": 1000000, "type": "integer", "message": { "zh": "必须 ≤ 1e6", "en": "Must be ≤ 1e6" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "image_height",
      "label": { "zh": "图像高度", "en": "Image Height" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "多模态", "en": "Multimodal" },
      "tooltip": { "zh": "多模态图像高度（≥1，≤1e6）。", "en": "Multimodal image height (≥1, ≤1e6)." },
      "validation": [
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] },
        { "rule": "max", "value": 1000000, "type": "integer", "message": { "zh": "必须 ≤ 1e6", "en": "Must be ≤ 1e6" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "image_width",
      "label": { "zh": "图像宽度", "en": "Image Width" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "多模态", "en": "Multimodal" },
      "tooltip": { "zh": "多模态图像宽度（≥1，≤1e6）。", "en": "Multimodal image width (≥1, ≤1e6)." },
      "validation": [
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] },
        { "rule": "max", "value": 1000000, "type": "integer", "message": { "zh": "必须 ≤ 1e6", "en": "Must be ≤ 1e6" }, "trigger": ["change", "blur"] }
      ]
    }
  ],
  validators: { stringValid, prefixCacheRate, lteNumDevices, batchRange, validParallelCombo, positiveOrInf, effectiveLenGe1, pdRatioMutexDisagg, mtpAcceptanceRatesPositive, mtpTokensVsAcceptanceRate },
}

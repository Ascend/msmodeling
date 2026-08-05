/* eslint-disable */
// AUTO-CONVERTED from JSON. This .ts is the source of truth (data + inlined
// validators). A build step regenerates the data-only .json for the backend
// (validators are stripped by JSON.stringify). Do not edit the .json by hand.
import {
  stringValid,
  prefixCacheRate,
  lteNumDevices,
  positiveIntegerIfProvided,
  dividesNumDevices,
  productEqNumDevices,
  moeProductEqNumDevices,
  perLayerProductEqNumDevices,
  sharedExpertMutex,
  effectiveLenGe1,
  profilingDbRequired,
  QUANTIZE_LINEAR_OPTIONS,
  QUANTIZE_ATTENTION_OPTIONS,
  REMOTE_SOURCE_OPTIONS,
  CLI_LOG_LEVEL_OPTIONS,
} from "./_validators"

export default {
"$schema": "form-schema/v1",
  "moduleId": "text_generate",
  "title": { "zh": "文本生成", "en": "Text Generation" },
  "runner": "ModelRunner",
  "version": "1.10.0",
  "optionSourceRegistry": {
    "devices": { "endpoint": "/api/options/devices", "cache": "session" }
  },
  "formValidation": [
    { "rule": "validator", "value": "productEqNumDevices", "message": { "zh": "tp_size × dp_size × pp_size 必须等于 num_devices", "en": "tp_size × dp_size × pp_size must equal num_devices" }, "dependsOn": ["tp_size", "dp_size", "pp_size", "num_devices"] },
    { "rule": "validator", "value": "moeProductEqNumDevices", "message": { "zh": "moe_tp_size × moe_dp_size × ep_size 必须等于 num_devices", "en": "moe_tp_size × moe_dp_size × ep_size must equal num_devices" }, "dependsOn": ["ep_size", "moe_tp_size", "moe_dp_size", "num_devices"] },
    { "rule": "validator", "value": "perLayerProductEqNumDevices", "message": { "zh": "每层 TP × DP 必须等于 num_devices", "en": "Per-layer TP × DP must equal num_devices" }, "dependsOn": ["o_proj_tp_size", "o_proj_dp_size", "mlp_tp_size", "mlp_dp_size", "lmhead_tp_size", "lmhead_dp_size", "num_devices"] },
    { "rule": "validator", "value": "sharedExpertMutex", "message": { "zh": "共享专家相关参数互斥或需 ep_size > 1", "en": "Shared-expert options are mutually exclusive or require ep_size > 1" }, "dependsOn": ["enable_shared_expert_tp", "host_external_shared_experts", "ep_size"] },
    { "rule": "validator", "value": "effectiveLenGe1", "message": { "zh": "有效请求长度必须 ≥ 1", "en": "Effective query length must be ≥ 1" }, "dependsOn": ["query_length", "prefix_cache_hit_rate"] }
  ],  "groups": [
    { "label": { "zh": "专家并行", "en": "Expert" }, "defaultCollapsed": true },
    { "label": { "zh": "高级并行", "en": "Advanced Parallelism" }, "defaultCollapsed": true },
    { "label": { "zh": "多模态", "en": "Multimodal" }, "defaultCollapsed": true },
    { "label": { "zh": "调试", "en": "Debug" }, "defaultCollapsed": true },
    { "label": { "zh": "模型", "en": "Model" }, "defaultCollapsed": true }
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
      "label": { "zh": "目标设备", "en": "Target Device" },
      "control": "multi-select",
      "dataType": "string[]",
      "default": ["ATLAS_350_425T_112G"],
      "group": { "zh": "通用", "en": "General" },
      "tooltip": { "zh": "选择用于仿真的设备 Profile（可多选；每个取值单独仿真，结果区生成多用例对比）。", "en": "Device profile(s) to simulate on (multi-select; each value runs independently and yields a multi-case comparison)." },
      "placeholder": { "zh": "请选择设备型号", "en": "Select device model(s)" },
      "optionSource": { "type": "dynamic", "name": "devices" },
      "validation": [
        { "rule": "required", "message": { "zh": "目标设备为必选项", "en": "Target Device is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "num_devices",
      "label": { "zh": "设备数量", "en": "Number of Devices" },
      "control": "number",
      "dataType": "integer",
      "default": 1,
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
      "default": 0.0,
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
      "optionSource": { "type": "inline", "values": CLI_LOG_LEVEL_OPTIONS },
      "validation": [
        { "rule": "required", "message": { "zh": "日志级别为必填项", "en": "Log Level is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "num_queries",
      "label": { "zh": "并行请求数", "en": "Number of Queries" },
      "control": "text",
      "dataType": "string",
      "default": "1",
      "group": { "zh": "请求", "en": "Request" },
      "tooltip": { "zh": "并发推理的请求数量；逗号分隔多值时每个取值单独仿真，结果区生成多用例对比（如 1,4,8）。", "en": "Concurrently inferred query count; with a comma-list each value runs independently and yields a multi-case comparison (e.g. 1,4,8)." },
      "placeholder": { "zh": "如 1,4,8", "en": "e.g. 1,4,8" },
      "validation": [
        { "rule": "required", "message": { "zh": "并行请求数为必填项", "en": "Number of Queries is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "query_length",
      "label": { "zh": "输入序列长度", "en": "Input Sequence Length" },
      "control": "number",
      "dataType": "integer",
      "default": 1,
      "group": { "zh": "请求", "en": "Request" },
      "tooltip": { "zh": "单条请求的输入 token 长度 + mtp token 数（≥1）。", "en": "Input token length per query + mtp token count (≥1)." },
      "validation": [
        { "rule": "required", "message": { "zh": "输入序列长度为必填项", "en": "Input sequence length is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "context_length",
      "label": { "zh": "上下文长度", "en": "Context Length" },
      "control": "number",
      "dataType": "integer",
      "default": 4500,
      "group": { "zh": "请求", "en": "Request" },
      "tooltip": { "zh": "上下文窗口大小（0 表示不限制）。", "en": "Context window size (0 = unlimited)." },
      "validation": [
        { "rule": "required", "message": { "zh": "上下文长度为必填项", "en": "Context Length is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 0, "type": "integer", "message": { "zh": "必须 ≥ 0", "en": "Must be ≥ 0" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "decode",
      "label": { "zh": "自回归解码", "en": "Autoregressive Decode" },
      "control": "switch",
      "dataType": "boolean",
      "default": true,
      "group": { "zh": "请求", "en": "Request" },
      "tooltip": { "zh": "启用自回归解码阶段。", "en": "Enable the autoregressive decode stage." }
    },
    {
      "id": "prefix_cache_hit_rate",
      "label": { "zh": "前缀缓存命中率", "en": "Prefix Cache Hit Rate" },
      "control": "number",
      "dataType": "number",
      "default": 0.0,
      "group": { "zh": "请求", "en": "Request" },
      "tooltip": { "zh": "模拟的前缀缓存命中率 [0, 1)。", "en": "Simulated prefix cache hit rate [0, 1)." },
      "validation": [
        { "rule": "required", "message": { "zh": "前缀缓存命中率为必填项", "en": "Prefix Cache Hit Rate is required" }, "trigger": ["change", "blur"] },
        { "rule": "validator", "value": "prefixCacheRate", "message": { "zh": "必须在 [0, 1) 区间", "en": "Must be in [0, 1)" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "num_mtp_tokens",
      "label": { "zh": "MTP token 数", "en": "MTP Token Count" },
      "control": "number",
      "dataType": "integer",
      "default": 0,
      "group": { "zh": "请求", "en": "Request" },
      "tooltip": { "zh": "Multi-Text Prediction token 数量（≥0）。", "en": "Multi-Text Prediction token count (≥0)." },
      "validation": [
        { "rule": "required", "message": { "zh": "MTP token 数为必填项", "en": "MTP Token Count is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 0, "type": "integer", "message": { "zh": "必须 ≥ 0", "en": "Must be ≥ 0" }, "trigger": ["change", "blur"] }
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
      "id": "quantize_lmhead",
      "label": { "zh": "量化 LM Head", "en": "Quantize LM Head" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "量化", "en": "Quantization" },
      "tooltip": { "zh": "对 LM Head 层应用量化。", "en": "Apply quantization to LM Head layer." }
    },
    {
      "id": "mxfp4_group_size",
      "label": { "zh": "MXFP4 分组大小", "en": "MXFP4 Group Size" },
      "control": "number",
      "dataType": "integer",
      "default": 32,
      "group": { "zh": "量化", "en": "Quantization" },
      "tooltip": { "zh": "MXFP4 量化的分组大小（>0）。此字段仅在 quantize_linear_action 或 quantize_non_expert_linear_action 包含 MXFP4 时有效", "en": "MXFP4 quantization group size (>0). This field is only effective when quantize_linear_action or quantize_non_expert_linear_action contains MXFP4" },
      "validation": [
        { "rule": "required", "message": { "zh": "MXFP4 分组大小为必填项", "en": "MXFP4 Group Size is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "quantize_attention_action",
      "label": { "zh": "KV cache 量化", "en": "Attention KV Cache Quantization" },
      "control": "multi-select",
      "dataType": "string[]",
      "default": ["DISABLED"],
      "group": { "zh": "量化", "en": "Quantization" },
      "tooltip": { "zh": "Attention KV Cache 的量化策略（可多选；每个取值单独仿真，结果区生成多用例对比）。", "en": "Quantization for attention KV cache (multi-select; each value runs independently and yields a multi-case comparison)." },
      "optionSource": { "type": "inline", "values": QUANTIZE_ATTENTION_OPTIONS },
      "validation": [
        { "rule": "required", "message": { "zh": "KV cache 量化为必填项", "en": "Attention KV Cache Quantization is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "tp_size",
      "label": { "zh": "张量并行", "en": "Tensor Parallel Size" },
      "control": "text",
      "dataType": "string",
      "default": "1",
      "group": { "zh": "并行", "en": "Parallelism" },
      "tooltip": { "zh": "张量并行度；逗号分隔多值时每个取值单独仿真，结果区生成多用例对比（如 1,2,4）。", "en": "Tensor-parallel degree; with a comma-list each value runs independently and yields a multi-case comparison (e.g. 1,2,4)." },
      "placeholder": { "zh": "如 1,2,4", "en": "e.g. 1,2,4" },
      "validation": [
        { "rule": "required", "message": { "zh": "张量并行为必填项", "en": "Tensor Parallel Size is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "pp_size",
      "label": { "zh": "流水线并行", "en": "Pipeline Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": 1,
      "group": { "zh": "并行", "en": "Parallelism" },
      "tooltip": { "zh": "流水线并行度（≥1）。tp_size × dp_size × pp_size 必须等于 num_devices。", "en": "Pipeline-parallel degree (≥1). tp_size × dp_size × pp_size must equal num_devices." },
      "validation": [
        { "rule": "required", "message": { "zh": "流水线并行为必填项", "en": "Pipeline Parallel Size is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "dp_size",
      "label": { "zh": "数据并行", "en": "Data Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "并行", "en": "Parallelism" },
      "tooltip": { "zh": "数据并行度，留空时自动按 num_devices // (tp_size × pp_size) 推导。", "en": "Data-parallel degree; auto-derived as num_devices // (tp_size × pp_size) when left empty." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "ep_size",
      "label": { "zh": "专家并行", "en": "Expert Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": 1,
      "group": { "zh": "专家并行", "en": "Expert" },
      "tooltip": { "zh": "专家并行度（MoE 模型，≥1）。", "en": "Expert-parallel degree (MoE models, ≥1)." },
      "validation": [
        { "rule": "required", "message": { "zh": "专家并行为必填项", "en": "Expert Parallel Size is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "o_proj_tp_size",
      "label": { "zh": "o_proj 张量并行", "en": "O-proj Tensor Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "高级并行", "en": "Advanced Parallelism" },
      "tooltip": { "zh": "O-proj 层张量并行度，留空时继承 tp_size。", "en": "O-proj layer tensor-parallel degree; inherits tp_size when empty." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "o_proj_dp_size",
      "label": { "zh": "o_proj 数据并行", "en": "O-proj Data Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "高级并行", "en": "Advanced Parallelism" },
      "tooltip": { "zh": "O-proj 层数据并行度，留空时自动计算。", "en": "O-proj layer data-parallel degree; auto-calculated when empty." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "mlp_tp_size",
      "label": { "zh": "MLP 张量并行", "en": "MLP Tensor Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "高级并行", "en": "Advanced Parallelism" },
      "tooltip": { "zh": "MLP 层张量并行度，留空时继承 tp_size。", "en": "MLP layer tensor-parallel degree; inherits tp_size when empty." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "mlp_dp_size",
      "label": { "zh": "MLP 数据并行", "en": "MLP Data Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "高级并行", "en": "Advanced Parallelism" },
      "tooltip": { "zh": "MLP 层数据并行度，留空时自动计算。", "en": "MLP layer data-parallel degree; auto-calculated when empty." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "lmhead_tp_size",
      "label": { "zh": "LM Head 张量并行", "en": "LM Head Tensor Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "高级并行", "en": "Advanced Parallelism" },
      "tooltip": { "zh": "LM Head 层张量并行度，留空时继承 tp_size。", "en": "LM Head layer tensor-parallel degree; inherits tp_size when empty." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "lmhead_dp_size",
      "label": { "zh": "LM Head 数据并行", "en": "LM Head Data Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "高级并行", "en": "Advanced Parallelism" },
      "tooltip": { "zh": "LM Head 层数据并行度，留空时自动计算。", "en": "LM Head layer data-parallel degree; auto-calculated when empty." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "vision_tp_size",
      "label": { "zh": "Vision 张量并行", "en": "Vision Tensor Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": 1,
      "group": { "zh": "高级并行", "en": "Advanced Parallelism" },
      "tooltip": { "zh": "视觉编码张量并行度，不得超过 num_devices 且 num_devices 必须能被其整除。", "en": "Vision tensor-parallel degree; must be ≤ num_devices and num_devices must be divisible by it." },
      "validation": [
        { "rule": "required", "message": { "zh": "Vision 张量并行为必填项", "en": "Vision Tensor Parallel Size is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] },
        { "rule": "validator", "value": "lteNumDevices", "message": { "zh": "vision_tp_size 不得超过 num_devices", "en": "vision_tp_size must not exceed num_devices" }, "dependsOn": ["num_devices", "vision_tp_size"], "trigger": ["change", "blur"] },
        { "rule": "validator", "value": "dividesNumDevices", "message": { "zh": "num_devices 必须能被 vision_tp_size 整除", "en": "num_devices must be divisible by vision_tp_size" }, "dependsOn": ["num_devices", "vision_tp_size"], "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "moe_tp_size",
      "label": { "zh": "MoE 张量并行", "en": "MoE Tensor Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "专家并行", "en": "Expert" },
      "tooltip": { "zh": "MoE 专家张量并行度，留空时自动计算。", "en": "MoE expert tensor-parallel degree; auto-calculated when empty." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "moe_dp_size",
      "label": { "zh": "MoE 数据并行", "en": "MoE Data Parallel Size" },
      "control": "number",
      "dataType": "integer",
      "default": 1,
      "group": { "zh": "专家并行", "en": "Expert" },
      "tooltip": { "zh": "MoE 专家数据并行度（≥1）。", "en": "MoE expert data-parallel degree (≥1)." },
      "validation": [
        { "rule": "required", "message": { "zh": "MoE 数据并行为必填项", "en": "MoE Data Parallel Size is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "enable_redundant_experts",
      "label": { "zh": "启用冗余专家", "en": "Enable Redundant Experts" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "专家并行", "en": "Expert" },
      "tooltip": { "zh": "启用冗余专家（仅 MoE 模型）。", "en": "Enable redundant experts (MoE models only)." }
    },
    {
      "id": "enable_shared_expert_tp",
      "label": { "zh": "启用共享专家 TP", "en": "Enable Shared Expert TP" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "专家并行", "en": "Expert" },
      "tooltip": { "zh": "对共享专家启用张量并行。此字段要求 ep_size > 1，且与 host_external_shared_experts 互斥（不能同时启用）", "en": "Tensor-parallelize the shared expert. This field requires ep_size > 1 and is mutually exclusive with host_external_shared_experts (cannot be enabled simultaneously)" }
    },
    {
      "id": "enable_external_shared_experts",
      "label": { "zh": "启用外部共享专家", "en": "Enable External Shared Experts" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "专家并行", "en": "Expert" },
      "tooltip": { "zh": "启用外部共享专家（仅 MoE 模型）。", "en": "Enable external shared experts (MoE models only)." }
    },
    {
      "id": "host_external_shared_experts",
      "label": { "zh": "宿主外部共享专家", "en": "Host External Shared Experts" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "专家并行", "en": "Expert" },
      "tooltip": { "zh": "将外部共享专家宿主于计算流。此字段与 enable_shared_expert_tp 互斥（不能同时启用）", "en": "Host external shared experts on the compute stream. This field is mutually exclusive with enable_shared_expert_tp (cannot be enabled simultaneously)" }
    },
    {
      "id": "word_embedding_tp",
      "label": { "zh": "词嵌入 TP 模式", "en": "Word Embedding TP Mode" },
      "control": "select",
      "dataType": "string",
      "default": null,
      "group": { "zh": "并行", "en": "Parallelism" },
      "tooltip": { "zh": "词嵌入张量并行模式。", "en": "Word embedding tensor parallelism mode." },
      "optionSource": {
        "type": "inline",
        "values": [
          { "value": "col", "label": { "zh": "col（列并行）", "en": "col (column-parallel)" } },
          { "value": "row", "label": { "zh": "row（行并行）", "en": "row (row-parallel)" } }
        ]
      }
    },
    {
      "id": "image_batch_size",
      "label": { "zh": "图像批大小", "en": "Image Batch Size" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "多模态", "en": "Multimodal" },
      "tooltip": { "zh": "每次处理的图像数量（多模态模型）。", "en": "Number of images processed per batch (multimodal models)." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "image_height",
      "label": { "zh": "图像高度", "en": "Image Height" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "多模态", "en": "Multimodal" },
      "tooltip": { "zh": "输入图像高度（多模态模型）。", "en": "Input image height (multimodal models)." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "image_width",
      "label": { "zh": "图像宽度", "en": "Image Width" },
      "control": "number",
      "dataType": "integer",
      "default": null,
      "group": { "zh": "多模态", "en": "Multimodal" },
      "tooltip": { "zh": "输入图像宽度（多模态模型）。", "en": "Input image width (multimodal models)." },
      "validation": [
        { "rule": "validator", "value": "positiveIntegerIfProvided", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "num_hidden_layers_override",
      "label": { "zh": "覆盖隐藏层数", "en": "Num Hidden Layers Override" },
      "control": "number",
      "dataType": "integer",
      "default": 0,
      "group": { "zh": "调试", "en": "Debug" },
      "tooltip": { "zh": "覆盖模型的隐藏层数量（0 = 不覆盖）。", "en": "Override the number of hidden layers (0 = no override)." },
      "validation": [
        { "rule": "required", "message": { "zh": "覆盖隐藏层数为必填项", "en": "Num Hidden Layers Override is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 0, "type": "integer", "message": { "zh": "必须 ≥ 0", "en": "Must be ≥ 0" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "disable_repetition",
      "label": { "zh": "禁用重复层优化", "en": "Disable Repetition Reuse" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "调试", "en": "Debug" },
      "tooltip": { "zh": "关闭后不再自动检测并利用模型中结构相同的重复层进行计算共享，每层独立建模，保留 Transformer 原始行为；仿真耗时会增加。", "en": "Disable automatic detection and reuse of structurally identical transformer layers. Each layer is modeled independently, preserving the original transformer behavior at increased runtime cost." }
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
      "id": "dump_input_shapes",
      "label": { "zh": "按输入形状分组", "en": "Group by Input Shapes" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "调试", "en": "Debug" },
      "tooltip": { "zh": "开启后算子平均表按输入 tensor 形状分组统计，显示 Input Shapes 列。", "en": "When enabled, the operator average table groups statistics by input tensor shapes and shows an Input Shapes column." }
    },
    {
      "id": "dump_op_bound_results",
      "label": { "zh": "显示算子瓶颈分析", "en": "Show Operator Bound Analysis" },
      "control": "switch",
      "dataType": "boolean",
      "default": false,
      "group": { "zh": "调试", "en": "Debug" },
      "tooltip": { "zh": "开启后算子平均表显示每个算子的性能瓶颈（Memory/Communication/MMA/GP）及百分比。", "en": "When enabled, the operator average table shows per-operator performance bound (Memory/Communication/MMA/GP) and percentages." }
    },
    {
      "id": "remote_source",
      "label": { "zh": "模型远程来源", "en": "Model Remote Source" },
      "control": "select",
      "dataType": "string",
      "default": "huggingface",
      "group": { "zh": "模型", "en": "Model" },
      "tooltip": { "zh": "模型加载的远程来源。", "en": "Remote source for model loading." },
      "optionSource": { "type": "inline", "values": REMOTE_SOURCE_OPTIONS },
      "validation": [
        { "rule": "required", "message": { "zh": "模型远程来源为必填项", "en": "Model Remote Source is required" }, "trigger": ["change", "blur"] }
      ]
    }
  ],
  validators: {
    stringValid,
    prefixCacheRate,
    lteNumDevices,
    positiveIntegerIfProvided,
    dividesNumDevices,
    productEqNumDevices,
    moeProductEqNumDevices,
    perLayerProductEqNumDevices,
    sharedExpertMutex,
    effectiveLenGe1,
    profilingDbRequired,
  },
}

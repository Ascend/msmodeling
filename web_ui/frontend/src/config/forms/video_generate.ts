/* eslint-disable */
// AUTO-CONVERTED from JSON. This .ts is the source of truth (data + inlined
// validators). A build step regenerates the data-only .json for the backend
// (validators are stripped by JSON.stringify). Do not edit the .json by hand.
import { intRangeOrdered, divisibleBy, cfgParallelRequiresWorldSize2, QUANTIZE_LINEAR_OPTIONS, CLI_LOG_LEVEL_OPTIONS } from "./_validators"

export default {
"$schema": "form-schema/v1",
  "moduleId": "video_generate",
  "title": { "zh": "视频生成", "en": "Video Generation" },
  "runner": "VideoGenerateRunner",
  "version": "1.4.5",
  "optionSourceRegistry": {
    "devices": { "endpoint": "/api/options/devices", "cache": "session" }
  },
  "formValidation": [
    {
      "rule": "validator",
      "value": "cfgParallelRequiresWorldSize2",
      "message": { "zh": "启用 CFG 并行时 world_size 必须 ≥ 2", "en": "cfg_parallel requires world_size ≥ 2" },
      "dependsOn": ["use_cfg", "cfg_parallel", "world_size"]
    }
  ],  "groups": [
    { "label": { "zh": "图像", "en": "Image" }, "defaultCollapsed": true },
    { "label": { "zh": "CFG", "en": "CFG" }, "defaultCollapsed": true },
    { "label": { "zh": "缓存", "en": "Cache" }, "defaultCollapsed": true },
    { "label": { "zh": "调试", "en": "Debug" }, "defaultCollapsed": true }
  ],
  "fields": [
    {
      "id": "device",
      "label": { "zh": "设备类型", "en": "Target Device" },
      "control": "multi-select",
      "dataType": "string[]",
      "default": ["ATLAS_350_425T_112G"],
      "group": { "zh": "通用", "en": "General" },
      "tooltip": { "zh": "选择用于仿真的设备 Profile（可多选；每个取值单独仿真，结果区生成多用例对比）。", "en": "Device profile(s) to simulate on (multi-select; each value runs independently and yields a multi-case comparison)." },
      "placeholder": { "zh": "请选择设备型号", "en": "Select device model(s)" },
      "optionSource": { "type": "dynamic", "name": "devices" },
      "validation": [
        { "rule": "required", "message": { "zh": "设备类型为必选项", "en": "Target Device is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "model_id",
      "label": { "zh": "模型 ID", "en": "Model ID" },
      "control": "text",
      "dataType": "string",
      "default": "tests/assets/model_config/Wan2.2-T2V-A14B-Diffusers",
      "group": { "zh": "通用", "en": "General" },
      "tooltip": { "zh": "待仿真模型的标准 HuggingFace 名称（组织/模型，如 Qwen/Qwen3-32B）或本地路径。", "en": "Standard HuggingFace model name (org/model, e.g. Qwen/Qwen3-32B) or local path." },
      "placeholder": { "zh": "如 Qwen/Qwen3-32B", "en": "e.g. Qwen/Qwen3-32B" },
      "validation": [
        { "rule": "required", "message": { "zh": "模型 ID 为必填项", "en": "Model ID is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "dtype",
      "label": { "zh": "数据类型", "en": "Data Type" },
      "control": "select",
      "dataType": "string",
      "default": "float16",
      "group": { "zh": "通用", "en": "General" },
      "tooltip": { "zh": "计算数据类型。", "en": "Computation data type." },
      "validation": [
        { "rule": "required", "message": { "zh": "数据类型为必填项", "en": "Data Type is required" }, "trigger": ["change", "blur"] }
      ],
      "optionSource": {
        "type": "inline",
        "values": [
          { "value": "float16", "label": { "zh": "float16", "en": "float16" } },
          { "value": "float32", "label": { "zh": "float32", "en": "float32" } },
          { "value": "bfloat16", "label": { "zh": "bfloat16", "en": "bfloat16" } }
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
      "id": "world_size",
      "label": { "zh": "设备数", "en": "World Size" },
      "control": "number",
      "dataType": "integer",
      "default": 8,
      "group": { "zh": "并行", "en": "Parallelism" },
      "tooltip": { "zh": "总设备数量（≥1）。", "en": "Total device count (≥1)." },
      "validation": [
        { "rule": "required", "message": { "zh": "设备数为必填项", "en": "World Size is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "ulysses_size",
      "label": { "zh": "Ulysses 并行", "en": "Ulysses Parallel Size" },
      "control": "text",
      "dataType": "string",
      "default": "4",
      "group": { "zh": "并行", "en": "Parallelism" },
      "tooltip": { "zh": "Ulysses 序列并行度数；逗号分隔多值时每个取值单独仿真，结果区生成多用例对比（如 1,2,4，须整除 world_size）。", "en": "Ulysses SP degree; with a comma-list each value runs independently and yields a multi-case comparison (e.g. 1,2,4; must divide world_size)." },
      "placeholder": { "zh": "如 1,2,4", "en": "e.g. 1,2,4" },
      "validation": [
        { "rule": "required", "message": { "zh": "Ulysses 并行为必填项", "en": "Ulysses Parallel Size is required" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "batch_size",
      "label": { "zh": "批大小", "en": "Batch Size" },
      "control": "number",
      "dataType": "integer",
      "default": 1,
      "group": { "zh": "输入", "en": "Input" },
      "tooltip": { "zh": "批大小（≥1）。", "en": "Batch size (≥1)." },
      "validation": [
        { "rule": "required", "message": { "zh": "批大小为必填项", "en": "Batch size is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "seq_len",
      "label": { "zh": "文本序列长度", "en": "Text Sequence Length" },
      "control": "number",
      "dataType": "integer",
      "default": 128,
      "group": { "zh": "输入", "en": "Input" },
      "tooltip": { "zh": "文本序列长度（≥1）。", "en": "Text sequence length (≥1)." },
      "validation": [
        { "rule": "required", "message": { "zh": "文本序列长度为必填项", "en": "Text sequence length is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "height",
      "label": { "zh": "图像高度", "en": "Image Height" },
      "control": "number",
      "dataType": "integer",
      "default": 1280,
      "group": { "zh": "图像", "en": "Image" },
      "tooltip": { "zh": "生成图像的高度（像素）。", "en": "Generated image height (pixels)." },
      "validation": [
        { "rule": "required", "message": { "zh": "图像高度为必填项", "en": "Image Height is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "width",
      "label": { "zh": "图像宽度", "en": "Image Width" },
      "control": "number",
      "dataType": "integer",
      "default": 720,
      "group": { "zh": "图像", "en": "Image" },
      "tooltip": { "zh": "生成图像的宽度（像素）。", "en": "Generated image width (pixels)." },
      "validation": [
        { "rule": "required", "message": { "zh": "图像宽度为必填项", "en": "Image Width is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "frame_num",
      "label": { "zh": "视频帧数", "en": "Frame Count" },
      "control": "number",
      "dataType": "integer",
      "default": 129,
      "group": { "zh": "视频", "en": "Video" },
      "tooltip": { "zh": "生成的视频帧数。", "en": "Number of video frames to generate." },
      "validation": [
        { "rule": "required", "message": { "zh": "视频帧数为必填项", "en": "Frame Count is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "sample_step",
      "label": { "zh": "采样步数", "en": "Sample Steps" },
      "control": "number",
      "dataType": "integer",
      "default": 50,
      "group": { "zh": "视频", "en": "Video" },
      "tooltip": { "zh": "DDPM 采样步数。", "en": "DDPM sampling steps." },
      "validation": [
        { "rule": "required", "message": { "zh": "采样步数为必填项", "en": "Sample Steps is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须为正整数", "en": "Must be a positive integer" }, "trigger": ["change", "blur"] }
      ]
    },
    {
      "id": "use_cfg",
      "label": { "zh": "启用 CFG", "en": "Enable CFG" },
      "control": "switch",
      "dataType": "boolean",
      "default": true,
      "group": { "zh": "CFG", "en": "CFG" },
      "tooltip": { "zh": "启用 Classifier-Free Guidance。", "en": "Enable Classifier-Free Guidance." }
    },
    {
      "id": "cfg_parallel",
      "label": { "zh": "CFG 并行", "en": "CFG Parallel" },
      "control": "switch",
      "dataType": "boolean",
      "default": true,
      "group": { "zh": "CFG", "en": "CFG" },
      "tooltip": { "zh": "CFG 使用并行计算（隐式要求 world_size ≥ 2）。此字段仅在 use_cfg 启用时有效", "en": "Use parallel computation for CFG (implicitly requires world_size ≥ 2). This field is only effective when use_cfg is enabled" }
    },
    {
      "id": "dit_cache",
      "label": { "zh": "启用 DiT 块缓存", "en": "Enable DiT Block Cache" },
      "control": "switch",
      "dataType": "boolean",
      "default": true,
      "group": { "zh": "缓存", "en": "Cache" },
      "tooltip": { "zh": "启用 DiT 块缓存以加速推理。", "en": "Enable DiT block cache to speed up inference." }
    },
    {
      "id": "cache_step_range",
      "label": { "zh": "缓存步区间", "en": "Cache Step Range" },
      "control": "text",
      "dataType": "string",
      "default": "20,30",
      "group": { "zh": "缓存", "en": "Cache" },
      "tooltip": { "zh": "缓存步范围（格式：start,end，end≥start）。此字段仅在 dit_cache 启用时有效且必填", "en": "Cache step range (format: start,end, end≥start). This field is only effective and required when dit_cache is enabled" },
      "validation": [
        { "rule": "required", "message": { "zh": "缓存步区间为必填项", "en": "Cache step range is required" }, "trigger": ["change", "blur"] },
        { "rule": "validator", "value": "intRangeOrdered", "message": { "zh": "格式应为 start,end 且 end≥start", "en": "Format should be start,end with end≥start" }, "trigger": ["blur"] }
      ],
      "conditions": {
        "enabled": { "field": "dit_cache", "op": "isTrue" }
      }
    },
    {
      "id": "cache_step_interval",
      "label": { "zh": "缓存更新间隔", "en": "Cache Update Interval" },
      "control": "number",
      "dataType": "integer",
      "default": 5,
      "group": { "zh": "缓存", "en": "Cache" },
      "tooltip": { "zh": "缓存更新间隔（1 表示禁用）。此字段仅在 dit_cache 启用时有效", "en": "Cache update interval (1 disables caching). This field is only effective when dit_cache is enabled" },
      "validation": [
        { "rule": "required", "message": { "zh": "缓存更新间隔为必填项", "en": "Cache Update Interval is required" }, "trigger": ["change", "blur"] },
        { "rule": "min", "value": 1, "type": "integer", "message": { "zh": "必须 ≥ 1", "en": "Must be ≥ 1" }, "trigger": ["change", "blur"] }
      ],
      "conditions": {
        "enabled": { "field": "dit_cache", "op": "isTrue" }
      }
    },
    {
      "id": "cache_block_range",
      "label": { "zh": "缓存块范围", "en": "Cache Block Range" },
      "control": "text",
      "dataType": "string",
      "default": null,
      "group": { "zh": "缓存", "en": "Cache" },
      "tooltip": { "zh": "缓存块范围（格式：start,end）。此字段仅在 dit_cache 启用时有效", "en": "Cache block range (format: start,end). This field is only effective when dit_cache is enabled" },
      "validation": [
        { "rule": "pattern", "value": "^\\s*\\d+\\s*,\\s*\\d+\\s*$", "message": { "zh": "格式应为 start,end", "en": "Format should be start,end" }, "trigger": ["blur"] }
      ]
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
      "id": "log_level",
      "label": { "zh": "日志级别", "en": "Log Level" },
      "control": "select",
      "dataType": "string",
      "default": "info",
      "group": { "zh": "调试", "en": "Debug" },
      "tooltip": { "zh": "日志输出级别。", "en": "Log output level." },
      "optionSource": { "type": "inline", "values": CLI_LOG_LEVEL_OPTIONS },
      "validation": [
        { "rule": "required", "message": { "zh": "日志级别为必填项", "en": "Log Level is required" }, "trigger": ["change", "blur"] }
      ]
    }
  ],
  validators: { intRangeOrdered, divisibleBy, cfgParallelRequiresWorldSize2 },
}

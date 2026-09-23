/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

/**
 * Shared option/enum constants + field factory helpers, co-located with the
 * form configs. Separated from _validators.ts (which retains only function
 * validators) so that high-frequency business data (model lists that change
 * per release) does not churn the validator module shared by all forms.
 *
 * Each form config (forms/*.ts) imports the options it uses and lists the
 * validators in its `validators` map; the engine (useFormValidation) resolves
 * a field's `validator` rule by name against it.
 */

const v = (s: string) => ({ value: s, label: s })

// === Option/enum constants (DRY — authoritative per the CLI doc) =============

/** quantize-{linear,backbone-linear}-action — 9 values (doc §2.1.3 / §5.2). */
export const QUANTIZE_LINEAR_OPTIONS = [
  v('DISABLED'), v('W8A16_STATIC'), v('W8A8_STATIC'), v('W4A8_STATIC'),
  v('W8A16_DYNAMIC'), v('W8A8_DYNAMIC'), v('W4A8_DYNAMIC'), v('FP8'), v('MXFP4'),
]

/** quantize-attention-action — 3 values (doc §2.1.3). */
export const QUANTIZE_ATTENTION_OPTIONS = [v('DISABLED'), v('INT8'), v('FP8')]

/** remote-source (doc §2.1.7). */
export const REMOTE_SOURCE_OPTIONS = [v('huggingface'), v('modelscope')]

/** log-level: cli/utils.py LOG_LEVELS = 5 levels (doc §5.4). */
export const CLI_LOG_LEVEL_OPTIONS = [
  v('debug'), v('info'), v('warning'), v('error'), v('critical'),
]

/** log-level: serving_cast LOG_LEVELS = 6 levels (adds 'fatal') (doc §5.4). */
export const SERVING_LOG_LEVEL_OPTIONS = [
  v('debug'), v('info'), v('warning'), v('error'), v('critical'), v('fatal'),
]

/** Supported model IDs for the model_id combobox.
 *  Source: support_matrix_user_guide.md + HuggingFace official repos (2026-07-17).
 *  Users may still type arbitrary HuggingFace names or local paths; the
 *  combobox is filterable + allow-create. HuggingFace standard format (org/model). */
export const MODEL_ID_OPTIONS = [
  // --- DeepSeek ---
  v('deepseek-ai/DeepSeek-V3'),
  v('deepseek-ai/DeepSeek-V3.2'),
  v('deepseek-ai/DeepSeek-V4-Flash'),
  v('deepseek-ai/DeepSeek-V4-Pro'),
  // --- Kimi (Moonshot) ---
  v('moonshotai/Kimi-K2.5'),
  v('moonshotai/Kimi-K2.6'),
  v('moonshotai/Kimi-K3'),
  // --- Qwen3 (Dense) ---
  v('Qwen/Qwen3-0.6B'),
  v('Qwen/Qwen3-1.7B'),
  v('Qwen/Qwen3-4B'),
  v('Qwen/Qwen3-8B'),
  v('Qwen/Qwen3-14B'),
  v('Qwen/Qwen3-32B'),
  v('Qwen/Qwen3.8-2.4T-A95B'),
  // --- Qwen3 (MoE) ---
  v('Qwen/Qwen3-30B-A3B'),
  v('Qwen/Qwen3-235B-A22B'),
  // --- Qwen3-Next ---
  v('Qwen/Qwen3-Next-80B-A3B-Instruct'),
  // --- Qwen3.5 (Dense) ---
  v('Qwen/Qwen3.5-0.8B'),
  v('Qwen/Qwen3.5-2B'),
  v('Qwen/Qwen3.5-4B'),
  v('Qwen/Qwen3.5-9B'),
  v('Qwen/Qwen3.5-27B'),
  // --- Qwen3.5 (MoE) ---
  v('Qwen/Qwen3.5-35B-A3B'),
  v('Qwen/Qwen3.5-122B-A10B'),
  v('Qwen/Qwen3.5-397B-A17B'),
  // --- GLM ---
  v('zai-org/GLM-4.5'),
  v('zai-org/GLM-4.6'),
  v('zai-org/GLM-4.7'),
  v('zai-org/GLM-5'),
  v('zai-org/GLM-5.1'),
  v('zai-org/GLM-5.2'),
  // --- ERNIE ---
  v('baidu/ERNIE-4.5-21B-A3B-PT'),
  v('baidu/ERNIE-4.5-300B-A47B-PT'),
  // --- MiMo ---
  v('XiaomiMiMo/MiMo-V2-Flash'),
  // --- MiniMax ---
  v('MiniMaxAI/MiniMax-M2'),
  v('MiniMaxAI/MiniMax-M2.5'),
  v('MiniMaxAI/MiniMax-M2.7'),
  v('MiniMaxAI/MiniMax-M3'),
  // --- Qwen3-VL (Dense) ---
  v('Qwen/Qwen3-VL-2B-Instruct'),
  v('Qwen/Qwen3-VL-4B-Instruct'),
  v('Qwen/Qwen3-VL-8B-Instruct'),
  v('Qwen/Qwen3-VL-32B-Instruct'),
  // --- Qwen3-VL (MoE) ---
  v('Qwen/Qwen3-VL-30B-A3B-Instruct'),
  v('Qwen/Qwen3-VL-235B-A22B-Instruct'),
  // --- GLM-4V ---
  v('zai-org/glm-4v-9b'),
  v('zai-org/GLM-4.5V'),
  v('zai-org/GLM-4.6V'),
  // --- InternVL2 ---
  v('OpenGVLab/InternVL2-1B'),
  v('OpenGVLab/InternVL2-2B'),
  v('OpenGVLab/InternVL2-4B'),
  v('OpenGVLab/InternVL2-8B'),
  // --- InternVL2.5 ---
  v('OpenGVLab/InternVL2_5-1B'),
  v('OpenGVLab/InternVL2_5-2B'),
  v('OpenGVLab/InternVL2_5-4B'),
  v('OpenGVLab/InternVL2_5-8B'),
  v('OpenGVLab/InternVL2_5-26B'),
  // --- InternVL3 ---
  v('OpenGVLab/InternVL3-1B'),
  v('OpenGVLab/InternVL3-2B'),
  v('OpenGVLab/InternVL3-8B'),
  v('OpenGVLab/InternVL3-14B'),
  v('OpenGVLab/InternVL3-38B'),
  v('OpenGVLab/InternVL3-78B'),
  // --- InternVL3.5 (Dense) ---
  v('OpenGVLab/InternVL3_5-1B'),
  v('OpenGVLab/InternVL3_5-2B'),
  v('OpenGVLab/InternVL3_5-4B'),
  v('OpenGVLab/InternVL3_5-8B'),
  v('OpenGVLab/InternVL3_5-14B'),
  v('OpenGVLab/InternVL3_5-38B'),
  // --- InternVL3.5 (MoE) ---
  v('OpenGVLab/InternVL3_5-30B-A3B'),
  v('OpenGVLab/InternVL3_5-241B-A28B'),
]

/** Supported video/DiT model IDs for the video_generate model_id combobox.
 *  Source: support_matrix_user_guide.md + HuggingFace official repos (2026-07-17). */
export const VIDEO_MODEL_ID_OPTIONS = [
  // --- Wan ---
  v('Wan-AI/Wan2.1-T2V-1.3B-Diffusers'),
  v('Wan-AI/Wan2.1-T2V-14B-Diffusers'),
  v('Wan-AI/Wan2.2-TI2V-5B-Diffusers'),
  v('Wan-AI/Wan2.2-T2V-A14B-Diffusers'),
  // --- HunyuanVideo ---
  v('tencent/HunyuanVideo'),
  v('tencent/HunyuanVideo-1.5'),
]

// === Field factory helpers ===================================================

export interface Option {
  value: string
  label: string
}

/**
 * Factory for the model_id combobox field, used by text_generate,
 * throughput_optimizer, and video_generate. Avoids copy-paste drift across
 * the three form configs (e.g. tooltip referencing the wrong model family).
 *
 * @param options  Inline option list (MODEL_ID_OPTIONS or VIDEO_MODEL_ID_OPTIONS)
 * @param defaultValue  The default model ID for this form
 * @param example  Example shown in tooltip/placeholder (should match the model family)
 * @param includeStringValid  Whether to include the stringValid validator (text/throughput yes, video no)
 */
export function makeModelIdField(
  options: Option[],
  defaultValue: string,
  example: string,
  includeStringValid = true,
) {
  const validation: any[] = [
    { rule: 'required', message: { zh: '模型 ID 为必填项', en: 'Model ID is required' }, trigger: ['change', 'blur'] },
  ]
  if (includeStringValid) {
    validation.push({
      rule: 'validator',
      value: 'stringValid',
      message: { zh: '模型 ID 含非法字符或过长', en: 'Model ID has invalid characters or is too long' },
      trigger: ['blur'],
    })
  }
  return {
    id: 'model_id',
    label: { zh: '模型 ID', en: 'Model ID' },
    control: 'combobox',
    dataType: 'string',
    default: defaultValue,
    group: { zh: '通用', en: 'General' },
    optionSource: { type: 'inline', values: options },
    tooltip: {
      zh: `待仿真模型的标准 HuggingFace 名称（组织/模型，如 ${example}）或本地路径。`,
      en: `Standard HuggingFace model name (org/model, e.g. ${example}) or local path.`,
    },
    placeholder: { zh: `如 ${example}`, en: `e.g. ${example}` },
    validation,
  }
}

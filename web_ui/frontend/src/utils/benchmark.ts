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
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the Mulan PSL v2
 * for more details.
 * -------------------------------------------------------------------------
 */

/**
 * Benchmark convention-field expansion/collapse — shared by SchemaForm
 * (benchmark↔full mode-switch migration) and ModuleWorkspace (submit-time
 * expansion). Single source of truth: the two call sites must stay in sync.
 *
 * Expansion semantics: a preset only fills quantize fields that are still at
 * their schema default (or absent). A value that differs from the default was
 * deliberately set — e.g., model-feature overrides like mixed quantization
 * (preset W4A8C8 + non-expert W8A8) — and must survive the expansion.
 */

export const PD_MODE_MAP: Record<
  string,
  { disagg: boolean; pd_ratio: boolean; disable_tpot?: boolean; disable_ttft?: boolean }
> = {
  pure_prefill: { disagg: true, pd_ratio: false, disable_tpot: true },
  pure_decode: { disagg: true, pd_ratio: false, disable_ttft: true },
  pd_separation: { disagg: true, pd_ratio: false },
  pd_mixed: { disagg: false, pd_ratio: false },
}

export const QUANTIZE_PRESETS: Record<
  string,
  { linear: string; attention: string; non_expert: string } | null
> = {
  none: { linear: 'DISABLED', attention: 'DISABLED', non_expert: 'DISABLED' },
  W8A8C8: { linear: 'W8A8_DYNAMIC', attention: 'FP8', non_expert: 'W8A8_DYNAMIC' },
  W8A8C16: { linear: 'W8A8_DYNAMIC', attention: 'DISABLED', non_expert: 'W8A8_DYNAMIC' },
  W8A16C8: { linear: 'W8A16_DYNAMIC', attention: 'FP8', non_expert: 'W8A16_DYNAMIC' },
  W8A16C16: { linear: 'W8A16_DYNAMIC', attention: 'DISABLED', non_expert: 'W8A16_DYNAMIC' },
  W4A8C8: { linear: 'W4A8_DYNAMIC', attention: 'FP8', non_expert: 'W4A8_DYNAMIC' },
  W4A8C16: { linear: 'W4A8_DYNAMIC', attention: 'DISABLED', non_expert: 'W4A8_DYNAMIC' },
  custom: null,
}

/** True when the field carries no deliberate value: absent, or still equal to
 *  its schema default (value comparison — model-feature writes create new
 *  array/string instances, so reference equality would misclassify them). */
function isUnset(result: Record<string, any>, field: string, fieldDefaults: Record<string, any>): boolean {
  if (!(field in result)) return true
  return JSON.stringify(result[field]) === JSON.stringify(fieldDefaults[field])
}

/**
 * Apply model-specific overrides to the standard QUANTIZE_PRESETS.
 *
 * When a model has override values for quantize-linear-action, quantize-attention-action,
 * or quantize-non-expert-linear-action, this function generates a new preset map where
 * the specified preset's values are replaced with the override values.
 *
 * @param overrides - Model's quantize field overrides (from useModelFeatures.getCurrentQuantizeOverrides)
 * @param overridePreset - The preset name that the overrides belong to (e.g., "W4A8C8")
 * @returns A new QUANTIZE_PRESETS map with overrides applied, or the original if no overrides
 */
export function applyOverridesToPresets(
  overrides: { linear: string; attention: string; nonExpert: string } | null,
  overridePreset: string | null,
): typeof QUANTIZE_PRESETS {
  if (!overrides || !overridePreset || !QUANTIZE_PRESETS[overridePreset]) {
    return QUANTIZE_PRESETS
  }

  // Create a new preset map with the override applied to the specific preset
  return {
    ...QUANTIZE_PRESETS,
    [overridePreset]: {
      linear: overrides.linear,
      attention: overrides.attention,
      non_expert: overrides.nonExpert,
    },
  }
}

/**
 * Expand quantize-preset to the three underlying quantization fields.
 *
 * Called when the user changes the preset dropdown — ensures the underlying
 * fields reflect the selected preset, overriding any model-feature fills.
 *
 * @param preset - The selected preset name (e.g., "W8A8C8", "none")
 * @param formState - The form state store (will be mutated)
 * @param overrides - Optional model-specific overrides (from useModelFeatures.getCurrentQuantizeOverrides)
 * @param overridePreset - Optional preset name that the overrides belong to
 */
export function expandQuantizePresetToFields(
  preset: string,
  formState: { setFieldValue: (field: string, value: any) => void },
  overrides?: { linear: string; attention: string; nonExpert: string } | null,
  overridePreset?: string | null,
): void {
  const presets = applyOverridesToPresets(overrides ?? null, overridePreset ?? null)
  if (preset === 'custom' || !presets[preset]) {
    return
  }
  const q = presets[preset]!
  formState.setFieldValue('quantize-linear-action', [q.linear])
  formState.setFieldValue('quantize-attention-action', [q.attention])
  formState.setFieldValue('quantize-non-expert-linear-action', q.non_expert)
}

/**
 * Expand benchmark convention fields (pd-mode, quantize-preset) into full-mode
 * params and delete the convention keys. Does not mutate the input.
 *
 * @param formData - Form values (may contain convention fields).
 * @param fieldDefaults - Field-id → schema-default map. Quantize fields whose
 *   current value differs from the default are left untouched so deliberate
 *   settings (e.g., mixed quantization) survive the preset expansion.
 * @param overrides - Optional model-specific overrides (from useModelFeatures.getCurrentQuantizeOverrides)
 * @param overridePreset - Optional preset name that the overrides belong to
 */
export function expandBenchmarkConventionFields(
  formData: Record<string, any>,
  fieldDefaults: Record<string, any> = {},
  overrides?: { linear: string; attention: string; nonExpert: string } | null,
  overridePreset?: string | null,
): Record<string, any> {
  const result = { ...formData }
  const presets = applyOverridesToPresets(overrides ?? null, overridePreset ?? null)

  // Expand pd-mode convention field
  if ('pd-mode' in result) {
    const pdMode = result['pd-mode'] || 'pd_mixed'
    const mapping = PD_MODE_MAP[pdMode] || PD_MODE_MAP.pd_mixed
    result.disagg = mapping.disagg
    result['enable-optimize-prefill-decode-ratio'] = mapping.pd_ratio
    if (mapping.disable_tpot) {
      result['tpot-limit'] = null
    }
    if (mapping.disable_ttft) {
      result['ttft-limit'] = null
    }
    delete result['pd-mode']
  }

  // Expand quantize-preset convention field
  if ('quantize-preset' in result) {
    const preset = result['quantize-preset'] || 'W8A8C8'
    if (preset !== 'custom' && presets[preset]) {
      const q = presets[preset]!
      if (isUnset(result, 'quantize-linear-action', fieldDefaults)) {
        result['quantize-linear-action'] = [q.linear]
      }
      if (isUnset(result, 'quantize-attention-action', fieldDefaults)) {
        result['quantize-attention-action'] = [q.attention]
      }
      if (isUnset(result, 'quantize-non-expert-linear-action', fieldDefaults)) {
        result['quantize-non-expert-linear-action'] = q.non_expert
      }
    }
    delete result['quantize-preset']
  }

  return result
}

/**
 * Collapse full-mode params back to benchmark convention fields (inverse of
 * expansion). Used when migrating form state full → benchmark. Does not mutate
 * the input.
 *
 * @param formData - Form values (may contain expanded fields).
 * @param overrides - Optional model-specific overrides (from useModelFeatures.getCurrentQuantizeOverrides)
 * @param overridePreset - Optional preset name that the overrides belong to
 */
export function collapseToBenchmarkConventionFields(
  formData: Record<string, any>,
  overrides?: { linear: string; attention: string; nonExpert: string } | null,
  overridePreset?: string | null,
): Record<string, any> {
  const result = { ...formData }
  const presets = applyOverridesToPresets(overrides ?? null, overridePreset ?? null)

  // Collapse disagg/ttft-limit/tpot-limit to pd-mode
  const disagg = result.disagg
  const ttftLimit = result['ttft-limit']
  const tpotLimit = result['tpot-limit']

  if (disagg === true) {
    if (tpotLimit === null || tpotLimit === undefined || tpotLimit === '') {
      result['pd-mode'] = 'pure_prefill'
    } else if (ttftLimit === null || ttftLimit === undefined || ttftLimit === '') {
      result['pd-mode'] = 'pure_decode'
    } else {
      result['pd-mode'] = 'pd_separation'
    }
  } else {
    result['pd-mode'] = 'pd_mixed'
  }

  // Collapse quantize-* fields to quantize-preset
  const linear = result['quantize-linear-action']?.[0]
  const attention = result['quantize-attention-action']?.[0]
  const nonExpert = result['quantize-non-expert-linear-action']

  // Find matching preset (use override-aware presets)
  let matchedPreset = 'custom'
  for (const [preset, config] of Object.entries(presets)) {
    if (preset === 'custom' || config === null) continue
    if (config.linear === linear && config.attention === attention && config.non_expert === nonExpert) {
      matchedPreset = preset
      break
    }
  }
  result['quantize-preset'] = matchedPreset

  // Remove expanded fields (the convention fields' expansion products — they
  // are regenerated by expandBenchmarkConventionFields). ttft-limit/tpot-limit
  // are NOT removed: they are independent shared fields (present in both
  // schemas), not expansion products — the mode-switch migration preserves
  // shared-field values regardless.
  delete result.disagg
  delete result['enable-optimize-prefill-decode-ratio']
  delete result['quantize-linear-action']
  delete result['quantize-attention-action']
  delete result['quantize-non-expert-linear-action']

  return result
}

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
 * Model features composable for benchmark mode.
 *
 * Provides model-specific default overrides (compilation-config, quantization, etc.)
 * that are automatically applied when the user selects a model in the form.
 *
 * Key features:
 * - Calls /api/model-features/{model_id} to get overrides
 * - Applies overrides to formState (always overwrites — the model-features
 *   JSON carries the model's best-known configuration)
 * - Auto-infers quantize-preset from three underlying quantization fields
 */

import { api } from '@/services/api'
import { useFormStateStore } from '@/stores/formState'

// Reverse mapping: infer quantize-preset from three underlying fields
const QUANTIZE_PRESET_REVERSE: Record<string, { linear: string; attention: string; non_expert: string }> = {
  none: { linear: 'DISABLED', attention: 'DISABLED', non_expert: 'DISABLED' },
  W8A8C8: { linear: 'W8A8_DYNAMIC', attention: 'FP8', non_expert: 'W8A8_DYNAMIC' },
  W8A8C16: { linear: 'W8A8_DYNAMIC', attention: 'DISABLED', non_expert: 'W8A8_DYNAMIC' },
  W8A16C8: { linear: 'W8A16_DYNAMIC', attention: 'FP8', non_expert: 'W8A16_DYNAMIC' },
  W8A16C16: { linear: 'W8A16_DYNAMIC', attention: 'DISABLED', non_expert: 'W8A16_DYNAMIC' },
  W4A8C8: { linear: 'W4A8_DYNAMIC', attention: 'FP8', non_expert: 'W4A8_DYNAMIC' },
  W4A8C16: { linear: 'W4A8_DYNAMIC', attention: 'DISABLED', non_expert: 'W4A8_DYNAMIC' },
}

/**
 * Infer quantize-preset from three underlying quantization fields.
 *
 * @param linear - quantize-linear-action value (string)
 * @param attention - quantize-attention-action value (string)
 * @param nonExpert - quantize-non-expert-linear-action value (string)
 * @returns The matching preset name, or "custom" if no match
 */
function inferQuantizePreset(
  linear: string,
  attention: string,
  nonExpert: string,
): string {
  const l = linear || 'DISABLED'
  const a = attention || 'DISABLED'
  const n = nonExpert || 'DISABLED'

  for (const [preset, config] of Object.entries(QUANTIZE_PRESET_REVERSE)) {
    if (config.linear === l && config.attention === a && config.non_expert === n) {
      return preset
    }
  }
  return 'custom'
}

// Fields that are multi-select (string[]) in full mode — wrap string values in array
const ARRAY_COMPATIBLE_FIELDS = new Set([
  'quantize-linear-action',
  'quantize-attention-action',
])

/**
 * Composable for applying model-specific overrides to the benchmark form.
 *
 * Usage:
 * ```typescript
 * const { applyModelOverrides } = useModelFeatures()
 *
 * // When model-id changes:
 * watch(() => formState.form['model-id'], (modelId) => {
 *   if (modelId) applyModelOverrides(modelId)
 * })
 * ```
 */
export function useModelFeatures() {
  const formState = useFormStateStore()

  // Track the model's original quantize-preset and underlying field values.
  // Used to distinguish "user selected same preset as model" (preserve model's
  // mixed quantization) from "user selected different preset" (use standard values).
  let modelQuantizePreset: string | null = null
  let modelQuantizeFields: { linear: string; attention: string; nonExpert: string } | null = null

  /**
   * Fetch and apply model-specific overrides.
   *
   * Always overwrites all override fields from the model config, regardless of
   * whether the user has manually modified them. This ensures that when a user
   * selects a model, they get the complete set of model-specific defaults.
   *
   * Automatically infers quantize-preset from the three underlying quantization
   * fields and sets it in the form.
   *
   * @param modelId - The model identifier (e.g., "zai-org/GLM-5.2")
   */
  async function applyModelOverrides(modelId: string): Promise<void> {
    try {
      const { overrides } = await api.getModelFeatures(modelId)

      if (!overrides || Object.keys(overrides).length === 0) {
        // No overrides — reset quantize fields to match the current preset selection
        // This ensures that when switching from a model with overrides to one without,
        // the override values from the previous model are cleared.
        const currentPreset = formState.form['quantize-preset'] || 'W8A8C8'
        const presets = await import('@/utils/benchmark').then(m => m.QUANTIZE_PRESETS)
        const presetConfig = presets[currentPreset]

        if (presetConfig) {
          formState.setFieldValue('quantize-linear-action', [presetConfig.linear])
          formState.setFieldValue('quantize-attention-action', [presetConfig.attention])
          formState.setFieldValue('quantize-non-expert-linear-action', presetConfig.non_expert)
        }

        modelQuantizePreset = null
        modelQuantizeFields = null
        return
      }

      // Apply each override field (always overwrite)
      for (const [field, value] of Object.entries(overrides)) {
        // For multi-select fields, wrap string values in array for compatibility
        let finalValue = value
        if (ARRAY_COMPATIBLE_FIELDS.has(field) && typeof value === 'string') {
          finalValue = [value]
        }
        formState.setFieldValue(field, finalValue)
      }

      // Track model's quantize config for preset-change detection
      const linearAction = overrides['quantize-linear-action'] as string | undefined
      const attentionAction = overrides['quantize-attention-action'] as string | undefined
      const nonExpertAction = overrides['quantize-non-expert-linear-action'] as string | undefined

      if (linearAction || attentionAction || nonExpertAction) {
        modelQuantizeFields = {
          linear: linearAction || 'DISABLED',
          attention: attentionAction || 'DISABLED',
          nonExpert: nonExpertAction || 'DISABLED',
        }
      } else {
        modelQuantizeFields = null
      }

      // Auto-infer quantize-preset if not explicitly provided
      const explicitPreset = overrides['quantize-preset'] as string | undefined
      if (explicitPreset) {
        // API returned explicit preset — use it directly
        formState.setFieldValue('quantize-preset', explicitPreset)
        modelQuantizePreset = explicitPreset
      } else {
        // No explicit preset — infer from the three underlying fields
        if (linearAction || attentionAction || nonExpertAction) {
          const preset = inferQuantizePreset(
            linearAction || 'DISABLED',
            attentionAction || 'DISABLED',
            nonExpertAction || 'DISABLED',
          )
          formState.setFieldValue('quantize-preset', preset)
          modelQuantizePreset = preset
        } else {
          modelQuantizePreset = null
        }
      }
    } catch (error) {
      console.warn('Failed to fetch model features:', error)
      modelQuantizePreset = null
      modelQuantizeFields = null
      // Don't throw — allow form submission to proceed even if API fails
    }
  }

  /**
   * Get the current model's quantize preset.
   * Returns the preset name if available, otherwise null.
   */
  function getCurrentQuantizePreset(): string | null {
    return modelQuantizePreset
  }

  /**
   * Get the current model's quantize overrides.
   * Returns the model's underlying field values if available, otherwise null.
   * Used by expand functions to generate preset-specific QUANTIZE_PRESETS.
   */
  function getCurrentQuantizeOverrides(): { linear: string; attention: string; nonExpert: string } | null {
    return modelQuantizeFields
  }

  /**
   * Get the model's quantize config for the given preset.
   * Returns the model's underlying field values if the preset matches the
   * model's preset, otherwise returns null.
   */
  function getModelQuantizeConfig(preset: string): { linear: string; attention: string; nonExpert: string } | null {
    if (!modelQuantizeFields || preset !== modelQuantizePreset) {
      return null
    }
    return modelQuantizeFields
  }

  return { applyModelOverrides, getModelQuantizeConfig, getCurrentQuantizePreset, getCurrentQuantizeOverrides }
}

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
 * Form state store. Per-module form model with dirty/valid flags.
 * Locale-agnostic — display strings resolved via useLocale.t (Principle II v2.2.0).
 */
import { defineStore } from 'pinia'
import { trackEvent } from '@/services/telemetrySink'
import { ref, computed } from 'vue'
import type { FormSchemaEnvelope } from '@/composables/useConfig'
import { evalPredicate } from '@/composables/usePredicate'

export interface FieldState {
  touched: boolean
  dirty: boolean
  error?: string
}

export type FormModel = Record<string, any>

export const useFormStateStore = defineStore('formState', () => {
  // Active module schema
  const schema = ref<FormSchemaEnvelope | null>(null)

  // Form model (locale-agnostic values)
  const form = ref<FormModel>({})

  // Per-field state
  const fieldStates = ref<Record<string, FieldState>>({})

  // Form-level validation state
  const isValid = ref(true)
  const isDirty = computed(() => {
    return Object.values(fieldStates.value).some((state) => state.dirty)
  })

  // Initialize form with default values from schema
  function initForm(newSchema: FormSchemaEnvelope) {
    schema.value = newSchema
    const newForm: FormModel = {}
    const newFieldStates: Record<string, FieldState> = {}

    for (const field of newSchema.fields) {
      const defaultValue = field.default !== undefined ? field.default : getDefaultValueForType(field.dataType)
      newForm[field.id] = defaultValue
      newFieldStates[field.id] = {
        touched: false,
        dirty: false,
      }
    }

    form.value = newForm
    fieldStates.value = newFieldStates
    isValid.value = true
  }

  // Update a field value
  function setFieldValue(fieldId: string, value: any) {
    const changed = form.value[fieldId] !== value
    if (changed) {
      form.value[fieldId] = value
    }
    if (fieldStates.value[fieldId]) {
      // touched: user interacted with the field (mirrors CLI argv-present
      // semantics — the flag was "passed" regardless of final value)
      fieldStates.value[fieldId].touched = true
      // dirty: value actually changed from its initial state
      if (changed) {
        fieldStates.value[fieldId].dirty = true
      }
    }
    // Telemetry: record field interaction. Debounced because text/number
    // inputs fire setFieldValue on every keystroke; 1s per (module,field)
    // counts repeated typing as one interaction.
    if (changed) {
      const moduleId = schema.value?.moduleId || 'unknown'
      trackEvent(moduleId, fieldId, 'change', true)
    }
  }

  // Update field error message
  function setFieldError(fieldId: string, error?: string) {
    if (fieldStates.value[fieldId]) {
      fieldStates.value[fieldId].error = error
    }
  }

  // Clear all field errors
  function clearErrors() {
    for (const state of Object.values(fieldStates.value)) {
      state.error = undefined
    }
    isValid.value = true
  }

  // Mark form as valid/invalid
  function setValid(valid: boolean) {
    isValid.value = valid
  }

  // Reset form to initial state
  function resetForm() {
    if (!schema.value) return
    initForm(schema.value)
  }

  // Get default value for data type
  function getDefaultValueForType(dataType: string): any {
    switch (dataType) {
      case 'boolean':
        return false
      case 'integer':
      case 'number':
        return 0
      case 'string[]':
      case 'integer[]':
        return []
      default:
        return ''
    }
  }

  // Whether a field currently participates in the submission, i.e. its
  // `conditions.enabled` predicate holds against the live form model. Fields
  // without such a predicate are always active.
  //
  // Shared by getFormValues (params) and getTouchedFields (provided) so the two
  // sets can never disagree. They MUST stay in lockstep: `provided` mirrors CLI
  // argv-present semantics, and wants_provided validators (e.g.
  // draftLegacyMtpMutex) reject a flag merely for being in `provided`. If a
  // condition-disabled field were stripped from params but kept in provided, the
  // backend would report a conflict against a value the submission does not
  // contain — and the user could not fix it, because the field is disabled.
  function isFieldActive(fieldId: string): boolean {
    const field = (schema.value?.fields ?? []).find((f: any) => f.id === fieldId)
    if (!field?.conditions?.enabled) return true
    return evalPredicate(field.conditions.enabled, form.value)
  }

  // Get current form values. Statically hidden fields (schema `hidden: true`
  // from backend ui_props) are INCLUDED — hidden only means "no visible
  // control"; the value (schema default or a programmatic override, e.g.
  // model-features) must still reach the backend. Conditionally-disabled
  // fields (conditions.enabled evaluates to false) are excluded to prevent
  // invalid CLI commands (e.g., dspark params with dflash).
  function getFormValues(): FormModel {
    const out: FormModel = {}
    for (const [k, v] of Object.entries(form.value)) {
      if (!isFieldActive(k)) continue // Skip disabled fields
      out[k] = v
    }
    return out
  }

  // Get set of field IDs the user has explicitly interacted with (touched).
  // Used at submit time to build the `provided` set for wants_provided=True
  // validators — mirrors CLI semantics where argv-present flags are "provided"
  // regardless of whether their value equals the default.
  //
  // Condition-disabled fields are excluded, matching getFormValues: a value that
  // is not submitted cannot have been "provided". Note `touched` is also set by
  // programmatic setFieldValue callers (model-features fills, benchmark
  // expand/collapse, prefill), so this filter is not only about user typing.
  function getTouchedFields(): string[] {
    return Object.entries(fieldStates.value)
      .filter(([id, state]) => state.touched && isFieldActive(id))
      .map(([id]) => id)
  }

  // Check if form is pristine (no changes)
  const isPristine = computed(() => !isDirty.value)

  return {
    schema,
    form,
    fieldStates,
    isValid,
    isDirty,
    isPristine,
    initForm,
    setFieldValue,
    setFieldError,
    clearErrors,
    setValid,
    resetForm,
    getFormValues,
    getTouchedFields,
  }
})

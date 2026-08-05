/**
 * Form state store. Per-module form model with dirty/valid flags.
 * Locale-agnostic — display strings resolved via useLocale.t (Principle II v2.2.0).
 */
import { defineStore } from 'pinia'
import { trackEvent } from '@/services/telemetrySink'
import { ref, computed } from 'vue'
import type { FormSchemaEnvelope } from '@/composables/useConfig'

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
    if (form.value[fieldId] !== value) {
      form.value[fieldId] = value
      if (fieldStates.value[fieldId]) {
        fieldStates.value[fieldId].dirty = true
        fieldStates.value[fieldId].touched = true
      }
      // Telemetry: record field interaction. Debounced because text/number
      // inputs fire setFieldValue on every keystroke; 1s per (module,field)
      // counts repeated typing as one interaction.
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

  // Get current form values (excluding hidden/disabled fields handled by renderer)
  function getFormValues(): FormModel {
    return { ...form.value }
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
  }
})

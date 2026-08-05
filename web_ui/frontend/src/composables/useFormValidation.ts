/**
 * Form validation engine. async-validator integration with
 * cross-field reactive re-validation via dependsOn reverse-dependency map,
 * and submit-time full pass (visible/enabled fields + formValidation[]).
 */
import Schema from 'async-validator'
import type { RuleItem } from 'async-validator'
import { ref, computed, watch, type Ref } from 'vue'
import type { FormSchemaEnvelope } from '@/composables/useConfig'
import type { LocalizedText } from '@/composables/useLocale'
import { t, useLocale } from '@/composables/useLocale'
import { evalPredicate } from './usePredicate'

/** A named validator function (frontend-only). */
type ValidatorFn = (ctx: { value: any; form: Record<string, any>; field: any }) =>
  boolean | string | Promise<boolean | string>

/** Resolve a named validator from the active form's own `validators` map. */
function resolveValidator(
  validators: Record<string, ValidatorFn> | undefined,
  name: string,
): ValidatorFn | undefined {
  return validators?.[name]
}

export interface ValidationRule {
  rule: string
  message?: LocalizedText
  trigger?: string | string[]
  value?: any
  type?: string
  min?: number
  max?: number
  len?: number
  dependsOn?: string[]
}

export interface FieldSchema {
  id: string
  dataType?: string
  validation?: ValidationRule[]
  conditions?: {
    visible?: any
    enabled?: any
    required?: any
  }
}

/**
 * Resolve a localized message to a string.
 */
function resolveMessage(message: LocalizedText | undefined): string {
  if (!message) return ''
  if (typeof message === 'string') return message
  return t(message)
}

/**
 * Convert validation rule to async-validator RuleItem format.
 */
function convertValidationRule(
  fieldId: string,
  validation: ValidationRule,
  formModel: Ref<Record<string, any>>,
  validators: Record<string, ValidatorFn> | undefined,
): RuleItem {
  const rule: RuleItem = {
    // Use the rule's own declared type. Do NOT force `type:'string'` for
    // `required` rules — that makes async-validator reject a NUMBER value (e.g.
    // num_queries=1) as a type mismatch and surface it as a "required" error,
    // even though the field has a value. Leaving type unset lets `required`
    // only check presence.
    type: validation.type as RuleItem['type'],
    required: validation.rule === 'required',
    pattern: validation.rule === 'pattern' ? validation.value as string : undefined,
    // Support both styles: {rule:'min', value:N} (config convention) and {min:N}.
    min: validation.rule === 'min' ? validation.value : validation.min,
    max: validation.rule === 'max' ? validation.value : validation.max,
    len: validation.rule === 'len' ? validation.value : validation.len,
    enum: validation.rule === 'enum' ? validation.value as any[] : undefined,
    message: resolveMessage(validation.message),
    validator:
      validation.rule === 'validator'
        ? (_rule, value, callback) => {
            const validatorFn = resolveValidator(validators, validation.value as string)
            if (!validatorFn) {
              callback('Validator not found')
              return
            }

            Promise.resolve(
              validatorFn({
                value,
                form: formModel.value,
                field: { id: fieldId },
              })
            )
              .then((result: boolean | string) => {
                if (result === true) {
                  callback()
                } else if (typeof result === 'string') {
                  callback(result)
                } else if (result === false) {
                  callback(resolveMessage(validation.message))
                } else {
                  callback()
                }
              })
              .catch((err: unknown) => {
                callback(err instanceof Error ? err.message : 'Validation error')
              })
          }
        : undefined,
  }

  // Remove undefined properties
  Object.keys(rule).forEach((key) => {
    if (rule[key as keyof RuleItem] === undefined) {
      delete rule[key as keyof RuleItem]
    }
  })

  return rule
}

/**
 * Build reverse dependency map from field dependsOn declarations.
 * Maps each field ID to the list of fields that depend on it.
 */
function buildReverseDependencyMap(fields: Record<string, any>[]): Map<string, Set<string>> {
  const reverseMap = new Map<string, Set<string>>()

  for (const field of fields) {
    for (const rule of field.validation || []) {
      if (rule.dependsOn) {
        for (const depFieldId of rule.dependsOn) {
          if (!reverseMap.has(depFieldId)) {
            reverseMap.set(depFieldId, new Set())
          }
          reverseMap.get(depFieldId)!.add(field.id)
        }
      }
    }
  }

  return reverseMap
}

/**
 * Composable for form validation with cross-field reactive re-validation.
 */
export function useFormValidation(
  schema: Ref<FormSchemaEnvelope | null>,
  formModel: Ref<Record<string, any>>,
  fieldStates: Ref<Record<string, { touched: boolean; error?: string }>>
) {
  const { locale } = useLocale()
  const validator = ref<Schema | null>(null)
  const reverseDepMap = computed(() => {
    return schema.value ? buildReverseDependencyMap(schema.value.fields) : new Map()
  })

  // Build async-validator schema when form schema changes
  function buildValidator() {
    const envelope = schema.value
    if (!envelope) return

    const descriptor: Record<string, any> = {}

    for (const field of envelope.fields) {
      if (!field.validation || field.validation.length === 0) continue

      const rules: RuleItem[] = field.validation.map((v: any) =>
        convertValidationRule(field.id, v, formModel, envelope.validators)
      )

      if (rules.length > 0) {
        descriptor[field.id] = rules
      }
    }

    validator.value = new Schema(descriptor)
  }

  // Rebuild validator when the schema OR locale changes. Validation messages are
  // localized at build time (resolveMessage → t), so a locale switch must rebuild
  // the validator to re-translate its messages into the new language.
  watch([() => schema.value, locale], buildValidator, { immediate: true })

  // Validate a single field. Only the target field's data is passed to
  // async-validator, but it validates against the FULL descriptor — so other
  // `required` fields that are absent from this partial payload can also surface
  // errors. Filter to keep ONLY errors whose `field` matches the target,
  // otherwise another field's message (e.g. device's "required") gets
  // misattributed to this field.
  async function validateField(fieldId: string): Promise<boolean> {
    if (!validator.value) return true

    try {
      await validator.value.validate({ [fieldId]: formModel.value[fieldId] })
      if (fieldStates.value[fieldId]) {
        fieldStates.value[fieldId].error = undefined
      }
      return true
    } catch (err: any) {
      const fieldErrors = (err?.errors || []).filter((e: any) => e.field === fieldId)
      if (fieldStates.value[fieldId]) {
        fieldStates.value[fieldId].error = fieldErrors[0]?.message
      }
      return fieldErrors.length === 0
    }
  }

  // Validate all visible/enabled fields (submit-time full pass)
  async function validateForm(): Promise<{
    valid: boolean
    errors: Record<string, string>
  }> {
    if (!schema.value) return { valid: true, errors: {} }

    // Build subset of form with only visible/enabled fields. A field hidden or
    // disabled by its `conditions` is skipped — so a static `required` rule on a
    // conditionally-visible field (e.g. profiling_database, the PD-ratio device
    // fields) does not spuriously block submit when it's hidden.
    const activeForm: Record<string, any> = {}
    const activeFields = new Set<string>()
    const model = formModel.value

    for (const field of schema.value.fields) {
      const cond = field.conditions
      if (cond?.visible !== undefined && !evalPredicate(cond.visible, model)) continue
      if (cond?.enabled !== undefined && !evalPredicate(cond.enabled, model)) continue
      activeForm[field.id] = model[field.id]
      activeFields.add(field.id)
    }

    // Build a descriptor restricted to ACTIVE fields only. The shared
    // `validator` holds rules for ALL fields (including hidden ones), and
    // async-validator reports a required field as "missing" when absent from the
    // validated payload — so a hidden required field (e.g. prefill/decode
    // devices when PD-ratio optimization is off) would spuriously fail submit.
    // Validating against an active-only descriptor avoids that.
    const envelope = schema.value
    const descriptor: Record<string, any> = {}
    for (const field of envelope.fields) {
      if (!activeFields.has(field.id)) continue
      if (!field.validation || field.validation.length === 0) continue
      descriptor[field.id] = field.validation.map((v: any) =>
        convertValidationRule(field.id, v, formModel, envelope.validators),
      )
    }

    try {
      await new Schema(descriptor).validate(activeForm)
      return { valid: true, errors: {} }
    } catch (err: any) {
      const errors: Record<string, string> = {}
      if (err?.errors) {
        for (const e of err.errors) {
          if (e.field) {
            errors[e.field] = e.message
            if (fieldStates.value[e.field]) {
              fieldStates.value[e.field].error = e.message
            }
          }
        }
      }
      return { valid: false, errors }
    }
  }

  // Validate form-level invariants (formValidation[] from envelope)
  async function validateFormInvariants(): Promise<{
    valid: boolean
    errors: string[]
  }> {
    if (!schema.value?.formValidation || schema.value.formValidation.length === 0) {
      return { valid: true, errors: [] }
    }

    const errors: string[] = []

    for (const rule of schema.value.formValidation) {
      if (rule.rule !== 'validator' || !rule.value) continue

      const validatorFn = resolveValidator(schema.value.validators, rule.value as string)
      if (!validatorFn) continue

      try {
        const result = await validatorFn({
          value: null,
          form: formModel.value,
          field: { id: '__form__' },
        })

        if (result !== true && typeof result === 'string') {
          errors.push(result)
        } else if (result === false) {
          errors.push(resolveMessage(rule.message))
        }
      } catch (err) {
        errors.push(err instanceof Error ? err.message : 'Validation error')
      }
    }

    return { valid: errors.length === 0, errors }
  }

  // Get fields that depend on a given field (for reactive re-validation)
  function getDependentFields(fieldId: string): string[] {
    return Array.from(reverseDepMap.value.get(fieldId) || [])
  }

  // Re-validate dependent fields when a field changes
  async function revalidateDependents(fieldId: string): Promise<void> {
    const dependents = getDependentFields(fieldId)
    for (const depId of dependents) {
      if (fieldStates.value[depId]?.touched) {
        await validateField(depId)
      }
    }
  }

  // Clear all validation errors
  function clearValidationErrors(): void {
    for (const state of Object.values(fieldStates.value)) {
      state.error = undefined
    }
  }

  // Locale changed → re-run the full visible-field validation so EVERY error
  // message re-translates to the new language (the validator was rebuilt above
  // with new-locale messages). Using the full pass — not just touched fields —
  // because submit-time validation can set errors on never-touched fields
  // (e.g. a required field left at its empty default), which would otherwise
  // keep their stale old-locale message.
  watch(locale, async () => {
    await validateForm()
  })

  return {
    validateField,
    validateForm,
    validateFormInvariants,
    getDependentFields,
    revalidateDependents,
    clearValidationErrors,
  }
}

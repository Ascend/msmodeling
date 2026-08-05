/**
 * Field conditions evaluator. Evaluates visible/enabled/required predicates.
 * Re-evaluates when dependent fields change (reactive).
 */
import { computed, type Ref } from 'vue'
import { evalPredicate } from './usePredicate'

export interface Condition {
  field?: string
  op?: string
  value?: any
  and?: Condition[]
  or?: Condition[]
  not?: Condition
}

export interface FieldConditions {
  visible?: Condition
  enabled?: Condition
  required?: Condition
}

/**
 * Evaluate a condition against the form model.
 * Returns true if the condition is satisfied. Delegates to the shared
 * ``evalPredicate`` engine, which resolves ``field``/``op``/``value`` from the
 * node itself and handles ``and``/``or``/``not`` combinators.
 *
 * For required conditions, the default (when undefined) is false (not required).
 * For visible/enabled conditions, the default (when undefined) is true.
 */
function evaluateCondition(
  condition: Condition | undefined,
  formModel: Record<string, any>,
  defaultValue: boolean = true
): boolean {
  if (!condition) return defaultValue
  return evalPredicate(condition as Record<string, any>, formModel)
}

/**
 * Composable to evaluate field conditions reactively.
 */
export function useFieldConditions(
  fieldConditions: FieldConditions | undefined,
  formModel: Ref<Record<string, any>>
) {
  // Compute visibility
  const isVisible = computed(() => {
    return evaluateCondition(fieldConditions?.visible, formModel.value)
  })

  // Compute enabled state
  const isEnabled = computed(() => {
    return evaluateCondition(fieldConditions?.enabled, formModel.value)
  })

  // Compute required state (default: not required)
  const isRequired = computed(() => {
    return evaluateCondition(fieldConditions?.required, formModel.value, false)
  })

  return {
    isVisible,
    isEnabled,
    isRequired,
  }
}

/**
 * Shared predicate engine. Evaluates form field ``conditions``
 * (visible / enabled / required). **Frontend-only** —
 * it drives LIVE field visibility/enabled/required feedback; the backend does
 * NOT re-validate (Principle I: thin orchestration — the Runner validates
 * internally → job ``failed``).
 *
 * Ops: eq | ne | include | exclude | contains | notContains | gt | gte | lt | lte
 *      | notEmpty | empty | isTrue | isFalse | present | absent.
 *      Combinators: and | or | not.
 *
 * Membership direction matters — the two pairs are NOT symmetric, pick by which
 * side is the array: ``include``/``exclude`` = scalar field value ∈ expected[]
 * (``expected.includes(value)``); ``contains``/``notContains`` = array field
 * value ∋ expected scalar (``value.includes(expected)``) — for multi-select
 * fields whose value is a list (e.g. quantize_linear_action = ["W8A8","MXFP4"]).
 */

export type PredicateNode = Record<string, any>

const MISSING = Symbol('missing')

function resolvePath(root: any, path: string): any {
  if (!path) return MISSING
  let current: any = root
  for (const part of String(path).split('.')) {
    if (current === MISSING) return MISSING
    if (current == null) return MISSING
    if (Array.isArray(current)) {
      const idx = Number.parseInt(part, 10)
      if (Number.isNaN(idx) || idx < 0 || idx >= current.length) return MISSING
      current = current[idx]
    } else if (typeof current === 'object') {
      if (part in current) current = current[part]
      else return MISSING
    } else {
      return MISSING
    }
  }
  return current
}

function operand(model: Record<string, any>, node: PredicateNode): any {
  const key = node.field ?? node.path
  if (key === undefined) throw new Error("predicate node needs 'field' or 'path'")
  return resolvePath(model, String(key))
}

export function evalPredicate(node: any, model: Record<string, any>): boolean {
  if (node == null) return true
  if (typeof node === 'boolean') return node
  if (typeof node !== 'object') throw new Error(`predicate node must be an object`)

  if (Array.isArray(node.and)) return node.and.every((child: PredicateNode) => evalPredicate(child, model))
  if (Array.isArray(node.or)) return node.or.some((child: PredicateNode) => evalPredicate(child, model))
  if (node.not !== undefined) return !evalPredicate(node.not, model)

  const op = node.op
  if (op === undefined) return true

  const value = operand(model, node)
  const expected = node.value

  switch (op) {
    case 'eq':
      return value === expected
    case 'ne':
      return value !== expected
    case 'include':
      return Array.isArray(expected) && value !== MISSING && expected.includes(value)
    case 'exclude':
      return Array.isArray(expected) && !(value !== MISSING && expected.includes(value))
    case 'contains':
      // expected scalar ∈ field value[] — multi-select membership (mirror of `in`).
      return Array.isArray(value) && value.includes(expected)
    case 'notContains':
      return !(Array.isArray(value) && value.includes(expected))
    case 'gt':
    case 'gte':
    case 'lt':
    case 'lte': {
      if (value === MISSING || value == null) return false
      switch (op) {
        case 'gt':
          return value > expected
        case 'gte':
          return value >= expected
        case 'lt':
          return value < expected
        default:
          return value <= expected
      }
    }
    case 'isTrue':
      return value === true
    case 'isFalse':
      return value === false
    case 'empty':
      return value === MISSING || value == null || value === '' || isEmptyCollection(value)
    case 'notEmpty':
      return !(value === MISSING || value == null || value === '' || isEmptyCollection(value))
    case 'present':
      return value !== MISSING
    case 'absent':
      return value === MISSING
    default:
      throw new Error(`unknown predicate op: ${op}`)
  }
}

function isEmptyCollection(value: any): boolean {
  return Array.isArray(value) ? value.length === 0 : typeof value === 'object' && value !== null && Object.keys(value).length === 0
}

/** Vue-friendly composable wrapper; the pure ``evalPredicate`` is the export. */
export function usePredicate() {
  return { evalPredicate }
}

/**
 * Config loader. Bundled form schemas are imported DIRECTLY via explicit
 * static imports — no `import.meta.glob` (per the project direct-import /
 * no-lazy-load directive: import required files directly). The frontend always renders from
 * the current config; version bumps apply on rebuild.
 *
 * Adding a module = add one `import` + one entry in `forms` below.
 * (Result rendering is per-module components — there is no viz/compare config.)
 */
import { computed } from 'vue'
import type { ValidatorFn } from '../config/forms/_validators'
// Import the TypeScript sources (data + inlined validators) — NOT the .json.
// The .json is a generated, data-only artifact for the backend (schema pinning).
import textGenerate from '../config/forms/text_generate'
import videoGenerate from '../config/forms/video_generate'
import throughputOptimizer from '../config/forms/throughput_optimizer'

export type LocalizedText = string | Record<string, string>

export interface FormSchemaEnvelope {
  moduleId: string
  title: LocalizedText
  runner: string
  version: string
  fields: Array<Record<string, any>>
  optionSourceRegistry?: Record<string, any>
  /** Optional whole-form invariants (cross-field validator rules). */
  formValidation?: Array<Record<string, any>>
  /**
   * Optional per-group metadata (order is informational; a field's group is
   * matched to an entry by its localized label). `defaultCollapsed` controls
   * whether the section renders collapsed initially.
   */
  groups?: Array<{ label: LocalizedText; defaultCollapsed?: boolean; description?: LocalizedText }>
  /**
   * Per-form validator map (frontend-only). A field's
   * `{ rule: 'validator', value: '<name>' }` resolves `<name>` against this map.
   * Functions live in the .ts source; the backend's generated .json has this
   * stripped to `{}`.
   */
  validators?: Record<string, ValidatorFn>
}

/** One explicit entry per bundled form config (direct imports — no glob). */
const forms: Record<string, FormSchemaEnvelope> = {
  text_generate: textGenerate as unknown as FormSchemaEnvelope,
  video_generate: videoGenerate as unknown as FormSchemaEnvelope,
  throughput_optimizer: throughputOptimizer as unknown as FormSchemaEnvelope,
}

export function useFormConfigs() {
  const moduleIds = computed(() => Object.keys(forms))
  const getForm = (moduleId: string): FormSchemaEnvelope | undefined => forms[moduleId]
  return { moduleIds, getForm }
}

export const useConfig = useFormConfigs

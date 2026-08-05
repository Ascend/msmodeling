<script setup lang="ts">
/**
 * SchemaForm. Config-driven form renderer with field dispatch,
 * validation engine integration, and submit-time full pass.
 */
import { ref, computed, watch, onMounted, nextTick, provide } from 'vue'
import { ElForm, ElMessage } from 'element-plus'
import { ArrowRight } from '@element-plus/icons-vue'
import { useFormStateStore } from '@/stores/formState'
import { useOptionSources } from '@/composables/useOptionSources'
import { useFormValidation } from '@/composables/useFormValidation'
import { useLocale } from '@/composables/useLocale'
import { useFormConfigs } from '@/composables/useConfig'
import type { Option } from '@/composables/useOptionSources'
import SchemaFormItem from './SchemaFormItem.vue'
import { trackEvent } from '@/services/telemetrySink'

interface Props {
  moduleId: string
  loading?: boolean
}

const props = defineProps<Props>()
const emit = defineEmits<{
  submit: [data: { moduleId: string; params: Record<string, any>; formSchemaVersion: string }]
}>()

const { t } = useLocale()
const formState = useFormStateStore()

// Dev-only test hook: expose the (singleton) form store so Playwright e2e can
// set field values directly by id. Element Plus el-form-item does NOT render a
// `prop` DOM attribute, so fields cannot be located by id in the DOM; setting
// via the reactive store is the reliable path. No-op in production builds.
if (import.meta.env?.DEV) {
  ;(window as any).__formState = formState
}
const { getForm } = useFormConfigs()
const { revalidateDependents, validateForm, validateFormInvariants } =
  useFormValidation(computed(() => formState.schema), computed(() => formState.form), computed(() => formState.fieldStates))

// Local state for dynamic options
const optionsMap = ref<Record<string, Option[]>>({})
const loadingOptions = ref<Set<string>>(new Set())
const optionsErrors = ref<Record<string, string>>({})

// Ref graph for scroll-to-error: keyed by field.id -> SchemaFormItem instance
// (which exposes its el-form-item ref). el-form-item does NOT render a `prop`
// DOM attribute, so document.querySelector('[prop=...]') is a dead no-op; we
// resolve each field's rendered root ($el) via this map instead. The callback
// form of :ref handles add/unmount without leaving stale entries.
const formItemRefs = new Map<string, any>()
function setItemRef(fieldId: string) {
  return (el: any) => {
    if (el) formItemRefs.set(fieldId, el)
    else formItemRefs.delete(fieldId)
  }
}

// Option sources registry from schema
const optionSourcesRegistry = computed(() => {
  return formState.schema?.optionSourceRegistry || {}
})

// Initialize option sources composable
const { fetchSourceOptions, isLoading: isOptionLoading, getSourceError } = useOptionSources(
  optionSourcesRegistry.value
)

// Fetch dynamic options on mount
async function fetchDynamicOptions() {
  if (!formState.schema) return

  // Fetch options for each dynamic source
  for (const field of formState.schema.fields) {
    if (field.optionSource?.type === 'dynamic' && field.optionSource.name) {
      const sourceName = field.optionSource.name
      if (!optionsMap.value[field.id]) {
        try {
          loadingOptions.value = new Set(loadingOptions.value).add(field.id)
          // Pass current optionSourcesRegistry value to ensure we have the loaded schema
          const options = await fetchSourceOptions(sourceName, optionSourcesRegistry.value)
          optionsMap.value[field.id] = options
        } catch (err) {
          optionsErrors.value[field.id] = getSourceError(sourceName) || 'Failed to load options'
        } finally {
          loadingOptions.value = new Set([...loadingOptions.value].filter((id) => id !== field.id))
        }
      }
    }
  }
}

// Resolve static (inline) options
function resolveStaticOptions(field: Record<string, any>): Option[] {
  if (field.optionSource?.type === 'inline' && field.optionSource.values) {
    return field.optionSource.values.map((opt: any) => ({
      value: opt.value,
      label: opt.label || opt.value,
    }))
  }
  return []
}

// Get options for a field
function getFieldOptions(fieldId: string): Option[] {
  const field = formState.schema?.fields.find((f) => f.id === fieldId)
  if (!field) return []

  // Direct `options` array on the field itself (e.g. QUANTIZE_LINEAR_OPTIONS,
  // REMOTE_SOURCE_OPTIONS, CLI_LOG_LEVEL_OPTIONS). This is a separate convention
  // from optionSource.{inline|dynamic} and must be handled here too, otherwise
  // those selects render empty.
  if (Array.isArray(field.options)) {
    return field.options.map((opt: any) => ({
      value: opt.value,
      label: opt.label || opt.value,
    }))
  }

  if (field.optionSource?.type === 'inline') {
    return resolveStaticOptions(field)
  }
  if (field.optionSource?.type === 'dynamic') {
    return optionsMap.value[fieldId] || []
  }
  return []
}

// Check if field options are loading (isLoading is a ComputedRef wrapping a fn)
function isFieldLoading(fieldId: string): boolean {
  return loadingOptions.value.has(fieldId) || isOptionLoading.value(fieldId)
}

// Retry a single field's dynamic option source — clears its error, re-runs the
// fetch. Previously `optionsErrors` was populated but never surfaced, so a
// failed fetch left the field stuck as an empty dropdown with no recovery path.
async function retryOptions(fieldId: string) {
  const field = formState.schema?.fields.find((f) => f.id === fieldId)
  if (!field || field.optionSource?.type !== 'dynamic' || !field.optionSource.name) return
  const sourceName = field.optionSource.name
  // Clear prior error + cache so we actually re-fetch
  delete optionsErrors.value[fieldId]
  optionsMap.value = { ...optionsMap.value, [fieldId]: undefined }
  try {
    loadingOptions.value = new Set(loadingOptions.value).add(fieldId)
    const options = await fetchSourceOptions(sourceName, optionSourcesRegistry.value)
    optionsMap.value = { ...optionsMap.value, [fieldId]: options }
  } catch {
    optionsErrors.value[fieldId] = getSourceError(sourceName) || 'Failed to load options'
  } finally {
    loadingOptions.value = new Set([...loadingOptions.value].filter((id) => id !== fieldId))
  }
}

// Initialize form with schema
onMounted(async () => {
  const schema = getForm(props.moduleId)
  if (schema) {
    formState.initForm(schema)
    await fetchDynamicOptions()
  }
})

// Watch for field changes and re-validate dependents.
//
// Implementation note (#18): Vue's `watch(fn, cb, { deep: true })` passes the
// SAME reference for `newForm` and `oldForm` — by the time the watcher fires,
// the mutation is already in place on the reactive object. So
// `newForm[fieldId] !== oldForm[fieldId]` was always FALSE and
// `revalidateDependents` was NEVER triggered. Rather than snapshot/serialize
// the form on every change, we call `revalidateDependents` directly from the
// `@update:model-value` handler below, where the changed `field.id` is known
// explicitly. That path is the ONLY writer to `formState.form` from the UI,
// so no watcher is needed.

// Handle form submission (exposed so parent pages can trigger it via a ref)
async function handleSubmit() {
  if (!formState.schema) return

  // Run all visible/enabled field validations
  const { valid: fieldsValid, errors: fieldErrors } = await validateForm()

  if (!fieldsValid) {
    ElMessage.error(t({ zh: '请修正表单错误', en: 'Please fix form errors' }))
    const firstErrorField = Object.keys(fieldErrors)[0]
    if (firstErrorField) {
      // Auto-expand the first group containing the error field so it is
      // actually visible before scrolling (collapsed sections hide their
      // fields' DOM roots).
      let expanded = false
      for (const group of groupedFields.value) {
        if (group.fields.some((f) => f.id === firstErrorField)) {
          if (collapsed.value[group.key]) {
            collapsed.value = { ...collapsed.value, [group.key]: false }
            expanded = true
          }
          break
        }
      }
      // v-show expand hasn't rendered yet when we just toggled collapse — wait
      // one tick so the field's DOM is actually visible before measuring it,
      // otherwise scrollIntoView targets a hidden element and is a no-op.
      if (expanded) await nextTick()
      // Scroll via the Vue ref graph: resolve the field's SchemaFormItem,
      // then its exposed el-form-item, then the rendered root ($el).
      const itemInstance = formItemRefs.get(firstErrorField)
      const formItemEl = (itemInstance?.formItemRef?.$el ?? itemInstance?.$el) as
        | HTMLElement
        | undefined
      formItemEl?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
    return
  }

  // Run form-level invariants
  const { valid: invariantsValid, errors: invariantErrors } = await validateFormInvariants()
  if (!invariantsValid) {
    ElMessage.error(invariantErrors.join('; '))
    return
  }

  // Strip hidden/disabled keys and emit submit
  const activeParams = formState.getFormValues()
  emit('submit', {
    moduleId: props.moduleId,
    params: activeParams,
    formSchemaVersion: formState.schema.version,
  })
}

// Group fields by their group attribute. Each group carries a stable key (the
// group's `en` label, locale-independent) for collapse state, plus the
// `defaultCollapsed` from the envelope's optional `groups` metadata.
const groupedFields = computed(() => {
  if (!formState.schema) return []
  const groupsMeta = formState.schema.groups || []
  const enOf = (lbl: any): string => (lbl && typeof lbl === 'object' ? lbl.en : typeof lbl === 'string' ? lbl : '')
  const metaByEn = new Map(groupsMeta.map((g) => [enOf(g.label), g]))

  const ordered: { key: string; label: string; fields: Record<string, any>[]; defaultCollapsed: boolean; description: string }[] = []
  const indexByKey = new Map<string, number>()

  for (const field of formState.schema.fields) {
    const g = field.group
    const en = g && typeof g === 'object' ? g.en : typeof g === 'string' ? g : ''
    const key = en || '__ungrouped__'
    let idx = indexByKey.get(key)
    if (idx === undefined) {
      const meta = en ? metaByEn.get(en) : undefined
      idx = ordered.length
      indexByKey.set(key, idx)
      ordered.push({
        key,
        label: g ? (typeof g === 'object' ? t(g) : g) : '',
        fields: [],
        defaultCollapsed: !!meta?.defaultCollapsed,
        description: meta?.description ? t(meta.description) : '',
      })
    }
    ordered[idx].fields.push(field)
  }
  return ordered
})

// Per-group error counts derived from fieldStates. A field counts as errored
// when its fieldStates entry carries a truthy `error` (set by both per-field
// validation and the submit-time full pass). Surfaces a red badge on a section
// header so a collapsed group still signals where validation failed.
const groupErrorCounts = computed<Record<string, number>>(() => {
  const counts: Record<string, number> = {}
  for (const group of groupedFields.value) {
    let n = 0
    for (const field of group.fields) {
      if (formState.fieldStates[field.id]?.error) n += 1
    }
    counts[group.key] = n
  }
  return counts
})

// Collapsible-section state, keyed by the group's stable key. Initialized from
// each group's `defaultCollapsed` ONLY when the schema changes — NOT on locale
// changes. `groupedFields` recomputes on locale change (it translates labels),
// so watching it would wipe the user's manual expand/collapse toggles every
// time they switch language. The group `key` is the locale-stable `en` label,
// so collapse state survives a re-render with translated labels.
const collapsed = ref<Record<string, boolean>>({})
watch(
  () => formState.schema,
  () => {
    const groups = groupedFields.value
    collapsed.value = Object.fromEntries(groups.map((g) => [g.key, g.defaultCollapsed]))
  },
  { immediate: true },
)

function toggleGroup(key: string) {
  collapsed.value = { ...collapsed.value, [key]: !collapsed.value[key] }
  trackEvent(props.moduleId, `group:${key}`, 'toggle')
}

// Resolve group label for display
function getGroupLabel(groupName: string): string {
  if (groupName === '__ungrouped__') return ''
  return groupName
}

// Whether a field should span two grid columns (wide controls whose selected
// tags need room). Multi-select dropdowns fit one column thanks to collapse-tags,
// so by default everything occupies a single cell -> aligned 4-column grid.
function isWideField(_field: Record<string, any>): boolean {
  return false
}

// Provide schema and options for child components
provide('schema', computed(() => formState.schema))
provide('optionsMap', optionsMap)

// Expose submit so parent pages can trigger it without DOM hacks
defineExpose({ submit: handleSubmit })
</script>

<template>
  <el-form
    v-if="formState.schema"
    ref="formRef"
    :model="formState.form"
    label-width="128px"
    label-position="left"
    class="schema-form"
    @submit.prevent="handleSubmit"
  >
    <!-- Each group is a collapsible section: a clickable header (chevron + title
         + field count) over a 4-column field grid. `defaultCollapsed` is read
         from the envelope's optional `groups` metadata. -->
    <div class="form-sections">
      <section
        v-for="group in groupedFields"
        :key="group.key"
        class="form-section"
        :class="{ 'is-collapsed': collapsed[group.key] }"
      >
        <button
          type="button"
          class="section-header"
          :aria-expanded="!collapsed[group.key]"
          :aria-controls="`section-${group.key}`"
          @click="toggleGroup(group.key)"
        >
          <el-icon class="section-chevron"><ArrowRight /></el-icon>
          <span class="section-title">{{ getGroupLabel(group.label) || group.label }}</span>
          <span class="section-count">{{ group.fields.length }}</span>
          <span
            v-if="groupErrorCounts[group.key] > 0"
            class="section-error-count"
            :title="t({ zh: `${groupErrorCounts[group.key]} 个校验错误`, en: `${groupErrorCounts[group.key]} validation error(s)` })"
          >
            {{ groupErrorCounts[group.key] }}
          </span>
        </button>

        <div v-if="group.description" class="section-desc">{{ group.description }}</div>

        <div :id="`section-${group.key}`" v-show="!collapsed[group.key]" class="form-grid">
          <schema-form-item
            v-for="field in group.fields"
            :key="field.id"
            :ref="setItemRef(field.id)"
            :class="{ 'cell-wide': isWideField(field) }"
            :field="field"
            :model-value="formState.form[field.id]"
            :field-error="formState.fieldStates[field.id]?.error"
            :options="getFieldOptions(field.id)"
            :loading="isFieldLoading(field.id)"
            :options-error="optionsErrors[field.id]"
            @update:model-value="(val) => {
              formState.setFieldValue(field.id, val)
              // Cross-field re-validation: when THIS field changes, re-run
              // validators on every field that declares `dependsOn: [field.id]`.
              // Called directly here (#18) rather than via a deep watch — Vue's
              // deep watch passes the same reference for new/old, so the
              // per-field diff was always empty and revalidation never fired.
              void revalidateDependents(field.id)
            }"
            @retry-options="retryOptions(field.id)"
          />
        </div>
      </section>
    </div>
  </el-form>

  <div v-else class="form-loading">
    <el-skeleton :rows="5" animated />
  </div>
</template>

<style scoped>
/* Collapsible sections stack vertically; each holds its own 4-column grid. */
.form-sections {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-section {
  border: 1px solid var(--msm-border);
  border-radius: 8px;
  background: var(--msm-bg-panel);
  overflow: hidden;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 10px 14px;
  cursor: pointer;
  user-select: none;
  text-align: left;
  font: inherit;
  color: inherit;
  background: var(--msm-bg-panel-2);
  border: none;
  border-left: 3px solid transparent;
  border-radius: 0;
  transition: background var(--msm-transition-fast) var(--msm-ease-out),
    border-color var(--msm-transition-fast) var(--msm-ease-out);
}

/* optional group-level description shown between header and field grid */
.section-desc {
  padding: 7px 14px 9px;
  font-size: 12px;
  line-height: 1.5;
  color: var(--msm-text-muted);
  background: var(--msm-bg-panel);
  border-bottom: 1px solid var(--msm-border);
}

/* expanded header gets a green accent stripe + separator border */
.form-section:not(.is-collapsed) .section-header {
  border-left-color: var(--msm-green);
  border-bottom: 1px solid var(--msm-border);
  background: linear-gradient(90deg, color-mix(in srgb, var(--msm-green) 8%, var(--msm-bg-panel-2)), var(--msm-bg-panel-2));
}

.section-header:hover {
  background: color-mix(in srgb, var(--msm-green) 14%, var(--msm-bg-panel-2));
}

/* visible keyboard focus ring (a11y) */
.section-header:focus-visible {
  outline: none;
  box-shadow: inset 0 0 0 2px var(--msm-green);
}

.section-chevron {
  font-size: 12px;
  color: var(--msm-text-muted);
  transition: transform var(--msm-transition-fast) var(--msm-ease-out),
    color var(--msm-transition-fast) var(--msm-ease-out);
}

.form-section:not(.is-collapsed) .section-chevron {
  transform: rotate(90deg);
  color: var(--msm-green);
}

.section-title {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--msm-text);
}

/* collapsed title is slightly muted (its fields are hidden) */
.form-section.is-collapsed .section-title {
  color: var(--msm-text);
  opacity: 0.92;
}

.section-count {
  margin-left: auto;
  font-size: 11px;
  font-family: 'Fira Code', monospace;
  color: var(--msm-text-muted);
  background: var(--msm-bg-deep);
  border: 1px solid var(--msm-border);
  border-radius: 10px;
  padding: 1px 8px;
  min-width: 22px;
  text-align: center;
}

/* when collapsed, the count badge is the only hint of contents — emphasize it */
.form-section.is-collapsed .section-count {
  color: var(--msm-green);
  border-color: color-mix(in srgb, var(--msm-green) 40%, var(--msm-border));
}

/* red error-count badge on the section header, shown only when the group has
   validation errors. Sits next to section-count so a collapsed group still
   signals where the failure is. */
.section-error-count {
  font-size: 11px;
  font-family: 'Fira Code', monospace;
  color: var(--msm-red);
  background: color-mix(in srgb, var(--msm-red) 10%, var(--msm-bg-deep));
  border: 1px solid color-mix(in srgb, var(--msm-red) 45%, var(--msm-border));
  border-radius: 10px;
  padding: 1px 8px;
  min-width: 22px;
  text-align: center;
}

/* Respect reduced-motion: drop the chevron rotation transition. */
@media (prefers-reduced-motion: reduce) {
  .section-chevron,
  .section-header {
    transition: none;
  }
}

/* 4-column grid inside each section. */
.form-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  column-gap: 18px;
  row-gap: 16px;
  align-items: start;
  padding: 14px 16px 16px;
}

.form-grid :deep(.cell-wide) {
  grid-column: span 2;
}

.form-loading {
  padding: 20px;
}

.schema-form :deep(.el-form-item) {
  margin-bottom: 0;
}

.schema-form :deep(.el-form-item__label) {
  white-space: normal;
  word-break: break-word;
  line-height: 1.3;
  height: auto;
  padding: 0 8px 0 0;
  font-size: 13px;
}

.schema-form :deep(.el-form-item__content) {
  line-height: 32px;
}

.schema-form :deep(.el-input),
.schema-form :deep(.el-select),
.schema-form :deep(.el-input-number),
.schema-form :deep(.el-checkbox-group) {
  width: 100%;
  max-width: 100%;
}
</style>

<script setup lang="ts">
/**
 * SchemaFormItem. Individual form field renderer with control dispatch,
 * localization, validation, and conditional visibility/enabled/required.
 */
import { computed, ref } from 'vue'
import { ElFormItem, ElInput, ElInputNumber, ElSelect, ElSwitch } from 'element-plus'
import { QuestionFilled } from '@element-plus/icons-vue'
import { useFieldConditions } from '@/composables/useFieldConditions'
import { useLocale } from '@/composables/useLocale'
import { useFormStateStore } from '@/stores/formState'
import type { LocalizedText } from '@/composables/useLocale'
import type { Option } from '@/composables/useOptionSources'

interface Props {
  field: Record<string, any>
  modelValue: any
  fieldError?: string
  options?: Option[]
  loading?: boolean
  // When a dynamic option source fails to load, the parent forwards the error
  // message here so we can surface it inline (with a retry) instead of
  // silently rendering an empty dropdown. See PR-632 #19.
  optionsError?: string
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
  optionsError: undefined,
})

const emit = defineEmits<{
  'update:modelValue': [value: any]
  'retry-options': []
}>()

const { t } = useLocale()

// Resolve localized text
const resolveLabel = (label: LocalizedText): string => {
  if (typeof label === 'string') return label
  return t(label)
}

const fieldLabel = computed(() => resolveLabel(props.field.label || ''))
const fieldTooltip = computed(() => (props.field.tooltip ? resolveLabel(props.field.tooltip) : ''))
const fieldPlaceholder = computed(() => (props.field.placeholder ? resolveLabel(props.field.placeholder) : ''))

// Evaluate field conditions against the FULL form model, not just this field's
// own value. Conditions commonly reference OTHER fields (e.g. enable_multistream
// depends on compile; mxfp4_group_size depends on quantize actions; profiling_*
// depends on performance_model). A per-field { id: value } snapshot would leave
// those cross-field references unresolved (operand -> MISSING -> always false).
const formState = useFormStateStore()
const formModel = computed(() => formState.form)
const { isVisible, isEnabled, isRequired } = useFieldConditions(props.field.conditions, formModel)

// Internal value state
const internalValue = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val),
})

// Get validation rules for Element Plus
const validationRules = computed(() => {
  const rules: any[] = []
  if (!props.field.validation) return rules

  for (const rule of props.field.validation) {
    const elRule: any = {
      required: rule.rule === 'required',
      message: rule.message ? resolveLabel(rule.message) : '',
      trigger: rule.trigger || 'blur',
    }

    // Handle min/max/len. Schema rules are declared as {rule:'min', value:N}
    // (the number control below also reads `.value` for its own min/max). Map
    // by rule type so Element Plus's min/max/len length/range checks actually
    // fire — the old `rule.min`/`rule.max`/`rule.len` reads were always undefined.
    if (rule.rule === 'min' && rule.value !== undefined) elRule.min = rule.value
    if (rule.rule === 'max' && rule.value !== undefined) elRule.max = rule.value
    if (rule.rule === 'len' && rule.value !== undefined) elRule.len = rule.value

    // Handle pattern
    if (rule.rule === 'pattern' && rule.value) {
      elRule.pattern = rule.value
    }

    // Handle enum
    if (rule.rule === 'enum' && rule.value) {
      elRule.type = 'enum'
      elRule.enum = rule.value
    }

    // Type-specific rules
    if (rule.type) {
      elRule.type = rule.type
    }

    rules.push(elRule)
  }

  return rules.length > 0 ? rules : undefined
})

// Determine required status
const required = computed(() => {
  const hasRequiredRule = props.field.validation?.some((v: any) => v.rule === 'required')
  return hasRequiredRule || isRequired.value
})

// A field is disabled when it carries a static `disabled: true` flag OR its
// `conditions.enabled` predicate currently evaluates false. The static flag
// locks fields that must never be user-editable (e.g. compile force-enabled).
const isDisabled = computed(() => props.field.disabled === true || !isEnabled.value)

// Mark field as touched on blur (validation handled by parent)
const onBlur = () => {}

// Emit value change (for select/switch)
const onChange = (value: any) => {
  emit('update:modelValue', value)
}

// el-form-item instance ref. Exposed so the parent SchemaForm can resolve a
// field's rendered DOM root (.$el) for scroll-to-error. el-form-item does NOT
// render a `prop` DOM attribute, so the parent cannot locate fields via
// document.querySelector('[prop=...]'); the Vue ref graph is the reliable path.
const formItemRef = ref<InstanceType<typeof ElFormItem> | null>(null)

defineExpose({ formItemRef })
</script>

<template>
  <el-form-item
    v-if="isVisible"
    ref="formItemRef"
    :label="fieldLabel"
    :prop="field.id"
    :required="required"
    :rules="validationRules"
    :show-message="!!fieldError"
    :error="fieldError"
  >
    <!-- Text input -->
    <el-input
      v-if="field.control === 'text'"
      v-model="internalValue"
      :placeholder="fieldPlaceholder"
      :disabled="isDisabled"
      @blur="onBlur"
    />

    <!-- Number input -->
    <el-input-number
      v-else-if="field.control === 'number'"
      v-model="internalValue"
      :placeholder="fieldPlaceholder"
      :disabled="isDisabled"
      :min="field.validation?.find((v: any) => v.rule === 'min')?.value"
      :max="field.validation?.find((v: any) => v.rule === 'max')?.value"
      :step="field.dataType === 'integer' ? 1 : 0.1"
      controls-position="right"
      @blur="onBlur"
    />

    <!-- Select dropdown -->
    <el-select
      v-else-if="field.control === 'select'"
      v-model="internalValue"
      :placeholder="fieldPlaceholder"
      :disabled="isDisabled || !!optionsError"
      :loading="loading"
      @blur="onBlur"
      @change="onChange"
    >
      <el-option
        v-for="opt in options"
        :key="opt.value"
        :label="typeof opt.label === 'string' ? opt.label : t(opt.label)"
        :value="opt.value"
      />
    </el-select>

    <!-- Multi-select dropdown (el-select multiple) -->
    <el-select
      v-else-if="field.control === 'multi-select'"
      v-model="internalValue"
      multiple
      collapse-tags
      collapse-tags-tooltip
      filterable
      clearable
      :placeholder="fieldPlaceholder"
      :disabled="isDisabled || !!optionsError"
      :loading="loading"
      class="msm-multi-select"
      @change="onChange"
    >
      <el-option
        v-for="opt in options"
        :key="opt.value"
        :label="typeof opt.label === 'string' ? opt.label : t(opt.label)"
        :value="opt.value"
      />
    </el-select>

    <!-- Switch -->
    <el-switch
      v-else-if="field.control === 'switch'"
      v-model="internalValue"
      :disabled="isDisabled"
      @change="onChange"
    />

    <!-- Options-load failure hint: surfaced inline with a retry button so the
         user isn't stuck staring at an empty dropdown with no explanation
         (previously `optionsErrors` was populated but never rendered). -->
    <div v-if="optionsError" class="msm-options-error">
      <span>{{ optionsError }}</span>
      <button type="button" class="msm-options-retry" @click="emit('retry-options')">
        {{ t({ zh: '重试', en: 'Retry' }) }}
      </button>
    </div>

    <!-- Tooltip help icon -->
    <template #label>
      <span>{{ fieldLabel }}</span>
      <el-tooltip v-if="fieldTooltip" :content="fieldTooltip" placement="top">
        <el-icon style="margin-left: 4px; color: var(--el-color-info);">
          <QuestionFilled />
        </el-icon>
      </el-tooltip>
    </template>
  </el-form-item>
</template>

<style scoped>
.msm-options-error {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 4px;
  font-size: 12px;
  color: var(--el-color-danger);
}
.msm-options-retry {
  padding: 0 6px;
  height: 20px;
  font-size: 12px;
  color: var(--el-color-primary);
  background: transparent;
  border: 1px solid currentColor;
  border-radius: 4px;
  cursor: pointer;
  transition: background var(--msm-transition-fast) var(--msm-ease-out);
}
.msm-options-retry:hover {
  background: color-mix(in srgb, var(--el-color-primary) 10%, transparent);
}
</style>
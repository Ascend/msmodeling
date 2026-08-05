<script setup lang="ts">
/**
 * ECharts wrapper component.
 *
 * Provides a reusable vue-echarts wrapper with:
 * - Auto resize handling
 * - Theme/locale-aware updates
 * - Error boundary for chart failures
 *
 * The <v-chart> is ALWAYS mounted (gated only by `error`) so the canvas paints
 * as soon as its option arrives — do NOT gate it behind a loading flag, or the
 * chart never mounts and its readiness event never fires (a previous skeleton
 * overlay created exactly that deadlock). The 200px min-height on the wrapper
 * reserves the slot so there is no layout jump before paint.
 */
import { ref, computed, onBeforeUnmount, watch } from 'vue'
import { useChartTheme } from '@/composables/useChartTheme'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart, PieChart, ScatterChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DatasetComponent,
  TransformComponent,
  ToolboxComponent,
  DataZoomComponent
} from 'echarts/components'

// Register required ECharts components
use([
  CanvasRenderer,
  BarChart,
  LineChart,
  PieChart,
  ScatterChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DatasetComponent,
  TransformComponent,
  ToolboxComponent,
  DataZoomComponent
])

interface Props {
  option: Record<string, any>
  height?: string
  width?: string
  theme?: string
}

const props = withDefaults(defineProps<Props>(), {
  height: '400px',
  width: '100%',
  theme: 'default'
})

// Forward raw ECharts mouse events so parents can react (e.g. drill-down on a
// bar click). Kept optional: parents that don't bind @chart-click are unaffected.
const emit = defineEmits<{
  (e: 'chart-click', params: any): void
}>()

// Theme-aware chart styling. ECharts options can't read CSS vars, so all
// colors come from useChartTheme(). baseOption (color palette, textStyle,
// legend, tooltip) is reactive to the light/dark toggle; the cartesian axis
// text / gridline family uses the light tokens (readable on both panels).
const { baseOption, axisText, splitLine } = useChartTheme()

// Normalize a cartesian axis (object OR array of objects) with the current
// theme's axis text / gridline colours.
function withAxis(axis: any): any {
  if (axis == null) return axis
  const fix = (a: any) => ({
    ...a,
    axisLabel: { color: axisText.value, ...(a.axisLabel || {}) },
    nameTextStyle: { color: axisText.value, ...(a.nameTextStyle || {}) },
    axisLine: { lineStyle: { color: splitLine.value }, ...(a.axisLine || {}) },
    splitLine: { lineStyle: { color: splitLine.value }, ...(a.splitLine || {}) },
  })
  return Array.isArray(axis) ? axis.map(fix) : fix(axis)
}

// Inject theme styling (color/textStyle/legend/tooltip from the reactive
// baseOption, then an axis pass) into options. Per-option explicit styles
// still win (caller spread last).
const mergedOption = computed(() => {
  const opt = props.option
  if (!opt) return opt
  const out: Record<string, any> = { ...baseOption.value, ...opt }
  if (opt.xAxis) out.xAxis = withAxis(opt.xAxis)
  if (opt.yAxis) out.yAxis = withAxis(opt.yAxis)
  return out
})

const error = ref<string | null>(null)

onBeforeUnmount(() => {
  // ECharts cleanup handled by vue-echarts
})

// An empty option is a genuine "not ready" — surface it as the error branch
// (which keeps the wrapper's reserved height) instead of rendering an empty chart.
watch(
  () => props.option,
  (newOption) => {
    error.value = newOption ? null : 'Chart option is empty'
  },
  { immediate: true, deep: true }
)
</script>

<template>
  <div class="chart-wrapper" :style="{ height, width }">
    <div v-if="error" class="chart-error">
      {{ error }}
    </div>
    <v-chart
      v-else
      ref="chartRef"
      :option="mergedOption"
      :theme="theme"
      :init-options="{ renderer: 'canvas' }"
      autoresize
      @click="(params: any) => emit('chart-click', params)"
    />
  </div>
</template>

<style scoped>
.chart-wrapper {
  min-height: 200px;
  position: relative;
}

.chart-error {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  height: 100%;
  color: var(--el-color-danger);
  font-size: 14px;
}
</style>

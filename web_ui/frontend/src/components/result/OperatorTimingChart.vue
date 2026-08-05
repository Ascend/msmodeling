<script setup lang="ts">
/**
 * OperatorTimingChart — top-half operator share (donut).
 *
 * Operators ranked by total time (desc); the top HALF (by count, e.g. 10 ops →
 * top 5) are shown individually, the rest are merged into a single "Other"
 * slice. Used alongside OperatorTimingTable (toggle views).
 *
 * op_breakdown items: { name, perf_total (s), perf_avg (s), call_times }
 */
import { computed } from 'vue'
import { useLocale } from '@/composables/useLocale'
import { useChartTheme } from '@/composables/useChartTheme'
import ChartWrapper from './ChartWrapper.vue'

const props = withDefaults(
  defineProps<{
    opBreakdown?: Array<Record<string, any>>
  }>(),
  {
    opBreakdown: () => [],
  },
)

const { t } = useLocale()
const { baseOption, axisText } = useChartTheme()

const otherLabel = computed(() => t({ zh: '其他', en: 'Other' }))

// Rank by total time desc; show the top ceil(n/2) individually, merge the rest
// into "Other" (e.g. 10 ops → top 5 + Other(rest 5)).
const chartData = computed(() => {
  const raw = props.opBreakdown
  if (!Array.isArray(raw) || raw.length === 0) return []
  const items = raw
    .map((r: any) => {
      // New format uses total_s; legacy uses perf_total
      const isNewFormat = 'total_s' in r || 'calls' in r
      const totalS = isNewFormat ? (Number(r.total_s) || 0) : (Number(r.perf_total) || 0)
      return {
        name: String(r.name || r.op_name || ''),
        totalMs: totalS * 1000,
      }
    })
    .filter((it) => it.totalMs > 0 && it.name)
    .sort((a, b) => b.totalMs - a.totalMs)
  if (items.length === 0) return []

  // Top HALF by count (e.g. 10 ops → top 5); the rest merge into "Other".
  // Was /3 which over-hid detail (10 → 4 standalone + 6 into Other); restored
  // to /2 to match this component's doc comment ("top half").
  const topCount = Math.max(1, Math.ceil(items.length / 2))
  const top = items
    .slice(0, topCount)
    .map((it) => ({ name: it.name, value: Number(it.totalMs.toFixed(3)) }))

  const otherTotal = items
    .slice(topCount)
    .reduce((s, it) => s + it.totalMs, 0)
  if (otherTotal > 0) {
    top.push({ name: otherLabel.value, value: Number(otherTotal.toFixed(3)) })
  }
  return top
})

const chartOption = computed(() => ({
  ...baseOption.value,
  tooltip: {
    ...baseOption.value.tooltip,
    trigger: 'item',
    formatter: (p: any) => `${p.name}: ${p.value} ms (${p.percent}%)`,
  },
  legend: {
    ...baseOption.value.legend,
    type: 'scroll',
    orient: 'vertical',
    left: 0,
    top: 'middle',
    formatter: (name: string) => (name.length > 28 ? name.slice(0, 27) + '…' : name),
  },
  series: [
    {
      type: 'pie',
      radius: ['38%', '66%'],
      center: ['62%', '50%'],
      avoidLabelOverlap: true,
      data: chartData.value,
      label: {
        formatter: '{d}%',
        color: axisText.value,
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 8,
          shadowOffsetX: 0,
          shadowColor: 'rgba(15, 23, 42, 0.10)',
        },
      },
    },
  ],
}))
</script>

<template>
  <div v-if="chartData.length > 0" class="op-timing-chart">
    <ChartWrapper :option="chartOption" height="360px" />
  </div>
</template>

<style scoped>
.op-timing-chart {
  margin-top: 8px;
}
</style>

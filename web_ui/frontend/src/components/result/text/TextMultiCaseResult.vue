<script setup lang="ts">
/**
 * Text multi-case comparison view (Phase D2).
 *
 * Shown when a text_generate job's result envelope has `multi_case: true`.
 * Sections:
 * - case summary table (each case × run_time / tps / peak_mem / error)
 * - TPS-across-cases bar chart
 * - op comparison table (top ops × cases; toggle analytic total / avg)
 * - op detail table (the selected case's Name / total / avg / # of Calls)
 * - full drill-down of the selected case (reuses TextGenerateResult)
 */
import { ref, computed } from 'vue'
import { useLocale } from '@/composables/useLocale'
import { useChartTheme } from '@/composables/useChartTheme'
import { ElTag } from 'element-plus'
import ChartWrapper from '../ChartWrapper.vue'
import TextGenerateResult from './TextGenerateResult.vue'
import ChromeTraceDownloads from '../ChromeTraceDownloads.vue'

interface Props {
  result: Record<string, any>
  jobId: string
}
const props = defineProps<Props>()
const { t } = useLocale()
const { baseOption, category, axisText, splitLine } = useChartTheme()

const cases = computed(() => props.result.cases || [])
const selectedIdx = ref(0)
const selectedCase = computed(() => cases.value[selectedIdx.value] ?? null)
const selectedEnvelope = computed(() => {
  if (!selectedCase.value) return {}
  const { config, summary, ...env } = selectedCase.value
  return { mode: 'text_generation', ...env }
})

// Add seq to cases for ChromeTraceDownloads
const casesWithSeq = computed(() =>
  cases.value.map((c: any, i: number) => ({ ...c, seq: i }))
)

// 'total' (ms) or 'avg' (us) for the op comparison table.
const compareMetric = ref<'total' | 'avg'>('total')

const fmt = (v: any, digits = 3) =>
  v === null || v === undefined || v === '' ? '-' : Number(v).toFixed(digits)

const summaryRows = computed(() =>
  cases.value.map((c: any, i: number) => ({
    idx: i,
    device: c.config?.device,
    num_queries: c.config?.num_queries,
    quant: c.config?.quantize_linear_action,
    att: c.config?.quantize_attention_action,
    tp: c.config?.tp_size,
    run_time: c.run_time_s,
    tps: Object.values(c.tps_per_model || {})[0] ?? null,
    peak_mem: c.memory_gb?.peak_usage ?? null,
    error: c.summary?.error || '',
    seq: i,
  })),
)

const tpsCompareOption = computed(() => {
  const labels = summaryRows.value.map((r: any) => `#${r.idx + 1}`)
  const values = summaryRows.value.map((r: any) => r.tps)
  return {
    ...baseOption.value,
    title: { text: t({ zh: '各用例 TPS 对比', en: 'TPS across cases' }), left: 'center' },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...baseOption.value.tooltip },
    xAxis: { type: 'category', data: labels, axisLabel: { interval: 0, color: axisText.value } },
    yAxis: { type: 'value', name: t({ zh: 'Token/秒', en: 'Token/s' }), axisLabel: { color: axisText.value }, splitLine: { lineStyle: { color: splitLine.value } } },
    series: [{ type: 'bar', data: values, itemStyle: { color: category.value[1] } }],
  }
})

// Union of op names across cases, each row = { name, values: [per-case number] }.
// Sorted by the max value across cases (top first), capped for readability.
const opComparisonRows = computed(() => {
  const nameMax = new Map<string, number>()
  const perCase: Map<string, number[]> = new Map()
  const n = cases.value.length
  cases.value.forEach((c: any) => {
    const items: any[] = c.op_breakdown || []
    items.forEach((it) => {
      const name = it.name
      // New format uses total_s/avg_s; legacy uses perf_total/perf_avg
      const isNewFormat = 'total_s' in it || 'calls' in it
      const val = compareMetric.value === 'total'
        ? (isNewFormat ? (Number(it.total_s) || 0) : (Number(it.perf_total) || 0))
        : (isNewFormat ? (Number(it.avg_s) || 0) : (Number(it.perf_avg) || 0))
      const arr = perCase.get(name) || new Array(n).fill(null)
      arr[selectedIdx.value === 0 ? cases.value.indexOf(c) : cases.value.indexOf(c)] = val
      perCase.set(name, arr)
      nameMax.set(name, Math.max(nameMax.get(name) ?? 0, val))
    })
  })
  // build full per-case arrays (fill nulls)
  const rows = [...perCase.entries()].map(([name]) => {
    // rebuild a clean per-case array (perCase was built naively/sparse):
    const vals = cases.value.map((c: any) => {
      const it = (c.op_breakdown || []).find((x: any) => x.name === name)
      if (!it) return null
      const isNewFormat = 'total_s' in it || 'calls' in it
      // Number(x) ?? null is a no-op fallback (NaN isn't nullish) and would let
      // NaN leak into cells ("NaN ms/us") and into Math.max, breaking the sort.
      // Coerce then keep only finite numbers; otherwise null so downstream
      // null-filters and sorts only ever see valid numbers.
      const raw = compareMetric.value === 'total'
        ? (isNewFormat ? it.total_s : it.perf_total)
        : (isNewFormat ? it.avg_s : it.perf_avg)
      const v = Number(raw)
      return Number.isFinite(v) ? v : null
    })
    return { name, values: vals, max: Math.max(...vals.filter((v: any) => v !== null).map(Number), 0) }
  })
  rows.sort((a, b) => b.max - a.max)
  return rows.slice(0, 15)
})

const compareUnit = computed(() =>
  compareMetric.value === 'total' ? 'ms' : 'us',
)
function fmtCompare(v: any) {
  if (v === null || v === undefined) return '-'
  // value is in seconds; total -> ms, avg -> us
  const n = Number(v)
  return (compareMetric.value === 'total' ? n * 1000 : n * 1e6).toFixed(3)
}

function selectRow(row: any) {
  selectedIdx.value = row.idx
}
</script>

<template>
  <div class="text-multi-case">
    <!-- 1. case summary -->
    <div class="mc-section">
      <div class="mc-hint">
        {{ t({ zh: 'sim_run = 仿真程序耗时（含编译）；tps / peak_mem = 模型仿真指标', en: 'sim_run = simulator wall-clock (incl. compile); tps / peak_mem = model sim metrics' }) }}
      </div>
      <el-table
        :data="summaryRows"
        size="small"
        border
        highlight-current-row
        @row-click="selectRow"
      >
        <el-table-column label="#" width="48" type="index" />
        <el-table-column label="device" prop="device" min-width="150" show-overflow-tooltip />
        <el-table-column label="num_queries" prop="num_queries" width="70" />
        <el-table-column label="quant" prop="quant" min-width="120" show-overflow-tooltip />
        <el-table-column label="att" prop="att" min-width="90" show-overflow-tooltip />
        <el-table-column label="tp" prop="tp" width="50" />
        <el-table-column label="tps" width="110">
          <template #default="{ row }">{{ row.tps === null ? '-' : fmt(row.tps, 1) }}</template>
        </el-table-column>
        <el-table-column label="peak_mem(GB)" width="120">
          <template #default="{ row }">{{ fmt(row.peak_mem, 2) }}</template>
        </el-table-column>
        <el-table-column label="sim_run(s)" width="110">
          <template #default="{ row }">{{ fmt(row.run_time) }}</template>
        </el-table-column>
        <el-table-column label="error" min-width="120">
          <template #default="{ row }">
            <span v-if="row.error" class="mc-error">{{ row.error }}</span>
            <span v-else class="mc-ok">OK</span>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 2. TPS comparison chart -->
    <div class="mc-section">
      <el-card>
        <ChartWrapper :option="tpsCompareOption" height="280px" />
      </el-card>
    </div>

    <!-- 3. op comparison table (toggle total/avg) -->
    <div class="mc-section">
      <div class="mc-section-head">
        <span>{{ t({ zh: '算子耗时对比', en: 'Op timing comparison' }) }}</span>
        <el-radio-group v-model="compareMetric" size="small">
          <el-radio-button value="total">{{ t({ zh: 'analytic total', en: 'analytic total' }) }}</el-radio-button>
          <el-radio-button value="avg">{{ t({ zh: 'analytic avg', en: 'analytic avg' }) }}</el-radio-button>
        </el-radio-group>
      </div>
      <el-table :data="opComparisonRows" size="small" border max-height="360">
        <el-table-column label="Name" prop="name" min-width="240" show-overflow-tooltip />
        <el-table-column
          v-for="(_, i) in cases"
          :key="i"
          :label="`#${i + 1}`"
          width="110"
        >
          <template #default="{ row }">{{ fmtCompare(row.values[i]) }} {{ compareUnit }}</template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 4. full drill-down (selected case) -->
    <div v-if="selectedCase" class="mc-section mc-detail">
      <h4>
        {{ t({ zh: '用例', en: 'Case' }) }} #{{ selectedIdx + 1 }}
        {{ t({ zh: '完整指标', en: 'full metrics' }) }}
        ({{ t({ zh: '点击最上方表格行切换', en: 'click a summary row to switch' }) }})
      </h4>
      <TextGenerateResult :result="selectedEnvelope" :job-id="jobId" hide-trace-downloads />
    </div>

    <!-- 5. Chrome Trace Downloads -->
    <ChromeTraceDownloads :job-id="jobId" :cases="casesWithSeq" />
  </div>
</template>

<style scoped>
.text-multi-case {
  padding: 12px 16px;
}
.mc-section {
  margin-bottom: 18px;
}
.mc-hint {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  margin-bottom: 8px;
}
.mc-section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 600;
}
.mc-error {
  color: var(--el-color-danger);
  font-size: 12px;
}
.mc-ok {
  color: var(--msm-green);
  font-size: 12px;
}
.mc-detail h4 {
  margin: 0 0 12px;
  font-size: 14px;
  font-weight: 600;
}
.op-name,
.input-shapes {
  font-family: 'Fira Code', monospace;
  font-size: 11px;
  color: var(--msm-text-muted);
  word-break: break-all;
}
</style>

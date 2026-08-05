<script setup lang="ts">
/**
 * Video multi-case comparison view (Phase D2).
 *
 * Shown when a video_generate job's result envelope has `multi_case: true`
 * (device / quantize_linear_action / ulysses_size had multiple values).
 * Sections: case summary table + op comparison table (toggle total/avg) +
 * op detail table + drill-down reusing VideoGenerateResult.
 */
import { ref, computed } from 'vue'
import { useLocale } from '@/composables/useLocale'
import VideoGenerateResult from './VideoGenerateResult.vue'
import ChromeTraceDownloads from '../ChromeTraceDownloads.vue'

interface Props {
  result: Record<string, any>
  jobId: string
}
const props = defineProps<Props>()
const { t } = useLocale()

const cases = computed(() => props.result.cases || [])
const selectedIdx = ref(0)
const selectedCase = computed(() => cases.value[selectedIdx.value] ?? null)
const selectedEnvelope = computed(() => {
  if (!selectedCase.value) return {}
  const { config, summary, ...env } = selectedCase.value
  return { mode: 'video_generation', ...env }
})

// Add seq to cases for ChromeTraceDownloads
const casesWithSeq = computed(() =>
  cases.value.map((c: any, i: number) => ({ ...c, seq: i }))
)

const compareMetric = ref<'total' | 'avg'>('total')
const compareUnit = computed(() => (compareMetric.value === 'total' ? 'ms' : 'us'))

const fmt = (v: any, digits = 3) =>
  v === null || v === undefined || v === '' ? '-' : Number(v).toFixed(digits)

function execTotal(c: any): number | null {
  const et = c.execution_time_s
  if (!et || typeof et !== 'object') return null
  const vals = Object.values(et).filter((v) => typeof v === 'number') as number[]
  return vals.length ? vals.reduce((a, b) => a + b, 0) : null
}

const summaryRows = computed(() =>
  cases.value.map((c: any, i: number) => ({
    idx: i,
    device: c.config?.device,
    quant: c.config?.quantize_linear_action,
    ulysses: c.config?.ulysses_size,
    exec_time: execTotal(c),
    error: c.summary?.error || '',
  })),
)

const opComparisonRows = computed(() => {
  const n = cases.value.length
  const rows = cases.value
    .map((c: any, ci: number) => {
      const items: any[] = c.op_breakdown || []
      return items.map((it) => ({
        name: it.name,
        caseIdx: ci,
        val: compareMetric.value === 'total' ? it.perf_total ?? null : it.perf_avg ?? null,
      }))
    })
    .flat()
  // group by op name -> per-case values
  const byName = new Map<string, (number | null)[]>()
  for (const r of rows) {
    const arr = byName.get(r.name) || new Array(n).fill(null)
    arr[r.caseIdx] = r.val
    byName.set(r.name, arr)
  }
  const out = [...byName.entries()].map(([name, vals]) => ({
    name,
    values: vals,
    max: Math.max(...vals.filter((v) => v !== null).map(Number), 0),
  }))
  out.sort((a, b) => b.max - a.max)
  return out.slice(0, 15)
})

function fmtCompare(v: any) {
  if (v === null || v === undefined) return '-'
  const num = Number(v)
  return (compareMetric.value === 'total' ? num * 1000 : num * 1e6).toFixed(3)
}

function selectRow(row: any) {
  selectedIdx.value = row.idx
}
</script>

<template>
  <div class="video-multi-case">
    <!-- 1. case summary -->
    <div class="mc-section">
      <el-table :data="summaryRows" size="small" border highlight-current-row @row-click="selectRow">
        <el-table-column label="#" width="48" type="index" />
        <el-table-column label="device" prop="device" min-width="150" show-overflow-tooltip />
        <el-table-column label="quant" prop="quant" min-width="120" show-overflow-tooltip />
        <el-table-column label="ulysses" prop="ulysses" width="80" />
        <el-table-column label="exec time(s)" width="130">
          <template #default="{ row }">{{ row.exec_time === null ? '-' : fmt(row.exec_time) }}</template>
        </el-table-column>
        <el-table-column label="error" min-width="120">
          <template #default="{ row }">
            <span v-if="row.error" class="mc-error">{{ row.error }}</span>
            <span v-else class="mc-ok">OK</span>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 2. op comparison table (toggle total/avg) -->
    <div class="mc-section">
      <div class="mc-section-head">
        <span>{{ t({ zh: '算子耗时对比', en: 'Op timing comparison' }) }}</span>
        <el-radio-group v-model="compareMetric" size="small">
          <el-radio-button value="total">{{ t({ zh: 'analytic total', en: 'analytic total' }) }}</el-radio-button>
          <el-radio-button value="avg">{{ t({ zh: 'analytic avg', en: 'analytic avg' }) }}</el-radio-button>
        </el-radio-group>
      </div>
      <el-table :data="opComparisonRows" size="small" border max-height="320">
        <el-table-column label="Name" prop="name" min-width="240" show-overflow-tooltip />
        <el-table-column v-for="(_, i) in cases" :key="i" :label="`#${i + 1}`" width="110">
          <template #default="{ row }">{{ fmtCompare(row.values[i]) }} {{ compareUnit }}</template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 3. drill-down -->
    <div v-if="selectedCase" class="mc-section mc-detail">
      <h4>
        {{ t({ zh: '用例', en: 'Case' }) }} #{{ selectedIdx + 1 }}
        {{ t({ zh: '完整指标（点击最上方表格行切换）', en: 'full metrics (click a row to switch)' }) }}
      </h4>
      <VideoGenerateResult :result="selectedEnvelope" :job-id="jobId" hide-trace-downloads />
    </div>

    <!-- 4. Chrome Trace Downloads -->
    <ChromeTraceDownloads :job-id="jobId" :cases="casesWithSeq" />
  </div>
</template>

<style scoped>
.video-multi-case { padding: 12px 16px; }
.mc-section { margin-bottom: 18px; }
.mc-section-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; font-size: 14px; font-weight: 600; }
.mc-error { color: var(--el-color-danger); font-size: 12px; }
.mc-ok { color: var(--msm-green); font-size: 12px; }
.mc-detail h4 { margin: 0 0 12px; font-size: 14px; font-weight: 600; }
</style>

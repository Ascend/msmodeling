<script setup lang="ts">
/**
 * Throughput multi-case comparison view (Phase D2).
 *
 * Shown when a throughput_optimizer job's result envelope has `multi_case: true`.
 *
 * Mode-aware:
 * - aggregation / pd_ratio: one summary table (best metric per case) + chart.
 * - disaggregated: TWO comparisons — Prefill (best prefill per case) and Decode
 *   (best decode per case) — mirroring the CLI's two independent tables. A single
 *   summary can't represent disagg because prefill throughput >> decode, so the
 *   rank=1 best_config is always a prefill row and Decode would be hidden.
 *
 * Always followed by a full drill-down of the selected case (click any row).
 */
import { ref, computed } from 'vue'
import { useLocale } from '@/composables/useLocale'
import { useChartTheme } from '@/composables/useChartTheme'
import ChartWrapper from '../ChartWrapper.vue'
import ThroughputOptimizerResult from './ThroughputOptimizerResult.vue'
import ChromeTraceDownloads from '../ChromeTraceDownloads.vue'

interface Props {
  result: Record<string, any>
  records?: any[]
  jobId?: string
}
const props = defineProps<Props>()
const { t } = useLocale()
const { baseOption, category, axisText, splitLine, tooltipBg, tooltipBorder, tooltipText } = useChartTheme()

const cases = computed(() => props.result.cases || [])
const selectedIdx = ref(0)
const selectedCase = computed(() => cases.value[selectedIdx.value] ?? null)

// Cases with seq numbers for ChromeTraceDownloads
const casesWithSeq = computed(() =>
  cases.value.map((c: any, i: number) => ({
    ...c,
    seq: i,
    config: c.case_config || c.config || {},
    chrome_trace: c.chrome_trace || { available: false }
  }))
)

// Filter raw records for the selected case (per-case curves). Uses case_hash
// when available; falls back to matching by device from case_config when
// case_hash is null (e.g. form_schema_version was not set).
const selectedCaseRecords = computed(() => {
  if (!props.records?.length || !selectedCase.value) return []
  const ch = selectedCase.value.case_hash
  if (ch) {
    return props.records.filter((r: any) => r.case_hash === ch)
  }
  // Fallback: match by device from case_config
  const device = selectedCase.value.case_config?.device
  if (device) {
    return props.records.filter((r: any) => {
      const d = r.config?.device
      return Array.isArray(d) ? d.includes(device) : d === device
    })
  }
  return []
})

const fmt = (v: any, digits = 2) =>
  v === null || v === undefined || v === '' ? '-' : Number(v).toFixed(digits)

// Detect PD-ratio mode (a case has balanced_qps, not throughput_token_s).
// Check ANY case — not just cases[0], which may be a failed case with no
// best_config and would wrongly collapse to plain aggregation, hiding the
// PD-Ratio / P-D QPS results. Mirrors isDisagg's .some() below.
const isPdRatio = computed(() =>
  cases.value.some((c: any) => c.best_config?.balanced_qps != null),
)

// Detect disaggregated mode: cases carry disagg_prefill / disagg_decode arrays.
const isDisagg = computed(
  () =>
    !isPdRatio.value &&
    cases.value.some(
      (c: any) => (c.disagg_prefill?.length || 0) + (c.disagg_decode?.length || 0) > 0,
    ),
)

const summaryRows = computed(() =>
  cases.value.map((c: any, i: number) => {
    const cc = c.case_config || {}
    const best = c.best_config || {}
    if (isPdRatio.value) {
      return {
        idx: i,
        device: cc.device,
        tpot: cc.tpot_limits,
        ttft: cc.ttft_limits,
        quant: cc.quantize_linear_action,
        att: cc.quantize_attention_action,
        primary: best.balanced_qps ?? null,
        primaryLabel: 'Balanced QPS',
        pd_ratio: best.pd_ratio ?? null,
        p_qps: best.p_qps ?? null,
        d_qps: best.d_qps ?? null,
        bestParallel: `P:${best.parallel_p || '-'} D:${best.parallel_d || '-'}`,
        error: c.best_config ? '' : 'no result',
      }
    }
    return {
      idx: i,
      device: cc.device,
      tpot: cc.tpot_limits,
      ttft: cc.ttft_limits,
      quant: cc.quantize_linear_action,
      att: cc.quantize_attention_action,
      primary: best.throughput_token_s ?? null,
      primaryLabel: 'Throughput',
      ttft_ms: best.ttft_ms ?? null,
      tpot_ms: best.tpot_ms ?? null,
      bestParallel: best.parallel || '-',
      error: c.best_config ? '' : 'no result',
    }
  }),
)

// Bar chart: primary metric across cases
const compareChartOption = computed(() => {
  const labels = summaryRows.value.map((r: any) => `#${r.idx + 1}`)
  const values = summaryRows.value.map((r: any) => r.primary)
  const metricName = isPdRatio.value
    ? t({ zh: 'Balanced QPS (req/s)', en: 'Balanced QPS (req/s)' })
    : t({ zh: '吞吐 (token/s)', en: 'Throughput (token/s)' })
  return {
    ...baseOption.value,
    title: { text: t({ zh: '各用例最佳指标对比', en: 'Best metric across cases' }), left: 'center' },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: tooltipBg.value,
      borderColor: tooltipBorder.value,
      borderWidth: 1,
      textStyle: { color: tooltipText.value },
    },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { interval: 0, color: axisText.value },
      axisLine: { lineStyle: { color: splitLine.value } },
      splitLine: { lineStyle: { color: splitLine.value } },
    },
    yAxis: {
      type: 'value',
      name: metricName,
      nameTextStyle: { color: axisText.value },
      axisLabel: { color: axisText.value },
      splitLine: { lineStyle: { color: splitLine.value } },
    },
    series: [{ type: 'bar', data: values, itemStyle: { color: category.value[1] } }],
  }
})

// --- Disaggregated: best row per phase per case -----------------------------
// Each phase's sweep is rank-sorted, so rows[0] is the best of that phase.
function phaseBest(c: any, phase: string): any | null {
  const rows = c?.[phase] || []
  return rows.length ? rows[0] : null
}
function caseTags(cc: any, i: number) {
  return {
    idx: i,
    device: cc.device,
    tpot: cc.tpot_limits,
    ttft: cc.ttft_limits,
    quant: cc.quantize_linear_action,
    att: cc.quantize_attention_action,
  }
}
const prefillSummary = computed(() =>
  cases.value.map((c: any, i: number) => {
    const p = phaseBest(c, 'disagg_prefill')
    return { ...caseTags(c.case_config || {}, i), ...(p || {}), error: p ? '' : 'no prefill' }
  }),
)
const decodeSummary = computed(() =>
  cases.value.map((c: any, i: number) => {
    const d = phaseBest(c, 'disagg_decode')
    return { ...caseTags(c.case_config || {}, i), ...(d || {}), error: d ? '' : 'no decode' }
  }),
)
function phaseChart(rows: any[], title: string) {
  return {
    ...baseOption.value,
    title: { text: title, left: 'center' },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: tooltipBg.value,
      borderColor: tooltipBorder.value,
      borderWidth: 1,
      textStyle: { color: tooltipText.value },
    },
    xAxis: {
      type: 'category',
      data: rows.map((r: any) => `#${r.idx + 1}`),
      axisLabel: { interval: 0, color: axisText.value },
      axisLine: { lineStyle: { color: splitLine.value } },
      splitLine: { lineStyle: { color: splitLine.value } },
    },
    yAxis: {
      type: 'value',
      name: 'token/s',
      nameTextStyle: { color: axisText.value },
      axisLabel: { color: axisText.value },
      splitLine: { lineStyle: { color: splitLine.value } },
    },
    series: [
      {
        type: 'bar',
        data: rows.map((r: any) => r.throughput_token_s),
        itemStyle: { color: category.value[1] },
      },
    ],
  }
}
const prefillChartOption = computed(() =>
  phaseChart(prefillSummary.value, t({ zh: '各用例 Prefill 最佳吞吐', en: 'Best Prefill throughput per case' })),
)
const decodeChartOption = computed(() =>
  phaseChart(decodeSummary.value, t({ zh: '各用例 Decode 最佳吞吐', en: 'Best Decode throughput per case' })),
)

function selectRow(row: any) {
  selectedIdx.value = row.idx
}
</script>

<template>
  <div class="thru-multi-case">
    <!-- Disaggregated: two independent phase comparisons (Prefill + Decode) -->
    <template v-if="isDisagg">
      <!-- Prefill comparison -->
      <div class="mc-section">
        <h4 class="phase-h">
          {{ t({ zh: 'Prefill 阶段对比（各用例最优，TTFT 约束）', en: 'Prefill comparison (best per case, TTFT-constrained)' }) }}
        </h4>
        <el-table :data="prefillSummary" size="small" border highlight-current-row @row-click="selectRow">
          <el-table-column label="#" width="48" type="index" />
          <el-table-column label="device" prop="device" min-width="150" show-overflow-tooltip />
          <el-table-column label="Throughput (token/s)" width="170">
            <template #default="{ row }">{{ row.throughput_token_s != null ? fmt(row.throughput_token_s) : '-' }}</template>
          </el-table-column>
          <el-table-column label="QPS (req/s)" width="120">
            <template #default="{ row }">{{ row.qps != null ? fmt(row.qps) : '-' }}</template>
          </el-table-column>
          <el-table-column label="TTFT (ms)" width="110">
            <template #default="{ row }">{{ row.ttft_ms != null ? fmt(row.ttft_ms) : '-' }}</template>
          </el-table-column>
          <el-table-column label="parallel" prop="parallel" min-width="160" show-overflow-tooltip />
          <el-table-column label="status" width="90">
            <template #default="{ row }">
              <span v-if="row.error" class="mc-error">{{ row.error }}</span>
              <span v-else class="mc-ok">OK</span>
            </template>
          </el-table-column>
        </el-table>
      </div>
      <div class="mc-section">
        <el-card><ChartWrapper :option="prefillChartOption" height="240px" /></el-card>
      </div>

      <!-- Decode comparison -->
      <div class="mc-section">
        <h4 class="phase-h">
          {{ t({ zh: 'Decode 阶段对比（各用例最优，TPOT 约束）', en: 'Decode comparison (best per case, TPOT-constrained)' }) }}
        </h4>
        <el-table :data="decodeSummary" size="small" border highlight-current-row @row-click="selectRow">
          <el-table-column label="#" width="48" type="index" />
          <el-table-column label="device" prop="device" min-width="150" show-overflow-tooltip />
          <el-table-column label="Throughput (token/s)" width="170">
            <template #default="{ row }">{{ row.throughput_token_s != null ? fmt(row.throughput_token_s) : '-' }}</template>
          </el-table-column>
          <el-table-column label="QPS (req/s)" width="120">
            <template #default="{ row }">{{ row.qps != null ? fmt(row.qps) : '-' }}</template>
          </el-table-column>
          <el-table-column label="TPOT (ms)" width="110">
            <template #default="{ row }">{{ row.tpot_ms != null ? fmt(row.tpot_ms) : '-' }}</template>
          </el-table-column>
          <el-table-column label="parallel" prop="parallel" min-width="160" show-overflow-tooltip />
          <el-table-column label="status" width="90">
            <template #default="{ row }">
              <span v-if="row.error" class="mc-error">{{ row.error }}</span>
              <span v-else class="mc-ok">OK</span>
            </template>
          </el-table-column>
        </el-table>
      </div>
      <div class="mc-section">
        <el-card><ChartWrapper :option="decodeChartOption" height="240px" /></el-card>
      </div>
    </template>

    <!-- Aggregation / PD-ratio: single summary + chart -->
    <template v-else>
      <div class="mc-section">
        <el-table :data="summaryRows" size="small" border highlight-current-row @row-click="selectRow">
          <el-table-column label="#" width="48" type="index" />
          <el-table-column label="device" prop="device" min-width="150" show-overflow-tooltip />
          <el-table-column label="TPOT" width="90">
            <template #default="{ row }">{{ row.tpot === null || row.tpot === undefined ? '∞' : fmt(row.tpot) }}</template>
          </el-table-column>
          <el-table-column label="TTFT" width="90">
            <template #default="{ row }">{{ row.ttft === null || row.ttft === undefined ? '∞' : fmt(row.ttft) }}</template>
          </el-table-column>
          <el-table-column label="Linear Quant" prop="quant" min-width="120" show-overflow-tooltip />
          <el-table-column label="Attn Quant" prop="att" min-width="90" show-overflow-tooltip />
          <el-table-column :label="isPdRatio ? 'Balanced QPS' : 'Throughput (token/s)'" width="170">
            <template #default="{ row }">{{ row.primary === null ? '-' : fmt(row.primary, 2) }}</template>
          </el-table-column>
          <!-- PD-ratio specific columns -->
          <template v-if="isPdRatio">
            <el-table-column label="PD Ratio" width="100">
              <template #default="{ row }">{{ row.pd_ratio === null ? '-' : fmt(row.pd_ratio) }}</template>
            </el-table-column>
            <el-table-column label="P QPS" width="100">
              <template #default="{ row }">{{ row.p_qps === null ? '-' : fmt(row.p_qps, 2) }}</template>
            </el-table-column>
            <el-table-column label="D QPS" width="100">
              <template #default="{ row }">{{ row.d_qps === null ? '-' : fmt(row.d_qps, 2) }}</template>
            </el-table-column>
          </template>
          <!-- Aggregation/disagg columns -->
          <template v-else>
            <el-table-column label="TTFT (ms)" width="110">
              <template #default="{ row }">{{ row.ttft_ms === null ? '-' : fmt(row.ttft_ms) }}</template>
            </el-table-column>
            <el-table-column label="TPOT (ms)" width="110">
              <template #default="{ row }">{{ row.tpot_ms === null ? '-' : fmt(row.tpot_ms) }}</template>
            </el-table-column>
          </template>
          <el-table-column label="best parallel" prop="bestParallel" min-width="200" show-overflow-tooltip />
          <el-table-column label="status" width="80">
            <template #default="{ row }">
              <span v-if="row.error" class="mc-error">{{ row.error }}</span>
              <span v-else class="mc-ok">OK</span>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <div class="mc-section">
        <el-card>
          <ChartWrapper :option="compareChartOption" height="280px" />
        </el-card>
      </div>
    </template>

    <!-- drill-down (click a table row to switch case) -->
    <Transition name="msm-fade" mode="out-in">
      <div v-if="selectedCase" :key="selectedIdx" class="mc-section mc-detail">
        <h4>
          {{ t({ zh: '用例', en: 'Case' }) }} #{{ selectedIdx + 1 }}
          {{ t({ zh: '完整结果（点击上表行切换）', en: 'full result (click a row to switch)' }) }}
        </h4>
        <ThroughputOptimizerResult :result="selectedCase" :records="selectedCaseRecords" :job-id="jobId" />
      </div>
    </Transition>

    <!-- Chrome trace downloads for all cases -->
    <ChromeTraceDownloads v-if="jobId" :job-id="jobId" :cases="casesWithSeq" />
  </div>
</template>

<style scoped>
.thru-multi-case { padding: 12px 16px; }
.mc-section { margin-bottom: 18px; }
.mc-error { color: var(--el-color-danger); font-size: 12px; }
.mc-ok { color: var(--msm-green); font-size: 12px; }
.mc-detail h4,
.phase-h { margin: 0 0 12px; font-size: 14px; font-weight: 600; }
.phase-h { color: var(--msm-text-muted); }

.thru-multi-case :deep(.el-table__row) {
  cursor: pointer;
  transition: background-color var(--msm-transition-fast) var(--msm-ease-out);
}
.thru-multi-case :deep(.el-table__row:hover) td {
  background: var(--msm-bg-panel-2) !important;
}
.thru-multi-case :deep(.el-table__row.current-row) td {
  background: var(--msm-accent-soft) !important;
}
</style>

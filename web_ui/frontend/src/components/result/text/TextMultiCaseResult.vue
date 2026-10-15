<script setup lang="ts">
/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

/**
 * Text multi-case comparison view (Phase D2).
 *
 * Shown when a text_generate job's result envelope has `multi-case: true`.
 * Sections:
 * - case summary table (each case × run-time / tps / peak-mem / error)
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
  return { mode: 'text-generation', ...env }
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
    num_queries: c.config?.["num_queries"],
    quant: c.config?.["quantize_linear_action"],
    att: c.config?.["quantize_attention_action"],
    tp: c.config?.["tp_size"],
    run_time: c["run_time_s"],
    tps: Object.values(c["tps_per_model"] || {})[0] ?? null,
    peak_mem: c["memory_gb"]?.["peak_usage"] ?? null,
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

// Aggregate op_breakdown entries by operator name.
//
// When `dump-input-shapes` is on, the backend emits one row per (op_name,
// input_shapes) tuple — the same operator name can appear multiple times
// with different shapes, each carrying its own `total_s` / `avg_s` / `calls`.
// The comparison table must NOT pick the first shape (`.find(x=>x.name===n)`)
// — it has to project onto the operator name by summing totals and computing
// a calls-weighted average. Otherwise the "analytic total" misses shapes and
// "analytic avg" reflects one shape, silently changing the ranking.
//
// Legacy payloads (perf_total / perf_avg, no `calls`) fall back to a simple
// sum of totals / simple average of avgs, since per-shape call counts are
// unavailable there. New-format rows whose `calls` are all zero across every
// shape for the same op name get the same simple-average treatment, since
// there are no call counts to weight by.
function aggregateByName(items: any[]): Map<string, { totalS: number; calls: number; avgS: number; isNew: boolean; hasCalls: boolean; count: number }> {
  const agg = new Map<string, { totalS: number; calls: number; avgS: number; isNew: boolean; hasCalls: boolean; count: number }>()
  for (const it of items) {
    const name = it.name as string
    const isNew = 'total_s' in it || 'calls' in it
    const totalS = isNew ? (Number(it['total_s']) || 0) : (Number(it['perf_total']) || 0)
    const calls = Number(it['calls']) || 0
    const avgS = isNew ? (Number(it['avg_s']) || 0) : (Number(it['perf_avg']) || 0)
    const cur = agg.get(name)
    if (!cur) {
      agg.set(name, { totalS, calls, avgS, isNew, hasCalls: calls > 0, count: 1 })
    } else {
      cur.totalS += totalS
      cur.calls += calls
      cur.avgS += avgS // accumulate; resolved to weighted or simple avg below
      cur.hasCalls = cur.hasCalls || calls > 0
      cur.count += 1
    }
  }
  // Derive the correct avg for each aggregated entry.
  // - New-format with recorded calls: calls-weighted avg = totalS / calls.
  // - Otherwise (legacy without calls, new-format with calls=0 across all
  //   shape rows, or a single-record name): use the simple average of
  //   per-row avgs when count > 1, else keep the recorded avgS as-is.
  for (const v of agg.values()) {
    if (v.isNew && v.hasCalls && v.calls > 0) {
      v.avgS = v.totalS / v.calls
    } else if (v.count > 1) {
      v.avgS = v.avgS / v.count
    }
  }
  return agg
}

// Union of op names across cases, each row = { name, values: [per-case number] }.
// Sorted by the max value across cases (top first), capped for readability.
const opComparisonRows = computed(() => {
  // Per-case aggregates: collapse same-name shape groups into one row per case.
  const perCaseAgg = cases.value.map((c: any) => aggregateByName(c['op_breakdown'] || []))

  // Collect the union of op names across all cases.
  const nameSet = new Set<string>()
  for (const agg of perCaseAgg) for (const k of agg.keys()) nameSet.add(k)

  const rows = [...nameSet].map((name) => {
    const vals = cases.value.map((_: any, i: number) => {
      const a = perCaseAgg[i].get(name)
      if (!a) return null
      const v = compareMetric.value === 'total' ? a.totalS : a.avgS
      return Number.isFinite(v) ? v : null
    })
    const finite = vals.filter((v: any) => v !== null).map(Number)
    return { name, values: vals, max: finite.length ? Math.max(...finite, 0) : 0 }
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

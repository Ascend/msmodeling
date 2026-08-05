<script setup lang="ts">
/**
 * Throughput optimizer exploration curves — scatter plots mirroring the CLI's
 * terminal ASCII curves (Throughput vs Concurrency + Throughput vs TPOT/TTFT).
 *
 * Uses RAW records (all explored configs), NOT the Top-N filtered sweep_rows.
 * Points are grouped by `parallel` (each parallel strategy = one colored series).
 *
 * Mode-aware:
 * - aggregation / pd_ratio: one chart group (all records)
 * - disagg: two chart groups (prefill rows with TTFT axis, decode rows with TPOT axis)
 */
import { computed } from 'vue'
import { useLocale } from '@/composables/useLocale'
import { useChartTheme } from '@/composables/useChartTheme'
import ChartWrapper from '../../ChartWrapper.vue'

interface Props {
  records?: any[]
  mode: string
}
const props = defineProps<Props>()
const { t } = useLocale()
const theme = useChartTheme()

const isPdRatio = computed(() => props.mode === 'pd_ratio')
const isDisagg = computed(() => props.mode === 'disagg_prefill' || props.mode === 'disagg_decode')

/** Extract the relevant fields from a record based on mode. */
function fields(rec: any, latencyKey: string) {
  const cfg = rec.config || {}
  const sm = rec.summary || {}
  if (isPdRatio.value) {
    // CLI's _pd_tps_curve_df: uses DECODE columns (parallel_d, concurrency_d,
    // tpot_d) and computes token/s = concurrency_d / tpot * 1000.
    const concurrency = cfg.d_concurrency ?? cfg.concurrency
    const tpot = sm.tpot_ms
    const tps = concurrency != null && tpot != null && tpot > 0
      ? (concurrency / tpot) * 1000
      : null
    return {
      parallel: cfg.parallel_d ?? cfg.parallel,
      concurrency,
      throughput: tps,
      latency: sm[latencyKey],
    }
  }
  return {
    parallel: cfg.parallel,
    concurrency: cfg.concurrency,
    throughput: sm.throughput_token_s,
    latency: sm[latencyKey],
  }
}

interface ChartGroup {
  label: string
  records: any[]
  latencyKey: string
  latencyLabel: string
}

/** Split records into chart groups: one for agg/pd_ratio, two for disagg (prefill + decode). */
const chartGroups = computed<ChartGroup[]>(() => {
  const recs = props.records
  if (!recs?.length) return []
  if (isDisagg.value) {
    const prefill = recs.filter(r => r.summary?.mode === 'disagg_prefill')
    const decode = recs.filter(r => r.summary?.mode === 'disagg_decode')
    return [
      ...(prefill.length ? [{ label: t({ zh: 'Prefill', en: 'Prefill' }), records: prefill, latencyKey: 'ttft_ms', latencyLabel: 'TTFT (ms)' }] : []),
      ...(decode.length ? [{ label: t({ zh: 'Decode', en: 'Decode' }), records: decode, latencyKey: 'tpot_ms', latencyLabel: 'TPOT (ms)' }] : []),
    ]
  }
  // Aggregation / PD ratio — all records, latency = TPOT
  return [{ label: '', records: recs, latencyKey: 'tpot_ms', latencyLabel: 'TPOT (ms)' }]
})

/** Build ECharts scatter series grouped by parallel.
 *  Mirrors the CLI's curve data selection:
 *  - _memory_filter: skip rows where device_memory_available_gb <= 0 (OOM)
 *  - drop_duplicates on (parallel, concurrency, latency) for PD ratio
 */
function buildSeries(recs: any[], xKey: 'concurrency' | 'latency', latencyKey: string) {
  const byParallel = new Map<string, [number, number][]>()
  const seen = new Set<string>()
  for (const rec of recs) {
    // Memory filter: skip OOM rows (mirrors CLI _memory_filter)
    const memAvail = rec.summary?.device_memory_available_gb
    if (memAvail != null && Number(memAvail) <= 0) continue

    const f = fields(rec, latencyKey)
    const x = xKey === 'concurrency' ? f.concurrency : f.latency
    const y = f.throughput
    if (x == null || y == null || isNaN(Number(x)) || isNaN(Number(y))) continue
    // Dedup: same (parallel, concurrency, latency) → skip (mirrors CLI drop_duplicates)
    const dedupKey = `${f.parallel}|${f.concurrency}|${f.latency}`
    if (seen.has(dedupKey)) continue
    seen.add(dedupKey)
    const p = String(f.parallel ?? '?')
    if (!byParallel.has(p)) byParallel.set(p, [])
    byParallel.get(p)!.push([Number(x), Number(y)])
  }
  const colors = theme.category.value
  return Array.from(byParallel.entries()).map(([name, data], i) => ({
    name,
    type: 'scatter' as const,
    data,
    itemStyle: { color: colors[i % colors.length] },
    symbolSize: 8,
  }))
}

/** ECharts option for Throughput vs Concurrency. */
function concurrencyOption(group: ChartGroup) {
  const series = buildSeries(group.records, 'concurrency', group.latencyKey)
  if (!series.length) return null
  return {
    ...theme.baseOption.value,
    tooltip: {
      ...theme.baseOption.value.tooltip,
      trigger: 'item',
      formatter: (p: any) =>
        `${p.seriesName}<br/>Concurrency: ${p.data[0]}<br/>Throughput: ${p.data[1].toFixed(1)} token/s`,
    },
    legend: { ...theme.baseOption.value.legend, top: 0 },
    grid: { top: 40, left: 60, right: 20, bottom: 50 },
    xAxis: {
      type: 'value',
      name: t({ zh: '并发数', en: 'Concurrency' }),
      nameLocation: 'middle',
      nameGap: 30,
      axisLine: { lineStyle: { color: theme.splitLine.value } },
      axisLabel: { color: theme.axisText.value },
      splitLine: { lineStyle: { color: theme.splitLine.value } },
    },
    yAxis: {
      type: 'value',
      name: isPdRatio.value
        ? t({ zh: '均衡 QPS', en: 'Balanced QPS' })
        : t({ zh: '吞吐 (token/s)', en: 'Throughput (token/s)' }),
      axisLine: { lineStyle: { color: theme.splitLine.value } },
      axisLabel: { color: theme.axisText.value },
      splitLine: { lineStyle: { color: theme.splitLine.value } },
    },
    series,
  }
}

/** ECharts option for Throughput vs Latency (TPOT or TTFT). */
function latencyOption(group: ChartGroup) {
  const series = buildSeries(group.records, 'latency', group.latencyKey)
  if (!series.length) return null
  return {
    ...theme.baseOption.value,
    tooltip: {
      ...theme.baseOption.value.tooltip,
      trigger: 'item',
      formatter: (p: any) =>
        `${p.seriesName}<br/>${group.latencyLabel}: ${p.data[0].toFixed(1)} ms<br/>Throughput: ${p.data[1].toFixed(1)} token/s`,
    },
    legend: { ...theme.baseOption.value.legend, top: 0 },
    grid: { top: 40, left: 60, right: 20, bottom: 50 },
    xAxis: {
      type: 'value',
      name: group.latencyLabel,
      nameLocation: 'middle',
      nameGap: 30,
      axisLine: { lineStyle: { color: theme.splitLine.value } },
      axisLabel: { color: theme.axisText.value },
      splitLine: { lineStyle: { color: theme.splitLine.value } },
    },
    yAxis: {
      type: 'value',
      name: isPdRatio.value
        ? t({ zh: '均衡 QPS', en: 'Balanced QPS' })
        : t({ zh: '吞吐 (token/s)', en: 'Throughput (token/s)' }),
      axisLine: { lineStyle: { color: theme.splitLine.value } },
      axisLabel: { color: theme.axisText.value },
      splitLine: { lineStyle: { color: theme.splitLine.value } },
    },
    series,
  }
}

// Precompute each chart group's options ONCE per reactive cycle. concurrencyOption
// and latencyOption each re-scan all RAW records (filter/sort/dedupe), and the
// template evaluated them twice per chart (v-if + :option) — so a single render
// re-scanned the records 4× per group. This computed caches both options per
// group so the heavy work runs at most once per group per update.
const groupOptions = computed(() =>
  chartGroups.value.map((group) => ({
    concurrency: concurrencyOption(group),
    latency: latencyOption(group),
  })),
)
</script>

<template>
  <div v-if="chartGroups.length" class="optimizer-curves">
    <div v-for="(group, gi) in chartGroups" :key="gi" class="curve-group">
      <div v-if="group.label" class="group-label">{{ group.label }}</div>
      <el-row :gutter="16">
        <el-col :xs="24" :md="12">
          <ChartWrapper
            v-if="groupOptions[gi]?.concurrency"
            :option="groupOptions[gi].concurrency!"
            height="320px"
          />
        </el-col>
        <el-col :xs="24" :md="12">
          <ChartWrapper
            v-if="groupOptions[gi]?.latency"
            :option="groupOptions[gi].latency!"
            height="320px"
          />
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<style scoped>
.optimizer-curves {
  margin-bottom: 16px;
}

.curve-group {
  margin-bottom: 12px;
}

.group-label {
  font-size: 14px;
  font-weight: 600;
  color: var(--msm-text);
  margin-bottom: 8px;
  padding-left: 4px;
  border-left: 3px solid var(--msm-accent);
}
</style>

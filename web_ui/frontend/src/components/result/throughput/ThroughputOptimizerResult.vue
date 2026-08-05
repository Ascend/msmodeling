<script setup lang="ts">
/**
 * Throughput optimizer result panel — mode-aware modular router.
 *
 * Routes to a mode-specific view based on result.mode:
 * - aggregation → AggregatedView (sweep table + cross-hw chart)
 * - disagg_prefill / disagg_decode → DisaggregatedView (prefill + decode tables)
 * - pd_ratio → PDRatioView (PD table + best PD ratio)
 *
 * The router renders a mode badge + the best_config card, then delegates the
 * mode-specific table(s)/chart(s) to the view component.
 */
import { computed } from 'vue'
import { useLocale } from '@/composables/useLocale'
import AggregatedView from './views/AggregatedView.vue'
import DisaggregatedView from './views/DisaggregatedView.vue'
import PDRatioView from './views/PDRatioView.vue'
import OptimizerCurves from './views/OptimizerCurves.vue'
import ChromeTraceDownloads from '../ChromeTraceDownloads.vue'

interface Props {
  result: Record<string, any>
  records?: any[]
  jobId?: string
}

const props = defineProps<Props>()
const { t } = useLocale()

const mode = computed(() => props.result.mode || 'aggregation')

const isDisagg = computed(() => mode.value === 'disagg_prefill' || mode.value === 'disagg_decode')

const modeView = computed(() => {
  if (mode.value === 'pd_ratio') return PDRatioView
  if (isDisagg.value) return DisaggregatedView
  return AggregatedView
})

const modeLabel = computed(() => {
  switch (mode.value) {
    case 'pd_ratio': return t({ zh: 'PD 配比优化', en: 'PD Ratio Optimization' })
    case 'disagg_prefill':
    case 'disagg_decode':
      return t({ zh: '分离部署', en: 'Disaggregated' })
    default: return t({ zh: '聚合', en: 'Aggregated' })
  }
})

// Chrome trace cases for download component
const traceCases = computed(() => {
  if (!props.jobId || !props.result.chrome_trace?.available) return []
  // Single case: seq=0
  return [{
    seq: 0,
    config: props.result.input_config || {},
    chrome_trace: props.result.chrome_trace
  }]
})

// In disaggregated mode, extract best configs from each phase table (top row
// per phase). Backend mode == "disagg_prefill" but both phases have data.
const prefillBest = computed(() => {
  const rows = props.result.disagg_prefill || []
  return rows.length ? rows[0] : null
})
const decodeBest = computed(() => {
  const rows = props.result.disagg_decode || []
  return rows.length ? rows[0] : null
})

// Best config summary metrics (mode-specific labels)
const bestMetrics = computed(() => {
  const bc = props.result.best_config || {}
  if (mode.value === 'pd_ratio') {
    return [
      { label: 'PD Ratio', value: bc.pd_ratio != null ? Number(bc.pd_ratio).toFixed(2) : '-' },
      { label: 'Balanced QPS', value: bc.balanced_qps != null ? Number(bc.balanced_qps).toFixed(2) : '-' },
      { label: 'P QPS', value: bc.p_qps != null ? Number(bc.p_qps).toFixed(2) : '-' },
      { label: 'D QPS', value: bc.d_qps != null ? Number(bc.d_qps).toFixed(2) : '-' },
    ]
  }
  if (isDisagg.value) {
    // Use prefillBest for the single-card fallback (see template — disagg
    // mode renders two per-phase cards below so this single card isn't used).
    return [
      { label: t({ zh: '吞吐', en: 'Throughput' }), value: bc.throughput_token_s != null ? `${Number(bc.throughput_token_s).toFixed(2)} token/s` : '-' },
      { label: 'TTFT', value: bc.ttft_ms != null ? `${Number(bc.ttft_ms).toFixed(2)} ms` : '-' },
      { label: 'TPOT', value: bc.tpot_ms != null ? `${Number(bc.tpot_ms).toFixed(2)} ms` : '-' },
    ]
  }
  return [
    { label: t({ zh: '吞吐', en: 'Throughput' }), value: bc.throughput_token_s != null ? `${Number(bc.throughput_token_s).toFixed(2)} token/s` : '-' },
    { label: 'TTFT', value: bc.ttft_ms != null ? `${Number(bc.ttft_ms).toFixed(2)} ms` : '-' },
    { label: 'TPOT', value: bc.tpot_ms != null ? `${Number(bc.tpot_ms).toFixed(2)} ms` : '-' },
  ]
})

const bestParallel = computed(() => {
  const bc = props.result.best_config || {}
  if (mode.value === 'pd_ratio') {
    return `P: ${bc.parallel_p || '-'} | D: ${bc.parallel_d || '-'}`
  }
  return bc.parallel || '-'
})
</script>

<template>
  <div class="throughput-result">
    <!-- Mode badge -->
    <div class="mode-header">
      <span class="mode-badge" :class="mode">{{ modeLabel }}</span>
    </div>

    <!-- Disaggregated: TWO best-config cards — one per phase -->
    <template v-if="isDisagg">
      <div class="disagg-best-row">
        <el-card v-if="prefillBest" class="best-config-card phase-card">
          <div class="phase-card-label">{{ t({ zh: 'Prefill 最优', en: 'Best Prefill' }) }}</div>
          <div class="best-metrics">
            <div class="metric">
              <span class="metric-label">{{ t({ zh: '吞吐', en: 'Throughput' }) }}</span>
              <span class="metric-value">{{ prefillBest.throughput_token_s != null ? Number(prefillBest.throughput_token_s).toFixed(2) + ' token/s' : '-' }}</span>
            </div>
            <div class="metric">
              <span class="metric-label">TTFT</span>
              <span class="metric-value">{{ prefillBest.ttft_ms != null ? Number(prefillBest.ttft_ms).toFixed(2) + ' ms' : '-' }}</span>
            </div>
            <div class="metric">
              <span class="metric-label">QPS</span>
              <span class="metric-value">{{ prefillBest.qps != null ? Number(prefillBest.qps).toFixed(2) : '-' }}</span>
            </div>
          </div>
          <div class="best-parallel">{{ prefillBest.parallel || '-' }}</div>
        </el-card>
        <el-card v-if="decodeBest" class="best-config-card phase-card">
          <div class="phase-card-label">{{ t({ zh: 'Decode 最优', en: 'Best Decode' }) }}</div>
          <div class="best-metrics">
            <div class="metric">
              <span class="metric-label">{{ t({ zh: '吞吐', en: 'Throughput' }) }}</span>
              <span class="metric-value">{{ decodeBest.throughput_token_s != null ? Number(decodeBest.throughput_token_s).toFixed(2) + ' token/s' : '-' }}</span>
            </div>
            <div class="metric">
              <span class="metric-label">TPOT</span>
              <span class="metric-value">{{ decodeBest.tpot_ms != null ? Number(decodeBest.tpot_ms).toFixed(2) + ' ms' : '-' }}</span>
            </div>
            <div class="metric">
              <span class="metric-label">QPS</span>
              <span class="metric-value">{{ decodeBest.qps != null ? Number(decodeBest.qps).toFixed(2) : '-' }}</span>
            </div>
          </div>
          <div class="best-parallel">{{ decodeBest.parallel || '-' }}</div>
        </el-card>
      </div>
    </template>

    <!-- Single best config card (aggregated / pd_ratio modes) -->
    <el-card v-else-if="result.best_config" class="best-config-card">
      <div class="best-metrics">
        <div v-for="m in bestMetrics" :key="m.label" class="metric">
          <span class="metric-label">{{ m.label }}</span>
          <span class="metric-value">{{ m.value }}</span>
        </div>
      </div>
      <div class="best-parallel">{{ bestParallel }}</div>
    </el-card>

    <!-- Exploration scatter curves (raw data, all explored configs) -->
    <OptimizerCurves v-if="records?.length" :records="records" :mode="mode" />

    <!-- Chrome trace downloads -->
    <ChromeTraceDownloads v-if="traceCases.length" :job-id="jobId" :cases="traceCases" />

    <!-- Mode-specific view (modular) -->
    <component :is="modeView" :result="result" />

    <!-- Empty state -->
    <el-empty
      v-if="!result.best_config && !result.sweep_rows?.length && !result.pd_ratio_rows?.length && !result.disagg_prefill?.length"
      :description="t({ zh: '暂无结果数据', en: 'No result data available' })"
    />
  </div>
</template>

<style scoped>
.thput-result, .throughput-result {
  padding: 12px 16px;
}
.mode-header {
  margin-bottom: 12px;
}
.mode-badge {
  display: inline-block;
  padding: 2px 12px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 600;
  background: var(--msm-bg-panel-2);
  color: var(--msm-text-muted);
  border: 1px solid var(--msm-border);
}
.mode-badge.pd_ratio { color: var(--msm-amber); border-color: color-mix(in srgb, var(--msm-amber) 30%, transparent); }
.mode-badge.disagg_prefill, .mode-badge.disagg_decode { color: var(--msm-accent); border-color: color-mix(in srgb, var(--msm-accent) 30%, transparent); }
.best-config-card {
  margin-bottom: 16px;
}
.best-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
}
.metric {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.metric-label {
  font-size: 12px;
  color: var(--msm-text-muted);
}
.metric-value {
  font-size: 18px;
  font-weight: 600;
  color: var(--msm-text);
}
.best-parallel {
  margin-top: 8px;
  font-size: 13px;
  color: var(--msm-text-muted);
  font-family: 'Fira Code', monospace;
}
.disagg-best-row {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
}
.disagg-best-row .phase-card {
  flex: 1;
}
.phase-card-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--msm-accent);
  margin-bottom: 8px;
}
</style>

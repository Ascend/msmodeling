<script setup lang="ts">
/**
 * Text generation result component.
 *
 * Renders the assembled `result` envelope for text_generate jobs:
 * - TPS per model (bar chart)
 * - Memory breakdown (nested structure)
 * - Execution time breakdowns (pie/stacked bar)
 *
 * Per the result rendering contract.
 */
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useLocale } from '@/composables/useLocale'
import { useChartTheme } from '@/composables/useChartTheme'
import ChartWrapper from '../ChartWrapper.vue'
import ChromeTraceDownloads from '../ChromeTraceDownloads.vue'
import OperatorTimingTable from '../OperatorTimingTable.vue'

interface Props {
  result: Record<string, any>
  jobId: string
  /** When embedded inside a MultiCase drill-down, hide the trace downloads
   *  (the parent's own table already lists every case). */
  hideTraceDownloads?: boolean
}

const props = defineProps<Props>()
const { t } = useLocale()
const router = useRouter()
const { baseOption, axisText, splitLine } = useChartTheme()

// Computed chart options
const tpsChartOption = computed(() => {
  const tpsData = props.result.tps_per_model || {}
  const models = Object.keys(tpsData)
  const values = Object.values(tpsData)

  return {
    ...baseOption.value,
    title: {
      text: t({ zh: 'TPS/设备', en: 'TPS per Device' }),
      left: 'center'
    },
    tooltip: {
      ...baseOption.value.tooltip,
      trigger: 'axis',
      axisPointer: { type: 'shadow' }
    },
    xAxis: {
      type: 'category',
      data: models,
      axisLine: { lineStyle: { color: splitLine.value } },
      axisLabel: { interval: 0, rotate: 30, color: axisText.value }
    },
    yAxis: {
      type: 'value',
      name: t({ zh: 'Token/秒', en: 'Token/s' }),
      splitLine: { lineStyle: { color: splitLine.value } },
      axisLabel: { color: axisText.value },
      nameTextStyle: { color: axisText.value }
    },
    series: [{
      type: 'bar',
      data: values
    }]
  }
})

const memoryChartOption = computed(() => {
  const memory = props.result.memory_gb || {}
  const data = [
    { name: t({ zh: '模型权重', en: 'Model Weight' }), value: memory.model_weight || 0 },
    { name: t({ zh: 'KV缓存', en: 'KV Cache' }), value: memory.kv_cache || 0 },
    { name: t({ zh: '模型激活', en: 'Model Activation' }), value: memory.model_activation || 0 },
    { name: t({ zh: '保留内存', en: 'Reserved' }), value: memory.reserved || 0 },
    { name: t({ zh: '可用内存', en: 'Available' }), value: memory.available || 0 }
  ].filter(d => d.value > 0)

  return {
    ...baseOption.value,
    title: {
      text: t({ zh: '内存分布 (GB)', en: 'Memory Distribution (GB)' }),
      left: 'center'
    },
    tooltip: {
      ...baseOption.value.tooltip,
      trigger: 'item',
      formatter: (p: any) => `${p.name}: ${p.value.toFixed(2)} GB (${p.percent}%)`
    },
    legend: {
      orient: 'vertical',
      left: 'left',
      textStyle: { color: axisText.value }
    },
    series: [{
      type: 'pie',
      radius: '50%',
      data: data,
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }]
  }
})

// OpBound category -> readable label (the classifier splits time into
// memory / communication / compute-mma / compute-gp bound buckets).
const _OPBOUND_LABELS: Record<string, { zh: string; en: string }> = {
  memory_bound: { zh: '访存', en: 'Memory' },
  communication_bound: { zh: '通信', en: 'Communication' },
  compute_bound_mma: { zh: '计算 (MMA)', en: 'Compute (MMA)' },
  compute_bound_gp: { zh: '计算 (GP)', en: 'Compute (GP)' },
}

// OpBound breakdown as a single compact sentence: "Memory 45% · Communication 30% · Compute(MMA) 20% · Compute(GP) 5%"
const opBoundSummary = computed(() => {
  const breakdowns = props.result.breakdowns_percent || {}
  const parts: { label: string; pct: number }[] = []
  for (const cats of Object.values(breakdowns)) {
    if (!cats || typeof cats !== 'object') continue
    for (const [cat, pct] of Object.entries(cats)) {
      const v = typeof pct === 'number' ? pct : 0
      parts.push({ label: t(_OPBOUND_LABELS[cat] || { zh: cat, en: cat }), pct: v })
    }
  }
  if (parts.length === 0) return ''
  return parts.map((p) => `${p.label} ${p.pct}%`).join(' · ')
})

// Whether any result data is present — mirrors the el-empty guard so a
// no-data result shows a clean empty state instead of all-dash metric cards.
const hasResult = computed(() =>
  !!props.result.tps_per_model
  || !!props.result.memory_gb
  || !!props.result.breakdowns_percent
  || (props.result.op_breakdown || []).length > 0
)

// Summary metrics
const summaryMetrics = computed(() => [
  {
    label: t({ zh: '批次大小', en: 'Batch Size' }),
    value: props.result.batch_size ?? '-'
  },
  {
    label: t({ zh: '执行时间', en: 'Execution Time' }),
    value: props.result.execution_time_s?.analytic ? `${props.result.execution_time_s.analytic.toFixed(3)} s` : '-'
  },
  {
    label: t({ zh: '峰值内存', en: 'Peak Memory' }),
    value: props.result.memory_gb?.peak_usage ? `${props.result.memory_gb.peak_usage.toFixed(2)} GB` : '-'
  },
  {
    label: t({ zh: '设备总内存', en: 'Total Device Memory' }),
    value: props.result.memory_gb?.total_device ? `${props.result.memory_gb.total_device.toFixed(2)} GB` : '-'
  }
])

// Chrome trace: wrap single case in array format for ChromeTraceDownloads
const traceCases = computed(() => {
  if (props.hideTraceDownloads || !props.result.chrome_trace?.available) return []
  return [{
    seq: 0,
    config: props.result.input_config || {},
    chrome_trace: props.result.chrome_trace
  }]
})
</script>

<template>
  <div class="text-generate-result">
    <!-- Summary Cards -->
    <el-row :gutter="16" class="summary-section" v-if="hasResult">
      <el-col
        v-for="metric in summaryMetrics"
        :key="metric.label"
        :xs="24"
        :sm="12"
        :md="8"
        :lg="6"
      >
        <el-card class="metric-card">
          <div class="metric-label">{{ metric.label }}</div>
          <div class="metric-value">{{ metric.value }}</div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Simulator run time (wall-clock, incl. compile; distinct from the model execution time above) -->
    <div v-if="result.run_time_s != null" class="runtime-note">
      <span class="rn-label">{{ t({ zh: '仿真程序耗时', en: 'Simulator run time' }) }}</span>
      <span class="rn-value">{{ result.run_time_s.toFixed(3) }} s</span>
      <span class="rn-hint">{{ t({ zh: '（含编译，非模型执行时间）', en: '(incl. compile; not model execution time)' }) }}</span>
    </div>

    <!-- Charts -->
    <el-row :gutter="16" class="charts-section">
      <el-col :xs="24" :lg="12" v-if="result.tps_per_model">
        <el-card>
          <ChartWrapper :option="tpsChartOption" height="350px" />
        </el-card>
      </el-col>
      <el-col :xs="24" :lg="12" v-if="result.memory_gb">
        <el-card>
          <ChartWrapper :option="memoryChartOption" height="350px" />
        </el-card>
      </el-col>
    </el-row>

    <!-- OpBound summary (compact text) -->
    <div v-if="opBoundSummary" class="opbound-summary">
      <span class="ob-label">{{ t({ zh: '算子瓶颈分布', en: 'OpBound' }) }}</span>
      <span class="ob-text">{{ opBoundSummary }}</span>
    </div>

    <!-- No Result State -->
    <el-empty
      v-if="!hasResult"
      :description="t({ zh: '暂无结果数据', en: 'No result data available' })"
    >
      <el-button type="primary" @click="router.push({ name: 'jobStatus', params: { jobId } })">
        {{ t({ zh: '查看任务状态', en: 'View Job Status' }) }}
      </el-button>
    </el-empty>

    <!-- Operator timing table (shared component) -->
    <OperatorTimingTable
      :op-breakdown="props.result.op_breakdown || []"
      :dump-input-shapes="props.result.dump_input_shapes"
      :dump-op-bound-results="props.result.dump_op_bound_results"
    />

    <!-- Chrome Trace Downloads -->
    <ChromeTraceDownloads :job-id="jobId" :cases="traceCases" />
  </div>
</template>

<style scoped>
.text-generate-result {
  padding: 16px;
}

.summary-section {
  margin-bottom: 16px;
}

.runtime-note {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 16px;
  padding: 8px 14px;
  background: var(--el-fill-color-light, #f5f7fa);
  border-left: 3px solid var(--el-color-info, #909399);
  border-radius: var(--msm-radius-sm);
  font-size: 13px;
}
.runtime-note .rn-label {
  color: var(--el-text-color-secondary);
}
.runtime-note .rn-value {
  font-weight: 600;
  color: var(--el-text-color-primary);
}
.runtime-note .rn-hint {
  color: var(--el-text-color-placeholder, #a8abb2);
  font-size: 12px;
}

.metric-card {
  text-align: center;
  padding: 16px;
}

.metric-label {
  font-size: 14px;
  color: var(--el-text-color-secondary);
  margin-bottom: 8px;
}

.metric-value {
  font-size: 20px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.charts-section {
  margin-bottom: 16px;
}

.charts-section:last-child {
  margin-bottom: 0;
}

.opbound-summary {
  display: flex;
  align-items: baseline;
  gap: 10px;
  padding: 8px 14px;
  background: var(--el-fill-color-light, #f5f7fa);
  border-left: 3px solid var(--el-color-primary, #409EFF);
  border-radius: var(--msm-radius-sm);
  font-size: 13px;
  margin-top: 16px;
}
.opbound-summary .ob-label {
  color: var(--el-text-color-secondary);
  white-space: nowrap;
  font-weight: 500;
}
.opbound-summary .ob-text {
  color: var(--el-text-color-primary);
}
</style>

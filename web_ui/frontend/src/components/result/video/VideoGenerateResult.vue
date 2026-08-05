<script setup lang="ts">
/**
 * Video generation result component.
 *
 * Renders the assembled `result` envelope for video_generate jobs:
 * - Execution time by perf model (bar chart)
 * - Breakdowns by model (pie charts)
 * - Operator average table (virtualized table)
 *
 * Per the result rendering contract.
 */
import { computed } from 'vue'
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
const { baseOption } = useChartTheme()

// Execution time chart (by perf model)
const executionTimeChartOption = computed(() => {
  const execTime = props.result.execution_time_s || {}
  const models = Object.keys(execTime)
  const values = Object.values(execTime).map(v => typeof v === 'number' ? v : 0)

  return {
    ...baseOption.value,
    title: {
      text: t({ zh: '执行时间 (秒)', en: 'Execution Time (s)' }),
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params: any) => {
        const param = params[0]
        return `${param.name}: ${param.value.toFixed(3)} s`
      }
    },
    xAxis: {
      type: 'category',
      data: models,
      axisLabel: { interval: 0 }
    },
    yAxis: {
      type: 'value',
      name: t({ zh: '秒', en: 'Seconds' })
    },
    series: [{
      type: 'bar',
      data: values
    }]
  }
})

// Breakdowns summary (compact text line instead of pie charts)
const breakdownSummary = computed(() => {
  const breakdowns: Record<string, Record<string, number>> = props.result.breakdowns || {}
  const parts: string[] = []
  for (const [modelName, modelBreakdown] of Object.entries(breakdowns)) {
    const opParts: string[] = []
    for (const [op, time] of Object.entries(modelBreakdown)) {
      const v = typeof time === 'number' ? time : 0
      opParts.push(`${op}: ${v.toFixed(1)} ms`)
    }
    parts.push(`${modelName}: ${opParts.join(' · ')}`)
  }
  return parts.join(' | ')
})

// Operator breakdown table is rendered by the shared OperatorTimingTable component.

// Whether any result data is present — mirrors the el-empty guard so a
// no-data result shows a clean empty state instead of all-dash metric cards.
const hasResult = computed(() =>
  !!props.result.execution_time_s
  || !!breakdownSummary.value
  || (props.result.op_breakdown || []).length > 0
)

// Summary metrics
const summaryMetrics = computed(() => {
  const execTime = props.result.execution_time_s || {}
  const totalTime = Object.values(execTime).reduce((sum: number, v) => sum + (typeof v === 'number' ? v : 0), 0)

  return [
    {
      label: t({ zh: '总执行时间', en: 'Total Execution Time' }),
      value: totalTime > 0 ? `${totalTime.toFixed(3)} s` : '-'
    },
    {
      label: t({ zh: '性能模型', en: 'Performance Models' }),
      value: Object.keys(execTime).join(', ') || '-'
    },
    {
      label: t({ zh: '模型细分', en: 'Model Breakdowns' }),
      value: Object.keys(props.result.breakdowns || {}).length || '-'
    },
    {
      label: t({ zh: '算子种类', en: 'Operator Types' }),
      value: (props.result.op_breakdown || []).length || '-'
    }
  ]
})

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
  <div class="video-generate-result">
    <!-- Summary Cards -->
    <el-row :gutter="16" class="summary-section" v-if="hasResult">
      <el-col
        v-for="metric in summaryMetrics"
        :key="metric.label"
        :xs="24"
        :sm="12"
        :md="6"
      >
        <el-card class="metric-card">
          <div class="metric-label">{{ metric.label }}</div>
          <div class="metric-value">{{ metric.value }}</div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Execution Time Chart -->
    <el-row :gutter="16" class="charts-section" v-if="result.execution_time_s">
      <el-col :xs="24">
        <el-card>
          <ChartWrapper :option="executionTimeChartOption" height="350px" />
        </el-card>
      </el-col>
    </el-row>

    <!-- Breakdowns Summary (compact text) -->
    <div v-if="breakdownSummary" class="breakdown-summary">
      <span class="ob-label">{{ t({ zh: '算子耗时分布', en: 'Operator Time Distribution' }) }}</span>
      <span class="ob-text">{{ breakdownSummary }}</span>
    </div>

    <!-- Operator timing table (shared component) -->
    <OperatorTimingTable :op-breakdown="props.result.op_breakdown || []" />

    <!-- No Result State -->
    <el-empty
      v-if="!result.execution_time_s && !breakdownSummary && !(props.result.op_breakdown || []).length"
      :description="t({ zh: '暂无结果数据', en: 'No result data available' })"
    />

    <!-- Chrome Trace Downloads -->
    <ChromeTraceDownloads :job-id="jobId" :cases="traceCases" />
  </div>
</template>

<style scoped>
.video-generate-result {
  padding: 16px;
}

.summary-section {
  margin-bottom: 16px;
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

.charts-section,
.table-section {
  margin-bottom: 16px;
}

.charts-section:last-child,
.table-section:last-child {
  margin-bottom: 0;
}

.breakdown-summary {
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
.breakdown-summary .ob-label {
  color: var(--el-text-color-secondary);
  white-space: nowrap;
  font-weight: 500;
}
.breakdown-summary .ob-text {
  color: var(--el-text-color-primary);
}
</style>

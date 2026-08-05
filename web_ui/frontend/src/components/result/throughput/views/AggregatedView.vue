<script setup lang="ts">
/**
 * Aggregated mode view: sweep table (Top | Throughput | TTFT | TPOT |
 * concurrency | num_devices | parallel | batch_size) + cross-hardware chart.
 */
import { computed } from 'vue'
import { useLocale } from '@/composables/useLocale'
import { useChartTheme } from '@/composables/useChartTheme'
import { exportCsv, type CsvColumn } from '@/composables/useCsvExport'
import ChartWrapper from '../../ChartWrapper.vue'

interface Props { result: Record<string, any> }
const props = defineProps<Props>()
const { t } = useLocale()
const theme = useChartTheme()

const sweepRows = computed(() => props.result.sweep_rows || [])
const crossHardware = computed(() => props.result.cross_hardware || [])

// CSV export columns (labels match the headers, already i18n'd).
const csvColumns = computed<CsvColumn[]>(() => [
  { key: 'rank', label: 'Top' },
  { key: 'throughput_token_s', label: t({ zh: '吞吐 (token/s)', en: 'Throughput (token/s)' }) },
  { key: 'qps', label: 'QPS (req/s)' },
  { key: 'ttft_ms', label: 'TTFT (ms)' },
  { key: 'tpot_ms', label: 'TPOT (ms)' },
  { key: 'concurrency', label: 'concurrency' },
  { key: 'num_devices', label: 'num_devices' },
  { key: 'parallel', label: 'parallel' },
  { key: 'batch_size', label: 'batch_size' },
  { key: 'model_weight_size_gb', label: 'weight_GB' },
  { key: 'kv_cache_size_gb', label: 'kv_cache_GB' },
  { key: 'model_activation_size_gb', label: 'activation_GB' },
  { key: 'device_memory_available_gb', label: 'avail_GB' },
])

function onExportCsv() {
  if (!sweepRows.value.length) return
  exportCsv('throughput-aggregated', sweepRows.value, csvColumns.value)
}

const crossChartOption = computed(() => {
  if (!crossHardware.value.length) return null
  return {
    ...theme.baseOption.value,
    title: { text: t({ zh: '跨设备最佳吞吐', en: 'Best Throughput per Device' }), left: 'center' },
    tooltip: { ...theme.baseOption.value.tooltip, trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: {
      type: 'category',
      data: crossHardware.value.map((r: any) => r.device),
      axisLine: { lineStyle: { color: theme.splitLine.value } },
      axisLabel: { rotate: 20, color: theme.axisText.value },
    },
    yAxis: {
      type: 'value',
      name: 'Token/s',
      splitLine: { lineStyle: { color: theme.splitLine.value } },
      axisLabel: { color: theme.axisText.value },
      nameTextStyle: { color: theme.axisText.value },
    },
    series: [{
      type: 'bar',
      data: crossHardware.value.map((r: any) => r.throughput_token_s),
      itemStyle: { color: theme.category.value[1] },
    }],
  }
})
</script>

<template>
  <div class="agg-view">
    <div class="view-toolbar">
      <el-button
        link
        size="small"
        :disabled="!sweepRows.length"
        @click="onExportCsv"
      >
        {{ t({ zh: '导出 CSV', en: 'Export CSV' }) }}
      </el-button>
    </div>
    <el-table
      :data="sweepRows"
      size="small"
      border
      stripe
      max-height="500"
      :default-sort="{ prop: 'rank', order: 'ascending' }"
    >
      <el-table-column :label="t({ zh: 'Top', en: 'Top' })" prop="rank" width="60" sortable />
      <el-table-column :label="t({ zh: '吞吐 (token/s)', en: 'Throughput (token/s)' })" prop="throughput_token_s" width="170" sortable>
        <template #default="{ row }">{{ row.throughput_token_s != null ? Number(row.throughput_token_s).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="TTFT (ms)" prop="ttft_ms" width="110" sortable>
        <template #default="{ row }">{{ row.ttft_ms != null ? Number(row.ttft_ms).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="TPOT (ms)" prop="tpot_ms" width="110" sortable>
        <template #default="{ row }">{{ row.tpot_ms != null ? Number(row.tpot_ms).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="concurrency" prop="concurrency" width="110" sortable />
      <el-table-column label="num_devices" prop="num_devices" width="100" sortable />
      <el-table-column label="parallel" prop="parallel" min-width="160" show-overflow-tooltip />
      <el-table-column label="batch_size" prop="batch_size" width="100" sortable />
      <el-table-column label="weight_GB" prop="model_weight_size_gb" width="110" sortable>
        <template #default="{ row }">{{ row.model_weight_size_gb != null ? Number(row.model_weight_size_gb).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="kv_cache_GB" prop="kv_cache_size_gb" width="120" sortable>
        <template #default="{ row }">{{ row.kv_cache_size_gb != null ? Number(row.kv_cache_size_gb).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="activation_GB" prop="model_activation_size_gb" width="140" sortable>
        <template #default="{ row }">{{ row.model_activation_size_gb != null ? Number(row.model_activation_size_gb).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="avail_GB" prop="device_memory_available_gb" width="110" sortable>
        <template #default="{ row }">{{ row.device_memory_available_gb != null ? Number(row.device_memory_available_gb).toFixed(2) : '-' }}</template>
      </el-table-column>
    </el-table>
    <el-card v-if="crossChartOption" class="chart-card">
      <ChartWrapper :option="crossChartOption" height="300px" />
    </el-card>
  </div>
</template>

<style scoped>
.agg-view { padding: 8px 0; }
.view-toolbar { display: flex; justify-content: flex-end; margin-bottom: 8px; }
.chart-card { margin-top: 12px; }
</style>

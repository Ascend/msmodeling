<script setup lang="ts">
/**
 * PD Ratio mode view: PD table (Top | PD Ratio | Balanced QPS | P QPS | D QPS |
 * TTFT | TPOT | P Parallel | D Parallel | P/D Devices | P/D Batch | P/D Conc).
 */
import { computed } from 'vue'
import { useLocale } from '@/composables/useLocale'
import { exportCsv, type CsvColumn } from '@/composables/useCsvExport'

interface Props { result: Record<string, any> }
const props = defineProps<Props>()
const { t } = useLocale()

const pdRows = computed(() => props.result.pd_ratio_rows || [])

const csvColumns = computed<CsvColumn[]>(() => [
  { key: 'rank', label: 'Top' },
  { key: 'pd_ratio', label: 'PD Ratio' },
  { key: 'balanced_qps', label: 'Balanced QPS' },
  { key: 'p_qps', label: 'P QPS' },
  { key: 'd_qps', label: 'D QPS' },
  { key: 'ttft_ms', label: 'TTFT (ms)' },
  { key: 'tpot_ms', label: 'TPOT (ms)' },
  { key: 'parallel_p', label: 'P Parallel' },
  { key: 'parallel_d', label: 'D Parallel' },
  { key: 'p_devices_per_instance', label: 'P Devices/Instance' },
  { key: 'd_devices_per_instance', label: 'D Devices/Instance' },
  { key: 'p_batch_size', label: 'P Batch Size' },
  { key: 'd_batch_size', label: 'D Batch Size' },
  { key: 'p_concurrency', label: 'P Concurrency' },
  { key: 'd_concurrency', label: 'D Concurrency' },
])

function onExportCsv() {
  if (!pdRows.value.length) return
  exportCsv('throughput-pd-ratio', pdRows.value, csvColumns.value)
}
</script>

<template>
  <div class="pd-view">
    <div class="view-toolbar">
      <el-button link size="small" :disabled="!pdRows.length" @click="onExportCsv">
        {{ t({ zh: '导出 CSV', en: 'Export CSV' }) }}
      </el-button>
    </div>
    <el-table :data="pdRows" size="small" border stripe max-height="500" :default-sort="{ prop: 'rank', order: 'ascending' }">
      <el-table-column :label="t({ zh: 'Top', en: 'Top' })" prop="rank" width="60" sortable />
      <el-table-column label="PD Ratio" prop="pd_ratio" width="100" sortable>
        <template #default="{ row }">{{ row.pd_ratio != null ? Number(row.pd_ratio).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="Balanced QPS" prop="balanced_qps" width="130" sortable>
        <template #default="{ row }">{{ row.balanced_qps != null ? Number(row.balanced_qps).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="P QPS" prop="p_qps" width="100" sortable>
        <template #default="{ row }">{{ row.p_qps != null ? Number(row.p_qps).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="D QPS" prop="d_qps" width="100" sortable>
        <template #default="{ row }">{{ row.d_qps != null ? Number(row.d_qps).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="TTFT (ms)" prop="ttft_ms" width="100" sortable>
        <template #default="{ row }">{{ row.ttft_ms != null ? Number(row.ttft_ms).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="TPOT (ms)" prop="tpot_ms" width="100" sortable>
        <template #default="{ row }">{{ row.tpot_ms != null ? Number(row.tpot_ms).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="P Parallel" prop="parallel_p" min-width="160" show-overflow-tooltip />
      <el-table-column label="D Parallel" prop="parallel_d" min-width="160" show-overflow-tooltip />
      <el-table-column label="P Devices/Instance" prop="p_devices_per_instance" width="160" sortable />
      <el-table-column label="D Devices/Instance" prop="d_devices_per_instance" width="160" sortable />
      <el-table-column label="P Batch Size" prop="p_batch_size" width="130" sortable />
      <el-table-column label="D Batch Size" prop="d_batch_size" width="130" sortable />
      <el-table-column label="P Concurrency" prop="p_concurrency" width="140" sortable />
      <el-table-column label="D Concurrency" prop="d_concurrency" width="140" sortable />
    </el-table>
  </div>
</template>

<style scoped>
.pd-view { padding: 8px 0; }
.view-toolbar { display: flex; justify-content: flex-end; margin-bottom: 8px; }
</style>

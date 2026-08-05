<script setup lang="ts">
/**
 * Disaggregated mode view: TWO tables — Prefill (Throughput | QPS | TTFT |
 * concurrency | num_devices | parallel | batch_size) and Decode (Throughput |
 * QPS | TPOT | concurrency | …). Each phase independently ranked.
 */
import { computed } from 'vue'
import { useLocale } from '@/composables/useLocale'
import { exportCsv, type CsvColumn } from '@/composables/useCsvExport'

interface Props { result: Record<string, any> }
const props = defineProps<Props>()
const { t } = useLocale()

const prefillRows = computed(() => props.result.disagg_prefill || [])
const decodeRows = computed(() => props.result.disagg_decode || [])

// Separate CSV column sets per phase. The page shows Prefill = TTFT only and
// Decode = TPOT only, so a single shared spec that includes BOTH ttft_ms and
// tpot_ms would add a stray TPOT column to the Prefill export (and a stray TTFT
// column to Decode). Each phase keeps only its own latency column.
const prefillCsvColumns = computed<CsvColumn[]>(() => [
  { key: 'rank', label: 'Top' },
  { key: 'throughput_token_s', label: t({ zh: '吞吐 (token/s)', en: 'Throughput (token/s)' }) },
  { key: 'qps', label: 'QPS (req/s)' },
  { key: 'ttft_ms', label: 'TTFT (ms)' },
  { key: 'concurrency', label: 'concurrency' },
  { key: 'num_devices', label: 'num_devices' },
  { key: 'parallel', label: 'parallel' },
  { key: 'batch_size', label: 'batch_size' },
  { key: 'model_weight_size_gb', label: 'weight_GB' },
  { key: 'kv_cache_size_gb', label: 'kv_cache_GB' },
  { key: 'model_activation_size_gb', label: 'activation_GB' },
  { key: 'device_memory_available_gb', label: 'avail_GB' },
])
const decodeCsvColumns = computed<CsvColumn[]>(() => [
  { key: 'rank', label: 'Top' },
  { key: 'throughput_token_s', label: t({ zh: '吞吐 (token/s)', en: 'Throughput (token/s)' }) },
  { key: 'qps', label: 'QPS (req/s)' },
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

function onExportPrefill() {
  if (!prefillRows.value.length) return
  exportCsv('throughput-prefill', prefillRows.value, prefillCsvColumns.value)
}
function onExportDecode() {
  if (!decodeRows.value.length) return
  exportCsv('throughput-decode', decodeRows.value, decodeCsvColumns.value)
}
</script>

<template>
  <div class="disagg-view">
    <!-- Prefill table -->
    <div class="phase-section">
      <div class="phase-header">
        <h4 class="phase-title">{{ t({ zh: 'Prefill 阶段（TTFT 约束）', en: 'Prefill Phase (TTFT-constrained)' }) }}</h4>
        <el-button link size="small" :disabled="!prefillRows.length" @click="onExportPrefill">
          {{ t({ zh: '导出 CSV', en: 'Export CSV' }) }}
        </el-button>
      </div>
      <el-table :data="prefillRows" size="small" border stripe max-height="400" :default-sort="{ prop: 'rank', order: 'ascending' }">
        <el-table-column :label="t({ zh: 'Top', en: 'Top' })" prop="rank" width="60" sortable />
        <el-table-column :label="t({ zh: '吞吐 (token/s)', en: 'Throughput (token/s)' })" prop="throughput_token_s" width="170" sortable>
          <template #default="{ row }">{{ row.throughput_token_s != null ? Number(row.throughput_token_s).toFixed(2) : '-' }}</template>
        </el-table-column>
        <el-table-column label="QPS (req/s)" prop="qps" width="120" sortable>
          <template #default="{ row }">{{ row.qps != null ? Number(row.qps).toFixed(2) : '-' }}</template>
        </el-table-column>
        <el-table-column label="TTFT (ms)" prop="ttft_ms" width="110" sortable>
          <template #default="{ row }">{{ row.ttft_ms != null ? Number(row.ttft_ms).toFixed(2) : '-' }}</template>
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
    </div>
    <!-- Decode table -->
    <div class="phase-section">
      <div class="phase-header">
        <h4 class="phase-title">{{ t({ zh: 'Decode 阶段（TPOT 约束）', en: 'Decode Phase (TPOT-constrained)' }) }}</h4>
        <el-button link size="small" :disabled="!decodeRows.length" @click="onExportDecode">
          {{ t({ zh: '导出 CSV', en: 'Export CSV' }) }}
        </el-button>
      </div>
      <el-table :data="decodeRows" size="small" border stripe max-height="400" :default-sort="{ prop: 'rank', order: 'ascending' }">
        <el-table-column :label="t({ zh: 'Top', en: 'Top' })" prop="rank" width="60" sortable />
        <el-table-column :label="t({ zh: '吞吐 (token/s)', en: 'Throughput (token/s)' })" prop="throughput_token_s" width="170" sortable>
          <template #default="{ row }">{{ row.throughput_token_s != null ? Number(row.throughput_token_s).toFixed(2) : '-' }}</template>
        </el-table-column>
        <el-table-column label="QPS (req/s)" prop="qps" width="120" sortable>
          <template #default="{ row }">{{ row.qps != null ? Number(row.qps).toFixed(2) : '-' }}</template>
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
    </div>
  </div>
</template>

<style scoped>
.disagg-view { padding: 8px 0; }
.phase-section { margin-bottom: 20px; }
.phase-header { display: flex; align-items: center; justify-content: space-between; margin: 0 0 8px; }
.phase-title { font-size: 14px; font-weight: 600; margin: 0; color: var(--msm-text-muted); }
</style>

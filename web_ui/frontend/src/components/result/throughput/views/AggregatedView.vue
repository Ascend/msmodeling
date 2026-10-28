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
 * Aggregated mode view: sweep table (Top | Throughput | TTFT | TPOT |
 * concurrency | num-devices | parallel | batch-size) + cross-hardware chart.
 */
import { computed } from 'vue'
import { ElTooltip, ElIcon } from 'element-plus'
import { InfoFilled } from '@element-plus/icons-vue'
import { useLocale } from '@/composables/useLocale'
import { useChartTheme } from '@/composables/useChartTheme'
import { exportCsv, type CsvColumn } from '@/composables/useCsvExport'
import ChartWrapper from '../../ChartWrapper.vue'

interface Props { result: Record<string, any> }
const props = defineProps<Props>()
const { t } = useLocale()
const theme = useChartTheme()

const sweepRows = computed(() => props.result["sweep_rows"] || [])
const crossHardware = computed(() => props.result["cross_hardware"] || [])

// CSV export columns (labels match the headers, already i18n'd).
const csvColumns = computed<CsvColumn[]>(() => [
  { key: 'rank', label: 'Top' },
  { key: 'throughput_token_s', label: t({ zh: '输出吞吐 (token/s)', en: 'Output Throughput (token/s)' }) },
  { key: 'per_card_output_throughput_token_s', label: t({ zh: '单卡输出吞吐 (token/s/卡)', en: 'Per-card Output (token/s/card)' }) },
  { key: 'input_throughput_token_s', label: t({ zh: '输入吞吐 (token/s)', en: 'Input Throughput (token/s)' }) },
  { key: 'e2e_throughput_token_s', label: t({ zh: 'e2e吞吐 (token/s)', en: 'E2E Throughput (token/s)' }) },
  { key: 'per_card_input_throughput_token_s', label: t({ zh: '单卡输入吞吐 (token/s/卡)', en: 'Per-card Input (token/s/card)' }) },
  { key: 'ttft_ms', label: 'TTFT (ms)' },
  { key: 'tpot_ms', label: 'TPOT (ms)' },
  { key: 'qps', label: 'QPS (req/s)' },
  { key: 'concurrency', label: 'concurrency' },
  { key: 'num_devices', label: 'num-devices' },
  { key: 'parallel', label: 'parallel' },
  { key: 'batch_size', label: 'batch-size' },
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
      data: crossHardware.value.map((r: any) => r["throughput_token_s"]),
      itemStyle: { color: theme.category.value[1] },
    }],
  }
})
</script>

<template>
  <div class="agg-view">
    <div class="view-toolbar">
      <el-tooltip placement="top" :show-after="200">
        <template #content>
          <div class="metric-notes">
            <p>{{ t({ zh: '输出吞吐：生成阶段吞吐（引擎计算）', en: 'Output Throughput: generation phase (engine-computed)' }) }}</p>
            <p>{{ t({ zh: '单卡输出吞吐 = 输出吞吐 / num_devices', en: 'Per-card Output = Output / num_devices' }) }}</p>
            <p>{{ t({ zh: '输入吞吐 = QPS × input_length（稳态输入流）', en: 'Input Throughput = QPS × input_length (steady-state input flow)' }) }}</p>
            <p>{{ t({ zh: 'e2e吞吐 = 输出吞吐 + 输入吞吐（输入+输出的总 token 流）', en: 'E2E Throughput = Output + Input (total token flow)' }) }}</p>
            <p>{{ t({ zh: '单卡输入吞吐 = 输入吞吐 / num_devices', en: 'Per-card Input = Input / num_devices' }) }}</p>
            <p>{{ t({ zh: 'QPS = 输出吞吐 / output_length（端到端完成速率，与吞吐同口径）', en: 'QPS = Output Throughput / output_length (end-to-end completion rate, same caliber as throughput)' }) }}</p>
          </div>
        </template>
        <el-icon class="metric-info-icon"><InfoFilled /></el-icon>
      </el-tooltip>
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
      <el-table-column :label="t({ zh: '输出吞吐 (token/s)', en: 'Output Throughput (token/s)' })" prop="throughput_token_s" width="180" sortable>
        <template #default="{ row }">{{ row["throughput_token_s"] != null ? Number(row["throughput_token_s"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column :label="t({ zh: '单卡输出吞吐 (token/s/卡)', en: 'Per-card Output (token/s/card)' })" prop="per_card_output_throughput_token_s" width="200" sortable>
        <template #default="{ row }">{{ row["per_card_output_throughput_token_s"] != null ? Number(row["per_card_output_throughput_token_s"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column :label="t({ zh: '输入吞吐 (token/s)', en: 'Input Throughput (token/s)' })" prop="input_throughput_token_s" width="170" sortable>
        <template #default="{ row }">{{ row["input_throughput_token_s"] != null ? Number(row["input_throughput_token_s"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column :label="t({ zh: 'e2e吞吐 (token/s)', en: 'E2E Throughput (token/s)' })" prop="e2e_throughput_token_s" width="160" sortable>
        <template #default="{ row }">{{ row["e2e_throughput_token_s"] != null ? Number(row["e2e_throughput_token_s"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column :label="t({ zh: '单卡输入吞吐 (token/s/卡)', en: 'Per-card Input (token/s/card)' })" prop="per_card_input_throughput_token_s" width="200" sortable>
        <template #default="{ row }">{{ row["per_card_input_throughput_token_s"] != null ? Number(row["per_card_input_throughput_token_s"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="TTFT (ms)" prop="ttft_ms" width="110" sortable>
        <template #default="{ row }">{{ row["ttft_ms"] != null ? Number(row["ttft_ms"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="TPOT (ms)" prop="tpot_ms" width="110" sortable>
        <template #default="{ row }">{{ row["tpot_ms"] != null ? Number(row["tpot_ms"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="QPS (req/s)" prop="qps" width="110" sortable>
        <template #default="{ row }">{{ row["qps"] != null ? Number(row["qps"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="concurrency" prop="concurrency" width="110" sortable />
      <el-table-column label="num_devices" prop="num_devices" width="100" sortable />
      <el-table-column label="parallel" prop="parallel" min-width="160" show-overflow-tooltip />
      <el-table-column label="batch_size" prop="batch_size" width="100" sortable />
      <el-table-column label="weight_GB" prop="model_weight_size_gb" width="110" sortable>
        <template #default="{ row }">{{ row["model_weight_size_gb"] != null ? Number(row["model_weight_size_gb"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="kv_cache_GB" prop="kv_cache_size_gb" width="120" sortable>
        <template #default="{ row }">{{ row["kv_cache_size_gb"] != null ? Number(row["kv_cache_size_gb"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="activation_GB" prop="model_activation_size_gb" width="140" sortable>
        <template #default="{ row }">{{ row["model_activation_size_gb"] != null ? Number(row["model_activation_size_gb"]).toFixed(2) : '-' }}</template>
      </el-table-column>
      <el-table-column label="avail_GB" prop="device_memory_available_gb" width="110" sortable>
        <template #default="{ row }">{{ row["device_memory_available_gb"] != null ? Number(row["device_memory_available_gb"]).toFixed(2) : '-' }}</template>
      </el-table-column>
    </el-table>
    <el-card v-if="crossChartOption" class="chart-card">
      <ChartWrapper :option="crossChartOption" height="300px" />
    </el-card>
  </div>
</template>

<style scoped>
.agg-view { padding: 8px 0; }
.view-toolbar { display: flex; justify-content: flex-end; align-items: center; gap: 4px; margin-bottom: 8px; }
.metric-info-icon { color: var(--msm-text-muted); cursor: help; font-size: 16px; }
.metric-notes p { margin: 2px 0; }
.chart-card { margin-top: 12px; }
</style>

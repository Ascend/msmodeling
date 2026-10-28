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
 * Disaggregated mode view: PD-pair estimate table + TWO phase tables —
 * Prefill (Input Throughput | QPS | TTFT | …) and Decode (Output Throughput |
 * QPS | TPOT | …). Each phase independently ranked.
 *
 * The PD-pair table renders the backend-assembled `pd_pairs` (system-level
 * UPPER-BOUND metrics for every prefill×decode combination; system QPS = min
 * of both sides'). Top 1 is the recommended pair.
 */
import { computed } from 'vue'
import { ElTooltip, ElIcon } from 'element-plus'
import { InfoFilled } from '@element-plus/icons-vue'
import { useLocale } from '@/composables/useLocale'
import { exportCsv, type CsvColumn } from '@/composables/useCsvExport'

interface Props { result: Record<string, any> }
const props = defineProps<Props>()
const { t } = useLocale()

const prefillRows = computed(() => props.result["disagg_prefill"] || [])
const decodeRows = computed(() => props.result["disagg_decode"] || [])

// ── PD-pair system estimate (upper bound, backend-assembled) ────────────────
const pdPairs = computed(() => props.result["pd_pairs"] || [])

const pairNotes = computed(() => [
  t({ zh: '系统指标为上界估计：两侧均为全部设备独立测量', en: 'System metrics are an UPPER BOUND: both sides measured with all devices' }),
  t({ zh: '系统 QPS = min(P QPS, D QPS)，由瓶颈侧决定；系统 QPS 相同的配对，TTFT 更低者（首 token 更快）排名靠前，仍相同再比 TPOT', en: 'System QPS = min(P QPS, D QPS), set by the bottleneck; among pairs with equal system QPS, lower TTFT (faster first token) ranks first, then lower TPOT' }),
  t({ zh: '系统输出/输入吞吐 = 系统 QPS × output-length / input-length，即稳态下系统实际产出/消耗的 token 流（瓶颈决定），非 P/D 单侧能力', en: 'System output/input throughput = System QPS × output/input-length — the token flow the pair actually sustains (bottleneck-limited), NOT either side\'s standalone capability' }),
  t({ zh: 'e2e吞吐 = 系统输出吞吐 + 系统输入吞吐（输入+输出的总 token 流）；P/D 单侧能力见下方各阶段表', en: 'E2E throughput = System Output + System Input throughput (total input+output token flow); each side\'s standalone capability is in the phase tables below' }),
  t({ zh: '实际 P/D 分割设备后系统 QPS ≤ 此上界；精确配比请使用 PD 配比优化模式', en: 'Actual system QPS with split devices ≤ this bound; use PD Ratio Optimization for exact pairing' }),
])

const pairCsvColumns = computed<CsvColumn[]>(() => [
  { key: 'rank', label: 'Top' },
  { key: 'parallel_p', label: 'P parallel' },
  { key: 'parallel_d', label: 'D parallel' },
  { key: 'system_qps', label: t({ zh: '系统 QPS (req/s)', en: 'System QPS (req/s)' }) },
  { key: 'system_output_throughput_token_s', label: t({ zh: '系统输出吞吐 (token/s)', en: 'System Output Throughput (token/s)' }) },
  { key: 'system_input_throughput_token_s', label: t({ zh: '系统输入吞吐 (token/s)', en: 'System Input Throughput (token/s)' }) },
  { key: 'e2e_throughput_token_s', label: t({ zh: 'e2e吞吐 (token/s)', en: 'E2E Throughput (token/s)' }) },
  { key: 'p_qps', label: 'P QPS' },
  { key: 'd_qps', label: 'D QPS' },
  { key: 'bottleneck', label: t({ zh: '瓶颈', en: 'Bottleneck' }) },
  { key: 'ttft_ms', label: 'TTFT (ms)' },
  { key: 'tpot_ms', label: 'TPOT (ms)' },
])

function onExportPairs() {
  if (!pdPairs.value.length) return
  exportCsv('throughput-pd-pairs', pdPairs.value, pairCsvColumns.value)
}

/** Top-1 row highlight (the recommended pair). */
function pairRowClass({ row }: { row: { rank: number } }): string {
  return row.rank === 1 ? 'pair-top1' : ''
}

function fmt(v: number | null | undefined, digits = 2): string {
  return v != null ? Number(v).toFixed(digits) : '-'
}

// Separate CSV column sets per phase. The page shows Prefill = TTFT only and
// Decode = TPOT only, so a single shared spec that includes BOTH ttft-ms and
// tpot-ms would add a stray TPOT column to the Prefill export (and a stray TTFT
// column to Decode). Each phase keeps only its own latency column.
// Prefill phase only processes input (prompt) tokens — its throughput IS the
// input processing rate (engine token/s = concurrency × input_length / drain
// makespan, the saturated caliber), so the table shows the input-throughput
// fields only; no separate output metric exists for this phase.
const prefillCsvColumns = computed<CsvColumn[]>(() => [
  { key: 'rank', label: 'Top' },
  { key: 'input_throughput_token_s', label: t({ zh: '输入吞吐 (token/s)', en: 'Input Throughput (token/s)' }) },
  { key: 'per_card_input_throughput_token_s', label: t({ zh: '单卡输入吞吐 (token/s/卡)', en: 'Per-card Input (token/s/card)' }) },
  { key: 'qps', label: 'QPS (req/s)' },
  { key: 'ttft_ms', label: 'TTFT (ms)' },
  { key: 'concurrency', label: 'concurrency' },
  { key: 'num_devices', label: 'num-devices' },
  { key: 'parallel', label: 'parallel' },
  { key: 'batch_size', label: 'batch-size' },
  { key: 'model_weight_size_gb', label: 'weight_GB' },
  { key: 'kv_cache_size_gb', label: 'kv_cache_GB' },
  { key: 'model_activation_size_gb', label: 'activation_GB' },
  { key: 'device_memory_available_gb', label: 'avail_GB' },
])
const decodeCsvColumns = computed<CsvColumn[]>(() => [
  { key: 'rank', label: 'Top' },
  { key: 'throughput_token_s', label: t({ zh: '输出吞吐 (token/s)', en: 'Output Throughput (token/s)' }) },
  { key: 'per_card_output_throughput_token_s', label: t({ zh: '单卡输出吞吐 (token/s/卡)', en: 'Per-card Output (token/s/card)' }) },
  { key: 'qps', label: 'QPS (req/s)' },
  { key: 'tpot_ms', label: 'TPOT (ms)' },
  { key: 'concurrency', label: 'concurrency' },
  { key: 'num_devices', label: 'num-devices' },
  { key: 'parallel', label: 'parallel' },
  { key: 'batch_size', label: 'batch-size' },
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

// Phase-specific metric semantics (shown as a tooltip next to each phase title).
const prefillNotes = computed(() => [
  t({ zh: 'Prefill 只处理输入 (prompt) token，其吞吐即输入处理速率', en: 'Prefill only processes input (prompt) tokens — its throughput IS the input rate' }),
  t({ zh: '输入吞吐 = 引擎计算的 token/s（concurrency × input_length / 排空时间，饱和口径）', en: 'Input Throughput = engine-computed token/s (concurrency × input_length / drain makespan, saturated caliber)' }),
  t({ zh: '单卡输入吞吐 = 引擎计算的 token/s/device', en: 'Per-card Input = engine-computed token/s/device' }),
  t({ zh: 'QPS = 输入吞吐 / input_length（与吞吐同口径，恒等自洽）', en: 'QPS = Input Throughput / input_length (same caliber as throughput — consistent by construction)' }),
])
const decodeNotes = computed(() => [
  t({ zh: 'Decode 生成输出 token，其吞吐即输出（生成）速率', en: 'Decode generates output tokens — its throughput IS the output (generation) rate' }),
  t({ zh: '输出吞吐：引擎计算的 token/s', en: 'Output Throughput: engine-computed token/s' }),
  t({ zh: '单卡输出吞吐 = 输出吞吐 / num_devices', en: 'Per-card Output = Output / num_devices' }),
  t({ zh: 'QPS = 输出吞吐 / output_length（与吞吐同口径，恒等自洽）', en: 'QPS = Output Throughput / output_length (same caliber as throughput — consistent by construction)' }),
])
</script>

<template>
  <div class="disagg-view">
    <!-- PD-pair system estimate (upper bound, backend-assembled) -->
    <div v-if="pdPairs.length" class="phase-section">
      <div class="phase-header">
        <h4 class="phase-title">
          {{ t({ zh: 'PD 配对估计（上界）', en: 'PD Pair Estimate (upper bound)' }) }}
          <el-tooltip placement="top" :show-after="200">
            <template #content>
              <div class="metric-notes">
                <p v-for="n in pairNotes" :key="n">{{ n }}</p>
              </div>
            </template>
            <el-icon class="metric-info-icon"><InfoFilled /></el-icon>
          </el-tooltip>
        </h4>
        <el-button link size="small" :disabled="!pdPairs.length" @click="onExportPairs">
          {{ t({ zh: '导出 CSV', en: 'Export CSV' }) }}
        </el-button>
      </div>
      <el-table
        :data="pdPairs"
        size="small"
        border
        stripe
        max-height="400"
        :row-class-name="pairRowClass"
        :default-sort="{ prop: 'rank', order: 'ascending' }"
      >
        <el-table-column :label="t({ zh: 'Top', en: 'Top' })" prop="rank" width="60" sortable />
        <el-table-column label="P parallel" prop="parallel_p" min-width="150" show-overflow-tooltip />
        <el-table-column label="D parallel" prop="parallel_d" min-width="150" show-overflow-tooltip />
        <el-table-column :label="t({ zh: '系统 QPS (req/s)', en: 'System QPS (req/s)' })" prop="system_qps" width="140" sortable>
          <template #default="{ row }"><span class="sys-qps">{{ fmt(row.system_qps) }}</span></template>
        </el-table-column>
        <el-table-column :label="t({ zh: '系统输出吞吐 (token/s)', en: 'System Output (token/s)' })" prop="system_output_throughput_token_s" width="180" sortable>
          <template #default="{ row }">{{ fmt(row.system_output_throughput_token_s) }}</template>
        </el-table-column>
        <el-table-column :label="t({ zh: '系统输入吞吐 (token/s)', en: 'System Input (token/s)' })" prop="system_input_throughput_token_s" width="180" sortable>
          <template #default="{ row }">{{ fmt(row.system_input_throughput_token_s) }}</template>
        </el-table-column>
        <el-table-column :label="t({ zh: 'e2e吞吐 (token/s)', en: 'E2E Throughput (token/s)' })" prop="e2e_throughput_token_s" width="160" sortable>
          <template #default="{ row }">{{ fmt(row.e2e_throughput_token_s) }}</template>
        </el-table-column>
        <el-table-column label="P QPS" prop="p_qps" width="100" sortable>
          <template #default="{ row }">{{ fmt(row.p_qps) }}</template>
        </el-table-column>
        <el-table-column label="D QPS" prop="d_qps" width="100" sortable>
          <template #default="{ row }">{{ fmt(row.d_qps) }}</template>
        </el-table-column>
        <el-table-column :label="t({ zh: '瓶颈', en: 'Bottleneck' })" prop="bottleneck" width="90" sortable>
          <template #default="{ row }">
            <span class="tag" :class="row.bottleneck === 'balanced' ? 'tag-balanced' : 'tag-bottleneck'">
              {{ row.bottleneck === 'p' ? 'P' : row.bottleneck === 'd' ? 'D' : t({ zh: '均衡', en: 'Balanced' }) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="TTFT (ms)" prop="ttft_ms" width="110" sortable>
          <template #default="{ row }">{{ fmt(row.ttft_ms) }}</template>
        </el-table-column>
        <el-table-column label="TPOT (ms)" prop="tpot_ms" width="110" sortable>
          <template #default="{ row }">{{ fmt(row.tpot_ms) }}</template>
        </el-table-column>
      </el-table>
      <div class="pair-caption">
        {{ t({ zh: 'Top 1 即推荐配对。系统指标为上界估计——两侧均为满配设备独立测量；精确 P/D 设备配比请使用 PD 配比优化模式。', en: 'Top 1 is the recommended pair. System metrics are upper bounds — both sides measured with all devices; use PD Ratio Optimization for exact device pairing.' }) }}
      </div>
    </div>
    <!-- Prefill table -->
    <div class="phase-section">
      <div class="phase-header">
        <h4 class="phase-title">
          {{ t({ zh: 'Prefill 阶段（TTFT 约束）', en: 'Prefill Phase (TTFT-constrained)' }) }}
          <el-tooltip placement="top" :show-after="200">
            <template #content>
              <div class="metric-notes">
                <p v-for="n in prefillNotes" :key="n">{{ n }}</p>
              </div>
            </template>
            <el-icon class="metric-info-icon"><InfoFilled /></el-icon>
          </el-tooltip>
        </h4>
        <el-button link size="small" :disabled="!prefillRows.length" @click="onExportPrefill">
          {{ t({ zh: '导出 CSV', en: 'Export CSV' }) }}
        </el-button>
      </div>
      <el-table :data="prefillRows" size="small" border stripe max-height="400" :default-sort="{ prop: 'rank', order: 'ascending' }">
        <el-table-column :label="t({ zh: 'Top', en: 'Top' })" prop="rank" width="60" sortable />
        <el-table-column :label="t({ zh: '输入吞吐 (token/s)', en: 'Input Throughput (token/s)' })" prop="input_throughput_token_s" width="170" sortable>
          <template #default="{ row }">{{ row["input_throughput_token_s"] != null ? Number(row["input_throughput_token_s"]).toFixed(2) : '-' }}</template>
        </el-table-column>
        <el-table-column :label="t({ zh: '单卡输入吞吐 (token/s/卡)', en: 'Per-card Input (token/s/card)' })" prop="per_card_input_throughput_token_s" width="200" sortable>
          <template #default="{ row }">{{ row["per_card_input_throughput_token_s"] != null ? Number(row["per_card_input_throughput_token_s"]).toFixed(2) : '-' }}</template>
        </el-table-column>
        <el-table-column label="QPS (req/s)" prop="qps" width="120" sortable>
          <template #default="{ row }">{{ row.qps != null ? Number(row.qps).toFixed(2) : '-' }}</template>
        </el-table-column>
        <el-table-column label="TTFT (ms)" prop="ttft_ms" width="110" sortable>
          <template #default="{ row }">{{ row["ttft_ms"] != null ? Number(row["ttft_ms"]).toFixed(2) : '-' }}</template>
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
    </div>
    <!-- Decode table -->
    <div class="phase-section">
      <div class="phase-header">
        <h4 class="phase-title">
          {{ t({ zh: 'Decode 阶段（TPOT 约束）', en: 'Decode Phase (TPOT-constrained)' }) }}
          <el-tooltip placement="top" :show-after="200">
            <template #content>
              <div class="metric-notes">
                <p v-for="n in decodeNotes" :key="n">{{ n }}</p>
              </div>
            </template>
            <el-icon class="metric-info-icon"><InfoFilled /></el-icon>
          </el-tooltip>
        </h4>
        <el-button link size="small" :disabled="!decodeRows.length" @click="onExportDecode">
          {{ t({ zh: '导出 CSV', en: 'Export CSV' }) }}
        </el-button>
      </div>
      <el-table :data="decodeRows" size="small" border stripe max-height="400" :default-sort="{ prop: 'rank', order: 'ascending' }">
        <el-table-column :label="t({ zh: 'Top', en: 'Top' })" prop="rank" width="60" sortable />
        <el-table-column :label="t({ zh: '输出吞吐 (token/s)', en: 'Output Throughput (token/s)' })" prop="throughput_token_s" width="180" sortable>
          <template #default="{ row }">{{ row["throughput_token_s"] != null ? Number(row["throughput_token_s"]).toFixed(2) : '-' }}</template>
        </el-table-column>
        <el-table-column :label="t({ zh: '单卡输出吞吐 (token/s/卡)', en: 'Per-card Output (token/s/card)' })" prop="per_card_output_throughput_token_s" width="200" sortable>
          <template #default="{ row }">{{ row["per_card_output_throughput_token_s"] != null ? Number(row["per_card_output_throughput_token_s"]).toFixed(2) : '-' }}</template>
        </el-table-column>
        <el-table-column label="QPS (req/s)" prop="qps" width="120" sortable>
          <template #default="{ row }">{{ row.qps != null ? Number(row.qps).toFixed(2) : '-' }}</template>
        </el-table-column>
        <el-table-column label="TPOT (ms)" prop="tpot_ms" width="110" sortable>
          <template #default="{ row }">{{ row["tpot_ms"] != null ? Number(row["tpot_ms"]).toFixed(2) : '-' }}</template>
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
    </div>
  </div>
</template>

<style scoped>
.disagg-view { padding: 8px 0; }
.phase-section { margin-bottom: 20px; }
.phase-header { display: flex; align-items: center; justify-content: space-between; margin: 0 0 8px; }
.phase-title { font-size: 14px; font-weight: 600; margin: 0; color: var(--msm-text-muted); display: inline-flex; align-items: center; gap: 4px; }
.metric-info-icon { color: var(--msm-text-muted); cursor: help; font-size: 15px; }
.metric-notes p { margin: 2px 0; }
.pair-caption { margin-top: 8px; font-size: 11.5px; color: var(--msm-text-muted); line-height: 1.6; }
/* Top-1 (recommended pair) row tint; deep selector needed — el-table rows
   render outside the scoped style's data attribute. */
:deep(.el-table .pair-top1) td.el-table__cell { background: color-mix(in srgb, var(--msm-green) 9%, transparent) !important; }
.sys-qps { color: var(--msm-green); font-weight: 700; }
.tag { display: inline-block; padding: 1px 9px; border-radius: 3px; font-size: 11.5px; font-weight: 600; }
.tag-bottleneck { background: color-mix(in srgb, var(--msm-amber) 16%, transparent); color: var(--msm-amber); }
.tag-balanced { background: color-mix(in srgb, var(--msm-green) 16%, transparent); color: var(--msm-green); }
</style>

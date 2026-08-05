<script setup lang="ts">
/**
 * OperatorTimingTable — shared operator-breakdown panel.
 *
 * Renders the `op_breakdown` array (aggregated by runner._aggregate_runtime_events)
 * with a view toggle:
 *   - Detail (table): operator name (monospace), total ms, average ms, call count
 *   - Top Share (chart): donut of the top-half operators (by count) by total
 *     time; the rest collapse into "Other".
 * Used by TextGenerateResult and VideoGenerateResult.
 *
 * op_breakdown items (new format):
 *   { name, bound?, input_shapes?, total_s, avg_s, calls, bound_pct? }
 * op_breakdown items (legacy format):
 *   { name, perf_model, perf_total (s), perf_avg (s), call_times }
 */
import { computed, ref } from 'vue'
import { useLocale } from '@/composables/useLocale'
import { ElTable, ElTableColumn, ElRadioGroup, ElRadioButton, ElButton, ElTag } from 'element-plus'
import { exportCsv, type CsvColumn } from '@/composables/useCsvExport'
import OperatorTimingChart from './OperatorTimingChart.vue'

type ViewMode = 'table' | 'chart'

const props = withDefaults(
  defineProps<{
    /** Raw op_breakdown array from the result envelope. */
    opBreakdown?: Array<Record<string, any>>
    /** Optional title override. */
    title?: string
    /** Whether dump_input_shapes was enabled for this run. */
    dumpInputShapes?: boolean
    /** Whether dump_op_bound_results was enabled for this run. */
    dumpOpBoundResults?: boolean
  }>(),
  {
    opBreakdown: () => [],
    title: '',
    dumpInputShapes: false,
    dumpOpBoundResults: false,
  },
)

const { t } = useLocale()
const viewMode = ref<ViewMode>('table')

// Raw op_breakdown → table rows (normalize field names, convert s → ms).
const rows = computed(() => {
  const raw = props.opBreakdown
  if (!Array.isArray(raw) || raw.length === 0) return []
  return raw.map((r: any) => {
    // New format uses total_s/avg_s/calls; legacy uses perf_total/perf_avg/call_times
    const isNewFormat = 'total_s' in r || 'calls' in r
    const totalS = isNewFormat ? (Number(r.total_s) || 0) : (Number(r.perf_total) || 0)
    const avgS = isNewFormat ? (Number(r.avg_s) || 0) : (Number(r.perf_avg) || 0)
    const count = isNewFormat ? (Number(r.calls) || 0) : (Number(r.call_times ?? r.call_count) || 0)

    const row: Record<string, any> = {
      op: String(r.name || r.op_name || ''),
      totalMs: totalS * 1000,
      avgMs: avgS * 1000,
      count,
    }

    if (props.dumpOpBoundResults && r.bound) {
      row.bound = r.bound
    }
    if (props.dumpInputShapes && r.input_shapes) {
      row.inputShapes = r.input_shapes
    }
    if (props.dumpOpBoundResults && r.bound_pct) {
      row.boundPct = r.bound_pct
      // Flatten bound_pct into top-level numeric fields matching the sortable
      // column props (memPct/commPct/mmaPct/gpPct). Without these the table
      // sorts on undefined and the % columns can't be ordered. Cell rendering
      // still reads row.boundPct below; these flat numbers exist for sorting.
      const bp = r.bound_pct
      row.memPct = typeof bp.memory === 'number' ? bp.memory : undefined
      row.commPct = typeof bp.comm === 'number' ? bp.comm : undefined
      row.mmaPct = typeof bp.mma === 'number' ? bp.mma : undefined
      row.gpPct = typeof bp.gp === 'number' ? bp.gp : undefined
    }

    return row
  })
})

const tableTitle = computed(
  () => props.title || t({ zh: '算子耗时明细', en: 'Operator Timing' }),
)

// Bound type display labels
const boundLabels: Record<string, string> = {
  memory_bound: 'Memory',
  communication_bound: 'Comm',
  compute_bound_mma: 'MMA',
  compute_bound_gp: 'GP',
}

// Bound tag color mapping
function boundTagType(bound: string): '' | 'success' | 'warning' | 'danger' | 'info' {
  switch (bound) {
    case 'memory_bound': return 'danger'
    case 'communication_bound': return 'warning'
    case 'compute_bound_mma': return 'success'
    case 'compute_bound_gp': return 'info'
    default: return 'info'
  }
}

const columns = computed(() => {
  const cols: Array<{ prop: string; label: string; width?: number; minWidth?: number; sortable: boolean }> = [
    { prop: 'op', label: t({ zh: '算子', en: 'Operator' }), minWidth: 200, sortable: false },
  ]

  if (props.dumpOpBoundResults) {
    cols.push({ prop: 'bound', label: t({ zh: '瓶颈', en: 'Bound' }), width: 100, sortable: true })
  }
  if (props.dumpInputShapes) {
    cols.push({ prop: 'inputShapes', label: t({ zh: '输入形状', en: 'Input Shapes' }), minWidth: 180, sortable: false })
  }

  cols.push(
    { prop: 'totalMs', label: t({ zh: '总耗时 (ms)', en: 'Total (ms)' }), width: 140, sortable: true },
    { prop: 'avgMs', label: t({ zh: '平均 (ms)', en: 'Avg (ms)' }), width: 120, sortable: true },
  )

  if (props.dumpOpBoundResults) {
    cols.push(
      { prop: 'memPct', label: t({ zh: 'Memory %', en: 'Memory %' }), width: 100, sortable: true },
      { prop: 'commPct', label: t({ zh: 'Comm %', en: 'Comm %' }), width: 90, sortable: true },
      { prop: 'mmaPct', label: t({ zh: 'MMA %', en: 'MMA %' }), width: 90, sortable: true },
      { prop: 'gpPct', label: t({ zh: 'GP %', en: 'GP %' }), width: 80, sortable: true },
    )
  }

  cols.push({ prop: 'count', label: t({ zh: '调用次数', en: 'Calls' }), width: 100, sortable: true })

  return cols
})

// CSV export column definitions (labels match the table headers, already i18n'd).
const csvColumns = computed<CsvColumn[]>(() => {
  const cols: CsvColumn[] = [
    { key: 'op', label: t({ zh: '算子', en: 'Operator' }) },
  ]

  if (props.dumpOpBoundResults) {
    cols.push({ key: 'bound', label: t({ zh: '瓶颈', en: 'Bound' }) })
  }
  if (props.dumpInputShapes) {
    cols.push({ key: 'inputShapes', label: t({ zh: '输入形状', en: 'Input Shapes' }) })
  }

  cols.push(
    { key: 'totalMs', label: t({ zh: '总耗时 (ms)', en: 'Total (ms)' }) },
    { key: 'avgMs', label: t({ zh: '平均 (ms)', en: 'Avg (ms)' }) },
  )

  if (props.dumpOpBoundResults) {
    cols.push(
      { key: 'memPct', label: t({ zh: 'Memory %', en: 'Memory %' }) },
      { key: 'commPct', label: t({ zh: 'Comm %', en: 'Comm %' }) },
      { key: 'mmaPct', label: t({ zh: 'MMA %', en: 'MMA %' }) },
      { key: 'gpPct', label: t({ zh: 'GP %', en: 'GP %' }) },
    )
  }

  cols.push({ key: 'count', label: t({ zh: '调用次数', en: 'Calls' }) })

  return cols
})

// Flatten bound_pct into row for CSV export
const exportRows = computed(() => {
  return rows.value.map(r => ({
    ...r,
    bound: r.bound ? (boundLabels[r.bound] || r.bound) : '',
    memPct: r.boundPct?.memory?.toFixed(2) ?? '',
    commPct: r.boundPct?.comm?.toFixed(2) ?? '',
    mmaPct: r.boundPct?.mma?.toFixed(2) ?? '',
    gpPct: r.boundPct?.gp?.toFixed(2) ?? '',
  }))
})

function onExportCsv() {
  if (!exportRows.value.length) return
  exportCsv('operator-timing', exportRows.value, csvColumns.value)
}
</script>

<template>
  <div v-if="rows.length > 0" class="op-timing-panel">
    <div class="op-panel-header">
      <div class="op-table-title">{{ tableTitle }}</div>
      <div class="op-panel-actions">
        <el-button
          v-if="viewMode === 'table'"
          link
          size="small"
          :disabled="!rows.length"
          @click="onExportCsv"
        >
          {{ t({ zh: '导出 CSV', en: 'Export CSV' }) }}
        </el-button>
        <el-radio-group v-model="viewMode" size="small">
          <el-radio-button value="table">{{ t({ zh: '明细', en: 'Detail' }) }}</el-radio-button>
          <el-radio-button value="chart">{{ t({ zh: 'Top占比', en: 'Top Share' }) }}</el-radio-button>
        </el-radio-group>
      </div>
    </div>

    <Transition name="msm-fade" mode="out-in">
      <!-- Detail table -->
      <el-table v-if="viewMode === 'table'" :data="rows" size="small" stripe max-height="420" border :default-sort="{ prop: 'totalMs', order: 'descending' }">
        <el-table-column
          v-for="col in columns"
          :key="col.prop"
          :prop="col.prop"
          :label="col.label"
          :width="col.width"
          :min-width="col.minWidth"
          :sortable="col.sortable"
        >
          <template #default="{ row }">
            <template v-if="col.prop === 'op'">
              <code class="op-name">{{ row.op }}</code>
            </template>
            <template v-else-if="col.prop === 'bound'">
              <el-tag v-if="row.bound" :type="boundTagType(row.bound)" size="small">
                {{ boundLabels[row.bound] || row.bound }}
              </el-tag>
            </template>
            <template v-else-if="col.prop === 'inputShapes'">
              <code class="input-shapes">{{ row.inputShapes }}</code>
            </template>
            <template v-else-if="col.prop === 'totalMs'">
              {{ row.totalMs.toFixed(3) }}
            </template>
            <template v-else-if="col.prop === 'avgMs'">
              {{ row.avgMs.toFixed(4) }}
            </template>
            <template v-else-if="col.prop === 'memPct'">
              {{ row.boundPct?.memory?.toFixed(2) ?? '-' }}
            </template>
            <template v-else-if="col.prop === 'commPct'">
              {{ row.boundPct?.comm?.toFixed(2) ?? '-' }}
            </template>
            <template v-else-if="col.prop === 'mmaPct'">
              {{ row.boundPct?.mma?.toFixed(2) ?? '-' }}
            </template>
            <template v-else-if="col.prop === 'gpPct'">
              {{ row.boundPct?.gp?.toFixed(2) ?? '-' }}
            </template>
            <template v-else>
              {{ row[col.prop] }}
            </template>
          </template>
        </el-table-column>
      </el-table>

      <!-- Top Share chart -->
      <OperatorTimingChart v-else :op-breakdown="props.opBreakdown" />
    </Transition>
  </div>
</template>

<style scoped>
.op-timing-panel {
  margin-top: 16px;
}
.op-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  padding-left: 2px;
}
.op-panel-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}
.op-table-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--msm-text);
}
.op-name {
  font-family: 'Fira Code', monospace;
  font-size: 12px;
  color: var(--msm-text-muted);
  word-break: break-all;
}
.input-shapes {
  font-family: 'Fira Code', monospace;
  font-size: 11px;
  color: var(--msm-text-muted);
  word-break: break-all;
}
</style>


<script setup lang="ts">
/**
 * Chrome Trace Downloads component.
 *
 * Displays per-case Chrome trace download links for jobs that had
 * chrome_trace enabled. Shows a table with case configuration and
 * download buttons. Only renders when at least one case has a trace
 * file available.
 *
 * Downloads use the service-layer getJobTrace() + Blob + programmatic <a> click
 * rather than a bare <a href download>, because the latter is unreliable across
 * browsers (Edge/older Chromium can fail with "no file" even when the server
 * returns a valid Content-Disposition: attachment).
 */
import { computed, ref, onBeforeUnmount } from 'vue'
import { getJobTrace } from '@/services/api'
import { Download, Loading } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useLocale } from '@/composables/useLocale'

interface CaseWithTrace {
  seq: number
  config: Record<string, any>
  chrome_trace: {
    available: boolean
  }
  [key: string]: any
}

interface Props {
  jobId: string
  cases: CaseWithTrace[]
}

const props = defineProps<Props>()
const { t } = useLocale()

// Filter cases that have chrome trace available
const availableCases = computed(() => {
  return props.cases.filter(c => c.chrome_trace?.available === true)
})

// No cases with traces → don't render
const shouldShow = computed(() => availableCases.value.length > 0)

// Format case config for display
const formatConfig = (config: Record<string, any> | undefined | null): string => {
  if (!config) return t({ zh: '默认配置', en: 'Default' })

  const parts: string[] = []

  // Common fields to show
  if (config.device) parts.push(config.device)
  if (config.model_id) {
    // Shorten model_id for display
    const modelId = config.model_id
    const match = modelId.match(/([^/]+)$/)?.[1] || modelId
    parts.push(match)
  }
  if (config.num_queries && config.num_queries !== '1') {
    parts.push(`queries=${config.num_queries}`)
  }
  if (config.quantize_linear_action && config.quantize_linear_action !== 'W8A8_DYNAMIC') {
    parts.push(config.quantize_linear_action)
  }
  if (config.batch_size && config.batch_size !== 1) {
    parts.push(`batch=${config.batch_size}`)
  }

  return parts.join(' · ') || t({ zh: '默认配置', en: 'Default' })
}

// Track which case is currently downloading (by seq)
const downloading = ref<number | null>(null)

// setTimeout handle used to delay revoking the blob URL; must be cleaned up on
// unmount (see frontend guide §5). The pending Blob URL is tracked alongside so
// that a subsequent download (which clears the timer) ALSO revokes the prior URL
// — clearing only the timer orphans the previous Blob, leaking one per download.
let revokeTimer: ReturnType<typeof setTimeout> | null = null
let pendingUrl: string | null = null

function revokePending() {
  if (revokeTimer) {
    clearTimeout(revokeTimer)
    revokeTimer = null
  }
  if (pendingUrl) {
    URL.revokeObjectURL(pendingUrl)
    pendingUrl = null
  }
}

onBeforeUnmount(revokePending)

// Defensive: props.jobId should always be set by the parent, but if the prop
// chain breaks (stale HMR / cached bundle) it can arrive as undefined. Fall
// back to parsing the job id from the result-page URL (/jobs/:jobId/result).
const effectiveJobId = computed(() => {
  if (props.jobId) return props.jobId
  const m = window.location.pathname.match(/\/jobs\/([^/]+)\//)
  return m ? m[1] : ''
})

// Fetch the trace via the service-layer getJobTrace (reusing the unified axios instance +
// error handling), then trigger the download with a Blob + programmatic <a> click
// (more reliable cross-browser than <a href download>).
const handleDownload = async (seq: number) => {
  if (downloading.value !== null) return
  downloading.value = seq
  try {
    const blob = await getJobTrace(effectiveJobId.value, seq)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `chrome_trace_case_${seq}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    // Delay revoke so the click has a chance to consume the URL. Revoke the
    // PREVIOUS pending URL too — clearing only its timer would leave that Blob
    // allocated forever (one leak per repeated download).
    revokePending()
    pendingUrl = url
    revokeTimer = setTimeout(() => {
      URL.revokeObjectURL(url)
      pendingUrl = null
      revokeTimer = null
    }, 1000)
  } catch (e) {
    ElMessage.error(
      t({ zh: '下载失败', en: 'Download failed' }) + ': ' + (e as Error).message
    )
  } finally {
    downloading.value = null
  }
}
</script>

<template>
  <div v-if="shouldShow" class="chrome-trace-downloads">
    <el-card class="trace-card">
      <template #header>
        <div class="card-header">
          <el-icon class="header-icon">
            <Download />
          </el-icon>
          <span class="card-title">
            {{ t({ zh: 'Chrome Trace 下载', en: 'Chrome Trace Downloads' }) }}
          </span>
          <span class="card-subtitle">
            {{ t({ zh: '每个用例导出的 Chrome trace 文件', en: 'Per-case Chrome trace export' }) }}
          </span>
        </div>
      </template>

      <el-table :data="availableCases" stripe class="trace-table" :header-cell-style="{ background: 'var(--msm-bg-panel)', color: 'var(--msm-text)' }">
        <el-table-column
          :label="t({ zh: '用例配置', en: 'Case Config' })"
          prop="config"
          min-width="200"
        >
          <template #default="{ row }">
            <span class="config-text">{{ formatConfig(row.config) }}</span>
          </template>
        </el-table-column>

        <el-table-column
          :label="t({ zh: '下载', en: 'Download' })"
          width="140"
          align="center"
        >
          <template #default="{ row }">
            <el-link
              type="primary"
              :underline="false"
              class="download-link"
              :disabled="downloading !== null"
              @click="handleDownload(row.seq)"
            >
              <el-icon class="dl-icon" :class="{ 'is-loading': downloading === row.seq }">
                <Loading v-if="downloading === row.seq" />
                <Download v-else />
              </el-icon>
              {{
                downloading === row.seq
                  ? t({ zh: '下载中…', en: 'Downloading…' })
                  : t({ zh: '下载', en: 'Download' })
              }}
            </el-link>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.chrome-trace-downloads {
  margin-top: 20px;
}

.trace-card {
  border: 1px solid var(--msm-border);
  background: var(--msm-bg-panel);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 12px;
}

.header-icon {
  font-size: 20px;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  background: rgba(22, 163, 74, 0.1);
  color: var(--msm-green);
}

.card-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--msm-text);
}

.card-subtitle {
  font-size: 13px;
  color: var(--msm-text-muted);
  margin-left: auto;
}

.trace-table {
  font-size: 13px;
}

.config-text {
  color: var(--msm-text);
  font-family: 'Fira Code', monospace;
}

.download-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  cursor: pointer;
}

.download-link:hover {
  opacity: 0.8;
}

.dl-icon.is-loading {
  animation: dl-spin 1s linear infinite;
}

@keyframes dl-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
</style>

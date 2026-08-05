<script setup lang="ts">
/**
 * ModuleWorkspace — one module's split-pane: TOP = config form, BOTTOM = result.
 * A draggable divider resizes the two panes. Owns the form (SchemaForm) and
 * reads/writes a shared `runner` (useJobRunner) so the result survives tab switches.
 */
import { ref, computed, onBeforeUnmount, watch, nextTick } from 'vue'
import { ElButton, ElIcon, ElDrawer, ElSkeleton, ElInput, ElMessage, ElMessageBox } from 'element-plus'
import { Document, DocumentCopy, VideoCamera, TrendCharts, Loading } from '@element-plus/icons-vue'
import { useLocale } from '@/composables/useLocale'
import SchemaForm from '@/components/form/SchemaForm.vue'
import ResultPane from './ResultPane.vue'
import { getJobLog, getCaseLog } from '@/services/api'
import { trackEvent } from '@/services/telemetrySink'

const props = defineProps<{
  moduleId: string
  runner: any
  splitPercent: number
}>()

const emit = defineEmits<{
  'update:splitPercent': [v: number]
  submit: [data: { moduleId: string; params: Record<string, any>; formSchemaVersion: string }]
}>()

const { t } = useLocale()

const statusText = computed(() => {
  switch (props.runner?.status) {
    case 'pending':
      return t({ zh: '排队等待', en: 'Queued' })
    case 'running':
      return t({ zh: '运行中', en: 'Running' })
    default:
      return ''
  }
})

const schemaFormRef = ref<InstanceType<typeof SchemaForm> | null>(null)
const containerRef = ref<HTMLElement | null>(null)
const logOpen = ref(false)
const logContent = ref('') // job-level CLI output (full log — banner, dedup info, all cases interleaved)
const logLoading = ref(false)
const logCaseFilter = ref('all') // 'all' (job log) | case index
const logSearch = ref('') // case-insensitive line filter for the displayed log
const logPreRef = ref<HTMLPreElement | null>(null)
let logCopying = false
const caseLogCache = ref<Record<string, string>>({}) // case_hash → fetched per-case log
const caseLogLoading = ref(false)
// Per-case load-failure flag (keyed by case_hash). A failed fetch is NOT cached
// as '' in caseLogCache, so re-selecting the case re-requests; this flag lets the
// UI distinguish "actually no logs" from "load failed" and offer a retry.
const caseLogError = ref<Record<string, boolean>>({})

// Cases exposed by this job's result envelope (each carries a case_hash for
// per-case log lookup — replaces the old regex-split of the job log by
// `===== Case i/N =====` headers, which broke when separators hit case bodies).
const caseList = computed(() => {
  const r = props.runner.result
  if (!r) return []
  if (r.multi_case && Array.isArray(r.cases)) {
    return r.cases.map((c: any, i: number) => ({
      idx: i,
      hash: c.case_hash as string | undefined,
      label: caseLabel(c, i),
    }))
  }
  if (r.case_hash) {
    return [{ idx: 0, hash: r.case_hash as string, label: t({ zh: 'Case 1', en: 'Case 1' }) }]
  }
  return []
})

function caseLabel(c: any, i: number): string {
  // throughput uses `case_config`; text/video use `config`.
  const dev = c?.config?.device ?? c?.case_config?.device
  const devStr = Array.isArray(dev) ? dev.join(',') : dev || ''
  return devStr ? `Case ${i + 1}: ${devStr}` : `Case ${i + 1}`
}

// Convert ANSI escape codes to HTML (bold/color) + escape raw HTML so the log
// renders as formatted terminal output, preserving Unicode box-drawing chars.
function cleanLog(text: string): string {
  if (!text) return ''
  // 1. Escape HTML entities (prevent XSS from log content)
  let out = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  // 2. Convert common ANSI codes to HTML spans
  out = out
    .replace(/\x1b\[1m/g, '<b>')
    .replace(/\x1b\[22m/g, '</b>')
    .replace(/\x1b\[3m/g, '<i>')
    .replace(/\x1b\[23m/g, '</i>')
    .replace(/\x1b\[4m/g, '<u>')
    .replace(/\x1b\[24m/g, '</u>')
    .replace(/\x1b\[90m/g, '<span style="opacity:.6">')
    .replace(/\x1b\[31m/g, '<span style="color:var(--msm-red)">')
    .replace(/\x1b\[32m/g, '<span style="color:var(--msm-green)">')
    .replace(/\x1b\[33m/g, '<span style="color:var(--msm-amber)">')
    .replace(/\x1b\[34m/g, '<span style="color:var(--msm-accent)">')
    .replace(/\x1b\[35m/g, '<span style="color:#7C3AED">')
    .replace(/\x1b\[36m/g, '<span style="color:#0891B2">')
    .replace(/\x1b\[38;2;(\d+);(\d+);(\d+)m/g, '<span style="color:rgb($1,$2,$3)">')
    .replace(/\x1b\[0m/g, '</span>') // reset closes the last span (or b/i/u)
  // 3. Strip any remaining unhandled ANSI codes
  out = out.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')
  return out
}

const displayedLog = computed(() => {
  if (logCaseFilter.value === 'all') return cleanLog(logContent.value)
  const idx = Number(logCaseFilter.value)
  const c = caseList.value[idx]
  if (!c) return cleanLog(logContent.value)
  return cleanLog(caseLogCache.value[c.hash || ''] || '')
})

// True when the currently-selected case log failed to load (and wasn't cached),
// so the template can show a load-failed hint + retry instead of "no logs".
const caseLoadFailed = computed(() => {
  if (logCaseFilter.value === 'all') return false
  const idx = Number(logCaseFilter.value)
  const c = caseList.value[idx]
  if (!c || !c.hash) return false
  return !!caseLogError.value[c.hash]
})

// The RAW (ANSI-uncleaned) text of whatever the drawer is currently viewing —
// used for "copy whole log". Mirrors displayedLog's case-vs-all selection so
// the copied text matches what the user sees (minus the ANSI→HTML transform).
const rawDisplayedLog = computed(() => {
  if (logCaseFilter.value === 'all') return logContent.value
  const idx = Number(logCaseFilter.value)
  const c = caseList.value[idx]
  if (!c) return logContent.value
  return caseLogCache.value[c.hash || ''] || ''
})

// Filter the displayed log LINES by a case-insensitive substring, keeping the
// ANSI-cleaned HTML formatting on each matching line. Returns '' (no <pre> body)
// when nothing matches so the "no logs" branch still reads sensibly.
const filteredLog = computed(() => {
  const html = displayedLog.value
  const q = logSearch.value.trim().toLowerCase()
  if (!q) return html
  // cleanLog escaped HTML entities, so split on the literal \n (unescaped) that
  // came from the raw log — newlines aren't translated by cleanLog.
  const lines = html.split('\n')
  const matched = lines.filter((ln) => ln.toLowerCase().includes(q))
  return matched.join('\n')
})

// 'N / M lines' — matched line count over total visible line count.
const logMatchCount = computed(() => {
  const total = displayedLog.value ? displayedLog.value.split('\n').length : 0
  const q = logSearch.value.trim().toLowerCase()
  if (!q || !total) return null
  const matched = displayedLog.value.split('\n').filter((ln) => ln.toLowerCase().includes(q)).length
  return { matched, total }
})

// Auto-scroll to bottom when the log grows — but only if the user is already
// near the bottom, so scrolling up to read isn't yanked away.
watch(displayedLog, () => {
  const el = logPreRef.value
  if (!el) return
  const nearBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30
  if (nearBottom) {
    nextTick(() => {
      el.scrollTop = el.scrollHeight
    })
  }
})

// On initial log load (logLoading goes false), the <pre> ref becomes available
// AFTER displayedLog already set — so the watcher above misses it. Scroll to
// bottom explicitly once the element is rendered.
watch([() => !logLoading.value, () => logCaseFilter.value], ([ready]) => {
  if (!ready) return
  nextTick(() => {
    const el = logPreRef.value
    if (el) el.scrollTop = el.scrollHeight
  })
})

async function handleCopyLog() {
  const text = rawDisplayedLog.value
  if (!text || logCopying) return
  logCopying = true
  try {
    await navigator.clipboard.writeText(text)
    ElMessage.success(t({ zh: '日志已复制到剪贴板', en: 'Log copied to clipboard' }))
  } catch (err: any) {
    console.error('Failed to copy log:', err)
    ElMessage.error(t({ zh: '复制失败', en: 'Failed to copy' }))
  } finally {
    logCopying = false
  }
}

// Fetch a case's standalone log (by case_hash) on first selection — cached so
// re-selecting doesn't re-fetch. Failures are NOT cached (so re-selecting
// re-requests) and are recorded in caseLogError for a load-failed hint + retry.
async function loadCaseLog(idx: number) {
  const c = caseList.value[idx]
  if (!c || !c.hash) return
  // Skip only when we already have a successful (incl. empty) entry. A failed
  // fetch leaves no cache entry, so this guard lets the retry go through.
  if (caseLogCache.value[c.hash] != null && !caseLogError.value[c.hash]) return
  caseLogLoading.value = true
  caseLogError.value = { ...caseLogError.value, [c.hash]: false }
  try {
    const text = await getCaseLog(c.hash)
    caseLogCache.value = { ...caseLogCache.value, [c.hash]: text }
  } catch {
    caseLogError.value = { ...caseLogError.value, [c.hash]: true }
  } finally {
    caseLogLoading.value = false
  }
}

watch(logCaseFilter, (v) => {
  if (v !== 'all') loadCaseLog(Number(v))
})

// Retry the failed per-case log fetch for the currently selected case.
function retryCaseLog() {
  if (logCaseFilter.value !== 'all') loadCaseLog(Number(logCaseFilter.value))
}

// --- draggable divider ------------------------------------------------------
const moduleMeta = {
  text_generate: { icon: Document, title: { zh: '文本生成', en: 'Text Generation' } },
  video_generate: { icon: VideoCamera, title: { zh: '视频生成', en: 'Video Generation' } },
  throughput_optimizer: { icon: TrendCharts, title: { zh: '吞吐优化', en: 'Throughput Optimizer' } },
} as Record<string, { icon: any; title: { zh: string; en: string } }>

const meta = computed(() => moduleMeta[props.moduleId])

const topStyle = computed(() => ({ height: props.splitPercent + '%' }))

function triggerSubmit() {
  trackEvent(props.moduleId, 'run', 'submit')
  schemaFormRef.value?.submit()
}

async function handleSubmit(data: { moduleId: string; params: Record<string, any>; formSchemaVersion: string }) {
  await props.runner.submit(data.params, data.formSchemaVersion)
}

async function handleCancel() {
  // Data-loss guard: cancelling interrupts the running job and cannot be undone.
  // Confirm before proceeding; treat the user dismissing the dialog as a no-op.
  try {
    await ElMessageBox.confirm(
      t({
        zh: '将中断当前运行中的任务，且无法恢复。确定要取消吗？',
        en: 'This will interrupt the running job and cannot be undone. Cancel?',
      }),
      t({ zh: '取消任务', en: 'Cancel Job' }),
      {
        type: 'warning',
        confirmButtonText: t({ zh: '取消任务', en: 'Cancel Job' }),
        cancelButtonText: t({ zh: '保留', en: 'Keep' }),
      },
    )
  } catch {
    // user dismissed — do nothing
    return
  }
  props.runner.cancel()
}

async function openLog() {
  if (!props.runner.jobId) return
  logOpen.value = true
  logLoading.value = true
  logCaseFilter.value = 'all'
  logSearch.value = ''
  caseLogCache.value = {}
  caseLogError.value = {}
  try {
    logContent.value = await getJobLog(props.runner.jobId, 300)
  } catch (e) {
    logContent.value = ''
  } finally {
    logLoading.value = false
  }
}

// --- draggable divider ------------------------------------------------------
let dragging = false

function onDividerDown(e: PointerEvent) {
  dragging = true
  ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
  window.addEventListener('pointermove', onDividerMove)
  window.addEventListener('pointerup', onDividerUp)
}

function onDividerMove(e: PointerEvent) {
  if (!dragging || !containerRef.value) return
  const rect = containerRef.value.getBoundingClientRect()
  const pct = ((e.clientY - rect.top) / rect.height) * 100
  const clamped = Math.min(75, Math.max(20, pct))
  emit('update:splitPercent', Math.round(clamped))
}

function onDividerUp() {
  dragging = false
  window.removeEventListener('pointermove', onDividerMove)
  window.removeEventListener('pointerup', onDividerUp)
}

// Keyboard resize of the divider — mirrors the pointer-drag clamp (20–75%).
function onDividerKey(e: KeyboardEvent) {
  const MIN = 20
  const MAX = 75
  const cur = props.splitPercent
  let next: number | null = null
  switch (e.key) {
    case 'ArrowUp':
      next = cur + 5
      break
    case 'ArrowDown':
      next = cur - 5
      break
    case 'Home':
      next = MIN
      break
    case 'End':
      next = MAX
      break
  }
  if (next == null) return
  e.preventDefault()
  emit('update:splitPercent', Math.min(MAX, Math.max(MIN, next)))
}

onBeforeUnmount(() => {
  // Only release the divider listeners here — the runner is parent-owned
  // (Console) so its polling continues across tab switches.
  onDividerUp()
})
</script>

<template>
  <div ref="containerRef" class="workspace">
    <!-- TOP: form -->
    <section class="pane pane-form" :style="topStyle">
      <header class="pane-header">
        <span class="pane-title">
          <el-icon class="pane-icon"><component :is="meta?.icon" /></el-icon>
          {{ meta ? t(meta.title) : moduleId }}
          <span class="pane-sub">{{ t({ zh: '配置', en: 'Config' }) }}</span>
        </span>
        <div class="header-actions">
          <el-button
            type="primary"
            :loading="runner.submitting || runner.isBusy"
            :disabled="runner.isBusy"
            class="run-btn"
            @click="triggerSubmit"
          >
            <span class="run-dot" />
            {{ t({ zh: '运行', en: 'Run' }) }}
          </el-button>
        </div>
      </header>

      <!-- 运行状态指示器 -->
      <div v-if="runner.isBusy" class="running-indicator">
        <el-icon class="is-loading"><Loading /></el-icon>
        <span class="running-text">{{ statusText }}</span>
      </div>

      <div class="pane-body">
        <Suspense>
          <schema-form ref="schemaFormRef" :module-id="moduleId" @submit="handleSubmit" />
          <template #fallback>
            <el-skeleton :rows="6" animated />
          </template>
        </Suspense>
      </div>
    </section>

    <!-- draggable divider -->
    <div
      class="divider"
      role="separator"
      aria-orientation="horizontal"
      aria-label="Resize panes"
      tabindex="0"
      @pointerdown="onDividerDown"
      @keydown="onDividerKey"
    >
      <span class="divider-grip" />
    </div>

    <!-- BOTTOM: result -->
    <section class="pane pane-result">
      <header class="pane-header">
        <span class="pane-title">
          <span class="pane-sub">{{ t({ zh: '结果', en: 'Result' }) }}</span>
        </span>
      </header>
      <result-pane :runner="runner" :module-id="moduleId" @cancel="handleCancel" @view-log="openLog" />
    </section>

    <!-- log drawer -->
    <el-drawer v-model="logOpen" size="55%" direction="rtl">
      <template #header>
        <div class="log-drawer-header">
          <span class="log-drawer-title">{{ t({ zh: '运行日志', en: 'Run Logs' }) }}</span>
          <el-button
            size="small"
            :disabled="!rawDisplayedLog"
            @click="handleCopyLog"
          >
            <el-icon><DocumentCopy /></el-icon>
            <span class="log-copy-label">{{ t({ zh: '复制日志', en: 'Copy Log' }) }}</span>
          </el-button>
        </div>
      </template>
      <div v-if="logLoading" class="log-loading">{{ t({ zh: '加载中…', en: 'Loading…' }) }}</div>
      <template v-else>
        <div v-if="caseList.length" class="log-case-filter">
          <el-radio-group v-model="logCaseFilter" size="small">
            <el-radio-button value="all">{{ t({ zh: '总日志', en: 'All' }) }}</el-radio-button>
            <el-radio-button
              v-for="c in caseList"
              :key="c.idx"
              :value="String(c.idx)"
            >{{ c.label }}</el-radio-button>
          </el-radio-group>
        </div>
        <div class="log-search">
          <el-input
            v-model="logSearch"
            clearable
            size="small"
            :placeholder="t({ zh: '搜索日志行…', en: 'Filter log lines…' })"
          />
          <span v-if="logMatchCount" class="log-match-count">
            {{ logMatchCount.matched }} / {{ logMatchCount.total }} {{ t({ zh: '行', en: 'lines' }) }}
          </span>
        </div>
        <pre v-if="caseLogLoading" ref="logPreRef" class="log-pre">{{ t({ zh: '加载中…', en: 'Loading…' }) }}</pre>
        <div v-else-if="caseLoadFailed" ref="logPreRef" class="log-pre log-pre-error">
          <span>{{ t({ zh: '日志加载失败', en: 'Failed to load log' }) }}</span>
          <el-button size="small" class="log-retry-btn" @click="retryCaseLog">
            {{ t({ zh: '重试', en: 'Retry' }) }}
          </el-button>
        </div>
        <pre v-else-if="filteredLog" ref="logPreRef" class="log-pre" v-html="filteredLog"></pre>
        <pre v-else ref="logPreRef" class="log-pre">{{ t({ zh: '暂无日志', en: 'No logs yet' }) }}</pre>
      </template>
    </el-drawer>
  </div>
</template>

<style scoped>
.workspace {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.pane {
  display: flex;
  flex-direction: column;
  min-height: 0;
  background: var(--msm-bg-panel);
  border: 1px solid var(--msm-border);
  border-radius: var(--msm-radius-lg);
  overflow: hidden;
}

.pane-form {
  flex: 0 0 auto;
}

.pane-result {
  flex: 1 1 0;
  margin-top: 0;
}

.pane-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  border-bottom: 1px solid var(--msm-border);
  background: var(--msm-bg-panel-2);
  flex: 0 0 auto;
}

.running-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: var(--msm-bg-panel-2);
  border-bottom: 1px solid var(--msm-border);
  color: var(--el-color-primary);
  font-size: 13px;
  font-weight: 500;
}

.running-indicator .is-loading {
  font-size: 16px;
}

.running-text {
  color: var(--msm-text);
}

.pane-title {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 14px;
  color: var(--msm-text);
}

.pane-icon {
  color: var(--el-color-primary);
}

.pane-sub {
  font-weight: 400;
  font-size: 12px;
  color: var(--msm-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.pane-body {
  flex: 1 1 0;
  overflow: auto;
  padding: 12px 18px 18px;
}

.run-btn {
  font-weight: 600;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.run-dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--msm-bg-panel);
  margin-right: 6px;
}

.divider {
  flex: 0 0 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: row-resize;
  position: relative;
  z-index: var(--msm-z-sticky);
}

/* Enlarge the touch/hit zone to ~45px around the 9px bar (visual bar unchanged). */
.divider::after {
  content: '';
  position: absolute;
  inset: -18px 0;
}

.divider::before {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  top: 3px;
  bottom: 3px;
  border-top: 1px solid var(--msm-border);
}

.divider-grip {
  width: 44px;
  height: 4px;
  border-radius: var(--msm-radius-sm);
  background: var(--msm-border-strong);
  transition: background 0.18s ease;
  position: relative;
}

.divider:focus-visible .divider-grip {
  outline: 2px solid var(--msm-accent);
  outline-offset: 4px;
}

.divider:hover .divider-grip {
  background: var(--el-color-primary);
}

.log-loading {
  padding: 24px;
  color: var(--msm-text-muted);
}

.log-case-filter {
  margin-bottom: 10px;
}

.log-drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
}

.log-drawer-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--msm-text);
}

.log-copy-label {
  margin-left: 4px;
}

.log-search {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.log-match-count {
  flex: 0 0 auto;
  font-size: 12px;
  color: var(--msm-text-muted);
  white-space: nowrap;
}
.log-pre {
  margin: 0;
  padding: 14px;
  background: var(--msm-bg-deep);
  border: 1px solid var(--msm-border);
  border-radius: var(--msm-radius);
  font-family: 'Fira Code', monospace;
  font-size: 12px;
  line-height: 1.55;
  color: var(--msm-text);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: calc(100vh - 140px);
  overflow: auto;
}

/* load-failed hint: same panel chrome, red accent + right-aligned retry. */
.log-pre-error {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--msm-red);
  border-color: color-mix(in srgb, var(--msm-red) 45%, var(--msm-border));
  white-space: normal;
}
</style>

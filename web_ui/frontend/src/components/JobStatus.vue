<script setup lang="ts">
/**
 * Job status component — orchestrator.
 *
 * Responsibilities: poll job status + hold job/error/cancelling state + status banner
 * (deduped) + compose three child components (Header / CommandCard / LogDrawer).
 * Display and interaction details are pushed down to the child components; this file
 * only does state-machine orchestration and navigation (<300 lines, per
 * the frontend development guide).
 *
 * Per the REST API contract.
 */
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import { ElMessageBox } from 'element-plus'
import { useRouter } from 'vue-router'
import { useLocale } from '@/composables/useLocale'
import { getJob, cancelJob } from '@/services/api'
import JobStatusHeader from './job-status/JobStatusHeader.vue'
import JobCommandCard from './job-status/JobCommandCard.vue'
import JobLogDrawer from './job-status/JobLogDrawer.vue'

interface Props {
  jobId: string
}

const props = defineProps<Props>()
const router = useRouter()
const { t } = useLocale()

// State
const job = ref<any>(null)
const loading = ref(true)
const error = ref<string | null>(null)
const logDrawerVisible = ref(false)
const cancelling = ref(false)

// Polling — then+setTimeout self-scheduling
let pollTimer: ReturnType<typeof setTimeout> | null = null
const POLL_INTERVAL = 1500

// Run id — same-route param navigation (/jobs/A/status -> /jobs/B/status)
// reuses this component instance; if we don't invalidate in-flight requests,
// the OLD getJob's response can land after the NEW one and overwrite job state,
// or the OLD success-redirect timer can fire with props.jobId stale.
// A monotonic counter checked after every await makes stale effects no-ops.
// See PR-632 #48.
let runId = 0

// Success redirect: countdown + timer references are stored as refs so they can be
// cleaned up in onBeforeUnmount, avoiding timers firing router.replace after the
// component is unmounted (navigating to an already-unmounted view).
const REDIRECT_DELAY_SECONDS = 3
const redirectTimer = ref<ReturnType<typeof setTimeout> | null>(null)
const redirectCountdown = ref(0)
let countdownInterval: ReturnType<typeof setInterval> | null = null

// True once we've seen the job in-flight (pending/running) during THIS session.
// Success auto-redirect fires only when watched live — not when navigating back
// to an already-finished job (which would bounce straight back to the result).
const sawInFlight = ref(false)

// Computed
const isPending = computed(() => job.value?.status === 'pending')
const isRunning = computed(() => job.value?.status === 'running')
const isSucceeded = computed(() => job.value?.status === 'succeeded')
const isFailed = computed(() => job.value?.status === 'failed')
const isCancelled = computed(() => job.value?.status === 'cancelled')
const isInterrupted = computed(() => job.value?.status === 'interrupted')
const isFinished = computed(() =>
  isSucceeded.value || isFailed.value || isCancelled.value || isInterrupted.value
)
const canCancel = computed(() =>
  (isPending.value || isRunning.value) && !job.value?.cancel_requested
)
const redirectPending = computed(() => isSucceeded.value && sawInFlight.value)

// Methods
const fetchJobStatus = async () => {
  const myRun = runId
  try {
    const fetched = await getJob(props.jobId)
    // Stale response: jobId changed (or polling restarted) while we were
    // waiting — drop the result so it doesn't overwrite the NEW job's state.
    if (myRun !== runId) return
    job.value = fetched
    error.value = null
    if (isPending.value || isRunning.value) sawInFlight.value = true

    if (isFinished.value) {
      stopPolling()
      // Auto-navigate to result ONLY when watched live; use replace() so this
      // transient status view doesn't linger in history. Drive the redirect
      // through a tracked timer + visible countdown so the user can see the
      // remaining seconds and abort via "Stay here".
      if (redirectPending.value) {
        startRedirectCountdown(myRun)
      }
    }
  } catch (err: any) {
    if (myRun !== runId) return // stale — don't stomp new state
    // A 404 means the job is gone (deleted / unknown id) — treat as terminal and
    // STOP polling, otherwise tick() reschedules every 1.5s forever on a dead id.
    // Other errors (flaky network / 5xx) stay transient: keep polling.
    if (err?.status === 404) {
      stopPolling()
      error.value = err.message || t({ zh: '作业不存在', en: 'Job not found' })
    } else {
      error.value = err.message || t({ zh: '获取作业状态失败', en: 'Failed to fetch job status' })
    }
    console.error('Failed to fetch job status:', err)
  } finally {
    if (myRun === runId) loading.value = false
  }
}

const handleCancel = async () => {
  try {
    await ElMessageBox.confirm(
      t({ zh: '确定要取消此作业吗？', en: 'Are you sure you want to cancel this job?' }),
      t({ zh: '取消作业', en: 'Cancel Job' }),
      {
        type: 'warning',
        confirmButtonText: t({ zh: '取消作业', en: 'Cancel Job' }),
        cancelButtonText: t({ zh: '保留', en: 'Keep' }),
      },
    )
  } catch {
    // user dismissed (cancel / Esc / backdrop) — no-op
    return
  }
  const myRun = runId
  cancelling.value = true
  try {
    await cancelJob(props.jobId)
    if (myRun !== runId) return // navigated away while confirming — don't mutate stale state
    job.value = { ...job.value, cancel_requested: true }
  } catch (err: any) {
    if (myRun !== runId) return
    error.value = err.message || t({ zh: '取消作业失败', en: 'Failed to cancel job' })
    console.error('Failed to cancel job:', err)
  } finally {
    if (myRun === runId) cancelling.value = false
  }
}

const viewResult = () => router.push({ name: 'jobResult', params: { jobId: props.jobId } })
const goBack = () => router.push({ name: 'history' })
const openLogDrawer = () => {
  logDrawerVisible.value = true
}

// Success-redirect countdown: visible countdown + interruptible + centralized timer cleanup.
const startRedirectCountdown = (forRun: number) => {
  // Guard against duplicate starts (e.g. status polling re-entering the completed state).
  if (redirectTimer.value !== null) return
  redirectCountdown.value = REDIRECT_DELAY_SECONDS
  countdownInterval = setInterval(() => {
    if (redirectCountdown.value > 0) redirectCountdown.value -= 1
  }, 1000)
  redirectTimer.value = setTimeout(() => {
    clearRedirectTimers()
    // Drop the redirect if the user navigated away while the countdown was
    // ticking — otherwise we'd route to the OLD jobId's result page.
    if (forRun !== runId) return
    router.replace({ name: 'jobResult', params: { jobId: props.jobId } })
  }, REDIRECT_DELAY_SECONDS * 1000)
}

const clearRedirectTimers = () => {
  if (redirectTimer.value !== null) {
    clearTimeout(redirectTimer.value)
    redirectTimer.value = null
  }
  if (countdownInterval !== null) {
    clearInterval(countdownInterval)
    countdownInterval = null
  }
}

// User-initiated "Stay here": clears the redirect timers and stays on the current status view.
const stayHere = () => {
  clearRedirectTimers()
  redirectCountdown.value = 0
}

// Polling control
const startPolling = () => {
  if (pollTimer !== null) return
  const tick = async () => {
    if (pollTimer === null) return
    await fetchJobStatus()
    if (pollTimer !== null) pollTimer = setTimeout(tick, POLL_INTERVAL)
  }
  pollTimer = setTimeout(tick, 0)
}

const stopPolling = () => {
  if (pollTimer) {
    clearTimeout(pollTimer)
    pollTimer = null
  }
}

// Lifecycle
onMounted(() => {
  startPolling()
})
onBeforeUnmount(() => {
  stopPolling()
  clearRedirectTimers()
})
watch(
  () => props.jobId,
  () => {
    // New job — invalidate any in-flight fetches / countdowns tied to the
    // previous jobId (their responses would otherwise overwrite state for a
    // different job). Also clears any active redirect so the OLD job's
    // success redirect doesn't fire after navigating away.
    runId += 1
    clearRedirectTimers()
    stopPolling()
    startPolling()
  },
  { immediate: true },
)
</script>

<template>
  <div class="job-status">
    <!-- Loading State -->
    <div v-if="loading" class="loading-state">
      <el-skeleton :rows="3" animated />
    </div>

    <!-- Error State -->
    <el-alert
      v-else-if="error"
      type="error"
      :title="t({ zh: '加载失败', en: 'Failed to Load' })"
      :description="error"
      show-icon
      :closable="false"
    />

    <!-- Job Status Content -->
    <div v-else-if="job" class="status-content">
      <JobStatusHeader
        :job="job"
        :can-cancel="canCancel"
        :cancelling="cancelling"
        @cancel="handleCancel"
        @view-result="viewResult"
        @view-log="openLogDrawer"
        @go-back="goBack"
      />

      <!-- Status banners (deduplicated: each appears exactly once) -->
      <el-alert
        v-if="isSucceeded"
        type="success"
        :title="t({ zh: '作业成功完成', en: 'Job Completed Successfully' })"
        show-icon
        closable
        class="success-banner"
      >
        <template #default>
          <template v-if="redirectPending && redirectTimer !== null">
            <span>{{
              t({ zh: '结果已准备就绪，', en: 'Results are ready. ' })
            }}{{
              t({ zh: `${redirectCountdown}s 后跳转到结果页面`, en: `Redirecting in ${redirectCountdown}s…` })
            }}</span>
            <el-link
              type="primary"
              :underline="false"
              class="stay-here-link"
              @click="stayHere"
            >
              {{ t({ zh: '留在此页', en: 'Stay here' }) }}
            </el-link>
          </template>
          <template v-else>
            {{ t({ zh: '结果已准备就绪，点击上方【查看结果】查看', en: 'Results are ready — click View Results above' }) }}
          </template>
        </template>
      </el-alert>

      <el-alert
        v-if="isFailed || isInterrupted"
        type="error"
        :title="t({ zh: '作业失败', en: 'Job Failed' })"
        :description="job.error || t({ zh: '作业执行失败', en: 'Job execution failed' })"
        show-icon
        class="error-banner"
      />

      <el-alert
        v-if="isCancelled"
        type="warning"
        :title="t({ zh: '作业已取消', en: 'Job Cancelled' })"
        :description="t({ zh: '此作业已被用户取消', en: 'This job was cancelled by the user' })"
        show-icon
        class="info-banner"
      />

      <JobCommandCard v-if="job.command" :command="job.command" :commands="job.commands" :params="job.params" />
    </div>

    <!-- Log Drawer -->
    <JobLogDrawer :job-id="jobId" v-model="logDrawerVisible" />
  </div>
</template>

<style scoped>
.job-status {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
  min-height: calc(100vh - 40px);
}

.loading-state {
  padding: 40px 20px;
}

.status-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* Alert banners */
.error-banner,
.info-banner,
.success-banner {
  margin-top: 0;
  border-radius: var(--msm-radius);
  border: 1px solid var(--el-border-color-lighter);
}

.error-banner {
  border-color: rgba(239, 68, 68, 0.3);
  background: color-mix(in srgb, var(--msm-red) 6%, var(--msm-bg-panel));
}

.info-banner {
  border-color: rgba(245, 158, 11, 0.3);
  background: color-mix(in srgb, var(--msm-amber) 6%, var(--msm-bg-panel));
}

.success-banner {
  border-color: rgba(34, 197, 94, 0.3);
  background: color-mix(in srgb, var(--msm-green) 6%, var(--msm-bg-panel));
}

.stay-here-link {
  margin-left: 8px;
  font-size: inherit;
  vertical-align: baseline;
}

@media (max-width: 768px) {
  .job-status {
    padding: 12px;
  }
}
</style>

/**
 * useJobRunner (workspace). One instance per module: owns a single job's
 * submit -> poll -> result lifecycle so a tab's result persists across tab
 * switches (the Console parent holds one instance per module).
 *
 * States: idle -> pending -> running -> (succeeded|failed|cancelled|interrupted)
 *
 * Polling: recursive `setTimeout` driven by the server's `poll_interval_ms`
 * hint (falling back to a local default) — never `setInterval`, so a slow request
 * can't stack calls. A mounted guard prevents state writes after unmount (race /
 * leak), and consecutive transient errors stop polling after a threshold instead
 * of looping forever. A 404 is treated as terminal (job gone), not transient.
 */
import { ref, computed, reactive, onMounted, onBeforeUnmount } from 'vue'
import { ElMessage } from 'element-plus'
import { submitJob, getJob, getJobResult, cancelJob } from '@/services/api'

export type RunStatus = 'idle' | 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted'

const DEFAULT_POLL_MS = 1500
const MAX_BACKOFF_MS = 8000
const MAX_CONSECUTIVE_ERRORS = 5

export function useJobRunner(moduleId: string) {
  const status = ref<RunStatus>('idle')
  const jobId = ref<string | null>(null)
  const progress = ref(0)
  const progressText = ref('')
  const error = ref<string | null>(null)
  const errorDetail = ref<string | null>(null)
  const result = ref<Record<string, any> | null>(null)
  const records = ref<any[] | null>(null)
  const resultError = ref<string | null>(null)
  const cancelRequested = ref(false)
  const submitting = ref(false)
  const schemaVersion = ref<string | null>(null)

  let timer: ReturnType<typeof setTimeout> | null = null
  let mounted = false
  let consecutiveErrors = 0
  // Generation counter: bumped on every `submit` so a stale in-flight
  // `getJob`/`getJobResult` request (from a previous submission) that returns
  // AFTER the new submit can't write its old job's state onto the current
  // runner. Each async entry captures `myRun = runId` and bails out if
  // `myRun !== runId` after the await — same pattern as Docs.vue (#46) and
  // JobStatus.vue (#48).
  let runId = 0

  const isBusy = computed(() => status.value === 'pending' || status.value === 'running')
  const isTerminal = computed(() =>
    ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(status.value),
  )
  const canCancel = computed(() => isBusy.value && !cancelRequested.value)

  function stopPolling() {
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
  }

  function schedulePoll(delay: number) {
    stopPolling()
    timer = setTimeout(poll, delay)
  }

  async function fetchResult() {
    if (!jobId.value) return
    const myRun = runId
    const expectedJobId = jobId.value
    try {
      const env = await getJobResult(expectedJobId)
      // Bail if unmounted OR a newer submit/cancel has moved us on OR the
      // jobId was swapped (defence-in-depth — a stale response for a different
      // job must never overwrite the current result).
      if (!mounted || myRun !== runId || jobId.value !== expectedJobId) return
      result.value = env.result || {}
      records.value = env.records || []
      resultError.value = null
    } catch (e: any) {
      if (!mounted || myRun !== runId || jobId.value !== expectedJobId) return
      // result fetch failed — expose the error so UI can show it instead of
      // leaving the user with a success icon and empty result area
      console.error('fetchResult failed', e)
      resultError.value = e?.message || 'Failed to fetch job result'
    }
  }

  async function poll() {
    if (!mounted || !jobId.value) return
    const myRun = runId
    const expectedJobId = jobId.value
    try {
      const job = await getJob(expectedJobId)
      // Bail on stale: unmounted / newer submit / jobId swapped. The stale
      // response must NOT overwrite current status/progress — that was the
      // original race (#11).
      if (!mounted || myRun !== runId || jobId.value !== expectedJobId) return
      consecutiveErrors = 0
      status.value = (job.status as RunStatus) || status.value
      progress.value = job.progress ?? progress.value
      progressText.value = job.progress_text ?? progressText.value
      cancelRequested.value = !!job.cancel_requested
      if (job.error) error.value = job.error
      if (job.error_detail) errorDetail.value = job.error_detail

      if (['succeeded', 'failed', 'cancelled', 'interrupted'].includes(job.status)) {
        stopPolling()
        if (job.status === 'succeeded') {
          await fetchResult()
        }
        return
      }
      // Non-terminal: poll again at the server-suggested cadence.
      schedulePoll(job.poll_interval_ms ?? DEFAULT_POLL_MS)
    } catch (e: any) {
      if (!mounted || myRun !== runId || jobId.value !== expectedJobId) return
      // 404: the job is gone (deleted/unknown) — stop with a clear error instead
      // of looping forever as if it were transient.
      if (e?.status === 404) {
        stopPolling()
        status.value = 'failed'
        const msg = e?.message || 'Job not found'
        error.value = msg
        errorDetail.value = msg
        return
      }
      // Transient error: back off, and give up after too many in a row.
      consecutiveErrors += 1
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        stopPolling()
        status.value = 'failed'
        const msg = 'Lost contact with server — polling stopped'
        error.value = msg
        errorDetail.value = msg
        return
      }
      const backoff = Math.min(DEFAULT_POLL_MS * 2 ** consecutiveErrors, MAX_BACKOFF_MS)
      schedulePoll(backoff)
    }
  }

  function startPolling() {
    stopPolling()
    void poll()
  }

  async function submit(params: Record<string, any>, version: string) {
    submitting.value = true
    // Bump generation: any in-flight poll/result-fetch from a previous submit
    // that returns AFTER this point will see myRun !== runId and bail out, so
    // stale data from the old job never overwrites the new runner state (#11).
    runId += 1
    const myRun = runId
    // reset previous run
    stopPolling()
    consecutiveErrors = 0
    status.value = 'pending'
    progress.value = 0
    progressText.value = ''
    error.value = null
    errorDetail.value = null
    result.value = null
    records.value = null
    resultError.value = null
    cancelRequested.value = false
    schemaVersion.value = version

    try {
      const res = await submitJob(moduleId, version, params)
      if (!mounted || myRun !== runId) return
      jobId.value = res.job_id
      status.value = (res.status as RunStatus) || 'pending'
      ElMessage.success(`Task submitted successfully (ID: ${res.job_id.slice(0, 8)}...)`)
      startPolling()
    } catch (e: any) {
      if (!mounted || myRun !== runId) return
      status.value = 'failed'
      const detail = e?.response?.data?.detail || e?.message || 'Submission failed'
      error.value = detail
      errorDetail.value = detail
    } finally {
      submitting.value = false
    }
  }

  async function cancel() {
    if (!jobId.value) return
    const myRun = runId
    const expectedJobId = jobId.value
    try {
      await cancelJob(expectedJobId)
      if (!mounted || myRun !== runId || jobId.value !== expectedJobId) return
      cancelRequested.value = true
    } catch (e: any) {
      console.error('cancel failed', e)
    }
  }

  function teardown() {
    stopPolling()
    // Bump generation so any in-flight request that returns after teardown
    // observes myRun !== runId and bails out (belt-and-braces with `mounted`).
    runId += 1
  }

  onMounted(() => {
    mounted = true
  })
  onBeforeUnmount(() => {
    mounted = false
    teardown()
  })

  // Wrap in reactive() so refs auto-unwrap when the runner is passed as a prop
  // and accessed in a child template (runner.status -> string, not a Ref).
  return reactive({
    moduleId,
    status,
    jobId,
    progress,
    progressText,
    error,
    errorDetail,
    result,
    records,
    resultError,
    cancelRequested,
    submitting,
    schemaVersion,
    isBusy,
    isTerminal,
    canCancel,
    submit,
    cancel,
    teardown,
  })
}

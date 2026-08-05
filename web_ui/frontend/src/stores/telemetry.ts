/**
 * Telemetry store — buffers UI interaction events (field changes + button
 * clicks) and flushes them in batches to POST /api/telemetry. Powers usage-
 * frequency analysis (which form fields/controls get used most) to inform
 * form-layout optimization (e.g. tier-based visibility).
 *
 * Best-effort, non-blocking: flush failures keep the buffer for the next retry
 * (capped); the backend POST never raises on error. No field VALUES are
 * collected — only (module, target, event_type, time).
 *
 * Lazy start: the periodic flush timer + beforeunload/visibility listeners are
 * wired on the first trackEvent() call, so consumers need no bootstrap.
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { logTelemetryBatch } from '@/services/api'
import { getFingerprint } from '@/services/fingerprint'

export interface TelemetryEvent {
  module_id: string
  target: string
  event_type: string
  fingerprint: string
}

const FLUSH_INTERVAL_MS = 30000
const FLUSH_THRESHOLD = 20
const FIELD_DEBOUNCE_MS = 1000
const RETRY_BUFFER_CAP = 100

export const useTelemetryStore = defineStore('telemetry', () => {
  const fingerprint = getFingerprint()
  const buffer = ref<TelemetryEvent[]>([])
  const lastEventAt = ref<Record<string, number>>({})
  let timer: ReturnType<typeof setTimeout> | null = null
  let started = false

  // Self-scheduled setTimeout (avoids setInterval piling up when flush is slow —
  // see frontend guide §5): the next round is only queued after the previous flush
  // completes, so a slow network never stacks multiple pending requests.
  function scheduleFlush() {
    if (timer) clearTimeout(timer)
    timer = setTimeout(async () => {
      timer = null
      await flush()
      // Only queue the next round while still running (stops if stop() was called mid-flush)
      if (started) scheduleFlush()
    }, FLUSH_INTERVAL_MS)
  }

  function start() {
    if (started) return
    started = true
    scheduleFlush()
    window.addEventListener('beforeunload', flushOnUnload)
    document.addEventListener('visibilitychange', onVisibility)
  }

  function stop() {
    if (timer) clearTimeout(timer)
    timer = null
    started = false
    window.removeEventListener('beforeunload', flushOnUnload)
    document.removeEventListener('visibilitychange', onVisibility)
  }

  /** Record an interaction. Pass debounce=true for high-frequency field changes
   * (text/number typing) so repeated setFieldValue calls within FIELD_DEBOUNCE_MS
   * count once per (module, target, event_type). */
  function trackEvent(moduleId: string, target: string, eventType: string, debounce = false) {
    if (!started) start()
    if (debounce) {
      const key = `${moduleId}|${target}|${eventType}`
      const now = Date.now()
      if (lastEventAt.value[key] && now - lastEventAt.value[key] < FIELD_DEBOUNCE_MS) return
      lastEventAt.value[key] = now
    }
    buffer.value.push({ module_id: moduleId, target, event_type: eventType, fingerprint })
    if (buffer.value.length >= FLUSH_THRESHOLD) flush()
  }

  async function flush() {
    if (buffer.value.length === 0) return
    const batch = buffer.value.splice(0)
    try {
      await logTelemetryBatch(batch)
    } catch {
      // Keep the batch for retry, capped to avoid unbounded growth on a down server.
      buffer.value.unshift(...batch.slice(-RETRY_BUFFER_CAP))
    }
  }

  function flushOnUnload() {
    if (buffer.value.length === 0) return
    const payload = JSON.stringify({ events: buffer.value.splice(0) })
    try {
      navigator.sendBeacon('/api/telemetry', new Blob([payload], { type: 'application/json' }))
    } catch {
      // ignore — best-effort on page unload
    }
  }

  function onVisibility() {
    if (document.visibilityState === 'hidden') flush()
  }

  return { buffer, trackEvent, flush, start, stop }
})

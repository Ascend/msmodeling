/**
 * API layer (guide §1.2 — endpoint constants + business functions). Endpoints
 * are centralized here (no hard-coded URLs in components/pages); business funcs
 * call the shared ``apiClient`` from ``request.ts``. Job submission / result
 * fetch live here; result rendering is per-module components.
 */
import { apiClient } from './request'

/** Centralized endpoint paths (the guide's "api.js" constants layer). */
export const endpoints = {
  devices: '/api/options/devices',
  modules: '/api/modules',
  formSchema: (moduleId: string) => `/api/modules/${moduleId}/form-schema`,
  jobs: '/api/jobs',
  job: (jobId: string) => `/api/jobs/${jobId}`,
  jobLog: (jobId: string) => `/api/jobs/${jobId}/log`,
  caseLog: (caseHash: string) => `/api/cases/${caseHash}/log`,
  jobTrace: (jobId: string, seq: number) => `/api/jobs/${jobId}/trace/${seq}`,
  jobCancel: (jobId: string) => `/api/jobs/${jobId}/cancel`,
  jobResult: ( jobId: string) => `/api/jobs/${jobId}/result`,
  telemetry: '/api/telemetry',
  telemetryStats: '/api/telemetry/stats',
  telemetryUsers: '/api/telemetry/users',
}

export const traceUrl = (jobId: string, seq: number): string =>
  endpoints.jobTrace(jobId, seq)


export const api = {
  /** Fetch the device option list once per session (cached by the caller). */
  async getDevices() {
    const res = await apiClient.get(endpoints.devices)
    return res.data as Array<{ value: string; label?: string }>
  },
  async getModules() {
    const res = await apiClient.get(endpoints.modules)
    return res.data
  },
  async getFormSchema(moduleId: string, version?: string) {
    const res = await apiClient.get(endpoints.formSchema(moduleId), {
      params: version ? { version } : {},
    })
    return res.data
  },
  /** Get options from a named dynamic source (e.g., devices). */
  async getOptions(endpoint: string) {
    const res = await apiClient.get(endpoint)
    return res.data as Array<{ value: string; label?: string }>
  },

  /** Submit a new modeling job. */
  async submitJob(moduleId: string, formSchemaVersion: string, params: Record<string, any>) {
    const res = await apiClient.post(endpoints.jobs, {
      module_id: moduleId,
      form_schema_version: formSchemaVersion,
      params,
    })
    return res.data
  },

  /** Poll job status/progress. */
  async getJob(jobId: string) {
    const res = await apiClient.get(endpoints.job(jobId))
    return res.data
  },

  /** Fetch captured runner logs (text/plain). */
  async getJobLog(jobId: string, tail: number = 200): Promise<string> {
    const res = await apiClient.get(endpoints.jobLog(jobId), {
      params: { tail },
      responseType: 'text',
    })
    return res.data as string
  },

  /** Fetch ONE case's CLI log by case_hash (per-case isolation — no regex split). */
  async getCaseLog(caseHash: string, tail: number = 0): Promise<string> {
    const res = await apiClient.get(endpoints.caseLog(caseHash), {
      params: { tail },
      responseType: 'text',
    })
    return res.data as string
  },

  /** Download a Chrome trace file for a specific case. */
  async getJobTrace(jobId: string, seq: number): Promise<Blob> {
    const res = await apiClient.get(endpoints.jobTrace(jobId, seq), {
      responseType: 'blob',
    })
    return res.data as Blob
  },

  /** Cooperatively request cancellation. */
  async cancelJob(jobId: string) {
    const res = await apiClient.post(endpoints.jobCancel(jobId))
    return res.data
  },

  /** Fetch job's assembled result envelope. */
  async getJobResult(jobId: string) {
    const res = await apiClient.get(endpoints.jobResult(jobId))
    return res.data
  },

  /** List jobs with filtering and pagination. */
  async listJobs(params?: {
    module_id?: string
    status?: string
    limit?: number
    offset?: number
  }) {
    const res = await apiClient.get(endpoints.jobs, { params })
    return res.data
  },

  /** Post a batch of UI interaction events (telemetry / usage analytics). */
  async logTelemetryBatch(events: Array<{ module_id: string; target: string; event_type: string; fingerprint?: string }>) {
    const res = await apiClient.post(endpoints.telemetry, { events })
    return res.data
  },

  /** Distinct-user counts (by browser fingerprint) overall + per module. */
  async getTelemetryUsers(moduleId?: string) {
    const res = await apiClient.get(endpoints.telemetryUsers, {
      params: moduleId ? { module_id: moduleId } : {},
    })
    return res.data
  },

  /** Aggregated interaction counts by (module, target, event_type). */
  async getTelemetryStats(params: Record<string, any> = {}) {
    const res = await apiClient.get(endpoints.telemetryStats, { params })
    return res.data
  },
}

// --- Named exports (consumers import these directly, e.g. JobStatus/JobResult) ---
export const submitJob = api.submitJob.bind(api)
export const getJob = api.getJob.bind(api)
export const getJobLog = api.getJobLog.bind(api)
export const getCaseLog = api.getCaseLog.bind(api)
export const getJobTrace = api.getJobTrace.bind(api)
export const cancelJob = api.cancelJob.bind(api)
export const getJobResult = api.getJobResult.bind(api)
export const listJobs = api.listJobs.bind(api)
export const logTelemetryBatch = api.logTelemetryBatch.bind(api)
export const getTelemetryUsers = api.getTelemetryUsers.bind(api)
export const getTelemetryStats = api.getTelemetryStats.bind(api)

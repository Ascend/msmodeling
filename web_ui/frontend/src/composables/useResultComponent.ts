/**
 * Result component resolution (single source of truth).
 *
 * text/video/throughput result components are resolved by module_id + multi_case flag.
 * The two render entry points (`pages/JobResult.vue` deep-link page +
 * `components/workspace/ResultPane.vue` Console inline) share this function to avoid
 * each maintaining its own switch (copy-paste-instead-of-reuse and props drift).
 *
 * The result-component props contract is unified as `{ result, jobId, hideTraceDownloads? }`.
 */
import type { Component } from 'vue'
import TextGenerateResult from '@/components/result/text/TextGenerateResult.vue'
import TextMultiCaseResult from '@/components/result/text/TextMultiCaseResult.vue'
import VideoGenerateResult from '@/components/result/video/VideoGenerateResult.vue'
import VideoMultiCaseResult from '@/components/result/video/VideoMultiCaseResult.vue'
import ThroughputOptimizerResult from '@/components/result/throughput/ThroughputOptimizerResult.vue'
import ThroughputMultiCaseResult from '@/components/result/throughput/ThroughputMultiCaseResult.vue'

export function resolveResultComponent(
  moduleId: string | undefined,
  multiCase: boolean | undefined,
): Component | null {
  switch (moduleId) {
    case 'text_generate':
      return multiCase ? TextMultiCaseResult : TextGenerateResult
    case 'video_generate':
      return multiCase ? VideoMultiCaseResult : VideoGenerateResult
    case 'throughput_optimizer':
      return multiCase ? ThroughputMultiCaseResult : ThroughputOptimizerResult
    default:
      return null
  }
}

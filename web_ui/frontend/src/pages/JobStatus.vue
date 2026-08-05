<script setup lang="ts">
/**
 * Job status page.
 *
 * Displays job execution status with polling, progress milestones, and error handling.
 * Wraps the JobStatus component for the specific job.
 */
import JobStatus from '@/components/JobStatus.vue'
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
// Reactive: same-route param changes (/jobs/A/status -> /jobs/B/status) reuse
// this component instance, so a one-time snapshot would keep polling the old job.
const jobId = computed(() => String(route.params.jobId))
</script>

<template>
  <job-status :job-id="jobId" />
</template>

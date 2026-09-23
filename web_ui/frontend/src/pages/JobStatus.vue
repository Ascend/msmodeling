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

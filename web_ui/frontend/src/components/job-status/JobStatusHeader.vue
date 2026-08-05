<script setup lang="ts">
/**
 * Job status header subcomponent.
 *
 * Status header card: status icon/tag + module + job id + progress bar + action buttons.
 * Pure display + emit events (orthogonal): navigation/cancellation handled by the parent.
 */
import { computed } from 'vue'
import { useLocale } from '@/composables/useLocale'
import {
  Document, Loading, CircleCheck, CircleClose, Warning, VideoPause,
  ArrowLeft, View,
} from '@element-plus/icons-vue'

interface Props {
  job: any
  canCancel: boolean
  cancelling: boolean
}

const props = defineProps<Props>()
const emit = defineEmits<{
  cancel: []
  'view-result': []
  'view-log': []
  'go-back': []
}>()
const { t } = useLocale()

const isPending = computed(() => props.job?.status === 'pending')
const isRunning = computed(() => props.job?.status === 'running')
const isSucceeded = computed(() => props.job?.status === 'succeeded')
const isFailed = computed(() => props.job?.status === 'failed')
const isCancelled = computed(() => props.job?.status === 'cancelled')
const isInterrupted = computed(() => props.job?.status === 'interrupted')

const statusText = computed(() => {
  switch (props.job?.status) {
    case 'pending': return t({ zh: '等待中', en: 'Pending' })
    case 'running': return t({ zh: '运行中', en: 'Running' })
    case 'succeeded': return t({ zh: '成功', en: 'Succeeded' })
    case 'failed': return t({ zh: '失败', en: 'Failed' })
    case 'cancelled': return t({ zh: '已取消', en: 'Cancelled' })
    case 'interrupted': return t({ zh: '中断', en: 'Interrupted' })
    default: return t({ zh: '未知', en: 'Unknown' })
  }
})

const statusType = computed(() => {
  if (isSucceeded.value) return 'success'
  if (isFailed.value || isInterrupted.value) return 'danger'
  if (isCancelled.value) return 'warning'
  return 'info'
})

const statusIcon = computed(() => {
  if (isSucceeded.value) return CircleCheck
  if (isFailed.value || isInterrupted.value) return CircleClose
  if (isCancelled.value) return VideoPause
  if (isRunning.value || isPending.value) return Loading
  return Document
})

const statusIconClass = computed(() => {
  if (isSucceeded.value) return 'icon-success'
  if (isFailed.value || isInterrupted.value) return 'icon-error'
  if (isCancelled.value) return 'icon-warning'
  if (isRunning.value || isPending.value) return 'icon-running'
  return 'icon-info'
})

const showProgress = computed(() => isRunning.value || isPending.value)
</script>

<template>
  <el-card class="status-card main-card">
    <template #header>
      <div class="card-header">
        <div class="header-left">
          <el-icon class="header-icon" :class="[statusIconClass, { 'msm-spin': isRunning || isPending }]">
            <component :is="statusIcon" />
          </el-icon>
          <div class="header-info">
            <div class="header-title">
              <el-tag :type="statusType" size="large" class="status-tag">
                {{ statusText }}
              </el-tag>
              <span class="module-label">{{ job.module_id }}</span>
            </div>
            <div class="job-id">Job ID: {{ job.id }}</div>
          </div>
        </div>
      </div>
    </template>

    <!-- Progress Section: no percentage bar (progress is not estimable for
         simulation jobs); show only the textual status detail while running. -->
    <div v-if="showProgress" class="progress-section">
      <div v-if="job.progress_text" class="progress-detail">
        {{ job.progress_text }}
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="action-buttons">
      <el-button
        v-if="canCancel"
        type="warning"
        :loading="cancelling"
        @click="emit('cancel')"
      >
        <el-icon><Warning /></el-icon>
        {{ t({ zh: '取消作业', en: 'Cancel Job' }) }}
      </el-button>
      <el-button v-if="job.cancel_requested" disabled>
        <el-icon><Loading /></el-icon>
        {{ t({ zh: '取消请求已发送', en: 'Cancel Requested' }) }}
      </el-button>
      <el-button v-if="isSucceeded" type="primary" @click="emit('view-result')">
        <el-icon><View /></el-icon>
        {{ t({ zh: '查看结果', en: 'View Results' }) }}
      </el-button>
      <el-button @click="emit('view-log')">
        <el-icon><Document /></el-icon>
        {{ t({ zh: '查看日志', en: 'View Logs' }) }}
      </el-button>
      <el-button @click="emit('go-back')">
        <el-icon><ArrowLeft /></el-icon>
        {{ t({ zh: '返回', en: 'Go Back' }) }}
      </el-button>
    </div>
  </el-card>
</template>

<style scoped>
.status-card.main-card {
  border: 1px solid var(--msm-border);
  background: var(--msm-bg-panel);
  box-shadow: var(--msm-shadow);
}

.status-card {
  text-align: left;
  border-radius: var(--msm-radius);
  overflow: hidden;
  transition: box-shadow var(--msm-transition) var(--msm-ease-out);
}

.status-card:hover {
  box-shadow: var(--msm-shadow-lg);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
}

.header-icon {
  font-size: 24px;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--msm-radius);
  flex-shrink: 0;
}

.header-icon.icon-success { background: color-mix(in srgb, var(--msm-green) 12%, transparent); color: var(--msm-green); }
.header-icon.icon-error { background: color-mix(in srgb, var(--msm-red) 12%, transparent); color: var(--msm-red); }
.header-icon.icon-warning { background: color-mix(in srgb, var(--msm-amber) 12%, transparent); color: var(--msm-amber); }
.header-icon.icon-running {
  background: color-mix(in srgb, var(--msm-text-muted) 12%, transparent);
  color: var(--msm-text-muted);
}
.header-icon.icon-info { background: color-mix(in srgb, var(--msm-accent) 12%, transparent); color: var(--msm-accent); }

.header-info { flex: 1; }

.header-title {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 4px;
}

.status-tag {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.module-label {
  font-size: 13px;
  color: var(--msm-text-muted);
  font-weight: 500;
  padding: 2px 8px;
  background: var(--el-fill-color-light);
  border-radius: 4px;
}

.job-id {
  color: var(--msm-text-muted);
  font-size: 13px;
  font-family: 'Fira Code', monospace;
  letter-spacing: 0.5px;
}

.progress-section {
  margin: 24px 0;
  padding: 0 4px;
}

.progress-detail {
  margin-top: 12px;
  color: var(--msm-text-muted);
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.action-buttons {
  display: flex;
  gap: 10px;
  justify-content: flex-start;
  flex-wrap: wrap;
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid var(--el-border-color-lighter);
}

.action-buttons .el-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

@media (max-width: 768px) {
  .header-title {
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
  }
  .action-buttons { justify-content: stretch; }
  .action-buttons .el-button { flex: 1; min-width: 0; }
}
</style>

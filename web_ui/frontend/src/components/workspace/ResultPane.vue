<script setup lang="ts">
/**
 * ResultPane (workspace bottom half). Renders the active module's job result
 * across its lifecycle: idle placeholder -> pending/running progress ->
 * succeeded (per-module result component) -> failed/cancelled notice.
 *
 * Reads from a shared `runner` (useJobRunner instance) so the result persists
 * across tab switches.
 */
import { computed } from 'vue'
import { ElButton, ElAlert, ElEmpty, ElTag, ElIcon } from 'element-plus'
import { DataLine, Loading, VideoCamera, TrendCharts, Cpu, Document } from '@element-plus/icons-vue'
import { useLocale } from '@/composables/useLocale'
import { resolveResultComponent } from '@/composables/useResultComponent'

const props = defineProps<{
  runner: any
  moduleId: string
}>()

const emit = defineEmits<{ cancel: []; 'view-log': [] }>()

const { t } = useLocale()

const moduleMeta: Record<string, { icon: any; title: { zh: string; en: string } }> = {
  text_generate: { icon: DataLine, title: { zh: '文本生成', en: 'Text Generation' } },
  video_generate: { icon: VideoCamera, title: { zh: '视频生成', en: 'Video Generation' } },
  throughput_optimizer: { icon: TrendCharts, title: { zh: '吞吐优化', en: 'Throughput Optimizer' } },
}

const resultComponent = computed(() =>
  resolveResultComponent(props.moduleId, props.runner?.result?.multi_case)
)

const statusText = computed(() => {
  switch (props.runner?.status) {
    case 'pending':
      return t({ zh: '排队等待', en: 'Queued' })
    case 'running':
      return t({ zh: '运行中', en: 'Running' })
    case 'succeeded':
      return t({ zh: '已完成', en: 'Completed' })
    case 'failed':
      return t({ zh: '失败', en: 'Failed' })
    case 'cancelled':
      return t({ zh: '已取消', en: 'Cancelled' })
    case 'interrupted':
      return t({ zh: '已中断', en: 'Interrupted' })
    default:
      return ''
  }
})
</script>

<template>
  <div class="result-pane">
    <!-- IDLE: no job yet -->
    <div v-if="runner.status === 'idle'" class="state state-idle">
      <el-empty :description="t({ zh: '填写上方表单并提交，结果将在此展示', en: 'Fill the form above and submit — results appear here' })">
        <template #image>
          <el-icon :size="56" class="idle-icon"><component :is="moduleMeta[moduleId]?.icon || Cpu" /></el-icon>
        </template>
      </el-empty>
    </div>

    <!-- PENDING / RUNNING: progress -->
    <div v-else-if="runner.status === 'pending' || runner.status === 'running'" class="state state-running">
      <div class="running-head">
        <el-icon class="is-loading spin"><Loading /></el-icon>
        <span class="running-label">{{ statusText }}</span>
        <el-button
          v-if="runner.canCancel"
          size="small"
          type="danger"
          plain
          class="cancel-btn"
          @click="emit('cancel')"
        >
          {{ t({ zh: '取消', en: 'Cancel' }) }}
        </el-button>
        <el-button v-else-if="runner.cancelRequested" size="small" disabled>
          {{ t({ zh: '取消中…', en: 'Cancelling…' }) }}
        </el-button>
        <el-button size="small" plain :icon="Document" class="log-btn" @click="emit('view-log')">
          {{ t({ zh: '日志', en: 'Logs' }) }}
        </el-button>
      </div>
      <div v-if="runner.progressText" class="progress-text">{{ runner.progressText }}</div>
    </div>

    <!-- SUCCEEDED: result component -->
    <div v-else-if="runner.status === 'succeeded'" class="state state-succeeded">
      <div class="success-strip">
        <el-tag type="success" effect="dark" round>{{ statusText }}</el-tag>
        <span class="job-id-mono">{{ runner.jobId }}</span>
        <el-button size="small" plain :icon="Document" class="log-btn" @click="emit('view-log')">{{ t({ zh: '日志', en: 'Logs' }) }}</el-button>
      </div>
      <!-- Result fetch failed — show error instead of empty result area -->
      <el-alert
        v-if="runner.resultError"
        type="warning"
        :title="t({ zh: '结果加载失败', en: 'Failed to load result' })"
        :description="runner.resultError"
        show-icon
        :closable="false"
        class="result-error-alert"
      />
      <div v-else class="result-host">
        <component v-if="resultComponent" :is="resultComponent" :result="runner.result || {}" :records="runner.records" :job-id="runner.jobId" />
      </div>
    </div>

    <!-- FAILED -->
    <div v-else-if="runner.status === 'failed'" class="state state-failed">
      <el-alert
        type="error"
        :title="t({ zh: '任务失败', en: 'Job Failed' })"
        :description="runner.error || t({ zh: '执行过程中出错', en: 'An error occurred during execution' })"
        show-icon
        :closable="false"
      />
      <pre v-if="runner.errorDetail" class="error-detail">{{ runner.errorDetail }}</pre>
      <el-button size="small" plain :icon="Document" class="log-btn" @click="emit('view-log')">{{ t({ zh: '查看日志', en: 'View Logs' }) }}</el-button>
    </div>

    <!-- CANCELLED / INTERRUPTED -->
    <div v-else class="state state-stopped">
      <el-alert
        :type="runner.status === 'interrupted' ? 'error' : 'warning'"
        :title="statusText"
        :description="runner.status === 'interrupted'
          ? t({ zh: '服务在运行时中断，任务未完成', en: 'The server interrupted during the run; the job did not finish' })
          : t({ zh: '任务已被取消', en: 'The job was cancelled' })"
        show-icon
        :closable="false"
      />
    </div>
  </div>
</template>

<style scoped>
.result-pane {
  height: 100%;
  overflow: auto;
  padding: 16px 20px;
}

.state {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.state-idle {
  align-items: center;
  justify-content: center;
}

.idle-icon {
  color: var(--el-color-primary);
  opacity: 0.65;
}

.state-running {
  justify-content: center;
  gap: 14px;
}

.running-head {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.spin {
  color: var(--el-color-primary);
}

.running-label {
  font-weight: 600;
  font-size: 15px;
}

.progress-text {
  color: var(--el-text-color-secondary);
  font-family: 'Fira Code', monospace;
  font-size: 12.5px;
}

/* Logs button — a discoverable secondary action. Outlined in the brand accent
   (icon + colored border + colored text) so it stands out on the dark panel,
   but kept flat (transparent bg, not filled) so it never out-stages the primary
   Run button. Replaces the old `text` button which blended in and customers
   reported as hard to find. */
.log-btn {
  --el-button-bg-color: transparent;
  --el-button-border-color: color-mix(in srgb, var(--msm-green) 42%, var(--msm-border));
  --el-button-text-color: var(--msm-green);
  --el-button-hover-bg-color: color-mix(in srgb, var(--msm-green) 14%, transparent);
  --el-button-hover-border-color: var(--msm-green);
  --el-button-hover-text-color: var(--msm-green);
  --el-button-active-bg-color: color-mix(in srgb, var(--msm-green) 22%, transparent);
  --el-button-active-border-color: var(--msm-green);
  --el-button-active-text-color: var(--msm-green);
  font-weight: 500;
}

.success-strip {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.job-id-mono {
  font-family: 'Fira Code', monospace;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.result-host {
  flex: 1;
  min-height: 0;
}

.result-error-alert {
  margin: 12px 0;
}

.error-detail {
  margin: 12px 0;
  padding: 12px;
  background: var(--msm-bg-deep);
  border: 1px solid var(--msm-border);
  border-radius: 6px;
  font-family: 'Fira Code', monospace;
  font-size: 12px;
  color: var(--msm-text-err);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 240px;
  overflow: auto;
}

.state-failed,
.state-stopped {
  justify-content: center;
}
</style>

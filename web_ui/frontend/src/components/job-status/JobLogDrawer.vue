<script setup lang="ts">
/**
 * Job log drawer subcomponent.
 *
 * Log drawer: visibility controlled via v-model; auto-fetches the log on open (getJobLog).
 * Self-contained fetch + loading state.
 */
import { ref, watch } from 'vue'
import { useLocale } from '@/composables/useLocale'
import { getJobLog } from '@/services/api'
import { Loading } from '@element-plus/icons-vue'

interface Props {
  jobId: string
  modelValue: boolean
}

const props = defineProps<Props>()
const emit = defineEmits<{ 'update:modelValue': [value: boolean] }>()
const { t } = useLocale()

const logContent = ref('')
const logLoading = ref(false)
const logError = ref<string | null>(null)

const fetchLog = async () => {
  logLoading.value = true
  logError.value = null
  try {
    logContent.value = await getJobLog(props.jobId, 200)
  } catch (err: any) {
    console.error('Failed to fetch log:', err)
    logError.value = err?.message || t({ zh: '日志加载失败', en: 'Failed to load logs' })
  } finally {
    logLoading.value = false
  }
}

// Fetch the log when the drawer opens (watch modelValue)
watch(
  () => props.modelValue,
  (visible) => {
    if (visible) fetchLog()
  },
)
</script>

<template>
  <el-drawer
    :model-value="modelValue"
    :title="t({ zh: '作业日志', en: 'Job Logs' })"
    size="60%"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <div v-if="logLoading" class="log-loading">
      <el-icon class="msm-spin"><Loading /></el-icon>
      <span>{{ t({ zh: '加载日志中...', en: 'Loading logs...' }) }}</span>
    </div>
    <el-alert
      v-else-if="logError"
      class="log-error"
      type="error"
      :title="logError"
      :closable="false"
      show-icon
    >
      <el-button type="primary" size="small" :loading="logLoading" @click="fetchLog">
        {{ t({ zh: '重试', en: 'Retry' }) }}
      </el-button>
    </el-alert>
    <div v-else-if="logContent" class="log-content">
      <pre>{{ logContent }}</pre>
    </div>
    <el-empty v-else :description="t({ zh: '暂无日志', en: 'No logs available' })" />
  </el-drawer>
</template>

<style scoped>
.log-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 40px;
  color: var(--msm-text-muted);
}

.log-error {
  margin: 16px;
}
.log-error .el-button {
  margin-top: 8px;
}

.log-content {
  padding: 0;
}

.log-content pre {
  margin: 0;
  padding: 16px;
  background: var(--msm-bg-deep);
  border-radius: 4px;
  font-size: 12px;
  line-height: 1.5;
  max-height: calc(100vh - 120px);
  overflow: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
  color: var(--msm-text-muted);
}
</style>

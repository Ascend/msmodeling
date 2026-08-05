<script setup lang="ts">
/**
 * Job result page.
 *
 * Resolves the module's result component by `module_id`, fetches the job result,
 * and renders the per-module result visualization.
 *
 */
import { ref, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useLocale } from '@/composables/useLocale'
import { getJobResult } from '@/services/api'
import { resolveResultComponent } from '@/composables/useResultComponent'

const route = useRoute()
const router = useRouter()
const { t } = useLocale()

// State
const jobData = ref<any>(null)
const loading = ref(true)
const error = ref<string | null>(null)

const jobId = computed(() => route.params.jobId as string)

// Computed
const moduleId = computed(() => jobData.value?.module_id)

// Resolve result component based on module_id (single source via composable)
const resultComponent = computed(() =>
  resolveResultComponent(moduleId.value, result.value?.multi_case)
)

const result = computed(() => jobData.value?.result || {})
const formSchema = computed(() => jobData.value?.form_schema || {})
const records = computed(() => jobData.value?.records || [])

// Methods
const fetchResult = async () => {
  // Clear stale state from a previous jobId so navigating between result pages
  // (/jobs/A/result -> /jobs/B/result reuses this component instance) doesn't
  // briefly show A's result/error while B loads.
  jobData.value = null
  error.value = null
  loading.value = true
  try {
    const response = await getJobResult(jobId.value)
    jobData.value = response
    error.value = null
  } catch (err: any) {
    error.value = err.message || t({ zh: '获取结果失败', en: 'Failed to fetch result' })
    console.error('Failed to fetch job result:', err)
  } finally {
    loading.value = false
  }
}

const goBack = () => {
  // A job result is a detail view drilled into from the History list —
  // "Back" goes back to the jobs hub (History), not the console. Navigate
  // explicitly rather than router.back(), which is history-dependent.
  router.push({ name: 'history' })
}

const viewStatus = () => {
  router.push({ name: 'jobStatus', params: { jobId: jobId.value } })
}

// Refetch whenever the route's jobId changes. Same-route param navigation
// (/jobs/A/result -> /jobs/B/result) reuses this component instance, so
// onMounted alone would not fire and jobData would stay stuck on A.
watch(jobId, fetchResult, { immediate: true })
</script>

<template>
  <div class="job-result-page">
    <!-- Loading State -->
    <div v-if="loading" class="loading-state">
      <el-skeleton :rows="5" animated />
    </div>

    <!-- Error State -->
    <el-alert
      v-else-if="error"
      type="error"
      :title="t({ zh: '加载失败', en: 'Failed to Load' })"
      :description="error"
      show-icon
      :closable="false"
      class="error-alert"
    />

    <!-- Job Result Content -->
    <div v-else-if="jobData" class="result-content">
      <!-- Header -->
      <el-card class="header-card">
        <div class="header-content">
          <div class="header-left">
            <h2 class="page-title">
              {{ t({ zh: '作业结果', en: 'Job Result' }) }}
            </h2>
            <div class="meta-info">
              <span class="job-id">Job ID: {{ jobId }}</span>
              <el-divider direction="vertical" />
              <span class="module-id">
                {{ t({ zh: '模块', en: 'Module' }) }}: {{ moduleId }}
              </span>
            </div>
          </div>
          <div class="header-actions">
            <el-button @click="viewStatus">
              {{ t({ zh: '查看状态', en: 'View Status' }) }}
            </el-button>
            <el-button @click="goBack">
              {{ t({ zh: '返回', en: 'Go Back' }) }}
            </el-button>
          </div>
        </div>
      </el-card>

      <!-- Unsupported Module Warning -->
      <el-alert
        v-if="!resultComponent"
        type="warning"
        :title="t({ zh: '暂不支持的结果类型', en: 'Unsupported Result Type' })"
        :description="`${t({ zh: '模块', en: 'Module' })}: ${moduleId}`"
        show-icon
        :closable="false"
        class="warning-alert"
      />

      <!-- Result Component -->
      <component
        v-if="resultComponent"
        :is="resultComponent"
        :result="result"
        :form-schema="formSchema"
        :records="records"
        :job-id="jobId"
      />
    </div>

    <!-- Empty State -->
    <el-empty
      v-else
      :description="t({ zh: '暂无数据', en: 'No data available' })"
    />
  </div>
</template>

<style scoped>
.job-result-page {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.loading-state {
  padding: 40px 20px;
}

.error-alert,
.warning-alert {
  margin-bottom: 20px;
}

.result-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.header-card {
  margin-bottom: 0;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}

.header-left {
  flex: 1;
  min-width: 200px;
}

.page-title {
  margin: 0 0 8px 0;
  font-size: 24px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.meta-info {
  display: flex;
  align-items: center;
  color: var(--el-text-color-secondary);
  font-size: 14px;
}

.job-id,
.module-id {
  font-family: monospace;
}

.header-actions {
  display: flex;
  gap: 12px;
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    align-items: flex-start;
  }

  .header-actions {
    width: 100%;
  }

  .header-actions .el-button {
    flex: 1;
  }
}
</style>

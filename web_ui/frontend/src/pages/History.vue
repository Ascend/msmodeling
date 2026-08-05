<script setup lang="ts">
/**
 * History page.
 *
 * Displays job history with:
 * - List view with filtering (by module/status)
 * - Pagination
 * - Job labels
 * - Reopen flow (click to view results)
 *
 */
import { ref, computed, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useLocale } from '@/composables/useLocale'
import { trackEvent } from '@/services/telemetrySink'
import { ElMessage } from 'element-plus'
import { Search } from '@element-plus/icons-vue'
import { api } from '@/services/api'

const router = useRouter()
const { t } = useLocale()

// State
const jobs = ref<any[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

// Filters
const filters = ref({
  module_id: '',
  status: '',
})

// Search (client-side, filters the currently-loaded page)
const searchText = ref('')
const debouncedSearch = ref('')
const searchIcon = Search
let searchTimer: number | undefined
watch(searchText, (val) => {
  if (searchTimer != null) {
    window.clearTimeout(searchTimer)
  }
  searchTimer = window.setTimeout(() => {
    debouncedSearch.value = val
  }, 300)
})

const filteredJobs = computed(() => {
  const q = debouncedSearch.value.trim().toLowerCase()
  if (!q) return jobs.value
  return jobs.value.filter(
    (job) =>
      String(job.job_id ?? '').toLowerCase().includes(q) ||
      String(job.label ?? '').toLowerCase().includes(q),
  )
})

const pagination = ref({
  page: 1,
  pageSize: 20,
  total: 0,
})

// Shown when the loaded page fills up — search only filters loaded rows, not the whole dataset.
const pageIsFull = computed(
  () => jobs.value.length >= pagination.value.pageSize,
)

// Module options
const moduleOptions = computed(() => [
  { label: t({ zh: '全部', en: 'All' }), value: '' },
  { label: 'text_generate', value: 'text_generate' },
  { label: 'video_generate', value: 'video_generate' },
  { label: 'throughput_optimizer', value: 'throughput_optimizer' },
])

// Status options
const statusOptions = computed(() => [
  { label: t({ zh: '全部', en: 'All' }), value: '' },
  { label: t({ zh: '等待中', en: 'Pending' }), value: 'pending' },
  { label: t({ zh: '运行中', en: 'Running' }), value: 'running' },
  { label: t({ zh: '成功', en: 'Succeeded' }), value: 'succeeded' },
  { label: t({ zh: '失败', en: 'Failed' }), value: 'failed' },
  { label: t({ zh: '已取消', en: 'Cancelled' }), value: 'cancelled' },
  { label: t({ zh: '中断', en: 'Interrupted' }), value: 'interrupted' },
])

// Computed

// Status tag type
const getStatusType = (status: string) => {
  switch (status) {
    case 'succeeded':
      return 'success'
    case 'failed':
    case 'interrupted':
      return 'danger'
    case 'cancelled':
      return 'warning'
    case 'running':
      return 'primary'
    default:
      return 'info'
  }
}

const getStatusText = (status: string) => {
  switch (status) {
    case 'pending':
      return t({ zh: '等待中', en: 'Pending' })
    case 'running':
      return t({ zh: '运行中', en: 'Running' })
    case 'succeeded':
      return t({ zh: '成功', en: 'Succeeded' })
    case 'failed':
      return t({ zh: '失败', en: 'Failed' })
    case 'cancelled':
      return t({ zh: '已取消', en: 'Cancelled' })
    case 'interrupted':
      return t({ zh: '中断', en: 'Interrupted' })
    default:
      return status
  }
}

// Methods
const fetchJobs = async () => {
  loading.value = true
  error.value = null
  try {
    const params: any = {
      limit: pagination.value.pageSize,
      offset: (pagination.value.page - 1) * pagination.value.pageSize,
    }

    if (filters.value.module_id) {
      params.module_id = filters.value.module_id
    }
    if (filters.value.status) {
      params.status = filters.value.status
    }

    const response = await api.listJobs(params) as { items: any[]; total: number }
    jobs.value = response.items
    pagination.value.total = response.total
  } catch (err: any) {
    const msg: string = err.message || t({ zh: '获取任务列表失败', en: 'Failed to fetch job list' })
    error.value = msg
    ElMessage.error(msg)
    console.error('Failed to fetch jobs:', err)
  } finally {
    loading.value = false
  }
}

const handleFilterChange = () => {
  pagination.value.page = 1
  fetchJobs()
}

// Fully reset every filter (module, status, AND the search box incl. its
// debounced shadow) then re-fetch. The empty-state "reset filters" button used
// to call handleFilterChange, which keeps searchText/debouncedSearch — so a
// prior search could leave the table looking empty even after "reset".
const resetFilters = () => {
  filters.value.module_id = ''
  filters.value.status = ''
  searchText.value = ''
  debouncedSearch.value = ''
  if (searchTimer != null) {
    window.clearTimeout(searchTimer)
    searchTimer = undefined
  }
  pagination.value.page = 1
  fetchJobs()
}

const handlePageChange = (page: number) => {
  pagination.value.page = page
  fetchJobs()
}

const handleViewResult = (jobId: string) => {
  trackEvent('global', 'view_result', 'click')
  router.push({ name: 'jobResult', params: { jobId } })
}

const formatDateTime = (dateTime: string | null) => {
  if (!dateTime) return '-'
  return new Date(dateTime).toLocaleString()
}

const goBack = () => {
  router.push({ name: 'console' })
}

// Lifecycle
onMounted(() => {
  fetchJobs()
})
</script>

<template>
  <div class="history-page">
    <el-card>
      <template #header>
        <div class="header-content">
          <h2>{{ t({ zh: '任务历史', en: 'Job History' }) }}</h2>
          <el-button @click="goBack">
            {{ t({ zh: '返回', en: 'Back' }) }}
          </el-button>
        </div>
      </template>

      <!-- Filters -->
      <el-row :gutter="16" class="filter-section">
        <el-col :xs="24" :sm="12" :md="4">
          <el-select
            v-model="filters.module_id"
            :placeholder="t({ zh: '选择模块', en: 'Select Module' })"
            clearable
            @change="handleFilterChange"
          >
            <el-option
              v-for="opt in moduleOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-col>
        <el-col :xs="24" :sm="12" :md="4">
          <el-select
            v-model="filters.status"
            :placeholder="t({ zh: '选择状态', en: 'Select Status' })"
            clearable
            @change="handleFilterChange"
          >
            <el-option
              v-for="opt in statusOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-col>
        <el-col :xs="24" :sm="12" :md="6">
          <el-input
            v-model="searchText"
            :prefix-icon="searchIcon"
            :placeholder="t({ zh: '搜索作业ID / 标签', en: 'Search job ID / label' })"
            clearable
          />
        </el-col>
        <el-col :xs="24" :sm="24" :md="10" class="action-col">
          <el-button type="primary" :loading="loading" :disabled="loading" @click="fetchJobs">
            {{ t({ zh: '刷新', en: 'Refresh' }) }}
          </el-button>
        </el-col>
      </el-row>
      <p v-if="pageIsFull" class="search-hint">
        {{ t({ zh: '搜索仅过滤当前已加载页', en: 'Search filters the currently-loaded page' }) }}
      </p>

      <!-- Job List -->
      <el-empty
        v-if="error"
        :description="error"
        style="margin-top: 16px"
      >
        <el-button type="primary" :loading="loading" @click="fetchJobs">
          {{ t({ zh: '重试', en: 'Retry' }) }}
        </el-button>
      </el-empty>
      <el-table
        v-else
        v-loading="loading"
        :data="filteredJobs"
        :default-sort="{ prop: 'created_at', order: 'descending' }"
        stripe
        style="width: 100%; margin-top: 16px"
      >
        <el-table-column prop="job_id" :label="t({ zh: '任务ID', en: 'Job ID' })" width="200" />
        <el-table-column prop="module_id" :label="t({ zh: '模块', en: 'Module' })" width="180" sortable />
        <el-table-column prop="label" :label="t({ zh: '标签', en: 'Label' })" width="150" />
        <el-table-column prop="status" :label="t({ zh: '状态', en: 'Status' })" width="120" sortable>
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)" size="small">
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" :label="t({ zh: '创建时间', en: 'Created' })" width="180" sortable>
          <template #default="{ row }">
            {{ formatDateTime(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column prop="completed_at" :label="t({ zh: '完成时间', en: 'Completed' })" width="180" sortable>
          <template #default="{ row }">
            {{ formatDateTime(row.completed_at) }}
          </template>
        </el-table-column>
        <el-table-column :label="t({ zh: '操作', en: 'Actions' })" width="200" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="row.status === 'succeeded'"
              type="primary"
              size="small"
              @click="handleViewResult(row.job_id)"
            >
              {{ t({ zh: '查看结果', en: 'View Result' }) }}
            </el-button>
            <el-button
              v-else-if="row.status === 'running' || row.status === 'pending'"
              type="default"
              size="small"
              @click="router.push({ name: 'jobStatus', params: { jobId: row.job_id } })"
            >
              {{ t({ zh: '查看状态', en: 'View Status' }) }}
            </el-button>
            <el-button
              v-else-if="['failed', 'cancelled', 'interrupted'].includes(row.status)"
              type="default"
              size="small"
              @click="router.push({ name: 'jobStatus', params: { jobId: row.job_id } })"
            >
              {{ t({ zh: '查看详情', en: 'View Details' }) }}
            </el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty :description="t({ zh: '暂无任务记录', en: 'No job records' })">
            <el-button @click="resetFilters">{{ t({ zh: '重置筛选', en: 'Reset filters' }) }}</el-button>
          </el-empty>
        </template>
      </el-table>

      <!-- Pagination -->
      <el-pagination
        v-if="!error"
        v-model:current-page="pagination.page"
        v-model:page-size="pagination.pageSize"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        :total="pagination.total"
        @size-change="handleFilterChange"
        @current-change="handlePageChange"
        style="margin-top: 16px; justify-content: center"
      />
    </el-card>
  </div>
</template>

<style scoped>
.history-page {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-content h2 {
  margin: 0;
}

.filter-section {
  margin-bottom: 16px;
}

.action-col {
  text-align: right;
}

.search-hint {
  margin: 0 0 4px;
  color: var(--msm-text-muted);
  font-size: 12px;
}
</style>

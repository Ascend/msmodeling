<script setup lang="ts">
/**
 * Console — the single-page tabbed workspace. A top tab bar switches the three
 * capability modules; each tab renders a ModuleWorkspace (form on top, result
 * below). One useJobRunner instance per module lives here so each module's
 * running job + result persists when the user switches tabs.
 */
import { ref, watch, onBeforeUnmount } from 'vue'
import { ElTabs, ElTabPane, ElIcon, ElMessage, ElLink } from 'element-plus'
import type { TabPaneName } from 'element-plus'
import { Document, VideoCamera, TrendCharts, WarningFilled, Link } from '@element-plus/icons-vue'
import { useLocale } from '@/composables/useLocale'
import { useJobRunner } from '@/composables/useJobRunner'
import ModuleWorkspace from '@/components/workspace/ModuleWorkspace.vue'
import { trackEvent } from '@/services/telemetrySink'

const { t } = useLocale()

// Source-of-truth repo link + disclaimer shown in the console footer.
const SOURCE_URL = 'https://gitcode.com/Ascend/msmodeling'

// One independent runner per module (state persists across tab switches).
const runners = {
  text_generate: useJobRunner('text_generate'),
  video_generate: useJobRunner('video_generate'),
  throughput_optimizer: useJobRunner('throughput_optimizer'),
}

// Block tab switching while the CURRENT tab's job is running (#5): leaving a
// running tab remounts the workspace (:key) and disrupts the live form/result.
// Dedupe the warning toast: a busy tab can fire beforeTabLeave on every click /
// arrow-key attempt, which would stack identical warnings. Only show it at most
// once per ~2.5s burst for the same module (still always return false to block).
const BLOCK_DEDUPE_MS = 2500
const lastBlockTs = ref(0)
let lastBlockModule: ModuleId | null = null

function beforeTabLeave(_active: TabPaneName, oldActive: TabPaneName): boolean {
  if (runners[oldActive as ModuleId]?.isBusy) {
    const now = Date.now()
    const sameModule = lastBlockModule === oldActive
    if (!sameModule || now - lastBlockTs.value >= BLOCK_DEDUPE_MS) {
      ElMessage.warning(
        t({
          zh: '当前任务执行中，请等待完成或取消后再切换标签页',
          en: 'A job is running — finish or cancel it before switching tabs',
        }),
      )
      lastBlockTs.value = now
      lastBlockModule = oldActive as ModuleId
    }
    return false
  }
  return true
}

// Per-module divider position (form/result split).
const splits = ref<Record<string, number>>({
  text_generate: 42,
  video_generate: 42,
  throughput_optimizer: 42,
})

const activeTab = ref<ModuleId>('text_generate')

// Telemetry: record module (tab) switches.
watch(activeTab, (tab) => trackEvent('global', `tab:${tab}`, 'click'))

type ModuleId = 'text_generate' | 'video_generate' | 'throughput_optimizer'

const tabs: { id: ModuleId; icon: any; title: { zh: string; en: string } }[] = [
  { id: 'text_generate', icon: Document, title: { zh: '文本生成', en: 'Text Generation' } },
  { id: 'video_generate', icon: VideoCamera, title: { zh: '视频生成', en: 'Video Generation' } },
  { id: 'throughput_optimizer', icon: TrendCharts, title: { zh: '吞吐优化', en: 'Throughput Optimizer' } },
]

onBeforeUnmount(() => {
  // Stop all background polling when the console is destroyed.
  Object.values(runners).forEach((r) => r.teardown())
})
</script>

<template>
  <div class="console">
    <el-tabs v-model="activeTab" :before-leave="beforeTabLeave" class="console-tabs">
      <el-tab-pane
        v-for="tab in tabs"
        :key="tab.id"
        :name="tab.id"
        lazy
      >
        <template #label>
          <span class="tab-label">
            <el-icon class="tab-icon"><component :is="tab.icon" /></el-icon>
            {{ t(tab.title) }}
          </span>
        </template>
      </el-tab-pane>
    </el-tabs>

    <div class="tab-body">
      <!-- Only the active module's workspace is mounted (:key forces a clean
           remount on tab switch so SchemaForm re-inits from the right schema;
           this is the sole consumer of the shared form store -> no clobbering).
           Job results persist across switches via the parent-owned runners. -->
      <module-workspace
        :key="activeTab"
        :module-id="activeTab"
        :runner="runners[activeTab]"
        v-model:split-percent="splits[activeTab]"
      />
    </div>

    <footer class="console-foot">
      <el-icon class="disclaimer-icon"><WarningFilled /></el-icon>
      <span class="disclaimer">
        {{ t({ zh: '免责声明：粗算过程，仅供学习参考', en: 'Disclaimer: rough estimation — for learning reference only' }) }}
      </span>
      <span class="foot-sep">·</span>
      <el-link
        class="source-link"
        :href="SOURCE_URL"
        target="_blank"
        rel="noopener"
        type="primary"
        :underline="false"
      >
        GitCode
        <el-icon class="source-link-icon"><Link /></el-icon>
      </el-link>
    </footer>
  </div>
</template>

<style scoped>
.console {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.console-tabs {
  flex: 0 0 auto;
  padding: 0 16px;
  background: var(--msm-bg-panel-2);
  border-bottom: 1px solid var(--msm-border);
}

/* pull the tab header flush to the top */
.console-tabs :deep(.el-tabs__header) {
  margin: 0;
}

.console-tabs :deep(.el-tabs__nav-wrap::after) {
  background-color: var(--msm-border);
}

.console-tabs :deep(.el-tabs__item) {
  height: 46px;
  font-size: 14px;
  font-weight: 600;
}

.tab-label {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}

.tab-icon {
  font-size: 16px;
}

.tab-body {
  flex: 1 1 0;
  min-height: 0;
  padding: 14px 16px 16px;
}

.tab-body > * {
  height: 100%;
}

/* Disclaimer banner: a prominent amber-tinted callout (warning-style) under
   the workspace. flex: 0 0 auto keeps it from eating into the workspace
   (tab-body stays flex: 1 1 0). Tint adapts to light/dark via color-mix on
   the theme's amber token over the panel surface. */
.console-foot {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 8px 16px;
  border-left: 3px solid var(--msm-amber);
  border-top: 1px solid var(--msm-border);
  background: color-mix(in srgb, var(--msm-amber) 12%, var(--msm-bg-panel-2));
  font-size: 13px;
  font-weight: 600;
  color: var(--msm-text);
}

.disclaimer-icon {
  color: var(--msm-amber);
  font-size: 16px;
  flex: 0 0 auto;
}

.foot-sep {
  opacity: 0.5;
  font-weight: 400;
}

.source-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  font-weight: 600;
  /* ElLink inherits --el-color-primary (blue light / green dark). */
}

/* Underline on hover confirms the affordance (the Link icon signals it at rest). */
.source-link:hover {
  text-decoration: underline;
}

.source-link-icon {
  font-size: 13px;
}
</style>

<script setup lang="ts">
// Root shell: app bar (brand + history + locale + plugin-contributed actions) over the workspace router-view.
import { computed, reactive, markRaw, defineAsyncComponent } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElIcon, ElButton } from 'element-plus'
import { HomeFilled, Clock, Document } from '@element-plus/icons-vue'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import en from 'element-plus/es/locale/lang/en'
import { useLocale } from './composables/useLocale'
import LocaleSwitcher from './components/LocaleSwitcher.vue'
import ThemeSwitcher from './components/ThemeSwitcher.vue'
import { pluginHost } from './plugins'

const { t, locale } = useLocale()
const router = useRouter()
const route = useRoute()
const elLocale = computed(() => (locale.value === 'zh' ? zhCn : en))

// Plugin-contributed app-bar buttons + global widgets (e.g. the feedback trigger
// + dialog). Async components are built ONCE and markRaw'd so Vue sees a stable
// definition — defining them inline in the template would rebuild the wrapper on
// every render and remount the widget, wiping its internal state.
const widgetVisible = reactive<Record<string, boolean>>({})
const globalWidgets = computed(() =>
  pluginHost
    .getPlugins()
    .filter((p) => p.globalWidget)
    .map((p) => ({ id: p.id, comp: markRaw(defineAsyncComponent(p.globalWidget!.component)) })),
)
const appBarPlugins = computed(() =>
  pluginHost
    .getPlugins()
    .filter((p) => p.appBar)
    .map((p) => ({
      id: p.id,
      label: p.appBar!.label,
      icon: markRaw(defineAsyncComponent(p.appBar!.icon)),
    })),
)
// Plugin-contributed nav buttons (plugins that declare a menuEntry, sorted by
// order). A plugin without menuEntry has its route reachable only by direct URL.
const menuPlugins = computed(() =>
  pluginHost
    .getPlugins()
    .filter((p) => p.menuEntry)
    .sort((a, b) => a.menuEntry!.order - b.menuEntry!.order)
    .map((p) => ({
      id: p.id,
      route: p.menuEntry!.route,
      label: p.menuEntry!.label,
      icon: p.menuEntry!.icon ? markRaw(defineAsyncComponent(p.menuEntry!.icon)) : null,
    })),
)

// Active-nav detection: '/' matches only the exact root; other paths match by prefix.
const isCurrent = (to: string) =>
  to === '/' ? route.path === '/' : route.path.startsWith(to)
</script>

<template>
  <el-config-provider :locale="elLocale">
    <div class="app-shell">
      <header class="app-bar">
        <div class="brand">
          <span class="brand-name">msmodeling</span>
          <span class="brand-sub">{{ t({ zh: '建模仿真控制台', en: 'Modeling Console' }) }}</span>
        </div>
        <div class="bar-actions">
          <el-button text class="link-btn" :class="{ 'is-active': isCurrent('/') }" @click="router.push('/')">
            <el-icon><HomeFilled /></el-icon>
            <span class="label">{{ t({ zh: '主页', en: 'Home' }) }}</span>
          </el-button>
          <el-button text class="link-btn" :class="{ 'is-active': isCurrent('/docs') }" @click="router.push('/docs')">
            <el-icon><Document /></el-icon>
            <span class="label">{{ t({ zh: '使用文档', en: 'Docs' }) }}</span>
          </el-button>
          <el-button
            v-for="p in appBarPlugins"
            :key="p.id"
            text
            class="link-btn"
            @click="widgetVisible[p.id] = true"
          >
            <el-icon><component :is="p.icon" /></el-icon>
            <span class="label">{{ t(p.label) }}</span>
          </el-button>
          <el-button text class="link-btn" :class="{ 'is-active': isCurrent('/history') }" @click="router.push('/history')">
            <el-icon><Clock /></el-icon>
            <span class="label">{{ t({ zh: '历史记录', en: 'History' }) }}</span>
          </el-button>
          <el-button
            v-for="p in menuPlugins"
            :key="p.id"
            text
            class="link-btn"
            :class="{ 'is-active': route.name === p.route }"
            @click="router.push({ name: p.route })"
          >
            <el-icon v-if="p.icon"><component :is="p.icon" /></el-icon>
            <span class="label">{{ t(p.label) }}</span>
          </el-button>
          <LocaleSwitcher />
          <ThemeSwitcher />
        </div>
      </header>
      <main class="app-main">
        <router-view />
      </main>
      <component
        v-for="w in globalWidgets"
        :key="w.id"
        :is="w.comp"
        v-model:visible="widgetVisible[w.id]"
      />
    </div>
  </el-config-provider>
</template>

<style scoped>
.app-shell {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--msm-bg);
}

.app-bar {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 54px;
  background: var(--msm-bg-deep);
  border-bottom: 1px solid var(--msm-border);
}

.brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.brand-name {
  font-weight: 700;
  font-size: 16px;
  letter-spacing: 0.01em;
  color: var(--msm-text);
  font-family: 'Fira Code', monospace;
}

.brand-sub {
  font-size: 13px;
  color: var(--msm-text-muted);
}

.bar-actions {
  display: flex;
  align-items: center;
  gap: 14px;
}

.link-btn {
  color: var(--msm-text-muted);
  transition: color var(--msm-transition-fast) var(--msm-ease-out),
    background var(--msm-transition-fast) var(--msm-ease-out);
  border-radius: var(--msm-radius-sm);
}

.link-btn:hover {
  color: var(--msm-text);
}

.link-btn:active {
  color: var(--msm-accent);
  background: var(--msm-bg-panel-2);
}

.link-btn.is-active {
  color: var(--msm-text);
  font-weight: 600;
}

.link-btn.is-active::after {
  content: '';
  display: block;
  height: 2px;
  margin-top: 2px;
  background: var(--msm-accent);
  border-radius: var(--msm-radius-sm);
}

.link-btn .label {
  margin-left: 6px;
}

.app-main {
  flex: 1 1 0;
  min-height: 0;
  /* Scroll document-flow pages (JobResult/History/JobStatus) whose content
     exceeds the viewport. Console fills height:100% exactly, so it never
     overflows here and keeps its own internal scroll panes. */
  overflow-y: auto;
  overflow-x: hidden;
}
</style>

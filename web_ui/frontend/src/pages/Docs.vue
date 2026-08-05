<script setup lang="ts">
/**
 * Docs page — renders markdown user guides from public/docs/.
 *
 * Layout: fixed sidebar (doc list from index.json + TOC of H2 headings)
 * + scrollable content area.
 *
 * To add a new doc:
 *   1. Put the .md file in public/docs/
 *   2. Add an entry to public/docs/index.json: {"file":"...", "label":"..."}
 */
import { ref, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElSkeleton, ElButton } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import { useLocale } from '@/composables/useLocale'
import { useMarkdown, type DocMeta } from '@/composables/useMarkdown'

const { t } = useLocale()
const router = useRouter()
const { render, loading, error } = useMarkdown()

// --- Doc list from index.json ----------------------------------------------
interface DocEntry { file: string; label: string }

const docs = ref<DocEntry[]>([])
const activeDoc = ref<string | null>(null)
const docMeta = ref<DocMeta | null>(null)
const listError = ref<string | null>(null)

async function fetchDocList() {
  try {
    const resp = await fetch('/docs/index.json')
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const items: DocEntry[] = await resp.json()
    docs.value = items
    if (items.length > 0 && !activeDoc.value) {
      activeDoc.value = items[0].file
    }
  } catch (e: any) {
    listError.value = e.message || 'Failed to load doc list'
  }
}

// --- Load doc --------------------------------------------------------------
// Request id — fast clicks through the sidebar can race, with the earlier
// fetch resolving AFTER a later one and overwriting `docMeta` with stale
// content. Guarding on a monotonically-incrementing id ensures only the most
// recent loadDoc call is allowed to publish its result. See PR-632 #46.
let loadDocId = 0
async function loadDoc(file: string) {
  const myId = ++loadDocId
  activeDoc.value = file
  docMeta.value = null
  try {
    const resp = await fetch(`/docs/${file}`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const raw = await resp.text()
    const result = await render(raw)
    if (myId !== loadDocId) return // superseded — drop stale result
    docMeta.value = result
    await nextTick()
    const area = document.querySelector('.docs-content-area')
    if (area) area.scrollTop = 0
  } catch (e: any) {
    if (myId !== loadDocId) return // superseded — don't stomp the active doc's error state
    error.value = e.message || 'Failed to load document'
  }
}

/** Smooth-scroll to a heading anchor in the content area. */
function scrollToAnchor(anchor: string) {
  const el = document.getElementById(anchor)
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

onMounted(async () => {
  await fetchDocList()
  if (activeDoc.value) {
    loadDoc(activeDoc.value)
  }
})
</script>

<template>
  <div class="docs-page">
    <!-- Sidebar -->
    <aside class="docs-sidebar">
      <!-- Back to home -->
      <div class="sidebar-back">
        <el-button text size="small" @click="router.push('/')">
          <el-icon><ArrowLeft /></el-icon>
          <span style="margin-left: 4px">{{ t({ zh: '返回主页', en: 'Back' }) }}</span>
        </el-button>
      </div>

      <!-- Doc selector -->
      <div class="sidebar-section">
        <div class="sidebar-label">{{ t({ zh: '使用文档', en: 'User Guides' }) }}</div>
        <div v-if="listError" class="sidebar-error">{{ listError }}</div>
        <ul v-else class="doc-nav">
          <li v-for="doc in docs" :key="doc.file" class="doc-nav-li">
            <button
              type="button"
              :class="['doc-nav-item', { active: doc.file === activeDoc }]"
              @click="loadDoc(doc.file)"
            >
              {{ doc.label }}
            </button>
          </li>
        </ul>
      </div>

      <!-- TOC for current doc -->
      <div v-if="docMeta && docMeta.headings.length > 0" class="sidebar-section">
        <div class="sidebar-label">{{ t({ zh: '目录', en: 'On this page' }) }}</div>
        <ul class="toc-nav">
          <li v-for="h in docMeta.headings" :key="h.anchor" class="toc-item">
            <a :href="'#' + h.anchor" @click.prevent="scrollToAnchor(h.anchor)">{{ h.text }}</a>
          </li>
        </ul>
      </div>
    </aside>

    <!-- Content area -->
    <main class="docs-content-area">
      <!-- Loading -->
      <el-skeleton v-if="loading" :rows="12" animated />

      <!-- Error -->
      <div v-else-if="error" class="docs-error">
        {{ error }}
      </div>

      <!-- Rendered content -->
      <div v-else-if="docMeta" class="markdown-body" v-html="docMeta.html" />

      <!-- Empty -->
      <div v-else class="docs-empty">
        {{ t({ zh: '暂无内容', en: 'No content' }) }}
      </div>
    </main>
  </div>
</template>

<style scoped>
.docs-page {
  display: flex;
  height: 100%;
  min-height: 0;
}

/* ---- Sidebar ---- */
.docs-sidebar {
  flex: 0 0 240px;
  overflow-y: auto;
  padding: 20px 16px;
  background: var(--msm-bg-panel);
  border-right: 1px solid var(--msm-border);
}

.sidebar-section {
  margin-bottom: 20px;
}

.sidebar-back {
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--msm-border);
}

.sidebar-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--msm-text-muted);
  margin-bottom: 8px;
}

.sidebar-error {
  font-size: 12px;
  color: var(--el-color-danger);
}

.doc-nav,
.toc-nav {
  list-style: none;
  margin: 0;
  padding: 0;
}

.doc-nav-item {
  display: block;
  width: 100%;
  padding: 6px 10px;
  border: none;
  background: transparent;
  border-radius: 6px;
  font-size: 13px;
  color: var(--msm-text);
  text-align: left;
  cursor: pointer;
  transition: background-color var(--msm-transition-fast) var(--msm-ease-out);
}

.doc-nav-item:hover {
  background: var(--msm-bg-panel-2);
}

.doc-nav-item:focus-visible {
  outline: 2px solid var(--msm-accent);
  outline-offset: -1px;
}

.doc-nav-item.active {
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary);
  font-weight: 600;
}

.toc-item a {
  display: block;
  padding: 4px 10px;
  font-size: 12px;
  color: var(--msm-text-muted);
  text-decoration: none;
  border-radius: 4px;
  transition: color var(--msm-transition-fast) var(--msm-ease-out),
              background-color var(--msm-transition-fast) var(--msm-ease-out);
  line-height: 1.4;
}

.toc-item a:hover {
  color: var(--msm-text);
  background: var(--msm-bg-panel-2);
}

/* ---- Content ---- */
.docs-content-area {
  flex: 1 1 0;
  overflow-y: auto;
  padding: 28px 40px;
  min-height: 60vh;
}

.docs-error {
  color: var(--el-color-danger);
  padding: 24px;
}

.docs-empty {
  color: var(--msm-text-muted);
  padding: 24px;
}

/* ---- Markdown rendering ---- */
.markdown-body {
  max-width: 860px;
  font-size: 14px;
  line-height: 1.75;
  color: var(--msm-text);
}

/* Headings */
.markdown-body :deep(h1) {
  font-size: 28px;
  font-weight: 700;
  margin: 0 0 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--msm-border);
  color: var(--msm-text);
}

.markdown-body :deep(h2) {
  font-size: 20px;
  font-weight: 600;
  margin: 32px 0 14px;
  color: var(--msm-text);
}

.markdown-body :deep(h3) {
  font-size: 16px;
  font-weight: 600;
  margin: 24px 0 10px;
  color: var(--msm-text);
}

/* Paragraphs & lists */
.markdown-body :deep(p) {
  margin: 0 0 14px;
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  margin: 0 0 14px;
  padding-left: 24px;
}

.markdown-body :deep(li) {
  margin-bottom: 4px;
}

/* Inline code */
.markdown-body :deep(code:not(pre code)) {
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--msm-bg-deep);
  font-family: 'Fira Code', monospace;
  font-size: 13px;
  color: var(--el-color-danger);
}

/* Code blocks */
.markdown-body :deep(pre) {
  margin: 0 0 16px;
  padding: 14px 18px;
  border-radius: 8px;
  background: var(--msm-bg-deep);
  overflow-x: auto;
  font-size: 13px;
  line-height: 1.55;
}

.markdown-body :deep(pre code) {
  font-family: 'Fira Code', monospace;
  background: transparent;
  color: var(--msm-text);
  padding: 0;
}

/* Tables */
.markdown-body :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 16px;
  font-size: 13px;
}

.markdown-body :deep(th),
.markdown-body :deep(td) {
  padding: 8px 12px;
  border: 1px solid var(--msm-border);
  text-align: left;
}

.markdown-body :deep(th) {
  background: var(--msm-bg-panel-2);
  font-weight: 600;
}

.markdown-body :deep(tr:nth-child(even)) {
  background: var(--msm-bg-panel);
}

/* Links */
.markdown-body :deep(a) {
  color: var(--el-color-primary);
  text-decoration: none;
}

.markdown-body :deep(a:hover) {
  text-decoration: underline;
}

/* Images */
.markdown-body :deep(img) {
  max-width: 100%;
  border-radius: 6px;
}

/* Blockquotes */
.markdown-body :deep(blockquote) {
  margin: 0 0 16px;
  padding: 10px 16px;
  border-left: 4px solid var(--el-color-primary);
  background: var(--msm-bg-panel);
  border-radius: 0 6px 6px 0;
}

.markdown-body :deep(blockquote p) {
  margin: 0;
}

/* Horizontal rules */
.markdown-body :deep(hr) {
  border: none;
  border-top: 1px solid var(--msm-border);
  margin: 24px 0;
}
</style>

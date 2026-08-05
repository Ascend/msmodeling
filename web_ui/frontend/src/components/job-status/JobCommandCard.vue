<script setup lang="ts">
/**
 * Job command card subcomponent.
 *
 * Run-command display card: CLI command (with $ prefix) + copy button + collapsible
 * raw params JSON (macOS-style code-editor look). Self-contained copy logic
 * (best-effort, toast on failure).
 */
import { useLocale } from '@/composables/useLocale'
import { ElMessage } from 'element-plus'
import { Document, DocumentCopy } from '@element-plus/icons-vue'
import { ref, computed } from 'vue'

interface Props {
  command: string
  /** Per-case CLI commands actually executed by the worker. For single-case
   *  jobs this contains one element equal to `command`; for multi-case jobs
   *  (multi-device / multi-quantize / etc.) it has one command per case. */
  commands?: string[]
  params?: any
}

const props = defineProps<Props>()
const { t } = useLocale()

const copying = ref(false)

/** Effective commands list — falls back to [command] when `commands` is absent. */
const effectiveCommands = computed(() => {
  if (props.commands && props.commands.length > 0) return props.commands
  return props.command ? [props.command] : []
})

const isMultiCase = computed(() => effectiveCommands.value.length > 1)

const handleCopyCommand = async () => {
  if (effectiveCommands.value.length === 0 || copying.value) return
  copying.value = true
  try {
    // Single command → copy it verbatim; multi-case → copy all commands joined
    // with a blank line between cases (easy to paste into a shell one by one).
    const text = effectiveCommands.value.join('\n\n')
    await navigator.clipboard.writeText(text)
    ElMessage.success(
      t({
        zh: isMultiCase.value
          ? `${effectiveCommands.value.length} 条命令已复制到剪贴板`
          : '命令已复制到剪贴板',
        en: isMultiCase.value
          ? `${effectiveCommands.value.length} commands copied to clipboard`
          : 'Command copied to clipboard',
      }),
    )
  } catch (err: any) {
    console.error('Failed to copy command:', err)
    ElMessage.error(t({ zh: '复制失败', en: 'Failed to copy' }))
  } finally {
    copying.value = false
  }
}
</script>

<template>
  <el-card class="status-card command-card">
    <template #header>
      <div class="card-header">
        <div class="header-left">
          <el-icon class="header-icon command-icon">
            <DocumentCopy />
          </el-icon>
          <span class="card-title">{{ t({ zh: '运行命令', en: 'Run Command' }) }}</span>
        </div>
        <el-button size="small" class="copy-btn" :loading="copying" :disabled="copying" @click="handleCopyCommand">
          <el-icon><DocumentCopy /></el-icon>
          {{ t({ zh: '复制', en: 'Copy' }) }}
        </el-button>
      </div>
    </template>
    <div class="command-wrapper">
      <!-- Single case: render as a single command block (original behavior). -->
      <template v-if="!isMultiCase">
        <pre class="command-content">{{ effectiveCommands[0] }}</pre>
      </template>
      <!-- Multi-case: render each case as its own command block with a case
           label. Each block gets the `$` prompt prefix via the existing
           `.command-content::before` rule. -->
      <template v-else>
        <div
          v-for="(cmd, idx) in effectiveCommands"
          :key="idx"
          class="case-block"
        >
          <div class="case-label">
            <el-tag size="small" class="case-tag">
              {{ t({ zh: `Case ${idx + 1}/${effectiveCommands.length}`, en: `Case ${idx + 1}/${effectiveCommands.length}` }) }}
            </el-tag>
          </div>
          <pre class="command-content">{{ cmd }}</pre>
        </div>
      </template>
    </div>
    <el-collapse v-if="params" class="params-collapse">
      <el-collapse-item>
        <template #title>
          <div class="collapse-header">
            <el-icon class="collapse-icon"><Document /></el-icon>
            <span>{{ t({ zh: '原始参数 (JSON)', en: 'Raw Params (JSON)' }) }}</span>
            <el-tag size="small" class="params-tag">JSON</el-tag>
          </div>
        </template>
        <div class="code-editor">
          <div class="editor-header">
            <span class="editor-title">params.json</span>
            <div class="editor-actions">
              <div class="editor-dot red"></div>
              <div class="editor-dot yellow"></div>
              <div class="editor-dot green"></div>
            </div>
          </div>
          <pre>{{ JSON.stringify(params, null, 2) }}</pre>
        </div>
      </el-collapse-item>
    </el-collapse>
  </el-card>
</template>

<style scoped>
.status-card.command-card {
  border: 1px solid var(--msm-border);
  background: var(--msm-bg-deep);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.status-card {
  text-align: left;
  border-radius: 8px;
  overflow: hidden;
  transition: box-shadow 0.2s ease;
}

.status-card:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
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
  border-radius: 8px;
  flex-shrink: 0;
}

.command-icon {
  background: rgba(22, 163, 74, 0.1);
  color: var(--msm-green);
}

.card-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--msm-text);
}

.copy-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}

.command-wrapper {
  position: relative;
  margin: 0 -12px -12px -12px;
}

.case-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 12px;
}

.case-block:last-child {
  margin-bottom: 0;
}

.case-label {
  padding: 0 16px;
}

.case-tag {
  font-family: 'Fira Code', monospace;
  letter-spacing: 0.3px;
}

.command-content {
  margin: 0;
  padding: 16px 16px 16px 32px;
  background: var(--msm-bg-deep);
  border: 1px solid var(--msm-border);
  border-radius: 0;
  font-family: 'Fira Code', monospace;
  font-size: 13px;
  line-height: 1.6;
  overflow-x: auto;
  white-space: pre;
  word-wrap: normal;
  color: var(--msm-text);
  position: relative;
}

.command-content::before {
  content: '$';
  position: absolute;
  left: 16px;
  color: var(--msm-green);
  font-weight: 600;
  opacity: 0.8;
}

.params-collapse {
  margin-top: 16px;
  border-top: 1px solid var(--msm-border);
  padding-top: 16px;
}

.params-collapse :deep(.el-collapse-item) {
  border: none;
  background: transparent;
}

.params-collapse :deep(.el-collapse-item__header) {
  background: transparent;
  border-radius: 8px;
  padding: 0;
  margin-bottom: 12px;
  height: auto;
  line-height: normal;
  border: none;
}

.params-collapse :deep(.el-collapse-item__wrap) {
  background: transparent;
  border: none;
}

.params-collapse :deep(.el-collapse-item__content) {
  padding: 0;
}

.collapse-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  background: var(--el-fill-color-light);
  border-radius: 8px;
  transition: all 0.2s ease;
  cursor: pointer;
  border: 1px solid var(--msm-border);
}

.collapse-header:hover {
  background: var(--el-border-color);
  border-color: var(--msm-text-muted);
}

.collapse-icon {
  font-size: 16px;
  color: var(--msm-text-muted);
  transition: color 0.2s ease;
}

.collapse-header:hover .collapse-icon {
  color: var(--msm-text);
}

.params-tag {
  margin-left: auto;
  font-family: 'Fira Code', monospace;
  font-size: 11px;
  letter-spacing: 0.5px;
  opacity: 0.8;
}

.code-editor {
  border-radius: 8px;
  overflow: hidden;
  background: var(--msm-bg-deep);
  border: 1px solid var(--msm-border);
  box-shadow: var(--msm-shadow);
  margin-top: 8px;
}

.editor-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: var(--msm-bg-panel-2);
  border-bottom: 1px solid var(--msm-border);
}

.editor-title {
  font-family: 'Fira Code', monospace;
  font-size: 12px;
  color: var(--msm-text-muted);
  font-weight: 500;
}

.editor-actions {
  display: flex;
  gap: 6px;
}

.editor-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  transition: opacity 0.2s ease;
}

.editor-dot:hover { opacity: 0.7; }
.editor-dot.red { background: #ff5f56; }
.editor-dot.yellow { background: #ffbd2e; }
.editor-dot.green { background: #27c93f; }

.code-editor pre {
  margin: 0;
  padding: 16px;
  background: var(--msm-bg-deep);
  font-family: 'Fira Code', monospace;
  font-size: 13px;
  line-height: 1.6;
  color: var(--msm-text);
  white-space: pre;
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
  border-radius: 0 0 8px 8px;
}

.code-editor pre::-webkit-scrollbar { width: 8px; height: 8px; }
.code-editor pre::-webkit-scrollbar-track { background: var(--msm-bg-panel-2); border-radius: 4px; }
.code-editor pre::-webkit-scrollbar-thumb { background: var(--msm-border-strong); border-radius: 4px; }
.code-editor pre::-webkit-scrollbar-thumb:hover { background: var(--msm-text-muted); }
</style>

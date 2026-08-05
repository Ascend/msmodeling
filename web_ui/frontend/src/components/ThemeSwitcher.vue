<script setup lang="ts">
/**
 * Theme switcher. Toggles light/dark via useTheme (mirrors LocaleSwitcher).
 * Shows a sun icon in light mode (click → go dark) and a moon icon in dark.
 */
import { ElButton, ElIcon } from 'element-plus'
import { Sunny, Moon } from '@element-plus/icons-vue'
import { useTheme } from '../composables/useTheme'
import { useLocale } from '../composables/useLocale'

const { theme, toggle } = useTheme()
const { t } = useLocale()
</script>

<template>
  <el-button
    text
    :aria-label="theme === 'dark' ? t({ zh: '切换到浅色', en: 'Switch to light' }) : t({ zh: '切换到深色', en: 'Switch to dark' })"
    :title="theme === 'dark' ? t({ zh: '切换到浅色', en: 'Switch to light' }) : t({ zh: '切换到深色', en: 'Switch to dark' })"
    @click="toggle"
  >
    <el-icon><Sunny v-if="theme === 'dark'" /><Moon v-else /></el-icon>
  </el-button>
</template>

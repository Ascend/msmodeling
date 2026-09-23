<script setup lang="ts">
/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

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

<script setup lang="ts">
/**
 * Locale switcher component. Toggle between ZH/EN.
 */
import { useLocale } from '../composables/useLocale'

const { locale, setLocale } = useLocale()

function toggleLocale() {
  setLocale(locale.value === 'zh' ? 'en' : 'zh')
}
</script>

<template>
  <el-button text @click="toggleLocale">
    {{ locale === 'zh' ? '中' : 'EN' }}
  </el-button>
</template>

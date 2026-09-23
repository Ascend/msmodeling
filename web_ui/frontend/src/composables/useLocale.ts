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
 * Locale management. Reactive locale state with localStorage persistence.
 * Supports bilingual UI (zh/en) per Constitution Principle II v2.2.0.
 */
import { ref, computed, watch } from 'vue'

export type LocalizedText = string | Record<string, string>

const LOCALE_STORAGE_KEY = 'msmodeling-locale'
const DEFAULT_LOCALE = 'zh'

/** Reactive locale state (initialize eagerly — ref takes a value, not a factory). */
const _storedLocale =
  typeof localStorage !== 'undefined'
    ? (localStorage.getItem(LOCALE_STORAGE_KEY) as 'zh' | 'en' | null)
    : null
const locale = ref<'zh' | 'en'>(_storedLocale ?? DEFAULT_LOCALE)

/** Persist locale changes to localStorage */
watch(locale, (newLocale) => {
  localStorage.setItem(LOCALE_STORAGE_KEY, newLocale)
}, { immediate: false })

/**
 * Resolve localized text to a string.
 * - Plain string: returned as-is (locale-neutral or zh fallback)
 * - Record<string, string>: lookup by current locale, fallback to 'zh' if missing
 */
export function t(localized: LocalizedText): string {
  if (typeof localized === 'string') {
    return localized
  }
  return localized[locale.value] || localized['zh'] || ''
}

/** Current locale value (read-only) */
export function useLocale() {
  return {
    locale: computed(() => locale.value),
    setLocale: (newLocale: 'zh' | 'en') => { locale.value = newLocale },
    t,
  }
}

/**
 * Theme management — reactive light/dark state with localStorage persistence.
 *
 * Mirrors useLocale. The CSS token system (styles/theme.css) already defines
 * both ramps: `:root` is light, `html.dark` opts into dark. Element Plus's dark
 * css-vars are imported in main.ts. This composable is the missing toggle: it
 * adds/removes the `html.dark` class on <html> and persists the choice.
 *
 * Initial value: stored choice → OS `prefers-color-scheme` → light. Applied at
 * import time (before mount) so the first paint is already correct.
 */
import { ref, watch } from 'vue'

export type Theme = 'light' | 'dark'

const STORAGE_KEY = 'msmodeling-theme'

function detectInitial(): Theme {
  const stored =
    typeof localStorage !== 'undefined'
      ? (localStorage.getItem(STORAGE_KEY) as Theme | null)
      : null
  if (stored === 'light' || stored === 'dark') return stored
  // No stored choice — honour the OS preference.
  if (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-color-scheme: dark)').matches
  ) {
    return 'dark'
  }
  return 'light'
}

const theme = ref<Theme>(detectInitial())

/** Apply a theme by toggling the `dark` class on <html> (the theme.css hook). */
function apply(next: Theme): void {
  if (typeof document === 'undefined') return
  const cls = document.documentElement.classList
  if (next === 'dark') cls.add('dark')
  else cls.remove('dark')
}

// Apply on import (before the app mounts) so there's no flash of the wrong theme.
apply(theme.value)

watch(theme, (next) => {
  apply(next)
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, next)
  }
})

export function useTheme() {
  return {
    /** Current theme (reactive — re-renders dependents on toggle). */
    theme,
    setTheme: (next: Theme) => {
      theme.value = next
    },
    toggle: () => {
      theme.value = theme.value === 'dark' ? 'light' : 'dark'
    },
  }
}

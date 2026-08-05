import { computed } from 'vue'
import { useTheme } from './useTheme'

/**
 * ECharts theme helper (theme-aware).
 *
 * Chart options are plain JS objects — they cannot read CSS custom properties
 * directly — so the categorical + axis palette lives here and must stay in
 * sync with the palette comment at the bottom of styles/theme.css.
 *
 * Both ramps are colorblind-aware. Light: 600-level hues on white. Dark: lifted
 * 400/500 hues on slate-800 panels. baseOption + axis tokens are reactive to
 * useTheme so charts re-render when the user toggles light/dark.
 */

// ── LIGHT palette (kept exported for any direct importer; AA on white) ──
export const CHART_CATEGORY = [
  '#1D4ED8', // blue-700  (primary)
  '#16A34A', // green-600
  '#D97706', // amber-600
  '#DC2626', // red-600
  '#7C3AED', // violet-600
  '#0891B2', // cyan-600
  '#DB2777', // pink-600
  '#475569', // slate-600
] as const
export const CHART_AXIS_TEXT = '#475569' // slate-600
export const CHART_SPLIT_LINE = '#E2E8F0' // slate-200
export const CHART_TOOLTIP_BG = '#FFFFFF'
export const CHART_TOOLTIP_BORDER = '#E2E8F0'
export const CHART_TOOLTIP_TEXT = '#1E293B'

// ── DARK palette (lifted for readability on slate-800 panels) ──
const DARK_CATEGORY = [
  '#60A5FA', // blue-400
  '#4ADE80', // green-400
  '#FBBF24', // amber-400
  '#F87171', // red-400
  '#A78BFA', // violet-400
  '#22D3EE', // cyan-400
  '#F472B6', // pink-400
  '#94A3B8', // slate-400
] as const
const DARK_AXIS_TEXT = '#94A3B8' // slate-400
const DARK_SPLIT_LINE = '#334155' // slate-700
const DARK_TOOLTIP_BG = '#18233A' // msm-bg-panel-2
const DARK_TOOLTIP_BORDER = '#334155'
const DARK_TOOLTIP_TEXT = '#E2E8F0'

/**
 * Shared ECharts defaults — `color`, `textStyle`, `legend`, `tooltip`.
 * Spread `baseOption.value` into a chart option and override per-chart.
 * Reactive: swaps to the dark palette when useTheme().theme === 'dark'.
 */
export function useChartTheme() {
  const { theme } = useTheme()
  const isDark = computed(() => theme.value === 'dark')

  const palette = computed(() => (isDark.value ? DARK_CATEGORY : CHART_CATEGORY))
  const axisText = computed(() => (isDark.value ? DARK_AXIS_TEXT : CHART_AXIS_TEXT))
  const splitLine = computed(() => (isDark.value ? DARK_SPLIT_LINE : CHART_SPLIT_LINE))
  const tooltipBg = computed(() => (isDark.value ? DARK_TOOLTIP_BG : CHART_TOOLTIP_BG))
  const tooltipBorder = computed(() => (isDark.value ? DARK_TOOLTIP_BORDER : CHART_TOOLTIP_BORDER))
  const tooltipText = computed(() => (isDark.value ? DARK_TOOLTIP_TEXT : CHART_TOOLTIP_TEXT))

  const baseOption = computed(() => ({
    color: [...palette.value],
    textStyle: {
      color: axisText.value,
      fontFamily: "'Fira Sans', system-ui, sans-serif",
    },
    legend: { textStyle: { color: axisText.value } },
    tooltip: {
      backgroundColor: tooltipBg.value,
      borderColor: tooltipBorder.value,
      borderWidth: 1,
      textStyle: { color: tooltipText.value },
      extraCssText: isDark.value
        ? 'box-shadow: 0 0 0 1px #334155; border-radius: 8px;'
        : 'box-shadow: 0 4px 12px rgba(15,23,42,0.10); border-radius: 8px;',
    },
  }))

  return {
    /** Theme-reactive ECharts defaults (color/textStyle/legend/tooltip). */
    baseOption,
    /** Reactive categorical palette — index as `category.value[i]`. */
    category: palette,
    /** Reactive axis label / legend text colour — read as `axisText.value`. */
    axisText,
    /** Reactive axis + split gridline colour — read as `splitLine.value`. */
    splitLine,
    /** Reactive tooltip surface colours — read as `.value`. */
    tooltipBg,
    tooltipBorder,
    tooltipText,
  }
}

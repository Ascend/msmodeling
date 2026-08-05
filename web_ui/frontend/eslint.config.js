// ESLint flat config for the web console frontend (Vue 3 + TypeScript).
// Constitution Principle V: quality gates. Run via `npm run lint`.
import js from '@eslint/js'
import vue from 'eslint-plugin-vue'
import tseslint from 'typescript-eslint'
import vueTsEslint from '@vue/eslint-config-typescript'

export default [
  {
    ignores: ['dist/**', 'node_modules/**', '*.config.ts', '*.config.js'],
  },
  js.configs.recommended,
  ...vue.configs['flat/recommended'],
  ...tseslint.configs.recommended,
  ...vueTsEslint.configs.recommended,

  {
    rules: {
      // Keep the schema-driven form renderer loose about explicit any in field
      // shapes; tighten in Phase 9 once config types are stable.
      '@typescript-eslint/no-explicit-any': 'warn',
      'vue/multi-word-component-names': 'off',
    },
  },
]

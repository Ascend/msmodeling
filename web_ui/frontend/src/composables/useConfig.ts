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
 * Config loader. Form schemas are fetched from the backend API at runtime.
 * Backend registry (ModuleSpec + UIProps) is the SSOT; JSONs are generated
 * at backend startup and served via /api/modules/{id}/form-schema.
 */
import { computed, reactive } from 'vue'
import { api } from '@/services/api'

export type LocalizedText = string | Record<string, string>

export interface FormSchemaEnvelope {
  moduleId: string
  title: LocalizedText
  runner: string
  version: string
  fields: Array<Record<string, any>>
  optionSourceRegistry?: Record<string, any>
  formValidation?: Array<Record<string, any>>
  groups?: Array<{ label: LocalizedText; defaultCollapsed?: boolean; description?: LocalizedText }>
  validators?: Record<string, any>
  schema_hash?: string
}

const schemaCache = reactive<Record<string, FormSchemaEnvelope>>({})
const loadingModules = reactive<Set<string>>(new Set())
const loadErrors = reactive<Record<string, string>>({})

async function loadFormSchema(moduleId: string): Promise<FormSchemaEnvelope | null> {
  if (schemaCache[moduleId]) return schemaCache[moduleId]
  if (loadingModules.has(moduleId)) return null

  loadingModules.add(moduleId)
  delete loadErrors[moduleId]

  try {
    const schema = await api.getFormSchema(moduleId)
    schemaCache[moduleId] = schema
    return schema
  } catch (err: any) {
    loadErrors[moduleId] = err?.message || 'Failed to load form schema'
    return null
  } finally {
    loadingModules.delete(moduleId)
  }
}

export function useFormConfigs() {
  const moduleIds = computed(() => Object.keys(schemaCache))
  const getForm = (moduleId: string): FormSchemaEnvelope | undefined => schemaCache[moduleId]
  const isLoading = (moduleId: string): boolean => loadingModules.has(moduleId)
  const getLoadError = (moduleId: string): string | undefined => loadErrors[moduleId]

  return { moduleIds, getForm, loadFormSchema, isLoading, getLoadError }
}

export const useConfig = useFormConfigs

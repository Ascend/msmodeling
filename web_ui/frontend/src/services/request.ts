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

import axios, { type AxiosInstance } from 'axios'

/**
 * Request layer (guide §1.2 — the axios instance + interceptors). This is the
 * ONLY place that touches axios directly: swapping the HTTP library changes
 * only this file. Business functions live in ``api.ts`` and call ``apiClient``.
 *
 * Error responses carrying ``{ detail, fieldErrors }`` (returned ONLY by
 * ``POST /api/device-config`` — ``POST /api/jobs`` does NOT field-validate)
 * are surfaced as a typed ``ApiError`` so callers can map ``fieldErrors`` onto
 * ``el-form-item``s.
 */
export interface FieldErrorBody {
  detail: string
  fieldErrors?: Record<string, string>
}

export class ApiError extends Error {
  fieldErrors?: Record<string, string>
  status: number
  constructor(message: string, status: number, fieldErrors?: Record<string, string>) {
    super(message)
    this.status = status
    this.fieldErrors = fieldErrors
  }
}

const baseURL = import.meta.env.VITE_API_BASE_URL ?? ''

export const apiClient: AxiosInstance = axios.create({
  baseURL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000,
})

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const body = error?.response?.data as FieldErrorBody | undefined
    const status = error?.response?.status ?? 0
    const detail = body?.detail ?? error?.message ?? 'Request failed'
    throw new ApiError(detail, status, body?.fieldErrors)
  },
)

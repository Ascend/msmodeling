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
 * Client-side CSV export util.
 *
 * Shared by History / result tables / Telemetry "Export CSV"
 * buttons. Reuses the Blob + programmatic `<a>` download pattern (same as
 * ChromeTraceDownloads). Pure client-side — exports the currently-loaded rows.
 */

export interface CsvColumn {
  /** Row object key to read. */
  key: string
  /** Column header label (already i18n-resolved by the caller). */
  label: string
}

/** RFC-4180 escape: wrap in quotes if the cell has comma/quote/newline; double internal quotes. */
function escapeCell(val: unknown): string {
  if (val == null) return ''
  let s = typeof val === 'object' ? JSON.stringify(val) : String(val)
  // CSV formula injection mitigation: a cell whose first non-whitespace char is
  // = + - @ is executed as a formula by Excel/LibreOffice. Prefix a single quote
  // to neutralize it BEFORE RFC-4180 quoting so the guard char survives.
  if (/^\s*[=+\-@]/.test(s)) {
    s = `'${s}`
  }
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

/** Build a CSV string (header row + body) from rows + a column spec. */
export function rowsToCsv(rows: Record<string, any>[], columns: CsvColumn[]): string {
  const head = columns.map((c) => escapeCell(c.label)).join(',')
  const body = rows.map((r) => columns.map((c) => escapeCell(r[c.key])).join(',')).join('\n')
  // Leading BOM so Excel reads UTF-8 (zh labels) correctly.
  return '﻿' + head + '\n' + body
}

/** Trigger a client-side download of `rows` as `filename.csv`. */
export function exportCsv(filename: string, rows: Record<string, any>[], columns: CsvColumn[]): void {
  const csv = rowsToCsv(rows, columns)
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename.endsWith('.csv') ? filename : `${filename}.csv`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

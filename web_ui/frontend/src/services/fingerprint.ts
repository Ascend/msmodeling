/**
 * Browser fingerprint — a stable per-browser id for telemetry user counting.
 *
 * Generation is abstracted behind a FingerprintProvider so the algorithm can be
 * upgraded later (more signals, different hash, or a library like FingerprintJS)
 * without touching callers — they depend only on getFingerprint(). The default
 * provider is a zero-dependency canvas+navigator+screen+timezone hash, cached in
 * localStorage so it is stable across reloads.
 */

export interface FingerprintProvider {
  /** Compute a stable per-browser fingerprint (hex string). */
  compute(): string
}

const STORAGE_KEY = 'msm_fp'
let _cached: string | null = null

// ---- default provider: canvas + navigator/screen/timezone signals → FNV-1a ----

function collectSignals(): string[] {
  const nav = navigator
  const scr = screen
  const sig: string[] = []
  sig.push(String(nav.userAgent || ''))
  sig.push(String(nav.platform || ''))
  sig.push(String(nav.language || ''))
  sig.push(String((nav.languages || []).join(',')))
  sig.push(String(nav.hardwareConcurrency || ''))
  sig.push(String((nav as unknown as { deviceMemory?: number }).deviceMemory || ''))
  sig.push(String(nav.maxTouchPoints || 0))
  sig.push(`${scr.width}x${scr.height}x${scr.colorDepth}x${window.devicePixelRatio || 1}`)
  try {
    sig.push(String(Intl.DateTimeFormat().resolvedOptions().timeZone || ''))
  } catch { /* ignore */ }
  sig.push(String(new Date().getTimezoneOffset()))
  sig.push(canvasSignature())
  sig.push(webglSignature())
  sig.push(mathSignature())
  return sig
}

function canvasSignature(): string {
  try {
    const canvas = document.createElement('canvas')
    canvas.width = 220
    canvas.height = 30
    const ctx = canvas.getContext('2d')
    if (!ctx) return ''
    ctx.textBaseline = 'top'
    ctx.font = "14px 'Arial'"
    ctx.fillStyle = '#f60'
    ctx.fillRect(0, 0, 100, 20)
    ctx.fillStyle = '#069'
    ctx.fillText('msmodeling·fingerprint', 2, 2)
    ctx.fillStyle = 'rgba(102,204,0,0.7)'
    ctx.fillText('msmodeling·fingerprint', 4, 4)
    return canvas.toDataURL()
  } catch {
    return ''
  }
}

/** WebGL vendor/renderer + key params — GPU/driver fingerprint (high entropy). */
function webglSignature(): string {
  try {
    const canvas = document.createElement('canvas')
    const gl = (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')) as WebGLRenderingContext | null
    if (!gl) return ''
    const dbg = gl.getExtension('WEBGL_debug_renderer_info')
    const vendor = dbg ? String(gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) || '') : ''
    const renderer = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || '') : ''
    const maxVp = gl.getParameter(gl.MAX_VIEWPORT_DIMS)
    const params = [
      String(gl.getParameter(gl.VENDOR) || ''),
      String(gl.getParameter(gl.RENDERER) || ''),
      String(gl.getParameter(gl.VERSION) || ''),
      String(gl.getParameter(gl.SHADING_LANGUAGE_VERSION) || ''),
      String(gl.getParameter(gl.MAX_TEXTURE_SIZE) || ''),
      maxVp ? Array.from(maxVp as Int32Array).join('x') : '',
    ]
    return [vendor, renderer, ...params].join('|')
  } catch {
    return ''
  }
}

/** Math function float results — differ across JS engines/CPUs (FingerprintJS technique). */
function mathSignature(): string {
  return [
    Math.acos(0.123),
    Math.acosh(1e308),
    Math.atanh(0.5),
    Math.cbrt(100),
    Math.cosh(10),
    Math.expm1(1),
    Math.log1p(1),
    Math.sinh(1),
    Math.tan(-1e300),
    Math.tanh(0.5),
  ].join(',')
}

/** FNV-1a 32-bit hash → 8-char hex (sync, no crypto dependency). */
function hashFingerprint(signals: string[]): string {
  let h = 0x811c9dc5
  const s = signals.join('||')
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0
  }
  return h.toString(16).padStart(8, '0')
}

class CanvasSignalFingerprint implements FingerprintProvider {
  compute(): string {
    return hashFingerprint(collectSignals())
  }
}

let provider: FingerprintProvider = new CanvasSignalFingerprint()

/** Return the per-browser fingerprint (cached in localStorage; stable across reloads). */
export function getFingerprint(): string {
  if (_cached) return _cached
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      _cached = stored
      return stored
    }
  } catch { /* localStorage unavailable */ }
  const fp = provider.compute()
  _cached = fp
  try {
    localStorage.setItem(STORAGE_KEY, fp)
  } catch { /* ignore */ }
  return fp
}

/** Swap the fingerprint provider (clears cache so the new value is computed). */
export function setFingerprintProvider(p: FingerprintProvider) {
  provider = p
  _cached = null
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch { /* ignore */ }
}

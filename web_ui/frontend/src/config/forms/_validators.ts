/**
 * Shared validator implementations + option constants, co-located with the form
 * configs. Each form config (forms/*.ts) imports the validators/options it uses
 * and lists the validators in its `validators` map; the engine
 * (useFormValidation) resolves a field's `validator` rule by name against it.
 *
 * Signature: ({ value, form, field }) => boolean | string | Promise<boolean | string>
 *   - true   => valid
 *   - false  => invalid (use the rule's `message`)
 *   - string => invalid (the string IS the error message)
 *
 * Frontend-only (live feedback). The backend does NOT run these — the
 * Runner validates internally (Principle I).
 */

export interface ValidatorContext {
  value: any
  form: Record<string, any>
  field: any
}

export type ValidatorFn = (ctx: ValidatorContext) => boolean | string | Promise<boolean | string>

// === Shared option/enum constants (DRY — authoritative per the CLI doc) =======

const v = (s: string) => ({ value: s, label: s })

/** quantize-{linear,backbone-linear}-action — 9 values (doc §2.1.3 / §5.2). */
export const QUANTIZE_LINEAR_OPTIONS = [
  v('DISABLED'), v('W8A16_STATIC'), v('W8A8_STATIC'), v('W4A8_STATIC'),
  v('W8A16_DYNAMIC'), v('W8A8_DYNAMIC'), v('W4A8_DYNAMIC'), v('FP8'), v('MXFP4'),
]

/** quantize-attention-action — 3 values (doc §2.1.3). */
export const QUANTIZE_ATTENTION_OPTIONS = [v('DISABLED'), v('INT8'), v('FP8')]

/** remote-source (doc §2.1.7). */
export const REMOTE_SOURCE_OPTIONS = [v('huggingface'), v('modelscope')]

/** log-level: cli/utils.py LOG_LEVELS = 5 levels (doc §5.4). */
export const CLI_LOG_LEVEL_OPTIONS = [
  v('debug'), v('info'), v('warning'), v('error'), v('critical'),
]

/** log-level: serving_cast LOG_LEVELS = 6 levels (adds 'fatal') (doc §5.4). */
export const SERVING_LOG_LEVEL_OPTIONS = [
  v('debug'), v('info'), v('warning'), v('error'), v('critical'), v('fatal'),
]

// === Field-level validators ===================================================

/** /^[a-zA-Z0-9_/.-]+$/ and length ≤ 256 (model_id). */
export const stringValid = ({ value }: ValidatorContext): boolean | string => {
  if (typeof value !== 'string') return false
  const regex = /^[a-zA-Z0-9_/.-]+$/
  if (!regex.test(value)) return false
  if (value.length > 256) return false
  return true
}

/** "start,end" comma-separated INTEGERS with end ≥ start (cache ranges, video).
 * Rejects blanks, decimals and Infinity — `Number('1.5')` / `Number('Infinity')`
 * are not NaN, so an isNaN check alone would pass them through to integer-only
 * CLI args. */
export const intRangeOrdered = ({ value }: ValidatorContext): boolean | string => {
  if (typeof value !== 'string') return false
  const parts = value.split(',').map((s) => s.trim())
  if (parts.length !== 2) return false
  if (parts.some((p) => p === '')) return false
  const [start, end] = parts.map(Number)
  if (!Number.isInteger(start) || !Number.isInteger(end)) return false
  if (start > end) return false
  return true
}

/** form.world_size % value === 0 (ulysses_size, video_generate). */
export const divisibleBy = ({ value, form }: ValidatorContext): boolean | string => {
  const a = Number(value)
  const b = Number(form.world_size)
  if (isNaN(a) || isNaN(b) || b === 0) return false
  return b % a === 0
}

/** value ∈ [0, 1) (prefix_cache_hit_rate). */
export const prefixCacheRate = ({ value }: ValidatorContext): boolean | string => {
  const num = Number(value)
  if (isNaN(num)) return false
  return num >= 0 && num < 1
}

/** 1–2 integers with min ≤ max (batch_range). Accepts an array OR a "start,end"
 * string — the field is free-text, not a fixed multi-select. Optional: null/empty passes. */
export const batchRange = ({ value }: ValidatorContext): boolean | string => {
  if (value === null || value === undefined || value === '') return true
  const arr = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(',').map((s) => s.trim()).filter(Boolean)
      : []
  if (arr.length < 1 || arr.length > 2) return false
  const nums = arr.map(Number)
  // Integer check, not merely isNaN: Number('1.5') is a valid number, so the
  // old isNaN check let fractional batch sizes through to a CLI that only
  // accepts integers.
  if (nums.some((n) => !Number.isInteger(n))) return false
  if (nums.length === 2 && nums[0] > nums[1]) return false
  return true
}

/** value ≤ num_devices. Accepts a number, an array, or a comma/space-separated
 * string (e.g. "1,2,4") — the free-text tp_sizes/ep_sizes/moe_dp_sizes fields
 * pass a string, while tp_size/vision_tp_size pass a scalar number. */
export const lteNumDevices = ({ value, form }: ValidatorContext): boolean | string => {
  const numDevices = Number(form.num_devices) || 0
  const arr = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(/[,\s]+/).filter(Boolean)
      : [value]
  return arr.every((x) => Number(x) <= numDevices)
}

/** value > 0 or 'inf' (ttft/tpot limits). Optional: null/empty passes (unset).
 * Accepts a scalar, an array, or a comma/space-separated string (e.g. "200,500")
 * so the free-text multi-value ttft_limits/tpot_limits fields are validated per
 * item — every token must be a strict positive number or 'inf'. */
export const positiveOrInf = ({ value }: ValidatorContext): boolean | string => {
  if (value === null || value === undefined || value === '') return true
  const parts = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(/[,\s]+/).filter(Boolean)
      : [value]
  return parts.every((p) => {
    if (p === 'inf' || p === Infinity) return true
    const num = Number(p)
    return !isNaN(num) && num > 0
  })
}

/** null/empty allowed, else a positive integer (optional parallelism knobs). */
export const positiveIntegerIfProvided = ({ value }: ValidatorContext): boolean | string => {
  if (value === null || value === undefined || value === '') return true
  const n = Number(value)
  return Number.isInteger(n) && n > 0
}

/** num_devices % value === 0 (vision_tp_size, text_generate). */
export const dividesNumDevices = ({ value, form }: ValidatorContext): boolean | string => {
  const numDevices = Number(form.num_devices)
  const x = Number(value)
  if (isNaN(numDevices) || isNaN(x) || x === 0) return false
  return numDevices % x === 0
}

/** every element of mtp_acceptance_rate is a finite positive number. Accepts
 * an array or a comma/space-separated string (the throughput form stores it as
 * free text like "0.8, 0.6, 0.4, 0.2"). Empty passes — the field's `required`
 * rule handles presence. */
export const mtpAcceptanceRatesPositive = ({ value }: ValidatorContext): boolean | string => {
  const arr = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(/[,\s]+/).filter(Boolean)
      : []
  if (arr.length === 0) return true
  return arr.every((r) => Number.isFinite(Number(r)) && Number(r) > 0)
}

// === Form-level validators (read the whole form) ==============================

/** text_generate §2.2.1: tp_size × dp_size × pp_size == num_devices (dp null→derived). */
export const productEqNumDevices = ({ form }: ValidatorContext): boolean | string => {
  const numDevices = Number(form.num_devices)
  if (!numDevices || isNaN(numDevices)) return true
  const tp = Number(form.tp_size)
  if (!tp || isNaN(tp)) return true
  const pp = Number(form.pp_size) || 1
  const dp = form.dp_size === null || form.dp_size === undefined || form.dp_size === ''
    ? Math.floor(numDevices / (tp * pp))
    : Number(form.dp_size)
  return tp * dp * pp === numDevices ? true : `tp_size(${tp}) × dp_size(${dp}) × pp_size(${pp}) must equal num_devices(${numDevices})`
}

/** text_generate §2.2.1: moe_tp_size × moe_dp_size × ep_size == num_devices. */
export const moeProductEqNumDevices = ({ form }: ValidatorContext): boolean | string => {
  const numDevices = Number(form.num_devices)
  if (!numDevices || isNaN(numDevices)) return true
  const ep = Number(form.ep_size) || 1
  const moeDp = form.moe_dp_size === null || form.moe_dp_size === undefined || form.moe_dp_size === ''
    ? 1
    : Number(form.moe_dp_size)
  const moeTp = form.moe_tp_size === null || form.moe_tp_size === undefined || form.moe_tp_size === ''
    ? Math.floor(numDevices / (ep * moeDp))
    : Number(form.moe_tp_size)
  return moeTp * moeDp * ep === numDevices
    ? true
    : `moe_tp(${moeTp}) × moe_dp(${moeDp}) × ep(${ep}) must equal num_devices(${numDevices})`
}

/** text_generate §2.2.1: per-layer TP×DP==num_devices for o_proj/mlp/lmhead. */
export const perLayerProductEqNumDevices = ({ form }: ValidatorContext): boolean | string => {
  const numDevices = Number(form.num_devices)
  if (!numDevices || isNaN(numDevices)) return true
  const groups: Array<[string, string, string]> = [
    ['o_proj_tp_size', 'o_proj_dp_size', 'o_proj'],
    ['mlp_tp_size', 'mlp_dp_size', 'mlp'],
    ['lmhead_tp_size', 'lmhead_dp_size', 'lmhead'],
  ]
  for (const [tpField, dpField, name] of groups) {
    const tpRaw = form[tpField]
    const dpRaw = form[dpField]
    if ((tpRaw === null || tpRaw === undefined || tpRaw === '') &&
      (dpRaw === null || dpRaw === undefined || dpRaw === '')) continue
    const tp = (tpRaw === null || tpRaw === undefined || tpRaw === '') ? 1 : Number(tpRaw)
    const dp = (dpRaw === null || dpRaw === undefined || dpRaw === '') ? Math.floor(numDevices / tp) : Number(dpRaw)
    if (tp * dp !== numDevices) return `${name}: ${tpField}(${tp}) × ${dpField}(${dp}) must equal num_devices(${numDevices})`
  }
  return true
}

/** text rel 4 / throughput cross 4: floor(len × (1 - prefix_cache_hit_rate)) ≥ 1. */
export const effectiveLenGe1 = ({ form }: ValidatorContext): boolean | string => {
  const len = Number(form.query_length ?? form.input_length)
  if (!len || isNaN(len)) return true
  const rate = Number(form.prefix_cache_hit_rate) || 0
  const eff = Math.floor(len * (1 - rate))
  return eff >= 1 ? true : `effective length must be ≥ 1 (got ${eff}); lower prefix_cache_hit_rate or raise length`
}

/** throughput §3.3 rel 3: at least one (tp, ep, moe_dp) combo is valid under num_devices. */
export const validParallelCombo = ({ form }: ValidatorContext): boolean | string => {
  const numDevices = Number(form.num_devices) || 0
  if (numDevices <= 0) return true
  const tpSizes = Array.isArray(form.tp_sizes) ? form.tp_sizes : []
  const epSizes = Array.isArray(form.ep_sizes) ? form.ep_sizes : []
  const moeDpSizes = Array.isArray(form.moe_dp_sizes) ? form.moe_dp_sizes : []
  if (tpSizes.length === 0 && epSizes.length === 0 && moeDpSizes.length === 0) return true
  for (const tp of tpSizes.length > 0 ? tpSizes : [1]) {
    for (const ep of epSizes.length > 0 ? epSizes : [1]) {
      for (const moeDp of moeDpSizes.length > 0 ? moeDpSizes : [1]) {
        const numTp = Number(tp) || 0
        const numEp = Number(ep) || 0
        const numMoeDp = Number(moeDp) || 0
        if (numTp === 0 || numEp === 0) continue
        const validTp = numDevices % numTp === 0
        const validEp = numDevices % numEp === 0
        const validMoeDp = numMoeDp === 0 || numDevices % (numEp * numMoeDp) === 0
        if (validTp && validEp && validMoeDp) return true
      }
    }
  }
  return false
}

/** throughput cross 1: num_mtp_tokens ≤ len(mtp_acceptance_rate) + 1.
 * Parses the acceptance-rate list from an array or a comma/space-separated
 * string (the form stores it as text). No-op when num_mtp_tokens is 0. */
export const mtpTokensVsAcceptanceRate = ({ form }: ValidatorContext): boolean | string => {
  const n = Number(form.num_mtp_tokens) || 0
  if (n === 0) return true
  const raw = form.mtp_acceptance_rate
  const arr = Array.isArray(raw)
    ? raw
    : typeof raw === 'string'
      ? raw.split(/[,\s]+/).filter(Boolean)
      : []
  return n <= arr.length + 1
    ? true
    : `num_mtp_tokens(${n}) must be ≤ len(mtp_acceptance_rate)+1 (${arr.length + 1})`
}

/** text rel 8+11: shared_expert_tp XOR host_external_shared_experts; EP>1 if shared_expert_tp. */
export const sharedExpertMutex = ({ form }: ValidatorContext): boolean | string => {
  if (form.enable_shared_expert_tp === true && form.host_external_shared_experts === true) {
    return 'enable_shared_expert_tp and host_external_shared_experts are mutually exclusive'
  }
  if (form.enable_shared_expert_tp === true && Number(form.ep_size) <= 1) {
    return 'enable_shared_expert_tp requires ep_size > 1'
  }
  return true
}

/** throughput cross 5: enable_optimize_prefill_decode_ratio XOR disagg. */
export const pdRatioMutexDisagg = ({ form }: ValidatorContext): boolean | string => {
  if (form.enable_optimize_prefill_decode_ratio === true && form.disagg === true) {
    return 'PD-ratio optimization cannot be used together with disagg'
  }
  return true
}

/** video §4.3 rel 5: use_cfg && cfg_parallel requires world_size ≥ 2. */
export const cfgParallelRequiresWorldSize2 = ({ form }: ValidatorContext): boolean | string => {
  if (form.use_cfg === true && form.cfg_parallel === true && Number(form.world_size) < 2) {
    return 'cfg_parallel requires world_size ≥ 2'
  }
  return true
}

/** text rel 1+2 / throughput: profiling_database required when performance_model includes 'profiling'. */
export const profilingDbRequired = ({ form }: ValidatorContext): boolean | string => {
  const pm = Array.isArray(form.performance_model) ? form.performance_model : []
  if (pm.includes('profiling')) {
    const db = form.profiling_database
    if (!db || (typeof db === 'string' && db.trim() === '')) {
      return 'profiling_database is required when performance_model includes "profiling"'
    }
  }
  return true
}

/**
 * Option sources composable. Fetch-once session cache for dynamic option sources.
 * Only genuinely-dynamic sources (like devices) are fetched from the backend.
 */
import { ref, computed } from 'vue'
import type { LocalizedText } from './useLocale'
import { api } from '@/services/api'

export interface Option {
  value: string
  label: LocalizedText
}

export interface OptionSourceRegistry {
  [name: string]: {
    endpoint: string
    cache: 'session'
  }
}

// Session cache for fetched options
const sessionCache = new Map<string, Option[]>()

// Pending fetches to avoid duplicate requests
const pendingFetches = new Map<string, Promise<Option[]>>()

// Per-endpoint generation counters. `clearCache` bumps these so in-flight
// fetches that were started BEFORE the clear don't write their (now-stale)
// results back into the empty cache — a race where a user-triggered refresh
// could be immediately clobbered by a still-pending prior fetch.
// See PR-632 #32.
const endpointGen = new Map<string, number>()

/**
 * Fetch options from a dynamic source endpoint.
 * Uses session cache and deduplicates concurrent requests.
 */
async function fetchOptions(endpoint: string): Promise<Option[]> {
  // Check cache first
  if (sessionCache.has(endpoint)) {
    return sessionCache.get(endpoint)!
  }

  // Check if a fetch is already in progress
  if (pendingFetches.has(endpoint)) {
    return pendingFetches.get(endpoint)!
  }

  // Snapshot the generation at fetch-start. If clearCache() bumps it while
  // we're in flight, we refuse to repopulate the cache below — the result is
  // still returned to this caller (they see a consistent snapshot), but
  // subsequent callers will re-fetch against a fresh cache.
  const myGen = endpointGen.get(endpoint) ?? 0

  // Start a new fetch
  const fetchPromise = api.getOptions(endpoint)
    .then((data) => {
      const options = data.map((item: any) => ({
        value: item.value,
        label: item.label || item.value, // Fallback to value if label is missing
      }))
      // Only populate cache if generation still matches — otherwise a
      // clearCache() happened mid-flight and writing would resurrect stale data.
      if ((endpointGen.get(endpoint) ?? 0) === myGen) {
        sessionCache.set(endpoint, options)
      }
      return options
    })
    .finally(() => {
      pendingFetches.delete(endpoint)
    })

  pendingFetches.set(endpoint, fetchPromise)
  return fetchPromise
}

/**
 * Clear the session cache (useful for testing or forced refresh).
 * Bumps per-endpoint generation counters so pending fetches can't
 * resurrect stale data.
 */
function clearCache() {
  const allEndpoints = new Set([...sessionCache.keys(), ...pendingFetches.keys()])
  for (const ep of allEndpoints) {
    endpointGen.set(ep, (endpointGen.get(ep) ?? 0) + 1)
  }
  sessionCache.clear()
}

/**
 * Get options for a named source.
 * Returns cached options immediately if available, otherwise fetches.
 */
async function getOptions(
  sourceName: string,
  registry: OptionSourceRegistry
): Promise<Option[]> {
  const source = registry[sourceName]
  if (!source) {
    console.warn(`Option source "${sourceName}" not found in registry`)
    return []
  }

  if (source.cache === 'session') {
    return await fetchOptions(source.endpoint)
  }

  console.warn(`Unknown cache type "${source.cache}" for source "${sourceName}"`)
  return []
}

/**
 * Composable to manage option sources.
 * The registry parameter can be a plain object or a Vue ComputedRef (for reactivity).
 */
export function useOptionSources(registry: OptionSourceRegistry | any) {
  const loading = ref<Set<string>>(new Set())
  const error = ref<Record<string, string>>({})

  // Fetch options for a named source with loading state.
  // Accepts an optional registry parameter to override the initial one (for reactive registries).
  async function fetchSourceOptions(sourceName: string, registryOverride?: OptionSourceRegistry): Promise<Option[]> {
    loading.value = new Set(loading.value).add(sourceName)
    delete error.value[sourceName]

    try {
      // Use override if provided, otherwise try to unwrap ComputedRef or use plain object
      const activeRegistry = registryOverride || (registry?.value !== undefined ? registry.value : registry)
      const options = await getOptions(sourceName, activeRegistry)
      return options
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      error.value[sourceName] = message
      console.error(`Failed to fetch options for "${sourceName}":`, message)
      return []
    } finally {
      loading.value = new Set([...loading.value].filter((name) => name !== sourceName))
    }
  }

  // Check if a source is currently loading
  const isLoading = computed(() => {
    return (sourceName: string) => loading.value.has(sourceName)
  })

  // Get error for a source
  const getSourceError = (sourceName: string) => error.value[sourceName]

  return {
    fetchSourceOptions,
    isLoading,
    getSourceError,
    clearCache,
  }
}

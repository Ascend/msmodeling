/**
 * Telemetry sink protocol for frontend telemetry collection.
 *
 * This provides a pluggable interface similar to the backend telemetry_sink.py,
 * allowing plugins to provide real telemetry implementations while the default
 * is a zero-cost no-op that does nothing.
 */

export interface TelemetryEvent {
  module_id: string
  target: string
  event_type: string
  fingerprint?: string
  metadata?: Record<string, any>
}

/**
 * Telemetry sink protocol. Implementations can buffer, batch, filter, or
 * transmit events; the default NoOpTelemetrySink is zero-cost.
 */
export interface TelemetrySink {
  /**
   * Record a single telemetry event.
   *
   * This should be non-blocking and not throw exceptions to avoid
   * impacting user experience.
   */
  logEvent(event: TelemetryEvent): void

  /**
   * Flush any buffered events.
   *
   * Called during application shutdown to ensure pending events are written.
   */
  flush(): void

  /**
   * Shutdown the sink and release resources.
   *
   * Called during application shutdown as the last operation.
   */
  shutdown(): void
}

/** No-op sink: the default for public builds — all methods empty (zero runtime overhead). */
class NoOpTelemetrySink implements TelemetrySink {
  private static instance: NoOpTelemetrySink | null = null

  private constructor() {}

  static getInstance(): NoOpTelemetrySink {
    if (!NoOpTelemetrySink.instance) {
      NoOpTelemetrySink.instance = new NoOpTelemetrySink()
    }
    return NoOpTelemetrySink.instance
  }

  logEvent(_event: TelemetryEvent): void {
  }

  flush(): void {
  }

  shutdown(): void {
  }
}

/**
 * Global telemetry sink instance.
 * Starts as NoOpTelemetrySink, can be replaced by plugins.
 */
let _sink: TelemetrySink = NoOpTelemetrySink.getInstance()
let _sinkSet = false

/**
 * Get the current telemetry sink instance.
 *
 * This always returns a valid sink, ensuring telemetry code never needs null checks.
 */
export function getTelemetrySink(): TelemetrySink {
  return _sink
}

/**
 * Set the telemetry sink (internal API for plugins).
 *
 * This allows plugins to replace the default NoOpTelemetrySink with a real
 * implementation. Can only be called once to prevent race conditions.
 *
 * @param sink - The new telemetry sink implementation
 * @throws Error if called more than once
 */
export function _setTelemetrySink(sink: TelemetrySink): void {
  if (_sinkSet) {
    throw new Error(
      'TelemetrySink can only be set once. ' +
      'Multiple sink replacements are not supported.'
    )
  }
  _sink = sink
  _sinkSet = true
  console.log(`[TelemetrySink] Set to: ${sink.constructor.name}`)
}

/**
 * Reset telemetry sink to NoOpTelemetrySink (testing only).
 *
 * This is provided for unit testing to reset state between tests.
 * Should never be called in production.
 */
export function _resetTelemetrySinkForTest(): void {
  _sink = NoOpTelemetrySink.getInstance()
  _sinkSet = false
}

/**
 * Convenience function to log a telemetry event.
 *
 * This provides a simple API for common use cases:
 * ```ts
 * logTelemetryEvent({
 *   module_id: 'text_generate',
 *   target: 'field:prompt',
 *   event_type: 'change',
 *   fingerprint: 'user123'
 * })
 * ```
 */
export function logTelemetryEvent(event: TelemetryEvent): void {
  getTelemetrySink().logEvent(event)
}

/**
 * Legacy-compatible trackEvent function for gradual migration.
 *
 * This maintains compatibility with existing code while allowing
 * gradual migration to the new sink-based API.
 *
 * @deprecated Use logTelemetryEvent() instead
 */
export function trackEvent(
  moduleId: string,
  target: string,
  eventType: string,
  _debounce = false
): void {
  // Convert old API to new TelemetryEvent format
  const event: TelemetryEvent = {
    module_id: moduleId,
    target,
    event_type: eventType,
  }
  logTelemetryEvent(event)
}
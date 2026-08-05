/**
 * Plugin system types for msmodeling Web Console frontend.
 *
 * Plugins are discovered at runtime via import.meta.glob over src/plugins/*.
 * A public build has no plugin dirs there, so no plugin code is bundled.
 */

/**
 * Plugin manifest - the contract each plugin must implement.
 *
 * Plugins export a manifest via definePlugin(), which is consumed by the host registry.
 */
export interface PluginManifest {
  /** Plugin identifier (^[a-z][a-z0-9_-]*$). Must be unique. */
  id: string

  /** Human-readable title (bilingual). */
  title: {
    zh: string
    en: string
  }

  /**
   * Optional nav-bar entry. When present, the host renders a nav button for this
   * plugin (sorted by `order`); when absent, the route (if any) is reachable only
   * by direct URL. Use this to opt a plugin into the top nav.
   */
  menuEntry?: {
    /** Bilingual label (rendered via useLocale.t()). */
    label: { zh: string; en: string }
    /** Route name (must match route.name in the route contribution). */
    route: string
    /** Display order in nav (lower = earlier). */
    order: number
    /** Optional icon loader (e.g. an Element Plus icon). */
    icon?: () => Promise<{ default: any }>
  }

  /** Optional route contribution. */
  route?: {
    path: string
    name: string
    /** Dynamic import of the component (lazy-loaded). */
    component: () => Promise<{ default: any }>
  }

  /**
   * Optional app-bar entry: a button rendered in the top app bar (e.g. a
   * feedback trigger). The host wires the click to toggle the matching
   * globalWidget (keyed by plugin id).
   */
  appBar?: {
    /** Dynamic import of an icon component (e.g. an Element Plus icon). */
    icon: () => Promise<{ default: any }>
    /** Bilingual label (rendered via useLocale.t()). */
    label: { zh: string; en: string }
  }

  /**
   * Optional global floating widget (e.g. a feedback dialog). Rendered once by
   * the host shell; visibility is toggled by the matching appBar entry.
   */
  globalWidget?: {
    /** Dynamic import of the widget component. Must accept v-model:visible. */
    component: () => Promise<{ default: any }>
  }

  /** Optional form config contribution (for future use). */
  formConfig?: () => Promise<any>

  /** Optional result component contribution (for future use - per-module rendering). */
  resultComponent?: {
    /** Module ID this result component handles. */
    moduleId: string
    /** Dynamic import of the component. */
    component: () => Promise<{ default: any }>
    /** Multi-case variant (optional). */
    multiCaseComponent?: () => Promise<{ default: any }>
  }

  /** Optional install callback - called by host during app bootstrap. */
  install?: (app: any) => void
}

/**
 * Define a plugin manifest.
 *
 * Plugins call this function at the top level of their index.ts to export their manifest:
 *   export default definePlugin({ id: 'device-upload', ... })
 */
export function definePlugin(manifest: PluginManifest): PluginManifest {
  return manifest
}

/**
 * Plugin host registry interface (internal, implemented by plugins/index.ts).
 */
export interface PluginHost {
  /** Get all enabled plugins (discovered via import.meta.glob). */
  getPlugins(): ReadonlyArray<PluginManifest>

  /** Get a plugin by ID. */
  getPlugin(id: string): PluginManifest | undefined

  /** Register a plugin (called by the glob import loop). */
  register(manifest: PluginManifest): void

  /** Install all plugins (call their install() callbacks). */
  install(app: any): void
}

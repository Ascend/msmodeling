/**
 * Plugin host registry for msmodeling Web Console frontend.
 *
 * Discovers plugins via import.meta.glob over each direct subdirectory of
 * src/plugins. Every plugin dir present there (copied by scripts/copy-plugins.mjs)
 * is enabled; a public build has no plugin dirs, so the glob is empty and no
 * plugin code is bundled. No hardcoded ID list — adding a plugin is just dropping
 * its dir into the internal package's frontend/plugins.
 */

import type { PluginManifest, PluginHost } from './types'

// Discover plugin manifests using import.meta.glob
// The pattern './*/index.ts' matches direct subdirectories (device-upload/index.ts, etc.)
// In a public build (no plugin dirs), this returns an empty object.
const pluginModules = import.meta.glob('./*/index.ts', { eager: true })

class PluginHostImpl implements PluginHost {
  private plugins: Map<string, PluginManifest> = new Map()

  getPlugins(): ReadonlyArray<PluginManifest> {
    return Array.from(this.plugins.values())
  }

  getPlugin(id: string): PluginManifest | undefined {
    return this.plugins.get(id)
  }

  register(manifest: PluginManifest): void {
    const { id } = manifest
    if (this.plugins.has(id)) {
      console.warn(`[PluginHost] Duplicate plugin ID "${id}", skipping`)
      return
    }
    this.plugins.set(id, manifest)
    console.log(`[PluginHost] Registered plugin "${id}"`)
  }

  install(app: any): void {
    for (const plugin of this.plugins.values()) {
      if (plugin.install) {
        try {
          plugin.install(app)
        } catch (err) {
          console.error(`[PluginHost] Install callback failed for plugin "${plugin.id}":`, err)
        }
      }
    }
  }
}

const pluginHost = new PluginHostImpl()

// Process discovered plugin modules
for (const [path, module] of Object.entries(pluginModules)) {
  const match = path.match(/^\.\/([^/]+)\/index\.ts$/)
  if (!match) {
    console.warn(`[PluginHost] Unmatched plugin path: ${path}`)
    continue
  }
  const pluginId = match[1]

  const manifest = (module as any).default as PluginManifest | undefined
  if (!manifest) {
    console.error(`[PluginHost] Plugin "${pluginId}" has no default export`)
    continue
  }

  pluginHost.register(manifest)
}

// Log final state
if (pluginHost.getPlugins().length === 0) {
  console.log('[PluginHost] No plugins loaded (system running without plugins)')
}

export function install(app: any) {
  ;(app.config.globalProperties as any).pluginHost = pluginHost
  pluginHost.install(app)
}

export { pluginHost }

export default {
  install,
  getPlugins: () => pluginHost.getPlugins(),
  getPlugin: (id: string) => pluginHost.getPlugin(id),
}

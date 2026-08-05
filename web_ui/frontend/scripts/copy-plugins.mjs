/**
 * Copy frontend plugin sources from installed plugin packages.
 *
 * This script runs before Vite build (via npm scripts) to copy plugin
 * frontend code from <site-packages>/msmodeling_internal/frontend/plugins/
 * to src/plugins/ for consumption by import.meta.glob().
 *
 * For public builds (mode != 'internal'), no plugins are installed, so
 * this script exits successfully with no files copied.
 */

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Source: installed plugin packages (search roots)
//
// For editable installs the frontend lives under the project-local venv
// (.venv-clean / .venv) without a platform-specific site-packages layout.
// For installed wheels we need to search both:
//   - Windows:       <venv>/Lib/site-packages
//   - Linux/macOS:   <venv>/lib/pythonX.Y/site-packages
// The old code hardcoded `Lib/site-packages` so a Linux/macOS build silently
// skipped the internal plugin frontend. Now we enumerate each venv's
// site-packages dir(s) with a small helper. See PR-632 #59.
function findSitePackages(venvRoot) {
  if (!venvRoot || !fs.existsSync(venvRoot)) return []
  const found = []
  // Windows layout
  const winPath = path.join(venvRoot, 'Lib', 'site-packages')
  if (fs.existsSync(winPath)) found.push(winPath)
  // POSIX layout — scan any `lib/pythonX.Y/` child
  const libDir = path.join(venvRoot, 'lib')
  if (fs.existsSync(libDir) && fs.statSync(libDir).isDirectory()) {
    for (const entry of fs.readdirSync(libDir)) {
      if (!entry.startsWith('python')) continue
      const sp = path.join(libDir, entry, 'site-packages')
      if (fs.existsSync(sp)) found.push(sp)
    }
  }
  return found
}

const venvRoots = [
  // Project local .venv-clean (editable install points here directly)
  path.join(process.cwd(), '..', '..', '.venv-clean'),
  // Project local .venv
  path.join(process.cwd(), '..', '..', '.venv'),
  // CONDA_PREFIX / VIRTUAL_ENV (system or user venv)
  process.env.CONDA_PREFIX || '',
  process.env.VIRTUAL_ENV || '',
].filter(Boolean)

// Each venv contributes its own `<root>/msmodeling_internal` lookup AND any
// discovered site-packages dirs. The msmodeling_internal paths (editable
// installs) exist directly at the venv root, not under site-packages.
const searchRoots = [
  ...venvRoots.map((r) => path.join(r, 'msmodeling_internal')),
  ...venvRoots.flatMap((r) => findSitePackages(r).map((sp) => path.join(sp, 'msmodeling_internal'))),
]

// Target: src/plugins/ (where import.meta.glob looks)
const targetDir = path.join(__dirname, '..', 'src', 'plugins')

console.log('Copying frontend plugins from installed packages...')

fs.mkdirSync(targetDir, { recursive: true })

let copied = 0

for (const pluginRoot of searchRoots) {
  if (!fs.existsSync(pluginRoot)) {
    continue
  }

  const frontendPath = path.join(pluginRoot, 'frontend', 'plugins')
  if (!fs.existsSync(frontendPath)) {
    continue
  }

  console.log(`Found plugin source: ${frontendPath}`)

  const pluginDirs = fs.readdirSync(frontendPath).filter(name => {
    const fullPath = path.join(frontendPath, name)
    return fs.statSync(fullPath).isDirectory()
  })

  for (const pluginId of pluginDirs) {
    const src = path.join(frontendPath, pluginId)
    const dst = path.join(targetDir, pluginId)

    console.log(`  Copying plugin: ${pluginId}`)

    // Remove existing if any
    if (fs.existsSync(dst)) {
      fs.rmSync(dst, { recursive: true, force: true })
    }

    // Copy plugin directory
    fs.mkdirSync(dst, { recursive: true })
    const files = fs.readdirSync(src)
    for (const file of files) {
      const srcFile = path.join(src, file)
      const dstFile = path.join(dst, file)
      fs.copyFileSync(srcFile, dstFile)
    }

    copied++
  }
}

if (copied === 0) {
  console.log('No plugin packages found (expected for public builds)')
} else {
  console.log(`Copied ${copied} plugin(s) to ${targetDir}`)
}

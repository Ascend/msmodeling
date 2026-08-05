#!/usr/bin/env node
/**
 * gen-form-schemas.mjs — regenerate the data-only JSON form configs from the
 * TypeScript sources (src/config/forms/*.ts).
 *
 * Why: the .ts configs are the single source of truth (schema data + inlined
 * validator functions). The backend needs pure JSON (it json.load's them to
 * hash + pin a snapshot). JSON.stringify drops function values, so the
 * generated .json contains the data with an empty `validators` map — exactly
 * what the backend needs. Functions stay frontend-only.
 *
 * Run: `node scripts/gen-form-schemas.mjs` (wired into npm run build / dev).
 */
import esbuild from 'esbuild'
import { readdirSync, writeFileSync, rmSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(__dirname, '..')
const formsDir = path.join(root, 'src', 'config', 'forms')

const tsFiles = readdirSync(formsDir).filter(
  (f) => f.endsWith('.ts') && !f.startsWith('_') && f !== '_validators.ts',
)

if (tsFiles.length === 0) {
  console.warn('[gen-form-schemas] no form .ts files found in', formsDir)
  process.exit(0)
}

let generated = 0
for (const f of tsFiles) {
  const entry = path.join(formsDir, f)
  const tmp = path.join(tmpdir(), `msm-form-${f.replace('.ts', '')}-${process.pid}.mjs`)
  try {
    // Bundle the .ts (resolving its ./_validators import) to a temp ESM module.
    await esbuild.build({
      entryPoints: [entry],
      bundle: true,
      format: 'esm',
      outfile: tmp,
      platform: 'node',
      logLevel: 'silent',
    })
    const mod = await import(pathToFileURL(tmp).href)
    const envelope = mod.default ?? {}
    // JSON.parse(JSON.stringify(...)) strips the validator functions (they are
    // not JSON-serializable). We also drop the now-empty `validators` key and
    // the frontend-only `groups` (collapse-default metadata) entirely: the
    // backend never uses either, and leaving them would change the canonical
    // schema_hash (refuse-on-mismatch for the same version).
    const dataOnly = JSON.parse(JSON.stringify(envelope))
    delete dataOnly.validators
    delete dataOnly.groups
    const outPath = path.join(formsDir, f.replace(/\.ts$/, '.json'))
    writeFileSync(outPath, JSON.stringify(dataOnly, null, 2) + '\n')
    generated += 1
    console.log(`[gen-form-schemas] ${path.relative(root, outPath)}`)
  } catch (err) {
    console.error(`[gen-form-schemas] FAILED on ${f}:`, err.message)
    process.exit(1)
  } finally {
    rmSync(tmp, { force: true })
  }
}

console.log(`[gen-form-schemas] generated ${generated} schema(s).`)

/**
 * useMarkdown — parse markdown to HTML with syntax highlighting.
 *
 * Uses marked (v18+) for parsing and highlight.js for code blocks.
 * Dynamically imports both libraries so they are only loaded when
 * the user visits the docs page (not in the main bundle).
 */

import { ref } from 'vue'

export interface DocMeta {
  /** Display title extracted from the first H1. */
  title: string
  /** Rendered HTML. */
  html: string
  /** Extracted H2 headings (anchor + text) for the sidebar TOC. */
  headings: Array<{ anchor: string; text: string }>
}

/** Slugify a heading text into an anchor id. */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w一-鿿]+/g, '-')
    .replace(/(^-|-$)/g, '')
}

// marked.use() appends to the extension list and is NOT idempotent — calling it
// on every render() accumulates code renderers without bound (memory leak as
// docs are opened/switched). The dynamic-import cache hands back the SAME marked
// instance each time, so a single module-level guard registers our highlight.js
// code renderer at most once per marked instance.
let markedConfigured = false

/**
 * Render markdown source to HTML with highlight.js applied to code blocks.
 * Returns { title, html, headings } for the doc page.
 */
export function useMarkdown() {
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function render(markdown: string): Promise<DocMeta | null> {
    loading.value = true
    error.value = null

    try {
      const [{ marked: markedFn }, hljsMod] = await Promise.all([
        import('marked'),
        import('highlight.js'),
      ])
      const hl = hljsMod.default

      // Register highlight.js code renderer via marked.use (v18 API) ONCE.
      // marked.use appends to the extension list — without this guard every
      // render() call would pile on another renderer (the dynamic-import cache
      // returns the same marked instance, so the renderer list grows forever).
      if (!markedConfigured) {
        markedFn.use({
          renderer: {
            code({ text, lang }: { text: string; lang?: string }) {
              const language = lang && hl.getLanguage(lang) ? lang : 'plaintext'
              const highlighted = hl.highlight(text, { language }).value
              return `<pre><code class="hljs language-${language}">${highlighted}</code></pre>`
            },
          },
        })
        markedConfigured = true
      }

      // Extract H2 headings before rendering (for sidebar TOC) and assign UNIQUE
      // anchor ids. Duplicate headings would otherwise share a slug, so both the
      // TOC link and the injected id would point at the FIRST occurrence and
      // later same-name sections couldn't be jumped to. A slug counter disambiguates
      // repeats with a stable -2/-3 suffix; the same sequence is reused below for
      // id injection so TOC anchors and HTML ids stay in lock-step.
      const headingRegex = /^## (.+)$/gm
      const headings: DocMeta['headings'] = []
      const seenSlugs = new Map<string, number>()
      let m: RegExpExecArray | null
      while ((m = headingRegex.exec(markdown)) !== null) {
        const text = m[1].trim()
        const base = slugify(text)
        const n = (seenSlugs.get(base) || 0) + 1
        seenSlugs.set(base, n)
        const anchor = n === 1 ? base : `${base}-${n}`
        headings.push({ anchor, text })
      }

      // Extract first H1 as title.
      const h1Match = markdown.match(/^# (.+)$/m)
      const title = h1Match ? h1Match[1].trim() : ''

      // Inject the SAME unique anchor ids into the H2 headings. This replace and
      // the TOC loop above run the identical /^## (.+)$/gm regex over the same
      // source, so headings are emitted in the same order — consume the
      // precomputed unique anchors by index to keep TOC and HTML aligned.
      let headingIdx = 0
      const withAnchors = markdown.replace(
        /^## (.+)$/gm,
        (_: string, text: string) => {
          const id = headings[headingIdx++]?.anchor ?? slugify(text.trim())
          return `## <span id="${id}">${text.trim()}</span>`
        },
      )

      const html = await markedFn.parse(withAnchors, {
        gfm: true,
        breaks: false,
      })

      return { title, html, headings }
    } catch (e: any) {
      error.value = e.message || 'Failed to render markdown'
      return null
    } finally {
      loading.value = false
    }
  }

  return { render, loading, error }
}

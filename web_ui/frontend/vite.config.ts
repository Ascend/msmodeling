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

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

// Vite config for the web console frontend.
// - @ alias -> src/ (Constitution Principle II: src/ holds the data-driven app)
// - Dev server proxies /api -> backend FastAPI.
//   ``MSMODELING_BACKEND_HOST`` (set by ``web_ui/main.py`` launcher from the
//   same loopback-detection the backend runs) overrides the default
//   ``localhost`` so the proxy target matches what the backend actually bound
//   to — matters on machines where IPv4 is disabled and backend is on ``::1``.
//   IPv6 literals need ``[...]`` in a URL; bare hostnames don't.
// - Plugin enablement is data-driven: src/plugins/index.ts enables every plugin
//   dir present under src/plugins/ (copied by scripts/copy-plugins.mjs). A public
//   build has no plugin dirs there, so no plugin code is bundled. Adding/removing
//   a plugin needs no edit here.
const backendHost = process.env.MSMODELING_BACKEND_HOST || 'localhost'
const backendTarget =
  backendHost.includes(':') ? `http://[${backendHost}]:8000` : `http://${backendHost}:8000`

// Frontend bind host: use env var if set, otherwise let Vite decide (IPv4/IPv6)
const frontendHost = process.env.MSMODELING_FRONTEND_HOST

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: frontendHost,
    port: 5173,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    target: 'es2020',
  },
})

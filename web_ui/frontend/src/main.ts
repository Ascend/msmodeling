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

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import './styles/theme.css'

import App from './App.vue'
import router from './router'
import pluginHost from './plugins'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.use(ElementPlus)
app.use(pluginHost) // Register plugin host (loads enabled plugins)

// Register plugin routes
for (const plugin of pluginHost.getPlugins()) {
  if (plugin.route) {
    router.addRoute({
      path: plugin.route.path,
      name: plugin.route.name,
      component: plugin.route.component,
      meta: {
        pluginId: plugin.id,
      },
    })
    console.log(`[Router] Added plugin route: ${plugin.route.path} -> ${plugin.route.name}`)
  }
}

// Light is the default theme. Dark css-vars.css is still imported above so the
// `html.dark` class opts into dark at runtime (toggle hook left for later).
app.mount('#app')

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

import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import Console from '../pages/Console.vue'
import History from '../pages/History.vue'
import JobStatus from '../pages/JobStatus.vue'
import JobResult from '../pages/JobResult.vue'
import Docs from '../pages/Docs.vue'
import { pluginHost } from '../plugins'

// Explicit static imports

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/console' },
  { path: '/console', name: 'console', component: Console },
  { path: '/home', redirect: '/console' },
  // Deep links (history browse + per-job status/result views) remain available.
  { path: '/history', name: 'history', component: History },
  { path: '/jobs/:jobId/status', name: 'jobStatus', component: JobStatus },
  { path: '/jobs/:jobId/result', name: 'jobResult', component: JobResult },
  { path: '/docs', name: 'docs', component: Docs },
]

// Register plugin-contributed routes
for (const plugin of pluginHost.getPlugins()) {
  if (plugin.route) {
    routes.push({
      path: plugin.route.path,
      name: plugin.route.name,
      component: plugin.route.component,
    })
  }
}

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router

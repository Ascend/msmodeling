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

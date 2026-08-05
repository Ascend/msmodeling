# Web UI 重构设计文档（Vue 3 + FastAPI）

## 修订记录

| 日期 | 修订版本 | 修改描述 | 作者 |
| -- | -- | -- | -- |
| 2026-07-28 | v1 | 从 Gradio 迁移到 Vue 3 + FastAPI 架构的重构设计文档 | zwt |

---

## 1. 概述

本文档描述 `web_ui/` 从 Gradio Blocks 迁移到 **Vue 3 + Element Plus 前端 + FastAPI + SQLite 后端** 的重构设计。重构已完成核心闭环，本文档作为架构基线，指导后续演进。

### 1.1 重构动机

Gradio 版本（参见 `web_ui_frontend_design.md`）存在以下核心问题：

| 问题 | Gradio 现状 | 重构后 |
|---|---|---|
| **文件膨胀** | `app.py` + `callbacks.py` 集中所有逻辑 | 前后端分离，职责拆分到独立模块 |
| **位置参数传递** | 回调依赖长参数列表 | 类型安全的 TypeScript 接口 + Pydantic schema |
| **状态管理分散** | `gr.State` 散落各处 | Vue 3 reactive + Pinia store |
| **结果展示** | 通用表格，专项语义弱 | 模块化结果组件，mode 感知渲染 |
| **任务治理** | 缺少任务中心 | Job 生命周期管理 + 历史 + 轮询 |
| **安全边界** | 前后端职责模糊 | 后端 FastAPI 强约束 + 前端纯展示 |

### 1.2 技术选型

| 层 | 技术 | 理由 |
|---|---|---|
| 前端框架 | Vue 3 (`<script setup>`) | 组合式 API，轻量，生态成熟 |
| UI 库 | Element Plus | 企业级组件，暗色主题支持 |
| 图表 | ECharts 5 + vue-echarts | 散点图 / 柱状图 / 主题感知 |
| 状态管理 | Pinia + reactive composables | 轻量，无 boilerplate |
| 构建 | Vite 6 | 快速 HMR，ESM 原生 |
| 后端框架 | FastAPI | 异步、自动 OpenAPI、Pydantic 校验 |
| 数据库 | SQLite (WAL mode) | 零配置，本地单机 |
| 迁移 | Alembic | 版本化 schema 管理 |
| 测试 | pytest + pytest-cov | 100% 后端覆盖率 |

---

## 2. 架构全景

### 2.1 系统分层

```text
┌─────────────────────────────────────────────────┐
│                  浏览器 (Vue 3 SPA)              │
│  ┌───────────┐  ┌───────────┐  ┌─────────────┐ │
│  │  Pages     │  │ Components│  │ Composables  │ │
│  │  (路由页)  │  │ (UI 组件) │  │ (业务逻辑)   │ │
│  └─────┬─────┘  └─────┬─────┘  └──────┬──────┘ │
│        └───────────────┴───────────────┘        │
│                        │ HTTP (axios)            │
└────────────────────────┼────────────────────────┘
                         │
┌────────────────────────┼────────────────────────┐
│                   FastAPI 后端                    │
│  ┌───────────┐  ┌───────────┐  ┌─────────────┐ │
│  │ API Router │  │ Services  │  │  Runners     │ │
│  │ (路由层)   │  │ (业务层)  │  │ (执行层)     │ │
│  └─────┬─────┘  └─────┬─────┘  └──────┬──────┘ │
│        └───────────────┴───────────────┘        │
│                        │ subprocess                │
└────────────────────────┼────────────────────────┘
                         │
┌────────────────────────┼────────────────────────┐
│              CLI (cli/inference/*)               │
│  text_generate / video_generate /                │
│  throughput_optimizer                             │
└─────────────────────────────────────────────────┘
```

### 2.2 前端目录结构

```text
web_ui/frontend/src/
├── App.vue                    # 根组件（app-bar + router-view）
├── main.ts                    # 入口（Vue + Element Plus + Pinia）
├── router/index.ts            # 路由（Console / History / JobResult / Docs）
├── pages/                     # 路由页
│   ├── Console.vue            # 工作台（tab 切换三个模块）
│   ├── History.vue            # 任务历史
│   ├── JobResult.vue          # 深链接结果页
│   ├── JobStatus.vue          # 任务状态详情
│   └── Docs.vue               # 使用文档
├── components/
│   ├── workspace/             # 工作区
│   │   ├── ModuleWorkspace.vue # 表单 + 结果分屏（可拖拽分割条）
│   │   └── ResultPane.vue     # 结果面板（idle/running/succeeded/failed）
│   ├── form/                  # 表单
│   │   ├── SchemaForm.vue     # 动态表单（从 config 驱动）
│   │   └── SchemaFormItem.vue # 单字段渲染 + 条件 + 校验
│   ├── result/                # 结果展示（模块化）
│   │   ├── text/              # TextGenerate 结果组件
│   │   ├── video/             # VideoGenerate 结果组件
│   │   ├── throughput/        # ThroughputOptimizer 结果组件
│   │   │   ├── ThroughputOptimizerResult.vue   # 模式路由
│   │   │   ├── ThroughputMultiCaseResult.vue   # 多用例
│   │   │   └── views/
│   │   │       ├── AggregatedView.vue          # 聚合模式
│   │   │       ├── DisaggregatedView.vue       # 分离模式
│   │   │       ├── PDRatioView.vue             # PD 配比
│   │   │       └── OptimizerCurves.vue         # 散点图
│   │   ├── ChartWrapper.vue   # ECharts 封装（主题感知）
│   │   └── OperatorTiming*    # 算子级耗时表/图
│   └── job-status/            # 任务状态卡
│       ├── JobCommandCard.vue # 命令展示（单/多用例）
│       └── JobLogDrawer.vue   # 日志抽屉
├── composables/               # 组合式函数（业务逻辑）
│   ├── useJobRunner.ts        # 任务提交→轮询→结果生命周期
│   ├── useFormValidation.ts   # 表单校验（依赖图 + 条件）
│   ├── useFieldConditions.ts  # 字段条件求值
│   ├── useChartTheme.ts       # 图表主题（light/dark）
│   ├── useResultComponent.ts  # 模块→结果组件路由
│   ├── useLocale.ts           # 国际化（zh/en）
│   └── ...
├── stores/                    # Pinia store
│   ├── formState.ts           # 表单值（跨组件共享）
│   └── telemetry.ts           # 遥测事件
├── services/                  # API 层
│   ├── api.ts                 # axios 封装 + 所有 endpoint
│   └── request.ts             # axios 实例
├── config/forms/              # 表单配置（source of truth）
│   ├── text_generate.ts       # 字段定义 + 校验 + 默认值
│   ├── video_generate.ts
│   ├── throughput_optimizer.ts
│   └── _validators.ts         # 校验函数
└── styles/theme.css           # CSS 变量主题
```

### 2.3 后端目录结构

```text
web_ui/backend/
├── main.py                    # FastAPI app + lifespan
├── db.py                      # SQLite engine + alembic 迁移
├── plugin_discovery.py        # 插件迁移路径发现
├── api/
│   ├── routers/               # API 路由
│   │   ├── jobs.py            # 提交/查询/取消/删除/日志/trace
│   │   ├── cases.py           # 用例管理
│   │   ├── modules.py         # 模块元数据
│   │   └── options.py         # 动态选项（设备列表等）
│   ├── schemas.py             # Pydantic 响应模型
│   └── errors.py              # 异常处理
├── models/
│   ├── entities.py            # 数据实体（Job, ResultRecord）
│   ├── enums.py               # 枚举（JobStatus 等）
│   └── orm.py                 # SQLModel ORM 定义
├── services/
│   ├── job_manager.py         # 异步任务管理
│   ├── job_runner.py          # 任务执行（subprocess + 进程池）
│   ├── result_view.py         # 结果组装（Top-N + SLO + 多用例）
│   ├── ranking.py             # 排名计算
│   ├── repositories.py        # 数据访问层
│   ├── schema_registry.py     # 表单 schema 快照 + hash
│   ├── capture.py             # 日志捕获（编码兼容）
│   ├── write_queue.py         # 异步写入队列
│   ├── params_hash.py         # 参数哈希（缓存键）
│   ├── sim_warmup.py          # 仿真栈预热
│   └── trace_store.py         # Chrome trace 文件管理
├── runners/                   # Runner 适配器
│   ├── _subprocess.py         # 通用 subprocess spawner
│   ├── _worker.py             # Worker 入口（序列化执行）
│   ├── _multicase.py          # 多用例展开（cartesian product）
│   ├── _cli_command.py        # CLI 命令重建（per-case）
│   ├── text_generate.py       # Text Generate runner
│   ├── video_generate.py      # Video Generate runner
│   ├── throughput_optimizer.py # Throughput Optimizer runner
│   ├── registry.py            # Runner 注册表
│   └── _video_generate_runner.py # Video 专用执行器
├── plugins/                   # 插件系统
│   ├── contract.py            # MsmdPlugin 数据契约
│   ├── loader.py              # entry_points 发现
│   └── manager.py             # 插件生命周期管理
└── migrations/                # Alembic 迁移
    ├── env.py
    └── versions/
        └── 0001_initial_schema.py
```

---

## 2.4 当前页面能力

### 全局层（App Shell）

| 能力 | 说明 |
|---|---|
| 顶部导航栏 | 品牌标识 + 主页/文档/历史 导航按钮 |
| 语言切换 | 中文 / English 实时切换（内联双语，非 vue-i18n） |
| 主题切换 | 亮色 / 暗色 实时切换（CSS 变量 + ECharts 主题同步） |
| 插件导航 | 插件贡献的菜单项 / app-bar 按钮 / 全局浮窗（动态发现） |
| 任务运行时锁 | 任务运行中阻止切换模块标签页（Toast 提示） |

### Console 页（工作台）

Console 是主工作区，采用 **Tab + 上下分屏** 布局：

```text
┌─────────────────────────────────────────────────┐
│ [文本生成] [视频生成] [吞吐优化]          ← Tab │
├─────────────────────────────────────────────────┤
│                                                 │
│    配置表单 (SchemaForm)                        │
│    ┌─────────────────────────────────────┐      │
│    │ 通用参数                             │      │
│    │ 请求参数                             │      │
│    │ 并行参数                             │      │
│    │ 量化参数                             │      │
│    │ MoE 参数（折叠）                      │      │
│    │ 高级并行（折叠）                      │      │
│    │ 调试参数（折叠）                      │      │
│    └─────────────────────────────────────┘      │
│                                                 │
├══════════════ 可拖拽分割条 ══════════════════════┤
│                                                 │
│    结果面板 (ResultPane)                         │
│    ┌─────────────────────────────────────┐      │
│    │ idle: 空占位                         │      │
│    │ running: 旋转图标 + 状态              │      │
│    │ succeeded: 结果组件                  │      │
│    │ failed: 错误告警                     │      │
│    └─────────────────────────────────────┘      │
│                                                 │
├─────────────────────────────────────────────────┤
│  [▶ 运行]  免责声明 · GitCode 链接              │
└─────────────────────────────────────────────────┘
```

**表单能力**：

- 配置驱动：字段从 `config/forms/*.ts` 动态生成，不硬编码 UI
- 分组折叠：通用 / 请求 / 并行 / 量化 / MoE / 高级并行 / 调试
- 控件类型：text / number / select / multi-select / switch
- 字段说明：每个字段有 tooltip（hover 显示中英文说明 + 关系描述）
- 校验：required / min / max / pattern / 自定义 validator（跨字段依赖图）
- 多值字段：逗号分隔列表 → 自动展开为多用例（如 device=A,B → 2 个 case）
- 命令预览：不展示（提交后在结果区查看 JobCommandCard）

**提交反馈**：

- 提交成功 → Toast 通知 `Task submitted successfully (ID: xxx...)`
- 运行中 → 表单上方显示旋转图标 + 状态文本（排队等待 / 运行中）
- 运行按钮禁用 + loading 状态
- Tab 切换锁定（任务运行中不可切换模块）

**结果面板状态机**：

| 状态 | 展示内容 |
|---|---|
| `idle` | 空占位："填写上方表单并提交，结果将在此展示" |
| `pending` | 旋转图标 + "排队等待" + 取消按钮 + 日志按钮 |
| `running` | 旋转图标 + "运行中" + 进度文本 + 取消按钮 + 日志按钮 |
| `succeeded` | 成功标签 + Job ID + 结果组件 + 日志按钮 |
| `failed` | 错误告警 + 错误详情 + 日志按钮 |
| `cancelled` | 警告告警："任务已被取消" |
| `interrupted` | 错误告警："服务在运行时中断" |

### Text Generation 结果

| 能力 | 说明 |
|---|---|
| 性能指标卡 | batch_size / run_time / execution_time / tps_per_model |
| 显存信息 | total_device / model_weight / kv_cache / peak_usage / available |
| 算子耗时表 | Name / total / avg / # of Calls（按耗时降序） |
| 算子耗时图 | ECharts 柱状图（Top N 算子） |
| 性能分解 | prefill / decode 阶段耗时占比 |
| Chrome trace | 下载链接（按 case/seq 索引） |
| 多用例 | summary 表 + drill-down 到单 case 指标卡 |

### Video Generation 结果

| 能力 | 说明 |
|---|---|
| 执行时间 | 分阶段耗时（encode / decode / sample 等） |
| 性能分解 | 各阶段耗时占比 |
| 结果表行 | table_averages 结构化行 |
| 算子耗时 | Name / total / avg / # of Calls |
| Chrome trace | 下载链接 |
| 多用例 | summary 表 + drill-down |

### Throughput Optimizer 结果

**散点图（OptimizerCurves）**：

- 数据来源：raw records（全量探索点）
- 分组：按 `parallel` 着色（每种并行策略一种颜色）
- 过滤：内存过滤（OOM 行 ≤0 剔除）+ 去重（drop_duplicates）
- 模式感知：
  - 聚合：2 张图（Throughput vs Concurrency + vs TPOT）
  - 分离：4 张图（Prefill: vs Concurrency + vs TTFT；Decode: vs Concurrency + vs TPOT）
  - PD 配比：2 张图（TPS vs Concurrency + vs TPOT，使用 decode 列）

- 主题感知：light/dark 切换

**聚合视图（AggregatedView）**：

- 跨设备柱状图（多设备时显示每设备最优吞吐）
- Sweep 表（Top-N 排序）：rank / throughput / TTFT / TPOT / concurrency / num_devices / parallel / batch_size
- CSV 导出

**分离视图（DisaggregatedView）**：

- Prefill 表（TTFT-oriented）
- Decode 表（TPOT-oriented）
- 各自的 Best 配置卡
- CSV 导出

**PD 配比视图（PDRatioView）**：

- PD Ratio 表：PD Ratio / Balanced QPS / P/D QPS / TTFT / TPOT / 并行配置
- Best PD 配比卡

**多用例视图（ThroughputMultiCaseResult）**：

- Summary 表（每 case 一行：设备 + 指标）
- 点击 drill-down → 单 case 完整结果（含散点图 + 模式视图）

### History 页（任务历史）

| 能力 | 说明 |
|---|---|
| 任务列表 | 表格展示：Job ID / 模块 / 标签 / 状态 / 创建时间 / 完成时间 |
| 状态标签 | 颜色编码：成功(绿) / 失败(红) / 运行中(蓝) / 取消(黄) |
| 模块过滤 | 按模块筛选（Text / Video / Throughput） |
| 状态过滤 | 按状态筛选 |
| 搜索 | 按 Job ID / 标签搜索（客户端过滤当前页） |
| 分页 | 可选每页 10/20/50/100 条 |
| 操作 | 查看结果（succeeded）/ 查看状态（running）/ 查看详情（failed） |

### Job Result 页（深链接结果）

通过 `/jobs/{jobId}` 深链接直接查看任务结果（从 History 点击进入）：

- Header：Job ID + 模块 + 查看状态按钮 + 返回按钮
- 结果组件：与工作区 ResultPane 渲染逻辑一致
- 加载态：骨架屏

### Job Status 页（任务状态）

- 任务状态头（进度 + 状态标签）
- 命令卡（JobCommandCard）：
  - 单 case：一条命令 + `$` 前缀 + 复制按钮 + 原始参数 JSON（折叠）
  - 多 case：每 case 独立命令块 + Case 标签 + 复制全部 + 原始参数 JSON（折叠）

### 日志抽屉（JobLogDrawer）

从工作区 / 状态页点击"日志"打开：

| 能力 | 说明 |
|---|---|
| 全量日志 | 任务主日志（banner + 所有 case 交织输出） |
| Per-case 日志 | 按 case_hash 过滤的独立日志（radio 切换） |
| 日志搜索 | 大小写不敏感行过滤（匹配行数 / 总行数） |
| ANSI 渲染 | 终端彩色 → HTML（bold / color / italic / underline） |

---

## 3. 核心数据流

### 3.1 任务提交 → 结果展示

```text
用户填写表单
→ SchemaForm 校验（useFormValidation）
→ ModuleWorkspace.handleSubmit
→ useJobRunner.submit(params, version)
→ POST /api/jobs
→ JobManager.submit_async
→ JobRunner（异步线程）
→ RunnerAdapter.run
→ _subprocess.run_module_subprocess
→ 子进程: runners._worker
→ Runner.execute(params)  # text/video/throughput
→ 多用例展开（expand_cases）
→ 逐 case 执行 CLI（ParallelRunner / ModelRunner）
→ 结果记录写入 SQLite
→ 返回 records + skipped_hashes
→ 主进程持久化 + ranking
→ 前端轮询 GET /api/jobs/{id}
→ 状态变为 succeeded
→ GET /api/jobs/{id}/result
→ result_view.assemble_result_envelope
→ 前端渲染结果组件
```

### 3.2 结果组装（result_view.py）

结果组装发生在 **API 请求时**（非存储时），确保 schema 变更后旧数据仍可正确渲染：

```text
records[] (raw, 全量探索点)
│
├─→ _apply_cli_topn_filter (SLO 过滤 + 每并行度去重)
│   ├─ aggregation: groupby(parallel, mode).first()
│   ├─ disagg: mode-aware SLO mask (prefill→ttft, decode→tpot)
│   └─ pd_ratio: best_pd_row_per_group + balanced_qps round-2 去重
│
├─→ _reassign_local_ranks (过滤后局部重排)
│
└─→ 组装 envelope
    ├─ best_config (rank=1)
    ├─ sweep_rows (排序后展示)
    ├─ cross_hardware (每设备最优)
    ├─ disagg_prefill / disagg_decode (分相表)
    └─ pd_ratio_rows
```

### 3.3 表单配置驱动

表单完全由 `config/forms/*.ts` 配置驱动，不再硬编码 UI：

```typescript
// 字段定义示例
{
  id: "quantize_linear_action",
  label: { zh: "线性层量化", en: "Linear-Layer Quantization" },
  control: "multi-select",           // 控件类型
  dataType: "string[]",
  default: ["W8A8_DYNAMIC"],
  group: { zh: "量化", en: "Quantization" },
  tooltip: { zh: "...", en: "..." },
  optionSource: { type: "inline", values: QUANTIZE_LINEAR_OPTIONS },
  validation: [
    { rule: "required", message: { zh: "...", en: "..." }, trigger: ["change", "blur"] }
  ]
}
```

**字段联动**：通过 tooltip 文字描述关系（非程序化禁用），让用户自主判断。

---

## 4. 关键设计决策

### 4.1 前后端通信

| 决策 | 选择 | 理由 |
|---|---|---|
| 传输格式 | JSON over HTTP | 简单、可调试 |
| 轮询方式 | `setTimeout`（非 `setInterval`） | 避免请求堆积 |
| 结果获取 | 按需组装（非存储时组装） | schema 变更兼容 |
| 错误处理 | 5 次连续错误后退避停止 | 避免无限轮询 |

### 4.2 任务执行模型

```text
主进程 (FastAPI)
├─ JobRunner (ThreadPoolExecutor, max_workers=1)
│  └─ RunnerAdapter.run()
│     └─ _subprocess.run_module_subprocess()
│        └─ 子进程: python -m runners._worker
│           └─ Runner.execute()
│              ├─ expand_cases() → 多用例展开
│              ├─ 逐 case 执行
│              │  └─ ParallelRunner (ProcessPoolExecutor)
│              │     └─ CLI 仿真核心
│              └─ 返回 (records, skipped_hashes)
│
├─ 主进程收到 records
│  ├─ 计算 global rank
│  ├─ 持久化到 SQLite
│  └─ 通知前端（轮询感知）
│
└─ Case 级去重：cached_hashes → 跳过已缓存的 case
```

**关键约束**：

- 主进程不 import torch/tensor_cast（FastAPI 可独立启动）
- 子进程承载所有重依赖（torch, tensor_cast, serving_cast）
- 子进程可被 hard-kill（树形 kill）实现即时取消

### 4.3 多用例（Multi-case）展开

用户在表单中选择多个值时（如多设备、多量化策略），系统自动展开为独立 case：

```text
device: [A, B] × quantize: [W8A8, FP8] → 4 个 case

每个 case：

- 独立的 CLI 命令（日志中打印 per-case 命令）
- 独立的 case_hash（用于缓存去重）
- 独立的 case_log（per-case 日志片段）
- 结果在多用例视图中分组展示
```

### 4.4 表单 Schema 版本管理

```text
前端 config/forms/*.ts (source of truth)
→ gen-form-schemas.mjs 生成 data-only JSON
→ 后端 schema_registry 快照存储（SQLite）
→ 参数哈希绑定 schema 版本
→ 版本不匹配时拒绝提交（要求 bump version）
```

### 4.5 结果组件模块化

```text
useResultComponent.ts
├─ text_generate → TextGenerateResult / TextMultiCaseResult
├─ video_generate → VideoGenerateResult / VideoMultiCaseResult
└─ throughput_optimizer
   ├─ multi_case=true → ThroughputMultiCaseResult
   └─ multi_case=false → ThroughputOptimizerResult
      ├─ mode=aggregation → AggregatedView
      ├─ mode=disagg_* → DisaggregatedView
      └─ mode=pd_ratio → PDRatioView
```

**两个渲染入口**（容易遗漏）：

- `JobResult.vue`（深链接页）：传 `:result` + `:records` + `:form-schema`
- `ResultPane.vue`（工作区面板）：传 `:result` + `:job-id`

---

## 5. Gradio 问题解决状态

### 5.1 已解决

| Gradio 问题 | 解决方案 | 状态 |
|---|---|---|
| 文件膨胀 | 前后端分离，职责拆分 | ✅ |
| 位置参数传递 | TypeScript 接口 + Pydantic | ✅ |
| 状态管理分散 | Pinia + reactive composables | ✅ |
| PD Disaggregated 结果错误 | mode-aware SLO mask + 分相展示 | ✅ |
| 缺少任务中心 | History 页 + 轮询 + 状态管理 | ✅ |
| 缺少 Stop/Cancel | 树形 kill + cancel flag | ✅ |
| 安全边界模糊 | 后端本地绑定 + 子进程隔离 | ✅ |
| 空列噪音 | 后端组装时按 mode 选择字段 | ✅ |
| 缺少 CLI 命令预览 | JobCommandCard（per-case 命令） | ✅ |
| 缺少日志查看 | JobLogDrawer（ANSI 解析 + 搜索） | ✅ |
| 缺少散点图 | OptimizerCurves（ECharts scatter） | ✅ |
| 测试覆盖 | 1415 个 UT，后端 100% 覆盖率 | ✅ |

---

## 6. 前端设计模式

### 6.1 组合式 API（Composables）

每个 composable 封装一块业务逻辑，可跨组件复用：

```typescript
// useJobRunner: 任务生命周期
const runner = useJobRunner('throughput_optimizer')
runner.submit(params, version)  // 提交
runner.status                   // 'idle' | 'pending' | 'running' | 'succeeded' | ...
runner.result                   // 结果 envelope
runner.cancel()                 // 取消
```

### 6.2 配置驱动的动态表单

```text
config/forms/*.ts → SchemaForm → SchemaFormItem → Element Plus 控件

字段类型映射：
  text        → ElInput
  number      → ElInputNumber
  select      → ElSelect + ElOption
  multi-select → ElSelect (multiple)
  switch      → ElSwitch
```

### 6.3 主题系统

```css
/* CSS 变量 + color-mix 实现 light/dark */
:root {
  --msm-bg: #ffffff;
  --msm-bg-deep: #f5f5f5;
  --msm-text: #1e293b;
  --msm-accent: #3b82f6;
  /* ... */
}
[data-theme="dark"] {
  --msm-bg: #0f172a;
  /* ... */
}
```

ECharts 主题通过 `useChartTheme` composable 同步切换。

### 6.4 国际化

内联式双语（非 vue-i18n）：

```typescript
const { t } = useLocale()
t({ zh: '运行', en: 'Run' })
```

---

## 7. 后端设计模式

### 7.1 模块化 Runner

每个 CLI 入口对应一个 Runner 适配器，实现统一接口：

```python
class ThroughputOptimizerAdapter:
    def run(self, params, *, job_id, on_progress, cancel_flag, cached_hashes, form_schema_version):
        return run_module_subprocess("throughput_optimizer", params, ...)
```

### 7.2 Subprocess 隔离

```python
# 主进程 → 子进程 → CLI 核心
result = subprocess.run(
    [python, "-m", "runners._worker", module_id, params_path, result_path],
    cwd=web_backend_dir,
    stdout=PIPE, stderr=STDOUT,  # 合并流 → 日志
    creationflags=CREATE_NEW_PROCESS_GROUP,  # Windows: 树形 kill
)
```

### 7.3 Case 级去重

```python
# Worker 侧
for case_params in cases:
    case_hash = compute_case_hash(module_id, version, case_params)
    if case_hash in cached_hashes:
        skipped.append(case_hash)  # 跳过
        continue
    records = run_one_case(case_params)

# 主进程侧：克隆已缓存 case 的 records
```

### 7.4 Alembic 迁移管理

```python
# db.py: init_db 自动检测并迁移
def init_db(db_path=None):
    engine = get_engine(db_path)
    _adopt_legacy_db_if_needed(engine, env)  # 孤儿版本戳 → 重戳 head
    _run_alembic("upgrade", "head")           # 应用迁移链
```

---

## 8. 安全设计

### 8.1 后端约束

| 约束 | 实现 |
|---|---|
| 仅本地通信 | uvicorn 绑定 `127.0.0.1` |
| 命令白名单 | Runner 只允许 3 个 CLI 入口 |
| 参数白名单 | Pydantic schema 校验 + form_schema 版本检查 |
| 子进程隔离 | 每个任务在独立子进程中执行 |
| 树形取消 | `taskkill /T` (Windows) / `killpg` (POSIX) |

### 8.2 前端职责

前端只负责：

- 展示任务状态
- 减少误操作（按钮禁用、确认弹窗）
- 不承担安全拦截

---

## 9. 测试策略

### 9.1 后端测试

```text
tests/regression/web_ui/
├── test_result_view*.py      # 结果组装（Top-N parity + envelope）
├── test_routers_jobs.py      # API 路由
├── test_runners_*.py         # Runner 适配器
├── test_db.py                # 数据库 + 迁移
├── test_capture.py           # 日志捕获
├── test_ranking.py           # 排名
├── test_schema_registry.py   # Schema 版本管理
├── test_plugins_loader.py    # 插件发现
└── ...                       # 每个后端模块一个测试文件
```

**覆盖率**：1415 个测试，后端 **100%** 覆盖率（line + branch）。

### 9.2 测试模式

```python
# asyncio.run 驱动异步端点（不依赖 pytest-asyncio）
def test_endpoint():
    result = asyncio.run(handler(request=None))

# mock 外部依赖
with patch("runners._cli_command.build_cli_command_string", return_value="cmd"):
    result = _run(get_job("job-1", repo))

# 每个 source 模块一个测试文件，per-module --cov 避免 OOM
```

---

## 10. 构建与部署

### 10.1 Web 启动

首次启动前，需安装前端依赖（只需一次）：

```bash
cd web_ui/frontend && npm install
```

同时确保已安装 Python 依赖（详见 [安装指南](../zh/install_guide/msmodeling_install_guide.md)）：

```bash
uv sync  # 或 pip install -e .
```

然后在仓库根目录（venv 已激活）运行启动器：

```bash
python web_ui/main.py
```

启动器自动处理：

- 并发启动 `backend/main.py`（uvicorn，`:8000`）和 `npm run dev`（vite，`:5173`）
- 两路输出合并并添加 `[backend]` / `[frontend]` 前缀
- Ctrl+C 或任一进程退出时，清理整个进程树（POSIX：SIGTERM → 宽限期 → SIGKILL；Windows：`taskkill /T /F`）

### 10.3 数据库

```text
.msmodeling_ui/
├── msmodeling.db              # SQLite 主库（WAL mode）
└── logs/
    ├── {job_id}.log           # 任务日志
    └── cases/
        └── {case_hash}.log    # Per-case 日志
```

---

## 11. 演进路线

### P0（已完成）

- ✅ Vue 3 + FastAPI 架构搭建
- ✅ 三模块表单 + 结果组件
- ✅ 任务提交 + 轮询 + 取消
- ✅ 历史页 + 深链接结果页
- ✅ 日志抽屉 + ANSI 解析
- ✅ 散点图（Throughput vs Concurrency / TPOT）
- ✅ 后端 100% 覆盖率

### P1（已完成）

- ✅ Toast 通知（任务提交反馈）
- ✅ CLI 命令 per-case 日志
- ✅ 视频结果表格去重）

---

## 12. 风险与约束

| 风险 | 应对 |
|---|---|
| CLI 参数变更 | form_schema 版本检查 + hash 拒绝 |
| 子进程 OOM | 树形 kill + 错误记录 |
| 大量并行 worker Windows 页面文件耗尽 | 可配置 `--jobs` + 错误诊断提示 |
| 数据库迁移失败 | 孤儿版本戳自动重戳 + alembic 回滚 |
| 表单 schema 与 CLI 不同步 | gen-form-schemas 构建步骤 + 版本绑定 |

---

## 13. 附录

### 13.1 Gradio → Vue 3 迁移映射

| Gradio 文件 | Vue 3 / FastAPI 对应 |
|---|---|
| `app.py` | `App.vue` + `Console.vue` + `main.py` |
| `callbacks.py` | `useJobRunner.ts` + `api/routers/jobs.py` |
| `command_builder.py` | `runners/_cli_command.py` |
| `runner.py` | `services/job_runner.py` + `runners/_subprocess.py` |
| `parsers.py` | `services/result_view.py` + `runners/*/execute()` |
| `result_store.py` | `services/repositories.py` + SQLite |
| `schemas.py` | `models/entities.py` + `models/orm.py` |
| `charts.py` | `components/result/ChartWrapper.vue` + ECharts |
| `components.py` | `components/**/*.vue` |
| `styles.py` | `styles/theme.css` |

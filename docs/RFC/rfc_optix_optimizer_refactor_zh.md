# RFC: optix 服务参数寻优可靠性增强

## 元数据

| 项目 | 内容 |
|:-----|:--------|
| **状态** | Draft |
| **作者** | msmodeling community |
| **创建日期** | 2026-07-03 |
| **最后更新** | 2026-07-03 |
| **相关链接** | PR #518（待合并） |

---

## 1. 概述

### 1.1 背景与问题

optix（`msmodeling optix`）通过 PSO 粒子寻优自动调节 MindIE / VLLM 推理服务参数，并在真实或仿真环境下执行压测以评估 fitness。随着工具在真机场景中的使用加深，用户在排障、自动化集成和长时间寻优任务中暴露出以下痛点：

| 痛点 | 说明 |
|------|------|
| 静默失败 | 部分配置或评测错误未以明确退出码反馈，脚本难以判断任务是否成功 |
| 日志难关联 | 控制台日志缺少稳定的 run 上下文，多轮寻优或并行任务时难以对齐 CSV 与控制台输出 |
| 评测结果歧义 | 压测输出目录异常（如 AISBench CSV 数量不为 1）时，错误信息不清晰 |
| 可维护性 | 调度、寻优与持久化逻辑耦合较深，回归测试覆盖不足 |

### 1.2 目标

本提案对 `optix/optimizer` 进行可靠性、可观测性与 CLI 语义增强，不改变用户侧插件注册与 `config.toml` 主流程。

**核心目标**：

- 结构化日志：控制台输出包含 `run_id`、`stage`、`engine` 等上下文字段，支持 `OPTIX_LOG_LEVEL` 分级
- fail-fast：配置缺失、非法 TOML、无可行解等场景以非零退出码或明确异常结束
- 评测健壮性：压测结果解析失败时立即报错，避免继续访问无效数据
- 测试与文档：补充回归与冒烟测试，同步用户指南与设计文档变更记录

**非目标**：

- 不改动 `serving_cast` 下另一套 optimizer
- 不重写 PSO 算法或插件对外 HTTP 契约
- 不在本次实现 sim/bench 子进程独立日志文件 sink
- 不迁移 stdlib `logging`；loguru 继续作为 optix 全局日志后端

### 1.3 核心价值

- **可观测**：单次寻优 run 的日志可通过 `run_id` 与 CSV 时间戳关联，降低排障成本
- **可集成**：CI / 脚本可通过退出码区分成功、配置错误与寻优失败
- **可信赖**：边界错误尽早暴露，减少「进程正常退出但无有效结果」的误报

---

## 2. 用例分析

### 2.1 寻优主流程

用户执行 `msmodeling optix -e vllm -b ais_bench`，工具完成 baseline 建立、PSO 搜索、可选 fine-tune，并输出最优参数 CSV。

| 维度 | 要求 |
|------|------|
| 功能 | CLI 解析、插件加载、Scheduler 启停服务与压测、DataStorage 落盘 |
| 性能 | 搜索成本与 `n_particles × iters` 近似线性；`sample_size` 可加速初筛 |
| 可靠性 | 单粒子评测失败记 `fitness=inf` 并继续；全局无可行解时明确失败 |
| 可测试性 | 核心路径可通过 mock simulator/benchmark 做回归测试 |

### 2.2 插件作者

自定义 simulator / benchmark 插件在寻优过程中被 Scheduler 调用；插件内使用全局 `logger` 时，应自动继承当前 run 的 `run_id` 与 `stage`。

### 2.3 运维排障

运维人员通过 `OPTIX_LOG_LEVEL=DEBUG` 获取参数细节与完整异常栈；通过退出码与 CSV `error` 列定位失败候选。

---

## 3. 方案设计

### 3.1 总体方案

```text
msmodeling optix (CLI)
        │
        ▼
optimizer.main()
  ├─ configure_logger()          # OPTIX_LOG_LEVEL / 兼容 MODELEVALSTATE_LEVEL
  ├─ logger.contextualize()    # run_id, engine, stage 透传全局 logger
  ├─ prepare_plugin / baseline
  ├─ PSOOptimizer.run_plugin()
  │     └─ Scheduler → Simulator + Benchmark + HealthCheck
  └─ DataStorage 持久化
```

**Run 上下文**：`main()` 入口通过 loguru `contextualize` 设置 `run_id` 与 `engine`；内层阶段（如 `search`、`evaluate`）嵌套 `contextualize(stage=...)`，保证 `op_func`、`Scheduler` 等使用全局 `logger` 的模块均可输出一致上下文。

**错误出口**：

| 场景 | 行为 |
|------|------|
| 自定义 config 文件不存在 | 抛出 `ConfigFileNotFoundError` |
| 非法 TOML | 抛出 `InvalidConfigError` |
| `run_plugin()` 无可行解 | 抛出 `NoFeasibleSolutionError` |
| baseline 服务启动失败 | 抛出 `BaselineRunError`（`from_scheduler` 生成可读 command/log 消息） |
| 未捕获异常 | `_main()` 在 `contextualize` 内 `logger.exception` 一次后以 `SystemExit(1)` 退出 |
| 压测 CSV 数量异常 | `BenchmarkResultError` 立即抛出 |

**边界 log-once**：Scheduler / 子进程层仅 `logger.debug` 记录诊断细节并 `raise`；`OptimizerError` 与未捕获异常仅在 `_main()` 的 `run_id` 上下文中打一条 `ERROR` / `exception()`。

**`configure_logger`**：仅替换 loguru 默认 stderr handler（`remove(0)`），不清除用户或插件预先注册的其他 sink。

### 3.2 技术选型（未采用方案）

| 方案 | 不采用原因 |
|------|------------|
| 向 Scheduler / PSO 传递 bound Logger 实例 | 侵入面大，与现有全局 logger 用法不一致 |
| `logger.remove()` 清空全部 handler | 破坏插件预注册 sink，与 docstring 不符 |
| 迁移 stdlib logging | 改动面过大，与现有 loguru 生态不一致 |

### 3.3 安全、隐私与 DFX

| 属性 | 设计 |
|------|------|
| 兼容性 | 保留 `MODELEVALSTATE_LEVEL`；未设置 `OPTIX_LOG_LEVEL` 时作为回退 |
| 可测试性 | 回归目录 `tests/regression/optix/test_optimizer/` + 冒烟 `tests/smoke/test_optix_optimizer_smoke.py` |
| 可维护性 | 日志阶段枚举 `LogStage` 集中定义；Scheduler 健康检查仍只 raise、不重复打日志 |
| 可靠性 | 单粒子失败降级为 warning；全局失败 fail-fast |

### 3.4 编程与调用设计

#### 环境变量

| 变量 | 说明 |
|------|------|
| `OPTIX_LOG_LEVEL` | 首选日志级别：`INFO`（默认）、`DEBUG`、`TRACE` 等 |
| `MODELEVALSTATE_LEVEL` | 历史兼容；仅当未设置 `OPTIX_LOG_LEVEL` 时生效 |

#### 控制台格式

`INFO` 及以上（默认）：

```text
HH:mm:ss | LEVEL    | run_id | stage | message
```

`DEBUG` / `TRACE` 额外包含源码位置列：

```text
HH:mm:ss | LEVEL    | run_id | stage | file:line | message
```

| 列 | 说明 |
|----|------|
| `run_id` | 单次 CLI 运行的 8 位十六进制 ID；未进入 run 上下文时为 `-` |
| `stage` | 当前阶段（见 LogStage）；未绑定时为 `-` |
| `file:line` | 仅 DEBUG/TRACE；日志调用点文件名与行号 |

#### CLI 退出码

| 退出码 | 含义 |
|--------|------|
| `0` | 找到可行最优解并完成输出 |
| `1` | `OptimizerError`（config 缺失、无可行解等）或未捕获致命错误；`main()` 边界记录后以 `SystemExit(1)` 退出 |

#### LogStage（用户可见 stage 列）

| stage | 含义 |
|-------|------|
| `init` | CLI 启动与配置加载 |
| `baseline` | 默认服务与压测建立 |
| `search` | PSO 主循环 |
| `fine_tune` | 候选二次调优 |
| `evaluate` | 单次 sim + bench |
| `done` | 任务正常结束 |

---

## 4. 测试设计

- **单元 / 回归**：mock simulator、benchmark 与 Scheduler，覆盖 CLI 分支、日志上下文传播、`configure_logger` 行为、压测解析与 PSO 边界
- **冒烟**：最小 mock 栈验证 `main()` 可完成一轮寻优编排
- **端到端**：在具备 NPU、MindIE/vLLM 与压测工具的环境中执行真实寻优（现有 E2E 策略不变）

验收标准：optix 回归套件与冒烟测试通过；用户指南与 design 变更记录与实现一致。

---

## 5. 风险与应对

| 风险 | 应对 |
|------|------|
| 插件依赖 loguru 预注册 handler | `configure_logger` 使用 `remove(0)` 仅替换默认 handler |
| Scheduler 行为变更引入回归 | 回归测试锁定 baseline、重试、健康检查路径 |
| 日志级别变更影响用户习惯 | 默认仍为 `INFO`；文档说明 `OPTIX_LOG_LEVEL` 与 `MODELEVALSTATE_LEVEL` 优先级 |

---

## 7. 未解决问题

- `RunOutcome` 结构化错误对象替代 `scheduler.error_info` 字符串的时机
- `optix/optimizer/utils.py` 模块拆分与职责边界

---

## 附录

### 参考资料

- [optix 用户指南（中文）](../zh/user_guide/optix_user_guide.md)
- [optix 用户指南（英文）](../en/user_guide/optix_user_guide.md)
- [设计文档：服务参数寻优](../design/optix-service-parameter-optimizer-design.md)

### 术语表

| 术语 | 含义 |
|------|------|
| optix | msmodeling 服务参数自动寻优 CLI |
| PSO | 粒子群优化算法 |
| run_id | 单次寻优运行的短标识，出现在日志与排障上下文中 |

### 文档更新计划

| 文档 | 变更 |
|------|------|
| design doc | 保持历史基线；末尾增加变更记录并链接本 RFC |
| user guide | 补充 `OPTIX_LOG_LEVEL`、退出码与排障说明 |
| plugin 开发指南 | 说明插件日志自动继承 run 上下文（如有需要） |

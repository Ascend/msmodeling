# Lessons Learned — msmodeling AI 协作经验池

本文件是 msmodeling AI agent 跨会话共享的**经验教训库**。每次 agent 在执行任务中遇到非显而易见的问题、发现项目约定、踩到坑，都应追加一条记录，供后续会话复用。

> **性质**：团队共享，随代码版本管理（`[commit]`）。与个人会话上下文（`INDEX.md`，`[ignore]`）区分。

---

## 1. 维护规则

### 1.1 何时追加

agent 在以下情况应追加一条 lesson：

- **踩坑**：耗时才解决的问题（非显而易见的）
- **项目约定**：发现文档化的或隐式的项目约定，AI 容易违反
- **反幻觉**：发现 agent 基于错误假设生成了内容
- **工具陷阱**：发现 pre-commit / CI / skill 脚本的隐藏行为
- **流程改进**：发现某类任务有更优执行路径

### 1.2 格式

```markdown
## {分类}
N. **{一句话标题}** — {详细说明，含命令/路径/错误信息}
   - 触发场景：{什么情况下会遇到}
   - 正确做法：{应该怎么做}
   - 引用：{issue / PR / commit / spec / 文档路径}
```

### 1.3 编号

- 全局连续编号，不重排
- 删除某条时保留编号空洞，标注 `deleted`

### 1.4 消费方式

agent 会话启动时，应：

1. 读本文件全文（当前规模小，全量读取）
2. 读当前任务的 skill SKILL.md
3. 执行任务时，若场景匹配某条 lesson，主动应用正确做法
4. 任务结束后，若有新教训，追加到本文件并 commit

---

## 2. 分类索引

| 分类 | 条目范围 | 说明 |
|------|------|------|
| 代码架构 | 1-4 | DeviceProfile / CommGrid / 上游依赖 / 性能模型 |
| 新模型适配 | 5 | model-adaptation 标准流程 |
| 文档与北向接口 | 6 | 接口变更同步文档 |
| Skill 治理 | 7-8 | 结构树 / frontmatter |
| Agent 可靠性 | 9 | 幻觉规避 |
| 测试与 CI | 10-11 | 虚拟环境 / 本地门禁 |

---

## 代码架构

### 1. **`DeviceProfile` name 必须唯一**

`tensor_cast/device.py` 的 `DeviceProfile.__post_init__` 会自动将 profile 注册到类变量 `all_device_profiles`，重复 name 抛 `ValueError`。

- 触发场景：新增设备 profile 时用了已存在的 name
- 正确做法：写入前先检查是否已存在同名 profile；不确定时优先写入 `tensor_cast/device_profiles/*.py`（用户自定义 profile）而非 `device.py`
- 引用：[AGENTS.md 代码架构注意事项 §1](../../AGENTS.md)

### 2. **`CommGrid` 约束**

`grid.ndim == len(topologies)`，每个 grid 维度至少为 2，`topologies` 的 key 是 `start_dim`（从 0 开始），不是任意层级编号。

- 触发场景：构建通信拓扑网格时
- 正确做法：严格按上述约束构造，违反会导致仿真结果错误或断言失败
- 引用：[AGENTS.md 代码架构注意事项 §2](../../AGENTS.md)

### 3. **不直接修改 vLLM/上游依赖**

TensorCast 引用上游模型代码时，必须通过适配层而非直接改上游。

- 触发场景：适配新模型时需要修改上游模型行为
- 正确做法：通过 `tensor_cast/transformers/builtin_model/` 中的 wrapper 或 patch 层适配；模型特定行为优先通过 composition 而非 monkey-patching 实现
- 引用：[AGENTS.md 代码架构注意事项 §3](../../AGENTS.md)

### 4. **不在 hot path 中使用 `tensor.item()`**

`tensor.item()` 会触发 CPU-NPU 同步，显著降低仿真性能。

- 触发场景：仿真热路径中取标量值
- 正确做法：用张量运算替代；确需取值时移到非热路径
- 引用：[AGENTS.md 代码架构注意事项 §4](../../AGENTS.md)

---

## 新模型适配

### 5. **新模型适配必须遵循 model-adaptation 标准流程**

适配新模型时不得凭模型名臆造字段或 patch。

- 触发场景：接入新 HuggingFace 模型到 TensorCast
- 正确做法：遵循 [model-adaptation skill](../../.agents/skills/model-adaptation/SKILL.md) 的标准 Workflow（确定性工具 + 人工审视），具体包括：不凭模型名臆造 `ModelProfile` 字段；不基于臆测写 patch method，必须基于 failure log 和已安装模型源码；`evidence.yaml` 必须导出 `evidence_draft` 后审视，不手写；私有路径、本地虚拟环境路径、临时 walkthrough 不进 commit
- 引用：`.agents/skills/model-adaptation/SKILL.md` Core Rule

---

## 文档与北向接口

### 6. **北向接口变更必须同步更新文档**

用户可见行为的任何变更，必须同 PR 更新对应文档。

- 触发场景：修改 CLI 参数、skill 触发词、默认行为、输出格式、配置项
- 正确做法：同 PR 更新 `README.md`、`docs/` 下对应 user_guide、AGENTS.md 的 skill 触发词表、CONTRIBUTING.md 的流程说明；漏更文档会导致用户和 AI agent 基于过期文档操作
- 引用：[CONTRIBUTING.md PR 规范](../../CONTRIBUTING.md)、[AGENTS.md Skills 体系](../../AGENTS.md)

---

## Skill 治理

### 7. **AGENTS.md 项目结构树易过期**

项目结构树是手工维护的，新增顶层目录或 skill 后常忘记同步。

- 触发场景：新增 `spec/`、`.loop/`、`optix/`、`web_ui/` 等目录或新 skill 后
- 正确做法：任何新增顶层目录或 skill 的 PR，必须同时更新 AGENTS.md 项目结构树
- 引用：[AGENTS.md 项目结构](../../AGENTS.md)

### 8. **skill frontmatter 必须用 `metadata.version` + `metadata.source`**

AGENTS.md 规定了标准 frontmatter 格式，但历史上存在顶层写法、metadata 嵌套、完全缺失三种并存。

- 触发场景：新增或修改 skill 的 SKILL.md
- 正确做法：统一用 `metadata.version` + `metadata.source` 嵌套写法，遵循 [Skill 开发规范](../../AGENTS.md)
- 引用：AGENTS.md「Skill 开发规范」章节

---

## Agent 可靠性

### 9. **agent 可能基于不存在的 issue/PR 编造内容**

被要求"检视 PR N"但 PR N 不存在时，agent 可能不报错而是编造 diff 内容。

- 触发场景：sig-review 检视不存在的 PR；分析不存在的 issue
- 正确做法：sig-review 流程开始前应确认 PR 真实存在（`review_api.py fetch` 返回非空）；agent 不应假设用户提供的编号一定有效，查不到时明确报错而非继续
- 引用：`.agents/skills/sig-review/SKILL.md`

---

## 测试与 CI

### 10. **运行 pytest 前必须先启动虚拟环境**

直接用系统 `pytest` 会缺依赖报错。

- 触发场景：本地运行单元测试或冒烟测试
- 正确做法：先 `source .venv/bin/activate`（Windows 用 `.venv\Scripts\activate`），或用 `uv run pytest` 前缀自动使用项目环境；首次需 `uv sync --group ci` 同步测试依赖
- 引用：[CONTRIBUTING.md §3 搭建开发环境](../../CONTRIBUTING.md)、[CONTRIBUTING.md §5 本地测试与检查](../../CONTRIBUTING.md)

### 11. **PR 推送前必须本地通过 pre-commit + 相关 UT/ST**

CI 门禁会重跑这些检查，本地不通过直接推会浪费 CI 资源并阻塞评审。

- 触发场景：准备推送 PR
- 正确做法：运行 `uv run pre-commit run --all-files`（代码风格、拼写、安全检查）；运行 `uv run pytest tests/regression/<对应模块>/`（相关单元测试）；如涉及集成运行 `uv run pytest tests/smoke/`
- 门禁详情见 CONTRIBUTING.md §5 与 §8
- 引用：[CONTRIBUTING.md §5 本地测试与检查](../../CONTRIBUTING.md)、[CONTRIBUTING.md §8 门禁流水线与 CI](../../CONTRIBUTING.md)

---

## 变更日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-07-21 | v0.1 | 初始化，含 11 条教训（代码架构 4 + 新模型适配 1 + 文档与北向接口 1 + Skill 治理 2 + Agent 可靠性 1 + 测试与 CI 2） |

---
name: optix-assistant
description: optix 寻优统一入口。当用户需要初始化或修改 config.toml、推荐 MindIE/vLLM 寻优参数与搜索范围、执行 PSO 参数寻优、或使用 agent 模式闭环调优时使用。触发词：寻优、参数推荐、生成寻优范围、配置 config.toml、config_writer、agent 模式寻优、推荐 optix 参数。
metadata:
  version: 0.1.0
  source: https://gitcode.com/Ascend/msmodeling/pull/778
  status: active
---

# Optix Assistant — 寻优统一入口

## 定位

msmodeling optix 寻优的唯一用户入口，覆盖三种意图：

| 模式 | 用户意图 | 流程文档 |
|---|---|---|
| **config 模式** | 只想初始化/修改 config.toml，不跑寻优 | 本节「入口」→ config 分支（流程内联，无独立文档） |
| **PSO 模式** | 用 optix 内置 PSO 做参数寻优 | 共享前置 → `references/pso-mode.md` |
| **agent 模式** | AI agent 逐轮生成候选的闭环调优 | 共享前置 → `workflows/agent.md` + `references/agent-mode.md` |

三种模式共享同一条前置链（见下「共享前置流程」），差异只在执行分支。

## 旧 Skill 关系（已归并删除）

下列旧 skill 的能力已并入本 skill，目录已删除、不再维护（路由见 AGENTS.md，仅此单一入口）：

- `optix-param-recommend` — 推荐能力已并入 PSO 分支（并吸收 agent 模式沉淀的经验资产）
- `optix-config` — 写入能力已并入（`config_writer.py`）
- `agent-optimizer` — 循环流程已并入 agent 分支（并吸收其 presets / experience_injector / known_patterns）

## 入口：意图识别与模式选择

**第一条消息必须完成模式判定，禁止默认选模式。**

1. 意图明确时直接进入对应模式：
   - "改下 config.toml / 加个搜索参数 / 配置 benchmark" → config 模式
     （流程：0a 确认 benchmark（如涉及）→ `config_writer.py --set/--target-field` 字段操作
     写后 `--check-only` 复检退出 0 → 结束）
   - "寻优 / 跑一轮参数优化" → 先问 PSO 还是 agent（用户不了解区别时给一句话对比：PSO 快、全自动、无 agent 成本；agent 慢、可解释、能吸收经验做方向决策）
   - "推荐参数/范围"（明确不跑）→ 按 PSO 分支只输出推荐，不启动 optix
2. 意图模糊（如"帮我调优服务"）→ 列出三种模式让用户选。
3. 选定 agent 模式后，再按 `workflows/agent.md` 选择"全自动/交互式"。

## 共享前置流程（两种寻优模式强制，config 模式跳过采集/注入）

> 本文件只维护**共享骨架**（步骤链与归属边界）。每步的完整命令、问询清单、回填细则由
> 对应分支文档与脚本 `--help` 承担，不在本文件展开——避免同一信息三处维护（#17/#18）。

| 步骤 | 做什么 | 归属 / 详情 |
|---|---|---|
| **0a. 定 benchmark（必问，禁猜）** | `vllm_benchmark` / `ais_bench` 二选一，无偏好默认 `ais_bench` | `references/benchmark.md` |
| **1. 配置就绪** | 发现 config.toml → 完整性检查 → 缺失项问用户回填 → 复检退出 0 → 生效配置确认表 | PSO：`config_writer.py --check-only`；agent：`config_preflight.py --check-only/--set-*`（agent 专属门禁，见 `workflows/agent.md`「配置就绪」） |
| **2. 上下文采集与知识注入** | `collect_context.py` → `experience_injector.py`（产出 experience_report.md，**两模式必读**）→ `check_search_space.py` | 脚本 `--help`；命中 presets 先 `apply_launch_config.py`（env：`source launch_env.sh`，轮次后 `source unset_env.sh`） |
| **3. 进入分支** | PSO → pso-mode.md；agent → agent.md + agent-mode.md（两文件进入 agent 模式前都须完整阅读） | — |

> **共享护栏（两种寻优模式强制）**：卡数 / 模型路径与服务模型名 / ais_bench `models`·`datasets`
> 短名 / 输入输出长度——**必须先问用户，禁止猜值、静默套默认或自行探测**；用户不知道且同意后才允许
> 脚本探测（`--discover` / `ais_bench --search`）。写入一律走脚本（`config_writer.py` 写搜索参数，
> `config_preflight.py --set-*` 只写 `[agent_optimizer]` 运行节），**禁止手写 TOML**。

## 完成标准

- config 模式：config.toml 通过 TOML 校验 + config_writer --check-only 退出 0（preflight 为 agent 模式门禁，config 模式不执行）
- PSO 模式：optix 正常启动且消费写入的搜索空间
- agent 模式：optix agent 模式跑通至少一轮并写回 results

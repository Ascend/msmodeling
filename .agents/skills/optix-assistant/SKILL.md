---
name: optix-assistant
description: optix 寻优统一入口。当用户需要初始化或修改 config.toml、推荐 MindIE/vLLM 寻优参数与搜索范围、执行 PSO 参数寻优、或使用 agent 模式闭环调优时使用。agent 模式的候选生成与失败恢复必须结合 knowledge 知识库（方法卡路由 + 组合合法性校验）。触发词：寻优、参数推荐、生成寻优范围、配置 config.toml、config_writer、agent 模式寻优、推荐 optix 参数。
metadata:
  version: 0.2.0
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
| **agent 模式** | AI agent 逐轮生成候选的闭环调优 | 共享前置 → `workflows/agent.md`（执行按 workflows/ 下阶段流程文件编排） |

agent 多机混部使用原 `multihost_infer` 插件；进入该场景先读 [多机约定](references/multihost.md)。

三种模式共享同一条前置链（见下「共享前置流程」），差异只在执行分支。

## Agent 决策与参数来源

> **前置**：`knowledge/` 是 git submodule，内容承载于 `Ascend/msmodeling.wiki` 仓 main 分支。
> Wiki 同时承载设计文档；为避免把与 OptiX 无关的文档拉入 agent 上下文，禁止直接初始化该
> submodule。若目录为空，在本仓根目录执行
> `python .agents/skills/optix-assistant/scripts/init_knowledge.py`。该入口使用 partial clone 和
> sparse checkout，只检出 OptiX 知识目录，并严格对齐主仓锁定的 Wiki commit。

`knowledge/` 参与提方向、选参数和失败诊断。方法卡提供机制、适用前提、组合约束与推荐起点；
案例提供配置参考，运行结果提供实测依据，均不能替代方法正文的适用性核对。

- 部署并行策略未定（新模型接入、卡数/硬件变更、重新评估并行形态）时，先按 `knowledge/modeling-knowledge/SKILL.md` 读取 [TP/DP/EP](knowledge/modeling-knowledge/methods/parallelism/tp-dp-ep.md) 确定可行拓扑，再按目标路由的“无基线”分支建立基线。
- Round 0 的 `context.json` 只记录环境事实、负载、约束和配置来源，不生成 `search_space`；`source.config_toml` 指向基础配置。
- 首次候选执行前由 agent 建立 `context.search_space`，后续每轮按方法卡和实测依据维护，候选 `params[name] = value`；未赋值参数继承空间默认值。新增、修改、删除及其依据写入 `directions[].candidates_analysis`，具体规则见 `workflows/data-analysis.md`「每轮搜索空间」。
- **候选生成前**：按 `workflows/data-analysis.md`「候选生成规则」与 `knowledge/modeling-knowledge/SKILL.md` 走完整路由（诊断先行 → 前提探测 → goals 路由 → 方法索引过滤 → relationships 组合校验 → 读方法卡正文和参数推荐章节）；首轮无实测时走无基线分支。
- Round 0 先收集环境、用户约束与 `knowledge/case-knowledge` 案例，按方法卡核对后集中写入 TOML 基础配置，再采集 context 并复检。案例不受已有参数名单限制；优先级与冲突裁决见 `workflows/data-analysis.md`「知识参考优先级」。
- Agent 模式跳过 vendor preset 注入，官方案例只走 case-knowledge；不把 presets JSON 中的值自动装配为搜索参数。
- **失败恢复时**：按 `workflows/execution-validation.md`「失败分类」与 `workflows/agent.md`「失败恢复」的升级契约进入 `knowledge/modeling-diagnosis/`。

**硬规则**：

1. 提方向前先查 `knowledge/modeling-knowledge/relationships.md` §1 依赖 / §2 组合限制，
   组合合法性一票否决，**不要用 trial 验证知识库已明确禁止的组合**。
2. 推荐值只作起点：采纳前读取对应方法正文，核对适用前提和组合约束；
   没有可用案例不代表方法不可用，继续按 modeling-knowledge 路由选择方法。
3. 引用必须可溯源：只有实际读取方法正文后才引用对应卡路径，读案例或推荐值不等于读过方法卡。
   运行结果也须引用实际读取的文件。
   每轮 `directions` 的 `evidence_analysis`、`refs_analysis`、`candidates_analysis` 均须为非空字符串，
   顶层 `routing_review` 须记录瓶颈假设、命中的机制层及暂缓理由，分析要求见 `workflows/data-analysis.md`「本轮分析记录」。执行前用
   `scripts/validate_candidates.py --run-dir <run-dir> --round N` 校验，缺少分析、routing_review 不完整或解析不到文件的引用会被拒。
   进程退出后、修改下一轮配置前，将 `context.json` 复制为 `context.round-N.json`，保留工作文件；
   归档规则见 `workflows/execution-validation.md`。

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

## 共享前置流程（两种寻优模式强制，config 模式跳过采集）

> 本文件只维护**共享骨架**（步骤链与归属边界）。每步的完整命令、问询清单、回填细则由
> 对应分支文档与脚本 `--help` 承担，不在本文件展开——避免同一信息三处维护（#17/#18）。

| 步骤 | 做什么 | 归属 / 详情 |
|---|---|---|
| **0a. 定 benchmark（必问，禁猜）** | `vllm_benchmark` / `ais_bench` 二选一，无偏好默认 `ais_bench` | `references/benchmark.md` |
| **1. 配置就绪** | 发现 config.toml → 收集配置依据、补齐缺失项 → 写入并预检退出 0 → 生效配置确认表 | PSO：`config_writer.py --check-only`；agent：收集环境与案例，按方法卡核对后集中写入基础配置和运行节（`config_writer.py` + `config_preflight.py`，见 `workflows/environment-preparation.md`） |
| **2. 上下文采集与配置检查** | `collect_context.py` → `check_search_space.py` | agent 的案例起点、参数登记和首轮候选见 `workflows/environment-preparation.md`；PSO：命中 presets 先 `apply_launch_config.py`，运行前 `source launch_env.sh`，结束后 `source unset_env.sh`（见 `references/pso-mode.md`） |
| **3. 进入分支** | PSO → pso-mode.md；agent → workflows/agent.md（进入 agent 模式前完整阅读；阶段细节与失败契约随 workflows/ 下流程文件阅读） | — |

> **共享护栏（两种寻优模式强制）**：卡数 / 模型路径与服务模型名 / ais_bench `models`·`datasets`
> 短名 / 输入输出长度——**必须先问用户，禁止猜值、静默套默认或自行探测**；用户不知道且同意后才允许
> 脚本探测（`--discover` / `ais_bench --search`）。写入一律走脚本（`config_writer.py` 写搜索参数，
> `config_preflight.py --set-*` 只写 `[agent_optimizer]` 运行节），**禁止手写 TOML**。

> **关键契约（两种寻优模式通用，headless 模式强制）**：若任务消息**已给出完整 benchmark
> 负载参数**（输入长度、输出长度、请求数 + 并发——并发可以是固定值，吞吐优先场景也可以是
> 范围 `lo~hi`）——
> 1. 无需再向用户确认（消息已给 = 已确认）；
> 2. 必须经 `collect_context --input-len/--output-len/--num-requests` + 并发旗标
>    （固定 `--concurrency`，可调 `--concurrency-min/--concurrency-max`）登记进
>    `context.json` 的 `benchmark_spec`（唯一事实源）；
> 3. ais_bench models/datasets 配置、共享 `synthetic_config.py` 真源、config.toml 的
>    others 与 CONCURRENCY target_field 必须按 spec 核对/重写（固定并发 → CONCURRENCY 钉死
>    `min==max==value`；并发范围 → CONCURRENCY 给 `min/max` 进搜索空间；短名建议带参数指纹，
>    如 `{model_abbr}_c{并发}` / `optix_synthetic_{输入}_{输出}`），
>    **禁止静默复用环境现存配置**；
> 4. 写后必须 `config_preflight --check-only --context <context.json>` 做 benchmark 一致性校验
>    （config.toml + ais_bench 配置 vs `benchmark_spec`），错配直接拦截，不进入第一个 trial。

## 完成标准

- config 模式：config.toml 通过 TOML 校验 + config_writer --check-only 退出 0（preflight 为 agent 模式门禁，config 模式不执行）
- PSO 模式：optix 正常启动且消费写入的搜索空间
- agent 模式：optix agent 模式跑通至少一轮并写回 results

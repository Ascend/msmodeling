# RFC: `text-generate-executor` Skill 设计方案

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | Draft |
| **作者** | lutean |
| **创建日期** | 2026-05-30 |
| **更新日期** | 2026-06-03 |
| **相关链接** | 无 |

---

## 1. 概述

本 RFC 提议新增 `text-generate-executor` Skill，用于将用户关于 `python -m cli.inference.text_generate` 的自然语言验证诉求，转换为可执行、可复核、可确认的 CLI 命令，并在用户确认后运行仿真、整理关键性能指标与失败原因。

该 Skill 面向“已有具体模型、硬件、batch、序列长度、并行策略，需要验证单个配置”的场景。它不负责搜索最优部署策略，而是补齐 `throughput_optimizer` 推荐结果到 `text_generate` 单点验证之间的操作链路。

## 2. 目标与非目标

目标：

- 支持用户通过对话生成 `python -m cli.inference.text_generate` 命令。
- 支持 Prefill-only、Decode、optimizer best-row validation、profiling、trace/debug 等常见验证路径。
- 在执行前展示参数摘要、假设、校验结果和完整命令，并要求用户显式确认。
- 执行后总结模型、设备、输入配置、并行策略、性能模型、延迟/吞吐/内存等关键指标。
- 失败时给出最小修正建议，避免让用户重新填写完整参数。

非目标：

- 不修改 `cli.inference.text_generate` 的业务逻辑或参数语义。
- 不替代 `throughput-optimizer-executor` 做 TP/EP/MOE-DP 搜索。
- 不承诺自动解析所有 stdout 格式；首版以关键指标摘要和错误分类为主。
- 不在 Skill 内固化硬件 profile 列表，必要时从当前仓库代码或 `--help` 读取。
- 不自动下载模型或安装环境依赖；环境问题交由现有环境安装流程处理。

## 3. 用例分析

| 用例 | 用户输入示例 | Skill 行为 | 输出 |
|:---|:---|:---|:---|
| Prefill 验证 | “帮我跑 Qwen 7B，A3，8 卡，输入 4k，batch 16” | 补齐 `model_id`、`device`、`num-devices`、`num-queries`、`query-length`，默认不加 `--decode` | 可确认的 prefill-only 命令与执行摘要 |
| Decode 验证 | “验证 decode，context 4096，batch 32” | 添加 `--decode`，要求 `--context-length`，保留 query length 语义 | decode 场景指标摘要 |
| Decode MTP 验证 | “decode 开 MTP，mtp tokens=2” | 添加 `--decode` 和 `--num-mtp-tokens 2`，并设置 `--query-length 3` | MTP decode 指标摘要和 forward 时间口径说明 |
| Prefix Cache 验证 | “prefill 开 prefix cache hit rate 0.8” | 仅 Prefill-only 阶段询问和添加 `--prefix-cache-hit-rate`，Decode 阶段不询问该参数 | Prefill cache 命中率仿真摘要 |
| Optimizer 推荐行复验 | “把 throughput_optimizer 最优行转 text_generate 跑一下” | 将 optimizer row 中的 batch/concurrency、TP/DP/EP/MOE-DP、输入输出长度映射为固定验证命令 | 单点配置验证结果 |
| Profiling 模式 | “用 profiling database 跑 text_generate” | 要求 `--performance-model profiling` 与 `--profiling-database` | profiling 命令与命中/缺失相关信息 |
| Debug/Trace | 用户未必知道 trace/debug 参数 | 必要参数交互完成并生成候选命令后，固定告知 `--chrome-trace`、`--graph-log-url`、`--dump-input-shapes` 的功能和限制，让用户选择是否追加；若用户主动要求 trace/debug，则直接进入对应选项确认 | 更新后的命令、trace/graph log 路径和执行状态 |

DFX 要求：

- 兼容性：Skill 只生成 CLI 命令，不改变现有代码路径。
- 可维护性：参数说明拆分到 `references/text-generate-params.md`，对话流程拆分到 `references/dialog-flow.md`。
- 可测试性：可通过示例 prompt 验证是否能生成稳定命令。
- 可靠性：执行前必须展示完整命令和关键假设，避免隐式修改量化、profiling 等参数；`--compile` 作为默认项需要显式展示，用户可要求关闭。
- 上下文隔离：除非用户明确要求沿用上一轮配置，否则每次新诉求都按新的参数收集流程处理，避免历史对话污染命令生成。

## 4. 方案设计

### 4.1 总体设计

新增 Skill 目录：

```text
skills/text-generate-executor/
├── SKILL.md
└── references/
    ├── dialog-flow.md
    └── text-generate-params.md
```

核心流程：

```text
用户自然语言诉求
        ↓
识别场景：Prefill / Decode / optimizer 复验 / profiling / debug
        ↓
渐进式收集缺失参数
        ↓
仅当用户不确定或要求推荐时提供建议值，并标注来源
        ↓
生成候选命令后固定展示 Debug/Trace 可选参数，并让用户选择是否追加
        ↓
执行前参数校验和假设说明
        ↓
展示完整 text_generate 命令并请求确认
        ↓
执行命令，捕获 stdout/stderr
        ↓
摘要结果或分类失败原因
```

### 4.2 Skill 触发条件

`SKILL.md` frontmatter 的 `description` 需要覆盖以下触发词和场景：

- `text_generate`
- 单点推理仿真 / 性能验证 / 配置复验
- Prefill / Decode
- TP / DP / EP / MOE-TP / MOE-DP 固定策略验证
- profiling database
- chrome trace / graph log / empirical metrics
- throughput_optimizer best row 映射验证

### 4.3 参数收集策略

首轮必须收集：

- `model_id`
- `--device`
- `--num-devices`
- `--num-queries`
- `--query-length`
- 执行模式：Prefill-only、Decode 或 optimizer best-row validation

按场景分支收集：

- Decode：必须确认 `--context-length` 并添加 `--decode`。
- MTP：无论模型类型和 Prefill/Decode 模式，均必须询问是否开启 MTP。开启时确认模型支持并收集 `--num-mtp-tokens`；Decode MTP 的 `--query-length` 必须设置为 `1 + --num-mtp-tokens`，其中 decode query length 为 `1`，MTP token 数来自用户输入的 `--num-mtp-tokens`。
- Prefix Cache：仅在 Prefill-only 仿真阶段询问是否设置 `--prefix-cache-hit-rate`；Decode 阶段不询问、不添加该参数。
- MoE：按需确认 `--ep-size`、`--moe-tp-size`、`--moe-dp-size`、shared expert 等参数。
- 多模态：必须同时确认 `--image-batch-size`、`--image-height`、`--image-width`。
- Profiling：必须同时确认 `--performance-model profiling` 与 `--profiling-database`。
- Debug/Trace：必要参数交互完成并生成候选命令后，必须固定告知用户以下可选参数和功能，再让用户选择是否追加：
  - `--chrome-trace`：导出 Chrome trace 文件，用于查看执行 timeline。
  - `--graph-log-url`：导出编译图日志，用于 debug compile graph dump；要求同时启用 `--compile`，且本地环境可能需要安装 `pydot`，否则该参数可能报错。
  - `--dump-input-shapes`：在表格平均值中按 input shape 分组，便于定位 shape 相关行为。
- 如果用户选择任一 Debug/Trace 参数，Skill 必须更新候选命令和参数摘要，然后再进入最终执行确认。

### 4.4 默认值和确认规则

- 默认使用 analytic performance model，除非用户明确要求 profiling。
- 默认添加 `--compile`，用于贴近编译路径仿真；用户明确要求 eager/non-compiled 行为时才移除。
- 量化默认值采用 CLI 默认语义：`W8A8_DYNAMIC` 与 `DISABLED`，不静默切换到自定义量化。
- `MXFP4` 需要确认 `--mxfp4-group-size`。
- 层级并行覆盖参数默认不设置，除非用户提供或来自 optimizer 推荐行。
- 用户只提供模型和场景时，不生成完整命令，必须先渐进式询问缺失核心参数。
- 不默认复用历史对话中的硬件、batch、context、MTP、TP/EP/MoE 等参数；仅当用户明确说“沿用/继续/保持上一轮配置”时才复用。
- 仅当用户表示“不确定”或主动要求建议时给出建议值。
- 建议值必须标注来源：CLI 默认、用户上一轮配置、当前轮检查过的本地测试/示例、文档示例、throughput_optimizer 结果或显式启发式。
- 不得声称某配置来自回归覆盖、仓库默认或本地示例，除非当前轮检查了具体文件或命令输出，并在摘要中说明来源。
- 显式启发式建议必须说明其为启发式，并简要说明实践理由。

### 4.5 校验规则

执行前校验：

- `--num-queries`、`--query-length`、`--num-devices`、显式并行尺寸必须为正整数。
- `--context-length` 必须非负。
- Prefill-only 阶段若设置 `--prefix-cache-hit-rate`，必须位于 `[0, 1)`；Decode 阶段不添加该参数。
- TP/DP/EP/MOE-TP/MOE-DP 与 `--num-devices` 需要具备合理兼容性。
- `--export-empirical-metrics` 必须搭配 profiling mode。
- `--graph-log-url` 必须搭配 `--compile`，并要求 Python 环境已安装 `pydot`。
- Debug/Trace 提示必须发生在候选命令生成之后、最终执行确认之前；即使用户没有主动提出 trace/debug，也需要展示 `--chrome-trace`、`--graph-log-url`、`--dump-input-shapes` 供选择。
- 多模态参数必须成组出现。
- Decode 模式开启 MTP 时，`--query-length` 必须等于 `1 + --num-mtp-tokens`。

### 4.6 输出摘要

成功执行后输出：

- 完整命令。
- 场景摘要：模型、设备、Prefill/Decode、batch/query/context。
- 并行策略：TP/DP/EP/MOE 相关参数。
- 性能模型：analytic 或 profiling。
- stdout 中可识别的延迟、吞吐、内存、算子/阶段 breakdown。
- Decode MTP 场景需要说明：仿真结果中的 forward 时间对应“原主模型 forward + MTP layer forward”。如果用户需要粗略换算开启 MTP 后的单 token 时延，可用该 forward 时间除以 `1 + sum(每步 MTP 的接受率)`；接受率来自用户预期，由用户按场景自行计算。
- trace、graph log、empirical metrics 的输出路径。
- 统一免责声明：结果来自 simulation，仅供规划和验证参考。

失败时输出：

- 完整命令。
- 关键错误行。
- 错误分类：参数、环境、模型加载、依赖、device profile、profiling database。
- 最小修复建议。

## 5. 实施计划

| 阶段 | 内容 | 状态 |
|:---|:---|:---|
| P0 | 创建 `text-generate-executor` Skill 基础目录和 `SKILL.md` | 已完成 |
| P0 | 创建 `dialog-flow.md` 和 `text-generate-params.md` | 已完成 |
| P0 | 创建 `agents/openai.yaml` | 已完成 |
| P1 | 将 Skill 安装到 `$CODEX_HOME/skills` | 已完成 |
| P1 | 使用典型 prompt 做人工验收 | 待执行 |
| P2 | 视 stdout 稳定性新增结构化解析脚本 | 可选 |

## 6. 测试与验收

| 测试场景 | 验收标准 |
|:---|:---|
| Prefill-only 命令生成 | 不添加 `--decode`，能生成包含必填参数的命令 |
| Decode 命令生成 | 添加 `--decode`，缺失 `--context-length` 时会追问 |
| MTP 固定询问 | 任意文本仿真场景都会询问是否开启 MTP；开启时收集 `--num-mtp-tokens` 并确认模型支持 |
| Decode MTP 命令生成 | 添加 `--decode` 和 `--num-mtp-tokens`，并将 `--query-length` 设置为 `1 + --num-mtp-tokens` |
| Prefix Cache 阶段限制 | 仅 Prefill-only 阶段询问/添加 `--prefix-cache-hit-rate`，Decode 阶段不询问 |
| 上下文隔离 | 用户只给模型和场景时不复用历史配置，不生成完整命令，而是询问缺失核心参数 |
| 建议来源标注 | 用户不确定时可以给建议，但每个建议值必须标注来源；未检查本地文件时不得声称来自回归覆盖 |
| Optimizer row 映射 | 明确说明固定候选验证，不执行搜索 |
| Profiling 参数校验 | 缺失 `--profiling-database` 时阻止执行并提示补齐 |
| Debug/Trace 固定提示 | 生成候选命令后、最终确认前必须说明 `--chrome-trace`、`--graph-log-url`、`--dump-input-shapes` 的功能和限制，并允许用户选择是否追加 |
| 多模态参数校验 | 任一 image 参数出现时要求补齐三元组 |
| 失败处理 | 能展示命令、关键错误和最小修正建议 |

## 7. 修改文件

| 文件 | 说明 |
|:---|:---|
| `skills/text-generate-executor/SKILL.md` | Skill 主说明和执行规则 |
| `skills/text-generate-executor/references/dialog-flow.md` | 渐进式问参流程 |
| `skills/text-generate-executor/references/text-generate-params.md` | text_generate 参数速查 |

## 8. 后续演进

- 新增 `scripts/extract_text_generate_metrics.py`，结构化提取 stdout 表格。
- 补充更多模型族模板，例如 dense、MoE、多模态、MTP。
- 在 Skill 安装后补充实际对话验收记录。

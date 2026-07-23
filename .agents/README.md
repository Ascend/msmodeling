# msmodeling skills

本目录存放 msmodeling 项目专用的 Claude Code skills，用于把常见性能建模、设备建模和 profiling 辅助任务沉淀为可复用的执行流程。

> **使用提示**：如需在 Claude Code 中启用这些 skills，请将本目录 `.agents/skills` 完整复制到 `.claude/skills`。
>
> **AI agents 必读**：请先阅读项目根目录的 [AGENTS.md](../AGENTS.md)，了解项目规范和 Skill 体系。

## Table of Contents

- [msmodeling skills](#msmodeling-skills)
- [msmodeling-env-installer](#msmodeling-env-installer)
- [model-adaptation](#model-adaptation)
- [device_config](#device_config)
- [op_mapping](#op_mapping)
- [microbench](#microbench)
- [text-generate-executor](#text-generate-executor)
- [throughput-optimizer-executor](#throughput-optimizer-executor)
- [throughput-optimizer-explainer](#throughput-optimizer-explainer)
- [sig-review](#sig-review)

## msmodeling-env-installer

msmodeling 环境安装器——将“安装 msmodeling 环境依赖”“`uv sync` 安装”“安装当前仓库 requirements.txt（legacy fallback）”“配置 PYTHONPATH / HF_ENDPOINT”等明确指向 msmodeling 的请求转换为可执行、可验证、可回溯的环境安装流程。用户只说“安装环境”或“安装依赖”时，需要先确认是否安装 msmodeling 当前仓库的环境依赖。

### What it does

引导 AI agent 按 RFC 中定义的流程完成开发环境初始化：

1. **仓库根目录校验**：确认当前目录包含 `README.md` 和 `pyproject.toml`。
2. **Python 与 uv 检查**：要求 Python `3.10+`，缺少 `uv` 时按镜像安装并解析真实可执行路径。
3. **安装路径选择**：默认在仓库根目录执行 `uv sync`（自动创建 `.venv`、可编辑安装 msmodeling）；legacy fallback 使用 `requirements.txt` 前检查 `torch_npu`、`torch-npu` 和 `cudatoolkit`。
4. **依赖安装与验证**：`uv sync` 后执行 `uv pip check` 与 `uv run msmodeling --help`。
5. **环境变量配置**：按需设置当前会话 `PYTHONPATH` 和 `HF_ENDPOINT=https://hf-mirror.com`。

### File layout

| File | Purpose |
| ---- | ------- |
| `msmodeling-env-installer/SKILL.md` | Skill 定义、触发场景、安装流程和安全规则 |
| `msmodeling-env-installer/scripts/install-current-project-deps.ps1` | Windows PowerShell 自动化安装脚本 |
| `msmodeling-env-installer/scripts/install-current-project-deps.sh` | Linux/macOS/WSL/Git Bash 自动化安装脚本 |

### Quick start

在对话中直接提出明确需求，例如“请帮我安装 msmodeling 环境依赖”“按 README 配置 msmodeling 环境”。如果只说“安装环境”，agent 需要先确认是否安装 msmodeling 当前仓库的环境依赖。

Windows PowerShell 可以从仓库根目录直接运行：

```powershell
.\.agents\skills\msmodeling-env-installer\scripts\install-current-project-deps.ps1
```

Linux/macOS/WSL/Git Bash 可以从仓库根目录直接运行：

```bash
bash ./.agents/skills/msmodeling-env-installer/scripts/install-current-project-deps.sh
```

### Key constraints

- 不修改 `requirements.txt`、README 或项目源码。
- `uv sync` 会复用已有 `.venv` 并更新依赖，不默认持久化系统级环境变量。
- 网络安装需要用户确认和工具权限授权。
- `scripts/install-current-project-deps.ps1` 当前仅适用于 Windows PowerShell；Linux/macOS 使用 README 通用命令。

---

## model-adaptation

TensorCast 新模型接入流程 skill——从仿真命令和 MindStudio Insight raw profiling 出发，引导 agent 运行 `model_adapter doctor`、审阅 `ModelProfile`、处理 patch/bug AI task、导出 `evidence.yaml` 并运行 verify。

### What it does

将新模型接入拆成确定性工具和人工 checkpoint：

1. 收集两个必需输入：仿真命令和匹配的 raw profiling。
2. 运行 doctor，审阅 candidate profile、evidence draft、human questions 和 ai tasks。
3. 对需要人工确认的字段生成精确问题，并把确认结果写入 `hints.yaml` 或 `evidence.yaml`。
4. 对 patch/bug 场景使用 `ai_tasks[].prompt_text` 驱动用户或用户的 AI 助手生成代码，并要求人工 review。
5. 使用 `export-evidence` 导出 `evidence.yaml`，再运行 verify。

### File layout

| File | Purpose |
| ---- | ------- |
| `model-adaptation/SKILL.md` | 新模型接入的核心工作流、人工 checkpoint 和验证要求 |

### Quick start

当用户说“接入新模型”“生成 ModelProfile”“根据 doctor report 继续适配”“处理 patch AI task”“从 doctor report 导出 evidence”时使用该 skill。

### Key constraints

- 不凭模型名猜 profile 字段。
- doctor 不生成模型专属 patch 代码，只生成 AI task 和 prompt。
- `evidence.yaml` 从 `doctor_after_profile.json.evidence_draft` 导出后再人工审阅。
- 不提交 raw profiling、本地 walkthrough、私人路径或临时材料。

---

## device_config

设备画像自然语言导入器——通过渐进式对话引导用户将自然语言硬件描述转换为 TensorCast `DeviceProfile`。

### What it does

引导 AI agent 通过渐进式对话流程：

1. **渐进收集信息**：首轮只问硬件名称、资料来源和粒度偏好，每轮最多 2-3 个问题。
2. **维护内部事实表**：`confirmed` / `ambiguous` / `missing` / `needs calibration`。
3. **生成可运行 profile**：将用户确认的值、临时估值和兜底默认值全部写入 `tensor_cast/device.py`。
4. **验证 + 输出 CLI 命令**：运行导入检查，输出 `--device <PROFILE_NAME>` 可执行命令。

### File layout

| File | Purpose |
| ---- | ------- |
| `device_config/SKILL.md` | Skill 定义、约束条件和执行流程 |

### Quick start

在 Claude Code 对话中直接提出需求，例如"我要导入新的设备拓扑"，遵循 agent 的渐进式提问，逐步提供硬件规格。

### Key constraints

- `DeviceProfile.__post_init__` 会自动注册 profile，`name` 必须唯一。

- 默认写入 `tensor_cast/device.py`，只有用户明确要求时才写入 `tensor_cast/device_profiles/`。

- 所有默认值、估值和假设必须对用户可见，列入 `needs calibration`。

---

## op_mapping

`op_mapping.yaml` 生成器——将 TensorCast 仿真算子映射到 NPU profiling 内核类型。

### What it does

通过并行子 Agent 团队（每个算子一个 Agent）追踪完整的 vLLM→CANN 调用链，生成 `op_mapping.yaml`。

### File layout

| File | Purpose |
| ---- | ------- |
| `op-mapping/SKILL.md` | 核心执行流程、六阶段工作流 |
| `op-mapping/op-mapping-template.yaml` | YAML 模板片段 |
| `op-mapping/single-op-worker-prompt.md` | 单算子 Worker Agent 指令 |
| `op-mapping/verifier-prompt.md` | 验证阶段指令 |
| `op-mapping/ref/shape_matching_catalog.md` | TC tensor 与 NPU profiling shape 的 10 种差异 |
| `op-mapping/ref/tc_input_count_rules.md` | `tc_input_count` 安全使用规则 |
| `op-mapping/ref/zero_cost_classification.md` | 零开销算子分类规则 |

### Quick start

收集完所有输入（model、device、profiling CSV、repo 版本）后，agent 自动执行六阶段流程：GATHER → FORWARD MAPPING → REVERSE MAPPING → VERIFY → WRITE → COMMIT。

### Key constraints

- `kernel_type` 必须与 CSV 文件名完全一致（无 `.csv` 后缀）。

- 三个映射路径：aten→op-plugin→aclnn、torch_npu.npu_*→op-plugin→aclnn、vllm-ascend 自定义/Triton。

- `alternate_kernel_types` 必须在同一抽象层级，禁止用融合大 op 作为子 op 的备选。

---

## microbench

Microbench Run Script 生成器——从 profiling CSV 生成可在 NPU 上重放的 `<KernelType>_run.py`。

### What it does

为 profiling 内核 CSV 生成可运行的 `tools/perf_data_collection/op_replay/<KernelType>_run.py`，用于 NPU 实测重放。

### File layout

| File | Purpose |
| ---- | ------- |
| `microbench/SKILL.md` | Skill 定义和 repo 搜索顺序 |

### Quick start

用户提供 `kernel_type`、设备 profile、vllm_ascend 版本和 CSV 路径后，agent 生成可重放的 run script。

### Key constraints

- 优先使用本地已克隆的 repos，按指定路径搜索。

- repo 缺失时按 `SKILL.md` 中提供的 clone 命令获取。

- 生成的 run script 由 `run_all_op.py` / `profile_and_update_db.py` 调用。

---

## text-generate-executor

`text_generate` 单点验证执行器。用于把用户关于 `python -m cli.inference.text_generate` 的验证诉求转换为可确认、可执行的 CLI 命令，并在确认后运行和总结结果。

### What it does

面向已有模型、硬件、batch/query length、prefill 或 decode 模式、固定 TP/DP/EP/MOE 策略、profiling database、trace/debug 或 throughput optimizer 最优行复验的场景，生成单点仿真命令。

### File layout

| File | Purpose |
| ---- | ------- |
| `text-generate-executor/SKILL.md` | Skill 主说明、默认策略、校验规则和 handoff 规则 |
| `text-generate-executor/references/dialog-flow.md` | 渐进式问参流程 |
| `text-generate-executor/references/text-generate-params.md` | `text_generate` 参数速查 |

### Quick start

提出“帮我跑 text_generate 验证”“把 throughput_optimizer 最优行转 text_generate 跑一下”“导出 chrome trace”等请求时，agent 会补齐缺失参数，展示命令和假设，并在用户确认后执行。

### Key constraints

- 执行前必须展示完整命令和关键假设，并要求显式确认。
- Decode 模式必须确认 `--context-length`；profiling 模式必须提供 `--profiling-database`。
- `text_generate` 只验证固定候选，不执行 TP/EP/MOE-DP 搜索。

---

## throughput-optimizer-executor

`throughput_optimizer` 部署规划执行器。用于把吞吐规划、硬件对比、并行搜索、PD 聚合/分离/配比优化等自然语言诉求转换为 `python -m cli.inference.throughput_optimizer` 命令。

### What it does

面向搜索和规划场景，收集模型、硬件、设备数、输入/输出长度、SLO、部署模式和搜索空间，生成 optimizer 命令，在确认后运行并总结最佳并行策略、batch、concurrency、throughput、TTFT、TPOT 和 PD ratio 信息。

### File layout

| File | Purpose |
| ---- | ------- |
| `throughput-optimizer-executor/SKILL.md` | Skill 主说明、默认策略、校验规则和 handoff 规则 |
| `throughput-optimizer-executor/references/dialog-flow.md` | 部署模式识别和渐进式问参流程 |
| `throughput-optimizer-executor/references/throughput-optimizer-params.md` | `throughput_optimizer` 参数速查 |
| `throughput-optimizer-executor/scripts/extract_throughput_optimizer_result.py` | optimizer stdout 结构化摘要脚本 |

### Quick start

提出“比较两种硬件”“搜索 Qwen 32B 最佳 TP”“做 PD 分离能力评估”“算 P/D 实例配比”等请求时，agent 会识别 aggregation、disagg 或 PD ratio 模式，补齐 SLO 和搜索空间，并在确认后执行。

### Key constraints

- `--enable-optimize-prefill-decode-ratio` 不能与 `--disagg` 同时使用。
- 多硬件对比共用同一个 `--num-devices`，需要在执行前说明。
- 执行前需要明确确认是否开启 prefix cache 和 MTP；开启后分别补齐 hit rate、MTP token 数和接受率假设。
- 该 skill 做候选搜索和规划；单点复验应 handoff 到 `text-generate-executor`。
- 结果合理性、硬件差异、Cube/Vec/Comm/Mem 和 best row 映射解释应 handoff 到 `throughput-optimizer-explainer`。

---

## throughput-optimizer-explainer

`throughput_optimizer` 结果解释器。用于分析 optimizer 输出是否合理、比较硬件或并行策略差异、解释 Prefill/Decode 阶段的 Cube/Vec/Comm/Mem 瓶颈，并把最优行映射为 `text_generate` 验证命令。

### What it does

围绕 optimizer 结果建立证据分级和解释边界：

1. 识别 aggregation、disaggregation 或 PD ratio 模式。
2. 提取模型、硬件、输入/输出长度、SLO、量化、compile、prefix cache、MTP 和搜索空间等可比条件。
3. 提取 best row、top candidates、throughput、TTFT、TPOT、concurrency、batch、并行策略、PD ratio、QPS 和 breakdown。
4. 按 `macro_only`、`optimizer_phase_breakdown`、`text_generate_phase_breakdown`、`text_generate_op_bound`、`profiler_trace` 判断证据等级。
5. aggregation 结果必须拆成 Prefill forward、Decode forward 和调度公式，不能当成单次 forward。
6. 需要 operator 级归因时，使用 `text_generate --dump-op-bound-results`，并明确它是 TensorCast 模拟归因而不是真实 profiler 证据。

### File layout

| File | Purpose |
| ---- | ------- |
| `throughput-optimizer-explainer/SKILL.md` | Skill 主说明、证据规则、工作流、映射规则和输出要求 |
| `throughput-optimizer-explainer/references/aggregation-mapping.md` | aggregation best row 到 Prefill/Decode 验证命令的映射 |
| `throughput-optimizer-explainer/references/disaggregation-mapping.md` | disaggregation 和 PD ratio 结果到 `text_generate` 的映射 |
| `throughput-optimizer-explainer/references/evidence-levels.md` | 证据等级、可支持结论和禁止过度推断的规则 |
| `throughput-optimizer-explainer/references/bottleneck-rules.md` | Cube/Vec/Comm/Mem 与并行策略解释规则 |
| `throughput-optimizer-explainer/references/output-template.md` | 简洁输出模板 |
| `throughput-optimizer-explainer/scripts/parse_optimizer_output.py` | 解析 optimizer、dump 表、`text_generate` breakdown 和 op-bound 表为 JSON |
| `throughput-optimizer-explainer/scripts/build_text_generate_commands.py` | 从 normalized best row JSON 生成 Prefill/Decode 验证命令 |
| `throughput-optimizer-explainer/scripts/compare_phase_breakdowns.py` | 对比 Cube/Vec/Comm/Mem 或 op-bound 表差异 |

### Quick start

当用户问“这个 throughput_optimizer 结果是否合理”“为什么 A3 比 A2 快/慢”“Cube/Vec/Comm/Mem 谁是瓶颈”“把 best row 转成 text_generate 验证命令”时使用该 skill。

### Key constraints

- 只有宏观输出时，只能做部署、阶段和策略层面的推断，不能断言具体 operator 或真实 kernel 瓶颈。
- `text_generate --dump-op-bound-results` 是 TensorCast 模拟 operator 归因，必须与真实 profiler/kernel 证据区分。
- aggregation throughput 不是单次 forward TPS；解释和复验时必须拆成 Prefill 与 Decode 两条验证命令。

---

## sig-review

GitCode PR 检视技能——支持三种工作流：(1) PR 作者推送后说"请求检视"，自动分析变更文件归属 SIG 并指派 chair；(2) reviewer 收到检视通知后说"检视PR {number}"，自动分析 diff 并提交结构化检视意见；(3) 收到检视意见后说"分析PR {number}的检视意见"，自动拉取 diff_comment 评论并逐条分析合理性、给出修改建议。全程自然语言交互，无需安装任何外部工具。

### What it does

引导 AI agent 完成端到端的 PR 代码检视：

1. **获取 PR 信息**：一条命令获取 PR 详情、变更文件、diff、已有评论。
2. **理解 PR**：分析标题、描述、变更范围，以 diff 为最关键依据。
3. **生成检视意见**：按类别（逻辑缺陷 / 性能 / 安全 / 架构 / 规范）和数量控制生成结构化意见。
4. **二次检查**：提交前检查问题准确性、行号准确性、建议可行性、措辞得体。
5. **提交检视意见**：通过 GitCode API 提交评论，自动添加 `【review】【类别】` 前缀。
6. **完成检视**：提交检视结论，通过则指派 approver，有意见则转回作者修改。

### File layout

| File | Purpose |
| ---- | ------- |
| `sig-review/SKILL.md` | Skill 定义、分配检视流程、代码检视流程、评论格式规范 |
| `sig-review/scripts/review_api.py` | 自包含 GitCode API 工具（零外部依赖，仅 Python 标准库），含 auth/fetch/assign/list/status/comment/handoff/complete/verdict/comments/withdraw 共 11 个命令 |
| `sig-review/sig_ownership.json` | SIG 目录归属映射表（10 个 SIG 的路径、chair、reviewer、approver） |
| `sig-review/ref/review-checklist.md` | 检视质量标准、什么该 / 不该检视、msmodeling 专项关注 |

### Quick start

**首次使用前配置令牌**（一次配置，持久生效）：

```bash
python3 .agents/skills/sig-review/scripts/review_api.py auth --token <你的GitCode令牌>
# 或从 stdin 读取（不在 shell 历史留痕）：
echo '<你的令牌>' | python3 .agents/skills/sig-review/scripts/review_api.py auth --stdin
```

令牌保存在 `~/.config/sig-review/config.json`（权限 600），后续所有命令自动读取。

PR 作者推送代码后，对 agent 说"请求检视"即可触发分配流程。Agent 自动分析变更文件归属哪个 SIG，指派对应 chair 并打标签。

reviewer 收到检视通知后，对 agent 说"我有哪些待检视PR"即可列出任务，然后说"检视PR 123"开始检视。

```bash
# 分配检视（PR 作者用）
python3 .agents/skills/sig-review/scripts/review_api.py assign 123 --dry-run  # 预览
python3 .agents/skills/sig-review/scripts/review_api.py assign 123            # 指派

# 查看待检视 PR（reviewer 用）
python3 .agents/skills/sig-review/scripts/review_api.py list

# 代码检视（reviewer 用）
python3 .agents/skills/sig-review/scripts/review_api.py fetch 123

# 提交检视意见
python3 .agents/skills/sig-review/scripts/review_api.py comment 123 \
  --file tensor_cast/core/config.py --line 42 --category 逻辑缺陷 \
  --content "缺少空指针检查"

# 完成检视，移交给 approver
python3 .agents/skills/sig-review/scripts/review_api.py complete 123 --to lutean --event approved
```

### Key constraints

- 脚本仅依赖 Python 标准库，零外部依赖，适用于任何 agent 环境。
- PR 变更唯一来源是 GitCode API 的 `patch` 字段，禁止用 `git diff` 获取变更。
- `assign` 命令使用最长前缀匹配路由文件到 SIG，支持 chair==author 自动改指派 reviewer、跨 SIG 双签、根目录文件标记架构共审。
- 检视意见自动添加 `【review】【类别】` 前缀，`--content` 只需提供正文。
- 评论必须提交在 diff 中新增或修改的行上，不能提交在未修改的上下文行上。

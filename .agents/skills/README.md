# msmodeling skills

本目录存放 msmodeling 项目专用的 Claude Code skills，供 TensorCast 等模块使用。

> **使用提示**：如需在 Claude Code 中启用这些 skills，请将本目录 `.agents/skills` 完整复制到 `.claude/skills`。
>
> **AI agents 必读**：请先阅读项目根目录的 [AGENTS.md](../../AGENTS.md)，了解项目规范和 Skill 体系。

## Table of Contents

- [msmodeling skills](#msmodeling-skills)
- [model-adaptation](#model-adaptation)
- [device_config](#device_config)
- [op_mapping](#op_mapping)
- [microbench](#microbench)


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

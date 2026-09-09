# TensorCast 新模型适配开发指导（跑通级）

本文介绍将 Hugging Face 风格的新模型适配到 TensorCast 的实际操作流程。当前流程的目标是**跑通级适配**：新模型仿真在基础 case 上能跑通不报错，且关键算子调用次数与模型公开结构一致（例如各类 attention 算子的调用次数）。不依赖任何实测数据，输入只有仿真命令、已安装的开源模型源码和配置。

全量 shape/dtype 校验与实测 profiling 精度对比由下游精度工作流负责，该工作流会串联本流程的产出继续处理。

架构设计与方案背景请参见：

```text
docs/design/model_adaptation_efficiency_design.md
```

## 1. 适用范围

当模型需要以下一种或多种适配时，可使用本文流程：

- 模型 Profile 元数据，例如 `model_type`、模块路径或配置字段映射。
- MoE 元数据、专家数量字段或专家路由兼容逻辑。
- MLA、MTP、视觉语言或多模态模块映射。
- meta device、`torch.compile` 或 shape 兼容性补丁。
- 新算子适配（算子声明与算力/内存性能属性）。

对于没有特殊结构的 dense text 模型，适配流程可能只需要一个最小模型 Profile。

## 2. 准备环境

请在已配置好的 Python 环境中，从 msModeling 仓库根目录执行所有命令。如果不在仓库根目录运行，请先设置 `PYTHONPATH`：

```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
```

检查模型适配 CLI 是否可用：

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter verify --help
```

如果您的 AI 助手支持加载项目 Skill，可使用模型适配 Skill：

```text
.agents/skills/model-adaptation
```

该 Skill 可辅助处理 doctor 报告、Profile 评审、人工确认点、新算子适配位置、AI 补丁任务与跑通验证。确定性的适配工具仍是解析、校验和通过/失败判断的准确信息来源。

## 3. 准备输入

在 `reports/` 下创建一个 case 目录（本地工作目录，已加入 `.gitignore`），并准备以下输入：

| 输入 | 保存为 | 要求 |
| --- | --- | --- |
| TensorCast 仿真命令 | `reports/<case_name>/command.txt` | 完整可运行的 workload 命令。 |

本流程不需要任何实测输入（无 MindStudio Insight 导出、无 profiling 数据）。

注意：`reports/<case_name>/` 下的所有文件（command.txt、doctor.json、verify.json、failure.log、st_cases 等）都是**开发过程中的工作产物，不上库**。它们供开发调试和后续精度对齐流程在仓库外使用；仓库只接收代码、测试、文档和（可选的）ST guardrail 用例。

可选输入：

| 输入 | 保存为 | 用途 |
| --- | --- | --- |
| 失败日志 | `reports/<case_name>/failure.log` | 让 doctor 对 dry-run 或 smoke 失败进行分类，判断是否需要补丁。 |

## 4. 创建 Case 目录

将 `<case_name>` 替换为简短且稳定的名称，例如 `qwen3_vl_8b_prefill`。

```bash
mkdir -p reports/<case_name>
```

保存完整 TensorCast 仿真命令：

```bash
cat > reports/<case_name>/command.txt <<'EOF'
python -m cli.inference.text_generate <model_id> \
  --device <device_profile> \
  --num-devices 1 \
  --num-queries 1 \
  --query-length 1 \
  --context-length 128
EOF
```

多模态与量化参数示例：

```bash
cat > reports/qwen3_vl_8b_prefill/command.txt <<'EOF'
python -m cli.inference.text_generate Qwen/Qwen3-VL-8B-Instruct \
  --device TEST_DEVICE \
  --num-devices 1 \
  --num-queries 1 \
  --query-length 128 \
  --context-length 0 \
  --image-batch-size 1 \
  --image-height 224 \
  --image-width 224 \
  --quantize-attention-action disabled \
  --quantize-linear-action W8A8_DYNAMIC
EOF
```

## 5. 运行 Doctor

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --profile-draft-output reports/<case_name>/<model_type>_draft.py \
  --output reports/<case_name>/doctor.json
```

预期输出：

```text
reports/<case_name>/doctor.json
reports/<case_name>/<model_type>_draft.py
```

重点检查 `doctor.json` 中的以下字段：

| 字段 | 含义 | 操作 |
| --- | --- | --- |
| `adaptation_context` | 解析后的命令和标准化 workload 参数。 | 确认与预期 workload 一致。 |
| `candidate_profile` | 最小候选 `ModelProfile` 字段。 | 对照已安装模型源码进行评审。 |
| `candidate_profile_validation` | 候选 Profile 的确定性校验结果。 | 注册 Profile 前修复错误。 |
| `candidate_profile_draft` | Profile 草稿 Python 模块内容。 | 仅作为起点使用。 |
| `profile` | 已注册 Profile（如存在）。 | 作为正常适配的参考。 |
| `profile_validation` | 已注册 Profile 的校验结果。 | 如存在错误需修复。 |
| `human_questions` | 需要用户确认的最小事实（低置信候选字段、结构缺口）。 | 对照源码确认后固化到 Profile。 |
| `ai_tasks` | 给 AI 助手的边界明确任务。 | 仅在评审确定性发现后使用。 |
| `patch_reports` | dry-run 补丁结果。 | 检查预期替换和跳过模块。 |
| `suggestions` | 推荐下一步操作。 | 用于决定下一轮迭代。 |

## 6. 定位模型源码

模型源码不一定来自已安装的 `transformers`，按以下顺序确定来源：

1. **仓库当前 `transformers` 版本已支持该模型**：直接读库内源码，通常为 `transformers.models.<model_name>.modeling_<model_name>`。
2. **最新 `transformers` 版本已支持、但仓库锁定版本不支持**：优先推动升级仓库的 `transformers` 依赖版本（`pyproject.toml` / `uv.lock`），升级后回到情况 1；不要为了省事把新版模型代码复制进仓库。
3. **最新 `transformers` 版本都不支持该模型**：去模型实际开源的地方获取源码（HuggingFace 模型库/remote code、vLLM 等推理框架的实现），并将模型源码作为模型专属模块落到 `tensor_cast/transformers/builtin_model/` 内适配，遵循仓库"patch/wrapper 层适配、不直接修改上游依赖"的约束。

无论哪种情况，都应在实际源码中确认真实类名、模块路径、配置字段和 forward 行为，不要仅凭模型名填写 Profile 字段。

## 7. 新算子适配

仅当模型 forward 用到了 TensorCast 尚不支持的算子语义时，才需要新增算子适配。判断依据是 dry-run 失败日志中的 `UNSUPPORTED_OP_ROUTING` 分类，或模型源码中确认的新计算语义。新增算子适配包含两件套，位置如下：

1. **算子声明（shape 传播）**：在 `tensor_cast/ops/` 下对应类别文件（或新文件）用 `@register_tensor_cast_op("<name>")` 声明算子，函数体只做 shape 传播（返回 `torch.empty_like` 等 meta 实现），并在 `tensor_cast/ops/__init__.py` 登记 import。可参考 `tensor_cast/ops/attention.py`、`tensor_cast/ops/layernorm.py` 等现有实现。

2. **性能属性（算力/内存公式）**：在 `tensor_cast/performance_model/__init__.py` 用 `@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.<name>.default)` 注册 functor，从 `OpInvokeInfo` 的输入 shape/dtype 推导 `mma_ops`/`gp_ops` 与内存读写量。模型专属算子可放在 `tensor_cast/performance_model/builtin_model/` 或对应 builtin model 模块中。算力与内存公式需要分析开源模型源码的数学语义（FLOPs 数量级、读写张量大小）推导，可对照同类现有算子的实现校验数量级。

补充约束：

- 如果新算子属于 attention 计算核心（每层调用一次）且算子名不含 `attention` 子串（例如 DeepSeek-V4 的 `sparse_attn_sharedkv`），必须同步登记到 `tensor_cast/adapter/expectations.py` 的 `_ATTENTION_CORE_OPS`，否则跑通验证无法正确计数。
- 实测（empirical）性能模型依赖 `op_mapping.yaml` 的算子映射，该映射由下游精度工作流校准，不在本流程范围内。
- 新算子需要配套单测：meta shape 传播正确、性能属性的数量级合理。

## 8. 注册已评审的 Profile

将评审后的草稿移动到：

```text
tensor_cast/transformers/builtin_model/<model_type>.py
```

使用 `register_model_profile(ModelProfile(...))` 注册。

保持 Profile 最小化：

- 包含 `model_type`。
- 仅包含已确认的非默认 MoE、MLA、MTP 或 VL 字段。
- `moe_field_names_override` 使用普通 `dict`。
- 嵌套专家数量 key 使用 list 形式，例如 `["text_config", "num_experts"]`。
- 不写空 override。
- 不写默认 `None` 字段。
- 除非已评审代码路径确实需要，否则不写默认 `moe_num_experts_key="num_experts"`。

Profile 示例：

```python
from tensor_cast.transformers.custom_model_registry import ModelProfile, register_model_profile


register_model_profile(
    ModelProfile(
        model_type="example_vl",
        model_family="example_vl",
        visual_module_path="visual",
        language_module_path="language_model",
        visual_layers_module_path="visual.blocks",
        visual_merger_linear_mapping={
            "visual.merger.linear_fc1": "colwise",
            "visual.merger.linear_fc2": "rowwise",
        },
        visual_mlp_linear_mapping={
            "visual.blocks.*.mlp.linear_fc1": "colwise",
            "visual.blocks.*.mlp.linear_fc2": "rowwise",
        },
    )
)
```

## 9. 处理运行时补丁需求

仅当已安装模型源码与 TensorCast 仿真不兼容时，才使用 `patch_method`。常见原因包括：

- 在 `meta` tensor 上读取依赖数据的 tensor 标量。
- 基于 tensor 值执行 Python 控制流。
- 严格的图片或视频 placeholder 检查。
- 使用动态输出 shape 的布尔 mask 索引。
- compile graph break。
- 不支持的算子路由。
- forward 签名不匹配。

采集完整失败日志：

```bash
set -o pipefail
bash reports/<case_name>/command.txt 2>&1 | tee reports/<case_name>/failure.log
```

带失败日志重新运行 doctor：

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --patch-failure-file reports/<case_name>/failure.log \
  --profile-draft-output reports/<case_name>/<model_type>_draft_with_patch.py \
  --output reports/<case_name>/doctor_with_failure.json
```

预期输出：

```text
reports/<case_name>/failure.log
reports/<case_name>/doctor_with_failure.json
reports/<case_name>/<model_type>_draft_with_patch.py
```

当需要补丁时，doctor 会在 `ai_tasks` 中生成 `PATCH_METHOD_AUTHORING`。可将 `ai_tasks[].prompt_text` 提供给 model-adaptation Skill 或其他 AI 助手。AI 助手应返回 patch-method 草稿、类/方法目标、语义说明和验证命令。

评审规则：

- Doctor 提供确定性证据和 prompt，但不会生成最终的模型专用补丁代码。
- AI 输出在人工评审前仅作为建议。
- 只修补与仿真不兼容的路径。
- 尽可能保持正常 tensor 行为。
- 添加补丁后重新运行 doctor 和验证。

## 10. Profile 注册后重新运行 Doctor

新增或更新 `tensor_cast/transformers/builtin_model/<model_type>.py` 后，重新运行 doctor：

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --output reports/<case_name>/doctor_after_profile.json
```

报告中应体现：

- `profile` 不为 null。
- `profile_validation.passed` 为 true。
- `candidate_profile_validation.passed` 为 true，或相关问题已被理解。
- `patch_reports` 与预期替换和跳过数量一致（MoE 层数、MLA 模块数等）。
- `human_questions` 为空，或已对照源码确认。

## 11. 运行跑通验证

verify 会执行一次基础 case 仿真，并将关键算子（attention 类、MoE gating 类）的实际调用次数与结构扫描推导的期望次数对账：

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --device <device_profile> \
  --num-queries 1 \
  --query-length 8 \
  --context-length 0 \
  --output reports/<case_name>/verify.json
```

视觉语言模型 prefill case 需要带图像参数（decode case 或无图像输入时 vision tower 不执行）：

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --image-batch-size 1 --image-height 224 --image-width 224 \
  --output reports/<case_name>/verify_vl_prefill.json
```

验证通过标准：`verify.json` 报告 `passed: true`（仿真跑通且所有 error 级 issue 清零）。

常见验证问题：

| 类型 | 常见原因 | 下一步 |
| --- | --- | --- |
| `SIMULATION_ERROR` | 仿真 case 崩溃（meta 兼容、placeholder 检查、动态 shape 等） | 报告内含 traceback 与 patch-discovery `ai_tasks`；按任务 prompt 修复后重跑 verify，无需手工采集日志 |
| `KEY_OP_MISSING` | attention 类算子从未被调用，模型回落到未适配 HF 模块 | 检查 ModelProfile 注册与模块替换；若消息中列出未分类 tensor_cast 算子，确认是否为新 attention 核心并登记 `_ATTENTION_CORE_OPS` |
| `OP_COUNT_MISMATCH` | 层数 override、MTP、vision 输入或部分层未替换 | 修复 Profile 或 case 输入 |
| `NO_TENSOR_CAST_OPS` | 仿真完全没有 tensor_cast 算子，模型未适配 | 注册 Profile 后重跑 |
| `MOE_NOT_ADAPTED` | 结构扫描发现 MoE 模块但无 MoE 补丁、无 gating 算子且无 TensorCast MoE 路径算子（init_routing_v2 等） | 注册或修复 MoE Profile 字段 |
| `STRUCTURE_SCAN_EMPTY` | 结构扫描未发现 attention 模块 | 检查模型构建路径与已安装源码 |
| `EXPECTATION_DEGRADED`（warning） | case 启用了 MTP 或 PP，次数校验降级 | 用基础 case（无 MTP、pp_size=1）重跑严格校验 |

注意：部分已正确适配的模型（如 DeepSeek V3.2）MoE gating 走标准 `torch.topk` 路径而非 tensor_cast 算子，此时验证依赖 doctor 的 MoE patch report 确认适配完整性，不视为问题。

## 12. 生成 ST Guardrail 用例

基于通过的验证结果生成回归用例（仅 passed 的验证会生成用例）：

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --device <device_profile> \
  --st-case-output reports/<case_name>/st_cases \
  --output reports/<case_name>/verify_with_st.json
```

预期输出：

```text
reports/<case_name>/verify_with_st.json
reports/<case_name>/st_cases/<case_name>.json
```

规则：

- 生成的用例 JSON 与 `tests/benchmark/models/test_model_regression.py` 的用例格式完全兼容，可将通过的用例提交到 `tests/benchmark/models/cases/` 加入现有模型的精度防护网。
- 验证失败不会生成用例；先修复问题再重新验证。
- 用例 `user_input` 来自标准化命令参数，而不是人工猜测。

## 13. Replay 或审计已有模型

Replay 模式用于检查流程是否能在不读取既有 Profile 作为答案的情况下，重新发现适配信息。

仅在 replay 或审计测试中使用 `--ignore-existing-profile`：

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/qwen3_vl_replay/command.txt \
  --ignore-existing-profile qwen3_vl \
  --ignore-existing-profile qwen3_vl_moe \
  --profile-draft-output reports/qwen3_vl_replay/qwen3_vl_replay_draft.py \
  --output reports/qwen3_vl_replay/doctor.json
```

Replay 约束：

- Replay 运行时不要把 `tensor_cast/transformers/builtin_model/qwen3_vl.py` 作为输入证据。
- 允许读取已安装的 `transformers` Qwen3-VL 源码，因为正常新模型适配也会使用已安装源码。
- 既有 TensorCast Qwen3-VL Profile 只能在 replay discovery 完成后作为 oracle 进行对比。

配套的 Qwen3-VL Blind Replay 测试使用仅配置 fixture（`tests/assets/model_config/qwen3_vl_tiny/config.json`），不需要下载权重：

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -k qwen3_vl -q
```

## 14. 验证检查清单

运行聚焦适配测试套：

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -q
```

检查 CLI 入口：

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter verify --help
```

如果运行时行为发生变化，运行相关 TensorCast 测试：

```bash
pytest tests/test_tensor_cast/test_runtime.py tests/test_tensor_cast/test_text_generate.py
```

如果适配涉及特定模型家族，请运行仓库中最接近的模型或 benchmark smoke 路径。

## 15. 提交检查清单

提交新适配前请确认：

- **不提交任何中间过程文件**：`reports/` 下的 command.txt、doctor.json、verify.json、failure.log、st_cases 等工作产物一律留在本地（目录已在 `.gitignore` 中），供开发过程和下游精度对齐流程在仓库外使用。
- 最终 built-in Profile 最小化且有源码依据；模型源码来源（库内 / 升级依赖 / 开源站获取）已在适配过程中明确。
- `candidate_profile_validation` 和 `profile_validation` 均通过。
- 所有 patch method 均来自确定性失败证据、AI 辅助和人工评审。
- 新增算子有声明 + 性能属性 + 单测，必要时已登记 attention 核心算子表。
- `verify.json` 报告 `passed: true`，或剩余 warning 已明确评审（报告本身不上库）。
- ST guardrail 用例仅从通过的验证生成；需要防回归看护时提交到 `tests/benchmark/models/cases/`。
- 未暂存临时本地文件、私有路径、原始内部笔记和 walkthrough。

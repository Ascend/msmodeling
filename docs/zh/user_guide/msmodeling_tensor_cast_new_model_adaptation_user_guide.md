# TensorCast 新模型适配开发指导

本文介绍将 Hugging Face 风格的新模型适配到 TensorCast 的实际操作流程。当模型在可靠仿真前需要补充 TensorCast 专用元数据、兼容性补丁或 profiling 证据时，可参考本文执行适配。

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
- 新支持模型的 profiling 证据与回归检查。

对于没有特殊结构的 dense text 模型，适配流程可能只需要一个最小模型 Profile。

## 2. 准备环境

请在已配置好的 Python 环境中，从 msModeling 仓库根目录执行所有命令。如果不在仓库根目录运行，请先设置 `PYTHONPATH`：

```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
```

检查模型适配 CLI 是否可用：

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter export-evidence --help
python -m cli.inference.model_adapter verify --help
```

如果您的 AI 助手支持加载项目 Skill，可使用模型适配 Skill：

```text
.agents/skills/model-adaptation
```

该 Skill 可辅助处理 doctor 报告、Profile 评审、人工确认点、AI 补丁任务、证据导出与验证。确定性的适配工具仍是解析、校验和通过/失败判断的准确信息来源。

## 3. 准备输入

在 `reports/` 下创建一个 case 目录，并准备以下输入：

| 输入 | 保存为 | 要求 |
| --- | --- | --- |
| TensorCast 仿真命令 | `reports/<case_name>/command.txt` | 与 profiling 导出匹配的完整 workload 命令。 |
| MindStudio Insight 原始 profiling 导出 | `reports/<case_name>/raw_insight.txt` | 原始表格导出，表头后需要紧跟 `Totals` 行。 |

原始 Insight 的 `Totals` 行很重要，其中 `Wall Duration(ms)` 会作为生成证据时的实测总前向耗时。

最小原始 Insight 形态如下：

```text
Name    Wall Duration(ms)    Self Time(ms)    Average Wall Duration(ms)    Max Wall Duration(ms)    Min Wall Duration(ms)    Occurrences
Totals  22.328398            22.328398        0.005782                     0.238545                 0.000000                 3862
FusedInferAttentionScore_*    3.055183         3.055183                    0.049277                 0.068602                 0.043541                 62
```

可选输入：

| 输入 | 保存为 | 用途 |
| --- | --- | --- |
| 已确认提示信息 | `reports/<case_name>/hints.yaml` | 记录已评审的 kernel 映射、调用次数、shape 说明或用户观察。 |
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
  --quantize-attention-action DISABLED \
  --quantize-linear-action W8A8_DYNAMIC
EOF
```

将匹配的 MindStudio Insight 原始导出复制到：

```text
reports/<case_name>/raw_insight.txt
```

完成后应得到以下文件：

```text
reports/<case_name>/command.txt
reports/<case_name>/raw_insight.txt
```

## 5. 添加可选提示信息

如果没有已确认的额外事实，可跳过本步骤。如果已经评审过 kernel 映射、调用次数、shape 或模块行为，请记录到 `reports/<case_name>/hints.yaml`。

```yaml
version: 1
hints:
  - kind: op_mapping_hint
    profiling_op: FusedInferAttentionScore
    tc_op: tensor_cast.attention.default
    confidence: medium
    note: "Derived from kernel name and matching call count."
```

常见 hint 类型如下：

| 类型 | 字段 | 用途 |
| --- | --- | --- |
| `op_mapping_hint` | `profiling_op`, `tc_op`, `confidence`, `note` | 将原始 Insight kernel 映射到 TensorCast 语义算子。 |
| `profiling_op_observation` | `op`, `count`, `confidence`, `note` | 记录 profiling 侧已确认的调用次数或解释。 |
| `tc_op_observation` | `op`, `count`, `shape_variants`, `confidence`, `note` | 记录 TensorCast 侧已确认的调用次数或 shape。 |

hint 应按需增量补充，不要为了填满文件而猜测字段。

## 6. 运行 Doctor

带 hints 运行 doctor：

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --profile-draft-output reports/<case_name>/<model_type>_draft.py \
  --output reports/<case_name>/doctor.json
```

不带 hints 运行 doctor：

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
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
| `adaptation_context` | 解析后的命令和标准化 workload 参数。 | 确认与 profiling workload 一致。 |
| `raw_insight_summary` | 解析后的 `Totals` 和主要 kernel。 | 确认总耗时和主要 kernel 合理。 |
| `candidate_profile` | 最小候选 `ModelProfile` 字段。 | 对照已安装模型源码进行评审。 |
| `candidate_profile_validation` | 候选 Profile 的确定性校验结果。 | 注册 Profile 前修复错误。 |
| `candidate_profile_draft` | Profile 草稿 Python 模块内容。 | 仅作为起点使用。 |
| `profile` | 已注册 Profile（如存在）。 | 作为正常适配的参考。 |
| `profile_validation` | 已注册 Profile 的校验结果。 | 如存在错误需修复。 |
| `evidence_draft` | 验证证据草稿。 | 后续导出并评审。 |
| `human_questions` | 需要用户确认的最小事实。 | 尽量通过 `hints.yaml` 回答。 |
| `ai_tasks` | 给 AI 助手的边界明确任务。 | 仅在评审确定性发现后使用。 |
| `patch_reports` | dry-run 补丁结果。 | 检查预期替换和跳过模块。 |
| `suggestions` | 推荐下一步操作。 | 用于决定下一轮迭代。 |

## 7. 检查已安装模型源码

以已安装的 `transformers` 实现作为准确信息来源：

```bash
python -c "import transformers; print(transformers.__file__)"
```

然后检查对应源码模块，通常为：

```text
transformers.models.<model_name>.modeling_<model_name>
```

不要仅凭模型名填写 Profile 字段。应在已安装源码中确认真实类名、模块路径、配置字段和 forward 行为。

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
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --patch-failure-file reports/<case_name>/failure.log \
  --profile-draft-output reports/<case_name>/<model_type>_draft_with_patch.py \
  --output reports/<case_name>/doctor_with_failure.json
```

如果没有 hints 文件，请省略 `--hints-file`。

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
- 添加补丁后重新运行 doctor、smoke 和验证。

## 10. Profile 注册后重新运行 Doctor

新增或更新 `tensor_cast/transformers/builtin_model/<model_type>.py` 后，重新运行 doctor：

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --output reports/<case_name>/doctor_after_profile.json
```

如果没有 hints 文件，请省略 `--hints-file`。

预期输出：

```text
reports/<case_name>/doctor_after_profile.json
```

报告中应体现：

- `profile` 不为 null。
- `profile_validation.passed` 为 true。
- `candidate_profile_validation.passed` 为 true，或相关问题已被理解。
- `patch_reports` 与预期替换和跳过数量一致。
- `human_questions` 为空，或已通过评审后的 hints 处理。

## 11. 导出证据

从注册 Profile 后的 doctor 报告中导出已评审的 `evidence_draft`：

```bash
python -m cli.inference.model_adapter export-evidence \
  --doctor-report reports/<case_name>/doctor_after_profile.json \
  --output reports/<case_name>/evidence.yaml
```

预期输出：

```text
reports/<case_name>/evidence.yaml
```

验证前检查 `evidence.yaml`：

| 字段 | 检查项 |
| --- | --- |
| `model.model_id` | 与已适配模型一致 |
| `model.raw_command` | 与 `command.txt` 一致 |
| `cases[].name` | 稳定且具有描述性 |
| `cases[].input` | 与 profiling workload 一致 |
| `expected.total_forward` | 来自 `raw_insight:Totals.wall_duration_ms`，并具备合理容差 |
| `expected.major_ops` | 包含已评审的 TensorCast 语义算子、调用次数、来源和置信度 |
| `shape_hints` | 仅在已确认或有帮助时出现 |
| `accepted_gaps` | 仅用于已评审的后端融合或建模差距 |

导出命令只执行确定性的格式转换，不能替代人工评审。

## 12. 验证证据

运行验证：

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --evidence-file reports/<case_name>/evidence.yaml \
  --device <device_profile> \
  --output reports/<case_name>/verify.json
```

如果 evidence 中已存在 `model.model_id`，可省略位置参数中的模型 ID：

```bash
python -m cli.inference.model_adapter verify \
  --evidence-file reports/<case_name>/evidence.yaml \
  --device <device_profile> \
  --output reports/<case_name>/verify.json
```

预期输出：

```text
reports/<case_name>/verify.json
```

常见验证问题：

| 类型 | 常见原因 | 下一步 |
| --- | --- | --- |
| `OP_MAPPING_MISSING` | evidence 算子名与实际 TensorCast 算子不一致，或后端融合没有直接对应的 TensorCast 算子 | 修复映射、添加 hints，或记录已评审的 accepted gap |
| `OP_COUNT_MISMATCH` | 层数、重复结构、MTP、MoE 路由或并行配置不匹配 | 修复 Profile 或 case 输入 |
| `LATENCY_MODEL_MISMATCH` | profiling 映射、性能数据库、融合策略或设备 Profile 存在问题 | 评审 profiling 覆盖范围和容差 |
| `PROFILING_SHAPE_MISSING` | 原始 profiling 缺少 shape 细节或数据库覆盖 | 添加 shape hints 或 profiling 数据 |
| `PATCH_SEMANTICS_MISSING` | 运行时补丁没有路由到预期 TensorCast 路径 | 修复补丁并重新运行 doctor |
| `COMMUNICATION_GAP` | TP/DP/EP 通信未被 evidence 覆盖 | 添加通信证据或 accepted gap |

当 `verify.json` 报告 `passed: true`，或所有剩余问题都已明确评审并记录为 accepted gap 时，该 case 才可认为就绪。

## 13. 生成 ST Guardrail 用例

基于验证结果生成 ST 用例：

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --evidence-file reports/<case_name>/evidence.yaml \
  --device <device_profile> \
  --st-case-output reports/<case_name>/st_cases \
  --output reports/<case_name>/verify_with_st.json
```

预期输出：

```text
reports/<case_name>/verify_with_st.json
reports/<case_name>/st_cases/*.json
```

规则：

- 验证通过会生成 `verified` ST 用例。
- 验证失败会生成 `draft` ST 用例。
- 不要将 draft 用例作为 verified guardrail 提交。
- 用例 `user_input` 应来自标准化命令和 evidence 输入，而不是人工猜测。

## 14. Replay 或审计已有模型

Replay 模式用于检查流程是否能在不读取既有 Profile 作为答案的情况下，重新发现适配信息。

仅在 replay 或审计测试中使用 `--ignore-existing-profile`：

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/qwen3_vl_replay/command.txt \
  --raw-insight-file reports/qwen3_vl_replay/raw_insight.txt \
  --ignore-existing-profile qwen3_vl \
  --ignore-existing-profile qwen3_vl_moe \
  --profile-draft-output reports/qwen3_vl_replay/qwen3_vl_replay_draft.py \
  --output reports/qwen3_vl_replay/doctor.json
```

Replay 约束：

- Replay 运行时不要把 `tensor_cast/transformers/builtin_model/qwen3_vl.py` 作为输入证据。
- 允许读取已安装的 `transformers` Qwen3-VL 源码，因为正常新模型适配也会使用已安装源码。
- 既有 TensorCast Qwen3-VL Profile 只能在 replay discovery 完成后作为 oracle 进行对比。

## 15. Qwen3-VL Blind Replay 测试

适配自动化测试套包含一个仅配置的 Qwen3-VL tiny fixture：

```text
tests/assets/model_config/qwen3_vl_tiny/config.json
```

该 fixture 可用于在不下载权重的情况下测试 replay 流程。

运行聚焦回归测试：

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -k qwen3_vl -q
```

该测试应验证 doctor 在隐藏 Profile 的情况下能够重新发现：

- `model_type=qwen3_vl`。
- `model_family=qwen3_vl`。
- `visual_module_path=visual`。
- `language_module_path=language_model`。
- `visual_layers_module_path=visual.blocks`。
- visual merger linear mappings。
- visual MLP linear mappings。
- 来自失败日志的 Qwen3-VL placeholder 或 dynamic-mask patch authoring 证据。

推荐运行更完整测试：

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -q
```

Replay discovery 完成后，可将 replay candidate 和 patch task 预期与既有 Qwen3-VL 适配作为 oracle 对比。oracle 对比不能用于提前填充 replay candidate。

## 16. 验证检查清单

运行聚焦适配测试套：

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -q
```

检查 CLI 入口：

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter export-evidence --help
python -m cli.inference.model_adapter verify --help
```

如果运行时行为发生变化，运行相关 TensorCast 测试：

```bash
pytest tests/test_tensor_cast/test_runtime.py tests/test_tensor_cast/test_text_generate.py
```

如果适配涉及特定模型家族，请运行仓库中最接近的模型或 benchmark smoke 路径。

## 17. 提交检查清单

提交新适配前请确认：

- 必需命令和原始 profiling 输入保存在 `reports/<case_name>/` 下；如果不能提交，则在仓库外记录清楚。
- 最终 built-in Profile 最小化且有源码依据。
- `candidate_profile_validation` 和 `profile_validation` 均通过。
- 所有 patch method 均来自确定性失败证据、AI 辅助和人工评审。
- `evidence.yaml` 已从 doctor 输出导出并完成评审。
- `verify.json` 已通过，或剩余 gap 已记录并接受。
- ST guardrail 用例仅从已验证或明确接受的 evidence 生成。
- 未暂存临时本地文件、私有路径、原始内部笔记和 walkthrough。

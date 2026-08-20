# Model Diagnostics 使用与模型接入指南

`model_diagnostics` 是一个源码内工具，用于捕获一次 msmodeling Runtime 执行，并按静态
Theory Spec 校验语义算子是否完整以及 Tensor shape/dtype 是否一致。它位于 `tools/` 下，
不作为顶层公共包发布，也不进入 wheel。

本文面向两类使用者：

1. 使用 Run Profile 运行一次诊断或导出 HTML 报告；
2. 为一类新模型编写 Spec、复用 Theory fragment，并补充端到端测试。

架构背景与设计理由见 [设计文档](../../docs/design/model_diagnostics_design.md)。本文只描述
当前代码已经支持的使用方式和字段契约。

## 1. 目录与职责

```text
tools/model_diagnostics/
├── profiles/                 # 可直接运行的 Run Profile 示例
├── specs/
│   ├── <model>_v1.yaml       # 模型类别 Spec：匹配、region、模型差异和组合关系
│   └── theory_fragments/     # 可复用 Theory/Runtime/comparison 片段
├── application/              # 诊断流程编排
├── specification/            # Profile/Spec 严格加载、表达式和 fragment 组合
├── sources/                  # Runtime 捕获与 Theory 执行记录
├── organization/             # 将有序调用切分为 region/layer/stage
├── comparison/               # shape/dtype 比较策略
└── rendering/                # Console、Runtime HTML、Comparison HTML

tests/
├── smoke/test_model_diagnostics.py       # 日常增量门禁的快速基础 guard
└── regression/model_diagnostics/         # 完整单元、契约、集成和真实模型 E2E
```

Run Profile 和 Model Spec 都是 YAML，但职责完全不同：

| YAML | 回答的问题 | 是否由普通使用者编写 |
| --- | --- | --- |
| `profiles/*.yaml` | 本次运行哪个模型、阶段、batch/query、量化和并行配置 | 是 |
| `specs/*.yaml` | 这类模型应有哪些 region/stage/operator，如何组织和比较 | 模型适配者 |
| `specs/theory_fragments/*.yaml` | 哪些 decoder/MTP 协议可以跨 region 或模型复用 | 模型适配者 |

## 2. 命令行入口

必须从 msmodeling 源码根目录执行：

```text
python -m tools.model_diagnostics PROFILE
    [--runtime-report [PATH]]
    [--theory-compare]
    [--comparison-report [PATH]]
    [--fail-only]
    [--show-all]
```

| 参数 | 作用 | 约束 |
| --- | --- | --- |
| `PROFILE` | 本次 Runtime capture 的 Run Profile YAML | 必须是存在的文件 |
| `--runtime-report [PATH]` | 导出完整 Runtime HTML | PATH 可省略；不触发 Theory 比较 |
| `--theory-compare` | 执行 Theory↔Runtime 比较并输出 Console | 与本次 Runtime capture 共用一个 Artifact |
| `--comparison-report [PATH]` | 额外导出 Comparison HTML | 必须同时给出 `--theory-compare`；PATH 可省略 |
| `--fail-only` | Console 只展开非 PASS finding | 未执行比较时不产生额外效果 |
| `--show-all` | 展开捕获日志、全部 finding 和 limitations | 与 `--fail-only` 同时出现时优先 |

### 2.1 只捕获 Runtime

```bash
python -m tools.model_diagnostics \
  tools/model_diagnostics/profiles/decode_example.yaml
```

裸命令只运行一次 Profile、捕获 Runtime Artifact 并打印简洁摘要。它不会加载 Theory、
不会比较，也不会生成文件：

```text
Runtime capture completed: 52 operator calls
Qwen/Qwen3-8B | decode | batch=1 query=1 context=128 | TP=1
No report or comparison requested.
```

### 2.2 导出完整 Runtime HTML

```bash
# 使用默认路径
python -m tools.model_diagnostics profile.yaml --runtime-report

# 指定路径
python -m tools.model_diagnostics profile.yaml --runtime-report work/runtime.html
```

Runtime HTML 包含完整有序算子调用、Tensor slot、shape/dtype 和来源引用，适合定位 Runtime
阶段边界。该功能默认关闭；HTML 仅用于阅读，不支持离线重放 Artifact。

### 2.3 执行 Theory↔Runtime 比较

```bash
python -m tools.model_diagnostics profile.yaml --theory-compare
```

默认输出 Console 详情。结果底部包含统一状态、运行上下文、PASS/非 PASS 计数和失败位置。
任何非 PASS finding（包括内部的 `INCOMPLETE`、`UNSUPPORTED`、`SKIP`）都会使命令返回失败。

```bash
# 只显示非 PASS finding；仍保留摘要
python -m tools.model_diagnostics profile.yaml --theory-compare --fail-only

# 显示全部 finding、limitations 和 Runtime 捕获日志
python -m tools.model_diagnostics profile.yaml --theory-compare --show-all
```

同时给出 `--fail-only --show-all` 不报错，`--show-all` 优先。

### 2.4 导出 Comparison HTML

```bash
# 使用默认路径
python -m tools.model_diagnostics profile.yaml \
  --theory-compare --comparison-report

# 同时导出两类报告，仍只捕获一次 Runtime
python -m tools.model_diagnostics profile.yaml \
  --runtime-report work/runtime.html \
  --theory-compare \
  --comparison-report work/diagnostics.html
```

`--comparison-report` 必须与 `--theory-compare` 同时使用。两个默认报告共享目录：

```text
outputs/model_diagnostics/<profile-stem>-<timestamp>/
├── runtime.html
└── theory_runtime.html
```

自定义的两个报告路径不能指向同一文件，也不能指向已有目录；父目录自动创建，文件采用
同目录临时文件加原子替换写入。

### 2.5 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 捕获成功，或比较的所有 finding 均为 PASS |
| `1` | 比较产生任一非 PASS finding |
| `2` | CLI 参数、Profile、捕获或报告写入错误 |

## 3. Run Profile YAML 编写说明

`tools/model_diagnostics/profiles/` 只保留两份可直接运行的样例，并随时覆盖为**当前
正在适配类别**的代表配置（不要在此目录另增 example YAML）。这两份文件供用户本地修改与
试跑，**不是**测试夹具；回归测试不得依赖其内容。

- [prefill_example.yaml](profiles/prefill_example.yaml)：当前为 Qwen3 MoE prefill；
- [decode_example.yaml](profiles/decode_example.yaml)：当前为 Qwen3 MoE decode，
  **默认开启 MTP**（`num_mtp_tokens>0` 且合法 window）。

这两份文件是**完整字段参考版**：列出 run profile 的全部可配字段，每个字段上方注释
是否可缺省及其默认值。实际使用时复制一份，只保留本次需要覆盖的字段即可。以下亦为一份
完整、可运行的 decode Profile 基准：

```yaml
schema_version: "1"
model_name: Qwen/Qwen3-30B-A3B
entrypoint: text_generate
phase: decode
batch_size: 1
query_length: 3
context_length: 128
num_mtp_tokens: 2
parallel:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  data_parallel_size: 1
  expert_parallel_size: 1
  moe_data_parallel_size: 1
selected_language_layers: [0]
selected_stage_regions: [input, output]
num_hidden_layers_override: 1
do_compile: true
device: TEST_DEVICE
quantize_linear_action: DISABLED
word_embedding_tp: null
enable_redundant_experts: false
enable_external_shared_experts: false
```

实际使用时建议从两个仓库样例中选择与 phase 对应的一份复制，再只保留本次确实需要的
字段；不要为了“完整”而在每个本地 Profile 中重复所有默认值。

### 3.1 字段约束

| 字段 | 必填 | 默认值 | 约束与含义 |
| --- | --- | --- | --- |
| `schema_version` | 否 | `"1"` | 只支持 `1`；字符串或整数均会规范为字符串 |
| `model_name` | 是 | 无 | 非空模型 ID 或本地模型标识；传给 msmodeling ModelRunner |
| `entrypoint` | 否 | `text_generate` | 非空字符串；还参与 Spec `matches.entrypoints` 匹配 |
| `phase` | 是 | 无 | `prefill` 或 `decode` |
| `batch_size` | 是 | 无 | 正整数 |
| `query_length` | 是 | 无 | 正整数；MTP decode 必须满足 `Q >= MTP + 1` |
| `context_length` | 否 | `null` | 非负整数；decode 通常应显式填写，Theory 中 `null` 按 0 处理 |
| `num_mtp_tokens` | 否 | `0` | 非负整数；只在合法 MTP decode 窗口启用 MTP region |
| `parallel` | 否 | 各维度均为 1 | mapping；见下表 |
| `selected_language_layers` | 否 | 所有已执行 language 层 | 非空、非负整数列表；排序去重；部分越界 warning 后跳过；全部越界则报错 |
| `selected_stage_regions` | 否 | Spec 中全部非分层 region | 非空字符串列表；仅在需要限制 input/output 等 region 时使用 |
| `num_hidden_layers_override` | 否 | `0` | 非负整数；控制本次实际执行的 language decoder 层数 |
| `do_compile` | 否 | `true` | 必须是布尔值；正式 Theory↔Runtime E2E 使用编译态语义算子 |
| `device` | 否 | `TEST_DEVICE` | 非空字符串 |
| `quantize_linear_action` | 否 | `DISABLED` | 见量化枚举 |
| `word_embedding_tp` | 否 | `null` | `col`、`row` 或省略；分别按 hidden/vocab 维切分 embedding |
| `enable_redundant_experts` | 否 | `false` | 按 msmodeling EP shard 规则增加冗余专家副本；要求 `expert_parallel_size > 1` |
| `enable_external_shared_experts` | 否 | `false` | 分配独立 rank 运行 shared experts；要求 `expert_parallel_size > 1` 且模型含 shared experts |

`parallel` 支持以下正整数（均为可缺省、默认 `1`）：

```yaml
parallel:
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
  data_parallel_size: 1
  expert_parallel_size: 1
  moe_data_parallel_size: 1    # --moe-dp-size；别名 moe_dp_size
```

> MoE 张量并行 `--moe-tp-size`（MTPt）**本模块固定为 1**，不支持配置。
> profile 中写入 `moe_tensor_parallel_size` / `moe_tp_size` 会直接报错。

量化枚举为：

```text
DISABLED
W8A16_STATIC  W8A8_STATIC  W4A8_STATIC
W8A16_DYNAMIC W8A8_DYNAMIC W4A8_DYNAMIC
FP8 MXFP4
```

### 3.2 禁止字段与 HF config

Profile 严格拒绝未知字段，并明确禁止：

```text
model_config
quantization_config
capture
```

开发者不应在 Profile 复制 HF config。捕获流程会根据 `model_name` 构建 ModelRunner，读取
模型/HF config，并把实际的 `model_type`、hidden size、head 数、dtype、有效层数等写入
`ModelRunContext.model_config`。Spec 匹配和 Theory 表达式消费的是这份捕获后上下文，因此
Profile 与模型真实配置不会形成两份权威来源。

### 3.3 language 与 MTP 选层

`num_hidden_layers_override` 决定 Runtime 执行多少个 language 层；
`selected_language_layers` 只决定其中哪些物理层参与组织、比较和报告：

```yaml
num_hidden_layers_override: 6
selected_language_layers: [0, 3, 5]
```

MTP 选层与上述字段无关，采用内置代表层策略：

```text
MTP=0  -> 不校验 MTP layer
MTP=1  -> layer[0]
MTP>=2 -> layer[0] 和 layer[1]
```

合法 MTP Profile 示例：

```yaml
model_name: Qwen/Qwen3-0.6B
phase: decode
batch_size: 1
query_length: 3
context_length: 128
num_mtp_tokens: 2
num_hidden_layers_override: 1
quantize_linear_action: W8A8_DYNAMIC
```

## 4. 端到端测试用例编写

这是本工具除 CLI 外的第二种核心用法。完整用例位于
`tests/regression/model_diagnostics/e2e/`，应直接组织 `DiagnosticsRunProfile`，而不是依赖
示例 YAML；只有公开 CLI smoke 才需要从 YAML 文件入口测试。

### 4.1 为什么 E2E 直接构造 Profile

- 测试意图直接可见，不受示例 YAML 默认值变化影响；
- 参数化模型、phase、量化、MTP 更自然；
- 同一个捕获 Artifact 可同时检查 Context、Runtime calls、组织结果和最终比较；
- Profile YAML Loader 的正确性由 application/specification 单测独立覆盖。

### 4.2 推荐结构

```python
from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, FindingStatus, ParallelContext
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.specification import DiagnosticsRunProfile
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile

profile = DiagnosticsRunProfile(
    schema_version="1",
    model_name="Vendor/NewModel-7B",
    entrypoint="text_generate",
    phase=ExecutionPhase.DECODE,
    batch_size=1,
    query_length=1,
    context_length=128,
    num_mtp_tokens=0,
    parallel=ParallelContext(),
    selected_language_layers=None,
    selected_stage_regions=(),
    num_hidden_layers_override=1,
    do_compile=True,
    device="TEST_DEVICE",
    quantize_linear_action="DISABLED",
    word_embedding_tp=None,
)

artifact = capture_artifact_for_profile(profile)
result = create_model_diagnostics_application().run_profile_against_artifact(profile, artifact)

assert artifact.operator_calls
assert result.summary.overall_status is FindingStatus.PASS
assert_diagnostics_passed(result)
```

不要把已知差异写成“预期 FAIL”的正常 E2E。模块的合格标准是用例最终全部 PASS；
`INCOMPLETE` 也属于失败，必须补 Theory、修正 Runtime 组织边界或明确修复模型实现。

### 4.3 新模型类别的最低覆盖

每个型号至少需要：

1. 一个非量化 prefill；
2. 一个非量化 decode；
3. 一个对应端到端量化用例；
4. 支持 MTP 时，一个 `query_length >= num_mtp_tokens + 1` 的 MTP decode。

CLI/YAML E2E 验证的是整个 `model_diagnostics` 共用的命令行、Profile Loader、捕获、比较和
输出链路，不属于每类模型的覆盖矩阵。模块范围保留一个有代表性的 CLI/YAML E2E 即可；
它不能替代各模型直接构造 Profile 的 E2E。

量化用例除最终 PASS 外，还应检查 Context 的量化配置和关键 Runtime kernel。主 decoder
projection 与 MTP fusion/predictor projection 可量化；embedding、attention/SwiGLU 和普通/
proposal lm-head 是否量化必须以模型 Runtime 实际行为及 Theory 契约为准。

### 4.4 smoke 与 regression

- `tests/smoke/test_model_diagnostics.py` 只验证快速 eager capture 和 Spec/request 解析，目标
  小于 10 秒；eager ATen 流不宣称与编译态 TensorCast Theory 完全一致。
- 完整 Theory↔Runtime、所有型号、量化和 MTP 矩阵保留在 regression。
- 不为本模块增加自定义层级 marker；遵循仓库 `tests/SKILL.md` 和 `tests/README.md`。

## 5. Model Spec YAML

一个模型类别对应 `specs/<spec_id>.yaml`。文件名 stem 必须与 `spec_id` 一致，composition
root 会加载 `specs/*.yaml` 并要求一个 Runtime Context 恰好匹配一个 Spec。

### 5.1 顶层字段

```yaml
schema_version: "1"
spec_id: new_model_dense_v1
spec_version: 1.0.0
model_category: new_model_dense
matches:
  entrypoints: [text_generate]
  model_types: [new_model]
  required_features: []       # 可省略
operator_aliases: {}          # 可省略
regions: {...}
```

| 字段 | 约束 |
| --- | --- |
| `schema_version` | 当前 Spec schema 版本；必须符合 Loader 支持值 |
| `spec_id` | 全局稳定且非空；建议 `<family>_<variant>_v1` |
| `spec_version` | 非空版本字符串；Theory/组织契约变化时更新 |
| `model_category` | 供 activation/诊断语义使用的稳定模型类别 |
| `matches.entrypoints` | 可选字符串列表；非空时必须匹配 Context entrypoint |
| `matches.model_types` | 可选字符串列表；来自捕获后 HF/model config 的 `model_type` |
| `matches.required_features` | 可选字符串列表；要求全部包含于 `model_config.features` |
| `operator_aliases` | 可选 `原始名: 规范名`；覆盖内置 alias，不改写 Artifact |
| `regions` | 非空且保持声明顺序；region id 必须唯一 |

匹配是精确的，不做模糊 fallback：零个匹配报 `UnsupportedModelSpec`，多个匹配报
`AmbiguousModelSpec`。新增 Spec 时必须检查它不会与现有 Spec 重叠。

### 5.2 Region、layer 和 stage

非重复 region 直接声明 `stages`：

```yaml
regions:
  input:
    stages:
      - id: embedding
        source_options:
          theory: ...
          runtime: ...
```

重复层使用 layout rule：

```yaml
language:
  layer_layout_rule:
    strategy: repeat
    layer_kind: dense
    count_from: model_config.effective_num_hidden_layers
  layer_specs:
    dense:
      include_fragment: new_model_decoder_v1
```

`layer_layout_rule` 支持两种严格结构：

- `strategy: repeat`：使用 `layer_kind` 将同一种 layer 重复 `count_from` 次；
- `strategy: prefix_then_repeat`：前 `prefix_count_from` 层使用
  `prefix_layer_kind`，其余层使用 `repeated_layer_kind`，适用于“Dense 前缀 + MoE 后缀”；
- 规则引用的每个 layer kind 都必须存在于 `layer_specs`；
- `count_from` 是点分 Context 路径，根只能是 `model_config` 或
  `quantization_config`，最终值必须是非负整数；
- language 通常使用 `model_config.effective_num_hidden_layers`；
- MTP 使用 materialize 阶段派生的 `model_config.effective_num_mtp_layers`。

DeepSeek V3 的完整示例直接整包导入两种 decoder fragment，不使用 stage group：

```yaml
layer_layout_rule:
  strategy: prefix_then_repeat
  count_from: model_config.effective_num_hidden_layers
  prefix_layer_kind: dense
  repeated_layer_kind: moe
  prefix_count_from: model_config.first_k_dense_replace
layer_specs:
  dense:
    include_fragment: deepseek_v3_dense_decoder_v1
  moe:
    include_fragment: deepseek_v3_moe_decoder_v1
```

`effective_num_hidden_layers` 是本次 Profile 实际捕获的层数；
`first_k_dense_replace` 来自捕获后的 HF/model config。TensorCast 的 DeepSeek V3.2
实现同样以 `layer_idx < first_k_dense_replace` 选择 Dense 层，之后选择 MoE 层。

一个 `layer_specs.<kind>` 必须且只能选择一种定义方式：

```text
stages | include_fragment | compose
```

优先级建议是：能整包复用 `include_fragment` 就不要逐 stage 重写；只有新模型确实改变
decoder 结构时才新增 decoder fragment。

### 5.3 Stage 与 Theory Tensor 声明

直接 stage 必须包含 `id`，并且必须且只能选择 `source_options` 或 `include_stage`：

```yaml
- id: example
  source_options:
    theory:
      modules:
        - name: semantic_operator
          activation: optional_policy_id
          tensors:
            INPUT[0]:  {shape: "[T, H]", dtype: ACT}
            OUTPUT[0]: {shape: "[T, H]", dtype: OUT}
    runtime:
      boundary_operators: [runtime_boundary]
      ignored_operators: [view, slice]
```

Theory 约束：

- `modules` 必须非空；每项只允许 `name`、`activation`、`tensors`；
- Tensor key 只允许 `INPUT[n]` 或 `OUTPUT[n]`，`n` 为非负整数；
- 每个 Tensor 至少声明 `shape` 或 `dtype`；
- Theory 声明哪些 slot 就比较哪些 slot。Runtime 多出的 weight、scale、cache 或控制 slot
  不会因为存在就自动参与比较；
- 不要为“看起来完整”机械声明 Runtime 所有输入。Theory 只描述稳定语义契约；
- `activation` 必须是已注册 policy。当前内置：`lm_head_token_selection`、
  `mtp_enabled`、`non_mtp_lm_head`。

Runtime 约束：

- `boundary_operators` 必填，可以有多个候选规范名；
- `ignored_operators` 可省略，空列表也应省略；
- ignored 只用于 stage 组织，Artifact 始终保留原始完整调用；
- `boundary_operators` 太普通时容易误切分，应先捕获 Runtime HTML，优先选择稳定的语义
  wrapper。只有 Runtime 没有更具体的可观测边界时才使用 `mm` 等通用线性算子；DeepSeek V3
  shared expert 的首个 gate/up linear 就属于这一例外，并同时列出各量化形态；
- operator 名会进行统一规范化，例如 `aten.mm.default -> mm`。模型 Spec 只在必要时用
  `operator_aliases` 覆盖内置映射。

## 6. Theory shape/dtype 表达式

shape 必须求值为非空 tuple/list，通常写成字符串：

```yaml
shape: "[B, Q, H]"
shape: "[T, 2 * Ftp]"
shape: "[B, max(MTP + 1 - LAYER * MTP, 1)]"
```

支持：

- 运算：`+ - * / // %`；整数 `/` 必须整除；
- 一元 `+/-`；
- 函数：`max`、`min`、`ceil`、`abs`；
- `?` 或 `unknown` 表示缺少 Theory 证据，会产生 `INCOMPLETE`，不能用来让正式 E2E PASS；
- 禁止 Python `eval`、属性访问、任意函数、关键字参数和未注册变量。

### 6.1 内置 shape 符号

以下变量由捕获后的 `ModelRunContext` 和 HF/model config 派生，不是 Profile 字段：

| 符号 | 定义 |
| --- | --- |
| `B` | DP 切分后的本地 batch：`ceil(batch_size / DP)` |
| `Q` | `query_length` |
| `C` | `context_length`，省略时为 0 |
| `S` | `C + Q` |
| `T` | 本地 token 行数：`B * Q` |
| `H` | hidden size |
| `V` | vocabulary size |
| `F` | intermediate size |
| `Nh` / `Nkv` | attention heads / KV heads |
| `Dh` | head dimension；未声明时为 `H / Nh` |
| `TP` / `DP` / `EP` | tensor/data/expert parallel size |
| `MLP_TP` / `OTP` / `LMTP` | MLP、o-projection、lm-head 使用的 TP size |
| `Lh` | 本地 attention heads：`Nh / TP` |
| `Lkv` | 本地 KV heads：`max(Nkv / TP, 1)` |
| `Ftp` | 本地 FFN size：`F / MLP_TP` |
| `Vtp` | 本地 lm-head vocab：`V / LMTP` |
| `EV` / `EH` / `EO` | embedding 本地 vocab、权重 hidden、输出 hidden |
| `Bs` | paged-attention block size |
| `Nblk` | 本次本地 paged-attention block pool 数 |
| `Mb` | 每个 sequence 最大 block 数 |
| `Rout` | 普通 lm-head 行数：prefill 为 `B`，decode 为 `T` |
| `MTP` | `num_mtp_tokens` |
| `LAYER` | 当前重复 layer index；在 layer Theory 物化时注入 |

DeepSeek V3 还使用以下由 HF/model config 派生的符号：

| 符号 | 定义 |
| --- | --- |
| `Qlora` / `KVlora` | query / KV LoRA rank |
| `QKnope` / `QKrope` / `Vh` / `Hmla` | MLA 的 non-RoPE、RoPE、value head 维度及本地输出宽度 |
| `Dsa_k` | Runtime 已稳定暴露的 DSA 有效 top-k；`Dsa_k=min(index_topk,S)` |
| `Nshared` / `Fshared` | shared expert 数及总中间宽度；`Fshared=moe_intermediate_size*Nshared` |
| `MOE_COMBINE_DTYPE` | routed expert 加权归并的 dtype；DeepSeek V3.2 为 `float32`，DeepSeek V3/GLM5/Kimi K2 跟随激活 dtype |
| `MOE_GATE_TOKENS` | routed gate 观测的 token 数；raw-logits 族（DeepSeek V3/V3.1、GLM-5/5.1、Kimi-K2-Base）EP>1 时为 `T`，其余布局/型号为 `Tmoe` |

这些值由 Runtime 模型加载完成后从其 text config 捕获；实现兼容模型直接暴露的
`text_config`、根 `hf_config` 的 text config，以及旧版 `config` 入口。
`ModelRunContext.model_config`，Profile 不重复声明，也不从算子列表反推。

### 6.2 通用 MoE shape 符号

MoE Spec 可以使用以下符号。配置字段来自捕获后的 HF/model config；并行度来自 Profile
的 `parallel`，但所有公式均在 `context_env.py` 中统一求值，YAML 不应重复实现。

| 符号 | 来源或公式 | 含义 |
| --- | --- | --- |
| `E` | `num_experts` 或 `n_routed_experts` | routed expert 总数 |
| `Ktop` | `num_experts_per_tok` | 每个 token 选择的 routed expert 数 |
| `Fmoe` | `moe_intermediate_size` | 单个 routed expert 的全局 intermediate size；不能用 Dense 的 `F` 代替 |
| `MTPt` | 固定为 `1` | MoE tensor-parallel size；本模块明确拒绝 `--moe-tp-size>1` |
| `MDP` | `parallel.moe_data_parallel_size` | MoE data-parallel size，用于并行布局校验 |
| `Fe` | `Fmoe / MTPt`；当前即 `Fmoe` | 当前 rank 上单个 expert 的 intermediate size |
| `Tmoe` | 见下方分段公式 | 进入 MoE dispatch 前的 token 行数 |
| `Te` | 见下方 dispatch 公式 | dispatch 后当前 EP rank 执行的 expert-token 行数 |

`Tmoe` 与 TensorCast `ParallelMoELayer._dp_transform_enter` 的 token-domain 转换保持一致。
设 `T=B*Q`：

```text
EP > 1 且 DP != EP:  Tmoe = ceil(T / TP)
EP = 1 且 DP != 1:   Tmoe = T * DP
其他情况:            Tmoe = T
```

第一种情况使用 MoE EP 路径，需要把普通 TP/DP token domain 转为 EP dispatch 输入；第二种
情况没有跨 EP dispatch，但普通 DP group 会先 all-gather；其余布局不做 token-domain 转换。
`MDP` 参与 `MTPt * MDP * EP == TP * DP` 的 MoE 并行布局校验，但不直接替代 Runtime 上述
转换公式，因此不能把 `Tmoe` 简化为 `T*MDP`。

令 routed expert-token 总数为：

```text
R = Tmoe * Ktop
```

当 `EP=1` 时没有跨 rank dispatch：

```text
Te = R = Tmoe * Ktop
```

当 `EP>1` 时，`Te` 必须复现 TensorCast `FusedMoETensorCast.get_split_sizes`，不能简单写成
`R/EP`。当前 Theory 按以下整数分配算法计算：

```text
per_expert = R // Eglobal
remainder  = R % Eglobal
tokens(expert[i]) = per_expert + (1 if i < remainder else 0)

local_share = sum(tokens(e) for e in experts_owned_by_current_ep_rank)
Te = local_share * EP
```

其中 `Eglobal` 包括启用的 redundant routed experts；expert ownership 使用与 Runtime 相同的
`assign_experts` 规则。启用 external shared experts 时，还要先为 external ranks 分配 shared
token，再在剩余 routing ranks 间分配 routed experts。乘以 `EP` 表示 Runtime 的解析模型
假设各发送 rank 具有同样的 split，并汇总对称 all-to-all 后当前 rank 收到的 token 数。

因此，即使 `R < EP` 或 `R` 不能被 expert 数整除，公式仍通过整数商和余数得到确定结果；
不要用浮点平均、向上取整或 `Tmoe*Ktop/EP` 替代它。

### 6.3 `Rtgt` 与 `Rprop`

这两个符号描述 MTP 中两个不同语义窗口：

```text
Rtgt  = B * (MTP + 1)
Rprop = B
```

- `Rtgt` 是 target verification 的行数。每个请求包含 `MTP` 个候选 token 加 1 个 bonus/
  target 行，因此 target selection 和 target lm-head 使用 `[Rtgt, ...]`；
- `Rprop` 是单个 proposal layer 为每个本地请求产生的 proposal 行数，因此 proposal
  selection、proposal lm-head 使用 `[Rprop, ...]`；
- 不要因为某次 Runtime 捕获恰好只有 1 行就把 `Rtgt` 写成 `Rprop`；两者在 `MTP>0`
  时语义和 shape 不同；
- MTP decode 的合法窗口要求 `phase=decode`、`MTP>0` 且 `Q >= MTP + 1`；否则 MTP
  region 不启用或 Profile 被拒绝。

### 6.4 dtype 符号

| 符号 | 来源/含义 |
| --- | --- |
| `D` | Runtime 实际模型 dtype |
| `ACT` | activation dtype |
| `LINEAR_IN` | linear 输入 dtype；动态量化时通常为 int8 |
| `WEIGHT` | weight dtype |
| `SCALE` | scale dtype，默认 float32 |
| `ACC` | accumulation dtype |
| `OUT` | output dtype，默认等于 ACT |
| `int64` | 固定 int64 |

Theory 绑定 Runtime 实际执行 dtype，而不是仅照抄 HF 声明 dtype。例如 HF 声明 BF16、
当前 Runtime 实际执行 FP16 时，比较以 FP16 为准，同时 limitation 保留声明差异。

## 7. Comparison 字段与缺省原则

comparison key 当前使用有方向的 `theory-runtime`：

```yaml
comparisons:
  theory-runtime:
    strategy: one_to_one
```

若 stage 未配置 comparison，Runner 默认使用 `one_to_one`。因此普通逐调用、逐声明 slot
比较应完全省略 `comparisons`；不要显式写空 mapping，也不要为了形式统一改成
`boundary_equal`。

所有内置逐 Tensor shape 比较先检查 tuple 完全相等；不相等且 rank 恰好相差 1 时，
允许把较长 shape 的前两维相乘后再比较。例如 `[T, ...]` 与 `[B, Q, ...]` 在
`T == B * Q` 且剩余维度逐维相等时视为等价。其他 rank/shape 差异仍为 `FAIL`，
dtype 始终严格比较。报告会用 `comparison.leading_product_equivalent` 显式标记该
PASS，而不是声称原始 shape 完全相等。

| strategy | 何时使用 | options |
| --- | --- | --- |
| `one_to_one` | 默认；Theory/Runtime 调用按位置一一对应 | 缺省 positional；必要时可 explicit mapping |
| `concat_shape` | 多个 Theory 输出沿一个 axis 对应单个融合 Runtime 输出 | 默认 composite、axis=-1；Q/K/V fusion 使用 |
| `boundary_equal` | Runtime stage 含多个调用，只比较一个明确边界 slot | 必须提供 explicit mapping |

显式 mapping 的 call index 是 **stage 内局部位置**，不是 Artifact 全局 `call_index`。
slot 对象必须完整写出 `direction/index/name`。只有重排、跨调用或融合关系才使用
`explicit/composite`；普通“只比较 OUTPUT[0]”应直接在 Theory 中只声明 `OUTPUT[0]`，由
默认 `one_to_one` 完成，不应再复制冗长 mapping。

## 8. Theory fragment 与复用

fragment 位于 `specs/theory_fragments/*.yaml`，顶层字段为：

```yaml
fragment_id: new_model_decoder_v1
fragment_kind: model_decoder
module_groups: {}   # 可选
stage_groups: {}    # 可选
stages: []          # 与 module_groups 至少一个非空
```

fragment schema 是严格的：

- 顶层只允许 `fragment_id/fragment_kind/module_groups/stages/stage_groups`；
- `module_groups.<id>` 是非空 Theory module 列表，供
  `theory.include_module_groups: [{fragment, group}]` 引用；
- `stages[]` 必须包含 `id/modules`，可选 `runtime/comparisons`，同一 fragment 中 stage id
  必须唯一；
- `stage_groups.<id>` 是无重复的 stage id 列表，只能引用本 fragment 已声明 stage；
- fragment 至少声明一个非空 `module_groups` 或 `stages`；
- fragment id 在注册表中必须全局唯一，引用不存在或 kind 不匹配都会加载失败。

支持的 `fragment_kind`：

| kind | 职责 |
| --- | --- |
| `model_decoder` | 一类模型 decoder 的 Theory、Runtime 默认边界和 comparison |
| `mtp_framework` | 公共 MTP request/proposal 框架及 stage groups |
| `mtp_predictor_adapter` | 少数模型在 predictor 前后增加的适配 stage |

### 8.1 复用层级

按优先顺序选择：

1. `include_fragment`：导入 fragment 全部 stage，首选方式；
2. region `include_fragment: {fragment, stage_group}`：导入一个完整 stage group；
3. `include_stage`：只导入单个 stage；仅在宿主 region 需要重新组织少量 stage 时使用；
4. `theory.include_module_groups/include_stages`：只复用 Theory operators，不复用完整
   Runtime/comparison 契约；谨慎使用；
5. 全量重写：只有结构和现有 fragment 实质不同才采用。

三种引用形式不要混淆：

```yaml
# layer：导入整个 decoder fragment
include_fragment: qwen3_dense_decoder_v1

# region：导入 fragment 中一个 stage group
include_fragment:
  fragment: mtp_framework_v1
  stage_group: request

# stage：导入一个 stage，可在宿主 stage 覆盖 runtime/comparison
include_stage:
  fragment: some_fragment_v1
  stage: attention
```

fragment stage 可以携带 `runtime` 和 `comparisons` 默认值。模型 Spec 对已导入 stage 的
差异使用按 stage id 的 override：

```yaml
layer_specs:
  dense:
    include_fragment: shared_decoder_v1
    runtime_options:
      attention:
        boundary_operators: [model_specific_attention]
    comparisons:
      attention_qkv:
        theory-runtime:
          strategy: concat_shape
```

override 只能引用 fragment 已有 stage id；未知 id 直接加载失败。默认字段正确时不要重复
写 override。

### 8.2 示例：decoder 相同，仅 Runtime kernel 名不同

假设新模型的 decoder 仍是 Qwen3 Dense 的三个语义 stage，Theory shape/dtype、stage 顺序和
comparison 都相同；唯一差异是 attention Runtime kernel 从
`tensor_cast.attention.default` 变成 `tensor_cast.new_attention.default`。不要复制一份新的
decoder fragment，应在新模型 Spec 中整包复用并只覆盖 `attention`：

```yaml
schema_version: "1"
spec_id: new_model_dense_v1
spec_version: 1.0.0
model_category: new_model_dense
matches:
  entrypoints: [text_generate]
  model_types: [new_model]

# one_to_one 比较时，把新 Runtime 名称对齐到 fragment 中的 Theory 语义名。
operator_aliases:
  tensor_cast.new_attention.default: attention

regions:
  language:
    layer_layout_rule:
      strategy: repeat
      layer_kind: dense
      count_from: model_config.effective_num_hidden_layers
    layer_specs:
      dense:
        # 继承 attention_qkv/attention/dense_ffn 的 Theory、默认 Runtime 和 comparison。
        include_fragment: qwen3_dense_decoder_v1

        # 只替换 attention stage 的完整 RuntimeStageOptions。
        runtime_options:
          attention:
            boundary_operators: [new_attention]
            ignored_operators:
              - view
              - index
              - reshape_and_cache
              - split_with_sizes
              - alias
              - copy_
              - slice
              - select
              - apply_rope
              - dynamic_quantize_symmetric
              - quantize
```

上述配置的实际效果是：

| 内容 | 来源 |
| --- | --- |
| `attention_qkv` 与 `dense_ffn` 全部配置 | 原 `qwen3_dense_decoder_v1` fragment |
| `attention` Theory modules/shape/dtype | 原 fragment |
| `attention` comparison | 原 fragment；未显式配置时仍走默认 `one_to_one` |
| `attention` Runtime boundary/ignored operators | 新模型 Spec 的 `runtime_options.attention` |
| `new_attention -> attention` 名称对齐 | 顶层 `operator_aliases` |

需要特别注意：`runtime_options.<stage_id>` 是**整段替换**，不是字段级深度合并。只写新的
`boundary_operators` 会令继承的 `ignored_operators` 变为空；如果新 kernel 仍需要原来的
ignored 集合，就像示例一样完整保留。相反，如果 Runtime 边界没有变化、只是比较时的算子
语义名称不同，则只增加 `operator_aliases`，不要覆盖 `runtime_options`。

如果新模型同时复用这个 decoder 作为 MTP predictor，`compose.predictor` 仍引用同一个
`qwen3_dense_decoder_v1`。当前 override 位于 language layer，只影响 language；MTP 的
Runtime 也存在同样差异时，应在 `mtp` region 对同一 stage id 增加对应 override，不能假设
language override 会跨 region 传播。

### 8.3 MTP 组合

普通共享 MTP 使用：

```yaml
mtp:
  activation: mtp_enabled
  layer_layout_rule:
    strategy: repeat
    layer_kind: dense_mtp
    count_from: model_config.effective_num_mtp_layers
  compose:
    framework: mtp_framework_v1
    predictor: new_model_decoder_v1
```

组合顺序固定为：

```text
framework.request                         -> region-level one-shot stages
framework.proposal_prefix
[adapter.before_predictor]
model decoder predictor
[adapter.after_predictor]
framework.proposal_suffix                 -> repeated MTP layer stages
```

普通模型只替换 `predictor`，不复制公共 MTP YAML。只有真实模型代码在 predictor 前后存在
额外语义适配时才新增 `mtp_predictor_adapter`，并在 compose 中增加
`predictor_adapter`。framework、adapter、predictor 的 stage id 不得重复。

## 9. 新增一类模型：推荐流程

### 步骤 1：先捕获 Runtime

编写一个本地 Profile，先不比较：

```bash
python -m tools.model_diagnostics work/new_model_decode.yaml \
  --runtime-report work/new_model_runtime.html --show-all
```

记录稳定语义阶段、边界 operator、机械 operator、量化前后差异、prefill/decode/MTP 差异。
不要从 Qwen3 YAML 猜测新模型 Runtime。

### 步骤 2：确定匹配边界

从捕获后的 HF/model config 确定稳定 `model_type` 和必要 features。新增
`specs/<new_spec_id>.yaml`，确保与现有 Spec 恰好一个匹配。

### 步骤 3：识别可复用部分

- decoder 与已有模型相同：直接 `include_fragment` 现有 decoder；
- decoder stage 相同、仅 Runtime kernel 名不同：复用 fragment 并按 stage override；
- decoder 结构不同：新增一个 `model_decoder` fragment；
- Dense/MoE 混合布局：用 `prefix_then_repeat` 展开物理层，并为两种 layer kind 分别
  `include_fragment`；不要为选取 fragment 引入 stage group；
- MTP 使用公共 wrapper：复用 `mtp_framework_v1`；
- predictor 前后有额外语义 stage：新增专属 `mtp_predictor_adapter`；
- 不要为每个模型复制公共 target/sampler/output/proposal 框架。

### 步骤 4：定义 Theory

按语义算子声明稳定 INPUT/OUTPUT slot，并使用本节定义的符号表达 shape/dtype。先保证普通
prefill/decode，再增加量化和 MTP。Theory 必须完整定义正式 E2E 所需 shape/dtype；不要用
`unknown` 掩盖缺口。

### 步骤 5：配置 Runtime 组织

基于 HTML 捕获选择 boundary/ignored operators。优先使用 fragment 默认值；差异最小化为
stage override。未知调用不得静默丢弃。

### 步骤 6：选择 comparison

先尝试缺省 `one_to_one`。只有 QKV 等融合使用 `concat_shape`，或多 Runtime call 中必须选择
稳定单边界时才使用 `boundary_equal`/显式 mapping。

### 步骤 7：补测试

- Loader/表达式/组织差异的单元测试放到对应 regression 镜像目录；
- 直接构造 Profile 的完整模型矩阵放到 `e2e/`；
- 为新类别保留极少量快速 smoke，不把完整矩阵加入日常增量门禁；
- 所有正式 E2E 必须 PASS。

### 步骤 8：验证

分类 `3` 的正式 E2E 复用同一份 `deepseek_v3_v1` 组合契约，并逐型号验证：

| 兼容入口 | 必测场景 |
| --- | --- |
| `deepseek-ai/DeepSeek-V3`、`deepseek-ai/DeepSeek-V3.2` | prefill、decode、W8A8_DYNAMIC、MTP；V3.2 额外覆盖 W4A8_DYNAMIC |
| `zai-org/GLM-5`、`zai-org/GLM-5.1` | prefill、decode、W8A8_DYNAMIC、MTP |
| `moonshotai/Kimi-K2-Base`、`moonshotai/Kimi-K2.5`、`moonshotai/Kimi-K2.6` | 文本路径的 prefill、decode、W8A8_DYNAMIC、MTP |

Kimi K2.5/K2.6 的顶层 VL config 均暴露 `text_config.model_type: kimi_k2`，因此文本诊断
共享分类 `3` 契约；带视觉输入时还需叠加分类 `6`，不由本节文本 E2E 代替。TensorCast 的
公开运行方式是一进程一模型；参数化测试在同一 pytest worker 中覆盖多个 Kimi 型号时，只在
测试侧重置远端类 patch guard，以模拟彼此独立的 CLI 运行，不扩展产品生命周期契约。

并行 shape 逐型号覆盖 `TP=2/EP=2` 与 `DP=2/MDP=2` 两个组合布局（DeepSeek V3/V3.1/V3.2、
GLM-5/5.1、Kimi K2/K2.5/K2.6）。所有场景均执行真实 Runtime capture，最终
Theory↔Runtime findings 必须全部 PASS；量化场景还必须断言对应量化 kernel 实际出现。

例行门禁保留**最小代表集**：每个 `model_type`（`deepseek_v32`/`deepseek_v3`/
`glm_moe_dsa`/`kimi_k2`）至少覆盖 prefill 与 decode，DeepSeek V3.2 作为旗舰型号
完整覆盖量化/MTP/并行；其余量化变体、MTP 与并行组合标记 `@pytest.mark.nightly`，
保留在仓库 nightly 层完整执行，不删除任何场景。

```bash
# 快速基础 guard
python -m pytest tests/smoke/test_model_diagnostics.py -q

# 完整诊断回归
python -m pytest tests/regression/model_diagnostics -q

# 检查新 YAML 严格加载、匹配唯一性和完整 E2E
python -m tools.model_diagnostics work/new_model_decode.yaml --theory-compare
```

## 10. 常见错误

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| `no Spec matched` | `matches.model_types/features` 与捕获后 Context 不一致 | 查看 Runtime Context/HF config，修正精确匹配 |
| `multiple Specs matched` | 新旧 Spec 匹配范围重叠 | 收紧 `required_features` 或模型类型边界 |
| `INCOMPLETE` | Theory 缺 shape/dtype、Runtime 边界未找到、stage/call/slot 缺失 | 不改成预期失败；补证据或修组织规则 |
| `operator.count_mismatch` | 默认 one_to_one 两侧调用数不同 | 先确认是否错误切段；确为融合/多边界再改策略 |
| shape 中 `unknown expression variable` | 使用了未注册变量或把 Profile 字段当 Theory env | 只使用本文环境符号，必要时先扩展 context_env 及测试 |
| `count_from path not found` | HF/model config 没有该派生字段 | 使用稳定捕获字段，或在物化阶段显式派生 |
| MTP Profile 被拒绝 | `query_length < num_mtp_tokens + 1` | 调整合法固定长度窗口 |
| eager 比较大量 INCOMPLETE | eager 展开为 ATen 流，Spec 面向编译态语义算子 | smoke 只验证基础 reachability；正式 E2E 使用 `do_compile=True` |

新增模型前建议先完整阅读当前
[Qwen3 Dense Spec](specs/qwen3_dense_v1.yaml)、
[decoder fragment](specs/theory_fragments/qwen3_dense_decoder_v1.yaml)、
[Qwen3 MoE Spec](specs/qwen3_moe_v1.yaml)、
[MoE decoder fragment](specs/theory_fragments/qwen3_moe_decoder_v1.yaml) 和
[DeepSeek V3 Spec](specs/deepseek_v3_v1.yaml)、
[DeepSeek Dense decoder](specs/theory_fragments/deepseek_v3_dense_decoder_v1.yaml)、
[DeepSeek MoE decoder](specs/theory_fragments/deepseek_v3_moe_decoder_v1.yaml) 以及
[MTP framework](specs/theory_fragments/mtp_framework_v1.yaml)，重点理解“模型 Spec 负责组合，
fragment 负责可复用权威定义，override 只表达真实差异”的边界。

分类 `2`（MoE）的 Theory 符号和公式见 6.2 节。Profile 可配：
`parallel.moe_data_parallel_size`（或 `moe_dp_size`）、
`enable_redundant_experts`、`enable_external_shared_experts`
（后两者需 `expert_parallel_size>1`；external 还要求模型有 shared experts）。

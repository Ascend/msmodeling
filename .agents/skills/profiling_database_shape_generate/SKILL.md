---
name: profiling-database-shape-generate
description: >-
  Generate spec-compliant profiling-database shapes for text_generate simulation cases
  (parse, grid generation, attention runtime metadata alignment) and verify case-level DB
  hit/interpolation without NPU. Invoke when generating shapes for the perf database,
  adapting model operators, building optimizer-args cases from text_generate commands/params,
  or checking that text_generate cases hit the DB without roofline fallback.
metadata:
  version: 1.0.0
  source: local-session-analysis
---

# 算子性能数据库 Shape 生成与端到端命中验证

## 目标与边界

为核心目标服务：生成的 shape 数据必须

1. 完全符合性能数据库 schema（`Input Shapes/Data Types/Formats` + runtime metadata 列），
   能通过 `start_microbench.py` 回放签名校验（契约错误行会被 `[DROP]` 删除）；
2. 默认只生成输入 case 直接捕获到的算子 shape 与 runtime metadata（cases yaml workload
   的精确 demand，量级为几十~几百行，而非数千行模型结构预置 shape）；需要 case 之外的
   插值锚点时，显式用 `--with-model-sweep` 启用模型结构 sweep（sweep 模型未显式传
   `--target-model` 时自动取 cases yaml 第一个 workload 的 `model` 字段；目标模型的
   结构 config 必须已存在于 `grid_generator/model_configs.py`，接入新模型见文末
   checklist）；
3. 复杂 attention 算子（`SparseFlashAttention`/FIA/`LightningIndexer`/`mla*` 等）的
   runtime metadata 与查询端 `_query_by_attn_params` 严格对齐；
4. 默认验收标准：仿真后按 analytic total 降序对**全体算子**取Top-N（默认 10），
   Top-N 中的**计算算子**全部被性能数据库命中（empirical total != analytic total，见"判定标准"）。
   **通信算子（all_reduce/all_gather/all_to_all/reduce_scatter 等集合通信 wrapper）
   不参与命中判定**——其时长由 base HCCL alpha-beta 模型给出，wrapper 插值主动跳过
   查库（`wrapper_communication_owned_by_base`），仅在报告中列出排除说明。已支持回放的
   算子（`tools/perf_data_collection/op_replay/` 下存在 `<KernelType>_run.py`）的
   全量命中作为 `--miss-scope replay` 补充口径保留。存在计算算子 MISS 时，必须分析
   原因并给出修复建议（运行收尾报告脚本自动产出，见"步骤 4：运行报告"）。

本 Skill 只做 shape 生成与本地命中验证；真实算子回放（NPU）、数据库发布、密度门禁
由 `profiling-database-lifecycle` / `profiling-database-axis-density` 负责。

## 双模式总览

**默认执行顺序：模式B → 模式A 完整闭环 → 收尾报告**——模式B 生成的性能数据库就是用户指定的
`<DB>`；模式A 的 mock 验证库固定为 `<DB>_mockverify`（用户指定路径 + `_mockverify`
后缀），且**每次模式B 运行后必须重新 mock**（generate 追加的新行不会出现在旧 mock 库，
用旧库验证会得到错误结论）。用户明确只要单一模式时才允许跳过另一个。

模式B 支持**追加模式**：目标库已有数据行时自动跳过 kernel_details 解析，新 case 的
shape 行直接追加到既有 CSV（传 `--force-parse` 可强制重新解析实测文件）；
mock 脚本按列级 union 语义填充：任一时长列缺失或 <=0 的行，其缺失的时长列填占位值，
已有实测值的时长列保持原值。

| 模式 | 用途 | 产物 | 入口 |
| --- | --- | --- | --- |
| cases 构建 | 把 text_generate 命令/参数组合转成 optimizer workload spec | cases 目录下 optimizer-args yaml | `scripts/build_cases.py` |
| 模式B 正式构建 | 生成干净（无 mock 时长）的数据库，供 `start_microbench.py` 正式回放；已有库时自动跳过解析、追加写入 | 目标库目录内 latency 为空/0 的新增行 | `scripts/generate_shape_db.py` |
| 模式A 本地验证 | 无 NPU 条件下验证输入 case 命中率/插值，确认无 roofline 回退 | mock 验证库 + 命中判定报告（回显日志解析，`<DB>_mockverify-verify-logs/` + `verify-summary.json`） | `scripts/mock_replay_durations.py` + `scripts/verify_db_coverage.py`（默认 top-total 口径） |
| 收尾报告 | 汇总一次完整运行：产物说明、仿真命令、日志位置、Top-N 计算算子命中、MISS 原因与修复建议 | `reports/<DB>-run-report.md` | `scripts/generate_run_report.py` |

目标数据库目录（下文 `<DB>`，所有脚本均可用 `--database-path` 覆盖）**必须由用户显式指定**，
禁止在用户指定的库不存在时静默回退到其他库；指定库不存在时按"输入 2"的规则新建。

```text
tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/<用户指定的库名>/
```

## 输入

本 Skill 接收两类输入：

### 输入 1：text_generate 仿真 case（二选一或混合）

case 是用户指定的仿真情况，格式可以是完整仿真命令，也可以是不全的参数集合：

- **形式 a：完整 text_generate 仿真命令**。用 `build_cases.py --command` 解析（基于仓库
  真实 CLI registry，校验与 CLI 完全一致），自动换算 `input_length`、`batch_range`、
  `tp_sizes` 等 spec 字段；
- **形式 b：部分参数（如仅输入输出长度、模型 id）**。用 `build_cases.py --set` 提供已指定
  参数，未指定的可选参数与指定参数做笛卡尔组合（每个未指定维度用默认小网格而非全集，
  防止组合爆炸），得到一个可能进行仿真的情况全集。组合先经 spec 级 preflight 淘汰非法
  组合，再经 `load_optimizer_spec` 全文件终校验；超过 `--max-workloads`（默认 64）时做
  确定性均匀采样。

两种形式最终都转换为 throughput-optimizer 场景的 workload spec yaml（字段语义与
`throughput_optimizer` CLI 一致），即 `generate_shape_grid.py --optimizer-args-file`
的消费格式。

### 输入 2：实测算子性能数据库位置

通过 `--database-path` 传入目标性能数据库目录，要求：

- 目录内已有 `op_mapping.yaml`（TensorCast op ↔ NPU kernel 映射，MISS 归类与回放支持
  判定均依赖它）；
- 模式B 首次运行还需要 NPU profiling 导出的 `kernel_details*.csv`（固定放在本 Skill
  目录下 `./kernel_details.csv`），用于写入实测行；仅重跑 shape 生成时加 `--skip-parse`。

**库不存在时的新建规则**（禁止静默改用其他库）：

1. 目标目录不存在时新建该目录，并要求用户在新建目录中存放 `op_mapping.yaml` 文件；
2. 实测行通过模式B 的 parse 步骤（`kernel_details.csv`）写入，不复制源库的任何
   CSV 数据行——新库是"实测解析 + 本次 case 需求"的干净组合；
3. 新建决策与映射来源必须在开始执行前向用户复述确认。

### 输出

1. **cases yaml**：`.agents/skills/profiling_database_shape_generate/cases/` 目录下的
   optimizer-args yaml 文件（命名建议 `<model>_cases.yaml`），供
   `generate_shape_grid.py --optimizer-args-file` 消费；单个 workload 场景同时覆盖
   prefill 与 decode 两阶段的查询需求；
2. **shape 生成后的性能数据库**：`<DB>` 内追加实测行与查询驱动的 shape 行（新行
   latency 为空/0），可直接交给 `tools/perf_data_collection/start_microbench.py` 或
   op_replay 脚本做正式算子回放——保证 op_replay 目录下已支持回放的算子在对应 case
   仿真时被精确命中；
3. （模式A 附带）mock 验证库 `<DB>_mockverify/`、逐 case 的命中判定报告
   （`<DB>_mockverify-verify-logs/<label>_run.log` 回显日志，含 `MISSes`/`MISS shapes`
   解析结果）与结构化汇总 `<DB>_mockverify-verify-logs/verify-summary.json`
   （仿真命令、Top-N 计算算子命中、MISS 原因，供收尾报告消费）；
4. （收尾报告）`<SKILL>/reports/<DB>-run-report.md`：一次完整运行的总报告，内容见
   "步骤 4：运行报告"。

## 事实源

按序读取，不得凭记忆改写 schema 或匹配语义：

1. `tools/perf_data_collection/parsers/parse_kernel_details.py`：解析逻辑与输出 schema；
2. `tools/perf_data_collection/generate_shape_grid.py` 与 `tools/perf_data_collection/grid_generator/`：
   查询驱动生成规则、preflight 淘汰、`optimizer_spec.py` 的 workload 字段；
3. `tools/perf_data_collection/start_microbench.py` 与 `signature_utils.py`：回放签名与匹配字段；
4. `tools/perf_data_collection/op_replay/`：回放支持算子清单（存在 `<KernelType>_run.py`
   即视为已支持回放）；
5. `tensor_cast/performance_model/profiling_database/profiling_data_source.py`：
   精确匹配（`_query_by_attn_params` 的 attention runtime 字段清单）；
6. `tensor_cast/performance_model/profiling_database/interpolating_data_source.py` 与
   `interpolation_index.py`：插值轴与不外推约束；
7. `tensor_cast/performance_model/empirical.py`（`process_op`）与 `metrics_collector.py`：
   PARTIAL/MISS→roofline 回退判定与 `MISSes`/`MISS shapes` 回显日志格式；
8. `tools/perf_data_collection/grid_generator/model_configs.py`：目标模型结构配置
   （内置多个模型的 config，如 GLM-5.2/Qwen3-32B/DeepSeek-V3；接入新模型需先在此
   补充对应 config，模型 ID 必须与 cases yaml 的 `model` 字段一致）。

## 工作流程

### 前置检查（步骤 0：输入齐全性检查）

运行任何脚本前，先核对以下输入清单；**任何一项缺失或存在歧义时，必须向用户交互确认
（AskUserQuestion）后再继续，禁止静默假设或回退默认值**：

1. **目标数据库路径**：用户是否显式给出了 `<DB>`？给出的目录是否存在？
   不存在时按"输入 2"的新建规则复述新建方案（目录 + op_mapping.yaml 种子来源）请用户确认；
2. **cases 输入**：cases yaml 文件是否存在且包含本次要仿真的 workload？
   用户给的是完整 text_generate 命令时，先走 `build_cases.py` 生成；
3. **output_length 确认（仅模式1，关键）**：text_generate CLI 没有 `--output-length` 参数，
   用 `build_cases.py --command`（模式1）解析完整命令时，未显式传 `--output-length`
   会静默取默认值 2048。`output_length` 直接决定 decode demand（decode 命令按
   `input_length = context_length - output_length` 反推）与生成的 shape 集合，
   不确认会导致生成的 yaml 与预期不一致。因此运行模式1 前必须与用户确认本次 case
   真实的 `output_length` 并显式传 `--output-length`，禁止静默使用默认值；
   `input_length` 由命令的 `--query-length`/`--context-length` 推导，无需单独确认。
   长度语义参照：prefill 场景 `--query-length = input_length`、`--context-length 0`
   （或带 `--prefill`）；decode 场景 `--query-length 1`、
   `--context-length = input_length + output_length`（带 `--decode`）。
   模式2（`--set` 网格组合）无需要严谨对齐的单一 case，output_length 走默认网格即可，
   不要求此确认；
4. **实测数据**：`kernel_details.csv` 是否存在？用户未说明时询问走 parse 还是 `--skip-parse`
   （库中已有实测行且 profiling 文件未变化时才可 skip）；
5. **环境**：`python -c "import tensor_cast, tools.perf_data_collection"` 可用
   （本机无 torch 时在装有环境的容器内执行）。

```bash
uv run python -c "import tensor_cast, tools.perf_data_collection"  # 环境可用
ls <DB>/op_mapping.yaml                                            # 输入 2 的数据库位置合法
```

### 步骤 1：构建 cases（build_cases.py）

把输入 case 转成 cases 目录下的 optimizer-args yaml：

```bash
# 形式 a：解析完整 text_generate 命令（prefill/decode 均可，自动换算 input_length/batch_range）
uv run python .agents/skills/profiling_database_shape_generate/scripts/build_cases.py \
  --output .agents/skills/profiling_database_shape_generate/cases/my_cases.yaml \
  --command "python -m cli.inference.text_generate <model_id> --num-queries 1 \
    --query-length 3584 --context-length 0 --device <device> \
    --tp-size 16 --num-devices 16 --quantize-linear-action W4A8_STATIC"

# 形式 b：部分参数 × 未指定维度默认网格组合
# 默认网格：input_length {1024,8192,65536,131072}、output_length {1024,2048}、
#           batch_range [1]、quantize W8A8_DYNAMIC（均可用 --*-grid 覆盖）
uv run python .agents/skills/profiling_database_shape_generate/scripts/build_cases.py \
  --output .agents/skills/profiling_database_shape_generate/cases/my_cases.yaml \
  --set model=<model_id> --set device=<device> --set num_devices=16 \
  --set tp_sizes=[16] --input-length-grid "1024,65536"
```

形式 b 必填：`model`、`device`、`num_devices`；`--set` 值按 YAML 解析（列表/整数直接写）。
字段映射表（text_generate CLI ↔ optimizer spec）见 `build_cases.py` 头部注释。并行维度
（tp_sizes/ep_sizes 等）是候选列表，非法组合（如 TP=3 不整除）由 `load_optimizer_spec`
展开时报 `illegal-skipped` 计数。

### 步骤 2：模式B 正式构建（干净数据库）

```bash
# 默认仅生成 cases yaml 中 workload 捕获的精确需求 shape（不跑模型结构 sweep）
# 内存有限的机器建议 --max-workers 1-2（链路工具默认 8 路并发，易触发
# Windows WinError 1450 资源耗尽）；失败 workload 自动断点重试至 --max-attempts 次
uv run python .agents/skills/profiling_database_shape_generate/scripts/generate_shape_db.py \
  --database-path <DB> --cases-yaml .agents/skills/profiling_database_shape_generate/cases/my_cases.yaml \
  --max-workers 2

# 需要输入 case 之外的插值锚点时，显式加 --with-model-sweep 启用模型结构 sweep
# （--rows 只约束 sweep/coverage 候选行，case 精确 demand 不受其截断）
uv run python .agents/skills/profiling_database_shape_generate/scripts/generate_shape_db.py \
  --database-path <DB> --cases-yaml .agents/skills/profiling_database_shape_generate/cases/my_cases.yaml \
  --target-model <model_id> --with-model-sweep --rows 1000 --max-workers 2
```

该脚本依次执行：

1. `parse_kernel_details.py --profiling-path ./kernel_details.csv --database-path <DB>`
   （实测行写入，含真实 `Profiling Average Duration(us)` 与 FIA runtime metadata 列）。
   **目标库已有数据行时自动跳过本步**（追加模式：新 case 的 shape 行直接追加到既有
   CSV，不重复解析实测文件）；实测文件变化需要重新解析时显式传 `--force-parse`；
2. `generate_shape_grid.py --database-path <DB>
   --optimizer-args-file <cases yaml> --rows 1000 --seed 0`
   （默认只跑 cases yaml 中 workload 的精确 demand；传 `--with-model-sweep` 时追加
   `--target-models` 启用模型结构 sweep。preflight 淘汰必败行后追加写入，
   新行 latency 为空/0，属 missing-only 回放的预期输入）。

可通过 `--rows` 调整每算子新增行预算（受 V1 密度门禁约束，参考
`profiling-database-axis-density`）。

### 步骤 3：模式A 本地命中验证（默认在模式B 之后必跑，无需 NPU、不执行回放）

验证逻辑与输入 case 的对应关系：对生成的算子性能数据库中的每个 shape 行构造对应时长
（mock 占位时长，模拟回放已完成），再运行输入 case 对应的 `text_generate` 仿真脚本，
判定 op_replay 已支持回放的算子是否 100% 被命中。

**硬约束：每次模式B（generate）运行之后必须重新 mock。** mock 库是 generate 前的静态副本，
generate 追加的新行不会出现在旧 mock 库中，用旧库验证会得到错误结论。
mock 库位置固定为 `<DB>_mockverify`（用户指定路径 + `_mockverify` 后缀）；
mock 脚本按列级 union 语义填充：任一时长列缺失或 <=0 的行，其缺失的时长列填占位值，
已有实测值的时长列保持原值。
`verify_db_coverage.py` 按 `<DB>_mockverify` 后缀推导源库并做行数新鲜度校验，
发现 stale 时报错退出（`--allow-stale` 可降级为警告）。

```bash
# 1. 复制 <DB> 为 mock 库（<DB>_mockverify），对缺失/<=0 的时长列填固定占位时长（默认 100us）
uv run python .agents/skills/profiling_database_shape_generate/scripts/mock_replay_durations.py \
  --database-path <DB> --output-dir <DB>_mockverify

# 2. 运行输入 case 对应的 prefill+decode text_generate 仿真并判定命中
# case id 是 --cases-yaml 中 workload 的 1-based 顺序号（任意个数，--cases 缺省跑全部）；
# model/device/tp/ep/num-devices/量化与模型专属 flag（word_embedding_tp、
# enable_shared_expert_tp、compile）默认从对应 workload 条目继承，CLI 参数仅作显式覆盖
uv run python .agents/skills/profiling_database_shape_generate/scripts/verify_db_coverage.py \
  --database-path <DB>_mockverify --cases-yaml .agents/skills/profiling_database_shape_generate/cases/my_cases.yaml \
  --cases 1,2,3
```

`verify_db_coverage.py` 对每个 case-phase 构造 `python -m cli.inference.text_generate`
命令并追加 `--log-level info`，捕获回显日志后做两级解析：默认 `top-total` 口径解析
runtime 汇总表（`empirical total`/`analytic total` 列）判定 Top-N **计算算子**命中
情况（通信算子排除，见"判定标准"）；`--miss-scope replay`/`all` 则解析 metrics_collector
输出的 ``MISSes``（完全 MISS 的算子与原因分类）与 ``MISS shapes``（未命中的算子 shape）
做 MISS 清单判定。运行结束把结构化汇总写入
`<DB>_mockverify-verify-logs/verify-summary.json`（仿真命令、Top-N 命中明细、MISS 原因），
供 `generate_run_report.py` 生成收尾报告。
`--dry-run` 可只打印命令，`--skip-run` 只分析已有回显日志
（`<DB>_mockverify-verify-logs/<label>_run.log`）。**模型/设备/并行/量化参数与模型专属
flag 默认从 `--cases-yaml` 对应 workload 条目继承**（case id = workload 的 1-based
顺序号，个数不限），`--cases-yaml` 为必填（无模型默认回退）；
`--model`/`--device`/`--tp-size`/`--ep-size`/`--num-devices`/
`--quantize-linear-action`/`--word-embedding-tp`/`--enable-shared-expert-tp`
按需显式覆盖。decode 的 context 按实测算子约定取 `input + output_length`
（`--query-length 1`，带 `--decode`），与 build_cases 的反推公式对齐。

### 步骤 4：运行报告（收尾）

```bash
uv run python .agents/skills/profiling_database_shape_generate/scripts/generate_run_report.py \
  --database-path <DB>
```

> 若步骤 3 的 verify 使用了自定义 `--log-dir`，`verify-summary.json` 会写在该目录而非
> 默认的 `<DB>_mockverify-verify-logs/`，此处必须显式传
> `--summary-json <log-dir>/verify-summary.json`，否则报告会以 "verify summary not found" 退出。

默认从 `<DB>_mockverify-verify-logs/verify-summary.json` 与
`<DB parent>/<DB name>-shape-generation-report.json` 汇总，报告写入
`<SKILL>/reports/<DB>-run-report.md`，内容包括：

1. **产物文件说明**：性能数据库位置与后续用途（正式回放/直接消费）、mock 库的作用与
   禁忌、逐 case 回显日志、生成报告 JSON、cases yaml；
2. **仿真命令清单**：每个 case-phase 实际执行的 `text_generate` 命令；
3. **仿真判定结果**（每 case-phase）：日志位置、通信算子排除说明、全体算子 Top-N 中
   MISS 的计算算子总览表（empirical/analytic/calls）；
4. **MISS 计算算子总览**：以算子为粒度聚合（避免逐 case 重复分析同一算子），
   按 Top-10 内最大 analytic total 降序罗列；每个算子列出 MISS 原因、根因方向、
   MISS 次数与 case 场景（case 数 + 明细，超过 3 个取前 3 加"等"）。修复建议是
   按 MISS 原因的静态映射，不在报告中罗列——处置方式对照"常见 MISS 处置"表执行；
5. **生成阶段概览与结论**：demand/落库统计、unsupported kernel 清单、整体 PASS/FAIL。

存在计算算子 MISS 时，以报告第 4 节为准向用户陈述根因与修复建议，不静默标记通过。

### 仿真命令构造规则（模型无关）

`verify_db_coverage.py` 自动为每个 workload 构造 prefill/decode 成对的
`python -m cli.inference.text_generate` 命令，规则统一、与具体模型解耦：

- **参数继承**：model/device/num_devices/tp_sizes/ep_sizes/quantize_linear_action/
  word_embedding_tp/enable_shared_expert_tp/compile 全部取自 cases yaml 的对应
  workload 条目；CLI 同名参数仅作显式覆盖。dense 模型不写
  `enable_shared_expert_tp`/`word_embedding_tp`/`ep_sizes` 字段即可，命令不会携带
  这些 flag（也不会附加默认值 `--ep-size 1` 的 MoE 专属噪音）；
- **长度语义**：prefill 用 `--query-length <input_length> --context-length 0`
  （或带 `--prefill`）；decode 用 `--query-length 1 --context-length
  <input_length + output_length>` 并追加 `--decode`（实测算子约定，与 build_cases
  的反推 `input_length = context_length - output_length` 对齐）；
- **必填字段**：workload 缺少 model/device/num_devices/tp_sizes 时报错退出，
  不会回退到任何模型的默认值；
- **dry-run 抽查**：换模型或换 case 数量后，先 `--dry-run` 检查自动构造的命令是否
  携带了目标模型不支持的 flag，再正式运行。

本地验证时 `<DB>` 用 mock 目录；verify 脚本自动追加 `--log-level info` 并把回显日志
保存到 `<DB>_mockverify-verify-logs/<label>_run.log`。

## 判定标准

### 默认验收口径（`--miss-scope top-total`，默认）

以 `--log-level info` 回显日志中 runtime 汇总表（`Name / empirical total /
empirical avg / analytic total / analytic avg / # of Calls`）为判定依据：

- **命中判据**：MISS 回退 roofline 时 empirical 与 analytic 完全相等，
  因此 `empirical total != analytic total` 即数据库命中；
- **参与判定的算子集合**：先按 analytic total 降序对**全体算子**（含通信算子）取
  Top-N（`--top`，默认 10）；Top-N 中命中判定只针对**计算算子**——通信算子（名称含
  all_reduce/all_gather/all_to_all/reduce_scatter 等片段，见 `verify_db_coverage.py`
  的 `COMM_OP_TOKENS`）的时长由 base HCCL alpha-beta 模型给出，wrapper 插值主动跳过
  查库（`wrapper_communication_owned_by_base`），仅在报告中列出排除说明、不参与
  HIT/MISS 判定；
- **验收条件**：全体算子 Top-N 中的计算算子全部命中才 PASS；报告中展示每个 case
  Top-N 里 MISS 的计算算子（empirical/analytic/calls）与被排除的通信算子；
- 该口径聚焦对仿真结果影响最大的头部算子，composite 算子（如
  `mla_sparse_attention`/`dsa_indexer`）的子 kernel 回退同样会被捕获
  （回退时其 empirical == analytic）；
- **存在计算算子 MISS 时**：必须按"常见 MISS 处置"分析根因并给出修复建议
  （收尾报告自动生成第 4 节），不得静默把 MISS 视为可接受。

### 补充口径（`--miss-scope replay` / `all`，非默认）

以 metrics_collector 的 `MISSes`/`MISS shapes` 段为判定依据：

1. `replay`（回放支持口径）：MISS 先按 `<DB>/op_mapping.yaml` 归到 kernel_type
   （`kernel_type` + `primary_kernel_type` + `alternate_kernel_types`），仅当
   kernel_type ∈ `tools/perf_data_collection/op_replay/*_run.py` 清单时才计为失败；
   未映射或无回放脚本的 kernel 计入 exempt 并在汇总中展示（处理方向见
   "常见 MISS 处置"）；
2. `all`（严格口径）：任何 MISS 都判失败。用于最终发布前的全面体检——exempt
   算子也需要通过 op_mapping 补映射（`op-mapping-generator`）或生成 `_run.py`
   （`microbench-run-script-generator`）后才能在此口径下通过。

## 常见 MISS 处置

| 现象 | 根因方向 | 处置 |
| --- | --- | --- |
| `shape_mismatch` | 缺该 shape 行 | 加大 `--rows` 或把场景加入 cases yaml 后重跑模式B |
| `outside_axis_boundary` | 插值不外推 | 在实测范围内补边界点（注意 ATTN_SEQ 平方复杂度轴） |
| `partial:<kernel>` | 复合算子叶子 kernel 缺行 | 按 `missed_kernels` 定位缺的 KernelType CSV 补行 |
| `compute_multidim_interpolation_failed` | sweep workload 部分失败导致 M/K/N 网格缺列 | 看 generate 输出的 `workloads: failed=N`；降低 `--max-workers` 重跑模式B（缓存续跑），再重新 mock 复验 |
| `compute_csv_not_found` | kernel 实测数据整库缺失 | 提供覆盖该 kernel 的 `kernel_details.csv` 重走解析；无 NPU 数据时无法补 |
| 生成报告 `unsupported_kernel_demands`（如 `MatMulCommon/MatMulV3/ConcatD`）且对应查询 MISS | demand 映射到无 CSV/replay 支持的 kernel，exact 需求被跳过，runtime 查询落到 alternate kernel 也无行 | 属 op_mapping/回放支持范围 gap：用 `op-mapping-generator` 核对该 kernel 的映射与 alternate_kernel_types，或为该 KernelType 生成 `_run.py`（`microbench-run-script-generator`）；SKILL 生成流程无法绕过 |
| attention 精确不命中 | runtime metadata 与运行时不一致 | 核对 `Runtime num_heads/sparse_mode/input_layout/block_size/topk/kv_cache_mode` 等列与 `text_generate` 运行时值；复杂 attention 必须带完整 runtime metadata，逐字段对齐 `profiling_data_source.py` 的 `_query_by_attn_params` |

## 接入新模型 checklist（如 Qwen3-32B、DeepSeek V3）

目标模型尚未接入本 Skill 时，按序确认；任何一步不满足都先停下来与用户确认，
禁止静默回退到任何默认模型：

1. **模型结构 config**：核对 `tools/perf_data_collection/grid_generator/model_configs.py`
   是否已有目标模型的 config（`Qwen/Qwen3-32B`、`deepseek-ai/DeepSeek-V3`、
   `zai-org/GLM-5.2` 已内置；模型 ID 必须与 cases yaml 的 `model` 字段一致）。
   没有时先补充 config（hidden/heads/kv_lora/experts 等来自模型开源配置），再继续；
2. **cases yaml**：用 `build_cases.py`（`--command` 或 `--set`）为目标模型生成新的
   cases yaml（命名建议 `<model>_cases.yaml`），workload 个数不限（2 个、5 个均可）；
   workload 条目必须包含该模型真实的并行与量化字段，不要从其他模型的 yaml 复制
   dense 模型不具备的字段（如 `enable_shared_expert_tp`/`word_embedding_tp`）；
3. **生成**：`generate_shape_db.py --database-path <DB> --cases-yaml <新yaml>`
   （`--database-path`/`--cases-yaml` 均为必填；`--with-model-sweep` 时
   `--target-model` 缺省自动取 yaml 第一个 workload 的 model）；
4. **mock + 验证**：重新 mock 后
   `verify_db_coverage.py --database-path <DB>_mockverify --cases-yaml <新yaml>
   --cases 1,...,N`（N 为 workload 个数）；先用 `--dry-run` 抽查自动构造的命令
   是否携带了目标模型不支持的 flag；
5. **op_mapping 覆盖**：不同模型的算子集合不同（如 MoE 模型的专家路由路径、
   MLA 模型的 LightningIndexer），确认 `<DB>/op_mapping.yaml` 已覆盖目标模型的 op；
   缺口用 `op-mapping-generator` 补映射。

## 完成标准

- 生成的 shape 与 metadata 字段与现有数据库 schema 完全一致（列名、分号分隔多槽、utf-8-sig）；
- cases yaml 已生成于 `.agents/skills/profiling_database_shape_generate/cases/`，
  覆盖输入 case 的精确需求（默认不跑模型结构 sweep；仅在显式 `--with-model-sweep`
  时才包含常用输入输出区间的插值锚点）；
- 模式A 对全部输入 case-phase 在默认 top-total 口径下验证 PASS：按 analytic total
  排序的 Top-N **计算算子**（通信算子排除）全部命中数据库，头部算子无任何 roofline
  回退（需要更严格体检时再跑 `--miss-scope replay`/`all`）；存在 MISS 时已在收尾
  报告中给出根因分析与修复建议；
- 模式B 产物（`<DB>`）可被 `start_microbench.py` 接受（无签名契约问题，preflight 已淘汰
  必败行），可直接进入正式 op_replay 回放；
- **收尾报告已生成**：`.agents/skills/profiling_database_shape_generate/reports/<DB>-run-report.md`，
  涵盖产物文件说明、仿真命令、日志位置、每 case Top-N 计算算子命中情况、MISS 原因
  与修复建议；
- `uv run python scripts/ai/validate_skills.py` 通过。

## 安全规则

- `kernel_details.csv`、`<DB>_mockverify/`、`<DB>_mockverify-verify-logs/`（含历史
  `*-verify-metrics/` metrics JSON 产物）与 `reports/` 下的运行报告为本地产物，
  禁止提交仓库、禁止上传或外发；
  mock 库严禁回写正式数据库或用于任何发布流程；
- 不发布 PR/Issue 评论，不触碰远端；仅在本地目录内操作；
- 不得为通过验证放宽判定口径（如把 PARTIAL 视为命中）或绕过 preflight/门禁；
- 路径引用一律用仓库相对路径；修改生成规则前先读事实源，禁止以算子名猜测字段。

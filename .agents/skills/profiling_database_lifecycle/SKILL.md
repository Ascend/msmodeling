---
name: profiling-database-lifecycle
description: >-
  Use when planning or executing an end-to-end profiling database update from
  axis-density rules and Shape generation through NPU collection, production
  query replay, anomaly audit, review, publication, and runtime feedback.
metadata:
  version: 0.1.0
  source: local-session-analysis
---

# 实测算子性能数据库生命周期

## 目标

按 [`profiling_database_lifecycle.md`](https://gitcode.com/Ascend/msmodeling.wiki/blob/main/design/profiling_database_lifecycle.md)
执行一次可追溯的数据库更新。本 Skill 只编排已有 Skill、脚本和测试，不复制轴密度、采集、查询或异常检测算法。

开始前确认仓库根目录、Git 分支、设备与软件版本、基线数据库、候选数据库、目标 workload 和报告目录。能从仓库读取的信息直接读取；
缺少会改变采集范围或验收结论的信息时，列为阻塞项，不猜测参数或阈值。

## 固定边界

- 基线数据库只读；所有 Shape 生成和 latency 写回仅指向独立候选目录。
- 报告目录必须位于数据库目录之外。
- YAML 是最低密度的数值依据，不是采集器运行时配置。
- 现有轴且规则不变时直接读取 YAML；只有新增轴或修改密度时才调用轴密度 Skill。
- 统计异常只能生成复测候选，未经独立硬件复测不得修改 latency。
- PR673 未合入、`find_database_anomalies.py` 不存在时，异常审计阶段标记为阻塞，不用替代算法绕过。
- 不把仿真 estimate 说成实机吞吐，不把无异常候选说成所有 query 均已通过。

## 输入与输出

| 名称 | 要求 |
| --- | --- |
| `<baseline-db>` | 已发布数据库目录，只读 |
| `<candidate-db>` | 从基线完整复制的候选目录，可写，包含 CSV 和 `op_mapping.yaml` |
| `<report-dir>` | 数据库外的独立目录，用于命令、hash、diff、回放和审计报告 |
| `<target-models>` | 采集计划明确支持的模型 ID；没有时不得假装是通用模型集合 |
| `<ops>` | 本轮新增或补测的 kernel 列表 |
| `<repeat-count>` | 项目或采集负责人批准的重复次数；未提供时先报告当前脚本默认值供确认 |

每轮至少保留：代码 commit、数据库路径与 hash、密度规则版本、生成参数、目标 workload、采集结果、候选库 diff、query replay、
异常报告、测试命令和未通过项。输入变化后，旧结论失效并从阶段 0 重新开始。

`start_microbench.py` 当前默认 `repeat-count=1`，只能作为 CLI 行为参考，不能直接视为满足重复测量质量。正式采集仍需由项目或
采集负责人确认次数；接口变化时以当前分支的 `--help` 和实现为准。

## 执行流程

### 0. 冻结基线

1. 阅读本 Skill、生命周期文档、仓库 `AGENTS.md` 和 `spec/README.md`。
2. 确认 Git 和工作树：

```bash
git status --short --branch
git rev-parse HEAD
```

3. 确认 `<baseline-db>`、其中的 `op_mapping.yaml`、目标设备/软件版本和 workload 清单。
4. 对基线数据库、mapping、密度 YAML 和 workload manifest 计算 hash，写入 `<report-dir>` 的执行记录。
5. 将基线完整复制为 `<candidate-db>`，复核复制前后文件数和 hash；后续命令不得写 `<baseline-db>`。

缺数据库版本、mapping、workload 或环境信息时停止。不要在看到 MISS 后改写已冻结的关键 query 清单。

### 1. 确定密度并生成采集计划

#### 1.1 选择规则入口

- 已有轴且密度不变：读取
  [轴密度 YAML](../profiling_database_axis_density/axis_collection_density.yaml)。
- 新轴、范围变化、间隔变化或必测值变化：调用
  [`profiling-database-axis-density`](../profiling_database_axis_density/SKILL.md)。该 Skill 负责确认轴语义、证据和 YAML 变更；
  规则评审通过并更新 YAML 后，才能继续生成正式计划。
- mapping 缺失或 kernel 到 CSV 的映射变化：调用 [`op-mapping-generator`](../op-mapping/SKILL.md)，不要按算子名猜映射。

#### 1.2 生成 Shape

本节命令对应 PR732 合入前的 `master` 接口。推荐合入顺序为 `PR732 -> PR673 -> PR738`；PR732 合入后、PR738 合入前，必须按最终 `--help` 更新本节命令和检查项，移除已废弃的 `--max-hbm-gb`，并按需传入新增的 `--ops`。更新完成前，本阶段标记为 `BLOCKED`，不得继续采集。

先查看当前分支实际参数：

```bash
python tools/perf_data_collection/generate_shape_grid.py --help
```

只对候选数据库执行：

```bash
python tools/perf_data_collection/generate_shape_grid.py \
  --database-path <candidate-db> \
  --target-models <target-models> \
  --rows 0 \
  --seed 0 \
  --max-hbm-gb <approved-limit>
```

在当前接口下，`--rows 0` 表示不随机截断网格；`--max-hbm-gb` 必须来自目标设备或已批准采集约束，不得照抄示例。生成后检查：

- YAML 的端点、必测值、最大间隔和最大相邻比是否落实到实际 Shape；
- 冻结的关键 query 和复合算子实际子 query 是否加入；
- Shape 是否满足代码中的对齐、dtype、format、layout 和 runtime metadata 要求；
- 严格签名去重后，候选库是否超过基线唯一签名数的 8 倍。

YAML 与生成点不一致时，修正生成器和一致性测试，不能只改 YAML 或手工补 CSV。
8 倍仅是防止采集规模失控的容量上限，不是目标规模，也不是密度甜点位证据。query-driven 生成降低点数时仍应保留该保护线；
只有新的容量或采集证据才能调整上限。

### 2. 采集并写入候选库

#### 2.1 准备 replay

对 `<ops>` 逐项检查 `tools/perf_data_collection/op_replay/<KernelType>_run.py`。缺失或接口不匹配时调用
[`microbench-run-script-generator`](../microbench/SKILL.md)，并完成其 `py_compile`、`--help` 和可用时的 NPU replay 验证。

#### 2.2 采集 compute kernel

先确认当前 CLI：

```bash
python tools/perf_data_collection/start_microbench.py --help
```

再在有目标 NPU 和正确软件栈的环境执行：

```bash
python tools/perf_data_collection/start_microbench.py \
  --database-path <candidate-db> \
  --ops <ops> \
  --repeat-count <repeat-count> \
  --update-mode missing-only \
  --fail-fast
```

`missing-only` 只填补无效 latency。需要覆盖已有正 latency 时必须单独说明原因、保存原始重复测量并人工审核冲突，不能改成 `all`
后直接覆盖。`DispatchFFNCombine` 按 `--help` 提供 EP、节点数、rank 和 master 参数；不得用单进程结果冒充目标并行配置。
当前采集入口若只写聚合 latency、不保留每次 repeat，应保存运行日志并将“缺少原始 repeat”列为证据缺口，不能宣称已具备重复测量追溯。

#### 2.3 可选入口

- 从整网 profiling 的 `kernel_details*.csv` 导入时调用：

```bash
python tools/perf_data_collection/parsers/parse_kernel_details.py \
  --profiling-path <profiling-output> \
  --database-path <candidate-db>
```

- 通信数据调用
  `tools/perf_data_collection/comm_bench/generate_comm_microbench.py`；执行前查看 `--help`，显式给出设备数、拓扑、消息网格和 dtype。

采集完成后，逐 Shape 对账“计划、执行结果、最终 CSV”。失败、超时和不支持必须保留状态；latency 只有正数且 finite 才计入覆盖。

### 3. 发布前质量检查与复测

#### 3.1 数据结构与回归测试

检查 Git LFS、CSV 字段、正 latency、严格签名重复、mapping 引用、候选库相对基线的新增/修改/删除。删除基线签名、未解释的
latency 冲突或 mapping 必需 CSV 缺失时失败。

至少运行与改动 kernel 相关的 profiling database 测试；公共 datasource 或插值行为变化时运行：

```bash
pytest -q tests/benchmark/ops/perf_database/test_profiling_data_source.py
pytest -q tests/benchmark/ops/perf_database/test_profiling_interpolation_non_regression.py
```

复合、FIA 或其他专用路径按改动追加对应测试，不能用上述两条替代。

#### 3.2 生产 query replay

调用 [`text-generate-executor`](../text-generate-executor/SKILL.md)，对冻结 workload 生成并确认实际命令，至少要求：

- `--performance-model profiling`
- `--profiling-database-path <candidate-db>`
- `--compile`
- `--dump-input-shapes`
- `--export-empirical-metrics-file <report-dir>/<workload>.json`

该 Skill 负责模型、设备、prefill/decode、并行和 workload 参数，不在本 Skill 中猜值。保存 stdout/stderr 和导出的 metrics；对关键
query 另外核对原始 CSV 严格签名。导出报告能区分 hit、partial 和 miss，但不能自动证明所有 hit 都是 strict exact；需要 exact、
interpolated、matched points 或复合子 CSV 明细时，应从真实 datasource result/trace 补证。当前工具不能提供该明细时，明确标记未验证。

#### 3.3 异常审计

确认脚本存在：

```bash
python tools/perf_data_collection/find_database_anomalies.py --help
```

本阶段的参数和退出码以当前分支脚本的 `--help` 与实际执行结果为准；RFC 只解释检测语义，不替代 CLI 契约。

然后运行 PR673 提供的只读审计：

```bash
python tools/perf_data_collection/find_database_anomalies.py \
  --database-path <candidate-db> \
  --output-dir <report-dir>/anomaly-audit \
  --residual-threshold 1.0 \
  --remeasure-limit <review-budget>
```

`1.0` 表示实测与 LOO 预测的相对残差达到 100% 才进入固定阈值候选；`review-budget` 只限制复测清单条数，不改变异常判定。
审核自动生成的 `anomaly_summary.md`、`anomaly_candidates.csv` 和 `remeasure_manifest.csv`：

脚本发现生产 mapping 引用但数据库缺失的 CSV 时，在完整写出报告后返回码 2；这表示确定性问题门禁失败，不是报告生成失败。

- 文件或字段错误、非法 latency、mapping 必需 CSV 缺失属于确定性问题，修复后重跑；
- 签名碰撞和 latency policy 分歧先修数据契约或解释语义；
- `REVIEW_REGIME` 先检查分桶或局部边界，不能直接改一整组 latency；
- 单点残差候选必须独立硬件复测；
- `INSUFFICIENT_EVIDENCE` 是审计器无法判断，不是通过。

复测清单是交接材料，不是 `start_microbench.py` 的直接输入。按其中的 CSV、行号、严格签名和 kernel 定位 replay case，复测后回到
阶段 2 写候选库，再完整重跑阶段 3。

### 4. 形成 PR

只有以下条件全部满足才进入发布：

- 基线、候选、mapping、密度 YAML、workload 和报告 hash 对应同一轮输入；
- 计划、采集和入库逐 Shape 对账；
- 关键 query strict exact 通过，其他支持 query 没有未解释的 roofline；
- 确定性数据问题已清零，统计候选已复测或明确不进入本次发布；
- 相关测试、pre-commit 和 Git LFS 检查通过；未验证项已经披露。

按 [`spec/workflows/pr-workflow.md`](../../../spec/workflows/pr-workflow.md) 执行 GitCode PR。调用
[`gitcode-precommit`](../gitcode-precommit/SKILL.md) 完成本地门禁，调用 [`gitcode-pr-create`](../gitcode-pr-create/SKILL.md)
生成 PR；远端创建、评论和状态操作必须使用 GitCode CLI。PR 说明至少附数据库 diff、query replay、异常摘要、测试、风险和回滚方式。

### 5. 处理运行反馈

按实际现象回到对应入口：

| 反馈 | 入口 |
| --- | --- |
| 新轴或轴密度不足 | `profiling-database-axis-density`，评审后更新 YAML，再回阶段 1 |
| kernel/CSV 映射错误 | `op-mapping-generator`，修 mapping 和测试 |
| 缺 replay 脚本 | `microbench-run-script-generator` |
| exact/interpolated/roofline MISS | 冻结 query 证据后回阶段 1 或 2 |
| latency 或数据契约可疑 | 运行异常审计，独立复测后回阶段 2 |
| 模型仿真回归 | `text-generate-executor` 复现，再区分数据、mapping、composite 或模型问题 |

不要因为运行反馈直接向正式 CSV 手工追加行；仍按基线、计划、候选库、检查和 PR 的顺序完成下一轮。

## 状态报告

每个阶段输出 `PASS`、`FAIL` 或 `BLOCKED`，并附证据路径：

```text
基线：commit、database/mapping/YAML/workload hash
计划：目标 models、ops、Shape 数、严格签名数、相对基线倍数
采集：成功、失败、无效 latency、冲突、repeat 和环境
回放：workload、hit/partial/miss、strict exact、roofline、未验证明细
审计：确定性问题、契约风险、固定阈值候选、弃权、复测结论
测试：命令、结果和未运行原因
发布：PR、数据库/LFS diff、风险、回滚和 reviewer 重点
```

任何阶段 `FAIL` 都返回对应修复阶段；`BLOCKED` 必须说明缺少的硬件、脚本、输入或证据，不得改写为通过。

## 完成标准

- 正式数据库未被本地流程直接修改。
- 密度规则、实际 Shape、采集结果、候选 CSV 和生产 query replay 能互相对账。
- 异常审计报告由脚本自动生成，统计候选经过独立复测后才改数据。
- 所有命令使用当前分支 `--help` 核验过的参数，没有写入 token、认证信息或本地绝对路径。
- PR 只包含计划内文件，验证证据、未通过项和回滚方式完整。

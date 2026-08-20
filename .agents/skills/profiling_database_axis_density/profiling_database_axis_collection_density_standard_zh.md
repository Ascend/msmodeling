# 实测算子数据库轴采集密度标准与依据

## 1. 目的与结论

本文规定通用实测算子性能数据库的 **V1 最低轴密度**。它面向多数模型的前置建库，不为单个模型逐算子重新设计轴；
算子使用哪些轴、轴如何从 Shape 或 Runtime metadata 提取，继续以采集和查询代码为准。

V1 采用保守工程基线：关键真实 query 必须 exact，其他支持 query 必须 exact 或在同一插值分桶内合法插值，
实测数据库支持范围内不得回退 roofline。数据库可以比本标准更密，不能更稀；唯一严格签名规模上限为当前数据库的 8 倍。

本目录只有三份交付物：

| 文件 | 职责 |
| --- | --- |
| 本文档 | 说明规则、证据、验收和证明边界 |
| `axis_collection_density.yaml` | 可机读的规范性数值基线 |
| `SKILL.md` | 新轴和密度变更的定标流程 |

`SKILL.md` 的 semver 记录工作流修订，YAML 的 `standard_version` 记录密度标准版本，两者独立演进。
YAML 是规范值，不是运行时配置文件；**不等于 YAML 已被采集代码读取**。本 PR 只提交规范，不宣称当前数据库已经达标。
是否达标必须在采集实施中比较 YAML、采集器生成结果和最终有效 CSV，不能只比较常量名称。

## 2. 术语和适用方式

- **轴**：采集或查询代码从 Shape、Runtime metadata 或通信参数中提取的一个独立取值。
- **严格签名**：规范化后的 kernel/query mode、全部输入输出 Shape/dtype/format、latency source，以及影响查询或执行分支的 Runtime metadata。
- **插值分桶**：从严格签名中只移除已声明的插值轴；其余影响执行路径的字段保持相同。生产代码明确实现并经 holdout 验证的归一化变换可以例外，但必须进入审计结果。
- **关键 query**：支持方在采集前冻结的 exact 清单。清单来自目标 workload、算法边界、瓶颈和寻优候选，不新增数据库 CSV 字段。
- **其他支持 query**：未进入关键清单，但声明由实测数据库支持的查询；允许 exact 或合法插值，不允许 roofline 回退。
- **最大相邻间隔**：相邻实测值之差不得超过 `max_gap`。
- **最大相邻比**：较大值除以较小值不得超过 `max_ratio`。
- **必测值**：`required_values`、关键 query、支持的拓扑规模、范围端点、对齐边界和算法切换点，必须有 exact 实测 latency。

`required_values` 只规定必须 exact 的锚点，`range`、`segments`、`max_gap` 和 `max_ratio` 规定其余点必须满足的最大间隔。
合格点集必须同时包含全部锚点并满足间隔约束；两者互不替代，也都不单独定义完整生成点集。

公共 `range` 是规则可覆盖的总轴域。具体 template 只使用其中一段时，代码或 query manifest 必须给出该子范围；
子范围端点必须实测，内部仍满足同一密度。不能因为 YAML 给出了总轴域，就声称每个 kernel 已覆盖整个总轴域。

关键 query 清单使用版本化 JSONL sidecar，不修改数据库 CSV。每行最少包含 `workload_id`、`critical` 和可直接送入生产查询的
`query` 对象；`query` 必须保留生成严格签名所需的完整 Shape、dtype、format 和 Runtime metadata。清单按
`(workload_id, strict_signature)` 去重并在采集前固定 hash。

## 3. V1 最低门禁

本节是 V1 门禁的唯一规范定义；YAML 只保存轴密度数值，Skill 只规定定标和验收流程。
一份数据库只有同时满足以下条件才算达到 V1：

1. 每个采集轴都能定位到代码中的提取公式和使用者；派生轴不重复生成。
2. 每个插值分桶在其声明支持范围内满足 YAML 的端点、必测值、`max_gap` 或 `max_ratio`。
3. 关键 query 清单 `100% strict exact`；清单必须在采集前冻结，不能在看到 MISS 后删除 query。
4. 其他支持 query `100% exact` 或在同一插值分桶内合法插值；禁止外推和跨执行分支插值。
5. 在同一同桶 holdout 上，插值 p90/p95、`>50%` 和 `>100%` 长尾数不得差于当前生产网格；项目已有更严格预算时从其规定。
6. 使用生产 compiled 查询路径回放支持 workload，要求 `roofline MISS=0`；复合算子按实际子查询逐项检查。
7. 每个计入密度的 Shape 都有正数、finite 的实测 latency；0、空值、失败行和插值结果不计入。
8. 采集实施交付时，YAML、采集器生成点集和最终有效 CSV 必须一致；采集失败造成的缺点必须补齐或移出支持范围。
9. 候选数据库按与基线相同的严格签名函数去重后不超过基线的 8 倍；不得拿 CSV 行数与简化 Shape 数混算。

这里的“最低”表示准入下限，不表示数学全局最优，也不限制补充真实 query、边界点或局部长尾点。

`strict exact` 不能由 datasource 返回的 `MEASURED` 标签直接判定。验收器必须把目标签名与正 latency 原始 CSV 行逐字段比较，
并输出 workload、目标签名 hash、匹配 CSV、匹配签名 hash 和最终分类。padding 邻点、跨 dtype 缩放或其他兼容命中统一记为
`interpolated/compatible`，不能冒充 exact。通信若以 `message_bytes` 归一化而忽略 dtype，或其他查询使用等价变换，必须由
生产策略明确声明并附同口径 holdout；否则按跨桶失败处理。

## 4. 证据及其结论

### 4.1 PR489 补库对比

PR489 变更 5 个 CSV 后，同一 holdout 的 success 从 11,449 增加到 12,377，rejected 从 21,191 降到 20,339；
`MatMulCommon` success 从 2 增加到 851，rejected 从 851 降到 0。这证明局部邻域补齐能提高数据库可用性。

补点并不自动消除异常：`DynamicQuant` success 从 433 增加到 442 时，`>100%` 长尾从 1 增加到 3；
`TransposeBatchMatMul` 补点后仍有固定 batch/K/N 下的局部 M 长尾。因此标准同时要求密度、严格分桶和真实 query，
不使用“总行数”代替质量。

### 4.2 固定快照回溯消融

在固定实验快照 master `9a665281` 按行叠加 PR654 `9a72d995` 的数据库上，以当前代码网格为 `L0`：
`L-1` 每隔一个非必测点保留一个，`L+1` 在相邻点间加入合法整数中点。固定 kernel、query mode、latency source、
core、dtype、format、其他 Shape 和 Runtime metadata 后，10,842 个一维 regime 中有 499 个可形成三档公共 holdout，
每档得到 6,977 个预测。

| 密度 | median | p90 | p95 | max | `>50%` | `>100%` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `L-1` | 6.35% | 32.96% | 45.62% | 245.70% | 249 | 39 |
| `L0` | 5.27% | 27.79% | 41.61% | 232.81% | 192 | 38 |
| `L+1` | 4.21% | 23.84% | 36.10% | 189.19% | 152 | 20 |

| 轴 | holdout | `L-1/L0/L+1` p90 | 能支持的结论 |
| --- | ---: | --- | --- |
| MatMul M | 1,557 | 18.86% / 16.97% / 15.32% | L0 优于更稀档，局部继续加密仍有收益 |
| elementwise token | 4,475 | 39.62% / 35.01% / 30.02% | 长尾最明显，应优先补真实点和局部点 |
| MoE token | 945 | 17.68% / 7.76% / 6.76% | L0 相对更稀档收益显著 |
| attention seq | 0 | 无 | 不能从本次消融推出 attention 的精确间隔 |

该实验支持“L0 不应再整体稀疏一档”，不证明每个分段边界都是最优值。只有 499/10,842 个 regime 可参与，
结论不能外推到未采区间、所有 kernel 或多维组合。

复核算法是：按上述严格签名固定非目标轴形成一维序列，只保留三档都有左右邻点的内部值，使用生产线性插值预测同一 holdout，
再按相同分母统计相对误差。原始逐点结果不在本三文件 PR 内，因此这些数字是固定历史分析摘要，不是当前数据库达标门禁。

### 4.3 GLM5.1 compiled query 回放

在固定代码快照 master `c35ac153` 和 PR654 `9a72d995` 数据上，使用 `--compile`、启用
`DispatchFFNCombine` 回放 GLM5.1 40K 的 24 个配置。Decode M5 为 98.42% 到 99.88%，Prefill 为 78.44% 到 99.97%，
两者都存在 roofline 回退。

`DispatchFFNCombine.csv` 的 9 个目标组合中，6 行 latency 为 0；另 3 行的物理权重 Shape 与目标 MOE-TP 严格签名不同，
因此实际可用为 0/9。`mlapo_quant` 的失败可定位到 `mla_preprocess_0_mix_aic.csv` 子查询。

这证明“CSV 有行”不等于“生产查询可用”，所以 V1 必须用真实 compiled query 验收完整严格签名和复合子查询。

### 4.4 规模与容量

在固定快照 master `990d4da5` 上，108 个 CSV 可归并为 20,489 个唯一语义 tuple：

| 方案 | 估算点数 | 相对当前 tuple |
| --- | ---: | ---: |
| 当前唯一语义 tuple | 20,489 | 1.00 倍 |
| 补齐每个单轴值 | 21,928 | 1.07 倍 |
| 受控两两组合 | 62,595 | 3.06 倍 |
| 完整笛卡尔积 | 1,005,253 | 49.06 倍 |

完整笛卡尔积是受控组合的 16.06 倍，因此默认由采集代码生成有效 tuple，不做所有轴全范围笛卡尔积。
该数据只证明完整组合成本过高，不证明其收益为零；真实 query 和已知多轴交互点仍必须保留。

固定 GLM5.1 trace 有 2,190 个可评估调用、195 个唯一基础签名。固定数据库快照有 38,265 行，叠加 PR654 的 10 个
严格签名后为 38,275 行。通用网格 dry-run 加 40 个 GLM5.1 真实 query 估算为 294,120 行，即 7.6844 倍。
这是**容量初估**，没有包含全部 Runtime metadata，也没有与基线使用完全相同的严格签名函数，不能作为 8 倍门禁已通过的证明。

## 5. YAML 数值规则的依据

### 5.1 MatMul M

M 是运行时工作量轴。小 M 区间的相对变化、启动和尾块开销占比更高，现有长尾也集中在低值和局部邻域；
因此 `1..8` 逐整数采集，随后按 `2/4/8/16/64/256` 逐段放宽，`1025` 以上用最大相邻比 2 控制跨数量级成本。
回溯消融证明该轴整体不能更稀一档；真实 query 和局部长尾点仍需额外 exact。

### 5.2 Token、序列和对齐边界

- `ELEM_TOKENS_GRID`：elementwise 通常接近线性内存流量，采用小值比 4、其余比 2；因回溯长尾较高，真实 query 优先 exact。
- `MOE_TOKENS_GRID`：token 改变专家负载和批量 GEMM，统一采用相邻比 2；回溯结果支持其明显优于更稀档。
- `ATTN_SEQ_GRID`：序列跨数量级，采用 `1..64` 相邻比 4、其余相邻比 2；这是保守基线，精确分段不能由现有 holdout证明。
- `PAD_TOKENS_GRID`：性能由对齐余数和边界决定，使用 `2^n±1` 一类必测值，不声明连续等距网格。

`tools/perf_data_collection/grid_generator/shape_grids.py` 中的公共 `ATTN_SEQ_GRID` 与 YAML 数值一致；Fused Attention 当前使用
`tools/perf_data_collection/grid_generator/generators/fused_attention.py` 中的场景点集，而不是直接消费该公共常量。YAML 的
`template_axes.fused_attention` 将 batch、seq 和 avg_seq_len 显式映射到公共规则；
场景点集必须在各自声明子范围内满足相邻比，并补齐关键 query。是否达标以实际生成结果为准，不能因为代码中存在
同名公共常量就判定通过。

### 5.3 结构宽度、Head 和模板特殊轴

`NK_GRID` 的必测值来自现有模型 hidden/intermediate width 及并行切分，低区间使用绝对间隔，高区间使用相邻比；
`HEADS_GRID`、`KV_HEADS_GRID`、`HEAD_DIM_GRID` 以离散结构值为主，真实结构值必须 exact。

模板特殊轴只用于代码中不能复用公共网格的局部 iterator，例如 TransposeBatchMatMul 的 batch/K/N、sampling vocab
和 cache batch。只有 `required_values` 的轴仅承诺这些 exact 锚点，不承诺锚点之间形成连续支持范围。

通信 `message_bytes` 使用 `tools/perf_data_collection/comm_bench/run_comm_bench.sh` 正式入口的 128 B 到 512 MB、相邻比 2，
覆盖已观察到的 272/528 B 小消息。`tools/perf_data_collection/comm_bench/generate_comm_microbench.py` 单独运行时默认从
1 KB 开始；正式采集必须通过 shell 入口或显式 `--bytes-grid` 传入 YAML 全范围，不能把较窄的 Python 默认值当作达标点集。
单节点 `MSG_BYTES` 和多节点完整 `MSG_BYTES_INTERPOD` 都必须一致；`QUICK=1` 的 5 点网格只用于连通性检查，不计入数据库
达标验收。

## 6. 多轴组合

YAML 只规定单轴最低密度，不规定算子使用哪些轴，也不要求完整笛卡尔积。采集器应先生成合法 tuple，再满足：

1. 关键真实 tuple、范围端点和算法切换 tuple 必测。
2. 改变一个轴时，其余轴和严格分桶字段固定，检查局部密度是否成立。
3. 发现强交互时，只在明确的轴组和局部范围加密；不得用跨桶插值掩盖缺失分桶字段。
4. 能由其他轴唯一推导的值不重复组合。

## 7. 新轴和密度变更

详细流程见 `SKILL.md`。最低要求是：先从代码确认轴语义、公式、使用者和严格分桶，再冻结支持范围及关键 query 清单；
使用已有密集 CSV 或专项校准点构造 `L-1/L0/L+1`，比较 exact、合法覆盖、holdout 误差、长尾和严格签名规模。

新轴只有少数真实值、尚无连续范围证据时，只写 `required_values`，不虚构 `range`、分段或间隔。要声明连续密度，
必须证明更稀候选降低覆盖或明显恶化同桶 holdout，并完成临时目录生成审计和生产 query 回放。

YAML 已有连续规则属于 V1 保守冻结值：当前证据不足时不得放宽，发现真实 MISS 或长尾时可以局部加密。上述新轴流程适用于
新增连续规则或放宽已有规则，不要求先删除现有 Attention 基线；这一区分避免用“缺少 holdout”反向把数据库变得更稀。

## 8. 交付验收

1. YAML 可解析，分段连续、无重叠，必测值位于声明范围内。
2. 每个公共轴和模板轴都能定位到真实采集或查询代码；未被任何生成器使用的规则必须说明用途或删除。
3. 将完整数据库版本目录复制到临时目录（必须包含 CSV 和 `op_mapping.yaml`），运行
   `tools/perf_data_collection/generate_shape_grid.py --data-dir <temp> --rows 0 --seed 0`，再对普通 iterator 和复杂 generator
   的生成结果逐子范围检查；脚本没有 `--dry-run`，禁止直接对正式库验收运行。
4. 通信没有无硬件生成模式，静态解析 `tools/perf_data_collection/comm_bench/run_comm_bench.sh` 的正式 `MSG_BYTES`、完整
   `MSG_BYTES_INTERPOD` 和 `tools/perf_data_collection/comm_bench/generate_comm_microbench.py` 的 `_DEFAULT_BYTES_GRID`；
   前两者必须相同并完整满足 YAML，后者必须是其子集，单/多节点入口都必须通过 `--bytes-grid` 传入正式点集。
5. 按第 3 节逐项提交 query 覆盖、holdout、复合子查询、有效 CSV 和严格签名容量证据，不在本节重复门禁阈值。
6. 规范 PR 可以只改本目录三份核心交付物及 `.agents/README.md`、`AGENTS.md` 索引；后续采集实施必须同步生成器和一致性测试。
   在此之前不能宣称数据库已经达标。

## 9. 证明边界与复现口径

- PR489 证据固定到远端 head `0bf424b9`，对同一 holdout 比较变更前后 5 个 CSV。
- 回溯消融固定到 master `9a665281`、PR654 `9a72d995` 和同一分桶/holdout 定义。
- GLM5.1 trace 使用 `docs/perf_database/forward_pass_traces/glm5-5.1_dc_1tok_ctx2500.csv`；compiled replay 固定到
  master `c35ac153` 和 PR654 `9a72d995`。
- 组合成本固定到 master `990d4da5`；容量 7.6844 倍是 Shape dry-run 代理，不是正式严格签名验收结果。
- LOOCV、holdout 和 profiling estimate 不能表述为真实硬件端到端吞吐提升。
- 修改具体间隔时必须保存输入 commit、query manifest 及 hash、严格签名定义、逐点结果和复现命令；没有这些材料时保留 V1 现值。
- 本节精确数字来自历史分析归档，三文件 PR 不携带脚本和原始结果；它们解释 V1 方向，但不单独证明当前数据库通过门禁。

---
name: profiling-database-axis-density
description: >-
  Use when adding a profiling database collection axis, changing an axis range or density,
  or checking whether collection code and generated shapes comply with the V1 density baseline.
metadata:
  version: 1.5.1
  source: local-session-analysis
---

# 实测算子数据库轴密度定标

## 目标

为新增轴或已有轴输出可执行的 V1 最低采集规则，并同步验证采集代码、生成 Shape 和真实查询。
本 Skill 不重新定义算子使用哪些轴，也不以算子名猜轴；轴语义和提取公式来自生产采集与查询代码。
Skill 的 semver 记录工作流修订，YAML 的 `standard_version` 记录密度标准版本，两者独立演进。

每次结论必须回答：

1. 轴是什么、单位是什么、如何从 Shape 或 Runtime metadata 得到。
2. 哪些 kernel/template 使用它，哪些字段必须严格分桶。
3. 支持范围、关键 query、必测值和候选密度是什么。
4. 为什么更稀会降低覆盖或恶化验证结果，当前规则是否满足 V1 门禁。
5. YAML、采集器、测试和数据库分别需要怎样修改。

默认先输出提案；只有用户明确要求实施后才修改文件。不得发布 PR 评论、resolve thread 或暴露认证信息。

## 事实源

按以下顺序读取：

1. `./axis_collection_density.yaml`：可机读的规范性数值基线。
2. `./profiling_database_axis_collection_density_standard_zh.md`：门禁、证据和证明边界。
3. `tools/perf_data_collection/grid_generator/shape_grids.py` 和
   `tools/perf_data_collection/grid_generator/config.yaml`：普通 iterator 和公共网格。
4. `tools/perf_data_collection/grid_generator/generators/` 和
   `tools/perf_data_collection/comm_bench/`：复杂 generator 和通信采集。
5. `tensor_cast/performance_model/profiling_database/`：生产查询、严格分桶和插值路径。
6. 相关测试、真实 query manifest、compiled query replay、profiling CSV 和 holdout 结果。

YAML 不是运行时配置。发现代码中存在同名常量，不代表复杂 generator 已消费该规则；必须检查实际展开点集。

## 输出契约

输出以下六部分。缺证据时写“未确定”，不得补造范围、间隔或阈值。

| 部分 | 必须包含 |
| --- | --- |
| 轴契约 | 名称、单位、类型、公式、合法域、使用者、严格分桶字段 |
| 查询契约 | 支持 workload、关键 query 清单来源、其他支持 query |
| 证据 | 代码位置、数据快照、候选点、holdout、回放和容量口径 |
| 候选规则 | `required_values`、`range`、分段、`max_gap/max_ratio` |
| 验收结果 | exact、合法插值、roofline、误差、长尾、有效 latency 和规模 |
| 变更清单 | YAML、采集实现、测试、数据库补点、复现命令和剩余风险 |

路径使用仓库相对路径和行号；结论中的 commit、数据库版本、query manifest 和严格签名定义必须固定。

## 工作流程

### 1. 找到轴的真实定义

先精确检索，再沿采集、CSV、查询和测试调用链阅读：

```bash
grep -R -n -E "<axis_name>|iterators|generator_function|Runtime .*|input_shapes|output_shapes" \
  tools/perf_data_collection tensor_cast tests
```

记录：

- 轴值的提取公式、数据类型、单位和最小合法步长。
- 合法上下界、整除/对齐限制和算法切换边界。
- 使用该轴的全部普通 iterator、复杂 generator、后端变体和复合子查询。
- dtype、format、layout、transpose、sparse mode、latency source、设备数、拓扑及其他严格分桶字段。
- 改变该轴时必须固定的其他 Shape。

CSV、采集器和查询端公式不一致时先修公式，不进入密度设计。轴没有代码定义时，不得仅根据名称添加 YAML。

### 2. 判断规则类型

| 代码事实 | 规则 |
| --- | --- |
| 有序数值轴，改变后仍在同一执行路径 | 可以定义连续 `range` 和密度 |
| 枚举值可能改变执行路径 | 全部支持值写入 `required_values`，不插值 |
| 只有少量真实值，没有连续范围证据 | 只写 `required_values` |
| 能由其他轴唯一推导 | 不新增规则，记录源轴 |
| dtype/layout/算法/拓扑等分支 | 严格分桶，不作为连续轴 |
| 语义、公式或合法范围不清楚 | 停止并列出缺失信息 |

同名轴在不同 template 中公式或性能行为不同，使用模板局部规则；不要把不同语义平均成公共密度。

### 3. 冻结支持范围和 exact 清单

支持范围来自代码能力和明确支持的 workload，两者缺一不可：

1. 汇总真实 query 的轴值、调用频次、严格签名和模型/配置来源。
2. 从代码提取合法域、对齐要求、容量限制和算法边界。
3. 在采集前冻结关键 query 清单；清单存于版本化 JSONL manifest，不增加数据库 CSV 列。每行使用
   `{"workload_id": ..., "critical": true/false, "query": {...}}`，其中 `query` 是完整生产查询对象。
4. 将关键 query、范围端点、对齐边界、算法切换点及合法相邻值加入 exact 必测集合。
5. 对其余支持 query 记录 `exact/interpolated/compatible/roofline` 分类，是否通过按主文档第 3 节判定。

manifest 按 `(workload_id, strict_signature)` 去重并固定 hash。关键 query 清单由支持方根据目标 workload、瓶颈和寻优候选确定；
密度定标不得在看到 MISS 后把失败 query 改成非关键。
范围外 query 不能记为已覆盖。只有少数真实值时，不得把最小值和最大值冒充连续支持范围。

`strict_signature` 至少包含规范化后的 kernel/query mode、全部输入输出 Shape/dtype/format、latency source 和影响查询的
Runtime metadata。exact 验收直接与正 latency 原始 CSV 行比较；不得只看 datasource 的 `MEASURED` 标签。生产代码的 padding、
跨 dtype 缩放或其他等价变换只能归类为 `interpolated/compatible`，并记录变换前后签名。插值分桶只移除已声明的插值轴；
额外归一化字段必须由生产策略明确声明并通过同口径 holdout，否则按跨桶失败处理。
逐 query 审计至少输出 `workload_id`、目标签名 hash、匹配 CSV、匹配签名 hash 和 `exact/interpolated/compatible/roofline` 分类。

### 4. 构造可比较的数据

优先复用同设备、软件版本、kernel、严格分桶和 latency source 下的现有有效 CSV。现有点不足以形成 holdout 时，
专项采集范围端点、真实值、边界以及相邻点的合法算术或几何中点，形成密集参考集。

参考集要求：

- 每个点 latency 为正数、finite，失败和 0 latency 单独报告。
- 改变目标轴时，其他 Shape 和严格分桶字段保持不变。
- 对波动点重复测量；先处理温度、频率、运行时或采集失败，不用增加密度掩盖脏数据。
- 采集顺序、设备/软件版本和原始结果可复现。
- 用于选择规则的点和最终 holdout 分开；现有数据太少时只输出 exact 规则。

### 5. 生成密度候选

有序数值轴从当前代码点集或密集参考集建立 `L0`：

1. 先放入全部 exact 必测值。
2. 绝对差值更能描述性能变化时，以算术中点补点，候选写成 `max_gap`。
3. 范围跨数量级且相对变化更重要时，以几何中点补点，候选写成 `max_ratio`。
4. 两种表示都合理时，在相同 holdout 上比较，不按轴名猜测。
5. 按性能变化、算法边界或长尾位置分段；没有数据支持时保持单段保守规则。

构造两个对照：

- `L-1`：从 L0 有序删除非必测点，保持范围、端点、关键 query 和边界不变。
- `L+1`：在 L0 的可插值相邻区间增加合法中点，保持严格分桶和支持范围不变。

`L-1/L0/L+1` 用同一组 holdout、同一生产插值方法和同一严格签名定义比较。L+1 用于发现仍需局部加密的位置；
V1 不以“L+1 完全没有收益”为前提，因为本标准声明的是保守准入下限，不是全局最优点集。

### 6. 检查多轴交互

对 MatMul、Attention、MoE 和复合算子，从真实 query 选择普通、边缘、瓶颈和算法切换切片：

1. 每次只改变目标轴，其余轴和严格分桶字段固定。
2. 每个使用该轴的 template 至少有可复现切片。
3. 不同切片结论不一致时，先检查是否遗漏严格分桶字段。
4. 确有局部交互时增加模板或局部分段，不修改无关公共轴。
5. 保留关键真实 tuple；不默认生成所有轴的完整笛卡尔积。

### 7. 执行 V1 验收

V1 门禁只以 `./profiling_database_axis_collection_density_standard_zh.md` 第 3 节为准，本 Skill 不复制数值判据。
对 `L-1/L0/L+1` 使用相同严格签名、holdout 和生产查询路径，并为每个候选输出：

| 类别 | 指标 |
| --- | --- |
| 数据质量 | 正/finite latency、重复签名、冲突 latency、失败点 |
| 覆盖 | 关键 query strict exact、其他 query exact/interpolated/roofline |
| 插值 | median、p90、p95、max 相对误差及长尾数量 |
| 规模 | 生成 Shape、有效严格签名、相对基线倍数 |
| 一致性 | YAML、普通 iterator、复杂 generator、通信脚本和最终 CSV |

逐项对照主文档第 3 节；任何门禁缺少证据即不通过。若 `L-1` 与 `L0` 结果等价，则采用更稀档；`L+1` 只用于定位
仍需局部加密的位置，不作为 V1 通过的前提。

新增连续轴或放宽已有规则时，无法满足连续密度证据就只提交 exact `required_values`，不得为了“规则完整”填入未经验证的
range、gap 或 ratio。YAML 已有数值是 V1 保守冻结值：证据不足时保持不变，真实 MISS 或长尾可以支持局部加密。

### 8. 同步 YAML、采集器和测试

用户确认实施后，按顺序修改：

1. 更新 `axis_collection_density.yaml`，只写已经验证的范围、分段和必测值。
2. 更新所有使用者：普通 `tools/perf_data_collection/grid_generator/shape_grids.py`、
   `tools/perf_data_collection/grid_generator/config.yaml`、复杂 generator 和通信脚本不能遗漏。
3. 将完整数据库版本目录复制到临时目录（包括 CSV 和 `op_mapping.yaml`），运行
   `python3 tools/perf_data_collection/generate_shape_grid.py --data-dir <temp> --rows 0 --seed 0`；该命令会写 CSV，不能直接指向正式库。
4. 增加或更新一致性测试，解析 YAML 并验证生成点集的端点、必测值、最大间隔和最大相邻比。
5. 复杂 generator 按每个 template/场景的声明子范围检查，不能只验证公共常量。
6. 通信没有无硬件生成模式；使用下面的静态检查读取正式 shell 点集和 Python 默认点集。正式点集必须满足 YAML，
   Python 默认点集必须是正式点集的子集，且 shell 必须显式传递 `--bytes-grid`。
7. 采集后验证有效 latency、严格签名去重、失败缺点和 8 倍容量。
8. 用生产查询路径回放，并独立核对原始 CSV 严格签名；复合 MISS 记录顶层 Shape、失败子 CSV、物理子 Shape 和 Runtime metadata。

```bash
python3 - <<'PY'
import ast
import re
from pathlib import Path

import yaml

root = Path.cwd()
rule_path = root / ".agents/skills/profiling_database_axis_density/axis_collection_density.yaml"
rule = yaml.safe_load(rule_path.read_text())["communication_axes"]["message_bytes"]
shell_path = root / "tools/perf_data_collection/comm_bench/run_comm_bench.sh"
shell = shell_path.read_text()
match = re.search(r'^MSG_BYTES="([0-9 ]+)"$', shell, re.MULTILINE)
interpod_matches = re.findall(r'^\s*MSG_BYTES_INTERPOD="([0-9 ]+)"$', shell, re.MULTILINE)
assert match and interpod_matches
assert "--bytes-grid $bytes" in shell and "--bytes-grid $MSG_BYTES_INTERPOD" in shell
formal = [int(value) for value in match.group(1).split()]
interpod = max(([int(value) for value in item.split()] for item in interpod_matches), key=len)
generator_path = root / "tools/perf_data_collection/comm_bench/generate_comm_microbench.py"
tree = ast.parse(generator_path.read_text())
default = next(
    ast.literal_eval(node.value)
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "_DEFAULT_BYTES_GRID"
        for target in node.targets
    )
)
assert interpod == formal
assert formal[0] == rule["range"]["min"] and formal[-1] == rule["range"]["max"]
assert set(default).issubset(formal)
assert all(right / left <= rule["max_ratio"] for left, right in zip(formal, formal[1:]))
PY
```

规范 PR 可以只修改本目录三份核心交付物及 `.agents/README.md`、`AGENTS.md` 索引；后续采集实施必须同步生成器和一致性测试。
在此之前不能宣称采集器或数据库已经达标。

## 报告模板

```text
结论：采用 / 拒绝 / 保留 exact-only
轴契约：语义、公式、合法域、使用者、严格分桶
查询契约：支持 workload、关键 query 清单及冻结方式
数据：commit、数据库版本、参考点、holdout 和 latency 质量
候选：required_values、L-1/L0/L+1、range、gap/ratio、分段
结果：exact、interpolated、roofline、median/p90/p95/max、长尾、严格签名规模
依据：更稀为何失败，L0 为何达到 V1，L+1 暴露了哪些局部风险
变更：YAML、采集实现、测试、数据库补点和复现命令
边界：未验证 kernel、区间、版本和剩余风险
```

## 完成标准

- 已逐项通过主文档第 3 节门禁，并保存对应的逐点证据和复现命令。
- 轴语义、公式、合法域、使用者和严格分桶都有代码证据。
- 新增连续轴和放宽规则附 `L-1/L0/L+1` 同口径结果；既有 V1 规则证据不足时不得放宽。
- YAML、采集实现、最终 CSV 和生产查询回放通过一致性检查。
- 未把插值值冒充实测 latency，未写入本地绝对路径、token 或认证信息。

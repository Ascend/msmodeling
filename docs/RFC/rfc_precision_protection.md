# 性能回归测试框架设计文档

状态 (Status): Draft
作者 (Authors): @yuyinkai
创建日期 (Created): 2026-05-19
更新日期 (Updated): 2026-09-22

---

## 1. 概述

### 1.1 简介

性能回归测试框架是一套自动化测试系统，用于检测代码变更是否导致推理性能退化。框架通过运行预设的推理场景，自动提取 `Total time for analytic` 指标和算子级耗时数据，与基线值对比，当偏差超过容忍度时自动报错，并在所有用例执行完毕后输出汇总报告。

核心价值：

- **自动化**：无需手动逐个运行 CLI 命令、肉眼对比数值
- **可量化**：将性能波动转化为可度量的百分比偏差，支持总耗时和算子级两个维度
- **可追溯**：每次运行产出结构化报告文件（`regression_report.txt`），便于追踪性能趋势
- **低门槛**：新增用例只需在 `cases/` 目录下创建 JSON 配置文件，无需修改测试框架代码
- **类型安全**：文本模型和视频模型使用独立的数据结构，字段约束清晰

### 1.2 动机

当前项目中，开发者通过 CLI 命令手动运行推理模拟，肉眼观察 `Total time for analytic` 输出来判断性能是否退化。这种方式存在以下痛点：

| 痛点 | 说明 |
|------|------|
| 效率低 | 每次代码变更后需要手动跑多条命令、逐一对比 |
| 易遗漏 | 人工对比容易忽略小幅退化（如 15% 的性能回退） |
| 无标准 | 缺乏统一的判定标准，不同开发者对"可接受偏差"认知不一致 |
| 难追溯 | 没有历史记录，无法追踪性能变化趋势 |
| 算子级盲区 | 只能看总耗时，无法定位具体哪个算子导致退化 |

### 1.3 目标

**目标**：

- 提供声明式用例配置方式，通过 JSON 文件描述"跑什么命令、基线值是多少"
- 自动运行所有用例，执行两项检测：总耗时对比 + 算子级对比
- 偏差超过容忍度自动报错，输出结构化汇总报告
- 文本模型和视频模型使用独立的数据结构，防止字段误用
- 支持 pytest 生态，可无缝集成 CI/CD
- 用例配置与测试代码分离，新增模型无需修改框架源码

**非目标**：

- 不做性能趋势存储与可视化（可后续扩展）
- 不做自动基线更新（基线需要显式生成和提交）
- 不做多设备并行执行优化

---

## 2. 用例分析

### 2.1 典型使用场景

| 场景 | 描述 |
|------|------|
| 日常开发 | 开发者修改了算子实现，提交前跑一次回归测试，确认没有性能退化 |
| Code Review | CI 自动运行回归测试，PR 中展示性能变化 |
| 发版检查 | 发布新版本前，全量运行回归用例，确保所有场景性能达标 |
| 新模型接入 | 在 `cases/` 目录添加 JSON 文件即可接入新模型，无需改框架代码 |
| 基线刷新 | 模型升级或性能优化后，删除旧基线重新生成并验证 |

### 2.2 功能点

1. **用例配置**：通过 `cases/*.json` 声明用例（模型类型、参数、基线值、容忍度），框架自动发现和加载
2. **双模型支持**：`TextPerfRegressionCase` 处理文本/VL/LLM 模型，`VideoPerfRegressionCase` 处理视频扩散模型
3. **自动执行**：pytest 参数化机制自动遍历所有用例
4. **检测一 — 总耗时对比**：实际耗时 vs 初版耗时 + vs 基线耗时，各带独立容忍度
5. **检测二 — 算子级对比**：Top-N 耗时算子 vs 算子基线，支持算子缺失/新增/耗时偏差/调用次数四种检测
6. **偏差计算**：`(actual - baseline) / baseline`，与容忍度对比
7. **汇总报告**：`tearDownClass` 中打印结构化表格，标注 PASS/FAIL，同时写入 `regression_report.txt`

### 2.3 关键性能指标

| 指标 | 说明 |
|------|------|
| 检出率 | 超过容忍度的性能退化 100% 被捕获（总耗时 + 算子级） |
| 误报率 | 正常波动范围内不触发报警 |
| 执行耗时 | 取决于用例数量和模型大小 |

---

## 3. 方案设计

### 3.1 总体架构

#### 3.1.1 目录结构

```text
tests/st/
├── regression.py               ← 统一入口：总耗时回归 + 算子回归
├── auto_baseline.py            ← 独立入口：自动基线运行器
├── __init__.py                 ← 包初始化
├── cases/                      ← 用例 JSON 配置文件目录（含算子基线数据）
│   ├── qwen3-8B-decode.json    ← 文本模型用例
│   ├── qwen3-8B-prefill.json
│   ├── wan2.2-ulysses8.json    ← 视频模型用例
│   └── ...
└── README.md                   ← 使用指南
```

#### 3.1.2 用例配置格式

**文本模型**（`type: "text"`）：

```json
{
  "type": "text",
  "name": "qwen3-8B-decode",
  "description": "Qwen3-8B decode, 32 queries, ctx=1536, TP=2, compile",
  "initial_time_s": 0.012733,
  "baseline_time_s": 0.015406,
  "initial_tolerance": 0.10,
  "baseline_tolerance": 0.20,
  "operator_top_n": 10,
  "operator_tolerance": 0.10,
  "user_input": {
    "device": "ATLAS_800_A2_376T_64G",
    "model_id": "Qwen/Qwen3-8B",
    "num_queries": 32,
    "query_len": 1,
    "context_length": 1536,
    "do_compile": true,
    "decode": true,
    "quantize_linear_action": "DISABLED",
    "tp_size": 2,
    "world_size": 2
  }
}
```

**视频模型**（`type: "video"`）：

```json
{
  "type": "video",
  "name": "wan2.2-ulysses8",
  "description": "Wan2.2-T2V-A14B ulysses=8, batch=1, seq=128",
  "initial_time_s": 8.542,
  "baseline_time_s": 7.625,
  "initial_tolerance": 0.10,
  "baseline_tolerance": 0.20,
  "operator_top_n": 10,
  "operator_tolerance": 0.10,
  "device": "ATLAS_800_A3_752T_128G_DIE",
  "model_id": "model_config/Wan2.2-T2V-A14B-Diffusers",
  "seq_len": 128,
  "batch_size": 1,
  "height": 720,
  "width": 1280,
  "frame_num": 81,
  "sample_step": 1,
  "dtype": "bfloat16",
  "use_cfg": true,
  "world_size": 8,
  "ulysses_size": 8,
  "cfg_parallel": false,
  "quantize_linear_action": "DISABLED"
}
```

JSON 中的 `model_id` 使用相对于 `tests/st/` 的路径，框架在加载时自动解析为绝对路径，保证跨平台和 CI 环境可移植。

#### 3.1.3 数据类设计

三个 dataclass 构成清晰的继承层次：

**BasePerfRegressionCase**：公共基类，包含所有用例共享的字段。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | 必填 | 用例唯一标识，算子基线存储在 `operators` 字段中 |
| `description` | `str` | 必填 | 用例描述，报错时展示 |
| `initial_time_s` | `float` | `0.0` | 初版总耗时（秒），填 0 跳过初版对比 |
| `baseline_time_s` | `float` | `0.0` | 基线总耗时（秒），填 0 跳过基线对比 |
| `initial_tolerance` | `float` | `0.10` | 与初版的容忍度（10%），对称判定 |
| `baseline_tolerance` | `float` | `0.20` | 与基线的容忍度（20%），对称判定 |
| `baseline_max_increase_pct` | `float` | `null` | 与基线相比的增幅上限（变慢方向硬顶），仅对 analytic 用例生效；显式配置优先于 case 类型默认值；profiling 用例固定不生效 |
| `operator_top_n` | `int` | `10` | 对比前 N 个耗时最高的算子 |
| `operator_tolerance` | `float` | `0.10` | 算子级容忍度（10%） |
| `operators` | `array` | `[]` | 算子基线数据列表，每项含 `name`、`total_time_s`、`num_calls` |

**TextPerfRegressionCase(BasePerfRegressionCase)**：文本/VL/LLM 模型用例。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `user_input` | `UserInputConfig` | `None` | 推理配置（等价于 CLI 参数），由 JSON 中的 `user_input` 对象反序列化 |

**VideoPerfRegressionCase(BasePerfRegressionCase)**：视频扩散模型用例。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | `str` | `""` | 目标设备名称 |
| `model_id` | `str` | `""` | 模型配置目录路径 |
| `seq_len` | `int` | `0` | 序列长度 |
| `batch_size` | `int` | `0` | 批大小 |
| `height` | `int` | `0` | 视频高度 |
| `width` | `int` | `0` | 视频宽度 |
| `frame_num` | `int` | `0` | 帧数 |
| `sample_step` | `int` | `0` | 采样步长 |
| `dtype` | `str` | `"float16"` | 数据类型 |
| `use_cfg` | `bool` | `False` | 是否启用 classifier-free guidance |
| `world_size` | `int` | `1` | 总设备数 |
| `ulysses_size` | `int` | `1` | Ulysses 序列并行度 |
| `cfg_parallel` | `bool` | `False` | 是否启用 CFG 并行 |
| `quantize_linear_action` | `QuantizeLinearAction` | `DISABLED` | 量化方式 |

#### 3.1.4 用例加载机制

框架通过 `_load_perf_regression_cases()` 函数自动从 `cases/` 目录加载所有 `*.json` 文件：

1. 遍历 `cases/` 目录下所有 `*.json` 文件，按文件名排序
2. 读取 JSON，根据 `type` 字段判断是 `"text"` 还是 `"video"`
3. 文本类型：从 `user_input` 对象构造 `UserInputConfig`，枚举字段（如 `QuantizeLinearAction`）自动从字符串还原
4. 视频类型：枚举字段自动还原，`model_id` 为相对路径时自动基于 `tests/st/` 解析为绝对路径
5. 构造对应的 `TextPerfRegressionCase` 或 `VideoPerfRegressionCase` 实例

**设计意图**：新增用例 = 新增 JSON 文件，零框架代码改动。"测试数据"和"测试逻辑"解耦，降低扩展成本。

#### 3.1.5 执行流程

每个用例的测试方法 `test_performance_regression` 执行以下流程：

1. `torch.compiler.reset()` 清理编译缓存
2. 根据用例类型分发执行：
   - 视频模型：调用 `cli.inference.video_generate.run_inference`，捕获 stdout 作为 `table_result`
   - 文本模型：创建 `ModelRunner` 并调用 `run_inference`，从 `ModelRunnerMetrics.table_result` 获取输出
3. 通过 `_resolve_table_performance_model_name` 自动推断表格指标名（profiling 模式为
   `empirical`，analytic 模式为 `analytic`），再从 `table_result` 中正则提取
   `Total time for {指标名}`，得到 `actual_time_s`
4. **检测一 — 总耗时对比**：
   - 如果 `initial_time_s > 0`：计算 `(actual - initial) / initial`，与 `initial_tolerance` 对比
   - 如果 `baseline_time_s > 0`：计算 `(actual - baseline) / baseline`，与 `baseline_tolerance` 对比；**analytic 用例**额外叠加增幅上限（文本 5%、视频 20%，case 级 `baseline_max_increase_pct` 显式配置优先）——analytic 理论时长是最优下界，实际大幅超过基线意味着仿真失真；**profiling 用例**（调用处将 `user_input.performance_model` 归一化为 `"profiling"`/`"analytic"` 后传入 `_is_baseline_time_acceptable` 判定）不叠加增幅上限，仅判断 `abs(diff_pct) <= tolerance`——empirical 基线来自实测，已反映真实执行成本，"analytic 最优下界"的前提不成立
   - 两项中任一未通过则记录为 FAIL
5. **检测二 — 算子级对比**：
   - 从 `table_result` 中解析 Top-N 算子（`_parse_top_operators`）
   - 从用例 JSON 的 `operators` 字段加载算子基线（`_load_baseline_operators`）
   - 如果 `operators` 字段为空：记 `NO_BASELINE` 告警（不自动生成基线，防止测试污染仓库），算子级对比跳过、不影响用例 PASS/FAIL 判定
   - 如果基线文件存在：从基线中取 `total_time_s` 最大的 Top-N 算子，与当前实际算子逐一对比
   - 检测四种异常：算子缺失、新增算子、耗时偏差超限、调用次数不匹配
   - 将 violations 详细信息一并存入 `_op_results`，失败时完整回放异常原因
6. 综合判定：任一检测失败则 `self.fail()`

#### 3.1.6 算子级对比机制

算子基线数据存储在用例 JSON 的 `operators` 字段中：

```json
{
  "type": "text",
  "name": "qwen3-8B-decode",
  "description": "...",
  "operators": [
    {"name": "aten::mm", "total_time_s": 0.003200, "num_calls": 64},
    {"name": "aten::addmm", "total_time_s": 0.002100, "num_calls": 32}
  ]
}
```

对比逻辑：

- 从基线中按 `total_time_s` 降序排序后取 Top-N 算子（不依赖 JSON 文件写入顺序）
- 检测基线中存在但当前结果中缺失的算子（MISSING OPERATOR）
- 检测当前结果中存在但不在基线 Top-N 中的新算子（NEW OPERATOR）
- 对共同算子计算时间偏差百分比，超限标记 FAIL
- 对比 `num_calls` 是否一致，不一致额外标记

#### 3.1.7 报告输出

全部用例执行完毕后，`tearDownClass` 输出两份汇总报告到终端和 `regression_report.txt`：

**检测一报告**（总耗时）：

```text
==============================================================================================================
  [检测一] 总耗时回归汇总
==============================================================================================================
Case                                       Actual      Init   InitDiff%      Base   BaseDiff%         Status
--------------------------------------------------------------------------------------------------------------
qwen3-8B-decode                          12.733ms   12.733ms     +0.00%   15.406ms    -17.35%           PASS
--------------------------------------------------------------------------------------------------------------
Total: 15 | Passed: 14 | Failed: 1 | No Baseline: 0
==============================================================================================================
```

**检测二报告**（算子级）：

```text
==============================================================================================================
  [检测二] 算子级回归汇总
==============================================================================================================
Case                              Operator                      Baseline    Actual         Diff       #Calls    Status
--------------------------------------------------------------------------------------------------------------
qwen3-8B-decode                   aten::mm                       3.200ms    3.580ms     +11.88%        64/64    FAIL
qwen3-8B-decode                   aten::addmm                    2.100ms    2.080ms      -0.95%        32/32     PASS
--------------------------------------------------------------------------------------------------------------
Total: 15 | Passed: 13 | Failed: 2 | No Baseline: 0
==============================================================================================================
```

失败时会输出详细的 violations 信息（算子名称、基线值、实际值、偏差百分比）。

---

### 3.2 基线生命周期

基线文件的生成和维护遵循显式流程，测试运行时不会自动创建基线：

1. **创建用例**：在 `cases/` 目录下创建 `<case_name>.json` 配置文件
2. **生成基线**：显式生成算子基线数据，填充到用例 JSON 的 `operators` 字段
3. **验证**：运行回归测试，确认总耗时与算子级对比结果稳定、可复现
4. **提交**：将包含算子基线数据的用例 JSON 文件提交到版本控制
5. **刷新基线**：当模型升级、性能优化或算子有意变更时，清空 `operators` 字段后重新走 2-4 步；提交时必须注明刷新原因

---

### 3.3 技术选型

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **pytest + parameterized** | 与项目测试框架一致，支持参数化，CI 友好 | 需依赖 pytest | 采用 |
| subprocess 调用 CLI | 完全模拟用户行为 | 启动开销大，输出解析不稳定 | 不采用 |
| 独立 Python 脚本 | 简单直接 | 无法利用 pytest 生态 | 不采用 |
| 目录化 JSON 配置 | 新增用例无需改代码，扩展成本低 | 需要 JSON 格式校验 | 采用 |
| 数据类继承层次 | 文本/视频字段约束清晰，防止误填 | Python dataclass 继承有默认参数限制 | 采用 |

---

### 3.4 工具函数

| 函数 | 职责 |
|------|------|
| `_parse_total_time_s(table_result, performance_model_name)` | 正则提取 `Total time for {performance_model_name}` 并转换为秒 |
| `_parse_top_operators(table_result, top_n, performance_model_name)` | 解析算子耗时表，返回前 N 个算子的 (名称, 耗时, 调用次数) |
| `_load_baseline_operators(case_name)` | 从用例 JSON 的 `operators` 字段加载算子基线 |
| `_save_baseline_operators(case_name, operators)` | 将算子基线写入用例 JSON 的 `operators` 字段（供基线生成工具使用） |
| `_load_perf_regression_cases()` | 从 `cases/` 目录加载所有 JSON 配置文件 |
| `_print_time_summary(results)` | 输出检测一汇总报告（过滤 `grouped` 行，组子用例由 Test 3 汇总呈现） |
| `_print_operator_summary(op_results, op_detail_rows, report_file)` | 输出检测二汇总报告 |
| `_print_group_summary(group_rows)` | 输出 Test 3 分组平均判定汇总：逐组渲染组行（Tolerance/AvgDiff/GroupStatus）与子用例明细表 |

---

### 3.5 auto_baseline.py 自动基线运行器

独立的 pytest 入口，用于自动跑两次对比：第一次建立基线，第二次对比，容忍度默认 5%。与 regression.py 不同，auto_baseline 聚焦于同一场景下两次运行的一致性验证，适合快速自测。

**关键保护**：

- 基线运行失败和对比运行失败分别捕获，返回结构化错误结果
- `baseline_time_s <= 0` 时有 ZeroDivisionError 守卫，返回明确错误信息而非算术异常

---

### 3.6 分组平均精度判定机制

#### 3.6.1 背景与动机

部分模型（如 GLM-5.2）的多个场景（不同输入长度 × prefill/decode）共享同一套模型代码路径与
profiling 数据库，单场景的 empirical 预估时长受算子查表边界效应影响，个别 case 的偏差天然偏大
但整体预估质量是稳定的。若对每个 case 单独硬判定，会产生与整体精度无关的误报。

因此引入**分组判定**机制：

- 组内单个 case 偏差超过阈值 → 仅告警（WARN），不判定失败；
- 取组内各 case 精度差距的**平均值**与阈值对比，平均值超阈值 → 判定失败。

未声明分组的 case 维持原有行为：单个 case 偏差超阈值即判定失败。

#### 3.6.2 配置格式（组用例单文件）

分组用**一个独立的组用例 JSON** 描述，组内子用例以子用例名为 key 内嵌在
`sub_cases` 对象中。value 结构与单个 text 用例 JSON 的顶层结构一致，但**不包含
`type` 和 `name`**——子用例名由 `sub_cases` 的 key 提供，其余字段
（`initial_time_s`/`baseline_time_s`/容忍度/`operators` 算子基线/`user_input`）
与 text 用例相同：

```json
{
  "type": "group",
  "name": "glm5.2-group-profiling",
  "description": "GLM-5.2 4 scenarios group average judgment",
  "group_tolerance": 0.15,
  "sub_cases": {
    "glm5.2-3.5k-1.5k-decode": {
      "description": "...",
      "initial_time_s": 0.04288,
      "baseline_time_s": 0.042342,
      "initial_tolerance": 0.1,
      "baseline_tolerance": 0.15,
      "operator_top_n": 10,
      "operator_tolerance": 0.1,
      "operators": [],
      "user_input": { "...": "..." }
    },
    "glm5.2-3.5k-1.5k-prefill": { "...": "..." },
    "glm5.2-64k-1k-decode": { "...": "..." },
    "glm5.2-64k-1k-prefill": { "...": "..." }
  }
}
```

> 示例中子用例的 `operators` 为空数组，表示该组未启用算子级防护：该子用例在
> Test 3 汇总的 `Op(TopN)` 列呈现为 `NO_BASELINE`（组子用例的算子结果不再进入
> Test 2 汇总，统一由 Test 3 呈现，见 3.6.5），但不影响组判定（算子级对比本身为
> warning-only，见 3.6.3）；如需启用算子级对比，须填入非空算子基线。注意组判定状态
> `FAIL(NO_BASELINE)`（见 3.6.5）指子用例缺失**时间基线**导致无法计算组平均，与
> 算子基线为空是两回事。

字段说明：

| 字段 | 层级 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `type` | 顶层 | `str` | 必填 | `"group"` 标识组用例，加载器据此走组解析路径 |
| `name` | 顶层 | `str` | 必填 | 组用例名，必须与文件名一致（命名约定见 3.7 节） |
| `group_tolerance` | 顶层 | `float` | `null` | 组级容忍度。为 `null` 时取组内各子用例 `baseline_tolerance` 的最小值（保守方向） |
| `sub_cases` | 顶层 | `object` | 必填 | 子用例集合，key 为子用例名；value 与单个 text 用例 JSON 顶层结构一致，但不包含 `type`/`name`（名称取自 key） |

设计动机：**执行顺序由代码保证**。子用例在组测试方法内部顺序执行，完成后立即做组
平均判定，不再依赖 pytest 收集顺序和跨用例的类级状态累积，也不受 `pytest -k` 把
成员与组判定拆散的影响。

代价（已接受的权衡）：

- pytest 报告不再提供子用例级 PASS/FAIL（子用例超阈值本来就只告警，明细进入
  Test 3 组汇总表与组用例日志）；
- `pytest -k` 只能整组过滤，无法单独运行组内某个子用例；
- 子用例的算子基线内嵌在组 JSON 中，`_load_baseline_operators` 的
  `{子用例名}.json` 独立文件定位方式对组子用例不再适用，改为从组 JSON 的
  `sub_cases[key].operators` 读取。

数据结构：新增 `GroupPerfRegressionCase` 数据类（`name`、`description`、
`group_tolerance`、`sub_cases: dict[str, TextPerfRegressionCase]`），复用现有
`TextPerfRegressionCase` 表示子用例，加载器按 `type` 分发。单个用例 JSON（`type:
"text"`）不新增任何字段，行为完全不变。

#### 3.6.3 判定流程

**单 case 测试方法 `test_performance_regression`**（仅处理 `type: "text"` /
`"video"` 的独立用例）：计算 `baseline_diff_pct = (actual - baseline) / baseline`，
偏差超阈值即 `self.fail()`，与现有逻辑一致。分组子用例不再出现在该方法中。

**组测试方法 `test_group_precision_judgment`（按组参数化）**：`@parameterized.expand`
按加载到的 `GroupPerfRegressionCase` 列表生成，每个组一个独立用例（如
`test_group_precision_judgment_0_glm5_2_group_profiling`），**在单个测试方法内
顺序完成"子用例执行 → 组平均判定"**：

1. 按声明顺序遍历 `sub_cases`，对每个子用例：
   - `torch.compiler.reset()` 清理编译缓存后执行仿真，复用与
     `test_performance_regression` 相同的执行辅助函数；
   - 计算 `baseline_diff_pct`，以带 `grouped` 标记的行写入 `_time_results`；
     组子用例**不出现在 Test 1 单 case 汇总表**，其完整明细由 Test 3 组汇总
     呈现（见 3.6.5）；
   - 随后执行算子级对比（与独立用例相同的 `_compare_operators`，基线从组 JSON
     内嵌的 `sub_cases[key].operators` 读取）：结果 dict 随 `_make_group_member`
     存入子用例明细，**不进入 Test 2 汇总表**；`operators` 为空时记
     `NO_BASELINE`，在 Test 3 的 `Op(TopN)` 列呈现；
   - 子用例偏差超其 `baseline_tolerance` → **仅告警（WARN(INIT)/WARN(BASE)），
     不中断**，继续下一个子用例；
   - 子用例执行抛异常（非判定失败）→ **捕获并记录该子用例为 ERROR，不中断**，
     继续执行剩余子用例；
2. 所有子用例执行完毕后做组判定：
   - **存在 ERROR 子用例 → 该组用例直接判定为失败（`self.fail()`）**，fail 信息
     列出 ERROR 子用例名与异常原因，再附已完成子用例的偏差明细；
   - 无 ERROR 时，`group_avg_diff = mean(各子用例的 |baseline_diff_pct|)`，
     `group_avg_diff > 生效阈值` → 该组用例 `self.fail()`，fail 信息列出组内每个
     子用例的单 case 偏差、平均值与阈值；
3. 生效阈值：`group_tolerance`；缺省时取组内各子用例 `baseline_tolerance` 的
   **最小值**（保守方向）并打 warning；
4. **组 JSON 的 `sub_cases` 为空（组内无任何子用例）**：属于无效配置，加载器输出
   warning 提示，该组用例运行时直接判定为失败，而非静默跳过；
5. `cases/` 目录下不存在任何组用例 JSON（`type: "group"`）时，组判定不产生用例，
   属正常场景（纯 analytic 防护的仓库）。

#### 3.6.4 精度差距的定义

单 case 精度差距统一取**基线对比**偏差的**绝对值**：

```text
diff_i = |(empirical_i - baseline_i) / baseline_i|
```

组平均偏差：`group_avg_diff = (diff_1 + diff_2 + ... + diff_n) / n`。

**设计决策**：采用绝对值平均而非有符号平均。有符号平均会让正负偏差互相抵消
（如 +14% 与 -14% 平均后为 0%），判定偏宽松；绝对值平均更保守，能捕获组内
"整体波动大但方向不一致"的预估质量问题。

#### 3.6.5 报告输出

`tearDownClass` 在现有两份汇总后新增第三份 **[Test 3] 分组平均判定汇总**。汇总
逐组渲染：组行的 Tolerance 为生效组阈值（显式 `group_tolerance`，缺省时回退为
子用例 `baseline_tolerance` 最小值）、AvgDiff 为组内各子用例 BaseDiff 绝对值的
平均、GroupStatus 为组判定结论；其下以子用例明细表逐行列出 Actual（实测总时长）/
Init（初值参考时长）/ InitDiff（Actual 相对 Init 的偏差）/ Baseline（基线时长）/
BaseDiff（Actual 相对 Baseline 的偏差）/ Status，末列为
`Op(TopN)` 算子状态（`PASS`/`WARN(n)`/`NO_BASELINE`，异常子用例为 `N/A`）；
组内存在算子级超差（WARN）时，在明细表下方追加该组的算子 WARN 汇总块：

```text
=======================================================================================================================================
  [Test 3] Group Average Precision Judgment Summary
=======================================================================================================================================
Group: glm5.2-group-profiling   Tolerance:   15.00%   AvgDiff:    8.51%   GroupStatus: PASS
---------------------------------------------------------------------------------------------------------------------------------------
Case                        Actual           Init   InitDiff       Baseline   BaseDiff  Status        Op(TopN)
glm5.2-3.5k-1.5k-decode      42.880ms       42.880ms     +0.00%       42.342ms     +1.27%  PASS          WARN(2)
glm5.2-3.5k-1.5k-prefill    1025.000ms     1025.000ms     +0.00%     1264.000ms    -18.91%  WARN(BASE)    WARN(2)
glm5.2-64k-1k-decode      43.389ms       43.389ms     +0.00%       43.373ms     +0.04%  PASS          NO_BASELINE
glm5.2-64k-1k-prefill    2229.000ms     2229.000ms     +0.00%     2586.000ms    -13.81%  PASS          NO_BASELINE
---------------------------------------------------------------------------------------------------------------------------------------
  Operator warnings in this group:
  Case                     Operator                                       Baseline      Actual        Diff      #Calls
  glm5.2-3.5k-1.5k-decode  tensor_cast.quantize.default                  100.000ms     5.406ms     -94.59%     384/384
  glm5.2-3.5k-1.5k-decode  tensor_cast.all_to_all.default                300.000ms     3.742ms     -98.75%     150/150
  glm5.2-3.5k-1.5k-prefill tensor_cast.all_reduce.default               2000.000ms   115.560ms     -94.22%     157/157
  glm5.2-3.5k-1.5k-prefill tensor_cast.static_quant_linear_int4.default   13.481ms    13.481ms      +0.00%     23/234!
---------------------------------------------------------------------------------------------------------------------------------------
Total: 1 group(s) | Passed: 1 | Failed: 0
=======================================================================================================================================
```

- Test 1 单 case 汇总表**仅覆盖独立用例**：组子用例行带 `grouped` 标记，不进入
  Test 1 表，其完整明细统一由 Test 3 组汇总逐行呈现，避免同一数据在两份汇总中
  口径不一；
- 组子用例的算子结果**同样不进入 Test 2 汇总表**：`_compare_operators` 对
  `from_group=True` 的用例仅返回结果 dict 供 Test 3 渲染，Test 2 保持只覆盖
  独立用例；
- 子用例 `Status` 沿用超阈值降级为 `WARN(INIT)`/`WARN(BASE)`/`WARN(BOTH)` 的
  约定；执行异常的子用例 Actual/InitDiff/BaseDiff 列渲染为 `N/A`（Init/Baseline
  保留子用例配置值）、`Status` 为 `ERROR`；
- `Op(TopN)` 列仅给结论：`PASS`（无超差）、`WARN(n)`（n 条算子级 violation）、
  `NO_BASELINE`（`operators` 为空）、异常子用例为 `N/A`；算子明细仅 WARN 行进入
  组内算子 WARN 汇总块，PASS 行不展开以控制篇幅；汇总块各列为 Case / Operator /
  Baseline / Actual / Diff / #Calls，Diff 为算子耗时相对基线的偏差，#Calls 为
  `基线/实际` 调用次数（后缀 `!` 表示调用数不一致）；算子结果仍为 WARNING-only，
  不参与组 PASS/FAIL 判定；
- `GroupStatus` 取值：`PASS`、`FAIL`（平均超阈值）、`FAIL(ERROR)`（存在异常
  子用例）、`FAIL(NO_BASELINE)`（存在无基线子用例，无法计算组平均）、
  `FAIL(EMPTY)`（`sub_cases` 为空）；Tolerance/AvgDiff 无法计算时显示 `N/A`；
- 组平均判定失败时，fail 消息完整回放组内明细，便于定位是哪几个场景拖高平均值；
- pytest 结果中每组独立展示，如 `test_group_precision_judgment_0_glm5_2_group_profiling PASSED`，
  多组一好一坏时报告直接可分辨，无需到 fail 消息中二次定位。

#### 3.6.6 边界情况

| 场景 | 行为 |
|------|------|
| 组内只有 1 个子用例 | 平均值即自身偏差。仅当生效阈值与该子用例的硬判定阈值一致（`group_tolerance` 缺省回退到该子用例 `baseline_tolerance`）时，数值上才退化为单 case 硬判定；显式设置不同的 `group_tolerance` 时以组阈值为准，且组路径不叠加 analytic 增幅上限，与独立路径可能得出不同结论 |
| 组用例被 `pytest -k` 过滤 | 整组不运行；组内子用例无法单独过滤（已接受的权衡，见 3.6.2） |
| `group_tolerance` 缺省 | 取组内各子用例 `baseline_tolerance` 的**最小值**作为生效阈值（保守方向），并打 warning |
| 组内某子用例执行抛异常（非判定失败） | 捕获并记 ERROR、不中断；所有子用例跑完后，存在 ERROR 即判定该组**失败**，fail 信息列出异常原因 |
| 组 JSON 的 `sub_cases` 为空 | 无效配置：加载时打 warning，该组用例运行时直接判定为失败 |
| 子用例 JSON 字段错误 | 加载组 JSON 时即构造 `TextPerfRegressionCase`，缺失必填字段抛出明确异常，整组用例报错 |
| `cases/` 下无任何组用例 JSON | 组判定不产生用例（正常场景，纯 analytic 防护仓库） |
| 独立用例 JSON（`type: "text"`） | 行为与现状完全一致（单 case 硬判定），不新增任何字段 |

#### 3.6.7 实现要点

| 改动点 | 位置 | 内容 |
|--------|------|------|
| 数据类 | 新增 `GroupPerfRegressionCase` | `name`、`description`、`group_tolerance`、`sub_cases: dict[str, TextPerfRegressionCase]` |
| 数据类 | 新增 `_TimeResultRow` | `_time_results` 行结构的 NamedTuple（`name`/`result_overall`/`actual`/`init_time`/`init_diff`/`base_time`/`base_diff`/`status`/`grouped`），`grouped=True` 标记组子用例：超阈值降级为 WARN，且不进入 Test 1 汇总表 |
| 加载器 | `_load_perf_regression_cases` | 按 `type: "group"` 分发，将 `sub_cases` 各项构造为 `TextPerfRegressionCase`；`sub_cases` 为空时打 warning |
| 执行辅助 | 抽取公共函数 | 从 `test_performance_regression` 中抽出"执行仿真 + 解析总时长"为独立函数，供独立用例与组子用例复用 |
| 算子基线 | `_load_baseline_operators` | 组子用例从组 JSON 的 `sub_cases[key].operators` 读取；独立用例仍按 `{name}.json` 定位 |
| 算子对比 | `_compare_operators` | 改为返回结果 dict（`case_name`/`op_passed`/`op_status`/`violations`/`detail_rows`）；独立用例照旧写入 Test 2 汇总列表，`from_group=True` 的组子用例只返回结果不写入（其算子明细由 Test 3 呈现） |
| 组判定 | `test_group_precision_judgment`（按组参数化，委托 `_run_group_case`） | 方法内顺序执行子用例（超阈值仅 WARN、异常捕获记 ERROR 且不中断），每个子用例经 `_make_group_member` 生成明细 dict（`case`/`actual`/`init`/`init_diff`/`baseline`/`base_diff`/`status`，另附 `op_status`/`op_warn_count`/`op_rows`）；随后：存在 ERROR 或 `sub_cases` 为空 → 直接 fail；否则按绝对值平均独立 fail |
| 报告 | `_print_time_summary` / `_print_group_summary` | `_print_time_summary` 过滤 `grouped` 行，Test 1 仅覆盖独立用例；`_print_group_summary` 逐组渲染组行（Tolerance/AvgDiff/GroupStatus）与子用例明细表（Test 3，含 `Op(TopN)` 算子状态列），异常子用例 Actual/InitDiff/BaseDiff 渲染为 N/A（Init/Baseline 保留配置值）；组内存在算子 WARN 时追加组级算子 WARN 汇总块 |
| 单元测试 | `tests/benchmark/models/` | 组平均计算、子用例 ERROR 触发组 FAIL、`sub_cases` 为空触发组 FAIL、组子用例 WARN 降级与明细 dict 字段（含 op 字段）、Test 1 隐藏 grouped 行、Test 3 组汇总渲染（含 Op 列与算子 WARN 块） |

不改动的时间判定部分：`_is_baseline_time_acceptable`、独立用例的 Test 2 算子对比
行为与基线生命周期均保持不变；仅组子用例的算子结果从 Test 2 迁移至 Test 3 呈现。
`auto_baseline.py` 仅做参数
重命名（`_parse_total_time_s` 的 `model_name` → `performance_model_name`，与
3.6.7 节命名保持一致），行为不变。

### 3.7 用例命名约定（profiling 后缀）

为在文件层面直观区分 empirical 时长防护与 analytic 时长防护，引入命名约定：

- **使用 profiling 模式仿真建模的用例**（`user_input.performance_model` 含
  `"profiling"`）：文件名（及 JSON `name` 字段）在原功能名后追加 `-profiling`
  后缀，如 `qwen3-32b-4k-1.5k-prefill-profiling.json`，且 JSON 的 `name` 字段必须
  与文件名（不含扩展名）保持一致——算子基线定位依赖 `{name}.json` 约定；
- **仅使用 analytic 模型的用例**：保持原命名，不加后缀；
- **组用例**：同样遵循该约定，如 `glm5.2-group-profiling.json`（组内子用例均为
  profiling 模式）；
- **统一后缀**：独立用例与组用例的后缀统一为 `-profiling`（连字符），保证命名
  规则单一、可被统一的 glob（`*-profiling`）筛选；
- **仅作约定，不做校验**：后缀与 `performance_model` 的一致性、JSON `name` 与
  文件名的一致性均靠评审和人工维护，框架不做加载时校验；`name` 与文件名不一致
  导致的实际影响（独立用例按 `{name}.json` 定位算子基线时静默 miss）由运行期
  的 NO_BASELINE 报告兜底暴露；
- 框架解析表格指标仍以 `_resolve_table_performance_model_name` 的自动推断为准，
  后缀仅为命名约定，不参与判定逻辑；
- **不叠加增幅上限**（见 3.1.5 节检测一）：empirical 基线来自实测而非理论最优
  下界，actual 超过基线并不代表仿真失真，因此仅按 `baseline_tolerance` 做对称判定。

该约定的收益：无需打开 JSON 即可从文件名区分防护模式，便于 CI 按模式筛选用例，
也避免 profiling 用例与 analytic 用例在基线数值上的混淆。

---

## 4. 测试设计

### 4.1 单元测试

| 测试项 | 测试内容 | 验证点 |
|--------|---------|--------|
| `_parse_total_time_s` 单位解析 | 输入 `"Total time for analytic: 321.540ms"` | 返回 `0.32154` |
| `_parse_total_time_s` 秒单位 | 输入 `"Total time for analytic: 1.744s"` | 返回 `1.744` |
| `_parse_total_time_s` 微秒单位 | 输入 `"Total time for analytic: 500.000us"` | 返回 `0.0005` |
| `_parse_total_time_s` 纳秒单位 | 输入 `"Total time for analytic: 100.000ns"` | 返回 `1e-7` |
| `_parse_total_time_s` 未找到 | 输入不包含目标字符串的文本 | 抛出 `ValueError` |
| `_parse_top_operators` 正常解析 | 输入算子耗时表 | 返回 Top-N 算子列表 |
| 空用例列表 | `cases/` 目录为空 | pytest 自动 skip，不报错 |
| JSON 配置加载 | 读取文本和视频两类 JSON | 分别构造正确的子类实例 |
| 相对路径解析 | 视频 JSON 中 `model_id` 为相对路径 | 自动解析为绝对路径 |

### 4.2 集成测试

| 测试项 | 测试内容 | 验证点 |
|--------|---------|--------|
| Smoke 用例 | 使用小模型跑通完整流程 | 框架各环节正常协作 |
| PASS 场景 | 基线值设为远大于实际值 | 状态为 PASS |
| FAIL 场景 | 基线值设为远小于实际值 | 状态为 FAIL，assertTrue 触发 |
| NO_BASELINE 场景 | 用例 JSON 中 `operators` 字段为空 | 状态为 NO_BASELINE，仅记 warning（Test 2 计入 No Baseline、Test 3 的 `Op(TopN)` 列呈现），不参与 PASS/FAIL 判定 |
| 算子 violations 持久化 | 算子检测失败时 | `_op_results` 包含完整 violations 列表 |

### 4.3 分组平均判定测试（3.6 节机制）

单元测试（不执行仿真，针对加载与判定辅助逻辑）：

| 测试项 | 测试内容 | 验证点 |
|--------|---------|--------|
| 组用例 JSON 加载 | `type: "group"` JSON，`sub_cases` 内嵌子用例 | 构造 `GroupPerfRegressionCase`，各子用例为 `TextPerfRegressionCase`，字段与 JSON 一致 |
| 空 `sub_cases` 加载 | 组 JSON 的 `sub_cases` 为 `{}` | 加载时输出 warning，组用例保留，运行时直接 FAIL |
| 组平均计算口径 | 子用例偏差 `+14%、-14%、+5%、-5%` | 平均值取绝对值平均（9.5%），正负不抵消 |
| 生效阈值回退 | `group_tolerance` 缺省且子用例 `baseline_tolerance` 不一致 | 取最小值作为生效阈值并打 warning |
| 平均恰等于阈值 | 组平均 = 生效阈值 | PASS（严格大于才 FAIL） |
| 组子用例 WARN 降级 | 子用例偏差超单 case 阈值（如 +14% 超文本 5% 上限）但走组路径 | 状态降级为 `WARN(BASE)`，成员明细 dict 字段齐全（`case`/`actual`/`init`/`init_diff`/`baseline`/`base_diff`/`status` 及 op 三字段），组判定不受单 case 超阈值影响 |
| Test 1 汇总过滤 grouped 行 | `_time_results` 仅含 `grouped=True` 行 | `_print_time_summary` 输出为空（组子用例由 Test 3 呈现） |
| Test 3 组汇总渲染 | 含 PASS/FAIL 组与 WARN(BASE)/ERROR 子用例的 `_group_rows` | 组行含 Tolerance/AvgDiff/GroupStatus；子用例明细行含数值、状态与 `Op(TopN)` 列，ERROR 子用例 Actual/InitDiff/BaseDiff 渲染为 N/A（Init/Baseline 保留配置值）；组内算子 WARN 渲染为组级算子 WARN 汇总块（仅 WARN 行）；footer 按组计数（`Total: N group(s)`） |

集成测试（pytest 运行级，可配合基线远大于/远小于实际值的桩用例）：

| 测试项 | 测试内容 | 验证点 |
|--------|---------|--------|
| 组 PASS 场景 | 子用例偏差均低于组阈值，其中部分超单 case 阈值 | 子用例行状态为 `WARN(BASE)`，组用例 PASSED，Test 3 汇总 PASS |
| 组 FAIL 场景 | 组平均偏差超过 `group_tolerance` | 该组用例 FAILED，fail 信息列出各子用例偏差、平均值与阈值；独立用例不受影响 |
| 子用例异常 → 组 FAIL | 某子用例执行抛异常（如仿真配置错误） | 异常被捕获不中断，其余子用例继续执行，组用例 FAILED 且信息含异常原因 |
| 执行顺序 | 组用例运行日志 | 子用例按声明顺序全部执行完毕后才输出组判定结论 |
| 算子基线内嵌读取 | 组子用例 `operators` 非空 | 算子对比从组 JSON 内嵌基线读取，不回退独立 `{name}.json` 定位 |
| 无组用例目录 | `cases/` 下无 `type: "group"` JSON | 不产生组判定用例，纯 analytic 场景行为与旧版一致 |

#### 运行命令

```bash
# 运行全部组用例（Test 3，按方法名过滤，不含独立用例）
uv run pytest tests/benchmark/models/test_model_regression.py -k "group" -v

# 运行指定组用例（按用例名过滤，即组 JSON 的 name 字段）
uv run pytest tests/benchmark/models/test_model_regression.py -k "glm5.2-group-profiling" -v

# 实时查看 WARNING 及以上日志（组容差回退提示、子用例异常堆栈等）
uv run pytest tests/benchmark/models/test_model_regression.py -k "group" -v -s -o log_cli=true --log-cli-level=WARNING
```

参数说明：

| 参数 | 说明 |
|------|------|
| `-k "group"` | pytest 关键字过滤；组判定方法名为 `test_group_precision_judgment`，含 `group` 即可只选中组用例 |
| `-k "<组用例名>"` | 精确运行某一组用例，用例名为组 JSON 的 `name`（如 `glm5.2-group-profiling`） |
| `-v` | 显示每个组用例的独立 PASS/FAIL 结果 |
| `-s` | 关闭 pytest 输出捕获，运行结束后可见 Test 1 / Test 2 / Test 3 汇总表（Test 3 为组判定汇总，含容差、平均偏差、各子用例偏差与算子状态） |
| `-o log_cli=true --log-cli-level=WARNING` | 实时输出运行日志；排查子用例异常时改为 `INFO` 可看到更多过程信息 |

注意：组用例集成测试会执行真实仿真，需要模型与运行环境就绪；本地快速验证加载逻辑可参考 4.1 节单元测试（不执行仿真）。

---

## 5. 缺点和风险

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| 基线值依赖环境 | 不同机器/GPU 基线值不同 | 文档明确说明基线采集环境要求 |
| 模型下载失败 | 首次运行需下载模型 | 使用已缓存模型 |
| 用例过多导致耗时过长 | CI 流水线超时 | 按优先级分组，pytest -k 按名称过滤 |
| 容忍度设置不当 | 可能过松或过严 | 支持 `initial_tolerance`、`baseline_tolerance`、`operator_tolerance` 分别配置 |
| JSON 配置编写错误 | 字段名拼写错误导致加载失败 | `_load_perf_regression_cases` 的 `**data` 解包会在缺失必填字段时抛出明确异常 |
| Python dataclass 继承限制 | 子类非默认字段需在基类默认字段之前 | 视频子类所有字段均提供默认值，文本子类 `user_input` 使用 Optional |

---

## 6. 现有技术

| 项目/工具 | 做法 | 借鉴与差异 |
|----------|------|-----------|
| pytest-benchmark | 自动校准、统计中位数/标准差 | 本框架更侧重与固定基线对比，且支持算子级细粒度对比 |
| Google Benchmark | C++ 微基准测试，支持统计 | 理念相似，本框架面向 LLM/视频推理全链路 |
| MLPerf | 标准化 AI 性能基准 | 本框架面向开发阶段快速回归，非标准化评测 |

---

## 附录

### 参考资料

- [pytest 官方文档](https://docs.pytest.org/)
- [parameterized 库](https://github.com/wolever/parameterized)
- 项目内参考：`tests/st/README.md`、`tests/st/auto_baseline.py`

### 术语表

| 术语 | 说明 |
|------|------|
| 基线值 (Baseline) | 在稳定环境中运行得到的参考耗时，作为性能对比基准 |
| 容忍度 (Tolerance) | 允许的性能波动范围，可分别配置总耗时和算子级 |
| 性能回归 (Performance Regression) | 代码变更导致推理耗时增加超过容忍度 |
| Total time for analytic | 分析性能模型计算出的总推理时间 |
| 算子基线 (Operator Baseline) | 用例 JSON 中 `operators` 字段记录的历史算子耗时和调用次数 |
| 目录化配置 | 通过 `cases/*.json` 文件声明用例，与测试代码分离 |

### 文档更新计划

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2026-05-19 | 初始版本 | @yuyinkai |
| v1.1 | 2026-05-19 | 重构为目录化 JSON 配置、数据类继承层次、双检测体系、算子级对比、基线显式管理 | @yuyinkai |
| v1.2 | 2026-09-22 | 新增 3.6 节：分组平均精度判定机制（组用例单 JSON 内嵌 `sub_cases`，按组参数化执行；组内单 case 超阈值仅告警，组内 ERROR 或空 `sub_cases` 直接判失败，否则按绝对值平均与组容差判定）；新增 3.7 节：profiling 用例 `-profiling` 命名后缀约定（纯约定不做框架校验，实际影响由运行期 NO_BASELINE 报告兜底），并补充判定差异：profiling 用例基线判定取消增幅上限，仅按 `abs(diff) <= baseline_tolerance` 对称判定（empirical 基线来自实测，analytic 保留 `diff <= max_increase_pct` 双重约束）；`_is_baseline_time_acceptable` 新增 `performance_model` 入参（默认 `"analytic"`），调用处归一化后传入；profiling 分支不再计算增幅上限；字段表补充 `baseline_max_increase_pct` | @zhenghaojie |

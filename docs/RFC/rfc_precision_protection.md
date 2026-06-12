# 性能回归测试框架设计文档

状态 (Status): Draft
作者 (Authors): @yuyinkai
创建日期 (Created): 2026-05-19
更新日期 (Updated): 2026-05-19

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
| `initial_tolerance` | `float` | `0.10` | 与初版的容忍度（10%） |
| `baseline_tolerance` | `float` | `0.20` | 与基线的容忍度（20%） |
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
3. 从 `table_result` 中正则提取 `Total time for analytic`，得到 `actual_time_s`
4. **检测一 — 总耗时对比**：
   - 如果 `initial_time_s > 0`：计算 `(actual - initial) / initial`，与 `initial_tolerance` 对比
   - 如果 `baseline_time_s > 0`：计算 `(actual - baseline) / baseline`，与 `baseline_tolerance` 对比
   - 两项中任一未通过则记录为 FAIL
5. **检测二 — 算子级对比**：
   - 从 `table_result` 中解析 Top-N 算子（`_parse_top_operators`）
   - 从用例 JSON 的 `operators` 字段加载算子基线（`_load_baseline_operators`）
   - 如果 `operators` 字段为空：直接 `self.fail()` 报错，不自动生成（防止测试污染仓库）
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
| `_parse_total_time_s(table_result, model_name)` | 正则提取 `Total time for {model_name}` 并转换为秒 |
| `_parse_top_operators(table_result, top_n, model_name)` | 解析算子耗时表，返回前 N 个算子的 (名称, 耗时, 调用次数) |
| `_load_baseline_operators(case_name)` | 从用例 JSON 的 `operators` 字段加载算子基线 |
| `_save_baseline_operators(case_name, operators)` | 将算子基线写入用例 JSON 的 `operators` 字段（供基线生成工具使用） |
| `_load_perf_regression_cases()` | 从 `cases/` 目录加载所有 JSON 配置文件 |
| `_print_time_summary(results, report_file)` | 输出检测一汇总报告 |
| `_print_operator_summary(op_results, op_detail_rows, report_file)` | 输出检测二汇总报告 |

---

### 3.5 auto_baseline.py 自动基线运行器

独立的 pytest 入口，用于自动跑两次对比：第一次建立基线，第二次对比，容忍度默认 5%。与 regression.py 不同，auto_baseline 聚焦于同一场景下两次运行的一致性验证，适合快速自测。

**关键保护**：

- 基线运行失败和对比运行失败分别捕获，返回结构化错误结果
- `baseline_time_s <= 0` 时有 ZeroDivisionError 守卫，返回明确错误信息而非算术异常

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
| NO_BASELINE 场景 | 用例 JSON 中 `operators` 字段为空 | 状态为 NO_BASELINE，框架直接 fail |
| 算子 violations 持久化 | 算子检测失败时 | `_op_results` 包含完整 violations 列表 |

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

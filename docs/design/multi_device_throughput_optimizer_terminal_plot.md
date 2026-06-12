# 特性设计：吞吐寻优多硬件展示与终端 ASCII Plot

## 修订记录

| 日期 | 修订版本 | 修改描述 | 作者 | RFC 文档 |
| -- | -- | -- | -- | -- |
| 2026-05-09 | 1.0 | 初稿：多 `--device` 对比、`plotext` 曲线、拆解/PD 比例路径说明 | — | 本文档 |
| 2026-05-19 | 1.1 | 同步当前实现：PD配比仅输出 TPS 双图；多 device 仅输出跨硬件汇总表 | — | 本文档 |
| 2026-05-19 | 1.2 | 术语统一：聚合→PD混部，拆解→PD分离，PD 比例→PD配比 | — | 本文档 |

---

## 功能描述

### 背景与问题

`throughput_optimizer` 用于在给定输入长度、输出长度和 SLO 约束下，搜索最优并行和并发配置。实际选型时通常同时存在两类诉求：

1. 横向比较多种 `DeviceProfile`，判断在相同约束下哪种硬件更优。
2. 纵向分析单一硬件，观察吞吐、QPS、并发和时延之间的关系。

当前建模寻优结果仅输出不同场景下的性能列表，整体结果不直观，不能同时对比多个硬件的性能。

### 目标

1. 支持 `--device DEVICE [DEVICE ...]` 一次传入多个硬件，在一次命令内完成逐设备寻优与跨硬件汇总，汇总输出硬件最优性能的配置对比。
2. 单设备时自动输出终端 ASCII 曲线，覆盖 PD混部、PD分离和 PD配比三类模式；每种模式输出 `Concurrency & TPS`、`TPOT & TPS` 两张图。
3. 将展示职责集中到两个 service 模块：
   - `optimizer_summary.py`：负责结果过滤、最佳配置提取、PrettyTable 输出。
   - `optimizer_curve_plots.py`：负责终端曲线绘制，以及多设备执行编排和跨硬件汇总调度。

---

## 当前实现概览

### 模块职责

| 模块 | 当前职责 |
|------|------|
| `cli/inference/throughput_optimizer.py` | 参数解析、模式校验、设备校验、调用执行入口 |
| `cli/utils.py` | `check_device_targets()` device参数校验 |
| `serving_cast/service/optimizer_curve_plots.py` | 终端 ASCII 曲线、单/多设备执行编排、跨硬件汇总调度 |
| `serving_cast/service/optimizer_summary.py` | 单设备结果过滤与打印、跨硬件表格渲染、PD分离/PD配比结果整理 |

### 当前调用链

```text
throughput_optimizer.main
  -> check_device_targets(args, logger)
  -> plot_curves_allowed = len(device_targets) == 1
  -> run_multi_device_loop(args, device_targets, plot_curves_allowed, logger)
       -> 单 device：每个 profile 完成后绘制 sweep 曲线
  -> render_cross_hardware_summary(args, device_targets, hw_rows, logger)
       -> 多 device：输出硬件画像表与跨硬件汇总表
```

说明：

- `run_multi_device_loop()` 与 `render_cross_hardware_summary()` 位于 `optimizer_curve_plots.py`。

---

## 详细设计

### 1. CLI 入口层

`cli/inference/throughput_optimizer.py` 当前负责：

1. 解析参数。
2. 调用 `check_device_targets(args, logger)` 统一校验 `--device`。
3. 根据设备数量计算 `plot_curves_allowed = len(device_targets) == 1`。
4. 调用 `run_multi_device_loop()` 执行寻优和结果采集。
5. 调用 `render_cross_hardware_summary()` 输出跨硬件摘要。

### 2. 设备校验

`--device` 由 `cli/utils.py` 中的 `check_device_targets()` 统一校验，职责包括：

1. 处理未传 `--device` 时的默认画像选择。
2. 去重并保持输入顺序。
3. 拒绝空白、未知设备名。
4. 校验设备画像通信网格与 `--num-devices` 的适配关系。

这样 CLI 主流程聚焦于参数解析和执行编排，设备校验细节由通用工具函数负责。

### 3. 多设备执行编排

`optimizer_curve_plots.py` 中的 `run_multi_device_loop()` 负责多设备执行编排。

执行流程如下：

1. 遍历 `device_targets`。
2. 对每个 `profile_name` 暂时写回 `args.device`。
3. 创建 `ParallelRunner(args)`。
4. 根据模式选择：
   - PD混部模式：`run_agg()`
   - PD分离模式或 PD配比模式：`run_disagg()`
5. 遍历本次结果列表 `results`：
   - 调用 `res.report_final_result(args, silent=False)` 输出单设备结果。
   - 若为多硬件模式，则从结果中抽取一行跨硬件对比摘要。
6. 若 `plot_curves_allowed` 为真，则在当前设备执行完成后绘制终端曲线。

### 4. 跨硬件摘要采集

跨硬件摘要依赖 `OptimizerSummary` 中的行提取方法：

| 模式 | 行提取方法 | 排序指标 |
|------|------|------|
| PD混部 | `collect_comparison_row()` | `throughput_tps` |
| PD分离 Prefill | `collect_disagg_prefill_row()` | `throughput_tps` |
| PD分离 Decode | `collect_disagg_decode_row()` | `throughput_tps` |
| PD配比 | `collect_pd_ratio_comparison_row()` | `balanced_qps` |

采集结果统一存入 `MultiDeviceComparisonRows`：

```text
aggregation
pd_ratio
disagg_prefill
disagg_decode
```

### 5. 跨硬件摘要输出

`render_cross_hardware_summary()` 的当前行为：

1. 当 `len(device_targets) <= 1` 时直接返回，不打印跨硬件表。
2. 先调用 `render_hardware_profile_comparison(device_targets)` 输出设备画像参数表。
3. 再按模式输出对应汇总表：
   - PD混部：`render_cross_device_comparison()`
   - PD分离：`render_cross_hardware_disagg_prefill()`、`render_cross_hardware_disagg_decode()`
   - PD配比：`render_cross_hardware_pd_ratio()`

当对应模式下没有有效汇总行时，仅记录 warning，不中断主流程。

---

## 终端 ASCII 曲线设计

### 1. 启用条件

终端曲线仅在单设备 sweep 场景启用：

#### 单设备 sweep 曲线

```text
plot_curves_allowed = len(device_targets) == 1
```

- 单设备：输出完整 sweep 曲线
- 每个 `parallel` 一组散点，展示该并行配置下不同 concurrency / 时延点

- 多设备：不输出终端曲线，只输出硬件画像表与跨硬件汇总表

### 2. 绘图入口

#### 单设备

`optimizer_curve_plots.py` 中由 `_plot_single_device_optimizer_curves()` 进行模式分发：

| 模式 | 入口 |
|------|------|
| PD混部 | `plot_concurrency_curves_from_optimizer_summaries()` |
| PD分离 | `plot_disagg_terminal_curves()` |
| PD配比 | `plot_pd_ratio_terminal_curves()` |

### 3. 曲线内容

#### PD混部模式

从多个 `OptimizerSummary.get_summary_df()` 合并结果后，绘制两张图：

1. `Throughput (token/s) vs Concurrency`
2. `Throughput (token/s) vs TPOT (ms)`

#### PD分离模式

按 `OptimizerSummary.data_config` 区分两类结果：

- Prefill：`ttft_limits is not None and tpot_limits is None`
- Decode：`tpot_limits is not None and ttft_limits is None`

分别绘制：

- Prefill：
  1. `Throughput (token/s) vs Concurrency`
  2. `Throughput (token/s) vs TTFT (ms)`
- Decode：
  1. `Throughput (token/s) vs Concurrency`
  2. `Throughput (token/s) vs TPOT (ms)`

#### PD配比模式

PD 单设备曲线不再分别绘制 Prefill-side / Decode-side 的 QPS 四张图，而是统一输出 Decode 侧 TPS 双图：

1. `Throughput (token/s) vs Concurrency`
2. `Throughput (token/s) vs TPOT (ms)`

数据来源与计算方式：

- 从 PD ratio summary DataFrame 中取 Decode 侧列：
  - `parallel_d`
  - `concurrency_d`
  - `tpot_d`
- TPS 计算公式：

```text
token/s = concurrency_d / tpot_d * 1000
```

说明：

- 表格输出仍保留 `p_qps`、`d_qps`、`balanced_qps` 等 PD 指标
- 终端 plot 仅展示 TPS 与 Concurrency / TPOT 的关系，不再输出 P/D QPS 曲线

### 4. 当前过滤规则

这里需要明确区分“表格过滤”和“曲线过滤”。

#### 表格/最佳配置过滤

`optimizer_summary.py` 中：

- PD混部/PD分离使用 TTFT/TPOT SLA 过滤后选最优。
- PD配比使用 `ttft_p` / `tpot_d` SLA 过滤后，再按 `balanced_qps` 去重和排序。

#### 曲线过滤

`optimizer_curve_plots.py` 中当前只保留两类处理：

1. 校验必需列是否存在。
2. 过滤 OOM 或显存不足点，即：
   - `memory_left_gb <= 0`
   - 或 `device_memory_available_gb <= 0`

当前实现**不再用 TTFT/TPOT SLA 过滤曲线点**。保留 `ttft_limit` / `tpot_limit` 参数主要是为了兼容外部调用接口，而不是实际参与曲线筛选。

这样做的目的，是在终端图里尽量保留完整 sweep 结果，只剔除明显无效的 OOM 点。

### 5. 绘图细节

当前绘图实现的几个关键点：

1. 使用 `plotext` 的模块级共享画布。
2. 单设备 sweep 图：每个 `parallel` 配置绘制一组散点，marker 为实心圆点 `●`。
3. 坐标完全重合的点会轻微错开，保证一个数据点对应一个可见点。
4. 坐标轴会根据当前图内所有点自动留白，避免点贴边。
5. 图尺寸由内部常量控制：
   - `_TERMINAL_PLOT_COLS = 128`
   - `_TERMINAL_PLOT_ROWS = 38`
6. 不同 `parallel` 使用轮转色板区分。

由于 `plotext` 是模块级状态式 API，当前假设调用过程为串行单线程；这与 CLI 当前的顺序执行方式一致。

---

## `optimizer_summary.py` 的当前角色

### 1. 单设备结果输出

`OptimizerSummary.report_final_result()` 按模式输出最终结果：

| 模式 | 输出逻辑 |
|------|------|
| PD混部/PD分离 | `_get_agg_disagg_final_out()` |
| PD配比 | `_get_pd_ratio_final_out()` |
| `--dump-original-results` | 打印原始或过滤后的 DataFrame |

### 2. PD混部/PD分离最佳点选择

`_prepare_agg_disagg_results()` 当前逻辑：

1. 使用 TTFT/TPOT 限制做过滤。
2. 以 `token/s` 降序排序。
3. 对每个 `parallel` 仅保留一条最佳记录。
4. 再按 `token/s` 总排序。

### 3. PD分离 QPS 计算

`_compute_disagg_request_qps()` 用于 PD分离汇总表中的 `QPS (req/s)` 字段：

- Prefill：`concurrency / ttft * 1000`
- Decode：`concurrency / (tpot * output_length) * 1000`

### 4. PD配比结果整理

`_prepare_pd_ratio_results()` 当前逻辑：

1. 按 `ttft_p` / `tpot_d` 做 SLA 过滤。
2. 对每个 `(parallel_p, parallel_d)` 组合仅保留 `balanced_qps` 最优项。
3. 按四舍五入到 2 位小数后的 `balanced_qps` 再去重。
4. 最终按 `balanced_qps` 降序返回。

### 5. 跨硬件表格渲染

当前由 `optimizer_summary.py` 统一输出：

- `render_hardware_profile_comparison()`
- `render_cross_device_comparison()`
- `render_cross_hardware_disagg_prefill()`
- `render_cross_hardware_disagg_decode()`
- `render_cross_hardware_pd_ratio()`

因此，`optimizer_curve_plots.py` 与 `optimizer_summary.py` 之间现在是**直接依赖关系**，而不是纯粹通过 DataFrame 间接配合。

---

## 接口设计

### CLI 参数

本特性对外接口为：

```bash
--device DEVICE [DEVICE ...]
```

### 取值规则

`--device` 的入参形式为一个或多个设备名：

```bash
--device AtlasA2
--device AtlasA2 Atlas800I
```

处理规则如下：

1. **不传值**
   - 即命令中不出现 `--device`
   - `args.device` 为 `None`
   - 由 `check_device_targets()` 补全为 `["TEST_DEVICE"]`，与原有单设备默认行为保持一致

2. **传入空值**
   - 由于参数定义为 `nargs="+"`，显式写出 `--device` 时必须至少跟一个值
   - 若没有值，CLI 在参数解析阶段直接报错退出
   - 若值为空白字符串，也会在校验阶段判定为非法并退出

3. **传入有效值**
   - 每个值都必须是已注册的 `DeviceProfile` 名称
   - 支持单值或多值输入
   - 重复值会去重并保留首次出现顺序
   - 规范化后得到 `device_targets: list[str]`

---

## DFX 说明

### 安全性

当前实现没有新增网络接口，也没有新增文件落盘行为。特性输出仅包括：

- stdout 表格
- stdout ASCII 曲线
- warning / exception 日志

### 可靠性

1. 曲线绘制路径对 `ImportError` 和绘图异常都有降级处理。
2. 缺列或过滤后为空时，曲线入口返回 `False` 并记录 warning，不中断寻优主流程。
3. Prefill 曲线直接使用 `ttft` 列，不再通过将 `ttft` 临时改名为 `tpot` 的方式复用逻辑，避免重复列名问题。

### 性能与可用性

1. 多设备模式按顺序执行，总耗时与设备数近似线性相关。
2. 多设备模式输出跨硬件汇总表，不输出终端曲线。
3. 单设备 sweep 图以 ASCII 终端可读性为目标，不追求高保真制图能力。

---

## 测试现状与建议

### 当前已有测试

当前仓库中可以确认的相关单测主要在：

- `serving_cast/tests/ut/test_service/test_optimizer_summary.py`

覆盖内容包括：

1. `OptimizerSummary` 初始化与 `summary_df` 读写。
2. early stop flag 逻辑。
3. PD混部结果输出基础路径。
4. PD配比模式判定、去重、实例分配和最终输出结构。

### 当前缺口

目前未看到独立的 `test_optimizer_curve_plots.py`。因此以下部分仍主要依赖代码阅读和手工验证：

1. 终端曲线入口分发。
2. OOM 过滤与缺列降级路径。
3. 单设备/多设备分支切换。
4. `run_multi_device_loop()` 与 `render_cross_hardware_summary()` 的协同行为。

### 建议测试方向

1. 新增 `optimizer_curve_plots` 相关 UT，覆盖：
   - `_prepare_curve_df()`
   - `_prepare_disagg_prefill_curve_df()`
   - `_pd_tps_curve_df()`
   - `plot_*_terminal_curves()` 的空数据与缺列分支
2. 对 `run_multi_device_loop()` 做 mock runner 测试，验证多设备行采集逻辑。
3. 手工验收以下场景：
   - 单 `--device` PD混部模式：有 sweep ASCII 图
   - 单 `--device --disagg`：Prefill/Decode sweep 曲线正常
   - 单 `--device` PD配比模式：仅输出 TPS vs Concurrency / TPOT 双图
   - 多 `--device`：输出硬件画像表与跨硬件汇总表，不输出 scatter 图

---

## 限制与后续演进

### 当前限制

1. 曲线仅输出到终端，不生成文件。
2. 多设备模式不输出终端曲线，只输出跨硬件表格。
3. 曲线 API 仍保留 `ttft_limit` / `tpot_limit` 参数，但当前实现不使用它们参与曲线筛选。
4. 终端图依赖 `plotext` 的共享画布模型，不适合并发交错调用。

### 后续可选方向

1. 增加 `--no-terminal-plots` 开关，允许单设备下显式关闭曲线。
2. 增加文件型后端，如 PNG / SVG / HTML 输出。
3. 为跨硬件汇总增加 CSV / JSON artifact 输出。
4. 为曲线与多设备编排补齐系统化单元测试。

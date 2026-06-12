# RFC: PD配比吞吐量寻优

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | 已批准 |
| **作者** | jiayanan |
| **创建日期** | 2026-03-19 |
| **相关链接** |  |

---

## 1. 概述

本提案旨在实现PD配比寻优功能：P和D**独立运行寻优**，分别找到各自最优的batch size和并行配置后，计算各自的QPS，再根据QPS均衡原则自动推导出最优PD配比（即P实例数与D实例数的比例），最终输出**Top N的PD配比结果**（每条结果包含对应的P配置和D配置）。

**QPS计算公式：**

```bash
P QPS = p_concurrency / ttft * 1000 (req/s)
D QPS = d_concurrency / (tpot * output_length) * 1000 (req/s)
```

**PD配比计算：** PD ratio = D_QPS / P_QPS。例如 P QPS=10 req/s，D QPS=15 req/s，则 PD ratio=1.5，即每1个D实例需要1.5个P实例来实现供需均衡。

---

## 2. 详细设计

### 2.1 CLI入口扩展

在 `cli/inference/throughput_optimizer.py` 的参数解析中新增以下参数：

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `--prefill-devices-per-instance` | int | 单个Prefill实例使用的设备数 |
| `--decode-devices-per-instance` | int | 单个Decode实例使用的设备数 |
| `--enable-optimize-prefill-decode-ratio` | flag | 开启PD配比寻优模式 |

**参数校验规则：**

- `--enable-optimize-prefill-decode-ratio` 开启时，`--prefill-devices-per-instance` 和 `--decode-devices-per-instance` 为必填
- P/D设备数必须为正整数

**使用示例：**

```bash
python -m cli.inference.throughput_optimizer \
    --input-length=2048 --output-length=512 \
    model_id \
    --device=DEVICE --num-devices=16 \
    --prefill-devices-per-instance=4 --decode-devices-per-instance=2 \
    --enable-optimize-prefill-decode-ratio \
    --ttft-limits=500 --tpot-limits=50
```

### 2.2 ParallelRunner扩展

在 `serving_cast/parallel_runner.py` 的 `ParallelRunner` 中新增 `run_pd_ratio()` 方法。

**核心设计：P和D独立寻优，最后组合计算PD配比。**

与现有 `run_agg()`/`run_disagg()` 不同，PD配比模式下P和D的并行配置可能不同（如P使用TP=4，D使用TP=2），因此需要分别创建各自的 `UserInputConfig` 和 `ModelRunner` 实例。P和D可以**分开独立运行**，各自遍历TP配置空间完成寻优，最后将所有P结果和D结果进行组合，计算PD配比。

```bash
run_pd_ratio():
    1. 独立运行P寻优
       - num_devices = p_devices_per_instance
       - 遍历可行的TP配置，对每个TP找最优batch size
       - 收集所有P结果列表: [(p_parallel, p_batch, ttft, p_qps), ...]
    2. 独立运行D寻优
       - num_devices = d_devices_per_instance
       - 遍历可行的TP配置，对每个TP找最优batch size
       - 收集所有D结果列表: [(d_parallel, d_batch, tpot, d_qps), ...]
    3. 组合P和D结果，计算PD配比
       - 对每对 (P_result, D_result) 计算 pd_ratio = D_QPS / P_QPS
       - 按 min(P_QPS, D_QPS) 即均衡QPS降序排序
       - 输出 Top N 配比结果
```

### 2.3 PDRatioThroughputOptimizer

新增 `serving_cast/service/pd_ratio_throughput_optimizer.py`，实现 `PDRatioThroughputOptimizer` 类。

**类设计：**

`PDRatioThroughputOptimizer` 是一个**组合器**，**不继承** `BaseThroughputOptimizer`。它接收已有的P和D优化结果（作为DataFrame），组合计算PD配比。

```python
class PDRatioThroughputOptimizer:
    """PD配比吞吐量优化的组合器。

    与继承 BaseThroughputOptimizer 的 Agg/Disagg 优化器不同，
    此类组合已有的P和D优化结果来计算PD配比。
    它不执行优化本身，而是作为纯组合器。
    """

    def __init__(self, output_length: int):
        self.output_length = output_length
        self._p_df: pd.DataFrame = None
        self._d_df: pd.DataFrame = None
        self._result_df: pd.DataFrame = None

    def set_p_results(self, df: pd.DataFrame):
        """设置prefill优化结果。"""
        self._p_df = df

    def set_d_results(self, df: pd.DataFrame):
        """设置decode优化结果。"""
        self._d_df = df

    def optimize(self) -> pd.DataFrame:
        """组合P和D结果并计算PD配比。

        返回包含所有PD配比组合的DataFrame。
        """
```

**关键设计决策：**

1. 使用DataFrame作为输入/输出接口，与 `parallel_runner` 保持一致
2. **不继承** `BaseThroughputOptimizer` - 它是处理现有结果的组合器

```python
def optimize(self) -> pd.DataFrame:
    # 过滤零值避免ZeroDivisionError
    p_df = self._p_df[self._p_df["ttft"] > 0]
    d_df = self._d_df[self._d_df["tpot"] > 0]

    # 带后缀的cross join
    merged = p_df.merge(d_df, how="cross", suffixes=("_p", "_d"))
    # 计算QPS和PD配比
    merged["p_qps"] = merged["concurrency_p"] / merged["ttft"] * 1000
    merged["d_qps"] = merged["concurrency_d"] / (merged["tpot"] * self.output_length) * 1000
    merged["pd_ratio"] = merged["d_qps"] / merged["p_qps"]
    merged["balanced_qps"] = merged[["p_qps", "d_qps"]].min(axis=1)

    return merged
```

**列命名规范：**

- 只出现在一个DataFrame中的列保持不变（如prefill的 `ttft`，decode的 `tpot`）
- 重叠列添加 `_p` 和 `_d` 后缀（`parallel_p`, `parallel_d` 等）
- 最终输出使用 `ttft_p` 和 `tpot_d` 保持一致性

### 2.4 结果输出

先输出 Overall Best Configuration，再输出 Top N PD配比表格，与现有功能输出风格保持一致。每行包含对应的P和D完整配置信息及concurrency：

**P/D Devices 计算逻辑（仅在用户指定 `--num-devices` 时生效）：**

当用户提供 `--num-devices`（总设备数）时，需要根据 PD ratio 和每个实例的卡数，计算出 P 和 D 侧的总设备分配数。约束条件：

```bash
P_instances * prefill_devices_per_instance + D_instances * decode_devices_per_instance = num_devices
P_instances / D_instances ≈ pd_ratio
P_instances >= 1, D_instances >= 1 (正整数)
```

计算方式：遍历所有满足总设备数约束的 `(P_instances, D_instances)` 正整数组合，选择实际比值 `P_instances / D_instances` 最接近 `pd_ratio` 的方案。

示例：`num_devices=16, prefill_devices_per_instance=4, decode_devices_per_instance=2, pd_ratio=1.5`

- P_instances=3, D_instances=2 → P_devices=12, D_devices=4 → 实际比值=1.5 → 总计=16 ✓

**示例（用户指定了 `--num-devices`）：**

```bash
********************************************************************************
  --------------------------------------------------------------------------
  Input Configuration:
    Model: model_id
    Devices: 16 DEVICE                        ← 仅在用户输入 --num-devices 时输出
    Prefill Devices Per Instance: 4
    Decode Devices Per Instance: 2
    TTFT Limits: 500 ms
    TPOT Limits: 50 ms
  --------------------------------------------------------------------------
  Overall Best Configuration:
    PD Ratio:    1.50 (P Instances:D Instances)
    Prefill QPS: 10.00 req/s  (TTFT: 100.00 ms, Parallel: tp4pp1dp1, Batch: 4, Concurrency: 4)
    Decode QPS:  15.00 req/s  (TPOT: 13.33 ms, Parallel: tp2pp1dp1, Batch: 8, Concurrency: 8)
    P Instances: 3 (12 devices)               ← 仅在用户输入 --num-devices 时输出
    D Instances: 2 (4 devices)                ← 仅在用户输入 --num-devices 时输出
  --------------------------------------------------------------------------
  Top N PD Ratio Configurations:
  +-----+----------+-----------+-----------+---------+---------+------------+------------+-----------+-----------+---------+---------+---------------+---------------+
  | Top | PD Ratio | P QPS     | D QPS     | P TTFT  | D TPOT  | P Parallel | D Parallel | P Devices | D Devices | P Batch | D Batch | P Concurrency | D Concurrency |
  |     | (P:D)    | (req/s)   | (req/s)   | (ms)    | (ms)    |            |            | /Instance | /Instance | Size    | Size    |               |               |
  +-----+----------+-----------+-----------+---------+---------+------------+------------+-----------+-----------+---------+---------+---------------+---------------+
  |  1  |   1.50   |     10.00 |     15.00 |  100.00 |   13.33 | tp4pp1dp1  | tp2pp1dp1  |         4 |         2 |       4 |       8 |             4 |             8 |
  |  2  |   1.20   |     12.00 |     14.40 |  120.00 |   15.00 | tp2pp1dp2  | tp2pp1dp1  |         4 |         2 |       6 |       6 |            12 |             6 |
  |  3  |   ...    |       ... |       ... |     ... |     ... | ...        | ...        |       ... |       ... |     ... |     ... |           ... |           ... |
  +-----+----------+-----------+-----------+---------+---------+------------+------------+-----------+-----------+---------+---------+---------------+---------------+
********************************************************************************
```

**说明：**

- `Devices` 行仅在用户输入 `--num-devices` 时输出
- 当用户指定 `--num-devices` 时，Top N 表格中包含 `P Devices` 和 `D Devices` 列，展示每个P/D实例的卡数
- 当用户指定 `--num-devices` 时，Overall Best Configuration 中输出 P/D 实例数和总设备分配（根据 PD ratio 和 num_devices 约束计算得出）
- 表格按均衡QPS降序排列
- 每行包含该配比对应的P和D完整配置信息（含concurrency）
- Overall Best Configuration在表格之前输出，与现有agg/disagg功能风格一致

### 2.5 整体架构流程

```bash
CLI参数解析 (throughput_optimizer.py)
    │
    ├─ --enable-optimize-prefill-decode-ratio 开启?
    │   ├─ 是 → ParallelRunner.run_pd_ratio()
    │   │       │
    │   │       ├─ Phase 1: P独立寻优
    │   │       │   ├─ _get_pd_instance_user_configs(p_devices) → P的TP配置列表
    │   │       │   ├─ ProcessPoolExecutor 并行执行
    │   │       │   │   └─ 每个P TP配置 → ModelRunner → DisaggOptimizer → 二分查找最优batch
    │   │       │   └─ 收集所有P结果: [p_result_1, p_result_2, ...]
    │   │       │
    │   │       ├─ Phase 2: D独立寻优
    │   │       │   ├─ _get_pd_instance_user_configs(d_devices) → D的TP配置列表
    │   │       │   ├─ ProcessPoolExecutor 并行执行
    │   │       │   │   └─ 每个D TP配置 → ModelRunner → DisaggOptimizer → 二分查找最优batch
    │   │       │   └─ 收集所有D结果: [d_result_1, d_result_2, ...]
    │   │       │
    │   │       └─ Phase 3: 组合计算PD配比
    │   │           ├─ 对每对 (P_result_i, D_result_j) 计算 QPS 和 PD ratio
    │   │           ├─ 按均衡QPS降序排序
    │   │           └─ 输出 Overall Best + Top N PD配比结果
    │   │
    │   └─ 否 → 原有 run_agg() / run_disagg() 流程
```

### 2.6 模块交互关系

```bash
┌──────────────────────────────────────────────────────────┐
│           throughput_optimizer.py (CLI)                   │
│  新增: --prefill-devices-per-instance                     │
│        --decode-devices-per-instance                      │
│        --enable-optimize-prefill-decode-ratio              │
└───────────────────┬──────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│            ParallelRunner                        │
│  新增: run_pd_ratio()                            │
│        _get_pd_instance_user_configs(num_devices) │
│        _submit_pd_task()                         │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Phase 1: P寻优          Phase 2: D寻优
  (独立并行)               (独立并行)
        │                       │
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ P ModelRunner │        │ D ModelRunner │
│ + DisaggOpt  │        │ + DisaggOpt  │
│ (prefill)    │        │ (decode)     │
└──────┬───────┘        └──────┬───────┘
       │                       │
       └───────────┬───────────┘
                   ▼
         Phase 3: 组合计算
    ┌──────────────────────────┐
    │ PDRatioThroughputOptimizer│
    │ - 组合P/D结果             │
    │ - 计算QPS & PD Ratio      │
    │ - 排序输出Top N           │
    └──────────────────────────┘
```

---

## 3. 实施计划

### 已完成功能

* [x] CLI参数扩展：新增 `--prefill-devices-per-instance`、`--decode-devices-per-instance`、`--enable-optimize-prefill-decode-ratio` 参数
* [x] PDRatioThroughputOptimizer：实现组合器类，使用DataFrame接口，向量化cross-join
* [x] 结果输出：RFC合规的Overall Best + Top N PD配比表格，支持去重
* [x] `--dump-original-results` 应用与正常输出相同的过滤
* [x] 单元测试：覆盖PD配比寻优的各场景

### 测试计划

| 测试场景 | 说明 |
| :--- | :--- |
| 基本功能测试 | PD配比寻优正常运行，输出正确的PD配比和Top N结果 |
| PD配比计算测试 | 验证 pd_ratio = D_QPS / P_QPS 计算正确 |
| 参数校验测试 | 缺少必要参数、无效参数值的错误处理 |
| 边界条件测试 | P或D的QPS为0的处理 |
| 输出格式测试 | 验证Overall Best在前、Top N表格在后，每行包含PD ratio、P/D QPS、P/D配置及concurrency |
| 与现有功能兼容性测试 | 确保不影响现有agg/disagg模式 |

---

## 4. 修改文件

| 文件 | 说明 |
| :--- | :--- |
| `serving_cast/service/pd_ratio_throughput_optimizer.py` | 核心组合器类，使用DataFrame接口 |
| `serving_cast/service/optimizer_summary.py` | PD配比输出格式化和过滤 |
| `serving_cast/parallel_runner.py` | `run_pd_ratio()` 集成，使用 `_add_summary_result` 模式 |
| `cli/inference/throughput_optimizer.py` | CLI参数 |
| `tests/test_pd_ratio_throughput_optimizer.py` | 单元测试 |

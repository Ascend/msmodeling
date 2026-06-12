# 特性设计：[算子性能数据采集能力建设]

## Revision History (修订记录)

| Date (日期) | Version (修订版本) | Change Description (修改描述) | Author (作者) | RFC Document (RFC文档) |
| --------- | -------------- | ------------------------- | ----------- | -------------------- |
| 2026-6-1  | 1.0            | 初稿完成，算子性能数据采集能力设计文档       | -           | -                    |
|           |                |                           |             |                      |
|           |                |                           |             |                      |

---

## 1. Background (背景描述)

当前仿真建模主要基于roofline模型进行算子性能估算，虽然能覆盖通用场景下的性能分析需求，但是面对特定硬件平台、复杂算子实现及组合场景时，估算结果与真实执行性能存在一定偏差。以此为出发点，为提升建模精度、增强方案评估可信度，需要建立一套基于实测性能数据的建模能力体系。

实测算子性能估算系统依赖 NPU Profiling 数据作为运行时性能模型的查询输入。这些数据需要从 NPU 设备上采集、解析、验证后才能使用，但此前缺乏一套标准化的离线数据生产流程。

本特性提供完整的性能数据采集工具链（位于 tools/perf_data_collection/），其核心价值在于：

* 将解析profiling数据、扩充算子形状、算子回放、数据库回填串成一条可重复执行的数据生产链
* 明确性能数据工具链的结构，将“解析profiling数据、扩充算子形状、算子回放、数据库回填”等流程解耦为职责清晰的子模块
* 提供结构化、可追溯的算子性能数据资产。

## 2. Design (方案设计)

### 2.1 总体架构

```python
原始 NPU Profiling 数据
       │
       ▼
parsers/parse_kernel_details.py     ← 解析 kernel_details.csv → per-kernel CSV
       │
       ▼
generate_shape_grid.py              ← shape 矩阵扩展
       │
       ▼
op_replay/*_run.py                  ← 逐 kernel 回放脚本 (25+)
       │
       ▼
start_microbench.py                 ← msprof 编排 & CSV 回写
       │
       ▼
带版本的 profiling_database/data 目录
  └── {device}/vllm_ascend/{version}/   ← 计算算子 CSV + op_mapping.yaml

comm_bench/generate_comm_microbench.py  ← 生成通信基准脚本
       │
       ▼
带版本的 profiling_database/data 目录
  └── {device}/hccl/{cann_version}/     ← HCCL 通信 CSV
```

代码分层结构

- 数据接入层：parse/

- 样本生成层： generate_shape_grid/

- 执行测量层：op_replay/、start_microbench/、comm_bench/

- 数据资产层：profiling_database/data/

### 2.2 工具范围与职责

| 范围              | 入口                                                                   | 数据库职责                                                      |
| --------------- | -------------------------------------------------------------------- | ---------------------------------------------------------- |
| 原始 profiling 解析 | `parsers/parse_kernel_details.py`                                    | 从 `kernel_details*.csv` 或 profiling 目录生成按 kernel 拆分的计算 CSV |
| Shape grid 扩展   | `generate_shape_grid.py` + `grid_generator/` + `memory_estimator.py` | 追加可 replay 的 theory shape 行，并按 HBM 预算过滤                    |
| 计算算子 replay     | `op_replay/*_run.py` (25+ scripts) + `start_microbench.py`           | 在 NPU 主机上重放 CSV 行并回写 microbench 耗时                         |
| HCCL 通信采集       | `comm_bench/generate_comm_microbench.py` + `run_comm_bench.sh`       | 生成 `hccl/{cann_version}/hcom_*.csv`                        |

### 2.3 数据库目录契约

数据库包含两类持久数据：

| 数据类型    | 目录                                                                                          | 内容                                     | 版本范围                                |
| ------- | ------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------- |
| 计算/算子数据 | `tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version_dir}/` | `op_mapping.yaml` 和 `{KernelType}.csv` | 设备 + vLLM-Ascend + PyTorch + CANN 栈 |
| 通信数据    | `tensor_cast/performance_model/profiling_database/data/{device}/hccl/{cann_version}/`       | `hcom_*.csv` benchmark 文件，可选拓扑/配置元数据   | 设备 + CANN/HCCL 栈                    |

`{version_dir}` 命名约定：
    vllm{vllm_version}_torch{torch_version}_cann{cann_version}

示例目录：

* `vllm0.13.0_torch2.8.0_cann8.3`
* `vllm0.15.0_torch2.9.0_cann8.5`
* `vllm0.18.0_torch2.9.0_cann8.5`
* `vllm0.18.0_torch2.9.0_cann8.5_shape_generated`（generated/staging 示例）

### 2.4 计算 CSV Schema 契约

每个计算 CSV 以 NPU profiling 中的 `Type` 列命名：{KernelType}.csv

**基础必需列**：

| 列组               | 列                                                                                                                                           | 生产者                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Kernel 标识与 shape | `OP State`, `Accelerator Core`, `Input Shapes`, `Input Data Types`, `Input Formats`, `Output Shapes`, `Output Data Types`, `Output Formats` | 原始 profiling parser      |
| Profiling 耗时     | `Profiling Average Duration(us)`, `Profiling Median Duration(us)`, `Profiling Std Duration(us)`                                             | 原始 profiling parser      |
| Profiling 计数器    | `Profiling Average aicore_time(us)`, `Profiling Average aic_total_cycles`, `Profiling Average aic_mac_time(us)` 及 AIC/AIV 利用率列              | 原始 profiling parser      |
| Microbench 耗时    | `Average Duration(us)`                                                                                                                      | `start_microbench.py` 回写 |
| Microbench 计数器   | `MicroBench aicore_time(us)`, `MicroBench aic_total_cycles` 及对应 `MicroBench ...` 列                                                          | `start_microbench.py` 回写 |

**关键规则**：

* Shape 字段使用分号分隔 tensor 槽位，逗号分隔维度：`"136,7168;7168,3584"`
* 槽位数量有语义。对于 FIA、自定义 kernel 等包含可选输入或标量参数的算子，空槽位必须保留
* `Input Data Types` 和 `Input Formats` 按槽位索引与 `Input Shapes` 对齐
* `FRACTAL_NZ` 是合法 format；replay 与运行时 lookup 都会在需要时做规范化

### 2.5 通信 CSV Schema 契约

通信数据保存在 HCCL 目录下，标准文件名：

* `hcom_allGather_.csv`
* `hcom_allReduce_.csv`
* `hcom_alltoallv_.csv`
* `hcom_reduceScatter_.csv`

| 列                                                                          | 级别      | 类型     | 含义                                         |
| -------------------------------------------------------------------------- | ------- | ------ | ------------------------------------------ |
| `message_bytes`                                                            | 运行时必需   | int    | 每 rank 消息大小，用于查询和插值                        |
| `num_devices`                                                              | 运行时必需   | int    | 参与通信的 rank 数                               |
| `Average Duration(us)` / `Profiling Average Duration(us)` / `Duration(us)` | 运行时必需其一 | float  | 通信时延                                       |
| `topology_tier`                                                            | 条件匹配    | int    | 设备拓扑层级，0为inter_pod，1为intra_pod，2为die_level |
| `dtype`                                                                    | 采集标准    | string | 测量 dtype，如 `DT_BF16`                       |
| `bandwidth_gbps`                                                           | 采集标准    | float  | 派生带宽，用于审计和校验                               |

### 2.6 原始 Profiling Parser

`parsers/parse_kernel_details.py`是把 Ascend profiling 输出转换成按 kernel 拆分 CSV 的入口。

**行为**：

* `--profiling-path` 可以是单个 `kernel_details*.csv` 文件或 profiling 目录
* 目录输入时递归扫描文件名包含 `kernel_details` 的 CSV
* 同时发现 `operator_details.csv` 和 `trace_view.json` 用于 FIA bundle 检查
* 要求 kernel detail 中存在 `Type`、`OP State`、`Accelerator Core`、shape/dtype/format 列、`Duration(us)` 以及 AIC/AIV 计数器列
* 规范化已知变体名：`split_qkv_rmsnorm_rope_kernel_0` → `split_qkv_rmsnorm_rope_kernel`、`muls_add_kernel_1` → `muls_add_kernel`
* 按 `(normalized Type, Input Shapes, Output Shapes)` 聚合，输出平均/median/标准差耗时列
* 每个算子类型写成一个 `{KernelType}.csv` 到目标数据库目录

**设计原则**：Parser 保持保守——不推断缺失 tensor 槽位，不改写 API 语义；保留 profiling 表示，把 shape/API 对齐留给 `op_mapping.yaml`、`generate_shape_grid.py` 和 replay 脚本。

### 2.7 Shape Grid 生成

`generate_shape_grid.py`通过 `grid_generator/` 包向数据库 CSV 追加 theory 生成行。

**架构**：

```python
    grid_generator/
    ├── config.py           # 配置加载
    ├── config.yaml         # 定义算子的shape结构和采样网格，用于生产理论性能测试的shape组合
    ├── runner.py           # 主调度：加载 CSV、运行 theory 模式、追加行
    ├── theory_router.py    # 根据config.yaml，获得kernel type 到 generator 的路由逻辑
    ├── shape_grids.py      # shape grid 定义
    ├── evaluator.py        # 解析和计算配置中的shape表达式
    ├── model_configs.py    # 模型配置（Qwen3 等）
    ├── utils.py            # 工具函数
    └── generators/
        ├── base.py                 # 基类
        ├── fused_attention.py      # FIA 复杂 generator
        └── moe.py                  # MoE 复杂 generator
```

**行为**：

* `--target-models` 使用已知模型配置裁剪 GEMM `(N, K)` 候选
* `--rows` 限制每个 CSV 的行数；`--seed` 保证抽样可复现
* `--max-hbm-gb` 通过 `memory_estimator.py` 过滤超出内存预算的行

**设计原则**：生成行继承源行中的稳定结构 metadata，不填充性能值（留空或置零）。只有经过 `start_microbench.py` 或后续 profiling 导入填充耗时后，才是可用的生产数据。

### 2.8 `op_replay` 逐 kernel Replay 脚本框架

`op_replay/` 存放按 kernel 编写的 replay 实现，共 25+ 个脚本。

**支持算子列表**：

| 类别          | 算子脚本                                                                                   | 说明                 |
| ----------- | -------------------------------------------------------------------------------------- | ------------------ |
| Matmul      | `MatMulV2_run.py`, `MatMulV3_run.py`, `MatMulCommon_run.py`                            | 标准矩阵乘法             |
| 量化          | `QuantBatchMatmulV3_run.py`, `AscendQuantV2_run.py`, `DynamicQuant_run.py`             | 量化 matmul/quantize |
| Attention   | `FusedInferAttentionScore_run.py`                                                      | FIA 增强 attention   |
| Norm        | `RmsNorm_run.py`, `AddRmsNormBias_run.py`                                              | 归一化 kernel         |
| RoPE        | `InterleaveRope_run.py`, `split_qkv_rmsnorm_rope_kernel_run.py`                        | RoPE 相关            |
| KV Cache    | `ReshapeAndCacheNdKernel_run.py`, `KvRmsNormRopeCache_run.py`                          | KV cache 操作        |
| Act         | `SwiGlu_run.py`, `SoftmaxV2_run.py`, `TensorMove_run.py`                               | 激活与 softmax        |
| Memory      | `Slice_run.py`, `PadV3_run.py`, `Index_run.py`, `GatherV2_run.py`, `MaskedFill_run.py` | 内存操作               |
| Shape       | `Transpose_run.py`, `Sort_run.py`                                                      | 转置/排序              |
| MoE         | `DispatchFFNCombine_run.py`, `RINGMLAPrefillBF16Kernel_run.py`                         | MoE 融合 kernel      |
| Sampling    | `ArgMaxV2_run.py`                                                                      | TopK/TopP 采样       |
| Elementwise | `Add_run.py`                                                                           | 逐元素加法              |

**共享组件**：

| 文件                         | 职责                                                                                             |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| `common.py`                | 数据库路径解析、版本目录命名、dtype/format 解析、tensor 构造、repeat count 处理、CSV 行加载、无效行跟踪                         |
| `replay_framework.py`      | `OpReplay` helper 类，提供标准 API 型 kernel 的 replay 基类                                              |
| `run_all_op.py`            | 发现 `*_run.py`，支持 `--execution-mode inprocess`（单 msprof session 采集），写出 `run_all_op_status.json` |
| `probe_dfc_constraints.py` | DFC kernel 约束探测                                                                                |

**共享约定**：

* `common.py` 负责所有公共逻辑：`load_csv_rows()`、`row_has_valid_duration()`、`csv_has_complete_microbench()`、`ensure_npu_available()` 等
* `replay_framework.py` 提供 `OpReplay` 类，封装 `run()` + `verify()` + `microbench_duration()` 的生命周期
* `run_all_op.py` 自动发现算子脚本，支持 `--execution-mode inprocess` 以便由单个外层 `msprof` session 采集
* 每个算子脚本读取匹配的 `{KernelType}.csv`，在 NPU 上 replay 每一行，打印 `[OK]` 信息
* 构造 replay case 失败时自动删除 CSV 中的无效行
* 通过 `MSMODELING_OP_REPLAY_REPEAT_COUNT` 环境变量可覆盖默认 repeat count

### 2.9 `start_microbench.py` msprof 编排与回写

`start_microbench.py`是 replay 测量回写的生产入口。

**流程**：
    1. 解析 --device / --vllm-version 参数 → 定位数据库目录
    2. 校验选中算子和自定义 OPP 所需环境变量
    3. 执行 msprof --output=... python op_replay/run_all_op.py --execution-mode inprocess
    4. 定位生成的 PROF_开头的目录和 op_summary_开头的表格文件
    5. 按 operator type 和 signature 聚合行
    6. 将 Average Duration(us) 和 MicroBench ... 计数器列回写到匹配的数据库 CSV 行
    7. 在 reports/ 子目录下生成kernel算子采集的结果报告和 duration gap CSV

**行匹配**：使用 `Input Shapes`、`Input Data Types`、`Input Formats`、`Output Shapes`、`Output Data Types` 五列作为 signature。`DispatchFFNCombine` 还会把 `EP Size` 纳入匹配。

**更新模式**：

| 模式             | 行为                                                       |
| -------------- | -------------------------------------------------------- |
| `all`          | 更新所有匹配行，并追加未匹配的 profiling 样本                             |
| `missing-only` | 只 replay/fill microbench 与 profiling 耗时都无效的行；未匹配样本只报告不追加 |

**其他能力**：

* `--prof-path`：直接解析已有 `PROF_*` 目录而不启动 `msprof`
* `--prune-empty-duration-rows`：删除 replay 和 profiling 耗时在回写后仍然无效的行
* 支持 `--kernel-list` 指定部分算子运行
* 自动生成 `reports/replay_report_{timestamp}.md` 和 `reports/duration_gap_{timestamp}.csv`
* 因为MatMulV2、MatMulV3、MatMulCommn算子均对应torch.mm，因此回放算子后回填数据库时，根据实际msprof采集得到的是算子的type，将数据回填到对应的算子csv中

**需要自定义 OPP 的算子**（需设置 `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`）：

* `AddRmsNormBias`
* `DispatchFFNCombine`
* `KvRmsNormRopeCache`
* `RINGMLAPrefillBF16Kernel`
* `split_qkv_rmsnorm_rope_kernel`

### 2.10 HCCL 通信 Benchmark

**生成通信基准**：`comm_bench/generate_comm_microbench.py`

* 使用 `torchrun` 多进程运行 HCCL 通信算子
* 支持 `all_reduce`、`all_gather`、`reduce_scatter`、`all_to_all` 四种算子
* 自动推导 `topology_tier`：根据 `--grid-shape` 和 rank group 确定拓扑层级
* 支持 event mode（仅 hcom_kernel，快速模式）和默认的 sync mode
* 输出 `message_bytes`、`num_devices`、`dtype`、`topology_tier`、`Duration(us)`、`bandwidth_gbps`

### 2.11 代码文件结构

```python
tools/perf_data_collection/
├── __init__.py
├── README.md                                          # 工具链使用说明
│
├── parsers/
│   ├── parse_kernel_details.py                        # 原始 Profiling 解析器
│   └── trace_to_csv.py                                # Chrome trace → CSV 转换
│
├── fill_fia_runtime_metadata.py                       # FIA JSONL runtime 回填
├── fia_common.py                                      # FIA 共享工具函数
│
├── generate_shape_grid.py                             # Shape grid 生成入口
├── grid_generator/
│   ├── __init__.py
│   ├── config.py                                      # 配置加载
│   ├── config.yaml                                    # kernel → theory pattern 路由
│   ├── runner.py                                      # 主调度
│   ├── theory_router.py                               # 路由逻辑
│   ├── shape_grids.py                                 # Shape grid 定义
│   ├── evaluator.py                                   # 生成行评估
│   ├── model_configs.py                               # 模型配置
│   ├── utils.py                                       # 工具函数
│   └── generators/
│       ├── __init__.py
│       ├── base.py                                    # 基类
│       ├── fused_attention.py                         # FIA generator
│       └── moe.py                                     # MoE generator
│
├── memory_estimator.py                                # HBM 内存估算
│
├── op_replay/
│   ├── common.py                                      # 共享公共逻辑
│   ├── replay_framework.py                            # OpReplay helper
│   ├── run_all_op.py                                  # 自动发现与运行
│   ├── probe_dfc_constraints.py                       # DFC 约束探测
│   │
│   ├── MatMulV2_run.py                                # MatMulV2 replay
│   ├── MatMulV3_run.py                                # MatMulV3 replay
│   ├── MatMulCommon_run.py                            # MatMulCommon replay
│   ├── QuantBatchMatmulV3_run.py                      # 量化 matmul replay
│   ├── AscendQuantV2_run.py                           # 量化 replay
│   ├── DynamicQuant_run.py                            # 动态量化 replay
│   ├── FusedInferAttentionScore_run.py                # FIA replay
│   ├── RmsNorm_run.py                                 # RmsNorm replay
│   ├── AddRmsNormBias_run.py                          # AddRmsNormBias replay
│   ├── InterleaveRope_run.py                          # RoPE replay
│   ├── split_qkv_rmsnorm_rope_kernel_run.py           # 融合 RoPE replay
│   ├── SwiGlu_run.py                                  # SwiGlu replay
│   ├── SoftmaxV2_run.py                               # Softmax replay
│   ├── TensorMove_run.py                              # TensorMove replay
│   ├── ReshapeAndCacheNdKernel_run.py                 # KV cache replay
│   ├── KvRmsNormRopeCache_run.py                      # KV norm cache replay
│   ├── Slice_run.py                                   # Slice replay
│   ├── PadV3_run.py                                   # Pad replay
│   ├── Index_run.py                                   # Index replay
│   ├── GatherV2_run.py                                # Gather replay
│   ├── MaskedFill_run.py                              # MaskedFill replay
│   ├── Transpose_run.py                               # Transpose replay
│   ├── Sort_run.py                                    # Sort replay
│   ├── ArgMaxV2_run.py                                # TopK sampling replay
│   ├── Add_run.py                                     # 逐元素加法 replay
│   ├── DispatchFFNCombine_run.py                      # DFC replay
│   └── RINGMLAPrefillBF16Kernel_run.py                # MLA prefetch replay
│
├── start_microbench.py                                # msprof 编排与回写
│
└── comm_bench/
    ├── generate_comm_microbench.py                    # HCCL 通信基准生成
    ├── run_comm_bench.sh                              # A3设备的集成通信基准运行脚本
    └── validate_comm_alignment.py                     # A3设备的对齐验证脚本

docs/perf_database/tutorial/
    ├── OP_PLUGIN_MAPPING_TUTORIAL.md                   # Op mappings 脚本编写教程
    └── MICROBENCH_RUN_SCRIPT_TUTORIAL_zh.md            # Replay 脚本编写教程

.agents/skills/
    ├── microbench/SKILL.md                      # Microbench Run 脚本生成skill
    └── op-mapping/SKILL.md                      # Op mappings 脚本生成skil
```

### 2.12 测试资产

```python
tests/tools/
├── test_fia_parser_backfill.py                        # FIA parser + backfill 测试
├── test_generate_comm_microbench.py                   # 通信基准生成测试
├── test_generate_shape_grid.py                        # Shape grid 生成测试
├── test_op_replay.py                                  # op_replay 框架测试
├── test_op_replay_common.py                           # common.py 测试
├── test_replay_framework.py                           # replay_framework 测试
├── test_start_microbench.py                           # start_microbench 测试
└── test_validate_comm_alignment.py                    # 通信对齐验证测试
```

**测试结果**：
    pytest tests/regression/cli/test_start_microbench.py
    pytest tests/regression/cli/test_generate_shape_grid.py
    pytest tests/regression/cli/test_shape_grid_model_configs.py
    pytest tests/regression/cli/test_generate_comm_microbench.p
    pytest tests/regression/cli/test_validate_comm_alignment.py
    pytest tests/regression/cli/test_op_replay.py
    pytest tests/regression/cli/test_op_replay_common.py
    pytest tests/regression/cli/test_replay_framework.py

## 3. Usage Instructions (使用说明)

### 3.1典型使用流程

```python
# kernel算子采集
# 1. 从 NPU profiling 原始数据生成 per-kernel CSV
python tools/perf_data_collection/parsers/parse_kernel_details.py \
    --profiling-path <kernel_details.csv> --database-path <data_dir>

# 2. 生成 shape 变异矩阵扩大覆盖
python tools/perf_data_collection/generate_shape_grid.py \
    --data-dir <data_dir> \
    --target-models dsv3,qwen3-32b \
    --max-hbm-gb 128

# 3. 运行微基准测试（需要 NPU 设备 + CANN + vLLM-Ascend 自定义 OPP）
export ASCEND_CUSTOM_OPP_PATH=/path/to/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/path/to/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/:${LD_LIBRARY_PATH}

python tools/perf_data_collection/start_microbench.py \
    --device ATLAS_800_A3_752T_128G_DIE \
    --vllm-version 0.18.0 \
    --torch-version 2.9.0 \
    --cann-version 8.5 \
    --prune-empty-duration-rows

# HCCL 通信基准采集
tools/perf_data_collection/comm_bench/generate_comm_microbench.py \
    --do-run --output-dir ./hccl_data \
    --ops all_reduce all_gather reduce_scatter all_to_all \
    --grid-shape 48 8 2 --num-devices 16 2

```

### 3.2 命令行参数说明

#### `parse_kernel_details.py`

| 参数                 | 是否必选 | 默认值                        | 说明                                                      |
| ------------------ | ---- | -------------------------- | ------------------------------------------------------- |
| `--profiling-path` | 是    |                            | 输入 profiling 路径（`kernel_details*.csv` 文件或 profiling 目录） |
| `--database-path`  | 否    |                            | 显式输出数据库目录                                               |
| `--device`         | 否    | ATLAS_800_A3_752T_128G_DIE | 目标设备名称，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用     |
| `--vllm-version`   | 否    |                            | vLLM 版本，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用    |
| `--torch-version`  | 否    |                            | PyTorch 版本，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用 |
| `--cann-version`   | 否    |                            | CANN 版本，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用    |

#### `start_microbench.py`

| 参数                                      | 是否必选 | 默认值                        | 说明                                                                      |
| --------------------------------------- | ---- | -------------------------- | ----------------------------------------------------------------------- |
| `--database-path`                       | 否    |                            | 数据库目录路径                                                                 |
| `--device`                              | 否    | ATLAS_800_A3_752T_128G_DIE | 目标设备名称，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用                     |
| `--vllm-version`                        | 否    |                            | vLLM-Ascend 版本号，如 `0.18.0`，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用 |
| `--torch-version`                       | 否    |                            | PyTorch 版本号，如 `2.9.0`，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用      |
| `--cann-version`                        | 否    |                            | CANN 版本号，如 `8.5`，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用           |
| `--prof-path`                           | 否    |                            | 已有的 PROF_* 目录，不启动 msprof                                                |
| `--op`                                  | 否    |                            | 要回放的算子列表，当不指定时默认回放全部算子                                                  |
| `--dispatch-ffn-combine-ep-size`        | 否    | 16                         | DispatchFFNCombine 的 EP 并行度，msprof回放DispatchFFNCombine算子的配置             |
| `--dispatch-ffn-combine-nproc-per-node` | 否    |                            | torchrun 每节点进程数，msprof回放DispatchFFNCombine算子的配置                         |
| `--dispatch-ffn-combine-nnodes`         | 否    | 1                          | torchrun 节点数，msprof回放DispatchFFNCombine算子的配置                            |
| `--dispatch-ffn-combine-node-rank`      | 否    | 0                          | torchrun 当前节点 rank，msprof回放DispatchFFNCombine算子的配置                      |
| `--dispatch-ffn-combine-master-addr`    | 否    | "127.0.0.1"                | torchrun master 地址，msprof回放DispatchFFNCombine算子的配置                      |
| `--dispatch-ffn-combine-master-port`    | 否    |                            | torchrun master 端口，msprof回放DispatchFFNCombine算子的配置                      |
| `--repeat-count`                        | 否    | 1                          | 每个kernel算子数据回放的重复次数                                                     |
| `--update-mode`                         | 否    | “all”                      | 更新模式： all 或 missing-only                                                |
| `--fail-fast`                           | 否    | False                      | 是否遇到错误立即退出                                                              |
| `--prune-empty-duration-rows`           | 否    | False                      | 是否删除回写后仍然无效的行                                                           |

#### `generate_shape_grid.py`

| 参数                | 是否必选 | 默认值                              | 说明                                                                  |
| ----------------- | ---- | -------------------------------- | ------------------------------------------------------------------- |
| `--target-models` | 否    |                                  | 逗号分隔的模型名（如 "qwen3-32b" ），指定时通过对 tp_sizes 枚举，只生成模型实际需要的 GEMM (N,K) 对 |
| `--data-dir`      | 否    |                                  | 数据库目录路径                                                             |
| `--device`        | 否    | ATLAS_800_A3_752T_128G_DIE       | 目标设备名称，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用                 |
| `--vllm-version`  | 否    | vLLM 版本                          | vLLM-Ascend 版本号，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用        |
| `--torch-version` | 否    | PyTorch 版本                       | PyTorch 版本，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用             |
| `--cann-version`  | 否    | CANN 版本                          | CANN 版本，用于自动推导输出数据库目录，当指定`--database-path`时，此参数无实际作用                |
| `--rows`          | 否    | 1000 | 每个 CSV 的最大理论行数（0=不限制，所有生成的理论形状行全部保留）                                |
| `--seed`          | 否    |                                  | 随机种子，保证可复现                                                          |
| `--max-hbm-gb`    | 否    | 32.0                             | HBM 内存上限（GB），过滤超出预算的行                                               |

#### `generate_comm_microbench.py`

| 参数                | 是否必选 | 默认值                                                          | 说明                                                                                    |
| ----------------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `--output-dir`    | 否    |                                                              | 每个算子写入 独立的 CSV 文件，此参数为CSV 输出父目录。当`--output-csv`指定时，此参数无效                              |
| `--output-csv`    | 否    |                                                              | 单个算子的输出位置，仅当--op为一个算子时有效，优先级比`--output-dir`高                                          |
| `--ops`           | 否    | ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"] | 采集的通信算子列表                                                                             |
| `--grid-shape`    | 否    | [48, 8, 2]                                                   | 实际采集设备的硬件拓扑网格形状，A2设备为"128 8"，A3设备为"48 8 2"                                            |
| `--num-devices`   | 否    | [16]                                                         | 采集的通信组设备数列表                                                                           |
| `--topology-tier` | 否    |                                                              | 拓扑层级，0表示inter_pod，1表示intra_pod，2表示die_level。不指定，将根据`--num-devices`与`--grid-shape`解析推导 |
| `--dtype`         | 否    | torch.bfloat16                                               | 张量数据类型，可选项为[torch.bfloat16, torch.float16, torch.float32, torch.int8]                 |
| `--bytes-grid`    | 否    | [1024, 2048, 4096, ..., 536870912]                           | 通信量大小，默认值为1KB ~ 512MB                                                                 |
| `--do-run`        | 否    | False                                                        | 实际运行 benchmark                                                                        |
| `--bench-mode`    | 否    | kernel                                                       | bench模式，可选项为kernel或event                                                              |

### 3.3 使用约束

* **硬件依赖**：`start_microbench.py`、`op_replay/*_run.py`、`comm_bench/generate_comm_microbench.py`、`run_comm_bench.sh` 需要昇腾 NPU 硬件（ATLAS 800 A3 系列），无 NPU 环境无法运行
* **软件依赖**：微基准测试需要安装 `torch_npu`、`msprof` 命令行工具、CANN 工具包；自定义算子需要配置 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`
* **无 NPU 测试**：`parse_kernel_details.py`、`fill_fia_runtime_metadata.py`、`generate_shape_grid.py`、`validate_comm_alignment.py` 以及所有单元测试可在无 NPU 环境运行
* **数据量**：shape grid 生成可能产生大量 CSV 行（默认 10,000 行上限），需监控磁盘空间
* **Replay 覆盖限制**：CSV 可能存在但没有对应 replay 脚本；部分脚本只支持其面向的已记录 shape/API 模式
* **工具链独立性**：`tools/` 目录可独立进行运行，不依赖实测算子建模部分。

## 4. Test Design (测试设计)

### 4.1 单元测试

运行所有工具链单元测试：
    pytest tests/regression/cli/test_start_microbench.py
    pytest tests/regression/cli/test_generate_shape_grid.py
    pytest tests/regression/cli/test_shape_grid_model_configs.py
    pytest tests/regression/cli/test_generate_comm_microbench.p
    pytest tests/regression/cli/test_validate_comm_alignment.py
    pytest tests/regression/cli/test_op_replay.py
    pytest tests/regression/cli/test_op_replay_common.py
    pytest tests/regression/cli/test_replay_framework.py

### 4.2 端到端验证

1、使用parse_kernel_detail.py脚本进行解析profiling数据，可以获取以每个算子Type为名称的csv

2、使用generate_shape_grid.py脚本成功扩样算子shape

3、使用start_microbench.py脚本对目前已支持的kernel算子进行回放及回填数据

4、使用torchrun命令运行generate_comm_microbench.py脚本,获取通信算子的数据

### 

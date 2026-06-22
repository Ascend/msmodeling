# RFC: 性能数据库独立采集工具链

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | 已完成 |
| **作者** | Horacehxw, Codex |
| **创建日期** | 2026-05-12 |
| **相关链接** | <https://gitcode.com/Ascend/msmodeling/pull/124> |

---

## 1. 概述

性能数据库只有在 CSV 数据、算子映射、replay 脚本、通信 benchmark 和校验报告都来自可重复的离线流程时，才适合作为运行时性能模型的查询输入。本 RFC 提出新增一套 `tools/perf_data_collection/` 离线工具链，覆盖原始 profiling 解析、FIA runtime metadata 回填、theory shape grid 生成、逐 kernel replay、`msprof` 回写、HCCL 通信 benchmark 和基础 trace 转换，并将这些文件化制品定义为稳定的数据生产契约。

产出的数据库供 `EmpiricalPerformanceModel` / `ProfilingDataSource` 消费：运行时只需要读取 `op_mapping.yaml`、按 kernel 拆分的计算 CSV，以及可选的 HCCL 通信 CSV。本文完整定义这些数据如何离线生成、校验和提升为可 review 的性能数据库，不要求读者依赖其他设计文档理解数据库生成流程。

本 RFC 定义性能数据库的独立采集和工具链侧边界：如何把原始 NPU profiling 结果转换成带版本的 CSV 数据，如何追加合成 shape 与 microbench 测量，如何采集 HCCL 通信数据，以及在运行时查询代码消费之前如何校验这些数据。

### 1.1 目标

- 提供一份不依赖临时设计或计划文档的离线数据生产契约。
- 定义数据库目录结构、计算算子 CSV schema、FIA 专用列、通信 CSV schema 和版本管理规则。
- 澄清 `op_mapping.yaml`、原始 profiling 解析、shape grid 生成、算子重放脚本、`msprof` 编排和回写之间的边界。
- 记录工具实现边界和限制，让后续贡献者清楚哪些步骤自动化，哪些步骤仍然需要手工维护。
- 让人工 review 能基于同一组检查项判断 CSV/YAML/report 是否可进入正式数据库目录。
- 给出大部分可在无 NPU 环境运行的 rollout 和测试计划，并隔离依赖硬件的 replay/HCCL 测试。

### 1.2 非目标

- 本 RFC 不修改 `ProfilingDataSource` 的运行时查询行为或插值逻辑。
- 本 RFC 不定义模型级对齐指标或端到端仿真策略，仅要求产出的数据能被这些消费者使用。
- 本 RFC 不要求外部数据库服务、远端制品仓或在线采集管线。
- 本 RFC 不声称已经覆盖所有 profiling kernel 的 replay。
- 本 RFC 不把临时计划文档定义为稳定接口。

## 2. 方案设计

### 2.1 推荐方案

推荐新增如下标准离线采集流水线：

```mermaid
flowchart TD
    A[原始 NPU profiling bundle] --> B[parsers/parse_kernel_details.py]
    B --> C[按 kernel 拆分的计算 CSV]
    C --> D[fill_fia_runtime_metadata.py]
    C --> E[generate_shape_grid.py]
    D --> E
    E --> F[op_replay/*_run.py]
    F --> G[start_microbench.py 通过 msprof 执行]
    G --> H[CSV 回写与报告]
    I[HCCL benchmark] --> J[hccl/{cann_version}/hcom_*.csv]
    J --> H
    K[op_mapping.yaml] --> F
    K --> H
    H --> L[带版本的 profiling_database/data 目录]
```

该流程保持文件化。CSV 和 YAML 便于 review、diff，也与运行时 data source 兼容。工具可以生成或重写行，但所有持久化数据制品仍然保存在带版本的数据库目录中。

#### 2.1.0 工具范围

本 RFC 定义 `tools/perf_data_collection/` 下的入口和职责：

| 范围 | 入口 | 数据库职责 |
| --- | --- | --- |
| 原始 profiling 解析 | `parsers/parse_kernel_details.py` | 从 `kernel_details*.csv` 或 profiling 目录生成按 kernel 拆分的计算 CSV |
| FIA runtime metadata 回填 | `fill_fia_runtime_metadata.py` | 将 FIA JSONL runtime 值合并到 `FusedInferAttentionScore.csv` |
| Shape grid 扩展 | `generate_shape_grid.py`、`grid_generator/`、`memory_estimator.py` | 追加可 replay 的 theory shape 行，并按 HBM 预算过滤 |
| 计算算子 replay | `op_replay/*_run.py`、`op_replay/run_all_op.py`、`start_microbench.py` | 在 NPU 主机上重放 CSV 行并回写 microbench 耗时 |
| HCCL 通信采集 | `comm_bench/generate_comm_microbench.py`、`comm_bench/run_comm_bench.sh`、`comm_bench/validate_comm_alignment.py` | 生成或校验 `hccl/{cann_version}/hcom_*.csv` |
| Trace 转换支持 | `parsers/trace_to_csv.py` | 将 TensorCast Chrome trace 转成 CSV，供离线对齐和报告使用；它不直接生产 operator database CSV |

#### 2.1.1 数据库目录契约

数据库包含两类持久数据：

| 数据类型 | 目录 | 内容 | 版本范围 |
| --- | --- | --- | --- |
| 计算/算子数据 | `tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version_dir}/` | `op_mapping.yaml` 和 `{KernelType}.csv` | 设备 + vLLM-Ascend + PyTorch + CANN 栈 |
| 通信数据 | `tensor_cast/performance_model/profiling_database/data/{device}/hccl/{cann_version}/` | `hcom_*.csv` benchmark 文件，可选拓扑/配置元数据 | 设备 + CANN/HCCL 栈 |

`{version_dir}` 遵循 helper 约定：

```text
vllm{vllm_version}_torch{torch_version}_cann{cann_version}
```

推荐目录示例包括：

- `vllm0.13.0_torch2.8.0_cann8.3`
- `vllm0.15.0_torch2.9.0_cann8.5`
- `vllm0.18.0_torch2.9.0_cann8.5`
- `vllm0.18.0_torch2.9.0_cann8.5_shape_generated`（generated/staging 示例，不应被误认为完整测量目录）

计算目录为该栈维护唯一的 `op_mapping.yaml`。通信 CSV 不复制到每个计算目录，而是由 `op_mapping.yaml` 通过 `communication_data_ref` 指向通信目录，例如：

```yaml
communication_data_ref: "../../hccl/v8.5/"
communication_fallback: analytic
```

当设备和 CANN/HCCL 栈不变时，这样可以在多个 vLLM-Ascend 版本之间复用 HCCL 测量数据。

#### 2.1.2 计算 CSV 契约

每个计算 CSV 以 NPU profiling 中的 `Type` 列命名，经过必要规范化后为：

```text
{KernelType}.csv
```

parser 和运行时 data source 使用 shape 与元数据形式的 CSV 契约，而不是 Python 对象 schema。基础必需列如下：

| 列组 | 列 | 生产者 |
| --- | --- | --- |
| Kernel 标识与 shape | `OP State`, `Accelerator Core`, `Input Shapes`, `Input Data Types`, `Input Formats`, `Output Shapes`, `Output Data Types`, `Output Formats` | 原始 profiling parser |
| Profiling 耗时 | `Profiling Average Duration(us)`, `Profiling Median Duration(us)`, `Profiling Std Duration(us)` | 原始 profiling parser |
| Profiling 计数器 | `Profiling Average aicore_time(us)`, `Profiling Average aic_total_cycles`, `Profiling Average aic_mac_time(us)` 以及相关 AIC/AIV 利用率列 | 原始 profiling parser |
| Microbench 耗时 | `Average Duration(us)` | `start_microbench.py` 回写 |
| Microbench 计数器 | `MicroBench aicore_time(us)`, `MicroBench aic_total_cycles` 以及对应的 `MicroBench ...` 计数器列 | `start_microbench.py` 回写 |

Shape 字段使用分号分隔 tensor 槽位，逗号分隔维度：

```text
"136,7168;7168,3584"
```

关键规则：

- 槽位数量有语义。对于 FIA、自定义 kernel 等包含可选输入或标量参数的算子，空槽位必须保留。
- `Input Data Types` 和 `Input Formats` 按槽位索引与 `Input Shapes` 对齐。
- `FRACTAL_NZ` 是合法 format；replay 与运行时 lookup 都会在需要时做规范化。
- 生成行可能一开始没有耗时或耗时为 0。只有在 replay/writeback 填入 `Average Duration(us)`，或 profiling 导入填入 `Profiling Average Duration(us)` 之后，才应视为测量数据。
- 运行时 lookup 选择 latency 列的优先级是：`Average Duration(us)`、`Profiling Average Duration(us)`、`Duration(us)`。

#### 2.1.3 FIA CSV 扩展契约

`FusedInferAttentionScore.csv` 需要额外 runtime metadata，因为仅凭 shape 无法完整描述 paged attention 行为。parser 和 backfill 工具在可用时使用以下列：

| 列 | 含义 |
| --- | --- |
| `Runtime source_profile` | metadata 来源 profiling 子目录或来源标签 |
| `Runtime actual_seq_lengths_shape` / `Runtime actual_seq_lengths_values` | query 序列长度 metadata |
| `Runtime actual_seq_lengths_kv_shape` / `Runtime actual_seq_lengths_kv_values` | KV 序列长度 metadata |
| `Runtime avg_seq_len` | attention lookup 和插值使用的平均 KV 序列长度 |
| `Runtime block_table_shape` / `Runtime block_table_valid_blocks` | paged cache block table metadata |
| `Runtime num_heads` / `Runtime num_key_value_heads` | 运行时 head 配置 |
| `Runtime sparse_mode`, `Runtime input_layout`, `Runtime block_size` | FIA 执行模式字段 |
| `Runtime attn_state`, `Runtime kv_cache_mode` | 可选状态/cache mode 字段 |
| `Runtime metadata_completeness` | 完备度标签，例如 `profile_shapes_only`、`runtime_values`、`runtime_values_dumped` 或 `shape_grid_scene_generated` |

parser 可以从 profiling bundle 中恢复 FIA shape 级 metadata。完整 runtime 值需要 JSONL dump 和 `fill_fia_runtime_metadata.py`。当这些值缺失时，replay 脚本会回退到启发式推断；这种行仍然有用，但不能被视为对整网运行状态的完全等价复现。

#### 2.1.4 通信 CSV 契约

通信数据保存在 HCCL 目录下，每种 HCCL kernel 一个 CSV。标准文件名为：

- `hcom_allGather_.csv`
- `hcom_allReduce_.csv`
- `hcom_alltoallv_.csv`
- `hcom_reduceScatter_.csv`

通信 CSV 列分为运行时必需列和采集标准列。`ProfilingDataSource` 运行时必需 `message_bytes`、`num_devices` 以及一个可用 latency 列；`topology_tier` 在 CSV 存在且 `DeviceProfile.comm_grid` 可解析时参与匹配；`dtype` 和 `bandwidth_gbps` 是采集标准列，用于审计和校验，lookup 不按 dtype 过滤。`communication_fallback` 记录期望 fallback 策略；实际 MISS fallback 由上层 empirical 模型处理。

| 列 | 级别 | 类型 | 含义 |
| --- | --- | --- | --- |
| `message_bytes` | 运行时必需 | int | 每 rank 消息大小，用于查询和插值 |
| `num_devices` | 运行时必需 | int | 参与通信的 rank 数 |
| `Average Duration(us)` / `Profiling Average Duration(us)` / `Duration(us)` | 运行时必需其一 | float | 通信时延；运行时按通用 latency 列优先级选择 |
| `topology_tier` | 条件匹配 | int | 设备拓扑层级；当可解析 topology 时参与匹配 |
| `dtype` | 采集标准 | string | 测量 dtype，例如 `DT_BF16` 或 `INT8` |
| `bandwidth_gbps` | 采集标准 | float | 派生带宽，用于审计和校验 |

`ProfilingDataSource` 会先查找完全匹配 `message_bytes` 和 `num_devices` 的通信行；当 `topology_tier` 可解析且 CSV 中存在该列时，`topology_tier` 也必须匹配。没有完全匹配时，它会在相同 `num_devices` 和可用 topology tier 内寻找左右 bracket 行，用这些匹配行拟合 alpha-beta latency model，并把预测时延 clamp 到 bracket latency bounds 之间。如果没有 `DeviceProfile.comm_grid`，则跳过 topology 过滤，此时 CSV 应避免同一 message/device count 下存在歧义行。

#### 2.1.5 `op_mapping.yaml` 与工具链关系

`op_mapping.yaml` 同时是运行时查询契约和工具输入。它把 TensorCast 算子映射到 NPU profiling kernel type，并记录每个映射的证据链。

关键顶层字段：

| 字段 | 用途 |
| --- | --- |
| `version`, `device`, `cann_version`, `pytorch_version`, `op_plugin_version`, `collection_date` | 标识软件栈和采集批次 |
| `communication_data_ref` | 指向 HCCL 通信 CSV 的相对路径 |
| `communication_fallback` | 记录期望 fallback 策略；实际 MISS fallback 由上层 empirical 模型处理 |
| `interpolation_policy` | 可选插值行为，例如 FIA 的平方根变换 |
| `operator_mappings` | 从 TensorCast op 名称到 kernel 查询规则的运行时映射 |
| `torch_npu_reference` | 用于辅助创建 replay 脚本的参考 API |

重要 `operator_mappings` 字段：

| 字段 | 含义 |
| --- | --- |
| `kernel_type` | 主 `{KernelType}.csv` 文件名 |
| `alternate_kernel_types` | 版本漂移或 kernel 变体的候选 CSV 名 |
| `category: communication` | 将查询路由到通信 lookup |
| `query_mode` | 使用特殊查询逻辑，如 `attention_special`、`elementwise` 或 `moe_fused` |
| `composite`, `sub_kernels`, `decomposer` | 将一个 TensorCast op 建模为多个 NPU kernel |
| `tc_input_count` | TC 与 NPU 签名不一致时截断 TC 输入参与匹配 |
| `zero_cost` | 将 shape-only 或被融合吸收的 op 标记为测量零耗时 |
| `accepted_miss` | 记录预期 miss，避免其阻塞校验 |
| `notes` | 证据链和 review 上下文 |

首期方案不要求自动消费 `torch_npu_reference.{KernelType}.microbench_api` 并生成 `op_replay/<KernelType>_run.py`。Replay 脚本仍由人工维护；新增或更新脚本时应使用 `docs/perf_database/skills/microbench/SKILL.md` 中的 Microbench Run Script Generator 工作流，按 CSV 行、`op_mapping.yaml` 中的 `microbench_api`、上游接口文档/测试和本目录 replay 约定生成脚本。未来可以在该工作流基础上扩展自动 generator，但本 RFC 将 `torch_npu_reference` 视为辅助 metadata，而不是首期自动化路径。

#### 2.1.6 原始 Profiling Parser

`tools/perf_data_collection/parsers/parse_kernel_details.py` 是把 Ascend profiling 输出转换成按 kernel 拆分 CSV 的入口。

Parser 行为：

- `--profiling-path` 可以是单个 `kernel_details*.csv` 文件，也可以是 profiling 目录。
- 当输入为目录时，递归扫描文件名包含 `kernel_details` 的 CSV。
- 同时发现 `operator_details.csv` 和 `trace_view.json`，用于 FIA bundle 检查和后续 enrich。
- 要求 kernel detail 中存在 `Type`、`OP State`、`Accelerator Core`、shape/dtype/format 列、`Duration(us)` 以及 AIC/AIV 计数器列。
- 规范化已知变体名，目前包括 `split_qkv_rmsnorm_rope_kernel_0` 到 `split_qkv_rmsnorm_rope_kernel`、`muls_add_kernel_1` 到 `muls_add_kernel`。
- 按 `(normalized Type, Input Shapes, Output Shapes)` 聚合。
- 输出平均、median、标准差耗时列，以及平均硬件计数器。
- 将每个算子类型写成一个 `{KernelType}.csv` 到目标数据库目录。

Parser 保持保守：不推断缺失 tensor 槽位，也不改写 API 语义；它保留 profiling 表示，把 shape/API 对齐留给 `op_mapping.yaml`、`generate_shape_grid.py` 和 replay 脚本。

#### 2.1.7 FIA Runtime Metadata Enrichment

FIA enrich 分两层：

1. `parsers/parse_kernel_details.py` 可以从 profiling bundle 中附加 shape 级 metadata，并将行标记为 `Runtime metadata_completeness=profile_shapes_only`。
2. `tools/perf_data_collection/fill_fia_runtime_metadata.py` 可以把 runtime JSONL dump 合并进 `FusedInferAttentionScore.csv`。

JSONL backfill 使用 LEFT JOIN 语义：

- CSV key：query/key/value shapes、KV 序列项个数、attention mask shape、block table shape。
- JSONL key：对应的 runtime tensor metadata。
- 如果多个 runtime record 匹配同一 CSV 行，则行会展开为 1:N。
- 匹配行写入可配置 metadata tag，默认是 `runtime_values_dumped`。

JSONL record 最小 join 字段为：`query_shape`、`key_shape`、`value_shape`、`actual_seq_lengths_kv`、`atten_mask_shape`、`block_table_shape`。可选 payload 字段包括 `actual_seq_lengths`、`block_table_valid_blocks`、`num_heads`、`num_key_value_heads`、`sparse_mode`、`input_layout`、`block_size`。每行是一个 JSON object，例如：

```json
{"query_shape":[8192,16,128],"key_shape":[8192,16,128],"value_shape":[8192,16,128],"actual_seq_lengths_kv":[40,998],"atten_mask_shape":null,"block_table_shape":null,"num_heads":16,"num_key_value_heads":2,"sparse_mode":0,"input_layout":"TND","block_size":128}
```

该设计把 metadata 的不完整性显式化。只有 shape 的 FIA 行仍然可用，但校验报告应将 `profile_shapes_only` 行视为低于真实 runtime 序列值行的置信度。

#### 2.1.8 Shape Grid 生成

`tools/perf_data_collection/generate_shape_grid.py` 向数据库 CSV 追加 theory 生成行。它不替代真实 profiling 数据，也不会自己产生测量耗时。

Shape grid 行为：

- `grid_generator/config.yaml` 将 kernel type 分配给 theory pattern。
- 模板 pattern 覆盖 GEMM、量化 matmul、elementwise、norm、quantization、RoPE、sampling、KV cache 和 shape manipulation 等类别。
- 对 fused attention、grouped matmul 和 `DispatchFFNCombine` 有复杂 Python generator。
- `--target-models` 使用已知模型配置裁剪 GEMM `(N, K)` 候选。
- `--rows` 限制每个 CSV 的行数；`--seed` 保证抽样可复现。
- `--max-hbm-gb` 会过滤估算输入/输出 tensor 超出内存预算的生成行。
- 没有匹配 theory generator 的文件会被跳过。

生成行应继承源行中的稳定结构 metadata，并将性能值留空或置零。只有经过 `start_microbench.py` 或后续 profiling 导入填充耗时后，它们才是可推广使用的生产数据。

#### 2.1.9 `op_replay` 逐 kernel replay 脚本框架

`tools/perf_data_collection/op_replay/` 存放按 kernel 编写的 replay 实现。每个已覆盖 kernel 应提供匹配的 `*_run.py` 脚本，用于覆盖 matmul、量化、FIA、KV cache、softmax、sort、transpose 和 vLLM-Ascend 自定义 kernel 等类别。

共享约定：

- `common.py` 负责数据库路径解析、软件栈版本目录命名、dtype/format 解析、tensor 构造、repeat count 处理和无效行跟踪。
- `replay_framework.py` 提供适用于标准 API 型 kernel 的 `OpReplay` helper。
- `run_all_op.py` 发现 `*_run.py`，支持 `--execution-mode inprocess` 以便由单个外层 `msprof` session 采集，并写出 `run_all_op_status.json`。
- 每个算子脚本读取匹配的 `{KernelType}.csv`，在 NPU 上 replay 每一行，打印简洁 `[OK]` 信息，并在构造 replay case 失败时删除无效行。
- 当未传 `--repeat-count` 时，使用 `common.py` 中的代码默认值（`30`）；repeat count 仅通过 CLI 配置，不再支持 `MSMODELING_OP_REPLAY_REPEAT_COUNT` 环境变量。
- 新增或修改 `<KernelType>_run.py` 时，应遵循 Microbench Run Script Generator skill 的核心步骤：读取目标 CSV，定位 `torch_npu_reference.<KernelType>.microbench_api`，从上游 repo 文档/测试确认真实 API，推断缺失的非 tensor 参数，复用 `common.py` / `replay_framework.py` 约定，并至少通过 `py_compile` 与 `--help` 校验。

覆盖限制：replay 覆盖可以逐步扩展。CSV 可能存在但没有对应 replay 脚本；部分脚本只支持其面向的已记录 shape/API 模式。自定义算子还要求正确设置 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`。常规 CI 中依赖 NPU 的 replay 测试必然有限；大部分测试覆盖 import、CLI help、纯解析逻辑和回写单元逻辑。

#### 2.1.10 `start_microbench.py` 的 `msprof` 编排与回写

`tools/perf_data_collection/start_microbench.py` 是 replay 测量回写的生产入口。

流程：

1. 通过 `--database-path` 或 device/version 参数解析目标数据库目录。
2. 校验选中的算子和自定义 OPP 所需环境变量。
3. 执行 `msprof --output=... python op_replay/run_all_op.py --execution-mode inprocess`。
4. 定位生成的 `PROF_*` 目录和 `op_summary_*.csv` 文件。
5. 按 operator type 和 signature 聚合行。
6. 将 `Average Duration(us)` 和 `MicroBench ...` 计数器列回写到匹配的数据库 CSV 行。
7. 在 `reports/` 子目录下生成 markdown 更新报告和 duration gap CSV。

行匹配使用：

```text
Input Shapes
Input Data Types
Input Formats
Output Shapes
Output Data Types
```

`DispatchFFNCombine` 还会把 `EP Size` 纳入 signature。更新模式为：

| 模式 | 行为 |
| --- | --- |
| `all` | 更新所有匹配行，并追加未匹配的 profiling 样本 |
| `missing-only` | 只 replay/fill microbench 与 profiling 耗时都无效的行；未匹配样本只报告不追加 |

`--prof-path` 可以直接解析已有 `PROF_*` 目录而不启动 `msprof`。`--prune-empty-duration-rows` 会删除 replay 和 profiling 耗时在回写后仍然无效的行。

计时口径限制：`Average Duration(us)` 是从 `op_summary_*.csv` 的 task duration 聚合得到的 microbench 值。它是运行时 lookup 的优先 latency，但对比报告仍应保留 `Profiling Average Duration(us)`，用于暴露 replay 与整网 profiling 的 gap。

#### 2.1.11 HCCL 通信 Benchmark

通信采集与计算 replay 分离，因为 HCCL 时延依赖 rank group 大小和拓扑。

`tools/perf_data_collection/comm_bench/generate_comm_microbench.py` 支持：

- 算子：all-reduce、all-gather、reduce-scatter、all-to-all
- 通过 `--bytes-grid` 指定 message bytes 网格
- `--num-devices`
- `--topology-tier`
- 通过 `--grid-shape` 指定硬件拓扑
- dtype 选择
- `--bench-mode kernel` 或 `--bench-mode event`
- 通过 `--do-run` 直接执行
- 通过 `--output-dir` 写 per-op 文件，或通过 `--output-csv` 写单个文件

`tools/perf_data_collection/comm_bench/run_comm_bench.sh` 是批处理包装脚本。`validate_comm_alignment.py` 使用可配置 ratio tolerance 检查测量行与 parser/query 模型是否对齐。

限制：

- 查询约定使用小写 `hcom_` 前缀，例如 `hcom_reduceScatter_.csv`；CamelCase 图编译名只能作为版本漂移线索，不能作为 HCCL CSV 主文件名。
- 通信 dtype 会记录在 CSV 中，但 lookup 路径主要按 message size、device count 和 topology tier 匹配。
- 如果无法从 `DeviceProfile` 解析 topology tier，通信数据需要按目录和行组织避免歧义。

#### 2.1.12 校验与报告

校验应覆盖三个层级：

| 层级 | 检查内容 | 代表命令 |
| --- | --- | --- |
| 静态/工具链 | Python 语法、CLI help、schema 单元测试、shape parser 测试、HCCL benchmark 纯逻辑测试 | `pytest tests/tools/test_op_replay.py tests/tools/test_op_replay_common.py tests/tools/test_start_microbench.py tests/tools/test_generate_shape_grid.py tests/tools/test_fia_parser_backfill.py tests/tools/test_generate_comm_microbench.py tests/tools/test_validate_comm_alignment.py` |
| 数据库/查询 | `op_mapping.yaml` schema、`ProfilingDataSource` lookup、插值、FIA enriched lookup | `pytest tests/perf_database` |
| 硬件依赖 | NPU replay、HCCL benchmark、从真实 `msprof` 输出回写 | 在 NPU 主机上运行 `pytest -m npu` 和选定 `start_microbench.py` / `comm_bench` 命令 |

每次数据刷新都应产出：

- 新建或更新的 CSV 文件列表。
- 来自 `run_all_op_status.json` 的 replay 状态摘要。
- 来自 `start_microbench.py` 的 profile update report。
- 当 replay 和整网 profiling 都有耗时时，产出 duration-gap hotspot CSV。
- 被跳过的算子或缺失 replay 脚本列表。
- 仍然没有有效耗时的生成行列表。

校验不应掩盖部分覆盖情况。Accepted miss、zero-cost op、插值、partial composite hit 和 FIA metadata 完备度都必须在报告或 CSV metadata 中保持可见。

#### 2.1.13 工具 CLI 契约

| 工具 | 必需输入 | 主要输出 | 关键失败模式 |
| --- | --- | --- | --- |
| `parsers/parse_kernel_details.py` | `--profiling-path`；输出目录用 `--database-path` 或 device/version 参数推导 | per-kernel `{KernelType}.csv` | 输入缺少必需 profiling 列时报错 |
| `fill_fia_runtime_metadata.py` | `--csv-path`、`--jsonl-path` | enriched `FusedInferAttentionScore.csv`；默认原地覆盖，可用 `--output-path` | CSV/JSONL 不存在、CSV header 为空或 JSONL 非法时报错 |
| `generate_shape_grid.py` | `--data-dir`，或 `--device --vllm-version [--torch-version --cann-version]` | 向 CSV 追加 theory shape 行 | data dir 不存在或无 CSV 时报错；无 generator 的 CSV 被跳过并报告 |
| `start_microbench.py` | `--database-path`，或 device/version 参数；可选 `--prof-path` | 回写 `Average Duration(us)` 和 `MicroBench ...` 计数器，生成 `reports/` | `msprof` 不存在、OPP 环境缺失、无 `PROF_*` / `op_summary_*.csv`；重复 signature、unmatched rows 和 gap 进入报告 |
| `op_replay/run_all_op.py` | 数据库路径参数；可选 `--op` | 执行匹配的 `*_run.py`，写 `run_all_op_status.json` | 单算子失败；默认 fail-fast，`--continue-on-error` 可继续执行剩余脚本 |
| `comm_bench/generate_comm_microbench.py` | `torchrun ... --do-run` | `hcom_*.csv` | 未传 `--do-run` 不会执行采集；`--output-csv` 只允许单个 `--ops` |
| `comm_bench/validate_comm_alignment.py` | `--csv-dir` | PASS/WARN/FAIL 对齐报告；FAIL 时退出码非 0 | `--csv-dir` 不是目录时报错；ratio 超过 tolerance 会失败；格式异常行不会计入有效行，review 时应检查 0-row 报告 |

#### 2.1.14 人工 Review Checklist

数据刷新或工具链变更提交给 human review 前，应至少提供以下信息：

- 目标软件栈：device、vLLM-Ascend、PyTorch、CANN、op-plugin、collection date。
- 原始输入来源：profiling bundle 名称或路径摘要、是否包含 `operator_details.csv` / `trace_view.json`、是否包含 FIA runtime JSONL。
- CSV schema 变更：新增、删除或重命名的列；是否保留空 tensor 槽位；是否仍满足 latency 列优先级。
- `op_mapping.yaml` 变更：新增/修改的 mapping、证据链、`alternate_kernel_types`、`tc_input_count`、`accepted_miss`、`zero_cost`、composite/decomposer 影响。
- Shape grid 变更：使用的 `--target-models`、`--rows`、`--seed`、`--max-hbm-gb`，以及 generated-only 行是否已经 replay。
- Replay 结果：`run_all_op_status.json` 摘要、profile update report、duration gap hotspot CSV、无效行和重复 signature 列表。
- HCCL 结果：`hcom_*.csv` 文件列表、`num_devices` / `topology_tier` 覆盖、`bench-mode`、`validate_comm_alignment.py` tolerance 和结果。
- 已知限制：缺失 replay 脚本、FIA metadata 不完整、只用于 staging 的 generated 目录、硬件依赖测试未执行原因。

#### 2.1.15 数据生命周期与版本管理

推荐生命周期：

1. 创建或选择一个软件栈对应的计算目录。
2. 使用 `parsers/parse_kernel_details.py` 导入原始 profiling。
3. 在同一软件栈目录添加或更新 `op_mapping.yaml`。
4. 如果有 JSONL runtime 值，则 enrich FIA metadata。
5. 在 staging copy 或命名清晰的 generated 目录中生成 shape 行。
6. 先使用 `start_microbench.py --update-mode missing-only` replay 选定算子。
7. Review 回写报告、missing shapes、无效行、重复 signature 和 duration gaps。
8. 将 CSV/YAML 变化与 changelog 或报告一起提交，完成目录 promotion。

版本管理规则：

- 按 `{version_dir}` 隔离软件栈变化。不要静默混用不同 vLLM-Ascend、PyTorch 或 CANN 版本的数据。
- HCCL 数据保存在 `{device}/hccl/{cann_version}` 下，并通过 `communication_data_ref` 复用。
- 当数据库目录接收新的 profiling 导入或较大 replay 刷新时，更新 `collection_date`。
- 版本目录一经发布应保留，除非有明确废弃公告。
- generated-only 目录应通过命名避免用户误认为它是完整测量数据。
- 除非 review 明确需要，不提交原始 `PROF_*` 目录；提交派生 CSV 和报告制品即可。

### 2.2 替代方案

#### 方案 A：仅使用整网原始 Profiling

只导入整网 profiling 行，不生成 shape grid，也不做 microbenchmark replay。

该方案简单且最贴近真实 workload，但 shape 覆盖稀疏。它无法补齐重要的邻近 shape，也会让运行时 lookup 过度依赖某次被采集的模型/profile。

#### 方案 B：完全从 `torch_npu_reference` 生成 Replay 脚本

读取 `op_mapping.yaml`，并基于 `torch_npu_reference.{KernelType}.microbench_api` 自动生成所有 `op_replay/<KernelType>_run.py`。

这是长期上更理想的方向，但不纳入首期范围。许多 kernel 需要非 tensor 参数、自定义 OPP 设置、槽位保留、合法 cache tensor，或版本相关 API 行为，这些信息不能从单个 API 字符串安全推断。

#### 方案 C：使用外部数据库

将 CSV/YAML 数据迁移到 SQLite、DuckDB 或服务型数据库。

这会增强索引和 schema 校验，但在首期会引入部署和 review 成本。推荐方案让运行时 data source 和 review 流程围绕文件制品构建。

#### 方案 D：用解析模型替代测量

避免 replay 采集，对计算和通信都使用解析公式。

解析模型适合作为 fallback，尤其适合通信，但它无法足够准确地捕捉 fusion、CANN kernel 选择、tensor format 和自定义 op 行为。

### 2.3 方案分析

#### 优点

- 保持数据生产离线、可重复、可 review。
- 采用运行时契约：`op_mapping.yaml` 加按 kernel 拆分的 CSV 文件。
- 将原始 profiling 导入、metadata enrich、shape 扩展、replay 测量和校验拆成职责明确的工具。
- 允许部分数据显式存在，而不是阻塞整个数据库刷新。
- 可在兼容的计算栈目录之间复用 HCCL 测量。
- 同时支持精确测量行，以及可逐步补齐的生成行。

#### 缺点与限制

- Replay 脚本需要人工维护；replay 覆盖是逐步扩展的。
- 首期没有自动消费 `torch_npu_reference` 的 generator。
- 常规 CI 中依赖 NPU 的 replay 与通信测试覆盖有限。
- CSV schema 便于 diff，但也容易被手工编辑破坏。
- 当 runtime sequence metadata 缺失时，FIA replay 仍然可能只是近似复现。
- Shape grid 可能生成表面合法但特定 operator API 无法 replay 的行。

## 3. 实施计划

### 3.1 Rollout 计划

时间安排采用按阶段排序的方式。Phase 0-2 可在任意开发主机上完成；Phase 3-4 依赖准备好的 NPU 主机可用窗口；Phase 5 只有在 replay 和通信报告产出后才开始。

| 里程碑 | Owner 角色与大致顺序 | 范围 | 退出条件 |
| --- | --- | --- | --- |
| Phase 0：RFC 与契约 | RFC owner；实现工作前的第一阶段 | 采用本 RFC 作为独立采集工具链契约 | 贡献者不依赖临时文档即可执行流程 |
| Phase 1：Parser/schema baseline | Tooling owner；Phase 0 后立即执行的无 NPU 阶段 | 校验原始解析、输出列顺序和软件栈目录命名 | 无 NPU parser 与 data-source 测试通过 |
| Phase 2：Shape-grid staging | Data collection owner；Phase 1 后的无 NPU staging 阶段 | 使用 `--rows`、`--seed`、`--max-hbm-gb` 在 staging 目录生成行 | 生成行保留槽位/dtype/format 契约，并报告 skipped kernels |
| Phase 3：Replay writeback | Data collection owner 与 NPU host owner；在 NPU 主机可用时排期 | 在 NPU 主机上通过 `start_microbench.py` 运行选定 `op_replay` 脚本 | 填充 `Average Duration(us)`，报告无效行，并生成 gap 报告 |
| Phase 4：通信采集 | HCCL/NPU owner；与相同或相邻 NPU 主机窗口一起排期 | 采集或刷新目标 CANN 版本的 HCCL CSV | `validate_comm_alignment.py` 在约定 tolerance 内通过 |
| Phase 5：Promotion | Data reviewer 与 runtime owner；Phase 3-4 报告 review 后执行 | Review CSV/YAML/report 变化并提升数据库目录 | 运行时 lookup 测试通过，相关限制在刷新报告中记录 |

### 3.2 测试计划

在任何硬件采集前先运行无 NPU 检查：

```bash
python -m pytest tests/tools/test_op_replay.py \
  tests/tools/test_op_replay_common.py \
  tests/tools/test_start_microbench.py \
  tests/tools/test_generate_shape_grid.py \
  tests/tools/test_fia_parser_backfill.py \
  tests/tools/test_generate_comm_microbench.py \
  tests/tools/test_validate_comm_alignment.py

python -m pytest tests/perf_database
```

对变更过的 replay 脚本运行语法和 help 检查：

```bash
python -m py_compile tools/perf_data_collection/op_replay/<KernelType>_run.py
python tools/perf_data_collection/op_replay/<KernelType>_run.py --help
```

仅在准备好的 NPU 主机上运行硬件依赖检查：

```bash
python tools/perf_data_collection/start_microbench.py \
  --database-path tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version_dir} \
  --op MatMulV2 RmsNorm \
  --repeat-count 1 \
  --update-mode missing-only

bash tools/perf_data_collection/comm_bench/run_comm_bench.sh \
  tensor_cast/performance_model/profiling_database/data/{device}/hccl/{cann_version}
```

### 3.3 后续工作

- 增加仓内 replay 脚本 generator，消费 `torch_npu_reference.{KernelType}.microbench_api` 和算子专用 metadata。
- 在回写前加入更严格的 CSV schema 校验和重复 signature 检查。
- 扩展带 NPU marker 的集成测试，覆盖代表性自定义算子。
- 改进 FIA runtime metadata 捕获，减少依赖 replay 启发式推断的行。
- 增加单一 collection 命令，在保持本文件契约的前提下编排 parse、enrich、shape generation、replay、通信校验和报告打包。

### 3.4 完成标准

- `tools/perf_data_collection/` 入口、数据库目录、CSV schema 和 `op_mapping.yaml` 契约与本文一致。
- 至少一个目标软件栈目录可以通过 parse、可选 enrich、shape generation、replay writeback 和报告流程生成。
- HCCL 目录可以通过 communication benchmark 工具生成，并能被 `communication_data_ref` 引用。
- 无 NPU 测试覆盖 parser、shape grid、FIA backfill、start_microbench 回写逻辑、communication CSV 校验和 data-source 查询。
- NPU 主机上可运行选定 replay 与 communication benchmark，并产出可供 human review 的状态报告和 gap 报告。

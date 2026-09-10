# GLM 5.2 实测算子仿真接入设计

## 修订记录

| 日期         | 修订版本 | 修改描述                                    | 作者  | RFC文档 |
| ---------- | ---- | --------------------------------------- | --- | ----- |
| 2026-08-29 | 1.0  | 初稿完成，归档 GLM 5.2 实测算子仿真的高频算子数据库接入与精度验收设计 | -   | -     |

## 背景描述

实测算子仿真建模基于 profiling 性能数据库记录每个算子的真实性能数据，通过 `EmpiricalPerformanceModel` 在算子级别命中实测 latency 并累加，从而推测整个模型的端到端仿真结果。相比 Roofline 理论模型（`AnalyticPerformanceModel`），实测建模能够覆盖特定硬件平台、复杂算子实现与组合场景下的真实性能，显著缩小估算与实测的偏差。

当前场景现状与诉求如下：

1. TensorCast 已完成 GLM5 系列模型的**结构适配**（见 [glm5-tensorcast-adaptation-design.md](glm5-tensorcast-adaptation-design.md)）：`glm_moe_dsa` 模型画像、Sparse MLA/DSA Indexer 语义算子、`tensor_cast.dsa_indexer.default` 与 `tensor_cast.mla_sparse_attention.default` 均已落地，且已支持 `zai-org/GLM-5.1` 的实测仿真建模。
2. `zai-org/GLM-5.2` 沿用同一 `model_type="glm_moe_dsa"` 与同一套算子集合（sparse attention、DSA indexer、MLA、MoE、MTP），仅结构上引入 IndexShare——`indexer_types` 区分 `full`（执行 indexer 生成 top-k）与 `shared`（复用前序 `full` layer 的 top-k）。因此 GLM5.2 不存在新的、GLM5.1 未覆盖的算子类型，其实测建模的关键差异落在**算子频次语义**与**高频实测数据库覆盖**上。
3. GLM5.2 在vllm-ascend 0.23.0rc1版本上支持，以目前仓上的实测算子性能数据库无法满足GLM 5.2的硬件配套，需要在新版软件栈（CANN 9.0.1 / vLLM 0.23.0rc1 / PyTorch 2.10.0）下构建 profiling 实测算子数据库，支持实测算子仿真建模。

本特性的目标是：在已完成的 GLM5.2 结构适配基础上，补齐 GLM5.2 **实测算子仿真**能力，同时新增 GLM5.2 相关的高频实测算子数据库，最终达到既定精度验收标准。

## 方案设计

### 整体设计思路

GLM5.2 的实测接入不新增加一套模型结构或性能模型，而是复用既有三层能力：

1. **模型识别与算子语义层（复用）**：直接沿用 `glm_moe_dsa` 模型画像（[glm5.py](../../tensor_cast/transformers/builtin_model/glm5.py)）与 `tensor_cast.dsa_indexer.default`、`tensor_cast.mla_sparse_attention.default` 等语义算子。GLM5.2 通过 `indexer_types` 表达 IndexShare，其 `full`/`shared` 语义已在结构适配中闭环。
2. **实测性能模型层（复用）**：沿用 `EmpiricalPerformanceModel` + `ProfilingDataSource` / `InterpolatingDataSource`，通过 `op_mapping.yaml` 将 TensorCast 语义算子映射到 NPU profiling 内核类型（`{KernelType}.csv`），未命中时回退 `AnalyticPerformanceModel`。
3. **高频实测数据库层（新增）**：在既有 `vllm0.18.0`/`vllm0.13.0` 等版本库之外，新增 GLM5.2 对应的新版软件栈实测数据库目录，补齐算子实测数据。

核心原则：**GLM5.2 的实测接入是"数据库 + 查询语义"维度的增量，而非结构维度的增量**。算子映射与查询策略尽量复用已有 `op_mapping.yaml`，新增能力聚焦在高频算子 CSV 覆盖率、语义唯一匹配约束与实测数据回填链路的正确性上。

### 模型识别与算子语义复用

GLM5.2 与 GLM5.1 共享 `model_profile`（`model_type="glm_moe_dsa"`），因此无需为 GLM5.2 单独注册模型画像。实测建模需要关注的语义要点为：

1. **`dsa_indexer` 仅计入 `full` source layer**：GLM5.2 的 `shared` layer 复用前序 `full` layer 的 top-k，不产生新的 `tensor_cast.dsa_indexer.default` 算子。相应地在 `op_mapping.yaml` 中，`tensor_cast.dsa_indexer.default` 已是 composite（`decomposer: true`），其 `primary_kernel_type` 为 `LightningIndexer`。只要模型结构层正确（结构适配已保证），实测层的查询次数自然与实际 `full` layer 数一致，`tensor_cast.dsa_indexer.default`共计为21个。
2. **`mla_sparse_attention` 命中 SparseFlashAttention**：GLM5.2 的 sparse attention 主路径由 `tensor_cast.mla_sparse_attention.default` / `tensor_cast.mla_sparse_attention_quant.default` 表达，其 composite decomposition 将 attention kernel 映射到 `SparseFlashAttention`（SFA），与 DeepSeek-V3.2 / GLM-5.1 的 DSA 路径一致。

### 新增 GLM5.2 在vllm0.23.0rc1_torch2.10.0_cann9.0.1软件栈下的实测算子数据库

为覆盖 GLM5.2 在vllm0.23.0rc1_torch2.10.0_cann9.0.1软件栈下的实测性能，在 profiling database 中新增 GLM5.2 对应版本的数据库目录：

```text
tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/
└── vllm0.23.0rc1_torch2.10.0_cann9.0.1/
    ├── op_mapping.yaml                 # 沿用 GLM5 稀疏 MLA 映射，版本字段对齐新软件栈
    ├── Add.csv                         # Add 实测数据
    ├── ...csv                          # 其他GLM 5.2涉及的算子实测数据
    ├── LightningIndexer.csv            # DSA indexer 实测数据
    └── SparseFlashAttention.csv        # sparse attention 实测数据
    
```

设计要点：

1. **版本化目录隔离**：数据库按 `{vllm 版本}_{torch 版本}_{cann 版本}` 组织，新软件栈单独建目录，不回写 `vllm0.18.0` 历史基线，保证可追溯与可回滚。
2. **CSV 契约一致**：`LightningIndexer.csv` 与 `SparseFlashAttention.csv` 的列结构与既有 `{KernelType}.csv` 契约一致，并包含 `Runtime case_id`、`Profiling Average Duration(us)` 等待查询/回填所依赖的字段。

### 精度验收与场景设计

验收在 A3（`ATLAS_800_A3_752T_128G_DIE`）上分别执行 Prefill、Decode 仿真，覆盖一组短序列与一组长序列场景，与已提供的 profiling 实测数据对比，在「相同配置 + 相同特性优化」下，性能耗时平均误差 < 15%。

| 场景  | 输入长度 | 输出长度 | 阶段               |
| --- | ---- | ---- | ---------------- |
| 短序列 | 3.5k | 1.5k | Prefill / Decode |
| 长序列 | 64k  | 1k   | Prefill / Decode |

2个case，4个场景的平均误差精度小于15%。「相同配置 + 相同特性优化」指：TP/EP/DP 并行策略一致、量化与特性开关（MTP、repetition、`--compile`、IndexShare）与 profiling 采集环境一致。

## 使用说明

### CLI 接口

GLM5.2 实测仿真复用 `text_generate` 的 profiling 性能模型能力（见 [profiling_driven_empirical_performance_model.md](profiling_driven_empirical_performance_model.md)），在结构适配命令基础上追加 `--performance-model profiling` 与 `--profiling-database`，具体命令如下：

```bash
输入3.5k输出1.5k的prefill仿真命令:
python -m cli.inference.text_generate zai-org/GLM-5.2 --num-queries 1 --query-length 3596 --context-length 0 --device ATLAS_800_A3_752T_128G_DIE --tp-size 16 --ep-size 16 --num-devices 16 --quantize-linear-action W4A8_STATIC --performance-model profiling --compile --profiling-database tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.23.0rc1_torch2.10.0_cann9.0.1/ --word-embedding-tp row --enable-shared-expert-tp

输入3.5k输出1.5k的decode仿真命令:
python -m cli.inference.text_generate zai-org/GLM-5.2 --num-queries 1 --query-length 1 --context-length 3585 --device ATLAS_800_A3_752T_128G_DIE --tp-size 16 --ep-size 16 --num-devices 16 --quantize-linear-action W4A8_STATIC  --performance-model analytic --performance-model profiling --compile --profiling-database tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.23.0rc1_torch2.10.0_cann9.0.1 --word-embedding-tp row --enable-shared-expert-tp --decode

输入64k输出1k的prefill仿真命令:
python -m cli.inference.text_generate zai-org/GLM-5.2 --num-queries 1 --query-length 8192 --context-length 0 --device ATLAS_800_A3_752T_128G_DIE --tp-size 16 --ep-size 16 --num-devices 16 --quantize-linear-action W4A8_STATIC --performance-model analytic --performance-model profiling --compile --profiling-database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.23.0rc1_torch2.10.0_cann9.0.1 --word-embedding-tp row  --enable-shared-expert-tp 

输入64k输出1k的decode仿真命令:
python -m cli.inference.text_generate zai-org/GLM-5.2 --num-queries 1 --query-length 1 --context-length 65536 --device ATLAS_800_A3_752T_128G_DIE --tp-size 16 --ep-size 16 --num-devices 16 --quantize-linear-action W4A8_STATIC --performance-model profiling --compile  --profiling-database tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.23.0rc1_torch2.10.0_cann9.0.1 --word-embedding-tp row  --enable-shared-expert-tp --decode 
```

关键参数说明：

| 参数                                    | 说明                                                           |
| ------------------------------------- | ------------------------------------------------------------ |
| `--performance-model profiling`       | 启用实测 `EmpiricalPerformanceModel`，命中 profiling 库时采用实测 latency |
| `--profiling-database`                | 指向包含 `op_mapping.yaml` 与 CSV 的版本库目录                          |
| `--context-length` / `--query-length` | 对应验收短序列（3.5k/1.5k）与长序列（64k/1k）场景                             |

### 配置约束

1. GLM5.2 实测建模依赖 `op_mapping.yaml` 中 `tensor_cast.dsa_indexer.default`（`LightningIndexer`）与 `tensor_cast.mla_sparse_attention.default`（`SparseFlashAttention`）的映射与语义唯一约束，不得移除。
2. 新增版本库目录必须与 `op_mapping.yaml` 顶层 `version`/`cann_version`/`pytorch_version` 字段一致，避免版本与数据不匹配导致的查询错误。
3. `LightningIndexer` / `SparseFlashAttention` CSV 必须包含 `Runtime case_id` 列，否则回填链路失败。

## 测试设计

### 单元测试

在 A3 上分别执行 Prefill、Decode，覆盖短序列（3.5k/1.5k）与长序列（64k/1k），在相同配置 + 相同特性优化下，与已提供的 profiling 实测数据对比：

1. **仿真成功**：GLM5.2 可完整仿真，无报错。
2. **精度达标**：Prefill / Decode × 短序列 / 长序列共四组场景，端到端性能平均耗时误差 < 15%。

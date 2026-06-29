# 性能数据库采集工具使用指南

本文档说明 `tools/perf_data_collection/` 下的性能数据库采集工具如何使用、需要什么环境、会生成哪些文件，以及常见问题如何处理。

主要工具按使用顺序分为 4 个：

1. `parsers/parse_kernel_details.py`：解析 profiling 输出，生成 per-operator CSV。
2. `generate_shape_grid.py`：生成 theory shape 行，扩展 CSV 覆盖。
3. `start_microbench.py`：运行 compute replay，通过 `msprof` 聚合并回写耗时。
4. `comm_bench/generate_comm_microbench.py`：采集 HCCL 通信微基准，生成 `hcom_*.csv`。

## 适用范围

本目录用于生成和刷新 TensorCast 性能数据库中的算子 CSV 数据，主要覆盖三类工作：

- 从 NPU profiling 结果中解析 per-operator CSV。
- 生成理论 shape 行并扩展 CSV 覆盖。
- 在 NPU 上 replay 算子或采集 HCCL 通信微基准，并把耗时写回数据库。

支持的设备目录名：

| 设备目录名 |
| --- |
| `ATLAS_800_A2_376T_64G` |
| `ATLAS_800_A2_313T_64G` |
| `ATLAS_800_A2_280T_64G` |
| `ATLAS_800_A2_280T_64G_PCIE` |
| `ATLAS_800_A2_280T_32G_PCIE` |
| `ATLAS_800_A3_752T_128G_DIE` |
| `ATLAS_800_A3_560T_128G_DIE` |

默认设备为 `ATLAS_800_A3_752T_128G_DIE`。通信批量采集脚本 `comm_bench/run_comm_bench.sh` 面向 **`ATLAS_800_A3_752T_128G_DIE`**，固定按 `48 8 2` 硬件网格采集。A3 系列中其他型号（如 ROCE 款）的网格形状可能不同（例如 `2 8 2`），需按实际硬件调整 `--grid-shape` 或改用 `generate_comm_microbench.py` 手动配置。

## 环境准备

1. 在昇腾环境安装配套版本的CANN Toolkit开发套件包和ops算子包并配置CANN环境变量，具体请参见[CANN快速安装](https://www.hiascend.com/cann/download)。
2. 完成 vLLM 和 vLLM-Ascend 的安装和配置并确认 vLLM-Ascend 可以正常运行，具体请参见《 [vLLM-Ascend安装指南](https://docs.vllm.ai/projects/ascend/zh-cn/latest/installation.html)》。

## 前置条件

在仓库根目录执行命令。项目要求 Python 版本不低于 `3.10`。

先激活 Python 环境，再运行下文命令：

```bash
source try/bin/activate
```

激活后使用 `python` 启动脚本（与英文 `readme_en.md` 命令一致；Windows 可选 `py -3`）。

| 场景 | 必需条件 |
| --- | --- |
| 解析 profiling | 已有 `kernel_details*.csv` 文件，或包含该文件的 profiling 目录。 |
| 生成 theory shape | 目标数据库目录中已有要扩展的 operator CSV。 |
| 运行 compute replay | NPU 可用；已安装 CANN、`msprof`、`torch`、`torch_npu`；部分自定义算子还需要 vLLM-Ascend custom OPP。 |
| 采集 HCCL 通信数据 | 使用 `torchrun` 启动；NPU/HCCL 通信环境可用。 |

数据库目录可以直接通过 `--database-path` 指定，也可以通过 `--device`、`--vllm-version`、`--torch-version`、`--cann-version` 自动推导。

示例数据库目录（仓库内已有路径）：

```text
tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5
tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5_shape_generated
```

前者为 profiling 解析后的基础数据库；后者为已追加 theory shape 行的扩样数据库。

## 目录结构

```text
tools/perf_data_collection/
  comm_bench/              # HCCL 通信微基准采集
  grid_generator/          # theory shape 生成核心逻辑
  op_replay/               # compute 算子 replay 框架与算子入口
  parsers/                 # profiling 解析入口
  fia_common.py            # FIA shape 与 metadata 共享工具
  generate_shape_grid.py   # theory shape 生成入口
  memory_estimator.py      # HBM 内存估算
  start_microbench.py      # msprof 编排、聚合与回写入口
  readme.md                # 中文说明
  readme_en.md             # 英文说明
```

## 推荐流程

### 1. 解析原始 profiling 输出

```bash
python tools/perf_data_collection/parsers/parse_kernel_details.py \
  --profiling-path /path/to/profiling_dir \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5
```

执行后，目标数据库目录下会生成或更新 per-operator CSV，例如 `MatMulV2.csv`、`FusedInferAttentionScore.csv`。

### 2. 生成 theory shape 行

```bash
python tools/perf_data_collection/generate_shape_grid.py \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5_shape_generated \
  --target-models deepseek-ai/DeepSeek-V3,Qwen/Qwen3-32B \
  --rows 2000 \
  --seed 20260409
```

该步骤会在已有 CSV 中追加理论生成的 shape 行，用于扩大 replay 覆盖。`--rows 0` 表示不限制每个 CSV 追加行数。

### 3. Replay 算子并回写耗时

```bash
python tools/perf_data_collection/start_microbench.py \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5_shape_generated \
  --repeat-count 1 \
  --update-mode missing-only
```

`start_microbench.py` 会通过 `msprof` 运行 `op_replay/run_all_op.py`，聚合 `op_summary_*.csv`，再把耗时写回数据库 CSV。

### 4. 采集 ATLAS_800_A3_752T_128G_DIE HCCL 通信数据

```bash
bash tools/perf_data_collection/comm_bench/run_comm_bench.sh ./hccl_bench_data
```

该脚本面向 **`ATLAS_800_A3_752T_128G_DIE`**（默认 `grid-shape 48 8 2`），输出 `hcom_*.csv`。其他 A3 型号请确认实际网格形状后再采集；如需更细控制，也可以直接使用 `generate_comm_microbench.py`。

## 主要工具

本节介绍性能数据库采集链路中的主要入口。它们覆盖 compute 算子的解析、扩样、回写，以及 HCCL 通信数据采集。

| 阶段 | 工具 | 主要输入 | 主要输出 | 适用场景 |
| --- | --- | --- | --- | --- |
| 解析 | `parsers/parse_kernel_details.py` | `kernel_details*.csv` 或 profiling 目录 | per-operator CSV | 把真实 profiling 结果转换成数据库格式。 |
| 扩样 | `generate_shape_grid.py` | 已有数据库 CSV、模型/shape 规则 | 追加 theory shape 行后的 CSV | 扩大 shape 覆盖，准备后续 replay。 |
| 回写 | `start_microbench.py` | 数据库 CSV、算子 replay 脚本 | 回写耗时后的 CSV、报告文件 | 在 NPU 上重新测量算子耗时并刷新数据库。 |
| 通信采集 | `comm_bench/generate_comm_microbench.py` | HCCL 通信配置、消息大小网格 | `hcom_*.csv` | 采集通信算子的性能数据库数据。 |
| 批量通信采集 | `comm_bench/run_comm_bench.sh` | `ATLAS_800_A3_752T_128G_DIE` 硬件网格（`48 8 2`）和输出目录 | 一组标准 `hcom_*.csv` | 按 752T DIE 推荐配置批量采集通信数据。 |

### 1. Profiling 解析入口：`parsers/parse_kernel_details.py`

`parse_kernel_details.py` 是采集链路的入口，用于把 NPU profiling 产物转换成性能数据库可以直接使用的 per-operator CSV。

- 输入：单个 `kernel_details*.csv` 文件，或包含 profiling 结果的目录。
- 输出：按算子类型拆分后的 CSV，例如 `MatMulV2.csv`、`FusedInferAttentionScore.csv`。
- 适用场景：已有真实 profiling 数据，希望沉淀到性能数据库中作为后续分析、扩样或 replay 的基础。

| 参数 | 是否必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--profiling-path` | 是 | 无 | 单个 `kernel_details*.csv` 文件或 profiling 目录。 |
| `--database-path` | 否 | 自动推导 | 显式输出数据库目录。 |
| `--device` | 否 | `ATLAS_800_A3_752T_128G_DIE` | 自动推导路径时使用的设备目录名。 |
| `--vllm-version` | 否 | 无 | 自动推导路径时使用的 vLLM 版本或完整版本目录名。 |
| `--torch-version` | 否 | 无 | 自动推导路径时使用的 PyTorch 版本。 |
| `--cann-version` | 否 | 无 | 自动推导路径时使用的 CANN 版本。 |

常用命令：

```bash
python tools/perf_data_collection/parsers/parse_kernel_details.py \
  --profiling-path ./PROF_001 \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5
```

### 2. Theory Shape 扩样入口：`generate_shape_grid.py`

`generate_shape_grid.py` 用于在已有数据库 CSV 中追加 theory shape 行，解决真实 profiling 覆盖不完整的问题。

- 输入：目标数据库目录，以及可选的模型名、行数、随机种子和 HBM 预算。
- 输出：追加了 theory shape 行的 operator CSV。
- 适用场景：某些算子或 shape 还没有真实样本，需要先生成候选行，再通过 replay 补齐耗时。

| 参数 | 是否必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--database-path` | 否 | 自动推导 | 显式 CSV 根目录。 |
| `--target-models` | 否 | 全量网格 | 逗号分隔模型 ID（如 `deepseek-ai/DeepSeek-V3,Qwen/Qwen3-32B`），命名与 `text_generate` 一致，用于裁剪 GEMM `(N, K)` 候选。 |
| `--device` | 否 | `ATLAS_800_A3_752T_128G_DIE` | 自动推导路径时使用的设备目录名。 |
| `--vllm-version` | 否 | 无 | 自动推导路径时使用的 vLLM 版本。 |
| `--torch-version` | 否 | 无 | 自动推导路径时使用的 PyTorch 版本。 |
| `--cann-version` | 否 | 无 | 自动推导路径时使用的 CANN 版本。 |
| `--rows` | 否 | `1000` | 每个 CSV 最多追加的行数，`0` 表示不限制。 |
| `--seed` | 否 | 无 | 随机采样种子。 |
| `--max-hbm-gb` | 否 | `32.0` | 单行输入/输出张量 HBM 预算，`0` 表示关闭过滤。 |

### 3. 算子 Replay 回写入口：`start_microbench.py`

`start_microbench.py` 负责真正执行算子 replay/microbench，并把采集到的耗时回写到数据库 CSV 中。

- 输入：目标数据库目录和待 replay 的算子列表；未指定 `--ops` 时默认处理全部支持算子。
- 输出：更新后的 operator CSV，以及 `reports/` 下的汇总报告。
- 适用场景：已经有数据库行，但缺少有效耗时，或需要用当前环境重新测量并刷新耗时。

| 参数 | 是否必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--database-path` | 否 | 自动推导 | 显式数据库目录。 |
| `--device` | 否 | `ATLAS_800_A3_752T_128G_DIE` | 自动推导路径时使用的设备目录名。 |
| `--vllm-version` | 否 | 无 | 自动推导路径时使用的 vLLM 版本。 |
| `--torch-version` | 否 | 无 | 自动推导路径时使用的 PyTorch 版本。 |
| `--cann-version` | 否 | 无 | 自动推导路径时使用的 CANN 版本。 |
| `--ops` | 否 | 全部支持算子 | 只 replay 指定算子，支持多个算子名。 |
| `--dispatch-ffn-combine-ep-size` | 否 | `16` | `DispatchFFNCombine` replay 和行匹配使用的 EP size。 |
| `--dispatch-ffn-combine-nproc-per-node` | 否 | 无 | `torchrun`每节点进程数，msprof回放`DispatchFFNCombine`算子的配置参数。 |
| `--dispatch-ffn-combine-nnodes` | 否 | `1` | `torchrun`节点数，msprof回放`DispatchFFNCombine`算子的配置参数。 |
| `--dispatch-ffn-combine-node-rank` | 否 | `0` | `torchrun`当前节点的rank，msprof回放`DispatchFFNCombine`算子的配置参数。 |
| `--dispatch-ffn-combine-master-addr` | 否 | `127.0.0.1` | `torchrun` master 地址，msprof回放`DispatchFFNCombine`算子的配置参数。 |
| `--dispatch-ffn-combine-master-port` | 否 | 无 | `torchrun` master 端口，msprof回放`DispatchFFNCombine`算子的配置参数。 |
| `--repeat-count` | 否 | `1` | 传给 `run_all_op.py` 的 replay 重复次数。 |
| `--update-mode` | 否 | `all` | `all` 表示更新全部匹配行；`missing-only` 只填充无有效耗时的行。 |
| `--fail-fast` | 否 | `false` | 任一 replay 失败时立即停止。 |
| `--prune-empty-duration-rows` | 否 | `false` | 回写后删除 replay/profiling 耗时仍无效的行。 |

单算子调试示例：

```bash
python tools/perf_data_collection/start_microbench.py \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5 \
  --ops MatMulV2 \
  --repeat-count 1 \
  --update-mode all
```

#### 3.1 底层 Replay 共享参数

`start_microbench.py` 会通过 `op_replay/run_all_op.py` 调度 `op_replay/*_run.py`。多数单算子 replay 脚本使用同一套基础参数。

| 参数 | 是否必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--database-path` | 否 | 自动推导 | 显式数据库目录。 |
| `--device` | 否 | `ATLAS_800_A3_752T_128G_DIE` | 设备目录名。 |
| `--vllm-version` | 否 | 无 | vLLM 版本或完整版本目录名。 |
| `--torch-version` | 否 | 无 | PyTorch 版本。 |
| `--cann-version` | 否 | 无 | CANN 版本。 |
| `--repeat-count` | 否 | `30` | 每行 replay 次数；未指定时使用代码默认值，不支持环境变量覆盖。 |
| `--update-mode` | 否 | `all` | `all` 或 `missing-only`。 |

单独运行某个 replay 脚本：

```bash
python tools/perf_data_collection/op_replay/MatMulV2_run.py \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5 \
  --repeat-count 10 \
  --update-mode missing-only
```

#### 3.2 底层 Replay 调度器：`op_replay/run_all_op.py`

`run_all_op.py` 会自动发现并运行所有 `*_run.py`，通常由 `start_microbench.py` 在 `msprof` 中调用；需要调试底层 replay 时也可以单独运行。

| 参数 | 是否必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--database-path` | 否 | 自动推导 | 显式数据库目录。 |
| `--ops` | 否 | 全部算子 | 限制运行指定算子。 |
| `--repeat-count` | 否 | 不传给子脚本 | 传给每个 operator replay 脚本。 |
| `--update-mode` | 否 | `all` | 回写模式。 |
| `--execution-mode` | 否 | `subprocess` | `subprocess` 或 `inprocess`。`start_microbench.py` 使用 `inprocess`。 |
| `--continue-on-error` | 否 | `false` | 单算子失败后继续执行剩余算子。 |

#### 3.3 单算子 Replay 入口

| 文件 | Kernel Type | 说明 |
| --- | --- | --- |
| `AddRmsNormBias_run.py` | `AddRmsNormBias` | Replay fused AddRmsNormBias。 |
| `Add_run.py` | `Add` | Replay elementwise add。 |
| `ArgMaxV2_run.py` | `ArgMaxV2` | Replay argmax。 |
| `AscendQuantV2_run.py` | `AscendQuantV2` | Replay Ascend quantize。 |
| `BatchMatMulV2_run.py` | `BatchMatMulV2` | Replay batch matmul。 |
| `DispatchFFNCombine_run.py` | `DispatchFFNCombine` | Replay DFC fused 算子。 |
| `DynamicQuant_run.py` | `DynamicQuant` | Replay dynamic quant。 |
| `FusedInferAttentionScore_run.py` | `FusedInferAttentionScore` | Replay FIA。 |
| `GatherV2_run.py` | `GatherV2` | Replay gather/embedding。 |
| `GroupedMatmul_run.py` | `GroupedMatmul` | Replay grouped matmul。 |
| `GroupedMatmulSwigluQuant_run.py` | `GroupedMatmulSwigluQuant` | Replay grouped matmul + SwiGlu + quant。 |
| `Index_run.py` | `Index` | Replay index。 |
| `InterleaveRope_run.py` | `InterleaveRope` | Replay interleaved RoPE。 |
| `KvRmsNormRopeCache_run.py` | `KvRmsNormRopeCache` | Replay KV RMSNorm RoPE cache。 |
| `LightningIndexer_run.py` | `LightningIndexer` | Replay LightningIndexer。 |
| `MaskedFill_run.py` | `MaskedFill` | Replay masked fill。 |
| `MatMulCommon_run.py` | `MatMulCommon` | Replay generic matmul。 |
| `MatMulV2_run.py` | `MatMulV2` | Replay MatMulV2。 |
| `MatMulV3_run.py` | `MatMulV3` | Replay MatMulV3。 |
| `MoeTokenPermute_run.py` | `MoeTokenPermute` | Replay MoE token permute。 |
| `MoeTokenUnpermute_run.py` | `MoeTokenUnpermute` | Replay MoE token unpermute。 |
| `PadV3_run.py` | `PadV3` | Replay pad。 |
| `QuantBatchMatmulV3_run.py` | `QuantBatchMatmulV3` | Replay quant batch matmul。 |
| `RINGMLAPrefillBF16Kernel_run.py` | `RINGMLAPrefillBF16Kernel` | Replay MLA prefill kernel。 |
| `ReshapeAndCacheNdKernel_run.py` | `ReshapeAndCacheNdKernel` | Replay reshape-and-cache。 |
| `RmsNorm_run.py` | `RmsNorm` | Replay RMSNorm。 |
| `ScatterNdUpdate_run.py` | `ScatterNdUpdate` | Replay scatter update。 |
| `Slice_run.py` | `Slice` | Replay slice。 |
| `SoftmaxV2_run.py` | `SoftmaxV2` | Replay softmax。 |
| `Sort_run.py` | `Sort` | Replay sort。 |
| `SparseFlashAttention_run.py` | `SparseFlashAttention` | Replay sparse flash attention。 |
| `SwiGlu_run.py` | `SwiGlu` | Replay SwiGlu。 |
| `TensorMove_run.py` | `TensorMove` | Replay tensor copy。 |
| `TransposeBatchMatMul_run.py` | `TransposeBatchMatMul` | Replay transpose batch matmul。 |
| `Transpose_run.py` | `Transpose` | Replay transpose。 |
| `split_qkv_rmsnorm_rope_kernel_run.py` | `split_qkv_rmsnorm_rope_kernel` | Replay custom QKV/RMSNorm/RoPE fused kernel。 |

### 4. HCCL 通信微基准采集入口：`comm_bench/generate_comm_microbench.py`

`generate_comm_microbench.py` 用于运行 HCCL 通信微基准，并把结果写入通信算子 CSV。

- 输入：通信算子列表、通信组设备数、拓扑层级、消息大小网格。
- 输出：`hcom_allReduce_.csv`、`hcom_allGather_.csv`、`hcom_reduceScatter_.csv`、`hcom_alltoallv_.csv`。
- 适用场景：需要采集或刷新通信算子性能数据，并希望精确控制算子、设备数或消息大小。

| 参数 | 是否必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--database-path` | 否 | 不写文件 | per-op 通信 CSV 输出目录。 |
| `--ops` | 否 | 全部通信算子 | 可选 `all_reduce`、`all_gather`、`reduce_scatter`、`all_to_all`。 |
| `--num-devices` | 否 | `16` | 每个通信组的设备数，可传多个值。 |
| `--topology-tier` | 否 | 自动推导 | 拓扑层级，`0` 表示 inter-pod，`1` 表示 intra-pod，`2` 表示 die-level。 |
| `--grid-shape` | 否 | `48 8 2` | 硬件网格形状。默认值对应 **`ATLAS_800_A3_752T_128G_DIE`**；A3 ROCE 等型号可能为 `2 8 2` 等，需按实际拓扑传入。 |
| `--dtype` | 否 | `torch.bfloat16` | 通信 tensor dtype。 |
| `--bytes-grid` | 否 | 内置消息大小网格 | 自定义 `message_bytes` 列表。 |
| `--output-csv` | 否 | 无 | 单 CSV 输出路径，只能配合单个 `--ops` 使用。 |

直接采集示例（`ATLAS_800_A3_752T_128G_DIE`，`grid-shape 48 8 2`）：

```bash
torchrun --nproc_per_node=16 \
  tools/perf_data_collection/comm_bench/generate_comm_microbench.py \
  --database-path ./hccl_data \
  --ops all_reduce all_gather reduce_scatter all_to_all \
  --grid-shape 48 8 2 \
  --num-devices 16 2
```

### 5. ATLAS_800_A3_752T_128G_DIE 通信批量采集脚本：`comm_bench/run_comm_bench.sh`

`run_comm_bench.sh` 是 **`ATLAS_800_A3_752T_128G_DIE`** 平台的批量通信采集脚本，对 `generate_comm_microbench.py` 做了封装。

- 输入：输出目录，以及可选的多节点环境变量。
- 输出：按 752T DIE 标准配置（`grid-shape 48 8 2`）采集的一组 `hcom_*.csv`。
- 适用场景：在 752T DIE 上按推荐拓扑和消息大小网格快速采集完整通信数据。
- 默认输出目录：`./hccl_bench_data/`。

```bash
bash tools/perf_data_collection/comm_bench/run_comm_bench.sh ./hccl_bench_data
```

多节点 inter-pod 采集时，在每个节点执行脚本，并设置不同的 `NODE_RANK`：

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<master_ip> \
  bash tools/perf_data_collection/comm_bench/run_comm_bench.sh ./hccl_inter_pod

NNODES=2 NODE_RANK=1 MASTER_ADDR=<master_ip> \
  bash tools/perf_data_collection/comm_bench/run_comm_bench.sh ./hccl_inter_pod
```

注意：多节点采集是为了模拟真实的多节点训练大模型，应在真实的多机组网环境下进行采集。

## 环境变量

本节列出 `tools/perf_data_collection/` 常用环境变量。replay 重复次数仅通过 `--repeat-count` CLI 参数配置，**不再支持** `MSMODELING_OP_REPLAY_REPEAT_COUNT` 环境变量。

| 环境变量 | 默认值 | 使用场景 |
| --- | --- | --- |
| `VLLM_ASCEND_PATH` | 兄弟目录 `../vllm-ascend` | vllm-ascend 仓库根目录，用于查找自定义 Triton kernel。 |
| `ASCEND_CUSTOM_OPP_PATH` | 无 | 自定义 OPP 算子 replay 所需；详见 `start_microbench.py` 模块文档。 |
| `LD_LIBRARY_PATH` | 无 | 自定义 OPP 的 `op_api/lib` 查找路径；部分算子需与 `ASCEND_CUSTOM_OPP_PATH` 配合使用。 |
| `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` / `ASCEND_TOOLKIT_HOME_PATH` / `ASCEND_INSTALL_PATH` | 自动探测 | CANN 安装目录和版本探测。 |
| `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE` / `LOCAL_RANK` | `torchrun` 注入 | 通信 benchmark 和 DFC 分布式 replay。 |

跨模块完整列表见 [Environment Variables](../../tests/README.md)。

自定义 OPP 环境示例：

```bash
export ASCEND_CUSTOM_OPP_PATH=/path/to/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/path/to/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib:${LD_LIBRARY_PATH}
```

## 输出文件

| 工具 | 主要输出 |
| --- | --- |
| `parse_kernel_details.py` | 目标数据库目录中的 `{KernelType}.csv`。 |
| `generate_shape_grid.py` | 向已有 `{KernelType}.csv` 追加 theory shape 行。 |
| `start_microbench.py` | 回写 `Average Duration(us)`、`MicroBench ...` 等列，并在 `reports/` 下生成报告。 |
| `generate_comm_microbench.py` | `hcom_allReduce_.csv`、`hcom_allGather_.csv`、`hcom_reduceScatter_.csv`、`hcom_alltoallv_.csv`。 |
| `run_all_op.py` | `op_replay/run_all_op_status.json`，供 `start_microbench.py` 汇总。 |

`start_microbench.py` 的报告目录位于目标数据库目录下：

```text
<database-path>/reports/
  profile_update_report_<timestamp>.md
  duration_gap_hotspots_full_<timestamp>.csv
```

## `start_microbench.py` 更新模式

- `all`：更新全部匹配行，并将未匹配的 profiling 样本追加到目标 CSV。
- `missing-only`：仅 replay/填充 `Average Duration(us)` 与 `Profiling Average Duration(us)` 均无效（`0` 或空）的行；已有有效耗时的行会跳过。

## 空行清理

- 默认行为：replay 与 profiling 耗时均无效的行会保留，以便 `missing-only` 后续重试。
- `--prune-empty-duration-rows`：回写后可选清理；仅在确认要删除无法恢复的空行时使用。

## 常见问题

### 找不到数据库目录

优先使用 `--database-path` 传入明确路径。若使用版本参数自动推导，需要同时确认 `--device`、`--vllm-version`、`--torch-version`、`--cann-version` 与实际目录一致。

### `msprof not found`

说明 Ascend toolkit 环境未生效。请先加载 CANN 环境，再运行 `start_microbench.py`。

### 提示自定义 OPP 环境缺失

部分自定义算子需要 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`。按 [环境变量](#环境变量) 中的示例设置后重试。

### `missing-only` 没有执行 replay

当目标 CSV 已经存在有效 `Average Duration(us)` 或 `Profiling Average Duration(us)` 时，`missing-only` 会跳过这些行。需要强制刷新时使用 `--update-mode all`。

### 通信采集没有写出 CSV

确认命令中传入了 `--database-path`，并且 rank 0 进程有目标目录写权限。使用 `run_comm_bench.sh` 时，输出目录是脚本的第一个位置参数，未传时默认为 `./hccl_bench_data/`。

### 多节点通信采集卡住

确认所有节点同时执行脚本，且 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`、`NPROC` 设置一致。不同节点只能有不同的 `NODE_RANK`。

## 文档维护检查项

修改本文档时，请同步检查 `readme_en.md`：

- 参数名与代码中的 argparse 定义一致。
- 默认值与代码默认值一致。
- 示例命令可以从仓库根目录直接复制执行（先 `source try/bin/activate`）。
- 新增工具如果依赖 NPU、CANN、`torchrun` 或自定义 OPP，需要在前置条件中说明。

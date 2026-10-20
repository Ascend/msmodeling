# 独立单算子性能测试工具

`run_op_microbench.py` 用于测试一个指定算子、指定输入 shape/dtype/format 的性能，并把结果输出为 JSON 或 CSV。

该工具与性能数据库回填链路解耦：不会调用 `start_microbench.py`，不会追加、覆盖或回填数据库 CSV。

## 当前能力

| 能力 | 说明 |
| --- | --- |
| 请求输入 | 所有算子统一填写 `replay_row`，字段与 replay CSV 行一致。 |
| JSON 输出 | 默认格式，包含输入输出元数据、耗时、Profiling 指标、校验结果和数据来源。 |
| CSV 输出 | 使用 `--output-format csv` 手动指定，输出一行扁平性能结果。 |
| 真实 NPU/msprof | 在 Ascend 环境中构造输入、执行 warmup 和 measurement，严格解析 `op_summary`。 |
| 数据库模拟 | 无 NPU 时，通过 `--simulate-from-database` 只读匹配已有 Profiling 记录。 |
| 匹配规则 | 真实模式要求目标 OP Type 记录数严格等于 warmup + repeat；模拟模式严格匹配六个基础描述列及请求中的 Runtime 字段。 |
| 计时约束 | 不使用 Event；真实模式读取 msprof 的 `Task Duration(us)`，模拟模式读取 `Profiling ...` 字段。 |

> [!IMPORTANT]
> 真实模式只支持仓库内注册的算子适配器，不会执行用户提供的 Python API 或脚本路径。若目标记录数量、
> warmup 边界或 `Task Duration(us)` 无法可靠确认，命令会失败，不会输出看似有效的性能数据。

## 快速开始

在仓库根目录运行以下命令。仓库已经提供 MatMul 请求示例：

```text
tools/perf_data_collection/examples/matmul_v2_request.json
```

### Ascend NPU 真实采集

```bash
python tools/perf_data_collection/run_op_microbench.py \
  --request tools/perf_data_collection/examples/matmul_v2_request.json \
  --output result.json \
  --artifact-dir ./op_microbench_artifacts
```

查询当前支持范围：

```bash
python tools/perf_data_collection/run_op_microbench.py --list-operators
```

查询某个算子的 replay CSV 行协议：

```bash
python tools/perf_data_collection/run_op_microbench.py \
  --describe-operator BatchMatMulV2
```

当前白名单覆盖 45 个已经完成软件侧 standalone bridge 验证的单卡算子。新增 `op_replay/*_run.py` 不会自动
进入白名单，必须先注册 adapter 并补充回归验证；`DispatchFFNCombine` 需要分布式多卡环境，明确不在本工具
范围内。当前没有 Ascend NPU，因此这里的“软件侧验证”不代表 45 个示例都已完成真实 NPU/msprof smoke。

### 无 NPU 数据库模拟 JSON（默认）

```powershell
.venv\Scripts\python.exe tools\perf_data_collection\run_op_microbench.py `
  --request tools\perf_data_collection\examples\matmul_v2_request.json `
  --simulate-from-database tensor_cast\performance_model\profiling_database\data\ATLAS_800_A3_752T_128G_DIE\vllm_ascend\vllm0.18.0_torch2.9.0_cann8.5 `
  --output result.json
```

没有指定 `--output-format` 时，即使输出文件名以 `.csv` 结尾，也仍然输出 JSON。

### 输出 CSV

```powershell
.venv\Scripts\python.exe tools\perf_data_collection\run_op_microbench.py `
  --request tools\perf_data_collection\examples\matmul_v2_request.json `
  --simulate-from-database tensor_cast\performance_model\profiling_database\data\ATLAS_800_A3_752T_128G_DIE\vllm_ascend\vllm0.18.0_torch2.9.0_cann8.5 `
  --output-format csv `
  --output result.csv
```

`result.csv` 是本次命令的独立结果，不会写回 `MatMulV2.csv`。六个 replay 描述列会原样保留请求中的分号槽位，
`Runtime ...` 和 `EP Size` 等执行身份字段也会保留；实际返回 Tensor 单独写入 `Actual Output Shapes`、
`Actual Output Data Types` 和 `Actual Output Formats`，不会覆盖请求描述。模拟 CSV 还会包含 `Result Source`、
`Simulated`、`Source CSV` 和 `Source Row`，用于追踪数据来源。

## 请求示例

下面直接使用 MatMul replay CSV 行的六个基础描述字段。工具把这行数据交给
`MatMulV2_run.py` 构造和执行。

```json
{
  "schema_version": 1,
  "kernel_type": "MatMulV2",
  "replay_row": {
    "Input Shapes": "1,7168;256,7168",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "1,256",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  },
  "profiling": {
    "device_id": 0,
    "warmup_count": 5,
    "repeat_count": 30,
    "include_raw_records": true
  }
}
```

### Replay 行规则

所有算子都使用同一种请求方式。请把目标 CSV 行的六个基础描述列放入 `replay_row`；如果原行还包含
`Runtime ...`、`EP Size` 等 replay 语义字段，也必须一并保留。空槽表示可选输入位置，不能删除或重排。
数据库中的耗时和 Profiling 指标不是执行参数，无需复制到请求中。

```json
{
  "schema_version": 1,
  "kernel_type": "GroupedMatmul",
  "replay_row": {
    "Input Shapes": "384,2048;16,2048,7168;;16,7168;;;;16;384",
    "Input Data Types": "DT_INT8;DT_INT8;;DT_BF16;;;;DT_INT64;DT_FLOAT",
    "Input Formats": "ND;FRACTAL_NZ;;ND;;;;ND;ND",
    "Output Shapes": "384,7168",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  },
  "profiling": {
    "device_id": 0,
    "warmup_count": 5,
    "repeat_count": 30
  }
}
```

输入值域、辅助 Tensor 和固定参数由 replay adapter 负责；结果中的输入会标记为
`origin=replay_metadata`。

## 输入参数

### CLI 参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | :---: | --- | --- |
| `--request` | 三选一 | 无 | 请求 JSON 文件路径。 |
| `--list-operators` | 三选一 | 无 | 输出支持的单卡算子、统一请求模式和多卡排除项。 |
| `--describe-operator` | 三选一 | 无 | 输出指定算子的 replay CSV 行协议。 |
| `--output-format` | 否 | `json` | 输出格式，可选 `json` 或 `csv`；不根据文件扩展名推断。 |
| `--output` | 否 | 标准输出 | 结果文件路径。 |
| `--artifact-dir` | 否 | 无 | 真实 msprof 产物目录；数据库模拟模式不使用。 |
| `--simulate-from-database` | 否 | 无 | 无 NPU 开发参数，可传版本目录或与 `kernel_type` 同名的 CSV 文件。 |

### 请求 JSON

| 字段 | 必填 | 说明 |
| --- | :---: | --- |
| `schema_version` | 是 | 当前固定为 `1`。 |
| `kernel_type` | 是 | 算子名称，同时用于定位 `{kernel_type}.csv`。 |
| `replay_row` | 是 | 与 replay CSV 行一致的描述对象；必须保留六个基础列、空槽和算子需要的 Runtime 字段。 |
| `profiling` | 否 | device、预热、重复次数、超时和原始记录开关。 |

`replay_row` 必须包含 `Input Shapes`、`Input Data Types`、`Input Formats`、`Output Shapes`、
`Output Data Types` 和 `Output Formats`。这些字段与原 CSV 的字符串表示完全一致，例如多个输入用分号分隔。
算子语义由 replay 行和 `<KernelType>_run.py` 共同确定。

## 逐算子使用示例

下面覆盖当前白名单中的全部 45 个算子。每个代码块都是完整的 `request.json`，保存后统一执行：

```bash
python tools/perf_data_collection/run_op_microbench.py \
  --request request.json \
  --output result.json
```

示例默认使用 `device_id=0`、预热 5 次、正式采集 30 次。需要调整时，在任意请求顶层增加 `profiling`。
每个示例都使用 `replay_row`。实际使用时应从目标 workload 的 replay 行复制完整描述，不能只改某一个 shape
后假设输出、辅助输入和 Runtime 参数仍然合法。

### `Add`

测试单 Tensor 加标量。第二个空 shape 槽来自 CSV，replay 脚本据此使用整数 `1` 作为第二个操作数。

```json
{
  "schema_version": 1,
  "kernel_type": "Add",
  "replay_row": {
    "Input Shapes": "8;",
    "Input Data Types": "INT32;INT32",
    "Input Formats": "ND;ND",
    "Output Shapes": "8",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `AddRmsNormBias`

测试融合的 Add + RMSNorm + Bias。该算子包含多个输入和多输出，使用 `replay_row` 保留完整槽位。

```json
{
  "schema_version": 1,
  "kernel_type": "AddRmsNormBias",
  "replay_row": {
    "Input Shapes": "1,1;1,1;1;1",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16",
    "Input Formats": "ND;ND;ND;ND",
    "Output Shapes": "1,1;1,1;1,1",
    "Output Data Types": "DT_BF16;FLOAT;DT_BF16",
    "Output Formats": "ND;ND;ND"
  }
}
```

### `ArgMaxV2`

测试沿 replay 记录维度求最大值下标。第二个空 shape 槽用于表达非 Tensor 维度参数，不能删除。replay 数据库
记录的是底层 Kernel 的 INT32 输出；`torch.argmax` 的 Python API 返回 INT64。standalone 校验会识别这组协议差异，
不会为了对齐元数据而在被测调用中额外插入 Cast。

```json
{
  "schema_version": 1,
  "kernel_type": "ArgMaxV2",
  "replay_row": {
    "Input Shapes": "1,1024;",
    "Input Data Types": "FLOAT;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "1",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `AscendQuantV2`

测试带 scale/offset 的 Ascend 量化。三个输入槽分别描述输入、scale 和 offset。

```json
{
  "schema_version": 1,
  "kernel_type": "AscendQuantV2",
  "replay_row": {
    "Input Shapes": "1,1;1;1",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "1,1",
    "Output Data Types": "INT8",
    "Output Formats": "ND"
  }
}
```

### `BatchMatMulV2`

测试批量矩阵乘。输入和输出 shape 一起交给现有 replay 脚本恢复矩阵乘语义。

```json
{
  "schema_version": 1,
  "kernel_type": "BatchMatMulV2",
  "replay_row": {
    "Input Shapes": "2,3,4;2,5,4",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "2,3,5",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `DynamicQuant`

测试动态量化，输出 INT8 Tensor 和动态 scale。该算子使用完整 `replay_row`。

```json
{
  "schema_version": 1,
  "kernel_type": "DynamicQuant",
  "replay_row": {
    "Input Shapes": "1,1",
    "Input Data Types": "DT_BF16",
    "Input Formats": "ND",
    "Output Shapes": "1,1;1",
    "Output Data Types": "INT8;FLOAT",
    "Output Formats": "ND;ND"
  }
}
```

### `FusedInferAttentionScore`

测试融合推理 Attention。必须保留 layout、head 数、稀疏模式和实际序列长度等 `Runtime ...` 信息；
只提供 Q/K/V shape 不足以可靠回放。

```json
{
  "schema_version": 1,
  "kernel_type": "FusedInferAttentionScore",
  "replay_row": {
    "Input Shapes": "12,16,128;12,16,128;12,16,128;;2048,2048;1;1;;;;;;;;;;;;;;;;;;12,16,64;12,16,64;;;;;",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;INT8;INT64;INT64;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED",
    "Input Formats": "ND;ND;ND;NULL;ND;ND;ND;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;ND;NULL;NULL;NULL;NULL;NULL",
    "Output Shapes": "12,16,128;12,16,1",
    "Output Data Types": "DT_BF16;FLOAT",
    "Output Formats": "ND;ND",
    "Runtime actual_seq_lengths_shape": "1",
    "Runtime actual_seq_lengths_values": "12",
    "Runtime actual_seq_lengths_kv_shape": "1",
    "Runtime actual_seq_lengths_kv_values": "12",
    "Runtime avg_seq_len": "12.000000",
    "Runtime num_heads": "16",
    "Runtime num_key_value_heads": "16",
    "Runtime sparse_mode": "3",
    "Runtime input_layout": "TND",
    "Runtime block_size": "0"
  }
}
```

### `GatherV2`

测试 embedding/gather。输入依次为参数表、索引和维度信息，建议整体复制 replay 行。

```json
{
  "schema_version": 1,
  "kernel_type": "GatherV2",
  "replay_row": {
    "Input Shapes": "1024,1;1;1",
    "Input Data Types": "DT_BF16;INT64;INT64",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "1,1",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `GroupedMatmul`

测试 MoE Grouped MatMul。分组权重使用 `FRACTAL_NZ`，中间空槽表示可选参数，不能压缩槽位。

```json
{
  "schema_version": 1,
  "kernel_type": "GroupedMatmul",
  "replay_row": {
    "Input Shapes": "6656,2048;8,2048,6144;;;;;;8",
    "Input Data Types": "DT_BF16;DT_BF16;FLOAT;UINT64;FLOAT;FLOAT16;FLOAT16;INT64",
    "Input Formats": "ND;FRACTAL_NZ;ND;ND;ND;ND;ND;ND",
    "Output Shapes": "6656,6144",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `GroupedMatmulSwigluQuant`

测试 Grouped MatMul + SwiGlu + Quant 融合算子。权重为 `FRACTAL_NZ`，同时返回量化结果和 scale。

```json
{
  "schema_version": 1,
  "kernel_type": "GroupedMatmulSwigluQuant",
  "replay_row": {
    "Input Shapes": "1,6144;8,128,384,16,32;8,4096;1;8",
    "Input Data Types": "INT8;INT8;FLOAT;FLOAT;INT64",
    "Input Formats": "ND;FRACTAL_NZ;ND;ND;ND",
    "Output Shapes": "1,2048;1",
    "Output Data Types": "INT8;FLOAT",
    "Output Formats": "ND;ND"
  }
}
```

### `Index`

测试 Tensor 多维索引。索引 Tensor 的数量和位置由 replay 槽位确定。

```json
{
  "schema_version": 1,
  "kernel_type": "Index",
  "replay_row": {
    "Input Shapes": "1;1;1;1",
    "Input Data Types": "INT32;INT64;INT64;INT64",
    "Input Formats": "ND;ND;ND;ND",
    "Output Shapes": "1",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `InterleaveRope`

测试交错式 RoPE，三个输入分别为待旋转 Tensor、cos 和 sin。

```json
{
  "schema_version": 1,
  "kernel_type": "InterleaveRope",
  "replay_row": {
    "Input Shapes": "1,1,1,64;1,1,1,64;1,1,1,64",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "1,1,1,64",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `KvRmsNormRopeCache`

测试 KV RMSNorm、RoPE 和 Cache 写入融合算子。该算子有固定空槽和原地 cache 输出，必须使用
`replay_row`。

```json
{
  "schema_version": 1,
  "kernel_type": "KvRmsNormRopeCache",
  "replay_row": {
    "Input Shapes": "1,1,1,576;512;1,1,1,64;1,1,1,64;1;16,128,1,64;16,128,1,512;;;;;",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16;INT64;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED",
    "Input Formats": "ND;ND;ND;ND;ND;ND;ND;NULL;NULL;NULL;NULL;NULL",
    "Output Shapes": "16,128,1,64;16,128,1,512;1,1,1,64;1,1,1,512",
    "Output Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16",
    "Output Formats": "ND;ND;ND;ND"
  }
}
```

### `LightningIndexer`

测试 Lightning Indexer 稀疏索引。paged cache、序列长度和 block table 的对应关系复杂，应从同一条
replay 记录整体复制。

```json
{
  "schema_version": 1,
  "kernel_type": "LightningIndexer",
  "replay_row": {
    "Input Shapes": "1,32,128;1766,128,1,128;1,32;1;1;1,1584",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;INT32;INT32;INT32",
    "Input Formats": "ND;ND;ND;ND;ND;ND",
    "Output Shapes": "1,1,2048;1,1,2048",
    "Output Data Types": "INT32;DT_BF16",
    "Output Formats": "ND;ND",
    "Runtime avg_seq_len": "256",
    "Runtime sparse_mode": "3",
    "Runtime num_key_value_heads": "1",
    "Runtime input_layout": "TND",
    "Runtime topk": "2048",
    "Runtime block_size": "128",
    "Runtime case_id": "li_8c110488d58d621d",
    "Runtime actual_seq_lengths_shape": "1",
    "Runtime actual_seq_lengths_values": "1",
    "Runtime actual_seq_lengths_kv_shape": "1",
    "Runtime actual_seq_lengths_kv_values": "256",
    "Runtime block_table_shape": "1,1584",
    "Runtime block_table_valid_blocks": "2",
    "Runtime num_heads": "32",
    "Runtime cache_layout": "PA_BSND",
    "Runtime kv_cache_mode": "paged",
    "Runtime source_profile": "glm51_decode_q1_tp1_dp32",
    "Runtime metadata_completeness": "complete"
  }
}
```

### `MaskedFill`

测试按 bool mask 原地填充值。第三个空 shape 槽表示标量填充值。

```json
{
  "schema_version": 1,
  "kernel_type": "MaskedFill",
  "replay_row": {
    "Input Shapes": "2;2;",
    "Input Data Types": "INT32;BOOL;INT32",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "2",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `MatMulCommon`

测试数据库中的通用 MatMul replay 语义：A 不转置、B 转置。该名字当前只支持 `replay_row`。

```json
{
  "schema_version": 1,
  "kernel_type": "MatMulCommon",
  "replay_row": {
    "Input Shapes": "3,6144;32,6144",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "3,32",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `MatMulV2`

测试二维矩阵乘。该行直接对应 MatMulV2 replay CSV 的输入和输出描述。

```json
{
  "schema_version": 1,
  "kernel_type": "MatMulV2",
  "replay_row": {
    "Input Shapes": "1,7168;256,7168",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "1,256",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `MatMulV3`

测试 MatMulV3 replay 行；实际 msprof `OP Type` 可能按 CANN 版本解析为白名单中的 MatMul alias，结果会如实
返回 `actual_op_types`。

```json
{
  "schema_version": 1,
  "kernel_type": "MatMulV3",
  "replay_row": {
    "Input Shapes": "4,128;64,128",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "4,64",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `MoeTokenPermute`

测试 MoE token 按 expert 索引重排，返回重排后的 token 和恢复索引。

```json
{
  "schema_version": 1,
  "kernel_type": "MoeTokenPermute",
  "replay_row": {
    "Input Shapes": "1988,1;1988",
    "Input Data Types": "FLOAT;INT32",
    "Input Formats": "ND;ND",
    "Output Shapes": "1988,1;1988",
    "Output Data Types": "FLOAT;INT32",
    "Output Formats": "ND;ND"
  }
}
```

### `MoeTokenUnpermute`

测试根据恢复索引和权重把 MoE token 还原到原顺序。

```json
{
  "schema_version": 1,
  "kernel_type": "MoeTokenUnpermute",
  "replay_row": {
    "Input Shapes": "2048,6144;2048;256,8",
    "Input Data Types": "DT_BF16;INT32;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "256,6144",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `PadV3`

测试 Tensor padding。第二个输入描述 padding 宽度，第三个空 shape 槽表示填充值。

```json
{
  "schema_version": 1,
  "kernel_type": "PadV3",
  "replay_row": {
    "Input Shapes": "1,1;4;",
    "Input Data Types": "DT_BF16;INT32;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "8,1",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `QuantBatchMatmulV3`

测试 INT8 量化矩阵乘，权重使用 `FRACTAL_NZ`，scale 决定 BF16 输出反量化语义。

```json
{
  "schema_version": 1,
  "kernel_type": "QuantBatchMatmulV3",
  "replay_row": {
    "Input Shapes": "1,2048;224,128,16,32;7168",
    "Input Data Types": "INT8;INT8;FLOAT",
    "Input Formats": "ND;FRACTAL_NZ;ND",
    "Output Shapes": "1,7168",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `RINGMLAPrefillBF16Kernel`

测试 Ring MLA Prefill BF16 融合 Kernel。其可选前缀状态通过尾部空槽表达，必须整体复制 replay 行。输出中的
空 shape + `UNDEFINED` 槽表示该返回值存在，但历史 replay 行没有记录其具体描述；standalone 仍校验输出数量，
但不对该槽的 shape、dtype 和 format 做猜测。前缀模式若提供了具体描述，则仍逐字段严格校验。

```json
{
  "schema_version": 1,
  "kernel_type": "RINGMLAPrefillBF16Kernel",
  "replay_row": {
    "Input Shapes": "12,16,128;12,16,64;12,16,128;12,16,64;12,16,128;512,512;;;;;;;;;",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16;DT_BF16;DT_BF16;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED",
    "Input Formats": "ND;ND;ND;ND;ND;ND;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED;UNDEFINED",
    "Output Shapes": "12,16,128;",
    "Output Data Types": "DT_BF16;UNDEFINED",
    "Output Formats": "ND;UNDEFINED"
  }
}
```

### `ReshapeAndCacheNdKernel`

测试把 key/value reshape 后写入 ND cache。cache 是原地更新输入，结果 JSON 会把更新后的 cache 作为输出。

```json
{
  "schema_version": 1,
  "kernel_type": "ReshapeAndCacheNdKernel",
  "replay_row": {
    "Input Shapes": "1,1,64;1,1,64;16,128,1,64;16,128,1,64;1",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16;INT32",
    "Input Formats": "ND;ND;ND;ND;ND",
    "Output Shapes": "16,128,1,64;16,128,1,64",
    "Output Data Types": "DT_BF16;DT_BF16",
    "Output Formats": "ND;ND"
  }
}
```

### `RmsNorm`

测试 RMSNorm。两个输入及两个输出均按 replay CSV 行描述，输入值和固定参数由原脚本生成。

```json
{
  "schema_version": 1,
  "kernel_type": "RmsNorm",
  "replay_row": {
    "Input Shapes": "2,8;8",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "2,8;2,1",
    "Output Data Types": "DT_BF16;FLOAT",
    "Output Formats": "ND;ND"
  }
}
```

### `ScatterNdUpdate`

测试按二维索引更新大 Tensor。索引值的合法范围由 replay 适配器生成和约束。

```json
{
  "schema_version": 1,
  "kernel_type": "ScatterNdUpdate",
  "replay_row": {
    "Input Shapes": "222592,128;3,1;3,128",
    "Input Data Types": "DT_BF16;INT32;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "222592,128",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `Slice`

测试按 offsets 和 size 对 Tensor 切片。本例把 `[32,256]` 切成 `[32,8]`。

```json
{
  "schema_version": 1,
  "kernel_type": "Slice",
  "replay_row": {
    "Input Shapes": "32,256;2;2",
    "Input Data Types": "INT32;INT64;INT64",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "32,8",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `SoftmaxV2`

测试 Softmax。归一化维度使用 replay 脚本的固定规则。

```json
{
  "schema_version": 1,
  "kernel_type": "SoftmaxV2",
  "replay_row": {
    "Input Shapes": "1,128",
    "Input Data Types": "FLOAT",
    "Input Formats": "ND",
    "Output Shapes": "1,128",
    "Output Data Types": "FLOAT",
    "Output Formats": "ND"
  }
}
```

### `Sort`

测试排序，输出排序后的值和底层 Kernel 的 INT32 索引；排序维度和顺序复用 replay 固定语义。`torch.sort`
Python API 返回 INT64 索引，standalone 校验会识别该协议差异，不会在旧 replay 的计时区间内增加 Cast。

```json
{
  "schema_version": 1,
  "kernel_type": "Sort",
  "replay_row": {
    "Input Shapes": "1,64",
    "Input Data Types": "FLOAT",
    "Input Formats": "ND",
    "Output Shapes": "1,64;1,64",
    "Output Data Types": "FLOAT;INT32",
    "Output Formats": "ND;ND"
  }
}
```

### `SparseFlashAttention`

测试带 paged cache 和稀疏索引的 Flash Attention。该算子要求 query、cache、block table、实际序列长度
相互一致，因此请求需要包含完整 Runtime 元数据。

```json
{
  "schema_version": 1,
  "kernel_type": "SparseFlashAttention",
  "replay_row": {
    "Input Shapes": "1,2,512;1766,128,1,512;1766,128,1,512;1,1,2048;1,1584;1;1;1,2,64;1766,128,1,64",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;INT32;INT32;INT32;INT32;DT_BF16;DT_BF16",
    "Input Formats": "ND;ND;ND;ND;ND;ND;ND;ND;ND",
    "Output Shapes": "1,2,512",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND",
    "Runtime avg_seq_len": "5769",
    "Runtime sparse_mode": "3",
    "Runtime num_key_value_heads": "1",
    "Runtime input_layout": "TND",
    "Runtime topk": "2048",
    "Runtime block_size": "128",
    "Runtime case_id": "sfa_d48f8c82b4169093",
    "Runtime actual_seq_lengths_shape": "1",
    "Runtime actual_seq_lengths_values": "1",
    "Runtime actual_seq_lengths_kv_shape": "1",
    "Runtime actual_seq_lengths_kv_values": "5769",
    "Runtime block_table_shape": "1,1584",
    "Runtime block_table_valid_blocks": "46",
    "Runtime num_heads": "2",
    "Runtime cache_layout": "PA_BSND",
    "Runtime kv_cache_mode": "paged",
    "Runtime source_profile": "glm51_serving_decode_q1_tp32_dp1",
    "Runtime metadata_completeness": "complete",
    "Runtime sparse_block_size": "1",
    "Runtime sparse_indices_pattern": "uniform",
    "Runtime sparse_indices_valid_count": "2048",
    "Runtime sparse_indices_seed": "0"
  }
}
```

### `SwiGlu`

测试 SwiGlu。replay 脚本固定沿最后一维切分，本例输出最后一维为 4。

```json
{
  "schema_version": 1,
  "kernel_type": "SwiGlu",
  "replay_row": {
    "Input Shapes": "2,8",
    "Input Data Types": "DT_BF16",
    "Input Formats": "ND",
    "Output Shapes": "2,4",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `TensorMove`

测试设备 Tensor copy/move。输入和输出 shape、dtype 相同。

```json
{
  "schema_version": 1,
  "kernel_type": "TensorMove",
  "replay_row": {
    "Input Shapes": "2,154880",
    "Input Data Types": "FLOAT",
    "Input Formats": "ND",
    "Output Shapes": "2,154880",
    "Output Data Types": "FLOAT",
    "Output Formats": "ND"
  }
}
```

### `Transpose`

测试 Transpose。CSV 中第二个输入是 `perm` Tensor 描述；replay 脚本根据输入和输出 shape 恢复交换维度。

```json
{
  "schema_version": 1,
  "kernel_type": "Transpose",
  "replay_row": {
    "Input Shapes": "2,3,4;3",
    "Input Data Types": "DT_BF16;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "2,4,3",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `TransposeBatchMatMul`

测试带固定转置语义的批量矩阵乘融合算子。应直接复制该算子的 replay 行，不能用其他算子的描述代替。

```json
{
  "schema_version": 1,
  "kernel_type": "TransposeBatchMatMul",
  "replay_row": {
    "Input Shapes": "1,1,128;1,128,256",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "1,1,256",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `split_qkv_rmsnorm_rope_kernel`

测试拆分 Q/K/V 并执行 RMSNorm、RoPE 的融合 Kernel。两个输入会产生三个输出，必须使用对应 replay 行。

```json
{
  "schema_version": 1,
  "kernel_type": "split_qkv_rmsnorm_rope_kernel",
  "replay_row": {
    "Input Shapes": "4112,768;128",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "4112,512;4112,128;4112,128",
    "Output Data Types": "DT_BF16;DT_BF16;DT_BF16",
    "Output Formats": "ND;ND;ND"
  }
}
```

### `Cast`

测试 ND Tensor 的 dtype 转换，输出 dtype 由 replay 行的 `Output Data Types` 指定。

```json
{
  "schema_version": 1,
  "kernel_type": "Cast",
  "replay_row": {
    "Input Shapes": "1",
    "Input Data Types": "BOOL",
    "Input Formats": "ND",
    "Output Shapes": "1",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `CastAiCore`

测试 Cast 的 AI Core replay 入口，输入输出描述直接复制对应 CSV 行。

```json
{
  "schema_version": 1,
  "kernel_type": "CastAiCore",
  "replay_row": {
    "Input Shapes": "1",
    "Input Data Types": "INT32",
    "Input Formats": "ND",
    "Output Shapes": "1",
    "Output Data Types": "INT64",
    "Output Formats": "ND"
  }
}
```

### `Fill`

测试根据 shape 元数据创建 Tensor。第二个空 shape 槽表示标量填充值。

```json
{
  "schema_version": 1,
  "kernel_type": "Fill",
  "replay_row": {
    "Input Shapes": "1;",
    "Input Data Types": "INT64;BOOL",
    "Input Formats": "ND;ND",
    "Output Shapes": "1",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `LayerNormV3`

测试 LayerNorm 的 input、weight、bias 三个输入和三个输出。

```json
{
  "schema_version": 1,
  "kernel_type": "LayerNormV3",
  "replay_row": {
    "Input Shapes": "3,128;128;128",
    "Input Data Types": "FLOAT;FLOAT;FLOAT",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "3,128;3,1;3,1",
    "Output Data Types": "FLOAT;FLOAT;FLOAT",
    "Output Formats": "ND;ND;ND"
  }
}
```

### `MoeGatingTopK`

测试 MoE gating top-k，输出包括权重、专家索引和辅助结果。

```json
{
  "schema_version": 1,
  "kernel_type": "MoeGatingTopK",
  "replay_row": {
    "Input Shapes": "1,256;256",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "1,8;1,8;1,256",
    "Output Data Types": "DT_BF16;INT32;FLOAT",
    "Output Formats": "ND;ND;ND"
  }
}
```

### `Mul`

测试 Tensor 与标量相乘。第二个空 shape 槽由 replay 脚本按 CSV dtype 构造标量操作数。

```json
{
  "schema_version": 1,
  "kernel_type": "Mul",
  "replay_row": {
    "Input Shapes": "1;",
    "Input Data Types": "INT32;INT32",
    "Input Formats": "ND;ND",
    "Output Shapes": "1",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `ScatterNdUpdateAiCore`

测试 ScatterNdUpdate 的 AI Core replay 入口，三个输入分别是目标 Tensor、索引和更新值。

```json
{
  "schema_version": 1,
  "kernel_type": "ScatterNdUpdateAiCore",
  "replay_row": {
    "Input Shapes": "226048,128;102,1;102,128",
    "Input Data Types": "DT_BF16;INT32;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "226048,128",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `SliceAiCore`

测试 Slice 的 AI Core replay 入口，第二、第三个输入表示起始位置和切片大小。

```json
{
  "schema_version": 1,
  "kernel_type": "SliceAiCore",
  "replay_row": {
    "Input Shapes": "16,192;2;2",
    "Input Data Types": "DT_BF16;INT64;INT64",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "16,128",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `_triton_rope_siso`

测试 vLLM-Ascend Triton RoPE 单输入单输出 Kernel，qk、cos、sin 的 token 维必须一致。

```json
{
  "schema_version": 1,
  "kernel_type": "_triton_rope_siso",
  "replay_row": {
    "Input Shapes": "102,32,128;102,64;102,64",
    "Input Data Types": "DT_BF16;FLOAT;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "102,32,128",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `mla_preprocess_0_mix_aic`

测试量化 MLA preprocess。该算子依赖完整 Runtime 描述，shape、量化模式、cache 模式和权重格式必须来自同一行。

```json
{
  "schema_version": 1,
  "kernel_type": "mla_preprocess_0_mix_aic",
  "replay_row": {
    "Input Shapes": "1,6144;1,192,2624,32;2624;2048;2048;1,64,512,32;512;512;1,64;1,64;2,192,512;1,128,1,512;1,128,1,64;1;1;1;2624;1;1;512;1;1",
    "Input Data Types": "DT_BF16;DT_INT8;DT_FLOAT;DT_BF16;DT_BF16;DT_INT8;DT_FLOAT;DT_BF16;DT_BF16;DT_BF16;DT_BF16;DT_BF16;DT_BF16;DT_INT32;DT_BF16;DT_INT8;DT_INT32;DT_BF16;DT_INT8;DT_INT32;DT_BF16;DT_BF16",
    "Input Formats": "ND;FRACTAL_NZ;ND;ND;ND;FRACTAL_NZ;ND;ND;ND;ND;ND;ND;ND;ND;ND;ND;ND;ND;ND;ND;ND;ND",
    "Output Shapes": "1,2,512;1,128,1,512;1,2,64;1,128,1,64;1,2048",
    "Output Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16;DT_BF16",
    "Output Formats": "ND;ND;ND;ND;ND",
    "Runtime case_id": "glm51_mlapo_tp32_h2_t1_w8a8_krope_ctkv",
    "Runtime num_tokens": "1",
    "Runtime hidden_size": "6144",
    "Runtime local_num_heads": "2",
    "Runtime q_lora_rank": "2048",
    "Runtime kv_lora_rank": "512",
    "Runtime qk_nope_head_dim": "192",
    "Runtime qk_rope_head_dim": "64",
    "Runtime block_size": "128",
    "Runtime cache_mode": "krope_ctkv",
    "Runtime quant_mode": "per_tensor_quant_asymm",
    "Runtime enable_inner_out": "true",
    "Runtime weight_quantized": "true",
    "Runtime weight_format": "FRACTAL_NZ",
    "Runtime source_profile": "shape_grid:glm51_w8a8_vllm018_cann85",
    "Runtime metadata_completeness": "generated"
  }
}
```


### `AddAiCore`

Add 的 AI_CORE 采集变体，复用 Add 回放逻辑（torch.add）。

```json
{
  "schema_version": 1,
  "kernel_type": "AddAiCore",
  "replay_row": {
    "Input Shapes": "\"1,1536;1536\"",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1,1536\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `AddRmsNormDynamicQuant`

融合 AddRmsNorm + 动态量化，走 torch.ops.npu.npu_add_rms_norm_dynamic_quant（签名对齐 vLLM-Ascend norm_quant_fusion_pass）。

```json
{
  "schema_version": 1,
  "kernel_type": "AddRmsNormDynamicQuant",
  "replay_row": {
    "Input Shapes": "\"1,7168;1,7168;7168;7168\"",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16",
    "Input Formats": "ND;ND;ND;ND",
    "Output Shapes": "\"1,7168;1,7168;1,7168;1;1\"",
    "Output Data Types": "INT8;INT8;DT_BF16;FLOAT;FLOAT",
    "Output Formats": "ND;ND;ND;ND;ND"
  }
}
```

### `AddRmsNormDynamicQuantAiCore`

AddRmsNormDynamicQuant 的 AI_CORE 采集变体。

```json
{
  "schema_version": 1,
  "kernel_type": "AddRmsNormDynamicQuantAiCore",
  "replay_row": {
    "Input Shapes": "\"1,7168;1,7168;7168;7168\"",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16",
    "Input Formats": "ND;ND;ND;ND",
    "Output Shapes": "\"1,7168;1,7168;1,7168;1;1\"",
    "Output Data Types": "INT8;INT8;DT_BF16;FLOAT;FLOAT",
    "Output Formats": "ND;ND;ND;ND;ND"
  }
}
```

### `AsStrided`

torch.as_strided，size 使用记录的输出 shape，stride 取连续布局，offset 取 0。

```json
{
  "schema_version": 1,
  "kernel_type": "AsStrided",
  "replay_row": {
    "Input Shapes": "\"12288;3;3;1\"",
    "Input Data Types": "DT_BF16;INT64;INT64;INT64",
    "Input Formats": "ND;ND;ND;ND",
    "Output Shapes": "\"16,4,128\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `AsStridedAiCore`

AsStrided 的 AI_CORE 采集变体。

```json
{
  "schema_version": 1,
  "kernel_type": "AsStridedAiCore",
  "replay_row": {
    "Input Shapes": "\"12288;3;3;1\"",
    "Input Data Types": "DT_BF16;INT64;INT64;INT64",
    "Input Formats": "ND;ND;ND;ND",
    "Output Shapes": "\"16,4,128\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `BatchMatMulNd`

ND 批量矩阵乘，复用 BatchMatMulV2 回放逻辑（torch.matmul）。

```json
{
  "schema_version": 1,
  "kernel_type": "BatchMatMulNd",
  "replay_row": {
    "Input Shapes": "\"16,1,128;16,128,512\"",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"16,1,512\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `BroadcastTo`

torch.Tensor.expand 到记录的输出 shape 后 contiguous 物化；第一槽可为标量。

```json
{
  "schema_version": 1,
  "kernel_type": "BroadcastTo",
  "replay_row": {
    "Input Shapes": "\"12,1,64;3\"",
    "Input Data Types": "DT_BF16;INT64",
    "Input Formats": "NCL;ND",
    "Output Shapes": "\"12,16,64\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "NCL"
  }
}
```

### `ClipByValueV2`

torch.clamp(input, min=1, max=1)，min/max 标量槽回放为常量边界。

```json
{
  "schema_version": 1,
  "kernel_type": "ClipByValueV2",
  "replay_row": {
    "Input Shapes": "\"1;;\"",
    "Input Data Types": "INT64;INT64;INT64",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "INT64",
    "Output Formats": "ND"
  }
}
```

### `ConcatD`

torch.cat，拼接维度从输入/输出 shape 推断。

```json
{
  "schema_version": 1,
  "kernel_type": "ConcatD",
  "replay_row": {
    "Input Shapes": "\"102,6144;102,6144\"",
    "Input Data Types": "DT_BF16;DT_BF16",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"102,12288\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `Cumsum`

torch.cumsum，针对记录的一维行取 dim=0。

```json
{
  "schema_version": 1,
  "kernel_type": "Cumsum",
  "replay_row": {
    "Input Shapes": "\"8;1\"",
    "Input Data Types": "INT64;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"8\"",
    "Output Data Types": "INT64",
    "Output Formats": "ND"
  }
}
```

### `Equal`

torch.eq，空槽行按 Tensor 与标量 1 比较。

```json
{
  "schema_version": 1,
  "kernel_type": "Equal",
  "replay_row": {
    "Input Shapes": "\"1;\"",
    "Input Data Types": "FLOAT;FLOAT",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `FloorDiv`

torch.floor_divide，空槽行按标量除法回放。

```json
{
  "schema_version": 1,
  "kernel_type": "FloorDiv",
  "replay_row": {
    "Input Shapes": "\"1;\"",
    "Input Data Types": "INT64;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "INT64",
    "Output Formats": "ND"
  }
}
```

### `FloorMod`

torch.remainder。

```json
{
  "schema_version": 1,
  "kernel_type": "FloorMod",
  "replay_row": {
    "Input Shapes": "\"1;1\"",
    "Input Data Types": "INT32;INT32",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `GatherElements`

torch.gather（沿最后一维），索引在 build 阶段转为 int64。

```json
{
  "schema_version": 1,
  "kernel_type": "GatherElements",
  "replay_row": {
    "Input Shapes": "\"2,65;1,1\"",
    "Input Data Types": "INT32;INT32",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1,1\"",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `GatherElementsV2`

torch.gather（沿最后一维），索引在 build 阶段转为 int64。

```json
{
  "schema_version": 1,
  "kernel_type": "GatherElementsV2",
  "replay_row": {
    "Input Shapes": "\"1,1584;1,1\"",
    "Input Data Types": "INT32;INT32",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1,1\"",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `GatherV2AiCore`

GatherV2 的 AI_CORE 采集变体，复用 GatherV2 回放逻辑。

```json
{
  "schema_version": 1,
  "kernel_type": "GatherV2AiCore",
  "replay_row": {
    "Input Shapes": "\"16160,7168;1;1\"",
    "Input Data Types": "DT_BF16;INT64;INT64",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"1,7168\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `GatherV3`

torch.index_select（沿 dim 0），对应 GatherV3 记录的行。

```json
{
  "schema_version": 1,
  "kernel_type": "GatherV3",
  "replay_row": {
    "Input Shapes": "\"81920,128;16;1\"",
    "Input Data Types": "DT_BF16;INT64;INT64",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"16,128\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `Greater`

torch.gt，空槽行按 Tensor 与标量 1 比较。

```json
{
  "schema_version": 1,
  "kernel_type": "Greater",
  "replay_row": {
    "Input Shapes": "\"1;\"",
    "Input Data Types": "INT64;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `GreaterEqual`

torch.ge，空槽行按 Tensor 与标量 1 比较。

```json
{
  "schema_version": 1,
  "kernel_type": "GreaterEqual",
  "replay_row": {
    "Input Shapes": "\"1;\"",
    "Input Data Types": "INT64;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `GroupedMatmulSwigluQuantV2`

W4A8 分组 GEMM + SwiGlu + 量化的融合算子，走 npu_grouped_matmul_swiglu_quant_v2，weight 以逻辑 (groups, K, N) int8 转 FRACTAL_NZ。

```json
{
  "schema_version": 1,
  "kernel_type": "GroupedMatmulSwigluQuantV2",
  "replay_row": {
    "Input Shapes": "\"209,6144;209;32;32,64,384,16,64;32,4096;32,4096\"",
    "Input Data Types": "INT8;FLOAT;INT64;DT_INT4;UINT64;FLOAT",
    "Input Formats": "ND;ND;ND;FRACTAL_NZ;ND;ND",
    "Output Shapes": "\"209,2048;209\"",
    "Output Data Types": "INT8;FLOAT",
    "Output Formats": "ND;ND"
  }
}
```

### `IndexFill`

torch.index_fill，index/value 槽未记录值，按单个 index/value 回放。

```json
{
  "schema_version": 1,
  "kernel_type": "IndexFill",
  "replay_row": {
    "Input Shapes": "\"1,3;;\"",
    "Input Data Types": "INT32;INT64;INT32",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"1,3\"",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `IndexPutV2`

torch.index_put，重建 (input, indices, values)。

```json
{
  "schema_version": 1,
  "kernel_type": "IndexPutV2",
  "replay_row": {
    "Input Shapes": "\"272;1;1;1;1\"",
    "Input Data Types": "INT32;INT32;INT64;INT64;INT32",
    "Input Formats": "ND;ND;ND;ND;ND",
    "Output Shapes": "\"272\"",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `LayerNormV3WithImplMode`

LayerNormV3 的 impl-mode 采集变体，复用 LayerNormV3 回放逻辑。

```json
{
  "schema_version": 1,
  "kernel_type": "LayerNormV3WithImplMode",
  "replay_row": {
    "Input Shapes": "\"102,128;128;128\"",
    "Input Data Types": "FLOAT;FLOAT;FLOAT",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"102,128;102,1;102,1\"",
    "Output Data Types": "FLOAT;FLOAT;FLOAT",
    "Output Formats": "ND;ND;ND"
  }
}
```

### `Less`

torch.lt，空槽行按 Tensor 与标量 1 比较。

```json
{
  "schema_version": 1,
  "kernel_type": "Less",
  "replay_row": {
    "Input Shapes": "\"1,3;\"",
    "Input Data Types": "INT64;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1,3\"",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `LessAiCore`

Less 的 AI_CORE 采集变体。

```json
{
  "schema_version": 1,
  "kernel_type": "LessAiCore",
  "replay_row": {
    "Input Shapes": "\"1;\"",
    "Input Data Types": "INT64;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `Log`

torch.log。

```json
{
  "schema_version": 1,
  "kernel_type": "Log",
  "replay_row": {
    "Input Shapes": "\"1,129280\"",
    "Input Data Types": "FLOAT",
    "Input Formats": "ND",
    "Output Shapes": "\"1,129280\"",
    "Output Data Types": "FLOAT",
    "Output Formats": "ND"
  }
}
```

### `LogicalAnd`

torch.logical_and。

```json
{
  "schema_version": 1,
  "kernel_type": "LogicalAnd",
  "replay_row": {
    "Input Shapes": "\"1,3;1,3\"",
    "Input Data Types": "BOOL;BOOL",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1,3\"",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `LogicalAndAiCore`

LogicalAnd 的 AI_CORE 采集变体。

```json
{
  "schema_version": 1,
  "kernel_type": "LogicalAndAiCore",
  "replay_row": {
    "Input Shapes": "\"102;102\"",
    "Input Data Types": "BOOL;BOOL",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"102\"",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `LogicalNot`

torch.logical_not。

```json
{
  "schema_version": 1,
  "kernel_type": "LogicalNot",
  "replay_row": {
    "Input Shapes": "\"102\"",
    "Input Data Types": "BOOL",
    "Input Formats": "ND",
    "Output Shapes": "\"102\"",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `LogicalNotAiCore`

LogicalNot 的 AI_CORE 采集变体。

```json
{
  "schema_version": 1,
  "kernel_type": "LogicalNotAiCore",
  "replay_row": {
    "Input Shapes": "\"1\"",
    "Input Data Types": "BOOL",
    "Input Formats": "ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `MaskedFillAiCore`

MaskedFill 的 AI_CORE 采集变体，value 槽为标量。

```json
{
  "schema_version": 1,
  "kernel_type": "MaskedFillAiCore",
  "replay_row": {
    "Input Shapes": "\"1,7168;1,1;\"",
    "Input Data Types": "DT_BF16;BOOL;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"1,7168\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `MulAiCore`

Mul 的 AI_CORE 采集变体，复用 Mul 回放逻辑。

```json
{
  "schema_version": 1,
  "kernel_type": "MulAiCore",
  "replay_row": {
    "Input Shapes": "\"1;\"",
    "Input Data Types": "INT64;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "INT64",
    "Output Formats": "ND"
  }
}
```

### `Muls`

torch.mul，单输入行按 Tensor 乘标量 1 回放。

```json
{
  "schema_version": 1,
  "kernel_type": "Muls",
  "replay_row": {
    "Input Shapes": "\"256,6144\"",
    "Input Data Types": "DT_BF16",
    "Input Formats": "ND",
    "Output Shapes": "\"256,6144\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `Neg`

torch.neg。

```json
{
  "schema_version": 1,
  "kernel_type": "Neg",
  "replay_row": {
    "Input Shapes": "\"1,129280\"",
    "Input Data Types": "FLOAT",
    "Input Formats": "ND",
    "Output Shapes": "\"1,129280\"",
    "Output Data Types": "FLOAT",
    "Output Formats": "ND"
  }
}
```

### `NotEqual`

torch.ne，空槽行按 Tensor 与标量 1 比较。

```json
{
  "schema_version": 1,
  "kernel_type": "NotEqual",
  "replay_row": {
    "Input Shapes": "\"1;\"",
    "Input Data Types": "INT64;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "BOOL",
    "Output Formats": "ND"
  }
}
```

### `Pack`

torch.stack(inputs, dim=0)。

```json
{
  "schema_version": 1,
  "kernel_type": "Pack",
  "replay_row": {
    "Input Shapes": "\"12,16,129;12,16,129\"",
    "Input Data Types": "FLOAT;FLOAT",
    "Input Formats": "NCL;NCL",
    "Output Shapes": "\"2,12,16,129\"",
    "Output Data Types": "FLOAT",
    "Output Formats": "ND"
  }
}
```

### `PadV3AiCore`

PadV3 的 AI_CORE 采集变体，paddings 从输入/输出 shape 差值推断。

```json
{
  "schema_version": 1,
  "kernel_type": "PadV3AiCore",
  "replay_row": {
    "Input Shapes": "\"1,256;4;\"",
    "Input Data Types": "DT_BF16;INT32;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"8,256\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `RealDiv`

torch.div，支持广播除法。

```json
{
  "schema_version": 1,
  "kernel_type": "RealDiv",
  "replay_row": {
    "Input Shapes": "\"1,129280;1,1\"",
    "Input Data Types": "FLOAT;FLOAT",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1,129280\"",
    "Output Data Types": "FLOAT",
    "Output Formats": "ND"
  }
}
```

### `ReduceSum`

torch.sum，归约维度从记录的输出 shape 推断。

```json
{
  "schema_version": 1,
  "kernel_type": "ReduceSum",
  "replay_row": {
    "Input Shapes": "\"1,3;1\"",
    "Input Data Types": "FLOAT;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "FLOAT",
    "Output Formats": "ND"
  }
}
```

### `RepeatInterleave`

torch.repeat_interleave(input, repeats)，repeats 在 build 阶段转为 int64。

```json
{
  "schema_version": 1,
  "kernel_type": "RepeatInterleave",
  "replay_row": {
    "Input Shapes": "\"256;256\"",
    "Input Data Types": "INT32;INT32",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1622\"",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `ScatterElementsV2`

torch.scatter(input, 0, index, source)，索引在 build 阶段转为 int64。

```json
{
  "schema_version": 1,
  "kernel_type": "ScatterElementsV2",
  "replay_row": {
    "Input Shapes": "\"340;1;1\"",
    "Input Data Types": "INT32;INT32;INT32",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"340\"",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `SelectV2`

torch.where(condition, x, y)，空槽的 x/y 回放为 0 维标量张量。

```json
{
  "schema_version": 1,
  "kernel_type": "SelectV2",
  "replay_row": {
    "Input Shapes": "\"102,1;;102,6144\"",
    "Input Data Types": "BOOL;DT_BF16;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"102,6144\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `Sub`

torch.sub，空槽行按 Tensor 减标量 1 回放。

```json
{
  "schema_version": 1,
  "kernel_type": "Sub",
  "replay_row": {
    "Input Shapes": "\"1;\"",
    "Input Data Types": "INT32;INT32",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"1\"",
    "Output Data Types": "INT32",
    "Output Formats": "ND"
  }
}
```

### `SubAiCore`

Sub 的 AI_CORE 采集变体。

```json
{
  "schema_version": 1,
  "kernel_type": "SubAiCore",
  "replay_row": {
    "Input Shapes": "\"102;102\"",
    "Input Data Types": "INT64;INT64",
    "Input Formats": "ND;ND",
    "Output Shapes": "\"102\"",
    "Output Data Types": "INT64",
    "Output Formats": "ND"
  }
}
```

### `TensorMoveAiCore`

TensorMove 的 AI_CORE 采集变体，复用 TensorMove 回放逻辑。

```json
{
  "schema_version": 1,
  "kernel_type": "TensorMoveAiCore",
  "replay_row": {
    "Input Shapes": "\"102,6144\"",
    "Input Data Types": "DT_BF16",
    "Input Formats": "ND",
    "Output Shapes": "\"102,6144\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `Tile`

torch.tile，multiples 从输入/输出 shape 推断。

```json
{
  "schema_version": 1,
  "kernel_type": "Tile",
  "replay_row": {
    "Input Shapes": "\"16,2,64;3\"",
    "Input Data Types": "DT_BF16;INT64",
    "Input Formats": "NCL;ND",
    "Output Shapes": "\"16,2,128\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "ND"
  }
}
```

### `TransData`

FRACTAL_NZ → ND 数据排布转换，走 torch_npu.npu_format_cast(input, ND)。

```json
{
  "schema_version": 1,
  "kernel_type": "TransData",
  "replay_row": {
    "Input Shapes": "\"192,128,16,32\"",
    "Input Data Types": "INT8",
    "Input Formats": "FRACTAL_NZ",
    "Output Shapes": "\"2048,6144\"",
    "Output Data Types": "INT8",
    "Output Formats": "ND"
  }
}
```

### `TransposeAiCore`

Transpose 的 AI_CORE 采集变体，复用 Transpose 回放逻辑（转置维度从 shape 推断）。

```json
{
  "schema_version": 1,
  "kernel_type": "TransposeAiCore",
  "replay_row": {
    "Input Shapes": "\"16,4,512;3\"",
    "Input Data Types": "DT_BF16;INT64",
    "Input Formats": "NCHW;ND",
    "Output Shapes": "\"4,16,512\"",
    "Output Data Types": "DT_BF16",
    "Output Formats": "NCHW"
  }
}
```

### `ZerosLike`

torch.zeros_like。

```json
{
  "schema_version": 1,
  "kernel_type": "ZerosLike",
  "replay_row": {
    "Input Shapes": "\"2,1\"",
    "Input Data Types": "INT64",
    "Input Formats": "ND",
    "Output Shapes": "\"2,1\"",
    "Output Data Types": "INT64",
    "Output Formats": "ND"
  }
}
```

### `_triton_rope`

vLLM-Ascend MHA Triton RoPE（rope_forward_triton），cos_sin_cache 单槽携带 rope_dim。

```json
{
  "schema_version": 1,
  "kernel_type": "_triton_rope",
  "replay_row": {
    "Input Shapes": "\"28736,4,128;28736,1,128;81920,128\"",
    "Input Data Types": "DT_BF16;FLOAT;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"28736,4,128;28736,1,128\"",
    "Output Data Types": "DT_BF16;FLOAT",
    "Output Formats": "ND;ND"
  }
}
```

### `AscendQuantV2Aicore`

AscendQuantV2 的 AI_CORE 采集变体，复用 AscendQuantV2 回放逻辑（torch_npu.npu_quantize）。

```json
{
  "schema_version": 1,
  "kernel_type": "AscendQuantV2Aicore",
  "replay_row": {
    "Input Shapes": "\"1,1536;1536;1536\"",
    "Input Data Types": "DT_BF16;DT_BF16;DT_BF16",
    "Input Formats": "ND;ND;ND",
    "Output Shapes": "\"1,1536\"",
    "Output Data Types": "INT8",
    "Output Formats": "ND"
  }
}
```

### `DequantSwigluQuant`

反量化 + SwiGlu + 量化融合算子，走 npu_dequant_swiglu_quant（keyword-only 参数，quant_mode=1）。

```json
{
  "schema_version": 1,
  "kernel_type": "DequantSwigluQuant",
  "replay_row": {
    "Input Shapes": "\"18,4096;4096;18;;;;1\"",
    "Input Data Types": "INT32;FLOAT;FLOAT;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;INT64",
    "Input Formats": "ND;ND;ND;NULL;NULL;NULL;ND",
    "Output Shapes": "\"18,2048;18\"",
    "Output Data Types": "INT8;FLOAT",
    "Output Formats": "ND;ND"
  }
}
```
## JSON 结果示例

使用仓库当前示例数据时，关键结果类似下面这样：

```json
{
  "status": "succeeded",
  "phase": "database_simulation",
  "simulated": true,
  "result_source": "profiling_database_simulation",
  "outputs": [
    {
      "logical_shape": [1, 256],
      "dtype": "DT_BF16",
      "format": "ND"
    }
  ],
  "duration": {
    "source": "profiling_database_simulation",
    "column": "Profiling Median Duration(us)",
    "unit": "us",
    "value_us": 10.84
  },
  "msprof": {
    "executed": false,
    "simulated": true,
    "source": "profiling_database_simulation"
  }
}
```

这里的 `10.84 us` 来自仓库已有 Profiling 聚合记录，只用于演示接口，不代表本次运行重新测量了算子。
完整结果还包括：

- `inputs` / `outputs`：Tensor shape、dtype、format 和字节数。
- `duration.statistics`：已有的 p50、mean 和 std；数据库没有的信息为 `null`。
- `msprof.aggregated_metrics`：已有的 AI Core、cycle、MTE、Cube 利用率等 Profiling 指标。
- `msprof.raw_target_records`：可选的匹配数据库聚合行。
- `validation`：请求、匹配、耗时和 replay 输出描述校验状态。
- `artifacts.simulation_source`：源 CSV、行号、大小、mtime 和 SHA256。

## 匹配与失败规则

模拟模式只有同时满足以下条件才成功：

1. 找到唯一的 `{kernel_type}.csv`。
2. 六个基础描述列，以及请求携带的 `Runtime ...`、`EP Size` 等 replay 字段与数据库记录严格一致。
3. 只有一条数据库记录匹配。
4. `Profiling Median Duration(us)` 是有限正数。

真实模式还会校验实际输出的 shape、dtype、format 与 `replay_row` 中的三个输出描述字段一致。

常见错误码：

| 错误码 | 说明 |
| --- | --- |
| `INVALID_REQUEST` | 请求字段、CSV 行描述或 profiling 参数不合法。 |
| `DATABASE_RECORD_NOT_FOUND` | 没有严格匹配的记录。 |
| `AMBIGUOUS_DATABASE_RECORD` | 多条记录匹配，结果不唯一。 |
| `NO_VALID_PROFILING_DURATION` | 没有有效的 Profiling 中位耗时。 |
| `REPLAY_OUTPUT_MISMATCH` | 真实执行结果与 `replay_row` 的输出描述不一致。 |
| `OUTPUT_INSIDE_DATABASE` | 结果路径位于数据源目录内；工具拒绝写入。 |
| `NPU_NOT_AVAILABLE` | 未使用模拟模式，且当前环境没有可用的 NPU/msprof 依赖。 |
| `OPERATOR_EXECUTION_FAILED` | 算子适配器在被 msprof 包裹的 worker 中执行失败。 |
| `MSPROF_TIMEOUT` / `MSPROF_FAILED` | msprof 超时或返回非零退出码。 |
| `MSPROF_SUMMARY_NOT_FOUND` | msprof 没有生成可解析的 `op_summary`。 |
| `AMBIGUOUS_MSPROF_TARGET_RECORDS` | 目标记录数不等于 warmup + repeat，无法可靠聚合。 |
| `NO_VALID_MSPROF_DURATION` | 正式样本中存在非有限或非正的 `Task Duration(us)`。 |

JSON 模式会返回结构化失败结果并退出 `1`。CSV 模式不会输出一行看似有效的数据，而是把结构化错误写到
stderr 并退出 `1`。

## 验证

运行单元测试：

```powershell
.venv\Scripts\pytest.exe -q tests\regression\cli\test_run_op_microbench.py
```

完整设计和真实 msprof 目标接口见
[独立算子 msprof Microbenchmark RFC](https://gitcode.com/Ascend/msmodeling.wiki/blob/main/RFC/rfc_standalone_msprof_operator_microbench_zh.md)。

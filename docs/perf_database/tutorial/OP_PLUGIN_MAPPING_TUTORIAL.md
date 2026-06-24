# Op Mapping 教程 v2

> 如何创建和维护 `op_mapping.yaml` —— TensorCast 仿真与 NPU profiling 数据之间的桥梁。

## 1. op_mapping.yaml 是什么？

`op_mapping.yaml` 将每个 TensorCast (TC) 虚拟算子映射到真实设备 profiling 数据中的 NPU 内核类型（kernel type）。这个映射使得 `EmpiricalPerformanceModel` 能够查找真实 profiling 延迟，而不是使用解析估算。

**文件位置：** `tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version}/op_mapping.yaml`

**版本命名规范：** `{version}` 编码了用于 profiling 的软件栈版本（例如 `v0.13.0` 或 `vllm0.13.0_torch2.8.0_cann8.3`）

### 文件结构

```yaml
version: "<vllm_ascend_version>"
device: <DEVICE>
cann_version: "<cann_version>"

interpolation_policy:
  default_method: linear
  kernel_overrides:
    FusedInferAttentionScore:
      shape_transform: sqrt    # O(seq²) 算子在 sqrt 空间中插值

operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2              # Profiling Type 列 = CSV 文件名
    notes: "[HIGH] Path A. op-plugin: ..."

  "tensor_cast.matmul_all_reduce.default":
    composite: true                     # 分解为多个内核
    sub_kernels: [MatMulV2, hcom_allReduce_]

  "aten.view.default":
    zero_cost: true                     # 纯元数据操作，无 NPU 执行

torch_npu_reference:
  MatMulV2:
    apis: [torch.mm, torch.matmul]
    aclnn: [aclnnMatmul, aclnnMatmulWeightNz]
```

### 条目类型（互斥）

| 类型 | 字段 | 含义 | 示例 |
|------|------|------|------|
| **计算** | `kernel_type: X` | 通过 Type=X 直接查询 CSV | MatMulV2, SwiGlu |
| **复合** | `composite: true` | 分解为 `sub_kernels`，分别查询 | matmul_all_reduce → [MatMulV2, hcom_allReduce_] |
| **零开销** | `zero_cost: true` | 返回 latency=0（纯元数据算子） | view, permute, split |

每个条目必须恰好具有以上三种类型之一。

### 可选字段

- `alternate_kernel_types: [Type1, Type2]` —— 主类型未命中时的备选 CSV 类型
- `category: communication` —— 触发 message_bytes+num_devices 查询而非 shape 匹配
- `query_mode: attention_special` —— 触发 (batch, seq, heads, head_dim) 匹配
- `query_mode: elementwise` —— 触发输出形状匹配 + dtype 松弛缩放（逐元素算子）
- `notes: "..."` —— 包含置信度级别的证据链

#### 逐元素算子 (Elementwise Ops)

对于内存带宽受限的逐元素算子 (`aten.add.Tensor`, `aten.mul.Tensor`, `aten.div.Tensor`),
使用 `query_mode: elementwise` 代替默认的输入形状匹配:

```yaml
"aten.add.Tensor":
  kernel_type: Add
  query_mode: elementwise
```

此模式按**输出形状**匹配 CSV,并支持 dtype 松弛匹配 (FP32 → BF16 × 2.0 字节比缩放)。
不需要设置 `tc_input_count`。

## 2. 数据流：PyTorch → NPU 内核

理解此流水线是创建正确映射的基础：

```text
PyTorch aten 算子 (如 aten.mm)
  → op-plugin 分发 (op_plugin_functions.yaml)
    → C++ 实现 (opapi/*.cpp)
      → EXEC_NPU_CMD(aclnn*) 调用
        → CANN aclnn Host API
          → L0 OpType 注册 (CMakeLists.txt)
            → NPU 内核执行
              → Profiling kernel_details.csv
```

**关键标识符：** `kernel_details.csv` 中的 `Type` 列 = CANN OPTYPE = op_mapping.yaml 中的 `kernel_type`。

**Name 列：** 具有3段结构：`aclnnAPI_DispatchFunc_L0OpType`。第3段 = Type 列的值。

## 3. 三种映射路径

### 路径 A：aten → op-plugin → aclnn（最常见）

适用于通过 Ascend op-plugin 分发的标准 PyTorch 算子：

1. **在 op-plugin 中查找算子：** `grep "aten::mm" op_plugin/config/op_plugin_functions.yaml`
2. **找到 C++ 实现：** `op_plugin/ops/opapi/MmKernelNpuOpApi.cpp`
3. **找到 aclnn 调用：** `EXEC_NPU_CMD(aclnnMm, ...)` 或 `EXEC_NPU_CMD(aclnnMatmulWeightNz, ...)`
4. **找到 OPTYPE：** 在 CANN 仓库中搜索 aclnn 名称 → 找到 `OP_TYPE_REGISTER(MatMulV2)`
5. **对照 profiling 验证：** 检查 `MatMulV2` 是否出现在 kernel_details.csv 的 Type 列中

**示例：aten.mm.default → MatMulV2**

```text
op-plugin YAML → MmKernelNpuOpApi.cpp → EXEC_NPU_CMD(aclnnMm)
  → cann-ops-nn/matmul/mat_mul_v2/ → OP_TYPE_REGISTER(MatMulV2)
  → Profiling: Type=MatMulV2
```

### 路径 B：torch_npu.npu_* → op-plugin → aclnn

适用于使用 torch_npu API 的 vLLM-ascend 专用算子：

1. **找到 vllm-ascend 调用：** `torch_npu.npu_grouped_matmul_swiglu_quant(...)`
2. **在 op-plugin 中查找：** `grep "npu_grouped_matmul_swiglu_quant" op_plugin/`
3. **沿相同的 aclnn → OPTYPE 链路追踪**

**示例：grouped_matmul_quant_swiglu → GroupedMatmulSwigluQuant**

```text
vllm-ascend moe_mlp.py → torch_npu.npu_grouped_matmul_swiglu_quant
  → op-plugin → aclnnGroupedMatmulSwigluQuantWeightNZ
  → cann-ops-transformer/gmm/ → OP_TYPE_REGISTER(GroupedMatmulSwigluQuant)
  → Profiling: Type=GroupedMatmulSwigluQuant
```

### 路径 C：vLLM-ascend 自定义算子 / Triton 内核

适用于不在 op-plugin 中的算子（自定义内核、Triton、ATB）：

1. **找到 vllm-ascend 自定义算子：** 如 `vllm_ascend/ops/attention.py`
2. **判断是 Triton、csrc 还是 ATB：** 函数名通常 = profiling Type
3. **在 profiling 数据中验证**

**示例：ATB 内核**

```text
vllm-ascend mla_v1.py → torch_npu.atb.npu_ring_mla()
  → ATB 内核: RINGMLAPrefillBF16Kernel
  → Profiling: Type=RINGMLAPrefillBF16Kernel
```

### 通信算子 (HCCL)

通信算子完全绕过 op-plugin：

```text
TC all_reduce → torch.distributed.all_reduce → HCCL → hcom_allReduce_
```

这些算子使用 `message_bytes + num_devices` 进行查询，而非 shape 匹配。

## 4. 如何追踪单个算子（分步指南）

**目标：** 将 `tensor_cast.swiglu.default` 映射到其 NPU 内核类型。

**步骤 1：理解 TC 算子**

```bash
grep -r "def swiglu" tensor_cast/ops/
# → tensor_cast/ops/activation.py: SwiGlu 激活函数 (gate * sigmoid(gate) * up)
```

**步骤 2：找到 aten/torch_npu 路径**

SwiGlu 是一个自定义 TC 算子，因此检查 vLLM-ascend：

```bash
grep -r "swiglu\|silu_and_mul" /path/to/vllm-ascend/
# → vllm_ascend/ops/activation.py → torch_npu.npu_swiglu(...)
```

**步骤 3：查找 op-plugin 条目**

```bash
grep "npu_swiglu" /path/to/op-plugin/op_plugin/config/op_plugin_functions.yaml
# → 第 5742 行: npu_swiglu
```

**步骤 4：查找 EXEC_NPU_CMD**

```bash
grep -r "npu_swiglu" /path/to/op-plugin/op_plugin/ops/
# → SwigluKernelNpuOpApi.cpp: EXEC_NPU_CMD(aclnnSwiglu, ...)
```

**步骤 5：查找 OPTYPE**

```bash
grep -r "SwiGlu\|SWIGLU" /path/to/cann-ops-transformer/ --include="CMakeLists.txt"
# → set(OPTYPE "SwiGlu")
```

**步骤 6：在 profiling 中验证**

```bash
grep "SwiGlu" kernel_details.csv | head -3
# → Type=SwiGlu，DSv3 中 390 次，Qwen3 中 670 次
```

**结果：**

```yaml
"tensor_cast.swiglu.default":
  kernel_type: SwiGlu
  notes: "[HIGH] Path B. op-plugin: SwigluKernelNpuOpApi.cpp → aclnnSwiglu → SwiGlu."
```

## 5. 8 种 Shape 差异

TC tensor shape 与 NPU profiling shape 存在差异。`profiling_data_source.py` 自动处理这些差异，但调试时需要理解它们：

| # | 差异类型 | TC Shape | NPU Profiling Shape | 处理方式 |
|---|---------|----------|---------------------|----------|
| 1 | 批次维度 | `(1,S,D)` | `(S,D)` | 移除两边的前导 batch=1 |
| 2 | 序列填充 | `S=144` | `S=136` | block-padding 容差（向上取整到 16/32） |
| 3 | FRACTAL_NZ | `(K,N)` ND 格式 | `[H,W,bh,bw]` 分块格式 | `fractal_nz_to_nd()` 还原 |
| 4 | ND 转置 | `(K,N)` | `(N,K)` | MatMul 权重转置检查 |
| 5 | SwiGlu 拼接 | 2×`(S,D/2)` | 1×`(S,D)` | 在最后维度上拼接输入 |
| 6 | RoPE 布局 | `(B,H,S,D)` Q,K | `(B,S,H,D)` K,Q | 转置维度 + 重排输入 |
| 7 | RoPE 内核 | 单个 TC 算子 | 多个 NPU 内核 | `alternate_kernel_types` |
| 8 | 复合算子 | 融合的 TC 算子 | 分离的 NPU 内核 | `sub_kernels` 分解 |

## 6. 使用 Profiling 数据

### kernel_details.csv 列说明

| 列名 | 含义 | 用途 |
|------|------|------|
| **Type** | CANN OPTYPE = 我们的 `kernel_type` | 聚合的主键 |
| **Name** | `aclnn_Dispatch_L0OpType` 三段式 | 追溯到 aclnn API |
| **Input Shapes** | tensor shape 字符串 | CSV 中的 shape 匹配 |
| **Duration(us)** | 内核执行时间 | 性能数据 |
| **Accelerator Core** | AI Core 或 AI Vector Core | 硬件利用率 |

### 解析 Profiling 数据

```bash
# 从 kernel_details.csv 生成按内核拆分的 CSV
python3.10 -m tools.perf_data_collection.parse_kernel_details \
  --device ATLAS_800_A3_752T_128G_DIE \
  --vllm-ascend-version <version_string> \
  --kernel-details-path /path/to/kernel_details.csv

# 验证生成的数据库
python3.10 -m tools.perf_data_collection.validate \
  --database tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version}/
```

### 获取唯一内核类型

```python
import csv
from collections import Counter
with open('kernel_details.csv') as f:
    types = Counter(row['Type'] for row in csv.DictReader(f))
for t, c in types.most_common():
    print(f"{c:6d}  {t}")
```

## 7. 查询分发类别

`ProfilingDataSource` 根据 op_mapping 配置通过 5 条路径路由查询：

```text
是复合算子？ → _lookup_composite(sub_kernels)
是通信算子？ → _lookup_comm(message_bytes, num_devices)
是特殊注意力？ → _lookup_attention(batch, seq, heads, head_dim)
是零开销？ → QueryResult(latency=0)
默认 → _lookup_compute(kernel_type, alternate_kernel_types)
```

| 类别 | 查询方式 | 匹配依据 |
|------|---------|---------|
| `compute` | CSV shape 查找 | 输入/输出 tensor shape |
| `communication` | 消息字节数 | `tensor_nbytes * dtype_size` |
| `attention_special` | 注意力维度 | `(batch, seq_len, num_heads, head_dim)` |
| `composite` | 分解 + 求和 | 每个 sub_kernel 独立查询 |
| `zero_cost` | 返回 0 | 无需查找 |

## 8. 处理 CANN 版本差异

内核类型会在不同 CANN 版本之间发生变化。常见模式：

| 变更类型 | 示例 | 处理方式 |
|---------|------|---------|
| **重命名** | `ScatterElements` → `ScatterElementsV2` | 更新 `kernel_type`，用 `alternate_kernel_types` 兼容 |
| **融合** | 独立的 matmul+activation → 单个融合内核 | 更新 `kernel_type`（不仅仅是 `alternate_kernel_types`） |
| **拆分** | 一个内核 → 两个独立内核 | 可能需要 `composite: true` + `sub_kernels` |
| **移除** | Triton 内核被 CANN 原生融合替代 | 删除条目或更新为新内核类型 |
| **新增内核** | 新的 ATB/CANN 融合内核 | 添加新条目，通过 5 层流水线追踪 |

### 如何发现版本差异

1. **对比两个 CANN 版本的 profiling type：**

   ```bash
   # 从每次 profiling 提取唯一类型
   awk -F',' 'NR>1 {print $2}' old_kernel_details.csv | sort -u > old_types.txt
   awk -F',' 'NR>1 {print $2}' new_kernel_details.csv | sort -u > new_types.txt
   diff old_types.txt new_types.txt
   ```

2. **通过 5 层流水线（路径 A/B/C）追踪每个差异**，确定正确映射
3. **使用 `alternate_kernel_types`**：当新旧名称可能出现在不同 profiling 数据集中时

**核心经验：** 更换 CANN 版本时务必重新生成 op_mapping。profiling 的 `Type` 列是唯一真相。

### aclgraph 一致性

vllm-ascend 的 aclgraph 确保 **eager 模式和 graph 模式产生完全相同的算子**（包括融合 pass）。两种模式的 profiling 数据对 op_mapping 同样有效。

## 9. 端到端验证

验证要求**从 profiling 数据本身推导**正确的 TC 仿真参数。使用错误参数（如 profiling 捕获的是 decode 但用了 prefill 参数）会导致大量 shape 不匹配，这**不是** op_mapping 的问题。

### 步骤 1：分析 profiling 数据推导参数

**判断负载类型（prefill vs decode）：**

```bash
# 检查计算内核的批次维度 —— 小值 (1-50) = decode，大值 (100+) = prefill
for f in MatMulV2.csv AddRmsNorm.csv SwiGlu.csv; do
  echo "=== $f ===" && awk -F',' 'NR>1 {print $3}' $DATA_DIR/$f | sort | uniq -c | sort -rn | head -5
done
```

**判断量化方式：**

```bash
# QuantBatchMatmulV3.csv 存在且含 INT8 → W8A8_STATIC；仅 BF16 MatMulV2 → DISABLED
ls $DATA_DIR/*.csv | grep -i quant
```

**判断并行度 (TP/DP/EP)：**

- 比较 CSV 中间维度与模型配置：`intermediate_per_card = model.intermediate_size / TP`
- 检查 FIA 头数：`q_heads_per_card = model.num_attention_heads / TP`
- 存在 MoE 算子 (GroupedMatmul*) → 启用 EP
- 推导：`world_size = TP × DP × EP_size`

**判断批次大小：**

- 计算内核中最高频的批次维度 = 目标 `--num-queries`
- Decode：`--num-queries=<batch> --query-length=1 --context-length=4500`
- Prefill：`--num-queries=1 --query-length=<batch>`（或 nq=2 ql=batch/2）
- block-padding 匹配：TC 向上填充到 16 的倍数，因此 `ceil(nq*ql/16)*16` 必须与 CSV seq 维度匹配

### 步骤 2：使用推导的参数运行 TC 仿真

```bash
python3.10 -m cli.inference.text_generate $MODEL \
  --num-queries $NQ --query-length $QL [--context-length $CL] \
  --device $DEVICE --num-devices $WS --tp-size $TP [--dp-size $DP] [--ep-size $EP] \
  --quantize-linear-action $QUANT \
  --performance-model profiling --compile \
  --profiling-database $DATA_DIR
```

如果你在验证 SequenceParallel 对齐，可额外添加 `--enable-sequence-parallel`。
该开关只在 `--compile` 打开时生效；同时它与 `matmul_allreduce` 这类
MC2 融合路径会竞争同一部分通信子图，因此通常应作为单独配置显式开启，
不要默认与其他 compile pass 一起混用。当前该开关仅用于 prefill 对齐；
原始 decode profiling 不启用 sequence parallel，因此 decode 场景下应保持关闭。

### 步骤 3：对每个 MISS 进行分类

输出会显示 `EmpiricalPerformanceModel: X/Y ops matched`。对每个 MISS 进行分类：

| 差距类别 | 示例 | 处理方式 |
|---------|------|---------|
| **算子映射错误** | kernel_type 错误或缺少条目 | 修复 op_mapping.yaml |
| **Shape 数据缺口** | 内核正确但 CSV 中没有对应 shape | 补充 profiling 数据或微基准测试 |
| **TC 分解不匹配** | TC 中间 shape ≠ 真实 vLLM-ascend | 已知限制，非映射问题 |
| **结构性缺失** | Embedding、KV cache、通信算子 | 预期之中 —— TC 与 NPU 接口不同 |
| **参数不匹配** | 错误的 batch/TP 导致缺失 | 从步骤 1 重新推导参数 |

**核心原则：** shape MISS + 正确的 kernel_type = 数据覆盖缺口。shape MISS + 错误的 kernel_type = 算子映射错误。只有后者需要修复。

### 步骤 4：迭代

1. 修复所有算子映射错误
2. 如发现参数不匹配，用修正后的参数重新运行
3. 重复直到无新的算子映射错误
4. 按类别记录剩余差距

### 步骤 5：验证匹配率

```bash
python3.10 -m cli.inference.text_generate $MODEL \
  --performance-model profiling --compile \
  --profiling-database $DATA_DIR 2>&1 | grep "matched"
```

### 各模型类型预期结果

| 模型类型 | 典型匹配率 | 说明 |
|---------|----------|------|
| Dense BF16（如 Qwen3-32B） | 80-90% | 剩余缺失：attention、KV cache、embedding、通信 |
| Dense W8A8 | 70-85% | 量化算子可能有不同的中间 shape |
| MoE W8A8（如 DSv3） | 35-50% | MoE 路由产生可变批次大小；MLA 分解复杂 |

MoE 模型匹配率较低是预期的，因为 TC 的 compile pass 产生的中间 shape 与真实 vLLM-ascend 不同（尤其是 MLA 投影和 MoE 分发部分）。

## 10. 常见陷阱

1. **缺少 `--compile`**：不加此参数时，融合算子（SwiGlu、AddRmsNorm、MC2、SequenceParallel）会分解为 70+ 个 aten 原始算子，无法匹配 profiling 内核。profiling 模式下务必使用 `--compile`。

2. **混用 `--enable-sequence-parallel` 与 MC2 配置**：SequenceParallel 与 `matmul_allreduce` 会改写部分重叠的通信模式，通常应视为二选一的 compile 配置。做 profiling 对齐时，先明确当前要验证哪条路径，再决定是否添加 `--enable-sequence-parallel`。另外，当前 SequenceParallel 只用于 prefill，对齐 decode profiling 时不要开启。

3. **验证参数错误**：用 prefill 参数（`--query-length 3500`）去验证 decode 的 profiling 数据（`batch=4, query-length=1`）会导致大量 shape 不匹配。务必先从 CSV shape 推导参数（见第 9 节步骤 1）。

4. **重命名内核类型错误**：CANN 版本可能重命名内核。务必核实 profiling 数据中的 `Type` 列，并用 `alternate_kernel_types` 实现跨版本兼容。

5. **混淆 Name 和 Type 列**：`Type` 列是干净的 OPTYPE（我们的查询键），`Name` 列是完整的层级路径。始终按 Type 聚合。

6. **复合 vs 单一**：某些 TC 算子映射到多个 NPU 内核（MLA decode = BatchMatMulV2 + FIA + batch_matmul_transpose）。使用 `composite: true` + `sub_kernels`。

7. **Shape 不匹配 ≠ 映射错误**：shape MISS（CSV 中无匹配 shape）与映射错误（kernel_type 错误）不同。shape 缺失是数据覆盖缺口，不是 op_mapping 的问题。

8. **MoE 模型低匹配率**：MoE 模型（DSv3）天然匹配率较低（35-50%），因为 TC 的 MoE 分发和 MLA 分解产生的中间 shape 与真实 vLLM-ascend 不同。这是已知的 TC 仿真限制，不是算子映射错误。

9. **通信算子不使用 shape**：HCCL 算子（allreduce、allgather 等）使用 message_bytes，不使用 tensor shape。不要尝试 shape 匹配。

## 11. 快速参考：常用映射

### 标准 aten 算子

| TC 算子 | NPU 内核 | 说明 |
|---------|---------|------|
| aten.mm | MatMulV2 | 标准矩阵乘法 |
| aten.bmm | BatchMatMulV2 | 批次矩阵乘法（备选：batch_matmul_transpose） |
| aten.addmm | MatMulV2 | Bias 融合到 MatMulV2 |
| aten.add.Tensor | Add | 逐元素加法 |
| aten.mul.Tensor | Mul | 逐元素乘法 |
| aten.div.Tensor | Div | 除法（备选：RealDiv） |
| aten.embedding | GatherV2 | Embedding 查找（备选：GatherV3） |
| aten.to.dtype | Cast | 类型转换（备选：TensorMove） |
| aten.clone | TensorMove | 内存拷贝 |
| aten.scatter.value | ScatterElementsV2 | Scatter 写入 |
| aten.sum.dim_IntList | ReduceSum | 求和归约 |
| aten.topk | TopKV2 | Top-K 选择 |

### TensorCast 融合算子

| TC 算子 | NPU 内核 | 说明 |
|---------|---------|------|
| tc.swiglu | SwiGlu | SwiGlu 激活 |
| tc.rms_norm | RmsNorm | RMS 归一化 |
| tc.add_rms_norm/2 | AddRmsNorm | 残差 + RmsNorm |
| tc.apply_rope | InterleaveRope | RoPE（备选：ApplyRotaryPosEmb） |
| tc.attention | FusedInferAttentionScore | 融合注意力 |
| tc.reshape_and_cache | ReshapeAndCacheNdKernel | KV cache 写入 |
| tc.kv_rmsnorm_rope_cache | KvRmsNormRopeCache | 融合 KV norm+RoPE+cache |

### 量化算子

| TC 算子 | NPU 内核 | 说明 |
|---------|---------|------|
| tc.static_quant_linear | QuantBatchMatmulV3 | INT8 矩阵乘法 |
| tc.static_quant_linear_int4 | QuantBatchMatmulV3 | INT4 矩阵乘法 |
| tc.fp8_linear | QuantBatchMatmulV3 | FP8 矩阵乘法 |
| tc.quantize | AscendQuantV2 | 静态量化 |
| tc.dynamic_quantize_symmetric | DynamicQuant | 动态量化 |
| tc.grouped_matmul_quant_swiglu | GroupedMatmulSwigluQuant | MoE 融合 gate-up |
| tc.grouped_matmul_quant | GroupedMatmul | MoE 矩阵乘法 |

### 通信算子

| TC 算子 | NPU 内核 | 说明 |
|---------|---------|------|
| tc.all_reduce | hcom_allReduce_ | HCCL all-reduce |
| tc.all_gather | hcom_allGather_ | HCCL all-gather |
| tc.all_to_all | hcom_alltoallv_ | HCCL all-to-all（MoE） |
| tc.reduce_scatter | HcomReduceScatter | HCCL reduce-scatter |

### 复合算子

| TC 算子 | 子内核 | 说明 |
|---------|-------|------|
| tc.matmul_all_reduce | MatMulV2 + hcom_allReduce_ | MC2 融合 |
| tc.static_quant_linear_all_reduce | QuantBatchMatmulV3 + hcom_allReduce_ | 量化 MC2 |
| tc.multihead_latent_attention | BatchMatMulV2 + FIA + batch_matmul_transpose | MLA decode |
| tc.mlapo | MatMulV2 + KvRmsNormRopeCache | MLA 预处理 |

### 零开销算子

view, permute, split, split_with_sizes, select, slice, transpose, unsqueeze, expand, full, detach, alias, arange, t, convert_element_type

## 12. 工具参考

| 工具 | 用途 | 关键参数 |
|------|------|---------|
| `parse_kernel_details.py` | 将 kernel_details.csv 拆分为逐内核 CSV | `--device`, `--vllm-ascend-version`, `--kernel-details-path` |
| `extract_tc_ops.py` | 从 chrome trace 提取 TC 算子 | `--chrome-trace`, `--output`, `--op-mapping` |
| `validate.py` | 验证 CSV 数据库质量 | `--database` |
| `discover_operators.py` | 对比 profiling 与 op_mapping 覆盖率 | — |
| `generate_shape_grid.py` | 生成微基准测试 shape 网格 | — |
| `generate_microbench.py` | 生成 torch_npu 基准测试脚本 | — |
| `build_database.py` | 合并多个 CSV 数据源 | `--sources`, `--target` |

## 13. 相关文档

- [Op Mapping 技能](../../../.agents/skills/op-mapping/SKILL.md) —— 使用并行子代理的自动化 op_mapping 生成
- 更多 Skill 用法和协作流程见 §16

## 14. 新增融合算子映射 SOP

本节描述从零开始添加**单个**新融合算子映射的标准化流程。适用于人类维护者和 AI Agent。

> **与 op-mapping SKILL 的关系**：本节是 `docs/perf_database/skills/op-mapping/SKILL.md`
> 的**单算子手动版本**。op-mapping SKILL 采用并行 sub-agent 架构批量处理所有未映射算子
> （GATHER → FORWARD → REVERSE → ASSEMBLE → VERIFY → CORRECT 六阶段），
> 而本节聚焦单个算子的 6 步操作流程。两者共享相同的核心原则：
>
> - 3 条追踪路径（A/B/C）
> - 4 种 YAML 条目类型（compute / composite / zero_cost / decomposer）
> - 置信度级别（HIGH / MEDIUM / LOW / ASSUMPTION）
> - 验证方法（运行 TC 仿真 → 检查 MISS → 分类 gap）
>
> **何时用哪个**：添加 1-2 个算子 → 本节 SOP；批量生成/更新整个 op_mapping.yaml → op-mapping SKILL。
> 更多 skill 用法见 §16。

### 14.1 前置条件

开始之前，确保具备：

- [ ] tensor_cast/ops/ 中已定义目标 TC 融合算子
- [ ] 有访问 vllm-ascend、op-plugin、CANN 相关仓库的权限
- [ ] 有目标设备的 profiling 数据（kernel_details.csv）

### 14.2 映射流程

```text
┌─────────────────────────────────────────────────────────────────┐
│                     新增融合算子映射流程                           │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: 确定 TC 算子签名                                         │
│    ↓                                                            │
│  Step 2: 追踪 NPU 内核类型（选路径 A/B/C）                          │
│    ↓                                                            │
│  Step 3: 分析 Shape 差异                                         │
│    ↓                                                            │
│  Step 4: 编写 op_mapping.yaml 条目                               │
│    ↓                                                            │
│  Step 5: 实现 Shape Normalization（如需要）                        │
│    ↓                                                            │
│  Step 6: 验证与迭代                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 14.3 详细步骤

#### Step 1: 确定 TC 算子签名

```bash
# 查找算子定义
grep -r "def <op_name>" tensor_cast/ops/

# 查找注册信息
grep -r "register_tensor_cast_op.*<op_name>" tensor_cast/ops/
```

**输出解析**：提取输入参数列表、dtype 约束、shape 推导规则。

#### Step 2: 追踪 NPU 内核类型

根据算子来源选择追踪路径：

**路径 A: 标准 aten 算子（op-plugin 分发）**

```bash
# 1. 在 op-plugin 中查找
grep "<op_name>" /path/to/op-plugin/op_plugin/config/op_plugin_functions.yaml

# 2. 找到 C++ 实现
grep -r "<op_name>" /path/to/op-plugin/op_plugin/ops/opapi/ | head -3

# 3. 找到 aclnn 调用（在 C++ 文件中）
grep "EXEC_NPU_CMD" /path/to/op-plugin/op_plugin/ops/opapi/<OpName>KernelNpuOpApi.cpp

# 4. 找到 OPTYPE（在 CANN 仓库中）
grep -r "<aclnn_name>" /path/to/cann-ops-*/ --include="CMakeLists.txt"
```

**路径 B: torch_npu.npu_* 扩展算子**

```bash
# 1. 找到 vllm-ascend 调用
grep -r "torch_npu.npu_<op_name>" /path/to/vllm-ascend/

# 2. 在 op-plugin 中查找（同路径 A）
grep "npu_<op_name>" /path/to/op-plugin/op_plugin/config/op_plugin_functions.yaml
```

**路径 C: 自定义内核 / Triton / ATB**

```bash
# 1. 找到 vllm-ascend 自定义算子
grep -r "<op_name>" /path/to/vllm-ascend/vllm_ascend/ops/
grep -r "<op_name>" /path/to/vllm-ascend/csrc/

# 2. 判断类型：Triton / ATB / 自定义 C++
#    - Triton: 通常有 @triton.jit 装饰器
#    - ATB: 通常通过 torch_npu.atb.* 调用
#    - 自定义 C++: 通常在 csrc/ 目录下有 pybind 绑定

# 3. 内核类型通常是函数名或注册名
grep -r "<op_name>" /path/to/vllm-ascend/csrc/ --include="*.cpp" --include="*.cu"
```

**验证 Profiling 数据**：

```bash
# 确认 Type 列存在
grep "<kernel_type>" /path/to/kernel_details.csv | head -5

# 确认有足够样本
grep "<kernel_type>" /path/to/kernel_details.csv | wc -l
```

#### Step 3: 分析 Shape 差异

比较 TC 仿真中的 shape 与 NPU profiling 中的 shape：

```bash
# 提取 profiling 中的 shape 模式
awk -F',' 'NR>1 && $2=="<kernel_type>" {print $3}' kernel_details.csv | sort | uniq -c | sort -rn | head -10
```

**10 种常见差异类型**（与 op-mapping SKILL Key Principle #4 对齐，详见第 5 节）：

| # | 差异 | 处理方式 | 示例 |
|---|------|---------|------|
| 1 | 批次维度差异 (1 vs 无) | `batch_strip` 自动处理 | TC `(1,4096,5120)` → CSV `(4096,5120)` |
| 2 | FRACTAL_NZ 格式 | `fractal_nz_to_nd()` 还原 | `(320,256,1,1,16)` → `(5120,4096)` |
| 3 | 转置权重 | 检测并 exchange row/col | TC `(4096,5120)` → CSV `(5120,4096)` |
| 4 | 输入数量不同 | `tc_input_count` 截断 NPU 内部参数 | TC 2 输入 vs CSV 4 输入 |
| 5 | Padding 对齐 | `padding` 到 16/32 对齐 | TC `(4096,5120)` → CSV `(4096,5136)` |
| 6 | Flatten/reshape | `flatten` 多维→二维 | TC `(1,32,128)` → CSV `(32,128)` |
| 7 | SwiGlu 2→1 合并 | 自定义 normalization | TC 2 个输入 → CSV 1 个拼接输入 |
| 8 | RoPE cos/sin 截断 | 自定义 normalization | TC 全序列 → CSV 当前 token |
| 9 | Elementwise 广播 | `query_mode: elementwise` 按输出 shape 匹配 | 不同 broadcast 语义 |
| 10 | 通信算子 | `message_bytes` + `num_devices` 替代 shape | AllReduce/AllGather 等 |

> **注意**：差异 1-6 由 `_inputs_match()` 自动处理，差异 7-8 需要自定义 normalization（§14 Step 5 场景 A），
> 差异 9-10 通过特殊 `query_mode` 处理。详见 op-mapping SKILL 的 `ref/shape_matching_catalog.md`。

#### Step 4: 编写 op_mapping.yaml 条目

根据映射类型选择模板：

**类型 1: 标准 1:1 映射**

```yaml
"tc.<op_name>.default":
  kernel_type: <NPU_KernelType>
  alternate_kernel_types: [<BackupType1>, <BackupType2>]
  notes: >
    [CONFIDENCE] Path X.
    op-plugin: <File>.cpp → aclnn<API> → <KernelType>.
    Profiling: <KernelType>(N times).
```

**类型 2: 复合算子（静态分解）**

```yaml
"tc.<op_name>.default":
  composite: true
  sub_kernels: [<Kernel1>, <Kernel2>, <Kernel3>]
  notes: >
    [CONFIDENCE] Composite mapping.
    NPU kernels: <Kernel1> + <Kernel2> + <Kernel3>.
```

**类型 3: 复合算子（动态分解）**

```yaml
"tc.<op_name>.default":
  composite: true
  decomposer: true
  notes: >
    [CONFIDENCE] Composite with Python decomposer.
    Decomposition logic: _decompose_<op_name>_common in profiling_data_source.py.
```

**类型 4: 零开销算子**

```yaml
"tc.<op_name>.default":
  zero_cost: true
  notes: "Shape-only op, no NPU kernel execution."
```

**notes 字段置信度级别**：

- `[HIGH]`: 高频/关键路径，已在多个模型 profiling 中验证
- `[MEDIUM]`: 中频，已在至少一个模型中验证
- `[LOW]`: 低频/placeholder，尚未在实际 profiling 中出现
- `[ASSUMPTION]`: 设计假设，待实际 profiling 验证

**关键约束规则**（与 op-mapping SKILL Key Principles 对齐）：

> 以下规则直接源自 `docs/perf_database/skills/op-mapping/SKILL.md` 的 11 条核心原则，
> 手动添加新算子时必须遵守。

1. **kernel_type = CSV 文件名**（SKILL Principle #10）：
   `kernel_type` 必须精确匹配 `kernel_details.csv` 的 `Type` 列值（即 per-kernel CSV 的文件名，不含 `.csv`）。
   禁止使用不匹配 CSV 的"规范"内核名。`csv_file` 字段已废弃。

2. **互斥条目类型**（SKILL Principle #5）：
   每个条目只能是 `kernel_type`、`composite`、`zero_cost` 三者之一，不可混用。

3. **tc_input_count 安全规则**（SKILL Principle #7）：
   `tc_input_count` 仅用于截断 NPU 内部参数（如 axis、scale），**不可**用于 elementwise 广播算子。
   详见 SKILL 的 `ref/tc_input_count_rules.md`。

4. **zero_cost 分类规则**（SKILL Principle #8）：
   标记 `zero_cost: true` 前必须验证：(a) 该 kernel Type 从未出现在 profiling 中，且 (b) 其延迟已被融合内核包含。
   详见 SKILL 的 `ref/zero_cost_classification.md`。

5. **elementwise query_mode**（SKILL Principle #9）：
   内存受限的逐元素算子（add、mul、div 等）应设置 `query_mode: elementwise`，按输出 shape 匹配并做 dtype-relaxed byte-ratio scaling。

6. **alternate_kernel_types 禁令**（SKILL Principle #11）：
   `alternate_kernel_types` 必须与主 `kernel_type` 处于**同一抽象层级**（如 MatMulV2 → MatMulV3 = 硬件变体，OK）。
   **禁止**用融合超算子作为子算子的 alternate（如 `DispatchFFNCombine` 不可作为 `init_routing_v2` 的 alternate，否则会导致延迟严重高估）。

7. **通信算子**（SKILL Principle #6）：
   通信算子使用 `message_bytes` + `num_devices` 匹配，不使用 shape 匹配。

#### Step 5: 实现代码扩展（如需要）

大多数新算子**只需修改 YAML**。但以下场景需要修改 `profiling_data_source.py`：

**场景 A: 需要自定义 shape normalization**

当 TC 输入 shape 与 profiling CSV shape 存在非通用差异时（如 SwiGlu 的 2→1 输入合并），
需要在 `_inputs_match()` 方法中添加处理。代码扩展点：

```python
# 1. 在文件顶部添加 kernel frozenset 常量
#    搜索 _SWIGLU_KERNELS 或 _ROPE_KERNELS 找到定义区域
_NEW_OP_KERNELS = frozenset({"NewKernelType"})

# 2. 如需复杂转换，添加模块级归一化函数
#    搜索 _normalize_rope_inputs() 或 _normalize_reshape_and_cache_inputs() 找到定义区域
def _normalize_new_op_inputs(
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
) -> List[Tuple[Tuple[int, ...], torch.dtype]]:
    """将 TC 输入布局转换为 profiling CSV 布局"""
    ...

# 3. 在 _inputs_match() 方法中添加归一化分支
#    搜索 "if kernel_type in _SWIGLU_KERNELS" 找到现有分支，在其后添加
if kernel_type in _NEW_OP_KERNELS and <条件>:
    tc_inputs_normalized = _normalize_new_op_inputs(tc_inputs)
```

**场景 B: 需要动态分解（composite + decomposer）**

当一个 TC 算子对应 NPU 上多个 kernel，且分解逻辑依赖运行时 shape 时：

```python
# 1. 编写分解函数
#    搜索 _decompose_mla_common() 或 _decompose_mlapo_common() 找到定义区域
def _decompose_new_op(
    op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[List[SubKernelSpec]]:
    """分解 new_op 为子 kernel 列表"""
    args = op_invoke_info.args
    # 从 args 提取 shape 信息，构造 SubKernelSpec 列表
    return [
        SubKernelSpec(kernel_type="SubKernel1", input_shapes=[...], dtype="DT_BF16"),
        SubKernelSpec(kernel_type="SubKernel2", input_shapes=[...], dtype="DT_BF16"),
    ]

# 2. 注册到 COMPOSITE_DECOMPOSERS 字典
#    搜索 "COMPOSITE_DECOMPOSERS" 找到字典定义
COMPOSITE_DECOMPOSERS["tensor_cast.new_op.default"] = _decompose_new_op
```

**场景 C: 需要全新查询模式**

当现有的 compute / attention_special / elementwise / moe_fused 都不适用时：

```python
# 1. 在 op_mapping.yaml 中定义新的 query_mode 值
#    query_mode: new_mode

# 2. 在 lookup() 分派链中添加新分支
#    搜索 "query_mode" 找到现有分派逻辑
if mapping.get("query_mode") == "new_mode":
    return self._lookup_new_mode(op_invoke_info, mapping)

# 3. 实现 _lookup_new_mode() 方法
def _lookup_new_mode(self, op_invoke_info, mapping) -> Optional[QueryResult]:
    ...
```

**场景 D: 需要新的 dtype 兼容性**

```python
# 1. 更新 DTYPE_MAP: 添加 torch dtype → profiling dtype 映射
#    搜索 "DTYPE_MAP" 找到定义位置
# 2. 更新 _DTYPE_COMPAT: 添加新的 dtype 等价组
#    搜索 "_DTYPE_COMPAT" 找到定义位置
# 3. 更新 _DTYPE_RELAXED_KERNELS: 添加允许宽松匹配的 kernel
#    搜索 "_DTYPE_RELAXED_KERNELS" 找到定义位置
```

**判断是否需要代码修改的决策树**：

```text
TC shape 与 CSV shape 完全一致？
  ├── 是 → 仅需 YAML（场景无需代码）
  └── 否 → 差异是否属于已有规则？
        ├── batch_strip / transpose / padding / flatten → 仅需 YAML
        └── 否 → 需要新的 normalization（场景 A）
                  或 composite + decomposer（场景 B）
                  或 新 query_mode（场景 C）
```

#### Step 6: 验证与迭代

```bash
# 运行 TC 仿真
python3.10 -m cli.inference.text_generate $MODEL \
  --num-queries $NQ --query-length $QL \
  --performance-model profiling --compile \
  --profiling-database $DATA_DIR

# 检查匹配率
# 输出中查找: EmpiricalPerformanceModel: X/Y ops matched
```

**验证检查点**：

- [ ] 目标算子出现在仿真输出中
- [ ] 目标算子无 MISS（或有预期的 MISS）
- [ ] 如果有 MISS，分类为 shape 数据缺口（非映射错误）

### 14.4 AI Agent 执行指令

> **自动化替代方案**：如果需要批量处理多个算子（>5 个），建议直接使用 op-mapping SKILL
> 而非逐个执行本节指令。使用方式：
>
> 1. 读取 `docs/perf_database/skills/op-mapping/SKILL.md`
> 2. 按 SKILL 的 "Required Inputs" 准备输入参数
> 3. 按 SKILL 的六阶段流程（GATHER → FORWARD → REVERSE → ASSEMBLE → VERIFY → CORRECT）执行
>
> 本节指令适用于添加 1-2 个算子的场景，是 SKILL 的简化单算子版本。

以下格式面向 AI Agent 设计，可直接执行：

````markdown
## TASK: 添加新融合算子映射

### INPUT

- op_name: <tc.<op_name>.default>
- vllm_ascend_path: /path/to/vllm-ascend
- op_plugin_path: /path/to/op-plugin
- cann_ops_path: /path/to/cann-ops-*
- profiling_data: /path/to/kernel_details.csv

### EXECUTION

1. **确认 TC 算子签名**

   ```bash
   grep -r "def <op_name>" tensor_cast/ops/
   ```

   OUTPUT: 提取函数签名和输入 dtype 约束

2. **追踪 NPU 内核类型（自动选路径）**
   - IF 算子名匹配 `aten.*`: 执行路径 A
   - IF 算子名匹配 `torch_npu.npu_*`: 执行路径 B
   - ELSE: 执行路径 C

3. **验证 profiling 数据**

   ```bash
   grep "<kernel_type>" kernel_details.csv | wc -l
   ```

   CONDITION: count > 0 → 继续; count = 0 → 警告: 缺少 profiling 数据

4. **生成 YAML 条目**
   根据映射类型选择模板，填充字段

5. **运行验证**

   ```bash
   python3.10 -m cli.inference.text_generate $MODEL \
     --performance-model profiling --compile \
     --profiling-database $DATA_DIR 2>&1 | grep -E "matched|MISS|<op_name>"
   ```

### SUCCESS CRITERIA

- 目标算子 0 MISS，或
- MISS 原因为 "shape data gap"（已 documented）

### FAILURE RECOVERY

- kernel_type 不在 profiling 中 → 检查 CANN 版本匹配
- shape 不匹配 → 分析 shape 差异，可能需要 normalization
- 复合算子失效 → 验证每个 sub_kernel 查询结果

### OUTPUT ARTIFACTS

- op_mapping.yaml 新增条目
- profiling_data_source.py 新增 normalization 函数（如需要）
- notes 字段包含完整证据链

````

## 15. CANN 版本升级 Checklist

本节提供 CANN 版本升级时的系统性检查清单，确保 op_mapping 的正确性和完整性。

### 15.1 升级流程概览

```text
┌─────────────────────────────────────────────────────────────────┐
│                    CANN 版本升级流程                             │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: 版本差异检测                                           │
│    ↓                                                            │
│  Phase 2: 映射更新                                               │
│    ↓                                                            │
│  Phase 3: CSV Schema 验证                                        │
│    ↓                                                            │
│  Phase 4: 代码适配                                               │
│    ↓                                                            │
│  Phase 5: 回归测试                                               │
│    ↓                                                            │
│  Phase 6: 文档更新                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 15.2 详细 Checklist

#### Phase 1: 版本差异检测

```bash
# 1.1 提取新旧 profiling 的 unique kernel types
awk -F',' 'NR>1 {print $2}' old_kernel_details.csv | sort -u > old_types.txt
awk -F',' 'NR>1 {print $2}' new_kernel_details.csv | sort -u > new_types.txt

# 1.2 对比差异
diff old_types.txt new_types.txt > types_diff.txt

# 1.3 分类变更
#    - 仅在旧版: 移除的内核
#    - 仅在新版: 新增的内核
#    - 两者都有: 可能重命名或保持不变
```

**检查点**：

- [ ] 1.1 提取了两个版本的 unique kernel types
- [ ] 1.2 生成了差异报告
- [ ] 1.3 将差异分类为：重命名、新增、移除、融合变更

**变更分类判断**：

| 分类 | 判断依据 | 示例 |
|------|---------|------|
| **重命名** | 新旧版本功能相同，仅名称变化 | ScatterElements → ScatterElementsV2 |
| **新增** | 全新功能或优化内核 | MatMulV3, DispatchFFNCombine |
| **移除** | 旧版存在，新版消失 | RINGMLAPrefillBF16Kernel (被融合) |
| **融合变更** | 多个独立内核合并为一个 | matmul+swiglu → GroupedMatmulSwigluQuant |

#### Phase 2: 映射更新

**检查点**：

- [ ] 2.1 对于重命名内核：
  - 更新 `kernel_type` 为新名称
  - 用 `alternate_kernel_types` 保留旧名称以确保向后兼容
  - 示例：

    ```yaml
    "aten.scatter.value":
      kernel_type: ScatterElementsV2          # CANN 8.5+
      alternate_kernel_types: [ScatterElements]  # CANN 8.3 兼容
    ```

- [ ] 2.2 对于新增内核：
  - 按 SOP（第 14 节）追踪映射路径
  - 确定是替代现有映射还是新算子
  - 添加新条目或更新现有条目

- [ ] 2.3 对于移除内核：
  - 检查是否有替代内核（通常是被融合）
  - 更新 affected mappings
  - 如果是独立算子被移除，标记为 `accepted_miss` 或删除条目

- [ ] 2.4 对于融合变更：
  - 可能需要从单一 kernel_type 转为 `composite: true`
  - 或从 composite 转为单一 kernel_type（反向融合）
  - 更新 notes 说明变化原因

#### Phase 3: CSV Schema 验证

```bash
# 3.1 检查列名
head -1 old_kernel_details.csv
head -1 new_kernel_details.csv

# 3.2 检查 shape 格式
awk -F',' 'NR>1 {print $3}' new_kernel_details.csv | head -5

# 3.3 检查 Type 列格式
awk -F',' 'NR>1 {print $2}' new_kernel_details.csv | grep -v "^[A-Za-z0-9_]*$" | head -5
```

**检查点**：

- [ ] 3.1 列名无变化（或已适配）
- [ ] 3.2 shape 格式无变化（或已适配）
- [ ] 3.3 Type 列格式符合预期

#### Phase 4: 代码适配

**检查点**：

- [ ] 4.1 检查 `profiling_data_source.py` 中的 shape normalization：
  - 新内核是否需要特殊处理？
  - 现有 normalization 是否仍然有效？

- [ ] 4.2 检查 dtype 映射：
  - 新版是否引入新的 dtype？
  - dtype 名称是否有变化？

- [ ] 4.3 检查复合算子分解逻辑：
  - `decomposer` 函数是否需要更新？
  - 分解产生的子内核是否仍然有效？

#### Phase 5: 回归测试

```bash
# 5.1 运行单元测试
python3.10 -m pytest tests/perf_database/ -v

# 5.2 检查匹配率
python3.10 -m cli.inference.text_generate $MODEL \
  --performance-model profiling --compile \
  --profiling-database $NEW_DATA_DIR 2>&1 | grep "matched"
```

**检查点**：

- [ ] 5.1 所有单元测试通过
- [ ] 5.2 参考模型测试通过
- [ ] 5.3 匹配率不低于升级前（或差异已 explain）
- [ ] 5.4 无新增 unexpected MISS

#### Phase 6: 文档更新

**检查点**：

- [ ] 6.1 更新 `op_mapping.yaml` 头部的版本信息：

  ```yaml
  version: "0.xx.0"
  cann_version: "8.x"
  collection_date: "20xx-xx-xx"
  ```

- [ ] 6.2 更新 TUTORIAL 中的示例（如有变化）

- [ ] 6.3 更新 CHANGELOG：

  ```text
  ## YYYY-MM-DD - CANN X.X 升级
  ### 变更摘要
  - 新增内核: MatMulV3, DispatchFFNCombine
  - 移除内核: RINGMLAPrefillBF16Kernel
  - 重命名: ScatterElements → ScatterElementsV2

  ### 映射更新
  - aten.mm: 添加 MatMulV3 作为 alternate
  - tensor_cast.dispatch_ffn_combine: 新增条目
  ```

### 15.3 常见升级场景处理

#### 场景 1: 内核重命名

**问题**: `Type` 列名称变化，功能不变

**处理**:

```yaml
"aten.scatter.value":
  kernel_type: ScatterElementsV2           # 新名称
  alternate_kernel_types: [ScatterElements]  # 旧名称兼容
```

#### 场景 2: 内核融合

**问题**: 多个独立内核合并为一个融合内核

**处理**:

```yaml
# 旧版: 多个独立算子
"tc.gate_up_proj":
  composite: true
  sub_kernels: [MatMulV2, MatMulV2]

# 新版: 单个融合内核
"tc.gate_up_proj":
  kernel_type: GroupedMatmulSwigluQuant
```

#### 场景 3: 内核拆分

**问题**: 一个内核拆分为多个独立内核

**处理**:

```yaml
# 旧版: 单个内核
"tc.mla":
  kernel_type: RINGMLAPrefillBF16Kernel

# 新版: 拆分为多个子内核
"tc.mla":
  composite: true
  decomposer: true
  notes: "MLA composite, 原融合内核被拆分"
```

#### 场景 4: 全新内核类型

**问题**: 新增从未存在过的内核类型

**处理**: 按 SOP（第 14 节）添加新条目

### 15.4 AI Agent 执行指令

````text
## TASK: CANN 版本升级适配

### INPUT
- old_profiling: /path/to/old/kernel_details.csv
- new_profiling: /path/to/new/kernel_details.csv
- old_op_mapping: /path/to/old/op_mapping.yaml
- cann_old_version: "8.3"
- cann_new_version: "8.5"

### EXECUTION

1. **差异检测**
   ```bash
   awk -F',' 'NR>1 {print $2}' $old_profiling | sort -u > old_types.txt
   awk -F',' 'NR>1 {print $2}' $new_profiling | sort -u > new_types.txt
   diff old_types.txt new_types.txt
   ```
   OUTPUT: 分类变更列表

2. **FOR EACH 变更**:
   - 重命名: 更新 kernel_type + 添加 alternate
   - 新增: 按 SOP 追踪并添加条目
   - 移除: 检查替代，更新或删除
   - 融合: 调整映射类型

3. **Schema 验证**
   ```bash
   head -1 $new_profiling
   awk -F',' 'NR>1 {print $3}' $new_profiling | head -10
   ```

4. **回归测试**
   ```bash
   python3.10 -m pytest tests/perf_database/ -v
   ```

### SUCCESS CRITERIA
- 所有测试通过
- 匹配率不低于升级前（或差异已 documented）
- 无新增 unexpected MISS

### FAILURE RECOVERY
- 测试失败: 检查 affected mappings，回滚变更
- 匹配率下降: 分析 MISS 原因，可能是 shape 数据缺口
- Schema 不匹配: 适配代码逻辑
````

## 16. 相关 Skill 用法指南

本项目使用多个 AI Agent Skill 来自动化 op_mapping 维护、MISS 分析和 CI 监控。
Skill 分为两类：**仓库内 Skill**（随代码版本管理）和**外部 Skill**（部署在 CI 环境中）。

### 16.1 仓库内 Skill

位于 `docs/perf_database/skills/` 目录下，随代码一起版本管理。

| Skill | 路径 | 用途 | 触发方式 |
|-------|------|------|---------|
| **op-mapping** | `skills/op-mapping/SKILL.md` | 批量生成/更新 op_mapping.yaml | 手动触发 |
| **microbench** | `skills/microbench/SKILL.md` | 生成 op_replay 脚本用于 NPU 微基准测试 | 手动触发 |
| **project-status** | `skills/project-status/SKILL.md` | 生成项目进展看板（M1-M6 指标） | `/project-status` |
| **doc-plan-update** | `skills/doc-plan-update/SKILL.md` | 更新文档和计划 | 手动触发 |
| **weekly-report** | `skills/weekly-report/SKILL.md` | 生成周报 | 手动触发 |
| **delivery-report-q1** | `skills/delivery-report-q1/SKILL.md` | Q1 交付报告 | 手动触发 |

#### op-mapping Skill 详解

**与本教程 §14 的关系**：§14 是单算子手动 SOP，op-mapping Skill 是批量并行自动化版本。
两者源自相同的设计原则，§14 的 6 步流程是 SKILL 六阶段的单算子简化。

**核心流程**：

```text
Phase 1: GATHER    — 运行 TC 仿真 + 提取 TC ops + 解析 profiling types
Phase 2: FORWARD   — 并行 sub-agent 逐算子追踪 TC→NPU 映射
Phase 3: REVERSE   — 并行 sub-agent 处理未覆盖的 profiling types
Phase 4: ASSEMBLE  — 合并 YAML 片段 + 添加元数据 + 按类别组织
Phase 5: VERIFY    — 运行 TC 仿真验证 + gap 分类
Phase 6: CORRECT   — 迭代修复 MISS
```

**使用场景**：

- 首次为新模型/新 CANN 版本生成完整 op_mapping.yaml
- 大批量更新（>5 个算子需要新增/修改映射）
- CANN 版本升级后全量重新验证（配合 §15 Checklist）

**具体使用步骤**：

1. **准备输入参数**（对应 SKILL 的 "Required Inputs"）：

   ```text
   - 目标模型: HuggingFace ID（如 Qwen/Qwen3-32B）
   - 设备配置: 如 ATLAS_800_A3_752T_128G_DIE
   - 并行配置: world-size, tp-size, dp-size, ep flag
   - 量化方式: DISABLED / W8A8_STATIC / W4A8_STATIC / FP8 / MXFP4
   - Profiling CSV: kernel_details.csv 路径
   - 软件栈版本: vLLM、vLLM-ascend、op-plugin、pytorch-npu、CANN ops 各仓库版本
   - 本地仓库路径: 各仓库的 checkout 路径（或 URL + tag 自动 clone）
   ```

2. **触发 Skill**：
   - 在 AI Agent（如 Claude/CodeFuse）对话中，提供上述输入参数
   - Agent 读取 `docs/perf_database/skills/op-mapping/SKILL.md` 后自动执行六阶段流程
   - 每个阶段的 sub-agent 独立追踪一个算子，避免上下文污染

3. **检查输出**：
   - 生成的 `op_mapping.yaml` 位于：
     `tensor_cast/performance_model/profiling_database/data/$DEVICE/vllm_ascend/$VERSION/op_mapping.yaml`
   - 验证报告包含：匹配率、gap 分类、每个 MISS 的详细原因

4. **迭代修正**：
   - Phase 5 VERIFY 会自动分类 gap（op_mapping 错误 vs shape 数据缺口 vs TC 建模偏差）
   - Phase 6 CORRECT 自动修复 op_mapping 错误，重新验证直到通过

**§14 SOP 与 SKILL 的流程对应关系**：

| §14 SOP 步骤 | op-mapping SKILL 阶段 | 差异 |
|-------------|---------------------|------|
| Step 1: 确定 TC 算子签名 | Phase 1: GATHER (1c) | SKILL 自动提取所有未映射算子 |
| Step 2: 追踪 NPU 内核类型 | Phase 2: FORWARD | SKILL 并行 sub-agent 处理 |
| Step 3: 分析 Shape 差异 | Phase 2: FORWARD (worker) | SKILL worker 自动分析 10 种差异 |
| Step 4: 编写 YAML 条目 | Phase 4: ASSEMBLE | SKILL 自动合并+去重+分类 |
| Step 5: 代码扩展 | Phase 6: CORRECT | SKILL 标记需要人工干预的项 |
| Step 6: 验证 | Phase 5: VERIFY | SKILL 自动运行 TC 仿真验证 |

**11 条核心原则**（与 §14 共享）：

1. Profiling Name 三段式结构
2. Type 列 = OPTYPE = CSV 文件名
3. 三条追踪路径（A/B/C）
4. 10 种 shape 差异
5. 互斥条目类型
6. 通信算子用 message_bytes
7. tc_input_count 安全规则
8. zero_cost 分类规则
9. elementwise query_mode
10. kernel_type = CSV 文件名
11. 禁止 sub-op → fused-op 作为 alternate

#### microbench Skill 详解

当 MISS 分析确定根因为 RC3（CSV 数据缺失）时，使用此 Skill 生成 NPU 微基准测试脚本。

**输入**：kernel_type + CSV 路径 + op_mapping.yaml 中的 `microbench_api`

**输出**：`tools/perf_data_collection/op_replay/<KernelType>_run.py`

**与 §14 的关系**：§14 Step 6 验证发现 shape 数据缺口时，用 microbench Skill 补采数据。

### 16.2 外部 Skill（CI 环境）

以下 Skill 部署在 CI 环境中，不在本仓库内，但与 op_mapping 维护流程紧密相关。

| Skill | 用途 | 与本教程的关系 |
|-------|------|---------------|
| **perf-db-ci** | 自动监控 feat/perf-database 新 commit，运行 4 场景 TC 仿真，更新飞书跟踪表 | 持续验证 §14/§15 的改动效果 |
| **perf-db-miss-analysis-workflow** | 提取 per-op MISS 详情，诊断根因（RC1-RC6），生成 CSV 跟踪表 | 提供 §14 Step 6 验证后的深度分析 |

#### perf-db-ci Skill

**功能**：每次 `feat/perf-database` 有新 commit 时自动执行：

1. 拉取最新代码
2. 运行 4 场景 TC 仿真（Qwen3 PF/DC + DSv3 PF/DC）
3. 计算 M1-M6 指标
4. 分类每个 MISS op 的解决状态
5. 更新飞书跟踪表

**4 场景参数基线**（与 project-status Skill 对齐）：

| 场景 | 模型 | NQ | QL | TP | 量化 |
|------|------|:--:|:--:|:--:|------|
| Qwen3 PF | Qwen3-32B | 1 | 4112 | 16 | DISABLED |
| Qwen3 DC | Qwen3-32B | 16 | 1 | 16 | DISABLED |
| DSv3 PF | DeepSeek-V3 | 1 | 4099 | 8 | W8A8_STATIC |
| DSv3 DC | DeepSeek-V3 | 1 | 1 | 8 | W8A8_STATIC |

#### perf-db-miss-analysis-workflow Skill

**根因分类体系**（RC1-RC6）：

| RC | 根因 | 修复方式 | 相关教程章节 |
|----|------|---------|-------------|
| RC1 | Shape 匹配规则不足 | 修改 `profiling_data_source.py` | §14 Step 5 场景 A |
| RC2 | TC 建模偏差 | 修改 TC transformers/layers | §9 端到端验证 |
| RC3 | CSV 数据缺失 | microbench 补采 | §16.1 microbench Skill |
| RC4 | op_mapping.yaml 配置错误 | 修改 op_mapping.yaml | §14 Step 4 |
| RC5 | 融合 Pass 缺失/多余 | 修改 compilation/patterns/ | §10 常见陷阱 #1 |
| RC6 | 数据采集工具问题 | 修改 tools/perf_data_collection/ | §12 工具参考 |

### 16.3 Skill 协作流程

典型的 op_mapping 维护闭环：

```text
1. perf-db-ci 检测到新 commit，运行 4 场景仿真
   ↓
2. perf-db-ci 发现新 MISS ops，更新飞书表
   ↓
3. perf-db-miss-analysis-workflow 对每个 MISS 做 RC1-RC6 根因诊断
   ↓
4. 根据根因选择修复路径：
   - RC4 (op_mapping 错误) → 手动修复 YAML (§14 SOP) 或 op-mapping Skill 重新生成
   - RC3 (CSV 缺失) → microbench Skill 生成 op_replay 脚本 → NPU 补采
   - RC1 (匹配规则) → 手动修改 profiling_data_source.py (§14 Step 5)
   - RC2/RC5 → 修改 TC 建模代码（超出 op_mapping 范围）
   ↓
5. 提交修复 → perf-db-ci 自动验证 → 飞书表更新
```

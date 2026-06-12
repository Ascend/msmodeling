# 形状匹配规则目录 (Shape Matching Catalog)

## 概述

TC 模拟的张量形状与 NPU profiling CSV 的形状存在系统性差异。
`profiling_data_source.py` 的 `_inputs_match()` 按顺序尝试以下匹配规则。

## 10 种形状差异类型

### 1. 批次维度剥离 (Batch dim strip)

- **TC**: `(1, S, D)` — 保持显式 batch dim
- **NPU**: `(S, D)` — 单 batch 时省略
- **处理**: `_strip_batch_dim()` 剥离第一维 =1 的情况
- **适用**: 所有 3D→2D 算子

### 2. 序列填充 (Seq padding)

- **TC**: `ceil(S/block) * block` — 填充到 NPU tile 对齐
- **NPU**: raw S — 原始序列长度
- **处理**: `_shapes_match_with_padding()` 容忍 block ∈ {16, 32, 64}
- **适用**: matmul, norm 等计算算子

### 3. FRACTAL_NZ 格式

- **TC**: ND `(K, N)` — 标准二维
- **NPU**: `[H, W, bh, bw]` — FRACTAL_NZ 分块布局
- **处理**: `fractal_nz_to_nd()` 还原: `(H*bw, W*bh)`
- **适用**: 所有标记 `FRACTAL_NZ` 格式的权重

### 4. ND 权重转置

- **TC**: `(K, N)` — 来自 `weight.T`
- **NPU**: `(N, K)` — 存储顺序不同
- **处理**: 对 `_MATMUL_KERNELS` 的第 2+ 个输入检查转置
- **适用**: MatMulV2, MatMulV3, MatMulCommon, QuantBatchMatmulV3, BatchMatMulV2

### 5. SwiGlu 输入拼接

- **TC**: 2 个输入 `(S, D/2)` — gate 和 up 分开
- **NPU**: 1 个输入 `(S, D)` — 拼接后的
- **处理**: `_SWIGLU_KERNELS` 特殊逻辑,合并 last dim
- **适用**: SwiGlu kernel

### 6. RoPE 布局转置

- **TC**: `(B, H, S, D)` 顺序 `[Q, K, cos, sin]`
- **NPU**: `(B, S, H, D)` 顺序 `[K, Q, cos, sin]`
- **处理**: `_normalize_rope_inputs()` 交换 Q↔K + 转置 H↔S
- **适用**: `_ROPE_KERNELS` (InterleaveRope, ApplyRotaryPosEmb, _triton_rope, etc.)

### 7. RoPE 多 kernel 变体

- **TC**: 单个 `apply_rope` op
- **NPU**: InterleaveRope (interleave mode) / ApplyRotaryPosEmb (neox) / _triton_rope (CANN 8.5)
- **处理**: `alternate_kernel_types` 列表
- **适用**: tensor_cast.apply_rope.default

### 8. 复合算子分解

- **TC**: 融合 op (如 matmul_all_reduce)
- **NPU**: 可能拆为独立 kernel
- **处理**: `composite: true` + `sub_kernels` 列表,`_lookup_composite()` 分别查询求和
- **适用**: MC2 (matmul+allreduce), MLA (bmm+FIA+transpose_bmm), MLAPO (matmul+kvnormrope)

### 9. 批次展平 (Flatten batch)

- **TC**: `(B, M, D)` — 3D (batch, seq, hidden)
- **NPU**: `(B*M, D)` — 2D (tokens, hidden)
- **处理**: `_FLATTEN_BATCH_KERNELS` 集合, 尝试 `(B*M, D)` 匹配
- **适用**: AscendQuantV2, DynamicQuant, RmsNorm, AddRmsNormBias, AddRmsNorm, **DispatchFFNCombine**

### 10. 末维合并 (Merge last dims)

- **TC**: `(T, H, D)` — per-head 量化 (MLA)
- **NPU**: `(T, H*D)` — 合并为 hidden_dim
- **处理**: `_MERGE_LAST_DIMS_KERNELS` 集合
- **适用**: AscendQuantV2, DynamicQuant (MLA 输出量化路径)

### 11. 输出形状匹配 (Output-shape matching for elementwise ops)

- **TC**: 匹配输出形状, 任何输入广播模式
- **NPU**: 输出形状确定 (CSV `Output Shapes` 列)
- **Dtype**: 松弛 — 按字节比缩放延迟 (FP32/BF16 = 4/2 = 2×)
- **处理**: `query_mode: elementwise` → `_lookup_elementwise()` / `_interpolate_elementwise()`
- **适用**: Add, Mul, Div (内存带宽受限的逐元素算子)
- **与 tc_input_count 互斥**: 设置 `query_mode: elementwise` 后不得设置 `tc_input_count`

## 新增形状规则的流程

1. 在 E2E 验证中发现 `shape_mismatch` MISS
2. 对比 TC 形状 vs CSV 形状,识别系统性差异
3. 确认差异不是数据缺失(需检查 CSV 是否有该 batch size)
4. 在 `_inputs_match()` 中添加新的匹配规则
5. 添加到对应的 kernel 集合 (如 `_FLATTEN_BATCH_KERNELS`)
6. 添加测试用例
7. 更新本文档

## tc_input_count 与形状匹配的交互

`tc_input_count` 在形状匹配之前截断输入:

```python
# _lookup_compute() 中:
tc_inputs = tc_inputs[:tc_input_count]  # 截断 TC 输入

# _inputs_match() 中:
csv_shapes = csv_shapes[:tc_input_count]  # 截断 CSV 输入
```

截断后,剩余输入按上述 10 种规则匹配。详见 `ref/tc_input_count_rules.md`。

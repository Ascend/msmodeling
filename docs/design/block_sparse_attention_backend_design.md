# Block Sparse Attention backend 实现设计

## 修订记录

| 日期 | 修订版本 | 修改描述 | 作者 |
| --- | --- | --- | --- |
| 2026-08-05 | 1.0 | 固化 Diffusers BSA dispatch、fallback、route lifecycle 与 analytic estimator 契约 | minghang_c |

## 1. 背景与范围

TensorCast 通过 Diffusers attention dispatch 和 `scaled_dot_product_attention` context 将 attention 调用转换为 TensorCast semantic ops。BSA backend 在该入口上增加 route generation 和 block sparse attention。本设计定义其跨层行为与实现契约。

范围包括：

- backend 配置和 capability validation；
- Diffusers dispatch 与 direct SDPA 路径；
- unsupported-call fallback 统计；
- route plan/route metadata 生命周期；
- BSA analytic performance properties；
- package registration 边界；
- video generation、cache、compile 和 sequence parallelism 交互；
- 测试、文档和回滚。

不修改真实数值 kernel，不增加 empirical mapping，不重构既有 SDPA monkey patch 和 SP 通信模型。

## 2. 决策摘要

| 决策点 | 结论 | 原因 |
| --- | --- | --- |
| Estimator | 离散 padded block-count | 连续 sparsity 无法表达 block size、取整和 edge tile |
| K/V memory | 保守 selected-union，单次完整 K/V HBM read | Route metadata 不提供 route union/reuse；禁止无依据线性缩放 |
| Unsupported calls | 聚合可见 dense fallback | 保留包含 cross-attention 的视频模型兼容性 |
| Route plan ownership | Run-scoped | 一个 context 包围 main/cache model，model config 状态重复且无效 |
| BSA + attention quantization | Backend boundary 拒绝 | 防止 direct API 产生混合 BSA/quantized-dense 语义 |
| Dense + BSA-only options | 拒绝非默认值 | 防止用户误以为参数生效 |
| Diffusers registration | Model/adapter 边界定向注册 | 减少无关 import 和私有 API 的进程级副作用 |

## 3. 配置模型

### 3.1 `AttentionRoutePlan`

Plan 保留三个字段：

- `backend`: `dense` 或 `block_sparse_attention`；
- `block_size`: 正整数，默认 `DEFAULT_BLOCK_SPARSE_ATTENTION_BLOCK_SIZE`（`128`）；
- `sparsity`: `[0.0, 1.0)`。

Plan 表达一次 inference run 的 backend policy，不表达每层生成的真实 route。CLI 构造 plan 后直接传入 `use_custom_sdpa(...)`。`DiffusersTransformerConfig` 不保存 plan；main/cache model config 不写入 plan。

Validation 规则：

1. `block_size > 0`；
2. `0.0 <= sparsity < 1.0`；
3. `dense` 只接受 `DEFAULT_BLOCK_SPARSE_ATTENTION_BLOCK_SIZE`（`128`）和 `sparsity == 0.0`；
4. BSA context 不接受非空 attention quant config。

CLI 的 quantization 检查只提供更早反馈，不能替代 context validation。

### 3.2 生命周期

`use_custom_sdpa(...)` 进入时保存原始 SDPA function、quant config、route plan 和 route stats；退出时按相反顺序恢复。嵌套 context 必须恢复外层状态。现有 `use_custom_sdpa(quant_config)` positional 调用保持兼容，route plan 和 stats 通过可选参数表达。

一次 `run_inference(...)` 只创建一个 plan。main model 与 cache model 在同一 context 内共享 plan。active model 切换不读取 model config 中的 backend 状态。

## 4. Dispatch 设计

### 4.1 Capability 判定

BSA eligibility 使用一个返回 `None | reason_code` 的判定函数。返回 `None` 表示支持；返回稳定 reason code 表示必须 dense fallback。

判定顺序固定，保证一个 call 只归入一个首要原因：

1. `non_4d_qkv`
2. `qkv_shape_mismatch`
3. `unsupported_attention_mask`
4. `dropout`
5. `causal`
6. `custom_scale`
7. `gqa`

支持条件：

- Q/K/V 都为 4D；
- Q/K/V shape 相同；
- mask 为 `None`，或 dtype 为 bool、rank 为 4、shape 为 `(batch, 1, query_len, key_len)`；
- dropout 为 0；
- 非 causal；
- 不提供 custom scale；
- 不启用 GQA。

Backend 不是 BSA 时不记录 fallback。

### 4.2 Canonical layout

Diffusers registry backend 接收 canonical `(batch, sequence, heads, head_dim)` layout。Direct SDPA 接收 `(batch, heads, sequence, head_dim)` layout；满足 BSA 条件时先转置为 canonical layout，执行 route 和 BSA op，再转回原 layout。

Unsupported direct SDPA calls 保持既有 dense path，不顺带改变 dense layout 语义。Fallback reason 在进入 dense path 前只记录一次。

### 4.3 执行路径

Eligible BSA call：

```text
Q/K/V
  -> attention_route_generate(Q, K, block_size, sparsity)
  -> block_sparse_attention(Q, K, V, mask, route_metadata, block_size, sparsity)
```

Unsupported BSA call：

```text
Q/K/V
  -> record fallback reason
  -> dense tensor_cast.attention
```

BSA context 进入前已拒绝 attention quantization，因此 unsupported BSA call 不会转入 quantized dense attention。

## 5. Fallback 可观测性

### 5.1 Run-scoped stats

Context 创建独立 stats，至少包含：

- `block_sparse_attention_calls`
- `dense_fallback_calls`
- `dense_fallback_reasons: Counter[str]`

Eligible call 只增加 BSA count。Unsupported call 只增加 dense fallback count 和一个 reason。Dense backend 不增加这些计数。

### 5.2 用户可见行为

若 BSA context 退出时存在 fallback，logger 输出一条聚合 warning，包含 BSA count、fallback 总数和按 reason 排序的计数。没有 fallback 时不输出 warning。

## 6. Route representation

`attention_route_generate` 返回 int32 metadata，shape：

```text
(batch, num_heads, ceil(query_len / block_size), ceil(key_len / block_size))
```

Route metadata 表示完整 block-pair route matrix。性能模型只使用 metadata shape 和 plan sparsity，不读取 metadata value。该表示兼容 meta/FakeTensor/compile，不引入 data-dependent shape。

如需更精确的 memory 模型，应增加显式 aggregate 输入或独立 route summary op，不应在性能注册中读取 meta tensor 内容。

## 7. Analytic performance model

### 7.1 维度

```text
N = batch size
Sq = query sequence length
Sk = key sequence length
H = query head count
D = head size
B = block size
s = sparsity
Qb = ceil(Sq / B)
Kb = ceil(Sk / B)
Kkeep = max(1, ceil(Kb * (1 - s)))
Pairs = N * H * Qb * Kkeep
Interactions = Pairs * B * B
```

`ceil` 保证 sparsity 小于 1 时每个 query block 至少保留一个 KV block。`B * B` 明确把 query/KV edge block 都视为 padded tile。本设计不使用 `min(Kkeep * B, Sk)` 的 token-exact edge 计数，因为 estimator 需要表达 block kernel 执行的完整 tile 成本。

### 7.2 Compute

```text
QK MMA ops = Interactions * D * 2
PV MMA ops = Interactions * D * 2
Softmax GP ops = Interactions * 4
```

总 MMA ops 为两个 BMM 之和。Softmax dtype 跟随 query dtype，与现有 BSA semantic-op contract 一致。

示例：`Sq = Sk = 5`、`B = 4`、`s = 0.5` 时：

```text
Qb = 2
Kb = 2
Kkeep = 1
Interactions per batch/head = 2 * 1 * 4 * 4 = 32
```

旧连续模型得到 `5 * 5 * 0.5 = 12.5` interactions；新模型表达两个 padded query tiles 各保留一个完整 KV tile。

### 7.3 Route generation

Route generation 检查完整 block-pair 空间：

```text
RoutePairs = N * H * Qb * Kb
PooledQK GP ops = RoutePairs * D * 2
Selection GP ops = RoutePairs
```

Route generation 不根据 `Kkeep` 减少 scan，因为 route 选择前必须评估候选 block pairs。

### 7.4 Memory

BSA op 的 HBM traffic：

- Q：完整 tensor 一次；
- K：完整 tensor 一次；
- V：完整 tensor 一次；
- mask：存在时完整 tensor 一次；
- route metadata：完整 tensor 一次；
- output：完整 tensor 一次写入。

K/V 规则代表：不同 query-block routes 的 union 保守覆盖全部 KV blocks，但跨 query-block tiles 存在一次 HBM load 后的 cache/kernel reuse。Memory 不按 sparsity 线性缩放。

不计入：score/probability scratch、中间 tile SRAM traffic、cache miss 放大和 route metadata cache efficiency。没有 profiling 证据前不增加经验常量。

### 7.5 限制与校准入口

Estimator 是 analytic approximation。获得 profiling 数据后，优先增加：

1. K/V query-tile reuse efficiency；
2. selected KV union ratio；
3. edge tile execution方式；
4. route metadata read efficiency。

只有在同一硬件/kernel contract 下有稳定证据时才增加 calibrated constant。若 route distribution 成为必要输入，扩展 semantic route summary，而不是从 scalar sparsity 推断。

## 8. Diffusers registration 边界

Package `__init__` 不导入 attention adapter。实际 model/adapter setup 负责导入并注册 TensorCast backend。目标行为：

- 导入 cache、resolver、utils 等无关子模块时，不加载私有 Diffusers attention registry API；
- 构建 Diffusers transformer model 时，backend 已注册；
- 重复导入保持幂等，不重复扩展 enum。

该变更不替换现有 registry API，只缩小副作用范围。

## 9. 与其他能力的交互

### 9.1 Attention quantization

BSA context 与任何非空 attention quant config 互斥。Linear quantization 不受影响。

### 9.2 Sequence parallelism

沿用既有 all-to-all。BSA capability 和 route generation 基于 all-to-all 后的 canonical Q/K/V shape。SP cleanup 只清理 SP group，不提前清理 route plan/stats；route context 退出时统一恢复。

### 9.3 DiT cache

main/cache model 共享 run-scoped plan。cache window 只切换 active model，不切换 backend policy。cache model config 不保存 plan。

### 9.4 `torch.compile`

主模型和 cache 模型可在进入 runtime 前调用 `torch.compile`，首次 forward/tracing 发生在 SDPA context 内。Eager 与 compiled path 使用相同 capability、fallback 和 stats 实现。

### 9.5 CFG

Batch concat 或 CFG parallel 只改变 batch/layout。满足 shape 条件时仍走 BSA；不新增专用 policy。

## 10. 测试设计

### 10.1 Estimator

- 两个不同 block size；
- sequence length 不能被 block size 整除；
- `ceil(Kb * (1 - s))` 的离散边界；
- `Kb == 1` 时任意合法 sparsity 仍保留一个 block；
- dense-equivalent BSA 的 edge padding 成本；
- K/V memory 不随 sparsity 线性缩放，route bytes 完整计入；
- route generation 扫描完整 block-pair 空间。

### 10.2 Capability validation

- Direct context 拒绝 BSA + attention quant config；
- CLI 在模型构建前拒绝同一组合；
- dense + nonzero sparsity 拒绝；
- dense + non-default block size 拒绝；
- 合法 dense、quantized dense、BSA 保持可用。

### 10.3 Fallback coverage

- equal-shape self-attention 计为 BSA；
- cross-attention/shape mismatch；
- causal；
- unsupported mask；
- dropout；
- custom scale；
- GQA；
- 相同 reason 多次只产生一条聚合 warning；
- 聚合 warning 包含 BSA call 总数、fallback 总数和 reason map；
- direct SDPA、Diffusers registry、eager、compiled 路径一致。

### 10.4 Ownership 与 registration

- nested SDPA contexts 恢复外层 plan/stats；
- main/cache model 使用同一 plan；
- model config 不再需要 plan 字段；
- fresh Python process 导入无关 Diffusers 子模块时不加载 attention adapter；
- model build 仍完成 backend 注册。

## 11. 文档与交付

中英文 TensorCast 用户指南需要同步：

- `--attention-backend`、`--attention-block-size`、`--attention-sparsity`；
- BSA 支持范围和 observable fallback；
- attention quantization incompatibility；
- discrete block rounding、edge padding 和 analytic memory assumption；
- 合法配置和错误配置示例。

最终交付运行受影响 regression tests、pre-commit、适用 build checks 和 diff check。未运行项必须记录原因。

## 12. 回滚

默认 backend 保持 dense。回滚 BSA 时不需要迁移模型或配置数据：停止选择 BSA、移除 route/BSA semantic-op dispatch，并保留 dense/quantized dense 路径。Registration 收缩可独立回退，不影响 estimator 数据结构。

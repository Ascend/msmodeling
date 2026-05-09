# zero_cost 分类规则

## 概述

`zero_cost: true` 表示该算子在 NPU 上无 kernel 执行,返回 0.0 us。分为两类:

1. **形状算子**: view, permute, reshape 等 — TC 和 NPU 都不执行
2. **TC 分解伪影**: TC 将融合算子拆为子步骤,但 NPU 已将子步骤融合到其他 kernel

## 类别 1: 形状算子 (Shape-only ops)

NPU 无数据移动,纯元数据操作:

```yaml
aten.view.default:           zero_cost: true
aten.permute.default:        zero_cost: true
aten.t.default:              zero_cost: true
aten.transpose.int:          zero_cost: true
aten._unsafe_view.default:   zero_cost: true
aten.unsqueeze.default:      zero_cost: true
aten.split.Tensor:           zero_cost: true
aten.split_with_sizes.default: zero_cost: true
aten.select.int:             zero_cost: true
aten.slice.Tensor:           zero_cost: true
aten.detach.default:         zero_cost: true
aten.alias.default:          zero_cost: true
aten.expand.default:         zero_cost: true
```

**判断标准**: Profiling 中永远不出现对应 kernel Type → zero_cost。

## 类别 2: TC 分解伪影 (Decomposition artifacts)

TC 将 NPU 融合算子分解为多个子步骤。子步骤的延迟已包含在融合算子中,重复计算会导致高估。

### MoE 路由分解 → MoeGatingTopK 融合

NPU 的 `MoeGatingTopK` 融合了:

- softmax + top-k + weight normalization + mask 操作

TC 将其分解为:

```text
moe_gating_top_k_softmax(logits, bias, k)  ← 有映射,捕获融合延迟
  ↓ 然后 TC 继续分解路由逻辑:
aten.topk()           ← 已被 MoeGatingTopK 包含 → zero_cost
aten.sum.dim_IntList() ← 已被 MoeGatingTopK 包含 → zero_cost
aten.sigmoid()        ← shared expert gate,negligible → zero_cost
aten.where.self()     ← 条件选择,已融合 → zero_cost
aten.bitwise_not()    ← mask 反转,TC 内部 → zero_cost
```

### MoE FFN 分解 → DispatchFFNCombine 融合

NPU 的 `DispatchFFNCombine` 融合了:

- InitRouting + Dispatch + 2×MatMul + SwiGlu + Combine + Unpermute

TC 将其分解为独立算子(当 DFC pass 不生效时):

```text
init_routing_v2    → 有映射 (MoeDistributeDispatchV2, alternate: DFC)
grouped_matmul×N   → 有映射 (GroupedMatmul, alternate: DFC)
swiglu             → 有映射 (SwiGlu)
unpermute_tokens   → 有映射 (MoeDistributeCombineV2, alternate: DFC)
```

注意: 这些子算子保持独立映射(不是 zero_cost),因为:

- DFC pass 生效时,它们不出现 → 无影响
- DFC pass 不生效时(当前 TC 实现),需要独立查询

### NPU 零拷贝 concat

```yaml
aten.cat.default:          zero_cost: true
tensor_cast.cat.default:   zero_cost: true
```

ConcatD 在 CANN 8.5 profiling 中从未出现。NPU concat 通常是零拷贝 view 操作。

## 决策流程

```text
Q: 该算子的 kernel Type 在 profiling 中出现过吗?
│
├─ 从未出现
│   ├─ 是形状/view 操作? → zero_cost (类别 1)
│   ├─ 延迟已包含在另一个融合算子中? → zero_cost (类别 2)
│   └─ 是独立计算但缺数据? → 保持 kernel_type 映射,等数据采集
│
└─ 出现过 → 不是 zero_cost,需要 kernel_type 映射
```

## 验证方法

确认 zero_cost 分类的证据:

1. 在所有 profiling kernel_details.csv 中搜索该 Type → 确认未出现
2. 追踪 vllm-ascend 源码,确认该操作被哪个融合 kernel 吸收
3. 在 notes 字段记录: 被哪个融合算子包含,profiling 数据来源

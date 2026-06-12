# tc_input_count 使用规则

## 概述

`tc_input_count` 截断 TC 和 CSV 的输入数量到 N 进行匹配。这是一个粗粒度工具,使用不当会导致误匹配。

## 安全场景: 截断 NPU 内部参数

当 CSV 额外输入是 CANN 内部参数(axis, scale, offset 等)时,`tc_input_count` 是安全的:

| Op | tc_input_count | TC 输入 | CSV 输入 | 截断原因 |
|---|---|---|---|---|
| `aten.scatter.value` | 2 | (self, index) | (self, index, **axis**) | axis 是 CANN 内部参数 |
| `aten.gather.default` | 2 | (self, index) | (self, index, **axis**) | 同上 |
| `aten.embedding.default` | 2 | (weight, indices) | (weight, indices, **axis**) | 同上 |
| `tensor_cast.quantize.default` | 1 | (tensor, scale, zp) | (tensor) | scale/zp 是 kernel 参数,不在 CSV |
| `tensor_cast.static_quant_linear.default` | 2 | (x, weight, scale, ...) | (x, weight_NZ, scale, scale) | 只匹配 x+weight |
| `tensor_cast.dispatch_ffn_combine.default` | 1 | (x, expert_indices) | (x, w1, w2, idx, s1, s2, probs) | 权重是模型固定的 |

**判断标准**: CSV 多出的输入是 NPU kernel 的**固定参数**(不随 batch/seq 变化)→ 安全截断。

## 不安全场景: 变量广播模式的逐元素算子

当 CSV 行的输入数量因广播模式不同而变化时,`tc_input_count` 不安全:

```text
# Add.csv 中的混合模式:
(16, 7168;)              BF16   5.2 us   ← 标量广播 (1 tensor read)
(16, 7168; 7168)         BF16   6.8 us   ← 向量广播 (2 tensor reads)
(16, 7168; 16, 7168)     BF16   7.1 us   ← 逐元素 (2 tensor reads, same shape)
```

设置 `tc_input_count=1` 后:

- TC 的 `add(x=(16,7168), y=(16,7168))` 会匹配第一行 (5.2 us)
- 正确应匹配第三行 (7.1 us)
- **~30% 延迟低估**

**影响的算子**: `aten.add.Tensor`, `aten.mul.Tensor`, `aten.div.Tensor`, `aten.sub.Tensor`

## 决策流程

```text
Q: CSV 多出的输入是什么?
│
├─ NPU 内部固定参数 (axis, scale, offset) → tc_input_count = N (安全)
├─ 广播操作数 (有时有,有时没有) → 不设 tc_input_count (让它 MISS)
└─ 不确定 → 不设,保守让 MISS,交给 analytic fallback
```

## 已解决: 逐元素算子使用 `query_mode: elementwise`

对于逐元素算子 (Add, Mul, Div), 不需要 `tc_input_count`。这些算子使用 `query_mode: elementwise`,
按**输出形状**匹配,完全绕过输入形状比较。

**规则**: `query_mode: elementwise` 与 `tc_input_count` **互斥**。设置了 `query_mode: elementwise`
的条目不得设置 `tc_input_count`。

## 长期解决方案

对逐元素算子,推荐使用 **输出形状匹配** (`query_mode: elementwise`):

- 输出形状 = 广播后的形状,无论输入是标量/向量/相同形状,输出确定
- 自然支持插值(沿输出维度插值有物理意义)
- 需要在 `profiling_data_source.py` 中新增 `_lookup_elementwise()` 方法

## 参考

- `profiling_data_source.py` `_inputs_match()` 第 942-951 行: tc_input_count 截断逻辑
- `profiling_data_source.py` `_lookup_compute()` 第 830-834 行: TC 输入截断
- 设计文档 S4.2: 查询调度逻辑

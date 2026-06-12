# RFC: CLI `text_generate` / `throughput_optimizer` 支持 Prefix Cache 命中率

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | 已批准 |
| **作者** | yaohan404, Codex |
| **创建日期** | 2026-3-23 |
| **相关链接** | 无 |

---

## 1. 概述

本 RFC 为以下两个 CLI 增加 `--prefix-cache-hit-rate`，用于快速评估 prefix cache 生效后的性能变化：

- `cli.inference.text_generate`
- `cli.inference.throughput_optimizer`

首版只做 token 粒度近似，不建模 block 粒度命中、prefix cache 管理开销，也不估算独立的 prefix cache 常驻内存。

## 2. 目标与非目标

目标：

- 允许用户通过 CLI 指定 prefix cache 命中率
- `text_generate` 能输出 prefix cache 场景下的延迟估算
- `throughput_optimizer` 能在 prefill 建模中纳入 prefix cache 影响

非目标：

- 不修改底层 performance model
- 不做 block 粒度命中建模
- 不做 hash / lookup / eviction / replacement 建模
- 不做 serving 系统级 prefix cache 仿真

## 3. 核心语义

- prefix cache 只作用于 `prefill`
- 命中率按 `token` 粒度近似
- 同一批请求默认 prompt 长度一致、命中率一致

### 3.1 `text_generate`

原始输入：

- `context_length = C`
- `query_length = Q`

若命中 `H = floor(Q * hit_rate)` 个 query token，则映射为：

- `effective_context_length = C + H`
- `effective_query_length = Q - H`

返回到建模层时：

- `RequestInfo.query_len = effective_query_length`
- `RequestInfo.seq_len = effective_context_length + effective_query_length`

因此总 `seq_len` 保持不变，变化的是本次仍需 prefill 的 token 数。

示例：

- 原始：`context_length = 1000`, `query_length = 200`
- `hit_rate = 0.5`
- 结果：`effective_context_length = 1100`, `effective_query_length = 100`, `seq_len = 1200`

### 3.2 `throughput_optimizer`

内部引入：

- `cached_prefix_tokens = floor(input_length * prefix_cache_hit_rate)`
- `effective_input_length = input_length - cached_prefix_tokens`

策略如下：

- 所有 prefill 相关路径使用 `effective_input_length`
- 所有 decode 相关路径保持原逻辑

## 4. 方案设计

### 4.1 CLI 与配置

两个入口均增加：

- `--prefix-cache-hit-rate`

参数约束：

- 类型：`float`
- 默认值：`0.0`
- 取值范围：`[0, 1)`
- 示例统一使用 `0.5`，不使用 `50%`

`UserInputConfig` 增加：

- `prefix_cache_hit_rate: float = 0.0`

### 4.2 `text_generate` 长度重写

在 `UserInputConfig.get_request_info()` 中完成 effective 长度计算，避免在下游重复改写长度。

### 4.3 `throughput_optimizer` 接入点

prefix cache 的 effective 长度语义应在共享的 forward-shape 构造层统一引入，而不是只在单个优化器类中做局部处理。

对 PD 混部模式：

- prefill wave 容量按 `effective_input_length` 计算
- prefill latency 使用 prefix cache 生效后的输入长度
- decode latency 保持原逻辑
- `TTFT` 会因 prefill latency 降低而下降

这里的 `prefill_batch_size = max_prefill_tokens // effective_input_length` 表示 prefill token budget 下单个 prefill wave 的容量，不是用户配置的 `batch_size`。

对 PD 分离模式：

- `disaggregation-prefill` 使用 `effective_input_length`
- `disaggregation-decode` 忽略 prefix cache

### 4.4 `max_prefill_tokens`

接入 prefix cache 后，以下逻辑都要基于 `effective_input_length`：

- `max_prefill_tokens` 的前置比较
- 混部模式下的 `prefill_batch_size = max_prefill_tokens // effective_input_length`

## 5. 指标与边界

### 5.1 指标语义

- `prefill latency`：受 prefix cache 影响
- `decode latency`：不因 prefix cache 直接变化
- `TTFT`：随 prefill latency 下降而下降
- `TPOT`：如果定义中包含 `TTFT`，可能随之变化；这不代表 decode kernel 被优化

### 5.2 边界条件

`text_generate` 下若同时指定：

- `--decode`
- `--prefix-cache-hit-rate > 0`

工具继续运行，但输出 warning，并忽略 prefix cache 命中率。

当前实现要求：

- `effective_query_len >= 1`
- `effective_input_length >= 1`

否则视为当前版本不支持的场景。

### 5.3 内存语义

当前方案只近似评估本次请求的计算开销下降：

- `text_generate` 保持总 `seq_len` 不变
- 不额外建模独立的 prefix cache 常驻内存

因此输出中的内存指标不应解释为真实 serving 系统的总缓存驻留成本。

### 5.4 首版不覆盖

- block 粒度命中与部分 block 复用
- 非均匀命中分布
- cache 管理开销
- decode 阶段额外优化
- serving 系统级高精度仿真

## 6. 测试与验收

参数测试：

- 默认值为 `0.0`
- `0.5` 合法
- `-0.1`、`1.0` 非法
- 导致 effective 长度为 `0` 的输入非法
- `text_generate --decode` 与 `--prefix-cache-hit-rate > 0` 组合应输出 warning

`text_generate` 测试：

- `context_length = 1000`, `query_length = 200`, `hit_rate = 0.5`
- 验证 effective 长度分别为 `1100`、`100`
- 验证 `seq_len = 1200`
- 验证 `hit_rate = 0` 与原实现一致

`throughput_optimizer` 测试：

- `input_length = 200`, `hit_rate = 0.5`
- 验证 `effective_input_length = 100`
- 验证 PD 混部 prefill 使用 effective 长度
- 验证 PD 分离 prefill 使用 effective 长度
- 验证 PD 分离 decode 保持原逻辑
- 验证 `max_prefill_tokens` 校验与 prefill wave 容量计算均基于 effective 长度
- 验证非法输入返回非零退出码

## 7. 后续演进

如需更高精度，可继续扩展：

- block 粒度 prefix cache 建模
- 命中分布建模
- prefix cache 管理与内存建模
- `serving_cast` 对 prefix cache 场景的进一步联动

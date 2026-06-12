# RFC: Transformer 重复层代表层优化

Status: Approved
Author(s): yaohan404
Created: 2026-05-21
Updated: 2026-05-21

## 1. 概述

本文描述 TensorCast 针对 Transformer 重复层的代表层优化方案。该方案面向大语言模型中高度重复的 decoder layer 结构：模型仍保留完整层数和完整建模结果，但模型构建阶段只对结构等价的一组层中的代表层执行 `patch_moe`、`quantize_model`、`shard_model` 等 host-side transform，从而降低建模启动成本。

代码中已有一个历史开关名为 `enable_repetition`。为了避免依赖内部讨论术语，本文将该能力称为“重复层复用建模”：当多个 Transformer layer 具有相同子模块结构时，TensorCast 只真实执行第一层，并用 copy region 表达后续等价层的执行效果。本文的优化是在这个机制之上，将 host-side transform 也收敛到代表层规模。

## 2. 重复层复用建模机制

Transformer decoder-only 模型通常由大量结构相同的 layer 组成。例如 dense 模型可能有 64 个结构相同的 decoder layer，MoE 模型可能包含少数 dense layer 加大量结构相同的 MoE layer。

TensorCast 的重复层复用建模机制基于两个 wrapper：

- `RegionMarkerWrapper`：包裹一组结构等价层中的第一层，也就是代表层。该层保留真实计算，并在 runtime 中标记一个可复用区域。
- `CopyLayerWrapper`：包裹后续结构等价层。forward 时通过 copy region 表达“复用代表层执行区域”的效果，而不是重复执行完整 layer 计算。

该机制的目标不是改变模型结构，而是减少性能建模过程中重复 layer 带来的执行和编译成本。外部仍应看到完整 `ModuleList` 层数，analytic model 也仍应统计完整层数对应的算子和权重。

优化前，该机制已经能降低 runtime/compile 中重复层的真实执行成本，但模型构建阶段仍有一个问题：`CopyLayerWrapper` 仍持有原始 `_inner` layer，默认 `named_modules()`、`named_parameters()` 会递归进入内部模块。因此后续转换逻辑仍会处理所有重复层内部的 MoE experts 和 Linear modules。

对于 DeepSeek-V3.2 这类 MoE 模型，这会放大成明显的启动耗时：

- `quantize_model` 会量化所有重复 MoE layer 内部的 experts/linear
- `patch_moe` 会遍历所有重复 MoE layer 的 expert 结构
- `shard_model` 会继续处理重复层中的参数和模块

本文方案解决的是这个 host-side transform 放大问题。

## 3. 背景与动机

典型 DeepSeek-V3.2 `text_generate` 命令：

```powershell
python -m cli.inference.text_generate deepseek-ai/DeepSeek-V3.2 `
  --num-queries 16 `
  --query-length 1 `
  --context-length 4096 `
  --device ATLAS_800_A3_752T_128G_DIE `
  --quantize-linear-action FP8 `
  --num-devices 16 `
  --tp-size 1 `
  --ep-size 16 `
  --compile
```

优化前的主要问题：

- 重复层复用建模已经降低 compile/runtime 重复层执行，但没有降低 host-side transform 的重复层遍历。
- `quantize_model` 会处理所有重复层内部 linear，DeepSeek-V3.2 中会放大到数万次 `QuantLinearBase` 构建和量化。
- `patch_moe` 会遍历重复 MoE layer 及 expert 结构，产生秒级额外耗时。
- `--num-hidden-layers-override 2` 可以显著加速，但会改变模型结构和完整建模结果，不能作为正式优化方案。

## 4. 目标与非目标

目标：

- 保留完整 `ModuleList` 层数，不改变模型外部结构。
- 保留原有 copy region runtime 语义，重复层仍由 copy region 表达。
- 后续 host-side transform 只处理代表层内部模块。
- 保持 full model weight accounting，不因隐藏 repeated layer 参数而低估模型权重。
- 验证 op table、analytic execution time、model weight、KV cache 等建模结果一致。

非目标：

- 不重写 torch.compile 或 copy region runtime 逻辑。
- 不优化 Python import/startup。
- 不重构 throughput_optimizer 搜索框架。
- 不引入 shape-only quantization 或 MoE proxy 新抽象。

## 5. 方案设计

### 5.1 代表层转换

`maybe_reuse_layers()` 根据 layer 子模块结构生成 key：

- 第一个结构 key 对应的 layer 包装为 `RegionMarkerWrapper`
- 后续相同 key 的 layer 包装为 `CopyLayerWrapper`

本方案保持完整层列表长度，但让 `CopyLayerWrapper` 在模块遍历中表现为叶子节点：

- `CopyLayerWrapper.named_modules()` 只返回自身
- `CopyLayerWrapper.modules()` 只返回自身
- `CopyLayerWrapper.named_parameters()` / `parameters()` 返回空
- `CopyLayerWrapper.named_buffers()` / `buffers()` 返回空
- `_inner` 仍保留，用于 forward 语义、元信息兼容和调试

这样后续 host-side transform 会跳过 repeated copy layer 的内部结构，只处理代表层。

### 5.2 repeat_count 与权重统计

因为 copy layers 隐藏了内部参数，直接用 `named_parameters()` 统计会低估模型权重。方案在 `RegionMarkerWrapper` 上引入 `repeat_count`：

- 代表层的 `repeat_count` 初始为 1
- 每遇到一个结构相同的 repeated layer，递增对应 representative wrapper 的 `repeat_count`
- `TransformerModel.get_weight_size_nested()` 额外调用 `get_represented_extra_weight_size()`
- 对每个 `RegionMarkerWrapper` 追加 `(repeat_count - 1) * representative_inner_weight_size`

这样 host transform 只处理代表层，但 full model weight size 仍按完整层数统计。

### 5.3 影响范围

代码变更范围：

- `tensor_cast/layers/internal.py`
  - `RegionMarkerWrapper` 增加 `repeat_count`
  - `CopyLayerWrapper` 隐藏内部 module/parameter/buffer traversal
- `tensor_cast/transformers/transformations.py`
  - `maybe_reuse_layers()` 保留完整层数，同时维护结构 key 到 representative wrapper 的映射
- `tensor_cast/transformers/model.py`
  - 权重统计补充 repeated layer represented weight
- `tests/test_tensor_cast/test_repetition.py`
  - 增加 representative layer 数量和 `repeat_count` 验证

## 6. 测试与验证结果

### 6.1 单元测试

已验证：

```powershell
python -m pytest -q tests\test_tensor_cast\test_repetition.py
```

结果：

- `4 passed`

补充验证：

```powershell
python -m pytest -q tests\test_tensor_cast\test_text_generate.py -k "num_hidden_layers_override or disable_repetition"
```

结果：

- `2 passed, 95 deselected`

### 6.2 DeepSeek-V3.2 text_generate no-compile

命令：

```powershell
python -m cli.inference.text_generate deepseek-ai/DeepSeek-V3.2 `
  --num-queries 16 `
  --query-length 1 `
  --context-length 4096 `
  --device ATLAS_800_A3_752T_128G_DIE `
  --quantize-linear-action FP8 `
  --num-devices 16 `
  --tp-size 1 `
  --ep-size 16
```

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| profiling total | 23.638s | 5.328s | 4.44x |
| `model_runner_init` | 17.989s | 0.749s | 24.0x |
| `patch_moe` | 3.309s | 0.046s | 72.0x |
| `quantize_model` | 13.741s | 0.239s | 57.5x |
| `shard_model` | 0.481s | 0.010s | 48.1x |
| `QuantLinearBase.__init__` calls | 45,215 | 790 | 57.2x fewer |
| `QuantLinearBase.quantize_weight` calls | 45,215 | 790 | 57.2x fewer |
| `TransformerModel._replace_module` calls | 3,154 | 62 | 50.9x fewer |
| analytic execution time | 0.061924s | 0.061924s | unchanged |
| model weight size | 56.663GB | 56.663GB | unchanged |
| KV cache | 0.276GB | 0.276GB | unchanged |

### 6.3 DeepSeek-V3.2 text_generate compile

命令：

```powershell
python -m cli.inference.text_generate deepseek-ai/DeepSeek-V3.2 `
  --num-queries 16 `
  --query-length 1 `
  --context-length 4096 `
  --device ATLAS_800_A3_752T_128G_DIE `
  --quantize-linear-action FP8 `
  --num-devices 16 `
  --tp-size 1 `
  --ep-size 16 `
  --compile
```

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| CLI wall time | 35.590s | 15.854s | 2.25x |
| `Model compilation and execution time` | 8.365s | 7.146s | 1.17x |
| analytic execution time | 0.045181s | 0.045181s | unchanged |
| model weight size | 56.663GB | 56.663GB | unchanged |
| KV cache | 0.276GB | 0.276GB | unchanged |
| `tensor_cast.cat.default` calls | 58 | 58 | unchanged |
| `aten.cat.default` calls | 58 | 58 | unchanged |

结论：

- compile 场景的主要收益来自 compile 前 host-side transform/init，而不是 compile graph 本身。
- 优化前已有 copy region runtime 机制，compile/runtime 已经只执行 representative region。
- 本方案保持 op table 中 repeated copy ops 的统计一致性。

### 6.4 Dense Qwen case

命令：

```powershell
python -m cli.inference.text_generate Qwen/Qwen3-32B `
  --device TEST_DEVICE `
  --tp-size 2 `
  --num-devices 2 `
  --num-queries 2 `
  --query-length 1000 `
  --compile `
  --quantize-linear-action FP8
```

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| phase total | 12.169s | 12.121s | 基本持平 |
| `model_runner_init` | 0.606s | 0.374s | -0.232s |
| `quantize_model` | 0.175s | 0.004s | 43.8x |
| `shard_model` | 0.064s | 0.004s | 16.0x |
| `QuantLinearBase.__init__` calls | 448 | 7 | 64.0x fewer |
| analytic execution time | 0.259494s | 0.259494s | unchanged |
| model weight size | 16.725504GB | 16.725504GB | unchanged |

结论：

- 代表层优化对 dense 模型也生效，但 dense 模型 host-side transform 原始绝对耗时较小。
- 端到端耗时主要由 import/startup 和 compile/run 构成，因此用户体感不明显。

### 6.5 throughput_optimizer

DeepSeek-V3.2 aggregation case：

```powershell
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.2 `
  --input-length 4096 `
  --output-length 128 `
  --device ATLAS_800_A3_752T_128G_DIE `
  --quantize-linear-action FP8 `
  --num-devices 16 `
  --tp-sizes 1 `
  --ep-sizes 16 `
  --moe-dp-sizes 1 `
  --batch-range 1 16 `
  --tpot-limits 200 `
  --max-prefill-tokens 8192 `
  --reserved-memory-gb 0 `
  --jobs 1
```

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| wall time | 43.727s | 21.644s |
| best throughput | 80.64 token/s | 80.64 token/s |
| TTFT | 8735.89ms | 8735.89ms |
| TPOT | 130.17ms | 130.17ms |

DeepSeek-V3.2 disagg single-config case：

```powershell
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.2 `
  --device ATLAS_800_A3_752T_128G_DIE `
  --num-devices 16 `
  --input-length 3500 `
  --output-length 1500 `
  --disagg `
  --ttft-limits 2000 `
  --tpot-limits 100 `
  --compile `
  --tp-sizes 1 `
  --jobs 1
```

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| `All experiments completed` | 117.11s | 63.06s |
| outer wall time | 122.175s | 68.128s |

完整 disagg 默认搜索中，收益会被 5 个 TP 配置、Prefill/Decode 两阶段、`jobs=8` 多进程并发和 batch search/compile 成本稀释。该部分后续优化应在 throughput_optimizer 搜索框架中继续推进。

## 7. 风险与边界

### 7.1 权重统计风险

`CopyLayerWrapper` 隐藏内部参数后，必须通过 `RegionMarkerWrapper.repeat_count` 补偿 represented weight。测试中已覆盖 weight size 一致性，但后续如果引入新的 wrapper 或非标准 layer list，需要确认是否仍能正确统计。

### 7.2 模块遍历语义变化

本方案改变了 `CopyLayerWrapper` 的 traversal 行为。预期收益正来自该变化，但如果未来有逻辑需要检查 repeated copy layer 的 `_inner` 结构，需要显式访问 `_inner`，不能依赖 `named_modules()`。

### 7.3 非完全重复结构

当前代表层检测仍基于子模块结构 key，不处理参数值差异。该行为与原有重复层复用机制一致，本方案不扩大适用范围。

### 7.4 compile 收益边界

本方案不承诺 compile graph 按层数线性下降。原有 copy region runtime 已经优化 compile/runtime；本文方案主要降低 compile 前模型构建和转换成本。

## 8. 后续工作

本 RFC 不包含但建议后续单独推进：

1. throughput_optimizer 优化：
   - 合并 Prefill/Decode 同配置任务
   - 复用 ProcessPoolExecutor
   - 减少 batch search forward/compile 次数
2. shape-only quantization：
   - 本方案后 `quantize_model` 已不再是主瓶颈，但仍可作为小模型/多 case benchmark 噪声优化。
3. import/startup 优化：
   - lazy builtin registration 已实验，DeepSeek-V3.2 compile 主 case 重复测试收益约 0.276s，暂不作为当前主要优化方向。

## 9. 验收标准

代码合入前至少满足：

- `tests/test_tensor_cast/test_repetition.py` 通过
- `tests/test_tensor_cast/test_text_generate.py -k "num_hidden_layers_override or disable_repetition"` 通过
- DeepSeek-V3.2 text_generate compile/no-compile 的 analytic execution time、weight、KV cache 一致
- DeepSeek-V3.2 compile op table 中 `tensor_cast.cat.default` 和 `aten.cat.default` call count 保持 58
- throughput_optimizer best result 不变


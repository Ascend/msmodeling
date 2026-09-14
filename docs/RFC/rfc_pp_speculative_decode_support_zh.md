# RFC: Pipeline Parallel 与投机解码（MTP/DFlash/DSpark）共存支持

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | 已实现（PR #773 已合入） |
| **作者** | hanxinlong |
| **创建日期** | 2026-09-03 |
| **相关链接** | <https://gitcode.com/Ascend/msmodeling/pull/773> |
| **前置 RFC** | [rfc_pipeline_parallel_support_zh.md](rfc_pipeline_parallel_support_zh.md) |

---

## 1. 问题陈述

Pipeline Parallel（PP）和投机解码（MTP/DFlash/DSpark）此前在吞吐优化器中互斥。PP>1 时，`build_pp_search_candidates`（`serving_cast/service/utils.py`）强制过滤所有非零 MTP 候选，注释称"model_builder rejects it"。但该注释**已过时**——模型构建层早已支持将 MTP proposal 层分配到 PP 最后一个 stage（`pipeline_parallel.py:348`）。

实际问题有两个层面：

1. **ServingCast 层面**：PP>1 的 decode 路径从未应用投机解码折算（`_fold_decode_latency_ms`），导致即使去掉候选门控，MTP/DFlash/DSpark 也无法在 PP 下体现 TPOT 收益。此外，PP 路径的 `_build_user_input` 不翻译投机方法，DFlash/DSpark 会因 `MtpConfig` 与 `DflashConfig` 互斥而崩溃。

2. **TensorCast 层面**：`_resolve_decoder_layers` 在 PP+MTP 下因 MTP 层索引为 None 抛 AttributeError，导致所有层回退到通用 KV 公式，丢失 V4 compressed / Bailing KDA 的 per-layer 特殊 KV shape。`_spec_decode_skips_runner_sampler` 不识别 MtpWrapper，PP 路径也完全没有 sampler guard。非 last stage 未清除 DFlash/DSpark config，导致 wrapper 在每个 stage 上构建。

### 1.1 动机

在**混布场景**（如 KimiK3）中，MTP/DFlash/DSpark 是保证 TPOT 收益的必要手段，PP 是释放显存压力的必要手段。二者必须在同一节点上同时开启。此前用户只能二选一：要么开 PP 牺牲 TPOT，要么开 MTP/DFlash/DSpark 牺牲显存。

### 1.2 目标

- 去掉 PP>1 与投机解码的候选门控，使 PP+MTP/DFlash/DSpark 候选可以正常枚举。
- 在 PP>1 decode 路径应用 `_fold_decode_latency_ms` 折算，使 TPOT/吞吐正确反映投机解码收益。
- 修复 TensorCast 侧的 decoder layer 解析、sampler guard、非 last stage config 清除、num_hidden_layers 计数、`extra_draft_layers` KV cache 切片。
- 保持 PP=1 或无投机解码时现有行为完全不变（折算为 no-op，config 清除不触发）。

### 1.3 非目标

- 不实现真实跨进程 PP 调度（PP 调度器是逻辑 max-plus 时间线模拟）。
- 不修改投机解码的模型构建逻辑（MtpWrapper/DflashWrapper/DsparkWrapper 本身不变）。
- 不修改 pipeline scheduler 的调度算法（max-plus cycle mean 不变）。
- 不为 DFlash/DSpark draft model 新增 `extra_tail_layers` 机制（draft model 在 wrapper 内部独立运行，不像 MTP proposal 层那样附加到 HF decoder）。

---

## 2. 方案设计

### 2.1 总体架构

三种投机解码在 PP 下的行为模式一致：proposal/draft 计算只在 PP 的**最后一个 stage** 上运行（因为它需要完整的 target model 输出）。因此适配策略是统一的——**确保 wrapper 只在 last stage 构建，折算在 decode 路径应用，sampler guard 覆盖所有三种 wrapper**。

下面用端到端流程图展示 8 项改动各自落在哪个环节。注意 ServingCast 和 TensorCast 是**交替执行**的——ServingCast 生成候选后，逐个交给 TensorCast 跑仿真，拿到结果后再做折算和调度：

```text
用户输入
  --pp-sizes 2 --speculative-method mtp --num-speculative-tokens 3
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  ServingCast（第一阶段：搜索 + 候选生成）                         │
│                                                                 │
│  ① build_pp_search_candidates                                   │
│     去掉 PP>1 候选门控 → MTP/DFlash/DSpark 候选正常枚举          │
│                                                                 │
│  ③ _build_user_input（PP 路径）                                  │
│     翻译投机方法：                                               │
│     MTP → num_mtp_tokens = N（构建 MtpWrapper）                 │
│     DFlash/DSpark → num_speculative_tokens = N,                  │
│                     num_mtp_tokens = 0（避免互斥崩溃）           │
│                                                                 │
│  产出：一批 UserInputConfig（每个含 TP/PP/EP/MTP 等参数）        │
└─────────────────────────────────────────────────────────────────┘
    │
    │  逐个候选交给 ModelRunner 评估
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  TensorCast（第二阶段：模型构建 + 仿真）                         │
│                                                                 │
│  ⑥ build_stage_model_config                                     │
│     非 last stage 清除 mtp_config / dflash_config / dspark_config│
│     → wrapper 只在 last stage 构建                                │
│                                                                 │
│  ⑦ PipelineModel.num_hidden_layers                              │
│     = decoder + MTP + draft（之前漏了 draft）                    │
│     → KV cache 分配覆盖全部层                                    │
│                                                                 │
│  ⑧ extra_draft_layers（PipelineStageSpec 新字段）                │
│     _slice_layer_cache 扩展切片范围                              │
│     → draft 层 KV cache 正确切到 last stage                      │
│                                                                 │
│  ④ _resolve_decoder_layers                                      │
│     只检查 decoder 层缺失，MTP/draft 层保持 None 不报错           │
│     → decoder 层保留特殊 KV shape，不再全部回退通用公式           │
│                                                                 │
│  ⑤ _spec_decode_skips_runner_sampler                            │
│     添加 MtpWrapper 检查 + PP 委托到 stages[-1].model            │
│     → 不 double-sample（wrapper 内部已采样）                      │
│                                                                 │
│  产出：PipelineProfile（含 stage 级 compute/comm/memory 数据）   │
│        ← TensorCast 的输出，传回给 ServingCast                    │
└─────────────────────────────────────────────────────────────────┘
    │
    │  PipelineProfile 传回 ServingCast
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  ServingCast（第三阶段：调度 + 折算 + 输出）                      │
│                                                                 │
│  调度器 estimate_forward_pipeline / estimate_repeated_pipeline   │
│    用 PipelineProfile 模拟 microbatch 填充 pipeline               │
│    → 算出 makespan / bubble_ratio / worst_tpot / interval        │
│                                                                 │
│  ② _fold_decode_latency_ms 折算                                  │
│     TPOT = worst_tpot × 1000 / (accept + 1)                     │
│     interval = measured_interval × 1000 / (accept + 1)          │
│                                                                 │
│  输出：TPOT / 吞吐量 / Top N 表格                                 │
└─────────────────────────────────────────────────────────────────┘
```

**流程说明**：ServingCast 先生成候选（①③），然后**逐个**交给 TensorCast 跑仿真（⑥⑦⑧④⑤），TensorCast 产出 PipelineProfile 传回 ServingCast，ServingCast 用它做调度和折算（②），最终输出 TPOT/吞吐量。三个阶段是**交替循环**的，不是一次性跑完。

**ServingCast 负责"搜索 + 评估"**：① 门控移除让候选通过，③ 翻译方法让 config 不冲突，② 折算让 TPOT 正确反映投机收益。

**TensorCast 负责"模型构建 + 仿真"**：⑥ wrapper 只在 last stage 构建，⑦⑧ KV cache 覆盖全部层且正确切片，④ decoder 层保留特殊 shape，⑤ sampler 不重复采样。

### 2.2 ServingCast 改动

#### 2.2.1 去掉候选门控 ①

`serving_cast/service/utils.py` 的 `build_pp_search_candidates`：

```python
# 旧（门控）：
if pp > 1:
    effective_mtp_list = [m for m in mtp_list if m == 0]
else:
    effective_mtp_list = mtp_list

# 新（无门控）：
effective_mtp_list = mtp_list
```

三种投机解码共享 MTP 搜索槽位（`num_mtp_tokens = N`），门控移除后均可通过。同时移除 `pp_blocked_by_mtp` 追踪和 MTP 专属 `ValueError`。

#### 2.2.2 PP decode 路径应用折算 ②

`serving_cast/service/agg_throughput_optimizer.py`（`_get_pp_metrics`）：

```python
# 旧（无折算）：
decode_latency = repeated.worst_tpot_s * 1000.0
decode_interval_ms = repeated.measured_interval_s * 1000.0

# 新（有折算）：
decode_latency = self._fold_decode_latency_ms(repeated.worst_tpot_s * 1000.0, optimizer_data)
decode_interval_ms = self._fold_decode_latency_ms(repeated.measured_interval_s * 1000.0, optimizer_data)
```

`serving_cast/service/disagg_throughput_optimizer.py`（`_get_pp_inference_info`）同理，`serving_cost_ms` 保持不折算（它是 per-step 服务开销，非计算延迟）。

折算函数 `_fold_decode_latency_ms` 已处理三种分支：

- DSpark：`latency / (clamp(accept, 0, block-1) + 1)`
- DFlash：同上
- 新 MTP（`speculative_method == "mtp"`）：`latency / (clamp(accept, 0, n) + 1)`
- 旧 MTP（无 `speculative_method`）：`latency / (sum(mtp_acceptance_rate[:N]) + 1)`

无投机方法时返回 `latency / 1.0 = latency`（no-op），PP 基线结果不变。

#### 2.2.3 PP `_build_user_input` 翻译投机方法 ③

`serving_cast/parallel_runner.py` 的 PP 路径 `_build_user_input`：

```python
# 旧（不翻译）：
tmp_user_input.num_mtp_tokens = candidate.num_mtp_tokens

# 新（翻译）：
method = getattr(base_user_input, "speculative_method", None)
if method in ("dflash", "dspark", "mtp"):
    # per-candidate acceptance 裁剪
    block = int(candidate.num_mtp_tokens) + 1 if int(candidate.num_mtp_tokens) >= 1 else 0
    if block >= 2:
        tmp_user_input.acceptance_length = clamp_acceptance_length(
            float(getattr(self.args, "acceptance_length", 5.0)), block, method)
    if method in ("dflash", "dspark"):
        tmp_user_input.num_speculative_tokens = candidate.num_mtp_tokens  # per-candidate N
        tmp_user_input.num_mtp_tokens = 0  # 避免 MtpConfig/DflashConfig 互斥
    else:  # mtp
        tmp_user_input.num_mtp_tokens = candidate.num_mtp_tokens
else:
    tmp_user_input.num_mtp_tokens = candidate.num_mtp_tokens
```

DFlash/DSpark 的 N 值必须显式设为 `num_speculative_tokens = candidate.num_mtp_tokens`（per-candidate）。单 N 时 `base_user_input` 已通过 `copy.copy` 继承正确值，但多 N 搜索（`--num-speculative-tokens 3 7`）时 CLI 只把第一个 N 写入 `args.num_speculative_tokens`，必须 per-candidate 覆盖。`num_mtp_tokens` 必须设为 0 以避免 `ConfigResolver` 同时构建 `MtpConfig`。

### 2.3 TensorCast 改动

#### 2.3.1 清除非 last stage 的 DFlash/DSpark config ⑥

`tensor_cast/pipeline_parallel.py` 的 `build_stage_model_config`：

```python
if not stage_spec.is_last:
    stage_config.mtp_config = None
    stage_config.dflash_config = None  # 新增
    stage_config.dspark_config = None  # 新增
```

与 MTP 的处理方式完全一致。DflashWrapper/DSparkWrapper 只在 last stage 构建（通过 `TransformerModel.__init__` → `maybe_enable_dflash`/`maybe_enable_dspark`），避免在非 last stage 上复制 proposal 计算。

#### 2.3.2 `PipelineModel.num_hidden_layers` 包含 draft 层 ⑦

```python
@property
def num_hidden_layers(self) -> int:
    mtp_layers = int(getattr(getattr(self.model_config, "mtp_config", None), "num_mtp_layers", 0) or 0)
    draft_layers = int(self.model_config.draft_num_layers()) if self.model_config.has_draft_spec() else 0
    return self.plan.num_hidden_layers + mtp_layers + draft_layers
```

确保 `_get_kv_cache_info` 的 `range(model.num_hidden_layers)` 覆盖 MTP 和 draft 层。`_resolve_decoder_layers` 返回的列表中 MTP/draft 索引为 None，下游用通用 KV 公式处理（MTP blocks 用标准 MLA，draft blocks 用 Qwen3 GQA，通用公式均正确）。

#### 2.3.3 `extra_draft_layers` 扩展 KV cache 切片范围 ⑧

`build_pipeline_plan` 原先只为 MTP 设置 `extra_tail_layers`，DFlash/DSpark draft 层不在切片范围内。`_slice_layer_cache` 的 `layer_end = stage_spec.layer_end + extra_tail_layers` 不含 draft 层索引，导致 last stage 的 local KV cache 缺少 draft 层条目。DFlash 的 `ensure_draft_kv_caches` 安全网虽能避免崩溃，但 draft KV 内存未计入 `PipelineStageCacheStats`，使 PP 内存估算偏乐观。

新增 `extra_draft_layers` 字段到 `PipelineStageSpec`，与 `extra_tail_layers` 平行：

```python
# PipelineStageSpec 新增字段
extra_draft_layers: int = 0  # draft layers (DFlash/DSpark) on last stage

# build_pipeline_plan
draft_layers = int(model_config.draft_num_layers()) if model_config.has_draft_spec() else 0
extra_draft_layers=draft_layers if is_last else 0

# _slice_layer_cache
layer_end = stage_spec.layer_end + stage_spec.extra_tail_layers + stage_spec.extra_draft_layers

# build_pipeline_stage_kwargs
if stage_spec.is_last and (extra_tail_layers + extra_draft_layers) > 0:
    stage_kwargs["input_ids"] = input_kwargs["input_ids"]
```

`extra_draft_layers` 不影响 `num_hidden_layers_override`（用 `num_layers` 而非 `num_local_layers`）——draft 层在 `DflashWrapper` 内部的独立模型中，不在 HF decoder ModuleList 里。字段只扩展 KV cache 切片范围和 `input_ids` 转发。

#### 2.3.4 `_resolve_decoder_layers` 不对 MTP/draft 层抛异常 ④

`tensor_cast/core/input_generator.py`：

```python
# 只对 decoder 层索引检查缺失
plan = getattr(inner, "plan", None)
decoder_count = int(plan.num_hidden_layers) if plan is not None else len(layers)
missing_decoder = [idx for idx in range(decoder_count) if layers[idx] is None]
if missing_decoder:
    raise AttributeError(...)
```

MTP/draft 层索引合法保持 None，下游 `_resolve_decoder_attention_layer(None)` 返回 None → 通用 KV 公式。

#### 2.3.5 `_spec_decode_skips_runner_sampler` PP 委托 ⑤

`tensor_cast/core/model_runner.py`：

```python
def _spec_decode_skips_runner_sampler(model, input_kwargs):
    ...
    # PipelineModel 无 _inner，wrapper 在 stages[-1].model 上
    if hasattr(model, "stages"):
        stages = list(getattr(model, "stages", []))
        if stages:
            last_stage_model = getattr(stages[-1], "model", None)
            if last_stage_model is not None:
                model = last_stage_model

    # 然后正常走 _inner 链
    node = model
    while node is not None:
        if isinstance(node, DflashWrapper):  # 覆盖 DsparkWrapper（子类）
            return DflashWrapper._is_decode_step(input_kwargs)
        if isinstance(node, MtpWrapper):
            return True  # MTP 始终内部采样
        node = getattr(node, "_inner", None)
    return False
```

PP 路径的 `run_inference`（`model_runner.py:235`）也应用 guard：

```python
skip_sampler = _spec_decode_skips_runner_sampler(self.model, input_kwargs)
pipeline_result = pipeline_runner.run(
    input_kwargs,
    with_sampler=with_sampler and not skip_sampler, ...
)
```

### 2.4 不需要改动的部分

| 能力 | 说明 |
|---|---|
| MtpWrapper/DflashWrapper/DsparkWrapper 构建 | 由 `TransformerModel.__init__` 的 transformation 链处理，PP 路径天然通过 |
| Hidden state 跨 stage 传递 | `PipelineStageModel` 硬编码 `output_intermediate_hidden_states=True` |
| MTP-tail KV cache remapping | `_slice_layer_cache` + `extra_tail_layers` 已正确处理 |
| Draft KV cache remapping | `_slice_layer_cache` + `extra_draft_layers` 已正确处理（见 §2.3.3） |
| MTP/draft compute 归属 last stage | wrapper 在 last stage 的 `Runtime` 上下文内执行 |
| pipeline scheduler | 纯 max-plus 数学，对 stage 内容无感知 |
| `_resolve_forward_shape` | decode shape 已处理 DFlash/DSpark 的 `block_size` |
| `_submit_task` | 已传播 `dspark_*`/`dflash_*`/`speculative_method` |
| OptimizerData | 已携带全部 draft 字段 |

---

## 3. 三种投机解码在 PP 下的行为对比

| 特性 | MTP | DFlash | DSpark |
|---|---|---|---|
| Wrapper | `MtpWrapper` | `DflashWrapper` | `DsparkWrapper(DflashWrapper)` |
| Config 字段 | `mtp_config` | `dflash_config` | `dspark_config` |
| 搜索槽位（候选生成层） | `num_mtp_tokens` | `num_mtp_tokens`（复用） | `num_mtp_tokens`（复用） |
| `num_mtp_tokens`（翻译后） | N（构建 MtpWrapper） | 0（避免互斥） | 0（避免互斥） |
| 非_last stage 清除 | `mtp_config = None` | `dflash_config = None` | `dspark_config = None` |
| `num_hidden_layers` 计入 | `num_mtp_layers` | `draft_num_layers()` | `draft_num_layers()` |
| KV cache 切片扩展 | `extra_tail_layers` | `extra_draft_layers` | `extra_draft_layers` |
| `num_speculative_tokens`（翻译后 per-candidate） | N/A（用 `num_mtp_tokens`） | `= candidate.num_mtp_tokens` | `= candidate.num_mtp_tokens` |
| sampler guard | `isinstance(MtpWrapper)` → True | `isinstance(DflashWrapper)` → `_is_decode_step` | 同 DFlash（子类） |
| 折算公式 | `latency / (clamp(accept, 0, n) + 1)` | `latency / (clamp(accept, 0, block-1) + 1)` | 同 DFlash |
| decode shape | `query_len = n + 1` | `query_len = block` | `query_len = block` |

> 注：搜索槽位与配置字段分属两层。**候选生成层**统一用 `num_mtp_tokens` 承载 N 值（`ParallelSearchCandidate.num_mtp_tokens`；CLI 侧 DFlash/DSpark 的 `--num-speculative-tokens` 在参数归一化阶段即写入 `num_mtp_token_sizes` 搜索槽）。**配置翻译层**（`_build_user_input`）在搜索完成后按 `speculative_method` 将槽位值写入 `UserInputConfig`：DFlash/DSpark 写 `num_speculative_tokens=N` 并清零 `num_mtp_tokens`（避免 ConfigResolver 同时构建互斥的 `MtpConfig` 与 `DflashConfig`），MTP 保留 `num_mtp_tokens=N`。表中标注"翻译后"的各行均为 `UserInputConfig` 状态；`num_speculative_tokens` 字段随 DFlash/DSpark 特性早已存在，本 RFC 新增的是 PP 多 N 搜索的 per-candidate 翻译。

---

## 4. 测试策略

### 4.1 ServingCast 测试

| 测试 | 文件 | 验证 |
|---|---|---|
| `test_pp_gt1_with_mtp_allowed` | `test_throughput_optimizer.py` | PP>1 接受 MTP 候选 |
| `test_pp_decode_applies_mtp_fold` (agg) | `test_agg_optimizer.py` | PP decode TPOT 按 (accept+1) 折算 |
| `test_pp_decode_applies_mtp_fold` (disagg) | `test_disagg_optimizer.py` | 同上 |
| `test_pp_build_user_input_sets_num_mtp_zero_for_dflash` | `test_parallel_runnner.py` | PP 路径 DFlash 下 `num_mtp_tokens=0`、`num_speculative_tokens=N` |
| `test_pp_build_user_input_multi_n_dflash_sets_per_candidate_spec_tokens` | `test_parallel_runnner.py` | 多 N DFlash per-candidate `num_speculative_tokens` |

### 4.2 TensorCast 测试

| 测试 | 文件 | 验证 |
|---|---|---|
| `test_resolve_decoder_layers_with_pp_and_mtp` | `test_input_generator.py` | PP+MTP decoder 层解析、MTP 层为 None |
| `test_spec_decode_skips_runner_sampler_for_mtp` | `test_pipeline_parallel_model_builder.py` | MtpWrapper sampler guard |
| `test_spec_decode_skips_runner_sampler_for_pp_with_dflash` | 同上 | PP+DFlash sampler guard |
| `test_build_stage_model_clears_dflash_dspark_on_non_last_stage` | 同上 | 非 last stage 清除 DFlash/DSpark config |
| `test_pipeline_model_num_hidden_layers_includes_draft_layers` | 同上 | num_hidden_layers 含 draft 层 |
| `test_build_pipeline_plan_sets_extra_draft_layers_for_dflash` | 同上 | `extra_draft_layers` 设置正确 |
| `test_pipeline_stage_kwargs_remaps_draft_cache_to_last_stage` | 同上 | draft 层 KV cache 切片到 last stage |

---

## 5. 兼容性

- **PP=1**：所有改动均为 no-op。折算在无投机方法时除以 1，config 清除不触发（PP=1 是 last stage），sampler guard 不影响 `with_sampler=False` 的标准路径。
- **无投机解码**：候选门控移除不影响（`num_mtp_tokens=0` 的候选始终通过），折算为 no-op。
- **PP+MTP**（已合入 master 的 PR #711 基础上）：门控移除 + 折算 + TensorCast 修复使 MTP 在 PP 下正确工作。
- **PP+DFlash/DSpark**：新增翻译逻辑 + config 清除 + layer 计数 + sampler guard + `extra_draft_layers` KV cache 切片 + per-candidate `num_speculative_tokens` 使 DFlash/DSpark 在 PP 下正确工作。

---

## 6. 验证情况

### 6.1 单元测试

全部通过（本地 Python 3.14 + CI Python 3.11），覆盖：

- 候选生成（PP>1 接受 MTP/DFlash/DSpark 候选）
- decode 折算（TPOT 按 `(accept+1)` 正确折算）
- decoder layer 解析（PP+MTP 不抛 AttributeError）
- sampler guard（MtpWrapper/DflashWrapper 在 PP 下正确跳过外部 sampler）
- config 清除（非 last stage 清除 `mtp_config`/`dflash_config`/`dspark_config`）
- layer 计数（`PipelineModel.num_hidden_layers` 含 MTP + draft 层）
- KV cache 切片（`extra_draft_layers` 正确切到 last stage）
- 多 N 搜索（per-candidate `num_speculative_tokens` 正确设置）
- 检视修复覆盖：prefill MTP 约束（PP 候选镜像非 PP 语义）、draft 深度守卫（末级层数不足抛 `UnsupportedPPConfigurationError`）、DFlash/DSpark target 输入二选一契约、draft 配置顶层 `target_layer_ids` 回退

### 6.2 Kimi-K3 混布验证

在 Kimi-K3 混布场景下进行了功能验证，确认 PP+MTP/DFlash/DSpark 可以同时开启。验证中发现一个问题并修复：

- **多 N DSpark 搜索**：`--num-speculative-tokens 3 7`（多 N 搜索）时，CLI 只把第一个 N 写入 `args.num_speculative_tokens`，PP 路径靠 `copy.copy(base_user_input)` 继承会导致候选 2（N=7）拿到错误的 N=3。修复：显式设 `num_speculative_tokens = candidate.num_mtp_tokens`（commit `33bf398`，与 legacy 路径对齐）。

### 6.3 Kimi-K3 PP×投机消融实验（2026-09-09）

在 28K 输入 / 64 die 混布负载上做了对照消融：对照组为 upstream/master（不含本 RFC 实现），实验组为本实现 + PR #794（PP>1 prefill 口径修复），矩阵为 PP∈{1,4} × DSpark 开/关：

| 配置 | TTFT | TPOT | 吞吐 |
|---|---|---|---|
| master，PP=1 TP=8/EP=64，无投机 | 5.44s | 71.98ms | 100.43 t/s |
| master，PP=4 TP=4/EP=16，无投机 | 2.96s | 69.61ms | 54.25 t/s |
| 本实现，PP=1 TP=8/EP=64，DSpark | 5.30s | 21.07ms | 279.62 t/s |
| 本实现，PP=4 TP=4/EP=16，DSpark | 2.97s | 28.63ms | 121.86 t/s |

结论：

- **投机收益与 PP 正交**：无投机下 PP=1/PP=4 的 TPOT 仅差 3.3%（71.98 vs 69.61ms），验证"PP 只影响显存、不影响 TPOT"的定性；DSpark 将 TPOT 压至 21-29ms（×2.4-3.4，理论 ×3.58），TTFT 代价可忽略（±1.5%）。PP=4 档位 TPOT 略高（28.63 vs 21.07ms）来自 TP 配套变化（TP=4 时 draft 每 rank 计算翻倍），而非 PP 调度本身。
- **master 对照两个硬失败**：master 上 PP=1 + DSpark 构建即失败（draft 配置顶层 `target_layer_ids` 回退缺失，已由 `fa9517c` 修复并作为两组共同前置）；PP=4 + DSpark 被旧门控以 `ValueError` **中止整个搜索**（非按候选跳过）——门控移除的直接证据。
- **零回归**：无投机基线在 master 与本实现上逐位一致（PP=1 与 PP=4 两组），确认共存改造不影响既有路径。
- **PP 的价值是显存边界**：PP=1 + TP=4 在 28K 下显存缺口 -54.81 GB（不可行），PP=4 使 TP=4 档位成立。
- **PR #794（PP>1 prefill 口径）影响评估**：agg 分块 prefill 下 `chunk_shapes` 自带 per-chunk `seq_len` 并优先于 wave 级参数，带/不带 #794 的 PP=4 结果仅差 +1.8% TTFT；#794 的主要价值在 disagg 吞吐分子口径。本 RFC 无需携带 #794，其合入 master 后 rebase 复核即可。

---

## 7. 未来工作

- PD 分离（disagg）场景的端到端实测：当前已覆盖 disagg 优化器的单元级折算回归（`test_pp_decode_applies_mtp_fold`），混布（aggregation）场景已端到端验证（§6.3）；P 节点不开 PP、D 节点开 PP+MTP/DFlash/DSpark 的真实负载实测仍待进行。
- DFlash/DSpark draft model 的 per-stage KV cache 精确内存归属（当前 `extra_draft_layers` 已覆盖 KV cache 切片范围，但 draft cache 的 per-layer shape 来自 `_get_kv_cache_info` 的通用 Qwen3-GQA 公式，而非 draft model 的实际 attention 配置——若 draft model 使用非标准 attention，可能需要 per-layer 精确 shape）。

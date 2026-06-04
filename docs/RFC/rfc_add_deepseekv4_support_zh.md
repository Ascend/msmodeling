# RFC: DeepSeek-V4 模型适配支持（Flash/Pro）

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | 已批准 |
| **作者** | — |
| **创建日期** | 2026-06-03 |
| **相关链接** | [TensorCast 适配 DeepSeek-V4 设计文档](../design/deepseek_v4_adaptation_design.md) |

---

## 1. 概述

本提案旨在为 TensorCast 增加 DeepSeek-V4（Flash/Pro）模型的支持，涵盖 Head Compression（HC）机制、分层 KV 压缩、共享 KV 稀疏注意力、Lightning Indexer 和 Hash Routing MoE 等关键新特性，使 TensorCast 能够正确编译和仿真 DeepSeek-V4 模型。

---

## 2. 动机

DeepSeek-V4 是 DeepSeek 系列的最新版本，相比 V3/V3.2 引入了多项关键架构升级：

1. **Head Compression 机制**：通过 Sinkhorn-based 混合实现 token 级别的信息压缩与还原，提升模型表达能力和推理效率。
2. **分层 KV 压缩**：Per-layer 可配置压缩策略（滑动窗口 / 索引压缩 / 重度压缩），平衡 KV 缓存大小与注意力覆盖。
3. **共享 KV 注意力**：使用单一 wkv 投影产生共享 KV 向量，配合 grouped O projection 降低计算复杂度。
4. **Lightning Indexer**：Learned sparse attention selection，在 ratio=4 层选择关键 KV 进行注意力计算。
5. **Hash Routing MoE**：使用 token-id based hash 替代 softmax routing，实现确定性且高效的专家路由。

若不进行 V4 适配，TensorCast 将无法加载和仿真该模型，影响对最新 DeepSeek 架构的性能评估能力。

---

## 3. 目标

### 3.1 目标

- [x] 支持 DeepSeek-V4 模型加载和编译
- [x] 支持 Head Compression（HC）机制的性能建模
- [x] 支持分层 KV 压缩策略（ratio=0/4/128）
- [x] 支持共享 KV sparse attention
- [x] 支持 Lightning Indexer（ratio=4）
- [x] 支持 Hash Routing MoE
- [x] 支持 Clamped SwiGLU expert activation

### 3.2 非目标

- 不支持 V4 与其他模型的并行组合测试
- 不支持 KV cache block 与 indexer 的联合优化
- 不支持 CP（Context Parallel）与 V4 稀疏注意力的组合

---

## 4. 用例分析

### 4.1 DeepSeek-V4 模型推理仿真

**场景描述**：使用 TensorCast 对 DeepSeek-V4-Flash/Pro 模型进行性能仿真，评估在不同并行配置下的推理耗时。

**功能点**：

- 模型加载：`AutoConfig` / `AutoModel` 注册
- HC pre/post wrapping：每个 decoder layer attention 和 FFN 前后
- 分层 attention：ratio=0/4/128 的不同 attention 路径
- MoE 路由：hash routing 和非 hash routing

**性能指标**：

- Prefill 阶段误差 < 30%
- Decode 阶段误差 < 20%

**DFX 要求**：

- 兼容性：支持与 V3.2 模型 profile 共存
- 可维护性：V4 特有逻辑集中在独立文件
- 可测试性：提供单元测试和集成测试覆盖

### 4.2 V4 特有算子性能评估

**场景描述**：评估 V4 新语义算子的计算和内存开销。

**功能点**：

- HC 算子：Sinkhorn iterations、weighted reduction
- Compressor：分层 KV 压缩成本
- Lightning Indexer：learned sparse selection 成本
- Sparse Attention：shared KV attention 成本

---

## 5. 方案设计

### 5.1 整体方案

V4 适配分为五个层次：

1. **模型注册层**：`deepseek_v4` model profile 注册
2. **HC 语义算子层**：4 个语义算子表达 Head Compression
3. **KV 压缩算子层**：compressor 和 scatter_nd_update_mla
4. **稀疏注意力算子层**：quant_lightning_indexer 和 sparse_attn_sharedkv
5. **MoE 路由算子层**：moe_gating_top_k/kash 和 v4_clamped_swiglu

### 5.2 关键设计决策

#### 5.2.1 HC 语义算子合并策略

`hc_pre_sinkhorn` 将 Sinkhorn iterations 和 weighted reduction 合并为一个语义算子，确保成本模型一起统计。如果分离为两个算子，会导致：

- 成本模型无法关联 Sinkhorn iterations 和后续 reduction 的依赖关系
- 性能估算可能低估 combined kernel fusion 的收益

#### 5.2.2 KV 写入功能句柄

`scatter_nd_update_mla` 返回功能句柄而非数据句柄，确保 torch.compile 无法对上游 KV producer 链进行 DCE。设计模式与 V3.2 的 `mlapo_quant` 返回 cache handle 一致。

#### 5.2.3 Dynamic Topk 宽度

`quant_lightning_indexer` 输出宽度使用 `min(topk_limit, active_seq_len)` 而非固定 `topk_limit`：

- 避免 decode 阶段 topk 超出实际可用压缩序列长度
- 确保性能估算与实际运行时行为一致

#### 5.2.4 O Projection 独立 TP Group

V4 的 O projection 使用独立的 `o_proj_tp_group`，与 attention 的 `tp_group` 可分离：

- 需要在 TP plan 中单独注册 `wo_a` 和 `o_proj`
- `o_proj_tp_group.world_size` 必须 ≤ `o_groups`

### 5.3 技术选型

**选择方案**：新增独立 V4 语义算子和性能模型

**替代方案**：复用 V3.2 算子，通过参数控制行为

不选择替代方案的原因：

- V4 HC 机制与 V3.2 完全正交，无法通过参数兼容
- V4 compressor 和 indexer 的语义与 V3.2 DSA indexer 差异显著
- 分层压缩策略需要 per-layer 级别的配置支持

### 5.4 安全隐私与 DFX 设计

**兼容性**：

- `deepseek_v4` 作为独立 model_type，不影响 V3/V3.2 路径
- V4 Config 子类化 `DeepseekV3Config`，复用公共字段

**可维护性**：

- V4 特有逻辑集中在 `deepseek_v4.py` 文件
- 性能模型集中在 `builtin_model/deepseek_v4.py`
- 使用 `_tensor_cast_patched` / `getattr` 标记防重复 patch

**可测试性**：

- 提供单元测试覆盖每个语义算子
- 提供集成测试覆盖端到端推理流程

### 5.5 编程与调用设计

#### 5.5.1 模型加载接口

```python
from tensor_cast.transformers.builtin_model.deepseek_v4 import DeepseekV4Config, DeepseekV4Model

# 通过 HF AutoModel 加载
from transformers import AutoModel, AutoConfig
config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V4-Flash")
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-V4-Flash", config=config)
```

#### 5.5.2 V4 特有配置参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `compress_ratios` | List[int] | 必需 | Per-layer KV 压缩策略 |
| `layer_types` | List[str] | 可选 | Per-layer attention 类型 |
| `topk_limit` | int | 可选 | Lightning indexer top-k 上限 |
| `num_hash_layers` | int | 0 | Hash routing MoE 层数 |
| `hc_mult` | int | 4 | HC expansion factor |
| `hc_sinkhorn_iters` | int | 20 | Sinkhorn 迭代次数 |
| `o_groups` | int | 1 | O projection 分组数 |
| `o_lora_rank` | int | 可选 | O projection lora rank |
| `score_func` | str | "sqrtsoftplus" | 路由评分函数 |
| `swiglu_limit` | float | 可选 | Clamped SwiGLU 限制值 |

#### 5.5.3 语义算子接口

**hc_pre_sinkhorn**

```python
torch.ops.tensor_cast.hc_pre_sinkhorn(
    x,                    # HC mix tensor [B,S,mix_hc]
    hidden_states,        # Original HC-expanded [B,S,Hc,D]
    hc_scale,             # Sinkhorn scale [3, Hc]
    hc_base,              # Sinkhorn base [3, Hc]
    hc_mult,              # HC expansion factor
    sinkhorn_iters,       # Sinkhorn iterations
    hc_eps,               # Epsilon for sinkhorn
) -> (reduced, post, comb)
```

**quant_lightning_indexer**

```python
torch.ops.tensor_cast.quant_lightning_indexer(
    q_states,             # RoPE-processed q [B,S,H,D]
    weights,              # weights_proj output [B,S,H]
    indexer_cache,        # Indexer KV cache
    topk_limit,           # Top-k 上限
    tp_world_size,        # TP world size
    seq_lens,             # Per-request seq lengths
    query_lens,           # Per-request query lengths
) -> topk_indices
```

---

## 6. 测试设计

### 6.1 单元测试

| 用例 | 验证要点 |
|------|----------|
| `test_v4_config_parsing` | compress_ratios 验证、ratio 范围检查 |
| `test_hc_semantic_ops` | hc_pre_*/hc_post/hc_head 在 trace 中出现 |
| `test_v4_attention_wrapper` | ratio=0/4/128 不同路径处理 |
| `test_lightning_indexer` | Dynamic topk 宽度、indexer_cache layout |
| `test_moe_routing` | Hash routing 和非 hash routing 区分 |
| `test_clamped_swiglu` | v4_clamped_swiglu 与标准 SiLU 行为差异 |

### 6.2 集成测试

```bash
python -m pytest tests/test_tensor_cast/test_deepseek_v4.py -v
python -m pytest tests/test_tensor_cast/test_mla.py -v
python -m pytest tests/test_tensor_cast/test_moe_layer.py -v
```

### 6.3 端到端验证

使用 DeepSeek-V4 text generation compile 命令进行端到端验证：

1. 模型构建成功，HF `DeepseekV4Config` 可被 TensorCast 识别
2. 编译过程不再出现 HC 相关属性缺失
3. trace 中出现 HC 语义算子
4. ratio=4 层出现 `quant_lightning_indexer`
5. 已有模型测试不受影响

---

## 7. 缺点和风险

### 7.1 潜在风险

1. **Breaking Change**：V4 的 O projection 使用独立 TP group，可能影响现有并行配置
2. **性能回退**：HC 算子在低 hc_mult 时可能引入额外开销
3. **复杂度提升**：新增 11 个语义算子和对应的性能模型

### 7.2 应对措施

1. 在 ModelProfile 中明确声明 `o_proj_tp_group` 约束，提供清晰的错误信息
2. 通过性能模型验证 HC 算子在各种 hc_mult 配置下的开销
3. 单元测试覆盖所有新算子，确保成本模型准确

---

## 8. 现有技术参考

- **DeepSeek-V3.2 适配**：GLM5 design doc 中有关于 V3.2 DSA indexer 的详细设计，本方案在此基础上扩展 V4 特有的 HC 和分层压缩机制
- **DeepSeek 官方实现**：`deepseek-ai/DeepSeek-V4-Flash/inference/model.py` 作为参考实现

---

## 9. 未解决问题

1. **KV Cache Block 联合优化**：当前 indexer 返回 topk_indices，但尚未实现根据 indexer 结果加载 KV cache blocks 的机制
2. **CP + V4 Sparse Attention**：Context Parallel 与 V4 分层稀疏注意力的组合尚未支持

---

## 附录

### 参考资料

- [TensorCast DeepSeek-V3.2 适配](./rfc_add_deepseekv32_support_zh.md)
- [GLM5 TensorCast 适配](../design/glm5-tensorcast-adaptation-design.md)

### 术语表

| 术语 | 描述 |
|------|------|
| HC | Head Compression，通过 Sinkhorn 混合实现 token 级别压缩 |
| Compressor | V4 KV 压缩器，分层产生 coarse-grained KV stream |
| Lightning Indexer | V4 learned sparse attention selection |
| Hash Routing | Token-id based MoE 路由，替代 softmax top-k |
| Clamped SwiGLU | 带 gate clamp 的 SwiGLU activation |
| Ratio | KV 压缩比率，0=滑动窗口，4=索引压缩，128=重度压缩 |

# 特性设计：TensorCast 适配 Kimi K3 模型

---

## 1. 背景与接入动机

### 1.1 模型概览

Kimi K3（`moonshotai/Kimi-K3`）是 Moonshot AI 发布的 **2.8T 参数原生多模态 agentic 模型**，核心规格：

| 维度 | 规格 |
|------|------|
| 总参 / 激活 | 2.8T 总参，约 104B 激活 |
| 上下文窗口 | 1M token（`max_position_embeddings=1048576`） |
| 文本架构 | KDA（Kimi Delta Attention，K3 自研线性注意力）+ AttnRes（跨层注意力残差）混合，69 层 KDA + 24 层 MLA |
| MoE 框架 | Stable LatentMoE（专家前后含降维/升维投影），896 专家选 16，2 共享专家 |
| 激活函数 | SiTU-GLU（K3 自定义激活，`beta=4.0`，`linear_beta=25.0`） |
| 视觉编码器 | MoonViT-V2（27 层 3D 视觉 transformer） |
| 量化 | 权重自带 MXFP4 压缩（`compressed-tensors`，group_size=32） |

### 1.2 部署拓扑

- **A3 四节点混合部署**：4 个 Atlas 800 A3（64G × 16）节点，vLLM 数据并行跨四节点，每节点一个 DP rank，节点内 16 NPU 张量并行，拓扑 DP4/TP16/EP64。
- **A3 十六节点 PD 分离部署**：16 个 Atlas 800 A3 节点，8 Prefill + 8 Decode，两侧均 DP8/TP16/PP1。
- **A2 八节点混合部署**：8 个 Atlas 800 A2 节点，每节点一个 DP rank，节点内 8 NPU 张量并行，拓扑 DP8/TP8/EP64。

### 1.3 接入动机

K3 引入线性注意力（KDA）、Latent MoE、SiTU 激活等新结构，现有仿真框架无法直接加载和编译。接入后 TensorCast 可对 K3 进行 prefill/decode 仿真，输出算子级耗时与瓶颈分析，支撑并行配置（TP/EP/PP）选型。

### 1.4 远端代码与适配缺口

K3 属于**社区远端模型**（`trust_remote_code=True`），其文本子模型 `model_type="kimi_linear"` 为全新自研架构，与 Kimi K2.5 复用 DeepseekV3 的路径完全不同。直接加载并编译 K3 的核心适配缺口：

| # | 缺口 | 影响 |
|---|------|------|
| 1 | `fla-core` 依赖缺失 | `modeling_kimi_linear.py` 顶部硬性 `raise ImportError`，import 阶段即失败，需 stub `fla` 到 `sys.modules` |
| 2 | `situ` 激活算子缺失 | expert MLP 用 `SituAndMul`（`beta*tanh(g/beta)*sigmoid(g)*up`），现有 `swiglu` / `v4_clamped_swiglu` 无法识别，grouped_matmul 性能模型 act_fn 映射失败 |
| 3 | 混合注意力路由 | 93 层中 69 层 KDA + 24 层 Gated MLA，由 `config.is_kda_layer(layer_idx)` 分支控制；KDA 层需路由到 `linear_attention` op |
| 4 | Latent MoE 结构差异 | `KimiSparseMoeBlock` 在专家前后增加 `routed_expert_down_proj` → 专家 → `routed_expert_norm` → `routed_expert_up_proj` 包裹层，`patch_moe` 会丢弃这些投影 |
| 5 | MLA Output Gate | K3 MLA 设 `mla_use_output_gate=True`，输出经 `g_proj` + sigmoid 门控乘法，`multihead_latent_attention` op 不含此路径 |
| 6 | AttnRes 跨层残差 | `attn_res_block_size=12` 的 `_forward_attn_residual` 含 `block_residual` 跨层状态，torch.compile 追踪需 stub |
| 7 | KimiDynamicCache | `_supports_default_dynamic_cache=False`，自定义 cache 含 `conv_states`/`recurrent_states`/`key_cache`/`value_cache`，需识别并 stub |
| 8 | VL Forward 接口差异 | kwargs 过滤、`image_grid_thw` vs `grid_thws` 参数名映射、meta device merge 失败（同 K2.5） |
| 9 | 视觉编码器适配 | K3 `MoonViT3dEncoder` 已显式定义 `use_deterministic_attn`（比 K2.5 简单），但仍需注册 `tensor_cast` attention backend；视觉 `nn.RMSNorm` 因 `_dynamo.disable` 无法被 pattern pass 融合 |
| 10 | Transformers 环境兼容 | 依赖 `is_torch_fx_available`（transformers v5.x 已移除）、`flash_attention_2`（未安装不可用）、`OutputRecorder`（5.13+ 移除）、`create_causal_mask` 签名变更 |

设计遵循**最小侵入 + 单文件隔离**原则：所有适配集中在 `kimi_k3.py`，通过 `model_type == "kimi_k3"` 门控严格隔离，不影响任何已有模型。

---

## 2. 模型架构

### 2.1 层分布概览

K3 文本 decoder 共 93 层 `KimiDecoderLayer`，由 `config.is_kda_layer(layer_idx)` 判定每层注意力类型：

- **69 层 KDA**（Kimi Delta Attention，线性注意力）：呈间隔式分布，每 4 层中 3 层 KDA
- **24 层 MLA**（Gated MLA，全注意力）：每 4 层中 1 层 MLA

此外，层 0 为 Dense MLP（`first_k_dense_replace=1`），层 1..92 为 MoE。AttnRes 跨层残差以 `attn_res_block_size=12` 为粒度，每 12 层一个 block，`block_residual` 跨层传递。

> K3 文本侧 93 层深网络用线性注意力（KDA）替代 softmax 注意力，换来 O(1) decode 复杂度，代价是长程记忆靠递归状态压缩会有信息损失。AttnRes 的跨层 `block_residual` 就是对这个代价的直接补偿；视觉层浅且用标准 attention 且只有 27 层，不需要这个补偿。

### 2.2 顶层结构（VL 三件套）

K3 是原生多模态模型，顶层 `KimiK3ForConditionalGeneration`（`model_type="kimi_k3"`）由三部分组成。

| 组件 | 类 | 职责 | 适配要点 |
|------|-----|------|----------|
| `vision_tower` | `MoonViT3dPretrainedModel` | 3D 视觉编码器（27 层 `MoonViTEncoderLayer`），含 patch 化 + 2D RoPE + 注意力；详细算子级结构见 §2.8 视觉部分 | `use_deterministic_attn` 已显式定义（比 K2.5 简单）；需注册 `tensor_cast` attention backend（含 sdpa key，见 P8）；视觉 `nn.RMSNorm` 需 P1.14 融合 |
| `mm_projector` | `PatchMergerMLPV2` | 多模态投影器，视觉特征对齐到语言维度 `hidden_size=7168`；`tpool_patch_merger` 对时间维度下采样；详细结构见 §2.8 MM Projector 部分 | 未 TP 切分（文本仿真不触发） |
| `language_model` | `KimiLinearForCausalLM`（`model_type="kimi_linear"`，全新自研） | 文本 decoder：`embed_tokens` + 93 层 `KimiDecoderLayer`（见 §2.3）+ 模型级最终 `norm` + AttnRes 输出归一化/投影 + `lm_head` | 实际切分树见 §2.8；`lm_head` 需 P1.13 修复 VL 嵌套 TP |

### 2.3 Decoder Layer 架构（核心创新区）

```text
KimiDecoderLayer (93 层)
│
├── 层索引判定: config.is_kda_layer(layer_idx)
│   ├── True  → KimiDeltaAttention (KDA 线性注意力, 69 层) ── [★结构变化点]
│   └── False → KimiMLAAttention (Gated MLA 全注意力, 24 层)
│
├── 层索引判定: layer_idx >= first_k_dense_replace (=1)
│   ├── False → KimiMLP (dense, 仅层 0)
│   └── True  → KimiSparseMoeBlock (MoE, 层 1..92) ── [★Latent MoE 结构变化点]
│
├── input_layernorm / post_attention_layernorm: KimiRMSNorm的两个实例
│  ※ input_layernorm：仿真记录的算子tensor_cast.rms_norm、
│  ※ post_attention_layernorm：仿真记录的算子tensor_cast.add_rms_norm2
│
└── AttnRes (attn_res_block_size=12) ── [★独有: 跨层残差]
    ├── self_attention_res_norm + self_attention_res_proj
    └── mlp_res_norm + mlp_res_proj
    ※ 每 12 层一个 block, block_residual 块级残差跨层传递
```

### 2.4 KDA 线性注意力数据流（KimiDeltaAttention）

```text
输入: hidden_states [B, S, 7168 (hidden_size)]
  │
  ├─① QKV 投影 (标准 Linear)
  │   q_proj → q  [B,S, 96(num_heads)×128(head_dim)]
  │   k_proj → k  [B,S, 96×128]
  │   v_proj → v  [B,S, 96×128]
  │
  ├─② 短卷积预处理 (ShortConvolution)
  │   q→q_conv1d→q'   k→k_conv1d→k'   v→v_conv1d→v'
  │   kernel=4 (short_conv_kernel_size) + silu
  │   作用: 补线性注意力"局部感知弱"缺陷
  │   ★变化点: 三路独立卷积, 现有算子接收 mixed_qkv 需 patch 合并
  │   复用: linear_attn_causal_conv
  │   备注：线性注意力用递归状态压缩记忆，所有历史 token 压进固定大小的
  │       recurrent_state，对"最近几个 token"的精细感知弱，短卷积 k=4 补此缺口
  │
  ├─③ 门控参数生成 (并行)
  │   hidden → f_a_proj → f_b_proj → g     [B,S,96,128] (控制状态矩阵 S 老状态衰减)
  │   hidden → b_proj                 → beta [B,S,96]   (控制 S 新信息写入)
  │   复用: linear_attn_fused_gdn_gating
  │
  ├─④ KDA 核心计算
  │   q',k',v' + g + beta → o [B,S,96,128]
  │   ├─ prefill: chunk_kda            (并行展开)
  │   └─ decode:  fused_recurrent_kda  (逐 token 递归)
  │   ★隐藏坑1: K3 据 q 形状和 KV cache 状态自动判路径，开启 MTP 会让 query-length从 1 变 1+n，改变内部路径选择，
  │             已由 P9 规避：_patched_kda_forward 把 forward 整体替换为单一 torch.ops.tensor_cast.linear_attention
  │             由 TC 性能模型按 cache_position/seq_len 自行区分 prefill/decode
  │   ★隐藏坑2: chunk_kda="矩阵乘+状态递推"混合，仿真器可能漏掉状态递推，待精度需求明确后拆分
  │   标志位: use_qk_l2norm / use_gate / use_beta_sigmoid / safe_gate / lower_bound
  │   复用: linear_attn_chunk/recurrent_gated_delta_rule + 补 KDA 标志位
  │   ★需 stub: fla.ops.kda (详见 4.2)
  │
  ├─⑤ 输出门控 + 归一化
  │   hidden → g_proj (full_rank_gate) → g  [B,S,96,128] (输出门控)
  │   ★变化点: full_rank_gate 变体 (use_full_rank_gate=true)，g_proj: 7168→12288 一步 GEMM
  │   o → FusedRMSNormGated(o, g) → o'
  │   复用: linear_attn_gated_rmsnorm
  │
  └─⑥ 输出投影 (标准 Linear)
      o' → o_proj → output [B, S, 7168 (hidden_size)]
```

### 2.5 Gated MLA 层数据流（KimiMLAAttention）

```text
输入: hidden_states [B, S, 7168 (hidden_size)]
  │
  ├─① Q 低秩投影 (7168→1536→12288, Q 不存 cache)
  │   hidden → q_a_proj → q_a_layernorm → q_b_proj → q
  │   形状: 7168 → 1536 (q_lora_rank) → 96×(128+64)
  │   维度拆分: 128 (qk_nope_head_dim, 不加 RoPE) + 64 (qk_rope_head_dim, 加 RoPE)
  │   复用: K2.5 MLA patch
  │
  ├─② KV 低秩投影 (7168→512→12288, 减小 KV cache)
  │   hidden → kv_a_proj_with_mqa → kv_a_layernorm → kv_b_proj → kv
  │   形状: 7168 → 512 (kv_lora_rank) → 96×128 + 64(rope)
  │   说明: 实际存进 KV cache 的是瓶颈 r=512 的压缩表示, 比标准 MHA 的 12288 维小 24 倍
  │   复用: K2.5 MLA patch
  │
  ├─③ RoPE 旋转位置编码
  │   对 q/k 的 qk_rope_head_dim(=64) 部分施加旋转
  │   复用: apply_rope  ※ 需 position_ids patch (同 K2.5 P9); mla_use_nope=True 时 identity
  │
  ├─④ MLA 核心计算
  │   q, kv → multihead_latent_attention → attn_out [B,S,96×128]
  │   复用: 现有 MLA op
  │
  ├─⑤ Output Gate (★K3 独有)
  │   hidden → g_proj → sigmoid → gate [B,S,96×128]
  │   attn_out × gate → gated_out
  │   ★变化点: mla_use_output_gate=true (K2.5 无此门控)
  │   ★TP 要点: g_proj 投影 7168→96×128(全量), 但 MSModeling 按 head 切 TP,
  │             attn_out 为 num_heads_per_rank×128; gate 须按 rank 切片后再乘 (见 P13b)
  │
  └─⑥ 输出投影 (标准 Linear)
      gated_out → o_proj → output [B, S, 7168 (hidden_size)]
```

### 2.6 Latent MoE 层数据流（KimiSparseMoeBlock）

```text
输入: hidden_states [B, S, 7168 (hidden_size)]
  │
  ├─(identity 保留, 用于最终残差)
  │
  ├─① 路由门控 (KimiMoEGate)
  │   hidden → sigmoid + noaux_tc top-k → topk_idx, topk_weight
  │   896(num_experts) 选 16(num_experts_per_token)
  │   复用: K2.5 gate patch
  │
  ├─② ★Latent 降维 (Latent MoE 变化点)
  │   routed_expert_down_proj: 7168 (hidden_size) → 3584 (routed_expert_hidden_size)
  │   ★新增: 额外 grouped_matmul（P1.7 保留, patch_moe 默认会丢弃）
  │
  ├─③ 专家计算 (896 个 KimiBlockSparseMLP)
  │   每专家: w1(gate) + w3(up) → situ → w2(down) (grouped_matmul)
  │   ★新算子: situ 替代 swiglu
  │   ★命名: 专家用 w1/w2/w3 命名（非标准 gate_proj/up_proj/down_proj）
  │   ※ patch 为 zeros_like stub, 真实语义由 patch_moe 接管
  │
  ├─④ 专家聚合
  │   dispatch_ffn_combine: topk_weight 加权求和 → moe_out
  │
  ├─⑤ ★Latent 升维 (Latent MoE 变化点)
  │   routed_expert_norm: KimiRMSNorm  (latent_moe_use_norm=true, 作用在 3584 维 latent)
  │   routed_expert_up_proj: 3584 → 7168 (hidden_size)
  │   ★新增: rms_norm + 额外 grouped_matmul（P1.7 保留）
  │
  ├─⑥ 共享专家 (并行)
  │   shared_experts: KimiMLP × 2 (num_shared_experts=2)
  │   → shared_out  ※ 命名标准 (gate_proj/up_proj/down_proj), 命中 TP plan
  │   复用: KimiMLP
  │
  └─⑦ 残差合并
      final = moe_out + shared_out
```

## 3. 算子设计

### 3.1 算子新旧分类总览

| 算子类别 | 现有算子（复用） | 新增算子 | 来源参考 |
|----------|-----------------|----------|----------|
| 激活 | `swiglu`、`v4_clamped_swiglu` | **`situ`** | 参考 `v4_clamped_swiglu` 模式 |
| 线性注意力 | `linear_attn_chunk_gated_delta_rule`、`linear_attn_recurrent_gated_delta_rule`、`linear_attn_fused_gdn_gating`、`linear_attn_causal_conv`、`linear_attn_gated_rmsnorm` | 无（补 KDA 标志位参数） | `qwen3_next.py` |
| MLA | `multihead_latent_attention`、`apply_rope`、`kv_rmsnorm_rope_cache` | 无（output gate 在 patch 层处理） | `kimi_k25.py` |
| MoE | `grouped_matmul`、`dispatch_ffn_combine`、`init_routing_v2`、`moe_gating_top_k_softmax` | 无（Latent MoE 的 down/up proj 走标准 grouped_matmul） | — |
| Norm | `rms_norm`、`add_rms_norm`、`add_rms_norm2` | 无 | — |
| 量化 | `static_quant_linear_int4`、`dynamic_quantize_symmetric` 等 | 无（W4A8 走现有 quant 路径；MXFP4 走现有 mxfp4 路径） | — |
| 通信 | `all_reduce`、`all_to_all`、`matmul_allreduce` | 无 | — |

### 3.2 普通 Transformer vs KDA

| 普通 Transformer | KDA 线性注意力 |
|------------------|----------------|
| 每层把 token 改一改再传给下一层 | 就地更新状态 S，token 不在层间传递 |
| 记忆 = KV cache（随长度增长） | 记忆 = 状态 S（固定大小） |

---

## 4. 适配方案设计

### 4.1 整体设计思路

适配拆为四个层次，全部集中在 `kimi_k3.py`（约 2522 行）：

1. **新算子层**：新增 `situ` 激活算子及对应性能模型，参考 `v4_clamped_swiglu` 实现模式。
2. **配置修补层（Phase 1，`_patch_hf_config_for_kimi_k3` + 各 `_install_*`）**：模型加载前修改 HF config 和全局 import 状态——fla-core stub、`is_torch_fx_available` 恢复、`flash_attention_2` 降级、vision config 字段桥接、专家计数字段复制、Latent MoE 投影保留、KDA TP plan、SiTU 融合链、VL 嵌套 TP、视觉 RMSNorm 融合等。
3. **类 Monkey-Patch 层（Phase 2，`_patch_model_classes_for_kimi_k3`）**：动态导入远端模型类后注入修补方法——VL forward、视觉 backend、KDA → linear_attention 路由、MoE stub、MLA output gate、AttnRes、DynamicCache stub、SituAndMul 直连。
4. **ModelProfile 注册层（Phase 3）**：在模块底部注册 `ModelProfile`，声明 MoE/MLA 模块名、专家计数字段、视觉路径等元数据。

### 4.2 Phase 0：fla-core stub

**问题**：Kimi K3源码[modeling_kimi_linear.py]顶部硬导入 fla-core，缺失即 `ImportError`；仿真环境装 fla 会引入 triton 依赖问题。

**策略**（`_install_fla_stub`，对应 §5 P1）：

1. 先检测 4 个必需子模块（`fla.modules`/`fla.ops.kda`/`fla.ops.utils.index`/`fla.utils`）是否都能干净导入（`_fla_submodules_importable`）。
2. 若真实 fla 完全可用 → 跳过（尊重真实包）；否则（含半装：包在但 triton 缺失导致子模块坏）→ 注入 stub。
3. 注入的 stub 符号：
   - `fla.modules.ShortConvolution` → `_ShortConvolutionStub`（带 learnable conv1d weight 参数防 weight-iteration 崩溃，原样返回输入）
   - `fla.modules.FusedRMSNormGated` → `_FusedRMSNormGatedStub`（返回 x 不变）
   - `fla.ops.kda.chunk_kda`/`fused_recurrent_kda` → `_chunk_kda_stub`（按 v.shape 推 shape 返回空张量 + None state）
   - `fla.ops.utils.index.prepare_cu_seqlens_from_mask`/`prepare_lens_from_mask` → 简化实现
   - `fla.utils.tensor_cache` → 返回原函数不变的 decorator
4. 通过 `sys.modules.update({...})` 注入 7 个模块，手工 wire 父子属性以支持 `from fla.ops.utils.index import X`。
5. 幂等：`_FLA_STUB_INSTALLED` 全局 flag 守护。

> **关键设计**：stub 只需提供正确 shape 推断，真实 KDA 计算由 P9 改路由到 `torch.ops.tensor_cast.linear_attention`，stub 永不执行。

### 4.3 复用策略

| 复用来源 | 复用内容 | 复用度 |
|----------|----------|--------|
| Kimi K2.5 | VL forward kwargs 过滤、meta merge stub、视觉 backend、MLA RoPE patch、MoE stub、gate 路由 | ~60% |
| Qwen3-Next | KDA → linear_attention 路由、meta tensor mask patch | 几乎照搬 |
| DeepSeek V4 | 新激活算子实现模式（situ 参考 v4_clamped_swiglu） | 模式参考 |

K3 独有适配项（无现成可复用）：fla-core import stub、Latent MoE 扩展、MLA Output Gate、AttnRes 跨层残差、KimiDynamicCache stub、`situ` 激活算子、视觉 RMSNorm 融合。

---

## 5. Patch 完整目录（单一权威清单）

> 本节是 K3 全部 patch 的权威索引，按 P 编号排列。`行号` 指向 `kimi_k3.py`。`状态` = 已实现并验证 / 已实现待校准 / no-op。

### 5.1 Phase 1（config 级，模型加载前）

| Patch | 函数/类 | 行号 | 问题 | 修复 | 来源 | 状态 |
|-------|---------|------|------|------|------|------|
| **P1** | `_install_fla_stub` | 141–212 | fla-core 缺失致 import 失败 | `sys.modules` 注入 7 个 stub 模块（详见 §4.2） | K3 独有 | 已验证 |
| **P1.5** | 内联于 `_patch_hf_config_for_kimi_k3` | 1223–1255 | `OutputRecorder` 在 transformers 5.13+ 移除 | stub 为 no-op class | K3 独有 | 已验证 |
| **P1.6** | 内联 | 1258–1296 | K3 误用 `input_embeds`；`cache_position` 已废弃 | 包装 `create_causal_mask`：别名修正 + 丢弃 `cache_position` | K3 独有 | 已验证 |
| **P1.7** | `_install_latent_moe_patch` | 236–386 | `patch_moe` 丢弃 Latent MoE 投影/归一化 | monkey-patch `MoELayer.__init__`/`ParallelMoELayer.__init__`/`FusedMoETensorCast.forward`，捕获并应用 `routed_expert_down_proj`/`norm`/`up_proj` | K3 独有 | 已验证 |
| **P1.8** | `_install_copy_layer_attr_patch` | 426–484 | `CopyLayerWrapper` 丢失 `is_linear_attn`/`block_residual` | monkey-patch `__init__`/`forward` 传递 K3 属性与跨层残差 | K3 独有 | 已验证 |
| **P1.9** | `_install_kda_tp_plan_patch` | 511–576 | KDA `k_proj`/`v_proj`/`g_proj` 未 TP 切分 | monkey-patch `build_tp_plan_extras` 添加 COLWISE 分片 | K3 独有 | 已验证 |
| **P1.10** | `_install_situ_pattern_patch` | 598–706 | SiTU 分解为大量 elementwise aten op | 注册 4 个 pattern（±linear_beta × bf16/fp16）→ `tensor_cast.situ.default` | K3 独有 | 已验证 |
| **P1.11** | `_install_situ_sink_split_patch` | 716–749 | `situ` 未注册到 SinkSplitPass | 添加到 `_sink_config_registry`（与 swiglu 一致） | K3 独有 | 已验证 |
| **P1.12** | `_install_situ_gmm_pass_patch` + `GroupedMatmulSituPass` | 759–1017 | GMM + situ 未融合 | FX pass 支持三种 pattern（split+getitem / slice.Tensor / cat→split→slice），monkey-patch `apply_freezing_passes` | K3 独有 | 已验证 |
| **P1.13** | `_install_lm_head_tp_patch` | 1041–1115 | VL 嵌套 `lm_head`/`embed_tokens` 不命中 TP pattern | monkey-patch `shard_model_by_tp` 手动替换为 `ColumnParallelLinear`/`ParallelEmbedding` | K3 独有 | 已验证 |
| **P1.14** | `_install_vision_rms_norm_patch` | 1121–1196 | 视觉 `nn.RMSNorm` 因 `_dynamo.disable` 无法被 pattern pass 融合 | monkey-patch `shard_model_by_tp` 替换为 `RMSNormFusedWrapper`，eps=None→1e-5 | K3 独有 | 已验证 |
| **P2** | 内联 | 1373–1385 | `is_torch_fx_available` 在 transformers v5.x 移除 | `find_spec("torch.fx")` 实现并注入 | 复用 K2.5 P1 | 已验证 |
| **P3** | 内联 | 1388–1419 | `flash_attention_2` 未安装 | text/vision/text_config `_attn_implementation` → `"tensor_cast"` | 复用 K2.5 P2 | 已验证 |
| **P4** | 内联 | 1422–1441 | vision config 字段命名/缺失 | `merge_kernel_size`→`spatial_merge_size`；补 `temporal_patch_time`/`in_channels` | 复用 K2.5 P3 | 已验证 |
| **P5** | 内联 | 1444–1474 | 专家计数字段在 `text_config` 内 | 复制到根 config，缺失回退 896/2 | 复用 K2.5 P12 | 已验证 |

### 5.2 Phase 2（类级，模型类导入后）

| Patch | 函数 | 行号 | 问题 | 修复 | 来源 | 状态 |
|-------|------|------|------|------|------|------|
| Pre | `_patched_resolve` | 1507–1518 | Windows SIGALRM 兼容，trust_remote_code=None | 默认 True | 复用 K2.5 | 已验证 |
| **P6** | `patched_vl_forward` | 1529–1590 | 不接受 TC kwargs；`image_grid_thw` vs `grid_thws` | 过滤 kwargs + 参数名映射 + attention_meta 注入 `_extra_forward_kwargs` | 复用 K2.5 P4 | 已验证 |
| **P7** | `patched_merge_input_ids_with_image_features` | 1603–1659 | meta device embedding 失败 | 返回同 shape meta tensor（4 元组含 position_ids） | 复用 K2.5 P5 | 已验证 |
| **P8** | `visual_tc_adapter` | 1689–1787 | 视觉注意力缺 `tensor_cast`/`sdpa` backend | 注册到 `VL_VISION_ATTENTION_FUNCTIONS` 三 key；meta 调 `tensor_cast.attention`，真实走 fallback | 复用 K2.5 P6 | 已验证 |
| **P9** | `_patched_kda_forward` | 1804–1848 | fla-core 不可追踪 | 改路由到 `torch.ops.tensor_cast.linear_attention`，传 TP-local head 数（96/tp_size） | 照搬 Q3N | 已验证 |
| **P10** | `_patched_update_linear_attn_mask` | 1858–1900 | meta tensor `.item()` 崩溃 | meta 时跳过 | 照搬 Q3N | 已验证 |
| **P11** | `patched_moe_forward`/`patched_moe_infer` | 1912–1934 | 动态派发不可追踪 | `zeros_like` stub | 复用 K2.5 P7 | 已验证 |
| **P12** | `patched_gate_forward` | 1944–1971 | 非确定性 top-k | 等权重确定性路由 | 复用 K2.5 P8 | 已验证 |
| **P13a** | `_patched_resolve_position_embeddings` | 1991–2047 | 缺 position_ids→RoPE；rotary_emb=None | 从 position_ids 计算 cos/sin；None 时 identity | 复用 K2.5 P9 | 已验证 |
| **P13b** | `_patched_mla_forward_with_gate_check` | 2052–2153 | output gate；TP>1 gate 须切片 | `sigmoid(g_proj(x))` 乘 attn 输出，o_proj pre-hook 应用，TP>1 按 rank 切片 | 复用 K2.5 P10 + 新增 | 已验证 |
| **P14** | `_patched_decoder_forward` | 2167–2271 | AttnRes 跨层残差不可追踪；TC kwargs 丢失 | meta 时跳过 AttnRes；从 `_extra_forward_kwargs` 恢复 TC kwargs | K3 独有 | 已验证（近似） |
| **P15** | （no-op） | 2274–2288 | `KimiDynamicCache` 不被识别 | Phase 1 no-op，仅文档说明 | K3 独有 | no-op |
| **P16** | `patched_patch_embed_forward` | 2299–2351 | 2D 扁平输入无法 Conv2d | reshape + linear projection | 复用 K2.5 P11 | 已验证 |
| **P17** | `_situ_forward_direct`/`_block_sparse_mlp_forward`/`_mlp_forward` | 2370–2443 | FX 图冗余 cat→slice 阻碍 GMM+situ 融合 | SituAndMul 直连 `(gate, up)`，MLP forward 调 `act_fn(gate, up)` | K3 独有 | 已验证 |

### 5.3 使用说明

新增依赖：当前模型的仿真要求在环境中安装einops，已同步将该依赖更新至 requirements.txt，重新执行 pip install -r requirements.txt 或者 pip install einops 命令即可

**纯文本推理仿真**（W4A8_DYNAMIC 量化 + DP4/TP16/EP64）：

prefill阶段：

```bash
python -m cli.inference.text_generate "moonshotai/Kimi-K3" \
  --device ATLAS_800_A3_560T_128G_DIE \
  --num-devices 64 \
  --num-queries 8 \
  --query-length 3500 \
  --context-length 0 \
  --compile \
  --quantize-linear-action W4A8_DYNAMIC \
  --quantize-non-expert-linear-action W4A8_DYNAMIC \
  --tp-size 16 \
  --dp-size 4 \
  --ep-size 64 \
  --compilation-config enable_multistream \
  --enable-shared-expert-tp 
```

decode阶段：

```bash
python -m cli.inference.text_generate "moonshotai/Kimi-K3" \
  --device ATLAS_800_A3_560T_128G_DIE \
  --num-devices 64 \
  --num-queries 32 \
  --query-length 1 \
  --context-length 4250 \
  --compile \
  --quantize-linear-action W4A8_DYNAMIC \
  --quantize-non-expert-linear-action W4A8_DYNAMIC \
  --tp-size 16 \
  --dp-size 4 \
  --ep-size 64 \
  --compilation-config enable_multistream \
  --enable-shared-expert-tp 
```

**多模态推理仿真**（W4A8_DYNAMIC 量化 + DP4/TP16/EP64）：：

prefill阶段：

```bash
python -m cli.inference.text_generate "moonshotai/Kimi-K3" \
  --device ATLAS_800_A3_560T_128G_DIE \
  --num-queries 4 \
  --num-devices 64 \
  --query-length 30 \
  --context-length 0 \
  --image-batch-size 1 \
  --image-height 1080 \
  --image-width 1920 \
  --compile \
  --quantize-linear-action W4A8_DYNAMIC \
  --quantize-non-expert-linear-action W4A8_DYNAMIC \
  --tp-size 16 \
  --dp-size 4 \
  --ep-size 64 \
  --compilation-config enable_multistream \
  --enable-shared-expert-tp 
```

decode阶段：

```bash
python -m cli.inference.text_generate "moonshotai/Kimi-K3" \
  --device ATLAS_800_A3_560T_128G_DIE \
  --num-queries 64 \
  --num-devices 64 \
  --query-length 1 \
  --context-length 2851 \
  --compile \
  --quantize-linear-action W4A8_DYNAMIC \
  --quantize-non-expert-linear-action W4A8_DYNAMIC \
  --tp-size 16 \
  --dp-size 4 \
  --ep-size 64 \
  --compilation-config enable_multistream \
  --enable-shared-expert-tp 
```

---

## 6. SiTU 融合效果验证

SiTU 融合链（P1.10 + P1.11 + P1.12 + P17）将分解的 elementwise aten op 融合为两个算子：

| 融合算子                  | 次数  | 含义                                               |
|-----------------------|-----|--------------------------------------------------|
| `tensor_cast.situ.default` | 185 | 共享专家 SiTU 激活（92 MoE 层共享专家 * 2 + 1 层 0 dense MLP） |

---

## 7. 与已有模型对比

> 将 K2.5 / V4-Flash / Qwen3-Next 横向对照。最后一列标注 K3 适配来源。

| 维度 | K3 | Kimi K2.5 | DeepSeek V4-Flash | Qwen3-Next | K3 适配来源 |
|------|-----|-----------|-------------------|------------|-------------|
| 模型形态 | VL 多模态 | VL 多模态 | 纯文本 | 纯文本 | 复用 K2.5 VL 框架 |
| 文本 model_type | `kimi_linear`（全新自研） | `kimi_k2`（复用 DeepseekV3） | 自定义类 | transformers 内置 | K3 全部 trust_remote_code |
| 注意力架构 | 混合 69 KDA + 24 MLA | 纯 MLA（61 层全 MLA） | shared KV + HC + 分层 KV 压缩 | 每 4 层 1 full + 3 linear | K3 按层分支路由 |
| 线性注意力类 | `KimiDeltaAttention` | 无 | 无 | `Qwen3NextGatedDeltaNet` | 同族不同实现 |
| 线性注意力后端 | fla-core `chunk_kda`/`fused_recurrent_kda` | — | — | transformers 内置 GatedDeltaNet | K3 依赖 fla，需 stub |
| MLA Output Gate | 有（`g_proj` + sigmoid） | 无 | 无 | 无 | **K3 独有** |
| MoE 结构 | Latent MoE（down/norm/up 包裹） | 标准 `DeepseekV3MoE` | 标准 MLP + clamped SwiGLU | 标准 | **K3 独有扩展** |
| MoE 路由 | sigmoid + noaux_tc top-k | 标准 top-k | 前 3 层 hash routing + 其余 top-k | 标准 | 复用 K2.5 思路 |
| 专家数 / top-k | 896 / 16 | 384 / 8 | — | — | 仅数值差异 |
| 共享专家 | 2 | 1 | — | — | K2.5 单数；K3 复数 |
| 激活函数 | situ | silu | clamped SwiGLU | silu | **新增 situ** |
| MTP | 无 | 有类（`num_nextn=0`） | 有（1 层） | — | K3 profile 设 None |
| AttnRes 跨层残差 | `attn_res_block_size=12` | 无 | 无 | 无 | **K3 独有** |
| DynamicCache | `KimiDynamicCache` | 标准 | 标准 | 标准 | **K3 独有 stub** |
| 远端依赖 | trust_remote_code + fla-core | trust_remote_code | — | — | K3 需 sys.modules stub |
| 视觉 `use_deterministic_attn` | 已显式定义 | 缺失（需 patch 补） | — | — | K3 视觉 patch 更简单 |
| 视觉 RMSNorm 融合 | P1.14 手动替换 | pattern pass 自动 | — | — | **K3 独有**（`_dynamo.disable` 阻碍） |
| VL forward patch | 需 | 需 | 不适用 | 不适用 | 复用 K2.5 P4 |
| KDA → linear_attention 路由 | 需 | 不适用 | 不适用 | 已实现 | 照搬 Q3N |
| 新增语义算子 | 1 个（situ）+ 3 融合变体 | 0 | 9 个 | 0 | V4 算子缺口远大于 K3 |
| 适配方式 | patch 式（大量 monkey-patch） | patch 式 | 声明式 | patch 式 | K3 靠 patch |
| 配置特有字段 | activation_situ_beta / routed_expert_hidden_size / mla_use_output_gate / attn_res_block_size | — | compress_ratios / hc_mult 等 | — | 字段集不重叠 |

---

## 8. 测试设计

| 测试项 | 覆盖内容 | 验收标准                             |
|--------|----------|----------------------------------|
| text-only 仿真 | 无图像输入的 prefill/decode | 纯文本链路跑通，关键算子计数合理 |
| VL 仿真 | 含图像输入的 prefill/decode | 视觉链路跑通，关键算子计数合理    |

### 新增算子

- `tensor_cast.situ.default`
- `tensor_cast.apply_attn_res.default`
- `tensor_cast.linear_attn_causal_conv.default`
- `tensor_cast.linear_attn_gated_rmsnorm.default`
- `tensor_cast.linear_attn_recurrent_gated_delta_rule.default` - 只有decode阶段会出现
- `tensor_cast.linear_attn_chunk_gated_delta_rule.default` - 只有prefill阶段会出现

---

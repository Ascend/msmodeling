# RFC: MsModeling 支持 Decode Context Parallel 仿真

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | 草案 |
| **作者** | Elrond |
| **创建日期** | 2026-05-14 |
| **相关链接** | [vllm-ascend Context Parallel Design](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/context_parallel.html) · [vllm-ascend Context Parallel 用户指南（中文）](https://docs.vllm.ai/projects/ascend/zh-cn/main/user_guide/feature_guide/context_parallel.html) · [vllm-project/vllm#25749](https://github.com/vllm-project/vllm/issues/25749) · [vllm-ascend#3260 support cp&dcp](https://github.com/vllm-project/vllm-ascend/pull/3260) · [vllm-ascend#4572 pcp+mtp](https://github.com/vllm-project/vllm-ascend/pull/4572) · [vllm-ascend#5672 dcp+mlapo](https://github.com/vllm-project/vllm-ascend/pull/5672) · [vllm-ascend#6563 dcp+sfa](https://github.com/vllm-project/vllm-ascend/pull/6563) · [知乎参考文章](https://zhuanlan.zhihu.com/p/2020086868914499979) |

---

## 1. 概述

### 1.1. 概述

本提案在 MsModeling 中新增对 **Decode Context Parallelism (DCP)** 的仿真支持。

DCP 的本质是：**沿序列维度切分 KV cache，使原本在 TP 组内冗余复制的 KV cache 改为分片存储**，从而：

- 单卡每条序列的 KV 占用降为 `1 / dcp_size`，等量 block 可装下 `dcp` 倍长度的序列，支撑更大 batch / 更长 context；
- 单次 decode 先在 DCP 组内 **all-gather Q**（沿 head 维），让每张卡持有 `h_q · dcp / tp` 个 head 后再用本地 `S/dcp` 的 KV 跑 attention；本地输出与 `lse` 经 **all-to-all + online-softmax merge** 重组成完整输出。MLA 与 GQA 走同一通信模式（每层 2 次集合通信），区别仅在 head 划分语义上；
- 由于复用 TP 通信域，**不引入新的设备**，仅在 TP 切分维度上叠加 sequence 分片语义。

### 1.2. 目标

1. 在 `ParallelConfig` / `UserInputConfig` 中增加 `dcp_size` 维度，并在 `ModelRunner` 仿真链路中正确建模 DCP 引起的 **显存变化** 与 **通信/计算变化**；
2. 在 `throughput_optimizer` 的寻优空间中加入 **DCP 维度**，使 Decode 寻优能够自动搜索最优 `(tp, dcp, ep, moe_dp, batch)` 组合；
3. 按张量 shape 改写清单的方式，明确 KV cache / attention / 通信原语 / metadata buffer 四类 op 在现有切分计算逻辑中需要修改的 shape，供 analytic performance model 按图实现，FLOPs / 显存 / 通信量由现有 estimator 自动推导。

### 1.3. 非目标（本期不做，留作后续）

1. **不建模 Prefill Context Parallel (PCP)**：PCP 涉及 head-tail 序列切分、独立通信组扩张 `world_size`、`AllGatherKV` / `AllGatherQ` / `Ring` 三种 chunked prefill fallback、`reorg_kvcache` 重排等额外语义，单独立 RFC 处理；
2. **不建模其他 CP 路线**：Ulysses（序列↔head 维 a2a 互换）与 Ring Attention（KV 在环上 P2P 滚动）不在本期仿真范围，本 RFC 仿真口径**仅对齐 vllm-ascend DCP 实现**；
3. **不建模 KV 跨实例搬运**：PD 分离 / KV pooling 场景下 `cp_kv_cache_interleave_size = block_size` 引起的 block-interleave KV 迁移开销不纳入 TPOT，仿真只确认配置合法性，不对 P→D 间传输时间建模；
4. **不建模 DCP 与稀疏 attention (SFA) 的交互**：DeepSeek-V3.2 的 SFA backend 把 `S` 替换为 `top_k`，attention FLOPs 公式需要 sparse/dense 双套，由后续 RFC 补齐；
5. **不建模 MLAPO 融合算子在 DCP 下的内部切分**：MLAPO（RMSNorm + QKV proj + RoPE + KV-write 融合 op）在 DCP 启用时按 `dcp_rank` 写入本地 KV 分片，本期把它视为一个普通融合算子按 `tp` 切，不细化 fused 内部的 dcp 子切分；
6. **不建模 ACLGraph 录制阶段的 launch-overhead 合并优化**：graph 模式下 kernel launch 时间会被合并，但分析模型仍按 kernel 累加，最坏高估 launch 开销；
7. **不修改 vllm-ascend 上游代码**：仿真行为对齐已合入 vllm-ascend PR #3260 / #4572 / #5672 / #6563 的实现，不引入新的算子或通信原语。

本期 **仅建模 Decode 路径**，Prefill Context Parallel (PCP) 留作后续 RFC。

## 2. 方案设计

### 2.1. 推荐方案

#### 2.1.1. MsModeling 仿真口径选择

MsModeling 的目标是为 Ascend 推理引擎做性能建模，**仿真口径完全对齐 vllm-ascend** 的 DCP 实现：

- KV cache 切分粒度 `cp_kv_cache_interleave_size` 这个字段只影响 token 在卡上的分布顺序，不影响计算量，所以忽略；
- DCP 通信域复用 TP，故 `world_size = tp_size * pp_size * dp_size`，**不额外扩张 world size**；
- DCP 维度在 TP 组内进一步划分，需满足约束；
- DCP 通信路径与 vllm-ascend 实现一致：MLA 与 GQA 均走 `all_gather Q` + `all_to_all_single(output ⊕ lse)`，最终的合并均在本地完成 online-softmax。

#### 2.1.2. 配置扩展

##### 2.1.2.1 `serving_cast/config.py` — `ParallelConfig`

```python
@dataclass
class ParallelConfig:
    world_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    dcp_size: int = 1                    # 新增：Decode Context Parallel size，复用 TP 设备
    # ... 其他字段保持不变
```

##### 2.1.2.2 `tensor_cast/core/user_config.py` — `UserInputConfig`

```python
@dataclass
class UserInputConfig:
    # ... existing ...
    dcp_size: int = 1
```

并在 `get_parallel_config()` 中透传：

```python
return ParallelConfig(
    ...,
    decode_context_parallel_size=self.dcp_size,
)
```

#### 2.1.3. 约束条件

复用 vllm-ascend 的约束：

  $$
  \texttt{tp\_size} \;\geq\; \texttt{dcp\_size}, \qquad
  \texttt{tp\_size} \bmod \texttt{dcp\_size} = 0
  $$

此外，**GQA 后端**下每个 dcp rank 持有 `h_kv · dcp / tp` 个 KV head，必须 ≥ 1，否则该 rank 无 KV head 可读，配置非法：

  $$
  \texttt{h\_kv} \;\geq\; \dfrac{\texttt{tp\_size}}{\texttt{dcp\_size}}
  $$

（MLA 后端 KV 不沿 TP 切分，此约束仅作用于 GQA 路径。）

上述约束的校验时机：

- `tp_size ≥ dcp_size` 与 `tp_size mod dcp_size = 0`：与模型无关，在 `ParallelRunner._get_user_config()` 生成配置时即可执行；
- `h_kv ≥ tp_size / dcp_size`：依赖具体模型的 `num_key_value_heads`，需在加载 `ModelConfig` 后、首次实例化 attention backend 前完成；建议在 `_get_user_config()` 拿到 model config 之后、构造 `ParallelConfig` 前一并校验，避免延后到 backend 内部抛错。

违反任一约束的组合直接报错。

#### 2.1.4. 仿真需要修改的张量 Shape 清单

MsModeling 的仿真本质是 **"用算子级 shape 驱动 analytic / profiling performance model"**——只要 decode 路径上各 op 的输入/输出 shape 按 DCP 切分语义改对，FLOPs、显存、通信量会被现有 estimator 自动算出。因此本节把 DCP 启用后**所有需要改 shape 的位置**作为 implementation checklist 罗列出来。

**符号约定**（后续 2.1.5 仍沿用）：

| 符号 | 含义 |
| :--- | :--- |
| `B` | decode batch size（已含 MTP 投机 token） |
| `S` | 单条请求 context 长度（即 KV cache 中已有 token 数） |
| `Q` | 单次 decode query 长度（普通 decode `Q = 1`，MTP-x 时 `Q = 1 + num_spec_tokens`） |
| `L` | 模型层数 |
| `h_q` | 每层 Q head 总数 |
| `h_kv` | 每层 KV head 总数（MLA 取 1） |
| `D` | 每个 head 的维度（MLA 取 `kv_lora_rank + qk_rope_head_dim`） |
| `h_hidden` | 隐层维度（= `h_q · D_q`） |
| `tp`, `dcp` | TP / DCP size |
| `dtype_bytes` | 元素字节数（fp16/bf16=2, fp32=4, kv-int8=1） |
| `N_d`, `N_p` | 本步 decode / prefill 请求数 |
| `T_max` | ACLGraph 录制时的最大 token 数 |

shape 改写分四类：**A. KV cache 与 paged storage**、**B. attention op 的 Q/K/V/out/lse**、**C. DCP 组内集合通信 op**、**D. metadata buffer**。每一类下面用"原 → 新"格式列出。

##### 2.1.4.1 KV cache 与 paged storage（A 类）

DCP 的核心节省是 "**每条序列在本卡的 KV 占用减为 `1/dcp`**"（dcp ranks 各持一段 seq），但**实现方式不是把 per-rank KV tensor 物理变小**——`num_blocks_per_rank` 由该 rank 可用显存预算决定，DCP 不动这个预算，所以 tensor 形状 `[2, N_blk, block_size, h_kv/tp, D]` 不变。"切分"发生在 **logical block_size 被放大**：`vllm_ascend/attention/context_parallel/mla_cp.py:75-79` 把 scheduler 视角下的 `block_size` 乘上 `cp_virtual_block_size = cp_kv_cache_interleave_size · dcp · pcp`，每条序列的 token 按 `cp_kv_cache_interleave_size` 粒度交错落到 dcp ranks 上 —— 等效效果：**本卡同样的 `num_blocks` 现在能装下 `dcp` 倍长度（或 `dcp` 倍 batch）的序列**。

| 张量 | 启用 DCP 前 | 启用 DCP 后 | 备注 |
| :--- | :--- | :--- | :--- |
| `block_size` | `block_size` | 物理 `block_size` 不变；调度器视角下逻辑 `block_size · dcp`（吞下 `dcp×` 更长的序列） | 模拟时 `per_seq_kv_bytes_per_rank ÷= dcp`，物理 tensor 维度不变（与下方 `kv_cache` 物理 shape 描述一致） |
| `kv_cache` (GQA) | `[2, N_blk, block_size, h_kv/tp, D]` | **物理 shape 不变**；但每条 seq 在本卡的 block 占用 `≈ 1/dcp`（block 内交错） | 仿真口径：`per_seq_kv_bytes_per_rank` 除以 `dcp`，`N_blk` 不动 |
| `kv_cache` (MLA) | `[N_blk, block_size, kv_lora_rank + qk_rope_head_dim]` | **物理 shape 不变**；同上 | latent KV 单独存 |
| `block_table` | `[N_d + N_p, max_blk_per_seq]` | **DCP-only 不变**；DCP + spec_decode 时扁平化为 `[N_d · decode_threshold + N_p, max_blk_per_seq]` | 扁平化路径需要计算这块空间 |
| `slot_mapping` | `[total_num_tokens]` int32 | **DCP-only 不变**（PCP 才有 `pcp_padded_slot_mapping` 扩到 `[total_num_tokens · pcp]`，见 `vllm_ascend/worker/pcp_utils.py:470-482`） | DCP 的 slot 路由通过 block 内交错完成，不改 slot_mapping 长度 |

##### 2.1.4.2 Attention op 的 Q/K/V/out/lse（B 类）

MLA 与 GQA decode 都走 "all_gather Q → 本地 attention → all_to_all_single(output⊕lse) → merge" 这同一模板，所以 attention op 内部的 shape 改写是同构的。两套后端只在 KV head / latent 维划分上有差异。

**MLA 后端**（`h_kv = 1`，`D = kv_lora_rank + qk_rope_head_dim`；TP 切 Q heads，不切 latent）：

| 张量 | 启用 DCP 前 | 启用 DCP 后 | 说明 |
| :--- | :--- | :--- | :--- |
| `Q_local` (输入 reorg_decode_q) | `[B, Q, h_q/tp, D]` | `[B, Q, h_q/tp, D]` → **all_gather Q 后** `[B, Q, h_q · dcp / tp, D]` | `mla_cp.py::reorg_decode_q` 沿 head 维 all_gather |
| `K_local` / `V_local` (本卡 paged cache 取数) | `[B, S, D]` | `[B, S/dcp, D]` | latent KV 沿 seq 切，本卡只读本地分片 |
| `attn_output_local` | `[B, Q, h_q/tp, v_head_dim]` | `[B, Q, h_q · dcp / tp, v_head_dim]`（partial，未 merge） | 进 attention 时 `num_heads = self.num_heads * dcp_size`，见 `mla_cp.py:625` |
| `attn_lse_local` ⭐**新增** | — | `[B, Q, h_q · dcp / tp]` fp32 | DCP merge 所需；算子调用时 `softmax_lse_flag=True` |

**GQA 后端**（TP 切 Q head；DCP 与 TP 共域）：

| 张量 | 启用 DCP 前 | 启用 DCP 后 | 说明 |
| :--- | :--- | :--- | :--- |
| `Q_local` | `[B, Q, h_q/tp, D]` | `[B, Q, h_q/tp, D]` → **all_gather Q 后** `[B, Q, h_q · dcp / tp, D]` | `attention_cp.py::_forward_decode_pcp_dcp` 沿 head 维 all_gather |
| `K_local` / `V_local` | `[B, S, h_kv/(tp), D]`（重复存） | `[B, S/dcp, h_kv/(tp/dcp), D]` | 沿 seq 切，**消除 TP 组内 KV 重复** |
| `attn_output_local` | `[B, Q, h_q/tp, D]` | `[B, Q, h_q · dcp / tp, D]`（partial，未 merge） | 同上 |
| `attn_lse_local` ⭐**新增** | — | `[B, Q, h_q · dcp / tp]` fp32 | 同 MLA |

##### 2.1.4.3 DCP 组内集合通信 op（C 类）

MLA 与 GQA 后端走**同一对** DCP 集合通信原语，对应 `vllm_ascend/attention/context_parallel/{mla_cp.py, attention_cp.py, common_cp.py}` 中的 `dist.all_gather`（沿 head 维）+ `dist.all_to_all_single`（沿 head 维拆 `dcp` 份）。

| 后端 | 通信原语 | 每层集合通信次数 |
| :--- | :--- | :--- |
| MLA | `all_gather Q`（head 维）+ `all_to_all_single(output ⊕ lse)`（head 维）+ 本地 online-softmax merge | **2** |
| GQA | `all_gather Q`（head 维）+ `all_to_all_single(output ⊕ lse)`（head 维）+ 本地 online-softmax merge | **2** |

> 备注（**实现要点**）：vllm-ascend 在 `_process_attn_out_lse` 内把 `attn_output` 与 `softmax_lse` 都 upcast 到 **fp32** 后再做 a2a。因此 `all_to_all_single(output ⊕ lse)` 的通信量必须按 **`bytes_per_element = 4`（fp32 固定）** 计算，**不要沿用模型权重 / KV 的 `dtype_bytes`（fp16/bf16 = 2、kv-int8 = 1）**，否则在低比特模型上会显著低估 a2a 字节数。`all_gather Q` 走原 dtype，不受此影响——即同一层的两次集合通信在 estimator 中需要传入不同的 `bytes_per_element`。

##### 2.1.4.4 Metadata buffer（D 类）

DCP-only 路径下，真正**新增**的 metadata buffer 只有 `attn_lse_local`（算子带 lse 输出，DCP merge 需要）。

| Buffer | shape | dtype | 用途 |
| :--- | :--- | :--- | :--- |
| `attn_lse_local` ⭐ | `[B · Q, h_q · dcp / tp]` | fp32 | 算子带 lse 输出，online-softmax merge 输入 |

数量级 KB–MB 级，对显存预算影响很小但需要计入，避免与 attention workspace 算重。

##### 2.1.5 Shape 改写位置总览

把 2.1.4.1-2.1.4.4 所有 shape 改动按代码文件归类，供实现时按图施工：

| 改写位置 | 涉及文件 | 涉及 shape |
| :--- | :--- | :--- |
| KV cache block_size 申请 | `tensor_cast/core/input_generator.py` | A 类 KV cache 单卡的 block_size |
| KV cache 申请 | `tensor_cast/core/model_runner.py` | A 类 KV cache 张量 |
| KV cache slot 索引 | `tensor_cast/ops/mla.py`、`tensor_cast/layers/attention.py` | A 类 `slot_mapping` |
| Attention kernel 输入/输出 | `tensor_cast/layers/mla.py`、`tensor_cast/layers/attention.py` | B 类 Q/K/V/out/lse |
| DCP 通信原语 | `tensor_cast/performance_model/comm_analytic.py`（复用现有 estimator） | C 类 `all_gather`（MLA / GQA 的 Q，用模型原 dtype）+ `all_to_all_single`（MLA / GQA 的 output⊕lse，**固定 `bytes_per_element = 4`（fp32），不要沿用 `dtype_bytes`**，详见 §2.1.4.3） |
| metadata buffer | `tensor_cast/core/model_runner.py`、`serving_cast/kv_cache_manager.py` | D 类 |
| block_table 扁平化 | `serving_cast/kv_cache_manager.py` | A 类 `block_table` |

> DCP 复用 TP 通信域，因此 DCP 组的 ranks 是 TP 组内的一段连续切片，`comm_grid` 解析无需新增拓扑。

### 2.2. 替代方案

| 方案 | 描述 | 不采纳原因 |
| :--- | :--- | :--- |
| **A. 建模 Ulysses 或者 Ring Attention** | Ulysses复用现有 ulysses 的 SP 抽象 | 与 vllm-ascend 部署不一致 |
| **B. PCP + DCP 同期建模** | 一次性补齐 prefill / decode | 改动面大；prefill 涉及 `head-tail` chunk 切分、`chunked prefill` 三种 fallback 策略，建模难度更高，应单独立项 |

## 3. 实施计划

### 3.1. 里程碑

| 阶段 | 任务 | 产出 |
| :--- | :--- | :--- |
| M1 | `ParallelConfig` / `UserInputConfig` 字段扩展 + 校验 | PR1：配置层支持 `dcp_size` |
| M2 | `ModelRunner` 的 KV 显存与 attention 计算修正 | PR2：DCP 显存 / 计算建模 |
| M3 | `tensor_cast/layers/mla.py`、`attention.py` 注入 DCP 通信 hook | PR3：DCP 通信建模 |
| M4 | `ParallelRunner` 寻优空间扩展 + CLI 参数 | PR4：寻优集成 |
| M5 | `optimizer_summary` 列扩展 + 输出格式更新 | PR5：输出展示 |
| M6 | 单测 + 端到端 vllm-ascend 对照基线 | PR6：测试与精度对齐 |

### 3.2. 测试计划

| 测试项 | 描述 | 通过标准 |
| :--- | :--- | :--- |
| 单元测试：约束校验 | MLA / GQA 场景下非法 `(tp, dcp)` 组合被剪枝 | 100% 覆盖 |
| 单元测试：KV 显存公式 | `dcp_size` 从 1 增到 8 时，`kv_cache_size_gb` 单调下降为 `1/dcp` | 数值误差 ≤ 0.5% |
| 单元测试：通信量公式 | MLA 与 GQA 后端的 `V_DCP` 与 通信量计算公式完全一致 | 完全匹配 |
| 集成测试：TPOT 收敛 | 固定 batch / 输入，扫 `dcp ∈ {1,2,4,8}`，TPOT 单调下降直到通信项占优 | 曲线形状符合预期 |
| 精度对照：vllm-ascend | 取 `DeepSeek-R1-W8A8 + DCP8`、`Qwen3-235B + DCP2` 两组实测 | 仿真 TPOT 与实测误差 ≤ 15% |
| 兼容性测试：与 PD 配比寻优 | `--enable-optimize-prefill-decode-ratio` 同时启用 `--dcp-sizes` | Prefill 强制 `dcp=1`，Decode 正常遍历 |

### 3.3. 后续工作

- **PCP 仿真**：单独立 RFC，建模 head-tail KV 切分、`AllGatherKV` / `AllGatherQ` / `Ring` 三种 chunked-prefill 策略，以及 `reorg_kvcache` 的重排开销（参考 [vllm-ascend#3260](https://github.com/vllm-project/vllm-ascend/pull/3260)、[vllm-ascend#4572](https://github.com/vllm-project/vllm-ascend/pull/4572)）；
- **MLAPO + DCP 融合算子建模**：MLAPO 把 RMSNorm + QKV proj + RoPE + KV-write 融合为单一 op，DCP 启用时融合算子按 `dcp_rank` 写入本地 KV 分片；需要确认 `tensor_cast/ops/mla.py` 中 `mlapo` op 的内存/计算 size 按 `dcp` 缩放（参考 [vllm-ascend#5672](https://github.com/vllm-project/vllm-ascend/pull/5672)）；
- **SFA (Sparse Attention) + DCP 建模**：DeepSeek-V3.2 走 SFA backend，attention 计算量由 `top_k` 而非 `S` 决定，DCP 通信公式仍适用但需要 sparse/dense 双套（参考 [vllm-ascend#6563](https://github.com/vllm-project/vllm-ascend/pull/6563)）；
- **KV 传输 + DCP 联合建模**：在 PD 分离场景下，建模 `cp_kv_cache_interleave_size = block_size` 时跨实例的 block-interleave KV 搬运开销，以及 Mooncake / Layerwise connector 的迁移路径；
- **DCP 与 KV 量化的交互**：`quantize_attention_action` 启用时 `dtype_bytes` 改变，需复核 KV 显存公式；
- **图模式（ACLGraph）下 DCP 的 launch overhead**：当前 analytic 模型按 kernel 时间累加，graph 模式下的合并优化暂未建模，可在 profiling 模式下补齐。理论上 analytic 和 profiling 模式变化不影响cp并行逻辑，需要测试。
- **PR #3260 + #4572 实测对照**：把 ACLGraph + MTP + DCP 三联开启的实测 TPOT 加入 3.2 测试基线，验证仿真误差 ≤15%。

### 3.4. 修改文件清单

| 文件 | 说明 |
| :--- | :--- |
| `tensor_cast/core/input_generator.py` | `_get_kv_cache_info` 增加 `block_size` 针对 `dcp_size` 的缩放 |
| `serving_cast/config.py` | `ParallelConfig` 增加 `dcp_size`、`cp_kv_cache_interleave_size` |
| `tensor_cast/core/user_config.py` | `UserInputConfig` 新增字段并透传到 `ParallelConfig` |
| `tensor_cast/model_config.py` | `ParallelConfig` 增加 `decode_context_parallel_size` |
| `tensor_cast/core/model_runner.py` | `kv_cache_size_gb` 公式按 `dcp_size` 缩放 |
| `tensor_cast/layers/mla.py` | MLA decode 路径注入 `all_gather Q` + `all_to_all_single(output ⊕ lse)`，KV 取数按 dcp_rank 切片；attention estimator 的 `num_heads` / `seq_len` 改为 `h_q · dcp / tp` 与 `S / dcp` |
| `tensor_cast/layers/attention.py` | GQA decode 路径注入 `all_gather Q` + `all_to_all_single(output ⊕ lse)`（与 MLA 同模板） |
| `tensor_cast/performance_model/comm_analytic.py` | 复用现有原语，新增 DCP 组解析（rank 列表是 TP 组的子集） |
| `serving_cast/parallel_runner.py` | `_get_user_config()` 扩展 DCP 搜索维度，区分 P/D phase |
| `cli/inference/throughput_optimizer.py` | 新增 `--dcp-sizes` |
| `serving_cast/service/optimizer_summary.py` | 输出列追加 `dcp`，`format_parallel_label` 增补 `dcp{n}` |
| `serving_cast/service/utils.py` | `DISAGG_COLUMNS` 增加 `dcp`；`resolve_search_sizes` 调用方对齐 |
| `tests/test_serving_cast/test_dcp_simulation.py` | DCP 仿真新增单测 |
| `tests/test_serving_cast/test_parallel_runner_dcp.py` | 寻优搜索新增单测 |

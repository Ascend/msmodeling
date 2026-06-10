# RFC: MsModeling Support for Decode Context Parallel Simulation

## Metadata

| Item | Content |
| :--- | :--- |
| **Status** | Draft |
| **Author** | Elrond |
| **Created** | 2026-05-14 |
| **Related Links** | [vllm-ascend Context Parallel Design](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/context_parallel.html) · [vllm-ascend Context Parallel User Guide (Chinese)](https://docs.vllm.ai/projects/ascend/zh-cn/main/user_guide/feature_guide/context_parallel.html) · [vllm-project/vllm#25749](https://github.com/vllm-project/vllm/issues/25749) · [vllm-ascend#3260 support cp&dcp](https://github.com/vllm-project/vllm-ascend/pull/3260) · [vllm-ascend#4572 pcp+mtp](https://github.com/vllm-project/vllm-ascend/pull/4572) · [vllm-ascend#5672 dcp+mlapo](https://github.com/vllm-project/vllm-ascend/pull/5672) · [vllm-ascend#6563 dcp+sfa](https://github.com/vllm-project/vllm-ascend/pull/6563) · [Zhihu reference article](https://zhuanlan.zhihu.com/p/2020086868914499979) |

---

## 1. Overview

### 1.1. Overview

This proposal adds simulation support for **Decode Context Parallelism (DCP)** in MsModeling.

The essence of DCP is: **slice the KV cache along the sequence dimension, so that KV cache which was originally redundantly replicated within the TP group is instead stored in shards**, thereby:

- Per-device KV occupancy per sequence is reduced to `1 / dcp_size`, so the same number of blocks can hold sequences `dcp` times longer, supporting larger batches / longer contexts;
- A single decode first performs **all-gather Q** within the DCP group (along the head dimension), so each device holds `h_q · dcp / tp` heads, then runs attention using the local `S/dcp` KV; local outputs and `lse` are reorganized into the complete output via **all-to-all + online-softmax merge**. MLA and GQA share the same communication pattern (2 collective communications per layer); the difference lies only in head partitioning semantics;
- Because it reuses the TP communication domain, **no new devices are introduced**; only sequence sharding semantics are layered on top of the TP partitioning dimension.

### 1.2. Goals

1. Add a `dcp_size` dimension to `ParallelConfig` / `UserInputConfig`, and correctly model the **memory changes** and **communication/computation changes** caused by DCP in the `ModelRunner` simulation pipeline;
2. Add a **DCP dimension** to the search space of `throughput_optimizer`, enabling Decode optimization to automatically search for the optimal `(tp, dcp, ep, moe_dp, batch)` combination;
3. Use a tensor-shape rewrite checklist to clearly identify the shapes that need to be modified in the existing sharding/computation logic for four categories of ops — KV cache / attention / communication primitives / metadata buffer — so that the analytic performance model can be implemented as a graph, with FLOPs / memory / communication volume automatically derived by the existing estimator.

### 1.3. Non-goals (deferred to future RFCs)

1. **No modeling of Prefill Context Parallel (PCP)**: PCP involves head-tail sequence partitioning, an independent communication group that expands `world_size`, three `AllGatherKV` / `AllGatherQ` / `Ring` chunked-prefill fallbacks, `reorg_kvcache` reordering, and other additional semantics — to be handled in a separate RFC;
2. **No modeling of other CP routes**: Ulysses (sequence ↔ head dimension a2a swap) and Ring Attention (KV P2P rolling along a ring) are not in the scope of this simulation. This RFC's simulation **only aligns with the vllm-ascend DCP implementation**;
3. **No modeling of KV inter-instance transfer**: In PD-disaggregated / KV pooling scenarios, the block-interleave KV migration overhead caused by `cp_kv_cache_interleave_size = block_size` is not counted in TPOT. The simulation only verifies configuration legality and does not model P→D transfer time;
4. **No modeling of DCP interaction with sparse attention (SFA)**: The SFA backend of DeepSeek-V3.2 replaces `S` with `top_k`, and the attention FLOPs formula needs a sparse/dense pair, to be supplemented in a future RFC;
5. **No modeling of MLAPO fused op's internal partitioning under DCP**: MLAPO (the RMSNorm + QKV proj + RoPE + KV-write fused op) writes to the local KV shard by `dcp_rank` when DCP is enabled. In this phase, we treat it as an ordinary fused op partitioned by `tp` and do not refine the dcp sub-partitioning inside the fused op;
6. **No modeling of launch-overhead consolidation during ACLGraph capture**: In graph mode, kernel launch times are consolidated, but the analytic model still sums per-kernel — in the worst case, overestimating launch overhead;
7. **No modifications to vllm-ascend upstream code**: Simulation behavior aligns with the implementation in merged vllm-ascend PRs #3260 / #4572 / #5672 / #6563, with no new ops or communication primitives introduced.

This phase **only models the Decode path**; Prefill Context Parallel (PCP) is left to a follow-up RFC.

## 2. Design

### 2.1. Recommended Approach

#### 2.1.1. MsModeling Simulation Scope Choice

MsModeling's goal is to do performance modeling for the Ascend inference engine, with **simulation scope fully aligned with vllm-ascend's** DCP implementation:

- The KV cache partitioning granularity field `cp_kv_cache_interleave_size` only affects the distribution order of tokens on devices, not the computation volume, so it is ignored;
- The DCP communication domain reuses TP, so `world_size = tp_size * pp_size * dp_size`, with **no additional world size expansion**;
- The DCP dimension is further subdivided within the TP group, subject to the constraints;
- The DCP communication path is consistent with the vllm-ascend implementation: both MLA and GQA go through `all_gather Q` + `all_to_all_single(output ⊕ lse)`, with the final merge performed locally via online-softmax.

#### 2.1.2. Configuration Extensions

##### 2.1.2.1 `serving_cast/config.py` — `ParallelConfig`

```python
@dataclass
class ParallelConfig:
    world_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    dcp_size: int = 1                    # New: Decode Context Parallel size, reuses TP devices
    # ... other fields remain unchanged
```

##### 2.1.2.2 `tensor_cast/core/user_config.py` — `UserInputConfig`

```python
@dataclass
class UserInputConfig:
    # ... existing ...
    dcp_size: int = 1
```

And pass it through in `get_parallel_config()`:

```python
return ParallelConfig(
    ...,
    decode_context_parallel_size=self.dcp_size,
)
```

#### 2.1.3. Constraints

Reuse the constraints from vllm-ascend:

  $$
  \texttt{tp\_size} \;\geq\; \texttt{dcp\_size}, \qquad
  \texttt{tp\_size} \bmod \texttt{dcp\_size} = 0
  $$

In addition, under the **GQA backend**, each dcp rank holds `h_kv · dcp / tp` KV heads, which must be ≥ 1 — otherwise the rank has no KV head to read and the configuration is illegal:

  $$
  \texttt{h\_kv} \;\geq\; \dfrac{\texttt{tp\_size}}{\texttt{dcp\_size}}
  $$

(The MLA backend does not partition KV along TP, so this constraint applies only to the GQA path.)

Validation timing for the constraints above:

- `tp_size ≥ dcp_size` and `tp_size mod dcp_size = 0`: model-independent, can be performed when `ParallelRunner._get_user_config()` generates the configuration;
- `h_kv ≥ tp_size / dcp_size`: depends on the concrete model's `num_key_value_heads`, must be completed after loading `ModelConfig` and before the attention backend is first instantiated; recommended to validate inside `_get_user_config()` once the model config is available and before constructing `ParallelConfig`, to avoid deferring the error to the backend internals.

Any combination violating these constraints raises an error directly.

#### 2.1.4. Tensor Shape Rewrite Checklist for Simulation

MsModeling simulation is essentially **"using op-level shapes to drive the analytic / profiling performance model"** — as long as the input/output shapes of each op on the decode path are correctly modified according to DCP partitioning semantics, FLOPs, memory, and communication volume will be automatically computed by the existing estimator. Therefore, this section lists **all shape changes required after enabling DCP** as an implementation checklist.

**Symbol convention** (also used in 2.1.5 below):

| Symbol | Meaning |
| :--- | :--- |
| `B` | decode batch size (including MTP speculative tokens) |
| `S` | context length per request (i.e., number of tokens already in the KV cache) |
| `Q` | query length per decode step (ordinary decode `Q = 1`, MTP-x: `Q = 1 + num_spec_tokens`) |
| `L` | number of model layers |
| `h_q` | total Q heads per layer |
| `h_kv` | total KV heads per layer (MLA: 1) |
| `D` | dimension per head (MLA: `kv_lora_rank + qk_rope_head_dim`) |
| `h_hidden` | hidden dimension (= `h_q · D_q`) |
| `tp`, `dcp` | TP / DCP size |
| `dtype_bytes` | bytes per element (fp16/bf16=2, fp32=4, kv-int8=1) |
| `N_d`, `N_p` | number of decode / prefill requests in this step |
| `T_max` | max token count during ACLGraph capture |

Shape rewrites fall into four categories: **A. KV cache and paged storage**, **B. Q/K/V/out/lse of the attention op**, **C. collective communication ops within the DCP group**, **D. metadata buffer**. Each category lists changes in the "original → new" format below.

##### 2.1.4.1 KV cache and paged storage (Category A)

The core savings of DCP is "**per-sequence KV occupancy on the current device is reduced to `1/dcp`**" (each dcp rank holds a segment of the seq), but **the implementation does not physically shrink the per-rank KV tensor** — `num_blocks_per_rank` is determined by the available memory budget on that rank, and DCP does not change this budget, so the tensor shape `[2, N_blk, block_size, h_kv/tp, D]` remains unchanged. The "partitioning" happens by **enlarging the logical block_size**: `vllm_ascend/attention/context_parallel/mla_cp.py:75-79` multiplies the scheduler-view `block_size` by `cp_virtual_block_size = cp_kv_cache_interleave_size · dcp · pcp`, with tokens of each sequence interleaved onto dcp ranks at `cp_kv_cache_interleave_size` granularity — equivalent effect: **the same `num_blocks` on this device can now hold sequences `dcp` times longer (or `dcp` times the batch)**.

| Tensor | Before DCP | After DCP | Notes |
| :--- | :--- | :--- | :--- |
| `block_size` | `block_size` | Physical `block_size` unchanged; logical `block_size · dcp` from the scheduler's view (absorbs sequences `dcp×` longer) | In simulation, `per_seq_kv_bytes_per_rank ÷= dcp`; the physical tensor dimensions are unchanged (consistent with the `kv_cache` physical shape rows below) |
| `kv_cache` (GQA) | `[2, N_blk, block_size, h_kv/tp, D]` | **Physical shape unchanged**; but each seq's block occupancy on the local device is `≈ 1/dcp` (interleaved within blocks) | Simulation scope: `per_seq_kv_bytes_per_rank` is divided by `dcp`, `N_blk` unchanged |
| `kv_cache` (MLA) | `[N_blk, block_size, kv_lora_rank + qk_rope_head_dim]` | **Physical shape unchanged**; same as above | latent KV stored separately |
| `block_table` | `[N_d + N_p, max_blk_per_seq]` | **Unchanged for DCP-only**; with DCP + spec_decode, flattened to `[N_d · decode_threshold + N_p, max_blk_per_seq]` | The flattened path requires accounting for this space |
| `slot_mapping` | `[total_num_tokens]` int32 | **Unchanged for DCP-only** (PCP introduces `pcp_padded_slot_mapping` extending to `[total_num_tokens · pcp]`, see `vllm_ascend/worker/pcp_utils.py:470-482`) | DCP's slot routing is done via intra-block interleaving and does not change `slot_mapping` length |

##### 2.1.4.2 Q/K/V/out/lse of the attention op (Category B)

Both MLA and GQA decode follow the same "all_gather Q → local attention → all_to_all_single(output⊕lse) → merge" template, so the shape rewrites inside the attention op are isomorphic. The two backends differ only in KV head / latent dimension partitioning.

**MLA backend** (`h_kv = 1`, `D = kv_lora_rank + qk_rope_head_dim`; TP partitions Q heads, latent is not partitioned):

| Tensor | Before DCP | After DCP | Description |
| :--- | :--- | :--- | :--- |
| `Q_local` (input reorg_decode_q) | `[B, Q, h_q/tp, D]` | `[B, Q, h_q/tp, D]` → **after all_gather Q** `[B, Q, h_q · dcp / tp, D]` | `mla_cp.py::reorg_decode_q` all_gathers along the head dimension |
| `K_local` / `V_local` (paged cache fetch on local device) | `[B, S, D]` | `[B, S/dcp, D]` | latent KV partitioned along seq, local device reads only its local shard |
| `attn_output_local` | `[B, Q, h_q/tp, v_head_dim]` | `[B, Q, h_q · dcp / tp, v_head_dim]` (partial, not yet merged) | Entering attention, `num_heads = self.num_heads * dcp_size`, see `mla_cp.py:625` |
| `attn_lse_local` ⭐**New** | — | `[B, Q, h_q · dcp / tp]` fp32 | Required by DCP merge; op call with `softmax_lse_flag=True` |

**GQA backend** (TP partitions Q heads; DCP shares the domain with TP):

| Tensor | Before DCP | After DCP | Description |
| :--- | :--- | :--- | :--- |
| `Q_local` | `[B, Q, h_q/tp, D]` | `[B, Q, h_q/tp, D]` → **after all_gather Q** `[B, Q, h_q · dcp / tp, D]` | `attention_cp.py::_forward_decode_pcp_dcp` all_gathers along the head dimension |
| `K_local` / `V_local` | `[B, S, h_kv/(tp), D]` (duplicated) | `[B, S/dcp, h_kv/(tp/dcp), D]` | Partitioned along seq, **eliminates KV duplication within the TP group** |
| `attn_output_local` | `[B, Q, h_q/tp, D]` | `[B, Q, h_q · dcp / tp, D]` (partial, not yet merged) | Same as above |
| `attn_lse_local` ⭐**New** | — | `[B, Q, h_q · dcp / tp]` fp32 | Same as MLA |

##### 2.1.4.3 Collective communication ops within the DCP group (Category C)

The MLA and GQA backends use **the same pair** of DCP collective communication primitives, corresponding to `dist.all_gather` (along head dimension) + `dist.all_to_all_single` (splitting `dcp` shards along the head dimension) in `vllm_ascend/attention/context_parallel/{mla_cp.py, attention_cp.py, common_cp.py}`.

| Backend | Communication Primitives | Collective Communications per Layer |
| :--- | :--- | :--- |
| MLA | `all_gather Q` (head dim) + `all_to_all_single(output ⊕ lse)` (head dim) + local online-softmax merge | **2** |
| GQA | `all_gather Q` (head dim) + `all_to_all_single(output ⊕ lse)` (head dim) + local online-softmax merge | **2** |

> Note (**implementation point**): vllm-ascend upcasts both `attn_output` and `softmax_lse` to **fp32** inside `_process_attn_out_lse` before doing a2a. Therefore the communication volume of `all_to_all_single(output ⊕ lse)` **must** be computed with **`bytes_per_element = 4` (fp32, fixed)**, and **must not** reuse the model weight / KV `dtype_bytes` (fp16/bf16 = 2, kv-int8 = 1) — otherwise the a2a byte count will be significantly underestimated on low-bit models. `all_gather Q` keeps the original dtype and is not affected — i.e., the two collective communications within the same layer must be passed different `bytes_per_element` values into the estimator.

##### 2.1.4.4 Metadata buffer (Category D)

On the DCP-only path, the only truly **newly added** metadata buffer is `attn_lse_local` (the op carries lse output, required by DCP merge).

| Buffer | shape | dtype | Purpose |
| :--- | :--- | :--- | :--- |
| `attn_lse_local` ⭐ | `[B · Q, h_q · dcp / tp]` | fp32 | Op-carried lse output, input to online-softmax merge |

Magnitude is at KB–MB level, with little impact on memory budget but needs to be counted to avoid double-counting with the attention workspace.

##### 2.1.5 Overview of Shape Rewrite Locations

All shape changes in 2.1.4.1-2.1.4.4 are grouped by code file, providing a roadmap for implementation:

| Rewrite Location | Files Involved | Shapes Involved |
| :--- | :--- | :--- |
| KV cache block_size allocation | `tensor_cast/core/input_generator.py` | Category A KV cache per-device block_size |
| KV cache allocation | `tensor_cast/core/model_runner.py` | Category A KV cache tensors |
| KV cache slot indexing | `tensor_cast/ops/mla.py`, `tensor_cast/layers/attention.py` | Category A `slot_mapping` |
| Attention kernel input/output | `tensor_cast/layers/mla.py`, `tensor_cast/layers/attention.py` | Category B Q/K/V/out/lse |
| DCP communication primitives | `tensor_cast/performance_model/comm_analytic.py` (reuse existing estimator) | Category C `all_gather` (MLA / GQA's Q, in the model's original dtype) + `all_to_all_single` (MLA / GQA's output⊕lse, **fixed `bytes_per_element = 4` (fp32), do not reuse `dtype_bytes`**, see §2.1.4.3) |
| Metadata buffer | `tensor_cast/core/model_runner.py`, `serving_cast/kv_cache_manager.py` | Category D |
| block_table flattening | `serving_cast/kv_cache_manager.py` | Category A `block_table` |

> DCP reuses the TP communication domain, so the ranks of the DCP group form a contiguous slice within the TP group, and `comm_grid` parsing does not need to add a new topology.

### 2.2. Alternatives

| Alternative | Description | Reason for Rejection |
| :--- | :--- | :--- |
| **A. Model Ulysses or Ring Attention** | Ulysses reuses the existing ulysses SP abstraction | Inconsistent with vllm-ascend deployment |
| **B. Model PCP + DCP together** | Complete prefill / decode in one shot | Large change surface; prefill involves `head-tail` chunk partitioning and three `chunked prefill` fallback strategies, with higher modeling difficulty — should be a separate project |

## 3. Implementation Plan

### 3.1. Milestones

| Phase | Task | Deliverable |
| :--- | :--- | :--- |
| M1 | `ParallelConfig` / `UserInputConfig` field extensions + validation | PR1: Configuration layer supports `dcp_size` |
| M2 | `ModelRunner` KV memory and attention computation corrections | PR2: DCP memory / computation modeling |
| M3 | Inject DCP communication hooks into `tensor_cast/layers/mla.py`, `attention.py` | PR3: DCP communication modeling |
| M4 | `ParallelRunner` search space extension + CLI argument | PR4: Optimization integration |
| M5 | `optimizer_summary` column extension + output format update | PR5: Output display |
| M6 | Unit tests + end-to-end vllm-ascend reference baseline | PR6: Tests and accuracy alignment |

### 3.2. Test Plan

| Test Item | Description | Pass Criteria |
| :--- | :--- | :--- |
| Unit test: constraint validation | Illegal `(tp, dcp)` combinations under MLA / GQA scenarios are pruned | 100% coverage |
| Unit test: KV memory formula | As `dcp_size` increases from 1 to 8, `kv_cache_size_gb` monotonically decreases to `1/dcp` | Numerical error ≤ 0.5% |
| Unit test: communication volume formula | `V_DCP` and communication volume formulas of MLA and GQA backends are fully consistent | Exact match |
| Integration test: TPOT convergence | With fixed batch / input, sweep `dcp ∈ {1,2,4,8}`, TPOT monotonically decreases until the communication term dominates | Curve shape matches expectations |
| Accuracy comparison: vllm-ascend | Take two real-world measurements: `DeepSeek-R1-W8A8 + DCP8`, `Qwen3-235B + DCP2` | Simulation TPOT error ≤ 15% vs. real measurements |
| Compatibility test: with PD ratio optimization | `--enable-optimize-prefill-decode-ratio` simultaneously with `--dcp-sizes` | Prefill forced `dcp=1`, Decode iterates normally |

### 3.3. Follow-up Work

- **PCP simulation**: A separate RFC modeling head-tail KV partitioning, three chunked-prefill strategies (`AllGatherKV` / `AllGatherQ` / `Ring`), and the reordering overhead of `reorg_kvcache` (refer to [vllm-ascend#3260](https://github.com/vllm-project/vllm-ascend/pull/3260), [vllm-ascend#4572](https://github.com/vllm-project/vllm-ascend/pull/4572));
- **MLAPO + DCP fused op modeling**: MLAPO fuses RMSNorm + QKV proj + RoPE + KV-write into a single op; when DCP is enabled, the fused op writes to the local KV shard by `dcp_rank`. Need to confirm that the memory/computation size of the `mlapo` op in `tensor_cast/ops/mla.py` scales by `dcp` (refer to [vllm-ascend#5672](https://github.com/vllm-project/vllm-ascend/pull/5672));
- **SFA (Sparse Attention) + DCP modeling**: DeepSeek-V3.2 uses the SFA backend, where attention computation is determined by `top_k` rather than `S`. The DCP communication formula still applies but requires a sparse/dense pair (refer to [vllm-ascend#6563](https://github.com/vllm-project/vllm-ascend/pull/6563));
- **Joint modeling of KV transfer + DCP**: In PD-disaggregated scenarios, model the cross-instance block-interleave KV transfer overhead when `cp_kv_cache_interleave_size = block_size`, and the migration paths of Mooncake / Layerwise connectors;
- **DCP interaction with KV quantization**: When `quantize_attention_action` is enabled, `dtype_bytes` changes; need to re-verify the KV memory formula;
- **DCP launch overhead under graph mode (ACLGraph)**: The current analytic model accumulates kernel time, while the consolidation optimization in graph mode is not yet modeled — can be supplemented in profiling mode. Theoretically, changes in analytic and profiling modes do not affect the cp parallel logic, but testing is needed.
- **Real-world comparison with PR #3260 + #4572**: Add measured TPOT with ACLGraph + MTP + DCP triple-enabled to the 3.2 test baseline, verifying simulation error ≤ 15%.

### 3.4. List of File Changes

| File | Description |
| :--- | :--- |
| `tensor_cast/core/input_generator.py` | `_get_kv_cache_info` adds scaling of `block_size` against `dcp_size` |
| `serving_cast/config.py` | `ParallelConfig` adds `dcp_size`, `cp_kv_cache_interleave_size` |
| `tensor_cast/core/user_config.py` | `UserInputConfig` adds new field and passes through to `ParallelConfig` |
| `tensor_cast/model_config.py` | `ParallelConfig` adds `decode_context_parallel_size` |
| `tensor_cast/core/model_runner.py` | `kv_cache_size_gb` formula scaled by `dcp_size` |
| `tensor_cast/layers/mla.py` | MLA decode path injects `all_gather Q` + `all_to_all_single(output ⊕ lse)`, KV fetch sliced by dcp_rank; attention estimator's `num_heads` / `seq_len` changed to `h_q · dcp / tp` and `S / dcp` |
| `tensor_cast/layers/attention.py` | GQA decode path injects `all_gather Q` + `all_to_all_single(output ⊕ lse)` (same template as MLA) |
| `tensor_cast/performance_model/comm_analytic.py` | Reuse existing primitives, add DCP group resolution (rank list is a subset of the TP group) |
| `serving_cast/parallel_runner.py` | `_get_user_config()` extends DCP search dimension, distinguishes P/D phase |
| `cli/inference/throughput_optimizer.py` | New `--dcp-sizes` |
| `serving_cast/service/optimizer_summary.py` | Output column appends `dcp`, `format_parallel_label` adds `dcp{n}` |
| `serving_cast/service/utils.py` | `DISAGG_COLUMNS` adds `dcp`; align callers of `resolve_search_sizes` |
| `tests/test_serving_cast/test_dcp_simulation.py` | New unit tests for DCP simulation |
| `tests/test_serving_cast/test_parallel_runner_dcp.py` | New unit tests for optimization search |

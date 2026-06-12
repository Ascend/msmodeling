# Single Operator Mapping — Sub-Agent Prompt

## Task

Map ONE operator between TensorCast (TC) virtual runtime and NPU profiling kernel types. Produce a YAML snippet for op_mapping.yaml following the template in `op-mapping-template.yaml`.

## Inputs

You will be given:

- `op_name`: TC op (e.g., `aten.mm.default`) OR profiling Type (e.g., `MatMulV2`)
- `direction`: `forward` (TC → NPU) or `reverse` (NPU → TC)
- `profiling_csv_path` (optional): kernel_details.csv to cross-reference
- `repo_urls`: dict of repo URLs with version tags to clone/webfetch
- `local_repo_paths` (optional): dict of local checkout paths to search directly

Before tracing, confirm every local checkout is at the requested version/tag/branch
and record `git log -1 --oneline --decorate` in your notes. Do not rely on `main`
branch paths when the target stack specifies a release tag.

## The 5-Layer Pipeline Model

Understanding the full call path from Python to device kernel is essential. Every NPU operator traverses these layers:

```text
Layer 0: vLLM Python — model.forward() dispatches ops via 3 paths
    ↓
Layer 1: op-plugin — C++ dispatch, routes aten/npu ops to CANN aclnn APIs
    ↓
Layer 2: CANN aclnn Host API — op_host/op_api/aclnn_*.cpp defines the C interface
    ↓
Layer 3: L0 Op Registration — OP_TYPE_REGISTER() or CMakeLists OPTYPE declares kernel name
    ↓
Layer 4: AI Core Execution — Profiling captures: Name=aclnnXxx_Yyy_OpType, Type=OpType
```

**Critical insight — Profiling Name 3-part structure:**

```text
aclnnMatmulWeightNz  _  MatMulCommon  _  MatMulV2
│                       │                │
├─ 1st: aclnn API       ├─ 2nd: Host     ├─ 3rd: L0 OpType
│  (op-plugin calls)    │  dispatch fn    │  (= Profiling Type column)
│                       │                 │  (= CSV filename for DB query)
```

Single-segment names (e.g., `FusedInferAttentionScore`, `GroupedMatmul`) indicate graph-compiled mode — the L0 OpType directly. Triton kernels use the Python function name directly (e.g., `split_qkv_rmsnorm_rope_kernel`).

### Handling CANN Version Differences

Kernel types can change between CANN versions — renames, fusions, or removals. Always verify the actual `Type` column in `kernel_details.csv` for the target CANN version rather than assuming a fixed name.

**When tracing the call chain:**

- If the L0 OPTYPE you find in CANN source doesn't match what appears in profiling, the CANN version may have renamed it. Check the profiling data as ground truth.
- Use `alternate_kernel_types` to list both old and new kernel names for cross-version compatibility.
- Some ops may be fused into larger kernels in newer CANN versions (e.g., separate matmul + activation → single fused kernel). These require updating `kernel_type`, not just adding alternates.

Prefer the CANN/op-plugin branch that matches the target CANN/torch-npu stack.
Use the user-provided version or profiling metadata to select the branch/tag,
and verify the resolved ref. For example, CANN 8.5 transformer kernels may use
`https://gitcode.com/cann/ops-transformer.git` branch `8.5.0`, but that is a
version-specific example rather than a global default.

If a TC op can take both a public `torch_npu.npu_*` path and a vLLM-Ascend custom
`torch.ops._C_ascend.*` path, choose the branch actually selected by the
version-pinned vLLM-Ascend Python code as `kernel_type`, and put the other
same-abstraction kernel in `alternate_kernel_types`.

**aclgraph parity:** vllm-ascend aclgraph ensures eager mode and graph mode produce exactly the same ops including fusion passes, so profiling from either mode is valid for op_mapping.

## Three-Path Decision Tree

```text
START: What is the op?
│
├─ aten.* elementwise (add, mul, div, sub)?
│  └─ ELEMENTWISE: use query_mode: elementwise
│     - Match on OUTPUT shape (not input)
│     - No tc_input_count needed (mutually exclusive)
│     - Dtype-relaxed with byte-ratio scaling
│
├─ aten.* op (standard PyTorch)?
│  └─ PATH A: ATen → op-plugin → aclnn → L0 OpType
│
├─ tensor_cast.* with known torch_npu.npu_* equivalent?
│  └─ PATH B: torch_npu pybind → op-plugin → aclnn → L0 OpType
│
├─ Communication op (all_reduce, all_gather, reduce_scatter, all_to_all)?
│  └─ HCCL DIRECT: torch.distributed → ProcessGroupHCCL → hcom_* kernel
│
├─ Triton kernel or vllm-ascend custom csrc/ op?
│  └─ PATH C: vllm-ascend custom → Profiling Type = function name or OP_ADD name
│
└─ Reverse direction (profiling Type → TC op)?
   └─ Parse aclnn prefix from Name → search op-plugin → find aten/npu function → match TC op
```

## Path A: ATen → op-plugin (Standard PyTorch Ops)

**When to use:** op_name starts with `aten.`

**Search sequence:**

1. Search op-plugin YAML for the function signature:

   ```bash
   grep -n "func: <op_name_without_aten_prefix>" $OP_PLUGIN/op_plugin/config/op_plugin_functions.yaml
   ```

2. Find the C++ implementation file:

   ```bash
   find $OP_PLUGIN/op_plugin/ops/opapi/ -iname "*<op_name>*"
   ```

3. Extract aclnn kernel names:

   ```bash
   grep "EXEC_NPU_CMD" $OP_PLUGIN/op_plugin/ops/opapi/<ImplFile>.cpp
   ```

4. Verify in CANN repos — search for the aclnn API:

   ```bash
   find $CANN/ -path "*/op_host/op_api/aclnn_*" -name "*<keyword>*"
   ```

5. Find L0 OPTYPE registration:

   ```bash
   grep -r "OPTYPE" $CANN/ops-*/*/op_host/CMakeLists.txt | grep -i "<keyword>"
   ```

6. Cross-reference with profiling CSV (if available):

   ```bash
   grep "<ProfilingType>" $PROFILING_CSV | head -3
   ```

**Example (aten.mm.default):**

- YAML: `func: mm(Tensor self, Tensor mat2) -> Tensor`
- Impl: `MmKernelNpuOpApi.cpp` → `EXEC_NPU_CMD(aclnnMm, ...)` and `EXEC_NPU_CMD(aclnnMatmulWeightNz, ...)`
- CANN: `ops-nn/matmul/mat_mul_v3/` → OPTYPE: `mat_mul_v3`, OP_TYPE_REGISTER(MatMulV2)
- Profiling: Type=`MatMulV2`

## Path B: torch_npu.npu_* → op-plugin (NPU Custom Ops)

**When to use:** TC op maps to a torch_npu custom function (fused kernels like SwiGlu, attention, RoPE, etc.)

**Search sequence:**

1. Search vllm-ascend for the torch_npu API call:

   ```bash
   grep -rn "torch_npu\." $VLLM_ASCEND/vllm_ascend/ --include="*.py" | grep "<keyword>"
   ```

2. Find op-plugin YAML entry for the npu_* function:

   ```bash
   grep -n "func: npu_<function_name>" $OP_PLUGIN/op_plugin/config/op_plugin_functions.yaml
   ```

3. Find C++ implementation and extract aclnn:

   ```bash
   grep -r "EXEC_NPU_CMD" $OP_PLUGIN/op_plugin/ops/opapi/ | grep -i "<function_name>"
   ```

4. Verify in CANN repos (same as Path A steps 4-6).

**Example (tensor_cast.swiglu.default):**

- vllm-ascend: `vllm_ascend/ops/fused_moe/moe_mlp.py` → `torch_npu.npu_swiglu()`
- YAML: `func: npu_swiglu(Tensor self, int dim=-1) -> Tensor`
- CANN: `ops-transformer/ffn/swiglu/` or `ops-nn/quant/swi_glu_quant/`
- Profiling: Type=`SwiGlu`

**Example (GLM sparse indexer / tensor_cast.dsa_indexer.default):**

- vLLM-Ascend v0.18.0: `vllm_ascend/attention/sfa_v1.py` sets
  `use_torch_npu_lightning_indexer=True` for `hf_config.model_type == "glm_moe_dsa"`.
- Selected path: `torch_npu.npu_lightning_indexer()` -> op-plugin
  `op_plugin_functions.yaml` -> `LightningIndexerKernelNpuOpApi.cpp`
  -> `aclnnLightningIndexer`.
- CANN: ops-transformer branch matching the target CANN version, e.g. 8.5.0
  for CANN 8.5, `attention/lightning_indexer/`,
  `OP_NAME LightningIndexer`, `OP_ADD(LightningIndexer)`.
- Independent vLLM-Ascend custom path:
  `torch.ops._C_ascend.npu_lightning_indexer` ->
  `csrc/lightning_indexer_vllm/lightning_indexer_vllm_torch_adpt.h`
  -> `aclnnLightningIndexerVllm` / `OP_ADD(LightningIndexerVllm)`.
  Treat `LightningIndexer` and `LightningIndexerVllm` as separate kernel types
  with separate `op_mapping.yaml` entries, not as `alternate_kernel_types` on the
  same entry.
- If `LightningIndexer.csv` is missing after adding the mapping, classify the result as
  `csv_not_found` data gap, not a mapping failure.

## Path C: vllm-ascend Custom / Triton

**When to use:** Op is NOT in op-plugin; it's a vllm-ascend C++ custom op or Triton kernel.

**Search sequence:**

1. Search vllm-ascend csrc/ for C++ custom ops:

   ```bash
   grep -rn "OP_ADD\|REGISTER" $VLLM_ASCEND/csrc/ --include="*.cpp" | grep "<keyword>"
   ```

2. Search vllm-ascend Triton kernels:

   ```bash
   find $VLLM_ASCEND/vllm_ascend/ops/triton/ -name "*.py" -exec grep -l "<keyword>" {} \;
   ```

3. Search vllm-ascend graph fusion passes:

   ```bash
   grep -rn "FusionPass\|fusion_pass" $VLLM_ASCEND/vllm_ascend/ --include="*.py"
   ```

4. For Triton kernels: Profiling Type = the `@triton.jit` function name
5. For csrc/ ops: check op_host/*_def.cpp for OP_ADD registration name
6. Verify the csrc path in the target tag; custom op directories often move between
   release tags and `main`.

**Example (Triton kernel):**

- vllm-ascend: `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py`
- QKNormRopeFusionPass in `vllm_ascend/compilation/passes/`
- Profiling: Type=`split_qkv_rmsnorm_rope_kernel`
- Note: Triton kernels may be replaced by native CANN fusions in newer versions — always verify against profiling data

## HCCL Communication Ops

**When to use:** op_name contains all_reduce, all_gather, reduce_scatter, all_to_all

Communication ops bypass op-plugin entirely:

- `torch.distributed.all_reduce()` → ProcessGroupHCCL → HCCL library → `hcom_allReduce_`
- Category must be set to `communication` in op_mapping.yaml
- Query uses message_bytes + num_devices, NOT shape matching

## Reverse Mapping (Profiling Type → TC Op)

**When to use:** direction=reverse, given a profiling Type that needs a TC op mapping

1. From profiling CSV, get example Name column entries for this Type:

   ```bash
   grep ",<Type>," $PROFILING_CSV | head -5
   ```

2. Parse the 3-part Name to extract the aclnn prefix (1st segment)
3. Search op-plugin for the aclnn name:

   ```bash
   grep -r "<aclnn_prefix>" $OP_PLUGIN/op_plugin/ops/ --include="*.cpp" -l
   ```

4. From the impl file, identify the function signature → match to aten op or npu_* API
5. Search TC ops for matching functionality:

   ```bash
   grep -rn "def <keyword>" $MSMODELING/tensor_cast/ops/ --include="*.py"
   ```

6. If no TC equivalent exists → create a `profiling.<Type>` placeholder entry

## 10 Shape Differences Checklist

After determining the mapping, flag which shape transforms apply to this op. The profiling_data_source.py `_inputs_match()` method handles these automatically, but workers must document them for correctness verification. Full details: `ref/shape_matching_catalog.md`.

- [ ] **Batch dim strip**: TC keeps `(1, S, D)`, NPU drops to `(S, D)`. Applies to most 3D→2D ops.
- [ ] **Seq padding**: TC pads sequence to block alignment (16/32/64). NPU records raw length. Applies to matmul, norm ops.
- [ ] **FRACTAL_NZ format**: NPU stores weights as `[H, W, bh, bw]` tiled layout. Restore via `H*bw, W*bh`. Check Input Formats column for `FRACTAL_NZ`.
- [ ] **ND weight transpose**: TC has `(K, N)` from `weight.T`, NPU may store `(N, K)`. Applies to MatMulV2 2nd input only.
- [ ] **SwiGlu input concat**: TC dispatches 2 inputs `(S, D/2)`, NPU expects 1 input `(S, D)`. Applies to SwiGlu kernel.
- [ ] **RoPE layout transpose**: TC `(B,H,S,D)` with `[Q,K,cos,sin]` order. NPU `(B,S,H,D)` with `[K,Q,cos,sin]`. Applies to InterleaveRope, ApplyRotaryPosEmb.
- [ ] **RoPE alternate kernels**: TC has one `apply_rope` op, NPU has InterleaveRope (interleave mode) and ApplyRotaryPosEmb (neox mode). Use `alternate_kernel_types`. Note: Some CANN versions may replace Triton-based RoPE kernels with native CANN fusions — verify against profiling data.
- [ ] **Composite decomposition**: TC fused op (e.g., matmul_all_reduce) maps to separate NPU kernels. Use `composite: true` + `sub_kernels`.
- [ ] **Flatten batch**: TC `(B, M, D)` → NPU `(B*M, D)`. Applies to quantize/norm/DFC kernels. Kernel must be in `_FLATTEN_BATCH_KERNELS`.
- [ ] **Merge last dims**: TC `(T, H, D)` → NPU `(T, H*D)`. Applies to MLA quantize (per-head → hidden_dim). Kernel must be in `_MERGE_LAST_DIMS_KERNELS`.

## tc_input_count Decision Guide

When TC tensor input count differs from CSV, decide whether to add `tc_input_count`. Full rules: `ref/tc_input_count_rules.md`.

**SAFE — CSV extras are NPU-internal fixed params:**

```yaml
# Scatter: CSV has (self, index, axis); TC has (self, index)
tc_input_count: 2   # truncate axis — it's a CANN-internal param
```

**UNSAFE — CSV extras are variable broadcast operands:**

```yaml
# Add: CSV has 1-input (scalar broadcast) AND 2-input (element-wise) rows
# tc_input_count: 1 would match scalar-broadcast row → 30% latency underestimate
# → Do NOT set tc_input_count. Let it MISS. Analytic fallback is safer.
```

**Decision rule**: Is the CSV extra input a **fixed NPU param** (axis, scale)? → Safe. Is it a **variable operand** (broadcast pattern)? → Unsafe.

## zero_cost Classification Guide

Some ops should be `zero_cost: true` instead of `kernel_type`. Full rules: `ref/zero_cost_classification.md`.

**Mark zero_cost when:**

1. The kernel Type **never appears** in profiling data, AND
2. The op's latency is **already captured** by a fused kernel (e.g., MoeGatingTopK absorbs topk+sum+sigmoid+where)

**Do NOT mark zero_cost when:**

- The kernel Type exists in profiling but CSV data hasn't been collected yet (= data gap, not fusion)
- The op has independent latency not captured by any fusion (= needs its own kernel_type)

## General Search Hints

**Where operators live in each repo:**

- **op-plugin**: `op_plugin/config/op_plugin_functions.yaml` (master YAML), `op_plugin/ops/opapi/` (C++ implementations)
- **vllm-ascend**: `vllm_ascend/ops/` (Python APIs), `csrc/` (C++ custom ops), `vllm_ascend/ops/triton/` (Triton kernels), `vllm_ascend/compilation/passes/` (graph fusion)
- **CANN ops-transformer**: `attention/`, `mc2/`, `moe/`, `gmm/`, `posembedding/`, `ffn/` — most LLM fusion kernels
- **CANN ops-nn**: `matmul/`, `norm/`, `quant/`, `index/`, `activation/` — basic building blocks
- **CANN ops-math**: basic element-wise ops (`add/`, `mul/`, `cast/`, `div/`, `reduce_mean/`)
- **CANN ascend-transformer-boost**: `src/ops/ops_infer/` — ATB high-level fused ops (reshape_and_cache, paged_attention, etc.)
- **CANN directory pattern**: each op has `op_host/CMakeLists.txt` with `OPTYPE` and `op_host/op_api/aclnn_*.cpp` for the aclnn C API

## Evidence Chain Format

Your notes field MUST include:

```text
[HIGH|MEDIUM|LOW] path: A|B|C|HCCL.
op-plugin: <yaml_entry_name> / <ImplFile.cpp> (or N/A for Path C / HCCL).
aclnn: <aclnn_kernel_name> (or N/A for Triton/HCCL).
CANN: <ops-xxx/category/opname/> OPTYPE: <value> (or N/A for Triton).
Profiling: <Type(count)> (or N/A if no profiling data).
shape_flags: [<applicable flags from checklist>].
```

Confidence levels:

- **HIGH**: Verified against profiling data (Type matches, count > 0)
- **MEDIUM**: op-plugin evidence exists but no profiling verification
- **LOW**: Placeholder or uncertain mapping (e.g., untested quant path)

## Output

Produce exactly ONE YAML snippet following `op-mapping-template.yaml` format. Include:

1. The operator_mappings entry
2. The torch_npu_reference entry (if applicable, not for zero_cost or profiling-only)
3. A summary line: `RESULT: <op_name> → <kernel_type|composite|zero_cost> [<confidence>]`

---
name: microbench-run-script-generator
description: Use when generating or updating tools/perf_data_collection/op_replay/<KernelType>_run.py from perf database CSV data, op_mapping.yaml torch_npu_reference.microbench_api, and upstream operator docs/tests across vLLM, vLLM-ascend, op-plugin, pytorch-npu, CANN, or ATB repos
version: 1.0.0
source: local-session-analysis
---

# Microbench Run Script Generator

Generate a runnable `tools/perf_data_collection/op_replay/<KernelType>_run.py` for a profiling kernel CSV, so the operator can be replayed on NPU and measured by `run_all_op.py` / `profile_and_update_db.py`.

## When to Use

- User asks to add a new `xxx_run.py` under `tools/perf_data_collection/op_replay`
- A CSV already exists under `tensor_cast/performance_model/profiling_database/data/<device>/vllm_ascend/<version>/`
- `op_mapping.yaml` already has a `torch_npu_reference.<KernelType>.microbench_api` entry
- The operator needs real NPU replay, not just theoretical mapping

## Inputs to Collect

- `kernel_type`: CSV / profiling kernel name, for example `AscendQuantV2`
- `device`: perf database device folder
- `vllm_ascend_version`: perf database version folder
- `csv_path`: target operator CSV
- `op_mapping.yaml` path for the same stack/version if available
- Local repo roots if they already exist

## Repo Search Order

Use local repos first. If the needed repo is missing, clone it next to `msmodeling`.

| Repo | Default sibling path |
|---|---|
| vLLM | `../vllm` |
| vLLM-ascend | `../vllm-ascend` |
| op-plugin | `../op-plugin` |
| pytorch-npu | `../pytorch` |
| CANN ops-nn | `../cann-ops-nn` |
| CANN ops-transformer | `../cann-ops-transformer` |
| CANN ops-math | `../cann-ops-math` |
| CANN ATB | `../ascend-transformer-boost` |

Suggested clone commands:

```bash
git clone https://github.com/vllm-project/vllm ../vllm
git clone https://github.com/vllm-project/vllm-ascend ../vllm-ascend
git clone https://github.com/Ascend/op-plugin ../op-plugin
git clone https://github.com/Ascend/pytorch ../pytorch
git clone https://gitee.com/ascend/cann-ops-nn ../cann-ops-nn
git clone https://gitee.com/ascend/cann-ops-transformer ../cann-ops-transformer
git clone https://gitee.com/ascend/cann-ops-math ../cann-ops-math
git clone https://gitee.com/ascend/ascend-transformer-boost ../ascend-transformer-boost
```

## Workflow

### Step 1: Locate `microbench_api`

Read the matching `op_mapping.yaml` and find:

```yaml
torch_npu_reference:
  <KernelType>:
    microbench_api: "..."
```

This API is the replay target. Do not guess it from the kernel name if the YAML already gives one.

### Step 2: Find the Real Interface

Search the local repos for the API string and operator name.

Priority:

1. Operator-specific markdown docs under `op-plugin/docs/context/`
2. Operator tests under `op-plugin/test/`
3. vLLM-ascend custom-op Python or Triton implementation
4. op-plugin / pytorch-npu C++ registration or op-api entry
5. CANN / ATB sources for cache/layout semantics

Typical search patterns:

```bash
git grep -n "<microbench_api>"
git grep -n "<kernel_type>"
git grep -n "npu_<op_name>"
```

If the operator is a custom `torch.ops.*` or Triton op, treat the implementation file and tests as the source of truth.

### Step 3: Read the CSV Before Coding

Inspect:

- `Input Shapes`
- `Input Data Types`
- `Input Formats`
- `Output Shapes`
- `Output Data Types`
- Number of rows and whether shapes are homogeneous

Important rule: generate the script for the *actual recorded rows first*. Do not over-generalize unless the CSV clearly needs it.

### Step 4: Classify the Operator

Choose one of these replay styles:

- **Direct torch op**: `torch.add`, `torch.Tensor.copy_`
- **torch_npu op**: `torch_npu.npu_dynamic_quant`, `torch_npu.npu_interleave_rope`
- **torch_npu custom / underscored op**: `torch_npu._npu_reshape_and_cache`
- **torch.ops custom op**: `torch.ops.vllm.qkv_rmsnorm_rope`
- **ATB path**: use ATB Python entry if `microbench_api` says so

### Step 5: Infer Missing Non-Tensor Arguments

CSV often omits scalar or optional args. Infer them from docs/tests and current rows.

Common examples:

- `axis=-1`, `div_mode=False`
- `epsilon=1e-5` or `1e-6`
- `cache_mode="PA_BNSD"`
- `is_output_kv=True`
- output quant dtype from `Output Data Types`
- optional tensors set to `None`
- legal `slot_mapping` / `index` tensors built from cache capacity

Do not invent broad heuristics unless the current CSV needs them. Prefer the smallest rule set that runs the recorded rows.

### Step 6: Follow Existing `op_replay` Conventions

Reuse patterns from existing files under `tools/perf_data_collection/op_replay/`:

- import helpers from `common.py`
- build `build_argparser()`
- parse CSV rows with `iter_csv_rows(...)`
- create a `build_row_case(...)` or `run_row(...)`
- call `runtime_torch.npu.synchronize()`
- print concise `[OK]` lines with row metadata

Keep scripts standalone and operator-specific. Do not add cross-operator abstractions unless already present in `common.py`.

### Step 7: Validate Minimally

Always run:

```bash
py -3 -m py_compile tools/perf_data_collection/op_replay/<KernelType>_run.py
py -3 tools/perf_data_collection/op_replay/<KernelType>_run.py --help
```

If an actual NPU replay is available, run one full command against the target CSV version.

### Step 8: Commit Scope

Normally commit only:

- `tools/perf_data_collection/op_replay/<KernelType>_run.py`

Do not accidentally add local data directories or unrelated generated CSVs.

## Implementation Rules

- Prefer existing helper functions in `common.py`
- Use `apply_patch` for file edits
- Match naming style of the current replay scripts
- Preserve ASCII by default
- Print enough metadata to debug row mismatches, but do not dump tensors
- If CSV has empty metadata slots, parse them explicitly instead of collapsing them away

## Operator-Specific Pitfalls

- `TensorMove`: CSV records only the source tensor; destination must be reconstructed
- `ReshapeAndCacheNdKernel`: `slot_mapping` is not random scalar metadata; build a legal in-range tensor
- `KvRmsNormRopeCache`: CSV may contain fixed empty slots for optional tensors; preserve slot count
- `split_qkv_rmsnorm_rope_kernel`: may require importing/registering custom vLLM-ascend ops before use
- `AscendQuantV2` / `DynamicQuant`: output dtype and return arity come from docs plus CSV, not just kernel name

## Done Criteria

- `xxx_run.py` exists under `tools/perf_data_collection/op_replay`
- `--help` works
- `py_compile` passes
- The script replays all current rows in the target CSV with the intended API
- Any inferred non-tensor args are explained in code comments or module docstring when non-obvious

# op_mapping Verifier — Sub-Agent Prompt

## Task

Verify an op_mapping.yaml for completeness and accuracy against TensorCast simulation and (optionally) NPU profiling data.

## Inputs

You will be given:

- `op_mapping_path`: path to op_mapping.yaml to verify
- `msmodeling_dir`: msmodeling project root
- `model`: HuggingFace model ID
- `device`: device profile name
- `tc_config`: dict with world_size, tp_size, dp_size, ep, quantize_linear_action
- `profiling_csv_path` (optional): kernel_details.csv from NPU run
- `python_path`: Python 3.10 executable path

## Verification Steps

### Step 1: Coverage Check

Run TC simulation and extract ops:

```bash
$python_path -m tensor_cast.scripts.text_generate $model \
  --num-queries 2 --query-length 3500 \
  --device $device --world-size $ws --tp-size $tp \
  [--dp-size $dp] [--ep] [--quantize-linear-action $quant] \
  --performance-model profiling --compile \
  --chrome-trace /tmp/verify_trace.json 2>&1 | tee /tmp/verify_run.log
```

Then extract ops:

```bash
$python_path tools/perf_data_collection/extract_tc_ops.py \
  --chrome-trace /tmp/verify_trace.json \
  --op-mapping $op_mapping_path \
  --output /tmp/verify_ops.json
```

Read the JSON output. Check `unmapped_ops` list — should be empty.
Check `mapped_count` vs total — compute coverage percentage.

### Step 2: Hit Rate Check

From the TC run log, count lines containing "MISS" or "ANALYTIC_FALLBACK" vs "HIT" or "PROFILING".
Target: >95% of compute ops should HIT.

Classify MISS reasons before recommending fixes:

- `unmapped`: missing or wrong `op_mapping.yaml` entry. Fix mapping.
- `csv_not_found`: mapping resolved to a kernel_type, but `<kernel_type>.csv` is absent.
  Treat this as a profiling database coverage gap; collect/generate CSV data rather
  than changing the mapping.
- `shape_mismatch`: CSV exists but lacks the TC shape, or the entry needs a shape
  transform/query mode. Inspect shapes before changing kernel_type.
- `elementwise_output_shape_mismatch`: usually a broadcast/output-shape query issue;
  prefer `query_mode: elementwise` or a shape matching fix over `tc_input_count`.

Expected misses (acceptable):

- communication ops (fallback to analytic is by design)
- attention_special ops without matching CSV data
- composite ops where sub-kernel CSV is missing

### Step 3: Op-Level Latency Comparison (requires profiling CSV)

If profiling_csv_path is provided:

1. Parse profiling CSV — group by Type, sum Duration(us) per Type
2. From TC run log or chrome trace — extract per-op latency
3. For each kernel_type present in both:
   - Compare TC total latency vs profiling total latency
   - Compute ratio: tc_latency / profiling_latency
   - Flag if ratio < 0.5 or > 2.0 (>2x discrepancy)

### Step 4: End-to-End Latency Comparison (requires profiling CSV)

Sum all TC op latencies → tc_total_us
Sum all profiling durations → profiling_total_us
Compute ratio: tc_total_us / profiling_total_us
Target: within ±30% (ratio 0.7 to 1.3)

### Step 5: Shape Matching Spot-Check

For top-5 ops by invocation count in TC trace:

1. Read the TC chrome trace — extract input shapes for one invocation
2. Read the corresponding CSV file (kernel_type.csv)
3. Verify that _inputs_match() in profiling_data_source.py would match
4. Document which shape transforms were needed

### Step 6: tc_input_count Safety Audit

For every op_mapping entry with `tc_input_count` set, verify it's safe. Full rules: `ref/tc_input_count_rules.md`.

For each entry with tc_input_count:

1. Read the corresponding CSV — check if input count is **fixed** (all rows have same count) or **variable** (mixed 1-input and 2-input rows)
2. If variable: `tc_input_count` is **UNSAFE** — flag for removal
3. If fixed and CSV count > TC count by exactly the truncated amount: **SAFE**

**Red flags:**

- `tc_input_count: 1` on elementwise ops (add, mul, div, sub) — almost always unsafe
- `tc_input_count` on ops where CSV has mixed broadcast patterns
- `tc_input_count` set without evidence in notes field
- `query_mode: elementwise` combined with `tc_input_count` — mutually exclusive, elementwise ops must not have tc_input_count

### Step 7: zero_cost Classification Audit

For every `zero_cost: true` entry, verify the classification is correct. Full rules: `ref/zero_cost_classification.md`.

For each zero_cost entry:

1. Search all profiling kernel_details.csv for the original kernel Type
2. If Type **found in profiling** → zero_cost is WRONG (should have kernel_type mapping)
3. If Type **not found** → verify the op's latency is captured by a fused kernel (document which one)

**Red flags:**

- zero_cost on ops that DO appear in profiling (miscategorized)
- zero_cost on compute ops without documenting which fusion absorbs them
- zero_cost on ops > 1% of total invocations without strong justification

### Step 8: Redundant profiling.* Placeholder Audit

Check for profiling.* entries whose kernel_type is already covered by a TC op mapping:

1. Collect all kernel_types from non-profiling entries (including alternate_kernel_types and sub_kernels)
2. For each profiling.* entry, check if its kernel_type is in the collected set
3. If yes → redundant, should be removed (TC op mapping already covers this kernel)

## Output

Write a verification report as markdown with:

```markdown
# op_mapping Verification Report

## Summary
- Model: <model>
- Device: <device>
- Config: <tc_config>
- Date: <today>

## Coverage
- Total TC ops: N
- Mapped: M (X%)
- Unmapped: [list]

## Hit Rate
- Compute hits: A/B (Y%)
- Expected misses: [list with reasons]

## Latency Comparison (if profiling available)
| kernel_type | TC (us) | Profiling (us) | Ratio | Status |
|---|---|---|---|---|
| MatMulV2 | ... | ... | ... | OK/FLAG |

## End-to-End
- TC total: X us
- Profiling total: Y us
- Ratio: Z (PASS/FAIL)

## Corrections Needed
corrections_needed:
  - op: "<op_name>"
    issue: "<description>"
    suggested_action: "<fix>"
```

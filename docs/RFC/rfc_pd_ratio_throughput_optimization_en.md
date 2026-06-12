# RFC: PD Ratio Throughput Optimization

## Metadata

| Item | Content |
| :--- | :--- |
| **Status** | Approved |
| **Author** | jiayanan |
| **Creation Date** | 2026-03-19 |
| **Related Links** |  |

---

## 1. Overview

This proposal implements a PD ratio optimization feature: P and D are **optimized independently**, each finding their optimal batch size and parallel configuration. After obtaining their respective QPS, the system automatically derives the optimal PD ratio (the ratio of P instances to D instances) based on QPS balancing. The final output is **Top N PD ratio results**, where each result includes the corresponding P and D configurations.

**QPS Formulas:**

```bash
P QPS = p_concurrency / ttft * 1000 (req/s)
D QPS = d_concurrency / (tpot * output_length) * 1000 (req/s)
```

**PD Ratio Calculation:** PD ratio = D_QPS / P_QPS. For example, if P QPS = 10 req/s and D QPS = 15 req/s, then PD ratio = 1.5, meaning 1.5 P instances are needed per D instance to achieve supply-demand balance.

---

## 2. Detailed Design

### 2.1 CLI Entry Extension

Add the following parameters to `cli/inference/throughput_optimizer.py`:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--prefill-devices-per-instance` | int | Number of devices per Prefill instance |
| `--decode-devices-per-instance` | int | Number of devices per Decode instance |
| `--enable-optimize-prefill-decode-ratio` | flag | Enable PD ratio optimization mode |

**Validation Rules:**

- When `--enable-optimize-prefill-decode-ratio` is enabled, `--prefill-devices-per-instance` and `--decode-devices-per-instance` are required
- P/D device counts must be positive integers

**Usage Example:**

```bash
python -m cli.inference.throughput_optimizer \
    --input-length=2048 --output-length=512 \
    model_id \
    --device=DEVICE --num-devices=16 \
    --prefill-devices-per-instance=4 --decode-devices-per-instance=2 \
    --enable-optimize-prefill-decode-ratio \
    --ttft-limits=500 --tpot-limits=50
```

### 2.2 ParallelRunner Extension

Add `run_pd_ratio()` method to `ParallelRunner` in `serving_cast/parallel_runner.py`.

**Core Design: P and D are optimized independently, then combined to compute PD ratio.**

Unlike `run_agg()`/`run_disagg()`, PD ratio mode requires different parallel configurations for P and D (e.g., P uses TP=4, D uses TP=2), thus separate `UserInputConfig` and `ModelRunner` instances must be created for each. P and D can **run independently**, each traversing the TP configuration space for optimization. After both complete, all P results and D results are combined to compute PD ratios.

```bash
run_pd_ratio():
    1. Run P optimization independently
       - num_devices = p_devices_per_instance
       - Iterate feasible TP configs, find optimal batch size for each
       - Collect all P results: [(p_parallel, p_batch, ttft, p_qps), ...]
    2. Run D optimization independently
       - num_devices = d_devices_per_instance
       - Iterate feasible TP configs, find optimal batch size for each
       - Collect all D results: [(d_parallel, d_batch, tpot, d_qps), ...]
    3. Combine P and D results to compute PD ratio
       - For each (P_result, D_result) pair, compute pd_ratio = D_QPS / P_QPS
       - Sort by balanced QPS (min(P_QPS, D_QPS)) in descending order
       - Output Top N ratio results
```

**Configuration generation method (unified):**

The configuration generation logic is identical for P and D, differing only in `num_devices`, so a single unified method is used with a parameter:

### 2.3 PDRatioThroughputOptimizer

Add `serving_cast/service/pd_ratio_throughput_optimizer.py` implementing the `PDRatioThroughputOptimizer` class.

**Class Design:**

`PDRatioThroughputOptimizer` is a **combiner** that does NOT inherit from `BaseThroughputOptimizer`. It takes existing P and D optimization results (as DataFrames) and combines them to compute PD ratios.

```python
class PDRatioThroughputOptimizer:
    """Combiner for PD ratio throughput optimization.

    Unlike Agg/Disagg optimizers that inherit from BaseThroughputOptimizer,
    this class combines existing P and D optimization results to compute PD ratios.
    It does not run optimization itself - it acts as a pure combiner.
    """

    def __init__(self, output_length: int):
        self.output_length = output_length
        self._p_df: pd.DataFrame = None
        self._d_df: pd.DataFrame = None
        self._result_df: pd.DataFrame = None

    def set_p_results(self, df: pd.DataFrame):
        """Set prefill optimization results."""
        self._p_df = df

    def set_d_results(self, df: pd.DataFrame):
        """Set decode optimization results."""
        self._d_df = df

    def optimize(self) -> pd.DataFrame:
        """Combine P and D results and compute PD ratios.

        Returns DataFrame with all PD ratio combinations.
        """
```

**Key Design Decisions:**

1. Uses DataFrame as input/output interface for consistency with `parallel_runner`
2. Does NOT inherit from `BaseThroughputOptimizer` - it's a combiner that processes existing results

**Column Naming Convention:**

- Columns unique to one DataFrame remain unchanged (`ttft` from prefill, `tpot` from decode)

- Overlapping columns get `_p` and `_d` suffixes (`parallel_p`, `parallel_d`, etc.)

- Final output uses `ttft_p` and `tpot_d` for consistency

### 2.4 Result Output

Overall Best Configuration is displayed first, followed by the Top N PD ratio table, consistent with the existing agg/disagg output style. Each row contains the corresponding P and D configuration information including concurrency:

**P/D Devices Calculation Logic (only when user specifies `--num-devices`):**

When the user provides `--num-devices` (total devices), the system calculates the total device allocation for P and D sides based on the PD ratio and per-instance device count. Constraints:

```bash
P_instances * prefill_devices_per_instance + D_instances * decode_devices_per_instance = num_devices
P_instances / D_instances ≈ pd_ratio
P_instances >= 1, D_instances >= 1 (positive integers)
```

Calculation: Enumerate all valid `(P_instances, D_instances)` positive integer pairs satisfying the total device constraint, and select the pair whose actual ratio `P_instances / D_instances` is closest to `pd_ratio`.

Example: `num_devices=16, prefill_devices_per_instance=4, decode_devices_per_instance=2, pd_ratio=1.5`

- P_instances=3, D_instances=2 → P_devices=12, D_devices=4 → actual ratio=1.5 → total=16 ✓

**Example (user specified `--num-devices`):**

```bash
********************************************************************************
  --------------------------------------------------------------------------
  Input Configuration:
    Model: model_id
    Devices: 16 DEVICE                        ← Only displayed when user provides --num-devices
    Prefill Devices Per Instance: 4
    Decode Devices Per Instance: 2
    TTFT Limits: 500 ms
    TPOT Limits: 50 ms
  --------------------------------------------------------------------------
  Overall Best Configuration:
    PD Ratio:    1.50 (P Instance:D Instance)
    Prefill QPS: 10.00 req/s  (TTFT: 100.00 ms, Parallel: tp4pp1dp1, Batch: 4, Concurrency: 4)
    Decode QPS:  15.00 req/s  (TPOT: 13.33 ms, Parallel: tp2pp1dp1, Batch: 8, Concurrency: 8)
    P Instances: 3 (12 devices)               ← Only displayed when user provides --num-devices
    D Instances: 2 (4 devices)                ← Only displayed when user provides --num-devices
  --------------------------------------------------------------------------
  Top N PD Ratio Configurations:
  +-----+----------+-----------+-----------+---------+---------+------------+------------+-----------+-----------+---------+---------+---------------+---------------+
  | Top | PD Ratio | P QPS     | D QPS     | P TTFT  | D TPOT  | P Parallel | D Parallel | P Devices | D Devices | P Batch | D Batch | P Concurrency | D Concurrency |
  |     | (P:D)    | (req/s)   | (req/s)   | (ms)    | (ms)    |            |            | /Instance | /Instance | Size    | Size    |               |               |
  +-----+----------+-----------+-----------+---------+---------+------------+------------+-----------+-----------+---------+---------+---------------+---------------+
  |  1  |   1.50   |     10.00 |     15.00 |  100.00 |   13.33 | tp4pp1dp1  | tp2pp1dp1  |         4 |         2 |       4 |       8 |             4 |             8 |
  |  2  |   1.20   |     12.00 |     14.40 |  120.00 |   15.00 | tp2pp1dp2  | tp2pp1dp1  |         4 |         2 |       6 |       6 |            12 |             6 |
  |  3  |   ...    |       ... |       ... |     ... |     ... | ...        | ...        |       ... |       ... |     ... |     ... |           ... |           ... |
  +-----+----------+-----------+-----------+---------+---------+------------+------------+-----------+-----------+---------+---------+---------------+---------------+
********************************************************************************
```

**Notes:**

- The `Devices` line is only displayed when the user provides `--num-devices`
- When user specifies `--num-devices`, the Top N table includes `P Devices` and `D Devices` columns showing the per-instance device count for P and D
- When user specifies `--num-devices`, Overall Best Configuration shows P/D instance count and total device allocation (calculated from PD ratio and num_devices constraint)
- Table is sorted by balanced QPS in descending order
- Each row contains the complete P and D configuration for that ratio (including concurrency)
- Overall Best Configuration is output before the table, consistent with existing agg/disagg output style

### 2.5 Overall Architecture Flow

```bash
CLI Argument Parsing (throughput_optimizer.py)
    │
    ├─ --enable-optimize-prefill-decode-ratio enabled?
    │   ├─ Yes → ParallelRunner.run_pd_ratio()
    │   │       │
    │   │       ├─ Phase 1: P independent optimization
    │   │       │   ├─ _get_pd_instance_user_configs(p_devices) → P TP config list
    │   │       │   ├─ ProcessPoolExecutor parallel execution
    │   │       │   │   └─ Each P TP config → ModelRunner → DisaggOptimizer → binary search best batch
    │   │       │   └─ Collect all P results: [p_result_1, p_result_2, ...]
    │   │       │
    │   │       ├─ Phase 2: D independent optimization
    │   │       │   ├─ _get_pd_instance_user_configs(d_devices) → D TP config list
    │   │       │   ├─ ProcessPoolExecutor parallel execution
    │   │       │   │   └─ Each D TP config → ModelRunner → DisaggOptimizer → binary search best batch
    │   │       │   └─ Collect all D results: [d_result_1, d_result_2, ...]
    │   │       │
    │   │       └─ Phase 3: Combine and compute PD ratio
    │   │           ├─ For each (P_result_i, D_result_j) compute QPS and PD ratio
    │   │           ├─ Sort by balanced QPS in descending order
    │   │           └─ Output Overall Best + Top N PD ratio results
    │   │
    │   └─ No → Existing run_agg() / run_disagg() flow
```

### 2.6 Module Interaction Diagram

```bash
┌──────────────────────────────────────────────────────────┐
│           throughput_optimizer.py (CLI)                   │
│  New: --prefill-devices-per-instance                      │
│       --decode-devices-per-instance                       │
│       --enable-optimize-prefill-decode-ratio               │
└───────────────────┬──────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│            ParallelRunner                        │
│  New: run_pd_ratio()                             │
│       _get_pd_instance_user_configs(num_devices)  │
│       _submit_pd_task()                          │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Phase 1: P Optimization  Phase 2: D Optimization
  (independent parallel)   (independent parallel)
        │                       │
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ P ModelRunner │        │ D ModelRunner │
│ + DisaggOpt  │        │ + DisaggOpt  │
│ (prefill)    │        │ (decode)     │
└──────┬───────┘        └──────┬───────┘
       │                       │
       └───────────┬───────────┘
                   ▼
         Phase 3: Combination
    ┌──────────────────────────┐
    │ PDRatioThroughputOptimizer│
    │ - Combine P/D results     │
    │ - Calculate QPS & PD Ratio│
    │ - Sort and output Top N   │
    └──────────────────────────┘
```

---

## 3. Implementation Plan

### Completed Features

* [x] CLI parameter extension: Add `--prefill-devices-per-instance`, `--decode-devices-per-instance`, `--enable-optimize-prefill-decode-ratio` parameters
* [x] ParallelRunner extension: Implement `run_pd_ratio()` using `_add_summary_result` pattern
* [x] PDRatioThroughputOptimizer: Implement combiner class with DataFrame interface, vectorized cross-join
* [x] Result output: RFC-compliant Overall Best + Top N PD ratio table with deduplication
* [x] `--dump-original-results` applies same filtering as normal output
* [x] Unit tests: Cover all PD ratio optimization scenarios

### Test Plan

| Test Scenario | Description |
| :--- | :--- |
| Basic functionality | PD ratio optimization runs correctly and outputs correct PD ratio with Top N results |
| PD ratio calculation | Verify pd_ratio = D_QPS / P_QPS is computed correctly |
| Parameter validation | Error handling for missing required parameters and invalid values |
| Edge cases | Handling of zero P or D QPS |
| Output format | Verify Overall Best before Top N table, each row contains PD ratio, P/D QPS, P/D configs and concurrency |
| Backward compatibility | Ensure existing agg/disagg modes are unaffected |

---

## 4. Files Modified

| File | Description |
| :--- | :--- |
| `serving_cast/service/pd_ratio_throughput_optimizer.py` | Core combiner class with DataFrame interface |
| `serving_cast/service/optimizer_summary.py` | PD ratio output formatting and filtering |
| `serving_cast/parallel_runner.py` | `run_pd_ratio()` integration using `_add_summary_result` pattern |
| `cli/inference/throughput_optimizer.py` | CLI arguments |
| `tests/test_pd_ratio_throughput_optimizer.py` | Unit tests |

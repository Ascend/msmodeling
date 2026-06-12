# Performance Regression Testing Framework

## Directory Structure

```text
tests/benchmark/models/
├── test_model_regression.py            ← Main entry: total time regression + operator regression
├── auto_baseline.py                    ← Standalone entry: auto-baseline runner (pytest)
├── __init__.py                         ← Package init
├── cases/                              ← Per-case JSON configuration files (includes operator baselines)
└── README.md                           ← This file
```

---

## Core Design: One Case Definition, Two Automatic Checks

Add a JSON configuration file under the `cases/` directory and the framework automatically performs two checks:

| Check | Description |
|-------|-------------|
| **Check 1: Total Time Comparison** | vs initial time (default 10%) + vs baseline time (default 20%) |
| **Check 2: Operator-Level Comparison** | Top-N operators vs initial operator baseline (default 10%) |

Execution policy in this repo:

- Put model-level fidelity cases only under `tests/benchmark/models/`
- Do not add `nightly` marker for these cases
- `run_benchmark.sh` and nightly full run will execute them; compile/regression incremental pipelines will not
- Shared model configs are stored in `tests/assets/model_config/`

---

## Case Configuration

### Case Types

The framework supports two model types with dedicated data structures:

- **`TextPerfRegressionCase`**: For text/VL/LLM models, configured via `UserInputConfig`
- **`VideoPerfRegressionCase`**: For video diffusion models, configured with video-specific parameters

Both share common fields from `BasePerfRegressionCase`.

### Adding a New Case

Create a JSON file under the `cases/` directory. The filename should match the `name` field.

#### Text Model Example (`cases/qwen3-8B-decode.json`)

```json
{
  "type": "text",
  "name": "qwen3-8B-decode",
  "description": "Qwen3-8B decode, 32 queries, ctx=1536, TP=2, compile",
  "initial_time_s": 0.012733,
  "baseline_time_s": 0.015406,
  "initial_tolerance": 0.10,
  "baseline_tolerance": 0.20,
  "operator_top_n": 10,
  "operator_tolerance": 0.10,
  "user_input": {
    "device": "ATLAS_800_A2_376T_64G",
    "model_id": "Qwen/Qwen3-8B",
    "num_queries": 32,
    "query_len": 1,
    "context_length": 1536,
    "do_compile": true,
    "decode": true,
    "quantize_linear_action": "DISABLED",
    "tp_size": 2,
    "world_size": 2
  }
}
```

#### Video Model Example (`cases/wan2.2-ulysses8.json`)

```json
{
  "type": "video",
  "name": "wan2.2-ulysses8",
  "description": "Wan2.2-T2V-A14B ulysses=8, batch=1, seq=128, 720x1280x81frames, bfloat16, use_cfg",
  "initial_time_s": 8.542,
  "baseline_time_s": 7.625,
  "initial_tolerance": 0.10,
  "baseline_tolerance": 0.20,
  "operator_top_n": 10,
  "operator_tolerance": 0.10,
  "device": "ATLAS_800_A3_752T_128G_DIE",
  "model_id": "assets/model_config/Wan2.2-T2V-A14B-Diffusers",
  "seq_len": 128,
  "batch_size": 1,
  "height": 720,
  "width": 1280,
  "frame_num": 81,
  "sample_step": 1,
  "dtype": "bfloat16",
  "use_cfg": true,
  "world_size": 8,
  "ulysses_size": 8,
  "cfg_parallel": false,
  "quantize_linear_action": "DISABLED"
}
```

### Common Fields (Base)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"text"` | Case type: `"text"` or `"video"` |
| `name` | `str` | **required** | Unique case identifier; operator baseline is stored in `operators` field of this file |
| `description` | `str` | **required** | Case description, shown on failure |
| `initial_time_s` | `float` | `0.0` | Initial total time (seconds). Set `0` to skip initial comparison |
| `baseline_time_s` | `float` | `0.0` | Baseline total time (seconds). Set `0` to skip baseline comparison |
| `initial_tolerance` | `float` | `0.10` | Tolerance vs initial time (10%) |
| `baseline_tolerance` | `float` | `0.20` | Tolerance vs baseline time (20%) |
| `operator_top_n` | `int` | `10` | Compare top-N most expensive operators |
| `operator_tolerance` | `float` | `0.10` | Operator-level tolerance (10%) |
| `operators` | `array` | `[]` | Operator baseline data: list of `{name, total_time_s, num_calls}` objects |

### Text-Specific Fields (`user_input`)

| Field | Type | Description |
|-------|------|-------------|
| `device` | `str` | Target device name |
| `model_id` | `str` | Model identifier or path |
| `num_queries` | `int` | Number of queries |
| `query_len` | `int` | Query token length |
| `context_length` | `int` | Context length for decode |
| `do_compile` | `bool` | Enable `torch.compile` |
| `decode` | `bool` | Enable decode mode |
| `quantize_linear_action` | `str` | Quantization action: `"DISABLED"`, `"W8A8_DYNAMIC"` |
| `quantize_attention_action` | `str` | Attention quantization: `"DISABLED"`, `"INT8"` |
| `tp_size` | `int` | Tensor parallelism degree |
| `dp_size` | `int` | Data parallelism degree |
| `ep_size` | `int` | Expert parallelism degree |
| `world_size` | `int` | Total device count |
| `num_mtp_tokens` | `int` | MTP token count |
| `image_batch_size` | `int` | Image batch size (VL models) |
| `image_height` | `int` | Image height (VL models) |
| `image_width` | `int` | Image width (VL models) |

### Video-Specific Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | `str` | `""` | Target device name |
| `model_id` | `str` | `""` | Path to model configuration directory |
| `seq_len` | `int` | `0` | Sequence length |
| `batch_size` | `int` | `0` | Batch size |
| `height` | `int` | `0` | Video height |
| `width` | `int` | `0` | Video width |
| `frame_num` | `int` | `0` | Number of frames |
| `sample_step` | `int` | `0` | Sampling step |
| `dtype` | `str` | `"float16"` | Data type |
| `use_cfg` | `bool` | `false` | Enable classifier-free guidance |
| `world_size` | `int` | `1` | Total device count |
| `ulysses_size` | `int` | `1` | Ulysses sequence parallelism degree |
| `cfg_parallel` | `bool` | `false` | Enable CFG parallel |
| `quantize_linear_action` | `str` | `"DISABLED"` | Quantization action |

---

## Running Tests

```bash
# Run all regression tests
python -m pytest tests/benchmark/models/test_model_regression.py -v --tb=short

# Filter by name
python -m pytest tests/benchmark/models/test_model_regression.py -k "qwen3_30b" -v --tb=short
```

---

## Output Example

```text
==============================================================================================================
  [Check 1] Total Time Regression Summary
==============================================================================================================
Case                                       Actual      Init   InitDiff%      Base   BaseDiff%         Status
--------------------------------------------------------------------------------------------------------------
qwen3_30b_a3b_prefill_w8a8_tp2_compile  330.123ms  300.000ms    +10.04%  322.000ms     +2.52%     FAIL(INIT)
qwen3_32b_prefill_w8a8_tp1              455.000ms  450.000ms     +1.11%  440.000ms     +3.41%           PASS
--------------------------------------------------------------------------------------------------------------
Total: 2 | Passed: 1 | Failed: 1 | No Baseline: 0
==============================================================================================================

==============================================================================================================
  [Check 2] Operator Regression Summary
==============================================================================================================
Case                                                         Status                              Details
--------------------------------------------------------------------------------------------------------------
qwen3_30b_a3b_prefill_w8a8_tp2_compile                       FAIL                   2 operator(s) exceeded
qwen3_32b_prefill_w8a8_tp1                                    PASS          All operators within tolerance
--------------------------------------------------------------------------------------------------------------
Total: 2 | Passed: 1 | Failed: 1 | No Baseline: 0
==============================================================================================================

*** Operator regression anomalies detected! ***

  [qwen3_30b_a3b_prefill_w8a8_tp2_compile]:
    aten::mm: +12.34% (baseline=45.123ms, actual=50.691ms)
    aten::addmm: +15.67% (baseline=32.456ms, actual=37.542ms)
```

---

## New Case Onboarding Process

Follow this standard lifecycle when adding a new performance regression case:

### Step 1: Create the Case Configuration

Create a new JSON file under `cases/<case_name>.json` with the appropriate `type` ("text" or "video") and all required fields. See the examples above for the correct format.

The framework automatically discovers and loads all `*.json` files from the `cases/` directory — no changes to the test source code are required.

### Step 2: First Run — Generate the Baseline

The operator baseline must be generated explicitly before regression tests can pass. On the first run, the test will fail with a message that no operator baseline was found. You need to capture the operator output and populate the `operators` field in your case JSON (`cases/<case_name>.json`):

```json
"operators": [
  {"name": "aten::mm", "total_time_s": 0.003200, "num_calls": 64},
  {"name": "aten::addmm", "total_time_s": 0.002100, "num_calls": 32}
]
```

Once the `operators` field is populated, subsequent runs will perform operator-level comparisons.

### Step 3: Second Run — Verify Stability

Run the same test a second time. The framework now has baseline data and will compare operator-level timings:

```bash
python -m pytest tests/benchmark/models/test_model_regression.py -k "your_case_name" -v --tb=short
```

Verify that:

- Total time comparisons (`initial_time_s` and `baseline_time_s`) are within tolerance
- Operator-level comparisons are stable (no unexpected regressions)
- Results are reproducible across multiple runs

### Step 4: Commit the Configuration

Once the case passes consistently, commit the case file:

- `cases/<case_name>.json` — the case configuration with operator baseline data

### Step 5: Refreshing Baselines

When a baseline refresh is needed (e.g., after a model update, performance optimization, or intentional operator change), clear the `operators` field in the case JSON and follow Steps 2–4 again:

```bash
# Manually edit the case JSON and set "operators": []
```

Then re-generate the operator baseline and re-verify.

**Important**: When committing a refreshed baseline, always include the reason in the commit message:

- Model version change (e.g., "Updated Qwen3-8B to v2.1")
- Performance baseline adjustment (e.g., "Adjusted baseline after compiler optimization")
- Intentional operator change (e.g., "Switched from aten::mm to aten::matmul")

---

## Auto-Baseline Runner (auto_baseline.py)

A pytest-based runner that automatically runs each case twice: the first run establishes a baseline, the second run compares against it (default tolerance: 5%).

### Adding a Case

Edit `auto_baseline.py` and add an `AutoBaselineCase` to the `AUTO_BASELINE_CASES` list:

```python
AUTO_BASELINE_CASES: List[AutoBaselineCase] = [
    AutoBaselineCase(
        name="qwen3-8B_auto",
        description="Qwen3-8B decode, baseline ctx=1536 vs compare ctx=1500",
        baseline_input=UserInputConfig(
            device="ATLAS_800_A2_376T_64G",
            model_id="Qwen/Qwen3-8B",
            num_queries=32,
            query_len=1,
            context_length=1536,
            do_compile=True,
            decode=True,
            tp_size=2,
            world_size=2,
        ),
        compare_input=UserInputConfig(
            device="ATLAS_800_A2_376T_64G",
            model_id="Qwen/Qwen3-8B",
            num_queries=32,
            query_len=1,
            context_length=1500,
            do_compile=True,
            decode=True,
            tp_size=2,
            world_size=2,
        ),
        tolerance=0.05,
    ),
]
```

### Auto-Baseline Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | **required** | Unique case identifier |
| `description` | `str` | **required** | Case description |
| `baseline_input` | `UserInputConfig` | **required** | Baseline inference configuration |
| `compare_input` | `UserInputConfig` | **required** | Comparison inference configuration |
| `tolerance` | `float` | `0.05` | Tolerance (5%) |

### Running

```bash
# Run all auto-baseline cases
python -m pytest tests/benchmark/models/auto_baseline.py -v -s

# Filter by name
python -m pytest tests/benchmark/models/auto_baseline.py -k "qwen3-8B" -v -s
```

---

## Quick Start

### 1. Add a Case

Create a JSON file under `cases/`:

```json
{
  "type": "text",
  "name": "your_case_name",
  "description": "your description",
  "initial_time_s": 0.300,
  "baseline_time_s": 0.322,
  "user_input": {
    "device": "YOUR_DEVICE",
    "model_id": "your/model/id",
    "num_queries": 1,
    "query_len": 6600,
    "do_compile": true,
    "tp_size": 2,
    "world_size": 2
  }
}
```

### 2. Run

```bash
python -m pytest tests/benchmark/models/test_model_regression.py -v --tb=short
```

**Note**: The `operators` field in each case JSON must be populated before the regression tests can pass. Without operator baseline data, the test will fail with a clear message directing you to generate the baseline first. See the onboarding process above for details.

### 3. Quick Self-Test

```bash
python -m pytest tests/benchmark/models/auto_baseline.py -v -s
```

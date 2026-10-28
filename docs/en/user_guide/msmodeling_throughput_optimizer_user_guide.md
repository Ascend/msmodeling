# Serving Performance Simulation User Guide 

## 1 Introduction

The throughput optimizer is a tool for optimizing throughput under Service Level Objective (SLO) constraints. It automatically searches for the optimal model configuration (parallelism strategy, batch size) under specified SLO constraints (for example, limits on TTFT and TPOT) to maximize token throughput.
This document is intended for developers, performance engineers, and capacity planners who need to evaluate LLM serving deployment plans, model serving throughput, parallelism strategies, and SLO constraint configurations. Before you start, complete the environment setup described in the [msModeling Installation Guide](../install_guide/msmodeling_install_guide.md) and confirm that the target model configuration can be loaded.

## 2 Main Scenarios

The throughput optimizer can be used for hardware planning, throughput optimization under SLO constraints, PD-disaggregation deployment design, and capacity planning that back-calculates the minimum device count required to meet a target QPS. By deployment form, it mainly supports the following scenarios:

| Mode | Applicable Scenario | Key Parameters |
| --- | --- | --- |
| PD aggregation | Prefill and decode run on the same instance, suitable for quickly evaluating overall throughput | `--tpot-limit`, `--ttft-limit`|
| PD disaggregation | Prefill and decode are deployed separately, requiring evaluation of the capability of each phase | `--disagg`, `--ttft-limit` or `--tpot-limit` |
| PD ratio | For planning the ratio of prefill to decode instance counts | `--enable-optimize-prefill-decode-ratio`, `--prefill-devices-per-instance`, `--decode-devices-per-instance` |
| KV Cache Offload | For evaluating how TTFT, TPOT, and the concurrency ceiling change after KV cache offload (Decode sparse/Prefill layerwise) | `--enable-decode-kv-offload`, `--enable-prefill-kv-offload` |
| Capacity planning | For back-calculating the minimum device count when the target QPS and SLO are known | `--target-qps` (can be combined with the parameters of any preceding mode) |

You can add `--speculative-method {dflash,dspark}` to any of the preceding scenarios to enable speculative decoding. For MTP, you are advised to use the unified interface `--speculative-method mtp`. For details, see [Combining Speculative Decoding (Dflash/DSpark)](#combining-speculative-decoding-dflashdspark).

### 2.1 PD Aggregation Scenario

Aggregation mode optimizes throughput for a combined prefill-decode serving architecture where both phases run on the same instance. The optimizer searches across all possible Tensor Parallelism (TP) and Data Parallelism (DP) configurations to find the best throughput under Service Level Objective (SLO) constraints.

Aggregation mode calculates the following metrics from a finite-batch timeline. Let `C` be the number of concurrent requests and let each request output `O` tokens. The arrival, first-token, and completion times of request `i` are `arrival_i`, `first_i`, and `finish_i`, respectively, all in ms. The batch drain time is `M = max(finish_i) - min(arrival_i)`. In the current model, all requests arrive at the same time. Therefore, `arrival_i = 0`.

| Metric | Formula | Unit |
| --- | --- | --- |
| TTFT | `mean(first_i - arrival_i)` | ms |
| TPOT | `mean(finish_i - first_i) / (O - 1)`, and 0 when `O = 1` | ms/token |
| QPS | `1000 × C / M` | req/s |
| Output throughput | `1000 × C × O / M = QPS × O` | token/s |

`--ttft-limit` and `--tpot-limit` check TTFT and TPOT separately. TPOT measures the average generation interval after the first token and excludes TTFT. QPS and output throughput use the full batch drain time; they cannot generally be derived from `TTFT + TPOT × O`. PP=1, PP>1, chunked prefill, and variable-length inputs use these definitions, while their scheduling timelines determine `first_i`, `finish_i`, and `M`. Decode TPOT in PD disaggregation is a separate phase metric and does not use this request-level aggregation timeline. In PD-disaggregated PP=1 prefill rows, `ttft` is the request-weighted drain average: the whole candidate batch arrives together and is drained wave by wave, so requests completed by later waves queue behind earlier ones and TTFT grows with concurrency — `--ttft-limit` prunes the concurrency search on this value.

#### Example

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action disabled \
    --tpot-limit 50
```

#### Using Prefix Cache

If you want to estimate aggregation throughput with prefix cache enabled, add `--prefix-cache-hit-rate`:

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action disabled \
    --tpot-limit 50 \
    --prefix-cache-hit-rate 0.5
```

#### Constraints

- `--max-batched-tokens` sets the token budget for one prefill or mixed prefill/decode step of a single data parallelism (DP) replica. With multiple DP replicas, the optimizer lets each replica use this budget independently. When the parameter is not passed in PD aggregation mode, the optimizer first uses `4 * input_length` and, if the prefill phase runs out of memory (OOM), falls back to `2 * input_length` and then `1 * input_length` in sequence. When the parameter is not passed for fixed-length PD disaggregation and PD-ratio prefill, no scheduling budget is set. Instead, the optimizer models the full concurrency of the result row so that the result maps directly to `text_generate`. To specify a budget for these serving simulations, pass the parameter explicitly. After you set the budget, if `effective_input_length` is greater than the effective `max_batched_tokens`, the optimizer automatically splits prefill into chunks.

### 2.2 PD Disaggregation Scenario

Disaggregation mode separates prefill and decode phases into independent optimization runs. This is useful when you need to characterize each phase independently or when planning disaggregated serving deployments.

#### Prerequisites

To enable disaggregation mode, you must provide:

- `--disagg`: Enables disaggregation mode.

#### Prefill Mode

Optimize prefill phase throughput under Time-to-First-Token (TTFT) constraints. In this mode, set both `--disagg` and `--ttft-limit`.

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action disabled \
    --disagg \
    --ttft-limit 2000
```

#### Decode Mode

Optimize decode phase throughput under Time-per-Output-Token (TPOT) constraints. In this mode, set both `--disagg` and `--tpot-limit`.

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action disabled \
    --disagg \
    --tpot-limit 50
```

#### Combining Speculative Decoding (Dflash/DSpark)

In PD aggregation, PD disaggregation, or PD ratio scenarios, you can add draft-based speculative decoding modeling through `--speculative-method {dflash,dspark}` to evaluate its impact on decode throughput and TPOT. This option is mutually exclusive with MTP (a non-zero `--num-mtp-tokens`). `--num-speculative-tokens`, `--acceptance-length`, `--num-draft-layers`, and `--draft-model-config-path` require `--speculative-method` to be set first. For DSpark, you can optionally use `--dspark-markov-rank` and `--dspark-markov-head` (defaults: 256 and `vanilla`).

> Note: For MTP, you are advised to use the unified interface `--speculative-method mtp --num-speculative-tokens N [--acceptance-length A]`. The legacy entry points `--num-mtp-tokens` and `--mtp-acceptance-rates` cannot be mixed with the new entry points `--speculative-method`, `--num-speculative-tokens`, and `--acceptance-length`. When you select `mtp`, you must also configure `--num-speculative-tokens`. The existing `--num-mtp-tokens` (and `--mtp-acceptance-rates`) still work alone for compatibility, but will be deprecated in later versions. `--num-speculative-tokens` supports multi-value search. `--acceptance-length` is clamped to `n` (`= block_size - 1`) for all methods.

The following example adds speculative decoding parameters to the PD disaggregation decode example:

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --disagg \
    --tpot-limit 50 \
    --speculative-method dflash \
    --num-speculative-tokens 7 \
    --acceptance-length 5 \
    --num-draft-layers 6
```

Notes:

- When `--speculative-method` is set, any candidate of 0 for `--num-speculative-tokens` causes a parse failure. To disable the feature, omit `--speculative-method`. When you omit the parameter, dflash and dspark still use the builtin `block_size`. You can also pass multiple candidate values (for example, 2 4) to `--num-speculative-tokens` for the search.
- `--num-draft-layers` and `--draft-model-config-path` apply only to `dflash` and `dspark` and cannot be used together with `--speculative-method mtp`.
- To use `dspark`, change `--speculative-method` to `dspark`. The Markov parameters default to `--dspark-markov-rank 256` and `--dspark-markov-head vanilla` (optional values: `gated` and `rnn`. MarkovHead is disabled when you set rank to 0).
- The default draft configuration is at `tensor_cast/runtime_configs/draft_configs/dflash_draft_builtin.json`. For field descriptions and custom methods, see [`tensor_cast/runtime_configs/draft_configs/README.md`](../../../tensor_cast/runtime_configs/draft_configs/README.md).

The following example enables pipeline parallelism (PP) search on top of PD disaggregation to evaluate the TTFT, TPOT, and throughput of different PP configurations:

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
    --device ATLAS_800_A3_560T_128G_DIE \
    --num-devices 32 \
    --input-length 2048 \
    --output-length 512 \
    --quantize-linear-action W8A8_DYNAMIC \
    --disagg \
    --tpot-limit 50 --ttft-limit 10000 \
    --tp-sizes 8 \
    --pp-sizes 1 2 4 \
    --ep-sizes 8 \
    --moe-dp-sizes 1
```

Notes:

- When `--pp-sizes` is not specified, PP is fixed at 1 and the original TP/EP/DCP search path runs, with fully compatible behavior and results. When you specify it, the optimizer enters the PP-aware search path.
- For PP>1, the optimizer derives `dp = num_devices // (tp * pp)` and `moe_tp = (num_devices // pp) // (ep * moe_dp)` with stage-local arithmetic, and uses a forward blocking scheduler to calculate the makespan, bubble ratio, and bottleneck stage. The result table adds columns such as `pp_size`, `pp_bubble_ratio`, `pp_bottleneck_stage`, and `pp_makespan_ms`.
- You can use `--pp-layer-partitions '[[31,30],[16,15,15,15]]'` to explicitly specify the layer partition (the length of each inner list equals the corresponding `pp_size`, and the sum of the elements equals `num_hidden_layers`). When you do not specify it, the optimizer uses even splitting.
- PP and DCP can be searched together (an explicit `--dcp-sizes` is not discarded).
- The following scenarios do not support PP>1: VL/multimodal models and variable-length input distributions (`length_distribution`). When these scenarios are matched, the optimizer skips the candidate and logs a warning. PP and speculative decoding (`--speculative-method mtp/dflash/dspark`) can be searched together, and the proposal/draft computation of speculative decoding runs on the last stage of the pipeline.
- The forward blocking (no-overlap) schedule gives a conservative lower bound for PP>1 throughput (it systematically overestimates the bubble). Therefore, do not conclude from it that PP is necessarily worse. Later 1F1B and interleaved evolution will improve this.

### 2.3 PD Ratio Scenario

Prefill-decode (PD) ratio optimization mode enables independent optimization of prefill and decode phases, then combines the results to find the optimal P/D instance ratio for maximum system throughput. This mode is particularly useful for disaggregated serving architectures where prefill and decode instances can be scaled independently.

#### Prerequisites

To enable PD ratio optimization, you must provide:

- `--enable-optimize-prefill-decode-ratio`: Enables PD ratio optimization mode.
- `--prefill-devices-per-instance`: number of devices per prefill instance
- `--decode-devices-per-instance`: number of devices per decode instance

#### Example

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
    --device TEST_DEVICE \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action disabled \
    --enable-optimize-prefill-decode-ratio \
    --prefill-devices-per-instance 16 \
    --decode-devices-per-instance 16 \
    --log-level info
```

#### Constraints

- `--enable-optimize-prefill-decode-ratio` cannot be used together with `--disagg`
- Both `--prefill-devices-per-instance` and `--decode-devices-per-instance` must be specified when PD ratio optimization is enabled

### 2.4 KV Cache Offload Scenario

Models the memory and transfer costs of KV cache offloading (Decode sparse/Prefill layerwise) into the search. Therefore, you can evaluate how TTFT, TPOT and the concurrency ceiling change with offload enabled. Currently targets DSA indexer-style models such as GLM-5.2 (the sparse window is read automatically from the model config `index_topk`, the indexer cache stays on device and is not offloaded, dense/GQA models do not support Decode offload, and PP>1 is mutually exclusive with offload).

#### Example

```bash
# Offload on/off pair (fixed concurrency, compare delta TPOT)
msmodeling inference throughput-optimizer zai-org/GLM-5.2 \
  --device ATLAS_800_A3_752T_128G_DIE --num-devices 32 --tp-sizes 8 --ep-sizes 4 \
  --input-length 80000 --output-length 2048 --batch-range 10 10 \
  --enable-decode-kv-offload --decode-offload-buffer-multiplier 2.0 --decode-offload-hit-rate 0.85
```

#### Notes and Constraints

- The sparse window top-k is resolved from the model checkpoint config (GLM-5.2 `index_topk=2048`) and is not a CLI flag. Enabling Decode offload is rejected when the config declares none.
- The host link bandwidth comes from the DeviceProfile (A3 = 70 GB/s/die/direction theoretical peak). Bandwidth-tier sensitivity requires profile variants.
- Prefill layerwise offload uses code-level defaults for B/I (`B=2`, `I=0`).

### 2.5 Capacity Planning Scenario

Capacity planning mode answers the reverse question: **given a target QPS and SLO, what is the minimum number of devices?** It adds linear back-calculation on top of the search results of the existing optimizer and does not change the search logic of the optimizer itself. The same command with or without `--target-qps` follows the same simulation path, but the former additionally reports a minimum device count.

Passing `--target-qps` triggers this mode, which you can combine with PD aggregation, PD disaggregation, and PD ratio.

#### Aggregation Mode

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device ATLAS_800_A3_752T_128G_DIE \
    --input-length 1024 \
    --output-length 512 \
    --ttft-limit 500 \
    --tpot-limit 50 \
    --target-qps 1000
```

#### PD Disaggregation Mode

When you combine `--disagg` with capacity planning, you must specify **exactly one** phase SLO, which determines the phase used for the back-calculation:

```bash
# Prefill phase
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device ATLAS_800_A3_752T_128G_DIE \
    --input-length 2048 \
    --output-length 512 \
    --disagg \
    --ttft-limit 500 \
    --target-qps 500

# Decode phase
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device ATLAS_800_A3_752T_128G_DIE \
    --input-length 2048 \
    --output-length 512 \
    --disagg \
    --tpot-limit 50 \
    --target-qps 500
```

#### PD Ratio Mode

```bash
# Fixed devices per instance
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
    --device ATLAS_800_A3_752T_128G_DIE \
    --input-length 3500 \
    --output-length 1500 \
    --enable-optimize-prefill-decode-ratio \
    --prefill-devices-per-instance 4 \
    --decode-devices-per-instance 2 \
    --ttft-limit 500 \
    --tpot-limit 50 \
    --target-qps 500

# Automatically search devices per instance: do not pass --prefill-devices-per-instance / --decode-devices-per-instance
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
    --device ATLAS_800_A3_752T_128G_DIE \
    --input-length 3500 \
    --output-length 1500 \
    --enable-optimize-prefill-decode-ratio \
    --ttft-limit 500 \
    --tpot-limit 50 \
    --target-qps 500
```

#### Constraints

- `--target-qps` must be a positive number. Passing it always triggers capacity planning mode.
- `--num-devices` becomes optional:
  - When specified, capacity planning runs the optimizer only once at that device count. The value also serves as the upper limit for the devices per instance in the automatic PD ratio search.
  - When not specified, the tool derives a starting budget from the `comm_grid` of the device profile and enumerates **all** power-of-2 budgets up to the `comm_grid` capacity. It runs the optimizer in full for each budget and takes the global minimum device count after merging the results. The per-device QPS is non-monotonic with respect to the device count (a larger budget may introduce a better parallelism structure). Therefore, the tool does not exit early. The number of budgets is limited. For example, the `comm_grid` capacity of A3 is 768, giving 8 budgets in total.
  - The tool directly judges a value that exceeds the `comm_grid` capacity of the device as infeasible and does not run an optimizer pass that is bound to fail.
- When you combine `--disagg` with capacity planning, exactly one of `--ttft-limit` and `--tpot-limit` must be present. Providing both or neither raises an explicit error, which avoids silently switching to the aggregation simulation path.
- The automatic PD ratio search enumerates all combinations of devices per instance for P and D (`p_budgets × d_budgets`), and the number of combinations grows quadratically with the number of budgets. The devices-per-instance budgets come from the device `comm_grid` and are independent of `--tp-sizes`. TP is a parallelism structure **internal** to an instance, and the device count of an instance should be on the order of its product (`TP × PP`). For precise control, pass `--prefill-devices-per-instance`/`--decode-devices-per-instance` explicitly.
- Search dimensions such as `--tp-sizes`/`--pp-sizes`/`--ep-sizes` keep the same semantics under capacity planning and still control the parallelism structure search **within each instance**.
- In capacity planning mode, `--device` takes only the first device. When you pass multiple devices, the tool logs a warning.

## 3 Result Information

After the command runs successfully, the terminal first prints the input configuration and the best configuration summary, then displays the candidate parallel configuration table. The output metrics include throughput, TTFT, TPOT, concurrency, and mode-specific fields (such as QPS or PD ratio). For example, the following is the output of the 2.1 PD aggregation scenario:

```text
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration:
    Model: Qwen/Qwen3-32B
    Quantize Linear action: W8A8_DYNAMIC
    Quantize Attention action: disabled
    Devices: 8 TEST_DEVICE
    TTFT Limits: None ms
    TPOT Limits: 50.0 ms
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    Best Throughput: 2888.45 tokens/s
    TTFT: 16032.05 ms
    TPOT: 49.90 ms
  ----------------------------------------------------------------------------
Top 4 Aggregation Configurations:
+-----+----------------------+-----------+-----------+-------------+-------------+--------------------+------------+
| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices |      parallel      | batch_size |
+-----+----------------------+-----------+-----------+-------------+-------------+--------------------+------------+
|  1  |       2888.45        |  16032.05 |   49.90   |     175     |       8     | TP=8 | PP=1 | DP=1 |    175     |
|  2  |       2013.49        |  22512.86 |   49.56   |     130     |       8     | TP=4 | PP=1 | DP=2 |     65     |
|  3  |       1140.23        |  25817.73 |   49.44   |      76     |       8     | TP=2 | PP=1 | DP=4 |     19     |
|  4  |        549.89        |  14214.54 |   48.72   |      32     |       8     | TP=1 | PP=1 | DP=8 |     4      |
+-----+----------------------+-----------+-----------+-------------+-------------+--------------------+------------+
********************************************************************************
```

Focus on the following fields:

- `TP`/`DP`: recommended parallelism strategy
- `concurrency`: number of concurrent requests that the current candidate configuration supports
- `batch size`: batch size under the SLO constraint
- `TTFT`/`TPOT`: time-to-first-token and time-per-output-token
- `Throughput (token/s)`: system-level output token throughput. A larger value indicates higher throughput.

Success criteria:

- The terminal outputs `Overall Best Configuration` or the candidate configuration table.
- The output includes metrics such as `Throughput`, `TTFT`, and `TPOT`.
- No model configuration loading failure or parameter conflict error occurs.

### 3.1 Comparing Multiple Hardware Profiles

One or more `--device` values can be passed in a single run to benchmark multiple `DeviceProfile` targets and compare their best configurations under the same model, workload, and SLO settings.

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device ATLAS_800_A2_280T_64G ATLAS_800_A3_560T_128G_DIE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action disabled \
    --tpot-limit 50
```

Behavior:

- Each device profile is optimized sequentially. Per-device tables (same format as a single-device run) are printed after each profile finishes.
- When two or more devices are specified, the tool additionally prints:
  1. A **hardware profile comparison** table with core modeling parameters (compute, memory, communication bandwidth, and related fields).
  2. A **cross-hardware summary** table with the best configuration per device, ranked for easy comparison.
- Cross-hardware summaries are mode-specific:
  - Aggregation: best throughput per device under TTFT/TPOT limits.
  - Disaggregation: separate prefill and decode cross-hardware tables when the corresponding limits are set.
  - PD ratio: best balanced QPS per device under TTFT/TPOT limits, including PD ratio and optional P/D instance counts when `--num-devices` is set.

#### Example Output When Using Multiple `--device` Values

When two or more device profiles are specified, the optimizer prints per-device results for each profile, followed by two additional cross-hardware tables:

**Hardware Profile Comparison Table**

This table shows core modeling parameters for all requested devices (compute, memory bandwidth, communication bandwidth, and so on):

```text
************************************************************************************************************
  Cross-hardware - device profile summary (modeling abstraction vs performance merge tables)
  Device profile parameter comparison (effective compute / memory BW / comm BW)
  --------------------------------------------------------------------------------------------------------
+-----------------------+-----------------------+-------------------------+---------------+-------------+-----------+----------------+
|         Device        | Cube Compute (TFLOPS) | Vector Compute (TFLOPS) | HBM BW (TB/s) | Memory (GB) | Comm Grid | Comm BW (GB/s) |
+-----------------------+-----------------------+-------------------------+---------------+-------------+-----------+----------------+
|      TEST_DEVICE      |         247.73        |           7.70          |     0.960     |     64.0    |  256 x 8  |   35 | 137.2   |
| ATLAS_800_A2_376T_64G |         247.73        |          15.40          |     0.960     |     64.0    |  128 x 8  |  17.5 | 137.2  |
+-----------------------+-----------------------+-------------------------+---------------+-------------+-----------+----------------+
```

**Cross-Hardware Summary Table (Mode-Specific)**

A ranked table of the best configuration per device under the active SLO constraints. Example for Aggregation mode:

```text
****************************************************************************************************
  Cross-hardware - PD Aggregated (best throughput config per device under TTFT/TPOT limits)
  ------------------------------------------------------------------------------------------------
+-----+-----------------------+----------------------+-----------+-----------+-------------+--------------------+-------+-------------+
| Top |         Device        | Throughput (token/s) | TTFT (ms) | TPOT (ms) | Concurrency |      Parallel      | Batch | num_devices |
+-----+-----------------------+----------------------+-----------+-----------+-------------+--------------------+-------+-------------+
|  1  | ATLAS_800_A2_376T_64G |       18435.99       |  4986.05  |   54.48   |     1184    | TP=1 | PP=1 | DP=8 |  148  |      8      |
|  2  |      TEST_DEVICE      |       18128.74       |  4973.39  |   53.39   |     1144    | TP=1 | PP=1 | DP=8 |  143  |      8      |
+-----+-----------------------+----------------------+-----------+-----------+-------------+--------------------+-------+-------------+
```

For PD Disaggregation and PD Ratio modes, the cross-hardware summary tables contain the corresponding phase-specific or QPS-related columns.

#### Terminal Sweep Curves (Single Device)

When exactly one device profile is evaluated, the optimizer can render **terminal ASCII scatter plots** after the sweep completes. These plots help inspect how throughput relates to concurrency and latency across parallel configurations.

Plots are produced for all three optimizer modes:

| Mode | Plots |
|------|-------|
| Aggregation | Throughput vs Concurrency, Throughput vs TPOT |
| Disaggregation (Prefill) | Throughput vs Concurrency, Throughput vs TTFT |
| Disaggregation (Decode) | Throughput vs Concurrency, Throughput vs TPOT |
| PD ratio | Throughput vs Concurrency, Throughput vs TPOT (decode-side TPS) |

Notes:

- Terminal curves are **not** printed when multiple `--device` values are used. Use cross-hardware summary tables instead.
- Curve points exclude OOM/insufficient-memory configurations. They are **not** filtered by TTFT/TPOT SLO limits. Therefore, the plots show the full valid sweep while tables still report SLO-constrained bests.
- Rendering uses the optional `plotext` dependency. If `plotext` is unavailable or plotting fails, optimization results are still printed and a warning is logged.

Example (single device, aggregation):

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device ATLAS_800_A2_280T_64G \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --tpot-limit 50 \
    --batch-range 1 256
```

### 3.2 Capacity Planning Result

When `--target-qps` is enabled, the command prints the capacity planning summary before the standard optimization table:

```text
****************************************************************************************************
  Capacity Planning Result
  --------------------------------------------------------------------------------------------------
  Target QPS:  1000.00 req/s
  SLO:         TTFT <= 500 ms, TPOT <= 50 ms
  --------------------------------------------------------------------------------------------------
  Minimum Devices: 64
  Achieved QPS:     1008.00 req/s
  Per-Die QPS:      15.75 req/s
  Replicas Needed:  8
  Per-Replica Dies: 8
  Parallel:         TP=8 | PP=1 | DP=1
  Batch Size:       64
  TTFT:             320.00 ms
  TPOT:             42.00 ms
  --------------------------------------------------------------------------------------------------
  Back-calculation comparison table (sorted by total device count ascending, highest per-die QPS for the same device count)
  ...
****************************************************************************************************
```

Fields:

| Field | Meaning |
| --- | --- |
| `Minimum Devices` | Minimum total device count that meets the target QPS and SLO |
| `Achieved QPS` | Actual request QPS back-calculated from the minimum-device-count plan, not lower than the target QPS |
| `Per-Die QPS` | Request QPS per device, used to compare plans with the same device count |
| `Replicas Needed` | Number of required instances (replicas) |
| `Per-Replica Dies` | Number of devices occupied by each instance |
| `Parallel`/`Batch Size`/`TTFT`/`TPOT` | Parallelism strategy and performance metrics of the optimal configuration |

In PD ratio mode, the summary reports `Topology` (in the form `xPyD`), `P QPS`, `D QPS`, and `PD Ratio` instead, and the back-calculation comparison table lists the total device count and per-device QPS of each (P, D) combination.

If no feasible configuration exists, the summary outputs `No feasible configuration found under the given constraints.` and logs the reason (the model does not fit, no configuration meets the SLO, or all budgets exceed the `comm_grid` capacity of the device).

## 4 Parameters

```bash
Options:
  --input-length INPUT_LENGTH
                        The input length of the prompt. (default: None)
  --output-length OUTPUT_LENGTH
                        The expected output length. (default: None)
  --mtp-acceptance-rates MTP_ACCEPTANCE_RATE [MTP_ACCEPTANCE_RATE ...]
                        Acceptance rate list for MTP (default: [0.9, 0.6, 0.4, 0.2])
  --dump-original-results
                        If set, dump the original results for analysis. (default: False)

General Options:
  model_id              Model source. Recommended safe mode: a reviewed absolute local model path. Model id mode also accepts Hugging Face or
                        ModelScope ids, but may execute remote Python code through trust_remote_code=True and is not security-guaranteed.
  --device DEVICE [DEVICE ...]
                        Device profile(s) to evaluate. One or more registered DeviceProfile names.
                        Supported values: any registered DeviceProfile name; see the TensorCast user guide.
                        Multiple values enable cross-hardware comparison tables.
                        Duplicate names are removed; input order is preserved.
                        If omitted, defaults to TEST_DEVICE. (default: TEST_DEVICE)
  --num-devices NUM_DEVICES
                        Specifies the total number of devices/processes to use. Must be a positive integer. A value of 1 indicates single-device
                        execution. (default: 1)
  --reserved-memory-gb RESERVED_MEMORY_GB
                        Amount of device memory (in gigabytes) reserved for system usage and unavailable for application. Set to 0 to disable
                        memory reservation. (default: 10.0)
  --log-level {debug,info,warning,error,critical}
                        Specifies the verbosity level for log output. Available levels: 'debug' (most verbose), 'info', 'warning', 'error', 'critical' (least verbose). [default: error]

Model & Quantization Options:
  --compile             If set, invoke torch.compile() on the model before inference. (default: False)
  --compile-allow-graph-break
                        If set, allows graph breaks during torch.compile() to improve compilation speed or handle unsupported ops. (default: False)
  --num-mtp-tokens {0,1,2,3,4,5,6,7,8,9} [{0,1,2,3,4,5,6,7,8,9} ...]
                        MTP token count candidate(s). Pass one value for a fixed configuration, or multiple values to
                        sweep during throughput optimization. 0 means disabled and only models with MTP support will
                        benefit from non-zero values. When combined with TP/EP/MOE-DP search, total combinations grow as
                        TP x EP x MOE-DP x MTP. Legacy MTP entry; cannot be mixed with
                        --speculative-method / --num-speculative-tokens / --acceptance-length.
                        MTP can also use --speculative-method mtp --num-speculative-tokens. (default: None)
  --speculative-method {mtp,dflash,dspark}
                        Enable speculative decoding: mtp, dflash, or dspark. Mutually exclusive with
                        the legacy MTP entry (--num-mtp-tokens / --mtp-acceptance-rate).
                        --speculative-method mtp requires --num-speculative-tokens. (default: None)
  --num-speculative-tokens NUM_SPECULATIVE_TOKENS [NUM_SPECULATIVE_TOKENS ...]
                        Requires --speculative-method. Speculative depth excluding anchor; block_size = n + 1 when
                        n >= 1. Pass multiple values to sweep. Any 0 candidate is rejected; omit
                        --speculative-method to disable. Omitting keeps builtin block_size for dflash/dspark.
                        For mtp, n is the MTP token count. (default: None)
  --acceptance-length ACCEPTANCE_LENGTH
                        Requires --speculative-method. Decode fold scalar; clamped to n (= block_size - 1) for all
                        methods. (default: 5.0)
  --num-draft-layers NUM_DRAFT_LAYERS
                        Requires --speculative-method dflash or dspark. Override draft num_hidden_layers; 0 = config
                        default. (default: 0)
  --draft-model-config-path DRAFT_MODEL_CONFIG_PATH
                        Requires --speculative-method dflash or dspark. Optional path to override builtin draft
                        config.json. (default: None)
  --dspark-markov-rank DSPARK_MARKOV_RANK
                        Requires --speculative-method dspark. Markov embedding rank; 0 disables MarkovHead. (default: 256)
  --dspark-markov-head {vanilla,gated,rnn}
                        Requires --speculative-method dspark. Markov head type. (default: vanilla)
  --quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}
                        Quantize all linear layers in the model from choices (currently only support symmetric quant) (default: W8A8_DYNAMIC)
  --quantize-non-expert-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}
                        Set a separate quantization type for non-expert linear layers, such as attention projections, dense MLP layers, and shared experts, while routed MoE experts keep --quantize-linear-action. This option is mainly intended for DeepSeek V4-style MoE models. (default: DISABLED)
  --mxfp4-group-size mxfp4_GROUP_SIZE
                        Group size for mxfp4 quantization (default: 32)
  --quantize-attention-action {DISABLED,INT8,FP8}
                        Quantize the KV cache with the given action (default: DISABLED)
  --tp-sizes [TP_SIZES ...]
                        Enable TP search. Optional explicit TP sizes. If no value is provided, defaults to powers of 2 up to world_size. (default: None)
  --ep-sizes [EP_SIZES ...]
                        Enable EP search. Optional explicit EP sizes. If no value is provided, defaults to powers of 2 up to world_size. (default: None)
  --moe-dp-sizes [MOE_DP_SIZES ...]
                        Enable MOE-DP search. Optional explicit MOE-DP sizes. If no value is provided, defaults to powers of 2 up to world_size. (default: None)
  --enable-shared-expert-tp
                        Enable vLLM-style tensor parallel for shared experts. (default: False)
  --compilation-config [{enable_multistream,enable_sequence_parallel,enable_matmul_allreduce,enable_dispatch_ffn_combine} ...]
                        Enable the specified compilation features as needed. Separate multiple features with spaces,
                        for example, --compilation-config enable_sequence_parallel enable_dispatch_ffn_combine.
                        When not specified, all compilation features remain disabled. (default: None)
  --word-embedding-tp {col,row}
                        Enable word embedding tensor parallel with mode {'col','row'}. If omitted, embedding TP is disabled. (default: None)

Debug Options:
  --chrome-trace-file CHROME_TRACE
                        Generate chrome trace file for visualization, for example trace.json. (default: None)

Performance Model Options:
  --performance-model {analytic,profiling}
                        Performance model type. 'analytic': Roofline model (default). 'profiling': empirical model backed by measured CSV data (requires --profiling-database-path). (default: analytic)
  --profiling-database-path PROFILING_DATABASE
                        Path to the profiling CSV database directory for 'profiling' mode. For example, tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/ (default: None)

Service Options:
  --ttft-limit TTFT_LIMITS
                        TTFT constraints under which to search for the best throughput. None means no constraint. (default: None)
  --tpot-limit TPOT_LIMITS
                        TPOT constraints under which to search for the best throughput. None means no constraint. (default: None)
  --target-qps TARGET_QPS
                        Target request QPS (req/s). When set, triggers capacity planning mode: the tool searches for the minimum device count that meets
                        this QPS under the given SLO. If --num-devices is also set, it controls the TP search width; otherwise the device count is
                        auto-derived: every power-of-2 budget from the device profile's comm_grid seed up to its capacity is searched and the results are
                        merged, so the reported minimum is global across those budgets.
  --max-batched-tokens MAX_BATCHED_TOKENS
                        Max batched tokens per data-parallel replica for one prefill or mixed prefill/decode step. If omitted, starts from 4 * input_length and falls back on Prefill OOM. (default: None)
  --prefix-cache-hit-rate PREFIX_CACHE_HIT_RATE
                        Prefix cache hit rate for token-level prefill reuse approximation. Valid range: [0, 1). (default: 0.0)
  --batch-range BATCH_RANGE [BATCH_RANGE ...]
                        Batch size range: [min max] or [max] (default: 1 for min, no limit for max) (default: None)
  --serving-cost SERVING_COST
                        Serving cost represents the cost of service delivery (default: 0)
  --disagg              If set, run disaggregation mode. disagg means disaggregation mode. (default: False)
  --jobs JOBS           Number of parallel jobs. (default: 8)
  --max-search-combinations MAX_SEARCH_COMBINATIONS
                        Warn when TP/EP/MOE-DP/MTP search combinations exceed this value. Set 0 to disable the warning. (default: 100)
  --concurrency-search-strategy {exponential,linear_exponential}
                        Concurrency search strategy. The default is exponential. (default: exponential)

MultiModal Options:
  --image-batch-size IMAGE_BATCH_SIZE
                        Number of images per request. If omitted, reuse batch_size for backward compatibility. (default: None)
  --image-height IMAGE_HEIGHT
                        Height of the input images (default: None)
  --image-width IMAGE_WIDTH
                        Width of the input images (default: None)

PD Ratio Optimization Options:
  --enable-optimize-prefill-decode-ratio
                        Enable Prefill-decode (PD) ratio optimization mode. This mode independently
                        optimizes Prefill and Decode phases, then combines results to find the optimal
                        P/D instance ratio. Cannot be used together with --disagg. (default: False)
  --prefill-devices-per-instance PREFILL_DEVICES_PER_INSTANCE
                        Number of devices per Prefill instance. Required when --enable-optimize-prefill-decode-ratio
                        is set. Determines the parallelism configuration search space for Prefill phase.
  --decode-devices-per-instance DECODE_DEVICES_PER_INSTANCE
                        Number of devices per Decode instance. Required when --enable-optimize-prefill-decode-ratio
                        is set. Determines the parallelism configuration search space for Decode phase.

KV Cache Offload Options:
  --enable-decode-kv-offload
                        Enable Decode sparse KV cache offload: move the device main KV to host, keep only indexer + a
                        top-k hot buffer on device; miss rows are fetched per step by hit rate. (default: False)
  --decode-offload-buffer-multiplier <FLOAT>
                        Hot buffer size as a multiple of topk (>= 1: the buffer must hold at least one full top-k window for a positive hit rate to be deployable; upstream measures 80-90 percent hit rate at 2.0).
                        (default: 2.0)
  --decode-offload-hit-rate <FLOAT>
                        Top-k hot buffer hit rate for decode KV offload (conservative default 0.85). (default: 0.85)
  --host-memory-available-gb <FLOAT>
                        Host memory budget for the offload KV pool in GB. 0 disables the host capacity check; when set, candidates whose offloaded KV pool exceeds the budget fold into the OOM early-stop path instead of returning undeployable results.
                        (default: 0.0)
  --enable-prefill-kv-offload
                        Enable Prefill layerwise KV cache offload: N layers share B physical NPU buffers, the rest of the
                        main KV lives on host. B/I use code-level defaults (see ModelRunner._PREFILL_OFFLOAD_*).
                        (default: False)
```

The main parameters are described as follows:

| Parameter                               | Category | Optional/Required | Description |
| --------------------------------------  | --- | --- | --- |
| --device                                | Options | Optional | Specifies one or more device profile names. When multiple devices are passed, cross-hardware comparison results are output.<br>1. Type: Str or List[Str].<br>2. Reference values: any registered `DeviceProfile` name. For built-in values, see the "Device Types" chapter of the TensorCast user guide.<br>3. Default: `TEST_DEVICE` when not specified.<br>4. You can pass multiple registered `DeviceProfile` names in one run. Duplicate names are removed and the input order is preserved. |
| --input-length                          | Options | Required | Token length of the input prompt.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: none. |
| --output-length                         | Options | Required | Expected output token length.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: none. |
| --mtp-acceptance-rates                  | Options | Optional | Specifies the acceptance rate list for MTP tokens.<br>1. Type: List[Float].<br>2. Value range: a list of float values.<br>3. Default: [0.9, 0.6, 0.4, 0.2].<br>4. It works with the legacy `--num-mtp-tokens` and cannot be mixed with `--speculative-method`/`--num-speculative-tokens`/`--acceptance-length`. It will be deprecated along with the legacy MTP interface in later versions. For the new interface, use `--acceptance-length`. |
| --prefix-cache-hit-rate                 | Options | Optional | Specifies the prefix cache hit rate.<br>1. Type: Float.<br>2. Value range: [0, 1).<br>3. Default: 0.0. |
| --dump-original-results                 | Options | Optional | Dumps the original search results for further analysis.<br>1. Type: Bool.<br>2. Value range: switch parameter.<br>3. Default: `False`. |
| model_id                                | General Options | Required | Model ID or a reviewed absolute local model path.<br>1. Type: Str.<br>2. Reference values: Hugging Face ID, ModelScope ID, or a local absolute path.<br>3. Default: none.<br>4. When you use a remote model ID, remote code may be executed through `trust_remote_code=True`. |
| --num-devices                           | General Options | Optional | Specifies the total number of devices involved in the simulation.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --reserved-memory-gb                    | General Options | Optional | Specifies the amount of device memory reserved for system usage on each device, in GB.<br>1. Type: Float.<br>2. Value range: non-negative. Setting 0 disables system memory reservation.<br>3. Default: 10.0. |
| --log-level                             | General Options | Optional | Specifies the log level.<br>1. Type: Str.<br>2. Reference values: `debug`, `info`, `warning`, `error`, `critical`.<br>3. Default: `error`. |
| --compile                               | Model & Quantization Options | Optional | Invokes `torch.compile()` on the model before inference.<br>1. Type: Bool.<br>2. Value range: switch parameter.<br>3. Default: `False`. |
| --compile-allow-graph-break             | Model & Quantization Options | Optional | Allows graph breaks during `torch.compile()`.<br>1. Type: Bool.<br>2. Value range: switch parameter.<br>3. Default: `False`. |
| --num-mtp-tokens                        | Model & Quantization Options | Optional | Specifies the MTP token count candidates. You can pass one or more values for the search. 0 means disabled.<br>1. Type: List[Int] (`nargs="+"`).<br>2. Value range: each candidate is 0 to 9. You can pass multiple values at once, for example, `--num-mtp-tokens 0 1 2`.<br>3. Default: equivalent to 0 (MTP disabled) when not specified.<br>4. When you pass a single value, that MTP configuration is fixed. When you pass multiple values, the optimizer searches the candidate combinations during throughput optimization and multiplies them by the TP/EP/MOE-DP search combinations.<br>5. Only models with MTP capability are supported. Each candidate value cannot exceed `len(--mtp-acceptance-rates) + 1` (the default acceptance rate list has a length of 4. Therefore, the upper limit is 5. When the value is exceeded, the runtime reports `exceed the supported mtp_acceptance_rate length`).<br>6. Legacy MTP entry. It cannot be mixed with `--speculative-method`/`--num-speculative-tokens`/`--acceptance-length`.<br>7. You are advised to use the unified interface `--speculative-method mtp --num-speculative-tokens` (semantically equivalent, also supports multi-value search). This parameter still works alone for compatibility, but will be gradually deprecated in later versions. |
| --speculative-method                    | Model & Quantization Options | Optional | Specifies the speculative decoding method.<br>1. Type: Str.<br>2. Values: `mtp`, `dflash`, or `dspark`.<br>3. Default: not specified (disabled).<br>4. New speculative entry. It cannot be mixed with the legacy MTP entry `--num-mtp-tokens`/`--mtp-acceptance-rates`.<br>5. When you select `mtp`, you must also configure `--num-speculative-tokens`. This parameter is the prerequisite switch for subordinate parameters such as `--num-speculative-tokens`/`--acceptance-length`. |
| --num-speculative-tokens                | Model & Quantization Options | Optional | Speculative token count/depth `n` (excluding anchor/bonus).<br>1. Type: List[Int] (`nargs="+"`, supports multi-value search).<br>2. Semantics: when `>= 1`, the internal `block_size = n + 1`. When **omitted**, dflash/dspark use the builtin or external config. When `--speculative-method` is set, any candidate that is an **explicit** 0 causes a parse failure (to disable, omit `--speculative-method`). For `mtp`, `n` is the MTP token count and must be configured explicitly.<br>3. Default: not specified.<br>4. Requires `--speculative-method`. When you pass multiple values, the optimizer searches them together with TP/EP/MOE-DP. |
| --acceptance-length                     | Model & Quantization Options | Optional | Acceptance length used for decode throughput folding.<br>1. Type: Float.<br>2. Value range: non-negative. It is uniformly clamped to `n` (`= block_size - 1`) for all methods.<br>3. Default: 5.0.<br>4. Requires `--speculative-method`. It does not participate in graph construction and only affects decode latency folding. |
| --dspark-markov-rank                    | Model & Quantization Options | Optional | Markov embedding dimension.<br>1. Type: Int.<br>2. Value range: non-negative integer. 0 disables MarkovHead.<br>3. Default: 256.<br>4. Requires `--speculative-method dspark`. |
| --dspark-markov-head                    | Model & Quantization Options | Optional | Markov head type.<br>1. Type: Str.<br>2. Reference values: `vanilla`, `gated`, `rnn`.<br>3. Default: `vanilla`.<br>4. Requires `--speculative-method dspark`. |
| --num-draft-layers                      | Model & Quantization Options | Optional | Overrides the draft `num_hidden_layers`.<br>1. Type: Int.<br>2. Value range: non-negative integer. 0 means using the configuration default.<br>3. Default: 0.<br>4. Requires `--speculative-method dflash` or `dspark` (cannot be used with `mtp`). |
| --draft-model-config-path               | Model & Quantization Options | Optional | Path to an external draft `config.json` (or the directory containing it).<br>1. Type: Str.<br>2. Default: `None` (uses `tensor_cast/runtime_configs/draft_configs/dflash_draft_builtin.json`).<br>3. Requires `--speculative-method dflash` or `dspark` (cannot be used with `mtp`).<br>4. For field descriptions, see [`tensor_cast/runtime_configs/draft_configs/README.md`](../../../tensor_cast/runtime_configs/draft_configs/README.md). |
| --quantize-linear-action                | Model & Quantization Options | Optional | Specifies the quantization mode for linear layers.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `W8A8_DYNAMIC`.<br>4. Linear layers owned by the draft are not quantized.<br>5. Values can use any case and underscore/hyphen style, for example, `W8A8_DYNAMIC` and `w8a8-dynamic`, or `DISABLED` and `disabled`. |
| --quantize-non-expert-linear-action     | Model & Quantization Options | Optional | Specifies a separate quantization mode for non-expert linear layers, such as attention projections, dense MLP layers, and shared experts.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `DISABLED`.<br>4. Mainly intended for DeepSeek V4-style MoE models. Routed MoE experts still use `--quantize-linear-action`.<br>5. Values can use any case and underscore/hyphen style, for example, `DISABLED` and `disabled`. |
| --mxfp4-group-size                      | Model & Quantization Options | Optional | Specifies the group size for mxfp4 quantization.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 32. |
| --quantize-attention-action             | Model & Quantization Options | Optional | Specifies the quantization mode for the KV cache.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `INT8`, `FP8`.<br>3. Default: `DISABLED`.<br>4. Values can use any case and underscore/hyphen style, for example, `DISABLED` and `disabled`, or `INT8` and `int8`. |
| --tp-sizes                              | Model & Quantization Options | Optional | Enables TP search and can explicitly specify the TP value range.<br>1. Type: List[Int].<br>2. Value range: a list of positive integers.<br>3. Default: `None`. When the argument is passed without values, powers of 2 up to `world_size` are searched by default. |
| --ep-sizes                              | Model & Quantization Options | Optional | Enables EP search and can explicitly specify the EP value range.<br>1. Type: List[Int].<br>2. Value range: a list of positive integers.<br>3. Default: `None`. When the argument is passed without values, powers of 2 up to `world_size` are searched by default. |
| --moe-dp-sizes                          | Model & Quantization Options | Optional | Enables MoE-DP search and can explicitly specify the MoE-DP value range.<br>1. Type: List[Int].<br>2. Value range: a list of positive integers.<br>3. Default: `None`. When the argument is passed without values, powers of 2 up to `world_size` are searched by default. |
| --pp-sizes                              | Model & Quantization Options | Optional | Enables pipeline parallelism (PP) search and can explicitly specify the PP value range.<br>1. Type: List[Int] (`nargs="*"`).<br>2. Value range: a list of positive integers. Each value must be a positive integer no greater than `num_devices` and no greater than the model `num_hidden_layers`.<br>3. Default: `None`. When not specified, PP is fixed at 1 (backward compatible, using the original TP/EP/DCP search path). When the argument is passed without values, powers of 2 up to `num_devices` (1, 2, 4, ...) are searched by default.<br>4. When specified, the optimizer enters the PP-aware search path: it derives `dp = num_devices // (tp * pp)` and `moe_tp = (num_devices // pp) // (ep * moe_dp)` with stage-local arithmetic, and uses a forward blocking pipeline scheduler to calculate the makespan, bubble ratio, and schedule-aware throughput of PP>1 candidates. PP and DCP can be searched together (an explicit `--dcp-sizes` is not discarded).<br>5. The following scenarios do not support PP>1: VL/multimodal models and variable-length input distributions (`length_distribution`). When these scenarios are matched, the optimizer skips the candidate and logs a warning. PP and speculative decoding (`--speculative-method mtp/dflash/dspark`) can be searched together, and the proposal/draft computation of speculative decoding runs on the last stage of the pipeline. |
| --pp-layer-partitions                   | Model & Quantization Options | Optional | Explicitly specifies the layer partition of each PP stage.<br>1. Type: Str (JSON list of lists).<br>2. Format: for example, `'[[31,30],[16,15,15,15]]'`. The length of each inner list must equal the corresponding `pp_size`, and the sum of the elements of all inner lists must equal the model `num_hidden_layers` (DeepSeek-V3.1 has 61 layers, 31+30=61, 16+15+15+15=61). The optimizer automatically matches inner lists to the corresponding `pp_size` by length.<br>3. Default: `None`. When not specified, even splitting is used (the remainder goes to the preceding stages so that the last stage does not carry both norm and lm_head).<br>4. Takes effect only when PP>1. |
| --enable-shared-expert-tp               | Model & Quantization Options | Optional | Enables vLLM-style tensor parallelism for shared experts.<br>1. Type: Bool.<br>2. Value range: switch parameter.<br>3. Default: `False`.<br>4. Shared experts use dense MLP TP and defer the `down_proj` reduction. |
| --compilation-config                    | Model & Quantization Options | Optional | Enables the specified compilation features as needed.<br>1. Type: List[Str].<br>2. Value range: `enable_multistream`, `enable_sequence_parallel`, `enable_matmul_allreduce`, `enable_dispatch_ffn_combine`. You can pass multiple values separated by spaces, for example, `--compilation-config enable_sequence_parallel enable_dispatch_ffn_combine`.<br>3. Default: `None`. When not specified, all compilation features remain disabled.<br>4. Since PR #573, this parameter uniformly replaces the original separate switches such as `--enable_sequence_parallel`/`--enable_dispatch_ffn_combine`.<br>5. Values can use any case and underscore/hyphen style, for example, `enable_multistream` and `enable-multistream`. |
| --word-embedding-tp                     | Model & Quantization Options | Optional | Enables word embedding tensor parallelism and specifies the parallel mode.<br>1. Type: Str.<br>2. Reference values: `col`, `row`.<br>3. Default: `None`, meaning embedding TP is disabled. |
| --performance-model                     | Performance Model Options | Optional | Specifies the performance model type.<br>1. Type: Str.<br>2. Reference values: `analytic`, `profiling`.<br>3. Default: `analytic`.<br>4. `profiling` mode must be used with `--profiling-database-path`. |
| --profiling-database-path               | Performance Model Options | Conditionally required | Specifies the measured operator CSV database directory used by profiling mode.<br>1. Type: Str.<br>2. Value range: a database directory path, for example, `tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/`.<br>3. Default: `None`. It is required when you use `--performance-model profiling`. |
| --chrome-trace-file                     | Debug Options | Optional | Generates a Chrome Trace file for visual analysis of operator-level performance.<br>1. Type: Str.<br>2. Reference values: a trace file path, for example, `trace.json`.<br>3. Default: `None`, meaning no Chrome Trace file is generated. |
| --ttft-limit                            | Service Options | Optional | Specifies the TTFT constraint under which to search for the best throughput.<br>1. Type: Float.<br>2. Value range: positive, in ms.<br>3. Default: `None`, meaning no TTFT limit. |
| --tpot-limit                            | Service Options | Optional | Specifies the TPOT constraint under which to search for the best throughput.<br>1. Type: Float.<br>2. Value range: positive, in ms.<br>3. Default: `None`, meaning no TPOT limit. |
| --target-qps                            | Service Options | Optional | Specifies the target request QPS (req/s). Setting it triggers capacity planning mode, which back-calculates the minimum device count that meets this QPS under the given SLO.<br>1. Type: Float.<br>2. Value range: positive number.<br>3. Default: `None`.<br>4. For how it works with `--num-devices`, and for the constraint that exactly one phase SLO is required under `--disagg`, see [2.5 Capacity Planning Scenario](#25-capacity-planning-scenario). |
| --max-batched-tokens                    | Service Options | Optional | Specifies the maximum number of batched tokens for a single data parallelism (DP) replica in one prefill or mixed prefill/decode step.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. Automatic mode first uses `4 * input_length` and falls back to `2 * input_length` and then `1 * input_length` on prefill OOM. |
| --batch-range                           | Service Options | Optional | Specifies the batch size search range.<br>1. Type: List[Int].<br>2. Format: `[min max]` or `[max]`.<br>3. Default: `None`. When `min` is not specified, the search starts from 1. When `max` is not specified, no upper limit is set. |
| --serving-cost                          | Service Options | Optional | Specifies the serving cost for cost-related metric calculation.<br>1. Type: Float.<br>2. Value range: non-negative.<br>3. Default: 0. |
| --disagg                                | Service Options | Optional | Enables PD disaggregation mode.<br>1. Type: Bool.<br>2. Value range: switch parameter.<br>3. Default: `False`. |
| --jobs                                  | Service Options | Optional | Specifies the number of parallel search jobs.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 8. |
| --max-search-combinations               | Service Options | Optional | Outputs a warning when the number of TP/EP/MOE-DP/MTP search combinations exceeds this value.<br>1. Type: Int.<br>2. Value range: non-negative integer. Setting it to 0 disables the warning.<br>3. Default: 100. |
| --concurrency-search-strategy           | Service Options | Optional | Specifies the concurrency search strategy.<br>1. Type: Str.<br>2. Reference values: `exponential`, `linear_exponential`.<br>3. Default: `exponential`.<br>4. Values can use any case and underscore/hyphen style, for example, `linear_exponential` and `linear-exponential`. |
| --image-batch-size                      | MultiModal Options | Optional | Specifies the number of images per request.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. When not set, the batch size is reused. |
| --image-height                          | MultiModal Options | Optional | Specifies the height of the input images.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --image-width                           | MultiModal Options | Optional | Specifies the width of the input images.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --prefill-devices-per-instance          | PD Ratio Optimization Options | Conditionally required | Required when PD ratio optimization is enabled. Specifies the number of devices per prefill instance.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: none.<br>4. Determines the parallelism configuration search space for the prefill phase. |
| --decode-devices-per-instance           | PD Ratio Optimization Options | Conditionally required | Required when PD ratio optimization is enabled. Specifies the number of devices per decode instance.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: none.<br>4. Determines the parallelism configuration search space for the decode phase. |
| --enable-optimize-prefill-decode-ratio  | PD Ratio Optimization Options | Optional | Enables the prefill/decode instance ratio optimization mode.<br>1. Type: Bool.<br>2. Value range: switch parameter.<br>3. Default: `False`.<br>4. Cannot be used together with `--disagg`. |
| --enable-decode-kv-offload              | KV Cache Offload Options | Optional | Enables Decode sparse KV cache offload modeling.<br>1. Type: Bool.<br>2. Value range: switch parameter.<br>3. Default: `False`.<br>4. The sparse window top-k is resolved from the model checkpoint config (`index_topk`, or `topk_limit` + `index_n_heads`). Enabling is rejected when the config declares none, and dense/GQA models (no indexer cache) are rejected as well. |
| --decode-offload-buffer-multiplier      | KV Cache Offload Options | Optional | Hot buffer capacity = multiplier × top-k.<br>1. Type: Float.<br>2. Value range: ≥ 1 (the buffer must hold at least one full top-k window for a deployable positive hit rate).<br>3. Default: 2.0 (upstream measures an 80% to 90% hit rate at 2×top-k). |
| --decode-offload-hit-rate               | KV Cache Offload Options | Optional | Hot buffer hit rate, determining the per-step miss-row reload volume.<br>1. Type: Float.<br>2. Value range: [0, 1].<br>3. Default: 0.85 (conservative lower bound).<br>4. Even at `hit_rate = 1` each step still pays the new-KV write-back cost — offload is never free. |
| --enable-prefill-kv-offload             | KV Cache Offload Options | Optional | Enables Prefill layerwise KV cache offload modeling (N layers share B physical buffers).<br>1. Type: Bool.<br>2. Value range: switch parameter.<br>3. Default: `False`.<br>4. B/I use code-level defaults (`B=2`, `I=0`, see `ModelRunner._PREFILL_OFFLOAD_*`). It is mutually exclusive with PP>1. |
| --host-memory-available-gb              | KV Cache Offload Options | Optional | Host memory budget for the offload KV pool (GB).<br>1. Type: Float.<br>2. Value range: ≥ 0.<br>3. Default: 0 (capacity check disabled).<br>4. When > 0, candidates whose offloaded KV pool exceeds the budget fold into the OOM early-stop path (the search shrinks and keeps feasible rows) instead of returning undeployable results. |

### Search Dimensions and Ranges

`throughput_optimizer` determines the search dimensions based on the provided search arguments:

- `--tp-sizes`: Enables TP search.
- `--ep-sizes`: Enables EP search.
- `--moe-dp-sizes`: Enables MoE-DP search.
- `--pp-sizes`: Enables pipeline parallelism (PP) search. When not specified, PP is fixed at 1 (backward compatible). For PP>1, the optimizer derives DP/MoE-TP with stage-local arithmetic and evaluates the makespan, bubble, and throughput with a forward blocking scheduler. PP can be searched together with DCP (an explicit `--dcp-sizes` is not discarded). You can use `--pp-layer-partitions` to specify an explicit layer partition.

Rules:

- If no search argument is provided, only TP search is performed by default with the default range.
- For dimensions not selected for search, fixed defaults are used:
  - `tp = num_devices`
  - `ep = num_devices`
  - `moe-dp = 1`
  - `pp = 1`
- If a search argument is provided, you must explicitly give the values. The commonly used range is:
  `powers of 2 up to world_size`
  (for example, when `num_devices=8`, you can set [1, 2, 4, 8]).

Examples:

```bash
# Search TP only (explicit range)
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limit 50 --tp-sizes 1 2 4 8

# Search TP/EP (MoE-DP fixed to 1)
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limit 50 --tp-sizes 1 2 4 8 --ep-sizes 1 2 4 8

# Search TP/EP/MoE-DP
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limit 50 --tp-sizes 1 2 4 8 --ep-sizes 1 2 4 8 --moe-dp-sizes 1 2 4 8

# Search EP only (explicit range)
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limit 50 --ep-sizes 1 2 4 8
```

### Performance Model Selection

`throughput_optimizer` uses the analytic (Roofline) model to estimate operator latency by default. You can switch to the profiling model backed by measured operator CSV data through `--performance-model`:

- `--performance-model analytic` (default): a pure analytic Roofline model that requires no extra data
- `--performance-model profiling`: models with measured operator data. You **must** also specify `--profiling-database-path <directory>`, otherwise the tool reports an error at startup. When the shape of an operator is missing from the CSV, the optimizer first tries interpolation, and falls back to the analytic model if it still cannot find a match.

Example:

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device ATLAS_800_A3_752T_128G_DIE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limit 50 \
    --performance-model profiling \
    --profiling-database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/
```

## 5 Additional Notes

### 5.1 How to Calculate the Performance Metrics in Aggregation Mode

- TTFT:

  When `effective_input_length <= max_batched_tokens`, we keep the original full-prefill formula.
  We get average `ttft = sum_for_ttft / concurrency`. For `sum_for_ttft`, we assume the prefill
  phase concurrency is the total token budget of all DP replicas divided by the effective input length.
  Therefore, `prefill_concurrency = (max_batched_tokens * dp_size) // effective_input_length`. Here,
  `prefill_concurrency` indicates the number of requests that a single prefill scheduling step can process at the same time, not the batch
  size of the entire workload. Requests are processed step by step at this concurrency, and `calc_nums_for_ttft` is the number of required prefill scheduling steps. We can get the total `ttft` time as follows:

  `sum_for_ttft = (prefill_latency * prefill_concurrency) * (1 + calc_nums_for_ttft) * calc_nums_for_ttft / 2`

  For example, if we have 3 DP replicas and 36 requests, and `max_batched_tokens` is 8192, `input_length` is 2048,
  then `prefill_concurrency` is 12, and 36 requests are completed in 3 prefill scheduling steps. The TTFT of the first 12 requests
  is 1 `prefill_latency`, the second batch is 2 `prefill_latency`, and the third batch is 3 `prefill_latency`. Therefore:

  `sum_for_ttft = (prefill_latency * 12) * (1 + 3) * 3 / 2 = prefill_latency * 72`

  `ttft = sum_for_ttft / 36 = 2 * prefill_latency`

  When `effective_input_length > max_batched_tokens`, the optimizer automatically splits prefill
  into multiple chunks. The current version uses a fixed decode-first mixed scheduler with 15% token
  budget slack. It does not expose a CLI parameter for scheduler selection.

- TPOT:

  We don't consider the bubble time in TPOT calculation.

  `tpot = (ttft + decode_latency * output_length) / output_length`

  When `--speculative-method` is enabled, the decode latency is folded by `--acceptance-length` (already clamped to `n = block_size - 1` for the method) before it participates in the TPOT calculation. The prefill latency is not affected by the acceptance folding.

- Output Throughput
  `output_throughput = 1000 * (output_length * concurrency) / (ttft + tpot * output_length)`

### 5.2 How to Calculate the Performance Metrics in PD Ratio Mode

PD ratio mode uses Queries Per Second (QPS) as the primary metric for matching prefill and decode capacities:

- **Prefill QPS (P QPS)**:

  P QPS represents the request processing capacity of a single prefill instance.

  `P QPS = p_concurrency / prefill_phase_makespan_ms * 1000` (req/s)

  Where:
  - `p_concurrency`: batch size (number of concurrent requests) in prefill phase
  - `prefill_phase_makespan_ms`: drain duration of the prefill phase in milliseconds. When an older result row lacks this column, the tool falls back to `ttft`
  - PP>1 (pipeline parallelism) rows: the throughput column is calculated from the steady-state wave interval, and P QPS accordingly takes `token/s / input_length` (full prompt token basis, consistent with the throughput numerator. When the prefix cache hits, the throughput numerator still counts the full input. Therefore, the denominator must not use the cache-miss length `effective_input_length`)

- **Decode QPS (D QPS)**:

  D QPS represents the request processing capacity of a single decode instance.

  `D QPS = d_concurrency / (tpot * max(output_length - 1, 1)) * 1000` (req/s)

  Where:
  - `d_concurrency`: batch size (number of concurrent requests) in decode phase
  - `tpot`: time-per-output-token in milliseconds
  - `max(output_length - 1, 1)`: number of decode tokens after the first token has been produced

- **PD Ratio**:

  PD Ratio indicates the optimal ratio between prefill and decode instances to achieve balanced throughput.

  `PD Ratio = D QPS / P QPS`

  Interpretation:
  - PD Ratio = 1.0: One prefill instance can feed one decode instance
  - PD Ratio = 2.0: Each decode instance requires about two prefill instances
  - PD Ratio = 0.5: One prefill instance can support about two decode instances

- **Instance Distribution**:

  When `--num-devices` is specified, the optimal number of prefill and decode instances is calculated:

  1. Calculate total instances that fit within device budget:
     `max_p_inst = total_devices / p_devices_per_instance`
     `max_d_inst = total_devices / d_devices_per_instance`

  2. Find the P:D instance combination that:
     - Maximizes the allocated `Balanced QPS` first
     - Uses more devices when `Balanced QPS` is the same
     - Approaches the theoretical PD ratio as closely as possible when device usage is also the same
     - Fits within the total device budget

  The allocated cluster throughput is calculated as follows:

  `Allocated P QPS = number of P instances * P QPS`

  `Allocated D QPS = number of D instances * D QPS`

  `Balanced QPS = min(Allocated P QPS, Allocated D QPS)`

### 5.3 How to Calculate the Minimum Device Count Under Capacity Planning

Capacity planning reuses the optimization results of the optimizer. It first runs the optimizer at candidate device counts to obtain the request QPS per instance, and then linearly back-calculates the minimum device count that meets the target QPS.

- **Aggregation/PD disaggregation mode**:

  First calculate the request QPS per instance, and then scale it by the number of instances:

  - Request QPS per instance:
    - Aggregation: `per_replica_qps = token/s ÷ output_length ÷ dp_size`
    - PD disaggregation Prefill: `per_replica_qps = concurrency ÷ prefill_phase_makespan_ms × 1000`. PP>1 rows use the steady-state basis `token/s ÷ input_length` (full prompt token basis, consistent with the throughput numerator)
    - PD disaggregation Decode: `per_replica_qps = concurrency ÷ (tpot × output_length) × 1000`
  - Required number of instances: `replicas_needed = ceil(target_qps ÷ per_replica_qps)`
  - Total device count: `total_gpus = replicas_needed × per_replica_gpus`
  - Per-device QPS: `per_die_qps = per_replica_qps ÷ per_replica_gpus`

  Here `per_replica_gpus` is the number of devices occupied by each instance (usually `num_devices ÷ dp_size` in aggregation/disaggregation mode). When EP spans multiple DP replicas (`num_devices ÷ dp_size < ep_size ≤ num_devices`), the entire instance serves as the replication unit. In this case, `per_replica_gpus = num_devices`, and `per_replica_qps` is no longer divided by `dp_size`.

- **PD ratio mode**: The P and D sides scale independently, and the QPS actually achieved is the smaller value of the two sides:

  - `p_inst = ceil(target_qps ÷ p_qps)`, `d_inst = ceil(target_qps ÷ d_qps)`
  - `total_gpus = p_inst × p_dev + d_inst × d_dev`
  - `achieved_qps = min(p_inst × p_qps, d_inst × d_qps)`

- **Sorting rule**: Sort by `total_gpus` in ascending order. For the same device count, sort by `per_die_qps` in descending order, and take the first entry as the minimum device count plan. The back-calculation comparison table shows all candidates in the same order.

> The back-calculation process keeps full precision and only formats the values to two decimal places when rendering the output. If you round the per-instance QPS to two decimal places before the back-calculation, the error of `achieved_qps` grows with the number of instances. For example, at 0.014 req/s per instance and a target of 1 req/s, you need 72 instances, and the correct result is 1.008 req/s, whereas rounding reports 0.72 req/s in error.

# Throughput Optimizer

## Introduction

Throughput optimizer is a tool to optimize the throughput under SLO (Service Level Objective) constraints. It automatically searches for the optimal model configuration (parallelism strategy, batch size) to maximize token throughput under specified SLO constraints (e.g., limits on TTFT, TPOT).

## Quick Start

## Run in aggregation mode

Aggregation mode optimizes throughput for a combined Prefill-Decode serving architecture where both phases run on the same instance. The optimizer searches across all possible TP (Tensor Parallelism) and DP (Data Parallelism) configurations to find the best throughput under SLO (Service Level Objective) constraints.

### Example

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --tpot-limits 50
```

### With Prefix Cache

If you want to estimate aggregation throughput with prefix cache enabled, add `--prefix-cache-hit-rate`:

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --tpot-limits 50 \
    --prefix-cache-hit-rate 0.5
```

### Constraints

- `--max-fill-tokens` must be greater than `--input-length` , which determines the maximum batch size in the Prefill phase.

## Run in disaggregation mode

Disaggregation mode separates Prefill and Decode phases into independent optimization runs. This is useful when you need to characterize each phase independently or when planning disaggregated serving deployments.

### Prerequisites

To enable disaggregation mode, you must provide:

- `--disagg`: Enable disaggregation mode

### Prefill Mode

Optimizes Prefill phase throughput under TTFT (Time-to-First-Token) constraints. `--disagg` flag and `--ttft-limits` flag should be set in this mode.

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
    --ttft-limits 2000
```

### Decode Mode

Optimizes Decode phase throughput under TPOT (Time-per-Output-Token) constraints. `--disagg` flag and `--tpot-limits` flag should be set in this mode.

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
    --tpot-limits 50
```

## Run in PD Ratio Optimization mode

PD (Prefill-Decode) Ratio Optimization mode enables independent optimization of Prefill and Decode phases, then combines the results to find the optimal P/D instance ratio for maximum system throughput. This mode is particularly useful for disaggregated serving architectures where Prefill and Decode instances can be scaled independently.

### Prerequisites

To enable PD ratio optimization, you must provide:

- `--enable-optimize-prefill-decode-ratio`: Enable PD ratio optimization mode
- `--prefill-devices-per-instance`: Number of devices per Prefill instance
- `--decode-devices-per-instance`: Number of devices per Decode instance

### Example

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
    --device TEST_DEVICE \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --enable-optimize-prefill-decode-ratio \
    --prefill-devices-per-instance 16 \
    --decode-devices-per-instance 16 \
    --log-level info
```

### Constraints

- `--enable-optimize-prefill-decode-ratio` cannot be used together with `--disagg`
- Both `--prefill-devices-per-instance` and `--decode-devices-per-instance` must be specified when PD ratio optimization is enabled

## Search dimensions and ranges

`throughput_optimizer` searches dimensions based on which search arguments are provided:

- `--tp-sizes`: enable TP search
- `--ep-sizes`: enable EP search
- `--moe-dp-sizes`: enable MOE-DP search

Rules:

- If no search argument is provided, default behavior is TP-only search with default range.
- For dimensions not selected for search, fixed defaults are used:
  - `tp = num_devices`
  - `ep = num_devices`
  - `moe-dp = 1`
- If a search argument is provided without values, that dimension uses default range:
  `powers of 2 up to world_size`
  (for example, when `num_devices=8`, default range is `[1, 2, 4, 8]`).

Examples:

```bash
# Search TP only (explicit range)
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50 --tp-sizes 1 2 4 8

# Search TP/EP (MOE-DP fixed to 1)
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50 --tp-sizes 1 2 4 8 --ep-sizes 1 2 4 8

# Search TP/EP/MOE-DP
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50 --tp-sizes 1 2 4 8 --ep-sizes 1 2 4 8 --moe-dp-sizes 1 2 4 8

# Search EP only with default range (argument provided without values)
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50 --ep-sizes
```

## Result Information

The script will output the performance metrics, including throughput, TTFT, TPOT, and concurrency. Like the example below:

```bash
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration:
    Model: Qwen/Qwen3-32B
    Quantize Linear action: W8A8_DYNAMIC
    Quantize Attention action: DISABLED
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

## Parameters

```bash
Options:
  --input-length INPUT_LENGTH
                        The input length of the prompt. (default: None)
  --output-length OUTPUT_LENGTH
                        The expected output length. (default: None)
                        The device type for benchmarking. (default: None)
  --mtp-acceptance-rate MTP_ACCEPTANCE_RATE [MTP_ACCEPTANCE_RATE ...]
                        Acceptance rate list for MTP (default: [0.9, 0.6, 0.4, 0.2])
  --dump-original-results
                        If set, dump the original results for analysis. (default: False)
                
General Options:
  model_id              The model identifier, which can be: 1) A Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b-hf') to load from the Hub;    
                        2) A local directory path containing a diffusers model (must include 'transformer/config.json').
  --device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE}
                        Specifies the target device profile to use for benchmarking and simulation. Must be a valid device name as defined in      
                        DeviceProfile. The default device 'TEST_DEVICE' is used for standard simulation runs. (default: TEST_DEVICE)
  --num-devices NUM_DEVICES
                        Specifies the total number of devices/processes to use. Must be a positive integer. A value of 1 indicates single-device   
                        execution. (default: 1)
  --reserved-memory-gb RESERVED_MEMORY_GB
                        Amount of device memory (in gigabytes) reserved for system usage and unavailable for application. Set to 0 to disable      
                        memory reservation. (default: 10.0)
  --log-level {debug,info,warning,error,critical}
                        Specifies the verbosity level for log output. Available levels: 'debug' (most verbose), 'info', 'warning', 'error',        
                        'critical' (least verbose). (default: error)

Model & Quantization Options:
  --compile             If set, invoke torch.compile() on the model before inference. (default: False)
  --compile-allow-graph-break
                        If set, invoke torch.compile() on the model before inference. (default: False)
  --num-mtp-tokens {0,1,2,3,4,5,6,7,8,9}
                        Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek (default: 0)
  --quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}
                        Quantize all linear layers in the model from choices (currently only support symmetric quant) (default: W8A8_DYNAMIC)
  --mxfp4-group-size MXFP4_GROUP_SIZE
                        Group size for MXFP4 quantization (default: 32)
  --quantize-attention-action {DISABLED,INT8,FP8}
                        Quantize the KV cache with the given action (default: DISABLED)
  --tp-sizes [TP_SIZES ...]
                        Enable TP search. Optional explicit TP sizes. If no value is provided, defaults to powers of 2 up to world_size. (default: None)
  --ep-sizes [EP_SIZES ...]
                        Enable EP search. Optional explicit EP sizes. If no value is provided, defaults to powers of 2 up to world_size. (default: None)
  --moe-dp-sizes [MOE_DP_SIZES ...]
                        Enable MOE-DP search. Optional explicit MOE-DP sizes. If no value is provided, defaults to powers of 2 up to world_size. (default: None)

Service Options:
  --ttft-limits TTFT_LIMITS
                        TTFT constraints under which to search for the best throughput. None means no constraint. (default: None)
  --tpot-limits TPOT_LIMITS
                        TPOT constraints under which to search for the best throughput. None means no constraint. (default: None)
  --max-prefill-tokens MAX_PREFILL_TOKENS
                        Max prefill tokens (default: 8192)
  --prefix-cache-hit-rate PREFIX_CACHE_HIT_RATE
                        Prefix cache hit rate for token-level prefill reuse approximation. Valid range: [0, 1). (default: 0.0)
  --batch-range BATCH_RANGE [BATCH_RANGE ...]
                        Batch size range: [min max] or [max] (default: 1 for min, no limit for max) (default: None)
  --serving-cost SERVING_COST
                        Serving cost represents the cost of service delivery (default: 0)
  --disagg              If set, run disaggregation mode. disagg means disaggregation mode. (default: False)
  --jobs JOBS           Number of parallel jobs. (default: 8)

PD Ratio Optimization Options:
  --enable-optimize-prefill-decode-ratio
                        Enable PD (Prefill-Decode) ratio optimization mode. This mode independently
                        optimizes Prefill and Decode phases, then combines results to find the optimal
                        P/D instance ratio. Cannot be used together with --disagg. (default: False)
  --prefill-devices-per-instance PREFILL_DEVICES_PER_INSTANCE
                        Number of devices per Prefill instance. Required when --enable-optimize-prefill-decode-ratio
                        is set. Determines the parallelism configuration search space for Prefill phase.
  --decode-devices-per-instance DECODE_DEVICES_PER_INSTANCE
                        Number of devices per Decode instance. Required when --enable-optimize-prefill-decode-ratio
                        is set. Determines the parallelism configuration search space for Decode phase.
```

## How to calculate the performance metrics in aggregation mode

- TTFT:

  We get average `ttft = sum_for_ttft / concurrency`. For sum_for_ttft, we assume the prefill batch size is the max prefill tokens divided by effective input length.
  So `prefill_batch_size = max_prefill_tokens // effective_input_length`. And request was processed in
  prefill_batch_size steps one by one. We can get the total ttft time as follows:

  `sum_for_ttft = (prefill_latency * prefill_batch_size) * (1 + calc_nums_for_ttft)) * (calc_nums_for_ttft) / 2`

  For example, if we have 12 requests, and max_prefill_tokens is 8192, input_length is 2048,
  then prefill_batch_size is 4. And 12 requests was processed in 3 steps.
  so 
  
  `sum_for_ttft = (prefill_latency * 4 ) * (1 + 3) * 3 / 2`

  `ttft = sum_for_ttft / 12`

- TPOT:

  We don't consider the bubble time in TPOT calculation.

  `tpot = (ttft + decode_latency * output_length) / output_length`

- Output Throughput
  `output_throughput = 1000 * (output_length * concurrency) / (ttft + tpot * output_length)`

## How to calculate the performance metrics in PD ratio mode

PD ratio mode uses QPS (Queries Per Second) as the primary metric for matching Prefill and Decode capacities:

- **Prefill QPS (P QPS)**:

  P QPS represents the request processing capacity of a single Prefill instance.

  `P QPS = p_concurrency / ttft * 1000` (req/s)

  Where:
  - `p_concurrency`: The batch size (number of concurrent requests) in Prefill phase
  - `ttft`: Time-to-first-token in milliseconds

- **Decode QPS (D QPS)**:

  D QPS represents the request processing capacity of a single Decode instance.

  `D QPS = d_concurrency / (tpot * output_length) * 1000` (req/s)

  Where:
  - `d_concurrency`: The batch size (number of concurrent requests) in Decode phase
  - `tpot`: Time-per-output-token in milliseconds
  - `output_length`: Expected output token length

- **PD Ratio**:

  PD Ratio indicates the optimal ratio between Prefill and Decode instances to achieve balanced throughput.

  `PD Ratio = D QPS / P QPS`

  Interpretation:
  - PD Ratio = 1.0: One Prefill instance can feed one Decode instance
  - PD Ratio = 2.0: One Prefill instance can feed two Decode instances
  - PD Ratio = 0.5: Two Prefill instances are needed to feed one Decode instance

- **Instance Distribution**:

  When `--num-devices` is specified, the optimal number of Prefill and Decode instances is calculated:

  1. Calculate total instances that fit within device budget:
     `max_p_inst = total_devices / p_devices_per_instance`
     `max_d_inst = total_devices / d_devices_per_instance`

  2. Find the P:D instance combination that:
     - Matches the PD ratio as closely as possible
     - Fits within the total device budget
     - Maximizes overall system throughput

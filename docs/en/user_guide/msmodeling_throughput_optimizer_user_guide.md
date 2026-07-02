# Throughput Optimizer Guide

## 1 Introduction

Throughput optimizer is a tool to optimize the throughput under SLO (Service Level Objective) constraints. It automatically searches for the optimal model configuration (parallelism strategy, batch size) to maximize token throughput under specified SLO constraints (e.g., limits on TTFT, TPOT).

This guide is intended for developers or performance engineers who need to evaluate model serving throughput, parallel strategies, and SLO constraints. Before you start, complete the environment setup in [Quick Start: Environment Setup and First Simulation](../install_guide/msmodeling_install_guide.md), and make sure that the target model configuration can be loaded.

## 2 Main Scenarios

| Mode | Use Case | Key Parameters |
| --- | --- | --- |
| PD Aggregation | Prefill and Decode run in the same instance. Suitable for quick end-to-end throughput evaluation. | `--tpot-limits`, `--ttft-limits` |
| PD Disaggregation | Prefill and Decode are deployed separately. Useful when phase-specific capacity needs to be evaluated. | `--disagg`, `--ttft-limits` or `--tpot-limits` |
| PD Ratio | Plan the instance ratio between Prefill and Decode. | `--enable-optimize-prefill-decode-ratio`, `--prefill-devices-per-instance`, `--decode-devices-per-instance` |

### 2.1 PD Aggregation Scenario

Aggregation mode optimizes throughput for a combined Prefill-Decode serving architecture where both phases run on the same instance. The optimizer searches across all possible TP (Tensor Parallelism) and DP (Data Parallelism) configurations to find the best throughput under SLO (Service Level Objective) constraints.

#### Example

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

#### With Prefix Cache

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

#### Constraints

- `--max-batched-tokens` sets the token budget for one prefill or mixed prefill/decode step. If `effective_input_length` is greater than `max_batched_tokens`, the optimizer automatically splits Prefill into chunks. Set `--max-batched-tokens` to match the serving engine's scheduling budget.

### 2.2 PD Disaggregation Scenario

Disaggregation mode separates Prefill and Decode phases into independent optimization runs. This is useful when you need to characterize each phase independently or when planning disaggregated serving deployments.

#### Prerequisites

To enable disaggregation mode, you must provide:

- `--disagg`: Enable disaggregation mode

#### Prefill Mode

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

#### Decode Mode

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

### 2.3 PD Ratio Scenario

PD (Prefill-Decode) Ratio Optimization mode enables independent optimization of Prefill and Decode phases, then combines the results to find the optimal P/D instance ratio for maximum system throughput. This mode is particularly useful for disaggregated serving architectures where Prefill and Decode instances can be scaled independently.

#### Prerequisites

To enable PD ratio optimization, you must provide:

- `--enable-optimize-prefill-decode-ratio`: Enable PD ratio optimization mode
- `--prefill-devices-per-instance`: Number of devices per Prefill instance
- `--decode-devices-per-instance`: Number of devices per Decode instance

#### Example

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

#### Constraints

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

## 3 Result Information

The script outputs performance metrics (throughput, TTFT, TPOT, concurrency, and mode-specific fields such as QPS or PD ratio). Example:

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

## 4 Parameters

```bash
Options:
  --input-length INPUT_LENGTH
                        The input length of the prompt. (default: None)
  --output-length OUTPUT_LENGTH
                        The expected output length. (default: None)
  --mtp-acceptance-rate MTP_ACCEPTANCE_RATE [MTP_ACCEPTANCE_RATE ...]
                        Acceptance rate list for MTP (default: [0.9, 0.6, 0.4, 0.2])
  --dump-original-results
                        If set, dump the original results for analysis. (default: False)

General Options:
  model_id              Model source. Recommended safe mode: a reviewed absolute local model path. Model id mode also accepts Hugging Face or
                        ModelScope ids, but may execute remote Python code through trust_remote_code=True and is not security-guaranteed.
  --device DEVICE [DEVICE ...]
                        Device profile(s) to evaluate. One or more registered DeviceProfile names.
                        Supported values: TEST_DEVICE, ATLAS_800_A2_376T_64G, ATLAS_800_A2_313T_64G,
                        ATLAS_800_A2_280T_64G, ATLAS_800_A2_280T_64G_PCIE, ATLAS_800_A2_280T_32G_PCIE,
                        ATLAS_800_A3_752T_128G_DIE, ATLAS_800_A3_560T_128G_DIE,
                        ATLAS_800_A3_560T_128G_DIE_ROCE, ATLAS_350_425T_112G, ATLAS_350_425T_84G.
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
                        Specifies the verbosity level for log output. Available levels: 'debug' (most verbose), 'info', 'warning', 'error',
                        'critical' (least verbose). (default: error)

Model & Quantization Options:
  --compile             If set, invoke torch.compile() on the model before inference. (default: False)
  --compile-allow-graph-break
                        If set, allows graph breaks during torch.compile() to improve compilation speed or handle unsupported ops. (default: False)
  --num-mtp-tokens {0,1,2,3,4,5,6,7,8,9}
                        Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek (default: 0)
  --quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}
                        Quantize all linear layers in the model from choices (currently only support symmetric quant) (default: W8A8_DYNAMIC)
  --quantize-non-expert-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}
                        Set a separate quantization type for non-expert linear layers, such as attention projections, dense MLP layers, and shared experts, while routed MoE experts keep --quantize-linear-action. This option is mainly intended for DeepSeek V4-style MoE models. (default: DISABLED)
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
  --max-batched-tokens MAX_BATCHED_TOKENS
                        Max batched tokens for one prefill or mixed prefill/decode step. (default: 8192)
  --prefix-cache-hit-rate PREFIX_CACHE_HIT_RATE
                        Prefix cache hit rate for token-level prefill reuse approximation. Valid range: [0, 1). (default: 0.0)
  --batch-range BATCH_RANGE [BATCH_RANGE ...]
                        Batch size range: [min max] or [max] (default: 1 for min, no limit for max) (default: None)
  --serving-cost SERVING_COST
                        Serving cost represents the cost of service delivery (default: 0)
  --disagg              If set, run disaggregation mode. disagg means disaggregation mode. (default: False)
  --jobs JOBS           Number of parallel jobs. (default: 8)
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

Main parameters:

| Parameter | Category | Required/Optional | Description |
| --- | --- | --- | --- |
| `--device` | Options | Optional | Specifies one or more device profile names. Multiple values enable cross-hardware comparison tables.<br>1. Type: Str or List[Str].<br>2. Reference values: `TEST_DEVICE`, `ATLAS_800_A2_376T_64G`, `ATLAS_800_A2_313T_64G`, `ATLAS_800_A2_280T_64G`, `ATLAS_800_A2_280T_64G_PCIE`, `ATLAS_800_A2_280T_32G_PCIE`, `ATLAS_800_A3_752T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE_ROCE`, `ATLAS_350_425T_112G`, `ATLAS_350_425T_84G`.<br>3. Default: uses `TEST_DEVICE` when omitted.<br>4. Duplicate registered `DeviceProfile` names are removed while preserving input order. |
| `--input-length` | Options | Required | Input prompt token length.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: none. |
| `--output-length` | Options | Required | Expected generated output token length.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: none. |
| `--mtp-acceptance-rate` | Options | Optional | MTP token acceptance rate list.<br>1. Type: List[Float].<br>2. Valid range: float list.<br>3. Default: `[0.9, 0.6, 0.4, 0.2]`. |
| `--prefix-cache-hit-rate` | Options | Optional | Prefix cache hit rate.<br>1. Type: Float.<br>2. Valid range: `[0, 1)`.<br>3. Default: `0.0`. |
| `--dump-original-results` | Options | Optional | Dumps original search results for further analysis.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `model_id` | General Options | Required | Model ID or reviewed local model absolute path.<br>1. Type: Str.<br>2. Reference values: Hugging Face ID, ModelScope ID, or local absolute path.<br>3. Default: none.<br>4. Remote model IDs may execute remote code through `trust_remote_code=True`. |
| `--num-devices` | General Options | Optional | Total number of devices for simulation.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`. |
| `--reserved-memory-gb` | General Options | Optional | Device memory reserved for system use, in GB.<br>1. Type: Float.<br>2. Valid range: non-negative number; set to `0` to disable reservation.<br>3. Default: `10.0`. |
| `--log-level` | General Options | Optional | Log level.<br>1. Type: Str.<br>2. Reference values: `debug`, `info`, `warning`, `error`, `critical`.<br>3. Default: `error`. |
| `--compile` | Model & Quantization Options | Optional | Invokes `torch.compile()` before inference.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--compile-allow-graph-break` | Model & Quantization Options | Optional | Allows graph breaks during `torch.compile()`.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--num-mtp-tokens` | Model & Quantization Options | Optional | Number of MTP tokens. `0` means disabled.<br>1. Type: Int.<br>2. Valid range: `0` to `9`.<br>3. Default: `0`. |
| `--quantize-linear-action` | Model & Quantization Options | Optional | Linear layer quantization mode.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `W8A8_DYNAMIC`. |
| `--quantize-non-expert-linear-action` | Model & Quantization Options | Optional | Separate quantization mode for non-expert linear layers.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `DISABLED`.<br>4. Mainly intended for DeepSeek V4-style MoE models. Routed MoE experts still use `--quantize-linear-action`. |
| `--mxfp4-group-size` | Model & Quantization Options | Optional | MXFP4 quantization group size.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `32`. |
| `--quantize-attention-action` | Model & Quantization Options | Optional | KV cache quantization mode.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `INT8`, `FP8`.<br>3. Default: `DISABLED`. |
| `--tp-sizes` | Model & Quantization Options | Optional | Enables TP search and optionally specifies TP candidates.<br>1. Type: List[Int].<br>2. Valid range: positive integer list.<br>3. Default: `None`; when provided without values, searches powers of 2 up to `world_size`. |
| `--ep-sizes` | Model & Quantization Options | Optional | Enables EP search and optionally specifies EP candidates.<br>1. Type: List[Int].<br>2. Valid range: positive integer list.<br>3. Default: `None`; when provided without values, searches powers of 2 up to `world_size`. |
| `--moe-dp-sizes` | Model & Quantization Options | Optional | Enables MOE-DP search and optionally specifies MOE-DP candidates.<br>1. Type: List[Int].<br>2. Valid range: positive integer list.<br>3. Default: `None`; when provided without values, searches powers of 2 up to `world_size`. |
| `--enable-shared-expert-tp` | Model & Quantization Options | Optional | Enables vLLM-style tensor parallel for shared experts.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`.<br>4. Shared experts use dense MLP TP with delayed `down_proj` reduction. |
| `--enable-sequence-parallel` | Model & Quantization Options | Optional | Enables the sequence parallel graph rewrite pass during compilation.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--enable-dispatch-ffn-combine` | Model & Quantization Options | Optional | Enables dispatch_ffn_combine fusion pattern during compilation.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--word-embedding-tp` | Model & Quantization Options | Optional | Enables word embedding tensor parallel and specifies mode.<br>1. Type: Str.<br>2. Reference values: `col`, `row`.<br>3. Default: `None`, meaning embedding TP is disabled. |
| `--chrome-trace` | Debug Options | Optional | Generates a Chrome Trace file for operator-level performance visualization.<br>1. Type: Str.<br>2. Reference value: trace file path, such as `trace.json`.<br>3. Default: `None`. |
| `--ttft-limits` | Service Options | Optional | TTFT constraint for throughput search.<br>1. Type: Float.<br>2. Valid range: positive number, in ms.<br>3. Default: `None`, meaning no TTFT constraint. |
| `--tpot-limits` | Service Options | Optional | TPOT constraint for throughput search.<br>1. Type: Float.<br>2. Valid range: positive number, in ms.<br>3. Default: `None`, meaning no TPOT constraint. |
| `--max-batched-tokens` | Service Options | Optional | Maximum batched tokens for one prefill or mixed prefill/decode step.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `8192`. |
| `--batch-range` | Service Options | Optional | Batch size search range.<br>1. Type: List[Int].<br>2. Format: `[min max]` or `[max]`.<br>3. Default: `None`; if `min` is omitted, search starts from `1`; if `max` is omitted, no upper limit is set. |
| `--serving-cost` | Service Options | Optional | Serving cost used for cost-related metrics.<br>1. Type: Float.<br>2. Valid range: non-negative number.<br>3. Default: `0`. |
| `--disagg` | Service Options | Optional | Enables PD disaggregation mode.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--jobs` | Service Options | Optional | Number of parallel search jobs.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `8`. |
| `--concurrency-search-strategy` | Service Options | Optional | Concurrency search strategy.<br>1. Type: Str.<br>2. Reference values: `exponential`, `linear_exponential`.<br>3. Default: `exponential`. |
| `--image-batch-size` | MultiModal Options | Optional | Number of images per request.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`; if omitted, batch size is reused. |
| `--image-height` | MultiModal Options | Optional | Input image height.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--image-width` | MultiModal Options | Optional | Input image width.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--prefill-devices-per-instance` | PD Ratio Optimization Options | Conditionally Required | Required when PD ratio optimization is enabled. Specifies devices per Prefill instance.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: none.<br>4. Determines the parallel configuration search space for Prefill phase. |
| `--decode-devices-per-instance` | PD Ratio Optimization Options | Conditionally Required | Required when PD ratio optimization is enabled. Specifies devices per Decode instance.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: none.<br>4. Determines the parallel configuration search space for Decode phase. |
| `--enable-optimize-prefill-decode-ratio` | PD Ratio Optimization Options | Optional | Enables Prefill/Decode instance ratio optimization mode.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`.<br>4. Cannot be used together with `--disagg`. |

## How to calculate the performance metrics in aggregation mode

- TTFT:

  When `effective_input_length <= max_batched_tokens`, we keep the original full-prefill formula.
  We get average `ttft = sum_for_ttft / concurrency`. For sum_for_ttft, we assume the prefill
  batch size is the max batched tokens divided by effective input length.
  So `prefill_batch_size = max_batched_tokens // effective_input_length`. And request was
  processed in prefill_batch_size steps one by one. We can get the total ttft time as follows:

  `sum_for_ttft = (prefill_latency * prefill_batch_size) * (1 + calc_nums_for_ttft)) * (calc_nums_for_ttft) / 2`

  For example, if we have 12 requests, and max_batched_tokens is 8192, input_length is 2048,
  then prefill_batch_size is 4. And 12 requests was processed in 3 steps.
  so

  `sum_for_ttft = (prefill_latency * 4 ) * (1 + 3) * 3 / 2`

  `ttft = sum_for_ttft / 12`

  When `effective_input_length > max_batched_tokens`, the optimizer automatically splits prefill
  into multiple chunks. The first version uses a fixed decode-first mixed scheduler with 15% token
  budget slack; it does not expose a scheduler selection CLI parameter.

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

  `D QPS = d_concurrency / (tpot * max(output_length - 1, 1)) * 1000` (req/s)

  Where:
  - `d_concurrency`: The batch size (number of concurrent requests) in Decode phase
  - `tpot`: Time-per-output-token in milliseconds
  - `max(output_length - 1, 1)`: Number of Decode tokens after the first token has been produced

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

## Compare multiple hardware profiles

One or more `--device` values can be passed in a single run to benchmark multiple `DeviceProfile` targets and compare their best configurations under the same model, workload, and SLO settings.

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device ATLAS_800_A2_280T_64G ATLAS_800_A3_560T_128G_DIE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --tpot-limits 50
```

Behavior:

- Each device profile is optimized sequentially. Per-device tables (same format as a single-device run) are printed after each profile finishes.
- When two or more devices are specified, the tool additionally prints:
  1. A **hardware profile comparison** table with core modeling parameters (compute, memory, communication bandwidth, and related fields).
  2. A **cross-hardware summary** table with the best configuration per device, ranked for easy comparison.
- Cross-hardware summaries are mode-specific:
  - Aggregation: best throughput per device under TTFT/TPOT limits.
  - Disaggregation: separate Prefill and Decode cross-hardware tables when the corresponding limits are set.
  - PD ratio: best balanced QPS per device under TTFT/TPOT limits, including PD ratio and optional P/D instance counts when `--num-devices` is set.

### Example output when using multiple `--device` values

When two or more device profiles are specified, the optimizer prints per-device results for each profile, followed by two additional cross-hardware tables:

**1. Hardware profile comparison table**

This table shows core modeling parameters for all requested devices (compute, memory bandwidth, communication bandwidth, etc.):

```plaintext
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

**2. Cross-hardware summary table (mode-specific)**

A ranked table of the best configuration per device under the active SLO constraints. Example for Aggregation mode:

```plaintext
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

## Terminal sweep curves (single device)

When exactly one device profile is evaluated, the optimizer can render **terminal ASCII scatter plots** after the sweep completes. These plots help inspect how throughput relates to concurrency and latency across parallel configurations.

Plots are produced for all three optimizer modes:

| Mode | Plots |
|------|-------|
| Aggregation | Throughput vs Concurrency; Throughput vs TPOT |
| Disaggregation (Prefill) | Throughput vs Concurrency; Throughput vs TTFT |
| Disaggregation (Decode) | Throughput vs Concurrency; Throughput vs TPOT |
| PD ratio | Throughput vs Concurrency; Throughput vs TPOT (Decode-side TPS) |

Notes:

- Terminal curves are **not** printed when multiple `--device` values are used; use cross-hardware summary tables instead.
- Curve points exclude OOM / insufficient-memory configurations. They are **not** filtered by TTFT/TPOT SLO limits, so the plots show the full valid sweep while tables still report SLO-constrained bests.
- Rendering uses the optional `plotext` dependency. If `plotext` is unavailable or plotting fails, optimization results are still printed and a warning is logged.

Example (single device, aggregation):

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device ATLAS_800_A2_280T_64G \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --tpot-limits 50 \
    --batch-range 1 256
```

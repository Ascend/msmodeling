# RFC: Pipeline Parallel + Speculative Decode (MTP/DFlash/DSpark) Coexistence

## Metadata

| Item | Content |
| :--- | :--- |
| **Status** | Implemented (PR #773, merged) |
| **Author** | hanxinlong |
| **Created** | 2026-09-03 |
| **Related** | <https://gitcode.com/Ascend/msmodeling/pull/773> |
| **Prerequisite RFC** | [rfc_pipeline_parallel_support_en.md](rfc_pipeline_parallel_support_en.md) |

---

## 1. Problem Statement

Pipeline Parallel (PP) and speculative decode (MTP/DFlash/DSpark) were previously mutually exclusive in the throughput optimizer. When PP>1, `build_pp_search_candidates` (`serving_cast/service/utils.py`) filtered out all non-zero MTP candidates, with a comment claiming "model_builder rejects it." However, this comment was **stale** — the model builder already supported placing MTP proposal layers on the last PP stage (`pipeline_parallel.py:348`).

The real problem had two layers:

1. **ServingCast**: The PP>1 decode path never applied the speculative decode fold (`_fold_decode_latency_ms`), so even if the gate were removed, MTP/DFlash/DSpark would provide no TPOT benefit under PP. Additionally, the PP `_build_user_input` did not translate the speculative method, causing DFlash/DSpark to crash due to `MtpConfig`/`DflashConfig` mutual exclusion.

2. **TensorCast**: `_resolve_decoder_layers` raised `AttributeError` under PP+MTP (MTP layer indices remained None). `_spec_decode_skips_runner_sampler` did not recognize `MtpWrapper`, and the PP path had no sampler guard at all. Non-last stages did not clear `dflash_config`/`dspark_config`, causing wrappers to be constructed on every stage. `PipelineModel.num_hidden_layers` omitted draft layers, breaking KV cache allocation.

### 1.1 Motivation

In **mixed-deployment scenarios** (e.g., KimiK3), MTP/DFlash/DSpark is the necessary means of keeping TPOT within SLO, and PP is the necessary means of relieving memory pressure. Both must be enabled simultaneously on the same nodes. Previously users had to pick one: either enable PP and sacrifice TPOT, or enable MTP/DFlash/DSpark and sacrifice memory.

### 1.2 Goals

- Remove the PP>1 speculative decode candidate gate so PP+MTP/DFlash/DSpark candidates can be enumerated normally.
- Apply `_fold_decode_latency_ms` in the PP>1 decode path so TPOT/throughput correctly reflect speculative decode benefits.
- Fix TensorCast gaps: decoder layer resolution, sampler guard, non-last stage config clearing, `num_hidden_layers` layer counting, and `extra_draft_layers` KV cache slicing.
- Maintain unchanged behavior for PP=1 or no speculative decode (fold is no-op, config clearing not triggered).

### 1.3 Non-Goals

- Does not implement real cross-process PP scheduling (the PP scheduler is a logical max-plus timeline simulation).
- Does not modify speculative decode model construction logic (MtpWrapper/DflashWrapper/DsparkWrapper themselves are unchanged).
- Does not modify the pipeline scheduler algorithm (max-plus cycle mean unchanged).
- Does not add `extra_tail_layers` for DFlash/DSpark draft layers (draft layers run inside the wrapper's separate model, not in the HF decoder ModuleList — `extra_draft_layers` is a parallel mechanism).

---

## 2. Design

### 2.1 Architecture

All three speculative decode methods share the same pattern under PP: proposal/draft computation runs only on the **last PP stage** (it needs the full target model output). The adaptation strategy is unified — **ensure the wrapper is only built on the last stage, the fold is applied in the decode path, and the sampler guard covers all three wrapper types**.

The following end-to-end flow diagram shows where each of the 8 changes fits. Note that ServingCast and TensorCast execute **interleaved** — ServingCast generates candidates, hands each one to TensorCast for simulation, then takes the results back for folding and scheduling:

```text
User Input
  --pp-sizes 2 --speculative-method mtp --num-speculative-tokens 3
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  ServingCast (Phase 1: search + candidate generation)           │
│                                                                 │
│  ① build_pp_search_candidates                                   │
│     Remove PP>1 gate → MTP/DFlash/DSpark candidates enumerated  │
│                                                                 │
│  ③ _build_user_input (PP path)                                  │
│     Translate speculative method:                               │
│     MTP → num_mtp_tokens = N (builds MtpWrapper)                │
│     DFlash/DSpark → num_speculative_tokens = N,                  │
│                     num_mtp_tokens = 0 (avoid mutual exclusion) │
│                                                                 │
│  Output: list of UserInputConfig (each with TP/PP/EP/MTP etc.)  │
└─────────────────────────────────────────────────────────────────┘
    │
    │  Each candidate handed to ModelRunner for evaluation
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  TensorCast (Phase 2: model building + simulation)             │
│                                                                 │
│  ⑥ build_stage_model_config                                     │
│     Non-last stage: clear mtp_config / dflash_config / dspark   │
│     → wrapper only built on last stage                          │
│                                                                 │
│  ⑦ PipelineModel.num_hidden_layers                              │
│     = decoder + MTP + draft (previously missed draft)           │
│     → KV cache allocation covers all layers                     │
│                                                                 │
│  ⑧ extra_draft_layers (new PipelineStageSpec field)             │
│     _slice_layer_cache extends slice range                      │
│     → draft layer KV cache sliced to last stage                 │
│                                                                 │
│  ④ _resolve_decoder_layers                                      │
│     Only check decoder layer indices; MTP/draft stay None       │
│     → decoder layers keep special KV shapes                    │
│                                                                 │
│  ⑤ _spec_decode_skips_runner_sampler                           │
│     Add MtpWrapper check + PP delegation to stages[-1].model    │
│     → no double-sampling (wrapper already sampled internally)   │
│                                                                 │
│  Output: PipelineProfile (stage-level compute/comm/memory data) │
│         ← TensorCast output, passed back to ServingCast          │
└─────────────────────────────────────────────────────────────────┘
    │
    │  PipelineProfile returned to ServingCast
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  ServingCast (Phase 3: scheduling + folding + output)          │
│                                                                 │
│  Scheduler estimate_forward_pipeline / estimate_repeated_pipeline│
│    Feed PipelineProfile to simulate microbatch pipeline fill    │
│    → compute makespan / bubble_ratio / worst_tpot / interval    │
│                                                                 │
│  ② _fold_decode_latency_ms                                       │
│     TPOT = worst_tpot × 1000 / (accept + 1)                     │
│     interval = measured_interval × 1000 / (accept + 1)          │
│                                                                 │
│  Output: TPOT / throughput / Top N table                        │
└─────────────────────────────────────────────────────────────────┘
```

**Flow**: ServingCast generates candidates (①③), hands each to TensorCast for simulation (⑥⑦⑧④⑤), TensorCast returns PipelineProfile to ServingCast, which uses it for scheduling and folding (②), then outputs TPOT/throughput. The three phases execute in an **interleaved loop**, not sequentially.

**ServingCast handles "search + evaluation"**: ① gate removal lets candidates through, ③ method translation prevents config conflicts, ② fold makes TPOT reflect speculative benefits.

**TensorCast handles "model building + simulation"**: ⑥ wrapper only on last stage, ⑦⑧ KV cache covers all layers and slices correctly, ④ decoder layers keep special shapes, ⑤ sampler doesn't double-sample.

### 2.2 ServingCast Changes

#### 2.2.1 Remove Candidate Gate ①

`serving_cast/service/utils.py`, `build_pp_search_candidates`:

```python
# Before (gated):
if pp > 1:
    effective_mtp_list = [m for m in mtp_list if m == 0]
else:
    effective_mtp_list = mtp_list

# After (no gate):
effective_mtp_list = mtp_list
```

All three methods share the MTP search slot (`num_mtp_tokens = N`). The gate removal lets all candidates through. The `pp_blocked_by_mtp` tracking and MTP-specific `ValueError` were also removed.

#### 2.2.2 Apply Fold in PP Decode Path ②

`serving_cast/service/agg_throughput_optimizer.py` (`_get_pp_metrics`):

```python
# Before (no fold):
decode_latency = repeated.worst_tpot_s * 1000.0
decode_interval_ms = repeated.measured_interval_s * 1000.0

# After (with fold):
decode_latency = self._fold_decode_latency_ms(repeated.worst_tpot_s * 1000.0, optimizer_data)
decode_interval_ms = self._fold_decode_latency_ms(repeated.measured_interval_s * 1000.0, optimizer_data)
```

`disagg_throughput_optimizer.py` (`_get_pp_inference_info`) applies the same fold. `serving_cost_ms` stays outside the fold (it is per-step overhead, not compute latency).

The fold function `_fold_decode_latency_ms` already handles all three branches:

- DSpark: `latency / (clamp(accept, 0, block-1) + 1)`
- DFlash: same
- New MTP (`speculative_method == "mtp"`): `latency / (clamp(accept, 0, n) + 1)`
- Legacy MTP: `latency / (sum(mtp_acceptance_rate[:N]) + 1)`

No-op when no speculative method is configured (divides by 1.0), so PP baseline results are unchanged.

#### 2.2.3 PP `_build_user_input` Speculative Method Translation ③

`serving_cast/parallel_runner.py`, PP path `_build_user_input`:

```python
# Before (no translation):
tmp_user_input.num_mtp_tokens = candidate.num_mtp_tokens

# After (with translation):
method = getattr(base_user_input, "speculative_method", None)
if method in ("dflash", "dspark", "mtp"):
    block = int(candidate.num_mtp_tokens) + 1 if int(candidate.num_mtp_tokens) >= 1 else 0
    if block >= 2:
        tmp_user_input.acceptance_length = clamp_acceptance_length(...)
    if method in ("dflash", "dspark"):
        tmp_user_input.num_speculative_tokens = candidate.num_mtp_tokens  # per-candidate N
        tmp_user_input.num_mtp_tokens = 0  # avoid MtpConfig/DflashConfig mutual exclusion
    else:  # mtp
        tmp_user_input.num_mtp_tokens = candidate.num_mtp_tokens
else:
    tmp_user_input.num_mtp_tokens = candidate.num_mtp_tokens
```

DFlash/DSpark N must be explicitly set as `num_speculative_tokens = candidate.num_mtp_tokens` (per-candidate). For single-N, `base_user_input` inherits the correct value via `copy.copy`. For multi-N search (`--num-speculative-tokens 3 7`), CLI only writes the first N to `args.num_speculative_tokens`, so per-candidate override is required. `num_mtp_tokens` must be 0 to avoid `ConfigResolver` building `MtpConfig` alongside `DflashConfig`.

### 2.3 TensorCast Changes

#### 2.3.1 Clear DFlash/DSpark Config on Non-Last Stages ⑥

`tensor_cast/pipeline_parallel.py`, `build_stage_model_config`:

```python
if not stage_spec.is_last:
    stage_config.mtp_config = None
    stage_config.dflash_config = None  # added
    stage_config.dspark_config = None  # added
```

Same pattern as MTP. DflashWrapper/DSparkWrapper is only constructed on the last stage (via `TransformerModel.__init__` → `maybe_enable_dflash`/`maybe_enable_dspark`).

#### 2.3.2 `PipelineModel.num_hidden_layers` Include Draft Layers ⑦

```python
@property
def num_hidden_layers(self) -> int:
    mtp_layers = int(getattr(getattr(self.model_config, "mtp_config", None), "num_mtp_layers", 0) or 0)
    draft_layers = int(self.model_config.draft_num_layers()) if self.model_config.has_draft_spec() else 0
    return self.plan.num_hidden_layers + mtp_layers + draft_layers
```

Ensures `_get_kv_cache_info`'s `range(model.num_hidden_layers)` covers MTP and draft layers. `_resolve_decoder_layers` returns None at MTP/draft indices, downstream uses generic KV formula (correct: MTP blocks use standard MLA, draft blocks use Qwen3 GQA).

#### 2.3.3 `extra_draft_layers` Extend KV Cache Slicing ⑧

`build_pipeline_plan` previously only set `extra_tail_layers` for MTP. DFlash/DSpark draft layers were not in the slicing range, causing `_slice_layer_cache` to exclude them from the last stage's local KV cache. DFlash's `ensure_draft_kv_caches` safety net prevented crashes, but draft KV memory was not counted in `PipelineStageCacheStats`.

Added `extra_draft_layers` field to `PipelineStageSpec`, parallel to `extra_tail_layers`:

```python
# PipelineStageSpec new field
extra_draft_layers: int = 0  # draft layers (DFlash/DSpark) on last stage

# build_pipeline_plan
draft_layers = int(model_config.draft_num_layers()) if model_config.has_draft_spec() else 0
extra_draft_layers = draft_layers if is_last else 0

# _slice_layer_cache
layer_end = stage_spec.layer_end + stage_spec.extra_tail_layers + stage_spec.extra_draft_layers

# build_pipeline_stage_kwargs
if stage_spec.is_last and (extra_tail_layers + extra_draft_layers) > 0:
    stage_kwargs["input_ids"] = input_kwargs["input_ids"]
```

`extra_draft_layers` does not affect `num_hidden_layers_override` (uses `num_layers`, not `num_local_layers`) — draft layers live inside `DflashWrapper`, not in the HF decoder ModuleList. The field only extends KV cache slicing range and `input_ids` forwarding.

#### 2.3.4 `_resolve_decoder_layers` Not Raise for MTP/Draft Layers ④

`tensor_cast/core/input_generator.py`:

```python
# Only check for missing decoder layers
plan = getattr(inner, "plan", None)
decoder_count = int(plan.num_hidden_layers) if plan is not None else len(layers)
missing_decoder = [idx for idx in range(decoder_count) if layers[idx] is None]
if missing_decoder:
    raise AttributeError(...)
```

MTP/draft layer indices legitimately remain None; downstream `_resolve_decoder_attention_layer(None)` returns None → generic KV formula.

#### 2.3.5 `_spec_decode_skips_runner_sampler` PP Delegation ⑤

`tensor_cast/core/model_runner.py`:

```python
def _spec_decode_skips_runner_sampler(model, input_kwargs):
    ...
    # PipelineModel has no _inner; wrapper sits on stages[-1].model
    if hasattr(model, "stages"):
        stages = list(getattr(model, "stages", []))
        if stages:
            last_stage_model = getattr(stages[-1], "model", None)
            if last_stage_model is not None:
                model = last_stage_model

    # Then walk _inner chain as usual
    node = model
    while node is not None:
        if isinstance(node, DflashWrapper):  # covers DsparkWrapper (subclass)
            return DflashWrapper._is_decode_step(input_kwargs)
        if isinstance(node, MtpWrapper):
            return True  # MTP always samples internally
        node = getattr(node, "_inner", None)
    return False
```

The PP path in `run_inference` also applies the guard:

```python
skip_sampler = _spec_decode_skips_runner_sampler(self.model, input_kwargs)
pipeline_result = pipeline_runner.run(
    input_kwargs,
    with_sampler=with_sampler and not skip_sampler, ...
)
```

### 2.4 Unchanged Components

| Component | Reason |
|---|---|
| MtpWrapper/DflashWrapper/DsparkWrapper construction | Handled by `TransformerModel.__init__` transformation chain, PP path naturally works |
| Hidden state passing across stages | `PipelineStageModel` hardcodes `output_intermediate_hidden_states=True` |
| MTP-tail KV cache remapping | `_slice_layer_cache` + `extra_tail_layers` already handles |
| Draft KV cache remapping | `_slice_layer_cache` + `extra_draft_layers` handles (see §2.3.3) |
| MTP/draft compute attribution to last stage | Wrapper runs inside last stage's `Runtime` context |
| Pipeline scheduler | Pure max-plus math, agnostic to stage content |
| `_resolve_forward_shape` | Decode shape already handles DFlash/DSpark `block_size` |
| `_submit_task` | Already propagates `dspark_*`/`dflash_*`/`speculative_method` |
| OptimizerData | Already carries all draft fields |

---

## 3. Comparison: MTP vs DFlash vs DSpark under PP

| Feature | MTP | DFlash | DSpark |
|---|---|---|---|
| Wrapper | `MtpWrapper` | `DflashWrapper` | `DsparkWrapper(DflashWrapper)` |
| Config field | `mtp_config` | `dflash_config` | `dspark_config` |
| Search slot (candidate-generation layer) | `num_mtp_tokens` | `num_mtp_tokens` (reused) | `num_mtp_tokens` (reused) |
| `num_mtp_tokens` (after translation) | N (builds MtpWrapper) | 0 (avoid mutual exclusion) | 0 (avoid mutual exclusion) |
| Non-last stage clearing | `mtp_config = None` | `dflash_config = None` | `dspark_config = None` |
| `num_hidden_layers` includes | `num_mtp_layers` | `draft_num_layers()` | `draft_num_layers()` |
| KV cache slicing extension | `extra_tail_layers` | `extra_draft_layers` | `extra_draft_layers` |
| `num_speculative_tokens` (per-candidate, after translation) | N/A (uses `num_mtp_tokens`) | `= candidate.num_mtp_tokens` | `= candidate.num_mtp_tokens` |
| Sampler guard | `isinstance(MtpWrapper)` → True | `isinstance(DflashWrapper)` → `_is_decode_step` | Same as DFlash (subclass) |
| Fold formula | `latency / (clamp(accept, 0, n) + 1)` | `latency / (clamp(accept, 0, block-1) + 1)` | Same as DFlash |
| Decode shape | `query_len = n + 1` | `query_len = block` | `query_len = block` |

> Note: the search slot and the config fields live on two layers. The **candidate-generation layer** carries N uniformly in `num_mtp_tokens` (`ParallelSearchCandidate.num_mtp_tokens`; on the CLI side, `--num-speculative-tokens` for DFlash/DSpark is normalized into the `num_mtp_token_sizes` search slot during argument parsing). The **config-translation layer** (`_build_user_input`) then maps the slot value into `UserInputConfig` per `speculative_method`: DFlash/DSpark set `num_speculative_tokens=N` and zero `num_mtp_tokens` (so ConfigResolver never builds the mutually exclusive `MtpConfig` alongside), while MTP keeps `num_mtp_tokens=N`. Rows marked "after translation" describe the `UserInputConfig` state; the `num_speculative_tokens` field predates this RFC (introduced with DFlash/DSpark), and this RFC adds the per-candidate translation for PP multi-N search.

---

## 4. Test Strategy

### 4.1 ServingCast Tests

| Test | File | Verifies |
|---|---|---|
| `test_pp_gt1_with_mtp_allowed` | `test_throughput_optimizer.py` | PP>1 accepts MTP candidates |
| `test_pp_decode_applies_mtp_fold` (agg) | `test_agg_optimizer.py` | PP decode TPOT folded by (accept+1) |
| `test_pp_decode_applies_mtp_fold` (disagg) | `test_disagg_optimizer.py` | Same |
| `test_pp_build_user_input_sets_num_mtp_zero_for_dflash` | `test_parallel_runnner.py` | PP DFlash: `num_mtp_tokens=0`, `num_speculative_tokens=N` |
| `test_pp_build_user_input_multi_n_dflash_sets_per_candidate_spec_tokens` | `test_parallel_runnner.py` | Multi-N DFlash per-candidate `num_speculative_tokens` |

### 4.2 TensorCast Tests

| Test | File | Verifies |
|---|---|---|
| `test_resolve_decoder_layers_with_pp_and_mtp` | `test_input_generator.py` | PP+MTP decoder layer resolution, MTP layers None |
| `test_spec_decode_skips_runner_sampler_for_mtp` | `test_pipeline_parallel_model_builder.py` | MtpWrapper sampler guard |
| `test_spec_decode_skips_runner_sampler_for_pp_with_dflash` | same | PP+DFlash sampler guard |
| `test_build_stage_model_clears_dflash_dspark_on_non_last_stage` | same | Non-last stage clears DFlash/DSpark config |
| `test_pipeline_model_num_hidden_layers_includes_draft_layers` | same | `num_hidden_layers` includes draft layers |
| `test_build_pipeline_plan_sets_extra_draft_layers_for_dflash` | same | `extra_draft_layers` set correctly |
| `test_pipeline_stage_kwargs_remaps_draft_cache_to_last_stage` | same | Draft layer KV cache sliced to last stage |

---

## 5. Compatibility

- **PP=1**: All changes are no-ops. The fold divides by 1 when no speculative method is configured, config clearing is never triggered (PP=1 has no non-last stage), and the sampler guard does not affect the standard `with_sampler=False` path.
- **No speculative decode**: The gate removal has no effect (candidates with `num_mtp_tokens=0` always pass), and the fold is a no-op.
- **PP+MTP** (on top of the merged PR #711): gate removal + fold + TensorCast fixes make MTP work correctly under PP.
- **PP+DFlash/DSpark**: the new translation logic + config clearing + layer counting + sampler guard + `extra_draft_layers` KV cache slicing + per-candidate `num_speculative_tokens` make DFlash/DSpark work correctly under PP.

---

## 6. Verification

### 6.1 Unit Tests

All pass (local Python 3.14 + CI Python 3.11), covering: candidate generation, decode fold, decoder layer resolution, sampler guard, config clearing, layer counting, KV cache slicing, and multi-N search. Review-fix coverage additionally includes: the prefill MTP constraint (PP candidates mirror the non-PP semantics), the draft-depth guard (`UnsupportedPPConfigurationError`), the DFlash/DSpark target input either-or contract, and the top-level `target_layer_ids` fallback in draft profiles.

### 6.2 Kimi-K3 Mixed Deployment Verification

Verified PP+MTP/DFlash/DSpark coexistence in the Kimi-K3 mixed deployment scenario. One issue was discovered and fixed during testing:

- **Multi-N DSpark search**: `--num-speculative-tokens 3 7` (multi-N) — CLI only writes the first N to `args.num_speculative_tokens`, and the PP path's `copy.copy(base_user_input)` would cause candidate 2 (N=7) to incorrectly inherit N=3. Fixed by explicitly setting `num_speculative_tokens = candidate.num_mtp_tokens` (commit `33bf398`, aligned with legacy path).

### 6.3 Kimi-K3 PP × Speculative-Decoding Ablation (2026-09-09)

Controlled ablation on the 28K-input / 64-die mixed deployment workload: control group on upstream/master (without this RFC), treatment group on this implementation + PR #794 (PP>1 prefill basis fix), matrix PP∈{1,4} × DSpark on/off:

| Configuration | TTFT | TPOT | Throughput |
|---|---|---|---|
| master, PP=1 TP=8/EP=64, spec off | 5.44s | 71.98ms | 100.43 t/s |
| master, PP=4 TP=4/EP=16, spec off | 2.96s | 69.61ms | 54.25 t/s |
| this RFC, PP=1 TP=8/EP=64, DSpark | 5.30s | 21.07ms | 279.62 t/s |
| this RFC, PP=4 TP=4/EP=16, DSpark | 2.97s | 28.63ms | 121.86 t/s |

Findings:

- **Speculative benefit is orthogonal to PP**: without speculation, TPOT differs by only 3.3% between PP=1 and PP=4 (71.98 vs 69.61ms), confirming that PP affects memory, not TPOT; DSpark cuts TPOT to 21-29ms (×2.4-3.4, theoretical ×3.58) at a negligible TTFT cost (±1.5%). The slightly higher PP=4 TPOT (28.63 vs 21.07ms) comes from the accompanying TP change (per-rank draft compute doubles at TP=4), not from PP scheduling itself.
- **Two hard failures on master**: PP=1 + DSpark fails at build time (the top-level `target_layer_ids` fallback in draft profiles is missing on master, fixed by `fa9517c` as a shared prerequisite); PP=4 + DSpark aborts the **whole search** with the legacy-gate `ValueError` (not a per-candidate skip) — direct evidence for the gate removal.
- **Zero regression**: the no-speculation baselines are bit-identical between master and this implementation for both PP=1 and PP=4, confirming the coexistence changes do not affect existing paths.
- **PP's value is a memory boundary**: PP=1 + TP=4 leaves -54.81 GB of headroom at 28K input (infeasible); PP=4 makes the TP=4 tier viable.
- **PR #794 (PP>1 prefill basis) assessment**: in chunked agg prefill, `chunk_shapes` carry per-chunk `seq_len` which takes precedence over the wave-level parameter modified by #794; PP=4 results differ by only +1.8% TTFT with/without it. #794's main value is the disagg throughput numerator. This RFC does not need to carry #794; rebase after it merges and re-verify.

---

## 7. Future Work

- End-to-end measurement of PD disaggregation: unit-level fold coverage for the disagg optimizer already exists (`test_pp_decode_applies_mtp_fold`) and the mixed deployment (aggregation) path is now verified end to end (§6.3); real-workload measurement with P nodes without PP and D nodes with PP+MTP/DFlash/DSpark remains.
- DFlash/DSpark draft model per-stage KV cache precise memory attribution (current `extra_draft_layers` covers KV cache slicing range, but draft cache per-layer shape comes from `_get_kv_cache_info`'s generic Qwen3-GQA formula, not the draft model's actual attention config — non-standard attention may need per-layer precise shapes).

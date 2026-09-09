# Design Document: TensorCast New Model Adaptation (Run-Through Scope)

## Revision History

| Version | Date | Description |
| --- | --- | --- |
| 1.0 | 2025 | Initial adaptation-efficiency design: doctor, evidence chain, and profiling-anchored verification. |
| 2.0 | 2026 | Streamlined to the run-through scope: removed the raw-insight/evidence chain; verification now derives key-op expectations from public model structure. Profiling-based precision comparison moved to the downstream precision workflow. |

## 1. Background and Scope

The model adapter onboards new HuggingFace-style models to TensorCast simulation. It is deliberately scoped to **run-through adaptation**:

1. The new model's simulation runs end to end on basic cases without errors.
2. Key operator call counts (attention-style ops, MoE gating ops) match what the model's public structure implies.

Out of scope (owned by the downstream precision workflow, which chains this adapter with shape/dtype validation and measured-profiling comparison):

- full op-shape/dtype coverage checks;
- latency or precision comparison against measured profiling;
- op-mapping (`op_mapping.yaml`) calibration for the empirical performance model.

**No measured data enters this flow.** The only inputs are the simulation command, the installed open-source model source/config, and optional simulation failure logs.

## 2. Architecture

### 2.1 Components

```text
reports/<case_name>/command.txt ──┐
                                  ├─> doctor ──> doctor.json + profile draft
reports/<case_name>/failure.log ──┘       │
                                          │ register ModelProfile
                                          v
                       tensor_cast/transformers/builtin_model/<model_type>.py
                                          │
                                          v
                    verify = structure scan + simulation run + reconciliation
                                          │
                                          ├──> verify.json (passed / issues)
                                          └──> ST guardrail case (passed runs only)
```

| Module | Responsibility |
| --- | --- |
| `tensor_cast/adapter/context.py` | Parse the simulation command into an adaptation context; serialize `UserInputConfig` into regression-case dicts. |
| `tensor_cast/adapter/inspect.py` | Scan the built model tree (structure facts, MoE/MLA/VL candidates). |
| `tensor_cast/adapter/recipes.py` | Materialize a minimal `ModelProfile` candidate from structure facts. |
| `tensor_cast/adapter/profile.py`, `profile_draft.py` | Validate profiles; render reviewable builtin drafts. |
| `tensor_cast/adapter/patch_discovery.py` | Classify simulation failure logs into deterministic findings and AI assistance tasks. |
| `tensor_cast/adapter/doctor.py` | Orchestrate the scan build; run verification (scan + simulation + reconciliation). |
| `tensor_cast/adapter/expectations.py` | Derive key-op call-count expectations from public structure; classify runtime ops into key-op categories. |
| `tensor_cast/adapter/runner.py`, `actual.py` | Run one simulation case and aggregate runtime op invocations. |
| `tensor_cast/adapter/verifier.py` | Reconcile actual op counts against expectations; produce issues. |
| `tensor_cast/adapter/st_case.py` | Emit benchmark-regression-compatible guardrail cases from passed verifications. |
| `tensor_cast/adapter/questions.py`, `advisor.py`, `ai_task.py` | Human checkpoints, next-step suggestions, and AI task packaging. |

Removed in v2.0: `insight.py`, `evidence.py`, `evidence_builder.py`, `evidence_export.py`, `hints.py` (the measured-profiling anchor chain).

### 2.2 Structure Scan and Repetition

`run_model_doctor` builds the model with `disable_repetition=True`: repeated-layer reuse collapses identical layers into copy wrappers, which would under-report module counts. The verification **run** keeps the user's repetition setting — region replay restores full per-layer op counts either way.

### 2.3 Key-Op Expectation Derivation

Expectations come from two patch-independent signals plus one patch signal:

- **Attention**: one core attention op per top-level attention module in the scanned tree (text modules, plus vision-tower modules when image input is present in prefill). Nested wrapper/indexer modules are deduplicated by path prefix.
- **MoE gating**: module count from the MoE patch report of the same doctor build (exact on patched trees, where field-based detection cannot see through the wrapper), falling back to the field-based structure scan for un-adapted models.
- MTP-path modules are excluded (the basic verification case does not exercise MTP).

### 2.4 Key-Op Classification

Runtime ops are classified by explicit op-name table plus a conservative fallback:

- Only `tensor_cast.*` ops participate; native `aten` ops (e.g. `aten.scaled_dot_product_attention`) never count, so an un-adapted HF fallback cannot mask a missing replacement.
- The attention family covers the core attention computation ops (`attention`, `attention_quant`, `multihead_latent_attention(_quant)`, `mla_sparse_attention(_quant)`, `linear_attention`, `linear_attn_chunk_gated_delta_rule`, `linear_attn_recurrent_gated_delta_rule`, `block_sparse_attention`, `minimax_sparse_attention`, `kimi_delta_attention_core`, `sparse_attn_sharedkv`, ...) plus any future op whose name contains `attention`. Auxiliary per-layer ops (indexers, KV-cache writes, linear-attention conv/gating helpers) are excluded.
- MoE gating covers `moe_gating_top_k*` variants (softmax/sigmoid/hash/plain).

A newly introduced attention-family op whose name does not contain `attention` must be registered in `_ATTENTION_CORE_OPS` (`tensor_cast/adapter/expectations.py`); the model-adaptation skill calls this out in its new-operator guidance.

### 2.5 Verification Semantics

| Situation | Outcome |
| --- | --- |
| All key-op counts match | `passed: true` |
| Simulation case raises | `SIMULATION_ERROR` error; the report embeds the traceback and patch-discovery `ai_tasks` so callers can drive the fix programmatically |
| Attention category missing (count 0 vs expected > 0) | `KEY_OP_MISSING` error; the message lists unclassified tensor_cast ops as new-attention-core candidates |
| Attention count mismatch (both > 0) | `OP_COUNT_MISMATCH` error |
| No `tensor_cast` ops at all | `NO_TENSOR_CAST_OPS` error |
| Structure scan found no attention modules | `STRUCTURE_SCAN_EMPTY` error |
| MoE modules found, no MoE patch, no gating op, no MoE-path ops (init_routing_v2 / dispatch / grouped_matmul) | `MOE_NOT_ADAPTED` error |
| MoE adapted via standard gating (patch report or MoE-path ops prove TensorCast execution) | no issue — models legitimately gate through `torch.topk` or pre-computed top-k |
| MoE gating op count mismatch (both > 0) | `OP_COUNT_MISMATCH` error |
| `num_mtp_tokens > 0` or `pp_size > 1` | checks degraded to warnings (`EXPECTATION_DEGRADED`) — rerun with a basic case for strict equality |

A crashing simulation is captured, never propagated: `verify` always exits with a JSON report (exit code 1 on failure), so the downstream precision workflow and the AI skill can consume `passed` / `issues` / `ai_tasks` deterministically.

### 2.6 Report Schema Stability

The adapter is a composable component invoked by the downstream precision workflow, so its report schema is a contract:

- Stable top-level keys of the verify report: `model_id`, `model_type`, `case_name`, `passed`, `case_input`, `simulation`, `expectations_basis`, `key_op_checks`, `issues`, `suggestions`, `actual_summary`, `ai_tasks`. New keys may be added; existing keys are not renamed or repurposed without a design-doc revision.
- Stable issue categories: `SIMULATION_ERROR`, `KEY_OP_MISSING`, `OP_COUNT_MISMATCH`, `NO_TENSOR_CAST_OPS`, `MOE_NOT_ADAPTED`, `STRUCTURE_SCAN_EMPTY`, `EXPECTATION_DEGRADED`.
- CLI contract: `verify` exits 0 on `passed: true` and 1 otherwise; `doctor` and `verify` always write a JSON report (stdout or `--output-file`).
- Key-op classification is part of the contract: new attention-family core ops must be registered in `_ATTENTION_CORE_OPS` (`tensor_cast/adapter/expectations.py`) and new MoE gating variants in the `moe_gating_top_k` prefix rule.

### 2.7 ST Guardrail Cases

`verify --st-case-output` emits a case JSON only for **passed** verifications. The format matches `tests/benchmark/models/test_model_regression.py` exactly (`TextPerfRegressionCase` fields), so a passed case can be committed directly under `tests/benchmark/models/cases/` to join the precision guardrail for existing models. Failed verifications never emit a case.

## 3. Workflow (10 Standard Steps)

The user guide `docs/zh/user_guide/msmodeling_tensor_cast_new_model_adaptation_user_guide.md` §4–§13 is the executable definition:

1. Create the case directory with `command.txt` (no measured inputs).
2. Run doctor (structure scan, candidate profile, validation).
3. Locate the model source: in the pinned `transformers`; if only a newer release supports it, upgrade the dependency first; if no release supports it, fetch the source from where the model is open-sourced (HuggingFace / vLLM / ...) and adapt it as a model-specific module under `tensor_cast/transformers/builtin_model/` via patch/wrapper layering.
4. Adapt new operators, if any (ops declaration + performance properties; see skill).
5. Register the reviewed profile in `tensor_cast/transformers/builtin_model/`.
6. Handle runtime patch needs (`failure.log` → doctor → AI task → human review).
7. Rerun doctor after registration.
8. Run verify (simulation runs through; key op counts reconcile).
9. Generate the ST guardrail case.
10. Replay/audit or submit.

Working artifacts under `reports/` (doctor/verify reports, drafts, failure logs, generated st_cases) are development-time outputs: they stay local (`.gitignore` covers `reports/`) and are consumed off-repo by the downstream precision workflow. Only code, tests, docs, and optionally the ST guardrail case (under `tests/benchmark/models/cases/`) are committed.

## 4. Test Design

- `tests/regression/tensor_cast/test_adapter_automation.py`: structure scan, candidate/recipe/profile, patch discovery, expectation derivation, verification semantics (match/mismatch/missing/un-adapted MoE/degraded MTP), ST case compatibility, end-to-end verification on the `qwen3_vl_tiny` fixture, and un-registered-MoE negative coverage.
- `tests/regression/cli/test_spec_cli.py`, `test_logo_cli_hooks.py`: CLI help contract and in-process `verify` execution.
- Cross-model validation: the adapter zoo (deepseek_v32, deepseek_v4, qwen3_5 dense/moe, glm5, minimax_m2, qwen3_vl) reconciles attention and MoE-gating counts via `model_adapter verify`.

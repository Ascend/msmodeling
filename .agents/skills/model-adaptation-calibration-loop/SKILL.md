---
name: model-adaptation-calibration-loop
description: Review non-pass TensorCast adaptation validation findings, obtain human mapping decisions, update decision evidence or narrowly scoped adapter code, and rerun affected checks until they pass or reach a documented blocker. Use after an end-to-end validation handoff, not for initial case execution.
metadata:
  version: 0.3.0
  source: local-rfc
---

# TensorCast Model Adaptation Calibration Loop

Turn a validation handoff into reviewed mapping decisions and minimal fixes, then rerun only the affected gates and their downstream checks.

## Entry Conditions

Require:

- the case directory and latest validation summary;
- the exact failing command and report artifacts;
- original Theory/Runtime/profiling evidence relevant to each finding;
- user authorization before modifying adaptation code, Theory, analyzer logic, or shared mappings.

If these are unavailable, return a concrete blocker instead of inferring the missing evidence.

## Core Rules

- Classify the failure before editing anything.
- Never change Theory merely to match Runtime. Theory changes require independent official source/config evidence.
- Never map a TensorCast sub-op to a fused NPU super-op merely because their names are related.
- Keep raw evidence immutable. Record mapping decisions separately with reviewer, evidence, confidence, and rationale.
- The current base `model-adaptation` flow does not consume measured profiling. Do not reintroduce removed `hints.yaml`, `evidence.yaml`, `export-evidence`, or profiling CLI arguments into doctor/verify.
- Preserve unrelated working-tree changes and upstream model code. Apply model-specific fixes through `tensor_cast/transformers/builtin_model/` or approved composition points.
- Do not weaken tolerances, ignore operators, disable quantization/parallelism, or reduce coverage solely to obtain a pass.
- When changing `op_mapping.yaml`, also use the existing `op-mapping-generator` skill; this skill does not replace its source-chain verification rules.

## Calibration Cycle

1. Read the latest stage results and group non-pass findings by root cause.
2. Reproduce the smallest failing gate without editing inputs.
3. Build an evidence table with expected, actual, provenance, confidence, and candidate action.
4. For uncertain mapping or accepted-gap decisions, ask one to three focused human questions. Do not proceed until the decision is explicit.
5. Record the decision using `assets/mapping-decision-template.yaml` or the case's established decision schema.
6. Apply the narrowest valid change:
   - input/forward issue: fix selection metadata or reacquire data;
   - case-only mapping conclusion: update the reviewed mapping-decision record;
   - performance `op_mapping.yaml`: use `op-mapping-generator` and its verification phase;
   - adapter/Profile bug: update the model-specific built-in adapter and tests;
   - key-op classification bug: update `tensor_cast/adapter/expectations.py` and tests;
   - Theory bug: update the independent YAML source plus specification tests;
   - analyzer bug: update shared analyzer logic plus a regression fixture.
7. Run focused tests for the changed file, then rerun the failed stage and every dependent downstream stage.
8. Update the validation summary with the before/after finding, commands, and residual risks.
9. Repeat while evidence produces new progress. Stop after three consecutive no-progress cycles on the same blocker, or immediately at a hard authorization/security/scope boundary.

## Finding Routing

| Finding | Default owner/action |
| --- | --- |
| Base doctor Profile issue | Validate against model source, obtain Profile approval, then rerun doctor |
| `KEY_OP_MISSING` / `NO_TENSOR_CAST_OPS` | Fix profile replacement or register a confirmed new attention core |
| `OP_COUNT_MISMATCH` | Check layer overrides, MTP, vision execution and patch replacement counts |
| `MOE_NOT_ADAPTED` | Fix model-specific MoE Profile/patch/operator coverage |
| `INPUT_CONFIG_MISMATCH` | Correct the case or acquire matching profiling; no code edit |
| `THEORY_PROFILE_MISSING` | Author an independent Spec/run profile, then add diagnostics tests |
| Shape/dtype mismatch | Decide Theory vs Runtime vs organization defect before patching |
| `FORWARD_BOUNDARY_AMBIGUOUS` | Human selects task id/time range or data is reacquired |
| `STRUCTURE_MAPPING_MISSING` | Record a reviewed mapping decision; use source-chain evidence |
| `COUNT_NORMALIZATION_CONFLICT` | Confirm forward/rank/layer denominator; never round silently |
| `OP_MAPPING_MISSING` | Use `op-mapping-generator` if production performance mapping is in scope |
| profiling CSV/Shape miss | Treat as database coverage, not automatically as mapping failure |
| latency mismatch | First verify context, mapping, count, and latency basis; only then inspect performance model |

## Verification Discipline

- A Profile change requires doctor, base verify, model-specific tests and target simulation.
- A key-op expectation change requires expectation/verifier tests and base verify on affected model families.
- A Theory change requires specification tests and Theory-to-Runtime E2E for affected layer kinds/scenarios.
- A mapping change requires the mapping skill's verification and an end-to-end profiling run.
- An analyzer change requires unit fixtures plus the full `npu_layer_compare.py` case.
- Final validation must restore the original target command; a reduced smoke command is supplementary only.

## Completion Criteria

- All required findings pass, or the same blocker is documented with evidence and an owner.
- Every mapping decision is traceable and has an honest confidence level.
- Code changes are minimal, tested, and do not overwrite unrelated work.
- No accepted gap hides a missing required structure.
- The top-level summary lists exact validation commands and remaining risks.

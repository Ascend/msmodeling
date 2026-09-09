---
name: model-adaptation
description: Use when adapting a new HuggingFace-style model to TensorCast, including collecting the simulation command, running model_adapter doctor, producing or reviewing ModelProfile fields, adapting new operators (op declaration plus performance properties), handling patch/bug AI assistance tasks, and running run-through verification of key operator call counts. No measured profiling data is involved.
metadata:
  version: 2.0.0
  source: local-session-analysis
---

# TensorCast New Model Adaptation (Run-Through Scope)

Guide TensorCast new model onboarding from one required input (the simulation command) to a registered profile, optional patch method, and a passed run-through verification. The goal is that the model's simulation runs on basic cases without errors and that key operator call counts (attention family, MoE gating) match the model's public structure. Full shape/dtype coverage and measured-profiling precision comparison belong to the downstream precision workflow.

## Core Rule

Treat the flow as deterministic tooling plus human review.

- Do not invent `ModelProfile` fields from model names alone; the model's open-source code/config is the only reference.
- Do not write a patch method unless it is based on a failure log and installed model source.
- No measured profiling data enters this flow; do not ask for or accept Insight exports, kernel counts, or measured latencies as inputs.
- Working artifacts under `reports/<case_name>/` (doctor.json, profile drafts, verify.json, failure logs, st_cases) are development-time outputs — they stay local and are never committed; the downstream precision workflow consumes them off-repo. Only code, tests, docs, and optionally the ST guardrail case (under `tests/benchmark/models/cases/`) are committed.
- Keep private paths, local virtualenv paths, raw internal notes, and temporary walkthroughs out of commits.

## Required Inputs

Collect these first. If the simulation command is missing, ask for it before running the workflow.

1. Exact TensorCast simulation command, saved as `reports/<case_name>/command.txt` (local working file, not committed).

Optional inputs:

- `reports/<case_name>/failure.log` for doctor/smoke failures that may need patch or bug-fix assistance.

## Workflow

### 1. Create or check the case workspace

Use repo-relative paths:

```bash
mkdir -p reports/<case_name>
```

The command file should contain the exact runnable simulation command, for example:

```bash
python -m cli.inference.text_generate <model_id> \
  --device <device_profile> \
  --num-devices 1 \
  --num-queries 1 \
  --query-length 1 \
  --context-length 128
```

### 2. Run doctor

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --profile-draft-output reports/<case_name>/<model_type>_draft.py \
  --output reports/<case_name>/doctor.json
```

Review:

- `candidate_profile`
- `candidate_profile_validation`
- `candidate_profile_draft`
- `human_questions`
- `ai_tasks`
- `patch_reports`
- `suggestions`

### 3. Handle human checkpoints

Use this decision table.

| Doctor output | Action |
| --- | --- |
| `candidate_profile_validation.passed=false` | Fix profile fields or confirm the uncertain field against installed model source. |
| `human_questions` is non-empty | Confirm against installed source; low-confidence candidate fields (e.g. `moe_gate_returns_raw_logits` safe default) and structural gaps (missing expert key, visual paths) are the typical questions. |
| `ai_tasks` contains `PATCH_METHOD_AUTHORING` | Give `ai_tasks[].prompt_text` to the user's AI assistant, review the generated patch, then add it to the built-in model profile. |
| failure is not expressible as a patch | Create an AI assistance task in the same style: evidence, suspected files, constraints, required output, verification commands, prompt text. |

Prefer asking one to three focused questions. Do not ask the user to explain the entire model.

### 4. Locate the model source code

Model source does not always come from the installed `transformers`. Resolve the source in this order:

1. **The repository's current `transformers` version already supports the model** — read it directly, usually `transformers.models.<model>.modeling_<model>`.
2. **A newer `transformers` release supports it, but the pinned version does not** — prefer upgrading the repository's `transformers` dependency (`pyproject.toml` / `uv.lock`) instead of copying model code into the repo; after the upgrade this becomes case 1.
3. **No `transformers` release supports it** — fetch the model source from wherever it is open-sourced (HuggingFace model repo / remote code, vLLM or other inference-framework implementations), and adapt it as a model-specific module under `tensor_cast/transformers/builtin_model/` following the repo constraint of patch/wrapper layering (never modify upstream dependencies in place).

In every case, confirm class names, module paths, config fields, and forward behavior in the actual source; do not fill profile fields from the model name alone.

### 5. Adapt new operators when needed

Only when the model uses semantics TensorCast does not yet support (evidence: `UNSUPPORTED_OP_ROUTING` in the failure classification, or a confirmed new computation in the model source). Two pieces are required, and their placement is fixed:

1. Op declaration (shape propagation): `@register_tensor_cast_op("<name>")` in the matching `tensor_cast/ops/` category file, meta-only body, import registered in `tensor_cast/ops/__init__.py`. Reference implementations: `tensor_cast/ops/attention.py`, `tensor_cast/ops/layernorm.py`.
2. Performance properties (compute/memory): `@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.<name>.default)` in `tensor_cast/performance_model/__init__.py` (model-specific operators may live in `performance_model/builtin_model/` or the builtin model module). Derive mma/gp ops and memory traffic from the open-source model's math semantics; cross-check magnitudes against comparable existing operators.

Extra rules:

- If the new op is an attention computation core invoked once per layer and its name does not contain `attention` (e.g. `sparse_attn_sharedkv`), register it in `_ATTENTION_CORE_OPS` in `tensor_cast/adapter/expectations.py` so run-through verification can count it.
- Add unit tests for shape propagation and performance-property magnitudes.
- `op_mapping.yaml` calibration for the empirical model belongs to the downstream precision workflow, not here.

### 6. Author and register the profile

Move reviewed profile code to:

```text
tensor_cast/transformers/builtin_model/<model_type>.py
```

Keep the profile minimal:

- Include only fields that are required or confirmed.
- Avoid empty overrides and default `None` fields.
- Register the reviewed `patch_method` only after it is implemented and reviewed.
- Validate `model_type` against the installed config.

### 7. Rerun doctor after profile registration

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --output reports/<case_name>/doctor_after_profile.json
```

The second report should have a non-null `profile`, passing `profile_validation`, and `patch_reports` matching expected replacement counts (MoE layer count, MLA module count, ...).

### 8. Run run-through verification

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --device <device_profile> \
  --num-queries 1 \
  --query-length 8 \
  --context-length 0 \
  --output reports/<case_name>/verify.json
```

For vision-language prefill cases add `--image-batch-size/--image-height/--image-width`; in decode mode or without image input the vision tower is not executed.

If verification fails, classify the gap:

- `SIMULATION_ERROR`: the simulation case crashed. The report embeds the traceback and patch-discovery `ai_tasks` (typically `PATCH_METHOD_AUTHORING`); follow the task prompt to author the fix, review it, then rerun verify. Do not collect logs manually — the report already carries what doctor needs.
- `KEY_OP_MISSING` / `NO_TENSOR_CAST_OPS`: the model fell back to un-adapted HF modules; fix profile registration/replacement. When the message lists unclassified tensor_cast ops, check whether one is a new attention core and register it in `_ATTENTION_CORE_OPS`.
- `OP_COUNT_MISMATCH`: layer overrides, MTP, vision input, or missing wrapper replacement.
- `MOE_NOT_ADAPTED`: MoE modules found but no MoE patch, no gating op, and no TensorCast MoE-path ops (init_routing_v2 / dispatch / grouped_matmul); fix the MoE profile fields. Note: a model that gates through standard `torch.topk` is fine when the patch report or MoE-path ops prove adaptation.
- `STRUCTURE_SCAN_EMPTY`: build path or installed source issue.
- `EXPECTATION_DEGRADED` (warning): the case enables MTP or PP; rerun a basic case for strict equality.

Then update the profile, operator adaptation, or AI task and rerun the relevant step.

### 9. Generate the ST guardrail case

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --device <device_profile> \
  --st-case-output reports/<case_name>/st_cases \
  --output reports/<case_name>/verify_with_st.json
```

Rules:

- Only passed verifications emit a case; failed runs never emit one.
- The case JSON is directly loadable by `tests/benchmark/models/test_model_regression.py`; passed cases can be committed under `tests/benchmark/models/cases/`.

## Patch and Bug-Fix Assistance

Doctor is deterministic and should not generate model-specific patch code. It should produce an AI task package.

For patch tasks, require the AI assistant to output:

- class and method names to patch
- original failure reason
- patch method diff
- simulation semantics preserved
- real-model checks bypassed, if any
- verification commands

For non-patch bugs, create a similar `BUG_FIX_INVESTIGATION` task with:

- failing command/log
- suspected repo files or stack frames
- constraints
- expected code/test output
- verification commands

Human review is mandatory before adding generated code to the repo.

## Validation

Run focused checks before finishing:

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter verify --help
pytest tests/regression/tensor_cast/test_adapter_automation.py -q
```

If runtime behavior changed, also run the relevant smoke or regression tests.

## Completion Criteria

- The adaptation is committed as code only: builtin profile, operator adaptation, patch methods, and tests. Working artifacts (`doctor*.json`, `verify.json`, drafts, failure logs) stay under the local `reports/` directory and are not committed.
- `ModelProfile` is minimal, reviewed, and validated.
- Any patch method is generated from an AI task plus source review, not from a hard-coded doctor template.
- New operators have declaration + performance properties + unit tests, and the attention-core op table was updated when needed.
- `verify` reports `passed: true` (or remaining warnings are explicitly reviewed and documented); the verify report itself is a development artifact — share it off-repo (e.g. with the downstream precision workflow), do not commit it.
- The ST guardrail case, if generated, comes from a passed verification and is committed under `tests/benchmark/models/cases/` when regression protection is wanted.
- Temporary files and local-only walkthroughs are not staged.

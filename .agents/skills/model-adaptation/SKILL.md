---
name: model-adaptation
description: Use when adapting a new HuggingFace-style model to TensorCast, including collecting the simulation command and MindStudio Insight raw profiling export, running model_adapter doctor, producing or reviewing ModelProfile fields, handling patch/bug AI assistance tasks, exporting evidence.yaml, and verifying the adaptation.
metadata:
  version: 1.0.0
  source: local-session-analysis
---

# TensorCast New Model Adaptation

Guide TensorCast new model onboarding from two required inputs to a reviewed profile, optional patch method, evidence YAML, and verification result.

## Core Rule

Treat the flow as deterministic tooling plus human review.

- Do not invent `ModelProfile` fields from model names alone.
- Do not write a patch method unless it is based on a failure log and installed model source.
- Do not hand-write `evidence.yaml` from scratch; export `evidence_draft` and then review it.
- Keep private paths, local virtualenv paths, raw internal notes, and temporary walkthroughs out of commits.

## Required Inputs

Collect these first. If either is missing, ask for it before running the workflow.

1. Exact TensorCast simulation command, saved as `reports/<case_name>/command.txt`.
2. Matching MindStudio Insight raw profiling export, saved as `reports/<case_name>/raw_insight.txt`.

Optional inputs:

- `reports/<case_name>/hints.yaml` for confirmed kernel mappings, counts, shape notes, or user observations.
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

### 2. Add optional hints only for confirmed facts

Use hints for information the user has already confirmed:

```yaml
version: 1
hints:
  - kind: op_mapping_hint
    profiling_op: FusedInferAttentionScore
    tc_op: tensor_cast.attention.default
    confidence: medium
    note: "Confirmed from kernel name and call count."
```

When information is uncertain, record it as a low/medium confidence hint or leave it for `human_questions`.

### 3. Run doctor

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --profile-draft-output reports/<case_name>/<model_type>_draft.py \
  --output reports/<case_name>/doctor.json
```

Omit `--hints-file` if no hints exist.

Review:

- `candidate_profile`
- `candidate_profile_validation`
- `candidate_profile_draft`
- `raw_insight_summary`
- `evidence_draft`
- `human_questions`
- `ai_tasks`

### 4. Handle human checkpoints

Use this decision table.

| Doctor output | Action |
| --- | --- |
| `candidate_profile_validation.passed=false` | Fix profile fields or ask the user to confirm the uncertain field. |
| `human_questions` is non-empty | Ask the user concise questions, then encode answers in `hints.yaml` or `evidence.yaml`. |
| `ai_tasks` contains `PATCH_METHOD_AUTHORING` | Give `ai_tasks[].prompt_text` to the user's AI assistant, review the generated patch, then add it to the built-in model profile. |
| failure is not expressible as a patch | Create an AI assistance task in the same style: evidence, suspected files, constraints, required output, verification commands, prompt text. |
| `evidence_draft` has low-confidence mappings | Keep the draft, but mark confidence honestly and ask for mapping/count confirmation if it blocks verification. |

Prefer asking one to three focused questions. Do not ask the user to explain the entire model.

### 5. Author and register the profile

Move reviewed profile code to:

```text
tensor_cast/transformers/builtin_model/<model_type>.py
```

Keep the profile minimal:

- Include only fields that are required or confirmed.
- Avoid empty overrides and default `None` fields.
- Register the reviewed `patch_method` only after it is implemented and reviewed.
- Validate `model_type` against the installed config.

### 6. Rerun doctor after profile registration

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --output reports/<case_name>/doctor_after_profile.json
```

The second report should have a non-null `profile` and passing `profile_validation`.

### 7. Export and review evidence

```bash
python -m cli.inference.model_adapter export-evidence \
  --doctor-report reports/<case_name>/doctor_after_profile.json \
  --output reports/<case_name>/evidence.yaml
```

Review the YAML before verification:

- case name
- input parameters
- `expected.total_forward`
- `expected.major_ops`
- counts and tolerances
- confidence
- optional `shape_hints` and notes

### 8. Verify

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --evidence-file reports/<case_name>/evidence.yaml \
  --device <device_profile> \
  --output reports/<case_name>/verify.json
```

If verification fails, classify the gap:

- profile field issue
- patch semantics issue
- missing or wrong op mapping
- accepted backend fusion gap
- profiling/database coverage gap
- communication evidence gap

Then update the profile, hints, evidence, or AI task and rerun the relevant step.

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
python -m cli.inference.model_adapter export-evidence --help
python -m cli.inference.model_adapter verify --help
pytest tests/regression/tensor_cast/test_adapter_automation.py -q
```

If runtime behavior changed, also run the relevant smoke or regression tests.

## Completion Criteria

- Required inputs are preserved under `reports/<case_name>/`.
- `ModelProfile` is minimal, reviewed, and validated.
- Any patch method is generated from an AI task plus source review, not from a hard-coded doctor template.
- `evidence.yaml` is exported from `doctor_after_profile.json.evidence_draft` and reviewed.
- `verify` passes or remaining gaps are explicitly documented.
- Temporary files and local-only walkthroughs are not staged.

# TensorCast New Model Adaptation Guide

This guide describes the operational workflow for adding a HuggingFace-style
model to TensorCast with the model adapter tools and the model-adaptation skill.
It is the authoritative place for step-by-step commands, required inputs,
outputs, review checkpoints, and replay/audit procedures.

Use the design document for architecture and rationale:

```text
docs/design/model_adaptation_efficiency_design.md
```

## 1. Scope

Use this guide when a model needs TensorCast-specific adaptation, such as:

- MoE module metadata or non-default expert count keys.
- MLA or MTP module metadata.
- Vision-language module paths and visual linear mappings.
- Meta-device or compile compatibility patches.
- Profiling evidence and regression guardrails for a newly adapted model.

For dense text models with no special structure, the workflow may produce a
minimal profile that only records `model_type`.

## 2. Prepare the Environment

Run all commands from the repository root in a configured Linux or WSL Python
environment.

Check the adapter CLI:

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter export-evidence --help
python -m cli.inference.model_adapter verify --help
```

If your AI assistant can load project skills, use:

```text
.agents/skills/model-adaptation
```

The skill helps process doctor reports, profile review, human checkpoints, AI
patch tasks, evidence export, and verification. The deterministic adapter tools
remain the source of truth for parsing, validation, and pass/fail checks.

## 3. Required Inputs

You need exactly two required inputs for a case:

| Input | Save As | Requirement |
| --- | --- | --- |
| TensorCast simulation command | `reports/<case_name>/command.txt` | The exact command for the workload that matches the profiling export |
| MindStudio Insight raw profiling export | `reports/<case_name>/raw_insight.txt` | A raw table export with a `Totals` row immediately after the header |

The raw Insight `Totals` row is important. Its `Wall Duration(ms)` value is the
measured total forward time and is used in generated evidence.

Minimal raw Insight shape:

```text
Name    Wall Duration(ms)    Self Time(ms)    Average Wall Duration(ms)    Max Wall Duration(ms)    Min Wall Duration(ms)    Occurrences
Totals  22.328398            22.328398        0.005782                     0.238545                 0.000000                 3862
FusedInferAttentionScore_*    3.055183         3.055183                    0.049277                 0.068602                 0.043541                 62
```

Optional input:

| Input | Save As | Use |
| --- | --- | --- |
| Confirmed hints | `reports/<case_name>/hints.yaml` | Record reviewed kernel mappings, counts, shape notes, or user observations |
| Failure log | `reports/<case_name>/failure.log` | Let doctor classify dry-run or smoke failures that may need a patch |

## 4. Create a Case Directory

Replace `<case_name>` with a short stable name, for example
`qwen3_vl_8b_prefill`.

```bash
mkdir -p reports/<case_name>
```

Save the exact TensorCast simulation command:

```bash
cat > reports/<case_name>/command.txt <<'EOF'
python -m cli.inference.text_generate <model_id> \
  --device <device_profile> \
  --num-devices 1 \
  --num-queries 1 \
  --query-length 1 \
  --context-length 128
EOF
```

Example with multimodal and parallel options:

```bash
cat > reports/qwen3_vl_8b_prefill/command.txt <<'EOF'
python -m cli.inference.text_generate Qwen/Qwen3-VL-8B-Instruct \
  --device TEST_DEVICE \
  --num-devices 1 \
  --num-queries 1 \
  --query-length 128 \
  --context-length 0 \
  --image-batch-size 1 \
  --image-height 224 \
  --image-width 224 \
  --quantize-attention-action DISABLED \
  --quantize-linear-action W8A8_DYNAMIC
EOF
```

Copy the matching MindStudio Insight raw export to:

```text
reports/<case_name>/raw_insight.txt
```

Expected outputs after this step:

```text
reports/<case_name>/command.txt
reports/<case_name>/raw_insight.txt
```

## 5. Add Optional Hints

Skip this step if no extra facts are confirmed.

Create `reports/<case_name>/hints.yaml`:

```yaml
version: 1
hints:
  - kind: op_mapping_hint
    profiling_op: FusedInferAttentionScore
    tc_op: tensor_cast.attention.default
    confidence: medium
    note: "Derived from kernel name and matching call count."
```

Common hint kinds:

| Kind | Fields | Use |
| --- | --- | --- |
| `op_mapping_hint` | `profiling_op`, `tc_op`, `confidence`, `note` | Map a raw Insight kernel to a TensorCast semantic op |
| `profiling_op_observation` | `op`, `count`, `confidence`, `note` | Record a confirmed profiling-side count or interpretation |
| `tc_op_observation` | `op`, `count`, `shape_variants`, `confidence`, `note` | Record a confirmed TensorCast-side count or shape |

Hints are incremental. Do not guess fields just to fill the file.

## 6. Run Doctor

Run doctor with hints:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --profile-draft-output reports/<case_name>/<model_type>_draft.py \
  --output reports/<case_name>/doctor.json
```

Run doctor without hints:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --profile-draft-output reports/<case_name>/<model_type>_draft.py \
  --output reports/<case_name>/doctor.json
```

Expected outputs:

```text
reports/<case_name>/doctor.json
reports/<case_name>/<model_type>_draft.py
```

Review these fields in `doctor.json`:

| Field | Meaning | Action |
| --- | --- | --- |
| `adaptation_context` | Parsed command and normalized workload arguments | Confirm it matches the profiled workload |
| `raw_insight_summary` | Parsed `Totals` and top kernels | Confirm total time and top kernels look plausible |
| `candidate_profile` | Minimal proposed `ModelProfile` fields | Review against installed source |
| `candidate_profile_validation` | Deterministic validation result for the candidate | Fix errors before registering the profile |
| `candidate_profile_draft` | Draft Python module content | Use as a starting point only |
| `profile` | Existing registered profile, if any | Use as reference in normal adaptation |
| `profile_validation` | Validation result for the existing profile | Fix errors if present |
| `evidence_draft` | Draft verification evidence | Export and review later |
| `human_questions` | Minimal facts needed from the user | Answer through `hints.yaml` when possible |
| `ai_tasks` | Bounded tasks for an AI assistant | Use only after reviewing deterministic findings |
| `patch_reports` | Dry-run patch pass results | Check expected replacements and skipped modules |
| `suggestions` | Recommended next actions | Use to decide the next iteration |

## 7. Inspect Installed Model Source

Use the installed `transformers` implementation as the source of truth:

```bash
python -c "import transformers; print(transformers.__file__)"
```

Then inspect the matching source module, usually:

```text
transformers.models.<model_name>.modeling_<model_name>
```

Do not fill profile fields from the model name alone. Confirm actual class
names, module paths, config fields, and forward behavior in the installed
source.

## 8. Register the Reviewed Profile

Move the reviewed draft to:

```text
tensor_cast/transformers/builtin_model/<model_type>.py
```

Use `register_model_profile(ModelProfile(...))`.

Keep the profile minimal:

- Include `model_type`.
- Include only non-default MoE/MLA/MTP/VL fields that are confirmed.
- Use a plain `dict` for `moe_field_names_override`.
- Use list form for nested expert count keys, for example
  `["text_config", "num_experts"]`.
- Do not write empty overrides.
- Do not write default `None` fields.
- Do not write default `moe_num_experts_key="num_experts"` unless required by
  the reviewed code path.

Example profile:

```python
from tensor_cast.transformers.custom_model_registry import ModelProfile, register_model_profile


register_model_profile(
    ModelProfile(
        model_type="example_vl",
        model_family="example_vl",
        visual_module_path="visual",
        language_module_path="language_model",
        visual_layers_module_path="visual.blocks",
        visual_merger_linear_mapping={
            "visual.merger.linear_fc1": "colwise",
            "visual.merger.linear_fc2": "rowwise",
        },
        visual_mlp_linear_mapping={
            "visual.blocks.*.mlp.linear_fc1": "colwise",
            "visual.blocks.*.mlp.linear_fc2": "rowwise",
        },
    )
)
```

## 9. Handle Runtime Patch Needs

Use `patch_method` only when the installed model source is incompatible with
TensorCast simulation. Common causes:

- Data-dependent tensor scalar reads on `meta` tensors.
- Python control flow based on tensor values.
- Strict image or video placeholder checks.
- Boolean mask indexing with dynamic output shape.
- Compile graph breaks.
- Unsupported operator routing.
- Forward signature mismatch.

Capture a full failure log:

```bash
set -o pipefail
bash reports/<case_name>/command.txt 2>&1 | tee reports/<case_name>/failure.log
```

Rerun doctor with the failure log:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --patch-failure-file reports/<case_name>/failure.log \
  --profile-draft-output reports/<case_name>/<model_type>_draft_with_patch.py \
  --output reports/<case_name>/doctor_with_failure.json
```

If no hints file exists, omit `--hints-file`.

Expected outputs:

```text
reports/<case_name>/failure.log
reports/<case_name>/doctor_with_failure.json
reports/<case_name>/<model_type>_draft_with_patch.py
```

When a patch is needed, doctor emits `PATCH_METHOD_AUTHORING` under `ai_tasks`.
Give `ai_tasks[].prompt_text` to the model-adaptation skill or another AI
assistant. The assistant should return a patch-method draft, class/method
targets, semantic explanation, and verification commands.

Review rules:

- Doctor provides deterministic evidence and a prompt; it does not produce final
  model-specific patch code.
- AI output is advisory until reviewed.
- Patch only the simulation-incompatible path.
- Preserve normal tensor behavior as closely as possible.
- Rerun doctor, smoke, and verification after adding the patch.

## 10. Rerun Doctor After Profile Registration

After adding or updating `tensor_cast/transformers/builtin_model/<model_type>.py`,
rerun doctor:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --output reports/<case_name>/doctor_after_profile.json
```

If no hints file exists, omit `--hints-file`.

Expected output:

```text
reports/<case_name>/doctor_after_profile.json
```

The report should show:

- `profile` is not null.
- `profile_validation.passed` is true.
- `candidate_profile_validation.passed` is true or any issue is understood.
- `patch_reports` match the expected replacement and skip counts.
- `human_questions` are either empty or handled through reviewed hints.

## 11. Export Evidence

Export the reviewed `evidence_draft` from the post-profile doctor report:

```bash
python -m cli.inference.model_adapter export-evidence \
  --doctor-report reports/<case_name>/doctor_after_profile.json \
  --output reports/<case_name>/evidence.yaml
```

Expected output:

```text
reports/<case_name>/evidence.yaml
```

Review `evidence.yaml` before verification:

| Field | Check |
| --- | --- |
| `model.model_id` | Matches the adapted model |
| `model.raw_command` | Matches `command.txt` |
| `cases[].name` | Stable and descriptive |
| `cases[].input` | Matches the profiled workload |
| `expected.total_forward` | Comes from `raw_insight:Totals.wall_duration_ms` with reasonable tolerance |
| `expected.major_ops` | Contains reviewed TensorCast semantic ops, counts, sources, and confidence |
| `shape_hints` | Present only when confirmed or useful |
| `accepted_gaps` | Used only for reviewed backend fusion or modeling gaps |

The export command performs deterministic format conversion. It does not replace
human review.

## 12. Verify Evidence

Run verification:

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --evidence-file reports/<case_name>/evidence.yaml \
  --device <device_profile> \
  --output reports/<case_name>/verify.json
```

If `model.model_id` is present in evidence, the positional model ID may be
omitted:

```bash
python -m cli.inference.model_adapter verify \
  --evidence-file reports/<case_name>/evidence.yaml \
  --device <device_profile> \
  --output reports/<case_name>/verify.json
```

Expected output:

```text
reports/<case_name>/verify.json
```

Common verification issues:

| Category | Typical Cause | Next Action |
| --- | --- | --- |
| `OP_MAPPING_MISSING` | Evidence op name does not match actual TensorCast op, or backend fusion has no direct TensorCast equivalent | Fix mapping, add hints, or record reviewed accepted gap |
| `OP_COUNT_MISMATCH` | Layer count, repetition, MTP, MoE routing, or parallel configuration mismatch | Fix profile or case input |
| `LATENCY_MODEL_MISMATCH` | Profiling mapping, performance database, fusion strategy, or device profile issue | Review profiling coverage and tolerances |
| `PROFILING_SHAPE_MISSING` | Raw profiling lacks shape detail or database coverage | Add shape hints or profiling data |
| `PATCH_SEMANTICS_MISSING` | Runtime patch did not route the intended TensorCast path | Fix patch and rerun doctor |
| `COMMUNICATION_GAP` | TP/DP/EP communication not covered by evidence | Add communication evidence or accepted gap |

The case is ready when `verify.json` reports `passed: true`, or when every
remaining issue is explicitly reviewed and documented as an accepted gap.

## 13. Generate ST Guardrail Cases

Generate ST cases from verification:

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --evidence-file reports/<case_name>/evidence.yaml \
  --device <device_profile> \
  --st-case-output reports/<case_name>/st_cases \
  --output reports/<case_name>/verify_with_st.json
```

Expected outputs:

```text
reports/<case_name>/verify_with_st.json
reports/<case_name>/st_cases/*.json
```

Rules:

- Passed verification produces `verified` ST cases.
- Failed verification produces `draft` ST cases.
- Do not submit a draft case as a verified guardrail.
- Case `user_input` should come from the normalized command and evidence input,
  not from manual guesses.

## 14. Replay or Audit an Existing Model

Replay mode checks whether the workflow can rediscover an adaptation without
reading the existing profile as the answer.

Use `--ignore-existing-profile` only for replay or audit tests:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/qwen3_vl_replay/command.txt \
  --raw-insight-file reports/qwen3_vl_replay/raw_insight.txt \
  --ignore-existing-profile qwen3_vl \
  --ignore-existing-profile qwen3_vl_moe \
  --profile-draft-output reports/qwen3_vl_replay/qwen3_vl_replay_draft.py \
  --output reports/qwen3_vl_replay/doctor.json
```

Replay constraints:

- Do not use `tensor_cast/transformers/builtin_model/qwen3_vl.py` as input
  evidence while replay is running.
- Reading the installed `transformers` Qwen3-VL source is allowed because normal
  new-model adaptation also uses installed source.
- Existing TensorCast Qwen3-VL profile may be used only after replay discovery
  completes, as an oracle for comparison.

## 15. Qwen3-VL Blind Replay Test

The adapter automation suite includes a tiny config-only Qwen3-VL fixture:

```text
tests/assets/model_config/qwen3_vl_tiny/config.json
```

Use it to test the replay workflow without downloading weights.

Run the focused regression test:

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -k qwen3_vl -q
```

The test should verify that doctor, under profile hiding, can rediscover:

- `model_type=qwen3_vl`.
- `model_family=qwen3_vl`.
- `visual_module_path=visual`.
- `language_module_path=language_model`.
- `visual_layers_module_path=visual.blocks`.
- Visual merger linear mappings.
- Visual MLP linear mappings.
- Qwen3-VL placeholder or dynamic-mask patch authoring evidence from a failure
  log.

Recommended broader test:

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -q
```

After replay discovery completes, compare the replay candidate and patch task
expectations against the existing Qwen3-VL adaptation as an oracle. The oracle
comparison should not be used to seed the replay candidate.

## 16. Validation Checklist

Run the focused adapter test suite:

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -q
```

Check CLI entry points:

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter export-evidence --help
python -m cli.inference.model_adapter verify --help
```

If runtime behavior changed, run relevant TensorCast tests:

```bash
pytest tests/test_tensor_cast/test_runtime.py tests/test_tensor_cast/test_text_generate.py
```

If the adaptation touches a specific model family, run the closest model or
benchmark smoke path available in the repository.

## 17. Submission Checklist

Before submitting a new adaptation:

- The required command and raw profiling inputs are preserved under
  `reports/<case_name>/` or documented outside the repository if they cannot be
  committed.
- The final built-in profile is minimal and source-backed.
- `candidate_profile_validation` and `profile_validation` pass.
- Any patch method came from deterministic failure evidence, AI assistance, and
  human review.
- `evidence.yaml` was exported from doctor output and reviewed.
- `verify.json` passes, or remaining gaps are documented and accepted.
- ST guardrail cases are generated only from verified or explicitly accepted
  evidence.
- Temporary local files, private paths, raw internal notes, and walkthroughs are
  not staged.

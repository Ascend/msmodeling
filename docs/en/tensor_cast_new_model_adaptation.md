# TensorCast New Model Adaptation Guide

This guide describes the minimal path for adding a HuggingFace-style transformer
model to TensorCast. Keep the change scoped to the model adapter, verification,
and nearby tests.

## 1. Prepare the development environment

Run development and verification from the repository root in a configured
Linux/WSL Python environment.

New work should start from `develop`. Existing adaptation branches can be used
as references, but avoid stacking new implementation on top of an old reference
branch.

If your AI assistant can load project skills, use the
`.agents/skills/model-adaptation` skill for this workflow. The skill guides the
assistant through doctor runs, profile review, human checkpoints, patch AI task
handling, evidence export, and verification without relying on local-only
paths or handwritten evidence.

## 2. Collect the required inputs

The required user inputs are:

- A MindStudio Insight raw profiling export.
- The exact TensorCast simulation command that produced the corresponding
  workload.

The raw Insight export must include a `Totals` row immediately after the
header. Its `Wall Duration(ms)` value is the measured forward total time and is
mapped to TensorCast analytic total time in generated evidence. A large total
forward mismatch should be treated as an adapter failure signal, even if some
individual operator counts look plausible.

Minimal raw Insight shape:

```text
Name    Wall Duration(ms)    Self Time(ms)    Average Wall Duration(ms)    Max Wall Duration(ms)    Min Wall Duration(ms)    Occurrences
Totals  22.328398            22.328398        0.005782                     0.238545                 0.000000                 3862
FusedInferAttentionScore_*    3.055183         3.055183                    0.049277                 0.068602                 0.043541                 62
```

Other information is optional. If the adapter flow later needs more facts, add
them incrementally through a hints YAML file instead of asking the user to fill a
complete operator mapping up front.

Example doctor run:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/model_command.txt \
  --raw-insight-file reports/raw_profiling_from_insight.txt \
  --hints-file reports/adapter_hints.yaml \
  --output reports/model_adapter_doctor.json
```

If a dry-run or smoke command fails in a way that may require a runtime patch,
save the stacktrace and let doctor classify it:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/model_command.txt \
  --raw-insight-file reports/raw_profiling_from_insight.txt \
  --patch-failure-file reports/model_failure.log \
  --profile-draft-output tensor_cast/transformers/builtin_model/<model_type>.py \
  --output reports/model_adapter_doctor.json
```

The doctor report includes `patch_discovery`, `ai_tasks`,
`human_questions`, and a `candidate_profile_draft`. When a runtime patch is
needed, doctor emits a `PATCH_METHOD_AUTHORING` task with deterministic
evidence, suspected traceback locations, constraints, verification commands,
and a ready-to-copy prompt for an AI assistant. Doctor does not generate model
specific patch code.

Replay or audit mode can temporarily ignore an existing built-in profile:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/qwen3_vl_command.txt \
  --raw-insight-file reports/qwen3_vl_raw.txt \
  --ignore-existing-profile qwen3_vl \
  --ignore-existing-profile qwen3_vl_moe \
  --output reports/qwen3_vl_replay_doctor.json
```

Use `--ignore-existing-profile` only for replay/audit tests. Normal adaptation
should use existing profiles and recipes as references.

Hints are optional and can be appended as the adapter author learns more:

```yaml
version: 1
hints:
  - kind: op_mapping_hint
    profiling_op: FusedInferAttentionScore
    tc_op: tensor_cast.attention.default
    confidence: medium
    note: "Derived from kernel name and matching call count."
```

## 3. Decide whether the model needs a profile

Add a profile file under `tensor_cast/transformers/builtin_model/` when the
model needs TensorCast-specific structure metadata, such as:

- MoE module location.
- Non-default MoE expert count config key.
- MoE field-name differences.
- MLA module location or custom MLA implementation.
- MTP/speculative decoding block location.
- Meta-device compatibility patches.
- Vision-language module paths and vision linear mapping.

For a dense text model with no special structure, a minimal profile may only
need `model_type`.

## 4. Start from runtime evidence

Use the currently installed `transformers` implementation as the source of
truth:

```bash
python -c "import transformers; print(transformers.__file__)"
```

Then inspect the matching source module, usually:

```text
transformers.models.<model_name>.modeling_<model_name>
```

Do not fill fields from model names alone. Confirm the actual module assignment,
config field, and forward behavior in the installed source.

## 5. Fill `ModelProfile` fields

Create or update:

```text
tensor_cast/transformers/builtin_model/<model_type>.py
```

Use `register_model_profile(ModelProfile(...))`.

### MoE fields

| Field | When to set | Value rule |
| --- | --- | --- |
| `moe_module_name` | MoE model | The model-defined MoE container class name. Do not use an inherited parent class unless that is the actual container. |
| `moe_num_experts_key` | MoE expert count is not the default top-level `num_experts` | Use a string for top-level keys, or a list for nested keys, for example `["text_config", "num_experts"]`. |
| `moe_gate_returns_raw_logits` | MoE model | Set from the router/gate return value. `True` means raw logits; `False` means ready-to-use normalized weights/probabilities. |
| `moe_field_names_override` | MoE attributes use non-standard names | Use a plain dict and include only fields that differ from defaults. |
| `custom_expert_module_type` | Expert storage/call pattern does not match the built-in expert wrapper | Provide a compatible expert wrapper class. |

Recommended override form:

```python
moe_field_names_override={
    "shared_experts": "shared_expert",
    "shared_experts_gate": "shared_expert_gate",
}
```

Do not write empty overrides. Do not write `None` fields.

### MLA and MTP fields

| Field | When to set | Value rule |
| --- | --- | --- |
| `mla_module_name` | The model has MLA attention | Use the attention class that implements MLA. |
| `mla_module_class_type` | TensorCast must use a custom MLA implementation | Import and pass the TensorCast MLA class. |
| `mtp_block_module_name` | The model has MTP/speculative decoding support | Use the independent MTP block class if present; otherwise use the decoder layer class that actually performs the MTP path. |

### Patch fields

Use `patch_method` only when the installed model source is incompatible with
TensorCast simulation. Common cases:

- Data-dependent tensor scalar reads on `meta` tensors.
- Python control flow based on tensor values.
- Strict image/video placeholder checks that index by data-dependent masks.
- Model methods that need TensorCast operator routing.

Patch the class that actually owns the problematic method. Keep normal tensor
behavior as close to the original implementation as possible.

### Vision-language fields

For VL models, prefer `resolve_visual_config()` and fill only what differs from
existing defaults. Confirm:

- Vision module path.
- Language module path.
- Vision layer list path.
- Vision merger linear mapping.
- Vision MLP linear mapping.

## 6. Use adapter inspection when possible

The adapter automation can infer a candidate profile from a built model:

- It scans attention-like, MoE-like, and MLP-like modules.
- It recognizes nested expert count keys under `text_config` and `llm_config`.
- It normalizes review output by omitting defaults and empty overrides.

Use the generated candidate as a review aid, not as proof. Validate uncertain
fields against the installed model source.

## 7. Validate the profile

At minimum, add or update tests near:

```text
tests/regression/tensor_cast/test_adapter_automation.py
```

Run a focused TensorCast test before finishing:

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -q
```

If the change affects runtime behavior, also run the relevant smoke test:

```bash
pytest tests/test_tensor_cast/test_runtime.py tests/test_tensor_cast/test_text_generate.py
```

For model-specific CLI behavior, verify the corresponding CLI help or smoke
path remains backward compatible.

The adapter automation test suite includes a tiny config-only Qwen3-VL fixture:

```text
tests/assets/model_config/qwen3_vl_tiny/config.json
```

It is used to replay doctor behavior with `--ignore-existing-profile qwen3_vl`
against the installed `transformers` Qwen3-VL source without downloading model
weights. This guards the flow that discovers VL paths, visual linear mapping,
and patch authoring evidence without reading the existing TensorCast built-in
`qwen3_vl.py` as the answer.

## 8. Run the end-to-end adapter flow

This sequence is the recommended user-facing workflow for one model and one
profiling case. Replace `<case_name>`, `<model_id>`, and shape/runtime options
with the target workload.

### 8.1 Prepare case inputs

Create a report directory and save the exact TensorCast simulation command:

```bash
mkdir -p reports/<case_name>
cat > reports/<case_name>/command.txt <<'EOF'
python -m cli.inference.text_generate <model_id> \
  --device <device_profile> \
  --num-devices 1 \
  --num-queries 1 \
  --query-length 1 \
  --context-length 128
EOF
```

Save the matching MindStudio Insight raw export as:

```text
reports/<case_name>/raw_insight.txt
```

Optional facts can be recorded in `reports/<case_name>/hints.yaml`. Use hints
for reviewed kernel mappings, operator counts, or shape observations that are
not obvious from the raw Insight export.

### 8.2 Run doctor

Run doctor to build the adaptation context, inspect the model structure, draft
a `ModelProfile`, and generate initial evidence:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --profile-draft-output reports/<case_name>/<model_type>_draft.py \
  --output reports/<case_name>/doctor.json
```

If no hints are available, omit `--hints-file`. Review these fields in
`doctor.json`:

- `candidate_profile`
- `candidate_profile_validation`
- `candidate_profile_draft`
- `evidence_draft`
- `human_questions`
- `ai_tasks`

### 8.3 Author runtime patches when needed

If doctor or a smoke run fails with a meta tensor, placeholder, dynamic shape,
signature, compile, or unsupported-op issue, save the full stacktrace:

```bash
set -o pipefail
bash reports/<case_name>/command.txt 2>&1 | tee reports/<case_name>/failure.log
```

Then rerun doctor with the failure log:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --patch-failure-file reports/<case_name>/failure.log \
  --profile-draft-output reports/<case_name>/<model_type>_draft_with_patch.py \
  --output reports/<case_name>/doctor_with_failure.json
```

When a patch is required, doctor emits a `PATCH_METHOD_AUTHORING` item under
`ai_tasks`. Give `ai_tasks[].prompt_text` to an AI assistant, review the
generated patch method, and add the reviewed implementation to the built-in
model profile. The generated draft contains only a placeholder patch method;
it is not a working implementation.

### 8.4 Register and validate the profile

Move the reviewed profile to:

```text
tensor_cast/transformers/builtin_model/<model_type>.py
```

Run doctor again after the profile is registered:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --raw-insight-file reports/<case_name>/raw_insight.txt \
  --hints-file reports/<case_name>/hints.yaml \
  --output reports/<case_name>/doctor_after_profile.json
```

The second report should show a non-null `profile` and a passing
`profile_validation`. If smoke still fails, capture the new failure and repeat
the patch authoring step.

### 8.5 Export and review evidence

`doctor_after_profile.json.evidence_draft` is the evidence body used by
verification. Export it to YAML with:

```bash
python -m cli.inference.model_adapter export-evidence \
  --doctor-report reports/<case_name>/doctor_after_profile.json \
  --output reports/<case_name>/evidence.yaml
```

This command only converts the deterministic `evidence_draft` JSON object to
YAML. Before verification, review `evidence.yaml` and adjust case names,
operator mappings, counts, confidence, tolerances, notes, and optional
`shape_hints` based on confirmed facts.

### 8.6 Verify

Run evidence verification:

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --evidence-file reports/<case_name>/evidence.yaml \
  --device <device_profile> \
  --output reports/<case_name>/verify.json
```

The case is ready when `verify.json` reports `passed: true`, or when any
remaining gaps are explicitly reviewed and accepted.

## 9. Review before submitting

Before submitting, check:

- The profile writes only required fields.
- `moe_field_names_override` is a dict, not a `MoEFieldNames(...)` object.
- Default `moe_num_experts_key="num_experts"` is not unnecessarily emitted in generated review output.
- Nested expert count keys use list form.
- Patch methods are scoped to the correct installed `transformers` class.
- Tests cover the new model behavior or the newly generalized adapter rule.

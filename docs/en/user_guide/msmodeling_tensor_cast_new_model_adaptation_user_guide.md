# TensorCast New Model Adaptation Guide (Run-Through Scope)

This guide describes the operational workflow for adding a HuggingFace-style
model to TensorCast with the model adapter tools and the model-adaptation
skill. The current scope is **run-through adaptation**: the new model's
simulation runs end to end on basic cases without errors, and key operator
call counts (for example, attention-family ops) match what the model's public
structure implies. No measured data is involved — the only inputs are the
simulation command, the installed open-source model source/config, and
optional simulation failure logs.

Full-coverage shape/dtype validation and measured-profiling precision
comparison are owned by the downstream precision workflow, which chains this
guide's outputs.

Use the design document for architecture and rationale:

```text
docs/design/model_adaptation_efficiency_design.md
```

## 1. Scope

Use the workflow in this guide when a model requires one or more of the following adaptations:

- MoE module metadata or non-default expert count keys.
- MLA or MTP module metadata.
- Vision-language module paths and visual linear mappings.
- Meta-device or compile compatibility patches.
- New operator adaptation (op declaration plus compute/memory performance
  properties).

For dense text models with no special structure, the adaptation workflow may require only a minimal model profile.

## 2. Preparing the Environment

Run all commands from the msmodeling repository root with a configured Python
environment. If you are not at the root, set `PYTHONPATH` first:

```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
```

Check that the model adapter CLI is available:

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter verify --help
```

If your AI assistant supports project skills, use the model-adaptation skill:

```text
.agents/skills/model-adaptation
```

The skill helps with doctor reports, profile review, human checkpoints,
new-operator placement, AI patch tasks, and run-through verification. The
deterministic adapter tools remain the accurate source for parsing,
validation, and pass/fail decisions.

## 3. Prepare Inputs

Create a case directory under `reports/` (a local working directory, covered
by `.gitignore`) with the following input:

| Input | Save as | Requirement |
| --- | --- | --- |
| TensorCast simulation command | `reports/<case_name>/command.txt` | The full runnable workload command. |

No measured input (no MindStudio Insight export, no profiling data) is
required by this flow.

Note: everything under `reports/<case_name>/` (command.txt, doctor.json,
verify.json, failure.log, st_cases, ...) is a **development-time working
artifact and is never committed**. Keep it for local debugging and for the
downstream precision alignment workflow to consume off-repo; the repository
only receives code, tests, docs, and optionally the ST guardrail case.

Optional inputs:

| Input | Save as | Purpose |
| --- | --- | --- |
| Failure log | `reports/<case_name>/failure.log` | Lets doctor classify dry-run or smoke failures and decide whether a patch is needed. |

## 4. Create the Case Directory

Choose a short, stable `<case_name>`, for example `qwen3_vl_8b_prefill`.

```bash
mkdir -p reports/<case_name>
```

Save the full TensorCast simulation command:

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

Multimodal and quantization example:

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

## 5. Run Doctor

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --profile-draft-output reports/<case_name>/<model_type>_draft.py \
  --output reports/<case_name>/doctor.json
```

Expected outputs:

```text
reports/<case_name>/doctor.json
reports/<case_name>/<model_type>_draft.py
```

Review the following `doctor.json` fields:

| Field | Meaning | Action |
| --- | --- | --- |
| `adaptation_context` | Parsed command and normalized workload parameters. | Confirm they match the intended workload. |
| `candidate_profile` | Minimal candidate `ModelProfile` fields. | Review against the installed model source. |
| `candidate_profile_validation` | Deterministic validation of the candidate. | Fix errors before registering. |
| `candidate_profile_draft` | Draft builtin profile module content. | Use only as a starting point. |
| `profile` | Registered profile, if any. | Reference for a healthy adaptation. |
| `profile_validation` | Validation of the registered profile. | Fix errors if present. |
| `human_questions` | Minimal facts needing user confirmation (low-confidence candidate fields, structural gaps). | Confirm against source and encode into the profile. |
| `ai_tasks` | Bounded tasks for an AI assistant. | Use only after reviewing deterministic findings. |
| `patch_reports` | Dry-run patch results. | Check expected replacements and skipped modules. |
| `suggestions` | Recommended next steps. | Drive the next iteration. |

## 6. Locate the Model Source Code

Model source does not always come from the installed `transformers`. Resolve
the source in this order:

1. **The repository's current `transformers` version already supports the
   model** — read it directly, usually
   `transformers.models.<model_name>.modeling_<model_name>`.
2. **A newer `transformers` release supports it, but the pinned version does
   not** — prefer upgrading the repository's `transformers` dependency
   (`pyproject.toml` / `uv.lock`) instead of copying model code into the
   repo; after the upgrade this becomes case 1.
3. **No `transformers` release supports it** — fetch the model source from
   wherever it is open-sourced (HuggingFace model repo / remote code, vLLM or
   other inference-framework implementations), and land it as a
   model-specific module under `tensor_cast/transformers/builtin_model/`,
   following the repo constraint of patch/wrapper layering (never modify
   upstream dependencies in place).

In every case, confirm real class names, module paths, config fields, and
forward behavior in the actual source; do not fill profile fields from the
model name alone.

## 7. Adapt New Operators

Only add operator adaptation when the model forward uses semantics TensorCast
does not yet support. Evidence: an `UNSUPPORTED_OP_ROUTING` classification in
a dry-run failure log, or a confirmed new computational semantic in the model
source. A new operator needs two pieces, placed as follows:

1. **Op declaration (shape propagation)**: declare the op with
   `@register_tensor_cast_op("<name>")` in the matching category file (or a
   new file) under `tensor_cast/ops/`, with a meta-only body that propagates
   shapes, and register the import in `tensor_cast/ops/__init__.py`. See
   `tensor_cast/ops/attention.py` or `tensor_cast/ops/layernorm.py` for
   reference implementations.

2. **Performance properties (compute/memory formulas)**: register a functor
   with `@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.<name>.default)`
   in `tensor_cast/performance_model/__init__.py` that derives `mma_ops` /
   `gp_ops` and memory traffic from the input shapes/dtypes in
   `OpInvokeInfo`. Model-specific operators may live in
   `tensor_cast/performance_model/builtin_model/` or the builtin model module.
   The formulas must be derived from the open-source model's mathematical
   semantics (order of magnitude of FLOPs, tensor sizes read/written);
   cross-check magnitudes against comparable existing operators.

Additional constraints:

- If the new op is an attention computation core (invoked once per layer) and
  its name does not contain the `attention` substring (for example
  DeepSeek-V4's `sparse_attn_sharedkv`), it must also be added to
  `_ATTENTION_CORE_OPS` in `tensor_cast/adapter/expectations.py`, otherwise
  run-through verification cannot count it.
- The empirical performance model relies on `op_mapping.yaml` op mapping; that
  mapping is calibrated by the downstream precision workflow and is out of
  scope here.
- A new operator needs unit tests: correct meta shape propagation and
  reasonable performance-property magnitudes.

## 8. Registering the Reviewed Profile

Move the reviewed draft to:

```text
tensor_cast/transformers/builtin_model/<model_type>.py
```

Register with `register_model_profile(ModelProfile(...))`.

Keep the profile minimal:

- Include `model_type`.
- Include only confirmed non-default MoE, MLA, MTP, or VL fields.
- Use a plain `dict` for `moe_field_names_override`.
- Use a list for nested expert count keys, e.g. `["text_config", "num_experts"]`.
- Do not write empty overrides.
- Do not write default `None` fields.
- Do not write the default `moe_num_experts_key="num_experts"` unless a
  reviewed code path needs it.

Profile example:

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

## 9. Handling Runtime Patch Needs

Use a `patch_method` only when the installed model source is incompatible
with TensorCast simulation. Common reasons:

- Reading data-dependent tensor scalars on `meta` tensors.
- Python control flow based on tensor values.
- Strict image/video placeholder validation.
- Boolean-mask indexing with dynamic output shapes.
- Compile graph breaks.
- Unsupported op routing.
- Forward signature mismatches.

Capture the full failure log:

```bash
set -o pipefail
bash reports/<case_name>/command.txt 2>&1 | tee reports/<case_name>/failure.log
```

Rerun doctor with the failure log:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --patch-failure-file reports/<case_name>/failure.log \
  --profile-draft-output reports/<case_name>/<model_type>_draft_with_patch.py \
  --output reports/<case_name>/doctor_with_failure.json
```

Expected outputs:

```text
reports/<case_name>/failure.log
reports/<case_name>/doctor_with_failure.json
reports/<case_name>/<model_type>_draft_with_patch.py
```

When a patch is needed, doctor emits a `PATCH_METHOD_AUTHORING` AI task. Give
`ai_tasks[].prompt_text` to the model-adaptation skill or another AI
assistant; it should return a patch-method draft, class/method targets,
semantics notes, and verification commands.

Review rules:

- Doctor provides deterministic evidence and prompts, never final
  model-specific patch code.
- AI output is advisory until human-reviewed.
- Patch only simulation-incompatible paths.
- Preserve normal tensor behavior where possible.
- Rerun doctor and verification after adding a patch.

## 10. Rerunning Doctor After Profile Registration

After adding or updating `tensor_cast/transformers/builtin_model/<model_type>.py`, rerun doctor:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/<case_name>/command.txt \
  --output reports/<case_name>/doctor_after_profile.json
```

The report should show:

- `profile` is not null.
- `profile_validation.passed` is true.
- `candidate_profile_validation.passed` is true, or the issues are understood.
- `patch_reports` match expected replacement/skip counts (MoE layer count,
  MLA module count, ...).
- `human_questions` is empty or has been confirmed against source.

## 11. Run Run-Through Verification

verify runs one basic simulation case and reconciles the actual call counts
of key ops (attention family, MoE gating) against structure-derived
expectations:

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --device <device_profile> \
  --num-queries 1 \
  --query-length 8 \
  --context-length 0 \
  --output reports/<case_name>/verify.json
```

Vision-language prefill cases need image parameters (the vision tower does
not execute in decode mode or without image input):

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --image-batch-size 1 --image-height 224 --image-width 224 \
  --output reports/<case_name>/verify_vl_prefill.json
```

Pass criteria: `verify.json` reports `passed: true` (the simulation ran and
all error-severity issues are clear).

Common verification issues:

| Type | Common cause | Next step |
| --- | --- | --- |
| `SIMULATION_ERROR` | The simulation case crashed (meta compatibility, placeholder validation, dynamic shapes, ...) | The report embeds the traceback and patch-discovery `ai_tasks`; follow the task prompt, fix, and rerun verify — no manual log collection needed |
| `KEY_OP_MISSING` | An attention-family op was never invoked; the model fell back to un-adapted HF modules | Check ModelProfile registration and module replacement; if the message lists unclassified tensor_cast ops, check whether one is a new attention core and register `_ATTENTION_CORE_OPS` |
| `OP_COUNT_MISMATCH` | Layer overrides, MTP, vision input, or partially replaced layers | Fix the profile or case input |
| `NO_TENSOR_CAST_OPS` | The simulation recorded no tensor_cast ops at all | Register a profile and rerun |
| `MOE_NOT_ADAPTED` | MoE modules were found with no MoE patch, no gating op, and no TensorCast MoE-path ops (init_routing_v2, ...) | Register or fix the MoE profile fields |
| `STRUCTURE_SCAN_EMPTY` | The structure scan found no attention modules | Check the model build path and installed source |
| `EXPECTATION_DEGRADED` (warning) | The case enables MTP or PP; count checks are degraded | Rerun with a basic case (no MTP, pp_size=1) for strict equality |

Note: correctly adapted models may legitimately gate MoE through the standard
`torch.topk` path instead of a tensor_cast op (for example DeepSeek V3.2). In
that case verification relies on the doctor MoE patch report to confirm
adaptation completeness; this is not an issue.

## 12. Generate the ST Guardrail Case

Generate a regression case from a passed verification (only passed runs emit
a case):

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --device <device_profile> \
  --st-case-output reports/<case_name>/st_cases \
  --output reports/<case_name>/verify_with_st.json
```

Expected outputs:

```text
reports/<case_name>/verify_with_st.json
reports/<case_name>/st_cases/<case_name>.json
```

Rules:

- The generated case JSON is fully compatible with
  `tests/benchmark/models/test_model_regression.py`; passed cases can be
  committed under `tests/benchmark/models/cases/` to join the precision
  guardrail for existing models.
- Failed verifications never emit a case; fix the issues and re-verify.
- The case `user_input` comes from the normalized command parameters, not
  manual guessing.

## 13. Replay or Audit an Existing Model

Replay mode checks whether the flow can rediscover adaptation information
without reading the existing profile as the answer.

Use `--ignore-existing-profile` only for replay or audit testing:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/qwen3_vl_replay/command.txt \
  --ignore-existing-profile qwen3_vl \
  --ignore-existing-profile qwen3_vl_moe \
  --profile-draft-output reports/qwen3_vl_replay/qwen3_vl_replay_draft.py \
  --output reports/qwen3_vl_replay/doctor.json
```

Replay constraints:

- Do not feed `tensor_cast/transformers/builtin_model/qwen3_vl.py` as input
  evidence during replay.
- Reading the installed `transformers` Qwen3-VL source is allowed, since a
  normal new-model adaptation also uses the installed source.
- The existing TensorCast Qwen3-VL profile may only be used as an oracle
  after replay discovery completes.

The Qwen3-VL blind replay test uses a config-only fixture
(`tests/assets/model_config/qwen3_vl_tiny/config.json`) and needs no weight
download:

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -k qwen3_vl -q
```

## 14. Verification Checklist

Run the focused adaptation suite:

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -q
```

Check the CLI entry points:

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter verify --help
```

If runtime behavior changed, run the related TensorCast tests:

```bash
pytest tests/test_tensor_cast/test_runtime.py tests/test_tensor_cast/test_text_generate.py
```

If the adaptation touches a specific model family, run the closest model or
benchmark smoke path in the repository.

## 15. Submission Checklist

Before submitting a new adaptation, confirm:

- **No working artifacts are committed**: command.txt, doctor.json,
  verify.json, failure.log, st_cases, and everything else under `reports/`
  stay local (the directory is covered by `.gitignore`); they serve the
  development process and the downstream precision alignment workflow
  off-repo.
- The final builtin profile is minimal and source-backed; the model source
  origin (in-repo transformers / dependency upgrade / open-source fetch) was
  made explicit during adaptation.
- `candidate_profile_validation` and `profile_validation` both pass.
- Every patch method comes from deterministic failure evidence, AI
  assistance, and human review.
- New operators have declaration + performance properties + unit tests, and
  the attention-core op table was updated when needed.
- `verify.json` reports `passed: true`, or remaining warnings were explicitly
  reviewed (the report itself stays off-repo).
- ST guardrail cases are generated only from passed verifications; commit
  them under `tests/benchmark/models/cases/` when regression protection is
  wanted.
- No temporary local files, private paths, raw internal notes, or
  walkthroughs are staged.

# TensorCast New Model Adaptation Development Guide (Smoke-Level)

This guide describes the operational workflow for adapting Hugging Face-style new models to TensorCast. The goal of the current process is **smoke-level adaptation**: the new model simulation runs through a basic case without errors, and the call counts of key operators match the public structure of the model (for example, the call counts of the various attention operators). The process does not depend on any measured data. Its only inputs are the simulation command, the installed open-source model source code, and the configuration.

Full shape/dtype validation and precision comparison against measured profiling are handled by the downstream precision workflow, which continues processing the outputs of this process.

Use the design document for architecture and rationale:

```text
https://gitcode.com/Ascend/msmodeling.wiki/blob/main/design/model_adaptation_efficiency_design.md
```

## 1. Scope

Use the workflow in this guide when a model requires one or more of the following adaptations:

- Model profile metadata, for example `model_type`, module paths, or config field mappings.
- MoE metadata, expert count fields, or expert routing compatibility logic.
- MLA, MTP, vision-language, or multimodal module mappings.
- meta device, `torch.compile`, or shape compatibility patches.
- New operator adaptation (operator declaration and compute/memory performance properties).

For dense text models with no special structure, the adaptation workflow may require only a minimal model profile.

## 2. Preparing the Environment

Run all commands from the msModeling repository root in a configured Python environment. If you are not running from the repository root, set `PYTHONPATH` first:

```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
```

Check that the model adaptation CLI is available:

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter verify --help
```

If your AI assistant supports loading project skills, use the model-adaptation skill:

```text
.agents/skills/model-adaptation
```

The skill helps process doctor reports, profile review, human checkpoints, new operator adaptation locations, AI patch tasks, and smoke verification. The deterministic adapter tools remain the source of truth for parsing, validation, and pass/fail checks.

## 3. Preparing Inputs

Create a case directory in `reports/` (a local working directory that is already in `.gitignore`), and prepare the following inputs:

| Input | Save As | Requirement |
| --- | --- | --- |
| TensorCast simulation command | `reports/<case_name>/command.txt` | The complete, runnable workload command |

This process does not require any measured input (no MindStudio Insight export, no profiling data).

Note: All files under `reports/<case_name>/` (`command.txt`, `doctor.json`, `verify.json`, `failure.log`, `st_cases`, and so on) are **working artifacts from the development process and are not committed**. They are for development debugging and for the downstream precision alignment workflow to use outside the repository. The repository only accepts code, tests, documentation, and (optionally) ST guardrail cases.

Optional inputs:

| Input | Save As | Use |
| --- | --- | --- |
| Failure log | `reports/<case_name>/failure.log` | Let doctor classify dry-run or smoke failures that may need a patch. |

## 4. Creating a Case Directory

Replace `<case_name>` with a short stable name, for example `qwen3_vl_8b_prefill`.

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

Example with multimodal and quantization options:

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
  --quantize-attention-action disabled \
  --quantize-linear-action W8A8_DYNAMIC
EOF
```

## 5. Running Doctor

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

Review these fields in `doctor.json`:

| Field                         | Meaning | Action |
| ----------------------------  | --- | --- |
| adaptation_context            | Parsed command and normalized workload arguments | Confirm it matches the expected workload. |
| candidate_profile             | Minimal proposed `ModelProfile` fields | Review against installed source. |
| candidate_profile_validation  | Deterministic validation result for the candidate | Fix errors before registering the profile. |
| candidate_profile_draft       | Draft Python module content | Use as a starting point only. |
| profile                       | Existing registered profile, if any | Use as reference in normal adaptation. |
| profile_validation            | Validation result for the existing profile | Fix errors if present. |
| human_questions               | Minimal facts needed from the user (low-confidence candidate fields, structural gaps) | Confirm against the source code, then finalize them in the profile. |
| ai_tasks                      | Bounded tasks for an AI assistant | Use only after reviewing deterministic findings. |
| patch_reports                 | Dry-run patch pass results | Check expected replacements and skipped modules. |
| suggestions                   | Recommended next actions | Use to decide the next iteration. |

## 6. Locating the Model Source

The model source does not necessarily come from the installed `transformers`. Determine the source in the following order:

1. **The current `transformers` version in the repository already supports the model**: Read the in-library source directly, usually `transformers.models.<model_name>.modeling_<model_name>`.
2. **The latest `transformers` version supports the model, but the version pinned by the repository does not**: First drive an upgrade of the `transformers` dependency version in the repository (`pyproject.toml`/`uv.lock`), then return to case 1 after the upgrade. Do not copy the new model code into the repository just to save effort.
3. **Not even the latest `transformers` version supports the model**: Obtain the source from where the model is actually open-sourced (the Hugging Face model repository or remote code, or implementations in inference frameworks such as vLLM), and adapt the model source as a model-specific module placed in `tensor_cast/transformers/builtin_model/`, following the repository constraint of "adapt at the patch/wrapper layer and do not directly modify upstream dependencies".

In every case, confirm the actual class names, module paths, config fields, and forward behavior in the actual source code. Do not fill profile fields from the model name alone.

## 7. New Operator Adaptation

Add operator adaptation only when the model forward pass uses operator semantics that TensorCast does not yet support. The criteria are the `UNSUPPORTED_OP_ROUTING` category in the dry-run failure log, or a new computation semantic confirmed in the model source. New operator adaptation has two parts, located as follows:

1. **Operator declaration (shape propagation)**: In the corresponding category file under `tensor_cast/ops/` (or a new file), declare the operator with `@register_tensor_cast_op("<name>")`. The function body performs only shape propagation (returns a meta implementation such as `torch.empty_like`), and you register the import in `tensor_cast/ops/__init__.py`. Refer to existing implementations such as `tensor_cast/ops/attention.py` and `tensor_cast/ops/layernorm.py`.

2. **Performance properties (compute/memory formulas)**: In `tensor_cast/performance_model/__init__.py`, register a functor with `@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.<name>.default)` that derives `mma_ops`/`gp_ops` and the memory read/write volumes from the input shape/dtype of `OpInvokeInfo`. You can place model-specific operators in `tensor_cast/performance_model/builtin_model/` or in the corresponding builtin model module. Derive the compute and memory formulas by analyzing the mathematical semantics of the open-source model source (the order of magnitude of the FLOPs and the size of the tensors read and written), and validate the order of magnitude against the implementations of existing operators of the same kind.

Additional constraints:

- If the new operator is an attention computation core (called once per layer) and its operator name does not contain the `attention` substring (for example, `sparse_attn_sharedkv` in DeepSeek-V4), you must also register it in `_ATTENTION_CORE_OPS` in `tensor_cast/adapter/expectations.py`. Otherwise, smoke verification cannot count it correctly.
- The empirical performance model depends on the operator mapping in `op_mapping.yaml`. The downstream precision workflow calibrates this mapping. Therefore, it is out of scope for this process.
- A new operator needs accompanying unit tests: correct meta shape propagation and a reasonable order of magnitude for the performance properties.

## 8. Registering the Reviewed Profile

Move the reviewed draft to:

```text
tensor_cast/transformers/builtin_model/<model_type>.py
```

Use `register_model_profile(ModelProfile(...))`.

Keep the profile minimal:

- Include `model_type`.
- Include only non-default MoE/MLA/MTP/VL fields that are confirmed.
- Use a plain `dict` for `moe_field_names_override`.
- Use list form for nested expert count keys, for example `["text_config", "num_experts"]`.
- Do not write empty overrides.
- Do not write default `None` fields.
- Do not write default `moe_num_experts_key="num_experts"` unless required by the reviewed code path.

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

## 9. Handling Runtime Patch Needs

Use `patch_method` only when the installed model source is incompatible with TensorCast simulation. Common causes:

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

When a patch is needed, doctor emits `PATCH_METHOD_AUTHORING` under `ai_tasks`. Give `ai_tasks[].prompt_text` to the model-adaptation skill or another AI assistant. The assistant should return a patch-method draft, class/method targets, semantic explanation, and verification commands.

Review rules:

- Doctor provides deterministic evidence and a prompt. It does not produce final model-specific patch code.
- AI output is advisory until reviewed.
- Patch only the simulation-incompatible path.
- Preserve normal tensor behavior as closely as possible.
- Rerun doctor and verification after adding the patch.

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
- `candidate_profile_validation.passed` is true or any issue is understood.
- `patch_reports` match the expected replacement and skip counts (MoE layer count, MLA module count, and so on).
- `human_questions` are either empty or confirmed against the source code.

## 11. Running Smoke Verification

verify runs one simulation of the basic case and reconciles the actual call counts of key operators (attention-type and MoE gating-type operators) against the expected counts derived from the structure scan:

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --device <device_profile> \
  --num-queries 1 \
  --query-length 8 \
  --context-length 0 \
  --output reports/<case_name>/verify.json
```

A vision-language model prefill case needs image parameters (the vision tower does not run in a decode case or with image-free input):

```bash
python -m cli.inference.model_adapter verify \
  <model_id> \
  --image-batch-size 1 --image-height 224 --image-width 224 \
  --output reports/<case_name>/verify_vl_prefill.json
```

Verification pass criterion: `verify.json` reports `passed: true` (the simulation runs through and all error-level issues are cleared).

Common verification issues:

| Category                 | Typical Cause | Next Action |
| -----------------------  | --- | --- |
| SIMULATION_ERROR         | The simulation case crashes (meta compatibility, placeholder checks, dynamic shapes, and so on) | The report contains the traceback and patch-discovery `ai_tasks`. Fix the problem per the task prompt and rerun verify. You do not need to collect logs manually. |
| KEY_OP_MISSING           | An attention-type operator was never called, and the model falls back to an unadapted HF module | Check the ModelProfile registration and module replacement. If the message lists an unclassified tensor_cast operator, confirm whether it is a new attention core and register it in `_ATTENTION_CORE_OPS`. |
| OP_COUNT_MISMATCH        | Layer count override, MTP, vision input, or some layers not replaced | Fix the profile or case input. |
| NO_TENSOR_CAST_OPS       | The simulation has no tensor_cast operator at all. Therefore, the model is not adapted. | Register the profile and rerun. |
| MOE_NOT_ADAPTED          | The structure scan finds an MoE module but there is no MoE patch, no gating operator, and no TensorCast MoE path operator (`init_routing_v2`, and so on) | Register or fix the MoE profile fields. |
| STRUCTURE_SCAN_EMPTY     | The structure scan finds no attention module | Check the model construction path and the installed source. |
| EXPECTATION_DEGRADED (warning) | The case enables MTP or PP. Therefore, count verification is degraded. | Rerun strict verification with a basic case (no MTP, `pp_size=1`). |

Note: For some correctly adapted models (such as DeepSeek V3.2), MoE gating uses the standard `torch.topk` path rather than a tensor_cast operator. In this case, verification relies on the MoE patch report from doctor to confirm adaptation completeness, and this is not treated as a problem.

## 12. Generating ST Guardrail Cases

Generate regression cases from passed verification (only passed verification generates cases):

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

- The generated case JSON is fully compatible with the case format in `tests/benchmark/models/test_model_regression.py`. You can submit passed cases to `tests/benchmark/models/cases/` to add them to the precision guardrail network for existing models.
- Failed verification does not generate cases. Fix the problem and verify again.
- Case `user_input` comes from the normalized command arguments, not from manual guesses.

## 13. Replaying or Auditing an Existing Model

Replay mode checks whether the workflow can rediscover an adaptation without reading the existing profile as the answer.

Use `--ignore-existing-profile` only for replay or audit tests:

```bash
python -m cli.inference.model_adapter doctor \
  --from-command-file reports/qwen3_vl_replay/command.txt \
  --ignore-existing-profile qwen3_vl \
  --ignore-existing-profile qwen3_vl_moe \
  --profile-draft-output reports/qwen3_vl_replay/qwen3_vl_replay_draft.py \
  --output reports/qwen3_vl_replay/doctor.json
```

Replay constraints:

- Do not use `tensor_cast/transformers/builtin_model/qwen3_vl.py` as input evidence while replay is running.
- Reading the installed `transformers` Qwen3-VL source is allowed because normal new-model adaptation also uses installed source.
- Existing TensorCast Qwen3-VL profile may be used only after replay discovery completes, as an oracle for comparison.

The accompanying Qwen3-VL blind replay test uses a config-only fixture (`tests/assets/model_config/qwen3_vl_tiny/config.json`) and does not require downloading weights:

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -k qwen3_vl -q
```

## 14. Validation Checklist

Run the focused adapter test suite:

```bash
pytest tests/regression/tensor_cast/test_adapter_automation.py -q
```

Check CLI entry points:

```bash
python -m cli.inference.model_adapter doctor --help
python -m cli.inference.model_adapter verify --help
```

If runtime behavior changed, run relevant TensorCast tests:

```bash
pytest tests/regression/tensor_cast/test_runtime.py tests/regression/tensor_cast/test_text_generate.py
```

If the adaptation touches a specific model family, run the closest model or benchmark smoke path available in the repository.

## 15. Submission Checklist

Before submitting a new adaptation:

- **Do not commit any intermediate process files**: Working artifacts under `reports/`, such as `command.txt`, `doctor.json`, `verify.json`, `failure.log`, and `st_cases`, all stay local (the directory is already in `.gitignore`) for use by the development process and the downstream precision alignment workflow outside the repository.
- The final built-in profile is minimal and source-backed, and the model source origin (in-library/upgraded dependency/obtained from an open-source site) is clear from the adaptation process.
- `candidate_profile_validation` and `profile_validation` pass.
- Any patch method came from deterministic failure evidence, AI assistance, and human review.
- New operators have a declaration, performance properties, and unit tests, and are registered in the attention core operator table when necessary.
- `verify.json` reports `passed: true`, or the remaining warnings are explicitly reviewed (the report itself is not committed).
- ST guardrail cases are generated only from passed verification. Submit them to `tests/benchmark/models/cases/` when regression protection is needed.
- Temporary local files, private paths, raw internal notes, and walkthroughs are not staged.

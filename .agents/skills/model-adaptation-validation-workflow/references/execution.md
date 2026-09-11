# Execution Procedure

Use repository-relative paths in committed examples and absolute paths for the current `npu_layer_compare.py --output-dir` invocation on Windows.

## 1. Case inventory and safe cleanup

Start by resolving the case directory and presenting this manifest before any command runs:

| File | What it represents | Stage that consumes it | Missing behavior |
| --- | --- | --- | --- |
| `command.txt` | Exact TensorCast simulation command: model, device, phase, workload, quantization and parallel options | doctor; verify translation; exact TensorCast simulation; cross-stage consistency | `BLOCKED` before doctor |
| ordinary/MTP run-profile YAML | Concrete input context for Theory-to-Runtime Shape/dtype materialization | semantic guardrail only | Requested scenario is `BLOCKED` with `THEORY_PROFILE_MISSING` |
| kernel-details CSV | Real-device NPU kernel execution details, possibly containing one or more forwards | profiling comparison only | With no trace either: `PENDING`; with trace present: `INCOMPLETE` |
| `trace.json` | TensorCast Chrome Trace containing simulated operator/layer timing and ordering | profiling comparison only | With no CSV either: `PENDING`; with CSV present: `INCOMPLETE` |
| `decisions/*` | Human confirmation of Profile fields, forward selection or operator mappings | checkpoint, summary and calibration loop | Optional until a decision is made; never synthesize acceptance |

Use this user-facing shape:

```text
[INPUT MANIFEST]
PRESENT command.txt — simulation command; used by doctor/verify/exact simulation
MISSING ordinary run profile — required by ordinary Shape/dtype diagnostics
MISSING MTP run profile — required by MTP Shape/dtype diagnostics
PRESENT kernel CSV — measured NPU data; used only by profiling comparison
PRESENT trace.json — TensorCast Chrome Trace; used only by profiling comparison
PRESENT/MISSING decisions/ — reviewed human decisions; preserved and never inferred
```

Do not merely say “file not found”. State what the file means, which stage needs it, whether its absence blocks the full workflow or only one branch, and the expected path. A missing case directory or `command.txt` stops the workflow before doctor. Do not create an immutable input from another same-named file unless the user explicitly identifies that source.

Classify files before deletion:

| Class | Examples | Cleanup rule |
| --- | --- | --- |
| Immutable input | `command.txt`, kernel CSV, `trace.json` | Never delete or overwrite |
| Reviewed decision | Profile checkpoint, mapping decision, accepted gaps | Preserve unless the user explicitly asks to discard decisions |
| Generated output | `doctor*.json`, generated drafts, `verify*.json`, analyzer output | May be regenerated after exact targets are verified |

Do not recursively delete a whole case directory. Remove only listed generated artifacts.

## 2. Base adaptation with the current run-through skill

The base stage has no measured-profiling input. Start with the exact command file:

```text
[STAGE INPUT: BASE ADAPTATION]
USE command.txt — doctor command fact source
DO NOT USE kernel CSV or trace.json — measured data must not influence ModelProfile generation
```

```powershell
python -m cli.inference.model_adapter doctor `
  --from-command-file reports/<case>/command.txt `
  --profile-draft-output-file reports/<case>/generated/<model_type>_draft.py `
  --output-file reports/<case>/generated/doctor.json
```

Do not add profiling or evidence arguments. Review:

- `candidate_profile` and `candidate_profile_validation`;
- `profile` and `profile_validation`;
- `human_questions` and `ai_tasks`;
- `patch_reports` and `suggestions`.

### Mandatory checkpoint after doctor

Do not treat successful candidate generation as registration. If `profile` is null, invalid, or materially different from `candidate_profile`:

1. Print the generated draft path and complete minimal `ModelProfile` block.
2. State whether `candidate_profile_validation` passed and list every issue.
3. Show a field-by-field diff against the registered profile; write `registered profile: absent` when it is null.
4. Keep Profile questions separate from later operator-mapping questions.
5. Ask the user to choose `确认注册`, `仅保留草稿`, or `修改字段`.
6. End the turn and wait. Do not continue merely because the original request asked for an end-to-end run.

After `确认注册`, validate the candidate against the actual model source/config and apply only the approved fields to `tensor_cast/transformers/builtin_model/<model_type>.py`. If new TensorCast semantics are required, follow the base skill's new-operator rules: op declaration, performance properties, focused tests, and `_ATTENTION_CORE_OPS` registration for a non-obvious attention core.

Before doctor revalidation, verify and exact simulation, announce that `command.txt` supplies workload options and that the approved registered Profile supplies adaptation behavior. Do not list the CSV or trace as inputs to these commands.

Rerun doctor without profiling arguments:

```powershell
python -m cli.inference.model_adapter doctor `
  --from-command-file reports/<case>/command.txt `
  --output-file reports/<case>/generated/doctor_after_profile.json
```

Continue only when `profile` is non-null, `profile_validation` passes, and `patch_reports` match the expected replacement structure.

Translate the applicable model, device, workload, quantization, vision, and parallel options from `command.txt` to `verify`. `verify` does not accept `--from-command-file`:

```powershell
python -m cli.inference.model_adapter verify <model_id> `
  --device <device_profile> `
  --num-devices <N> `
  --num-queries <B> `
  --query-length <Q> `
  --context-length <C> `
  --tp-size <TP> `
  --ep-size <EP> `
  --output-file reports/<case>/generated/verify.json
```

Carry over applicable `--decode`, quantization, image, MoE parallel and compile flags. If an original command option has no `verify` equivalent, record the omission explicitly. Treat `EXPECTATION_DEGRADED` for MTP/PP as a warning requiring a strict basic-case verify, not as full strict coverage.

Finally run the exact command stored in `command.txt`. Do not silently remove quantization, compilation, TP, EP, MTP, cache, or multimodal flags to obtain a pass. Optionally generate an ST guardrail only from a passed verification:

```powershell
python -m cli.inference.model_adapter verify <model_id> `
  <translated-options> `
  --st-case-output-path reports/<case>/generated/st_cases `
  --output-file reports/<case>/generated/verify_with_st.json
```

After `仅保留草稿`, record the base stage as `INCOMPLETE`. Stop unless the user separately authorizes downstream validation without a registered Profile.

## 3. Semantic guardrail

Announce the semantic inputs before running:

```text
[STAGE INPUT: SHAPE/DTYPE]
USE <scenario>.yaml — concrete phase, Shape, quantization, parallel and MTP inputs
USE matching Theory Spec — source-derived model structure and symbolic contracts
CHECK command.txt — consistency reference only; it is not loaded by model_diagnostics
DO NOT USE kernel CSV or trace.json — Theory must remain independent of measured Runtime/profiling evidence
```

Find candidates without assuming a filename:

```powershell
rg -n "<model_type>|<model_id_fragment>" tools/model_diagnostics/specs tools/model_diagnostics/profiles tests
```

Then run each matching scenario profile:

```powershell
python -m tools.model_diagnostics <run_profile.yaml> `
  --theory-compare `
  --comparison-report reports/<case>/semantic/theory_runtime.html
```

The Theory Spec describes model structure and optional regions. A run profile describes one invocation, including phase, query/context length, `num_mtp_tokens`, image inputs, quantization and parallelism. Do not remove MTP capability from Theory merely because the ordinary scenario uses `num_mtp_tokens=0`; use a separate MTP run profile when that path must be validated.

Current CLI emits console status and optional HTML. Until a JSON diagnostics renderer exists, preserve stdout/stderr and do not parse prose into invented findings.

If no matching Spec/profile exists, report:

```text
semantic_guardrail: BLOCKED
code: THEORY_PROFILE_MISSING
action: author an independent Theory Spec and run profile from official model source/config, then rerun
```

## 4. Profiling branch

Full ordered comparison requires both sides:

```text
[STAGE INPUT: PROFILING COMPARISON]
USE <kernel-details.csv> — measured NPU side
USE trace.json — matching TensorCast Chrome Trace side
CHECK command.txt — context consistency only
```

```powershell
$env:PYTHONUTF8 = "1"
python tests/npu_layer_analyzer/npu_layer_compare.py `
  --csv <absolute-kernel-details.csv> `
  --json <absolute-tensorcast-trace.json> `
  --output-dir <absolute-output-directory>
```

`PYTHONUTF8=1` prevents Windows' default GBK console from failing while the analyzer prints Unicode layer summaries. This changes only process text encoding, not analyzer semantics or input data.

Interpret outputs conservatively:

- `npu_out.xlsx`: forward candidates and extracted NPU layers;
- `layer_out.xlsx`: TensorCast trace layer extraction;
- `compare_result.xlsx`: positional Stage presentation, not a reviewed semantic mapping;
- if 62-layer work appears 124 times in a two-forward file, first verify `forward_count=2`; never silently divide.

## 5. MiniMax-M2.7 example

For `reports/minimax2_decode`, use `command.txt` as the base-adaptation input. Kernel CSV and trace files belong only to the profiling branch and stay outside `doctor`/`verify`.

After the Profile checkpoint is resolved, run ordinary Decode and MTP as separate diagnostics scenarios when matching profiles exist:

```powershell
python -m tools.model_diagnostics `
  reports/minimax2_decode/minimax_m2_7_decode_w8a8_tp8_ep16.yaml `
  --theory-compare `
  --comparison-report reports/minimax2_decode/generated/minimax_m2_7_theory_runtime.html

python -m tools.model_diagnostics `
  reports/minimax2_decode/minimax_m2_7_decode_mtp3_w8a8_tp8_ep16.yaml `
  --theory-compare `
  --comparison-report reports/minimax2_decode/generated/minimax_m2_7_mtp3_theory_runtime.html
```

These paths are case inputs, not assets bundled with this skill. If the MiniMax Theory Spec or profiles are absent on the current branch, report `THEORY_PROFILE_MISSING` and hand off their source-based implementation instead of substituting another model family's contract.

## 6. Summary status

Before writing the summary, announce that it consumes the current run's generated reports plus preserved `decisions/*`; it must not reuse conclusions from an old `generated/` directory.

Use this precedence:

```text
FAIL > BLOCKED > INCOMPLETE > PENDING > UNSUPPORTED > PASS
```

List every command exactly as executed and mark commands not run. Do not say “all passed” when a stage was unavailable.

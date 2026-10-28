# Execution Procedure

Use repository-relative paths in committed examples and absolute paths for the current `npu_layer_compare.py --output-dir` invocation on Windows.

## Mode dispatch

Read only the sections required by the selected execution mode:

| Mode | Sections to execute | Earlier-stage prerequisite |
| --- | --- | --- |
| `full` | [intake](#0-natural-language-intake) → [inventory](#1-case-inventory-and-safe-cleanup) → [base adaptation](#2-base-adaptation-with-the-current-run-through-skill) → [semantic guardrail](#3-semantic-guardrail) → [profiling](#4-profiling-branch) → [summary](#5-summary-status) | none; the workflow establishes each prerequisite |
| `adaptation-only` | [intake](#0-natural-language-intake) → [inventory](#1-case-inventory-and-safe-cleanup) → [base adaptation](#2-base-adaptation-with-the-current-run-through-skill) → [summary](#5-summary-status) | approved simulation command |
| `semantic-only` | [inventory](#1-case-inventory-and-safe-cleanup) → [semantic guardrail](#3-semantic-guardrail) → [summary](#5-summary-status) | matching Theory Spec and run-profile YAML; no doctor/Profile prerequisite |
| `profiling-only` | [inventory](#1-case-inventory-and-safe-cleanup) → [profiling](#4-profiling-branch) → [summary](#5-summary-status) | matching kernel CSV and TensorCast trace; no semantic prerequisite |
| `calibration-only` | inventory the supplied finding, then switch to `model-adaptation-calibration-loop` | latest summary/finding, evidence, and explicit repair authorization |

Do not execute or report unrequested sections. Mark them `NOT_REQUESTED`. `PENDING` means an optional stage was reached but lacks optional evidence; it does not mean the stage was outside the selected mode.

## 0. Natural-language intake

In `full` or `adaptation-only`, when the user has not supplied an existing `reports/<case_name>/command.txt`, do not fail immediately and do not display the whole workflow questionnaire. Read [interactive-dialog.md](interactive-dialog.md), load `text-generate-executor`, and progressively collect at most three missing inputs per turn. Show and confirm the complete simulation command before creating the case directory or writing `command.txt`.

In `semantic-only` or `profiling-only`, skip command construction unless the user separately requests a TensorCast simulation. Gather only the inputs required by that standalone mode.

Once the command is confirmed for `full` or `adaptation-only`, `command.txt` becomes the immutable workload fact source used by those sections. `semantic-only` and `profiling-only` do not create a command merely to satisfy the full-workflow layout. If the user requested simulation only, stop after that confirmed executor flow and offer the full validation workflow as an optional next step.

## 1. Case inventory and safe cleanup

Start by resolving the case directory and presenting this manifest before any command runs:

| File | What it represents | Stage that consumes it | Missing behavior |
| --- | --- | --- | --- |
| `command.txt` | Exact TensorCast simulation command: model, device, phase, workload, quantization and parallel options | doctor; verify translation; exact TensorCast simulation; optional cross-stage consistency | Required by `full`/`adaptation-only`; optional in standalone semantic/profiling modes |
| ordinary/MTP run-profile YAML | Concrete input context for Theory-to-Runtime Shape/dtype materialization | semantic guardrail only | Requested scenario is `BLOCKED` with `THEORY_PROFILE_MISSING` |
| kernel-details CSV | Real-device NPU kernel execution details, possibly containing one or more forwards | profiling comparison only | With no trace either: `PENDING`; with trace present: `INCOMPLETE` |
| `trace.json` | TensorCast Chrome Trace containing simulated operator/layer timing and ordering | profiling comparison only | With no CSV either: `PENDING`; with CSV present: `INCOMPLETE` |
| `decisions/*` | Human confirmation of Profile fields, forward selection or operator mappings | checkpoint, summary and calibration loop | Optional until a decision is made; never synthesize acceptance |

For `full`, use this user-facing shape:

```text
[INPUT MANIFEST]
PRESENT command.txt — simulation command; used by doctor/verify/exact simulation
MISSING ordinary run profile — required by ordinary Shape/dtype diagnostics
MISSING MTP run profile — required by MTP Shape/dtype diagnostics
PRESENT kernel CSV — measured NPU data; used only by profiling comparison
PRESENT trace.json — TensorCast Chrome Trace; used only by profiling comparison
PRESENT/MISSING decisions/ — reviewed human decisions; preserved and never inferred
```

For `adaptation-only`, report only `command.txt` and any existing human decision evidence. For `semantic-only`, report the run profile and matching Theory Spec; `command.txt`, kernel CSV, and `trace.json` are not prerequisites. For `profiling-only`, report the kernel CSV and `trace.json`; a run profile and Theory Spec are not prerequisites unless semantic validation is also explicitly requested.

Do not merely say “file not found”. State what the file means, which requested mode needs it, and the expected path. In `full`/`adaptation-only`, a missing `command.txt` stops before doctor. In `semantic-only`, a missing command is not an error; a missing Spec/profile is. In `profiling-only`, a missing command is not an error; a missing CSV or trace is. Do not create an immutable input from another same-named file unless the user explicitly identifies that source.

Classify files before deletion:

| Class | Examples | Cleanup rule |
| --- | --- | --- |
| Immutable input | `command.txt`, kernel CSV, `trace.json` | Never delete or overwrite |
| Reviewed decision | Profile checkpoint, mapping decision, accepted gaps | Preserve unless the user explicitly asks to discard decisions |
| Generated output | `doctor*.json`, generated drafts, `verify*.json`, analyzer output | May be regenerated after exact targets are verified |

Do not recursively delete a whole case directory. Remove only listed generated artifacts.

## 2. Base adaptation with the current run-through skill

Applies only to `full` and `adaptation-only`. Skip as `NOT_REQUESTED` in all other modes.

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

Applies to `full` and `semantic-only`. In `semantic-only`, start here after the mode-scoped inventory; do not require doctor, Profile registration, verify, or exact simulation.

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

### Mandatory checkpoint when Theory assets are missing

`THEORY_PROFILE_MISSING` is a hard interactive checkpoint. It does not merely block the semantic branch while the workflow continues automatically.

In `semantic-only`, report the missing Theory asset and stop that standalone run. Do not offer “skip Shape/dtype and continue profiling”, because profiling is outside the requested mode. The supplement/skip/end choices below apply to `full` only.

If the model has no matching Theory Spec, show:

```text
[CHECKPOINT: THEORY SPEC]

workflow_status: PAUSED
semantic_guardrail: BLOCKED
code: THEORY_PROFILE_MISSING

COMPLETED  doctor; Profile validation; verify; exact TensorCast simulation
MISSING    matching Theory Spec and the requested scenario run profile
NOT RUN    Shape/dtype comparison; profiling comparison

Choose one:
1. 确认补充 Theory Spec
2. 跳过 Shape/dtype，继续 profiling
3. 结束并生成阶段性报告
```

List the concrete source-derived assets needed for the current model, such as the main Spec, decoder/attention/FFN fragments and scenario run profile. Write or update an interim `workflow_summary.md` with `Overall status: PAUSED` and the current checkpoint, then end the turn. Do not inspect or execute the profiling comparison beyond the input inventory already shown.

The original full-workflow request is not approval to choose option 1 or 2. Resume only after an explicit reply:

- `确认补充 Theory Spec`: author the missing Theory assets from official source/config, validate them, and resume at the semantic stage;
- `跳过 Shape/dtype，继续 profiling`: record the semantic gap in `decisions/` and the summary, then enter the profiling branch;
- `结束并生成阶段性报告`: keep downstream stages `NOT_RUN` and finish with the interim report.

If the Theory Spec exists but only the concrete run profile is missing, generate the draft from the approved command and public source/config facts, show the complete YAML, validation result and field mapping, then end the turn for approval. Do not save or execute the draft before that approval.

## 4. Profiling branch

Applies to `full` and `profiling-only`.

In `full`, enter this section only after Shape/dtype validation passes or the user explicitly chooses `跳过 Shape/dtype，继续 profiling` and that accepted gap is recorded. A missing Theory checkpoint never falls through automatically. In `profiling-only`, enter directly after confirming that CSV and trace describe the same scenario; no Theory result is required.

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
- if work for `N` model layers appears `2N` times in a two-forward file, first verify `forward_count=2`; never silently divide.

## 5. Summary status

Before writing the summary, announce that it consumes the current run's generated reports plus preserved `decisions/*`; it must not reuse conclusions from an old `generated/` directory.

When a mandatory checkpoint is waiting for user input, write an interim summary with `Overall status: PAUSED`, record `Current checkpoint`, and leave later stages in the selected mode as `NOT_RUN`. Mark stages outside the selected mode `NOT_REQUESTED`. Do not convert the pause into a final `BLOCKED` workflow merely because the current stage is blocked.

Use this precedence for stages that were requested and executed:

```text
FAIL > BLOCKED > INCOMPLETE > PENDING > UNSUPPORTED > PASS
```

`PAUSED` is a workflow-control state while waiting for required human input. `NOT_RUN` and `NOT_REQUESTED` are reported explicitly and excluded from the precedence calculation.

List every command exactly as executed and mark commands not run. Do not say “all passed” when a stage was unavailable.

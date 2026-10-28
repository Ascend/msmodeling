---
name: model-adaptation-validation-workflow
description: "Interactively run either the full TensorCast model-adaptation validation workflow or one standalone stage: base adaptation, Theory-to-Runtime shape/dtype diagnostics, NPU profiling comparison, or calibration handoff. Use for natural-language guided requests and existing reports cases; use model-adaptation-calibration-loop for mapping decisions or code repair."
metadata:
  version: 0.7.0
  source: local-rfc
---

# TensorCast Model Adaptation Validation Workflow

Run one auditable validation case from a natural-language goal or existing case inputs to a run-through, semantic and, when supported by evidence, profiling comparison result.

## Execution Modes and Natural-Language Routing

Resolve the requested mode before collecting files or running commands. An explicit narrow request never authorizes a broader workflow.

| Mode | Natural-language intent | Stages | Required inputs |
| --- | --- | --- | --- |
| `full` | “完整工作流”“模型适配精度闭环” | adaptation → semantic → optional profiling → report/handoff | approved simulation command; matching Theory run profile for semantic; CSV + trace only when profiling is requested/available |
| `adaptation-only` | “只做模型适配”“只跑 doctor/verify/仿真” | current `model-adaptation` doctor/Profile checkpoint/verify/exact simulation | approved simulation command |
| `semantic-only` | “只跑算子精度防护网”“只检查 Shape/dtype”“只做 theory compare” | `tools.model_diagnostics` only | matching Theory Spec and diagnostics run-profile YAML |
| `profiling-only` | “只做实测对账”“只比较 CSV 和 trace” | `npu_layer_compare.py` only | matching kernel-details CSV and TensorCast `trace.json` |
| `calibration-only` | “进入校准”“审核并修复非通过项” | hand off immediately to `model-adaptation-calibration-loop` | latest finding/summary, exact evidence, and user authorization required by that skill |

`simulation-only` is not an adaptation-validation mode. Route it to `text-generate-executor` and do not expand scope automatically.

Standalone modes do not require earlier workflow stages to have run. They do require their own inputs and invariants:

- `semantic-only` does not require `command.txt`, doctor, Profile registration, CSV, or trace. When `command.txt` is available, use it only as a consistency reference.
- `profiling-only` does not require doctor or a Theory pass. Require CSV and trace to describe the same scenario; use `command.txt` only when available to verify context.
- `adaptation-only` never consumes measured profiling evidence.
- `calibration-only` cannot start without an existing non-pass finding and evidence; it is not an initial execution shortcut.
- `full` retains the ordered semantic checkpoint and must not fall through to profiling while Theory is unresolved unless the user explicitly accepts that gap.

If the user's wording does not distinguish `full` from a standalone mode, ask one short scope question before requesting stage inputs. Record the chosen mode in the conversation ledger and output summary.

For copyable natural-language invocation examples for all five modes, read [references/interactive-dialog.md](references/interactive-dialog.md#7-copyable-mode-invocation-examples).

## Interaction Contract

Support two input entry styles independently of the execution mode:

1. **Natural-language guided mode**: the user may provide only a model alias, scenario, and desired validation scope. Do not require a pre-existing case directory. Summarize confirmed facts, ask at most three missing questions per turn, and follow [references/interactive-dialog.md](references/interactive-dialog.md). Use `text-generate-executor` for command construction only when the selected mode needs a simulation command (`full` or `adaptation-only`).
2. **Existing-case mode**: when the user identifies `reports/<case_name>`, inventory the existing inputs and continue from the first unresolved checkpoint.

Maintain a visible conversation ledger with `CONFIRMED`, `MISSING`, `PROPOSED`, and `NEXT`. Never silently fill a missing value from prior conversation or an unrelated case. When the user is unsure, offer a small recommendation and label its source as a CLI default, inspected repository example, optimizer result, or explicit heuristic.

Do not ask for inputs belonging to unrequested modes. Collect only what the selected mode's next stage needs:

- simulation intent first for `full` or `adaptation-only`;
- Profile approval only after doctor produces evidence;
- Theory Spec/run-profile identity first for `semantic-only`;
- matching kernel CSV/trace paths and scenario identity first for `profiling-only`.

Before executing a generated command or writing it to `command.txt`, show the full command, assumptions and selected scope, then ask for explicit confirmation. This applies only to modes that need a simulation command. After confirmation, create or reuse `reports/<case_name>`, save the approved command as its fact source, and continue. `semantic-only` and `profiling-only` may operate on explicitly identified inputs without creating `command.txt`. If the user requested only simulation, finish through `text-generate-executor` and offer—without automatically expanding scope—to continue with the full validation workflow.

## Boundaries

- In `full` and `adaptation-only`, execute `.agents/skills/model-adaptation/SKILL.md` and use its current CLI contract. That stage accepts the simulation command and public model source/config only; never pass measured profiling, kernel counts, or measured latency to `doctor` or `verify`.
- In `full` and `adaptation-only`, add a mandatory generated-Profile approval checkpoint around the base skill. A broad workflow request does not authorize registering a generated candidate. Pause when the registered Profile is absent, invalid, or materially different from the candidate.
- Use `tools.model_diagnostics` as the only Theory-to-Runtime shape/dtype oracle. Never derive or edit Theory to match a captured Runtime trace.
- Use `tests/npu_layer_analyzer/npu_layer_compare.py` for the current profiling comparison prototype. Do not claim that its positional operator table proves semantic one-to-one equivalence.
- This skill executes and reports. It does not approve uncertain mappings or repair adaptation code. Hand those findings to `model-adaptation-calibration-loop`.
- Preserve `command.txt`, kernel-details CSV, traces, mapping decisions, and other human-reviewed inputs. Clean only reproducible generated outputs.

## Read Before Running

For a natural-language or incomplete request, first read [references/interactive-dialog.md](references/interactive-dialog.md). Once the execution mode and its inputs are known, read the applicable sections of [references/execution.md](references/execution.md). Do not reuse conclusions from old generated reports after the user requests a clean rerun.

## Input and Evidence Rules

- `full` and `adaptation-only` use an approved `command.txt`; create it under `reports/<case_name>/` only after command confirmation.
- `semantic-only` requires a matching Theory Spec and diagnostics run-profile YAML. A command is optional consistency evidence, not a prerequisite.
- `profiling-only` requires a matching kernel-details CSV and TensorCast `trace.json`; neither file enters doctor, verify, or Theory comparison.
- Preserve reviewed `decisions/` records and original inputs. Never copy, guess, or derive a missing immutable input from unrelated cases or Runtime evidence.
- Before running, print the selected mode's input manifest; before each stage, identify files used, deliberately omitted, and missing. Follow the exact notices and missing-input behavior in [references/execution.md](references/execution.md#1-case-inventory-and-safe-cleanup).
- Missing optional profiling does not block earlier `full` stages. Missing required semantic assets causes `THEORY_PROFILE_MISSING`; missing CSV or trace makes requested standalone profiling `INCOMPLETE`.

## Workflow

0. Resolve both the input entry style and execution mode. In guided `full`/`adaptation-only`, collect and confirm the simulation command before persisting it. In `semantic-only` or `profiling-only`, collect only that mode's inputs. A request for simulation alone does not authorize Profile registration, Theory authoring, profiling comparison, or code repair.
1. Inventory the selected mode's inputs, print the mode-scoped manifest, and classify every supplied file as immutable input, reviewed decision, or generated output. Record the execution mode, repository branch, commit, dirty paths, Python executable, and input hashes. Do not modify unrelated dirty files.
2. For `full` or `adaptation-only`, run the complete current `model-adaptation` flow:
   - run `doctor` from `command.txt` without profiling evidence and review all candidate, validation, question, task, patch, and suggestion fields;
   - when registration is unresolved, present the complete draft and field diff, then pause for `确认注册`, `仅保留草稿`, or `修改字段`;
   - after approval, register only reviewed fields, rerun doctor, run `model_adapter verify`, and execute the exact simulation command. Label any reduced smoke case as supplementary.
3. For `full` or `semantic-only`, locate diagnostics profiles whose model id/type and execution contexts match the requested scenarios. If no matching Theory Spec exists, present a `THEORY_PROFILE_MISSING` checkpoint. In `full`, offer supplement/skip/end choices and do not continue automatically. In `semantic-only`, report the missing Spec/profile and stop; there is no downstream profiling stage to skip to. If a matching Theory Spec exists but the concrete run profile is missing, build a draft only from an approved command or explicitly confirmed scenario plus public source/config facts, show the complete draft and field mapping, then pause for approval before saving or running it. Never derive Theory from Runtime evidence. After the checkpoint is resolved, run `python -m tools.model_diagnostics <profile> --theory-compare`. Run ordinary and MTP/multimodal/parallel scenarios separately when requested.
4. For `full`, enter profiling only after semantic passes or the user records an accepted semantic gap. For `profiling-only`, enter profiling directly after validating that CSV and trace match the same scenario; no semantic result is required. Run `npu_layer_compare.py` with an absolute output directory. If either side is missing, mark the standalone mode `INCOMPLETE`; do not infer the missing side.
5. Summarize executed stages as `PASS`, `FAIL`, `INCOMPLETE`, `PENDING`, `UNSUPPORTED`, or `BLOCKED`; use `PAUSED` for a required human checkpoint, `NOT_RUN` for a requested but unexecuted stage, and `NOT_REQUESTED` for stages outside the selected mode. A pending profiling stage cannot be reported as a complete performance pass.
6. Write or update `workflow_summary.md` using `assets/workflow-summary-template.md`. Record the execution mode. Mark stages outside the requested mode `NOT_REQUESTED`, not `PENDING` or `PASS`. Include exact commands, artifacts, findings, limitations, human decisions, and the next action.
7. For `calibration-only`, or when another mode produces a finding that needs a mapping judgment or code change, use `model-adaptation-calibration-loop` and obey its entry conditions.

## Stop Conditions

Stop and report instead of guessing when:

- a generated `ModelProfile` is awaiting explicit user approval;
- `full`/`adaptation-only` has no approved command, `semantic-only` has no matching Spec/profile, or `profiling-only` lacks either CSV or trace;
- multiple possible command files or profiling CSV files exist and the intended input cannot be proven;
- the simulation and profiling contexts differ materially;
- the Theory profile is absent or does not match the model;
- forward selection has multiple plausible candidates;
- the current tool input cannot prove order;
- a mapping is low-confidence or requires source-chain evidence;
- a code fix would alter public behavior beyond the case.

For every stop condition, show the current checkpoint, what has completed, what is missing, the exact allowed responses, and the consequence of each response. End the current turn immediately after the checkpoint. A prior request for the full workflow does not implicitly authorize skipping or authoring missing Theory assets.

## Completion Criteria

- Original inputs and human decisions remain intact and identifiable; every executed stage has reproducible commands and reports.
- The mode is explicit, inputs and notices explain what was used or missing, and unrequested stages are `NOT_REQUESTED`.
- Any generated command and Profile were confirmed before persistence or downstream use; profiling evidence never entered doctor or verify.
- Requested semantic validation either ran or stopped at an explicit blocker. In `full`, profiling never crossed an unresolved Theory checkpoint.
- Requested profiling used matching CSV/trace evidence or is explicitly `INCOMPLETE`.
- The summary distinguishes facts, inferences, accepted gaps, remaining risks, and any repair handoff to `model-adaptation-calibration-loop`.

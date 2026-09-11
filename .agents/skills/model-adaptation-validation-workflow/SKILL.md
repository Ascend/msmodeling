---
name: model-adaptation-validation-workflow
description: Run an end-to-end TensorCast model adaptation validation case by chaining the current run-through model-adaptation flow, Theory-to-Runtime shape/dtype diagnostics, and tests/npu_layer_analyzer profiling comparison. Use for full case execution and report handoff; use model-adaptation-calibration-loop for mapping decisions or code repair.
metadata:
  version: 0.5.0
  source: local-rfc
---

# TensorCast Model Adaptation Validation Workflow

Run one auditable validation case from its original inputs to a run-through, semantic and, when supported by evidence, profiling comparison result.

## Boundaries

- Execute `.agents/skills/model-adaptation/SKILL.md` as the first validation stage and use its current CLI contract. That stage accepts the simulation command and public model source/config only; never pass measured profiling, kernel counts, or measured latency to `doctor` or `verify`.
- Add a mandatory generated-Profile approval checkpoint around the base skill. A full-workflow request does not authorize registering a generated candidate. Pause when the registered Profile is absent, invalid, or materially different from the candidate.
- Use `tools.model_diagnostics` as the only Theory-to-Runtime shape/dtype oracle. Never derive or edit Theory to match a captured Runtime trace.
- Use `tests/npu_layer_analyzer/npu_layer_compare.py` for the current profiling comparison prototype. Do not claim that its positional operator table proves semantic one-to-one equivalence.
- This skill executes and reports. It does not approve uncertain mappings or repair adaptation code. Hand those findings to `model-adaptation-calibration-loop`.
- Preserve `command.txt`, kernel-details CSV, traces, mapping decisions, and other human-reviewed inputs. Clean only reproducible generated outputs.

## Read Before Running

Read [references/execution.md](references/execution.md). Use its MiniMax section for a MiniMax-M2.7 case. Do not reuse conclusions from old generated reports after the user requests a clean rerun.

## Inputs

| File | Meaning | Used by | Requirement |
| --- | --- | --- | --- |
| `command.txt` | Exact TensorCast simulation command and workload fact source | doctor, verify option translation, exact simulation, scenario-consistency checks | Required in the case root |
| diagnostics run-profile YAML | One concrete ordinary/MTP/multimodal Shape/dtype scenario | `tools.model_diagnostics` semantic guardrail | Required for every requested semantic scenario |
| kernel-details CSV | Measured NPU kernel records from the target device | `npu_layer_compare.py` profiling side | Required only for profiling comparison |
| `trace.json` | TensorCast Chrome Trace timeline for the matching simulation | `npu_layer_compare.py` TensorCast side | Required only for profiling comparison |
| `decisions/` records | Human-approved Profile, forward and operator-mapping decisions | approval checkpoints, summary and calibration handoff | Optional initially; preserve once created |

The case directory is normally `reports/<case_name>`. `command.txt` must be in its root. CSV and trace files never enter doctor, verify or Theory comparison.

The absence of profiling does not block run-through or semantic validation. The absence of a matching Theory Spec/run profile blocks the semantic gate with `THEORY_PROFILE_MISSING`; do not invent one from Runtime evidence.

## Mandatory Input Notices

Before doing work, print an input manifest with `file`, `meaning`, `stages`, `resolved path`, and `PRESENT`/`MISSING` status. Explain every missing input instead of only printing a path error.

Before each stage, print a short `STAGE INPUT` notice that says which files will be read and which available files are deliberately not used in that stage. In particular:

- doctor reads only `command.txt` plus public model source/config;
- verify and exact simulation derive their workload from `command.txt` and use the approved registered Profile;
- Shape/dtype diagnostics read a run-profile YAML and Theory Spec; `command.txt` is used only to check scenario consistency;
- profiling comparison reads both kernel CSV and `trace.json`;
- the final summary reads generated reports and preserved human decisions.

If the case directory or `command.txt` is missing, stop before doctor, show the expected directory layout, and ask the user to provide or identify the exact command source. Do not create or guess an immutable input. If both CSV and trace are absent, mark profiling `PENDING`; if only one is present, mark it `INCOMPLETE` and name the missing counterpart. Missing profiling must not block earlier stages.

## Workflow

1. Inventory the case, print the mandatory input manifest, and classify every file as immutable input, reviewed decision, or generated output. Record the repository branch, commit, dirty paths, Python executable, and input hashes. Do not modify unrelated dirty files.
2. Run the complete current `model-adaptation` flow first:
   - run `doctor` from `command.txt` without any profiling/evidence arguments;
   - review `candidate_profile`, both Profile validation results, `human_questions`, `ai_tasks`, `patch_reports`, and `suggestions`;
   - if registration is unresolved, present the draft and field diff, ask for `确认注册`, `仅保留草稿`, or `修改字段`, then stop the turn;
   - after approval, register only the reviewed fields, run focused checks, and rerun `doctor` until the registered Profile is non-null and valid;
   - run `model_adapter verify` with the applicable workload, quantization, vision, and parallel options translated from `command.txt` to check run-through and key Attention/MoE call counts;
   - run the exact simulation command. A reduced smoke case is supplementary and must be labelled as such.
3. Locate diagnostics profiles whose model id/type and execution contexts match the requested scenarios. Run `python -m tools.model_diagnostics <profile> --theory-compare`. Run ordinary and MTP/multimodal/parallel scenarios separately when requested; a Theory Spec describes model capability while each run profile describes one concrete case.
4. If both kernel-details CSV and TensorCast trace are available, run the NPU layer comparison with an absolute output directory. If either side is missing, mark profiling comparison `PENDING` or `INCOMPLETE`; do not infer the missing side.
5. Summarize each stage as `PASS`, `FAIL`, `INCOMPLETE`, `PENDING`, `UNSUPPORTED`, or `BLOCKED`. A pending profiling stage cannot be reported as a complete performance pass.
6. Write or update `workflow_summary.md` using `assets/workflow-summary-template.md`. Include exact commands, artifacts, findings, limitations, human decisions, and the next action.
7. If any finding needs a mapping judgment or code change, create a handoff section and use `model-adaptation-calibration-loop`.

## Stop Conditions

Stop and report instead of guessing when:

- a generated `ModelProfile` is awaiting explicit user approval;
- the case directory or required `command.txt` is absent;
- multiple possible command files or profiling CSV files exist and the intended input cannot be proven;
- the simulation and profiling contexts differ materially;
- the Theory profile is absent or does not match the model;
- forward selection has multiple plausible candidates;
- the current tool input cannot prove order;
- a mapping is low-confidence or requires source-chain evidence;
- a code fix would alter public behavior beyond the case.

## Completion Criteria

- Original inputs remain intact and are identifiable.
- The initial manifest and every stage input notice identify what was used, omitted, missing, and why.
- The base stage uses the current run-through `model-adaptation` interface; measured profiling never enters `doctor` or `verify`.
- Any generated-Profile decision is recorded, and downstream validation did not silently cross an unresolved approval checkpoint.
- The base stage has reproducible doctor and verify reports plus the exact simulation result.
- The semantic stage ran, or has an explicit blocker explaining the missing Spec/profile.
- The profiling branch ran when sufficient evidence exists; otherwise its status is `PENDING` or `INCOMPLETE`.
- The summary distinguishes facts, inferences, accepted gaps, and unresolved risks.
- Every repair need has a concrete handoff to `model-adaptation-calibration-loop`.

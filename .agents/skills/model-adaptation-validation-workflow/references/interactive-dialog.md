# Interactive Dialog Procedure

Use this procedure when the user starts from a natural-language goal or incomplete stage inputs. It resolves the execution mode first. It converts the conversation into a reviewed `command.txt` only for `full` or `adaptation-only`; standalone semantic/profiling modes collect their own inputs instead.

## 1. Resolve intent and scope

Extract only facts explicitly present in the latest request:

- model alias or exact model id;
- Prefill, Decode, or unspecified phase;
- requested outcome: simulation only, `adaptation-only`, `semantic-only`, `profiling-only`, `calibration-only`, or `full`;
- any supplied workload, device, parallel, quantization, MTP, multimodal or trace options.

If the user asks only to simulate one forward, route the run through `text-generate-executor`. Do not silently expand it into model adaptation or profiling. If the user asks for `full` or `adaptation-only`, use the same executor to build the command and then continue through the selected mode. If the user asks only for Shape/dtype or profiling comparison, do not collect a simulation command unless it is actually missing evidence needed for that standalone request.

## 2. Keep a conversation ledger

At each turn show a compact ledger:

```text
[INTAKE]
CONFIRMED  model=DeepSeek V3.1; phase=Decode forward
MISSING    device profile; total devices; batch/num-queries
PROPOSED   none
NEXT       answer the three questions below
```

Rules:

- Ask no more than three short questions in one turn.
- Ask only fields needed by the next decision; do not print the complete CLI option catalog.
- Keep an alias as an alias until the exact model id is verified from user input or an inspected source.
- Do not reuse an earlier case unless the user explicitly says to reuse or continue it.
- A proposed value is not confirmed. Label its source and wait for acceptance.
- If the user says “不知道” or asks for a recommendation, inspect the CLI/local examples when useful, then offer at most two choices with trade-offs.

## 3. Collect parameters progressively

For `full` or `adaptation-only`, load and follow `../../text-generate-executor/SKILL.md` and its dialog reference for the authoritative `text_generate` parameter rules. The normal question order is:

1. identity and deployment: exact model id when ambiguous, device profile, total devices, batch/`--num-queries`;
2. workload: Prefill/Decode, query length, and Decode context length;
3. parallelism: TP/DP and, only for MoE, EP/MOE-TP/MOE-DP choices;
4. capability branches: MTP, multimodal input, prefix cache where applicable;
5. performance and precision: analytic/profiling mode, quantization and compile;
6. optional diagnostics: Chrome trace, graph log and input-shape dump.

Skip questions already answered. Group coupled values, such as all three image dimensions, while still keeping the turn to at most three questions.

For `semantic-only`, ask only for the run-profile path or the concrete scenario fields needed to review a run-profile draft, then confirm the matching Theory Spec. Do not ask for CSV or trace.

For `profiling-only`, ask only for the kernel CSV, TensorCast trace, output directory, and enough scenario identity to establish that both inputs match. Do not ask for doctor/Profile/Theory inputs.

For `calibration-only`, ask for the latest summary/finding and evidence location, then load `model-adaptation-calibration-loop`; do not start a fresh validation run.

## 4. Confirm before writing or running

When `full` or `adaptation-only` command values are complete, show:

- requested scope;
- scenario and workload;
- parallelism;
- performance model and quantization;
- optional debug/trace choices;
- assumptions and their sources;
- the exact one-line command.

Ask the user to choose `确认执行`, `修改参数`, or `仅保存命令`. Do not run the command, create a case, or write `command.txt` before this confirmation.

Handle the selected response explicitly:

- `修改参数`: return to parameter collection. Do not create the case, save the command, or execute any stage.
- `仅保存命令`: resolve and, when needed, confirm the case directory; create it; save the exact approved command as `reports/<case_name>/command.txt`; record the command status as `saved only`; then end the current turn. Do not print a stage-execution manifest or run doctor, verify, simulation, semantic diagnostics, or profiling comparison.
- `确认执行`: continue with the execution steps below.

After `确认执行`:

1. derive a readable case name from the confirmed model and phase; if an existing directory would be reused or overwritten, show its path and ask which case to use;
2. create `reports/<case_name>/` when needed;
3. save the exact approved command as `reports/<case_name>/command.txt`;
4. print the mode-scoped case input manifest;
5. execute only the confirmed mode. For `full`, continue with `model-adaptation`, semantic diagnostics and the optional profiling branch. For `adaptation-only`, stop after the exact simulation and report.

For `semantic-only` or `profiling-only`, show the resolved input paths and scenario assumptions, ask for confirmation only when an input choice is ambiguous, and execute without manufacturing `command.txt`.

## 5. Introduce later evidence only when needed

Do not ask a first-time simulation user for run-profile YAML, kernel CSV or `trace.json`.

- Before Shape/dtype validation, locate an existing matching Theory Spec and run profile. If the Spec exists but the concrete run profile does not, construct a profile draft from an approved command or explicitly confirmed scenario plus source/config facts, show the complete draft and validation result, and end the turn for approval. Keep Theory independent of Runtime evidence. If the Spec is absent in `full`, offer `确认补充 Theory Spec`, `跳过 Shape/dtype，继续 profiling`, or `结束并生成阶段性报告`. In `semantic-only`, report the missing asset and stop without offering an unrelated profiling continuation.
- Before profiling comparison, explain that kernel CSV is measured NPU data and `trace.json` is the matching TensorCast Chrome Trace. Ask for their paths only if the user wants this stage or requested the full workflow.
- Before any mapping or code change, hand off to `model-adaptation-calibration-loop` and wait for the required human decision.

## 6. Example: DeepSeek V3.1 Decode forward

User:

```text
我想仿真 DeepSeek V3.1 的 Decode forward 性能。
```

First response:

```text
可以开始。目前已知：模型别名是 DeepSeek V3.1，场景是 Decode forward。

[INTAKE]
CONFIRMED  model=DeepSeek V3.1; phase=Decode forward
MISSING    device profile; total devices; batch/num-queries
PROPOSED   none

本轮请补充：
1. 目标硬件 profile，例如 ATLAS_800_A3_752T_128G_DIE；
2. 总卡数，例如 64；
3. batch/--num-queries，例如 256。

如果某项不确定，可以回复“请推荐”，我会说明推荐依据。
```

After those values are supplied, ask the next group—for example context length, query length and TP/EP strategy—rather than repeating already confirmed inputs. Once all required values are known, show the full command and wait for confirmation. This example demonstrates dialog shape only; its sample numbers are not defaults.

## 7. Copyable mode invocation examples

The mode name is optional when the intent is already clear, but using it makes the requested boundary explicit.

### Full workflow

```text
请使用 model-adaptation-validation-workflow skill 的 full 模式，
处理 reports/<case_name>。
保留原始输入和人工决定，不复用旧 generated 结论；
依次执行基础模型适配、Shape/dtype 防护和可用的 NPU profiling 对账，
在 Profile、Theory、forward 或映射需要人工决定时暂停。
```

### Base adaptation only

```text
请使用 model-adaptation-validation-workflow skill 的 adaptation-only 模式。
根据我提供的 TensorCast 仿真命令，只执行 doctor、Profile 检查点、
model_adapter verify 和原始仿真；不要读取 profiling CSV/trace，
也不要执行 Shape/dtype 或 NPU 实测对账。
```

### Shape/dtype guardrail only

```text
请使用 model-adaptation-validation-workflow skill 的 semantic-only 模式。
使用 <run-profile.yaml> 和匹配的 Theory Spec，
只执行 Theory↔Runtime Shape/dtype 对比；
不要要求先运行 doctor/verify，也不要要求 kernel CSV 或 trace.json。
```

### NPU profiling comparison only

```text
请使用 model-adaptation-validation-workflow skill 的 profiling-only 模式。
使用 <kernel-details.csv> 和 <trace.json> 做 NPU 与 TensorCast 分层对比；
先核对模型、设备、phase、工作负载、并行和量化是否一致，
不要补跑 doctor、verify 或 Theory。
```

### Calibration review only

```text
请使用 model-adaptation-validation-workflow skill 的 calibration-only 模式，
审查 reports/<case_name>/workflow_summary.md 中最新的非 PASS finding。
先展示证据和需要的人工作决定；收到修复授权后，
转交 model-adaptation-calibration-loop 做最小修复和受影响项复验。
```

For any standalone mode, mark all stages outside that mode as `NOT_REQUESTED`. Do not ask for their inputs merely to make the case directory resemble a full-workflow case.

## 8. Resume behavior

On every resumed turn:

1. restate the current stage and confirmed ledger briefly;
2. consume the user's latest answers;
3. validate contradictions immediately;
4. ask only the next unresolved questions or present the next confirmation;
5. do not restart inventory or repeat completed commands unless the user requests a rerun.

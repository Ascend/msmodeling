---
name: fusion-eval
description: Use when a user wants to estimate whole-model E2E performance of a hypothetical fusion (e.g. "evaluate rms_norm on GLM5") BEFORE the fx pattern + AscendC kernel are implemented. Captures the real compiled graph (Phase 0.5) to provide ground-truth op overloads and structure, generates a fusion plugin .py with scalar_workaround for eps, validates it (Validator L1-L4 + L3-real subprocess fire-count check), and runs ModelRunner to report fused latency/TPS. Verified first-attempt success rate on GLM5/Kimi-K2.6/DSv4/Qwen3.5-MoE.
metadata:
  version: 2.0.0
  source: local-session-analysis
---

# fusion-eval

## Overview

Estimate the whole-model E2E time of a **hypothetical fusion** without writing
any fx pattern or AscendC kernel first. The user names an op sequence (e.g.
`aten.mm, aten.relu`); this skill generates a self-contained fusion plugin
`.py`, validates it, then drives the existing TensorCast flow via the Python API
to print the fused latency / TPS.

This skill is the user-facing layer of the Fusion Plugin Framework
(`docs/RFC/rfc_manual_fusion_eval_en.md`). It contains **no executable code** —
it guides you (the agent) to produce a protocol-compliant plugin. The Loader,
Validator, and Python API are repo-side Python (`tensor_cast/plugins/`,
`tensor_cast/plugin_framework/`).

**Non-invasive contract:** never modify `scripts/text_generate.py`, its
argparse, or any main-repo source. The user entry is this skill / the Python
API only.

## Input Forms (RFC §3.1)

The user triggers this skill with natural language. Accept all three forms and
parse each into a uniform `{ops, dtype, model, device}` structure:

```text
# Form 1: concise command
/fusion-eval mm+relu fp16 Qwen3-32B

# Form 2: natural description
"evaluate the fusion of attention output followed by layernorm, on Qwen3-32B prefill"

# Form 3: explicit parameters
/fusion-eval ops=aten.mm,aten.relu dtype=fp16 model=Qwen3-32B device=ATLAS_800_A3_752T_128G_DIE
```

Parsing rules:

- **Form 1** (positional): `<ops> <dtype> <model>` — split `mm+relu` on `+`
  into the op sequence, normalize bare names to `aten.<name>` when unqualified,
  map the model short name to its HuggingFace id.
- **Form 2** (free text): extract the op sequence from the description
  ("attention output followed by layernorm" → `[attention, layer_norm]`), plus
  dtype / model / decode-vs-prefill if mentioned.
- **Form 3** (key=value): parse `ops=` / `dtype=` / `model=` / `device=`
  directly.
- After parsing, confirm the resolved `{ops, dtype, model, device}` back to the
  user when anything was inferred, and **ask to clarify when the op sequence is
  ambiguous** — never guess it.

## Required Inputs

Collect these from the user; ask to clarify when ambiguous (do not guess the
op sequence):

- [ ] **Op sequence** — the ops to fuse, in dataflow order (e.g.
      `aten.mm, aten.relu`). The first op is the subgraph head.
- [ ] **Model** — HuggingFace id (e.g. `Qwen/Qwen3-32B`)
- [ ] **Device** — a registered `DeviceProfile` full name (e.g.
      `ATLAS_800_A3_752T_128G_DIE`; short aliases are rejected)
- [ ] **dtype** — fp16 / bf16 / etc. (affects compute_ops accounting)
- [ ] **Shape / scenario** (optional) — num_queries, query_len, decode vs
      prefill; defaults come from `UserInputConfig`.

## Scope (v1)

- ✅ Single-input single-output chained sequences (mm+epilogue, swiglu,
  rms_norm-style).
- ❌ Residual/bypass merges, multi-output head ops, in-place ops, non-tensor
  intermediates. For these, give an "unsupported in v1" conclusion rather than
  a wrong estimate.

## Workflow

### Phase 0: CACHE CHECK

Look for an existing plugin matching the op sequence in the plugin directory
(default `./plugins/`). If found, skip generation (Phase 1-2) and go straight
to Phase 3 — reuse saves the minute-scale generation cost.

### Phase 0.5: GRAPH CAPTURE (before Phase 1)

Run `extract_subgraph` on the pre-rewrite compiled graph to get the exact
op sequence, overload names, fan-out structure, and boundary inputs. This
eliminates F3/F4/F5 failure modes (wrong op overloads, wrong boundary structure)
before the LLM writes a single line.

```python
import tensor_cast.compilation.compile_backend as cb
from tensor_cast.plugins.graph_extractor import extract_subgraph
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig

captured = {}
orig = cb.CompilerBackend.apply_pattern_match_passes

def spy(self, gm, inp):
    if "info" not in captured:
        captured["info"] = extract_subgraph(gm, seed_op)
    return orig(self, gm, inp)

cb.CompilerBackend.apply_pattern_match_passes = spy
try:
    ui = UserInputConfig(model_id=model_id, device=device, do_compile=True,
                         num_hidden_layers_override=1)
    ModelRunner(ui).run_inference(generate_inputs_func=generate_inputs)
finally:
    cb.CompilerBackend.apply_pattern_match_passes = orig

info = captured.get("info")
```

- If `info is None`: the seed op does not exist in this model → report
  **"unsupported: seed op not found in compiled graph"** and stop. Do NOT
  proceed to Phase 1 with a guessed pattern.
- If `info` is returned: pass `info.to_prompt_str()` to Phase 1 as
  **ground truth**. The LLM must transcribe it, not invent it.

`seed_op` = the first op in the user's requested fusion sequence, full overload
spelling. Use this mapping for common fusions:

| Fusion | seed_op |
|--------|---------|
| swiglu / silu_mul | `"aten.sigmoid.default"` or `"aten.silu.default"` |
| rms_norm | `"aten.rsqrt.default"` (do NOT use `"aten.add.Tensor"` — the residual add appears earlier in the graph and produces an unrelated SubgraphInfo) |
| custom epilogue | the first non-matmul op in the chain |

### Phase 1: GENERATE

Produce `<plugin_dir>/<name>.py` following `generate-prompt.md`.

**If Phase 0.5 produced a `SubgraphInfo`**: pass `info.to_prompt_str()` as
`captured_graph_section` to generate-prompt.md — the LLM transcribes the
captured section into `_pattern()` verbatim, no op spellings to invent.

**If Phase 0.5 was skipped** (cache hit in Phase 0): generate from template
using `ref/pattern-examples.md` as reference as before.

### Phase 2: VALIDATE (loop-back)

Run the Validator following `validate-prompt.md`. On failure, read the specific
layer signal (L1-L4), fix the plugin targetedly, and re-validate. Cap at 3
iterations; if still failing, stop and hand the draft + cause to the user.
**Never report metrics for a plugin that has not passed validation.**

### Phase 2b: L3-REAL CHECK

After Validator passes (layer == "OK"), run the L3-real fire-count check in a
subprocess to confirm the pattern actually fires in the real compiled graph:

```python
from tensor_cast.plugins.l3_real import check_fire_count, L3RealError

try:
    result = check_fire_count(
        plugin_path="<plugin_dir>/<name>.py",
        model_id=model_id,
        seed_op=seed_op,
        device=device,
    )
except L3RealError as e:
    # subprocess crashed — treat as unknown, proceed to evaluate but warn user
    print(f"L3-real check failed: {e}")
    result = None
```

- `result.ok` (fire_count > 0): proceed to Phase 3.
- `result.candidate_count == 0`: the seed op is not in this model's compiled
  graph — report **"unsupported: seed op absent in real graph"** and stop.
- `result.fire_count == 0` and `candidate_count > 0`: the pattern registered
  but never matched. Go back to Phase 1 with `result.diagnostic_section` as
  the corrected ground truth — same loop-back protocol as validate-prompt.md.
  Cap combined Phase 1+2+2b iterations at 3.

### Phase 3: EVALUATE

Drive the Python API (it takes over the CLI's parse_args job; do_compile=True
is forced inside):

```python
from tensor_cast.plugin_framework import evaluate_fusion_plugin

metrics = evaluate_fusion_plugin(
    plugin_path="./plugins/<name>.py",
    model_id="<model>",
    device="<DEVICE_FULL_NAME>",
    num_queries=<n>, query_len=<n>,   # optional scenario fields
)
```

`evaluate_fusion_plugin` validates again (idempotent), loads the plugin, forces
`do_compile=True`, and runs one inference. `metrics.print_info()` is called for
you.

### Phase 4: REPORT

Report to the user:

- fused latency / TPS from `metrics` (`execution_time_s` / `tps_per_model` are
  per-model dicts keyed by model name, NOT scalars — iterate them)
- whether the virtual op node actually appears in the rewritten fx graph
- baseline comparison: run the same config with `plugin_path=None` **in a
  separate process** (plugin registration is process-level and irreversible),
  then compare. See `validate-prompt.md` for the isolation rationale.

## Boundaries

| Action | Allowed |
|--------|---------|
| Write `.py` files in the plugin directory | Yes |
| Call the Validator (`tensor_cast.plugins.validator`) | Yes |
| Run `evaluate_fusion_plugin()` (Python API) | Yes |
| Modify `text_generate` CLI / argparse | No |
| Modify any main-repo source | No |
| Import repo-private (underscore) symbols in the plugin | No |
| Report metrics without passing the Validator | No |

## Completion Criteria

- [ ] Plugin passes the Validator (L1-L4, `layer == "OK"`)
- [ ] `evaluate_fusion_plugin()` returns metrics
- [ ] Fused latency / TPS reported, with baseline comparison if requested
- [ ] Plugin `.py` left in the plugin directory for reuse

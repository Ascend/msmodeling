# validate-prompt — Validate & Loop-Back Fix

The Validator is the **sole quality gatekeeper** (RFC §3.4 / §9.1). A plugin
that does not pass MUST NOT be run through ModelRunner — never report metrics
for an unvalidated plugin.

## Run the Validator

```python
from tensor_cast.plugins.validator import validate_plugin

result = validate_plugin("<plugin_dir>/<name>.py")
print(result.layer, result.detail)   # layer == "OK" means all of L1-L4 passed
```

`result` is truthy only when `layer == "OK"`. Otherwise `result.layer` is the
first failing layer and `result.detail` is the specific signal.

> Note: `validate_plugin` has a side effect — its L2 imports and registers the
> plugin into the in-process global tables. Re-validation in the SAME process
> hits the loader's idempotency guard (returns "already loaded"). For a truly
> clean re-run after editing, use a fresh process/subprocess.

## The four layers and how to fix each

| Layer | Means | Typical fix |
|-------|-------|-------------|
| **L1** static | bad syntax / missing `register_all_patterns` / private import / op name lacks namespace prefix | fix the named issue; ensure declared op starts with `<namespace>_` |
| **L2** register | a `register_*` call raised (op-name clash / `already registered`) | change the namespace prefix or op name to something unique |
| **L3** hit | `matched_cnt = 0` or hit ratio below threshold (0.9) | the `_pattern()` body does not match the real graph — fix op overload spelling / in-place variant / operand order; register missing variants |
| **L4** estimate | props functor missing or returns invalid (negative bytes / incomplete dtype) | register `@register_op_properties` for the declared op; fill non-negative `compute_ops` |

## Loop-back protocol

1. Read `result.layer` + `result.detail` — it names the specific failure.
2. Edit the plugin targetedly (usually just `_pattern()` for L3, or
   `compute_ops` for L4).
3. Re-validate (fresh process if L2 already registered the op this run).
4. **Cap at 3 iterations.** If still failing, STOP: report the cause + the
   current draft to the user, and either hand off to manual edit (escape hatch,
   same as editing a built-in `patterns/<name>.py`) or give an
   "unsupported in v1" conclusion. Never silently pass.

## After OK: L3-real check, then evaluate

Only once `result.layer == "OK"`, run Phase 2b (L3-real) before evaluating:

```python
from tensor_cast.plugins.l3_real import check_fire_count, L3RealError

result = check_fire_count(
    plugin_path="<plugin_dir>/<name>.py",
    model_id=model_id,
    seed_op=seed_op,   # e.g. "aten.sigmoid.default"
    device=device,
)
```

| Outcome | Action |
|---------|--------|
| `result.ok` (fire_count > 0) | Proceed to Phase 3 |
| `result.candidate_count == 0` | Report unsupported: seed op absent in model |
| `result.fire_count == 0`, `candidate_count > 0` | Loop back: fix `_pattern()` using `result.diagnostic_section` as corrected ground truth, re-validate |

**Loop-back with L3-real diagnostic:**

`result.diagnostic_section` is the same format as `SubgraphInfo.to_prompt_str()` —
the real graph section around the seed op. Use it exactly as you would a Phase 0.5
`captured_graph_section`: transcribe its op sequence into a corrected `_pattern()`.

Combined Phase 1+2+2b iteration cap: **3 total**. If still failing after 3 rounds,
stop and report: the current draft plugin + the diagnostic_section + cause.



## Baseline comparison (separate processes)

Plugin registration is process-level and irreversible (RFC §4.4), and
`patterns.lazy_init()` is `@lru_cache`-cached so in-process disabling of
built-ins is unreliable. To compare with vs. without the plugin, run **two
separate processes**:

```python
# Process A (baseline): plugin_path=None  -> no plugin loaded
# Process B (fused):     plugin_path="./plugins/<name>.py"
# Compare metrics.execution_time_s / metrics.tps_per_model (per-model dicts).
```

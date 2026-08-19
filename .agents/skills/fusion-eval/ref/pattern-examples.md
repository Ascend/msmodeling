# Pattern Examples — Built-in References

When generating a plugin, mimic the repo's **validated** built-in patterns.
They are the ground truth for "what a correct fx pattern + replacement +
props functor looks like". Read the source before writing — do not invent
shapes or op spellings.

## The four reverse-calibration anchors (all `register_pattern` path)

These are the anchors the Validator calibrates against (RFC §3.5). All reach
`patterns.all_passes` via `register_pattern`, so a plugin can faithfully
reproduce them:

| Anchor | Source | Character | Note |
|--------|--------|-----------|------|
| swiglu | `tensor_cast/compilation/patterns/swiglu.py` | activation, memory-bound | two variants (mul order); fp32 up/down-casts in the pattern |
| rms_norm | `tensor_cast/compilation/patterns/rms_norm.py` | normalization | residual variants |
| rms_norm_quant | `tensor_cast/compilation/patterns/rms_norm.py` | norm + quant | `register_pattern` path (the 4th anchor) |
| rotary_embedding | `tensor_cast/compilation/patterns/rotary_embedding.py` | position embed | compute-bound |

> **Excluded:** `grouped_matmul_swiglu` is a *freezing pass*
> (`GroupedMatmulSwigluPass`), NOT a `register_pattern` path. The Plugin
> protocol cannot reproduce it, so it is never an anchor (RFC §3.5).

## What to copy from swiglu

`swiglu.py` shows the canonical skeleton:

- **`get_inputs()`** — minimal meta tensors (shape `(1, 1)`) so the pattern can
  be traced without real data. Replicate this exactly.
- **`pattern(...)`** — the low-level op chain. Note swiglu spells ops in full:
  `torch.ops.aten.sigmoid.default`, `torch.ops.aten.mul.Tensor`, and uses
  `prims.convert_element_type` for dtype casts. **Match the traced graph's op
  spelling precisely** — `.Tensor` vs `.default` overloads and in-place
  variants (`aten.relu_`) are the #1 cause of `matched_cnt = 0`.
- **multiple variants** — swiglu registers both `mul_up_first` and
  `mul_silu_first` because the real graph can emit either operand order. If
  your fusion has a commutative tail, register both variants too, or the hit
  ratio (RFC §3.5 L3) drops below threshold.
- **`replacement(...)`** — one line: `torch.ops.tensor_cast.<op>(...)`, same
  boundary args as the pattern.

## Boundary memory = why fusion saves time

The virtual op's schema declares only **boundary inputs/outputs**. Intermediate
tensors are *not* in the schema, so `info.get_memory_access_properties()` does
not count their HBM read/write — modeling "intermediates stay in on-chip SRAM".
Do not manually add intermediate-tensor bytes; let the schema bucketing do it.
Only fill `compute_ops` (accumulated from every low-level op).

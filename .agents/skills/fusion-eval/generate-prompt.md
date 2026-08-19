# generate-prompt — Produce a Fusion Plugin

Goal: turn a parsed `{ops, dtype, model, device}` request into a
protocol-compliant plugin `.py`. Keep degrees of freedom low — the structure is
fixed; only the `_pattern()` body is genuinely free-form.

## Inputs

- `ops`: ordered op sequence (e.g. `[aten.mm, aten.relu]`), first op = head
- `dtype`: fp16 / bf16 / ...
- `name`: short fusion name (e.g. `mm_relu`); op name becomes
  `<namespace>_<name>` with namespace defaulting to `user_fusion`
- `plugin_dir`: output dir (default `./plugins/`)
- `captured_graph_section` *(optional)*: output of `SubgraphInfo.to_prompt_str()`
  from Phase 0.5 — the real compiled graph structure around the seed op.

## Steps

1. **Start from the template.** Copy `ref/plugin-template.py` to
   `<plugin_dir>/<name>.py`. Every `<name>` placeholder must be replaced
   consistently (op declaration, replacement, register_pattern, meta `default`).

2. **Set the namespace prefix.** Keep `__plugin_namespace__ = "user_fusion"`
   unless the user gave a team prefix. The declared op name
   (`register_tensor_cast_op("...")`) MUST start with `<namespace>_` or L1
   fails (RFC §3.5 L1).

3. **Write `_pattern()` — guided by captured_graph_section when available.**

   **If `captured_graph_section` was provided (Phase 0.5 ran):**
   Transcribe it line-by-line into Python. Do NOT invent op spellings or
   add/remove nodes — copy the exact overload names from the section.
   The section format is:
   ```
   # x_1  [BOUNDARY — external tensor input]
   # w_1  [BOUNDARY — external tensor input]
   #
   t0 = prims.convert_element_type.default(x_1)  # _users=2
   t1 = aten.pow.Tensor_Scalar(t0)
   ...
   t7 = aten.mul.Tensor(w_1, t6)  ← OUTPUT
   ```
   Translation rules:
   - Each `BOUNDARY` name becomes a function parameter of `_pattern()`.
   - Each `var = op(args)` line becomes `var = torch.ops.<op>(<args>)`.
   - Nodes annotated `# _users=N` with N≥2 are fan-out variables — reuse the
     same Python variable name in all downstream calls (do NOT duplicate it).
   - The node marked `← OUTPUT` is the return value.
   - Scalar args (e.g. `2` in `pow.Tensor_Scalar`) and list args (e.g.
     `[-1]` in `mean.dim`) are baked as literals; they are ignored by inductor
     during matching (no need for `scalar_workaround` for dims/exponents).
   - Numeric scalars that appear as operands of arithmetic ops (e.g. `1e-5`
     in `add.Tensor`) may use `scalar_workaround` for clarity, but it is
     optional — inductor treats unlisted floats as wildcards anyway.

   **If `captured_graph_section` was NOT provided:**
   Read the actual traced fx graph spelling (mimic `ref/pattern-examples.md`).
   Rules:
   - Use full overload spelling (`torch.ops.aten.mul.Tensor`, not `aten.mul`).
   - Watch in-place variants: `aten.relu_` vs `aten.relu`. If unsure, register
     both or check the traced graph.
   - Insert `prims.convert_element_type` casts only if the real graph has them
     (swiglu does, for its fp32 activation core).
   - If the tail op is commutative (operand order can flip), register both
     variants like swiglu's `mul_up_first` / `mul_silu_first`.

4. **Fill `_meta_impl()` output shape.** Return an empty meta tensor with the
   fused op's true output shape/dtype. Only boundary I/O — intermediates are
   excluded by design (that is what makes the estimate reflect fusion).

5. **Fill `compute_ops`.** Accumulate EVERY low-level op's compute:
   - matmul: `mma_ops = m * n * k * 2`
   - elementwise (relu/sigmoid/mul): `gp_ops += elem_count`
   Missing a term under-counts compute and biases the estimate. Do NOT add
   intermediate-tensor memory bytes — the schema bucketing handles memory.

6. **Fill `__plugin_meta__`.** Record `ops`, `dtype_support`, `notes`,
   `plugin_schema_version: "2.0"`. Set `expected_match_count` only if you know
   the head op appears a known number of times and want to override the L3
   default.

7. **Build `example_inputs` as independent tensors (CRITICAL).** In
   `register_all_patterns`, create each example input with its own
   `torch.empty(...)` call. **Never** write `[torch.empty(...)] * N`:

   ```python
   # WRONG — N references to ONE object
   example_inputs = [torch.empty(1, 1, dtype=torch.float16, device="meta")] * 3
   # RIGHT — N independent objects
   example_inputs = [
       torch.empty(1, 1, dtype=torch.float16, device="meta"),
       torch.empty(1, 1, dtype=torch.float16, device="meta"),
       torch.empty(1, 1, dtype=torch.float16, device="meta"),
   ]
   ```

   Why: inductor traces the pattern with these inputs and **dedups args by
   object identity**. A shared object collapses distinct params (`a, b, c`)
   into one var — the registered pattern degenerates (e.g. `mul(c, c) + c`) and
   never matches the real graph, so the plugin loads and validates L1/L2 but
   silently fires 0 times (L3 hit ratio = 0). Distinct shapes per param are
   fine; the rule is one `torch.empty` call per slot.

   If `captured_graph_section` was provided, use the boundary metadata shapes:
   ```python
   # boundary_meta from to_prompt_str: "x_1  f16[512, 4096]"
   torch.empty(512, 4096, dtype=torch.float16, device="meta"),  # x_1
   ```

## Do not

- Do not import repo-private (underscore-prefixed) symbols from `tensor_cast`
  (L1 rejects this).
- Do not touch `text_generate` / argparse / any main-repo source.
- Do not hand-roll a new roofline formula — reuse
  `info.get_memory_access_properties()` + `ComputeOps`.

Output: the path to the written `.py`. Then go to `validate-prompt.md`.


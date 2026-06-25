---
name: throughput-optimizer-explainer
description: Analyze, compare, and explain results from `python -m cli.inference.throughput_optimizer`. Use when the user asks whether throughput, TTFT, TPOT, PD ratio, Cube/Vec/Comm/Mem breakdowns, text_generate op-bound output, or best parallel strategies are reasonable; wants to compare hardware results; needs bottleneck attribution across Prefill, Decode, compute, memory, communication, TP, DP, PP, EP, MOE-TP, or MOE-DP; or asks follow-up questions after running throughput_optimizer. Also use when mapping throughput_optimizer best rows to `python -m cli.inference.text_generate` validation commands.
metadata:
  version: 0.1.0
  source: local-session-analysis
---

# Throughput Optimizer Explainer

Use this skill to explain optimizer results without overstating what the evidence supports.

## Evidence Rule

Classify the evidence before explaining:

- `macro_only`: ordinary `throughput_optimizer` summary or extractor JSON with no phase breakdown.
- `optimizer_phase_breakdown`: `--dump-original-results` output containing `percentage_breakdowns(p)`, `percentage_breakdowns(d)`, or `percentage_breakdowns`.
- `text_generate_phase_breakdown`: `text_generate` output containing Stats breakdowns.
- `text_generate_op_bound`: `text_generate --dump-op-bound-results` output containing an operator average table with Bound and memory/comm/mma/gp percentage columns.
- `profiler_trace`: real profiler or chrome trace data.

Do not describe operator-level facts unless the input includes text_generate op-bound output or profiler data. Treat text_generate op-bound output as simulated operator attribution, not real kernel/profiler evidence. If only macro output is available, explain at deployment/phase/strategy level and call the bottleneck attribution an inference.

## Input Handling

Accept any of these inputs:

- raw `throughput_optimizer` command
- raw `throughput_optimizer` output
- JSON produced by an extractor script
- `--dump-original-results` table output
- `text_generate` output
- `text_generate --dump-op-bound-results` output
- hardware specs or hardware names
- profiler traces or summarized profiler reports

The extractor JSON is convenient but not required. If it lacks fields needed for explanation, ask for the raw command, raw output, `--dump-original-results`, or `text_generate` output.

## Workflow

1. Parse the optimizer mode: aggregation, disaggregation, or PD ratio.
2. Extract comparable conditions: model, device, num devices, input length, output length, SLO limits, quantization, compile, prefix cache, MTP, and search space when available.
3. Extract best rows and top candidates: throughput, TTFT, TPOT, concurrency, batch size, parallel strategy, PD ratio, QPS, and breakdowns.
4. Classify evidence level.
5. For aggregation, treat the result as Prefill forward + Decode forward + scheduling formula. Never map aggregation to a single forward.
6. For disaggregation, map each result to the corresponding Prefill or Decode phase.
7. If phase breakdowns are missing and the user needs Cube/Vec/Comm/Mem analysis, generate `text_generate` validation commands using the mapping rules. Add `--dump-op-bound-results` when bottleneck attribution needs simulated per-operator evidence.
8. If text_generate op-bound output is available, inspect the top total-time operators, their dominant bounds, and memory/comm/mma/gp percentages before making operator-level simulated-attribution claims.
9. Compare hardware or strategies using phase breakdowns first, op-bound evidence for operator attribution, macro metrics second, and hardware ratios only as supporting context.
10. State a reasonableness level:
   - `basically reasonable`
   - `partly explainable`
   - `suspicious`
   - `insufficient evidence`
11. End with the smallest useful validation action.

## References

Load only the reference needed for the current question:

- `references/aggregation-mapping.md`: mapping aggregation best rows to Prefill and Decode validation.
- `references/disaggregation-mapping.md`: mapping disaggregation results to `text_generate`.
- `references/evidence-levels.md`: how to phrase certainty and missing evidence.
- `references/bottleneck-rules.md`: Cube/Vec/Comm/Mem and parallel strategy interpretation rules.
- `references/output-template.md`: concise answer template.

## Scripts

Use scripts when structured extraction or command generation helps:

- `scripts/parse_optimizer_output.py`: parse raw optimizer output, dump tables, text_generate breakdowns, and text_generate op-bound tables into JSON.
- `scripts/build_text_generate_commands.py`: generate Prefill/Decode validation commands from a normalized best row JSON; pass `--include-op-bound` to append `--dump-op-bound-results`.
- `scripts/compare_phase_breakdowns.py`: compare Cube/Vec/Comm/Mem percentages across two phase results, or pass `--op-bound` to compare text_generate op-bound tables.

## Aggregation Mapping Summary

Aggregation best throughput is not a single forward. It is built from:

- Prefill forward latency and breakdown.
- Decode forward latency and breakdown.
- Scheduling formulas for TTFT, TPOT, concurrency, and output throughput.

For aggregation best row values:

```text
I = input_length
O = output_length
B = batch_size
C = concurrency
M = max_prefill_tokens
H = prefix_cache_hit_rate
N = num_mtp_tokens
```

Compute:

```text
I_eff = max(1, I - floor(I * H))
prefill_batch_size = M // I_eff
decode_query_length = N + 1
decode_context_length = I + O // 2
decode_num_queries = C
```

Generate two validation commands: one Prefill and one Decode. Do not compare a single `text_generate` TPS directly with aggregation throughput.
For simulated operator attribution, generate the same commands with `--dump-op-bound-results` or use `scripts/build_text_generate_commands.py --include-op-bound`.

## Output Requirements

Always make clear:

- whether the conclusion is based on raw macro output, optimizer phase breakdowns, text_generate, or profiler data
- whether aggregation was decomposed into Prefill and Decode
- when using op-bound output, that it is TensorCast simulated operator attribution rather than real profiler/kernel evidence
- what field or command is missing if the conclusion is uncertain
- that optimizer and text_generate are simulations unless the user supplies real profiler/runtime measurements

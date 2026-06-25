# Evidence Levels

Use these levels to keep conclusions honest.

## macro_only

Available fields usually include throughput, TTFT, TPOT, concurrency, batch size, and parallel strategy.

Allowed conclusions:

- strategy-level explanations
- SLO-limited versus capacity-limited inferences
- Prefill-heavy or Decode-heavy hypotheses from TTFT/TPOT shape
- requests for `--dump-original-results` or `text_generate`

Avoid:

- claiming a specific Cube/Vec/Comm/Mem bottleneck as fact
- claiming a specific operator is slow

## optimizer_phase_breakdown

Available fields include `percentage_breakdowns(p)`, `percentage_breakdowns(d)`, or `percentage_breakdowns`.

Allowed conclusions:

- phase-level Cube/Vec/Comm/Mem comparison
- hardware difference explanation tied to phase metrics
- parallel strategy interpretation with communication/memory hints

Avoid:

- kernel-level claims
- real runtime claims

## text_generate_phase_breakdown

Available data comes from `text_generate` Stats breakdowns.

Allowed conclusions:

- validation of optimizer phase assumptions
- phase-level breakdown comparison under the best-row configuration

Avoid:

- treating text_generate TPS as equal to aggregation throughput
- claiming real kernel behavior

## text_generate_op_bound

Available data comes from `text_generate --dump-op-bound-results` operator average tables.

Allowed conclusions:

- simulated operator-level attribution for the modeled phase
- top total-time operator comparison
- dominant-bound comparison using `memory_bound`, `communication_bound`, `compute_bound_mma`, and `compute_bound_gp`
- mapping op-bound ratios back to phase-level Mem/Comm/Cube/Vec hypotheses

Avoid:

- treating simulated op-bound output as profiler or kernel trace evidence
- mixing Prefill and Decode op-bound tables without labeling the phase
- treating text_generate TPS as equal to aggregation throughput

## profiler_trace

Available data is a real profiler, chrome trace, or summarized operator/kernel trace.

Allowed conclusions:

- operator-level and kernel-level attribution if the trace supports it
- validation or rejection of optimizer assumptions

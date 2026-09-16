"""Shared assertions for model-diagnostics tests."""

from __future__ import annotations

import math

from tools.model_diagnostics.domain import ParallelContext


def assert_parallel_contract(env: dict[str, object], parallel: ParallelContext, *, raw_logits_gate: bool) -> None:
    """Validate rank-local heads and MoE token domains for a parallel layout."""

    assert env["Lh"] == env["Nh"] // parallel.tensor_parallel_size
    if parallel.expert_parallel_size > 1:
        assert env["Tmoe"] == math.ceil(env["T"] / parallel.tensor_parallel_size)
    else:
        assert env["Tmoe"] == env["T"] * parallel.data_parallel_size
    if raw_logits_gate and parallel.expert_parallel_size > 1:
        assert env["MOE_GATE_TOKENS"] == env["T"]
    else:
        assert env["MOE_GATE_TOKENS"] == env["Tmoe"]

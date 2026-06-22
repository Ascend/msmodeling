# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StepDecision:
    """Prefill/decode concurrency selected for one simulated mixed step."""

    p_step: int
    d_step: int


@dataclass(frozen=True)
class SchedulerState:
    """Queue and token-budget snapshot used by a scheduler decision."""

    # Requests that have completed prefill and can produce one decode token in this step.
    ready_decode: int
    # Requests waiting for the current prefill chunk shape.
    pending_prefill: int
    # New tokens computed by one prefill request in the current chunk.
    chunk_query_len: int
    # User-facing token budget for one prefill or mixed prefill/decode step.
    max_batched_tokens: int


class Scheduler(ABC):
    """Policy interface for selecting prefill/decode work in each simulation step."""

    @abstractmethod
    def decide(self, state: SchedulerState) -> StepDecision:
        """Return how many prefill and decode requests should run next.

        Implementations must base the decision only on SchedulerState. Latency is unavailable
        during dry-run schedule construction and is applied later during replay via step_latency().
        """
        ...

    @abstractmethod
    def step_latency(self, prefill_latency: float, decode_latency: float) -> float:
        """Combine the modeled latency of prefill and decode work in one step."""
        ...


class DecodeFirstWithSlack(Scheduler):
    """Default scheduler: prioritize decode, then admit prefill with a small slack budget."""

    slack_ratio = 1.15

    def decide(self, state: SchedulerState) -> StepDecision:
        """Prefer ready decode work, then admit prefill within the slack-adjusted budget."""
        # Slack only affects mixed-step admission; chunk sizes still respect max_batched_tokens.
        limit = math.floor(state.max_batched_tokens * self.slack_ratio)
        # Decode is prioritized because ready requests already exposed their first token to users.
        d_step = min(state.ready_decode, state.max_batched_tokens)
        if state.chunk_query_len <= 0:
            p_step = 0
        else:
            # Each prefill request consumes chunk_query_len budget; each decode request consumes one.
            p_step = max(
                0,
                min(
                    state.pending_prefill,
                    math.floor((limit - d_step) / state.chunk_query_len),
                ),
            )
        return StepDecision(p_step=p_step, d_step=d_step)

    def step_latency(self, prefill_latency: float, decode_latency: float) -> float:
        """Model mixed prefill/decode work as overlapped in one scheduling step."""
        # Prefill and decode are modeled as overlapped within a mixed step.
        return max(prefill_latency, decode_latency)

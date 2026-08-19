# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, isfinite
from typing import Callable, Dict, Iterable, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

from ..config.config import BenchmarkEarlyExitConfig, PerformanceIndex


METRICS_SAMPLE_INTERVAL_SECONDS = 5
MIN_WARMUP_SECONDS = 15
MAX_WARMUP_SECONDS = 90
LOAD_READY_RATIO = 0.8
LOAD_READY_CONSECUTIVE_SAMPLES = 2
MIN_VALID_WARMUP_SAMPLES = 3


class EarlyExitPhase(str, Enum):
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"


@dataclass
class VllmMetricsSnapshot:
    timestamp: float
    output_tokens: Optional[float] = None
    completed_requests: Optional[float] = None
    failed_requests: Optional[float] = None
    ttft_sum: Optional[float] = None
    ttft_count: Optional[float] = None
    tpot_sum: Optional[float] = None
    tpot_count: Optional[float] = None
    running_requests: Optional[float] = None
    waiting_requests: Optional[float] = None

    @staticmethod
    def _counter_delta(current: Optional[float], previous: Optional[float]) -> Optional[float]:
        if current is None or previous is None:
            return None
        delta = current - previous
        if delta < 0:
            return None
        return delta

    def output_tokens_since(self, previous: "VllmMetricsSnapshot") -> Optional[float]:
        return self._counter_delta(self.output_tokens, previous.output_tokens)

    def completed_requests_since(self, previous: "VllmMetricsSnapshot") -> Optional[float]:
        return self._counter_delta(self.completed_requests, previous.completed_requests)

    def failed_requests_since(self, previous: "VllmMetricsSnapshot") -> Optional[float]:
        return self._counter_delta(self.failed_requests, previous.failed_requests)

    def output_tokens_per_second_since(self, previous: "VllmMetricsSnapshot") -> Optional[float]:
        delta = self.output_tokens_since(previous)
        elapsed = self.timestamp - previous.timestamp
        if delta is None or elapsed <= 0:
            return None
        return delta / elapsed

    @staticmethod
    def _average_delta(
        current_sum: Optional[float],
        previous_sum: Optional[float],
        current_count: Optional[float],
        previous_count: Optional[float],
    ) -> Optional[float]:
        if current_sum is None or previous_sum is None or current_count is None or previous_count is None:
            return None
        sum_delta = current_sum - previous_sum
        count_delta = current_count - previous_count
        if sum_delta < 0 or count_delta <= 0:
            return None
        return sum_delta / count_delta

    def performance_since(self, previous: "VllmMetricsSnapshot") -> PerformanceIndex:
        completed_delta = self.completed_requests_since(previous)
        failed_delta = self.failed_requests_since(previous)
        success_rate = None
        if completed_delta is not None:
            failed_delta = failed_delta or 0
            total = completed_delta + failed_delta
            if total > 0:
                success_rate = completed_delta / total
        return PerformanceIndex(
            generate_speed=self.output_tokens_per_second_since(previous),
            time_to_first_token=self._average_delta(
                self.ttft_sum, previous.ttft_sum, self.ttft_count, previous.ttft_count
            ),
            time_per_output_token=self._average_delta(
                self.tpot_sum, previous.tpot_sum, self.tpot_count, previous.tpot_count
            ),
            success_rate=success_rate,
        )


@dataclass
class EarlyExitDecision:
    would_early_exit: bool
    early_exit: bool
    reason: str
    performance: PerformanceIndex
    observed_score: Optional[float] = None
    reference_score: Optional[float] = None
    observed_generate_speed: Optional[float] = None
    reference_generate_speed: Optional[float] = None
    slo_violations: Dict[str, bool] = field(default_factory=dict)


@dataclass
class WarmupResult:
    end_timestamp: float
    elapsed_seconds: float
    reason: str
    sample_count: int
    effective_target: Optional[int]
    load_threshold: Optional[int]
    running_requests: Optional[float]
    waiting_requests: Optional[float]
    load_ratio: Optional[float]
    forced: bool


@dataclass
class MetricsSampleRecord:
    timestamp: float
    pass_elapsed_seconds: float
    sample_index: int
    phase: str
    scrape_success: bool = True
    scrape_error: Optional[str] = None
    output_tokens_total: Optional[float] = None
    completed_requests_total: Optional[float] = None
    failed_requests_total: Optional[float] = None
    running_requests: Optional[float] = None
    waiting_requests: Optional[float] = None
    sample_window_seconds: Optional[float] = None
    output_tokens_delta: Optional[float] = None
    completed_requests_delta: Optional[float] = None
    failed_requests_delta: Optional[float] = None
    generate_speed: Optional[float] = None
    time_to_first_token: Optional[float] = None
    time_per_output_token: Optional[float] = None
    success_rate: Optional[float] = None
    effective_target: Optional[int] = None
    load_threshold: Optional[int] = None
    load_ratio: Optional[float] = None
    load_ready: bool = False
    load_ready_consecutive_count: int = 0
    warmup_state: str = "not_applicable"
    warmup_end_event: bool = False
    warmup_end_reason: Optional[str] = None
    formal_evaluation_window: bool = False
    eligible_for_early_exit: bool = False
    evaluation_generate_speed: Optional[float] = None
    evaluation_time_to_first_token: Optional[float] = None
    evaluation_time_per_output_token: Optional[float] = None
    evaluation_success_rate: Optional[float] = None
    reference_generate_speed: Optional[float] = None
    reference_score: Optional[float] = None
    observed_score: Optional[float] = None
    bad_window: bool = False
    consecutive_bad_windows: int = 0
    would_early_exit: bool = False
    early_exit: bool = False
    decision_reason: Optional[str] = None


class EarlyExitTriggered(Exception):
    def __init__(self, decision: EarlyExitDecision):
        super().__init__(decision.reason)
        self.decision = decision


class MetricsUnavailableError(RuntimeError):
    """Raised when benchmark metrics cannot be fetched due to an IO failure."""


class VllmMetricsClient:
    SUPPORTED_SCHEMES = frozenset({"http", "https"})
    OUTPUT_TOKEN_ALIASES = (
        "output",
        "output_total",
        "vllm_output_tokens_total",
        "vllm_generation_tokens_total",
        "vllm:generation_tokens_total",
        "vllm_request_generation_tokens_total",
        "vllm:request_generation_tokens_total",
    )
    TTFT_SUM_ALIASES = (
        "fine_grained_ttft_sum",
        "fine_grained_ttft_seconds_sum",
        "vllm_time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_sum",
    )
    TTFT_COUNT_ALIASES = (
        "fine_grained_ttft_count",
        "fine_grained_ttft_seconds_count",
        "vllm_time_to_first_token_seconds_count",
        "vllm:time_to_first_token_seconds_count",
    )
    TPOT_SUM_ALIASES = (
        "fine_grained_tpot_sum",
        "fine_grained_tpot_seconds_sum",
        "vllm_time_per_output_token_seconds_sum",
        "vllm:time_per_output_token_seconds_sum",
        "vllm_request_time_per_output_token_seconds_sum",
        "vllm:request_time_per_output_token_seconds_sum",
    )
    TPOT_COUNT_ALIASES = (
        "fine_grained_tpot_count",
        "fine_grained_tpot_seconds_count",
        "vllm_time_per_output_token_seconds_count",
        "vllm:time_per_output_token_seconds_count",
        "vllm_request_time_per_output_token_seconds_count",
        "vllm:request_time_per_output_token_seconds_count",
    )
    COMPLETED_REQUEST_ALIASES = (
        "completed_requests",
        "completed_requests_total",
        "vllm_request_success_total",
        "vllm:request_success_total",
    )
    FAILED_REQUEST_ALIASES = (
        "failed_requests",
        "failed_requests_total",
        "vllm_request_failure_total",
        "vllm:request_failure_total",
    )
    RUNNING_REQUEST_ALIASES = (
        "num_requests_running",
        "vllm_num_requests_running",
        "vllm:num_requests_running",
    )
    WAITING_REQUEST_ALIASES = (
        "num_requests_waiting",
        "vllm_num_requests_waiting",
        "vllm:num_requests_waiting",
    )

    _SAMPLE_PATTERN = re.compile(
        r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
    )

    def __init__(
        self,
        metrics_url: str = "",
        fetch_metrics: Optional[Callable[[], str]] = None,
        time_func: Callable[[], float] = time.time,
        timeout_seconds: float = 1.0,
    ):
        self.metrics_url = metrics_url
        self.fetch_metrics = fetch_metrics
        self.time_func = time_func
        self.timeout_seconds = timeout_seconds

    def _fetch_metrics(self) -> str:
        if self.fetch_metrics is not None:
            try:
                return self.fetch_metrics()
            except OSError as error:
                raise MetricsUnavailableError("Failed to fetch benchmark metrics") from error
        parsed_url = urlparse(self.metrics_url)
        try:
            hostname = parsed_url.hostname
        except ValueError as error:
            raise ValueError("metrics_url must be a valid HTTP or HTTPS URL") from error
        if parsed_url.scheme not in self.SUPPORTED_SCHEMES or not hostname:
            raise ValueError("metrics_url must be a valid HTTP or HTTPS URL")
        try:
            with urlopen(self.metrics_url, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")
        except OSError as error:
            raise MetricsUnavailableError(f"Failed to fetch benchmark metrics from {self.metrics_url}") from error

    @classmethod
    def _parse_samples(cls, metrics_text: str) -> Dict[str, float]:
        samples: Dict[str, float] = {}
        for line in metrics_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = cls._SAMPLE_PATTERN.match(line)
            if not match:
                continue
            name, raw_value = match.groups()
            try:
                samples[name] = samples.get(name, 0.0) + float(raw_value)
            except ValueError:
                continue
        return samples

    @staticmethod
    def _first_value(samples: Dict[str, float], aliases: Iterable[str]) -> Optional[float]:
        for alias in aliases:
            if alias in samples:
                return samples[alias]
        return None

    def snapshot(self, timestamp: Optional[float] = None) -> VllmMetricsSnapshot:
        samples = self._parse_samples(self._fetch_metrics())
        return VllmMetricsSnapshot(
            timestamp=self.time_func() if timestamp is None else timestamp,
            output_tokens=self._first_value(samples, self.OUTPUT_TOKEN_ALIASES),
            completed_requests=self._first_value(samples, self.COMPLETED_REQUEST_ALIASES),
            failed_requests=self._first_value(samples, self.FAILED_REQUEST_ALIASES),
            ttft_sum=self._first_value(samples, self.TTFT_SUM_ALIASES),
            ttft_count=self._first_value(samples, self.TTFT_COUNT_ALIASES),
            tpot_sum=self._first_value(samples, self.TPOT_SUM_ALIASES),
            tpot_count=self._first_value(samples, self.TPOT_COUNT_ALIASES),
            running_requests=self._first_value(samples, self.RUNNING_REQUEST_ALIASES),
            waiting_requests=self._first_value(samples, self.WAITING_REQUEST_ALIASES),
        )


class EarlyExitController:
    def __init__(
        self,
        config: BenchmarkEarlyExitConfig,
        metrics_client: VllmMetricsClient,
        reference: Optional[PerformanceIndex] = None,
        fitness_evaluator=None,
        max_num_seqs: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        sample_sink: Optional[Callable[[MetricsSampleRecord], None]] = None,
        sample_interval_seconds: float = METRICS_SAMPLE_INTERVAL_SECONDS,
        min_warmup_seconds: float = MIN_WARMUP_SECONDS,
        max_warmup_seconds: float = MAX_WARMUP_SECONDS,
    ):
        self.config = config
        self.metrics_client = metrics_client
        self.reference = reference
        self.fitness_evaluator = fitness_evaluator
        self.max_num_seqs = self._positive_int(max_num_seqs)
        self.max_concurrency = self._positive_int(max_concurrency)
        self.sample_sink = sample_sink
        self.sample_interval_seconds = sample_interval_seconds
        self.min_warmup_seconds = min_warmup_seconds
        self.max_warmup_seconds = max_warmup_seconds
        self.warmup_result: Optional[WarmupResult] = None
        self._start_timestamp: Optional[float] = None
        self._first_snapshot: Optional[VllmMetricsSnapshot] = None
        self._last_snapshot: Optional[VllmMetricsSnapshot] = None
        self._evaluation_snapshot: Optional[VllmMetricsSnapshot] = None
        self._last_sample_attempt_timestamp: Optional[float] = None
        self._window_performances: list[PerformanceIndex] = []
        self._bad_windows = 0
        self._sample_index = 0
        self._warmup_sample_count = 0
        self._load_ready_consecutive_count = 0

    @staticmethod
    def _positive_int(value) -> Optional[int]:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @property
    def effective_target(self) -> Optional[int]:
        candidates = [value for value in (self.max_num_seqs, self.max_concurrency) if value]
        return min(candidates) if candidates else None

    @property
    def load_threshold(self) -> Optional[int]:
        if self.effective_target is None:
            return None
        return ceil(self.effective_target * LOAD_READY_RATIO)

    def update_runtime_context(
        self,
        max_num_seqs: Optional[int],
        max_concurrency: Optional[int],
        sample_sink: Optional[Callable[[MetricsSampleRecord], None]],
    ) -> None:
        self.max_num_seqs = self._positive_int(max_num_seqs)
        self.max_concurrency = self._positive_int(max_concurrency)
        self.sample_sink = sample_sink

    def reset(self):
        self.warmup_result = None
        self._start_timestamp = None
        self._first_snapshot = None
        self._last_snapshot = None
        self._evaluation_snapshot = None
        self._last_sample_attempt_timestamp = None
        self._window_performances = []
        self._bad_windows = 0
        self._sample_index = 0
        self._warmup_sample_count = 0
        self._load_ready_consecutive_count = 0

    def _emit(self, record: MetricsSampleRecord) -> None:
        if self.sample_sink is not None:
            self.sample_sink(record)

    def _new_record(
        self,
        snapshot: VllmMetricsSnapshot,
        phase: EarlyExitPhase,
        previous: Optional[VllmMetricsSnapshot],
    ) -> MetricsSampleRecord:
        performance = snapshot.performance_since(previous) if previous is not None else PerformanceIndex()
        load_ratio = None
        if snapshot.running_requests is not None and self.effective_target:
            load_ratio = snapshot.running_requests / self.effective_target
        load_ready = bool(
            self.load_threshold is not None
            and snapshot.running_requests is not None
            and snapshot.running_requests >= self.load_threshold
        )
        return MetricsSampleRecord(
            timestamp=snapshot.timestamp,
            pass_elapsed_seconds=snapshot.timestamp - self._start_timestamp,
            sample_index=self._sample_index,
            phase=phase.value,
            output_tokens_total=snapshot.output_tokens,
            completed_requests_total=snapshot.completed_requests,
            failed_requests_total=snapshot.failed_requests,
            running_requests=snapshot.running_requests,
            waiting_requests=snapshot.waiting_requests,
            sample_window_seconds=(snapshot.timestamp - previous.timestamp) if previous is not None else None,
            output_tokens_delta=snapshot.output_tokens_since(previous) if previous is not None else None,
            completed_requests_delta=snapshot.completed_requests_since(previous) if previous is not None else None,
            failed_requests_delta=snapshot.failed_requests_since(previous) if previous is not None else None,
            generate_speed=performance.generate_speed,
            time_to_first_token=performance.time_to_first_token,
            time_per_output_token=performance.time_per_output_token,
            success_rate=performance.success_rate,
            effective_target=self.effective_target,
            load_threshold=self.load_threshold,
            load_ratio=load_ratio,
            load_ready=load_ready,
            warmup_state="not_applicable" if phase != EarlyExitPhase.EVALUATION else "warming",
            reference_generate_speed=self.reference.generate_speed if self.reference else None,
            reference_score=self._score(self.reference),
        )

    def _finish_warmup(self, snapshot: VllmMetricsSnapshot, record: MetricsSampleRecord, reason: str) -> None:
        self.warmup_result = WarmupResult(
            end_timestamp=snapshot.timestamp,
            elapsed_seconds=record.pass_elapsed_seconds,
            reason=reason,
            sample_count=self._warmup_sample_count,
            effective_target=self.effective_target,
            load_threshold=self.load_threshold,
            running_requests=snapshot.running_requests,
            waiting_requests=snapshot.waiting_requests,
            load_ratio=record.load_ratio,
            forced=reason == "max_warmup_timeout",
        )
        self._evaluation_snapshot = snapshot
        record.warmup_state = "finished"
        record.warmup_end_event = True
        record.warmup_end_reason = reason

    def finalize_warmup_on_case_end(self) -> None:
        if self.warmup_result is not None or self._last_snapshot is None or self._first_snapshot is None:
            return
        snapshot = self._last_snapshot
        load_ratio = None
        if snapshot.running_requests is not None and self.effective_target:
            load_ratio = snapshot.running_requests / self.effective_target
        self.warmup_result = WarmupResult(
            end_timestamp=snapshot.timestamp,
            elapsed_seconds=max(0.0, snapshot.timestamp - self._start_timestamp),
            reason="case_completed_before_warmup",
            sample_count=self._warmup_sample_count,
            effective_target=self.effective_target,
            load_threshold=self.load_threshold,
            running_requests=snapshot.running_requests,
            waiting_requests=snapshot.waiting_requests,
            load_ratio=load_ratio,
            forced=False,
        )

    def _sample(self, phase: EarlyExitPhase, make_decision: bool) -> Optional[EarlyExitDecision]:
        if not self.config.enabled:
            return None
        now = self.metrics_client.time_func()
        if (
            self._last_sample_attempt_timestamp is not None
            and now - self._last_sample_attempt_timestamp < self.sample_interval_seconds
        ):
            return None
        self._last_sample_attempt_timestamp = now
        if self._start_timestamp is None:
            self._start_timestamp = now
        self._sample_index += 1
        try:
            snapshot = self.metrics_client.snapshot(timestamp=now)
        except MetricsUnavailableError as error:
            self._emit(
                MetricsSampleRecord(
                    timestamp=now,
                    pass_elapsed_seconds=max(0.0, now - self._start_timestamp),
                    sample_index=self._sample_index,
                    phase=phase.value,
                    scrape_success=False,
                    scrape_error=type(error).__name__,
                )
            )
            raise
        previous = self._last_snapshot
        if self._first_snapshot is None:
            self._first_snapshot = snapshot
        self._last_snapshot = snapshot
        record = self._new_record(snapshot, phase, previous)

        if phase != EarlyExitPhase.EVALUATION:
            self._emit(record)
            return None

        if self.warmup_result is None:
            if snapshot.running_requests is not None or snapshot.waiting_requests is not None:
                self._warmup_sample_count += 1
            if record.load_ready:
                self._load_ready_consecutive_count += 1
            else:
                self._load_ready_consecutive_count = 0
            record.load_ready_consecutive_count = self._load_ready_consecutive_count
            elapsed = record.pass_elapsed_seconds
            normal_ready = (
                elapsed >= self.min_warmup_seconds
                and self._warmup_sample_count >= MIN_VALID_WARMUP_SAMPLES
                and self._load_ready_consecutive_count >= LOAD_READY_CONSECUTIVE_SAMPLES
            )
            if normal_ready:
                self._finish_warmup(snapshot, record, "load_ready")
            elif elapsed >= self.max_warmup_seconds:
                self._finish_warmup(snapshot, record, "max_warmup_timeout")
            self._emit(record)
            return None

        record.warmup_state = "finished"
        record.warmup_end_reason = self.warmup_result.reason
        if (
            self._evaluation_snapshot is None
            or snapshot.timestamp - self._evaluation_snapshot.timestamp < self.config.window_seconds
        ):
            self._emit(record)
            return None

        evaluation_previous = self._evaluation_snapshot
        self._evaluation_snapshot = snapshot
        performance = snapshot.performance_since(evaluation_previous)
        record.formal_evaluation_window = True
        record.evaluation_generate_speed = performance.generate_speed
        record.evaluation_time_to_first_token = performance.time_to_first_token
        record.evaluation_time_per_output_token = performance.time_per_output_token
        record.evaluation_success_rate = performance.success_rate
        output_delta = snapshot.output_tokens_since(evaluation_previous)
        completed_delta = snapshot.completed_requests_since(evaluation_previous)
        eligible = (
            output_delta is not None
            and output_delta >= self.config.min_output_tokens
            and (completed_delta is None or completed_delta >= self.config.min_completed_requests)
        )
        record.eligible_for_early_exit = eligible
        if not eligible:
            self._emit(record)
            return None
        self._window_performances.append(performance)

        decision = None
        if make_decision and self.reference is not None and self.reference.generate_speed:
            decision = self._build_decision(performance)
            record.observed_score = self._score(performance)
            if decision is None:
                self._bad_windows = 0
            else:
                self._bad_windows += 1
                record.bad_window = True
                record.decision_reason = decision.reason
        record.consecutive_bad_windows = self._bad_windows
        if decision is not None and self._bad_windows >= self.config.consecutive_bad_windows:
            record.would_early_exit = True
            record.early_exit = decision.early_exit
        self._emit(record)
        if decision is None or self._bad_windows < self.config.consecutive_bad_windows:
            return None
        decision.performance.would_early_exit = True
        decision.performance.early_exit = decision.early_exit
        decision.performance.early_exit_reason = decision.reason
        decision.performance.result_source = "early_exit_metrics"
        decision.performance.usable_as_best = False
        decision.performance.reference_generate_speed = decision.reference_generate_speed
        decision.performance.observed_generate_speed = decision.observed_generate_speed
        decision.performance.reference_score = decision.reference_score
        decision.performance.observed_score = decision.observed_score
        decision.performance.slo_violations = dict(decision.slo_violations)
        return decision

    def observe(self, phase: EarlyExitPhase) -> None:
        """Collect metrics without making an early-exit decision."""
        self._sample(phase, make_decision=False)

    def representative_window(self) -> tuple[Optional[PerformanceIndex], int]:
        """Return a conservative median-speed window and the eligible sample count."""
        eligible_windows = [
            performance
            for performance in self._window_performances
            if performance.generate_speed is not None and performance.generate_speed > 0
        ]
        if not eligible_windows:
            return None, 0
        eligible_windows.sort(key=lambda performance: performance.generate_speed)
        representative = eligible_windows[(len(eligible_windows) - 1) // 2]
        return representative, len(eligible_windows)

    def check(self, phase: EarlyExitPhase) -> Optional[EarlyExitDecision]:
        return self._sample(phase, make_decision=True)

    def _build_decision(self, performance: PerformanceIndex) -> Optional[EarlyExitDecision]:
        slo_violations = self._slo_violations(performance)
        reference_score = self._score(self.reference)
        observed_score = self._score(performance)
        if (
            reference_score is not None
            and reference_score > 0
            and observed_score is not None
            and observed_score >= reference_score * self.config.relative_score_threshold
        ):
            reason = (
                f"score {observed_score:.6g} exceeded reference score {reference_score:.6g} "
                f"by threshold {self.config.relative_score_threshold}"
            )
            return self._decision(reason, performance, slo_violations, observed_score, reference_score)
        if performance.generate_speed is None:
            return None
        observed_speed = performance.generate_speed
        reference_speed = self.reference.generate_speed
        if observed_speed <= reference_speed * self.config.relative_generate_speed_threshold:
            reason = (
                f"generate_speed {observed_speed:.6g} is below reference {reference_speed:.6g} "
                f"by threshold {self.config.relative_generate_speed_threshold}"
            )
            return self._decision(reason, performance, slo_violations, observed_score, reference_score)
        return None

    def _decision(
        self,
        reason: str,
        performance: PerformanceIndex,
        slo_violations: Dict[str, bool],
        observed_score: Optional[float],
        reference_score: Optional[float],
    ) -> EarlyExitDecision:
        return EarlyExitDecision(
            would_early_exit=True,
            early_exit=self.config.action == "terminate",
            reason=reason,
            performance=performance,
            observed_score=observed_score,
            reference_score=reference_score,
            observed_generate_speed=performance.generate_speed,
            reference_generate_speed=self.reference.generate_speed if self.reference else None,
            slo_violations=slo_violations,
        )

    def _score(self, performance: Optional[PerformanceIndex]) -> Optional[float]:
        if performance is None:
            return None
        if self.fitness_evaluator is None:
            return None
        score = self.fitness_evaluator.calculate_cost(performance, require_success_rate=False)
        if not isfinite(score):
            return None
        return score

    def _slo_violations(self, performance: PerformanceIndex) -> Dict[str, bool]:
        if self.fitness_evaluator is None:
            return {}
        violations = {}
        if performance.time_to_first_token is not None:
            violations["time_to_first_token"] = performance.time_to_first_token > self.fitness_evaluator.ttft_slo
        if performance.time_per_output_token is not None:
            violations["time_per_output_token"] = performance.time_per_output_token > self.fitness_evaluator.tpot_slo
        if performance.success_rate is not None:
            violations["success_rate"] = performance.success_rate < self.fitness_evaluator.success_rate_slo
        return violations

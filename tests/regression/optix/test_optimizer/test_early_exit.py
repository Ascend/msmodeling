from unittest.mock import MagicMock, patch

import pytest

from optix.config.config import BenchmarkEarlyExitConfig, PerformanceIndex
from optix.optimizer.early_exit import (
    EarlyExitController,
    EarlyExitPhase,
    MetricsUnavailableError,
    VllmMetricsClient,
    VllmMetricsSnapshot,
)
from optix.optimizer.performance_tunner import PerformanceTuner


@pytest.mark.parametrize(
    "metrics_url",
    [
        "file:///etc/passwd",
        "ftp://127.0.0.1/metrics",
        "data:text/plain,output%201",
        "http:///metrics",
        "metrics",
    ],
)
def test_vllm_metrics_client_rejects_non_http_or_hostless_urls(metrics_url):
    client = VllmMetricsClient(metrics_url=metrics_url)

    with pytest.raises(ValueError, match="metrics_url must be a valid HTTP or HTTPS URL"):
        client.snapshot()


def test_vllm_metrics_client_fetches_valid_http_url():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"output 1\n"
    client = VllmMetricsClient(metrics_url="http://127.0.0.1:8000/metrics", timeout_seconds=2.0)

    with patch("optix.optimizer.early_exit.urlopen", return_value=response) as mock_urlopen:
        snapshot = client.snapshot()

    assert snapshot.output_tokens == 1
    mock_urlopen.assert_called_once_with("http://127.0.0.1:8000/metrics", timeout=2.0)


def test_vllm_metrics_client_test_fetcher_does_not_require_url():
    client = VllmMetricsClient(fetch_metrics=lambda: "output 1\n")

    assert client.snapshot().output_tokens == 1


def test_vllm_metrics_client_wraps_io_failures():
    client = VllmMetricsClient(metrics_url="http://127.0.0.1:8000/metrics")

    with patch("optix.optimizer.early_exit.urlopen", side_effect=OSError("connection refused")):
        with pytest.raises(MetricsUnavailableError) as error_info:
            client.snapshot()

    assert isinstance(error_info.value.__cause__, OSError)


def test_vllm_metrics_client_uses_counter_delta_and_aggregates_dp_labels():
    metrics = iter(
        [
            'output{dp="0"} 100\noutput{dp="1"} 50\n',
            'output{dp="0"} 220\noutput{dp="1"} 130\n',
        ]
    )
    timestamps = iter([10.0, 20.0])
    client = VllmMetricsClient(fetch_metrics=lambda: next(metrics), time_func=lambda: next(timestamps))

    first = client.snapshot()
    second = client.snapshot()

    assert first.output_tokens == 150
    assert second.output_tokens == 350
    assert second.output_tokens_per_second_since(first) == pytest.approx(20.0)


def test_vllm_metrics_client_supports_vllm_018_metric_names():
    metrics = iter(
        [
            "\n".join(
                [
                    'vllm:generation_tokens_total{model_name="smoke"} 10',
                    'vllm:request_success_total{model_name="smoke"} 1',
                    'vllm:request_failure_total{model_name="smoke"} 0',
                    'vllm:time_to_first_token_seconds_sum{model_name="smoke"} 0.5',
                    'vllm:time_to_first_token_seconds_count{model_name="smoke"} 1',
                    'vllm:request_time_per_output_token_seconds_sum{model_name="smoke"} 0.04',
                    'vllm:request_time_per_output_token_seconds_count{model_name="smoke"} 1',
                ]
            ),
            "\n".join(
                [
                    'vllm:generation_tokens_total{model_name="smoke"} 50',
                    'vllm:request_success_total{model_name="smoke"} 3',
                    'vllm:request_failure_total{model_name="smoke"} 0',
                    'vllm:time_to_first_token_seconds_sum{model_name="smoke"} 1.7',
                    'vllm:time_to_first_token_seconds_count{model_name="smoke"} 3',
                    'vllm:request_time_per_output_token_seconds_sum{model_name="smoke"} 0.16',
                    'vllm:request_time_per_output_token_seconds_count{model_name="smoke"} 3',
                ]
            ),
        ]
    )
    timestamps = iter([0.0, 10.0])
    client = VllmMetricsClient(fetch_metrics=lambda: next(metrics), time_func=lambda: next(timestamps))

    first = client.snapshot()
    second = client.snapshot()
    performance = second.performance_since(first)

    assert second.output_tokens_since(first) == 40
    assert performance.generate_speed == pytest.approx(4.0)
    assert performance.time_to_first_token == pytest.approx(0.6)
    assert performance.time_per_output_token == pytest.approx(0.06)
    assert performance.success_rate == pytest.approx(1.0)


def test_vllm_metrics_client_parses_running_and_waiting_requests():
    client = VllmMetricsClient(fetch_metrics=lambda: "vllm:num_requests_running 52\nvllm:num_requests_waiting 3\n")

    snapshot = client.snapshot()

    assert snapshot.running_requests == 52
    assert snapshot.waiting_requests == 3


@pytest.mark.parametrize(
    ("method_name", "field_name"),
    [
        ("output_tokens_since", "output_tokens"),
        ("completed_requests_since", "completed_requests"),
        ("failed_requests_since", "failed_requests"),
    ],
)
def test_vllm_metrics_snapshot_counter_delta_rejects_missing_or_decreased_values(method_name, field_name):
    previous = VllmMetricsSnapshot(timestamp=0.0, **{field_name: 10.0})
    missing = VllmMetricsSnapshot(timestamp=1.0)
    decreased = VllmMetricsSnapshot(timestamp=1.0, **{field_name: 9.0})

    assert getattr(missing, method_name)(previous) is None
    assert getattr(decreased, method_name)(previous) is None


def test_early_exit_controller_fetches_metrics_only_after_window_expires():
    fetch_metrics = MagicMock(side_effect=["output 0\n", "output 100\n"])
    timestamps = iter([0.0, 5.0, 10.0])
    controller = EarlyExitController(
        config=BenchmarkEarlyExitConfig(
            enabled=True,
            window_seconds=10,
            min_output_tokens=1,
        ),
        metrics_client=VllmMetricsClient(
            fetch_metrics=fetch_metrics,
            time_func=lambda: next(timestamps),
        ),
        sample_interval_seconds=10,
    )

    assert controller.check(EarlyExitPhase.CALIBRATION) is None
    assert controller.check(EarlyExitPhase.CALIBRATION) is None
    assert controller.check(EarlyExitPhase.CALIBRATION) is None

    assert fetch_metrics.call_count == 2


def test_early_exit_controller_collects_representative_window_without_reference():
    metrics = iter(
        [
            "output 0\ncompleted_requests 0\nfine_grained_tpot_sum 0\nfine_grained_tpot_count 0\n",
            "output 100\ncompleted_requests 1\nfine_grained_tpot_sum 1\nfine_grained_tpot_count 1\n",
            "output 400\ncompleted_requests 2\nfine_grained_tpot_sum 4\nfine_grained_tpot_count 2\n",
            "output 600\ncompleted_requests 3\nfine_grained_tpot_sum 6\nfine_grained_tpot_count 3\n",
        ]
    )
    timestamps = iter([0.0, 10.0, 20.0, 30.0])
    controller = EarlyExitController(
        config=BenchmarkEarlyExitConfig(
            enabled=True,
            window_seconds=1,
            min_output_tokens=1,
            min_completed_requests=1,
        ),
        metrics_client=VllmMetricsClient(
            fetch_metrics=lambda: next(metrics),
            time_func=lambda: next(timestamps),
        ),
        max_warmup_seconds=0,
    )

    for _ in range(4):
        assert controller.check(EarlyExitPhase.EVALUATION) is None

    representative, sample_count = controller.representative_window()

    assert sample_count == 3
    assert representative is not None
    assert representative.generate_speed == pytest.approx(20.0)
    assert representative.time_per_output_token == pytest.approx(2.0)


def test_early_exit_controller_uses_slo_aware_score_when_slo_metrics_are_available():
    metrics = iter(
        [
            "\n".join(
                [
                    'output{dp="0"} 100',
                    "fine_grained_ttft_sum 1.0",
                    "fine_grained_ttft_count 2",
                    "fine_grained_tpot_sum 0.08",
                    "fine_grained_tpot_count 2",
                    "completed_requests 2",
                    "failed_requests 0",
                ]
            ),
            "\n".join(
                [
                    'output{dp="0"} 1000',
                    "fine_grained_ttft_sum 6.0",
                    "fine_grained_ttft_count 6",
                    "fine_grained_tpot_sum 0.60",
                    "fine_grained_tpot_count 6",
                    "completed_requests 6",
                    "failed_requests 0",
                ]
            ),
        ]
    )
    timestamps = iter([0.0, 10.0])
    client = VllmMetricsClient(fetch_metrics=lambda: next(metrics), time_func=lambda: next(timestamps))
    controller = EarlyExitController(
        config=BenchmarkEarlyExitConfig(
            enabled=True,
            action="terminate",
            window_seconds=1,
            min_output_tokens=1,
            min_completed_requests=1,
            relative_score_threshold=1.2,
            consecutive_bad_windows=1,
        ),
        metrics_client=client,
        reference=PerformanceIndex(
            generate_speed=1000,
            time_to_first_token=0.4,
            time_per_output_token=0.04,
            success_rate=1,
        ),
        fitness_evaluator=PerformanceTuner(
            generate_speed_target=1000,
            ttft_slo=0.5,
            tpot_slo=0.05,
            success_rate_slo=1.0,
        ),
        max_warmup_seconds=0,
    )

    assert controller.check(EarlyExitPhase.EVALUATION) is None
    decision = controller.check(EarlyExitPhase.EVALUATION)

    assert decision is not None
    assert decision.early_exit is True
    assert decision.performance.usable_as_best is False
    assert decision.performance.result_source == "early_exit_metrics"
    assert decision.performance.generate_speed == pytest.approx(90.0)
    assert decision.slo_violations["time_to_first_token"] is True
    assert decision.slo_violations["time_per_output_token"] is True
    assert "score" in decision.reason


def test_early_exit_controller_ignores_zero_reference_score():
    metrics = iter(["output 0\n", "output 1000\n"])
    timestamps = iter([0.0, 10.0])
    client = VllmMetricsClient(fetch_metrics=lambda: next(metrics), time_func=lambda: next(timestamps))
    fitness_evaluator = PerformanceTuner(generate_speed_target=0)
    fitness_evaluator.w_gen = 0
    fitness_evaluator.w_ft = 0
    fitness_evaluator.w_pot = 0
    fitness_evaluator.w_succ = 0
    controller = EarlyExitController(
        config=BenchmarkEarlyExitConfig(
            enabled=True,
            action="terminate",
            window_seconds=1,
            min_output_tokens=1,
            relative_generate_speed_threshold=0.5,
            relative_score_threshold=2.0,
            consecutive_bad_windows=1,
        ),
        metrics_client=client,
        reference=PerformanceIndex(generate_speed=100),
        fitness_evaluator=fitness_evaluator,
        max_warmup_seconds=0,
    )

    assert controller.check(EarlyExitPhase.EVALUATION) is None
    assert controller.check(EarlyExitPhase.EVALUATION) is None


def test_early_exit_controller_report_action_records_would_exit_without_terminating():
    metrics = iter(['output 100\ncompleted_requests 1\n', 'output 110\ncompleted_requests 2\n'])
    timestamps = iter([0.0, 10.0])
    client = VllmMetricsClient(fetch_metrics=lambda: next(metrics), time_func=lambda: next(timestamps))
    controller = EarlyExitController(
        config=BenchmarkEarlyExitConfig(
            enabled=True,
            action="report",
            window_seconds=1,
            min_output_tokens=1,
            min_completed_requests=1,
            relative_generate_speed_threshold=0.5,
            consecutive_bad_windows=1,
        ),
        metrics_client=client,
        reference=PerformanceIndex(generate_speed=100),
        max_warmup_seconds=0,
    )

    assert controller.check(EarlyExitPhase.EVALUATION) is None
    decision = controller.check(EarlyExitPhase.EVALUATION)

    assert decision is not None
    assert decision.would_early_exit is True
    assert decision.early_exit is False
    assert decision.performance.would_early_exit is True
    assert decision.performance.early_exit is False
    assert decision.performance.usable_as_best is False


def test_adaptive_warmup_ends_after_load_is_ready_twice():
    metrics = iter(
        [
            "num_requests_running 10\nnum_requests_waiting 0\n",
            "num_requests_running 30\nnum_requests_waiting 0\n",
            "num_requests_running 52\nnum_requests_waiting 0\n",
            "num_requests_running 52\nnum_requests_waiting 0\n",
        ]
    )
    timestamps = iter([0.0, 5.0, 10.0, 15.0])
    samples = []
    controller = EarlyExitController(
        config=BenchmarkEarlyExitConfig(enabled=True),
        metrics_client=VllmMetricsClient(
            fetch_metrics=lambda: next(metrics),
            time_func=lambda: next(timestamps),
        ),
        max_num_seqs=64,
        max_concurrency=100,
        sample_sink=samples.append,
    )

    for _ in range(4):
        assert controller.check(EarlyExitPhase.EVALUATION) is None

    assert controller.effective_target == 64
    assert controller.load_threshold == 52
    assert controller.warmup_result is not None
    assert controller.warmup_result.reason == "load_ready"
    assert controller.warmup_result.elapsed_seconds == 15
    assert controller.warmup_result.sample_count == 4
    assert samples[-1].warmup_end_event is True


def test_adaptive_warmup_forces_end_at_ninety_seconds():
    metrics = iter(["num_requests_running 13\nnum_requests_waiting 57\n"] * 4)
    timestamps = iter([0.0, 30.0, 60.0, 90.0])
    samples = []
    controller = EarlyExitController(
        config=BenchmarkEarlyExitConfig(enabled=True),
        metrics_client=VllmMetricsClient(
            fetch_metrics=lambda: next(metrics),
            time_func=lambda: next(timestamps),
        ),
        max_num_seqs=64,
        max_concurrency=100,
        sample_sink=samples.append,
    )

    for _ in range(4):
        controller.check(EarlyExitPhase.EVALUATION)

    assert controller.warmup_result is not None
    assert controller.warmup_result.reason == "max_warmup_timeout"
    assert controller.warmup_result.elapsed_seconds == 90
    assert controller.warmup_result.forced is True
    assert controller.warmup_result.waiting_requests == 57
    assert all(not sample.load_ready for sample in samples)


def test_calibration_collects_samples_without_starting_warmup():
    samples = []
    controller = EarlyExitController(
        config=BenchmarkEarlyExitConfig(enabled=True),
        metrics_client=VllmMetricsClient(
            fetch_metrics=lambda: "num_requests_running 64\nnum_requests_waiting 1\n",
            time_func=lambda: 0.0,
        ),
        max_num_seqs=64,
        max_concurrency=100,
        sample_sink=samples.append,
    )

    controller.observe(EarlyExitPhase.CALIBRATION)

    assert len(samples) == 1
    assert samples[0].warmup_state == "not_applicable"
    assert controller.warmup_result is None

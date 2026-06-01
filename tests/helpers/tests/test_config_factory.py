import json
from pathlib import Path

from tests.helpers.assert_utils import assert_latency_within
from tests.helpers.config_factory import build_case_matrix, build_latency_thresholds

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_FIXTURES_DIR = Path("tests/helpers/tests/fixtures")
_TENSOR_CLOSE_CASES = _FIXTURES / "tensor_close_cases.json"
_LATENCY_THRESHOLD_CONFIG = _FIXTURES / "latency_threshold_config.json"
_LATENCY_SAMPLE_MEASUREMENTS = _FIXTURES / "latency_sample_measurements.json"


def test_build_case_matrix_expands_fixture_meta():
    with _TENSOR_CLOSE_CASES.open(encoding="utf-8") as f:
        payload = json.load(f)

    assert len(payload["cases"]) >= 3
    for case in payload["cases"]:
        meta = case["meta"]
        matrix = build_case_matrix(
            model=[meta["model_id"]],
            precision=[meta["precision"]],
            batch=[meta["batch"]],
        )
        assert len(matrix) == 1
        assert matrix[0]["model"] == meta["model_id"]
        assert matrix[0]["precision"] == meta["precision"]
        assert matrix[0]["batch"] == meta["batch"]


def test_build_latency_thresholds_with_fixture_measurements():
    with _LATENCY_THRESHOLD_CONFIG.open(encoding="utf-8") as f:
        threshold_config = json.load(f)
    with _LATENCY_SAMPLE_MEASUREMENTS.open(encoding="utf-8") as f:
        measurements_payload = json.load(f)

    thresholds = build_latency_thresholds(
        ttft_ms=threshold_config["ttft_ms"],
        tpot_ms=threshold_config["tpot_ms"],
        tolerance_ms=threshold_config["tolerance_ms"],
    )
    measurements = measurements_payload["measurements"]

    assert_latency_within(
        measurements["ttft_ms"],
        thresholds["ttft_ms"],
        metric="ttft_ms",
        tolerance_ms=thresholds["tolerance_ms"],
    )
    assert_latency_within(
        measurements["tpot_ms"],
        thresholds["tpot_ms"],
        metric="tpot_ms",
        tolerance_ms=thresholds["tolerance_ms"],
    )
    assert threshold_config["source"] == str(_FIXTURES_DIR / "latency_threshold_config.json")
    assert measurements_payload["source"] == str(_FIXTURES_DIR / "latency_sample_measurements.json")

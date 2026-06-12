from serving_cast.request import Request
from serving_cast.utils import summarize
from tests.helpers.assert_utils import assert_latency_within
from tests.helpers.config_factory import build_latency_thresholds


def test_serving_cast_ttft_tpot_smoke(ttft_ctx, capfd):
    cast_model = ttft_ctx["cast_model"]
    assert cast_model["model_id"]
    assert cast_model["op_meta"]
    thresholds = build_latency_thresholds(ttft_ms=0.5, tpot_ms=0.3, tolerance_ms=0.05)
    req = Request(num_input_tokens=8, num_output_tokens=4)
    req.leaves_client_time = 0.5
    req.arrives_server_time = 0.6
    req.prefill_done_time = 1.0
    req.decode_done_time = 1.9

    ttft = req.time_to_first_token()
    tpot = req.time_per_output_token()
    assert_latency_within(ttft, thresholds["ttft_ms"], tolerance_ms=thresholds["tolerance_ms"])
    assert_latency_within(tpot, thresholds["tpot_ms"], tolerance_ms=thresholds["tolerance_ms"])
    assert ttft <= thresholds["ttft_ms"] + thresholds["tolerance_ms"]
    assert tpot <= thresholds["tpot_ms"] + thresholds["tolerance_ms"]
    summarize([req])
    out, _ = capfd.readouterr()
    assert "output_token_throughput(tok/s)" in out

from serving_cast.request import Request
from tests.helpers.assert_utils import assert_latency_within


def test_ttft_metric_only(ttft_ctx):
    cast_model = ttft_ctx["cast_model"]
    assert cast_model["model_id"]
    assert cast_model["op_meta"]
    req = Request(num_input_tokens=16, num_output_tokens=8)
    req.leaves_client_time = 1.0
    req.prefill_done_time = 1.45
    assert_latency_within(req.time_to_first_token(), 0.45)

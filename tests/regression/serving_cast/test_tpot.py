from serving_cast.request import Request
from tests.helpers.assert_utils import assert_latency_within


def test_tpot_metric_only(cast_model):
    assert cast_model["model_id"]
    assert cast_model["op_meta"]
    req = Request(num_input_tokens=16, num_output_tokens=5)
    req.prefill_done_time = 2.0
    req.decode_done_time = 3.2
    assert_latency_within(req.time_per_output_token(), 0.3)

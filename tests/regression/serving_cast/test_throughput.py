from serving_cast.request import Request
from serving_cast.utils import summarize


def test_output_throughput_summary_contains_key(cast_model, capfd):
    assert cast_model["model_id"]
    assert cast_model["op_meta"]
    req = Request(num_input_tokens=10, num_output_tokens=10)
    req.leaves_client_time = 0.0
    req.arrives_server_time = 0.1
    req.prefill_done_time = 0.6
    req.decode_done_time = 1.6

    summarize([req])
    out, _ = capfd.readouterr()
    assert "output_token_throughput(tok/s)" in out

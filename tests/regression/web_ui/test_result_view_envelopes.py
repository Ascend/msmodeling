"""Coverage for ``services.result_view`` envelope assembly.

The Top-N SLO/dedup parity lives in ``test_result_view_topn_parity.py``; this
file covers the rest of the module so result_view reaches 100% line+branch
coverage: ``assemble_result_envelope`` dispatch, the text/video envelopes
(single + multi-case), and the throughput envelope's edge branches (top empty
guard, cross_hardware device miss, disagg phase splits, pd_ratio no-key,
``_get_devices_in_config`` variants).
"""

from __future__ import annotations

from typing import Any

from services.result_view import (
    _assemble_text_result,
    _assemble_throughput_result,
    _assemble_video_result,
    _get_devices_in_config,
    assemble_result_envelope,
)


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def _text_record(seq: int = 0, **tables: Any) -> dict[str, Any]:
    return {
        "seq": seq,
        "config": {"device": "D1"},
        "summary": {"mode": "text_generation"},
        "case_hash": f"ch{seq}",
        "tables": dict(tables),
    }


def _video_record(seq: int = 0, **tables: Any) -> dict[str, Any]:
    return {
        "seq": seq,
        "config": {"device": "D1"},
        "summary": {"mode": "video_generation"},
        "case_hash": f"ch{seq}",
        "tables": dict(tables),
    }


def _disagg_records(tpot_limits, ttft_limits) -> list[dict[str, Any]]:
    """A prefill row (parallel TP=1) + a decode row (parallel TP=2). Different
    parallels so both survive the per-parallel dedup when limits allow.
    """
    base_cfg = {
        "device": "D1",
        "concurrency": 10,
        "num_devices": 1,
        "batch_size": 8,
        "mode": "disagg_prefill",
        "tpot_limits": tpot_limits,
        "ttft_limits": ttft_limits,
        "quantize_linear_action": "W8A8_DYNAMIC",
        "quantize_attention_action": "DISABLED",
    }
    prefill = {
        "config": {**base_cfg, "parallel": "TP=1"},
        "summary": {
            "throughput_token_s": 800.0,
            "qps": None,
            "ttft_ms": 100.0,
            "tpot_ms": None,
            "mode": "disagg_prefill",
        },
        "tables": {},
    }
    decode = {
        "config": {**base_cfg, "parallel": "TP=2", "concurrency": 12},
        "summary": {
            "throughput_token_s": 600.0,
            "qps": 50.0,
            "ttft_ms": None,
            "tpot_ms": 20.0,
            "mode": "disagg_decode",
        },
        "tables": {},
    }
    return [prefill, decode]


# ---------------------------------------------------------------------------
# assemble_result_envelope dispatch
# ---------------------------------------------------------------------------


def test_dispatch_empty_records():
    env = assemble_result_envelope("text_generate", [], input_config={"device": ["D1"]})
    assert env["mode"] == "empty"
    assert env["best_config"] is None


def test_dispatch_text_video_throughput_unknown():
    assert assemble_result_envelope("text_generate", [_text_record()])["mode"] == "text_generation"
    assert assemble_result_envelope("video_generate", [_video_record()])["mode"] == "video_generation"
    agg_rec = {
        "config": {
            "device": "D1",
            "parallel": "TP=1",
            "mode": "aggregation",
            "tpot_limits": None,
            "ttft_limits": None,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quantize_attention_action": "DISABLED",
        },
        "summary": {"throughput_token_s": 100.0, "qps": None, "ttft_ms": 10.0, "tpot_ms": 5.0, "mode": "aggregation"},
        "tables": {},
    }
    assert (
        assemble_result_envelope("throughput_optimizer", [agg_rec], input_config={"device": ["D1"]})["mode"]
        == "aggregation"
    )
    # Unknown module -> warning + empty envelope.
    env = assemble_result_envelope("nope", [_text_record()])
    assert env["mode"] == "empty"


# ---------------------------------------------------------------------------
# Text envelope: single (job_id None -> chrome_trace False branch) + multi
# (job_id set -> chrome_trace True branch via trace_path).
# ---------------------------------------------------------------------------


def test_text_single_envelope():
    env = _assemble_text_result([_text_record(0, batch_size=4, run_time_s=1.2)], {"device": ["D1"]})
    assert env["mode"] == "text_generation"
    assert env["batch_size"] == 4
    assert env["op_breakdown"] == []
    assert env["chrome_trace"] == {"available": False}  # job_id None -> False branch


def test_text_multi_case_envelope():
    env = _assemble_text_result([_text_record(0), _text_record(1)], {"device": ["D1"]}, job_id="job1")
    assert env["multi_case"] is True
    assert len(env["cases"]) == 2
    # job_id set -> trace_path(...) evaluated (file absent -> available False).
    assert env["cases"][0]["chrome_trace"] == {"available": False}


def test_text_empty_records_returns_empty_dict():
    assert _assemble_text_result([], None) == {}


# ---------------------------------------------------------------------------
# Video envelope: single + multi (both chrome_trace branches).
# ---------------------------------------------------------------------------


def test_video_single_envelope():
    env = _assemble_video_result([_video_record(0, execution_time_s={"a": 1})], {"device": ["D1"]}, job_id="job1")
    assert env["mode"] == "video_generation"
    assert env["execution_time_s"] == {"a": 1}
    assert env["table_rows"] == []
    assert env["chrome_trace"] == {"available": False}  # job_id set -> trace_path branch


def test_video_multi_case_envelope():
    env = _assemble_video_result([_video_record(0), _video_record(1)], {"device": ["D1"]})
    assert env["multi_case"] is True
    assert len(env["cases"]) == 2
    assert env["cases"][0]["chrome_trace"] == {"available": False}  # job_id None -> False branch


def test_video_empty_records_returns_empty_dict():
    assert _assemble_video_result([], None) == {}


# ---------------------------------------------------------------------------
# Throughput envelope edge branches
# ---------------------------------------------------------------------------


def test_throughput_top_empty_guard():
    env = _assemble_throughput_result([], {"device": ["D1"]})
    assert env["mode"] == "aggregation"
    assert env["best_config"] is None
    assert env["sweep_rows"] == []


def test_cross_hardware_device_not_in_records():
    # Records are all device D1; input asks for D1 + D2 -> D2 misses (inner
    # `if device == device_name` False; `if best_for_device` False), D1 found.
    rec = {
        "config": {
            "device": "D1",
            "parallel": "TP=1",
            "concurrency": 10,
            "num_devices": 1,
            "batch_size": 8,
            "mode": "aggregation",
            "tpot_limits": None,
            "ttft_limits": None,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quantize_attention_action": "DISABLED",
        },
        "summary": {"throughput_token_s": 100.0, "qps": None, "ttft_ms": 10.0, "tpot_ms": 5.0, "mode": "aggregation"},
        "tables": {},
    }
    env = _assemble_throughput_result([rec], {"device": ["D1", "D2"]})
    assert [c["device"] for c in env["cross_hardware"]] == ["D1"]


def test_disagg_both_phases_survive():
    env = _assemble_throughput_result(_disagg_records(None, None), {"device": ["D1"]})
    assert env["disagg_prefill"] is not None and len(env["disagg_prefill"]) == 1
    assert env["disagg_decode"] is not None and len(env["disagg_decode"]) == 1


def test_disagg_prefill_filtered_only_decode():
    # ttft_limit set restrictively -> prefill (only ttft matters) filtered;
    # decode (only tpot matters) survives since tpot_limit is unset.
    env = _assemble_throughput_result(_disagg_records(None, 50.0), {"device": ["D1"]})
    assert env["disagg_prefill"] is None  # prefill ttft=100 > 50 filtered
    assert env["disagg_decode"] is not None  # decode tpot=20 <= inf survives


def test_disagg_decode_filtered_only_prefill():
    # tpot_limit set restrictively -> decode (only tpot matters) filtered;
    # prefill (only ttft matters) survives since ttft_limit is unset.
    env = _assemble_throughput_result(_disagg_records(5.0, None), {"device": ["D1"]})
    assert env["disagg_prefill"] is not None  # prefill ttft=100 <= inf survives
    assert env["disagg_decode"] is None  # decode tpot=20 > 5 filtered


def test_pd_ratio_record_without_pd_keys_skipped():
    # A pd-mode record carrying neither pd_ratio nor balanced_qps -> the
    # `if "pd_ratio" in summary or "balanced_qps" in summary` guard is False.
    rec = {
        "config": {
            "device": "D1",
            "parallel_p": "TP=1",
            "parallel_d": "TP=1",
            "mode": "pd_ratio",
            "tpot_limits": None,
            "ttft_limits": None,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quantize_attention_action": "DISABLED",
        },
        "summary": {"mode": "pd_ratio", "ttft_ms": 100.0, "tpot_ms": 20.0},
        "tables": {},
    }
    env = _assemble_throughput_result([rec], {"device": ["D1"]})
    assert env["mode"] == "pd_ratio"
    assert env["pd_ratio_rows"] is None  # guard False -> nothing collected


# ---------------------------------------------------------------------------
# _get_devices_in_config variants
# ---------------------------------------------------------------------------


def test_get_devices_in_config_variants():
    assert _get_devices_in_config({"device": ["D1", "D2"]}) == ["D1", "D2"]  # list
    assert _get_devices_in_config({"device": "D1"}) == ["D1"]  # str
    assert _get_devices_in_config({}) == []  # missing
    assert _get_devices_in_config({"device": None}) == []  # None/other

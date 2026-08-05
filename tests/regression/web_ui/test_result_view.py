"""Unit tests for result_view module."""

from __future__ import annotations

from services.result_view import (
    _assemble_text_result,
    _assemble_throughput_result,
    _assemble_video_result,
    _get_devices_in_config,
    _reassign_local_ranks,
    _single_text_envelope,
    _single_video_envelope,
    _sweep_row_from_record,
    assemble_result_envelope,
)


class TestGetDevicesInConfig:
    """Tests for _get_devices_in_config helper."""

    def test_get_devices_from_list(self):
        """Extracts devices from list config."""
        config = {"device": ["A100", "H100"]}
        result = _get_devices_in_config(config)
        assert result == ["A100", "H100"]

    def test_get_devices_from_string(self):
        """Wraps single device string in list."""
        config = {"device": "A100"}
        result = _get_devices_in_config(config)
        assert result == ["A100"]

    def test_get_devices_empty_when_missing(self):
        """Returns empty list when device is missing."""
        config = {}
        result = _get_devices_in_config(config)
        assert result == []

    def test_get_devices_empty_when_null(self):
        """Returns empty list when device is None."""
        config = {"device": None}
        result = _get_devices_in_config(config)
        assert result == []


class TestSingleTextEnvelope:
    """Tests for _single_text_envelope."""

    def test_single_text_envelope_extracts_tables(self):
        """Extracts all text generation metrics from tables."""
        record = {
            "tables": {
                "batch_size": 32,
                "run_time_s": 10.5,
                "execution_time_s": 8.2,
                "tps_per_model": 100.5,
                "memory_gb": 16.2,
                "breakdowns_raw": {},
                "breakdowns_percent": {},
                "perf_model_name": "gpt2",
                "op_breakdown": [],
            }
        }
        result = _single_text_envelope(record)
        assert result["batch_size"] == 32
        assert result["tps_per_model"] == 100.5
        assert result["op_breakdown"] == []

    def test_single_text_envelope_missing_tables(self):
        """Handles missing tables gracefully."""
        record = {}
        result = _single_text_envelope(record)
        assert result["batch_size"] is None
        assert result["run_time_s"] is None

    def test_single_text_envelope_partial_tables(self):
        """Handles partial tables data."""
        record = {"tables": {"batch_size": 32}}
        result = _single_text_envelope(record)
        assert result["batch_size"] == 32
        assert result["tps_per_model"] is None


class TestSingleVideoEnvelope:
    """Tests for _single_video_envelope."""

    def test_single_video_envelope_extracts_tables(self):
        """Extracts all video generation metrics from tables."""
        record = {
            "tables": {
                "execution_time_s": {"encode": 5.0, "decode": 3.0},
                "breakdowns": {},
                "table_rows": [],
                "op_breakdown": [],
            }
        }
        result = _single_video_envelope(record)
        assert result["execution_time_s"] == {"encode": 5.0, "decode": 3.0}
        assert result["op_breakdown"] == []

    def test_single_video_envelope_missing_tables(self):
        """Handles missing tables gracefully."""
        record = {}
        result = _single_video_envelope(record)
        assert result["execution_time_s"] == {}
        assert result["breakdowns"] == {}


class TestAssembleTextResult:
    """Tests for _assemble_text_result."""

    def test_assemble_text_single_record(self):
        """Assembles single text generation record."""
        records = [
            {
                "seq": 0,
                "case_hash": "hash123",
                "config": {"model": "gpt2"},
                "summary": {"status": "success"},
                "tables": {"batch_size": 32},
            }
        ]
        result = _assemble_text_result(records, {"model": "gpt2"}, "job123")
        assert result["mode"] == "text_generation"
        assert result["case_hash"] == "hash123"
        assert result["batch_size"] == 32
        assert result["input_config"] == {"model": "gpt2"}

    def test_assemble_text_multi_case(self):
        """Assembles multiple text generation records as multi-case."""
        records = [
            {"seq": 0, "case_hash": "hash1", "config": {}, "summary": {}, "tables": {}},
            {"seq": 1, "case_hash": "hash2", "config": {}, "summary": {}, "tables": {}},
        ]
        result = _assemble_text_result(records, {}, "job123")
        assert result["mode"] == "text_generation"
        assert result["multi_case"] is True
        assert len(result["cases"]) == 2

    def test_assemble_text_empty_records(self):
        """Empty records returns empty dict."""
        result = _assemble_text_result([], {})
        assert result == {}


class TestAssembleVideoResult:
    """Tests for _assemble_video_result."""

    def test_assemble_video_single_record(self):
        """Assembles single video generation record."""
        records = [
            {
                "seq": 0,
                "case_hash": "hash123",
                "config": {"frames": 100},
                "summary": {"status": "success"},
                "tables": {"execution_time_s": {}},
            }
        ]
        result = _assemble_video_result(records, {"frames": 100}, "job123")
        assert result["mode"] == "video_generation"
        assert result["case_hash"] == "hash123"

    def test_assemble_video_multi_case(self):
        """Assembles multiple video generation records as multi-case."""
        records = [
            {"seq": 0, "case_hash": "hash1", "config": {}, "summary": {}, "tables": {}},
            {"seq": 1, "case_hash": "hash2", "config": {}, "summary": {}, "tables": {}},
        ]
        result = _assemble_video_result(records, {}, "job123")
        assert result["mode"] == "video_generation"
        assert result["multi_case"] is True
        assert len(result["cases"]) == 2

    def test_assemble_video_empty_records(self):
        """Empty records returns empty dict."""
        result = _assemble_video_result([], {})
        assert result == {}


class TestSweepRowFromRecord:
    """Tests for _sweep_row_from_record."""

    def test_sweep_row_extracts_fields(self):
        """Extracts sweep row fields from record."""
        record = {
            "rank": 1,
            "config": {"device": "A100", "parallel": 2, "concurrency": 4, "num_devices": 1, "batch_size": 32},
            "summary": {"throughput_token_s": 100.5, "qps": 10.2, "ttft_ms": 50.0, "tpot_ms": 20.0},
        }
        result = _sweep_row_from_record(record)
        assert result["rank"] == 1
        assert result["device"] == "A100"
        assert result["throughput_token_s"] == 100.5

    def test_sweep_row_handles_missing_fields(self):
        """Handles missing fields gracefully."""
        record = {}
        result = _sweep_row_from_record(record)
        assert result["rank"] is None
        assert result["device"] is None


class TestReassignLocalRanks:
    """Tests for _reassign_local_ranks."""

    def test_reassign_ranks_copies_records(self):
        """Creates copy of records."""
        records = [
            {"rank": 1, "summary": {"throughput_token_s": 100}},
            {"rank": 2, "summary": {"throughput_token_s": 50}},
        ]
        result = _reassign_local_ranks(records)
        # Original records unchanged
        assert records[0]["rank"] == 1
        assert records[1]["rank"] == 2
        # Result has re-ranked
        assert len(result) == 2

    def test_reassign_ranks_uses_assign_optimizer_ranks(self):
        """Uses assign_optimizer_ranks for ranking."""
        records = [
            {"rank": 10, "summary": {"throughput_token_s": 50}},
            {"rank": 20, "summary": {"throughput_token_s": 100}},
        ]
        result = _reassign_local_ranks(records)
        # Highest throughput should get rank 1
        assert result[1]["rank"] == 1


class TestAssembleThroughputResult:
    """Tests for _assemble_throughput_result."""

    def test_assemble_throughput_empty_records(self):
        """Empty records returns default structure."""
        result = _assemble_throughput_result([], {})
        assert result["mode"] == "aggregation"
        assert result["best_config"] is None
        assert result["sweep_rows"] == []

    def test_assemble_throughput_finds_best_config(self):
        """Finds rank=1 record as best_config."""
        records = [
            {
                "rank": 1,
                "config": {
                    "device": "A100",
                    "parallel": 2,
                    "concurrency": 4,
                    "num_devices": 1,
                    "batch_size": 32,
                    "tpot_limits": None,
                    "ttft_limits": None,
                    "quantize_linear_action": None,
                    "quantize_attention_action": None,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 100, "qps": 10, "ttft_ms": 50, "tpot_ms": 20},
            },
            {
                "rank": 2,
                "config": {
                    "device": "A100",
                    "parallel": 1,
                    "concurrency": 1,
                    "num_devices": 1,
                    "batch_size": 32,
                    "tpot_limits": None,
                    "ttft_limits": None,
                    "quantize_linear_action": None,
                    "quantize_attention_action": None,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 50, "qps": 5, "ttft_ms": 60, "tpot_ms": 30},
            },
        ]
        result = _assemble_throughput_result(records, {})
        assert result["best_config"] is not None
        assert result["best_config"]["device"] == "A100"
        assert result["best_config"]["throughput_token_s"] == 100

    def test_assemble_throughput_builds_sweep_rows(self):
        """Builds sweep_rows from all records."""
        records = [
            {
                "rank": 1,
                "config": {
                    "device": "A100",
                    "parallel": 2,
                    "concurrency": 4,
                    "num_devices": 1,
                    "batch_size": 32,
                    "tpot_limits": None,
                    "ttft_limits": None,
                    "quantize_linear_action": None,
                    "quantize_attention_action": None,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 100, "qps": 10, "ttft_ms": 50, "tpot_ms": 20},
            },
            {
                "rank": 2,
                "config": {
                    "device": "A100",
                    "parallel": 1,
                    "concurrency": 1,
                    "num_devices": 1,
                    "batch_size": 32,
                    "tpot_limits": None,
                    "ttft_limits": None,
                    "quantize_linear_action": None,
                    "quantize_attention_action": None,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 50, "qps": 5, "ttft_ms": 60, "tpot_ms": 30},
            },
        ]
        result = _assemble_throughput_result(records, {})
        assert len(result["sweep_rows"]) == 2
        # Sorted by rank
        assert result["sweep_rows"][0]["rank"] == 1
        assert result["sweep_rows"][1]["rank"] == 2

    def test_assemble_throughput_pd_ratio_mode(self):
        """Extracts PD-ratio specific fields."""
        records = [
            {
                "rank": 1,
                "config": {"device": "A100", "parallel_p": 2, "parallel_d": 2},
                "summary": {
                    "mode": "pd_ratio",
                    "balanced_qps": 50,
                    "pd_ratio": 1.0,
                    "p_qps": 30,
                    "d_qps": 20,
                },
            }
        ]
        result = _assemble_throughput_result(records, {})
        assert result["mode"] == "pd_ratio"
        assert result["best_config"]["balanced_qps"] == 50
        assert result["pd_ratio_rows"] is not None

    def test_assemble_throughput_disagg_mode(self):
        """Partitions records by disagg phase."""
        # Distinct parallels so both phases survive the per-parallel Top-N dedup.
        records = [
            {
                "rank": 1,
                "config": {"device": "A100", "parallel": 2},
                "summary": {"mode": "disagg_prefill"},
            },
            {
                "rank": 1,
                "config": {"device": "A100", "parallel": 4},
                "summary": {"mode": "disagg_decode"},
            },
        ]
        result = _assemble_throughput_result(records, {})
        assert result["mode"].startswith("disagg")
        assert result["disagg_prefill"] is not None
        assert result["disagg_decode"] is not None

    def test_assemble_throughput_cross_hardware(self):
        """Cross_hardware grouping requires multiple devices in config."""
        # Use same case fields for both records
        case_fields = {
            "tpot_limits": None,
            "ttft_limits": None,
            "quantize_linear_action": None,
            "quantize_attention_action": None,
        }
        records = [
            {
                "rank": 1,
                "config": {
                    "device": "A100",
                    "parallel": 2,
                    "concurrency": 4,
                    "num_devices": 1,
                    "batch_size": 32,
                    **case_fields,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 100, "qps": 10, "ttft_ms": 50, "tpot_ms": 20},
            },
            {
                "rank": 1,
                "config": {
                    "device": "H100",
                    "parallel": 1,
                    "concurrency": 1,
                    "num_devices": 1,
                    "batch_size": 32,
                    **case_fields,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 80, "qps": 8, "ttft_ms": 55, "tpot_ms": 25},
            },
        ]
        result = _assemble_throughput_result(records, {"device": ["A100", "H100"]})
        # Should return multi_case since devices differ
        assert result["multi_case"] is True
        assert len(result["cases"]) == 2

    def test_assemble_throughput_multi_case(self):
        """Groups records by case config - different device creates different cases."""
        records = [
            {
                "rank": 1,
                "config": {
                    "device": "A100",
                    "tpot_limits": None,
                    "ttft_limits": None,
                    "quantize_linear_action": None,
                    "quantize_attention_action": None,
                    "parallel": 2,
                    "concurrency": 4,
                    "num_devices": 1,
                    "batch_size": 32,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 100, "qps": 10, "ttft_ms": 50, "tpot_ms": 20},
                "case_hash": "case1",
            },
            {
                "rank": 1,
                "config": {
                    "device": "H100",
                    "tpot_limits": None,
                    "ttft_limits": None,
                    "quantize_linear_action": None,
                    "quantize_attention_action": None,
                    "parallel": 2,
                    "concurrency": 4,
                    "num_devices": 1,
                    "batch_size": 32,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 80, "qps": 8, "ttft_ms": 55, "tpot_ms": 25},
                "case_hash": "case2",
            },
        ]
        result = _assemble_throughput_result(records, {})
        assert result["multi_case"] is True
        assert len(result["cases"]) == 2
        # Each case should have best_config from _assemble_throughput_result
        assert "best_config" in result["cases"][0]

    def test_cross_hardware_single_case_multiple_devices(self):
        """Records sharing the same case_key (same device) + multiple devices
        listed in input_config exercises the cross_hardware device loop body.
        Only devices with matching records appear in cross_hardware.
        """
        case_fields = {
            "tpot_limits": None,
            "ttft_limits": None,
            "quantize_linear_action": None,
            "quantize_attention_action": None,
        }
        # Both records are for A100 (same case_key → single case, not multi_case).
        records = [
            {
                "rank": 1,
                "config": {
                    "device": "A100",
                    "parallel": 2,
                    "concurrency": 4,
                    "num_devices": 1,
                    "batch_size": 32,
                    **case_fields,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 100, "qps": 10, "ttft_ms": 50, "tpot_ms": 20},
            },
            {
                "rank": 2,
                "config": {
                    "device": "A100",
                    "parallel": 1,
                    "concurrency": 1,
                    "num_devices": 1,
                    "batch_size": 32,
                    **case_fields,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 80, "qps": 8, "ttft_ms": 55, "tpot_ms": 25},
            },
        ]
        # input_config lists two devices; only A100 has records → one cross_hardware row.
        result = _assemble_throughput_result(records, {"device": ["A100", "H100"]})
        assert result.get("multi_case") is not True
        assert len(result["cross_hardware"]) == 1
        assert result["cross_hardware"][0]["device"] == "A100"
        # rank=1 record is the best for A100.
        assert result["cross_hardware"][0]["throughput_token_s"] == 100

    def test_reranks_records_so_best_config_always_set(self):
        """Input ranks are recomputed locally: even records passed WITHOUT a
        rank=1 are reranked, so best_config is always set for non-empty input
        (the persisted/global rank is not trusted at read time).
        """
        case_fields = {
            "tpot_limits": None,
            "ttft_limits": None,
            "quantize_linear_action": None,
            "quantize_attention_action": None,
        }
        records = [
            {
                "rank": 5,  # no rank=1 in input → rerank assigns rank=1
                "config": {
                    "device": "A100",
                    "parallel": 2,
                    "concurrency": 4,
                    "num_devices": 1,
                    "batch_size": 32,
                    **case_fields,
                },
                "summary": {"mode": "aggregation", "throughput_token_s": 100, "qps": 10, "ttft_ms": 50, "tpot_ms": 20},
            },
        ]
        result = _assemble_throughput_result(records, {"device": ["A100"]})
        assert result["best_config"] is not None  # rerank → rank=1 present
        assert result["best_config"]["throughput_token_s"] == 100
        assert result["cross_hardware"] != []  # A100 matches a record

    def test_multi_case_groups_multiple_records_same_key(self):
        """Multiple records sharing a case_key land in the same group
        (covers the key-already-in-groups branch).
        """
        case_fields = {
            "tpot_limits": None,
            "ttft_limits": None,
            "quantize_linear_action": None,
            "quantize_attention_action": None,
        }
        base = {"parallel": 2, "concurrency": 4, "num_devices": 1, "batch_size": 32, **case_fields}
        # Two records for A100 (same key), one for H100 (different key).
        records = [
            {
                "rank": 1,
                "config": {"device": "A100", **base},
                "summary": {"mode": "aggregation", "throughput_token_s": 100},
            },
            {
                "rank": 2,
                "config": {"device": "A100", **base},
                "summary": {"mode": "aggregation", "throughput_token_s": 90},
            },
            {
                "rank": 1,
                "config": {"device": "H100", **base},
                "summary": {"mode": "aggregation", "throughput_token_s": 80},
            },
        ]
        result = _assemble_throughput_result(records, {})
        assert result["multi_case"] is True
        assert len(result["cases"]) == 2  # A100 group + H100 group

    def test_disagg_mode_empty_partitions(self):
        """disagg mode with no prefill/decode records yields None partitions."""
        records = [
            {
                "rank": 1,
                "config": {"device": "A100"},
                "summary": {"mode": "disagg_prefill"},  # only prefill, no decode
            },
        ]
        result = _assemble_throughput_result(records, {})
        assert result["disagg_prefill"] is not None
        assert result["disagg_decode"] is None

    def test_disagg_mode_no_matching_phase_records(self):
        """disagg mode where no record matches prefill OR decode → both None.
        (mode inferred from config, but records lack the phase summary.mode.)
        """
        records = [
            {
                "rank": 1,
                "config": {"device": "A100", "mode": "disagg_prefill"},
                "summary": {},  # no 'mode' in summary → doesn't match either phase
            },
        ]
        result = _assemble_throughput_result(records, {})
        assert result["disagg_prefill"] is None
        assert result["disagg_decode"] is None

    def test_pd_ratio_mode_no_matching_rows(self):
        """pd_ratio mode with no pd_ratio/balanced_qps rows → None."""
        records = [
            {
                "rank": 1,
                "config": {"device": "A100"},
                "summary": {"mode": "pd_ratio"},  # mode set but no pd_ratio fields
            },
        ]
        result = _assemble_throughput_result(records, {})
        assert result["pd_ratio_rows"] is None


class TestAssembleResultEnvelope:
    """Tests for assemble_result_envelope main function."""

    def test_assemble_empty_records(self):
        """Empty records returns empty envelope."""
        result = assemble_result_envelope("text_generate", [])
        assert result["mode"] == "empty"
        assert result["best_config"] is None

    def test_assemble_text_generate(self):
        """Routes to text generation assembler."""
        records = [
            {
                "seq": 0,
                "config": {},
                "summary": {},
                "tables": {},
            }
        ]
        result = assemble_result_envelope("text_generate", records)
        assert result["mode"] == "text_generation"

    def test_assemble_video_generate(self):
        """Routes to video generation assembler."""
        records = [
            {
                "seq": 0,
                "config": {},
                "summary": {},
                "tables": {},
            }
        ]
        result = assemble_result_envelope("video_generate", records)
        assert result["mode"] == "video_generation"

    def test_assemble_throughput_optimizer(self):
        """Routes to throughput optimizer assembler."""
        records = [
            {
                "rank": 1,
                "config": {},
                "summary": {},
            }
        ]
        result = assemble_result_envelope("throughput_optimizer", records)
        assert result["mode"] == "aggregation"

    def test_assemble_unknown_module(self):
        """Unknown module_id returns empty envelope."""
        records = [{"config": {}, "summary": {}}]
        result = assemble_result_envelope("unknown_module", records)
        assert result["mode"] == "empty"

    def test_assemble_with_input_config(self):
        """Passes input_config through to result."""
        records = [{"seq": 0, "config": {}, "summary": {}, "tables": {}}]
        input_config = {"model": "gpt2"}
        result = assemble_result_envelope("text_generate", records, input_config=input_config)
        assert result["input_config"] == input_config

    def test_assemble_with_job_id(self):
        """Passes job_id for trace checking."""
        records = [
            {
                "seq": 0,
                "config": {},
                "summary": {},
                "tables": {},
            }
        ]
        # Should not raise, trace_path will be called
        result = assemble_result_envelope("text_generate", records, job_id="job123")
        assert result["mode"] == "text_generation"

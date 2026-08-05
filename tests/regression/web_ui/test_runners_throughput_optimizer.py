"""Real unit tests for runners/throughput_optimizer.py.

Real analytic inference via ``_run_throughput_sweep`` on TEST_DEVICE (the
pattern from tests/regression/serving_cast — ParallelRunner models analytically,
no weight download). Pure helpers (``_split_*``, ``_build_namespace``,
``_infer_mode``) run directly. Per tests/SKILL.md — real imports, fixture-scoped
mocks only.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from runners.throughput_optimizer import (
    ThroughputOptimizerAdapter,
    _agg_disagg_row,
    _build_namespace,
    _disagg_qps,
    _enum_val,
    _fnum,
    _infer_mode,
    _pd_ratio_row,
    _run_throughput_sweep,
    _split_floats,
    _split_ints,
    _summary_to_rows,
    execute,
)

from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

# Tiny analytic config for the real sweep (~15s on TEST_DEVICE).
_TINY_THROUGHPUT_PARAMS = {
    "model_id": "Qwen/Qwen3-32B",
    "device": "TEST_DEVICE",
    "num_devices": 1,
    "query_length": 8,
    "num_queries": 1,
    "quantize_linear_action": "W8A8_DYNAMIC",
    "quantize_attention_action": "DISABLED",
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestEnumVal:
    def test_enum_to_value(self):
        assert _enum_val(QuantizeLinearAction.W8A8_DYNAMIC) == "W8A8_DYNAMIC"

    def test_passthrough_non_enum(self):
        assert _enum_val("plain") == "plain"
        assert _enum_val(5) == 5


class TestSplitInts:
    def test_none_passthrough(self):
        assert _split_ints(None) is None

    def test_list_coerced(self):
        assert _split_ints(["1", "2"]) == [1, 2]

    def test_string_parsed(self):
        assert _split_ints("1, 2 4") == [1, 2, 4]


class TestSplitFloats:
    def test_none_empty(self):
        assert _split_floats(None) == []

    def test_list_coerced(self):
        assert _split_floats(["1", "2.5"]) == [1.0, 2.5]

    def test_string_parsed(self):
        assert _split_floats("0.5, 1.0") == [0.5, 1.0]


class TestInferMode:
    def test_pd_ratio_wins(self):
        assert _infer_mode(MagicMock(), True, False) == "pd_ratio"

    def test_disagg_decode_when_tpot_only(self):
        args = SimpleNamespace(tpot_limits=10.0, ttft_limits=None)
        assert _infer_mode(args, False, True) == "disagg_decode"

    def test_disagg_prefill_otherwise(self):
        args = SimpleNamespace(tpot_limits=None, ttft_limits=5.0)
        assert _infer_mode(args, False, True) == "disagg_prefill"

    def test_aggregation_default(self):
        args = SimpleNamespace()
        assert _infer_mode(args, False, False) == "aggregation"


class TestBuildNamespace:
    """Tests for _build_namespace (param dict -> argparse Namespace)."""

    def test_basic_fields_mapped(self):
        ns = _build_namespace({**_TINY_THROUGHPUT_PARAMS, "parallel": "tp=1"})
        # device is set to None by _build_namespace (set per-loop in the sweep).
        assert ns.device is None
        assert ns.model_id == "Qwen/Qwen3-32B"
        assert ns.num_devices == 1

    def test_null_int_field_uses_default(self):
        """A present-but-null int field is coerced to its default."""
        ns = _build_namespace({"model_id": "m", "num_devices": None})
        assert ns.num_devices == 1

    def test_empty_string_int_field_uses_default(self):
        ns = _build_namespace({"model_id": "m", "num_devices": ""})
        assert ns.num_devices == 1

    def test_non_numeric_int_field_uses_default(self):
        """A non-numeric int field hits the except -> default (covers _int except)."""
        ns = _build_namespace({"model_id": "m", "num_devices": "abc"})
        assert ns.num_devices == 1

    def test_non_numeric_num_field_uses_default(self):
        ns = _build_namespace({"model_id": "m", "reserved_memory_gb": "xyz"})
        assert ns.reserved_memory_gb == 10.0

    def test_empty_num_field_uses_default(self):
        """A None/'' float field (_num) is coerced to its default."""
        ns = _build_namespace({"model_id": "m", "reserved_memory_gb": ""})
        assert ns.reserved_memory_gb == 10.0

    def test_num_or_none_for_empty_limit(self):
        ns = _build_namespace({"model_id": "m", "ttft_limits": ""})
        assert ns.ttft_limits is None

    def test_num_or_none_non_numeric_is_none(self):
        ns = _build_namespace({"model_id": "m", "ttft_limits": "notanum"})
        assert ns.ttft_limits is None

    def test_tp_sizes_default_when_all_unset(self):
        """When tp/ep/moe-dp sizes are all unset, tp_sizes defaults to [] (search)."""
        ns = _build_namespace({"model_id": "m"})
        assert ns.tp_sizes == []

    def test_split_ints_fields_parsed(self):
        ns = _build_namespace(
            {
                "model_id": "m",
                "tp_sizes": "1,2,4",
                "ep_sizes": ["1", "2"],
            }
        )
        assert ns.tp_sizes == [1, 2, 4]
        assert ns.ep_sizes == [1, 2]


# ---------------------------------------------------------------------------
# Real analytic sweep
# ---------------------------------------------------------------------------


class TestRunThroughputSweepReal:
    """Real ParallelRunner sweep on TEST_DEVICE."""

    def test_sweep_returns_tagged_records(self):
        records = _run_throughput_sweep(dict(_TINY_THROUGHPUT_PARAMS))
        assert len(records) >= 1
        rec = records[0]
        assert "config" in rec and "summary" in rec and "tables" in rec
        assert rec["config"]["device"] == "TEST_DEVICE"
        # case_tag fields merged into config.
        assert "quantize_linear_action" in rec["config"]
        # seq assigned 0..N-1.
        assert all("seq" in r for r in records)

    def test_report_final_result_failure_swallowed(self):
        """summary.report_final_result raising is swallowed (best-effort)."""
        # Patch ParallelRunner to return a runner whose summaries' report raises,
        # but still yield summaries so the loop runs.
        with patch("serving_cast.parallel_runner.ParallelRunner") as mock_pr:
            runner = MagicMock()
            summary = MagicMock()
            summary.report_final_result.side_effect = RuntimeError("report failed")
            # get_summary_df returns a non-empty df so _summary_to_rows yields rows.
            import pandas as pd

            summary.get_summary_df.return_value = pd.DataFrame({"a": [1]})
            runner.run_agg.return_value = [summary]
            mock_pr.return_value = runner
            with patch(
                "runners.throughput_optimizer._summary_to_rows",
                return_value=[{"config": {}, "summary": {}, "tables": {}}],
            ):
                records = _run_throughput_sweep(dict(_TINY_THROUGHPUT_PARAMS))
        assert len(records) >= 1  # report failure didn't abort

    def test_device_list_branch(self):
        """device passed as a list is coerced to its first element (covers the
        list branch in _run_throughput_sweep, ParallelRunner mocked for speed).
        """
        import pandas as pd

        with patch("serving_cast.parallel_runner.ParallelRunner") as mock_pr:
            runner = MagicMock()
            summary = MagicMock()
            summary.get_summary_df.return_value = pd.DataFrame(
                {
                    "parallel": ["TP=1"],
                    "concurrency": [4],
                    "num_devices": [1],
                    "batch_size": [8],
                    "token/s": [100.0],
                    "ttft": [10.0],
                    "tpot": [5.0],
                }
            )
            runner.run_agg.return_value = [summary]
            mock_pr.return_value = runner
            records = _run_throughput_sweep({**_TINY_THROUGHPUT_PARAMS, "device": ["TEST_DEVICE"]})
        assert len(records) >= 1
        assert records[0]["config"]["device"] == "TEST_DEVICE"


# ---------------------------------------------------------------------------
# Row helpers (pure data reshape — tested with fake DataFrame rows)
# ---------------------------------------------------------------------------


class TestRowHelpers:
    """Tests for _fnum / _disagg_qps / _agg_disagg_row / _pd_ratio_row / _summary_to_rows."""

    def _row(self, **kw):
        import pandas as pd

        return pd.Series(kw)

    def test_fnum_coerces(self):
        assert _fnum("1.5") == 1.5
        assert _fnum(2) == 2.0

    def test_fnum_none_for_bad(self):
        assert _fnum("abc") is None
        assert _fnum(None) is None
        assert _fnum(float("nan")) is None

    def test_disagg_qps_prefill(self):
        r = self._row(concurrency=10.0, ttft=5.0)
        assert _disagg_qps(r, 100, "disagg_prefill") == 10.0 / 5.0 * 1000.0

    def test_disagg_qps_decode(self):
        r = self._row(concurrency=10.0, tpot=2.0)
        assert _disagg_qps(r, 50, "disagg_decode") == 10.0 / (2.0 * 50) * 1000.0

    def test_disagg_qps_none_when_no_concurrency(self):
        r = self._row(concurrency=None)
        assert _disagg_qps(r, 100, "disagg_prefill") is None

    def test_disagg_qps_none_decode_no_output_length(self):
        r = self._row(concurrency=10.0, tpot=2.0)
        assert _disagg_qps(r, None, "disagg_decode") is None

    def test_disagg_qps_none_prefill_missing_ttft(self):
        """Branch 540->542: mode is disagg_prefill but ttft is None/0 → falls
        through to the decode check (mismatched mode) and returns None.
        """
        r = self._row(concurrency=10.0, ttft=None)
        assert _disagg_qps(r, 100, "disagg_prefill") is None
        # ttft=0 also exercises the falsy branch.
        r_zero = self._row(concurrency=10.0, ttft=0.0)
        assert _disagg_qps(r_zero, 100, "disagg_prefill") is None

    def test_agg_disagg_row_aggregation(self):
        r = self._row(
            parallel="TP=1", concurrency=4, num_devices=1, batch_size=8, **{"token/s": 100.0, "ttft": 10.0, "tpot": 5.0}
        )
        args = SimpleNamespace(output_length=100)
        row = _agg_disagg_row(r, "TEST_DEVICE", "aggregation", args)
        assert row["summary"]["throughput_token_s"] == 100.0
        assert row["config"]["device"] == "TEST_DEVICE"
        assert row["summary"]["mode"] == "aggregation"

    def test_agg_disagg_row_disagg_prefill(self):
        """A disagg row with ttft set is labeled disagg_prefill."""
        r = self._row(
            parallel="TP=1",
            concurrency=4,
            num_devices=1,
            batch_size=8,
            **{"token/s": 100.0, "ttft": 10.0, "tpot": None},
        )
        args = SimpleNamespace(output_length=100)
        row = _agg_disagg_row(r, "TEST_DEVICE", "disagg_prefill", args)
        assert row["summary"]["mode"] == "disagg_prefill"

    def test_agg_disagg_row_disagg_decode(self):
        """A disagg row without ttft is labeled disagg_decode."""
        r = self._row(
            parallel="TP=1", concurrency=4, num_devices=1, batch_size=8, **{"token/s": 100.0, "ttft": None, "tpot": 5.0}
        )
        args = SimpleNamespace(output_length=100)
        row = _agg_disagg_row(r, "TEST_DEVICE", "disagg_prefill", args)
        assert row["summary"]["mode"] == "disagg_decode"

    def test_pd_ratio_row(self):
        r = self._row(
            balanced_qps=50.0,
            pd_ratio=1.5,
            p_qps=30.0,
            d_qps=20.0,
            ttft_p=10.0,
            tpot_d=5.0,
            parallel_p="TP=1",
            parallel_d="TP=2",
            num_devices_p=1,
            num_devices_d=2,
            batch_size_p=8,
            batch_size_d=4,
            concurrency_p=4,
            concurrency_d=2,
        )
        row = _pd_ratio_row(r, "TEST_DEVICE")
        assert row["summary"]["balanced_qps"] == 50.0
        assert row["summary"]["pd_ratio"] == 1.5
        assert row["config"]["parallel_p"] == "TP=1"
        assert row["summary"]["mode"] == "pd_ratio"

    def test_summary_to_rows_empty_df(self):
        summary = MagicMock()
        summary.get_summary_df.return_value = None
        assert _summary_to_rows(summary, "d", "aggregation", SimpleNamespace()) == []

    def test_summary_to_rows_pd_ratio(self):
        import pandas as pd

        summary = MagicMock()
        summary.get_summary_df.return_value = pd.DataFrame(
            [{"balanced_qps": 50.0, "pd_ratio": 1.0, "p_qps": 30.0, "d_qps": 20.0, "ttft_p": 10.0, "tpot_d": 5.0}]
        )
        rows = _summary_to_rows(summary, "TEST_DEVICE", "pd_ratio", SimpleNamespace())
        assert len(rows) == 1
        assert rows[0]["summary"]["mode"] == "pd_ratio"

    def test_summary_to_rows_aggregation(self):
        import pandas as pd

        summary = MagicMock()
        summary.get_summary_df.return_value = pd.DataFrame(
            [
                {
                    "parallel": "TP=1",
                    "concurrency": 4,
                    "num_devices": 1,
                    "batch_size": 8,
                    "token/s": 100.0,
                    "ttft": 10.0,
                    "tpot": 5.0,
                }
            ]
        )
        rows = _summary_to_rows(summary, "TEST_DEVICE", "aggregation", SimpleNamespace(output_length=100))
        assert len(rows) == 1
        assert rows[0]["config"]["parallel"] == "TP=1"


# ---------------------------------------------------------------------------
# execute orchestration
# ---------------------------------------------------------------------------


class TestExecute:
    def test_single_case_real_sweep(self):
        """execute() with one case runs the real sweep (covers orchestration)."""
        records, skipped = execute(dict(_TINY_THROUGHPUT_PARAMS), form_schema_version="1.0.0")
        assert len(records) >= 1
        assert skipped == []
        assert records[0]["case_hash"] is not None
        assert records[0]["case_log"]  # captured CLI output

    def test_cached_case_skipped(self):
        """Run once to capture the case_hash, then re-run cached -> skipped."""
        with patch("runners.throughput_optimizer._run_throughput_sweep") as mock_sweep:
            mock_sweep.return_value = [{"config": {}, "summary": {}, "tables": {}}]
            recs1, _ = execute(dict(_TINY_THROUGHPUT_PARAMS), form_schema_version="1.0.0")
            ch = recs1[0]["case_hash"]
            mock_sweep.reset_mock()
            recs2, skipped = execute(dict(_TINY_THROUGHPUT_PARAMS), cached_hashes={ch}, form_schema_version="1.0.0")
        assert recs2 == []
        assert skipped == [ch]
        mock_sweep.assert_not_called()

    def test_pd_ratio_validation_raises(self):
        """PD ratio enabled without the required device counts raises ValueError."""
        with pytest.raises(ValueError, match="prefill-devices-per-instance"):
            execute(
                {**_TINY_THROUGHPUT_PARAMS, "enable_optimize_prefill_decode_ratio": True},
                form_schema_version="1.0.0",
            )

    def test_pd_ratio_valid_proceeds(self):
        """PD ratio enabled WITH the required counts does not raise at validation."""
        with patch("runners.throughput_optimizer._run_throughput_sweep") as mock_sweep:
            mock_sweep.return_value = [{"config": {}, "summary": {}, "tables": {}}]
            # Provide the required PD device counts so validation passes.
            records, _ = execute(
                {
                    **_TINY_THROUGHPUT_PARAMS,
                    "enable_optimize_prefill_decode_ratio": True,
                    "prefill_devices_per_instance": 1,
                    "decode_devices_per_instance": 1,
                },
                form_schema_version="1.0.0",
            )
        assert len(records) >= 1

    def test_failing_sweep_recorded_loop_continues(self):
        """A case whose sweep raises is recorded as failed; loop continues."""
        with patch("runners.throughput_optimizer._run_throughput_sweep") as mock_sweep:
            mock_sweep.side_effect = [
                RuntimeError("case 0 boom"),
                [{"config": {}, "summary": {}, "tables": {}}],
            ]
            records, _ = execute(
                {**_TINY_THROUGHPUT_PARAMS, "tpot_limits": [10.0, 20.0]},  # 2 cases
                form_schema_version="1.0.0",
            )
        assert len(records) == 2
        failed = [r for r in records if "error" in r.get("summary", {})]
        assert len(failed) == 1
        assert "case 0 boom" in failed[0]["summary"]["error"]

    def test_chrome_trace_path_synthesized(self):
        """When chrome_trace=True, the trace path is synthesized from job_id and case_hash."""
        from pathlib import Path

        with patch("runners.throughput_optimizer._run_throughput_sweep") as mock_sweep:
            mock_sweep.return_value = [{"config": {}, "summary": {}, "tables": {}}]
            with patch("services.trace_store.legacy_hash_path", return_value=Path("/trace/path.json")) as mock_path:
                records, _ = execute(
                    {**_TINY_THROUGHPUT_PARAMS, "chrome_trace": True},
                    job_id="job-123",
                    form_schema_version="1.0.0",
                )
        # Verify legacy_hash_path was called
        assert mock_path.called
        # The call should be with (job_id, case_hash)
        call_args = mock_path.call_args
        assert call_args[0][0] == "job-123"

    def test_max_search_combinations_warning(self, capsys):
        """When total_combinations > max_search_combinations, a warning is printed."""
        # Call _run_throughput_sweep directly to trigger the warning
        from runners.throughput_optimizer import _run_throughput_sweep

        params = {
            **_TINY_THROUGHPUT_PARAMS,
            "tp_sizes": [1, 2, 4],  # 3 options
            "ep_sizes": [1, 2],  # 2 options
            "max_search_combinations": 2,  # But allow only 2
        }
        # Mock the ParallelRunner to avoid actual execution
        with patch("serving_cast.parallel_runner.ParallelRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.run.return_value = []
            mock_runner.return_value = mock_instance
            _run_throughput_sweep(params)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "Large number of parallel search combinations" in captured.err


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TestThroughputOptimizerAdapter:
    def test_run_delegates_to_subprocess(self):
        adapter = ThroughputOptimizerAdapter()
        with patch("runners._subprocess.run_module_subprocess", return_value=([], [])) as mock_sub:
            records, _skipped = adapter.run(
                {"model_id": "m"},
                job_id="j1",
                cancel_flag=lambda: False,
                cached_hashes={"h1"},
                form_schema_version="1.0.0",
            )
        assert records == []
        assert mock_sub.call_args.args[0] == "throughput_optimizer"
        assert mock_sub.call_args.kwargs["job_id"] == "j1"
        assert mock_sub.call_args.args[1] == {"model_id": "m"}
        assert mock_sub.call_args.kwargs["cached_hashes"] == {"h1"}
        assert mock_sub.call_args.kwargs["form_schema_version"] == "1.0.0"

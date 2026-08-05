"""Real unit tests for runners/text_generate.py.

Covers the pure helpers (``_metrics_to_envelope``, ``_as_list``,
``_parse_int_list``, ``_expand_cases``), the adapter ``run()`` (subprocess
spawner — mocked), and real analytic inference via ``_build_user_input`` /
``_run_one_case`` / ``execute`` on TEST_DEVICE (the pattern from
tests/regression/serving_cast & tensor_cast — analytic modelling, no weight
download). Per tests/SKILL.md — real imports, fixture-scoped mocks only.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from runners.text_generate import (
    TextGenerateRunnerAdapter,
    _as_list,
    _build_user_input,
    _expand_cases,
    _metrics_to_envelope,
    _parse_int_list,
    _run_one_case,
    execute,
)


class TestAsList:
    """Tests for _as_list."""

    def test_none(self):
        assert _as_list(None) == []

    def test_list_filters_none(self):
        assert _as_list([1, None, 2]) == [1, 2]

    def test_scalar(self):
        assert _as_list("x") == ["x"]


class TestParseIntList:
    """Tests for _parse_int_list."""

    def test_none(self):
        assert _parse_int_list(None) == []

    def test_list(self):
        assert _parse_int_list(["1", "2"]) == [1, 2]

    def test_int(self):
        assert _parse_int_list(5) == [5]

    def test_string(self):
        assert _parse_int_list("1, 2 3") == [1, 2, 3]

    def test_list_with_non_int_raises(self):
        """Line 78-79: a list containing non-integer values raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="invalid integer list"):
            _parse_int_list(["1", "abc", "3"])

    def test_string_with_non_int_raises(self):
        """Line 85-86: a string containing non-integer values raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="invalid integer list"):
            _parse_int_list("1, abc 3")


class TestExpandCases:
    """Tests for _expand_cases."""

    def test_single_case_no_multi_fields(self):
        cases = _expand_cases({"model_id": "m"})
        assert len(cases) == 1
        assert cases[0]["model_id"] == "m"

    def test_cartesian_product_two_fields(self):
        cases = _expand_cases({"device": ["A", "B"], "tp_size": [1, 2]})
        # 2 devices × 2 tp_sizes = 4 cases.
        assert len(cases) == 4

    def test_single_value_field_collapses(self):
        """A single-value multi-case field collapses to one case."""
        cases = _expand_cases({"device": "A"})
        assert len(cases) == 1

    def test_too_many_cases_raises(self):
        """Line 172: cartesian product exceeding max_search_combinations raises ValueError."""
        import pytest

        # 11 devices × 11 num_queries × 1 tp_size = 121 > max_search_combinations (100)
        params = {
            "device": [f"d{i}" for i in range(11)],
            "num_queries": list(range(1, 12)),  # 11 values
            "tp_size": 1,
            "max_search_combinations": 100,
        }
        with pytest.raises(ValueError, match="too many cases.*> 100"):
            _expand_cases(params)

    def test_no_limit_without_max_search_combinations(self):
        """Without max_search_combinations, no limit is enforced."""
        # 11 devices × 11 num_queries × 1 tp_size = 121 cases — should succeed
        params = {
            "device": [f"d{i}" for i in range(11)],
            "num_queries": list(range(1, 12)),
            "tp_size": 1,
        }
        cases = _expand_cases(params)
        assert len(cases) == 121

    def test_limit_not_exceeded(self):
        """When max_search_combinations is set and NOT exceeded, expand succeeds.

        Covers the 176->180 partial branch: limit is not None, for-loop completes
        without raising ValueError.
        """
        params = {
            "device": ["d1", "d2"],
            "num_queries": [1, 2],
            "tp_size": 1,
            "max_search_combinations": 100,  # 2 × 2 × 1 = 4 cases < 100
        }
        cases = _expand_cases(params)
        assert len(cases) == 4


class TestMetricsToEnvelope:
    """Tests for _metrics_to_envelope."""

    def test_serializes_basic_fields(self):
        m = SimpleNamespace(
            batch_size=32,
            run_time_s=1.5,
            execution_time_s={"default": 1.0},
            total_device_memory_gb=80.0,
            model_weight_size_gb=40.0,
            peak_memory_usage_gb=60.0,
            kv_cache_size_gb=10.0,
            kv_cache_per_token_gb=0.001,
            model_activation_size_gb=5.0,
            reserved_memory_gb=4.0,
            device_memory_available_gb=76.0,
            breakdowns={},
            perf_model_name="default",
        )
        env = _metrics_to_envelope(m)
        assert env["batch_size"] == 32
        assert env["run_time_s"] == 1.5
        assert env["memory_gb"]["total_device"] == 80.0
        assert env["perf_model_name"] == "default"

    def test_breakdowns_percent_computed(self):
        m = SimpleNamespace(
            breakdowns={"layer_type": {"matmul": 60, "attention": 40}},
        )
        env = _metrics_to_envelope(m)
        assert env["breakdowns_percent"]["layer_type"]["matmul"] == 60.0
        assert env["breakdowns_percent"]["layer_type"]["attention"] == 40.0

    def test_breakdowns_zero_total_skipped(self):
        m = SimpleNamespace(breakdowns={"zeros": {"a": 0, "b": 0}})
        env = _metrics_to_envelope(m)
        assert "zeros" not in env["breakdowns_percent"]

    def test_breakdowns_non_numeric_skipped(self):
        m = SimpleNamespace(breakdowns={"bad": {"a": "not a number"}})
        env = _metrics_to_envelope(m)
        assert "bad" not in env["breakdowns_percent"]

    def test_none_metrics_uses_defaults(self):
        """A None/empty metrics object yields None fields (getattr fallback)."""
        env = _metrics_to_envelope(SimpleNamespace())
        assert env["batch_size"] is None
        assert env["breakdowns_raw"] == {}


class TestTextGenerateRunnerAdapter:
    """Tests for the TextGenerateRunnerAdapter.run spawner."""

    def test_run_delegates_to_subprocess(self):
        """run() calls run_module_subprocess with module_id='text_generate'."""
        adapter = TextGenerateRunnerAdapter()
        with patch("runners._subprocess.run_module_subprocess", return_value=([], [])) as mock_sub:
            records, skipped = adapter.run(
                {"model": "gpt2"},
                job_id="j1",
                on_progress=lambda *a: None,
                cancel_flag=lambda: False,
                cached_hashes={"h"},
                form_schema_version="1.0.0",
            )
        assert records == []
        assert skipped == []
        mock_sub.assert_called_once()
        call = mock_sub.call_args
        assert call.args[0] == "text_generate"
        assert call.args[1] == {"model": "gpt2"}
        assert call.kwargs["job_id"] == "j1"
        assert call.kwargs["cached_hashes"] == {"h"}
        assert call.kwargs["form_schema_version"] == "1.0.0"

    def test_run_minimal_kwargs(self):
        """run() works with only the required job_id."""
        adapter = TextGenerateRunnerAdapter()
        with patch("runners._subprocess.run_module_subprocess", return_value=([MagicMock()], [])):
            records, _ = adapter.run({"model": "gpt2"}, job_id="j1")
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Real analytic-inference tests (the pattern from tests/regression/serving_cast
# & tensor_cast): build ModelRunner on TEST_DEVICE with a remote model_id so it
# models analytically — no weight download, no local-path security check.
# ---------------------------------------------------------------------------

# A tiny, analytically-modeled request (TEST_DEVICE + remote id). ~7s per case.
# quantize_* are set explicitly because execute() -> _expand_cases injects an
# explicit None for unset multi-case fields, overriding UserInputConfig defaults.
_TINY_TEXT_PARAMS = {
    "model_id": "Qwen/Qwen3-32B",
    "device": "TEST_DEVICE",
    "num_devices": 1,
    "query_length": 8,
    "num_queries": 1,
    "do_compile": False,
    "quantize_linear_action": "DISABLED",
    "quantize_attention_action": "DISABLED",
    "tp_size": 1,
}


class TestBuildUserInput:
    """Tests for _build_user_input (config construction — no inference)."""

    def test_constructs_user_input_config(self):
        ui = _build_user_input(
            {
                "model_id": "Qwen/Qwen3-32B",
                "device": "TEST_DEVICE",
                "num_devices": 2,
                "query_length": 16,
                "do_compile": False,
            }
        )
        assert ui.world_size == 2  # num_devices -> world_size
        assert ui.query_len == 16  # query_length -> query_len
        assert ui.device == "TEST_DEVICE"

    def test_do_compile_name_map(self):
        ui = _build_user_input({"model_id": "m", "device": "TEST_DEVICE", "compile": True})
        assert ui.do_compile is True  # compile -> do_compile

    def test_unknown_keys_silently_dropped(self):
        ui = _build_user_input(
            {
                "model_id": "m",
                "device": "TEST_DEVICE",
                "export_empirical_metrics": "out.json",  # not a UIC field
            }
        )
        assert not hasattr(ui, "export_empirical_metrics")

    def test_quantize_string_coerced_to_enum(self):
        from tensor_cast.core.quantization.datatypes import (
            QuantizeAttentionAction,
            QuantizeLinearAction,
        )

        ui = _build_user_input(
            {
                "model_id": "m",
                "device": "TEST_DEVICE",
                "quantize_linear_action": "FP8",
                "quantize_attention_action": "FP8",
            }
        )
        assert isinstance(ui.quantize_linear_action, QuantizeLinearAction)
        assert isinstance(ui.quantize_attention_action, QuantizeAttentionAction)


class TestRunOneCaseReal:
    """Real analytic inference via _run_one_case (mirrors serving_cast tests)."""

    def test_run_one_case_produces_full_record(self):
        rec = _run_one_case(dict(_TINY_TEXT_PARAMS))
        assert rec["config"]["model_id"] == "Qwen/Qwen3-32B"
        assert rec["config"]["device"] == "TEST_DEVICE"
        assert "run_time_s" in rec["tables"]
        assert "memory_gb" in rec["tables"]
        assert "breakdowns_percent" in rec["tables"]
        # op_breakdown aggregated from the captured runtime.
        assert isinstance(rec["tables"]["op_breakdown"], list)
        assert rec["summary"]["run_time_s"] == rec["tables"]["run_time_s"]

    def test_op_breakdown_falls_back_when_aggregate_raises(self):
        """The defensive `except` around _extract_op_breakdown yields [].
        ModelRunner is mocked (a MagicMock metrics supports print_info); the
        captured runtime triggers the aggregate call, which is stubbed to raise.
        """
        with patch("tensor_cast.core.model_runner.ModelRunner") as mock_mr_cls:
            runner = MagicMock()

            def run_inference(*, generate_inputs_func=None, runtime_observer=None):
                # Capture a runtime with an event_list so the aggregate runs.
                rt_mock = MagicMock(event_list=[MagicMock()])
                rt_mock._aggregate_average_table_data.side_effect = RuntimeError("boom")
                runtime_observer(rt_mock)
                # A MagicMock metrics supports print_info + _metrics_to_envelope's getattr reads.
                return MagicMock(execution_time_s={"default": 1.0}, breakdowns={}, perf_model_name="default")

            runner.run_inference.side_effect = run_inference
            mock_mr_cls.return_value = runner
            rec = _run_one_case(dict(_TINY_TEXT_PARAMS))
        assert rec["tables"]["op_breakdown"] == []

    def test_op_breakdown_empty_when_runtime_is_none(self):
        """When runtime is None (not captured), op_breakdown is empty list."""
        from runners.text_generate import _extract_op_breakdown

        result = _extract_op_breakdown(None, "default", False, False)
        assert result == []

    def test_op_breakdown_with_dump_op_bound_results(self):
        """When dump_op_bound_results=True, bound_pct is included in the output."""
        with patch("tensor_cast.core.model_runner.ModelRunner") as mock_mr_cls:
            runner = MagicMock()

            def run_inference(*, generate_inputs_func=None, runtime_observer=None):
                # Create a mock runtime with proper aggregate behavior
                rt_mock = MagicMock()

                # Mock the aggregated data structure
                mock_key = MagicMock()
                mock_key.op_name = "test_op"
                mock_key.bound = "memory"
                mock_key.input_shapes = {"input": [1, 2, 3]}

                mock_data = MagicMock()
                mock_data.total_runtimes = {"default": 1.5}
                mock_data.count = 10
                mock_data.bound_components = {
                    "default": {
                        "memory": 50.0,
                        "communication": 20.0,
                        "mma": 20.0,
                        "gp": 10.0,
                    }
                }

                rt_mock._aggregate_average_table_data.return_value = {mock_key: mock_data}
                rt_mock._sort_average_table_items.return_value = [(mock_key, mock_data)]

                runtime_observer(rt_mock)
                return MagicMock(
                    execution_time_s={"default": 1.0},
                    breakdowns={},
                    perf_model_name="default",
                )

            runner.run_inference.side_effect = run_inference
            mock_mr_cls.return_value = runner

            params = dict(_TINY_TEXT_PARAMS)
            params["dump_op_bound_results"] = True
            rec = _run_one_case(params)

        assert len(rec["tables"]["op_breakdown"]) == 1
        op = rec["tables"]["op_breakdown"][0]
        assert op["name"] == "test_op"
        assert op["bound"] == "memory"
        assert "bound_pct" in op
        assert "memory" in op["bound_pct"]
        assert "comm" in op["bound_pct"]
        assert "mma" in op["bound_pct"]
        assert "gp" in op["bound_pct"]


class TestExecuteReal:
    """Real-inference tests for execute() orchestration."""

    def test_single_case_runs_and_returns_record(self):
        records, skipped = execute(dict(_TINY_TEXT_PARAMS), form_schema_version="1.0.0")
        assert len(records) == 1
        assert skipped == []
        assert records[0]["case_hash"] is not None
        assert len(records[0]["case_hash"]) == 64
        assert records[0]["case_log"]  # captured CLI output

    def test_single_case_cached_is_skipped(self):
        """A single case whose hash is in cached_hashes is skipped, not run."""
        from runners._multicase import compute_case_hash
        from runners.text_generate import _expand_cases, resolve_model_id_path

        params = dict(_TINY_TEXT_PARAMS)
        resolved = {**params, "model_id": resolve_model_id_path(params["model_id"])}
        ch = compute_case_hash("text_generate", "1.0.0", _expand_cases(resolved)[0])
        records, skipped = execute(params, cached_hashes={ch}, form_schema_version="1.0.0")
        assert records == []
        assert skipped == [ch]

    def test_multi_case_runs_each_case(self):
        """Two tp_sizes fan out to 2 distinct cases; each runs (real inference)."""
        params = {**_TINY_TEXT_PARAMS, "tp_size": [1, 2]}
        records, skipped = execute(params, form_schema_version="1.0.0")
        assert len(records) == 2
        assert skipped == []
        # Distinct cases carry distinct hashes.
        assert records[0]["case_hash"] != records[1]["case_hash"]

    def test_multi_case_partial_cache_skips_some(self):
        """When one of two cases is cached, only the other runs."""
        from runners._multicase import compute_case_hash
        from runners.text_generate import _expand_cases, resolve_model_id_path

        params = {**_TINY_TEXT_PARAMS, "tp_size": [1, 2]}
        resolved = {**params, "model_id": resolve_model_id_path(params["model_id"])}
        cases = _expand_cases(resolved)
        # Cache the first case's hash.
        first_hash = compute_case_hash("text_generate", "1.0.0", cases[0])
        records, skipped = execute(params, cached_hashes={first_hash}, form_schema_version="1.0.0")
        assert len(records) == 1  # only the second ran
        assert skipped == [first_hash]

    def test_failing_case_recorded_loop_continues(self):
        """A case whose _run_one_case raises is recorded as failed; the loop
        continues. (The error-orchestration branch — _run_one_case is stubbed
        to raise only here, since engineering a real analytic failure is brittle.)
        """
        with patch("runners.text_generate._run_one_case") as mock_run:
            mock_run.side_effect = [
                RuntimeError("case 0 boom"),
                {"config": {}, "summary": {}, "tables": {}},
            ]
            records, _skipped = execute(
                {**_TINY_TEXT_PARAMS, "tp_size": [1, 2]},
                form_schema_version="1.0.0",
            )
        assert len(records) == 2
        failed = [r for r in records if "error" in r.get("summary", {})]
        assert len(failed) == 1
        assert "case 0 boom" in failed[0]["summary"]["error"]

    def test_chrome_trace_path_synthesized(self):
        """chrome_trace=True is replaced with a trace path inside execute."""
        from pathlib import Path as _Path

        captured = {}

        original_run = _run_one_case

        def spy(cp):
            captured["chrome_trace"] = cp.get("chrome_trace")
            return original_run(cp)

        with (
            patch("runners.text_generate._run_one_case", side_effect=spy),
            patch("services.trace_store.legacy_hash_path", return_value=_Path("/tmp/t.json")),
        ):
            execute(
                {**_TINY_TEXT_PARAMS, "chrome_trace": True},
                form_schema_version="1.0.0",
                job_id="job-1",
            )
        assert str(captured["chrome_trace"]).endswith("t.json")

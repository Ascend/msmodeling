"""Real unit tests for runners/_multicase.py.

Pure-Python case-expansion + aggregation helpers (no torch/tensor_cast). Per
tests/SKILL.md — real imports, fixture-scoped mocks only.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from runners._multicase import (
    aggregate_runtime_events,
    as_list,
    compute_case_hash,
    expand_cases,
    parse_float_list,
    parse_int_list,
    resolve_model_id_path,
    run_cases,
)


class TestResolveModelIdPath:
    """Tests for resolve_model_id_path."""

    def test_absolute_path_unchanged(self):
        assert resolve_model_id_path("/abs/path/model") == "/abs/path/model"

    def test_remote_repo_id_unchanged(self):
        # A non-existent relative path that isn't a dir under repo root stays as-is.
        assert resolve_model_id_path("Org/Model-123") == "Org/Model-123"

    def test_none_unchanged(self):
        assert resolve_model_id_path(None) is None

    def test_existing_relative_dir_resolved_to_repo_root(self, tmp_path, monkeypatch):
        # Point _REPO_ROOT at tmp_path so the candidate is found.
        import runners._multicase as mc

        monkeypatch.setattr(mc, "_REPO_ROOT", tmp_path)
        (tmp_path / "assets" / "model").mkdir(parents=True)
        result = resolve_model_id_path("assets/model")
        assert result == str(tmp_path / "assets" / "model")

    def test_existing_local_dir_unchanged(self, tmp_path):
        # An existing dir (relative to cwd) is returned as-is.
        assert resolve_model_id_path(str(tmp_path)) == str(tmp_path)


class TestAsList:
    """Tests for as_list."""

    def test_none_returns_empty(self):
        assert as_list(None) == []

    def test_list_filters_none(self):
        assert as_list([1, None, 2, None]) == [1, 2]

    def test_scalar_wrapped(self):
        assert as_list("x") == ["x"]
        assert as_list(5) == [5]


class TestParseIntList:
    """Tests for parse_int_list."""

    def test_none_empty(self):
        assert parse_int_list(None) == []

    def test_list_of_ints(self):
        assert parse_int_list([1, 2, 3]) == [1, 2, 3]

    def test_single_int(self):
        assert parse_int_list(7) == [7]

    def test_string_comma_space(self):
        assert parse_int_list("1, 2 3") == [1, 2, 3]

    def test_string_mixed_separators(self):
        assert parse_int_list("4,5, 6") == [4, 5, 6]


class TestParseFloatList:
    """Tests for parse_float_list."""

    def test_none_empty(self):
        assert parse_float_list(None) == []

    def test_list_of_numbers(self):
        assert parse_float_list([1, 2.5]) == [1.0, 2.5]

    def test_single_int_to_float(self):
        assert parse_float_list(3) == [3.0]

    def test_single_float(self):
        assert parse_float_list(2.5) == [2.5]

    def test_string(self):
        assert parse_float_list("1.0, 2.5") == [1.0, 2.5]


class TestAggregateRuntimeEvents:
    """Tests for aggregate_runtime_events."""

    def _event(self, func, exec_time=None, perf_model="default"):
        ev = MagicMock()
        ev.op_invoke_info.func = func
        if exec_time is not None:
            result = MagicMock()
            result.execution_time_s = exec_time
            ev.perf_results = {perf_model: result}
        else:
            ev.perf_results = {}
        return ev

    def test_empty_list(self):
        assert aggregate_runtime_events([], "default") == []
        assert aggregate_runtime_events(None) == []

    def test_no_perf_model_returns_zero_totals(self):
        ev = self._event("op_a", exec_time=1.0)
        result = aggregate_runtime_events([ev], None)
        assert result[0]["perf_total"] == 0.0
        assert result[0]["call_times"] == 1
        assert result[0]["perf_model"] is None

    def test_aggregates_and_sorts_by_total(self):
        events = [
            self._event("op_a", exec_time=1.0),
            self._event("op_a", exec_time=3.0),
            self._event("op_b", exec_time=10.0),
        ]
        result = aggregate_runtime_events(events, "default")
        # op_b (10.0) sorts before op_a (4.0).
        assert result[0]["name"] == "op_b"
        assert result[1]["name"] == "op_a"
        assert result[1]["perf_total"] == 4.0
        assert result[1]["perf_avg"] == 2.0
        assert result[1]["call_times"] == 2

    def test_missing_perf_result_skipped(self):
        ev = MagicMock()
        ev.op_invoke_info.func = "op_x"
        ev.perf_results = {}  # no result for the perf_model
        result = aggregate_runtime_events([ev], "default")
        assert result[0]["perf_total"] == 0.0
        assert result[0]["call_times"] == 1


class TestComputeCaseHash:
    """Tests for compute_case_hash."""

    def test_none_when_no_version(self):
        assert compute_case_hash("m", None, {"a": 1}) is None
        assert compute_case_hash("m", "", {"a": 1}) is None

    def test_returns_hash_with_version(self):
        h = compute_case_hash("m", "1.0.0", {"a": 1})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_deterministic(self):
        assert compute_case_hash("m", "1.0.0", {"a": 1}) == compute_case_hash("m", "1.0.0", {"a": 1})


class TestExpandCases:
    """Tests for expand_cases."""

    def test_single_field_multiple_values(self):
        cases = expand_cases({"a": [1, 2]}, {"a": as_list})
        assert len(cases) == 2
        assert {c["a"] for c in cases} == {1, 2}

    def test_expand_two_fields(self):
        cases = expand_cases(
            {"a": [1, 2], "b": [3, 4]},
            {"a": as_list, "b": as_list},
        )
        assert len(cases) == 4

    def test_cartesian_product(self):
        cases = expand_cases(
            {"a": [1, 2], "b": [3, 4], "fixed": "x"},
            {"a": as_list, "b": as_list},
        )
        assert len(cases) == 4
        # Each case carries 'fixed' unchanged.
        assert all(c["fixed"] == "x" for c in cases)
        combos = {(c["a"], c["b"]) for c in cases}
        assert combos == {(1, 3), (1, 4), (2, 3), (2, 4)}

    def test_empty_field_collapses_to_none(self):
        cases = expand_cases({"a": None}, {"a": parse_int_list})
        assert len(cases) == 1
        assert cases[0]["a"] is None

    def test_no_fields_single_case(self):
        cases = expand_cases({"a": 1}, {})
        assert len(cases) == 1
        assert cases[0] == {"a": 1}

    def test_too_many_cases_raises(self):
        """Line 142: cartesian product exceeding max_search_combinations raises ValueError."""
        import pytest

        # 11 × 11 = 121 > max_search_combinations (100)
        with pytest.raises(ValueError, match="too many cases.*> 100"):
            expand_cases(
                {"a": list(range(11)), "b": list(range(11)), "max_search_combinations": 100},
                {"a": as_list, "b": as_list},
            )

    def test_no_limit_without_max_search_combinations(self):
        """Without max_search_combinations, no limit is enforced."""
        # 11 × 11 = 121 cases — should succeed without limit
        cases = expand_cases(
            {"a": list(range(11)), "b": list(range(11))},
            {"a": as_list, "b": as_list},
        )
        assert len(cases) == 121

    def test_limit_with_empty_fields(self):
        """When limit is set but fields is empty, the for-loop doesn't execute.

        Covers the 151->155 partial branch: limit is not None, but keys is empty.
        """
        cases = expand_cases(
            {"max_search_combinations": 10},
            {},  # empty fields → keys is empty → for-loop skipped
        )
        assert len(cases) == 1  # one case with just the original params


class TestRunCases:
    """Tests for run_cases."""

    def test_single_case_runs_once(self):
        def run_one(cp):
            return {"config": cp, "summary": {}, "tables": {}}

        records, skipped = run_cases({"a": 1}, {}, run_one)
        assert len(records) == 1
        assert skipped == []
        assert records[0]["case_hash"] is None  # no case_hash_ctx

    def test_multiple_cases_all_run(self):
        def run_one(cp):
            return {"config": cp, "summary": {"ok": True}, "tables": {}}

        records, skipped = run_cases(
            {"a": [1, 2, 3]},
            {"a": as_list},
            run_one,
        )
        assert len(records) == 3
        assert skipped == []

    def test_failing_case_recorded_loop_continues(self):
        calls = []

        def run_one(cp):
            calls.append(cp["a"])
            if cp["a"] == 2:
                raise RuntimeError("boom on 2")
            return {"config": cp, "summary": {}, "tables": {}}

        records, _skipped = run_cases({"a": [1, 2, 3]}, {"a": as_list}, run_one)
        assert len(calls) == 3  # all cases attempted
        assert len(records) == 3
        # The failed case carries an error summary.
        failed = [r for r in records if "error" in r.get("summary", {})]
        assert len(failed) == 1
        assert "boom on 2" in failed[0]["summary"]["error"]

    def test_single_case_cached_skipped(self):
        """A single case whose hash is cached is skipped."""
        ctx = ("text_generate", "1.0.0")
        params = {"a": 1}
        cached_hash = compute_case_hash(ctx[0], ctx[1], params)

        def run_one(cp):
            raise AssertionError("should not run a cached case")

        records, skipped = run_cases(params, {}, run_one, cached_hashes={cached_hash}, case_hash_ctx=ctx)
        assert records == []
        assert skipped == [cached_hash]

    def test_multi_case_cached_skipped_and_run_mixed(self):
        """Some cases cached (skipped), others run."""
        ctx = ("text_generate", "1.0.0")
        run_one = MagicMock(side_effect=lambda cp: {"config": cp, "summary": {}, "tables": {}})
        # Cache the hash of case a=1 only.
        cached = {compute_case_hash(ctx[0], ctx[1], {"a": 1, "b": 3})}
        _records, skipped = run_cases(
            {"a": [1, 2], "b": [3, 4]},
            {"a": as_list, "b": as_list},
            run_one,
            cached_hashes=cached,
            case_hash_ctx=ctx,
        )
        # 4 cases total; case (1,3) cached → skipped. 3 run.
        assert len(skipped) == 1
        assert run_one.call_count == 3

    def test_case_hash_attached_to_records(self):
        ctx = ("text_generate", "1.0.0")

        def run_one(cp):
            return {"config": cp, "summary": {}, "tables": {}}

        records, _ = run_cases({"a": 1}, {}, run_one, case_hash_ctx=ctx)
        assert records[0]["case_hash"] is not None
        assert len(records[0]["case_hash"]) == 64

    def test_chrome_trace_path_synthesized(self):
        """When chrome_trace is True, the bool is replaced with a path."""
        ctx = ("text_generate", "1.0.0")
        captured = {}

        def run_one(cp):
            captured["chrome_trace"] = cp.get("chrome_trace")
            return {"config": cp, "summary": {}, "tables": {}}

        with patch("services.trace_store.legacy_hash_path", return_value=Path("/tmp/trace.json")):
            run_cases({"a": 1, "chrome_trace": True}, {}, run_one, case_hash_ctx=ctx, job_id="job-1")
        assert captured["chrome_trace"] == "/tmp/trace.json" or str(captured["chrome_trace"]).endswith("trace.json")

    def test_single_case_failure_recorded(self):
        """Lines 244-250: when a single case's run_one raises, the error is
        recorded (not propagated) — mirrors the multi-case error path.
        """
        ctx = ("text_generate", "1.0.0")

        def run_one(cp):
            raise RuntimeError("single case boom")

        records, skipped = run_cases({"a": 1}, {}, run_one, case_hash_ctx=ctx)
        assert skipped == []
        assert len(records) == 1
        rec = records[0]
        assert "error" in rec["summary"]
        assert "single case boom" in rec["summary"]["error"]
        assert rec["case_hash"] is not None
        # case_log should contain the traceback.
        assert "Traceback" in rec["case_log"]
        assert "single case boom" in rec["case_log"]

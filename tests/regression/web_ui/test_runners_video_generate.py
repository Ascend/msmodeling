"""Real unit tests for runners/video_generate.py.

video_generate's DiT inference needs a real Diffusers model directory (no
analytic mode like text), so ``_run_one_video_case`` mocks
``VideoGenerateRunner`` to return a fake Runtime and exercises the real
data-flow logic (config extraction, chrome-trace export, op-breakdown
aggregation, envelope assembly) — the same pattern as the regression diffusers
tests, which mock the HF/MS snapshot calls. The pure helpers
(``_runtime_to_envelope`` / ``_try_structured_table_rows``) run against mock
Runtime objects. Per tests/SKILL.md — real imports, fixture-scoped mocks only.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from runners.video_generate import (
    VideoGenerateRunnerAdapter,
    _run_one_video_case,
    _runtime_to_envelope,
    _try_structured_table_rows,
    execute,
)

# ---------------------------------------------------------------------------
# Fake Runtime objects (stand-ins for the DiT Runtime the real runner produces)
# ---------------------------------------------------------------------------


def _runtime(
    *, exec_times=None, breakdowns=None, table_rows=None, table_text="TABLE", event_list=None, structured_helper=None
):
    """A fake Runtime with the methods _runtime_to_envelope reads."""
    rt = MagicMock()
    rt.total_execution_time_s.return_value = exec_times if exec_times is not None else {"default": 1.0}
    rt.get_breakdowns.return_value = breakdowns if breakdowns is not None else {"layer": {"matmul": 60}}
    rt.table_averages.return_value = table_text
    rt.event_list = event_list if event_list is not None else []
    if structured_helper is not None:
        rt._aggregate_average_table_data = structured_helper
    else:
        # No structured helper -> _try_structured_table_rows returns [].
        rt._aggregate_average_table_data = None
    return rt


# ---------------------------------------------------------------------------
# _try_structured_table_rows
# ---------------------------------------------------------------------------


class TestTryStructuredTableRows:
    """Tests for _try_structured_table_rows."""

    def test_no_helper_returns_empty(self):
        rt = MagicMock()
        del rt._aggregate_average_table_data  # attribute absent
        assert _try_structured_table_rows(rt) == []

    def test_helper_not_callable_returns_empty(self):
        rt = MagicMock()
        rt._aggregate_average_table_data = "not callable"
        assert _try_structured_table_rows(rt) == []

    def test_helper_raises_returns_empty(self):
        rt = MagicMock()
        rt._aggregate_average_table_data = MagicMock(side_effect=RuntimeError("boom"))
        assert _try_structured_table_rows(rt) == []

    def test_flattens_aggregated_data_to_rows(self):
        from collections import namedtuple

        Key = namedtuple("Key", "op_name bound input_shapes")
        key = Key(op_name="matmul", bound="compute", input_shapes=(1, 8))
        data = SimpleNamespace(total_runtimes={"default": 4.0, "analytic": 2.0}, count=2)
        rt = MagicMock()
        rt._aggregate_average_table_data = MagicMock(return_value={key: data})
        rows = _try_structured_table_rows(rt)
        assert len(rows) == 2
        row = rows[0]
        assert row["op_name"] == "matmul"
        assert row["bound"] == "compute"
        assert row["call_count"] == 2
        assert row["avg"] == 2.0  # 4.0 / 2

    def test_zero_count_yields_none_avg(self):
        """count = getattr(data,'count',0) or 1 — so count is always >= 1
        (the `else None` is a dead guard); avg = total / count.
        """
        from collections import namedtuple

        Key = namedtuple("Key", "op_name bound input_shapes")
        key = Key(op_name="op", bound=None, input_shapes=None)
        data = SimpleNamespace(total_runtimes={"m": 2.0}, count=0)  # 0 -> coerced to 1
        rt = MagicMock()
        rt._aggregate_average_table_data = MagicMock(return_value={key: data})
        rows = _try_structured_table_rows(rt)
        assert rows[0]["avg"] == 2.0  # 2.0 / 1 (count 0 coerced to 1)
        assert rows[0]["call_count"] == 1


# ---------------------------------------------------------------------------
# _runtime_to_envelope
# ---------------------------------------------------------------------------


class TestRuntimeToEnvelope:
    """Tests for _runtime_to_envelope."""

    def test_assembles_envelope_from_runtime(self):
        rt = _runtime(
            exec_times={"default": 5.0},
            breakdowns={"layer": {"matmul": 60, "attn": 40}},
            structured_helper=dict,  # no rows -> falls back to table_averages text
        )
        env = _runtime_to_envelope(rt)
        assert env["execution_time_s"] == {"default": 5.0}
        assert env["breakdowns"]["layer"] == {"matmul": 60, "attn": 40}
        assert env["table_averages_text"] == "TABLE"

    def test_execution_time_failure_falls_back_to_empty(self):
        rt = MagicMock()
        rt.total_execution_time_s.side_effect = RuntimeError("no metrics")
        rt.get_breakdowns.return_value = {}
        rt._aggregate_average_table_data = None
        rt.table_averages.return_value = ""
        env = _runtime_to_envelope(rt)
        assert env["execution_time_s"] == {}

    def test_breakdowns_failure_falls_back_to_empty(self):
        rt = MagicMock()
        rt.total_execution_time_s.return_value = {}
        rt.get_breakdowns.side_effect = RuntimeError("no breakdowns")
        rt._aggregate_average_table_data = None
        rt.table_averages.return_value = ""
        env = _runtime_to_envelope(rt)
        assert env["breakdowns"] == {}

    def test_structured_rows_preferred_over_text(self):
        from collections import namedtuple

        Key = namedtuple("Key", "op_name bound input_shapes")
        key = Key(op_name="op", bound="b", input_shapes=None)
        data = SimpleNamespace(total_runtimes={"m": 1.0}, count=1)
        rt = MagicMock()
        rt.total_execution_time_s.return_value = {}
        rt.get_breakdowns.return_value = {}
        rt._aggregate_average_table_data = MagicMock(return_value={key: data})
        env = _runtime_to_envelope(rt)
        assert len(env["table_rows"]) == 1
        assert "table_averages_text" not in env  # structured rows took precedence

    def test_table_averages_failure_falls_back_to_empty_string(self):
        rt = MagicMock()
        rt.total_execution_time_s.return_value = {}
        rt.get_breakdowns.return_value = {}
        rt._aggregate_average_table_data = None  # no structured rows
        rt.table_averages.side_effect = RuntimeError("no table")
        env = _runtime_to_envelope(rt)
        assert env["table_averages_text"] == ""


# ---------------------------------------------------------------------------
# _run_one_video_case (VideoGenerateRunner mocked)
# ---------------------------------------------------------------------------


class TestRunOneVideoCase:
    """Tests for _run_one_video_case with a mocked VideoGenerateRunner."""

    def _patch_runner(self, runtime):
        return patch("runners._video_generate_runner.VideoGenerateRunner")

    def test_assembles_record_from_runtime(self):
        rt = _runtime(
            exec_times={"default": 3.0},
            breakdowns={"layer": {"matmul": 100}},
            event_list=[],  # no events -> op_breakdown aggregation sees []
        )
        with self._patch_runner(rt) as mock_cls:
            mock_cls.return_value.run_inference.return_value = rt
            rec = _run_one_video_case(
                {
                    "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
                    "device": "TEST_DEVICE",
                    "world_size": 2,
                    "quantize_linear_action": "W8A8_DYNAMIC",
                    "ulysses_size": 4,
                }
            )
        # Assert ALL 5 config fields the production code populates.
        assert rec["config"]["device"] == "TEST_DEVICE"
        assert rec["config"]["model_id"] == "Wan-AI/Wan2.1-T2V-1.3B"
        assert rec["config"]["world_size"] == 2
        assert rec["config"]["quantize_linear_action"] == "W8A8_DYNAMIC"
        assert rec["config"]["ulysses_size"] == 4
        assert rec["summary"]["execution_time_s"] == {"default": 3.0}
        assert rec["tables"]["execution_time_s"] == {"default": 3.0}
        assert rec["tables"]["op_breakdown"] == []
        # VideoGenerateRunner was constructed with the right kwargs.
        cls_kwargs = mock_cls.call_args.kwargs
        assert cls_kwargs["device"] == "TEST_DEVICE"
        assert cls_kwargs["model_id"] == "Wan-AI/Wan2.1-T2V-1.3B"
        assert cls_kwargs["world_size"] == 2
        assert cls_kwargs["ulysses_size"] == 4

    def test_invalid_quantize_defaults_to_w8a8_dynamic(self):
        """An unrecognized quantize string falls back to W8A8_DYNAMIC."""
        rt = _runtime()
        with self._patch_runner(rt) as mock_cls:
            mock_cls.return_value.run_inference.return_value = rt
            _run_one_video_case(
                {
                    "model_id": "m",
                    "device": "TEST_DEVICE",
                    "quantize_linear_action": "not-a-real-quant",
                }
            )
            # The runner was constructed with the default quant enum.
            kwargs = mock_cls.call_args.kwargs
            from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

            assert kwargs["quantize_linear_action"] == QuantizeLinearAction.W8A8_DYNAMIC

    def test_chrome_trace_exported_and_failure_swallowed(self):
        """chrome_trace set -> export_chrome_trace called; its failure swallowed."""
        rt = _runtime()
        rt.export_chrome_trace.side_effect = RuntimeError("export failed")
        with self._patch_runner(rt) as mock_cls:
            mock_cls.return_value.run_inference.return_value = rt
            _run_one_video_case(
                {
                    "model_id": "m",
                    "device": "TEST_DEVICE",
                    "chrome_trace": "/tmp/trace.json",
                }
            )
            rt.export_chrome_trace.assert_called_once_with("/tmp/trace.json")

    def test_op_breakdown_aggregated_from_event_list(self):
        """When the runtime carries perf events, op_breakdown is aggregated."""
        ev = MagicMock()
        ev.op_invoke_info.func = "op_x"
        ev.perf_results = {"default": MagicMock(execution_time_s=2.0)}
        rt = _runtime(event_list=[ev, ev])
        with self._patch_runner(rt) as mock_cls:
            mock_cls.return_value.run_inference.return_value = rt
            rec = _run_one_video_case({"model_id": "m", "device": "TEST_DEVICE"})
        assert len(rec["tables"]["op_breakdown"]) == 1  # one distinct op
        assert rec["tables"]["op_breakdown"][0]["call_times"] == 2

    def test_table_averages_print_failure_swallowed(self):
        """runtime.table_averages printing failure is swallowed (best-effort)."""
        rt = _runtime()
        rt.table_averages.side_effect = RuntimeError("print failed")
        with self._patch_runner(rt) as mock_cls:
            mock_cls.return_value.run_inference.return_value = rt
            rec = _run_one_video_case({"model_id": "m", "device": "TEST_DEVICE"})
        # Still produces a record (the envelope falls back to empty table text).
        assert "tables" in rec

    def test_empty_string_numeric_fields_guarded(self):
        """Empty-string number fields (cleared in the form) are guarded to defaults."""
        rt = _runtime()
        with self._patch_runner(rt) as mock_cls:
            mock_cls.return_value.run_inference.return_value = rt
            _run_one_video_case(
                {
                    "model_id": "m",
                    "device": "TEST_DEVICE",
                    "batch_size": "",
                    "seq_len": "",
                    "height": "",
                    "width": "",
                    "frame_num": "",
                    "sample_step": "",
                    "mxfp4_group_size": "",
                }
            )
            # run_inference called with coerced ints (no crash from int("")).
            mock_cls.return_value.run_inference.assert_called_once()

    def test_relative_model_id_resolved_to_repo_root_dir(self, tmp_path, monkeypatch):
        """A repo-relative model_id that is an existing dir is resolved against
        _REPO_ROOT (covers the candidate.is_dir() True branch).
        """
        import runners.video_generate as vg

        fake_repo = tmp_path / "repo"
        (fake_repo / "assets" / "vid_model").mkdir(parents=True)
        monkeypatch.setattr(vg, "_REPO_ROOT", fake_repo)
        rt = _runtime()
        with self._patch_runner(rt) as mock_cls:
            mock_cls.return_value.run_inference.return_value = rt
            _run_one_video_case({"model_id": "assets/vid_model", "device": "TEST_DEVICE"})
            # model_id resolved to the absolute repo-rooted path.
            assert mock_cls.call_args.kwargs["model_id"] == str(fake_repo / "assets" / "vid_model")

    def test_absolute_model_id_left_unchanged(self, tmp_path):
        """An absolute model_id skips the repo-root resolution entirely
        (covers the outer-if False branch).
        """
        abs_model = tmp_path / "abs_model"  # an existing absolute dir
        abs_model.mkdir()
        rt = _runtime()
        with self._patch_runner(rt) as mock_cls:
            mock_cls.return_value.run_inference.return_value = rt
            _run_one_video_case({"model_id": str(abs_model), "device": "TEST_DEVICE"})
            assert mock_cls.call_args.kwargs["model_id"] == str(abs_model)

    def test_op_breakdown_falls_back_when_aggregate_raises(self):
        """The defensive `except` around op_breakdown aggregation yields []."""
        ev = MagicMock()
        ev.op_invoke_info.func = "op"
        ev.perf_results = {"default": MagicMock(execution_time_s=1.0)}
        rt = _runtime(event_list=[ev])
        with (
            self._patch_runner(rt) as mock_cls,
            patch("runners.video_generate.aggregate_runtime_events", side_effect=RuntimeError("boom")),
        ):
            mock_cls.return_value.run_inference.return_value = rt
            rec = _run_one_video_case({"model_id": "m", "device": "TEST_DEVICE"})
        assert rec["tables"]["op_breakdown"] == []


# ---------------------------------------------------------------------------
# execute (delegates to run_cases — orchestration already covered by
# test_runners_multicase; here we confirm the wiring + cached-skip)
# ---------------------------------------------------------------------------


class TestExecute:
    """Tests for execute() wiring."""

    def test_single_case_runs_and_returns_record(self):
        with patch("runners.video_generate._run_one_video_case") as mock_run:
            mock_run.return_value = {"config": {}, "summary": {}, "tables": {}}
            records, _skipped = execute(
                {"model_id": "m", "device": "TEST_DEVICE"},
                form_schema_version="1.0.0",
            )
        assert len(records) == 1
        assert records[0]["case_hash"] is not None
        mock_run.assert_called_once()

    def test_multi_case_runs_each(self):
        with patch("runners.video_generate._run_one_video_case") as mock_run:
            mock_run.return_value = {"config": {}, "summary": {}, "tables": {}}
            records, _skipped = execute(
                {"model_id": "m", "device": "TEST_DEVICE", "ulysses_size": [1, 2]},
                form_schema_version="1.0.0",
            )
        assert len(records) == 2
        assert mock_run.call_count == 2

    def test_cached_case_skipped(self):
        # Run once to capture the exact case_hash execute() computes (the
        # expanded case includes None for unset multi-case fields), then re-run
        # with that hash cached -> the case is skipped.
        with patch("runners.video_generate._run_one_video_case") as mock_run:
            mock_run.return_value = {"config": {}, "summary": {}, "tables": {}}
            recs1, _ = execute({"model_id": "m", "device": "TEST_DEVICE"}, form_schema_version="1.0.0")
            ch = recs1[0]["case_hash"]
            mock_run.reset_mock()
            recs2, skipped = execute(
                {"model_id": "m", "device": "TEST_DEVICE"},
                cached_hashes={ch},
                form_schema_version="1.0.0",
            )
        assert recs2 == []
        assert skipped == [ch]
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# VideoGenerateRunner class (not mocked — real constructor / run_inference)
# ---------------------------------------------------------------------------


class TestVideoGenerateRunnerClass:
    """Tests for the real VideoGenerateRunner class (not mocked)."""

    def test_constructor_stores_params(self):
        """Constructor stores all parameters as attributes."""
        from runners._video_generate_runner import VideoGenerateRunner

        from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

        runner = VideoGenerateRunner(
            device="TEST_DEVICE",
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            dtype="float16",
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            mxfp4_group_size=64,
            world_size=2,
            ulysses_size=1,
        )
        assert runner.device == "TEST_DEVICE"
        assert runner.model_id == "Wan-AI/Wan2.1-T2V-1.3B"
        assert runner.dtype == "float16"
        assert runner.quantize_linear_action == QuantizeLinearAction.W8A8_DYNAMIC
        assert runner.mxfp4_group_size == 64
        assert runner.world_size == 2
        assert runner.ulysses_size == 1

    def test_constructor_defaults(self):
        """Constructor applies defaults for optional params."""
        from runners._video_generate_runner import VideoGenerateRunner

        from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

        runner = VideoGenerateRunner(device="TEST_DEVICE", model_id="m")
        assert runner.dtype == "float16"
        assert runner.quantize_linear_action == QuantizeLinearAction.W8A8_DYNAMIC
        assert runner.mxfp4_group_size == 32
        assert runner.world_size == 1
        assert runner.ulysses_size == 1

    def test_run_inference_delegates_to_cli(self):
        """run_inference delegates to cli.inference.video_generate.run_inference with
        the constructed params (the Runtime-recovery path is covered end-to-end
        in the worker; here we assert the delegation contract).
        """
        from runners._video_generate_runner import VideoGenerateRunner

        with patch("cli.inference.video_generate.run_inference", return_value=None) as mock_cli:
            runner = VideoGenerateRunner(device="TEST_DEVICE", model_id="m")
            runner.run_inference(
                batch_size=2,
                seq_len=256,
                height=832,
                width=400,
                frame_num=81,
                sample_step=50,
                use_cfg=True,
                cfg_parallel=False,
                dit_cache=False,
                cache_step_range=None,
                cache_step_interval=1,
                cache_block_range=None,
            )
        mock_cli.assert_called_once_with(
            device="TEST_DEVICE",
            model_id="m",
            batch_size=2,
            seq_len=256,
            height=832,
            width=400,
            frame_num=81,
            sample_step=50,
            dtype="float16",
            quantize_linear_action=runner.quantize_linear_action,
            mxfp4_group_size=32,
            world_size=1,
            ulysses_size=1,
            use_cfg=True,
            cfg_parallel=False,
            dit_cache=False,
            cache_step_range=None,
            cache_step_interval=1,
            cache_block_range=None,
        )

    def test_run_inference_returns_captured_runtime(self):
        """When the CLI creates a Runtime inside its ``with`` block, the stub
        captures and returns it — even though cli.run_inference returns None.
        """
        from runners._video_generate_runner import VideoGenerateRunner

        import tensor_cast.runtime as _rt_mod

        MagicMock(name="captured_runtime")

        # Drive the CLI's real ``with Runtime(...)`` machinery so the stub's
        # __enter__ patch actually fires and stashes an instance. We fake the
        # instance by making Runtime() construct our mock via __new__-free patch:
        with (
            patch("cli.inference.video_generate.run_inference") as mock_cli,
            patch.object(_rt_mod.Runtime, "__enter__", lambda self: self) as _,
            patch.object(_rt_mod.Runtime, "__exit__", lambda self, *a: False),
        ):
            # Make the CLI body create a Runtime whose __enter__ stashes it.
            def _fake_cli(**kwargs):
                rt = _rt_mod.Runtime.__new__(_rt_mod.Runtime)  # pylint: disable=no-value-for-parameter
                with rt:  # triggers the patched __enter__ -> stub captures `rt`
                    pass

            mock_cli.side_effect = _fake_cli

            runner = VideoGenerateRunner(device="TEST_DEVICE", model_id="m")
            result = runner.run_inference(batch_size=1, seq_len=128)
        assert isinstance(result, _rt_mod.Runtime)

    def test_run_inference_returns_none_when_cli_creates_no_runtime(self):
        """If the CLI exits early without a Runtime (e.g. raises before the with),
        the stub returns None rather than crashing.
        """
        from runners._video_generate_runner import VideoGenerateRunner

        with patch("cli.inference.video_generate.run_inference", return_value=None):
            runner = VideoGenerateRunner(device="TEST_DEVICE", model_id="m")
            result = runner.run_inference(batch_size=1, seq_len=128)
        assert result is None

    def test_run_inference_restores_enter_patch(self):
        """The __enter__ monkeypatch is always reverted (even though cli returns None)."""
        from runners._video_generate_runner import VideoGenerateRunner

        import tensor_cast.runtime as _rt_mod

        original_enter = _rt_mod.Runtime.__enter__
        with patch("cli.inference.video_generate.run_inference", return_value=None):
            runner = VideoGenerateRunner(device="TEST_DEVICE", model_id="m")
            runner.run_inference(batch_size=1, seq_len=128)
        # Patch reverted after the call (no leak).
        assert _rt_mod.Runtime.__enter__ is original_enter


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TestVideoGenerateRunnerAdapter:
    """Tests for the VideoGenerateRunnerAdapter.run spawner."""

    def test_run_delegates_to_subprocess(self):
        adapter = VideoGenerateRunnerAdapter()
        with patch("runners._subprocess.run_module_subprocess", return_value=([], [])) as mock_sub:
            records, _skipped = adapter.run(
                {"model_id": "m"},
                job_id="j1",
                cancel_flag=lambda: False,
                cached_hashes=set(),
                form_schema_version="1.0.0",
            )
        assert records == []
        assert mock_sub.call_args.args[0] == "video_generate"
        assert mock_sub.call_args.kwargs["job_id"] == "j1"

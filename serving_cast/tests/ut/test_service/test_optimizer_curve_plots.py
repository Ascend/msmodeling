# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Unit tests for ``serving_cast/service/optimizer_curve_plots.py`` (serving_cast UT suite)."""

import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from serving_cast.service import optimizer_curve_plots as ocp

try:
    import torch as _torch_for_parallel_runner  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class TestCurvePlotHelpers(TestCase):
    def test_axis_metric_name_strips_unit_suffix(self):
        self.assertEqual(ocp._axis_metric_name("Throughput (token/s)"), "Throughput")
        self.assertEqual(ocp._axis_metric_name("foo"), "foo")

    def test_parallel_label_truncates_long_string(self):
        short = "tp2pp1dp2"
        self.assertEqual(ocp._parallel_label(short), short)
        long_p = "x" * 100
        got = ocp._parallel_label(long_p)
        self.assertEqual(len(got), 80)
        self.assertTrue(got.endswith("..."))

    def test_padded_axis_limits_empty_and_values(self):
        self.assertIsNone(ocp._padded_axis_limits([]))
        lim = ocp._padded_axis_limits([10.0])
        self.assertIsNotNone(lim)
        self.assertEqual(lim[1] - lim[0] > 0, True)
        lim2 = ocp._padded_axis_limits([0.0, 10.0])
        self.assertIsNotNone(lim2)
        self.assertGreaterEqual(lim2[0], 0.0)

    def test_compact_scatter_legend_collapses_double_marker(self):
        label = "parallel_a"
        marker = ocp._TERMINAL_MARKER
        line = f"| {marker}{marker} {label} |"
        compacted = ocp._compact_scatter_legend(line, [label])
        self.assertIn(f"{marker}{label}", compacted)
        self.assertNotIn(f"{marker}{marker} {label}", compacted)

    def test_jitter_overlapping_points_offsets_duplicates(self):
        xs = [1.0, 1.0, 2.0]
        ys = [3.0, 3.0, 4.0]
        out = ocp._jitter_overlapping_points(xs, ys)
        self.assertEqual(len(out), 3)
        self.assertNotEqual(out[0], out[1])

    def test_sorted_curve_subset_sorts_by_batch_and_concurrency(self):
        df = pd.DataFrame(
            {
                "parallel": ["p1", "p1", "p1"],
                "concurrency": [2, 1, 1],
                "batch_size": [2, 1, 2],
                "token/s": [10.0, 20.0, 15.0],
                "tpot": [5.0, 5.0, 5.0],
            }
        )
        sub = ocp._sorted_curve_subset(df, "p1", ["concurrency", "batch_size", "tpot"])
        self.assertEqual(sub.iloc[0]["concurrency"], 1)
        self.assertEqual(sub.iloc[0]["batch_size"], 1)

    def test_memory_filter_drops_non_positive_when_column_present(self):
        df = pd.DataFrame(
            {
                "parallel": ["a", "b"],
                "concurrency": [1, 1],
                "token/s": [1.0, 2.0],
                "tpot": [1.0, 1.0],
                "memory_left_gb": [1.0, -0.1],
            }
        )
        filt = ocp._memory_filter(df.copy())
        self.assertEqual(len(filt), 1)
        self.assertEqual(filt.iloc[0]["parallel"], "a")

    def test_require_columns_raises(self):
        df = pd.DataFrame({"parallel": []})
        with self.assertRaises(ValueError) as ctx:
            ocp._require_columns(df, {"parallel", "tpot"}, "missing")
        self.assertIn("tpot", str(ctx.exception))

    def test_sort_curve_df_empty(self):
        self.assertTrue(ocp._sort_curve_df(pd.DataFrame()).empty)

    def test_prepare_latency_curve_df_drops_na_and_sorts(self):
        df = pd.DataFrame(
            {
                "parallel": ["tp1", "tp1"],
                "concurrency": [1.0, 2.0],
                "token/s": [10.0, float("nan")],
                "tpot": [30.0, 20.0],
            }
        )
        work = ocp._prepare_latency_curve_df(
            df,
            latency_col="tpot",
            missing_message="test_missing",
        )
        self.assertEqual(len(work), 1)
        self.assertAlmostEqual(work.iloc[0]["token/s"], 10.0)


class TestPdTpsAndMerge(TestCase):
    def test_pd_tps_curve_df_computes_token_per_s(self):
        df = pd.DataFrame(
            {
                "parallel_d": ["d1", "d1"],
                "concurrency_d": [100.0, 200.0],
                "tpot_d": [10.0, 20.0],
            }
        )
        out = ocp._pd_tps_curve_df(df)
        self.assertIn("token/s", out.columns)
        self.assertAlmostEqual(out.loc[out["parallel"] == "d1", "token/s"].iloc[0], 10000.0)

    def test_pd_tps_curve_df_drops_non_positive_tpot(self):
        df = pd.DataFrame(
            {
                "parallel_d": ["d1"],
                "concurrency_d": [100.0],
                "tpot_d": [0.0],
            }
        )
        out = ocp._pd_tps_curve_df(df)
        self.assertTrue(out.empty)


class TestPlotEntryPoints(TestCase):
    def test_plot_concurrency_curves_from_optimizer_summaries_empty(self):
        self.assertFalse(
            ocp.plot_concurrency_curves_from_optimizer_summaries(
                [],
                basename_prefix="x",
                ttft_limit=None,
                tpot_limit=None,
            )
        )

    def test_emit_curve_df_empty_returns_false(self):
        with self.assertLogs(ocp.logger, level="WARNING") as logctx:
            ok = ocp._emit_curve_df(
                pd.DataFrame(),
                title_prefix="t",
                skip_label="unittest empty",
            )
        self.assertFalse(ok)
        self.assertTrue(any("no rows after filtering" in m for m in logctx.output))

    @patch.object(ocp, "_emit_terminal_optimizer_curve_ascii")
    def test_plot_concurrency_optimizer_curves_success(self, mock_emit):
        df = pd.DataFrame(
            {
                "parallel": ["tp2pp1dp1"],
                "concurrency": [4.0],
                "batch_size": [1],
                "token/s": [12.34],
                "tpot": [18.0],
            }
        )
        self.assertTrue(
            ocp.plot_concurrency_optimizer_curves(df, basename_prefix="unit_pref", ttft_limit=None, tpot_limit=None)
        )
        mock_emit.assert_called_once()

    @patch.object(ocp, "_emit_terminal_optimizer_curve_ascii")
    def test_plot_concurrency_optimizer_curves_value_error_returns_false(self, mock_emit):
        df = pd.DataFrame({"parallel": []})
        self.assertFalse(
            ocp.plot_concurrency_optimizer_curves(df, basename_prefix="bad", ttft_limit=None, tpot_limit=None)
        )
        mock_emit.assert_not_called()

    def test_first_non_empty_summary_df(self):
        empty = MagicMock()
        empty.get_summary_df.return_value = None
        nonempty = MagicMock()
        nonempty.get_summary_df.return_value = pd.DataFrame({"x": [1]})
        self.assertIsNotNone(ocp._first_non_empty_summary_df([empty, nonempty]))
        only_empty_df = MagicMock()
        only_empty_df.get_summary_df.return_value = pd.DataFrame()
        self.assertIsNone(ocp._first_non_empty_summary_df([empty, only_empty_df]))


class TestRenderCrossHardwareSummary(TestCase):
    @patch.object(ocp, "render_hardware_profile_comparison", return_value="")
    @patch.object(ocp, "render_cross_device_comparison", return_value="")
    def test_render_cross_hardware_summary_skips_single_device(self, _mock_render_table, _mock_render_hw):
        args = MagicMock()
        args.disagg = False
        args.enable_optimize_prefill_decode_ratio = False
        rows = ocp.MultiDeviceComparisonRows()
        logger = MagicMock()
        ocp.render_cross_hardware_summary(args, ["only_one"], rows, logger=logger)
        _mock_render_table.assert_not_called()

    @patch("builtins.print")
    @patch.object(ocp, "render_hardware_profile_comparison", return_value="hw")
    @patch.object(ocp, "render_cross_device_comparison", return_value="table")
    def test_render_cross_hardware_summary_prints_when_multi_device(
        self, _mock_render_table, _mock_render_hw, _mock_print
    ):
        args = MagicMock()
        args.disagg = False
        args.enable_optimize_prefill_decode_ratio = False
        rows = ocp.MultiDeviceComparisonRows(aggregation=[{"device": "a"}])
        logger = MagicMock()
        ocp.render_cross_hardware_summary(args, ["d1", "d2"], rows, logger=logger)
        self.assertTrue(_mock_print.called)


def _install_fake_plotext():
    """Minimal fake ``plotext`` so _emit_terminal_optimizer_curve_ascii runs without the real package."""
    import types

    m = types.ModuleType("plotext")
    for name in (
        "plot_size",
        "theme",
        "scatter",
        "xlim",
        "ylim",
        "title",
        "xlabel",
        "ylabel",
        "grid",
        "clear_data",
    ):
        setattr(m, name, MagicMock())
    m.build = MagicMock(return_value="[fake plotext ascii]\n")
    return m


class TestOptimizerCurvePlotsWithFakePlotext(TestCase):
    """Drive high-coverage paths through ``_emit_terminal_optimizer_curve_ascii`` and plot orchestration."""

    def setUp(self):
        import sys

        self._saved_plotext = sys.modules.pop("plotext", None)

    def tearDown(self):
        import sys

        if self._saved_plotext is not None:
            sys.modules["plotext"] = self._saved_plotext
        elif "plotext" in sys.modules:
            del sys.modules["plotext"]

    def test_emit_terminal_optimizer_curve_runs_with_fake_plotext(self):
        import sys

        sys.modules["plotext"] = _install_fake_plotext()
        df = pd.DataFrame(
            {
                "parallel": ["tp1", "tp1"],
                "concurrency": [1.0, 4.0],
                "batch_size": [1, 1],
                "token/s": [10.0, 12.0],
                "tpot": [30.0, 25.0],
            }
        )
        with patch("builtins.print"):
            ocp._emit_terminal_optimizer_curve_ascii(
                df, title_prefix="ut", chart2_x_col="tpot", chart2_x_label="TPOT (ms)"
            )

    def test_emit_terminal_plotext_build_failure_is_handled(self):
        import sys

        fake = _install_fake_plotext()
        fake.build = MagicMock(side_effect=RuntimeError("build fail"))
        sys.modules["plotext"] = fake
        df = pd.DataFrame(
            {
                "parallel": ["p"],
                "concurrency": [2.0],
                "batch_size": [1],
                "token/s": [9.0],
                "tpot": [11.0],
            }
        )
        with patch("builtins.print"):
            ocp._emit_terminal_optimizer_curve_ascii(df, title_prefix="ut")

    def test_plot_import_error_skips_emit(self):
        import builtins
        import sys

        sys.modules.pop("plotext", None)
        real_import = builtins.__import__

        def _import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "plotext":
                raise ImportError("stub no plotext")
            return real_import(name, globals, locals, fromlist, level)

        df = pd.DataFrame(
            {
                "parallel": ["p"],
                "concurrency": [1.0],
                "batch_size": [1],
                "token/s": [5.0],
                "tpot": [10.0],
            }
        )
        with patch("builtins.__import__", side_effect=_import):
            with self.assertLogs(ocp.logger, level="WARNING") as log_ctx:
                ocp._emit_terminal_optimizer_curve_ascii(df, title_prefix="ut")
        self.assertTrue(any("plotext" in m.lower() for m in log_ctx.output))

    def test_plot_concurrency_optimizer_curves_end_to_end_fake_plotext(self):
        import sys

        sys.modules["plotext"] = _install_fake_plotext()
        df = pd.DataFrame(
            {
                "parallel": ["a"],
                "concurrency": [2.0],
                "batch_size": [1],
                "token/s": [8.0],
                "tpot": [15.0],
            }
        )
        with patch("builtins.print"):
            self.assertTrue(
                ocp.plot_concurrency_optimizer_curves(df, basename_prefix="e2e", ttft_limit=None, tpot_limit=None)
            )

    def test_plot_disagg_prefill_and_decode_fake_plotext(self):
        import sys
        from types import SimpleNamespace

        sys.modules["plotext"] = _install_fake_plotext()

        prefill_df = pd.DataFrame(
            {
                "parallel": ["tp1"],
                "concurrency": [3.0],
                "batch_size": [1],
                "token/s": [50.0],
                "ttft": [90.0],
                "tpot": [pd.NA],
            }
        )
        dec_df = pd.DataFrame(
            {
                "parallel": ["tp2"],
                "concurrency": [4.0],
                "batch_size": [1],
                "token/s": [40.0],
                "ttft": [pd.NA],
                "tpot": [12.0],
            }
        )

        class _Res:
            def __init__(self, df, ttft_limits, tpot_limits):
                self._df = df
                self.data_config = SimpleNamespace(ttft_limits=ttft_limits, tpot_limits=tpot_limits)

            def get_summary_df(self):
                return self._df

        results = [
            _Res(prefill_df, ttft_limits=100.0, tpot_limits=None),
            _Res(dec_df, ttft_limits=None, tpot_limits=20.0),
        ]
        with patch("builtins.print"):
            ok = ocp.plot_disagg_terminal_curves(results, basename_prefix="dis", ttft_limit=None, tpot_limit=None)
        self.assertTrue(ok)

    def test_plot_pd_ratio_terminal_curves_fake_plotext(self):
        import sys

        sys.modules["plotext"] = _install_fake_plotext()
        pd_df = pd.DataFrame(
            {
                "parallel_d": ["d1", "d1"],
                "concurrency_d": [8.0, 10.0],
                "tpot_d": [2.0, 2.5],
            }
        )
        with patch("builtins.print"):
            self.assertTrue(
                ocp.plot_pd_ratio_terminal_curves(pd_df, basename_prefix="pd", ttft_limit=None, tpot_limit=None)
            )

    def test_sort_curve_df_and_prepare_base_curve_df(self):
        raw = pd.DataFrame(
            {
                "parallel": ["b", "a"],
                "concurrency": [2, 1],
                "batch_size": [1, 2],
                "token/s": [1.0, 2.0],
                "tpot": [5.0, 4.0],
            }
        )
        out = ocp._sort_curve_df(raw)
        self.assertFalse(out.empty)
        base = ocp._prepare_base_curve_df(raw, latency_col="tpot", missing_message="ut")
        self.assertIn("parallel", base.columns)

    def test_collect_cross_hardware_row_aggregation_and_pd(self):
        from types import SimpleNamespace

        rows = ocp.MultiDeviceComparisonRows()
        res = MagicMock()
        res.collect_comparison_row.return_value = {"device": "X", "throughput_tps": 1.0}
        args = SimpleNamespace(disagg=False, enable_optimize_prefill_decode_ratio=False)
        ocp._collect_cross_hardware_row(rows, res, "dev1", args)
        self.assertEqual(len(rows.aggregation), 1)

        res2 = MagicMock()
        res2.collect_pd_ratio_comparison_row.return_value = {
            "device": "Y",
            "balanced_qps": 2.0,
        }
        args_pd = SimpleNamespace(disagg=False, enable_optimize_prefill_decode_ratio=True)
        ocp._collect_cross_hardware_row(rows, res2, "dev2", args_pd)
        self.assertEqual(len(rows.pd_ratio), 1)

    @unittest.skipUnless(_TORCH_AVAILABLE, "ParallelRunner import requires torch")
    @patch.object(ocp, "_plot_single_device_optimizer_curves")
    @patch("serving_cast.parallel_runner.ParallelRunner")
    def test_run_multi_device_loop_calls_report(self, mock_pr_class, _mock_plot):
        from types import SimpleNamespace

        mock_inst = MagicMock()
        fake_res = MagicMock()
        mock_inst.run_agg.return_value = [fake_res]
        mock_pr_class.return_value = mock_inst

        args = SimpleNamespace(
            device=["PROFILE_A"],
            enable_optimize_prefill_decode_ratio=False,
            disagg=False,
            model_id="m",
        )
        logger = MagicMock()
        ocp.run_multi_device_loop(
            args,
            ["PROFILE_A"],
            plot_curves_allowed=False,
            logger=logger,
        )
        run_args = mock_pr_class.call_args.args[0]
        self.assertIsNot(run_args, args)
        self.assertEqual(run_args.device, "PROFILE_A")
        self.assertEqual(args.device, ["PROFILE_A"])
        fake_res.report_final_result.assert_called_once_with(run_args, silent=False)

    @patch("builtins.print")
    def test_render_cross_hardware_summary_disagg_branch(self, _mock_print):
        args = MagicMock()
        args.disagg = True
        rows = ocp.MultiDeviceComparisonRows(disagg_prefill=[{"device": "p"}])
        logger = MagicMock()
        with patch.object(ocp, "render_hardware_profile_comparison", return_value=""):
            with patch.object(ocp, "render_cross_hardware_disagg_prefill", return_value=""):
                with patch.object(ocp, "render_cross_hardware_disagg_decode", return_value=""):
                    ocp.render_cross_hardware_summary(args, ["a", "b"], rows, logger=logger)

    def test_prepare_curve_df_filters_device_memory_column(self):
        df = pd.DataFrame(
            {
                "parallel": ["z"],
                "concurrency": [1.0],
                "batch_size": [1],
                "token/s": [3.0],
                "tpot": [9.0],
                "device_memory_available_gb": [-1.0],
            }
        )
        work = ocp._prepare_curve_df(df, None, None)
        self.assertTrue(work.empty)

    def test_plot_concurrency_curves_from_optimizer_summaries_merges_frames(self):
        import sys

        sys.modules["plotext"] = _install_fake_plotext()

        class R:
            def get_summary_df(self):
                return pd.DataFrame(
                    {
                        "parallel": ["x"],
                        "concurrency": [1.0],
                        "batch_size": [1],
                        "token/s": [7.0],
                        "tpot": [8.0],
                    }
                )

        with patch("builtins.print"):
            self.assertTrue(
                ocp.plot_concurrency_curves_from_optimizer_summaries(
                    [R(), R()],
                    basename_prefix="merge",
                    ttft_limit=None,
                    tpot_limit=None,
                )
            )


class TestOptimizerCurvePlotsBranchCoverage(TestCase):
    """Extra branches toward ~90%+ file coverage."""

    def test_axis_metric_name_empty_after_strip_falls_back(self):
        self.assertEqual(ocp._axis_metric_name(" (x)"), " (x)")

    def test_padded_axis_limits_non_finite_filtered(self):
        self.assertEqual(
            ocp._padded_axis_limits([2.0, float("nan")]),
            ocp._padded_axis_limits([2.0]),
        )

    def test_padded_axis_limits_negative_region(self):
        lim = ocp._padded_axis_limits([-3.0, -1.0])
        self.assertIsNotNone(lim)
        self.assertLessEqual(lim[0], lim[1])
        self.assertLess(lim[0], 0.0)

    def test_compact_scatter_border_pad_and_secondary_pattern(self):
        marker = ocp._TERMINAL_MARKER
        legend_line = "\x1b[1mfake\x1b[0m" + f"{marker}{marker}\x1b[0m L2" + " │"
        out = ocp._compact_scatter_legend(legend_line, ["L2"])
        self.assertNotIn(marker + marker, out)

    def test_compact_scatter_no_pipe_appends_spaces(self):
        self.assertEqual(
            ocp._compact_scatter_legend("single line sans border", ["nope"]),
            "single line sans border",
        )

    def test_jitter_empty_inputs(self):
        self.assertEqual(ocp._jitter_overlapping_points([], []), [])

    def test_jitter_no_duplicates_keeps_coordinates(self):
        self.assertEqual(
            ocp._jitter_overlapping_points([1.0, 2.0], [3.0, 4.0]),
            [(1.0, 3.0), (2.0, 4.0)],
        )

    def test_sorted_curve_subset_no_batch_column_in_frame(self):
        df = pd.DataFrame(
            {
                "parallel": ["q", "q"],
                "concurrency": [2.0, 1.0],
                "token/s": [1.0, 2.0],
                "tpot": [9.0, 8.0],
            }
        )
        sub = ocp._sorted_curve_subset(df, "q", ["concurrency", "batch_size"])
        self.assertGreater(len(sub), 0)

    def test_memory_filter_uses_memory_left_gb_first_when_both_exist(self):
        df = pd.DataFrame(
            {
                "parallel": ["z"],
                "memory_left_gb": [5],
                "device_memory_available_gb": [-99],
            }
        )
        out = ocp._memory_filter(df.copy())
        self.assertFalse(out.empty)

    def test_memory_filter_missing_means_na_cells_kept(self):
        df = pd.DataFrame(
            {
                "parallel": ["z"],
                "device_memory_available_gb": [pd.NA],
            }
        )
        out = ocp._memory_filter(df.copy())
        self.assertFalse(out.empty)

    def test_emit_prepared_curve_value_error_logs(self):
        with self.assertLogs(ocp.logger, level="WARNING"):
            ok = ocp._emit_prepared_curve(
                lambda: (_ for _ in ()).throw(ValueError("bad df")),
                title_prefix="t",
                skip_label="skipme",
                emit_kwargs=ocp._DECODE_EMIT_KWARGS,
            )
        self.assertFalse(ok)

    def test_plot_pd_ratio_empty_returns_immediately(self):
        self.assertFalse(
            ocp.plot_pd_ratio_terminal_curves(
                pd.DataFrame(),
                basename_prefix="x",
                ttft_limit=None,
                tpot_limit=None,
            )
        )


class TestOptimizerCurvePlotsHighCoverage(TestCase):
    """Uses fake plotext for remaining uncovered paths."""

    def setUp(self):
        import sys

        self._saved = sys.modules.pop("plotext", None)

    def tearDown(self):
        import sys

        if self._saved is not None:
            sys.modules["plotext"] = self._saved
        elif "plotext" in sys.modules:
            del sys.modules["plotext"]

    def test_emit_empty_parallels_early_return(self):
        import sys

        sys.modules["plotext"] = _install_fake_plotext()
        empty = pd.DataFrame(
            {
                "parallel": pd.Series([], dtype=object),
                "concurrency": pd.Series([], dtype=float),
                "batch_size": pd.Series([], dtype=float),
                "token/s": pd.Series([], dtype=float),
                "tpot": pd.Series([], dtype=float),
            }
        )
        with patch("builtins.print") as printed:
            ocp._emit_terminal_optimizer_curve_ascii(empty, title_prefix="e")
            printed.assert_not_called()

    def test_emit_outer_exception_logs_terminal_failure(self):
        import sys

        sys.modules["plotext"] = _install_fake_plotext()
        df = pd.DataFrame(
            {
                "parallel": ["a"],
                "concurrency": [1.0],
                "batch_size": [1],
                "token/s": [9.0],
                "tpot": [11.0],
            }
        )
        # Do not patch builtins.print: logger.exception/traceback formatting also calls print.
        with patch.object(ocp, "_compact_scatter_legend", side_effect=RuntimeError("legend boom")):
            with self.assertLogs(ocp.logger, level="ERROR") as log_ctx:
                ocp._emit_terminal_optimizer_curve_ascii(df, title_prefix="boom")
        self.assertTrue(any("optimizer curves failed" in m.lower() for m in log_ctx.output))

    def test_emit_palette_wraps_multiple_parallels(self):
        import sys

        sys.modules["plotext"] = _install_fake_plotext()
        parallels = [f"p{i}" for i in range(9)]
        rows = []
        for i, p in enumerate(parallels):
            rows.append(
                {
                    "parallel": p,
                    "concurrency": float(i + 1),
                    "batch_size": 1,
                    "token/s": float(10 + i),
                    "tpot": 15.0 + i,
                }
            )
        df = pd.DataFrame(rows)
        with patch("builtins.print"):
            ocp._emit_terminal_optimizer_curve_ascii(df, title_prefix="palette", y_axis_label="QPS (req/s)")

    def test_emit_empty_build_skips_compact_scatter_print(self):
        import sys

        fake = _install_fake_plotext()
        fake.build = MagicMock(return_value="")
        sys.modules["plotext"] = fake
        df = pd.DataFrame(
            {
                "parallel": ["w"],
                "concurrency": [1.0],
                "batch_size": [1],
                "token/s": [10.0],
                "tpot": [14.0],
            }
        )
        with patch("builtins.print") as printed:
            ocp._emit_terminal_optimizer_curve_ascii(df, title_prefix="noprint")
            printed.assert_not_called()

    def test_emit_curve_passes_prefill_emit_kwargs(self):
        import sys

        sys.modules["plotext"] = _install_fake_plotext()
        df = pd.DataFrame(
            {
                "parallel": ["w"],
                "concurrency": [1.0],
                "batch_size": [1],
                "token/s": [4.0],
                "tpot": [6.0],
                "ttft": [55.0],
            }
        )
        with patch("builtins.print"):
            ocp._emit_curve_df(
                df,
                title_prefix=" pre ",
                skip_label="lbl",
                emit_kwargs=ocp._PREFILL_EMIT_KWARGS,
            )

    @patch.object(ocp, "_emit_terminal_optimizer_curve_ascii")
    def test_basename_fallback_strip_for_plot_concurrency(self, mocked):
        df = pd.DataFrame(
            {
                "parallel": ["x"],
                "concurrency": [1.0],
                "batch_size": [1],
                "token/s": [1.0],
                "tpot": [2.0],
            }
        )
        ocp.plot_concurrency_optimizer_curves(df, basename_prefix="   ", ttft_limit=None, tpot_limit=None)
        kw = mocked.call_args.kwargs
        self.assertEqual(kw.get("title_prefix"), "optimizer")

    def test_plot_disagg_skips_and_returns_false_when_no_phase(self):
        from types import SimpleNamespace

        class Rskip:
            def get_summary_df(self):
                return pd.DataFrame(
                    {
                        "parallel": ["z"],
                        "concurrency": [1.0],
                        "batch_size": [1],
                        "token/s": [1.0],
                        "tpot": [1.0],
                        "ttft": [10.0],
                    }
                )

            data_config = SimpleNamespace(ttft_limits=50.0, tpot_limits=50.0)

        self.assertFalse(
            ocp.plot_disagg_terminal_curves([Rskip()], basename_prefix="x", ttft_limit=None, tpot_limit=None)
        )

    def test_collect_cross_hardware_disagg_collectors(self):
        from types import SimpleNamespace

        rows = ocp.MultiDeviceComparisonRows()
        res = MagicMock()
        res.collect_disagg_prefill_row.return_value = {"device": "P"}
        res.collect_disagg_decode_row.return_value = None
        args = SimpleNamespace(disagg=True, enable_optimize_prefill_decode_ratio=False)
        ocp._collect_cross_hardware_row(rows, res, "dev", args)
        self.assertEqual(rows.disagg_prefill, [{"device": "P"}])
        self.assertEqual(rows.disagg_decode, [])

    def test_render_cross_hardware_disagg_warnings_when_tables_empty(self):
        args = MagicMock()
        args.disagg = True
        rows = ocp.MultiDeviceComparisonRows()
        logger = MagicMock()
        with patch.object(ocp, "render_hardware_profile_comparison", return_value=""):
            with patch.object(ocp, "render_cross_hardware_disagg_prefill", return_value=""):
                with patch.object(ocp, "render_cross_hardware_disagg_decode", return_value=""):
                    ocp.render_cross_hardware_summary(args, ["a", "b"], rows, logger=logger)
        logger.warning.assert_called_once()

    @patch("builtins.print")
    def test_render_cross_hardware_pd_ratio_branch(self, printed):
        args = MagicMock()
        args.disagg = False
        args.enable_optimize_prefill_decode_ratio = True
        rows = ocp.MultiDeviceComparisonRows(pd_ratio=[{"device": "z"}])
        logger = MagicMock()
        with patch.object(ocp, "render_hardware_profile_comparison", return_value=""):
            with patch.object(ocp, "render_cross_hardware_pd_ratio", return_value="body"):
                ocp.render_cross_hardware_summary(args, ["a", "b"], rows, logger=logger)
                printed.assert_called()

    def test_render_cross_hardware_pd_logs_when_no_render_no_rows(self):
        args = MagicMock()
        args.disagg = False
        args.enable_optimize_prefill_decode_ratio = True
        rows = ocp.MultiDeviceComparisonRows()
        logger = MagicMock()
        with patch.object(ocp, "render_hardware_profile_comparison", return_value=""):
            with patch.object(ocp, "render_cross_hardware_pd_ratio", return_value=""):
                ocp.render_cross_hardware_summary(args, ["a", "b"], rows, logger=logger)
        logger.warning.assert_called_once()

    @patch.object(ocp, "plot_pd_ratio_terminal_curves")
    @patch.object(ocp, "plot_disagg_terminal_curves")
    @patch.object(ocp, "plot_concurrency_curves_from_optimizer_summaries")
    def test_plot_single_device_dispatcher(self, mock_agg, mock_dis, mock_pd):
        args = MagicMock(ttft_limits=1.0, tpot_limits=1.0)

        mock_res_empty = MagicMock()
        mock_res_empty.get_summary_df.return_value = None
        mock_res_df = MagicMock()
        mock_res_df.get_summary_df.return_value = pd.DataFrame({"x": [1]})

        args.enable_optimize_prefill_decode_ratio = True
        args.disagg = False
        ocp._plot_single_device_optimizer_curves([mock_res_empty], args, basename_prefix="p")
        mock_pd.assert_not_called()

        ocp._plot_single_device_optimizer_curves([mock_res_df], args, basename_prefix="p")
        mock_pd.assert_called_once()

        mock_agg.reset_mock()
        mock_pd.reset_mock()
        args.enable_optimize_prefill_decode_ratio = False
        args.disagg = True
        ocp._plot_single_device_optimizer_curves([mock_res_df], args, basename_prefix="d")
        mock_dis.assert_called_once()

        mock_dis.reset_mock()
        mock_agg.reset_mock()
        args.disagg = False
        ocp._plot_single_device_optimizer_curves([mock_res_df], args, basename_prefix="a")
        mock_agg.assert_called_once()

    @unittest.skipUnless(_TORCH_AVAILABLE, "torch required for ParallelRunner import")
    @patch.object(ocp, "_plot_single_device_optimizer_curves")
    @patch("serving_cast.parallel_runner.ParallelRunner")
    def test_run_multi_device_loop_run_disagg(self, mock_pc, _plot):
        from types import SimpleNamespace

        mock_inst = MagicMock()
        mock_inst.run_disagg.return_value = []
        mock_pc.return_value = mock_inst

        args = SimpleNamespace(
            enable_optimize_prefill_decode_ratio=False,
            disagg=True,
            model_id="m",
        )
        ocp.run_multi_device_loop(args, ["D1"], plot_curves_allowed=False, logger=MagicMock())
        mock_inst.run_disagg.assert_called_once()

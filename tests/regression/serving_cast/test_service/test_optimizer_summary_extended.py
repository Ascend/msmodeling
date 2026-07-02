# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Extended unit tests for ``serving_cast/service/optimizer_summary.py`` (serving_cast UT suite).

Complements ``test_optimizer_summary.py`` in this directory with helper/render/branch coverage.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import serving_cast.service.optimizer_summary as optimizer_summary_module
from prettytable import PrettyTable
from serving_cast.service.optimizer_summary import (
    OptimizerSummary,
    _add_table_row,
    _compute_disagg_request_qps,
    _fmt_optional,
    _get_agg_table_buf,
    _get_disagg_table_buf,
    _get_pd_ratio_table_buf,
    _positive_float,
    _sorted_rows,
    render_cross_device_comparison,
    render_cross_hardware_disagg_decode,
    render_cross_hardware_disagg_prefill,
    render_cross_hardware_pd_ratio,
    render_hardware_profile_comparison,
)


class TestPositiveFloat(TestCase):
    def test_positive_float_accept_and_reject(self):
        self.assertEqual(_positive_float(1.5), 1.5)
        self.assertIsNone(_positive_float(0))
        self.assertIsNone(_positive_float(None))
        self.assertIsNone(_positive_float("bad"))


class TestFmtOptionalSortedRows(TestCase):
    def test_fmt_optional_formats_or_dash(self):
        self.assertEqual(_fmt_optional(3.14159), "3.14")
        self.assertEqual(_fmt_optional(None), "-")

    def test_sorted_rows_orders_by_metric(self):
        rows = [{"k": 1}, {"k": 3}, {"k": 2}]
        ordered = _sorted_rows(rows, "k")
        self.assertEqual([r["k"] for r in ordered], [3, 2, 1])


class TestComputeDisaggRequestQps(TestCase):
    def test_prefill_formula(self):
        row = pd.Series({"concurrency": 10.0, "ttft": 50.0, "tpot": None})
        self.assertAlmostEqual(_compute_disagg_request_qps(row, 64), 10.0 / 50.0 * 1000.0)

    def test_decode_formula_requires_output_length(self):
        row = pd.Series({"concurrency": 8.0, "ttft": None, "tpot": 2.0})
        self.assertIsNone(_compute_disagg_request_qps(row, None))
        self.assertIsNone(_compute_disagg_request_qps(row, 0))
        self.assertAlmostEqual(
            _compute_disagg_request_qps(row, 4),
            8.0 / (2.0 * 4.0) * 1000.0,
        )

    def test_returns_none_when_both_ttft_and_tpot(self):
        row = pd.Series({"concurrency": 1.0, "ttft": 1.0, "tpot": 1.0})
        self.assertIsNone(_compute_disagg_request_qps(row, 8))

    def test_returns_none_when_concurrency_invalid(self):
        row = pd.Series({"concurrency": 0.0, "ttft": 50.0, "tpot": None})
        self.assertIsNone(_compute_disagg_request_qps(row, 8))


class TestDisaggPdRatioTableBuf(TestCase):
    def test_table_row_guard_raises_on_column_mismatch(self):
        table = PrettyTable()
        table.field_names = ["A", "B"]
        with self.assertRaisesRegex(RuntimeError, "row length 1"):
            _add_table_row(table, ["only-one"], ["A", "B"])

    def test_disagg_prefill_table_title_and_qps_cell(self):
        df = pd.DataFrame(
            {
                "token/s": [88.0],
                "ttft": [110.0],
                "tpot": [pd.NA],
                "concurrency": [22.0],
                "num_devices": [1],
                "parallel": ["tp2"],
                "batch_size": [1],
            }
        )
        buf = _get_disagg_table_buf(df, output_length=None)
        self.assertRegex(buf, r"PD Disaggregated Prefill Configurations:")
        expected_qps = 22.0 / 110.0 * 1000.0
        self.assertIn(f"{expected_qps:.2f}", buf)

    def test_disagg_decode_table_uses_decode_title(self):
        df = pd.DataFrame(
            {
                "token/s": [50.0],
                "ttft": [pd.NA],
                "tpot": [2.5],
                "concurrency": [5.0],
                "num_devices": [1],
                "parallel": ["tp1"],
                "batch_size": [1],
            }
        )
        buf = _get_disagg_table_buf(df, output_length=4)
        self.assertRegex(buf, r"PD Disaggregated Decode Configurations:")
        expected_qps = 5.0 / (2.5 * 4.0) * 1000.0
        self.assertIn(f"{expected_qps:.2f}", buf)


class TestPdRatioTableBuf(TestCase):
    def test_get_pd_ratio_table_buf_contains_banner_and_columns(self):
        df = pd.DataFrame(
            {
                "pd_ratio": [0.5],
                "balanced_qps": [12.34],
                "p_qps": [10.0],
                "d_qps": [20.0],
                "ttft_p": [30.0],
                "tpot_d": [1.5],
                "parallel_p": ["Pa"],
                "parallel_d": ["Da"],
                "num_devices_p": [4],
                "num_devices_d": [4],
                "batch_size_p": [1],
                "batch_size_d": [2],
                "concurrency_p": [3],
                "concurrency_d": [4],
            }
        )
        buf = _get_pd_ratio_table_buf(df)
        self.assertIn("PD Ratio Configurations:", buf)
        self.assertIn("Balanced QPS", buf)


class TestRenderComparisonTables(TestCase):
    def test_render_helpers_empty_lists(self):
        self.assertEqual(render_cross_device_comparison([]), "")
        self.assertEqual(render_cross_hardware_pd_ratio([]), "")
        self.assertEqual(render_cross_hardware_disagg_prefill([]), "")
        self.assertEqual(render_cross_hardware_disagg_decode([]), "")

    def test_render_cross_device_comparison_non_empty(self):
        txt = render_cross_device_comparison(
            [
                {
                    "device": "D1",
                    "throughput_tps": 99.9,
                    "concurrency": 1,
                    "parallel": "p",
                    "batch_size": 1,
                    "num_devices": 1,
                }
            ]
        )
        self.assertIn("Cross-hardware", txt)
        self.assertIn("D1", txt)

    def test_render_cross_hardware_pd_ratio_shows_banner(self):
        rows = [
            {
                "device": "X",
                "balanced_qps": 1.23,
                "pd_ratio": 0.25,
                "p_qps": 4.0,
                "d_qps": 1.0,
                "ttft_p": 50.0,
                "tpot_d": 2.0,
                "p_instances": 2,
                "d_instances": 1,
                "total_devices": 8,
            }
        ]
        txt = render_cross_hardware_pd_ratio(rows)
        self.assertIn("PD Ratio", txt)
        self.assertIn("num-devices=", txt.lower())

    def test_render_cross_hardware_disagg_prefill_decode(self):
        pref = render_cross_hardware_disagg_prefill(
            [
                {
                    "device": "P1",
                    "throughput_tps": 80.0,
                    "qps_req_s": None,
                    "ttft_ms": 100.0,
                    "concurrency": 2,
                }
            ]
        )
        self.assertIn("PD Disaggregated Prefill", pref)
        dec = render_cross_hardware_disagg_decode(
            [
                {
                    "device": "D2",
                    "throughput_tps": 90.0,
                    "qps_req_s": 1.23,
                    "tpot_ms": 20.0,
                    "concurrency": 3,
                }
            ]
        )
        self.assertIn("PD Disaggregated Decode", dec)

    def test_render_cross_hardware_pd_ratio_num_devices_banner(self):
        rows = [
            {
                "device": "Hw1",
                "balanced_qps": 9.0,
                "pd_ratio": 0.5,
                "p_qps": 18.0,
                "d_qps": 9.0,
                "ttft_p": 60.0,
                "tpot_d": 1.5,
                "p_instances": 4,
                "d_instances": 2,
                "total_devices": 16,
            }
        ]
        txt = render_cross_hardware_pd_ratio(rows)
        self.assertIn("--num-devices=16", txt)


class TestOptimizerSummaryBranches(TestCase):
    def test_report_final_result_silent_returns_immediately(self):
        cfg = SimpleNamespace(ttft_limits=10.0, tpot_limits=10.0, output_length=8)
        s = OptimizerSummary(cfg)
        s.set_summary_df(pd.DataFrame({"token/s": [1.0], "ttft": [1.0], "tpot": [1.0], "concurrency": [1]}))
        with patch("builtins.print") as p:
            s.report_final_result(SimpleNamespace(disagg=False, dump_original_results=False), silent=True)
            p.assert_not_called()

    def test_report_final_result_warns_when_no_summary(self):
        cfg = SimpleNamespace(ttft_limits=10.0, tpot_limits=10.0, output_length=8)
        s = OptimizerSummary(cfg)
        with self.assertLogs("serving_cast.service.optimizer_summary", level="WARNING") as log_ctx:
            s.report_final_result(SimpleNamespace(disagg=False, dump_original_results=False))
        self.assertTrue(any("empty or unset" in m for m in log_ctx.output))

    def test_get_agg_disagg_final_out_empty_after_filters(self):
        cfg = SimpleNamespace(ttft_limits=10.0, tpot_limits=10.0, output_length=None)
        s = OptimizerSummary(cfg)
        s.set_summary_df(
            pd.DataFrame(
                {
                    "token/s": [1.0],
                    "ttft": [1000.0],
                    "tpot": [500.0],
                    "concurrency": [1],
                    "num_devices": [1],
                    "parallel": ["x"],
                    "batch_size": [1],
                }
            )
        )
        args = SimpleNamespace(
            model_id="m",
            num_devices=1,
            device="TEST",
            quantize_linear_action="DISABLED",
            quantize_attention_action="DISABLED",
            disagg=False,
            input_length=1024,
        )
        with self.assertLogs("serving_cast.service.optimizer_summary", level="WARNING") as log_ctx:
            out = s._get_agg_disagg_final_out(args)
        self.assertTrue(any("TTFT/TPOT filters" in m for m in log_ctx.output))
        self.assertIn("No configurations satisfy", "\n".join(out))

    def test_collect_comparison_row_via_best_agg_disagg_row(self):
        cfg = SimpleNamespace(ttft_limits=1000.0, tpot_limits=50.0, output_length=32)
        s = OptimizerSummary(cfg)
        s.set_summary_df(
            pd.DataFrame(
                {
                    "token/s": [10.0, 30.0],
                    "ttft": [90.0, 80.0],
                    "tpot": [5.0, 6.0],
                    "concurrency": [1, 2],
                    "num_devices": [8, 8],
                    "parallel": ["tp1", "tp2"],
                    "batch_size": [1, 1],
                }
            )
        )
        row = s.collect_comparison_row("device_a")
        self.assertEqual(row["device"], "device_a")
        self.assertEqual(row["throughput_tps"], 30.0)

    def test_collect_disagg_prefill_decode_guards(self):
        # Prefill collector requires TTFT limit and forbids simultaneous TPOT limit.
        cfg_no_ttft = SimpleNamespace(ttft_limits=None, tpot_limits=None, output_length=None)
        self.assertIsNone(OptimizerSummary(cfg_no_ttft).collect_disagg_prefill_row("d"))

        cfg_ttft_and_tpot = SimpleNamespace(ttft_limits=100.0, tpot_limits=10.0, output_length=None)
        self.assertIsNone(OptimizerSummary(cfg_ttft_and_tpot).collect_disagg_prefill_row("d"))

        # Decode collector requires TPOT limit and forbids TTFT limit simultaneously.
        cfg_decode_but_ttft_set = SimpleNamespace(ttft_limits=100.0, tpot_limits=10.0, output_length=4)
        self.assertIsNone(OptimizerSummary(cfg_decode_but_ttft_set).collect_disagg_decode_row("d"))

    def test_collect_pd_ratio_comparison_row_needs_pd_mode_and_data(self):
        cfg_plain = SimpleNamespace(ttft_limits=100.0, tpot_limits=10.0, num_devices=None)
        s_plain = OptimizerSummary(cfg_plain)
        s_plain.set_summary_df(pd.DataFrame({"x": [1]}))
        self.assertIsNone(s_plain.collect_pd_ratio_comparison_row("d"))


def _baseline_agg_row(updates=None):
    base = {
        "token/s": 10.0,
        "ttft": 120.0,
        "tpot": 40.0,
        "concurrency": 8,
        "num_devices": 4,
        "parallel": "tp1",
        "batch_size": 1,
    }
    if updates:
        base.update(updates)
    return base


def _baseline_pd_ratio_row(**overrides):
    base = {
        "balanced_qps": 222.222,
        "pd_ratio": 0.625,
        "p_qps": 40.0,
        "d_qps": 25.0,
        "ttft_p": 200.0,
        "tpot_d": 12.5,
        "parallel_p": "Pa",
        "parallel_d": "Db",
        "num_devices_p": 2,
        "num_devices_d": 4,
        "batch_size_p": 1,
        "batch_size_d": 2,
        "concurrency_p": 4,
        "concurrency_d": 16,
    }
    base.update(overrides)
    return base


class TestOptimizerSummaryEarlyStopHelpers(TestCase):
    def test_set_get_summary_accessor(self):
        cfg = SimpleNamespace(ttft_limits=1.0, tpot_limits=1.0)
        obj = OptimizerSummary(cfg)
        df = pd.DataFrame({"z": [1]})
        obj.set_summary_df(df)
        pd.testing.assert_frame_equal(obj.get_summary_df(), df)

    def test_early_stop_flags(self):
        cfg = SimpleNamespace(ttft_limits=50.0, tpot_limits=30.0, output_length=8)
        s = OptimizerSummary(cfg)

        s.set_early_stop_flag(memory_left=-1, tpot=None, ttft=None)
        self.assertTrue(s.check_early_stop_flag())

        s.set_early_stop_flag(memory_left=1, tpot=None, ttft=None)
        self.assertFalse(s.check_early_stop_flag())

        s.set_early_stop_flag(memory_left=1, tpot=60.0, ttft=None)
        self.assertTrue(s.check_early_stop_flag())

        s.set_early_stop_flag(memory_left=1, tpot=None, ttft=100.0)
        self.assertTrue(s.check_early_stop_flag())

        cfg_no_limits = SimpleNamespace(ttft_limits=None, tpot_limits=None, output_length=None)
        s2 = OptimizerSummary(cfg_no_limits)
        s2.set_early_stop_flag(memory_left=1, tpot=999.0, ttft=999.0)
        self.assertFalse(s2.check_early_stop_flag())


class TestOptimizerSummaryReportAndCollect(TestCase):
    def test_best_agg_disabled_in_pd_ratio_mode(self):
        cfg = SimpleNamespace(
            ttft_limits=1000.0,
            tpot_limits=100.0,
            output_length=8,
            prefill_devices_per_instance=2,
            decode_devices_per_instance=2,
        )
        s = OptimizerSummary(cfg)
        s.set_summary_df(pd.DataFrame([_baseline_agg_row()]))
        self.assertIsNone(s.collect_comparison_row("x"))

    def test_collect_disagg_prefill_and_decode_success(self):
        pref_cfg = SimpleNamespace(ttft_limits=500.0, tpot_limits=None, output_length=8)
        s_pref = OptimizerSummary(pref_cfg)
        s_pref.set_summary_df(
            pd.DataFrame(
                [
                    _baseline_agg_row(
                        {
                            "token/s": 111.1,
                            "ttft": 200.0,
                            "tpot": float("nan"),
                            "parallel": "pref",
                        }
                    ),
                ]
            )
        )
        row_p = s_pref.collect_disagg_prefill_row("Pdev")
        self.assertEqual(row_p["device"], "Pdev")
        self.assertAlmostEqual(row_p["throughput_tps"], 111.1)

        dec_cfg = SimpleNamespace(ttft_limits=None, tpot_limits=50.0, output_length=4)
        s_dec = OptimizerSummary(dec_cfg)
        s_dec.set_summary_df(
            pd.DataFrame(
                [
                    _baseline_agg_row(
                        {
                            "token/s": 50.0,
                            "tpot": 5.0,
                            "ttft": float("nan"),
                            "parallel": "dec",
                        }
                    ),
                ]
            )
        )
        row_d = s_dec.collect_disagg_decode_row("Ddev")
        self.assertEqual(row_d["device"], "Ddev")

    def test_row_dict_na_latency_fields(self):
        cfg = SimpleNamespace(ttft_limits=None, tpot_limits=None, output_length=None)
        s = OptimizerSummary(cfg)
        s.set_summary_df(
            pd.DataFrame(
                [
                    _baseline_agg_row(
                        {
                            "token/s": 333.3,
                            "ttft": pd.NA,
                            "tpot": pd.NA,
                            "parallel": "na_row",
                            "concurrency": 9,
                            "num_devices": 1,
                            "batch_size": 2,
                        }
                    ),
                ]
            )
        )
        rc = s.collect_comparison_row("uut")
        self.assertIsNone(rc["ttft_ms"])
        self.assertIsNone(rc["tpot_ms"])
        self.assertEqual(rc["parallel"], "na_row")

    def test_prepare_pd_ratio_dedupe_and_comparison_row_instances(self):
        cfg = SimpleNamespace(
            ttft_limits=9999.0,
            tpot_limits=9999.0,
            num_devices=32,
            prefill_devices_per_instance=2,
            decode_devices_per_instance=4,
        )
        s = OptimizerSummary(cfg)
        r0 = _baseline_pd_ratio_row(
            balanced_qps=500.501,
            parallel_p="P1",
            parallel_d="D1",
        )
        r1 = _baseline_pd_ratio_row(
            balanced_qps=490.1,
            parallel_p="P1",
            parallel_d="D9",
            pd_ratio=0.75,
            num_devices_d=8,
            num_devices_p=2,
        )
        dup_balanced = dict(r1)
        dup_balanced["balanced_qps"] = r1["balanced_qps"] + 0.008
        s.set_summary_df(pd.DataFrame([r0, r1, dup_balanced]))

        filt = s._prepare_pd_ratio_results()
        self.assertFalse(filt.empty)
        self.assertTrue((filt["balanced_qps"] <= 500.601).any())

        comp = s.collect_pd_ratio_comparison_row("hw-X")
        self.assertEqual(comp["device"], "hw-X")
        self.assertAlmostEqual(comp["balanced_qps"], filt.iloc[0]["balanced_qps"], places=6)
        self.assertIsNotNone(comp.get("p_instances"))
        self.assertIsNotNone(comp.get("d_instances"))
        self.assertEqual(comp.get("total_devices"), 32)

    def test_get_agg_disagg_final_out_disagg_branch(self):
        cfg = SimpleNamespace(ttft_limits=500.0, tpot_limits=40.0, output_length=None)
        s = OptimizerSummary(cfg)
        s.set_summary_df(
            pd.DataFrame(
                [
                    _baseline_agg_row(
                        {
                            "token/s": 555.5,
                            "ttft": 35.0,
                            "tpot": 8.0,
                            "parallel": "pref",
                        }
                    ),
                ]
            )
        )
        args = SimpleNamespace(
            model_id="m",
            num_devices=32,
            device="DEVICE",
            quantize_linear_action="OFF",
            quantize_attention_action="OFF",
            disagg=True,
            input_length=1024,
        )
        out = s._get_agg_disagg_final_out(args)
        joined = "\n".join(out)
        self.assertIn("PD Disaggregated Prefill", joined)

    def test_get_agg_table_buf_contains_rows(self):
        df = pd.DataFrame(
            [
                _baseline_agg_row({"token/s": 777.77, "ttft": 10.0, "tpot": 3.33, "parallel": "px"}),
                _baseline_agg_row({"token/s": 5.5, "ttft": 20.0, "tpot": 1.1, "parallel": "py"}),
            ]
        )
        buf = _get_agg_table_buf(df)
        self.assertIn("Aggregated Configurations", buf)
        self.assertIn("777.77", buf)

    def test_report_final_agg_dump_original_and_normal(self):
        cfg = SimpleNamespace(ttft_limits=500.0, tpot_limits=40.0, output_length=8)
        s = OptimizerSummary(cfg)
        s.set_summary_df(pd.DataFrame([_baseline_agg_row({"token/s": 12.34})]))
        dump_args = SimpleNamespace(
            disagg=False,
            dump_original_results=True,
            model_id="_",
            num_devices=1,
            device="-",
            quantize_linear_action="",
            quantize_attention_action="",
            input_length=1024,
        )
        with patch("builtins.print") as pr:
            s.report_final_result(dump_args, silent=False)
            self.assertGreaterEqual(pr.call_count, 1)

        norm_args = SimpleNamespace(
            disagg=False,
            dump_original_results=False,
            model_id="m",
            num_devices=1,
            device="dev",
            quantize_linear_action="QL",
            quantize_attention_action="QA",
            input_length=1024,
        )
        with patch("builtins.print") as pr2:
            s.report_final_result(norm_args, silent=False)
            merged = "".join(call.args[0] for call in pr2.call_args_list if call.args)
            self.assertIn("Overall Best", merged)

    def test_report_pd_ratio_dump_empty_filtered_infos(self):
        cfg = SimpleNamespace(
            ttft_limits=500.0,
            tpot_limits=40.0,
            output_length=8,
            prefill_devices_per_instance=4,
            decode_devices_per_instance=4,
            num_devices=128,
        )
        s = OptimizerSummary(cfg)
        s.set_summary_df(
            pd.DataFrame(
                [
                    _baseline_pd_ratio_row(
                        balanced_qps=999.99,
                        ttft_p=1e9,
                        tpot_d=1e9,
                        parallel_p="_",
                        parallel_d="_",
                        num_devices_p=2,
                        num_devices_d=2,
                        batch_size_p=1,
                        batch_size_d=1,
                        concurrency_p=1,
                        concurrency_d=1,
                    ),
                ]
            )
        )

        args = SimpleNamespace(dump_original_results=True, device="CARD", model_id="MID")
        with self.assertLogs(optimizer_summary_module.logger, level="INFO"), patch("builtins.print"):
            s.report_final_result(args, silent=False)

    def test_report_pd_ratio_dump_non_empty_df(self):
        cfg = SimpleNamespace(
            ttft_limits=800.0,
            tpot_limits=80.0,
            output_length=8,
            prefill_devices_per_instance=2,
            decode_devices_per_instance=2,
            num_devices=None,
        )
        row = dict(_baseline_pd_ratio_row(parallel_p="PP", parallel_d="DD"))
        s = OptimizerSummary(cfg)
        s.set_summary_df(pd.DataFrame([row]))
        args_dump = SimpleNamespace(dump_original_results=True, device="CARD", model_id="MID")

        captured = ""

        def _capture(*parts, **_kwargs):
            nonlocal captured
            captured += " ".join(str(p) for p in parts)

        with patch("builtins.print", side_effect=_capture):
            s.report_final_result(args_dump, silent=False)
        self.assertIn("balanced_qps", captured)

    def test_report_pd_ratio_pretty_best_with_instances(self):
        cfg = SimpleNamespace(
            ttft_limits=450.0,
            tpot_limits=45.0,
            output_length=8,
            prefill_devices_per_instance=2,
            decode_devices_per_instance=2,
            num_devices=32,
        )
        row = dict(_baseline_pd_ratio_row(pd_ratio=0.5))
        df = pd.DataFrame([row])
        s = OptimizerSummary(cfg)
        s.set_summary_df(df)

        filt = s._prepare_pd_ratio_results()
        self.assertFalse(filt.empty)
        fout = s._get_pd_ratio_final_out(
            SimpleNamespace(model_id="model-x", device="mydev"),
            filt,
        )
        body = "\n".join(fout)
        self.assertIn("model-x", body)
        self.assertIn("P Instances:", body)

    def test_pd_ratio_final_out_includes_memory_info(self):
        cfg = SimpleNamespace(
            ttft_limits=450.0,
            tpot_limits=45.0,
            output_length=8,
            prefill_devices_per_instance=2,
            decode_devices_per_instance=2,
            num_devices=32,
        )
        s = OptimizerSummary(cfg)
        s.set_memory_info({"total_device_memory_gb": 64.0, "reserved_memory_gb": 4.0})
        fout = s._get_pd_ratio_final_out(
            SimpleNamespace(model_id="model-x", device="mydev"),
            pd.DataFrame([dict(_baseline_pd_ratio_row(pd_ratio=0.5))]),
        )
        body = "\n".join(fout)
        self.assertIn("Memory Info:", body)
        self.assertIn("Total device memory:", body)

    def test_report_pd_ratio_pretty_not_dump_calls_print(self):
        cfg = SimpleNamespace(
            ttft_limits=450.0,
            tpot_limits=45.0,
            output_length=8,
            prefill_devices_per_instance=2,
            decode_devices_per_instance=2,
            num_devices=32,
        )
        s = OptimizerSummary(cfg)
        s.set_summary_df(pd.DataFrame([dict(_baseline_pd_ratio_row(pd_ratio=0.5))]))
        args = SimpleNamespace(
            dump_original_results=False,
            device="CARD",
            model_id="MID",
            quantize_linear_action="OFF",
            quantize_attention_action="OFF",
        )
        with patch("builtins.print") as pr:
            s.report_final_result(args, silent=False)
        self.assertGreaterEqual(pr.call_count, 1)

    def test_collect_comparison_returns_none_when_all_rows_filtered_out(self):
        cfg = SimpleNamespace(ttft_limits=1e-6, tpot_limits=1e-6, output_length=None)
        s = OptimizerSummary(cfg)
        s.set_summary_df(
            pd.DataFrame([_baseline_agg_row({"token/s": 999.9, "ttft": 1e9, "tpot": 1e9, "parallel": "gone"})])
        )
        self.assertIsNone(s.collect_comparison_row("dev"))

    def test_collect_pd_ratio_summary_unset_or_filtered_returns_none(self):
        pd_cfg = SimpleNamespace(
            ttft_limits=100.0,
            tpot_limits=100.0,
            output_length=None,
            prefill_devices_per_instance=1,
            decode_devices_per_instance=1,
            num_devices=None,
        )
        unset = OptimizerSummary(pd_cfg)
        self.assertIsNone(unset.collect_pd_ratio_comparison_row("d"))

        s = OptimizerSummary(pd_cfg)
        s.set_summary_df(
            pd.DataFrame(
                [
                    dict(_baseline_pd_ratio_row(ttft_p=1e9, tpot_d=1e9)),
                ]
            )
        )
        self.assertIsNone(s.collect_pd_ratio_comparison_row("d"))


class TestRenderHardwareProfileComparisonStubbedImports(TestCase):
    """Exercise ``render_hardware_profile_comparison`` inner branches without requiring ``torch``."""

    _saved_sys_modules: dict

    def setUp(self):
        self._saved_sys_modules = dict(sys.modules)

    def tearDown(self):
        extras = [
            k
            for k in list(sys.modules)
            if k not in self._saved_sys_modules and (k.startswith("tensor_cast") or k == "torch")
        ]
        for k in extras:
            sys.modules.pop(k, None)
        sys.modules.clear()
        sys.modules.update(self._saved_sys_modules)

    def _stub_modules_for_hardware_render(self, profile_map, tor_stub):
        class DeviceProfile:
            all_device_profiles = profile_map

        dev_pkg = ModuleType("tensor_cast.device")
        dev_pkg.DeviceProfile = DeviceProfile

        tc_pkg = ModuleType("tensor_cast")
        tc_pkg.__path__ = []
        device_profiles_stub = ModuleType("tensor_cast.device_profiles")
        tc_pkg.device_profiles = device_profiles_stub

        sys.modules["torch"] = tor_stub
        sys.modules["tensor_cast"] = tc_pkg
        sys.modules["tensor_cast.device_profiles"] = device_profiles_stub
        sys.modules["tensor_cast.device"] = dev_pkg

    def test_render_profiles_hits_torch_branch_and_notes(self):
        bf = object()
        hf = object()
        tor = ModuleType("torch")
        tor.bfloat16 = bf
        tor.half = hf

        def _grid(shape):
            g = SimpleNamespace()
            g.shape = shape
            return g

        prof_full = SimpleNamespace(
            name="full_bf16",
            mma_ops={bf: 400e12},
            gp_ops={
                bf: 80e12,
                hf: 50e12,
            },
            compute_efficiency=0.93,
            memory_bandwidth_bytes_ps=900e9,
            memory_efficiency=0.88,
            memory_size_bytes=96 * (1024**3),
            comm_grid=SimpleNamespace(
                topologies={
                    0: SimpleNamespace(bandwidth_bytes_ps=120e9, comm_efficiency=0.8),
                    3: SimpleNamespace(bandwidth_bytes_ps=60e9, comm_efficiency=0.92),
                },
                grid=_grid((2, 4)),
            ),
        )
        prof_half_only = SimpleNamespace(
            name="half_only",
            mma_ops={hf: 200e12},
            gp_ops={},
            compute_efficiency=1.0,
            memory_bandwidth_bytes_ps=450e9,
            memory_efficiency=0.5,
            memory_size_bytes=32 * (1024**3),
            comm_grid=SimpleNamespace(
                topologies={1: SimpleNamespace(bandwidth_bytes_ps=30e9, comm_efficiency=0.61)},
                grid=_grid((8,)),
            ),
        )
        prof_empty_peak = SimpleNamespace(
            name="empty_ops",
            mma_ops={},
            gp_ops={},
            compute_efficiency=1.0,
            memory_bandwidth_bytes_ps=210e9,
            memory_efficiency=1.0,
            memory_size_bytes=128 * (1024**3),
            comm_grid=SimpleNamespace(topologies={}, grid=_grid((1, 1))),
        )
        profiles = {
            prof_full.name: prof_full,
            prof_half_only.name: prof_half_only,
            prof_empty_peak.name: prof_empty_peak,
        }
        self._stub_modules_for_hardware_render(profiles, tor)

        txt = render_hardware_profile_comparison(
            [
                "missing_device",
                prof_full.name,
                prof_half_only.name,
                prof_empty_peak.name,
                prof_full.name,
            ]
        )
        self.assertIn("missing_device", txt)
        self.assertIn("Notes:", txt)
        self.assertIn("empty_ops", txt)
        self.assertIn(prof_half_only.name, txt)
        self.assertGreaterEqual(txt.count(prof_full.name), 1)

    def test_gp_ops_max_peak_when_bf16_half_missing(self):
        alt = object()
        tor = ModuleType("torch")
        tor.bfloat16 = object()
        tor.half = object()

        def _grid(shape):
            g = SimpleNamespace()
            g.shape = shape
            return g

        prof = SimpleNamespace(
            name="mixed_keys",
            mma_ops={alt: 500e12},
            gp_ops={"z_other": 200e11},
            compute_efficiency=1.0,
            memory_bandwidth_bytes_ps=320e9,
            memory_efficiency=1.0,
            memory_size_bytes=128 * (1024**3),
            comm_grid=SimpleNamespace(
                topologies={0: SimpleNamespace(bandwidth_bytes_ps=10e9, comm_efficiency=0.55)},
                grid=_grid((4, 2)),
            ),
        )

        profiles = {prof.name: prof}
        self._stub_modules_for_hardware_render(profiles, tor)

        txt = render_hardware_profile_comparison([prof.name])
        self.assertIn(prof.name, txt)
        self.assertIn("500.00", txt)


class TestRenderHardwareProfileShortcuts(TestCase):
    def test_empty_device_name_list_returns_empty_string(self):
        self.assertEqual(render_hardware_profile_comparison([]), "")

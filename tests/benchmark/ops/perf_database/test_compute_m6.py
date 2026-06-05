"""Tests for compute_m6.py v2: TC trace vs Prof trace comparison."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[4] / "tools" / "perf_data_analysis"),
)
from compute_m6 import (
    _format_report,
    _sum_kernels_with_dedup,
    build_argparser,
    compute_m6,
)
from tests.helpers.cli_runner import run_module_main


def _make_tc_trace(tmp_path, events=None):
    """Create a chrome trace JSON fixture."""
    if events is None:
        events = [
            # 64 FIA invocations (1 per layer), source=MEASURED
            *[
                {
                    "name": "tensor_cast.attention.default",
                    "ph": "X",
                    "ts": i * 100,
                    "dur": 53,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "source": "MEASURED",
                        "kernel_type": "FusedInferAttentionScore",
                        "confidence": 0.9,
                    },
                }
                for i in range(64)
            ],
            # 128 MatMulV2 (2 per layer)
            *[
                {
                    "name": "aten.mm.default",
                    "ph": "X",
                    "ts": 10000 + i * 50,
                    "dur": 20,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "source": "MEASURED",
                        "kernel_type": "MatMulV2",
                        "confidence": 0.9,
                    },
                }
                for i in range(128)
            ],
            # 100 zero_cost ops (dur=0)
            *[
                {
                    "name": "aten.view.default",
                    "ph": "X",
                    "ts": 20000 + i * 10,
                    "dur": 0,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "source": "MEASURED",
                        "kernel_type": "zero_cost",
                        "confidence": 1.0,
                    },
                }
                for i in range(100)
            ],
            # 3 MISS ops (no source, analytic fallback)
            *[
                {
                    "name": "tensor_cast.apply_rope.default",
                    "ph": "X",
                    "ts": 30000 + i * 10,
                    "dur": 2,
                    "pid": 0,
                    "tid": 0,
                    "args": {},
                }
                for i in range(3)
            ],
            # Metadata events (should be ignored)
            {"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "test"}},
        ]
    path = tmp_path / "tc_trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return path


def _make_prof_trace(tmp_path, rows=None):
    """Create a prof trace CSV fixture (clean forward pass)."""
    if rows is None:
        t = 0
        rows = []
        for _ in range(64):
            rows.append(("FusedInferAttentionScore", "50.0", str(t), str(t + 50), '"16,4,128"'))
            t += 60
        for _ in range(128):
            rows.append(("MatMulV2", "25.0", str(t), str(t + 25), '"16,5120"'))
            t += 30
        for _ in range(64):
            rows.append(("hcom_allReduce_", "100.0", str(t), str(t + 100), '""'))
            t += 110  # unique start times → no dedup
        rows.append(("Sort", "200.0", str(t), str(t + 200), '""'))
    path = tmp_path / "prof_trace.csv"
    lines = ["Type,Duration(us),Start Time(us),End Time(us),Input Shapes"]
    for row in rows:
        lines.append(",".join(str(x) for x in row))
    path.write_text("\n".join(lines))
    return path


class TestComputeM6TraceMode:
    """Tests for the new tc-trace + prof-trace interface."""

    def test_basic_m6(self, tmp_path):
        tc_path = _make_tc_trace(tmp_path)
        prof_path = _make_prof_trace(tmp_path)
        result = compute_m6(tc_trace=str(tc_path), prof_trace=str(prof_path))

        # TC MEASURED dur>0: 64*53 + 128*20 = 3392 + 2560 = 5952
        # Prof total: 64*50 + 128*25 + 64*100 + 200 = 3200+3200+6400+200 = 13000
        assert result["empirical_hit_us"] == pytest.approx(5952.0)
        assert result["real_per_fwd_us"] == pytest.approx(13000.0)
        assert result["m6_ratio"] == pytest.approx(5952.0 / 13000.0, rel=1e-3)

    def test_compute_hcom_split(self, tmp_path):
        tc_path = _make_tc_trace(tmp_path)
        prof_path = _make_prof_trace(tmp_path)
        result = compute_m6(tc_trace=str(tc_path), prof_trace=str(prof_path))

        # Prof: 64*50 + 128*25 = 6400 compute, 64*100 = 6400 hcom, 200 Sort
        assert result["selected_fwd_compute_us"] == pytest.approx(6600.0)
        assert result["selected_fwd_hcom_us"] == pytest.approx(6400.0)

    def test_source_filter_measured_only(self, tmp_path):
        """--source-filter MEASURED excludes INTERPOLATED events."""
        events = [
            {
                "name": "op_a",
                "ph": "X",
                "ts": 0,
                "dur": 100,
                "pid": 0,
                "tid": 0,
                "args": {"source": "MEASURED", "kernel_type": "MatMulV2"},
            },
            {
                "name": "op_b",
                "ph": "X",
                "ts": 100,
                "dur": 50,
                "pid": 0,
                "tid": 0,
                "args": {"source": "INTERPOLATED", "kernel_type": "RmsNorm"},
            },
        ]
        tc_path = tmp_path / "tc.json"
        tc_path.write_text(json.dumps({"traceEvents": events}))
        prof_path = _make_prof_trace(tmp_path)

        # Default: both MEASURED and INTERPOLATED
        result_all = compute_m6(tc_trace=str(tc_path), prof_trace=str(prof_path))
        assert result_all["empirical_hit_us"] == pytest.approx(150.0)

        # Filter: MEASURED only
        result_m = compute_m6(
            tc_trace=str(tc_path),
            prof_trace=str(prof_path),
            source_filter={"MEASURED"},
        )
        assert result_m["empirical_hit_us"] == pytest.approx(100.0)

    def test_miss_ops_excluded(self, tmp_path):
        """Events without source (MISS/analytic) are excluded from empirical_hit."""
        tc_path = _make_tc_trace(tmp_path)
        prof_path = _make_prof_trace(tmp_path)
        result = compute_m6(tc_trace=str(tc_path), prof_trace=str(prof_path))
        # 3 MISS ops with dur=2 each should NOT be in empirical_hit
        # empirical_hit = MEASURED only = 64*53 + 128*20 = 5952
        assert result["empirical_hit_us"] == pytest.approx(5952.0)

    def test_no_per_kernel_delta_in_result(self, tmp_path):
        """per_kernel_delta was removed — result should not contain it."""
        tc_path = _make_tc_trace(tmp_path)
        prof_path = _make_prof_trace(tmp_path)
        result = compute_m6(tc_trace=str(tc_path), prof_trace=str(prof_path))
        assert "per_kernel_delta" not in result

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_m6(tc_trace="/nonexistent.json", prof_trace=str(tmp_path / "x.csv"))

    def test_hcom_dedup_in_prof_trace(self, tmp_path):
        """Prof trace hcom dedup works correctly."""
        tc_path = _make_tc_trace(tmp_path, events=[])
        prof_rows = [
            ("hcom_allReduce_", "100.0", "1000.0", "1100.0", '""'),
            ("hcom_allReduce_", "100.0", "1000.0", "1100.0", '""'),  # dup
            ("MatMulV2", "50.0", "2000.0", "2050.0", '""'),
        ]
        prof_path = _make_prof_trace(tmp_path, prof_rows)
        result = compute_m6(tc_trace=str(tc_path), prof_trace=str(prof_path))
        # hcom deduped: 100 (not 200) + MatMulV2 50 = 150
        assert result["real_per_fwd_us"] == pytest.approx(150.0)


class TestSumKernelsWithDedupPreserved:
    """Ensure _sum_kernels_with_dedup still works (shared utility)."""

    def test_hcom_dedup(self):
        events = [
            (1000.0, 1010.0, "hcom_allReduce_", ""),
            (1000.0, 1010.0, "hcom_allReduce_", ""),
            (2000.0, 2006.0, "MatMulV2", ""),
        ]
        compute_us, hcom_us, aicpu_us, kc, ktd = _sum_kernels_with_dedup(events)
        assert hcom_us == pytest.approx(10.0)
        assert compute_us == pytest.approx(6.0)
        assert kc == 2

    def test_hcom_dedup_keeps_max(self):
        events = [
            (1000.0, 1008.0, "hcom_allReduce_", ""),  # dur=8
            (1000.0, 1015.0, "hcom_allReduce_", ""),  # dur=15, larger
        ]
        _, hcom_us, _, kc, _ = _sum_kernels_with_dedup(events)
        assert hcom_us == pytest.approx(15.0)
        assert kc == 1

    def test_aicpu_excluded_from_compute(self):
        events = [
            (100.0, 200.0, "allgatherAicpuKernel", ""),
            (200.0, 300.0, "MatMulV2", ""),
        ]
        compute_us, hcom_us, aicpu_us, kc, ktd = _sum_kernels_with_dedup(events)
        assert aicpu_us == pytest.approx(100.0)
        assert compute_us == pytest.approx(100.0)
        assert "allgatherAicpuKernel" not in ktd

    def test_empty_events(self):
        compute_us, hcom_us, aicpu_us, kc, ktd = _sum_kernels_with_dedup([])
        assert compute_us == 0.0
        assert hcom_us == 0.0
        assert kc == 0

    def test_kernel_type_durations_tracked(self):
        events = [
            (100.0, 110.0, "MatMulV2", ""),
            (200.0, 205.0, "RmsNorm", ""),
        ]
        _, _, _, _, ktd = _sum_kernels_with_dedup(events)
        assert ktd["MatMulV2"] == pytest.approx(10.0)
        assert ktd["RmsNorm"] == pytest.approx(5.0)


class TestFormatReport:
    def test_output_contains_all_fields(self):
        result = {
            "m6_ratio": 0.95,
            "empirical_hit_us": 5952.0,
            "real_per_fwd_us": 6265.26,
            "selected_fwd_compute_us": 3200.0,
            "selected_fwd_hcom_us": 3065.26,
            "tc_trace": "/path/to/tc.json",
            "prof_trace": "/path/to/prof.csv",
            "source_filter": ["MEASURED"],
        }
        report = _format_report(result)
        assert "M6" in report
        assert "0.950" in report
        assert "/path/to/tc.json" in report
        assert "/path/to/prof.csv" in report
        assert "MEASURED" in report
        assert "5,952.0" in report
        assert "6,265.3" in report

    def test_ratio_greater_than_one(self):
        result = {
            "m6_ratio": 1.5,
            "empirical_hit_us": 150.0,
            "real_per_fwd_us": 100.0,
            "selected_fwd_compute_us": 80.0,
            "selected_fwd_hcom_us": 20.0,
            "tc_trace": "a.json",
            "prof_trace": "b.csv",
            "source_filter": ["MEASURED", "INTERPOLATED"],
        }
        report = _format_report(result)
        assert "1.500" in report


class TestBuildArgparser:
    def test_required_args(self):
        parser = build_argparser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_tc_and_prof_trace(self):
        parser = build_argparser()
        args = parser.parse_args(
            [
                "--tc-trace",
                "trace.json",
                "--prof-trace",
                "prof.csv",
            ]
        )
        assert args.tc_trace == "trace.json"
        assert args.prof_trace == "prof.csv"
        assert args.source_filter is None
        assert args.json_output is None

    def test_parses_source_filter(self):
        parser = build_argparser()
        args = parser.parse_args(
            [
                "--tc-trace",
                "t.json",
                "--prof-trace",
                "p.csv",
                "--source-filter",
                "MEASURED",
            ]
        )
        assert args.source_filter == "MEASURED"

    def test_parses_json_output(self):
        parser = build_argparser()
        args = parser.parse_args(
            [
                "--tc-trace",
                "t.json",
                "--prof-trace",
                "p.csv",
                "--json-output",
                "out.json",
            ]
        )
        assert args.json_output == "out.json"


class TestJsonOutput:
    def test_json_output_written(self, tmp_path):
        tc_path = _make_tc_trace(tmp_path)
        prof_path = _make_prof_trace(tmp_path)
        json_out = tmp_path / "m6_out.json"
        result = compute_m6(tc_trace=str(tc_path), prof_trace=str(prof_path))
        json_out.write_text(json.dumps(result, indent=2))
        assert json_out.exists()
        loaded = json.loads(json_out.read_text())
        assert "m6_ratio" in loaded
        assert "empirical_hit_us" in loaded


class TestMainCli:
    def test_main_exits_cleanly(self, tmp_path):
        tc_path = _make_tc_trace(tmp_path)
        prof_path = _make_prof_trace(tmp_path)
        result = run_module_main(
            "tools.perf_data_analysis.compute_m6",
            [
                "--tc-trace",
                str(tc_path),
                "--prof-trace",
                str(prof_path),
            ],
        )
        assert result.returncode == 0
        assert "M6" in result.stdout

    def test_main_with_json_output(self, tmp_path):
        tc_path = _make_tc_trace(tmp_path)
        prof_path = _make_prof_trace(tmp_path)
        json_out = tmp_path / "result.json"
        result = run_module_main(
            "tools.perf_data_analysis.compute_m6",
            [
                "--tc-trace",
                str(tc_path),
                "--prof-trace",
                str(prof_path),
                "--json-output",
                str(json_out),
            ],
        )
        assert result.returncode == 0
        assert json_out.exists()
        data = json.loads(json_out.read_text())
        assert "m6_ratio" in data

    def test_main_with_source_filter(self, tmp_path):
        tc_path = _make_tc_trace(tmp_path)
        prof_path = _make_prof_trace(tmp_path)
        result = run_module_main(
            "tools.perf_data_analysis.compute_m6",
            [
                "--tc-trace",
                str(tc_path),
                "--prof-trace",
                str(prof_path),
                "--source-filter",
                "MEASURED",
            ],
        )
        assert result.returncode == 0

    def test_main_file_not_found_exits_nonzero(self, tmp_path):
        result = run_module_main(
            "tools.perf_data_analysis.compute_m6",
            [
                "--tc-trace",
                str(tmp_path / "nonexistent.json"),
                "--prof-trace",
                str(tmp_path / "prof.csv"),
            ],
        )
        assert result.returncode != 0

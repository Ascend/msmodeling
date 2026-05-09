"""Tests for generate_per_shape_comparison.py — per-(kernel_type, shape) delta."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "tools" / "perf_data_analysis"),
)
from generate_per_shape_comparison import generate_per_shape_comparison  # noqa: E402


def _make_trace(tmp_path, events):
    path = tmp_path / "tc_trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return str(path)


def _make_prof(tmp_path, rows):
    """Create prof forward-pass CSV. rows: list of (Type, Input Shapes, Duration)."""
    path = tmp_path / "prof.csv"
    lines = ["Type,Duration(us),Start Time(us),Input Shapes"]
    t = 0
    for ktype, shapes, dur in rows:
        lines.append(f'{ktype},{dur},{t},"{shapes}"')
        t += dur + 10
    path.write_text("\n".join(lines))
    return str(path)


def _x(name, dur, pid=0, **kwargs):
    return {
        "name": name,
        "ph": "X",
        "ts": 0,
        "dur": dur,
        "pid": pid,
        "tid": 0,
        "args": {k: str(v) for k, v in kwargs.items()},
    }


class TestBasicComparison:
    def test_single_kernel_exact_match(self, tmp_path):
        """TC and prof both have MatMulV2 with same shape."""
        trace = _make_trace(
            tmp_path,
            [
                _x(
                    "aten.mm.default",
                    20,
                    kernel_type="MatMulV2",
                    source="MEASURED",
                    simulation_shapes="[[4112, 5120], [5120, 768]]",
                ),
            ],
        )
        prof = _make_prof(
            tmp_path,
            [
                ("MatMulV2", "4112,5120;5120,768", 25.0),
            ],
        )
        out = tmp_path / "out.csv"
        generate_per_shape_comparison(trace, prof, str(out))

        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 1
        r = rows[0]
        assert r["kernel_type"] == "MatMulV2"
        assert float(r["tc_dur_us"]) == pytest.approx(20.0)
        assert float(r["prof_dur_us"]) == pytest.approx(25.0)
        assert float(r["delta_pct"]) == pytest.approx(-20.0, abs=0.1)

    def test_multiple_shapes_same_kernel(self, tmp_path):
        """Two different shapes for MatMulV2."""
        trace = _make_trace(
            tmp_path,
            [
                _x(
                    "mm",
                    20,
                    kernel_type="MatMulV2",
                    source="MEASURED",
                    simulation_shapes="[[100, 200], [200, 300]]",
                ),
                _x(
                    "mm",
                    40,
                    kernel_type="MatMulV2",
                    source="MEASURED",
                    simulation_shapes="[[500, 200], [200, 300]]",
                ),
            ],
        )
        prof = _make_prof(
            tmp_path,
            [
                ("MatMulV2", "100,200;200,300", 22.0),
                ("MatMulV2", "500,200;200,300", 38.0),
            ],
        )
        out = tmp_path / "out.csv"
        generate_per_shape_comparison(trace, prof, str(out))

        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 2


class TestCompositeExpansion:
    def test_composite_sub_kernels_expanded(self, tmp_path):
        """Composite op with sub_kernel_durations creates separate rows."""
        trace = _make_trace(
            tmp_path,
            [
                _x(
                    "mla",
                    77,
                    kernel_type="BMNd,FIA,TBMM",
                    source="MEASURED",
                    composite="True",
                    sub_kernel_durations="[('BMNd', 9.0), ('FIA', 55.0), ('TBMM', 13.0)]",
                    simulation_shapes="[[4, 512]]",
                ),
            ],
        )
        prof = _make_prof(
            tmp_path,
            [
                ("BMNd", "4,512", 10.0),
                ("FIA", "4,512", 50.0),
                ("TBMM", "4,512", 14.0),
            ],
        )
        out = tmp_path / "out.csv"
        generate_per_shape_comparison(trace, prof, str(out))

        rows = list(csv.DictReader(out.open()))
        kts = {r["kernel_type"] for r in rows}
        assert "FIA" in kts
        fia_row = next(r for r in rows if r["kernel_type"] == "FIA")
        assert float(fia_row["tc_dur_us"]) == pytest.approx(55.0)
        assert float(fia_row["prof_dur_us"]) == pytest.approx(50.0)


class TestUnmatchedEntries:
    def test_tc_only_kernel(self, tmp_path):
        """Kernel in TC but not in prof → prof_dur_us = 0."""
        trace = _make_trace(
            tmp_path,
            [
                _x(
                    "op",
                    10,
                    kernel_type="OnlyInTC",
                    source="MEASURED",
                    simulation_shapes="[[100]]",
                ),
            ],
        )
        prof = _make_prof(tmp_path, [])
        out = tmp_path / "out.csv"
        generate_per_shape_comparison(trace, prof, str(out))

        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 1
        assert float(rows[0]["prof_dur_us"]) == 0

    def test_prof_only_kernel(self, tmp_path):
        """Kernel in prof but not in TC → tc_dur_us = 0."""
        trace = _make_trace(tmp_path, [])
        prof = _make_prof(
            tmp_path,
            [
                ("OnlyInProf", "100,200", 30.0),
            ],
        )
        out = tmp_path / "out.csv"
        generate_per_shape_comparison(trace, prof, str(out))

        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 1
        assert float(rows[0]["tc_dur_us"]) == 0


class TestAggregation:
    def test_same_shape_aggregated(self, tmp_path):
        """Multiple invocations of same (kernel_type, shape) are summed."""
        trace = _make_trace(
            tmp_path,
            [
                _x(
                    "mm",
                    20,
                    kernel_type="MatMulV2",
                    source="MEASURED",
                    simulation_shapes="[[100, 200]]",
                ),
                _x(
                    "mm",
                    30,
                    kernel_type="MatMulV2",
                    source="MEASURED",
                    simulation_shapes="[[100, 200]]",
                ),
            ],
        )
        prof = _make_prof(
            tmp_path,
            [
                ("MatMulV2", "100,200", 22.0),
                ("MatMulV2", "100,200", 28.0),
            ],
        )
        out = tmp_path / "out.csv"
        generate_per_shape_comparison(trace, prof, str(out))

        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 1
        assert float(rows[0]["tc_dur_us"]) == pytest.approx(50.0)
        assert float(rows[0]["prof_dur_us"]) == pytest.approx(50.0)
        assert int(rows[0]["tc_count"]) == 2

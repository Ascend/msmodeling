"""Regression tests for extract_tc_from_chrome_trace (generate_op_comparison.py).

- Bug 1: Composite sub-kernel duration split (equal-split → sub_kernel_durations)
- Bug 2: hcom_allReduce_ zeroing (MC2 fused vs standalone)
- Bug 4: DynamicQuant must not be remapped to AscendQuantV2
- Bug 6: MISS ops with analytic duration should be labeled, not mixed
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[4] / "tools" / "perf_data_analysis"),
)
from generate_op_comparison import extract_tc_from_chrome_trace

# Minimal op_mapping for tests
OP_MAPPING = {
    "operator_mappings": {
        "tensor_cast.multihead_latent_attention.default": {
            "composite": True,
            "decomposer": True,
        },
        "tensor_cast.static_quant_linear_all_reduce.default": {
            "composite": True,
            "sub_kernels": ["QuantBatchMatmulV3", "hcom_allReduce_"],
        },
        "aten.view.default": {"zero_cost": True},
        "tensor_cast.all_reduce.default": {
            "kernel_type": "hcom_allReduce_",
        },
    }
}


def _make_trace(tmp_path, events):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return str(path)


def _x_event(
    name,
    dur,
    kernel_type=None,
    source="MEASURED",
    composite=False,
    sub_kernel_durations=None,
    pid=0,
):
    """Helper to create a chrome trace X event."""
    args = {}
    if source:
        args["source"] = source
    if kernel_type:
        args["kernel_type"] = kernel_type
    if composite:
        args["composite"] = "True"
    if sub_kernel_durations is not None:
        args["sub_kernel_durations"] = str(sub_kernel_durations)
    return {
        "name": name,
        "ph": "X",
        "ts": 0,
        "dur": dur,
        "pid": pid,
        "tid": 0,
        "args": args,
    }


class TestCompositeSubKernelDurations:
    """Bug 1: Composite ops should use sub_kernel_durations, not equal-split."""

    def test_decomposed_composite_uses_sub_kernel_durations(self, tmp_path):
        """MLA composite with sub_kernel_durations should split by actual ratio."""
        trace = _make_trace(
            tmp_path,
            [
                {
                    "name": "tensor_cast.multihead_latent_attention.default",
                    "ph": "X",
                    "ts": 0,
                    "dur": 77,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "source": "MEASURED",
                        "kernel_type": "BatchMatMulV2,FusedInferAttentionScore,TransposeBatchMatMul",
                        "composite": "True",
                        "sub_kernel_durations": str(
                            [
                                ("BatchMatMulV2", 9.0),
                                ("FusedInferAttentionScore", 55.0),
                                ("TransposeBatchMatMul", 13.0),
                            ]
                        ),
                    },
                },
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)

        # Should use actual durations, not 77/3 = 25.67 each
        assert abs(stats["FusedInferAttentionScore"]["total_us"] - 55.0) < 0.1
        assert abs(stats["BatchMatMulV2"]["total_us"] - 9.0) < 0.1
        assert abs(stats["TransposeBatchMatMul"]["total_us"] - 13.0) < 0.1

    def test_mc2_composite_uses_sub_kernel_durations(self, tmp_path):
        """MC2 composite with sub_kernel_durations should split correctly."""
        trace = _make_trace(
            tmp_path,
            [
                {
                    "name": "tensor_cast.static_quant_linear_all_reduce.default",
                    "ph": "X",
                    "ts": 0,
                    "dur": 31,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "source": "MEASURED",
                        "kernel_type": "QuantBatchMatmulV3",
                        "composite": "True",
                        "sub_kernel_durations": str(
                            [
                                ("QuantBatchMatmulV3", 22.77),
                                ("hcom_allReduce_", 8.29),
                            ]
                        ),
                    },
                },
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)

        assert abs(stats["QuantBatchMatmulV3"]["total_us"] - 22.77) < 0.1
        assert abs(stats["hcom_allReduce_"]["total_us"] - 8.29) < 0.1


class TestHcomAllReduceZeroing:
    """Bug 2: hcom_allReduce_ should not be zeroed for standalone ops."""

    def test_standalone_allreduce_preserves_duration(self, tmp_path):
        """A standalone allReduce (not MC2) should keep its duration."""
        trace = _make_trace(
            tmp_path,
            [
                # One standalone allReduce
                _x_event("tensor_cast.all_reduce.default", 8, kernel_type="hcom_allReduce_"),
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)

        assert stats["hcom_allReduce_"]["total_us"] == pytest.approx(8.0)

    def test_mc2_plus_standalone_allreduce(self, tmp_path):
        """MC2 fused allReduce (dur=0) + standalone allReduce (dur=8)."""
        trace = _make_trace(
            tmp_path,
            [
                # MC2 with sub_kernel_durations
                {
                    "name": "tensor_cast.static_quant_linear_all_reduce.default",
                    "ph": "X",
                    "ts": 0,
                    "dur": 31,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "source": "MEASURED",
                        "kernel_type": "QuantBatchMatmulV3",
                        "composite": "True",
                        "sub_kernel_durations": str(
                            [
                                ("QuantBatchMatmulV3", 22.77),
                                ("hcom_allReduce_", 8.29),
                            ]
                        ),
                    },
                },
                # Standalone allReduce
                _x_event("tensor_cast.all_reduce.default", 8, kernel_type="hcom_allReduce_"),
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)

        # hcom_allReduce_ should have MC2 portion (8.29) + standalone (8) = 16.29
        assert stats["hcom_allReduce_"]["total_us"] == pytest.approx(16.29, abs=0.1)


class TestMissOpsLabeling:
    """Bug 6: MISS ops should be clearly labeled."""

    def test_miss_ops_have_miss_status(self, tmp_path):
        """Ops with no source (analytic fallback) should have MISS status."""
        trace = _make_trace(
            tmp_path,
            [
                # MISS op: no source, no kernel_type
                _x_event("aten.add.Tensor", 2, source=""),
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)

        # Should exist under op name (no op_mapping entry) and be marked MISS
        assert "aten.add.Tensor" in stats
        assert stats["aten.add.Tensor"]["miss_count"] > 0


class TestDynamicQuantNoRemap:
    """Bug 4: DynamicQuant should not be remapped to AscendQuantV2."""

    def test_dynamicquant_preserved(self, tmp_path):
        """quantize.default matching DynamicQuant should keep that kernel name."""
        trace = _make_trace(
            tmp_path,
            [
                _x_event("tensor_cast.quantize.default", 4, kernel_type="DynamicQuant"),
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)

        assert "DynamicQuant" in stats
        assert stats["DynamicQuant"]["total_us"] == pytest.approx(4.0)
        # Should NOT be remapped to AscendQuantV2
        assert "AscendQuantV2" not in stats or stats["AscendQuantV2"]["total_us"] == 0


class TestMalformedSubKernelDurations:
    """Malformed sub_kernel_durations should fall back to equal-split."""

    def test_flat_list_falls_back(self, tmp_path):
        """A flat list (not list of pairs) should be rejected."""
        trace = _make_trace(
            tmp_path,
            [
                {
                    "name": "tensor_cast.multihead_latent_attention.default",
                    "ph": "X",
                    "ts": 0,
                    "dur": 90,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "source": "MEASURED",
                        "kernel_type": "A,B,C",
                        "composite": "True",
                        "sub_kernel_durations": "[10, 20, 60]",
                    },
                },
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)
        # Falls back to equal-split: 90/3 = 30 each
        assert abs(stats["A"]["total_us"] - 30.0) < 0.1
        assert abs(stats["B"]["total_us"] - 30.0) < 0.1

    def test_non_numeric_duration_falls_back(self, tmp_path):
        """Non-parseable sub_kernel_durations should fall back."""
        trace = _make_trace(
            tmp_path,
            [
                {
                    "name": "tensor_cast.multihead_latent_attention.default",
                    "ph": "X",
                    "ts": 0,
                    "dur": 60,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "source": "MEASURED",
                        "kernel_type": "X,Y",
                        "composite": "True",
                        "sub_kernel_durations": "garbage",
                    },
                },
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)
        assert abs(stats["X"]["total_us"] - 30.0) < 0.1


class TestParseProfilingByType:
    def test_basic_counting(self, tmp_path):
        csv_path = tmp_path / "prof.csv"
        csv_path.write_text(
            "Type,Duration(us),Start Time(us),Input Shapes\n"
            'MatMulV2,50.0,1000.0,""\n'
            'MatMulV2,30.0,2000.0,""\n'
            'RmsNorm,10.0,3000.0,""\n'
        )
        from generate_op_comparison import parse_profiling_by_type

        stats = parse_profiling_by_type(str(csv_path))
        assert stats["MatMulV2"]["total_us"] == pytest.approx(80.0)
        assert stats["MatMulV2"]["count"] == 2
        assert stats["RmsNorm"]["total_us"] == pytest.approx(10.0)
        assert stats["RmsNorm"]["count"] == 1

    def test_aicpu_skipped(self, tmp_path):
        csv_path = tmp_path / "prof.csv"
        csv_path.write_text(
            "Type,Duration(us),Start Time(us),Input Shapes\n"
            'allgatherAicpuKernel,200.0,1000.0,""\n'
            'MatMulV2,50.0,2000.0,""\n'
        )
        from generate_op_comparison import parse_profiling_by_type

        stats = parse_profiling_by_type(str(csv_path))
        assert "allgatherAicpuKernel" not in stats
        assert "MatMulV2" in stats


class TestResolveKernelType:
    def test_resolve_via_op_mapping(self):
        import yaml
        from generate_op_comparison import resolve_kernel_type

        mapping = yaml.safe_load("aten.mm: {kernel_type: MatMulV2}")
        assert resolve_kernel_type("aten.mm", mapping) == "MatMulV2"
        assert resolve_kernel_type("unknown.op", mapping) is None


class TestIsZeroCost:
    def test_zero_cost_op(self):
        import yaml
        from generate_op_comparison import is_zero_cost

        mapping = yaml.safe_load("aten.view: {zero_cost: true}\naten.add: {}")
        assert is_zero_cost("aten.view", mapping) is True
        assert is_zero_cost("aten.add", mapping) is False
        assert is_zero_cost("unknown", mapping) is False


class TestLoadOpMapping:
    def test_missing_file_returns_empty(self, tmp_path):
        from generate_op_comparison import load_op_mapping

        result = load_op_mapping(str(tmp_path / "nonexistent"))
        assert result == {}


class TestExtractTcEdgeCases:
    def test_partial_status(self, tmp_path):
        trace = _make_trace(
            tmp_path,
            [
                _x_event("aten.mm.default", 20, kernel_type="MatMulV2", source="MEASURED"),
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)
        assert "MatMulV2" in stats
        assert stats["MatMulV2"]["hit_count"] == 1

    def test_miss_only_status(self, tmp_path):
        trace = _make_trace(
            tmp_path,
            [
                _x_event("aten.add.Tensor", 5, source=""),
                _x_event("aten.add.Tensor", 3, source=""),
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)
        assert "aten.add.Tensor" in stats
        assert stats["aten.add.Tensor"]["miss_count"] == 2
        assert stats["aten.add.Tensor"]["hit_count"] == 0

    def test_non_x_events_ignored(self, tmp_path):
        trace = _make_trace(
            tmp_path,
            [
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": 1,
                    "args": {"name": "profiling"},
                },
                _x_event("aten.mm", 20, kernel_type="MatMulV2", source="MEASURED", pid=1),
            ],
        )
        stats = extract_tc_from_chrome_trace(trace, OP_MAPPING)
        assert stats["MatMulV2"]["count"] == 1


class TestBuildComparison:
    def test_basic(self, tmp_path):
        from generate_op_comparison import build_comparison

        m6_path = tmp_path / "m6.json"
        m6_path.write_text(json.dumps({"m6_ratio": 0.95}))

        tc_path = tmp_path / "tc.json"
        tc_path.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        _x_event("aten.mm", 20, kernel_type="MatMulV2", source="MEASURED"),
                    ]
                }
            )
        )

        prof_path = tmp_path / "prof.csv"
        prof_path.write_text(
            "Type,Duration(us),Start Time(us),Input Shapes\nMatMulV2,50.0,1000.0,\"\"\nRmsNorm,10.0,2000.0,\"\"\n"
        )

        scenario = {
            "name": "test",
            "m6": str(m6_path),
            "tc_trace": str(tc_path),
            "trace_csv": str(prof_path),
        }
        rows = build_comparison(scenario, OP_MAPPING)
        assert len(rows) >= 2
        mm_row = next(r for r in rows if r["kernel_type"] == "MatMulV2")
        assert mm_row["tc_total_us"] == pytest.approx(20.0)
        assert mm_row["prof_per_fwd_us"] == pytest.approx(50.0)

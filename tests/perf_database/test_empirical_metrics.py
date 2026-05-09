"""Unit tests for M4/M5/M6 evaluation metrics."""

from unittest.mock import MagicMock

import torch

from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.empirical import (
    EmpiricalOpRecord,
    EmpiricalPerformanceModel,
)
from tensor_cast.performance_model.metrics_collector import (
    compute_fused_op_stats,
    compute_per_shape_stats,
    MetricsCollector,
)
from tensor_cast.performance_model.profiling_database.data_source import (
    DataSourcePerformanceModel,
    QueryResult,
    QuerySource,
)


# --- MetricsCollector Unit Tests ---


class TestMetricsCollector:
    """Unit tests for MetricsCollector class."""

    def test_collect_hit(self):
        """collect_from_records() with full HIT result updates hit count and latency."""
        collector = MetricsCollector()
        result = QueryResult(
            latency_us=100.0,
            confidence=1.0,
            source=QuerySource.MEASURED,
            details={"kernel_type": "MatMulV2"},
        )
        collector.collect_from_records(
            [
                EmpiricalOpRecord(
                    func_name="aten.mm.default",
                    lookup_result=result,
                    analytic_latency_s=50e-6,
                    tc_shapes=[(2048, 5120), (5120, 5120)],
                )
            ]
        )

        stats = collector.get_stats()
        assert stats["hit"] == 1
        assert stats["miss"] == 0
        assert collector._hit_latency_sum == 50e-6
        assert collector._total_latency_sum == 50e-6

    def test_collect_miss(self):
        """collect_from_records() with None result updates miss count."""
        collector = MetricsCollector()
        collector.collect_from_records(
            [
                EmpiricalOpRecord(
                    func_name="aten.mm.default",
                    lookup_result=None,
                    analytic_latency_s=50e-6,
                    tc_shapes=[(2048, 5120), (5120, 5120)],
                )
            ]
        )

        stats = collector.get_stats()
        assert stats["hit"] == 0
        assert stats["miss"] == 1
        assert collector._hit_latency_sum == 0.0
        assert collector._total_latency_sum == 50e-6

    def test_collect_partial(self):
        """collect_from_records() with PARTIAL result counts as MISS but uses empirical latency."""
        collector = MetricsCollector()
        result = QueryResult(
            latency_us=100.0,
            confidence=0.5,
            source=QuerySource.PARTIAL,
            details={
                "kernel_type": ["MatMulV2"],
                "missed_kernels": ["X"],
            },
        )
        collector.collect_from_records(
            [
                EmpiricalOpRecord(
                    func_name="aten.mm.default",
                    lookup_result=result,
                    analytic_latency_s=50e-6,
                    tc_shapes=[(2048, 5120)],
                )
            ]
        )

        stats = collector.get_stats()
        assert stats["hit"] == 0
        assert stats["miss"] == 1
        # PARTIAL: analytic latency in total, but NOT in hit
        assert collector._hit_latency_sum == 0.0
        assert collector._total_latency_sum == 50e-6

    def test_collect_zero_cost(self):
        """collect_from_records() with zero_cost flag uses sentinel kernel_type."""
        collector = MetricsCollector()
        result = QueryResult(
            latency_us=0.0,
            confidence=1.0,
            source=QuerySource.MEASURED,
            details={"kernel_type": "View", "zero_cost": True},
        )
        collector.collect_from_records(
            [
                EmpiricalOpRecord(
                    func_name="aten.view.default",
                    lookup_result=result,
                    analytic_latency_s=1e-9,
                    tc_shapes=[(2048, 5120)],
                )
            ]
        )

        assert collector._hit_details[0][1] == "zero_cost"

    def test_collect_with_miss_reason(self):
        """collect_from_records() accepts miss_reason parameter for full MISS."""
        collector = MetricsCollector()
        collector.collect_from_records(
            [
                EmpiricalOpRecord(
                    func_name="aten.mm.default",
                    lookup_result=None,
                    analytic_latency_s=50e-6,
                    tc_shapes=[(2048, 5120)],
                    miss_reason="shape_mismatch",
                )
            ]
        )

        assert collector._miss_details[0][1] == "shape_mismatch"

    def test_get_stats(self):
        """get_stats() returns correct M1 stats."""
        collector = MetricsCollector()
        result = QueryResult(
            latency_us=100.0,
            confidence=1.0,
            source=QuerySource.MEASURED,
            details={"kernel_type": "MatMulV2"},
        )
        collector.collect_from_records(
            [
                EmpiricalOpRecord("op1", result, 10e-6, [(1, 2)]),
                EmpiricalOpRecord("op2", None, 20e-6, [(3, 4)]),
            ]
        )

        stats = collector.get_stats()
        assert stats["hit"] == 1
        assert stats["miss"] == 1
        assert stats["total"] == 2
        assert abs(stats["m1_raw_op_count_hr"] - 0.5) < 1e-9

    def test_export_hit_miss_report_structure(self):
        """export_hit_miss_report() returns correct structure."""
        collector = MetricsCollector()
        result = QueryResult(
            latency_us=100.0,
            confidence=1.0,
            source=QuerySource.MEASURED,
            details={"kernel_type": "MatMulV2"},
        )
        collector.collect_from_records(
            [
                EmpiricalOpRecord("aten.mm.default", result, 50e-6, [(2048, 5120)]),
                EmpiricalOpRecord("aten.add.default", None, 30e-6, [(1024, 512)]),
            ]
        )

        report = collector.export_hit_miss_report()

        assert "m1" in report
        assert "m2" in report
        assert "m3" in report
        assert "m4" in report
        assert "m5" in report
        assert "misses" in report
        assert report["m1"]["m1_hit"] == 1
        assert report["m1"]["m1_miss"] == 1


class TestPartialMetrics:
    def test_partial_uses_latency_but_counts_as_miss(self):
        """PARTIAL result: latency is used in E2E, but counted as MISS in metrics."""
        mock_ds = MagicMock(spec=DataSourcePerformanceModel)
        mock_ds.lookup.return_value = QueryResult(
            latency_us=100.0,
            confidence=0.5,
            source=QuerySource.PARTIAL,
            details={
                "kernel_type": ["QuantBatchMatmulV3"],
                "missed_kernels": ["KvRmsNormRopeCache"],
                "composite": True,
                "partial": True,
            },
        )

        mock_device = MagicMock()
        mock_device.flops = 1e12
        mock_device.bandwidth = 1e12

        mock_fallback = MagicMock(spec=PerformanceModel)
        mock_fallback.process_op.return_value = PerformanceModel.Result(
            execution_time_s=200e-6,
            statistics={},
        )
        mock_fallback.get_classifiers.return_value = []

        pm = EmpiricalPerformanceModel(mock_device, mock_ds, mock_fallback)

        op = MagicMock()
        op.func = torch.ops.tensor_cast.mlapo_quant.default
        op.args = (torch.empty(4099, 7168, device="meta", dtype=torch.bfloat16),)

        result = pm.process_op(op)

        # PARTIAL uses empirical latency
        assert abs(result.execution_time_s - 100e-6) < 1e-9

        # But counts as MISS in stats
        collector = MetricsCollector()
        collector.collect_from_records(pm.op_records)
        stats = collector.get_stats()
        assert stats["miss"] == 1
        assert stats["hit"] == 0

    def test_partial_shown_separately_in_log_stats(self, caplog):
        """PARTIAL entries are shown in a separate line, not mixed into MISSes."""
        import logging

        # Test log_stats() directly on MetricsCollector with pre-populated state
        collector = MetricsCollector()
        # Manually populate state to simulate processed ops
        collector._stats = {"hit": 3, "miss": 4}
        collector._hit_details = [
            ("tensor_cast.swiglu.default", "SwiGlu", ((2048, 6912),), 12e-6),
            ("tensor_cast.swiglu.default", "SwiGlu", ((2048, 6912),), 12e-6),
            ("tensor_cast.swiglu.default", "SwiGlu", ((2048, 6912),), 12e-6),
        ]
        collector._miss_details = [
            (
                "tensor_cast.mlapo_quant.default",
                "partial:KvRmsNormRopeCache",
                [(4099, 7168)],
                200e-6,
            ),
            (
                "tensor_cast.mlapo_quant.default",
                "partial:KvRmsNormRopeCache",
                [(4099, 7168)],
                200e-6,
            ),
            (
                "tensor_cast.multihead_latent_attention.default",
                "partial:FusedInferAttentionScore",
                [(4099, 512)],
                200e-6,
            ),
            (
                "aten.mm.default",
                "shape_mismatch",
                [(4096, 5120), (5120, 5120)],
                200e-6,
            ),
        ]

        with caplog.at_level(logging.INFO):
            collector.log_stats()

        log_text = caplog.text

        # PARTIAL line should appear with count and op names
        assert "PARTIAL: 3/7" in log_text
        assert "mlapo_quant" in log_text
        assert "multihead_latent_attention" in log_text

        # MISSes section should NOT contain the partial reasons
        # Find the MISSes line and verify it only has 1 unique reason
        assert "MISSes (1 unique reasons)" in log_text
        assert "[shape_mismatch]" in log_text


class TestM4PerShapeMatchRate:
    """M4: Per-Shape Match HR -- unique (func_name, shape) pairs, excl zero_cost."""

    def test_mixed_hit_miss(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("tensor_cast.swiglu.default", "SwiGlu", ((2048, 6912),), 12.1e-6),
        ]
        miss_details = [
            ("aten.mm.default", "shape_mismatch", [(4096, 5120), (5120, 5120)]),
            ("tensor_cast.swiglu.default", "shape_mismatch", [(4096, 6912)]),
        ]
        stats = compute_per_shape_stats(hit_details, miss_details)
        assert stats["m4_hit_shapes"] == 2
        assert stats["m4_total_shapes"] == 4
        assert abs(stats["m4_per_shape_hr"] - 0.5) < 1e-9

    def test_all_hit(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
        ]
        stats = compute_per_shape_stats(hit_details, [])
        assert stats["m4_per_shape_hr"] == 1.0

    def test_all_miss(self):
        miss_details = [
            ("aten.mm.default", "shape_mismatch", [(2048, 5120), (5120, 5120)]),
        ]
        stats = compute_per_shape_stats([], miss_details)
        assert stats["m4_per_shape_hr"] == 0.0

    def test_zero_cost_excluded(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("aten.view.default", "zero_cost", ((2048, 5120),), 0.0),
            ("aten.permute.default", "zero_cost", ((2048, 5120),), 0.0),
        ]
        miss_details = [
            ("aten.mm.default", "shape_mismatch", [(4096, 5120), (5120, 5120)]),
        ]
        stats = compute_per_shape_stats(hit_details, miss_details)
        assert stats["m4_hit_shapes"] == 1
        assert stats["m4_total_shapes"] == 2
        assert abs(stats["m4_per_shape_hr"] - 0.5) < 1e-9

    def test_accepted_miss_excluded(self):
        """accepted_miss ops are excluded from M4 same as zero_cost."""
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("aten.index.Tensor", "accepted_miss", ((163840, 128),), 0.0),
            (
                "tensor_cast.concat_and_cache_mla.default",
                "accepted_miss",
                ((4099, 512),),
                0.0,
            ),
        ]
        miss_details = []
        stats = compute_per_shape_stats(hit_details, miss_details)
        # Only aten.mm counted; accepted_miss excluded
        assert stats["m4_hit_shapes"] == 1
        assert stats["m4_total_shapes"] == 1

    def test_duplicate_shape_calls_unique(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
        ]
        stats = compute_per_shape_stats(hit_details, [])
        assert stats["m4_hit_shapes"] == 1
        assert stats["m4_total_shapes"] == 1

    def test_empty_inputs(self):
        stats = compute_per_shape_stats([], [])
        assert stats["m4_per_shape_hr"] == 0.0
        assert stats["m4_hit_shapes"] == 0
        assert stats["m4_total_shapes"] == 0

    def test_miss_shape_list_sorted(self):
        miss_details = [
            ("z_op", "unmapped", [(10, 20)]),
            ("a_op", "unmapped", [(30, 40)]),
        ]
        stats = compute_per_shape_stats([], miss_details)
        assert stats["m4_miss_shape_list"][0][0] == "a_op"
        assert stats["m4_miss_shape_list"][1][0] == "z_op"


# --- M5: Simulated Latency Coverage ---


def _make_op(shape_pairs):
    """Create a mock OpInvokeInfo with given tensor shapes."""
    mock = MagicMock()
    mock.func = torch.ops.aten.mm.default
    mock.args = tuple(torch.empty(*s, device="meta") for s in shape_pairs)
    return mock


def _make_device():
    mock = MagicMock()
    mock.name = "TEST_DEVICE"
    return mock


class ControlledDataSource(DataSourcePerformanceModel):
    """Data source that returns HIT for shapes in hit_set, MISS otherwise."""

    def __init__(self, hit_set: set):
        self.hit_set = hit_set
        self.last_miss_reason = "shape_mismatch"

    def lookup(self, op_invoke_info):
        shapes = tuple(
            tuple(a.shape) for a in op_invoke_info.args if isinstance(a, torch.Tensor)
        )
        if shapes in self.hit_set:
            return QueryResult(
                latency_us=100.0,
                confidence=1.0,
                source=QuerySource.MEASURED,
                details={"kernel_type": "MatMulV2"},
            )
        return None


class TestM5SimulatedLatencyCoverage:
    """M5: Roofline-latency-weighted coverage of HIT ops."""

    def _make_model(self, hit_shapes, analytic_latency_s=50e-6):
        device = _make_device()
        ds = ControlledDataSource(hit_shapes)
        fallback = MagicMock(spec=PerformanceModel)
        fallback.process_op.return_value = PerformanceModel.Result(
            execution_time_s=analytic_latency_s,
        )
        return EmpiricalPerformanceModel(
            device, data_source=ds, fallback_model=fallback
        )

    def test_all_hit(self):
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))
        c = MetricsCollector()
        c.collect_from_records(model.op_records)
        assert c._total_latency_sum > 0
        assert abs(c._hit_latency_sum / c._total_latency_sum - 1.0) < 1e-9

    def test_all_miss(self):
        model = self._make_model(hit_shapes=set())
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))
        c = MetricsCollector()
        c.collect_from_records(model.op_records)
        assert c._total_latency_sum > 0
        assert c._hit_latency_sum == 0.0

    def test_mixed_coverage(self):
        """2 HITs + 1 MISS, all same analytic weight -> M5 = 2/3."""
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT
        model.process_op(_make_op([(4096, 5120), (5120, 768)]))  # MISS
        c = MetricsCollector()
        c.collect_from_records(model.op_records)
        m5 = c._hit_latency_sum / c._total_latency_sum
        assert abs(m5 - 2.0 / 3.0) < 1e-9

    def test_weighted_by_analytic_latency(self):
        """HIT op 50us, MISS op 150us -> M5 = 50/200 = 0.25, not 0.5.

        Roofline weighting means a high-latency MISS drags M5 down more
        than a low-latency HIT pulls it up.
        """
        device = _make_device()
        hit_shape = ((2048, 5120), (5120, 768))
        ds = ControlledDataSource({hit_shape})

        call_count = [0]
        latencies = [50e-6, 150e-6]  # HIT gets 50us, MISS gets 150us

        fallback = MagicMock(spec=PerformanceModel)

        def side_effect(_op):
            idx = call_count[0]
            call_count[0] += 1
            return PerformanceModel.Result(execution_time_s=latencies[idx])

        fallback.process_op.side_effect = side_effect

        model = EmpiricalPerformanceModel(device, ds, fallback)
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT, 50us
        model.process_op(_make_op([(4096, 5120), (5120, 768)]))  # MISS, 150us
        c = MetricsCollector()
        c.collect_from_records(model.op_records)
        m5 = c._hit_latency_sum / c._total_latency_sum
        assert abs(m5 - 0.25) < 1e-9

    def test_partial_contributes_to_total_but_not_hit(self):
        """PARTIAL op counts toward M5 denominator (total) but not numerator (hit)."""
        device = _make_device()
        mock_ds = MagicMock(spec=DataSourcePerformanceModel)
        fallback = MagicMock(spec=PerformanceModel)
        fallback.process_op.return_value = PerformanceModel.Result(
            execution_time_s=50e-6,
        )

        model = EmpiricalPerformanceModel(device, mock_ds, fallback)

        # PARTIAL result
        mock_ds.lookup.return_value = QueryResult(
            latency_us=100.0,
            confidence=0.5,
            source=QuerySource.PARTIAL,
            details={
                "kernel_type": ["MatMulV2"],
                "missed_kernels": ["X"],
                "composite": True,
                "partial": True,
            },
        )
        op = MagicMock()
        op.func = torch.ops.aten.mm.default
        op.args = (torch.empty(2048, 5120, device="meta"),)
        model.process_op(op)

        # PARTIAL: analytic latency in total (denominator) but NOT in hit (numerator)
        c = MetricsCollector()
        c.collect_from_records(model.op_records)
        assert c._total_latency_sum == 50e-6
        assert c._hit_latency_sum == 0.0

    def test_empty(self):
        model = self._make_model(hit_shapes=set())
        c = MetricsCollector()
        c.collect_from_records(model.op_records)
        assert c._hit_latency_sum == 0.0
        assert c._total_latency_sum == 0.0


# --- export_hit_miss_report ---


class TestExportHitMissReport:
    """Tests for EmpiricalPerformanceModel.export_hit_miss_report()."""

    def _make_model(self, hit_shapes, analytic_latency_s=50e-6):
        device = _make_device()
        ds = ControlledDataSource(hit_shapes)
        fallback = MagicMock(spec=PerformanceModel)
        fallback.process_op.return_value = PerformanceModel.Result(
            execution_time_s=analytic_latency_s,
        )
        return EmpiricalPerformanceModel(
            device, data_source=ds, fallback_model=fallback
        )

    def test_report_structure(self):
        """Report contains all expected top-level keys."""
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT
        model.process_op(_make_op([(4096, 5120), (5120, 768)]))  # MISS

        collector = MetricsCollector()
        collector.collect_from_records(model.op_records)
        report = collector.export_hit_miss_report()

        assert "m1" in report
        assert "m2" in report
        assert "m3" in report
        assert "m4" in report
        assert "m5" in report
        assert "misses" in report
        # hits[] removed — per-op HIT data is in chrome trace
        assert "hits" not in report
        assert "m6_input" not in report

    def test_m1_keys(self):
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))
        model.process_op(_make_op([(4096, 5120), (5120, 768)]))

        collector = MetricsCollector()
        collector.collect_from_records(model.op_records)
        m1 = collector.export_hit_miss_report()["m1"]
        assert m1["m1_hit"] == 1
        assert m1["m1_miss"] == 1
        assert m1["m1_total"] == 2
        assert abs(m1["m1_raw_op_count_hr"] - 0.5) < 1e-9

    def test_write_json(self, tmp_path):
        """export_hit_miss_report writes valid JSON when output_path given."""
        import json

        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))

        collector = MetricsCollector()
        collector.collect_from_records(model.op_records)
        out = tmp_path / "report.json"
        collector.export_hit_miss_report(output_path=out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["m1"]["m1_hit"] == 1
        assert "m6_input" not in data

    def test_empty_report(self):
        """Report works with no ops processed."""
        model = self._make_model(hit_shapes=set())
        collector = MetricsCollector()
        collector.collect_from_records(model.op_records)
        report = collector.export_hit_miss_report()
        assert report["m1"]["m1_total"] == 0
        assert report["m5"]["m5_simulated_latency_coverage"] == 0.0
        assert "m6_input" not in report


# --- model_runner.py: profiling mode MetricsCollector path ---


class TestModelRunnerProfilingMetrics:
    """Verify model_runner.run_inference() triggers MetricsCollector.log_stats()
    via the external collector path when using EmpiricalPerformanceModel."""

    def _make_empirical_pm(self, hit_shapes):
        """Build an EmpiricalPerformanceModel with a controlled data source."""
        device = _make_device()
        ds = ControlledDataSource(hit_shapes)
        fallback = MagicMock(spec=PerformanceModel)
        fallback.process_op.return_value = PerformanceModel.Result(
            execution_time_s=50e-6,
        )
        fallback.get_classifiers.return_value = []
        return EmpiricalPerformanceModel(
            device, data_source=ds, fallback_model=fallback
        )

    def test_op_records_populated_after_process_op(self):
        """op_records is populated after process_op() calls — the data
        that model_runner feeds into MetricsCollector."""
        pm = self._make_empirical_pm(hit_shapes={((2048, 5120), (5120, 768))})

        pm.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT
        pm.process_op(_make_op([(4096, 5120), (5120, 768)]))  # MISS

        assert len(pm.op_records) == 2
        assert pm.op_records[0].lookup_result is not None  # HIT
        assert pm.op_records[1].lookup_result is None  # MISS

    def test_log_stats_called_via_external_collector(self, caplog):
        """Simulate the model_runner.py log path:
        MetricsCollector().collect_from_records(pm.op_records).log_stats()
        produces the expected log line."""
        import logging

        pm = self._make_empirical_pm(hit_shapes={((2048, 5120), (5120, 768))})
        pm.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT
        pm.process_op(_make_op([(4096, 5120), (5120, 768)]))  # MISS

        # Replicate exactly what model_runner.py does
        collector = MetricsCollector()
        collector.collect_from_records(pm.op_records)
        with caplog.at_level(logging.INFO):
            collector.log_stats()

        assert "1/2" in caplog.text or "ops matched" in caplog.text

    def test_collect_from_records_matches_direct_collect(self):
        """collect_from_records(pm.op_records) produces identical M1 stats
        to calling _collect_one() directly with the same data — verifies the
        model_runner path is equivalent to the old inline path."""
        from tensor_cast.performance_model.empirical import EmpiricalOpRecord

        hit_result = QueryResult(
            latency_us=100.0,
            confidence=1.0,
            source=QuerySource.MEASURED,
            details={"kernel_type": "MatMulV2"},
        )

        # New path: via op_records
        pm = self._make_empirical_pm(hit_shapes={((2048, 5120), (5120, 768))})
        pm.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT
        pm.process_op(_make_op([(4096, 5120), (5120, 768)]))  # MISS
        new_collector = MetricsCollector()
        new_collector.collect_from_records(pm.op_records)
        new_stats = new_collector.get_stats()

        # Equivalent path: direct collect_from_records with hand-built records
        old_collector = MetricsCollector()
        old_collector.collect_from_records(
            [
                EmpiricalOpRecord(
                    "aten.mm.default", hit_result, 50e-6, [(2048, 5120), (5120, 768)]
                ),
                EmpiricalOpRecord(
                    "aten.mm.default", None, 50e-6, [(4096, 5120), (5120, 768)]
                ),
            ]
        )
        old_stats = old_collector.get_stats()

        assert new_stats["hit"] == old_stats["hit"] == 1
        assert new_stats["miss"] == old_stats["miss"] == 1
        assert new_stats["m1_raw_op_count_hr"] == old_stats["m1_raw_op_count_hr"]


# --- compute_fused_op_stats unit tests ---


def test_fused_op_hr_groups_dfc_as_one():
    """DFC constituent ops should be counted as 1 fused op."""
    hit_details = [
        ("aten.mm.default", "MatMulV2", ((136, 5120), (5120, 768)), 45.3e-6),
        ("tensor_cast.swiglu.default", "SwiGlu", ((136, 6912),), 12.1e-6),
        ("aten.mm.default", "MatMulV2", ((136, 5120), (5120, 768)), 45.3e-6),
    ]
    miss_details = [
        ("tensor_cast.init_routing_v2.default", "csv_not_found", []),
        ("tensor_cast.grouped_matmul_quant_swiglu.default", "csv_not_found", []),
        ("tensor_cast.unpermute_tokens.default", "csv_not_found", []),
        ("tensor_cast.all_to_all.default", "csv_not_found", []),
        ("aten.embedding.default", "shape_mismatch", []),
    ]
    fused_groups = {
        "DispatchFFNCombine": [
            "tensor_cast.init_routing_v2",
            "tensor_cast.grouped_matmul",
            "tensor_cast.unpermute_tokens",
            "tensor_cast.all_to_all",
        ],
    }

    stats = compute_fused_op_stats(hit_details, miss_details, fused_groups)

    assert stats["m2_fused_total"] == 4
    assert stats["m2_fused_hit"] == 2
    assert stats["m2_fused_miss"] == 2


def test_fused_op_hr_excludes_zero_cost():
    """Reference view should exclude zero_cost ops from count."""
    hit_details = [
        ("aten.mm.default", "MatMulV2", ((136, 5120), (5120, 768)), 45.3e-6),
        ("aten.view.default", "zero_cost", ((136, 5120),), 0.0),
        ("aten.permute.default", "zero_cost", ((136, 5120),), 0.0),
    ]
    miss_details = [
        ("aten.embedding.default", "shape_mismatch", []),
    ]

    stats = compute_fused_op_stats(hit_details, miss_details, fused_groups={})

    assert stats["m2_fused_total"] == 4
    assert stats["m2_fused_hit"] == 3
    assert stats["m3_fused_total_no_zc"] == 2
    assert stats["m3_fused_hit_no_zc"] == 1


def test_fused_op_hr_pessimistic_partial_shape():
    """Op that HITs for some shapes and MISSes for others → MISS (pessimistic)."""
    hit_details = [
        ("tensor_cast.quantize.default", "AscendQuantV2", ((8, 16, 128),), 9.8e-6),
        ("aten.mm.default", "MatMulV2", ((136, 5120), (5120, 768)), 45.3e-6),
        ("aten.view.default", "zero_cost", ((136, 5120),), 0.0),
    ]
    miss_details = [
        ("tensor_cast.quantize.default", "shape_mismatch", [(16, 128)]),
        ("aten.mm.default", "shape_mismatch", [(16, 7168)]),
        ("aten.embedding.default", "shape_mismatch", [(9496, 5120)]),
    ]

    stats = compute_fused_op_stats(hit_details, miss_details, fused_groups={})

    assert stats["m2_fused_total"] == 4
    assert stats["m2_fused_hit"] == 1
    assert stats["m2_fused_miss"] == 3
    assert stats["m3_fused_hit_no_zc"] == 0
    assert stats["m3_fused_total_no_zc"] == 3

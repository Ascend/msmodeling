"""Synthetic CI guard for profiling interpolation on/off metrics."""

from tensor_cast.performance_model.empirical import EmpiricalOpRecord
from tensor_cast.performance_model.metrics_collector import MetricsCollector
from tensor_cast.performance_model.profiling_database.data_source import QueryResult, QuerySource


def _hit(kernel_type: str, source: QuerySource = QuerySource.MEASURED) -> QueryResult:
    return QueryResult(
        latency_us=10.0,
        confidence=0.7,
        source=source,
        details={"kernel_type": kernel_type},
    )


def _metrics(records: list[EmpiricalOpRecord]) -> dict:
    collector = MetricsCollector()
    collector.collect_from_records(records)
    return collector.export_hit_miss_report()


def test_interpolation_enabled_metrics_do_not_regress_against_disabled():
    disabled = [
        EmpiricalOpRecord("aten.mm.default", _hit("MatMulV2"), 10e-6, [(128, 64), (64, 256)]),
        EmpiricalOpRecord("aten.add.Tensor", None, 5e-6, [(128, 7168), (7168,)], "exact_miss"),
        EmpiricalOpRecord("tensor_cast.attention.default", None, 20e-6, [(2, 512)], "exact_miss"),
    ]
    enabled = [
        EmpiricalOpRecord("aten.mm.default", _hit("MatMulV2"), 10e-6, [(128, 64), (64, 256)]),
        EmpiricalOpRecord(
            "aten.add.Tensor",
            _hit("Add", QuerySource.INTERPOLATED),
            5e-6,
            [(128, 7168), (7168,)],
        ),
        EmpiricalOpRecord("tensor_cast.attention.default", None, 20e-6, [(2, 512)], "exact_miss"),
    ]

    disabled_metrics = _metrics(disabled)
    enabled_metrics = _metrics(enabled)

    assert enabled_metrics["m1"]["m1_raw_op_count_hr"] >= disabled_metrics["m1"]["m1_raw_op_count_hr"]
    assert enabled_metrics["m3"]["m3_fused_op_hr_no_zc"] >= disabled_metrics["m3"]["m3_fused_op_hr_no_zc"]
    assert (
        enabled_metrics["m5"]["m5_simulated_latency_coverage"]
        >= disabled_metrics["m5"]["m5_simulated_latency_coverage"]
    )

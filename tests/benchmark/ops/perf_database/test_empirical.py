from unittest.mock import MagicMock

import torch
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.empirical import EmpiricalPerformanceModel
from tensor_cast.performance_model.profiling_database.data_source import (
    DataSourcePerformanceModel,
    QueryResult,
    QuerySource,
    ShapeMatchInfo,
    SubKernelShapeInfo,
)


class HitDataSource(DataSourcePerformanceModel):
    def lookup(self, op_invoke_info):
        return QueryResult(
            latency_us=45.3,
            confidence=1.0,
            source=QuerySource.MEASURED,
            details={"kernel_type": "MatMulV2"},
        )


class MissDataSource(DataSourcePerformanceModel):
    def lookup(self, op_invoke_info):
        return None


def _make_mock_op_invoke_info():
    mock = MagicMock()
    mock.func = torch.ops.aten.mm.default
    mock.args = (
        torch.empty(136, 5120, device="meta"),
        torch.empty(5120, 768, device="meta"),
    )
    return mock


def _make_mock_device_profile():
    mock = MagicMock()
    mock.name = "TEST_DEVICE"
    return mock


def test_empirical_uses_datasource_when_hit():
    """data_source.lookup() hit → use measured latency."""
    device = _make_mock_device_profile()
    fallback = MagicMock(spec=PerformanceModel)
    fallback.process_op.return_value = PerformanceModel.Result(execution_time_s=200e-6)
    model = EmpiricalPerformanceModel(device, data_source=HitDataSource(), fallback_model=fallback)
    result = model.process_op(_make_mock_op_invoke_info())
    assert abs(result.execution_time_s - 45.3e-6) < 1e-12
    assert result.statistics.get("source") == "MEASURED"
    assert result.statistics.get("kernel_type") == "MatMulV2"
    # M5: fallback also called for analytic weight
    fallback.process_op.assert_called_once()


def test_empirical_falls_back_when_miss():
    """data_source.lookup() miss → fallback_model.process_op()."""
    device = _make_mock_device_profile()
    fallback = MagicMock(spec=PerformanceModel)
    fallback.process_op.return_value = PerformanceModel.Result(execution_time_s=100e-6)

    model = EmpiricalPerformanceModel(
        device,
        data_source=MissDataSource(),
        fallback_model=fallback,
    )
    result = model.process_op(_make_mock_op_invoke_info())
    fallback.process_op.assert_called_once()
    assert abs(result.execution_time_s - 100e-6) < 1e-12


def test_hit_with_shape_match_info_populates_statistics():
    """HIT with shape_match_info → statistics contains kernel_shapes and shape_match_rule."""

    class ShapeHitDataSource(DataSourcePerformanceModel):
        def lookup(self, op_invoke_info):
            return QueryResult(
                latency_us=45.3,
                confidence=1.0,
                source=QuerySource.MEASURED,
                details={"kernel_type": "MatMulV2"},
                shape_match_info=ShapeMatchInfo(
                    simulation_shapes=[[136, 5120]],
                    kernel_shapes=[[128, 5120]],
                    shape_match_rule="padding",
                ),
            )

    device = _make_mock_device_profile()
    fallback = MagicMock(spec=PerformanceModel)
    fallback.process_op.return_value = PerformanceModel.Result(execution_time_s=200e-6)
    model = EmpiricalPerformanceModel(device, data_source=ShapeHitDataSource(), fallback_model=fallback)
    result = model.process_op(_make_mock_op_invoke_info())
    assert result.statistics["kernel_shapes"] == "[[128, 5120]]"
    assert result.statistics["shape_match_rule"] == "padding"


def test_hit_with_sub_kernel_shapes_populates_statistics():
    """HIT with sub_kernel_shapes → statistics contains JSON-encoded sub_kernel_shapes."""
    import json

    class CompositeHitDataSource(DataSourcePerformanceModel):
        def lookup(self, op_invoke_info):
            return QueryResult(
                latency_us=60.0,
                confidence=0.9,
                source=QuerySource.MEASURED,
                details={"kernel_type": "composite"},
                sub_kernel_shapes=[
                    SubKernelShapeInfo(
                        kernel_type="MatMulV2",
                        simulation_shapes=[[136, 5120]],
                        kernel_shapes=[[128, 5120]],
                        shape_match_rule="padding",
                    )
                ],
            )

    device = _make_mock_device_profile()
    fallback = MagicMock(spec=PerformanceModel)
    fallback.process_op.return_value = PerformanceModel.Result(execution_time_s=200e-6)
    model = EmpiricalPerformanceModel(device, data_source=CompositeHitDataSource(), fallback_model=fallback)
    result = model.process_op(_make_mock_op_invoke_info())
    assert "sub_kernel_shapes" in result.statistics
    parsed = json.loads(result.statistics["sub_kernel_shapes"])
    assert len(parsed) == 1
    assert parsed[0]["kernel_type"] == "MatMulV2"
    assert parsed[0]["shape_match_rule"] == "padding"


def test_miss_sets_shape_match_rule_analytic():
    """MISS path → statistics['shape_match_rule'] == 'analytic'."""
    device = _make_mock_device_profile()
    fallback = MagicMock(spec=PerformanceModel)
    fallback.process_op.return_value = PerformanceModel.Result(execution_time_s=100e-6)
    model = EmpiricalPerformanceModel(device, data_source=MissDataSource(), fallback_model=fallback)
    result = model.process_op(_make_mock_op_invoke_info())
    assert result.statistics.get("shape_match_rule") == "analytic"


# --- C5: Interpolation toggle tests ---


def test_interpolating_data_source_wraps_profiling():
    """Interpolated lookup populates shape_match metadata for empirical results."""
    from tensor_cast.performance_model.profiling_database.interpolating_data_source import (
        InterpolatingDataSource,
    )

    base_ds = MagicMock()
    base_ds._op_mapping = {}
    base_ds.lookup.return_value = None

    wrapped_ds = InterpolatingDataSource(base_ds)
    wrapped_ds._interpolate = MagicMock(
        return_value=QueryResult(
            latency_us=15.0,
            confidence=0.7,
            source=QuerySource.INTERPOLATED,
            details={"kernel_type": "MatMulV2", "method": "linear_1d"},
        )
    )

    device = _make_mock_device_profile()
    fallback = MagicMock(spec=PerformanceModel)
    fallback.process_op.return_value = PerformanceModel.Result(execution_time_s=200e-6)
    model = EmpiricalPerformanceModel(device, data_source=wrapped_ds, fallback_model=fallback)

    result = model.process_op(_make_mock_op_invoke_info())

    assert abs(result.execution_time_s - 15.0e-6) < 1e-12
    assert result.statistics["source"] == "INTERPOLATED"
    assert result.statistics["kernel_type"] == "MatMulV2"
    assert result.statistics["shape_match_rule"] == "interpolated"
    base_ds.lookup.assert_called_once()
    wrapped_ds._interpolate.assert_called_once()
    fallback.process_op.assert_called_once()

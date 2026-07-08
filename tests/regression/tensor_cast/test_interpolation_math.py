import math

import pytest

from tensor_cast.performance_model.profiling_database.interpolation_math import (
    find_boundary,
    griddata_linear_interp,
    linear_interp,
    validate_interpolated_latency,
    validate_positive_latency,
)


def test_interpolation_index_definitions_are_covered_under_test_context():
    import importlib
    import sys

    original = importlib.import_module("tensor_cast.performance_model.profiling_database.interpolation_index")
    module_name = "tensor_cast.performance_model.profiling_database._test_coverage_interpolation_index"
    spec = importlib.util.spec_from_file_location(module_name, original.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    regime_key = module.make_regime_key({"kernel": "MatMulV2", "dtype": "DT_BF16"})
    point = module.CandidatePoint(
        kernel_type="MatMulV2",
        axes={"M": 128.0},
        latency_us=10.0,
        regime_key=regime_key,
    )
    target = module.InterpolationTarget(
        func_name="aten.mm.default",
        kernel_type="MatMulV2",
        axes={"M": 192.0},
        regime_key=regime_key,
    )
    result = module.InterpolationResult(
        latency_us=12.0,
        confidence=0.8,
        method="linear_1d",
        interpolation_dim=1,
        axes=("M",),
        details={},
        shape_match_rule="interpolated_1d_linear",
        matched_points=[point],
    )
    group = module.CandidateGroup(regime_key, [point])
    candidate_index = module.CandidateIndex([point])

    assert target.kernel_type == "MatMulV2"
    assert result.matched_points == [point]
    assert group.points == [point]
    assert candidate_index.points == [point]


def test_find_boundary_inside_and_outside_range():
    assert find_boundary([100, 200, 400], 300) == (200.0, 400.0)
    assert find_boundary([100, 200, 400], 100) == (100.0, 100.0)
    assert find_boundary([100, 200, 400], 50) is None
    assert find_boundary([100, 200, 400], 500) is None


def test_linear_interp():
    assert linear_interp(150, 100, 10, 200, 20) == pytest.approx(15.0)


def test_griddata_linear_interp_2d_and_outside_hull():
    pytest.importorskip("scipy")
    points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    values = [1.0, 2.0, 11.0, 12.0]
    assert griddata_linear_interp(points, values, (0.5, 0.5)) == pytest.approx(6.5)
    assert griddata_linear_interp(points, values, (2.0, 2.0)) is None


def test_griddata_linear_interp_3d():
    pytest.importorskip("scipy")
    points = []
    values = []
    for x in (0.0, 1.0):
        for y in (0.0, 1.0):
            for z in (0.0, 1.0):
                points.append((x, y, z))
                values.append(x + 10 * y + 100 * z)

    assert griddata_linear_interp(points, values, (0.5, 0.5, 0.5)) == pytest.approx(55.5)


def test_griddata_linear_interp_rejects_degenerate_points():
    pytest.importorskip("scipy")
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    values = [0.0, 1.0, 2.0]
    assert griddata_linear_interp(points, values, (0.5, 0.5)) is None


def test_griddata_linear_interp_falls_back_when_scipy_unavailable(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def reject_scipy_interpolate(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.interpolate":
            raise ImportError("scipy unavailable in this environment")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_scipy_interpolate)

    value, details = griddata_linear_interp(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        [0.0, 1.0, 10.0],
        (0.5, 0.5),
        return_details=True,
    )

    assert value is None
    assert details["failure_reason"] == "scipy_unavailable"
    assert details["exception_type"] == "ImportError"


def test_griddata_linear_interp_warns_and_falls_back_on_memory_error(monkeypatch, caplog):
    scipy_interpolate = pytest.importorskip("scipy.interpolate")

    def raise_memory_error(*args, **kwargs):
        raise MemoryError("unexpected allocation failure")

    monkeypatch.setattr(scipy_interpolate, "LinearNDInterpolator", raise_memory_error)
    caplog.set_level("WARNING", logger="tensor_cast.performance_model.profiling_database.interpolation_math")

    value, details = griddata_linear_interp(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        [0.0, 1.0, 10.0],
        (0.5, 0.5),
        return_details=True,
    )

    assert value is None
    assert details["failure_reason"] == "scipy_exception"
    assert details["exception_type"] == "MemoryError"
    warning_messages = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any("griddata_linear_interp encountered MemoryError" in message for message in warning_messages)
    assert all(record.exc_info is None for record in caplog.records if record.levelname == "WARNING")


def test_griddata_linear_interp_debug_logs_non_memory_exception(monkeypatch, caplog):
    scipy_interpolate = pytest.importorskip("scipy.interpolate")

    def raise_value_error(*args, **kwargs):
        raise ValueError("bad triangulation")

    monkeypatch.setattr(scipy_interpolate, "LinearNDInterpolator", raise_value_error)
    caplog.set_level("DEBUG", logger="tensor_cast.performance_model.profiling_database.interpolation_math")

    value, details = griddata_linear_interp(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        [0.0, 1.0, 10.0],
        (0.5, 0.5),
        return_details=True,
    )

    assert value is None
    assert details["failure_reason"] == "scipy_exception"
    assert details["exception_type"] == "ValueError"
    debug_records = [record for record in caplog.records if record.levelname == "DEBUG"]
    assert any("griddata_linear_interp scipy exception" in record.getMessage() for record in debug_records)
    assert any(record.exc_info is not None for record in debug_records)


def test_validate_positive_latency_rejects_invalid_values():
    assert not validate_positive_latency(0.0)
    assert validate_positive_latency(1.2)
    assert not validate_positive_latency(-1.0)
    assert not validate_positive_latency(math.nan)
    assert not validate_positive_latency(math.inf)


def test_validate_interpolated_latency_requires_positive_value_and_candidate():
    assert not validate_interpolated_latency(0.0, [10.0, 20.0])
    assert validate_interpolated_latency(0.5, [10.0, 20.0])
    assert validate_interpolated_latency(10.0, [10.0, 20.0])
    assert not validate_interpolated_latency(10.0, [0.0, math.nan])

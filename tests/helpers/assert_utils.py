"""Custom assertions for tensor and latency checks."""

import math

import torch


def assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> None:
    """Assert two tensors are element-wise close (torch.testing.assert_close semantics)."""
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)


def assert_latency_within(
    actual_ms: float,
    expected_ms: float,
    *,
    metric: str = "latency",
    tolerance_ms: float | None = None,
    rel_tolerance: float = 0.0,
) -> None:
    """Assert latency is within absolute and/or relative tolerance.

    When ``tolerance_ms`` and ``rel_tolerance`` are both unset/zero, uses exact
    match with a small float epsilon for deterministic unit tests. Callers
    comparing noisy measurements must pass ``tolerance_ms`` and/or ``rel_tolerance``.
    """
    if actual_ms < 0 or expected_ms < 0:
        raise AssertionError(f"Latency must be non-negative, got actual={actual_ms}, expected={expected_ms}")

    if tolerance_ms is None and rel_tolerance == 0:
        if not math.isclose(actual_ms, expected_ms, rel_tol=0.0, abs_tol=1e-9):
            delta = abs(actual_ms - expected_ms)
            raise AssertionError(
                f"{metric} out of range: metric={metric}, baseline={expected_ms}, "
                f"actual={actual_ms}, delta={delta}, allowed=1e-9"
            )
        return

    allowed_abs = 0.0 if tolerance_ms is None else tolerance_ms
    if rel_tolerance > 0:
        allowed_delta = max(allowed_abs, rel_tolerance * max(abs(actual_ms), abs(expected_ms)))
    else:
        allowed_delta = allowed_abs
    delta = abs(actual_ms - expected_ms)
    if delta > allowed_delta:
        raise AssertionError(
            f"{metric} out of range: metric={metric}, baseline={expected_ms}, "
            f"actual={actual_ms}, delta={delta}, allowed={allowed_delta}"
        )

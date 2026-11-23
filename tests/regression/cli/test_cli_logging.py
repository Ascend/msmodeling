from __future__ import annotations

import argparse
import logging

import pytest

from cli.utils import LOG_FORMAT, LOG_LEVELS, check_device_targets, configure_logging
from tensor_cast.device import DeviceProfile


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Isolate tests from force=True's effect on the process-wide root logger.

    configure_logging() calls ``logging.basicConfig(force=True)``, which removes
    and closes existing root handlers (including pytest's log-capture handler) and
    leaves a new level + stderr handler behind. That state leaks across tests and
    makes log capture order-dependent. Save the original handlers and level, then
    restore them -- and close any handler a test created -- afterwards.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        for handler in root.handlers:
            if handler not in saved_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_configure_logging_default_level_is_error() -> None:
    configure_logging()
    assert logging.getLogger().level == logging.ERROR


@pytest.mark.parametrize(
    ("log_level", "expected"),
    [
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
        ("critical", logging.CRITICAL),
    ],
)
def test_configure_logging_applies_requested_level(log_level: str, expected: int) -> None:
    configure_logging(log_level)
    assert logging.getLogger().level == expected


def test_configure_logging_normalizes_unknown_level_to_error() -> None:
    configure_logging("nonsense")
    assert logging.getLogger().level == logging.ERROR


def test_configure_logging_uses_consistent_format() -> None:
    configure_logging("info")
    root = logging.getLogger()
    assert root.handlers, "expected a root handler after configure_logging"
    formatter = root.handlers[0].formatter
    assert formatter is not None
    assert formatter._fmt == LOG_FORMAT


def test_log_levels_mapping_is_complete() -> None:
    assert LOG_LEVELS == {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }


@pytest.fixture
def device_profiles(monkeypatch: pytest.MonkeyPatch) -> dict:
    profiles = {"TEST_DEVICE": object()}
    monkeypatch.setattr(DeviceProfile, "all_device_profiles", profiles, raising=False)
    return profiles


def test_check_device_targets_logs_unknown_device(
    caplog: pytest.LogCaptureFixture,
    device_profiles: dict,
) -> None:
    caplog.set_level(logging.ERROR)
    args = argparse.Namespace(device=["unknown-device"], num_devices=1)
    result = check_device_targets(args, logging.getLogger("cli.utils.test"))

    assert result is None
    assert any(record.levelno == logging.ERROR and "Unknown" in record.message for record in caplog.records)

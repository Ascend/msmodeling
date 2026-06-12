from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace
from unittest import TestCase

import pytest

from cli.utils import (
    check_device_targets,
    check_non_negative_integer,
    check_prefix_cache_hit_rate,
    check_positive_integer,
    check_string_valid,
    get_common_argparser,
    parse_int_range,
)
from tensor_cast.device import DeviceProfile


class _DummyGrid:
    def __init__(self, size: int) -> None:
        self._size = size

    def nelement(self) -> int:
        return self._size


class _DummyProfile:
    def __init__(self, size: int) -> None:
        self.comm_grid = SimpleNamespace(grid=_DummyGrid(size))


class TestCliUtils(TestCase):
    def test_common_argparser_reserved_memory_default_is_zero(self):
        parser = get_common_argparser()

        args = parser.parse_args(["Qwen/Qwen3-32B"])

        self.assertEqual(args.reserved_memory_gb, 0.0)

    def test_common_argparser_reserved_memory_default_can_be_overridden(self):
        parser = get_common_argparser(reserved_memory_gb_default=10.0)

        args = parser.parse_args(["Qwen/Qwen3-32B"])

        self.assertEqual(args.reserved_memory_gb, 10.0)


@pytest.fixture
def device_profiles(monkeypatch: pytest.MonkeyPatch) -> dict[str, _DummyProfile]:
    profiles = {
        "TEST_DEVICE": _DummyProfile(4),
        "NPU_A": _DummyProfile(8),
    }
    monkeypatch.setattr(DeviceProfile, "all_device_profiles", profiles, raising=False)
    return profiles


def test_common_argparser_parses_device_and_num_devices(device_profiles: dict[str, _DummyProfile]) -> None:
    parser = get_common_argparser(reserved_memory_gb_default=10.0)

    args = parser.parse_args(
        [
            "Qwen/Qwen3-32B",
            "--device",
            "TEST_DEVICE",
            "--num-devices",
            "2",
            "--log-level",
            "debug",
        ]
    )

    assert args.device == "TEST_DEVICE"
    assert args.num_devices == 2
    assert args.log_level == "debug"
    assert args.reserved_memory_gb == 10.0


@pytest.mark.parametrize("value", ["1", "100", 5])
def test_check_positive_integer_accepts_valid_values(value: int | str) -> None:
    assert check_positive_integer(value) == int(value)


@pytest.mark.parametrize("value", ["abc", "0", "-1"])
def test_check_positive_integer_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        check_positive_integer(value)


@pytest.mark.parametrize("value", ["0", "1", "42"])
def test_check_non_negative_integer_accepts_valid_values(value: str) -> None:
    assert check_non_negative_integer(value) == int(value)


@pytest.mark.parametrize("value", ["abc", "-1"])
def test_check_non_negative_integer_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        check_non_negative_integer(value)


@pytest.mark.parametrize("value", ["0", "0.5", "0.999999"])
def test_check_prefix_cache_hit_rate_accepts_valid_values(value: str) -> None:
    assert check_prefix_cache_hit_rate(value) == pytest.approx(float(value))


@pytest.mark.parametrize("value", ["1", "-0.1", "abc"])
def test_check_prefix_cache_hit_rate_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        check_prefix_cache_hit_rate(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1,2", (1, 2)),
        (" 0 , 4 ", (0, 4)),
    ],
)
def test_parse_int_range_accepts_valid_values(value: str, expected: tuple[int, int]) -> None:
    assert parse_int_range(value, "--range") == expected


@pytest.mark.parametrize(
    "value",
    [
        "1",
        "1,",
        ",2",
        "a,b",
        "-1,2",
        "3,2",
    ],
)
def test_parse_int_range_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_int_range(value, "--range")


@pytest.mark.parametrize("value", ["valid_string123/test-path.file", "abc/DEF-123.txt"])
def test_check_string_valid_accepts_valid_values(value: str) -> None:
    assert check_string_valid(value, max_len=100) == value


@pytest.mark.parametrize("value", ["invalid value", "bad#value"])
def test_check_string_valid_rejects_invalid_characters(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        check_string_valid(value)


def test_check_string_valid_rejects_overlong_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        check_string_valid("x" * 257)


def test_check_device_targets_returns_default_and_dedupes(device_profiles: dict[str, _DummyProfile]) -> None:
    args = argparse.Namespace(device=None, num_devices=2)
    logger = logging.getLogger("cli.utils.test")

    result = check_device_targets(args, logger)

    assert result == ["TEST_DEVICE"]
    assert args.device == ["TEST_DEVICE"]

    args = argparse.Namespace(device=["NPU_A", "NPU_A", "TEST_DEVICE"], num_devices=2)
    result = check_device_targets(args, logger)

    assert result == ["NPU_A", "TEST_DEVICE"]


def test_check_device_targets_rejects_missing_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DeviceProfile, "all_device_profiles", {}, raising=False)
    args = argparse.Namespace(device=["TEST_DEVICE"], num_devices=1)

    assert check_device_targets(args, logging.getLogger("cli.utils.test")) is None


@pytest.mark.parametrize(
    "device_values",
    [
        [""],
        ["unknown-device"],
    ],
)
def test_check_device_targets_rejects_blank_and_unknown_devices(
    device_profiles: dict[str, _DummyProfile],
    device_values: list[str],
) -> None:
    args = argparse.Namespace(device=device_values, num_devices=1)

    assert check_device_targets(args, logging.getLogger("cli.utils.test")) is None


def test_check_device_targets_rejects_undersized_comm_grid(device_profiles: dict[str, _DummyProfile]) -> None:
    args = argparse.Namespace(device=["TEST_DEVICE"], num_devices=8)

    assert check_device_targets(args, logging.getLogger("cli.utils.test")) is None

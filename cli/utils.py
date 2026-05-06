import argparse
import logging
import re

from tensor_cast.device import DeviceProfile


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
LOG_FORMAT = "[%(levelname)s] [%(name)s] %(message)s"


def check_positive_integer(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid integer value: %r", value) from None
    if value <= 0:
        raise argparse.ArgumentTypeError("%r is not a positive integer", value)

    return value


def check_prefix_cache_hit_rate(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid float value for prefix cache hit rate: %r", value
        ) from None
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("%r is not in the valid range [0, 1)", value)
    return value


def parse_int_range(value: str, name: str) -> tuple[int, int]:
    """Parse a range string in the form 'start,end'.

    Semantics:
    - Surrounding spaces are allowed around both numbers.
    - Both values must be integers and non-negative.
    - `end` must be greater than or equal to `start`.

    Args:
        value: Raw CLI string, for example '11,45' or ' 0 , 54 '.
        name: Argument name used in error messages, for example '--cache-step-range'.

    Returns:
        A tuple `(start, end)`.

    Raises:
        ValueError: If input format or bounds are invalid.
    """
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{name} must be 'start,end', got {value!r}.")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{name} must be 'start,end', got {value!r}.") from exc
    if start < 0 or end < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    if end < start:
        raise ValueError(
            f"{name} must be 'start,end' with end >= start, got {value!r}."
        )
    return start, end


def check_string_valid(string: str, max_len=256):
    if len(string) > max_len:
        raise argparse.ArgumentTypeError(
            "String length exceeds %d characters: %r", max_len, string
        )
    if not re.match(r"^[a-zA-Z0-9_/.-]+$", string):
        raise argparse.ArgumentTypeError(
            "String contains invalid characters: %r", string
        )
    return string


def get_common_argparser(reserved_memory_gb_default: float = 0.0):
    common_parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general_group = common_parser.add_argument_group("General Options")

    general_group.add_argument(
        "model_id",
        type=check_string_valid,
        help="The model identifier, which can be: "
        "1) A Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b-hf') to load from the Hub; "
        "2) A local directory path containing a diffusers model (must include 'transformer/config.json').",
    )

    general_group.add_argument(
        "--device",
        type=str,
        choices=list(DeviceProfile.all_device_profiles.keys()),
        default="TEST_DEVICE",
        help=(
            "Specifies the target device profile to use for benchmarking and simulation. "
            "Must be a valid device name as defined in DeviceProfile. "
            "The default device 'TEST_DEVICE' is used for standard simulation runs."
        ),
    )

    general_group.add_argument(
        "--num-devices",
        type=check_positive_integer,
        default=1,
        help=(
            "Specifies the total number of devices/processes to use. "
            "Must be a positive integer. "
            "A value of 1 indicates single-device execution."
        ),
    )

    general_group.add_argument(
        "--enable-multistream",
        action="store_true",
        default=True,
        help=(
            "Enable compiler-driven multi-stream simulation for torch.compile path. "
            "Enabled by default."
        ),
    )

    general_group.add_argument(
        "--reserved-memory-gb",
        type=float,
        default=reserved_memory_gb_default,
        help=(
            "Amount of device memory (in gigabytes) reserved for system usage and unavailable for application. "
            "Set to 0 to disable memory reservation."
        ),
    )

    general_group.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="error",
        help=(
            "Specifies the verbosity level for log output. "
            "Available levels: 'debug' (most verbose), 'info', 'warning', 'error', 'critical' (least verbose)."
        ),
    )

    return common_parser

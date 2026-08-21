import argparse
import logging
import re
import sys
from typing import Any, Optional

from cli.spec_cli import (
    METAVAR_FLOAT,
    METAVAR_N,
    METAVAR_NAME,
    add_log_options,
    add_option,
    add_version_option,
)
from tensor_cast.device import DeviceProfile

DRAFT_METHODS = ("dflash", "dspark")

_METHOD_DEPENDENT_OPTIONS = (
    "--num-speculative-tokens",
    "--acceptance-length",
    "--num-draft-layers",
    "--draft-model-config-path",
)
_DSPARK_ONLY_OPTIONS = (
    "--dspark-markov-rank",
    "--dspark-markov-head",
)

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
LOG_FORMAT = "[%(levelname)s] [%(name)s] %(message)s"


def check_device_targets(args: argparse.Namespace, logger: logging.Logger) -> list[str] | None:
    """Validate ``--device``: default if omitted, de-dupe, reject invalid names, check comm grid."""
    profiles = DeviceProfile.all_device_profiles
    if not profiles:
        logger.error(
            "No device profiles are registered. Import tensor_cast.device_profiles before defining CLI defaults."
        )
        return None

    if not args.device:
        args.device = ["TEST_DEVICE"]
        logger.info("No --device specified; using default profile %r.", args.device[0])

    targets = list(dict.fromkeys(args.device))

    blank = [name for name in targets if not str(name).strip()]
    if blank:
        logger.error("Empty --device name is not allowed.")
        return None

    unknown = [name for name in targets if name not in profiles]
    if unknown:
        logger.error(
            "Unknown --device name(s): %s. Valid profiles: %s",
            ", ".join(repr(name) for name in unknown),
            ", ".join(sorted(profiles.keys())),
        )
        return None

    for name in targets:
        grid_n = profiles[name].comm_grid.grid.nelement()
        if grid_n < args.num_devices:
            logger.error(
                "Device profile %r cannot model num_devices=%s (communication grid size is %s).",
                name,
                args.num_devices,
                grid_n,
            )
            return None

    return targets


def check_positive_integer(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a positive integer")

    return value


def check_non_negative_integer(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a non-negative integer")

    return value


def check_prefix_cache_hit_rate(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value for prefix cache hit rate: {value!r}") from None
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError(f"{value!r} is not in the valid range [0, 1)")
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
        raise ValueError(f"{name} must be 'start,end' with end >= start, got {value!r}.")
    return start, end


def check_string_valid(string: str, max_len=256):
    if len(string) > max_len:
        raise argparse.ArgumentTypeError(f"String length exceeds {max_len} characters: {string!r}")
    if not re.match(r"^[a-zA-Z0-9_/.-]+$", string):
        raise argparse.ArgumentTypeError(f"String contains invalid characters: {string!r}")
    return string


def get_common_argparser(reserved_memory_gb_default: float = 0.0):
    common_parser = argparse.ArgumentParser(add_help=False)
    add_version_option(common_parser)

    general_group = common_parser.add_argument_group("General Options")
    add_model_id_source(
        general_group,
        positional_help=(
            "Model source. Recommended safe mode: a reviewed absolute local model path. "
            "Model id mode also accepts Hugging Face or ModelScope ids, but may execute remote Python code through "
            "trust_remote_code=True and is not security-guaranteed. Equivalent to --model-id."
        ),
        option_help=(
            "Model source. Recommended safe mode: a reviewed absolute local model path. "
            "Equivalent to the positional model id."
        ),
    )

    general_group.add_argument(
        "--device",
        type=str,
        choices=list(DeviceProfile.all_device_profiles.keys()),
        default="TEST_DEVICE",
        metavar=METAVAR_NAME,
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
        metavar=METAVAR_N,
        help=(
            "Specifies the total number of devices/processes to use. "
            "Must be a positive integer. "
            "A value of 1 indicates single-device execution."
        ),
    )

    general_group.add_argument(
        "--reserved-memory-gb",
        type=float,
        default=reserved_memory_gb_default,
        metavar=METAVAR_FLOAT,
        help=(
            "Amount of device memory (in gigabytes) reserved for system usage and unavailable for application. "
            "Set to 0 to disable memory reservation."
        ),
    )

    add_log_options(general_group)
    return common_parser


def add_model_id_source(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    *,
    positional_help: str,
    option_help: str | None = None,
    value_type: Any = check_string_valid,
    public_snake_alias: bool = False,
) -> None:
    """Register positional model source and ``--model-id`` on separate dests.

    Positional dest is ``model_id_positional``. Option dest is ``model_id``.
    ``require_model_id`` merges them. ``--model_id`` is a hidden alias unless
    ``public_snake_alias`` is true (model-adapter dual public names).
    """
    if option_help is None:
        option_help = "Model source. Equivalent to the positional model id."
    parser.add_argument(
        "model_id_positional",
        nargs="?",
        metavar=METAVAR_NAME,
        type=value_type,
        help=positional_help,
    )
    if public_snake_alias:
        parser.add_argument(
            "--model-id",
            "--model_id",
            dest="model_id",
            type=value_type,
            default=None,
            metavar=METAVAR_NAME,
            help=option_help,
        )
        return
    add_option(
        parser,
        "--model-id",
        dest="model_id",
        type=value_type,
        default=None,
        metavar=METAVAR_NAME,
        help=option_help,
        aliases=("--model_id",),
    )


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def require_model_id(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    option = getattr(args, "model_id", None)
    positional = getattr(args, "model_id_positional", None)
    if option and positional:
        parser.error("pass either a positional model id or --model-id, not both")
    model_id = option or positional
    if not model_id:
        if _parser_has_option(parser, "--model-id"):
            parser.error("model_id is required; pass a positional model id or use --model-id <MODEL_ID>.")
        parser.error("model_id is required; pass a positional model id.")
    args.model_id = model_id
    if hasattr(args, "model_id_positional"):
        delattr(args, "model_id_positional")


def argv_has_option(argv, *option_names: str) -> bool:
    """Return True if any option name appears in argv (supports ``--opt=value``)."""
    names = set(option_names)
    for arg in argv:
        if arg.split("=", 1)[0] in names:
            return True
    return False


def draft_method(args) -> Optional[str]:
    """Return ``dflash`` / ``dspark`` / None from ``--speculative-method``."""
    method = getattr(args, "speculative_method", None)
    if method in DRAFT_METHODS:
        return str(method)
    return None


def add_draft_spec_arguments(
    group: argparse._ActionsContainer,
    *,
    include_acceptance: bool = False,
) -> None:
    """Register unified draft-spec CLI flags on an argument group/parser."""
    group.add_argument(
        "--speculative-method",
        type=str,
        default=None,
        choices=list(DRAFT_METHODS),
        help="Enable draft speculative decoding: dflash or dspark. "
        "Mutually exclusive with MTP. Required before draft-dependent options.",
    )
    group.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=0,
        help="Requires --speculative-method. Number of speculative tokens excluding anchor/bonus "
        "(vLLM-aligned). When >= 1, internal block_size = n + 1. "
        "0 = use builtin/config block_size.",
    )
    if include_acceptance:
        group.add_argument(
            "--acceptance-length",
            type=float,
            default=5.0,
            help="Requires --speculative-method. Decode fold scalar. "
            "Clamped to block_size-1 for dflash, block_size for dspark.",
        )
    group.add_argument(
        "--num-draft-layers",
        type=int,
        default=0,
        help="Requires --speculative-method. Override draft num_hidden_layers from builtin/config. "
        "0 = use config default.",
    )
    group.add_argument(
        "--draft-model-config-path",
        type=str,
        default=None,
        help="Requires --speculative-method. Optional path to override builtin draft config.json.",
    )
    group.add_argument(
        "--dspark-markov-rank",
        type=int,
        default=256,
        help="Requires --speculative-method dspark. Markov embedding rank (0 disables MarkovHead). Default: 256.",
    )
    group.add_argument(
        "--dspark-markov-head",
        type=str,
        default="vanilla",
        choices=["vanilla", "gated", "rnn"],
        help="Requires --speculative-method dspark. Markov head type: vanilla (default), gated, or rnn.",
    )


def mtp_active(args) -> bool:
    """True when MTP is enabled via fixed value or any search candidate > 0 (RFC G2)."""
    if int(getattr(args, "num_mtp_tokens", 0) or 0) > 0:
        return True
    candidates = getattr(args, "num_mtp_token_sizes", None) or []
    return any(int(c) > 0 for c in candidates)


def validate_draft_spec_cli_args(
    parser: argparse.ArgumentParser,
    args,
    argv=None,
    *,
    check_mtp_candidates: bool = False,
) -> None:
    """Validate draft-spec G2/G3: dependents require --speculative-method; vs MTP exclusive."""
    argv = sys.argv[1:] if argv is None else argv
    method = draft_method(args)

    if method is None and argv_has_option(argv, *_METHOD_DEPENDENT_OPTIONS):
        parser.error(
            "--num-speculative-tokens / --acceptance-length / "
            "--num-draft-layers / --draft-model-config-path require --speculative-method"
        )
    if method != "dspark" and argv_has_option(argv, *_DSPARK_ONLY_OPTIONS):
        parser.error("--dspark-markov-rank / --dspark-markov-head require --speculative-method dspark")

    if method is not None:
        mtp_on = mtp_active(args) if check_mtp_candidates else int(getattr(args, "num_mtp_tokens", 0) or 0) > 0
        if mtp_on:
            label = "DSpark" if method == "dspark" else "Dflash"
            parser.error(f"{label} and MTP are mutually exclusive")


def resolve_num_speculative_tokens_to_block(
    num_speculative_tokens: int,
    *,
    draft_model_config_path: Optional[str] = None,
) -> tuple[int, int]:
    """Map CLI n to (n, block_size). ``n<=0`` loads builtin/config ``block_size``."""
    from tensor_cast.layers.dflash import load_dflash_draft_config_dict

    source = load_dflash_draft_config_dict(draft_model_config_path)
    builtin_block = int(source.get("block_size") or 8)
    if builtin_block < 2:
        builtin_block = 8

    n = int(num_speculative_tokens or 0)
    if n <= 0:
        block = builtin_block
        n = block - 1
    else:
        block = n + 1
    return n, block


def clamp_acceptance_length(accept: float, block: int, method: str) -> float:
    """Clamp acceptance by method: dflash → B-1, dspark → B."""
    accept = float(accept)
    if accept < 0:
        accept = 0.0
    if method == "dspark":
        max_accept = float(block)
    else:
        max_accept = float(block - 1)
    if accept > max_accept:
        accept = max_accept
    return accept


def resolve_draft_block_and_acceptance(args) -> None:
    """Resolve ``num_speculative_tokens`` → block; clamp ``acceptance_length`` when present."""
    method = draft_method(args)
    if method is None:
        return

    n, block = resolve_num_speculative_tokens_to_block(
        int(getattr(args, "num_speculative_tokens", 0) or 0),
        draft_model_config_path=getattr(args, "draft_model_config_path", None),
    )
    args.num_speculative_tokens = n
    # Bridge fields for OptimizerData / labels (block includes anchor).
    args.draft_block_size = block

    if hasattr(args, "acceptance_length"):
        args.acceptance_length = clamp_acceptance_length(
            float(getattr(args, "acceptance_length", 5.0)),
            block,
            method,
        )

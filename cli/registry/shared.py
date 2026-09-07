"""Cross-module shared parameter definitions.

These Param instances are the single source of truth for parameters that appear
in multiple CLI modules (e.g. MODEL_ID, DEVICE, NUM_DEVICES). Each module's
ModuleSpec reuses these shared instances via ``.override()`` for module-specific
tweaks (different defaults, added nargs, etc.).

Design note on log_level
------------------------
log_level is NOT a Param — it's provided by ``spec_cli.add_log_options()`` as
public infrastructure (--log-level / -v / -q / --debug / --log-file).
When a module's ``ModuleSpec.log_options=True`` (default), the argparse
adapter injects these flags automatically. The Web UI exposes log_level as a
convention field in ``ui_props`` (not in ModuleSpec — cli/ doesn't know about
UI concepts); choices match spec_cli.STANDARD_LOG_LEVELS, default="info".
"""

from __future__ import annotations

from tensor_cast.core.compilation_config import COMPILATION_CONFIG_OPTIONS

# Enum imports (pure StrEnum, safe to import — no torch dependency)
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)

from .datatypes import Param


def get_device_choices() -> list[str]:
    """Runtime device profile names.

    Deferred import: tensor_cast.device_profiles may not be importable at module
    load time (e.g. in tests without the sim stack). At build_argparser() time
    the profiles are expected to be registered.

    Falls back to ["TEST_DEVICE"] if no profiles are available — keeps CLI
    usable in degraded environments (matches legacy get_common_argparser
    behavior for --device choices).
    """
    try:
        import tensor_cast.device_profiles  # noqa: F401  (registers built-in profiles)
        from tensor_cast.device import DeviceProfile

        # Registration order (TEST_DEVICE first, built-ins next) — matches the
        # legacy get_common_argparser() choices list and its error output.
        names = list(DeviceProfile.all_device_profiles.keys())
        return names or ["TEST_DEVICE"]
    except Exception:  # noqa: BLE001  # Intentional broad catch: fallback for degraded environments
        return ["TEST_DEVICE"]


# ── Common positional: model_id ─────────────────────────────────────
# Used by text_generate, throughput_optimizer, video_generate.
# Positional with nargs="?" + --model-path/--model-id same-dest formal flags
# + runtime require_model_id validation. argparse_adapter handles this pattern
# when cli_positional=True + cli_flag="model-path". cli_aliases are DEPRECATED
# names (one-shot warning); --model-id is a formal sibling registered separately
# by the adapter.
MODEL_ID = Param(
    name="model-id",
    data_type="string",
    required=True,
    nargs="?",
    pattern=r"^[a-zA-Z0-9_/.-]+$",
    max_length=256,
    cli_positional=True,
    cli_flag="model-path",  # positional+flag combo: formal --model-path flag (different from positional name)
    cli_aliases=("model_id",),  # no -- prefix; deprecated underscore form
    cli_metavar="<NAME>",
    group="General Options",
    cli_help=(
        "Model source. Recommended safe mode: a reviewed absolute local model path. "
        "Model id mode also accepts Hugging Face or ModelScope ids, but may execute "
        "remote Python code through trust_remote_code=True and is not security-guaranteed. "
        "Equivalent to --model-id."
    ),
)

# ── Common: device ──────────────────────────────────────────────────
# choices placeholder ["TEST_DEVICE"]; real choices injected at runtime from
# get_device_choices() via argparse_adapter's DEVICE-id special handling.
#
# Module override examples:
#   throughput_optimizer: DEVICE.override(nargs="+", cli_help="...")
#   text_generate / video_generate: no override (single-value default)
DEVICE = Param(
    name="device",
    data_type="string",
    default="TEST_DEVICE",
    group="General Options",
    cli_metavar="<NAME>",
    cli_help=(
        "Specifies the target device profile to use for benchmarking and simulation. "
        "Must be a valid device name as defined in DeviceProfile. "
        "The default device 'TEST_DEVICE' is used for standard simulation runs."
    ),
)

# ── General parameters (from get_common_argparser) ──────────────────

NUM_DEVICES = Param(
    name="num-devices",
    data_type="integer",
    default=1,
    min=1,
    group="General Options",
    cli_metavar="<N>",
    cli_help=(
        "Specifies the total number of devices/processes to use. "
        "Must be a positive integer. "
        "A value of 1 indicates single-device execution."
    ),
)

RESERVED_MEMORY_GB = Param(
    name="reserved-memory-gb",
    data_type="number",
    default=0.0,
    min=0,
    group="General Options",
    cli_metavar="<FLOAT>",
    cli_help=(
        "Amount of device memory (in gigabytes) reserved for system usage "
        "and unavailable for application. "
        "Set to 0 to disable memory reservation."
    ),
)

# ── Quantization parameters (3 enums) ───────────────────────────────

QUANTIZE_LINEAR_ACTION = Param(
    name="quantize-linear-action",
    data_type="string",
    default=QuantizeLinearAction.W8A8_DYNAMIC,
    choices=list(QuantizeLinearAction),
    group="Quantization Options",
    cli_help="Quantize all linear layers (symmetric quant).",
)

QUANTIZE_NON_EXPERT_LINEAR_ACTION = Param(
    name="quantize-non-expert-linear-action",
    data_type="string",
    default=QuantizeLinearAction.DISABLED,
    choices=list(QuantizeLinearAction),
    group="Quantization Options",
    cli_help="Separate quantization type for non-expert linear layers. Routed MoE experts keep --quantize-linear-action.",
)

QUANTIZE_ATTENTION_ACTION = Param(
    name="quantize-attention-action",
    data_type="string",
    default=QuantizeAttentionAction.DISABLED,
    choices=list(QuantizeAttentionAction),
    group="Quantization Options",
    cli_help="Quantize the KV cache with the given action.",
)

MXFP4_GROUP_SIZE = Param(
    name="mxfp4-group-size",
    data_type="integer",
    default=32,
    min=1,
    group="Quantization Options",
    cli_metavar="<N>",
    cli_help="Group size for MXFP4 quantization. [default: 32]",
)

# ── Optimization parameters ─────────────────────────────────────────

COMPILE = Param(
    name="compile",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Optimization Options",
    cli_help="If set, invoke torch.compile() on the model before inference.",
)

COMPILE_ALLOW_GRAPH_BREAK = Param(
    name="compile-allow-graph-break",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Optimization Options",
    cli_help="Allow graph breaks during torch.compile() for models with dynamic control flow.",
)

COMPILATION_CONFIG = Param(
    name="compilation-config",
    data_type="string[]",
    nargs="*",
    choices=COMPILATION_CONFIG_OPTIONS,
    group="Optimization Options",
    cli_help="Enable specific compilation features. If omitted, all compilation features stay disabled.",
)

# ── Debug/Request parameters ────────────────────────────────────────

PREFIX_CACHE_HIT_RATE = Param(
    name="prefix-cache-hit-rate",
    data_type="number",
    default=0.0,
    min=0,
    exclusive_max=1,
    group="Request",
    cli_metavar="<FLOAT>",
    cli_help="Prefix cache hit rate for prefill token reuse in [0, 1). [default: 0.0]",
)

CHROME_TRACE = Param(
    name="chrome-trace-file",
    data_type="string",
    default=None,
    group="Debug",
    cli_aliases=("chrome-trace",),
    cli_metavar="<FILE>",
    cli_help="Write a chrome trace JSON file.",
)

# ── Performance model parameters ────────────────────────────────────

PERFORMANCE_MODEL = Param(
    name="performance-model",
    data_type="string",
    default=None,
    choices=["analytic", "profiling"],
    cli_action="append",
    group="Performance Model",
    cli_metavar="{analytic,profiling}",
    cli_help="Performance model type(s). Repeat the option to select more than one. 'analytic': Roofline model. 'profiling': empirical model (requires --profiling-database-path).",
)

PROFILING_DATABASE = Param(
    name="profiling-database-path",
    data_type="string",
    default=None,
    group="Performance Model",
    cli_aliases=("profiling-database",),
    cli_metavar="<DIR>",
    cli_help="Directory of the profiling database for 'profiling' mode.",
)

DISABLE_PROFILING_INTERPOLATION = Param(
    name="disable-profiling-interpolation",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Performance Model",
    cli_aliases=("no-profiling-interpolation",),
    cli_help="Use exact and partial profiling matches only with --performance-model profiling.",
)

# ── Other shared parameters ─────────────────────────────────────────

REMOTE_SOURCE = Param(
    name="remote-source",
    data_type="string",
    default="huggingface",
    choices=["huggingface", "modelscope"],
    group="Model Source",
    cli_metavar="{huggingface,modelscope}",
    cli_help="The remote source for the model. [default: huggingface]",
)

WORD_EMBEDDING_TP = Param(
    name="word-embedding-tp",
    data_type="string",
    default=None,
    choices=["col", "row"],
    group="Parallelism",
    cli_aliases=("word-embedding-tensor-parallel",),
    cli_metavar="{col,row}",
    cli_help="Word embedding tensor parallel mode. Omitted disables embedding TP.",
)

# ── Unified speculative decoding (mtp/dflash/dspark) ───────────────
# Shared by text_generate (6 params) and throughput_optimizer (adds
# ACCEPTANCE_LENGTH via override + overrides NUM_SPECULATIVE_TOKENS to
# multi-value for search). Registered in both modules' fields;
# G2/G3 cross-field checks live in cli/registry/validators.py
# (draft_dependents_require_method, draft_mtp_mutex).

SPECULATIVE_METHOD = Param(
    name="speculative-method",
    data_type="string",
    default=None,
    choices=["mtp", "dflash", "dspark"],
    group="Request",
    cli_metavar="{mtp,dflash,dspark}",
    cli_help="Enable speculative decoding: mtp, dflash, or dspark. "
    "Mutually exclusive with the legacy MTP entry (--num-mtp-tokens). "
    "Required before speculative-dependent options.",
)

NUM_SPECULATIVE_TOKENS = Param(
    name="num-speculative-tokens",
    data_type="integer",
    default=0,
    min=0,
    group="Request",
    cli_metavar="<N>",
    cli_help="Requires --speculative-method. Number of speculative tokens excluding anchor/bonus "
    "(vLLM-aligned). When >= 1, internal block_size = n + 1. "
    "Omitting keeps builtin block_size for dflash/dspark. "
    "mtp always requires an explicit value; explicit 0 with --speculative-method is rejected.",
)

ACCEPTANCE_LENGTH = Param(
    name="acceptance-length",
    data_type="number",
    default=5.0,
    group="Model & Quantization Options",
    cli_help="Requires --speculative-method. Decode fold scalar. "
    "Clamped to num_speculative_tokens (n) for all methods.",
)

NUM_DRAFT_LAYERS = Param(
    name="num-draft-layers",
    data_type="integer",
    default=0,
    min=0,
    group="Request",
    cli_metavar="<N>",
    cli_help="Requires --speculative-method dflash or dspark. Override draft num_hidden_layers from builtin/config. "
    "0 = use config default. Not allowed with mtp.",
)

DRAFT_MODEL_CONFIG_PATH = Param(
    name="draft-model-config-path",
    data_type="string",
    default=None,
    group="Request",
    cli_help="Requires --speculative-method dflash or dspark. Optional path to override builtin draft config.json. "
    "Not allowed with mtp.",
)

DSPARK_MARKOV_RANK = Param(
    name="dspark-markov-rank",
    data_type="integer",
    default=256,
    min=0,
    group="Request",
    cli_metavar="<N>",
    cli_help="Requires --speculative-method dspark. Markov embedding rank (0 disables MarkovHead). Default: 256.",
)

DSPARK_MARKOV_HEAD = Param(
    name="dspark-markov-head",
    data_type="string",
    default="vanilla",
    choices=["vanilla", "gated", "rnn"],
    group="Request",
    cli_metavar="{vanilla,gated,rnn}",
    cli_help="Requires --speculative-method dspark. Markov head type: vanilla (default), gated, or rnn.",
)

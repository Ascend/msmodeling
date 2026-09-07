"""throughput_optimizer module specification for CLI/WebUI parameter registry.

Defines all throughput_optimizer CLI parameters including shared params (from shared.py)
and module-specific params.
"""

from cli.registry import shared as S
from cli.registry import validators as V
from cli.registry.datatypes import ModuleSpec, Param, ValidatorRef
from serving_cast.service.utils import check_positive_integer_and_string

# ── throughput_optimizer-specific parameters ─────────────────────────

# General Options (device with nargs="+")
DEVICES = Param(
    name="device",
    data_type="string[]",
    nargs="+",
    default=None,
    group="Optional arguments",
    cli_extra_flags=("devices",),
    cli_metavar="<NAME>",
    cli_help="Device profile(s) to evaluate. Multiple values enable cross-hardware summaries.",
)

# Request (2 params)
INPUT_LENGTH = Param(
    name="input-length",
    data_type="string",  # Can be int or YAML file path
    required=True,
    group="Request",
    cli_metavar="<N>",
    cli_type=check_positive_integer_and_string,
    cli_help="Prompt length in tokens, or a YAML file describing a variable-length input distribution.",
)

OUTPUT_LENGTH = Param(
    name="output-length",
    data_type="integer",
    required=True,
    min=1,
    group="Request",
    cli_metavar="<N>",
    cli_help="Expected output length in tokens.",
)

# Model & Quantization Options
NUM_MTP_TOKENS = Param(
    name="num-mtp-tokens",
    data_type="integer[]",
    nargs="+",
    default=None,
    min=0,
    max=9,
    group="Model & Quantization Options",
    cli_metavar="<N>",
    cli_help=(
        "MTP token count candidate(s) in 0-9. Pass one value for a fixed configuration, "
        "or multiple values to sweep during throughput optimization. "
        "0 means disabled and only models with MTP support will benefit from non-zero values. "
        "When combined with TP/EP/MOE-DP search, total combinations grow as TP x EP x MOE-DP x MTP."
    ),
)

MTP_ACCEPTANCE_RATES = Param(
    name="mtp-acceptance-rates",
    data_type="number[]",
    nargs="+",
    default=[0.9, 0.6, 0.4, 0.2],
    group="Model & Quantization Options",
    cli_metavar="<FLOAT>",
    cli_aliases=("mtp-acceptance-rate",),
    cli_dest="mtp_acceptance_rate",
    cli_help="Acceptance rates for MTP. [default: [0.9, 0.6, 0.4, 0.2]]",
)

# throughput_optimizer supports multi-N search for speculative tokens,
# so we need a local definition (integer[], nargs="+") instead of the shared version.
NUM_SPECULATIVE_TOKENS = Param(
    name="num-speculative-tokens",
    data_type="integer[]",
    nargs="+",
    default=None,
    min=0,
    group="Model & Quantization Options",
    cli_metavar="<N>",
    cli_help="Requires --speculative-method. Speculative depth (excluding anchor). "
    "Pass multiple values to sweep during throughput optimization. "
    "Omitting keeps builtin block_size for dflash/dspark. "
    "mtp always requires an explicit value; any 0 candidate is rejected.",
)

TENSOR_PARALLEL_SIZES = Param(
    name="tp-sizes",
    data_type="integer[]",
    nargs="*",
    default=None,
    group="Model & Quantization Options",
    cli_metavar="<N>",
    cli_help="Enable TP search. Optional explicit sizes; default is powers of 2 up to world size.",
)

EXPERT_PARALLEL_SIZES = Param(
    name="ep-sizes",
    data_type="integer[]",
    nargs="*",
    default=None,
    group="Model & Quantization Options",
    cli_metavar="<N>",
    cli_help="Enable EP search. Optional explicit sizes; default is powers of 2 up to world size.",
)

MOE_DATA_PARALLEL_SIZES = Param(
    name="moe-dp-sizes",
    data_type="integer[]",
    nargs="*",
    default=None,
    group="Model & Quantization Options",
    cli_metavar="<N>",
    cli_help="Enable MOE-DP search. Optional explicit sizes; default is powers of 2 up to world size.",
)

PIPELINE_PARALLEL_SIZES = Param(
    name="pp-sizes",
    data_type="integer[]",
    nargs="*",
    default=None,
    group="Model & Quantization Options",
    cli_metavar="<N>",
    cli_help="Enable pipeline-parallel (PP) size search. Optional explicit PP sizes. "
    "If no value is provided, defaults to powers of 2 up to num_devices. "
    "When absent, only PP=1 is searched (legacy behavior).",
)

PIPELINE_LAYER_PARTITIONS = Param(
    name="pp-layer-partitions",
    data_type="string",
    default=None,
    group="Model & Quantization Options",
    cli_metavar="<JSON>",
    cli_help="Explicit PP layer partitions as a JSON list of lists. "
    "Example: --pp-layer-partitions '[[31,30],[16,15,15,15]]' "
    "Each inner list's length must equal its pp_size and sum to num_hidden_layers. "
    "When absent, the default balanced partition is used for each PP size.",
)

DECODE_CONTEXT_PARALLEL_SIZES = Param(
    name="dcp-sizes",
    data_type="integer[]",
    nargs="*",
    default=None,
    group="Model & Quantization Options",
    cli_metavar="<N>",
    cli_help="Enable DCP search. Optional explicit sizes; default is powers of 2 up to world size.",
)

ENABLE_SHARED_EXPERT_TP = Param(
    name="enable-shared-expert-tp",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Model & Quantization Options",
    cli_help="Enable vLLM-style tensor parallel for shared experts. This uses dense-MLP TP for shared_experts with delayed down_proj reduction.",
)

WORD_EMBEDDING_TP = Param(
    name="word-embedding-tp",
    data_type="string",
    default=None,
    choices=["col", "row"],
    group="Model & Quantization Options",
    cli_metavar="{col,row}",
    cli_help="Word embedding tensor parallel mode. Omitted disables embedding TP.",
)

# Performance Model Options (1 param)
PERFORMANCE_MODEL = Param(
    name="performance-model",
    data_type="string",
    default="analytic",
    choices=["analytic", "profiling"],
    group="Performance Model Options",
    cli_metavar="{analytic,profiling}",
    cli_help="Performance model type.",
)

# Debug Options (1 param - chrome_trace already in shared.py)

# Service Options
TTFT_LIMIT = Param(
    name="ttft-limit",
    data_type="number",
    default=None,
    group="Service Options",
    cli_metavar="<FLOAT>",
    cli_aliases=("ttft-limits",),
    cli_dest="ttft_limits",
    cli_help="TTFT constraint under which to search for the best throughput.",
)

TPOT_LIMIT = Param(
    name="tpot-limit",
    data_type="number",
    default=None,
    group="Service Options",
    cli_metavar="<FLOAT>",
    cli_aliases=("tpot-limits",),
    cli_dest="tpot_limits",
    cli_help="TPOT constraint under which to search for the best throughput.",
)

MAX_BATCHED_TOKENS = Param(
    name="max-batched-tokens",
    data_type="integer",
    default=None,
    min=1,
    group="Service Options",
    cli_metavar="<N>",
    cli_help="Max batched tokens for one prefill or mixed prefill/decode step.",
)

BATCH_RANGE = Param(
    name="batch-range",
    data_type="integer[]",
    nargs="+",
    default=None,
    group="Service Options",
    cli_metavar="<N>",
    cli_help="Batch size range: min max, or a single max (min defaults to 1).",
    # Legacy behavior: argparse-level BatchRangeAction validates [min max]/[max]
    # ordering and positivity AND stores the int list. The adapter registers the
    # custom action class directly (see argparse_adapter cli_action handling).
    cli_action="batch_range",
)

SERVING_COST = Param(
    name="serving-cost",
    data_type="number",
    default=0,
    group="Service Options",
    cli_metavar="<FLOAT>",
    cli_help="Serving cost of service delivery.",
)

DISAGGREGATION = Param(
    name="disagg",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Service Options",
    cli_help="Run disaggregation mode.",
)

JOBS = Param(
    name="jobs",
    data_type="integer",
    default=8,
    min=1,
    group="Service Options",
    cli_short="j",
    cli_metavar="<N>",
    cli_help="Number of parallel jobs. Must be a positive integer.",
)

MAX_SEARCH_COMBINATIONS = Param(
    name="max-search-combinations",
    data_type="integer",
    default=100,  # DEFAULT_MAX_SEARCH_COMBINATIONS
    min=0,
    group="Service Options",
    cli_metavar="<N>",
    cli_help="Warn when TP/EP/MOE-DP/MTP search combinations exceed this value. Set 0 to disable the warning.",
)

CONCURRENCY_SEARCH_STRATEGY = Param(
    name="concurrency-search-strategy",
    data_type="string",
    default="exponential",
    choices=["exponential", "linear_exponential"],
    group="Service Options",
    cli_metavar="{exponential,linear_exponential}",
    cli_help="Concurrency search strategy.",
)

DUMP_ORIGINAL_RESULTS = Param(
    name="dump-original-results",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Debug Options",
    cli_help="If set, dump the original results for analysis.",
)

# MultiModal Options
IMAGE_BATCH_SIZE = Param(
    name="image-batch-size",
    data_type="integer",
    default=None,
    min=1,
    group="Multimodal Options",
    cli_metavar="<N>",
    cli_help="Number of images per request. If omitted, reuse batch_size for backward compatibility.",
)

IMAGE_HEIGHT = Param(
    name="image-height",
    data_type="integer",
    default=None,
    min=1,
    group="Multimodal Options",
    cli_metavar="<N>",
    cli_help="Height of the input images.",
)

IMAGE_WIDTH = Param(
    name="image-width",
    data_type="integer",
    default=None,
    min=1,
    group="Multimodal Options",
    cli_metavar="<N>",
    cli_help="Width of the input images.",
)

# PD Ratio Optimization Options
PREFILL_DEVICES_PER_INSTANCE = Param(
    name="prefill-devices-per-instance",
    data_type="integer",
    default=None,
    min=1,
    group="PD Ratio Optimization Options",
    cli_metavar="<N>",
    cli_help="Number of devices per Prefill instance for PD ratio optimization.",
)

DECODE_DEVICES_PER_INSTANCE = Param(
    name="decode-devices-per-instance",
    data_type="integer",
    default=None,
    min=1,
    group="PD Ratio Optimization Options",
    cli_metavar="<N>",
    cli_help="Number of devices per Decode instance for PD ratio optimization.",
)

ENABLE_OPTIMIZE_PREFILL_DECODE_RATIO = Param(
    name="enable-optimize-prefill-decode-ratio",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="PD Ratio Optimization Options",
    cli_help="Enable PD ratio optimization mode",
)


# ── ModuleSpec definition ───────────────────────────────────────────

SPEC = ModuleSpec(
    module_id="throughput_optimizer",
    title="Throughput Optimizer",
    cli_prog="msmodeling inference throughput-optimizer",
    cli_description=(
        "Get best throughput for given input/output sequence length and SLO limits "
        "in aggregation mode or disaggregation mode."
    ),
    cli_examples=(
        "# Search TP on 8 devices\n"
        "msmodeling inference throughput-optimizer Qwen/Qwen3-32B "
        "--device TEST_DEVICE --num-devices 8 --input-length 1024 --output-length 512\n"
        "# Disaggregated prefill/decode search\n"
        "msmodeling inference throughput-optimizer Qwen/Qwen3-32B "
        "--num-devices 16 --input-length 1024 --output-length 512 --disagg"
    ),
    cli_output_help="Best-strategy tables on stdout. Optional chrome trace via --chrome-trace-file.",
    log_options=True,
    log_options_after_group="General Options",
    requires_model_id=True,
    validators=[
        ValidatorRef(name="validParallelCombo", fn=V.valid_parallel_combo),
        ValidatorRef(name="effectiveLenGe1", fn=V.effective_len_ge1_to),
        ValidatorRef(name="pdRatioMutexDisagg", fn=V.pd_ratio_mutex_disagg),
        ValidatorRef(name="mtpTokensVsAcceptanceRate", fn=V.mtp_tokens_vs_acceptance_rate),
        ValidatorRef(
            name="draftDependentsRequireMethod",
            fn=V.draft_dependents_require_method,
            wants_provided=True,
        ),
        ValidatorRef(name="draftMtpMutex", fn=V.draft_mtp_mutex, wants_provided=True),
        ValidatorRef(name="draftLegacyMtpMutex", fn=V.draft_legacy_mtp_mutex, wants_provided=True),
        ValidatorRef(
            name="mtpRequiresNumSpeculativeTokens",
            fn=V.mtp_requires_num_speculative_tokens,
            wants_provided=True,
        ),
        ValidatorRef(name="mtpNoDraftLayers", fn=V.mtp_no_draft_layers),
        ValidatorRef(
            name="speculativeTokensPositive",
            fn=V.speculative_tokens_positive,
            wants_provided=True,
        ),
        ValidatorRef(
            name="mtpNoLegacyAcceptanceRate",
            fn=V.mtp_no_legacy_acceptance_rate,
            wants_provided=True,
        ),
        ValidatorRef(
            name="legacyMtpNoAcceptanceLength",
            fn=V.legacy_mtp_no_acceptance_length,
            wants_provided=True,
        ),
    ],
    fields=[
        # General Options
        S.MODEL_ID,
        S.NUM_DEVICES,
        S.RESERVED_MEMORY_GB.override(default=10.0),
        DEVICES,
        # Request
        INPUT_LENGTH,
        OUTPUT_LENGTH,
        # Model & Quantization Options
        S.COMPILE,
        S.COMPILE_ALLOW_GRAPH_BREAK.override(
            cli_help="If set, invoke torch.compile() on the model before inference. [default: off]",
        ),
        NUM_MTP_TOKENS,
        MTP_ACCEPTANCE_RATES,
        # Unified speculative decoding (multi-value NUM_SPECULATIVE_TOKENS; G2/G3 in validators.py)
        S.SPECULATIVE_METHOD,
        NUM_SPECULATIVE_TOKENS,
        S.ACCEPTANCE_LENGTH.override(group="Model & Quantization Options"),
        S.NUM_DRAFT_LAYERS,
        S.DRAFT_MODEL_CONFIG_PATH,
        S.DSPARK_MARKOV_RANK,
        S.DSPARK_MARKOV_HEAD,
        S.PREFIX_CACHE_HIT_RATE.override(
            cli_help="Prefix cache hit rate for prefill token reuse. This is a token-level approximation in [0, 1). [default: 0.0]",
        ),
        S.QUANTIZE_LINEAR_ACTION,
        S.QUANTIZE_NON_EXPERT_LINEAR_ACTION.override(
            cli_help="Separate quantization type for non-expert linear layers.",
        ),
        S.MXFP4_GROUP_SIZE,
        S.QUANTIZE_ATTENTION_ACTION,
        TENSOR_PARALLEL_SIZES,
        EXPERT_PARALLEL_SIZES,
        MOE_DATA_PARALLEL_SIZES,
        PIPELINE_PARALLEL_SIZES,
        PIPELINE_LAYER_PARTITIONS,
        DECODE_CONTEXT_PARALLEL_SIZES,
        ENABLE_SHARED_EXPERT_TP,
        S.COMPILATION_CONFIG,
        WORD_EMBEDDING_TP,
        # Performance Model Options
        PERFORMANCE_MODEL,
        S.PROFILING_DATABASE.override(cli_help="Profiling CSV database directory for 'profiling' mode."),
        # Debug Options
        S.CHROME_TRACE,
        # Service Options
        TTFT_LIMIT,
        TPOT_LIMIT,
        MAX_BATCHED_TOKENS,
        BATCH_RANGE,
        SERVING_COST,
        DISAGGREGATION,
        JOBS,
        MAX_SEARCH_COMBINATIONS,
        CONCURRENCY_SEARCH_STRATEGY,
        DUMP_ORIGINAL_RESULTS,
        # MultiModal Options
        IMAGE_BATCH_SIZE,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        # PD Ratio Optimization Options
        PREFILL_DEVICES_PER_INSTANCE,
        DECODE_DEVICES_PER_INSTANCE,
        ENABLE_OPTIMIZE_PREFILL_DECODE_RATIO,
    ],
)

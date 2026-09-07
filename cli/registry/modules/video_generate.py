"""video_generate module specification for CLI/WebUI parameter registry.

Defines all video_generate CLI parameters including shared params (from shared.py)
and module-specific params.
"""

from cli.registry import shared as S
from cli.registry import validators as V
from cli.registry.datatypes import ModuleSpec, Param, ValidatorRef
from tensor_cast.model_config import AttentionBackend

# ── video_generate-specific parameters ─────────────────────────────

# Core Options
BATCH_SIZE = Param(
    name="batch-size",
    data_type="integer",
    required=True,
    min=1,
    group="Core",
    cli_metavar="<N>",
    cli_help="Batch size.",
)

SEQ_LEN = Param(
    name="seq-len",
    data_type="integer",
    required=True,
    min=1,
    group="Core",
    cli_metavar="<N>",
    cli_help="Text sequence length.",
)

# Video Dimensions
HEIGHT = Param(
    name="height",
    data_type="integer",
    default=400,
    min=1,
    group="Video Dimensions",
    cli_metavar="<N>",
    cli_help="Frame height. [default: 400]",
)

WIDTH = Param(
    name="width",
    data_type="integer",
    default=832,
    min=1,
    group="Video Dimensions",
    cli_metavar="<N>",
    cli_help="Frame width. [default: 832]",
)

FRAME_NUM = Param(
    name="frame-num",
    data_type="integer",
    default=81,
    min=1,
    group="Video Dimensions",
    cli_metavar="<N>",
    cli_help="Number of frames. [default: 81]",
)

# Sampling
SAMPLE_STEP = Param(
    name="sample-step",
    data_type="integer",
    default=1,
    min=1,
    group="Sampling",
    cli_metavar="<N>",
    cli_help="Number of sampling steps. [default: 1]",
)

# Data Type
DTYPE = Param(
    name="dtype",
    data_type="string",
    default="float16",
    choices=["float16", "float32", "bfloat16"],
    group="Data Type",
    cli_metavar="{float16,float32,bfloat16}",
    cli_help="Activation dtype. [default: float16]",
)

# CFG
USE_CFG = Param(
    name="use-cfg",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="CFG",
    cli_help="Enable classifier-free guidance. [default: off]",
)

# Attention Options
ATTENTION_BACKEND = Param(
    name="attention-backend",
    data_type="string",
    default=AttentionBackend.dense,
    choices=list(AttentionBackend),
    group="Attention Options",
    cli_help="Attention backend semantics for simulation.",
)

ATTENTION_BLOCK_SIZE = Param(
    name="attention-block-size",
    data_type="integer",
    default=128,
    min=1,
    group="Attention Options",
    cli_metavar="<N>",
    cli_help="Block size for block sparse attention route planning. [default: 128]",
)

ATTENTION_SPARSITY = Param(
    name="attention-sparsity",
    data_type="number",
    default=0.0,
    min=0.0,
    exclusive_max=1.0,
    group="Attention Options",
    cli_metavar="<FLOAT>",
    cli_help="Skipped KV-block ratio for block sparse attention in [0.0, 1.0). [default: 0.0]",
)

# Optimization Options (2 params) - COMPILE, COMPILE_ALLOW_GRAPH_BREAK from shared.py

# Parallel Options
# Canonical CLI flag is --num-devices but pydantic field is world_size
NUM_DEVICES = Param(
    name="num-devices",
    data_type="integer",
    default=1,
    min=1,
    group="Parallel Options",
    cli_aliases=("world-size",),
    cli_metavar="<N>",
    cli_help="Number of devices. [default: 1]",
)

ULYSSES_PARALLEL_SIZE = Param(
    name="ulysses-size",
    data_type="integer",
    default=1,
    min=1,
    group="Parallel Options",
    cli_metavar="<N>",
    cli_help="Ulysses sequence-parallel size. [default: 1]",
)

CFG_PARALLEL = Param(
    name="cfg-parallel",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Parallel Options",
    cli_help="Enable classifier-free guidance parallelism. [default: off]",
)

# Cache Options
DIT_CACHE = Param(
    name="dit-cache",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Cache Options",
    cli_help="Enable DiT block cache. [default: off]",
)

CACHE_STEP_RANGE = Param(
    name="cache-step-range",
    data_type="string",
    default=None,
    group="Cache Options",
    cli_metavar="<RANGE>",
    cli_help="Cache step range 'start,end' (inclusive). Required with --dit-cache.",
)

CACHE_STEP_INTERVAL = Param(
    name="cache-step-interval",
    data_type="integer",
    default=1,
    min=1,
    group="Cache Options",
    cli_metavar="<N>",
    cli_help="Update every N steps (1 disables). [default: 1]",
)

CACHE_BLOCK_RANGE = Param(
    name="cache-block-range",
    data_type="string",
    default=None,
    group="Cache Options",
    cli_metavar="<RANGE>",
    cli_help="Cache block range 'start,end' (start inclusive, end exclusive).",
)


# ── Module Specification ────────────────────────────────────────────

SPEC = ModuleSpec(
    module_id="video_generate",
    title="Video Generation",
    cli_prog="msmodeling inference video-generate",
    cli_description="Run a simulated diffusion transformer forward and dump perf stats.",
    cli_examples=(
        "# Single-device video generate\n"
        "msmodeling inference video-generate Wan-AI/Wan2.1-T2V-1.3B "
        "--batch-size 1 --seq-len 512 --device TEST_DEVICE"
    ),
    cli_output_help="Perf stats on stdout. Optional chrome trace via --chrome-trace-file.",
    log_options=True,
    log_options_after_group="Sampling",
    requires_model_id=True,
    validators=[
        ValidatorRef(
            name="cfgParallelRequiresWorldSize2",
            fn=V.cfg_parallel_requires_world_size_2,
        ),
    ],
    fields=[
        # General (baseline order: device first, then model_id positional)
        S.DEVICE.override(cli_help="Device profile used for simulation. [default: TEST_DEVICE]"),
        S.MODEL_ID.override(
            cli_help="Diffusers model dir, remote repo id, or remote repo id plus subfolder (needs transformer/config.json or a compatible transformer config). Recommended safe mode: a reviewed absolute local directory; remote model ids are not security-guaranteed. Equivalent to --model-id.",
            cli_flag_help="Diffusers model source. Equivalent to the positional model id.",
        ),
        # Core Options
        BATCH_SIZE,
        SEQ_LEN,
        # Debug
        S.CHROME_TRACE.override(cli_help="Write chrome trace JSON."),
        # Video Dimensions
        HEIGHT,
        WIDTH,
        FRAME_NUM,
        # Sampling
        SAMPLE_STEP,
        # Data Type
        DTYPE,
        # Remote Source
        S.REMOTE_SOURCE.override(cli_help="The remote source for non-local Diffusers repo ids. [default: huggingface]"),
        # Quantization
        S.QUANTIZE_LINEAR_ACTION.override(cli_help="Quantize linear layers."),
        S.QUANTIZE_ATTENTION_ACTION.override(cli_help="Quantize attention computation."),
        # CFG
        USE_CFG,
        # Attention Options
        ATTENTION_BACKEND,
        ATTENTION_BLOCK_SIZE,
        ATTENTION_SPARSITY,
        # Optimization Options
        S.COMPILE,
        S.COMPILE_ALLOW_GRAPH_BREAK,
        # Parallel Options
        NUM_DEVICES,
        ULYSSES_PARALLEL_SIZE,
        CFG_PARALLEL,
        # Cache Options
        DIT_CACHE,
        CACHE_STEP_RANGE,
        CACHE_STEP_INTERVAL,
        CACHE_BLOCK_RANGE,
    ],
)

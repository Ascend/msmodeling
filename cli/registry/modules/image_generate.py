# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""image_generate module specification for CLI/WebUI parameter registry.

Defines all image_generate CLI parameters including shared params (from shared.py)
and module-specific params.
"""

from cli.registry import shared as S
from cli.registry import validators as V
from cli.registry.datatypes import ModuleSpec, Param, ValidatorRef
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

# ── image_generate-specific parameters ─────────────────────────────

# Request Options
BATCH_SIZE = Param(
    name="batch-size",
    data_type="integer",
    required=True,
    min=1,
    cli_metavar="<N>",
    cli_help="Base workload batch size, not prompt or source-image count.",
)

OUTPUT_IMAGE_SIZE = Param(
    name="output-image-size",
    data_type="integer[]",
    required=True,
    nargs=2,
    cli_action="append",
    cli_metavar=("HEIGHT", "WIDTH"),
    cli_help="Output image size HEIGHT WIDTH. Provide exactly once. Used only to derive shapes.",
)

TEXT_SEQ_LEN = Param(
    name="text-seq-len",
    data_type="integer",
    required=True,
    min=1,
    cli_metavar="<N>",
    cli_help="Text condition length that enters the Transformer. Encoding is not executed.",
)

SOURCE_IMAGE_SIZE = Param(
    name="source-image-size",
    data_type="integer[]",
    default=[],
    nargs=2,
    cli_action="append",
    cli_metavar=("HEIGHT", "WIDTH"),
    cli_help="Source image size HEIGHT WIDTH. Repeatable. Editing kinds only.",
)

SAMPLE_STEP = Param(
    name="sample-step",
    data_type="integer",
    default=1,
    min=1,
    cli_metavar="<N>",
    cli_help="Number of identical Transformer workload iterations.",
)

# Data Type & Model Source
DTYPE = Param(
    name="dtype",
    data_type="string",
    default="float16",
    choices=["float16", "float32", "bfloat16"],
    cli_metavar="{float16,float32,bfloat16}",
    cli_help="Activation dtype.",
)

# Quantization Options (3 params) - QUANTIZE_LINEAR_ACTION, MXFP4_GROUP_SIZE, QUANTIZE_ATTENTION_ACTION from shared.py

# Optimization Options (2 params) - COMPILE, COMPILE_ALLOW_GRAPH_BREAK from shared.py

# Parallel Options
# Formal CLI flag is --num-devices; --world-size is a deprecated alias
NUM_DEVICES = Param(
    name="num-devices",
    data_type="integer",
    default=1,
    min=1,
    cli_aliases=("world-size",),
    cli_metavar="<N>",
    cli_help="Number of devices. Must equal --ulysses-size, or 2 * --ulysses-size with --cfg-parallel.",
)

ULYSSES_SIZE = Param(
    name="ulysses-size",
    data_type="integer",
    default=1,
    min=1,
    cli_metavar="<N>",
    cli_help="Ulysses sequence-parallel size.",
)

CFG_PARALLEL = Param(
    name="cfg-parallel",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    cli_help="Enable CFG parallelism. Requires --use-cfg.",
)

# CFG Options
USE_CFG = Param(
    name="use-cfg",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    cli_help="Enable classifier-free guidance workload approximation.",
)

# Cache Options
DIT_CACHE = Param(
    name="dit-cache",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    cli_help="Enable DiT block cache.",
)

CACHE_STEP_RANGE = Param(
    name="cache-step-range",
    data_type="string",
    default=None,
    cli_metavar="<RANGE>",
    cli_help="Cache step range 'start,end' (inclusive). Required with --dit-cache when interval > 1.",
)

CACHE_STEP_INTERVAL = Param(
    name="cache-step-interval",
    data_type="integer",
    default=1,
    min=1,
    cli_metavar="<N>",
    cli_help="Update cache every N steps (1 disables reuse).",
)

CACHE_BLOCK_RANGE = Param(
    name="cache-block-range",
    data_type="string",
    default=None,
    cli_metavar="<RANGE>",
    cli_help="Cache block range 'start,end' (start inclusive, end exclusive).",
)


# ── Module assembly ────────────────────────────────────────────────

SPEC = ModuleSpec(
    module_id="image_generate",
    title="Image Generation",
    cli_prog="msmodeling inference image-generate",
    cli_description=(
        "Simulate image Transformer denoising workloads and report their critical path "
        "and logical measured work only. Prompt encoding, VAE, scheduler, and image I/O "
        "are excluded."
    ),
    cli_examples=(
        "# Single-device image generate\n"
        "msmodeling inference image-generate black-forest-labs/FLUX.1-dev "
        "--batch-size 1 --output-image-size 512 512 --text-seq-len 512 --device TEST_DEVICE"
    ),
    cli_output_help="Perf stats on stdout. Optional chrome trace via --chrome-trace-file.",
    log_options=True,
    log_options_after_field="compile_allow_graph_break",
    requires_model_id=True,
    fields=[
        # General (baseline order)
        S.DEVICE.override(group=None, cli_help="Device profile used for simulation."),
        S.MODEL_ID.override(
            group=None,
            cli_help=(
                "Reviewed local Diffusers config directory or exact remote model ID. "
                "Equivalent to --model-id. Remote model ids are not security-guaranteed."
            ),
            cli_flag_help="Image model source. Equivalent to the positional model id.",
        ),
        # Request
        BATCH_SIZE,
        OUTPUT_IMAGE_SIZE,
        TEXT_SEQ_LEN,
        SOURCE_IMAGE_SIZE,
        SAMPLE_STEP,
        # CFG
        USE_CFG,
        # Data Type
        DTYPE,
        S.REMOTE_SOURCE.override(group=None, cli_help="Remote source for non-local Diffusers repo ids."),
        # Quantization
        S.QUANTIZE_LINEAR_ACTION.override(
            group=None,
            default=QuantizeLinearAction.DISABLED,
            cli_help="Quantize linear layers.",
        ),
        S.MXFP4_GROUP_SIZE.override(group=None),
        S.QUANTIZE_ATTENTION_ACTION.override(group=None, cli_help="Quantize attention computation."),
        # Optimization
        S.COMPILE.override(group=None, cli_help="Compile the transformer before simulation."),
        S.COMPILE_ALLOW_GRAPH_BREAK.override(group=None, cli_help="Allow graph breaks during torch.compile()."),
        # Parallelism
        NUM_DEVICES,
        ULYSSES_SIZE,
        CFG_PARALLEL,
        # Cache
        DIT_CACHE,
        CACHE_STEP_RANGE,
        CACHE_STEP_INTERVAL,
        CACHE_BLOCK_RANGE,
        # Debug
        S.CHROME_TRACE.override(group=None, cli_help="Write chrome trace JSON."),
    ],
    validators=[
        ValidatorRef(name="cfgParallelRequiresUseCfg", fn=V.cfg_parallel_requires_use_cfg),
        ValidatorRef(name="numDevicesMatchesUlyssesSize", fn=V.num_devices_matches_ulysses_size),
        ValidatorRef(name="cfgParallelRequiresWorldSize2", fn=V.cfg_parallel_requires_world_size_2),
    ],
)

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

"""text_generate module specification for CLI/WebUI parameter registry.

Defines all text_generate CLI parameters including shared params (from shared.py)
and module-specific params.
"""

from cli.registry import shared as S
from cli.registry import validators as V
from cli.registry.datatypes import ModuleSpec, Param, ValidatorRef

# ── text_generate-specific parameters ─────────────────────────────

# LLM Options (6 params)
NUM_QUERIES = Param(
    name="num-queries",
    data_type="integer",
    required=True,
    min=1,
    group="Request",
    cli_metavar="<N>",
    cli_help="Number of parallel inference queries to execute in a single batch.",
)

QUERY_LENGTH = Param(
    name="query-length",
    data_type="integer",
    required=True,
    min=1,
    group="Request",
    cli_metavar="<N>",
    cli_help="Length (in tokens) of new input sequence for each query.",
)

CONTEXT_LENGTH = Param(
    name="context-length",
    data_type="integer",
    default=0,
    min=0,
    group="Request",
    cli_metavar="<N>",
    cli_help="Length (in tokens) of existing context for each query. [default: 0]",
)

DECODE = Param(
    name="decode",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Request",
    cli_help="Enable autoregressive decoding mode for text generation. [default: off]",
)

PREFILL = Param(
    name="prefill",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Request",
    cli_help=(
        "Explicitly run in prefill phase. Mutually exclusive with --decode. "
        "When neither flag is passed, prefill is assumed by default (backward compatible). "
        "[default: off]"
    ),
)

NUM_MTP_TOKENS = Param(
    name="num-mtp-tokens",
    data_type="integer",
    default=0,
    min=0,
    group="Request",
    cli_metavar="<N>",
    cli_help="Number of Multi-Token Prediction (MTP) tokens. 0 = disabled. Only supports models with MTP capability (e.g., DeepSeek). [default: 0]",
)

DISABLE_REPETITION = Param(
    name="no-repetition",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    cli_aliases=("disable-repetition",),
    cli_dest="disable_repetition",  # Backward compat: downstream code uses this attr name
    group="Request",
    cli_help="Do not reuse repeated transformer layers to save runtime cost. [default: off]",
)

# Optimization Options (shared: COMPILE, COMPILE_ALLOW_GRAPH_BREAK, COMPILATION_CONFIG + FUSION_PLUGIN)
FUSION_PLUGIN = Param(
    name="fusion-plugin",
    data_type="string",
    default=None,
    cli_action="append",
    group="Optimization Options",
    cli_metavar="PATH",
    cli_help=(
        "Path to a fusion plugin .py to load before model construction. "
        "May be repeated to load several. "
        "Requires --compile; without it the pattern registers but never fires."
    ),
)

# Quantization Options (shared: QUANTIZE_LINEAR_ACTION, QUANTIZE_NON_EXPERT_LINEAR_ACTION, QUANTIZE_ATTENTION_ACTION, MXFP4_GROUP_SIZE + LMHEAD)
QUANTIZE_LMHEAD = Param(
    name="quantize-lmhead",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Quantization Options",
    cli_help="Quantize the LM Head. Off by default because it usually hurts accuracy. [default: off]",
)

# Debug Options (5 params)
GRAPH_LOG_URL = Param(
    name="graph-log-path",
    data_type="string",
    default=None,
    group="Debug",
    cli_aliases=("graph-log-url", "graph-log-file"),
    cli_dest="graph_log_url",  # Backward compat: downstream code uses this attr name
    cli_metavar="<DIR>",
    cli_help="Directory for dumping compiled graphs when --compile is on. Each compile pass writes files under this directory.",
)

DUMP_INPUT_SHAPES = Param(
    name="dump-input-shapes",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Debug",
    cli_help="Group the result table average by input shapes. [default: off]",
)

DUMP_OP_BOUND_RESULTS = Param(
    name="dump-op-bound-results",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Debug",
    cli_help="Dump per-operator memory/communication/MMA/GP bound ratios. [default: off]",
)

NUM_HIDDEN_LAYERS_OVERRIDE = Param(
    name="num-hidden-layers-override",
    data_type="integer",
    default=0,
    min=0,
    group="Debug",
    cli_metavar="<N>",
    cli_help="Override the number of hidden layers, for debugging only. [default: 0]",
)

EXPORT_EMPIRICAL_METRICS = Param(
    name="export-empirical-metrics-file",
    data_type="string",
    default=None,
    group="Debug",
    cli_aliases=("export-empirical-metrics",),
    cli_metavar="<FILE>",
    cli_help="(developer only) Export M1-M5 metrics report as JSON. Requires --performance-model profiling.",
)

# Parallelism Options (14 params)
TP_SIZE = Param(
    name="tp-size",
    data_type="integer",
    default=1,
    min=1,
    group="Parallelism",
    cli_metavar="<N>",
    cli_help="Tensor parallel size for the whole model. [default: 1]",
)

PP_SIZE = Param(
    name="pp-size",
    data_type="integer",
    default=1,
    min=1,
    group="Parallelism",
    cli_metavar="<N>",
    cli_help="Pipeline parallel size for the whole model. [default: 1]",
)

DP_SIZE = Param(
    name="dp-size",
    data_type="integer",
    default=None,
    group="Parallelism",
    cli_metavar="<N>",
    cli_help="Data parallel size for the whole model.",
)

EP_SIZE = Param(
    name="ep-size",
    data_type="integer",
    default=1,
    min=1,
    group="Expert",
    cli_metavar="<N>",
    cli_help="Expert parallel size. [default: 1]",
)

DCP_SIZE = Param(
    name="dcp-size",
    data_type="integer",
    default=1,
    min=1,
    group="Parallelism",
    cli_metavar="<N>",
    cli_help="Decode Context Parallel size. Reuses TP devices and must divide --tp-size.",
)

MOE_TP_SIZE = Param(
    name="moe-tp-size",
    data_type="integer",
    default=None,
    group="Expert",
    cli_metavar="<N>",
    cli_help="Tensor parallel size for experts.",
)

MOE_DP_SIZE = Param(
    name="moe-dp-size",
    data_type="integer",
    default=1,
    min=1,
    group="Expert",
    cli_metavar="<N>",
    cli_help="Data parallel size for experts. [default: 1]",
)

O_PROJ_TP_SIZE = Param(
    name="o-proj-tp-size",
    data_type="integer",
    default=None,
    group="Advanced Parallelism",
    cli_metavar="<N>",
    cli_help="Tensor parallel size for attn o_proj.",
)

O_PROJ_DP_SIZE = Param(
    name="o-proj-dp-size",
    data_type="integer",
    default=None,
    group="Advanced Parallelism",
    cli_metavar="<N>",
    cli_help="Data parallel size for attn o_proj.",
)

MLP_TP_SIZE = Param(
    name="mlp-tp-size",
    data_type="integer",
    default=None,
    group="Advanced Parallelism",
    cli_metavar="<N>",
    cli_help="Tensor parallel size for MLP layers.",
)

MLP_DP_SIZE = Param(
    name="mlp-dp-size",
    data_type="integer",
    default=None,
    group="Advanced Parallelism",
    cli_metavar="<N>",
    cli_help="Data parallel size for MLP layers.",
)

LMHEAD_TP_SIZE = Param(
    name="lmhead-tp-size",
    data_type="integer",
    default=None,
    group="Advanced Parallelism",
    cli_metavar="<N>",
    cli_help="Tensor parallel size for the LM head.",
)

LMHEAD_DP_SIZE = Param(
    name="lmhead-dp-size",
    data_type="integer",
    default=None,
    group="Advanced Parallelism",
    cli_metavar="<N>",
    cli_help="Data parallel size for the LM head.",
)

VISION_TP_SIZE = Param(
    name="vision-tp-size",
    data_type="integer",
    default=1,
    min=1,
    group="Advanced Parallelism",
    cli_metavar="<N>",
    cli_help="Vision tensor parallel degree. Default 1 keeps vision modules unsharded. [default: 1]",
)

# Expert Options (4 params)
ENABLE_REDUNDANT_EXPERTS = Param(
    name="enable-redundant-experts",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Expert",
    cli_help="Use redundant experts. If shared-expert externalization is off, each device adds one redundant expert. If it is on and every device has the same number of routing experts, each device hosting routing experts also adds one redundant expert. [default: off]",
)

ENABLE_SHARED_EXPERT_TP = Param(
    name="enable-shared-expert-tp",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Expert",
    cli_help="Enable vLLM-style tensor parallel for shared experts. This uses dense-MLP TP for shared_experts with delayed down_proj reduction. [default: off]",
)

ENABLE_EXTERNAL_SHARED_EXPERTS = Param(
    name="enable-external-shared-experts",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Expert",
    cli_help="Whether or not to implement external shared experts [default: off]",
)

HOST_EXTERNAL_SHARED_EXPERTS = Param(
    name="host-external-shared-experts",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    group="Expert",
    cli_help="Whether to have the current device host the external shared experts [default: off]",
)

# MultiModal Options (3 params)
IMAGE_BATCH_SIZE = Param(
    name="image-batch-size",
    data_type="integer",
    default=None,
    min=1,
    group="Multimodal",
    cli_metavar="<N>",
    cli_help="Batch size for image processing.",
)

IMAGE_HEIGHT = Param(
    name="image-height",
    data_type="integer",
    default=None,
    min=1,
    group="Multimodal",
    cli_metavar="<N>",
    cli_help="Height of the input images.",
)

IMAGE_WIDTH = Param(
    name="image-width",
    data_type="integer",
    default=None,
    min=1,
    group="Multimodal",
    cli_metavar="<N>",
    cli_help="Width of the input images.",
)


# ── ModuleSpec definition ─────────────────────────────────────────

SPEC = ModuleSpec(
    module_id="text_generate",
    title="Text Generation",
    cli_prog="msmodeling inference text-generate",
    cli_description="Run a simulated LLM inference pass and dump the perf result.",
    cli_examples=(
        "# Prefill one query of 128 tokens\n"
        "msmodeling inference text-generate Qwen/Qwen3-32B --num-queries 1 --query-length 128 --device TEST_DEVICE\n"
        "# Decode with tensor parallel\n"
        "msmodeling inference text-generate Qwen/Qwen3-32B --num-queries 8 --query-length 1 --context-length 4096 --decode --tp-size 8"
    ),
    cli_output_help="Metrics table on stdout. Optional chrome trace via --chrome-trace-file.",
    log_options=True,
    requires_model_id=True,
    fields=[
        # General Options
        S.MODEL_ID,
        S.DEVICE,
        S.NUM_DEVICES,
        S.RESERVED_MEMORY_GB,
        # Request
        NUM_QUERIES,
        QUERY_LENGTH,
        CONTEXT_LENGTH,
        DECODE,
        PREFILL,
        S.PREFIX_CACHE_HIT_RATE,
        NUM_MTP_TOKENS,
        # Unified speculative decoding (G2/G3 checks in validators.py)
        S.SPECULATIVE_METHOD,
        S.NUM_SPECULATIVE_TOKENS,
        S.NUM_DRAFT_LAYERS,
        S.DRAFT_MODEL_CONFIG_PATH,
        S.DSPARK_MARKOV_RANK,
        S.DSPARK_MARKOV_HEAD,
        DISABLE_REPETITION,
        # Optimization Options
        S.COMPILE,
        S.COMPILE_ALLOW_GRAPH_BREAK,
        S.COMPILATION_CONFIG,
        FUSION_PLUGIN,
        # Quantization Options
        S.QUANTIZE_LINEAR_ACTION,
        S.QUANTIZE_NON_EXPERT_LINEAR_ACTION,
        QUANTIZE_LMHEAD,
        S.MXFP4_GROUP_SIZE,
        S.QUANTIZE_ATTENTION_ACTION,
        # Debug Options
        GRAPH_LOG_URL,
        DUMP_INPUT_SHAPES,
        DUMP_OP_BOUND_RESULTS,
        S.CHROME_TRACE,
        NUM_HIDDEN_LAYERS_OVERRIDE,
        # Parallelism Options
        TP_SIZE,
        PP_SIZE,
        DCP_SIZE,
        DP_SIZE,
        EP_SIZE,
        O_PROJ_TP_SIZE,
        O_PROJ_DP_SIZE,
        MLP_TP_SIZE,
        MLP_DP_SIZE,
        LMHEAD_TP_SIZE,
        LMHEAD_DP_SIZE,
        MOE_TP_SIZE,
        MOE_DP_SIZE,
        S.WORD_EMBEDDING_TP,
        ENABLE_REDUNDANT_EXPERTS,
        ENABLE_SHARED_EXPERT_TP,
        ENABLE_EXTERNAL_SHARED_EXPERTS,
        HOST_EXTERNAL_SHARED_EXPERTS,
        VISION_TP_SIZE,
        # MultiModal Options
        IMAGE_BATCH_SIZE,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        # Top-level
        S.REMOTE_SOURCE,
        S.PERFORMANCE_MODEL,
        S.PROFILING_DATABASE,
        S.DISABLE_PROFILING_INTERPOLATION,
        S.ANALYTIC_CALIBRATION_PROFILE,
        S.ANALYTIC_CALIBRATION_STACK,
        EXPORT_EMPIRICAL_METRICS,
    ],
    validators=[
        ValidatorRef(name="prefillDecodeMutex", fn=V.prefill_decode_mutex, wants_provided=True),
        ValidatorRef(name="productEqNumDevices", fn=V.product_eq_num_devices),
        ValidatorRef(name="moeProductEqNumDevices", fn=V.moe_product_eq_num_devices),
        ValidatorRef(name="perLayerProductEqNumDevices", fn=V.per_layer_product_eq_num_devices),
        ValidatorRef(name="sharedExpertMutex", fn=V.shared_expert_mutex),
        ValidatorRef(name="effectiveLenGe1", fn=V.effective_len_ge1),
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
    ],
)

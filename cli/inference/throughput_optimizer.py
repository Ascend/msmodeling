# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import sys
import time

from cli.logo import print_logo
from cli.spec_cli import (
    METAVAR_DIR,
    METAVAR_FILE,
    METAVAR_FLOAT,
    METAVAR_N,
    METAVAR_NAME,
    SpecArgumentParser,
    add_option,
    configure_std_logging,
    inherit_deprecated,
    make_enum_type,
    make_token_type,
    parse_args as spec_parse_args,
)
from serving_cast.service.optimizer_curve_plots import (
    render_cross_hardware_summary,
    run_multi_device_loop,
)
from serving_cast.service.utils import (
    BatchRangeAction,
    DEFAULT_MAX_SEARCH_COMBINATIONS,
    check_positive_float,
    check_positive_integer,
    check_positive_integer_and_string,
    count_search_combinations,
    load_length_distribution,
    resolve_parallel_search_candidates,
    resolve_search_sizes,
)
from tensor_cast import device_profiles  # noqa: F401
from tensor_cast.core.compilation_config import (
    COMPILATION_CONFIG_OPTIONS,
    apply_compilation_config,
)
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.model_config import WordEmbeddingTPMode

from ..utils import (
    LOG_FORMAT,
    add_draft_spec_arguments,
    check_device_targets,
    check_non_negative_integer,
    check_prefix_cache_hit_rate,
    draft_method,
    get_common_argparser,
    require_model_id,
    resolve_draft_block_and_acceptance,
    validate_draft_spec_cli_args,
)


def arg_parse():
    common_parser = get_common_argparser(reserved_memory_gb_default=10.0)
    parser = SpecArgumentParser(
        prog="msmodeling inference throughput-optimizer",
        description="Get best throughput for given input/output sequence length and SLO limits "
        "in aggregation mode or disaggregation mode.",
        parents=[common_parser],
        conflict_handler="resolve",
        examples=(
            "# Search TP on 8 devices\n"
            "msmodeling inference throughput-optimizer Qwen/Qwen3-32B "
            "--device TEST_DEVICE --num-devices 8 --input-length 1024 --output-length 512\n"
            "# Disaggregated prefill/decode search\n"
            "msmodeling inference throughput-optimizer Qwen/Qwen3-32B "
            "--num-devices 16 --input-length 1024 --output-length 512 --disagg"
        ),
        output_help="Best-strategy tables on stdout. Optional chrome trace via --chrome-trace-file.",
    )
    inherit_deprecated(parser, common_parser)
    parse_linear, linear_meta = make_enum_type(QuantizeLinearAction, "--quantize-linear-action")
    parse_attn, attn_meta = make_enum_type(QuantizeAttentionAction, "--quantize-attention-action")
    parse_cc, cc_meta = make_token_type(COMPILATION_CONFIG_OPTIONS, "--compilation-config", store_canonical="snake")
    parse_strategy, strategy_meta = make_token_type(
        ("exponential", "linear_exponential"),
        "--concurrency-search-strategy",
        store_canonical="snake",
    )
    parser.add_argument(
        "--device",
        "--devices",
        dest="device",
        type=str,
        nargs="+",
        default=None,
        metavar=METAVAR_NAME,
        help="Device profile(s) to evaluate. Multiple values enable cross-hardware summaries.",
    )
    parser.add_argument(
        "--input-length",
        type=check_positive_integer_and_string,
        required=True,
        metavar=METAVAR_N,
        help="Prompt length in tokens, or a YAML file describing a variable-length input distribution.",
    )
    parser.add_argument(
        "--output-length",
        type=check_positive_integer,
        required=True,
        metavar=METAVAR_N,
        help="Expected output length in tokens.",
    )
    model_group = parser.add_argument_group("Model & Quantization Options")
    model_group.add_argument(
        "--compile",
        action="store_true",
        help="If set, invoke torch.compile() on the model before inference.",
    )
    model_group.add_argument(
        "--compile-allow-graph-break",
        action="store_true",
        help="If set, invoke torch.compile() on the model before inference.",
    )
    model_group.add_argument(
        "--num-mtp-tokens",
        type=int,
        choices=range(0, 10),
        nargs="+",
        default=None,
        metavar=METAVAR_N,
        help="MTP token count candidate(s) in 0-9. Pass one value for a fixed configuration, "
        "or multiple values to sweep during throughput optimization. "
        "0 means disabled and only models with MTP support will benefit from non-zero values. "
        "When combined with TP/EP/MOE-DP search, total combinations grow as TP x EP x MOE-DP x MTP.",
    )
    add_option(
        parser,
        "--mtp-acceptance-rates",
        dest="mtp_acceptance_rate",
        type=float,
        default=[0.9, 0.6, 0.4, 0.2],
        nargs="+",
        metavar=METAVAR_FLOAT,
        help="Acceptance rates for MTP.",
        aliases=("--mtp-acceptance-rate",),
    )
    add_draft_spec_arguments(model_group, include_acceptance=True)
    parser.add_argument(
        "--prefix-cache-hit-rate",
        type=check_prefix_cache_hit_rate,
        default=0.0,
        metavar=METAVAR_FLOAT,
        help="Prefix cache hit rate for prefill token reuse. This is a token-level approximation in [0, 1).",
    )
    model_group.add_argument(
        "--quantize-linear-action",
        type=parse_linear,
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        metavar=linear_meta,
        help="Quantize all linear layers (symmetric quant).",
    )
    model_group.add_argument(
        "--quantize-non-expert-linear-action",
        type=parse_linear,
        default=QuantizeLinearAction.DISABLED,
        metavar=linear_meta,
        help="Separate quantization type for non-expert linear layers.",
    )
    model_group.add_argument(
        "--mxfp4-group-size",
        type=check_positive_integer,
        default=32,
        metavar=METAVAR_N,
        help="Group size for MXFP4 quantization.",
    )
    model_group.add_argument(
        "--quantize-attention-action",
        type=parse_attn,
        default=QuantizeAttentionAction.DISABLED,
        metavar=attn_meta,
        help="Quantize the KV cache with the given action.",
    )
    add_option(
        model_group,
        "--tp-sizes",
        dest="tp_sizes",
        type=check_positive_integer,
        nargs="*",
        default=None,
        metavar=METAVAR_N,
        help="Enable TP search. Optional explicit sizes; default is powers of 2 up to world size.",
    )
    add_option(
        model_group,
        "--ep-sizes",
        dest="ep_sizes",
        type=check_positive_integer,
        nargs="*",
        default=None,
        metavar=METAVAR_N,
        help="Enable EP search. Optional explicit sizes; default is powers of 2 up to world size.",
    )
    add_option(
        model_group,
        "--moe-dp-sizes",
        dest="moe_dp_sizes",
        type=check_positive_integer,
        nargs="*",
        default=None,
        metavar=METAVAR_N,
        help="Enable MOE-DP search. Optional explicit sizes; default is powers of 2 up to world size.",
    )
    add_option(
        model_group,
        "--dcp-sizes",
        dest="dcp_sizes",
        type=check_positive_integer,
        nargs="*",
        default=None,
        metavar=METAVAR_N,
        help="Enable DCP search. Optional explicit sizes; default is powers of 2 up to world size.",
    )
    model_group.add_argument(
        "--enable-shared-expert-tp",
        action="store_true",
        help="Enable vLLM-style tensor parallel for shared experts. "
        "This uses dense-MLP TP for shared_experts with delayed down_proj reduction.",
    )
    model_group.add_argument(
        "--compilation-config",
        nargs="*",
        default=None,
        type=parse_cc,
        metavar=cc_meta,
        help="Enable specific compilation features. If omitted, all compilation features stay disabled.",
    )
    model_group.add_argument(
        "--word-embedding-tp",
        dest="word_embedding_tp",
        type=str,
        choices=[mode.value for mode in WordEmbeddingTPMode],
        default=None,
        metavar="{col,row}",
        help="Word embedding tensor parallel mode. Omitted disables embedding TP.",
    )
    perf_group = parser.add_argument_group("Performance Model Options")
    perf_group.add_argument(
        "--performance-model",
        type=str,
        default="analytic",
        dest="performance_model",
        choices=["analytic", "profiling"],
        metavar="{analytic,profiling}",
        help="Performance model type.",
    )
    add_option(
        perf_group,
        "--profiling-database-path",
        dest="profiling_database",
        type=str,
        default=None,
        metavar=METAVAR_DIR,
        help="Profiling CSV database directory for 'profiling' mode.",
        aliases=("--profiling-database",),
    )
    debug_group = parser.add_argument_group("Debug Options")
    add_option(
        debug_group,
        "--chrome-trace-file",
        dest="chrome_trace",
        type=str,
        default=None,
        metavar=METAVAR_FILE,
        help="Write a chrome trace JSON file.",
        aliases=("--chrome-trace",),
    )

    service_group = parser.add_argument_group("Service Options")
    add_option(
        service_group,
        "--ttft-limit",
        dest="ttft_limits",
        type=check_positive_float,
        default=None,
        metavar=METAVAR_FLOAT,
        help="TTFT constraint under which to search for the best throughput.",
        aliases=("--ttft-limits",),
    )
    add_option(
        service_group,
        "--tpot-limit",
        dest="tpot_limits",
        type=check_positive_float,
        default=None,
        metavar=METAVAR_FLOAT,
        help="TPOT constraint under which to search for the best throughput.",
        aliases=("--tpot-limits",),
    )
    service_group.add_argument(
        "--max-batched-tokens",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Max batched tokens for one prefill or mixed prefill/decode step.",
    )
    service_group.add_argument(
        "--batch-range",
        type=int,
        nargs="+",
        action=BatchRangeAction,
        default=None,
        metavar=METAVAR_N,
        help="Batch size range: min max, or a single max (min defaults to 1).",
    )
    service_group.add_argument(
        "--serving-cost",
        type=float,
        default=0,
        metavar=METAVAR_FLOAT,
        help="Serving cost of service delivery.",
    )
    add_option(
        service_group,
        "--disagg",
        dest="disagg",
        action="store_true",
        help="Run disaggregation mode.",
    )
    service_group.add_argument(
        "--jobs",
        "-j",
        type=check_positive_integer,
        default=8,
        metavar=METAVAR_N,
        help="Number of parallel jobs. Must be a positive integer.",
    )
    service_group.add_argument(
        "--max-search-combinations",
        type=check_non_negative_integer,
        default=DEFAULT_MAX_SEARCH_COMBINATIONS,
        metavar=METAVAR_N,
        help="Warn when TP/EP/MOE-DP/MTP search combinations exceed this value. Set 0 to disable the warning.",
    )
    service_group.add_argument(
        "--concurrency-search-strategy",
        type=parse_strategy,
        default="exponential",
        metavar=strategy_meta,
        help="Concurrency search strategy.",
    )
    parser.add_argument(
        "--dump-original-results",
        action="store_true",
        help="If set, dump the original results for analysis.",
    )
    multimodal_group = parser.add_argument_group("MultiModal Options")
    multimodal_group.add_argument(
        "--image-batch-size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Number of images per request. If omitted, reuse batch_size for backward compatibility.",
    )
    multimodal_group.add_argument(
        "--image-height",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Height of the input images.",
    )
    multimodal_group.add_argument(
        "--image-width",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Width of the input images.",
    )
    pd_ratio_group = parser.add_argument_group("PD Ratio Optimization Options")
    pd_ratio_group.add_argument(
        "--prefill-devices-per-instance",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Number of devices per Prefill instance for PD ratio optimization.",
    )
    pd_ratio_group.add_argument(
        "--decode-devices-per-instance",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Number of devices per Decode instance for PD ratio optimization.",
    )
    pd_ratio_group.add_argument(
        "--enable-optimize-prefill-decode-ratio",
        action="store_true",
        help="Enable PD ratio optimization mode",
    )
    args = spec_parse_args(parser)
    require_model_id(parser, args)

    if all(x is None for x in (args.tp_sizes, args.ep_sizes, args.moe_dp_sizes)):
        # Backward-compatible default: search TP only with default range.
        args.tp_sizes = []

    def _normalize_mtp_token_values(values: list[int] | None) -> tuple[int, list[int]]:
        if values is None:
            return 0, []

        normalized = []
        for val in values:
            if val not in normalized:
                normalized.append(val)

        if not normalized:
            parser.error("--num-mtp-tokens expects at least one candidate when provided.")

        return normalized[0], normalized

    if args.performance_model == "profiling" and not args.profiling_database:
        parser.error("--profiling-database-path is required when using --performance-model profiling")

    def _normalize_and_validate(values: list[int] | None, arg_name: str, num_devices: int) -> list[int] | None:
        if values is None:
            return None
        normalized = []
        for val in values:
            if val > num_devices:
                raise ValueError(
                    f"--{arg_name} contains value {val}, which is larger than --num-devices ({num_devices})."
                )
            if val not in normalized:
                normalized.append(val)
        return normalized

    args.num_mtp_tokens, args.num_mtp_token_sizes = _normalize_mtp_token_values(args.num_mtp_tokens)
    # G3 / G2: dependents + three-way mutual exclusion before search / worker build.
    validate_draft_spec_cli_args(parser, args, check_mtp_candidates=True)
    if draft_method(args) is not None:
        # RFC: draft path MUST keep MTP disabled for search combinations.
        args.num_mtp_tokens = 0
        args.num_mtp_token_sizes = []
        resolve_draft_block_and_acceptance(args)

    args.tp_sizes = _normalize_and_validate(args.tp_sizes, "tp-sizes", args.num_devices)
    args.ep_sizes = _normalize_and_validate(args.ep_sizes, "ep-sizes", args.num_devices)
    args.moe_dp_sizes = _normalize_and_validate(args.moe_dp_sizes, "moe-dp-sizes", args.num_devices)
    # DCP reuses TP devices, so its candidates are bounded by TP (hence num_devices), not
    # by a separate device budget; the per-combination tp % dcp == 0 check happens below.
    args.dcp_sizes = _normalize_and_validate(args.dcp_sizes, "dcp-sizes", args.num_devices)

    tp_candidates, ep_candidates, moe_dp_candidates, mtp_candidates = resolve_parallel_search_candidates(
        args.tp_sizes,
        args.ep_sizes,
        args.moe_dp_sizes,
        args.num_mtp_token_sizes,
        args.num_mtp_tokens,
        args.num_devices,
    )
    dcp_candidates = resolve_search_sizes(args.dcp_sizes, args.num_devices, 1)
    total_combinations = count_search_combinations(
        tp_candidates,
        ep_candidates,
        moe_dp_candidates,
        mtp_candidates,
    ) * len(dcp_candidates)

    has_valid_combination = any(
        args.num_devices % tp == 0
        and args.num_devices % ep == 0
        and args.num_devices % (ep * moe_dp) == 0
        and tp % dcp == 0
        for tp in tp_candidates
        for ep in ep_candidates
        for moe_dp in moe_dp_candidates
        for dcp in dcp_candidates
    )
    if not has_valid_combination:
        parser.error(
            "No valid parallel combination is produced by the provided search arguments under current --num-devices."
        )

    args.search_combination_warning_emitted = False
    if args.max_search_combinations and total_combinations > args.max_search_combinations:
        args.search_combination_warning_emitted = True
        print(
            "[WARNING] Large number of parallel search combinations "
            f"({total_combinations} = TP:{len(tp_candidates)} x EP:{len(ep_candidates)} "
            f"x MOE-DP:{len(moe_dp_candidates)} x MTP:{len(mtp_candidates)} x DCP:{len(dcp_candidates)}). "
            "Optimization may take a long time. Consider narrowing --tp-sizes, --ep-sizes, "
            "--moe-dp-sizes, --num-mtp-tokens, or --dcp-sizes; "
            "or increase --max-search-combinations.",
            file=sys.stderr,
            flush=True,
        )

    return args


def main():
    start_time = time.time()
    args = arg_parse()
    print_logo()
    configure_std_logging(args, log_format=LOG_FORMAT)
    logger = logging.getLogger(__name__)

    apply_compilation_config(args.compilation_config)

    device_targets = check_device_targets(args, logger)
    if device_targets is None:
        return 1

    if isinstance(args.input_length, str) and (
        args.enable_optimize_prefill_decode_ratio
        or (args.disagg and (args.ttft_limits is None or args.tpot_limits is not None))
    ):
        logger.warning(
            "--input-length FILE currently supports aggregation runs or disaggregation "
            "prefill-only runs with --ttft-limit and without --tpot-limit."
        )
        return 1

    if isinstance(args.input_length, str):
        try:
            load_length_distribution(args.input_length)
        except ValueError as err:
            logger.error("Failed to load length distribution from %s: %s", args.input_length, err)
            return 1

    mtp_candidates = args.num_mtp_token_sizes or [args.num_mtp_tokens]
    invalid_num_mtp_tokens = [value for value in mtp_candidates if value > len(args.mtp_acceptance_rate) + 1]
    if invalid_num_mtp_tokens:
        logger.error(
            "num_mtp_tokens candidates %r exceed the supported mtp_acceptance_rate length (%r). Please check.",
            invalid_num_mtp_tokens,
            len(args.mtp_acceptance_rate),
        )
        return 1

    # Validate PD ratio optimization parameters. Use getattr for compatibility
    # with programmatic callers that provide a minimal argparse namespace.
    prefill_devices_per_instance = getattr(args, "prefill_devices_per_instance", None)
    decode_devices_per_instance = getattr(args, "decode_devices_per_instance", None)
    if args.enable_optimize_prefill_decode_ratio:
        if args.disagg:
            logger.error("--enable-optimize-prefill-decode-ratio cannot be used together with --disagg.")
            return 1
        if prefill_devices_per_instance is None or decode_devices_per_instance is None:
            logger.error(
                "Both --prefill-devices-per-instance and --decode-devices-per-instance "
                "are required when PD ratio optimization is enabled."
            )
            return 1
    elif prefill_devices_per_instance is not None or decode_devices_per_instance is not None:
        logger.error(
            "--prefill-devices-per-instance and --decode-devices-per-instance require "
            "--enable-optimize-prefill-decode-ratio. This mode cannot be used together with --disagg."
        )
        return 1

    # Terminal ASCII curves (plotext) run automatically when structurally allowed.
    plot_curves_allowed = len(device_targets) == 1 and not isinstance(args.input_length, str)

    logger.info("Starting experiments.")
    hw_rows = run_multi_device_loop(
        args,
        device_targets,
        plot_curves_allowed=plot_curves_allowed,
        logger=logger,
    )
    render_cross_hardware_summary(args, device_targets, hw_rows, logger=logger)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"All experiments completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    sys.exit(main() or 0)

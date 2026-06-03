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

import argparse
import logging
import sys
import time

from serving_cast.service.optimizer_curve_plots import (
    render_cross_hardware_summary,
    run_multi_device_loop,
)
from serving_cast.service.utils import (
    BatchRangeAction,
    OptimizerData,
    check_positive_float,
    check_positive_integer,
    resolve_search_sizes,
)
from tensor_cast import device_profiles  # noqa: F401
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)

from ..utils import (
    LOG_FORMAT,
    LOG_LEVELS,
    check_device_targets,
    check_prefix_cache_hit_rate,
    get_common_argparser,
)


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Get Best Throughput for given input/output sequence length and SLO limitations "
        "in aggregation mode or disaggregation mode.",
        parents=[get_common_argparser(reserved_memory_gb_default=10.0)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="+",
        default=None,
        metavar="DEVICE",
        help="Device profile(s) to evaluate. Multiple values enable cross-hardware summaries.",
    )
    parser.add_argument(
        "--input-length",
        type=check_positive_integer,
        required=True,
        help="The input length of the prompt.",
    )
    parser.add_argument(
        "--output-length",
        type=check_positive_integer,
        required=True,
        help="The expected output length.",
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
        default=0,
        help="Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek",
    )
    parser.add_argument(
        "--mtp-acceptance-rate",
        type=float,
        default=[0.9, 0.6, 0.4, 0.2],
        nargs="+",
        help="Acceptance rate list for MTP",
    )
    parser.add_argument(
        "--prefix-cache-hit-rate",
        type=check_prefix_cache_hit_rate,
        default=0.0,
        help="Prefix cache hit rate for prefill token reuse. This is a token-level approximation in [0, 1).",
    )
    model_group.add_argument(
        "--quantize-linear-action",
        type=QuantizeLinearAction,
        choices=list(QuantizeLinearAction),
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        help="Quantize all linear layers in the model from choices (currently only support symmetric quant)",
    )
    model_group.add_argument(
        "--mxfp4-group-size",
        type=check_positive_integer,
        default=32,
        help="Group size for MXFP4 quantization",
    )
    model_group.add_argument(
        "--quantize-attention-action",
        type=QuantizeAttentionAction,
        choices=list(QuantizeAttentionAction),
        default=QuantizeAttentionAction.DISABLED,
        help="Quantize the KV cache with the given action",
    )
    model_group.add_argument(
        "--tp-sizes",
        type=check_positive_integer,
        nargs="*",
        default=None,
        help="Enable TP search. Optional explicit TP sizes. "
        "If no value is provided, defaults to powers of 2 up to world_size.",
    )
    model_group.add_argument(
        "--ep-sizes",
        type=check_positive_integer,
        nargs="*",
        default=None,
        help="Enable EP search. Optional explicit EP sizes. "
        "If no value is provided, defaults to powers of 2 up to world_size.",
    )
    model_group.add_argument(
        "--moe-dp-sizes",
        type=check_positive_integer,
        nargs="*",
        default=None,
        help="Enable MOE-DP search. Optional explicit MOE-DP sizes. "
        "If no value is provided, defaults to powers of 2 up to world_size.",
    )
    service_group = parser.add_argument_group("Service Options")
    service_group.add_argument(
        "--ttft-limits",
        type=check_positive_float,
        default=None,
        help="TTFT constraints under which to search for the best throughput. None means no constraint.",
    )
    service_group.add_argument(
        "--tpot-limits",
        type=check_positive_float,
        default=None,
        help="TPOT constraints under which to search for the best throughput. None means no constraint.",
    )
    service_group.add_argument(
        "--max-prefill-tokens",
        type=check_positive_integer,
        default=8192,
        help="Max prefill tokens",
    )
    service_group.add_argument(
        "--batch-range",
        type=int,
        nargs="+",
        action=BatchRangeAction,
        default=None,
        help="Batch size range: [min max] or [max] (default: 1 for min, no limit for max)",
    )
    service_group.add_argument(
        "--serving-cost",
        type=float,
        default=0,
        help="Serving cost represents the cost of service delivery",
    )
    service_group.add_argument(
        "--disagg",
        action="store_true",
        help="If set, run disaggregation mode. disagg means disaggregation mode.",
    )
    service_group.add_argument(
        "--jobs",
        type=check_positive_integer,
        default=8,
        help="Number of parallel jobs.",
    )
    service_group.add_argument(
        "--concurrency-search-strategy",
        choices=["exponential", "linear_exponential"],
        default="exponential",
        help="Concurrency search strategy. The default is exponential.",
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
        help="Number of images per request. If omitted, reuse batch_size for backward compatibility.",
    )
    multimodal_group.add_argument(
        "--image-height",
        type=check_positive_integer,
        default=None,
        help="Height of the input images",
    )
    multimodal_group.add_argument(
        "--image-width",
        type=check_positive_integer,
        default=None,
        help="Width of the input images",
    )
    pd_ratio_group = parser.add_argument_group("PD Ratio Optimization Options")
    pd_ratio_group.add_argument(
        "--prefill-devices-per-instance",
        type=check_positive_integer,
        default=None,
        help="Number of devices per Prefill instance for PD ratio optimization",
    )
    pd_ratio_group.add_argument(
        "--decode-devices-per-instance",
        type=check_positive_integer,
        default=None,
        help="Number of devices per Decode instance for PD ratio optimization",
    )
    pd_ratio_group.add_argument(
        "--enable-optimize-prefill-decode-ratio",
        action="store_true",
        help="Enable PD ratio optimization mode",
    )
    args = parser.parse_args()
    if all(x is None for x in (args.tp_sizes, args.ep_sizes, args.moe_dp_sizes)):
        # Backward-compatible default: search TP only with default range.
        args.tp_sizes = []

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

    args.tp_sizes = _normalize_and_validate(args.tp_sizes, "tp-sizes", args.num_devices)
    args.ep_sizes = _normalize_and_validate(args.ep_sizes, "ep-sizes", args.num_devices)
    args.moe_dp_sizes = _normalize_and_validate(args.moe_dp_sizes, "moe-dp-sizes", args.num_devices)

    tp_candidates = resolve_search_sizes(args.tp_sizes, args.num_devices, args.num_devices)
    ep_candidates = resolve_search_sizes(args.ep_sizes, args.num_devices, args.num_devices)
    moe_dp_candidates = resolve_search_sizes(args.moe_dp_sizes, args.num_devices, 1)

    has_valid_combination = any(
        args.num_devices % tp == 0 and args.num_devices % ep == 0 and args.num_devices % (ep * moe_dp) == 0
        for tp in tp_candidates
        for ep in ep_candidates
        for moe_dp in moe_dp_candidates
    )
    if not has_valid_combination:
        parser.error(
            "No valid parallel combination is produced by the provided search arguments under current --num-devices."
        )

    return args


def main():
    start_time = time.time()
    args = arg_parse()
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level.lower()],
        format=LOG_FORMAT,
    )
    logger = logging.getLogger(__name__)

    device_targets = check_device_targets(args, logger)
    if device_targets is None:
        return 1

    effective_input_length = OptimizerData(
        input_length=args.input_length,
        prefix_cache_hit_rate=args.prefix_cache_hit_rate,
    ).get_effective_input_length()

    if (
        not args.disagg
        and not args.enable_optimize_prefill_decode_ratio
        and args.max_prefill_tokens < effective_input_length
    ):
        logger.error(
            "max_prefill_tokens (%r) is smaller than effective_input_length (%r). "
            "We currently do not have support for this scenario.",
            args.max_prefill_tokens,
            effective_input_length,
        )
        return 1

    if args.num_mtp_tokens > 0 and args.num_mtp_tokens > len(args.mtp_acceptance_rate) + 1:
        logger.error(
            "num_mtp_tokens (%r) is greater than the length of mtp_acceptance_rate (%r). Please check.",
            args.num_mtp_tokens,
            len(args.mtp_acceptance_rate),
        )
        return 1

    # Validate PD ratio optimization parameters
    if args.enable_optimize_prefill_decode_ratio:
        if args.disagg:
            logger.error("--enable-optimize-prefill-decode-ratio cannot be used together with --disagg.")
            return 1
        if args.prefill_devices_per_instance is None or args.decode_devices_per_instance is None:
            logger.error(
                "Both --prefill-devices-per-instance and --decode-devices-per-instance "
                "are required when PD ratio optimization is enabled."
            )
            return 1

    # Terminal ASCII curves (plotext) run automatically when structurally allowed.
    plot_curves_allowed = len(device_targets) == 1

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

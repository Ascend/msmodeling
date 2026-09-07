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

import json
import logging
import sys
import time

from cli.logo import print_logo
from cli.registry.argparse_adapter import build_argparser, parse_module_args
from cli.registry.modules import get_spec
from cli.spec_cli import configure_std_logging
from serving_cast.service.optimizer_curve_plots import (
    render_cross_hardware_summary,
    run_multi_device_loop,
)
from serving_cast.service.utils import (
    count_search_combinations,
    load_length_distribution,
    resolve_parallel_search_candidates,
    resolve_search_sizes,
)
from tensor_cast import device_profiles  # noqa: F401  (registers device profiles)
from tensor_cast.core.compilation_config import apply_compilation_config

from ..utils import (
    LOG_FORMAT,
    check_device_targets,
    draft_method,
    resolve_draft_block_and_acceptance,
    resolve_num_speculative_tokens_to_block,
)


def arg_parse():
    """Build the parser from the parameter registry and run post-parse normalization.

    Parser construction is declarative (cli/registry/modules/throughput_optimizer.py
    via build_argparser); everything below is search-candidate normalization and
    cross-field validation kept from the legacy entry point.
    """
    spec = get_spec("throughput_optimizer")
    parser = build_argparser(spec)
    args = parse_module_args(spec, parser)

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

    if args.performance_model == "profiling" and not args.profiling_database_path:
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
    # G2/G3 (draft vs MTP) run as registry validators inside parse_module_args
    # (wants_provided=True). When the draft path is taken, RFC requires MTP
    # disabled for search combinations — clear the candidates here.
    method = draft_method(args)
    if method is not None:
        args.num_mtp_tokens = 0
        args.num_mtp_token_sizes = []

        n_raw = getattr(args, "num_speculative_tokens", None)

        if isinstance(n_raw, list):
            # Explicit multi-value: dedup + validate non-negative (check_non_negative_integer already rejects < 0).
            n_candidates = []
            for v in n_raw:
                if v not in n_candidates:
                    n_candidates.append(v)
            if not n_candidates:
                parser.error("--num-speculative-tokens expects at least one candidate.")
        elif n_raw is None:
            # Omitted: dflash/dspark → builtin (C2); mtp requires explicit n.
            if method == "mtp":
                parser.error("--speculative-method mtp requires --num-speculative-tokens")
            builtin_n, _ = resolve_num_speculative_tokens_to_block(
                0,
                draft_model_config_path=getattr(args, "draft_model_config_path", None),
                explicit=False,
            )
            n_candidates = [builtin_n]
        else:
            n_candidates = [int(n_raw)]

        if any(int(v) == 0 for v in n_candidates):
            parser.error(
                "--speculative-method is set but --num-speculative-tokens contains 0. "
                "Omit --speculative-method for a baseline run, or pass n >= 1."
            )

        args.num_speculative_token_sizes = n_candidates
        args.num_mtp_token_sizes = list(n_candidates)
        # num_mtp_tokens stays 0 for the new entry (forced off); the search
        # uses num_mtp_token_sizes (N candidates) via the MTP search slot.
        args.num_mtp_tokens = 0

        # For single-N (builtin or single explicit), resolve block/acceptance on args.
        # For multi-N, resolution happens per-iteration in ParallelRunner.
        if len(n_candidates) == 1:
            args.num_speculative_tokens = n_candidates[0]
            resolve_draft_block_and_acceptance(args, argv=sys.argv[1:])
        else:
            args.num_speculative_tokens = n_candidates[0]

    args.tp_sizes = _normalize_and_validate(args.tp_sizes, "tp-sizes", args.num_devices)
    args.ep_sizes = _normalize_and_validate(args.ep_sizes, "ep-sizes", args.num_devices)
    args.moe_dp_sizes = _normalize_and_validate(args.moe_dp_sizes, "moe-dp-sizes", args.num_devices)
    # DCP reuses TP devices, so its candidates are bounded by TP (hence num_devices), not
    # by a separate device budget; the per-combination tp % dcp == 0 check happens below.
    args.dcp_sizes = _normalize_and_validate(args.dcp_sizes, "dcp-sizes", args.num_devices)

    # Parse pp_layer_partitions from JSON string to list[list[int]]
    if args.pp_layer_partitions is not None:
        try:
            parsed = json.loads(args.pp_layer_partitions)
        except json.JSONDecodeError as exc:
            parser.error(f"--pp-layer-partitions must be valid JSON: {exc}")
        if not isinstance(parsed, list) or not all(isinstance(p, list) for p in parsed):
            parser.error("--pp-layer-partitions must be a JSON list of lists, e.g. '[[31,30],[16,15,15,15]]'")
        args.pp_layer_partitions = [list(p) for p in parsed]

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
        spec_label = "Spec" if draft_method(args) is not None else "MTP"
        print(
            "[WARNING] Large number of parallel search combinations "
            f"({total_combinations} = TP:{len(tp_candidates)} x EP:{len(ep_candidates)} "
            f"x MOE-DP:{len(moe_dp_candidates)} x {spec_label}:{len(mtp_candidates)} x DCP:{len(dcp_candidates)}). "
            "Optimization may take a long time. Consider narrowing --tp-sizes, --ep-sizes, "
            "--moe-dp-sizes, --num-mtp-tokens/--num-speculative-tokens, or --dcp-sizes; "
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
    # The acceptance_rate length check only applies to the legacy MTP fold path
    # (sum(rates[:N])+1). New speculative entry (dflash/dspark/--speculative-method mtp)
    # uses accept+1 fold and does not need rates alignment.
    if draft_method(args) is None:
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

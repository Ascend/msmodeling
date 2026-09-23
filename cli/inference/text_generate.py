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

import logging

from cli.logo import print_logo
from cli.registry.argparse_adapter import build_argparser, parse_module_args
from cli.registry.modules import get_spec
from cli.spec_cli import configure_std_logging
from tensor_cast import config, device_profiles  # noqa: F401
from tensor_cast.core.compilation_config import apply_compilation_config

from ..utils import (
    LOG_FORMAT,
    draft_method,
    resolve_draft_block_and_acceptance,
)

# Supported performance model types
SUPPORTED_PERFORMANCE_MODELS = ["analytic", "calibrated", "profiling"]


def arg_parse(argv=None):
    """Parse CLI args and apply draft-spec block resolution.

    Parser is generated from the parameter registry; draft-spec G2/G3 checks
    run as registry validators inside parse_module_args (wants_provided=True
    — they see which flags were explicitly passed). Only the block-size
    resolution remains here: it MUTATES args (fills draft_block_size), which
    is post-parse business logic, not validation.
    """
    spec = get_spec("text_generate")
    parser = build_argparser(spec)
    args = parse_module_args(spec, parser, argv)
    resolve_draft_block_and_acceptance(args)
    # Bridge --speculative-method mtp --num-speculative-tokens N onto the
    # legacy num_mtp_tokens field so ConfigResolver/MtpWrapper match
    # --num-mtp-tokens N.
    method = draft_method(args)
    if method is not None:
        n = int(getattr(args, "num_speculative_tokens", 0) or 0)
        if n == 0:
            parser.error(
                "--speculative-method is set but --num-speculative-tokens is explicitly 0 "
                "(disabled). Omit --speculative-method for a baseline run, or pass n >= 1."
            )
        if method == "mtp" and int(getattr(args, "num_mtp_tokens", 0) or 0) == 0:
            args.num_mtp_tokens = n
    if args.performance_model is None:
        args.performance_model = ["analytic"]
    if "calibrated" in args.performance_model and not args.analytic_calibration_profile:
        parser.error("--analytic-calibration-profile is required when using --performance-model calibrated")
    if args.export_empirical_metrics_file and "profiling" not in args.performance_model:
        parser.error("--export-empirical-metrics requires --performance-model profiling")
    if args.fusion_plugin and not args.compile:
        parser.error("--fusion-plugin requires --compile (else the fusion never fires)")
    # Validate fusion plugin paths eagerly so an invalid plugin is caught
    # before ModelRunner construction rather than silently falling back.
    if args.fusion_plugin:
        from tensor_cast.plugins.validator import validate_plugin

        for plugin_path in args.fusion_plugin:
            result = validate_plugin(plugin_path)
            if not result:
                parser.error(f"--fusion-plugin {plugin_path}: validation failed at {result.layer}: {result.detail}")
    return args


def align_decode_query_length(args, logger=None) -> None:
    """Align decode ``--query-length`` to ``n + 1`` for MTP / DFlash / DSpark.

    Both MTP entries (legacy ``--num-mtp-tokens`` and
    ``--speculative-method mtp --num-speculative-tokens``) use the same window
    so decode simulation shapes stay identical.
    """
    method = draft_method(args)
    n = int(getattr(args, "num_speculative_tokens", 0) or 0)
    if method is None and int(getattr(args, "num_mtp_tokens", 0) or 0) > 0:
        n = int(args.num_mtp_tokens)
        method = "mtp"
    if not args.decode or n < 1:
        return
    decode_query_len = n + 1
    if int(args.query_length) == decode_query_len:
        return
    label = {"dspark": "DSpark", "dflash": "Dflash", "mtp": "MTP"}.get(method, method)
    if logger is not None:
        logger.warning(
            "%s decode sets --query-length to num_speculative_tokens+1 (%d); was %d",
            label,
            decode_query_len,
            args.query_length,
        )
    args.query_length = decode_query_len


def main():
    """
    Main function to parse arguments and run the inference simulation.

    Parser is generated from the declarative parameter registry
    (cli/registry/modules/text_generate.py) via build_argparser(); this
    module only holds the post-parse business logic.
    """
    args = arg_parse()
    print_logo()
    configure_std_logging(args, log_format=LOG_FORMAT)
    logger = logging.getLogger(__name__)

    if args.graph_log_url:
        config.compilation.debug.graph_log_url = args.graph_log_url
    apply_compilation_config(args.compilation_config)

    align_decode_query_length(args, logger=logger)

    # import here to make sure the logger level is set
    logger.info("Importing core modules...")
    from tensor_cast.core.input_generator import generate_inputs
    from tensor_cast.core.model_runner import ModelRunner
    from tensor_cast.core.user_config import UserInputConfig

    logger.debug("Core modules imported")

    logger.info("Initializing user configuration...")
    user_input = UserInputConfig.from_args(args)
    logger.debug("User configuration initialized: %s", user_input)

    # Load fusion plugin(s) into the global tables before ModelRunner is built,
    # so Phase 3 picks them up at compile time. Additive hook (RFC §3.3a):
    # arg_parse() already validated each plugin path; load_plugin is
    # idempotent so calling it again here is harmless.
    if args.fusion_plugin:
        from tensor_cast.plugins.loader import load_plugin

        for plugin_path in args.fusion_plugin:
            load_plugin(plugin_path)

    logger.info("Initializing ModelRunner")
    model_runner = ModelRunner(user_input)
    logger.info("ModelRunner initialization completed: %s", model_runner)

    logger.info("Running inference...")
    metrics = model_runner.run_inference(generate_inputs_func=generate_inputs)
    metrics.print_info()

    # Export metrics JSON for offline M6 computation
    if args.export_empirical_metrics_file:
        from pathlib import Path

        from tensor_cast.performance_model.empirical import EmpiricalPerformanceModel
        from tensor_cast.performance_model.metrics_collector import MetricsCollector

        for pm in model_runner.perf_models:
            if isinstance(pm, EmpiricalPerformanceModel):
                collector = MetricsCollector()
                collector.collect_from_records(pm.op_records)
                collector.export_hit_miss_report(
                    output_path=Path(args.export_empirical_metrics_file),
                )
                break


if __name__ == "__main__":
    main()

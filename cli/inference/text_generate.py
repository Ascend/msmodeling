import logging

from cli.logo import print_logo
from cli.spec_cli import (
    METAVAR_DIR,
    METAVAR_FILE,
    METAVAR_FLOAT,
    METAVAR_N,
    SpecArgumentParser,
    add_option,
    configure_std_logging,
    inherit_deprecated,
    kebab_choice_metavar,
    make_enum_type,
    make_token_type,
    parse_args as spec_parse_args,
)
from tensor_cast import config, device_profiles  # noqa: F401
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
    check_positive_integer,
    check_prefix_cache_hit_rate,
    get_common_argparser,
    require_model_id,
)

# Supported performance model types
SUPPORTED_PERFORMANCE_MODELS = ["analytic", "profiling"]


def main():
    """
    Main function to parse arguments and run the inference simulation.
    """
    common_parser = get_common_argparser()
    parser = SpecArgumentParser(
        prog="msmodeling inference text-generate",
        description="Run a simulated LLM inference pass and dump the perf result.",
        parents=[common_parser],
        examples=(
            "# Prefill one query of 128 tokens\n"
            "msmodeling inference text-generate Qwen/Qwen3-32B "
            "--num-queries 1 --query-length 128 --device TEST_DEVICE\n"
            "# Decode with tensor parallel\n"
            "msmodeling inference text-generate Qwen/Qwen3-32B "
            "--num-queries 8 --query-length 1 --context-length 4096 --decode "
            "--tp-size 8"
        ),
        output_help="Metrics table on stdout. Optional chrome trace via --chrome-trace-file.",
    )
    inherit_deprecated(parser, common_parser)
    parse_linear, linear_meta = make_enum_type(QuantizeLinearAction, "--quantize-linear-action")
    parse_attn, attn_meta = make_enum_type(QuantizeAttentionAction, "--quantize-attention-action")
    parse_cc, cc_meta = make_token_type(COMPILATION_CONFIG_OPTIONS, "--compilation-config", store_canonical="snake")

    llm_group = parser.add_argument_group("LLM Options")
    llm_group.add_argument(
        "--num-queries",
        type=check_positive_integer,
        required=True,
        metavar=METAVAR_N,
        help="Number of parallel inference queries to execute in a single batch.",
    )
    llm_group.add_argument(
        "--query-length",
        type=check_positive_integer,
        required=True,
        metavar=METAVAR_N,
        help="Length (in tokens) of new input sequence for each query.",
    )
    llm_group.add_argument(
        "--context-length",
        type=int,
        default=0,
        metavar=METAVAR_N,
        help="Length (in tokens) of existing context for each query.",
    )
    llm_group.add_argument(
        "--decode",
        action="store_true",
        help="Enable autoregressive decoding mode for text generation.",
    )
    llm_group.add_argument(
        "--prefix-cache-hit-rate",
        type=check_prefix_cache_hit_rate,
        default=0.0,
        metavar=METAVAR_FLOAT,
        help="Prefix cache hit rate for prefill token reuse in [0, 1).",
    )
    llm_group.add_argument(
        "--num-mtp-tokens",
        type=int,
        default=0,
        metavar=METAVAR_N,
        help="Number of Multi-Token Prediction (MTP) tokens. 0 = disabled. "
        "Only supports models with MTP capability (e.g., DeepSeek).",
    )
    add_option(
        llm_group,
        "--no-repetition",
        dest="disable_repetition",
        action="store_true",
        default=False,
        help="Do not reuse repeated transformer layers to save runtime cost.",
        aliases=("--disable-repetition",),
    )

    optim_group = parser.add_argument_group("Optimization Options")
    optim_group.add_argument(
        "--compile",
        action="store_true",
        help="If set, invoke torch.compile() on the model before inference.",
    )
    optim_group.add_argument(
        "--compile-allow-graph-break",
        action="store_true",
        help="Allow graph breaks during torch.compile() for models with dynamic control flow.",
    )
    optim_group.add_argument(
        "--compilation-config",
        nargs="*",
        default=None,
        type=parse_cc,
        metavar=cc_meta,
        help="Enable specific compilation features. If omitted, all compilation features stay disabled.",
    )
    optim_group.add_argument(
        "--fusion-plugin",
        action="append",
        default=None,
        metavar="PATH",
        help="Path to a fusion plugin .py to load before model construction "
        "(see RFC manual_fusion_eval §3.3a). May be repeated to load several. "
        "Requires --compile; without it the pattern registers but never fires.",
    )

    quant_group = parser.add_argument_group("Quantization Options")
    quant_group.add_argument(
        "--quantize-linear-action",
        type=parse_linear,
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        metavar=linear_meta,
        help="Quantize all linear layers (symmetric quant).",
    )
    quant_group.add_argument(
        "--quantize-non-expert-linear-action",
        type=parse_linear,
        default=QuantizeLinearAction.DISABLED,
        metavar=linear_meta,
        help=(
            "Separate quantization type for non-expert linear layers. Routed MoE experts keep --quantize-linear-action."
        ),
    )
    quant_group.add_argument(
        "--quantize-lmhead",
        action="store_true",
        help="Quantize the LM Head. Off by default because it usually hurts accuracy.",
    )
    quant_group.add_argument(
        "--mxfp4-group-size",
        type=check_positive_integer,
        default=32,
        metavar=METAVAR_N,
        help="Group size for MXFP4 quantization.",
    )
    quant_group.add_argument(
        "--quantize-attention-action",
        type=parse_attn,
        default=QuantizeAttentionAction.DISABLED,
        metavar=attn_meta,
        help="Quantize the KV cache with the given action.",
    )

    debug_group = parser.add_argument_group("Debugging Options")
    add_option(
        debug_group,
        "--graph-log-path",
        dest="graph_log_url",
        metavar=METAVAR_DIR,
        help=(
            "Directory for dumping compiled graphs when --compile is on. "
            "Each compile pass writes files under this directory."
        ),
        aliases=("--graph-log-url", "--graph-log-file"),
    )
    debug_group.add_argument(
        "--dump-input-shapes",
        action="store_true",
        help="Group the result table average by input shapes.",
    )
    debug_group.add_argument(
        "--dump-op-bound-results",
        action="store_true",
        help="Dump per-operator memory/communication/MMA/GP bound ratios.",
    )
    add_option(
        debug_group,
        "--chrome-trace-file",
        dest="chrome_trace",
        metavar=METAVAR_FILE,
        help="Write a chrome trace JSON file.",
        aliases=("--chrome-trace",),
    )
    debug_group.add_argument(
        "--num-hidden-layers-override",
        type=int,
        default=0,
        metavar=METAVAR_N,
        help="Override the number of hidden layers, for debugging only.",
    )

    par_group = parser.add_argument_group("Parallelism Options")
    par_group.add_argument(
        "--tp-size",
        dest="tp_size",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Tensor parallel size for the whole model.",
    )
    par_group.add_argument(
        "--pp-size",
        dest="pp_size",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Pipeline parallel size for the whole model.",
    )
    par_group.add_argument(
        "--dcp-size",
        dest="dcp_size",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Decode Context Parallel size. Reuses TP devices and must divide --tp-size.",
    )
    par_group.add_argument(
        "--dp-size",
        dest="dp_size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Data parallel size for the whole model.",
    )
    par_group.add_argument(
        "--ep-size",
        dest="ep_size",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Expert parallel size.",
    )
    par_group.add_argument(
        "--o-proj-tp-size",
        dest="o_proj_tp_size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Tensor parallel size for attn o_proj.",
    )
    par_group.add_argument(
        "--o-proj-dp-size",
        dest="o_proj_dp_size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Data parallel size for attn o_proj.",
    )
    par_group.add_argument(
        "--mlp-tp-size",
        dest="mlp_tp_size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Tensor parallel size for MLP layers.",
    )
    par_group.add_argument(
        "--mlp-dp-size",
        dest="mlp_dp_size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Data parallel size for MLP layers.",
    )
    par_group.add_argument(
        "--lmhead-tp-size",
        dest="lmhead_tp_size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Tensor parallel size for the LM head.",
    )
    par_group.add_argument(
        "--lmhead-dp-size",
        dest="lmhead_dp_size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Data parallel size for the LM head.",
    )
    par_group.add_argument(
        "--moe-tp-size",
        dest="moe_tp_size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Tensor parallel size for experts.",
    )
    par_group.add_argument(
        "--moe-dp-size",
        dest="moe_dp_size",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Data parallel size for experts.",
    )
    par_group.add_argument(
        "--word-embedding-tp",
        dest="word_embedding_tp",
        type=str,
        choices=[mode.value for mode in WordEmbeddingTPMode],
        default=None,
        metavar="{col,row}",
        help="Word embedding tensor parallel mode. Omitted disables embedding TP.",
    )
    par_group.add_argument(
        "--enable-redundant-experts",
        action="store_true",
        help="Use redundant experts. If shared-expert externalization is off, each device adds one "
        "redundant expert. If it is on and every device has the same number of routing experts, "
        "each device hosting routing experts also adds one redundant expert.",
    )
    par_group.add_argument(
        "--enable-shared-expert-tp",
        action="store_true",
        help="Enable vLLM-style tensor parallel for shared experts. "
        "This uses dense-MLP TP for shared_experts with delayed down_proj reduction.",
    )
    par_group.add_argument(
        "--enable-external-shared-experts",
        action="store_true",
        help="Whether or not to implement external shared experts",
    )
    par_group.add_argument(
        "--host-external-shared-experts",
        action="store_true",
        help="Whether to have the current device host the external shared experts",
    )
    par_group.add_argument(
        "--vision-tp-size",
        dest="vision_tp_size",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Vision tensor parallel degree. Default 1 keeps vision modules unsharded.",
    )

    multimodal_group = parser.add_argument_group("MultiModal Options")
    multimodal_group.add_argument(
        "--image-batch-size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Batch size for image processing.",
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

    parser.add_argument(
        "--remote-source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        metavar="{huggingface,modelscope}",
        help="The remote source for the model.",
    )
    parser.add_argument(
        "--performance-model",
        action="append",
        default=None,
        choices=SUPPORTED_PERFORMANCE_MODELS,
        metavar=kebab_choice_metavar(SUPPORTED_PERFORMANCE_MODELS),
        help="Performance model type(s). Repeat the option to select more than one. "
        "'analytic': Roofline model. 'profiling': empirical model (requires --profiling-database-path).",
    )
    add_option(
        parser,
        "--profiling-database-path",
        dest="profiling_database",
        type=str,
        default=None,
        metavar=METAVAR_DIR,
        help="Directory of the profiling database for 'profiling' mode.",
        aliases=("--profiling-database",),
    )
    add_option(
        parser,
        "--disable-profiling-interpolation",
        dest="disable_profiling_interpolation",
        action="store_true",
        help="Use exact and partial profiling matches only with --performance-model profiling.",
    )
    add_option(
        parser,
        "--export-empirical-metrics-file",
        dest="export_empirical_metrics",
        type=str,
        default=None,
        metavar=METAVAR_FILE,
        help="(developer only) Export M1-M5 metrics report as JSON. Requires --performance-model profiling.",
        aliases=("--export-empirical-metrics",),
    )

    args = spec_parse_args(parser)
    require_model_id(parser, args)
    print_logo()
    configure_std_logging(args, log_format=LOG_FORMAT)
    logger = logging.getLogger(__name__)

    if args.graph_log_url:
        config.compilation.debug.graph_log_url = args.graph_log_url
    apply_compilation_config(args.compilation_config)

    # Set default performance_model if not specified
    if args.performance_model is None:
        args.performance_model = ["analytic"]

    # Validate developer-only options
    if args.export_empirical_metrics and "profiling" not in args.performance_model:
        parser.error("--export-empirical-metrics requires --performance-model profiling")

    # Fusion plugin requires compilation: the fusion is a compile-time fx graph
    # rewrite (Phase 3); without --compile the pattern registers but never fires
    # and the estimate silently equals the no-plugin baseline (RFC §3.3a).
    if args.fusion_plugin and not args.compile:
        parser.error("--fusion-plugin requires --compile (else the fusion never fires)")

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
    # shares validate_plugin+load_plugin with the Python API; validate first so
    # an invalid plugin is caught before ModelRunner construction rather than
    # silently falling back to the no-plugin baseline.
    if args.fusion_plugin:
        from tensor_cast.plugins.loader import load_plugin
        from tensor_cast.plugins.validator import validate_plugin

        for plugin_path in args.fusion_plugin:
            result = validate_plugin(plugin_path)
            if not result:
                parser.error(f"--fusion-plugin {plugin_path}: validation failed at {result.layer}: {result.detail}")
            # validate_plugin's L2 already imported+registered the plugin;
            # load_plugin is an idempotent no-op here, kept for consistency.
            load_plugin(plugin_path)

    logger.info("Initializing ModelRunner")
    model_runner = ModelRunner(user_input)
    logger.info("ModelRunner initialization completed: %s", model_runner)

    logger.info("Running inference...")
    metrics = model_runner.run_inference(generate_inputs_func=generate_inputs)
    metrics.print_info()

    # Export metrics JSON for offline M6 computation
    if args.export_empirical_metrics:
        from pathlib import Path

        from tensor_cast.performance_model.empirical import EmpiricalPerformanceModel
        from tensor_cast.performance_model.metrics_collector import MetricsCollector

        for pm in model_runner.perf_models:
            if isinstance(pm, EmpiricalPerformanceModel):
                collector = MetricsCollector()
                collector.collect_from_records(pm.op_records)
                collector.export_hit_miss_report(
                    output_path=Path(args.export_empirical_metrics),
                )
                break


if __name__ == "__main__":
    main()

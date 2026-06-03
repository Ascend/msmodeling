import argparse
import logging

from tensor_cast import config, device_profiles  # noqa: F401
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.model_config import WordEmbeddingTPMode
from ..utils import (
    check_positive_integer,
    check_prefix_cache_hit_rate,
    get_common_argparser,
    LOG_FORMAT,
    LOG_LEVELS,
)

# Supported performance model types
SUPPORTED_PERFORMANCE_MODELS = ["analytic", "profiling"]


def main():
    """
    Main function to parse arguments and run the inference simulation.
    """
    common_parser = get_common_argparser()
    parser = argparse.ArgumentParser(
        description="Run a simulated LLM inference pass and dump the perf result.",
        parents=[common_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    llm_group = parser.add_argument_group("LLM Options")
    llm_group.add_argument(
        "--num-queries",
        type=check_positive_integer,
        required=True,
        help="Number of parallel inference queries to execute in a single batch.",
    )
    llm_group.add_argument(
        "--query-length",
        type=check_positive_integer,
        required=True,
        help="Length (in tokens) of new input sequence for each query.",
    )
    llm_group.add_argument(
        "--context-length",
        type=int,
        default=0,
        help="Length (in tokens) of existing context for each query. Default: 0.",
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
        help="Prefix cache hit rate for prefill token reuse. This is a token-level approximation in [0, 1).",
    )
    llm_group.add_argument(
        "--num-mtp-tokens",
        type=int,
        default=0,
        help="Number of Multi-Token Prediction (MTP) tokens. 0 = disabled. "
        "Only supports models with MTP capability (e.g., DeepSeek).",
    )
    llm_group.add_argument(
        "--disable-repetition",
        action="store_true",
        help="Preserve the original behavior of the transformer models. Do not leverage the repetition "
        "pattern of the transformer models to save runtime cost",
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
        "--enable-sequence-parallel",
        action="store_true",
        help="Enable the sequence parallel graph rewrite pass during compilation.",
    )

    quant_group = parser.add_argument_group("Quantization Options")
    quant_group.add_argument(
        "--quantize-linear-action",
        type=QuantizeLinearAction,
        choices=list(QuantizeLinearAction),
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        help="Quantize all linear layers in the model from choices (currently only support symmetric quant)",
    )
    quant_group.add_argument(
        "--quantize-lmhead",
        action="store_true",
        help="Whether to quantize LM Head, off by default since quantizing LM Head usually impact accuracy a lot",
    )
    quant_group.add_argument(
        "--mxfp4-group-size",
        type=check_positive_integer,
        default=32,
        help="Group size for MXFP4 quantization",
    )
    quant_group.add_argument(
        "--quantize-attention-action",
        type=QuantizeAttentionAction,
        choices=list(QuantizeAttentionAction),
        default=QuantizeAttentionAction.DISABLED,
        help="Quantize the KV cache with the given action",
    )

    debug_group = parser.add_argument_group("Debugging Options")
    debug_group.add_argument(
        "--graph-log-url",
        help="For debug: the path for dumping the compiled graphs if compile is on",
    )
    debug_group.add_argument(
        "--dump-input-shapes",
        action="store_true",
        help="If set, group the table average by input shapes",
    )
    debug_group.add_argument(
        "--chrome-trace",
        help="Generate chrome trace file",
    )
    debug_group.add_argument(
        "--num-hidden-layers-override",
        type=int,
        default=0,
        help="Override the number of hidden layers, for debugging only",
    )

    par_group = parser.add_argument_group("Parallelism Options")
    par_group.add_argument(
        "--tp-size",
        type=check_positive_integer,
        default=1,
        help="The tp size for the whole model",
    )
    par_group.add_argument(
        "--dp-size",
        type=check_positive_integer,
        default=None,
        help="The dp size for the whole model",
    )
    par_group.add_argument(
        "--ep-size",
        type=check_positive_integer,
        default=1,
        help="The ep size for experts",
    )
    par_group.add_argument(
        "--o-proj-tp-size",
        type=check_positive_integer,
        default=None,
        help="The tp size for attn o_proj layer",
    )
    par_group.add_argument(
        "--o-proj-dp-size",
        type=check_positive_integer,
        default=None,
        help="The dp size for attn o_proj layer",
    )
    par_group.add_argument(
        "--mlp-tp-size",
        type=check_positive_integer,
        default=None,
        help="The tp size for mlp layer, can override tp-size for mlp layer",
    )
    par_group.add_argument(
        "--mlp-dp-size",
        type=check_positive_integer,
        default=None,
        help="The dp size for mlp layer, can override dp-size for mlp layer",
    )
    par_group.add_argument(
        "--lmhead-tp-size",
        type=check_positive_integer,
        default=None,
        help="The tp size for lm head, can override tp-size for lm head",
    )
    par_group.add_argument(
        "--lmhead-dp-size",
        type=check_positive_integer,
        default=None,
        help="The dp size for lm head, can override dp-size for lm head",
    )
    par_group.add_argument(
        "--moe-tp-size",
        type=check_positive_integer,
        default=None,
        help="The tp size for experts, can override tp-size for experts",
    )
    par_group.add_argument(
        "--moe-dp-size",
        type=check_positive_integer,
        default=1,
        help="The dp size for experts, can override dp-size for experts",
    )
    par_group.add_argument(
        "--word-embedding-tp",
        type=str,
        choices=[mode.value for mode in WordEmbeddingTPMode],
        default=None,
        help="Enable word embedding tensor parallel with mode {'col','row'}. If omitted, embedding TP is disabled.",
    )
    par_group.add_argument(
        "--enable-redundant-experts",
        action="store_true",
        help="Whether or not to use redundant experts. When this flag is True: "
        "if the externalization of shared experts is not enabled at this time, "
        "each device will add one redundant expert. If the externalization of shared experts is enabled "
        "and the number of routing experts on each device is the same, "
        "then each device hosting the routing experts will also add one redundant expert.",
    )
    par_group.add_argument(
        "--enable-shared-expert-tp",
        action="store_true",
        help="Enable vLLM-style tensor parallel for shared experts. "
        "This uses dense-MLP TP for shared_experts with delayed down_proj reduction.",
    )
    par_group.add_argument(
        "--enable-dispatch-ffn-combine",
        action="store_true",
        help="Enable dispatch_ffn_combine fusion pattern during compilation.",
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

    multimodal_group = parser.add_argument_group("MultiModal Options")
    multimodal_group.add_argument(
        "--image-batch-size",
        type=check_positive_integer,
        default=None,
        help="Batch size for image processing",
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

    parser.add_argument(
        "--remote-source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="The remote source for the model",
    )
    parser.add_argument(
        "--performance-model",
        action="append",
        default=None,
        choices=SUPPORTED_PERFORMANCE_MODELS,
        help="Performance model type(s). Can specify one or more models. "
        "'analytic': Roofline model (default, no data required). "
        "'profiling': EmpiricalPerformanceModel backed by Profiling CSV database "
        "(exact match, requires --profiling-database). "
        "Example: --performance-model analytic --performance-model profiling",
    )
    parser.add_argument(
        "--profiling-database",
        type=str,
        default=None,
        help="Path to the performance database directory for 'profiling' mode. "
        "The directory must contain op_mapping.yaml and per-kernel-type CSV files, "
        "e.g. tensor_cast/performance_model/profiling_database/data/atlas_a3_752t_128g/vllm_ascend/v0.13.0/",
    )
    parser.add_argument(
        "--export-empirical-metrics",
        type=str,
        default=None,
        help="(developer only) Export M1-M5 metrics report as JSON for offline M6 computation. "
        "Requires --performance-model profiling",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level.lower()],
        format=LOG_FORMAT,
    )
    logger = logging.getLogger(__name__)

    if args.graph_log_url:
        config.compilation.debug.graph_log_url = args.graph_log_url
    config.compilation.passes.enable_sequence_parallel = args.enable_sequence_parallel
    config.compilation.fusion_patterns.enable_dispatch_ffn_combine = args.enable_dispatch_ffn_combine

    selected_embedding_tp_mode = args.word_embedding_tp
    args.word_embedding_tp = selected_embedding_tp_mode is not None
    args.word_embedding_tp_mode = selected_embedding_tp_mode or WordEmbeddingTPMode.col.value

    # Set default performance_model if not specified
    if args.performance_model is None:
        args.performance_model = ["analytic"]

    # Validate developer-only options
    if args.export_empirical_metrics and "profiling" not in args.performance_model:
        parser.error("--export-empirical-metrics requires --performance-model profiling")

    # import here to make sure the logger level is set
    logger.info("Importing core modules...")
    from tensor_cast.core.input_generator import generate_inputs
    from tensor_cast.core.model_runner import ModelRunner
    from tensor_cast.core.user_config import UserInputConfig

    logger.debug("Core modules imported")

    logger.info("Initializing user configuration...")
    user_input = UserInputConfig.from_args(args)
    logger.debug("User configuration initialized: %s", user_input)

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

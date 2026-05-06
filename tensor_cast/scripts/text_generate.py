import argparse
import logging

from .. import config, device_profiles  # noqa: F401
from ..core.input_generator import generate_inputs
from ..core.model_runner import ModelRunner
from ..core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from ..core.user_config import UserInputConfig
from ..device import DeviceProfile
from ..model_config import WordEmbeddingTPMode

from ..utils import check_dependencies
from .utils import check_positive_integer, LOG_LEVELS


def main():
    check_dependencies()
    """
    Main function to parse arguments and run the inference simulation.
    """
    # TODO: add parallel configuration
    # TODO: add quantization configuration
    parser = argparse.ArgumentParser(
        description="Run a simulated LLM inference pass and dump the perf result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=list(DeviceProfile.all_device_profiles.keys()),
        default="TEST_DEVICE",
        help="The device type for simulation.",
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID from Hugging Face (e.g., 'meta-llama/Llama-2-7b-hf').",
    )
    parser.add_argument(
        "--num-queries",
        type=check_positive_integer,
        required=True,
        help="Number of inference queries to run in a batch.",
    )
    parser.add_argument(
        "--query-length",
        type=check_positive_integer,
        required=True,
        help="The length of the new input tokens for each query.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=0,
        help="The context length for each query. Defaults to 0.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="If set, invoke torch.compile() on the model before inference.",
    )
    parser.add_argument(
        "--compile-allow-graph-break",
        action="store_true",
        help="If set, invoke torch.compile() on the model before inference.",
    )
    parser.add_argument(
        "--enable-multistream",
        action="store_true",
        default=True,
        help=(
            "Enable compiler-driven multi-stream simulation for torch.compile path. "
            "Enabled by default."
        ),
    )
    parser.add_argument(
        "--dump-input-shapes",
        action="store_true",
        help="If set, group the table average by input shapes",
    )
    parser.add_argument(
        "--chrome-trace",
        type=str,
        default=None,
        help="Generate chrome trace file",
    )
    parser.add_argument(
        "--quantize-linear-action",
        type=QuantizeLinearAction,
        choices=list(QuantizeLinearAction),
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        help="Quantize all linear layers in the model from choices (currently only support symmetric quant)",
    )
    parser.add_argument(
        "--quantize-lmhead",
        action="store_true",
        help="Whether to quantize LM Head, off by default since quantizing LM Head usually impact accuracy a lot",
    )
    parser.add_argument(
        "--mxfp4-group-size",
        type=check_positive_integer,
        default=32,
        help="Group size for MXFP4 quantization",
    )
    parser.add_argument(
        "--quantize-attention-action",
        type=QuantizeAttentionAction,
        choices=list(QuantizeAttentionAction),
        default=QuantizeAttentionAction.DISABLED,
        help="Quantize the KV cache with the given action",
    )
    parser.add_argument(
        "--graph-log-url",
        type=str,
        default=None,
        help="For debug: the path for dumping the compiled graphs if compile is on",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="info",
        help="Set the logging level",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Whether we are doing decode",
    )
    parser.add_argument(
        "--num-mtp-tokens",
        type=int,
        default=0,
        help="Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek",
    )
    parser.add_argument(
        "--num-hidden-layers-override",
        type=int,
        default=0,
        help="Override the number of hidden layers, for debugging only",
    )
    parser.add_argument(
        "--disable-repetition",
        action="store_true",
        help="Preserve the original behavior of the transformer models. Do not leverage the repetition "
        "pattern of the transformer models to save runtime cost",
    )
    parser.add_argument(
        "--reserved-memory-gb",
        type=float,
        default=0,
        help="Size of reserved device memory (in GB) that we cannot use from applications.",
    )
    # ========== ParallelConfig Parameters ==========
    parser.add_argument(
        "--world-size",
        type=check_positive_integer,
        default=1,
        help="The total number of processes",
    )
    parser.add_argument(
        "--tp-size",
        type=check_positive_integer,
        default=1,
        help="The tp size for the whole model",
    )
    parser.add_argument(
        "--dp-size",
        type=check_positive_integer,
        default=None,
        help="The dp size for the whole model",
    )
    parser.add_argument(
        "--o-proj-tp-size",
        type=check_positive_integer,
        default=None,
        help="The tp size for attn o_proj layer",
    )
    parser.add_argument(
        "--o-proj-dp-size",
        type=check_positive_integer,
        default=None,
        help="The dp size for attn o_proj layer",
    )
    parser.add_argument(
        "--mlp-tp-size",
        type=check_positive_integer,
        default=None,
        help="The tp size for mlp layer, can override tp-size for mlp layer",
    )
    parser.add_argument(
        "--mlp-dp-size",
        type=check_positive_integer,
        default=None,
        help="The dp size for mlp layer, can override dp-size for mlp layer",
    )
    parser.add_argument(
        "--lmhead-tp-size",
        type=check_positive_integer,
        default=None,
        help="The tp size for lm head, can override tp-size for lm head",
    )
    parser.add_argument(
        "--lmhead-dp-size",
        type=check_positive_integer,
        default=None,
        help="The dp size for lm head, can override dp-size for lm head",
    )
    parser.add_argument(
        "--moe-dp-size",
        type=check_positive_integer,
        default=1,
        help="The dp size for experts, can override dp-size for experts",
    )
    parser.add_argument(
        "--moe-tp-size",
        type=check_positive_integer,
        default=None,
        help="The tp size for experts, can override tp-size for experts",
    )
    parser.add_argument(
        "--ep-size",
        type=check_positive_integer,
        default=1,
        help="The ep size for experts",
    )
    parser.add_argument(
        "--word-embedding-tp",
        type=str,
        choices=[mode.value for mode in WordEmbeddingTPMode],
        default=None,
        help="Enable word embedding tensor parallel with mode {'col','row'}. "
        "If omitted, embedding TP is disabled.",
    )
    parser.add_argument(
        "--enable-redundant-experts",
        action="store_true",
        help="Whether or not to use redundant experts. When this flag is True: "
        "if the externalization of shared experts is not enabled at this time, "
        "each device will add one redundant expert. If the externalization of shared experts is enabled "
        "and the number of routing experts on each device is the same, "
        "then each device hosting the routing experts will also add one redundant expert.",
    )
    parser.add_argument(
        "--enable-external-shared-experts",
        action="store_true",
        help="Whether or not to implement external shared experts",
    )
    parser.add_argument(
        "--host-external-shared-experts",
        action="store_true",
        help="Whether to have the current device host the external shared experts",
    )
    parser.add_argument(
        "--performance-model",
        nargs="+",
        default=["analytic"],
        help="Performance model type(s). Can specify one or more models. "
        "'analytic': Roofline model (default, no data required). "
        "'profiling': EmpiricalPerformanceModel backed by Profiling CSV database "
        "(exact match, requires --profiling-database). "
        "Example: --performance-model analytic profiling",
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
        "--remote-source",
        type=str,
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="The remote source for the model",
    )

    # Image parameters
    parser.add_argument(
        "--image-batch-size",
        type=check_positive_integer,
        default=None,
        help="Batch size for image processing",
    )
    parser.add_argument(
        "--image-height",
        type=check_positive_integer,
        default=None,
        help="Height of the input images",
    )
    parser.add_argument(
        "--image-width",
        type=check_positive_integer,
        default=None,
        help="Width of the input images",
    )

    args = parser.parse_args()
    log_level = LOG_LEVELS[args.log_level.lower()]
    logging.getLogger().setLevel(log_level)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=log_level)

    if args.graph_log_url:
        config.compilation.debug.graph_log_url = args.graph_log_url

    selected_embedding_tp_mode = args.word_embedding_tp
    args.word_embedding_tp = selected_embedding_tp_mode is not None
    args.word_embedding_tp_mode = (
        selected_embedding_tp_mode or WordEmbeddingTPMode.col.value
    )

    user_input = UserInputConfig.from_args(args)
    model_runner = ModelRunner(user_input)
    metrics = model_runner.run_inference(generate_inputs_func=generate_inputs)
    metrics.print_info()


if __name__ == "__main__":
    main()

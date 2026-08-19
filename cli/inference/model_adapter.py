import argparse
import json

from cli.logo import print_logo
from cli.spec_cli import (
    METAVAR_DIR,
    METAVAR_FILE,
    METAVAR_N,
    METAVAR_NAME,
    SpecArgumentParser,
    add_log_options,
    add_option,
    add_version_option,
    configure_std_logging,
    make_enum_type,
    parse_args as spec_parse_args,
)
from tensor_cast import device_profiles  # noqa: F401
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile

from ..utils import (
    add_model_id_source,
    check_non_negative_integer,
    check_positive_integer,
    require_model_id,
)

SUPPORTED_PERFORMANCE_MODELS = ["analytic", "profiling"]


def _add_output_file_option(parser: argparse.ArgumentParser, help_text: str) -> None:
    add_option(
        parser,
        "-o",
        "--output-file",
        dest="output",
        type=str,
        default=None,
        metavar=METAVAR_FILE,
        help=help_text,
        aliases=("--output",),
    )


def _add_adapter_common_args(parser: argparse.ArgumentParser) -> None:
    add_version_option(parser)
    general_group = parser.add_argument_group("General Options")
    add_model_id_source(
        general_group,
        positional_help="Model source. Prefer a reviewed absolute local model path. Equivalent to --model-id.",
        option_help="Model source. Prefer a reviewed absolute local model path.",
        public_snake_alias=True,
    )
    general_group.add_argument(
        "--device",
        type=str,
        choices=list(DeviceProfile.all_device_profiles.keys()),
        default="TEST_DEVICE",
        metavar=METAVAR_NAME,
        help="Target device profile used for simulation.",
    )
    general_group.add_argument(
        "--num-devices",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Total number of simulated devices.",
    )
    general_group.add_argument(
        "--reserved-memory-gb",
        type=float,
        default=0.0,
        metavar="<FLOAT>",
        help="Reserved device memory in GB.",
    )
    add_log_options(general_group)


def _normalize_adapter_common_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    require_model_id(parser, args)


def _configure_logging(args: argparse.Namespace) -> None:
    configure_std_logging(args, log_format="[%(levelname)s] [%(name)s] %(message)s")


def _write_report(report: dict, output: str | None) -> None:
    content = json.dumps(report, indent=2, sort_keys=True)
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(content + "\n")
    else:
        print(content)


def _add_doctor_runtime_options(parser: argparse.ArgumentParser) -> None:
    runtime_group = parser.add_argument_group("Runtime Options")
    runtime_group.add_argument(
        "--num-queries",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Number of parallel inference queries.",
    )
    runtime_group.add_argument(
        "--query-length",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="New input length in tokens.",
    )
    runtime_group.add_argument(
        "--context-length",
        type=check_non_negative_integer,
        default=0,
        metavar=METAVAR_N,
        help="Existing context length in tokens.",
    )
    runtime_group.add_argument("--decode", action="store_true", help="Enable decode mode.")
    runtime_group.add_argument("--compile", action="store_true", help="Compile the model before the dry-run.")
    runtime_group.add_argument(
        "--compile-allow-graph-break",
        action="store_true",
        help="Allow graph breaks during torch.compile().",
    )
    runtime_group.add_argument(
        "--dump-input-shapes",
        action="store_true",
        help="Group the result table by input shapes.",
    )
    runtime_group.add_argument(
        "--num-hidden-layers-override",
        type=int,
        default=0,
        metavar=METAVAR_N,
        help="Override model layers for a fast adapter dry-run.",
    )
    runtime_group.add_argument(
        "--remote-source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        metavar="{huggingface,modelscope}",
        help="The remote source for the model.",
    )
    add_option(
        runtime_group,
        "--no-repetition",
        dest="disable_repetition",
        action="store_true",
        help="Disable automatic repeated-layer reuse during dry-run.",
        aliases=("--disable-repetition",),
    )
    parse_linear, linear_meta = make_enum_type(QuantizeLinearAction, "--quantize-linear-action")
    parse_attn, attn_meta = make_enum_type(QuantizeAttentionAction, "--quantize-attention-action")
    runtime_group.add_argument(
        "--quantize-linear-action",
        type=parse_linear,
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        metavar=linear_meta,
        help="Quantize linear layers.",
    )
    runtime_group.add_argument(
        "--quantize-attention-action",
        type=parse_attn,
        default=QuantizeAttentionAction.DISABLED,
        metavar=attn_meta,
        help="Quantize attention.",
    )
    runtime_group.add_argument(
        "--image-batch-size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Batch size for image processing.",
    )
    runtime_group.add_argument(
        "--image-height",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Height of the input images.",
    )
    runtime_group.add_argument(
        "--image-width",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Width of the input images.",
    )
    _add_parallelism_options(parser)


def _add_parallelism_options(parser: argparse.ArgumentParser) -> None:
    parallel_group = parser.add_argument_group("Parallelism Options")
    add_option(
        parallel_group,
        "--tp-size",
        dest="tp_size",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Tensor parallel size.",
    )
    add_option(
        parallel_group,
        "--dp-size",
        dest="dp_size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="Data parallel size.",
    )
    add_option(
        parallel_group,
        "--ep-size",
        dest="ep_size",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Expert parallel size.",
    )
    add_option(
        parallel_group,
        "--moe-tp-size",
        dest="moe_tp_size",
        type=check_positive_integer,
        default=None,
        metavar=METAVAR_N,
        help="MoE tensor parallel size.",
    )
    add_option(
        parallel_group,
        "--moe-dp-size",
        dest="moe_dp_size",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="MoE data parallel size.",
    )
    add_option(
        parallel_group,
        "--vision-tp-size",
        dest="vision_tp_size",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Vision tensor parallel size.",
    )


def _make_doctor_user_input(args: argparse.Namespace) -> UserInputConfig:
    args.word_embedding_tp = None
    args.performance_model = getattr(args, "performance_model", None) or ["analytic"]
    return UserInputConfig.from_args(args)


def _run_doctor(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    adaptation_context = None
    raw_insight = None
    hints = None
    if args.from_command_file:
        from tensor_cast.adapter.context import (
            apply_context_to_namespace,
            load_context_from_command_file,
        )

        adaptation_context = load_context_from_command_file(
            args.from_command_file,
            raw_insight_file=args.raw_insight_file,
            hints_file=args.hints_file,
        )
        apply_context_to_namespace(args, adaptation_context)
    _normalize_adapter_common_args(args, parser)
    _configure_logging(args)

    from tensor_cast.adapter.doctor import run_model_doctor

    if args.raw_insight_file:
        from tensor_cast.adapter.insight import load_raw_insight

        raw_insight = load_raw_insight(args.raw_insight_file)
    if args.hints_file:
        from tensor_cast.adapter.hints import load_hints

        hints = load_hints(args.hints_file)
    patch_failure_text = None
    if args.patch_failure_file:
        with open(args.patch_failure_file, "r", encoding="utf-8") as handle:
            patch_failure_text = handle.read()

    report = run_model_doctor(
        _make_doctor_user_input(args),
        adaptation_context=adaptation_context,
        raw_insight=raw_insight,
        hints=hints,
        ignore_existing_profiles=args.ignore_existing_profile,
        patch_failure_text=patch_failure_text,
    ).to_dict()
    if args.profile_draft_output:
        from tensor_cast.adapter.profile_draft import write_builtin_profile_draft

        patch_method_name = None
        patch_discovery = report.get("patch_discovery")
        if patch_discovery and patch_discovery.get("requires_patch"):
            patch_method_name = patch_discovery.get("suggested_patch_method_name")
        path = write_builtin_profile_draft(
            report["candidate_profile"],
            args.profile_draft_output,
            patch_method_name=patch_method_name,
        )
        report["profile_draft_output"] = str(path)
    _write_report(report, args.output)


def _add_verify_case_options(parser: argparse.ArgumentParser) -> None:
    case_group = parser.add_argument_group("Evidence Case Defaults")
    case_group.add_argument(
        "--num-queries",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="Number of parallel inference queries.",
    )
    case_group.add_argument(
        "--query-length",
        type=check_positive_integer,
        default=1,
        metavar=METAVAR_N,
        help="New input length in tokens.",
    )
    case_group.add_argument(
        "--context-length",
        type=check_non_negative_integer,
        default=0,
        metavar=METAVAR_N,
        help="Existing context length in tokens.",
    )
    case_group.add_argument("--decode", action="store_true", help="Enable decode mode.")
    case_group.add_argument(
        "--num-hidden-layers-override",
        type=int,
        default=0,
        metavar=METAVAR_N,
        help="Override model layers for a fast adapter dry-run.",
    )
    add_option(
        case_group,
        "--no-repetition",
        dest="disable_repetition",
        action="store_true",
        help="Disable automatic repeated-layer reuse.",
        aliases=("--disable-repetition",),
    )

    perf_group = parser.add_argument_group("Performance Model Options")
    perf_group.add_argument(
        "--performance-model",
        action="append",
        default=None,
        choices=SUPPORTED_PERFORMANCE_MODELS,
        metavar="{analytic,profiling}",
        help="Performance model type(s). Defaults to analytic unless evidence case overrides it.",
    )
    add_option(
        perf_group,
        "--profiling-database-path",
        dest="profiling_database",
        type=str,
        default=None,
        metavar=METAVAR_DIR,
        help="Profiling database directory.",
        aliases=("--profiling-database",),
    )
    _add_parallelism_options(parser)

    parser.add_argument(
        "--remote-source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        metavar="{huggingface,modelscope}",
        help="The remote source for the model.",
    )


def _make_verify_user_input(args: argparse.Namespace) -> UserInputConfig:
    args.word_embedding_tp = None
    if args.performance_model is None:
        args.performance_model = ["analytic"]
    return UserInputConfig.from_args(args)


def _run_verify(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not (args.model_id or args.model_id_positional):
        from tensor_cast.adapter.evidence import load_evidence

        model_id = load_evidence(args.evidence_file).model.get("model_id")
        if model_id:
            args.model_id = str(model_id)
    _normalize_adapter_common_args(args, parser)
    _configure_logging(args)

    from tensor_cast.adapter.doctor import run_evidence_verification

    report = run_evidence_verification(args.evidence_file, _make_verify_user_input(args)).to_dict()
    if args.st_case_output:
        from tensor_cast.adapter.st_case import (
            build_st_cases_from_report,
            write_st_cases,
        )

        st_cases = build_st_cases_from_report(report)
        report["st_case_outputs"] = [str(path) for path in write_st_cases(st_cases, args.st_case_output)]
    _write_report(report, args.output)
    if not report["passed"]:
        raise SystemExit(1)


def _run_export_evidence(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.doctor_report:
        parser.error("--doctor-report-file is required")
    from tensor_cast.adapter.evidence_export import export_evidence_from_doctor_report

    content = export_evidence_from_doctor_report(args.doctor_report, args.output)
    if not args.output:
        print(content, end="")


def _build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = SpecArgumentParser(
        prog="msmodeling inference model-adapter",
        description="Inspect and verify TensorCast model adapter onboarding artifacts.",
        examples=(
            "# Inspect a local model\n"
            "msmodeling inference model-adapter doctor --model-id Qwen/Qwen3-32B\n"
            "# Verify reviewed evidence\n"
            "msmodeling inference model-adapter verify --evidence-file evidence.yaml"
        ),
    )
    add_version_option(parser)
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=SpecArgumentParser)
    command_parsers = {}

    doctor_parser = subparsers.add_parser(
        "doctor",
        description="Inspect a model adapter profile, patch result, and deterministic suggestions.",
        examples="# Run doctor on a local model\nmsmodeling inference model-adapter doctor --model-id Qwen/Qwen3-32B",
        output_help="JSON report on stdout, or --output-file.",
    )
    _add_adapter_common_args(doctor_parser)
    _add_doctor_runtime_options(doctor_parser)
    doctor_parser.add_argument(
        "--from-command-file",
        type=str,
        default=None,
        metavar=METAVAR_FILE,
        help="Read a TensorCast simulation command and use it as the adaptation context.",
    )
    doctor_parser.add_argument(
        "--raw-insight-file",
        type=str,
        default=None,
        metavar=METAVAR_FILE,
        help="MindStudio Insight raw profiling export that corresponds to the simulation command.",
    )
    doctor_parser.add_argument(
        "--hints-file",
        type=str,
        default=None,
        metavar=METAVAR_FILE,
        help="Optional iterative user hints YAML file.",
    )
    doctor_parser.add_argument(
        "--patch-failure-file",
        type=str,
        default=None,
        metavar=METAVAR_FILE,
        help="Optional stacktrace/failure log used for patch discovery classification.",
    )
    add_option(
        doctor_parser,
        "--ignore-existing-profiles",
        dest="ignore_existing_profile",
        action="append",
        default=[],
        metavar=METAVAR_NAME,
        help="Replay/audit mode only: temporarily ignore an existing registered ModelProfile.",
        aliases=("--ignore-existing-profile",),
    )
    add_option(
        doctor_parser,
        "--profile-draft-output-file",
        dest="profile_draft_output",
        type=str,
        default=None,
        metavar=METAVAR_FILE,
        help="Optional output path for a generated built-in ModelProfile draft module.",
        aliases=("--profile-draft-output",),
    )
    _add_output_file_option(
        doctor_parser,
        "Optional JSON output path. Prints JSON to stdout when omitted.",
    )
    doctor_parser.set_defaults(handler=_run_doctor)
    command_parsers["doctor"] = doctor_parser

    verify_parser = subparsers.add_parser(
        "verify",
        description="Run profiling evidence verification for a TensorCast model adapter.",
        examples="# Verify evidence\nmsmodeling inference model-adapter verify --evidence-file evidence.yaml",
        output_help="JSON report on stdout, or --output-file.",
    )
    _add_adapter_common_args(verify_parser)
    verify_parser.add_argument(
        "--evidence-file",
        required=True,
        metavar=METAVAR_FILE,
        help="YAML file with manually reviewed expected op counts and latency.",
    )
    _add_output_file_option(
        verify_parser,
        "Optional JSON output path. Prints JSON to stdout when omitted.",
    )
    add_option(
        verify_parser,
        "--st-case-output-path",
        dest="st_case_output",
        type=str,
        default=None,
        metavar=METAVAR_FILE,
        help="Optional file or directory for generated ST guardrail case JSON.",
        aliases=("--st-case-output",),
    )
    _add_verify_case_options(verify_parser)
    verify_parser.set_defaults(handler=_run_verify)
    command_parsers["verify"] = verify_parser

    export_evidence_parser = subparsers.add_parser(
        "export-evidence",
        description="Export doctor report evidence_draft as evidence YAML.",
        examples=(
            "# Export evidence YAML\n"
            "msmodeling inference model-adapter export-evidence --doctor-report-file doctor.json"
        ),
        output_help="Evidence YAML on stdout, or --output-file.",
    )
    add_version_option(export_evidence_parser)
    add_option(
        export_evidence_parser,
        "--doctor-report-file",
        dest="doctor_report",
        metavar=METAVAR_FILE,
        help="Doctor JSON report that contains an evidence_draft field.",
        aliases=("--doctor-report",),
    )
    _add_output_file_option(
        export_evidence_parser,
        "Optional evidence YAML output path. Prints YAML to stdout when omitted.",
    )
    export_evidence_parser.set_defaults(handler=_run_export_evidence)
    command_parsers["export-evidence"] = export_evidence_parser

    return parser, command_parsers


def main() -> None:
    # See docs/RFC/rfc_uv_dependency_management_en.md: dependency versions are
    # governed by pyproject.toml/uv.lock, so the old runtime check_dependencies
    # hook is intentionally not called here.
    parser, command_parsers = _build_parser()
    args = spec_parse_args(parser)
    print_logo()
    args.handler(args, command_parsers[args.command])


if __name__ == "__main__":
    main()

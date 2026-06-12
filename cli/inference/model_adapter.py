import argparse
import json
import logging

from cli.logo import print_logo
from tensor_cast import config, device_profiles  # noqa: F401
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile
import tensor_cast.utils as tensor_cast_utils

from ..utils import (
    LOG_LEVELS,
    check_non_negative_integer,
    check_positive_integer,
    check_string_valid,
)

SUPPORTED_PERFORMANCE_MODELS = ["analytic", "profiling"]


def _enum_values(enum_type: type) -> list[str]:
    return [str(item) for item in enum_type]


def _add_adapter_common_args(parser: argparse.ArgumentParser) -> None:
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "model_id_positional",
        nargs="?",
        type=check_string_valid,
        help="Model identifier or local model path. Equivalent to --model-id.",
    )
    general_group.add_argument(
        "--model-id",
        "--model_id",
        dest="model_id",
        type=check_string_valid,
        default=None,
        help="Model identifier or local model path.",
    )
    general_group.add_argument(
        "--device",
        type=str,
        choices=list(DeviceProfile.all_device_profiles.keys()),
        default="TEST_DEVICE",
        help="Target device profile used for simulation.",
    )
    general_group.add_argument(
        "--num-devices",
        type=check_positive_integer,
        default=1,
        help="Total number of simulated devices.",
    )
    general_group.add_argument(
        "--reserved-memory-gb",
        type=float,
        default=0.0,
        help="Reserved device memory in GB.",
    )
    general_group.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="error",
        help="Log verbosity.",
    )


def _normalize_adapter_common_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    model_id = args.model_id or args.model_id_positional
    if not model_id:
        parser.error("model_id is required; pass positional model_id or --model-id.")
    args.model_id = model_id
    delattr(args, "model_id_positional")


def _configure_logging(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level.lower()],
        format="[%(levelname)s] [%(name)s] %(message)s",
    )


def _write_report(report: dict, output: str | None) -> None:
    content = json.dumps(report, indent=2, sort_keys=True)
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(content + "\n")
    else:
        print(content)


def _add_doctor_runtime_options(parser: argparse.ArgumentParser) -> None:
    runtime_group = parser.add_argument_group("Runtime Options")
    runtime_group.add_argument("--num-queries", type=check_positive_integer, default=1)
    runtime_group.add_argument("--query-length", type=check_positive_integer, default=1)
    runtime_group.add_argument("--context-length", type=check_non_negative_integer, default=0)
    runtime_group.add_argument("--decode", action="store_true")
    runtime_group.add_argument("--compile", action="store_true")
    runtime_group.add_argument("--compile-allow-graph-break", action="store_true")
    runtime_group.add_argument("--dump-input-shapes", action="store_true")
    runtime_group.add_argument(
        "--num-hidden-layers-override",
        type=int,
        default=0,
        help="Override model layers for fast adapter dry-run.",
    )
    runtime_group.add_argument(
        "--remote-source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="The remote source for the model.",
    )
    runtime_group.add_argument(
        "--disable-repetition",
        action="store_true",
        help="Disable automatic repeated-layer reuse during dry-run.",
    )
    runtime_group.add_argument(
        "--quantize-linear-action",
        type=str,
        choices=_enum_values(QuantizeLinearAction),
        default=str(QuantizeLinearAction.W8A8_DYNAMIC),
    )
    runtime_group.add_argument(
        "--quantize-attention-action",
        type=str,
        choices=_enum_values(QuantizeAttentionAction),
        default=str(QuantizeAttentionAction.DISABLED),
    )
    runtime_group.add_argument("--image-batch-size", type=check_positive_integer, default=None)
    runtime_group.add_argument("--image-height", type=check_positive_integer, default=None)
    runtime_group.add_argument("--image-width", type=check_positive_integer, default=None)

    parallel_group = parser.add_argument_group("Parallelism Options")
    parallel_group.add_argument("--tp-size", type=check_positive_integer, default=1)
    parallel_group.add_argument("--dp-size", type=check_positive_integer, default=None)
    parallel_group.add_argument("--ep-size", type=check_positive_integer, default=1)
    parallel_group.add_argument("--moe-tp-size", type=check_positive_integer, default=None)
    parallel_group.add_argument("--moe-dp-size", type=check_positive_integer, default=1)


def _make_doctor_user_input(args: argparse.Namespace) -> UserInputConfig:
    args.word_embedding_tp = None
    args.performance_model = getattr(args, "performance_model", None) or ["analytic"]
    if isinstance(args.quantize_linear_action, str):
        args.quantize_linear_action = QuantizeLinearAction(args.quantize_linear_action)
    if isinstance(args.quantize_attention_action, str):
        args.quantize_attention_action = QuantizeAttentionAction(args.quantize_attention_action)
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
    config.compilation.multistream.enable = False

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
    case_group.add_argument("--num-queries", type=check_positive_integer, default=1)
    case_group.add_argument("--query-length", type=check_positive_integer, default=1)
    case_group.add_argument("--context-length", type=check_non_negative_integer, default=0)
    case_group.add_argument("--decode", action="store_true")
    case_group.add_argument("--num-hidden-layers-override", type=int, default=0)
    case_group.add_argument("--disable-repetition", action="store_true")

    perf_group = parser.add_argument_group("Performance Model Options")
    perf_group.add_argument(
        "--performance-model",
        action="append",
        default=None,
        choices=SUPPORTED_PERFORMANCE_MODELS,
        help="Performance model type(s). Defaults to analytic unless evidence case overrides it.",
    )
    perf_group.add_argument("--profiling-database", type=str, default=None)

    parallel_group = parser.add_argument_group("Parallelism Options")
    parallel_group.add_argument("--tp-size", type=check_positive_integer, default=1)
    parallel_group.add_argument("--dp-size", type=check_positive_integer, default=None)
    parallel_group.add_argument("--ep-size", type=check_positive_integer, default=1)
    parallel_group.add_argument("--moe-tp-size", type=check_positive_integer, default=None)
    parallel_group.add_argument("--moe-dp-size", type=check_positive_integer, default=1)

    parser.add_argument(
        "--remote-source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
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
    config.compilation.multistream.enable = False

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


def _run_export_evidence(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> None:
    from tensor_cast.adapter.evidence_export import export_evidence_from_doctor_report

    content = export_evidence_from_doctor_report(args.doctor_report, args.output)
    if not args.output:
        print(content, end="")


def _build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        description="Inspect and verify TensorCast model adapter onboarding artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    command_parsers = {}

    doctor_parser = subparsers.add_parser(
        "doctor",
        description="Inspect a model adapter profile, patch result, and deterministic suggestions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_adapter_common_args(doctor_parser)
    _add_doctor_runtime_options(doctor_parser)
    doctor_parser.add_argument(
        "--from-command-file",
        type=str,
        default=None,
        help="Read a TensorCast simulation command and use it as the adaptation context.",
    )
    doctor_parser.add_argument(
        "--raw-insight-file",
        type=str,
        default=None,
        help="MindStudio Insight raw profiling export that corresponds to the simulation command.",
    )
    doctor_parser.add_argument(
        "--hints-file",
        type=str,
        default=None,
        help="Optional iterative user hints YAML file.",
    )
    doctor_parser.add_argument(
        "--patch-failure-file",
        type=str,
        default=None,
        help="Optional stacktrace/failure log used for patch discovery classification.",
    )
    doctor_parser.add_argument(
        "--ignore-existing-profile",
        action="append",
        default=[],
        help="Replay/audit mode only: temporarily ignore an existing registered ModelProfile.",
    )
    doctor_parser.add_argument(
        "--profile-draft-output",
        type=str,
        default=None,
        help="Optional output path for a generated built-in ModelProfile draft module.",
    )
    doctor_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path. Prints JSON to stdout when omitted.",
    )
    doctor_parser.set_defaults(handler=_run_doctor)
    command_parsers["doctor"] = doctor_parser

    verify_parser = subparsers.add_parser(
        "verify",
        description="Run profiling evidence verification for a TensorCast model adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_adapter_common_args(verify_parser)
    verify_parser.add_argument(
        "--evidence-file",
        required=True,
        help="YAML file with manually reviewed expected op counts and latency.",
    )
    verify_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path. Prints JSON to stdout when omitted.",
    )
    verify_parser.add_argument(
        "--st-case-output",
        type=str,
        default=None,
        help="Optional file or directory for generated ST guardrail case JSON.",
    )
    _add_verify_case_options(verify_parser)
    verify_parser.set_defaults(handler=_run_verify)
    command_parsers["verify"] = verify_parser

    export_evidence_parser = subparsers.add_parser(
        "export-evidence",
        description="Export doctor report evidence_draft as evidence YAML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    export_evidence_parser.add_argument(
        "--doctor-report",
        required=True,
        help="Doctor JSON report that contains an evidence_draft field.",
    )
    export_evidence_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional evidence YAML output path. Prints YAML to stdout when omitted.",
    )
    export_evidence_parser.set_defaults(handler=_run_export_evidence)
    command_parsers["export-evidence"] = export_evidence_parser

    return parser, command_parsers


def main() -> None:
    getattr(tensor_cast_utils, "check_dependencies")()
    parser, command_parsers = _build_parser()
    args = parser.parse_args()
    print_logo()
    args.handler(args, command_parsers[args.command])


if __name__ == "__main__":
    main()

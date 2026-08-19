"""Unified console entry for the msmodeling package."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from cli.spec_cli import SpecArgumentParser, add_version_option

if TYPE_CHECKING:
    from collections.abc import Callable


def _dispatch(main_callable: Callable[[], object], remaining: list[str]) -> int:
    sys.argv = [sys.argv[0], *remaining]
    result = main_callable()
    if isinstance(result, int):
        return result
    return 0


def _handle_inference_command(
    args: argparse.Namespace,
    remaining: list[str],
    inference_parser: argparse.ArgumentParser,
) -> int:
    if args.inference_command == "text-generate":
        from cli.inference.text_generate import main as text_generate_main

        return _dispatch(text_generate_main, remaining)
    if args.inference_command == "throughput-optimizer":
        from cli.inference.throughput_optimizer import main as throughput_optimizer_main

        return _dispatch(throughput_optimizer_main, remaining)
    if args.inference_command == "model-adapter":
        from cli.inference.model_adapter import main as model_adapter_main

        return _dispatch(model_adapter_main, remaining)
    if args.inference_command == "video-generate":
        from cli.inference.video_generate import main as video_generate_main

        return _dispatch(video_generate_main, remaining)
    if args.inference_command == "image-generate":
        from cli.inference.image_generate import main as image_generate_main

        return _dispatch(image_generate_main, remaining)
    if args.inference_command is None:
        inference_parser.print_help()
        return 0
    print(f"Unknown inference command: {args.inference_command}", file=sys.stderr)
    inference_parser.print_help()
    return 1


def main() -> int:
    parser = SpecArgumentParser(
        prog="msmodeling",
        description="MindStudio Modeling CLI",
        examples=(
            "# Simulated LLM inference\n"
            "msmodeling inference text-generate MODEL --num-queries 1 --query-length 128 --device DEV\n"
            "# Throughput search\n"
            "msmodeling inference throughput-optimizer MODEL --device DEV --num-devices 8 "
            "--input-length 1024 --output-length 512\n"
            "# Model adapter doctor\n"
            "msmodeling inference model-adapter doctor --model-id MODEL\n"
            "# Video generate\n"
            "msmodeling inference video-generate MODEL --batch-size 1 --seq-len 512\n"
            "# Image generate\n"
            "msmodeling inference image-generate MODEL --batch-size 1\n"
            "# OptiX service parameter optimizer\n"
            "msmodeling optix -e vllm -b ais_bench"
        ),
    )
    add_version_option(parser)
    subparsers = parser.add_subparsers(dest="command", parser_class=SpecArgumentParser)

    subparsers.add_parser("optix", help="Service parameter optimizer", add_help=False)

    inference_parser = subparsers.add_parser(
        "inference",
        help="Inference simulation commands",
        description="Inference simulation commands.",
        examples=(
            "# Simulated LLM inference\n"
            "msmodeling inference text-generate MODEL --num-queries 1 --query-length 128 --device DEV\n"
            "# Throughput search\n"
            "msmodeling inference throughput-optimizer MODEL --device DEV --num-devices 8 "
            "--input-length 1024 --output-length 512\n"
            "# Image generate\n"
            "msmodeling inference image-generate MODEL --batch-size 1"
        ),
    )
    add_version_option(inference_parser)
    inference_sub = inference_parser.add_subparsers(dest="inference_command", parser_class=SpecArgumentParser)
    inference_sub.add_parser("text-generate", help="Run a simulated LLM inference pass")
    inference_sub.add_parser("throughput-optimizer", help="Search serving throughput strategies")
    inference_sub.add_parser("model-adapter", help="Model adaptation doctor, verify, and export-evidence")
    inference_sub.add_parser("video-generate", help="Run a simulated video generation pass")
    inference_sub.add_parser(
        "image-generate",
        help="Run a simulated image generation pass",
        # Defer --help to cli.inference.image_generate's own parser so the full
        # image-specific flag list is rendered instead of just this help line.
        add_help=False,
    )

    args, remaining = parser.parse_known_args()

    if args.command == "optix":
        from optix.optimizer.optimizer import main as optix_main

        return _dispatch(optix_main, remaining)

    if args.command == "inference":
        return _handle_inference_command(args, remaining, inference_parser)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

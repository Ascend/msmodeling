# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Capture one Runtime profile and optionally compare or report it."""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import FindingStatus
from tools.model_diagnostics.rendering import (
    ComparisonHtmlRenderer,
    ConsoleResultRenderer,
    RuntimeHtmlRenderer,
    write_html_report,
)
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
from tools.model_diagnostics.specification import load_diagnostics_run_profile

_DEFAULT_REPORT_PATH = object()
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.model_diagnostics",
        description="Run one profile capture and optionally produce Runtime or Theory-to-Runtime reports.",
    )
    parser.add_argument("profile", type=Path, help="Path to the diagnostics run-profile YAML")
    parser.add_argument("--runtime-report", nargs="?", const=_DEFAULT_REPORT_PATH, type=Path, help="Write a Runtime HTML report (optional path)")
    parser.add_argument("--theory-compare", action="store_true", help="Compare Theory with the captured Runtime artifact")
    parser.add_argument("--comparison-report", nargs="?", const=_DEFAULT_REPORT_PATH, type=Path, help="Write a Theory-to-Runtime HTML report (optional path)")
    parser.add_argument("--fail-only", action="store_true", help="Show only non-PASS comparison findings")
    parser.add_argument("--show-all", action="store_true", help="Show full capture logs and comparison details; overrides --fail-only")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.comparison_report is not None and not args.theory_compare:
        parser.error("--comparison-report requires --theory-compare")
    if not args.profile.is_file():
        print(f"error: run profile not found: {args.profile}", file=sys.stderr)
        return 2
    try:
        runtime_path, comparison_path = _report_paths(args)
        profile, artifact = _capture(args.profile, show_all=args.show_all)
        if runtime_path is not None:
            write_html_report(runtime_path, RuntimeHtmlRenderer().render(artifact))
            print(f"Runtime report: {runtime_path}")
        if not args.theory_compare:
            print(f"Runtime capture completed: {len(artifact.operator_calls)} operator calls")
            print(_artifact_context_line(artifact))
            if runtime_path is None:
                print("No report or comparison requested.")
            return 0

        result = create_model_diagnostics_application().run_profile_against_artifact(profile, artifact)
        sys.stdout.write(ConsoleResultRenderer(show_all=args.show_all, fail_only=args.fail_only).render(result))
        if comparison_path is not None:
            write_html_report(comparison_path, ComparisonHtmlRenderer().render(result))
            print(f"Comparison report: {comparison_path}")
        return 0 if result.summary.overall_status is FindingStatus.PASS else 1
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _capture(profile_path: Path, *, show_all: bool):
    if show_all:
        profile = load_diagnostics_run_profile(profile_path)
        return profile, capture_artifact_for_profile(profile)
    captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
    previous_logging_disable = logging.root.manager.disable
    try:
        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr), warnings.catch_warnings():
            logging.disable(logging.WARNING)
            warnings.simplefilter("ignore")
            profile = load_diagnostics_run_profile(profile_path)
            artifact = capture_artifact_for_profile(profile)
    except BaseException:
        _replay_captured_runtime_output(captured_stdout, captured_stderr)
        raise
    finally:
        logging.disable(previous_logging_disable)
    _print_user_actionable_warnings(captured_stdout, captured_stderr)
    return profile, artifact


def _report_paths(args) -> tuple[Path | None, Path | None]:
    requested = (args.runtime_report is not None, args.comparison_report is not None)
    default_dir = None
    if _DEFAULT_REPORT_PATH in (args.runtime_report, args.comparison_report):
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.profile.stem)
        default_dir = _REPO_ROOT / "outputs" / "model_diagnostics" / f"{stem}-{datetime.now():%Y%m%d-%H%M%S}"
    runtime = default_dir / "runtime.html" if args.runtime_report is _DEFAULT_REPORT_PATH else args.runtime_report
    comparison = default_dir / "theory_runtime.html" if args.comparison_report is _DEFAULT_REPORT_PATH else args.comparison_report
    paths = [path for path, enabled in zip((runtime, comparison), requested, strict=True) if enabled]
    if len(paths) == 2 and paths[0].resolve() == paths[1].resolve():
        raise ValueError("Runtime and comparison reports must use different paths")
    for path in paths:
        if path.suffix.lower() != ".html":
            warnings.warn(f"HTML report path does not end in .html: {path}", UserWarning, stacklevel=2)
    return runtime, comparison


def _artifact_context_line(artifact) -> str:
    context = artifact.run_context
    phase = context.phase.value if context.phase is not None else "?"
    ctx = 0 if context.context_length is None else context.context_length
    return f"{context.model_name} | {phase} | batch={context.batch_size} query={context.query_length} context={ctx} | TP={context.parallel.tensor_parallel_size}"


def _replay_captured_runtime_output(stdout: io.StringIO, stderr: io.StringIO) -> None:
    sys.stderr.write(stdout.getvalue())
    sys.stderr.write(stderr.getvalue())


def _print_user_actionable_warnings(stdout: io.StringIO, stderr: io.StringIO) -> None:
    for line in (*stdout.getvalue().splitlines(), *stderr.getvalue().splitlines()):
        if line.startswith("[msmodeling security]"):
            print(line, file=sys.stderr)
        elif "DiagnosticsSelectionWarning:" in line:
            message = line.split("DiagnosticsSelectionWarning:", maxsplit=1)[1].strip()
            print(f"[model_diagnostics warning] {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

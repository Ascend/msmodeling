"""Regression tests for the unified msmodeling CLI entry (cli.main)."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import cli
import pytest

from cli.main import _dispatch, _handle_inference_command, main
from tests.helpers.cli_runner import run_cli_main


def test_cli_package_docstring() -> None:
    assert cli.__doc__


def test_dispatch_returns_integer_exit_code() -> None:
    assert _dispatch(lambda: 7, ["--flag"]) == 7


def test_dispatch_defaults_non_integer_to_zero() -> None:
    assert _dispatch(lambda: None, []) == 0


def test_handle_inference_command_prints_help_when_subcommand_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inference_parser = argparse.ArgumentParser(prog="msmodeling inference")
    inference_parser.add_subparsers(dest="inference_command")
    args = argparse.Namespace(inference_command=None)

    assert _handle_inference_command(args, [], inference_parser) == 0
    assert "usage:" in capsys.readouterr().out


def test_handle_inference_command_rejects_unknown_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inference_parser = argparse.ArgumentParser(prog="msmodeling inference")
    inference_parser.add_subparsers(dest="inference_command")
    args = argparse.Namespace(inference_command="not-a-command")

    assert _handle_inference_command(args, [], inference_parser) == 1
    captured = capsys.readouterr()
    assert "Unknown inference command" in captured.err
    assert "usage:" in captured.out


@pytest.mark.parametrize(
    ("inference_command", "target", "remaining"),
    [
        ("text-generate", "cli.inference.text_generate.main", ["MODEL", "--help"]),
        ("throughput-optimizer", "cli.inference.throughput_optimizer.main", ["MODEL", "--help"]),
        ("model-adapter", "cli.inference.model_adapter.main", ["--help"]),
        ("video-generate", "cli.inference.video_generate.main", ["MODEL", "--help"]),
    ],
)
def test_handle_inference_command_dispatches_registered_subcommands(
    inference_command: str,
    target: str,
    remaining: list[str],
) -> None:
    inference_parser = argparse.ArgumentParser(prog="msmodeling inference")
    inference_parser.add_subparsers(dest="inference_command")
    args = argparse.Namespace(inference_command=inference_command)

    with patch(target, return_value=0) as sub_main:
        assert _handle_inference_command(args, remaining, inference_parser) == 0
        sub_main.assert_called_once()


def test_main_prints_top_level_help_without_subcommand() -> None:
    result = run_cli_main(main, [], prog="msmodeling")
    assert result.returncode == 0
    assert "MindStudio Modeling CLI" in result.stdout
    assert "msmodeling inference" in result.stdout


def test_main_dispatches_optix_subcommand() -> None:
    with patch("optix.optimizer.optimizer.main", return_value=0) as optix_main:
        result = run_cli_main(main, ["optix", "--help"], prog="msmodeling")

    assert result.returncode == 0
    optix_main.assert_called_once()


def test_main_dispatches_inference_subcommand() -> None:
    with patch("cli.inference.text_generate.main", return_value=0) as text_generate_main:
        result = run_cli_main(
            main,
            ["inference", "text-generate", "Qwen/Qwen3-32B"],
            prog="msmodeling",
        )

    assert result.returncode == 0
    text_generate_main.assert_called_once()

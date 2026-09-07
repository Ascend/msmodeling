"""Tests for CLI validators integration (RFC §3.7 / gap #2.1).

parse_module_args() calls spec.validators after parse_args(). A failing
validator exits via parser.error() (stderr message + exit code 2, matching
argparse's built-in error style).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from cli.registry.argparse_adapter import build_argparser, parse_module_args
from cli.registry.modules import get_spec
from tests.helpers.cli_runner import run_cli_main


# ── Validator fires on invalid args ──────────────────────────────────


class TestValidatorIntegration:
    """Each sim module's cross-field validators are invoked by parse_module_args."""

    def _parse(self, module_id: str, argv: list[str]):
        """Run parse_module_args with given argv, capture SystemExit."""
        spec = get_spec(module_id)
        parser = build_argparser(spec)
        with patch.object(sys, "argv", ["prog", *argv]):
            # parse_module_args calls parser.error() → SystemExit(2) on failure
            return parse_module_args(spec, parser)

    def test_text_generate_tp_product_mismatch(self, capsys):
        """tg: tp_size × dp_size × pp_size != num_devices → validator error."""
        with pytest.raises(SystemExit) as exc_info:
            self._parse(
                "text_generate",
                [
                    "--model-id",
                    "Qwen/Qwen3-32B",
                    "--tp-size",
                    "3",
                    "--num-devices",
                    "8",
                    "--num-queries",
                    "1",
                    "--query-length",
                    "128",
                ],
            )
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "tp_size" in captured.err
        assert "num_devices" in captured.err

    def test_text_generate_valid_args_pass(self):
        """tg: valid args → parse_module_args returns Namespace (no exit)."""
        ns = self._parse(
            "text_generate",
            [
                "--model-id",
                "Qwen/Qwen3-32B",
                "--tp-size",
                "8",
                "--num-devices",
                "8",
                "--num-queries",
                "1",
                "--query-length",
                "128",
            ],
        )
        assert ns.tp_size == 8
        assert ns.num_devices == 8

    def test_throughput_optimizer_parallel_combo(self, capsys):
        """to: invalid parallel combo → validator error."""
        # throughput_optimizer's validators include a parallel combo check.
        # We trigger it by setting inconsistent sizes.
        with pytest.raises(SystemExit) as exc_info:
            self._parse(
                "throughput_optimizer",
                [
                    "--model-id",
                    "Qwen/Qwen3-32B",
                    "--tp-sizes",
                    "3",
                    "--num-devices",
                    "8",
                    "--input-length",
                    "128",
                    "--output-length",
                    "128",
                    "--max-batched-tokens",
                    "256",
                ],
            )
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        # The error should mention parallel-related fields
        assert len(captured.err) > 0

    def test_video_generate_valid_args_pass(self):
        """vi: valid args → parse_module_args returns Namespace."""
        ns = self._parse(
            "video_generate",
            [
                "--model-id",
                "test-model",
                "--batch-size",
                "1",
                "--seq-len",
                "16",
                "--frame-num",
                "1",
                "--sample-step",
                "1",
            ],
        )
        assert ns.batch_size == 1

    def test_image_generate_no_validators(self):
        """ig: no validators defined → parse_module_args returns Namespace."""
        # image_generate has validators=[] (empty), so no cross-field check fires.
        ns = self._parse(
            "image_generate",
            [
                "--model-id",
                "test-model",
                "--batch-size",
                "1",
                "--output-image-size",
                "256",
                "256",
                "--text-seq-len",
                "16",
            ],
        )
        assert ns.model_id == "test-model"


# ── Validator contract ───────────────────────────────────────────────


class TestValidatorContract:
    """Validators follow the contract: fn(dict) → None(pass) | str(fail)."""

    def test_all_module_validators_callable(self):
        """Every module's spec.validators entries are callable."""
        for module_id in ["text_generate", "video_generate", "throughput_optimizer", "image_generate"]:
            spec = get_spec(module_id)
            for v in spec.validators:
                assert callable(v.fn), f"{module_id}.{v.name}.fn is not callable"
                assert isinstance(v.name, str), f"{module_id} validator name is not str"

    def test_validator_fn_accepts_kebab_keys(self):
        """Validator fn accepts a dict keyed by kebab-case Param.name.

        wants_provided=True validators additionally receive the explicitly-
        provided field set (empty here — nothing explicitly passed).
        """
        spec = get_spec("text_generate")
        # Build a valid params dict (all None should be accepted by validators
        # that check "if not value: return None")
        params = {p.name: None for p in spec.fields}
        for v in spec.validators:
            result = v.fn(params, set()) if v.wants_provided else v.fn(params)
            assert result is None or isinstance(result, str), (
                f"{v.name}.fn() returned {type(result)}, expected None|str"
            )


# ── Help snapshots unaffected ────────────────────────────────────────


class TestHelpUnaffected:
    """Validator integration doesn't change --help output."""

    @pytest.mark.parametrize("module_id", ["text_generate", "throughput_optimizer", "video_generate"])
    def test_help_still_works(self, module_id):
        """--help exits 0 and includes expected sections."""
        from cli.inference import text_generate, video_generate, throughput_optimizer

        main_map = {
            "text_generate": text_generate.main,
            "video_generate": video_generate.main,
            "throughput_optimizer": throughput_optimizer.main,
        }
        result = run_cli_main(main_map[module_id], ["--help"], prog="msmodeling")
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

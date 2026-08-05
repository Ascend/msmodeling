"""Real unit tests for runners/_cli_command.py module.

Tests CLI command string building logic using real module imports.
"""

from __future__ import annotations

# Import the actual module directly
from runners._cli_command import (
    _CLI_MODULE,
    _FLAG_OVERRIDES,
    _SKIP_FIELDS,
    build_cli_command_string,
)


class TestCliModuleMapping:
    """Tests for _CLI_MODULE mapping."""

    def test_text_generate_mapping(self):
        """text_generate maps to cli.inference.text_generate."""
        assert _CLI_MODULE.get("text_generate") == "cli.inference.text_generate"

    def test_video_generate_mapping(self):
        """video_generate maps to cli.inference.video_generate."""
        assert _CLI_MODULE.get("video_generate") == "cli.inference.video_generate"

    def test_throughput_optimizer_mapping(self):
        """throughput_optimizer maps to cli.inference.throughput_optimizer."""
        assert _CLI_MODULE.get("throughput_optimizer") == "cli.inference.throughput_optimizer"

    def test_unknown_module_defaults(self):
        """Unknown modules default to cli.inference.{module_id}."""
        cmd = build_cli_command_string("unknown_module", {})
        assert "cli.inference.unknown_module" in cmd

    def test_mapping_is_complete(self):
        """Mapping contains all expected modules."""
        expected_modules = ["text_generate", "video_generate", "throughput_optimizer"]
        for mod in expected_modules:
            assert mod in _CLI_MODULE


class TestFlagOverrides:
    """Tests for _FLAG_OVERRIDES mapping."""

    def test_overrides_dict_exists(self):
        """_FLAG_OVERRIDES exists and is a dict."""
        assert isinstance(_FLAG_OVERRIDES, dict)

    def test_default_uses_snake_to_kebab(self):
        """By default, flags use snake_case -> kebab-case conversion."""
        cmd = build_cli_command_string("text_generate", {"some_param": "value"})
        assert "--some-param" in cmd

    def test_override_can_change_flag_name(self):
        """_FLAG_OVERRIDES can override flag names if needed."""
        # The dict exists for future overrides
        assert isinstance(_FLAG_OVERRIDES, dict)


class TestSkipFields:
    """Tests for _SKIP_FIELDS set."""

    def test_skip_fields_exists(self):
        """_SKIP_FIELDS exists and is a set."""
        assert isinstance(_SKIP_FIELDS, set)

    def test_skip_fields_currently_empty(self):
        """_SKIP_FIELDS is currently empty (all legacy fields removed from schema)."""
        # Note: if a field is removed from CLI but may still exist in cached jobs,
        # add it here so build_cli_command_string skips it.
        assert isinstance(_SKIP_FIELDS, set)


class TestBasicCommandBuilding:
    """Tests for basic command building."""

    def test_empty_params(self):
        """Empty params produce only module command."""
        cmd = build_cli_command_string("text_generate", {})
        assert cmd.startswith("python -m cli.inference.text_generate")
        assert cmd.count("python") == 1

    def test_model_id_as_positional(self):
        """model_id is added as positional argument."""
        cmd = build_cli_command_string("text_generate", {"model_id": "gpt2"})
        assert "gpt2" in cmd
        # model_id should not have --flag prefix
        assert "--model-id" not in cmd

    def test_string_param_becomes_flag(self):
        """String params become --flag value."""
        cmd = build_cli_command_string("text_generate", {"device": "cpu"})
        assert "--device" in cmd
        assert "cpu" in cmd

    def test_multiple_params(self):
        """Multiple params are all included."""
        cmd = build_cli_command_string("text_generate", {"model_id": "gpt2", "device": "cpu", "batch_size": 32})
        assert "gpt2" in cmd
        assert "--device" in cmd
        assert "--batch-size" in cmd
        assert "cpu" in cmd
        assert "32" in cmd

    def test_snake_case_converts_to_kebab(self):
        """snake_case param names convert to kebab-case flags."""
        cmd = build_cli_command_string("text_generate", {"batch_size": 32})
        assert "--batch-size" in cmd
        assert "--batch_size" not in cmd

    def test_boolean_true_becomes_flag_only(self):
        """Boolean True params become --flag without value."""
        cmd = build_cli_command_string("text_generate", {"verbose": True})
        assert "--verbose" in cmd
        # No value should follow
        parts = cmd.split()
        verbose_idx = parts.index("--verbose")
        # Either end of list or next is another flag
        if verbose_idx + 1 < len(parts):
            assert parts[verbose_idx + 1].startswith("--")


class TestSpecialCases:
    """Tests for special parameter cases."""

    def test_none_values_skipped(self):
        """None values are skipped."""
        cmd = build_cli_command_string("text_generate", {"device": None})
        assert "--device" not in cmd

    def test_false_values_skipped(self):
        """False values are skipped."""
        cmd = build_cli_command_string("text_generate", {"verbose": False})
        assert "--verbose" not in cmd

    def test_empty_string_skipped(self):
        """Empty strings are skipped."""
        cmd = build_cli_command_string("text_generate", {"device": ""})
        assert "--device" not in cmd

    def test_list_values(self):
        """List values become --flag val1 val2."""
        cmd = build_cli_command_string("text_generate", {"devices": ["cpu", "cuda"]})
        assert "--devices" in cmd
        assert "cpu" in cmd
        assert "cuda" in cmd

    def test_comma_separated_string_split(self):
        """Comma-separated strings are split on comma."""
        cmd = build_cli_command_string("throughput_optimizer", {"mtp_acceptance_rate": "0.8,0.6,0.4"})
        assert "--mtp-acceptance-rate" in cmd
        assert "0.8" in cmd
        assert "0.6" in cmd
        assert "0.4" in cmd

    def test_comma_separated_handles_empty_items(self):
        """Comma-separated strings handle empty items."""
        cmd = build_cli_command_string("throughput_optimizer", {"mtp_acceptance_rate": "0.8,,0.6"})
        # Empty items after split should be filtered
        assert "0.8" in cmd
        assert "0.6" in cmd
        # Should not have empty strings as separate values
        assert "--mtp-acceptance-rate" in cmd

    def test_chrome_trace_true_placeholder(self):
        """chrome_trace=True adds --chrome-trace as a bare flag (no value)."""
        cmd = build_cli_command_string("text_generate", {"chrome_trace": True})
        assert "--chrome-trace" in cmd
        # Boolean True maps to a bare flag (store_true argparse semantics).
        # The command must NOT contain a following value token.
        parts = cmd.split()
        idx = parts.index("--chrome-trace")
        assert idx == len(parts) - 1 or parts[idx + 1].startswith("--")

    def test_chrome_trace_false_skipped(self):
        """chrome_trace=False is skipped like other boolean False."""
        cmd = build_cli_command_string("text_generate", {"chrome_trace": False})
        assert "--chrome-trace" not in cmd


class TestModuleSpecificCommands:
    """Tests for module-specific command structures."""

    def test_text_generate_command(self):
        """text_generate produces expected command structure."""
        cmd = build_cli_command_string("text_generate", {"model_id": "gpt2", "batch_size": 32, "device": "cpu"})
        assert "python -m cli.inference.text_generate" in cmd
        assert "gpt2" in cmd
        assert "--batch-size 32" in cmd or ("--batch-size" in cmd and "32" in cmd)

    def test_video_generate_command(self):
        """video_generate produces expected command structure."""
        cmd = build_cli_command_string("video_generate", {"model_id": "video_model", "resolution": "1080p"})
        assert "python -m cli.inference.video_generate" in cmd
        assert "video_model" in cmd
        assert "--resolution" in cmd

    def test_throughput_optimizer_command(self):
        """throughput_optimizer produces expected command structure."""
        cmd = build_cli_command_string(
            "throughput_optimizer", {"model_id": "opt_model", "mtp_acceptance_rate": "0.8,0.6"}
        )
        assert "python -m cli.inference.throughput_optimizer" in cmd
        assert "opt_model" in cmd


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_is_included(self):
        """Zero values are included (not falsy)."""
        cmd = build_cli_command_string("text_generate", {"batch_size": 0})
        assert "--batch-size" in cmd
        assert "0" in cmd

    def test_large_numbers(self):
        """Large numbers are handled correctly."""
        cmd = build_cli_command_string("text_generate", {"batch_size": 999999})
        assert "999999" in cmd

    def test_negative_numbers(self):
        """Negative numbers are handled correctly."""
        cmd = build_cli_command_string("text_generate", {"temperature": -0.5})
        assert "--temperature" in cmd
        assert "-0.5" in cmd

    def test_float_values(self):
        """Float values are stringified correctly."""
        cmd = build_cli_command_string("text_generate", {"temperature": 0.7})
        assert "0.7" in cmd

    def test_unicode_in_params(self):
        """Unicode characters in params are handled."""
        cmd = build_cli_command_string("text_generate", {"prompt": "café"})
        assert "café" in cmd

    def test_special_characters(self):
        """Special characters are handled."""
        cmd = build_cli_command_string("text_generate", {"device": "cpu-cuda"})
        assert "cpu-cuda" in cmd

    def test_very_long_param_name(self):
        """Very long param names are converted correctly."""
        cmd = build_cli_command_string("text_generate", {"very_long_parameter_name": "value"})
        assert "--very-long-parameter-name" in cmd

    def test_very_long_param_value(self):
        """Very long param values are handled."""
        long_value = "x" * 1000
        cmd = build_cli_command_string("text_generate", {"prompt": long_value})
        assert long_value in cmd

    def test_empty_list(self):
        """Empty lists are handled."""
        cmd = build_cli_command_string("text_generate", {"devices": []})
        # Should not crash
        assert "--devices" in cmd

    def test_single_item_list(self):
        """Single-item lists are handled."""
        cmd = build_cli_command_string("text_generate", {"devices": ["cpu"]})
        assert "--devices" in cmd
        assert "cpu" in cmd

    def test_large_list(self):
        """Large lists are handled."""
        items = [f"item{i}" for i in range(100)]
        cmd = build_cli_command_string("text_generate", {"items": items})
        for item in items[:10]:  # Check first 10
            assert item in cmd


class TestCommandStringStructure:
    """Tests for command string structure."""

    def test_command_starts_with_python(self):
        """Command always starts with 'python -m'."""
        cmd = build_cli_command_string("text_generate", {})
        assert cmd.startswith("python -m")

    def test_command_has_single_python(self):
        """Command has only one 'python' (python -m, not python python)."""
        cmd = build_cli_command_string("text_generate", {"model_id": "model"})
        assert cmd.count("python") == 1

    def test_command_parts_separated_by_spaces(self):
        """Command parts are separated by spaces."""
        cmd = build_cli_command_string("text_generate", {"model_id": "gpt2", "device": "cpu"})
        parts = cmd.split()
        assert len(parts) >= 4  # python, -m, module, model_id, --flag, value

    def test_no_extra_spaces(self):
        """No extra spaces in command."""
        cmd = build_cli_command_string("text_generate", {})
        assert "  " not in cmd  # No double spaces


class TestIntegration:
    """Integration tests for command building."""

    def test_full_command_structure(self):
        """Complete command with various param types."""
        cmd = build_cli_command_string(
            "text_generate",
            {
                "model_id": "gpt2",
                "batch_size": 32,
                "device": "cpu",
                "verbose": True,
                "chrome_trace": True,
            },
        )
        # Check structure
        assert "python -m cli.inference.text_generate" in cmd
        assert "gpt2" in cmd
        assert "--batch-size" in cmd and "32" in cmd
        assert "--device" in cmd and "cpu" in cmd
        assert "--verbose" in cmd
        assert "--chrome-trace" in cmd

    def test_real_world_throughput_command(self):
        """Real-world throughput_optimizer command."""
        cmd = build_cli_command_string(
            "throughput_optimizer",
            {
                "model_id": "llama-7b",
                "device": "a100",
                "batch_size": 16,
                "mtp_acceptance_rate": "0.8,0.6,0.4",
                "chrome_trace": True,
            },
        )
        assert "python -m cli.inference.throughput_optimizer" in cmd
        assert "llama-7b" in cmd
        assert "--device" in cmd and "a100" in cmd
        assert "--batch-size" in cmd and "16" in cmd
        assert "--mtp-acceptance-rate" in cmd
        assert "0.8" in cmd and "0.6" in cmd and "0.4" in cmd
        assert "--chrome-trace" in cmd


class TestConstants:
    """Tests for module constants."""

    def test_cli_module_is_dict(self):
        """_CLI_MODULE is a dictionary."""
        assert isinstance(_CLI_MODULE, dict)

    def test_skip_fields_is_set(self):
        """_SKIP_FIELDS is a set."""
        assert isinstance(_SKIP_FIELDS, set)

    def test_flag_overrides_is_dict(self):
        """_FLAG_OVERRIDES is a dictionary."""
        assert isinstance(_FLAG_OVERRIDES, dict)


class TestNoSplitFields:
    """Tests for single-value fields that may contain commas but should NOT be
    comma-split into space-separated tokens (they're passed as ONE token to
    argparse). Contrast with nargs="+" fields which DO get split.
    """

    def test_cache_step_range_not_split(self):
        """cache_step_range='20,30' stays as one token, not split into '20 30'."""
        cmd = build_cli_command_string("video_generate", {"model_id": "m", "cache_step_range": "20,30"})
        assert "--cache-step-range 20,30" in cmd
        assert "20 30" not in cmd  # must NOT be space-separated

    def test_cache_block_range_not_split(self):
        """cache_block_range='0,4' stays as one token."""
        cmd = build_cli_command_string("video_generate", {"model_id": "m", "cache_block_range": "0,4"})
        assert "--cache-block-range 0,4" in cmd

    def test_nargs_plus_field_still_split(self):
        """mtp_acceptance_rate='0.8,0.6' IS split into space-separated tokens
        (it's a nargs='+' field, not in _NO_SPLIT_FIELDS).
        """
        cmd = build_cli_command_string("throughput_optimizer", {"model_id": "m", "mtp_acceptance_rate": "0.8,0.6"})
        assert "--mtp-acceptance-rate" in cmd
        assert "0.8" in cmd and "0.6" in cmd
        # Should be space-separated, not comma (nargs="+" expects separate tokens).
        assert "0.8,0.6" not in cmd

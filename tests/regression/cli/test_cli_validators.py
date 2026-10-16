# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for CLI validators integration (RFC §3.7 / gap #2.1).

parse_module_args() calls spec.validators after parse_args(). A failing
validator exits via parser.error() (stderr message + exit code 2, matching
argparse's built-in error style).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from cli.registry import validators as V
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

    @pytest.mark.parametrize(
        ("parallel_args", "num_devices", "ulysses_size", "cfg_parallel"),
        [
            ([], 1, 1, False),
            (["--num-devices", "2", "--ulysses-size", "2"], 2, 2, False),
            (
                ["--num-devices", "4", "--ulysses-size", "2", "--use-cfg", "--cfg-parallel"],
                4,
                2,
                True,
            ),
        ],
    )
    def test_video_generate_valid_parallel_contract(self, parallel_args, num_devices, ulysses_size, cfg_parallel):
        """vi: non-CFG uses N=U; CFG-parallel uses N=2U and requires CFG."""
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
                *parallel_args,
            ],
        )
        assert ns.num_devices == num_devices
        assert ns.ulysses_size == ulysses_size
        assert ns.cfg_parallel is cfg_parallel

    @pytest.mark.parametrize(
        ("parallel_args", "error"),
        [
            (
                ["--num-devices", "2", "--ulysses-size", "1"],
                "num-devices (2) must equal ulysses-size (1)",
            ),
            (
                ["--num-devices", "2", "--ulysses-size", "2", "--use-cfg", "--cfg-parallel"],
                "num-devices (2) must equal 2 * ulysses-size (2) = 4",
            ),
            (
                ["--num-devices", "2", "--ulysses-size", "1", "--cfg-parallel"],
                "cfg-parallel requires use-cfg to be enabled",
            ),
        ],
    )
    def test_video_generate_invalid_parallel_contract_rejected_during_parse(self, parallel_args, error, capsys):
        """vi: invalid parallel combinations fail registry parsing before inference."""
        with pytest.raises(SystemExit) as exc_info:
            self._parse(
                "video_generate",
                [
                    "--model-id",
                    "test-model",
                    "--batch-size",
                    "1",
                    "--seq-len",
                    "16",
                    *parallel_args,
                ],
            )
        assert exc_info.value.code == 2
        assert error in capsys.readouterr().err

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


# ── effective_len_ge1 rounding parity (issue #496) ───────────────────


class TestEffectiveLenGe1:
    """effective_len_ge1 matches UserInputConfig.get_request_info rounding.

    Formula: length - floor(length × rate) ≥ 1 — the cached prefix is floored
    first, then subtracted; decode phase ignores the hit rate (mirrors
    get_effective_prefix_cache_hit_rate). Regression for issue #496: the old
    int(length × (1 - rate)) truncated the remaining fraction and wrongly
    rejected single-token queries at hit rate 0.5.
    """

    _ISSUE_BASE_ARGS = [
        "--model-id",
        "tests/assets/model_config/qwen3_dense_0_6b",
        "--device",
        "TEST_DEVICE",
        "--num-devices",
        "1",
        "--num-queries",
        "1",
        "--query-length",
        "1",
        "--context-length",
        "128",
    ]

    def _parse(self, module_id: str, argv: list[str]):
        """Run parse_module_args with given argv, capture SystemExit."""
        spec = get_spec(module_id)
        parser = build_argparser(spec)
        with patch.object(sys, "argv", ["prog", *argv]):
            return parse_module_args(spec, parser)

    # -- unit level: effective_len_ge1 (text_generate) --

    @pytest.mark.parametrize(
        ("length", "rate"),
        [
            (1, 0.0),
            (1, 0.5),  # issue #496: int(1 × 0.5) == 0 rejected this before
            (1, 0.9),
            (2, 0.6),
            (10, 0.91),  # issue #496: high hit rate on a short length
            (128, 0.99),
        ],
    )
    def test_prefill_boundaries_pass(self, length, rate):
        """length - floor(length × rate) ≥ 1 holds for any rate < 1."""
        params = {"query-length": length, "prefix-cache-hit-rate": rate}
        assert V.effective_len_ge1(params) is None

    @pytest.mark.parametrize("length", [1, 2, 128])
    def test_prefill_rate_one_rejected(self, length):
        """Rate 1 caches every token → effective length 0 → rejected."""
        params = {"query-length": length, "prefix-cache-hit-rate": 1.0}
        msg = V.effective_len_ge1(params)
        assert msg is not None
        assert "effective length must be ≥ 1" in msg

    @pytest.mark.parametrize("rate", [0.5, 1.0])
    def test_decode_ignores_rate(self, rate):
        """Decode phase ignores the hit rate entirely."""
        params = {"query-length": 1, "prefix-cache-hit-rate": rate, "decode": True}
        assert V.effective_len_ge1(params) is None

    @pytest.mark.parametrize("decode", [None, False])
    def test_prefill_phase_still_applies_rate(self, decode):
        """Without an explicit decode flag the rate applies (default prefill)."""
        params = {"query-length": 1, "prefix-cache-hit-rate": 1.0, "decode": decode}
        assert V.effective_len_ge1(params) is not None

    def test_missing_or_zero_length_skipped(self):
        """No/zero length (throughput-less context) skips the check."""
        assert V.effective_len_ge1({"prefix-cache-hit-rate": 0.5}) is None
        assert V.effective_len_ge1({"query-length": 0, "prefix-cache-hit-rate": 0.5}) is None

    def test_missing_rate_defaults_to_zero(self):
        assert V.effective_len_ge1({"query-length": 1}) is None

    # -- unit level: effective_len_ge1_to (throughput) --

    @pytest.mark.parametrize(
        ("length", "rate"),
        [
            (1, 0.5),
            (2, 0.6),
            (10, 0.91),
        ],
    )
    def test_throughput_boundaries_pass(self, length, rate):
        """Same rounding as OptimizerData.get_effective_input_length."""
        params = {"input-length": length, "prefix-cache-hit-rate": rate}
        assert V.effective_len_ge1_to(params) is None

    def test_throughput_rate_one_rejected(self):
        params = {"input-length": 1, "prefix-cache-hit-rate": 1.0}
        msg = V.effective_len_ge1_to(params)
        assert msg is not None
        assert "effective input length must be ≥ 1" in msg

    def test_throughput_yaml_path_skipped(self):
        """input-length may be a YAML distribution path; the check is skipped."""
        params = {"input-length": "length_dist.yaml", "prefix-cache-hit-rate": 0.5}
        assert V.effective_len_ge1_to(params) is None

    # -- CLI level: the four issue #496 commands parse successfully --

    @pytest.mark.parametrize("rate", ["0", "0.5"])
    @pytest.mark.parametrize("phase_flag", ["--prefill", "--decode"])
    def test_issue_496_commands_parse(self, phase_flag, rate):
        """Issue #496 repro matrix: query-length 1 × both phases × rate {0, 0.5}."""
        ns = self._parse(
            "text_generate",
            [
                *self._ISSUE_BASE_ARGS,
                phase_flag,
                "--prefix-cache-hit-rate",
                rate,
            ],
        )
        assert ns.query_length == 1

    def test_issue_496_rate_one_still_rejected(self, capsys):
        """Rate 1 stays invalid: the param declares exclusive_max=1."""
        with pytest.raises(SystemExit) as exc_info:
            self._parse(
                "text_generate",
                [*self._ISSUE_BASE_ARGS, "--prefill", "--prefix-cache-hit-rate", "1"],
            )
        assert exc_info.value.code == 2
        assert "must be < 1" in capsys.readouterr().err


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


# ── Capacity planning: derived --num-devices ─────────────────────────


class TestTargetQpsNumDevicesSemantics:
    """--target-qps turns --num-devices into a derived value.

    The Param default is 1, so "explicitly 1" and "not passed" are only
    distinguishable through the parser's provided set.
    """

    def test_auto_mode_clears_num_devices(self):
        from cli.registry.validators import target_qps_num_devices_semantics

        params = {"target-qps": 1000.0, "num-devices": 1}
        assert target_qps_num_devices_semantics(params, {"target-qps"}) is None
        assert params["num-devices"] is None

    def test_explicit_num_devices_is_preserved(self):
        from cli.registry.validators import target_qps_num_devices_semantics

        params = {"target-qps": 1000.0, "num-devices": 1}
        assert target_qps_num_devices_semantics(params, {"target-qps", "num-devices"}) is None
        assert params["num-devices"] == 1

    def test_without_target_qps_num_devices_is_untouched(self):
        from cli.registry.validators import target_qps_num_devices_semantics

        params = {"target-qps": None, "num-devices": 1}
        assert target_qps_num_devices_semantics(params, set()) is None
        assert params["num-devices"] == 1

    def test_validator_is_declared_before_parallel_combo_check(self):
        """The cleared value must reach validParallelCombo, which reads params."""
        spec = get_spec("throughput_optimizer")
        names = [v.name for v in spec.validators]
        assert names.index("targetQpsNumDevicesSemantics") < names.index("validParallelCombo")

    def test_auto_mode_clears_num_devices_through_the_parser(self):
        spec = get_spec("throughput_optimizer")
        parser = build_argparser(spec)
        argv = [
            "--model-id",
            "Qwen/Qwen3-32B",
            "--input-length",
            "1024",
            "--output-length",
            "512",
            "--target-qps",
            "1000",
            "--tp-sizes",
            "8",
        ]
        with patch.object(sys, "argv", ["prog", *argv]):
            args = parse_module_args(spec, parser)
        # Cleared so CapacityPlanner derives the budget range from comm_grid, and
        # --tp-sizes is no longer rejected against the placeholder value of 1.
        assert args.num_devices is None
        assert args.tp_sizes == [8]


class TestValidatorAdjustmentWriteBack:
    """Validator param adjustments are applied back, conservatively."""

    def test_changed_value_is_written_back(self):
        from types import SimpleNamespace

        from cli.registry.argparse_adapter import _apply_validator_adjustments

        field = SimpleNamespace(name="num-devices", dest_name="num_devices")
        spec = SimpleNamespace(fields=[field])
        args = SimpleNamespace(num_devices=1)
        _apply_validator_adjustments(args, spec, {"num-devices": None})
        assert args.num_devices is None

    def test_unchanged_value_leaves_namespace_untouched(self):
        from types import SimpleNamespace

        from cli.registry.argparse_adapter import _apply_validator_adjustments

        field = SimpleNamespace(name="num-devices", dest_name="num_devices")
        spec = SimpleNamespace(fields=[field])
        args = SimpleNamespace(num_devices=8)
        _apply_validator_adjustments(args, spec, {"num-devices": 8})
        assert args.num_devices == 8

    def test_absent_namespace_attribute_is_not_created(self):
        """argparse.SUPPRESS off-flags must not be resurrected as None."""
        from types import SimpleNamespace

        from cli.registry.argparse_adapter import _apply_validator_adjustments

        field = SimpleNamespace(name="no-foo", dest_name="no_foo")
        spec = SimpleNamespace(fields=[field])
        args = SimpleNamespace()
        _apply_validator_adjustments(args, spec, {"no-foo": None})
        assert not hasattr(args, "no_foo")

    def test_param_missing_from_params_is_skipped(self):
        from types import SimpleNamespace

        from cli.registry.argparse_adapter import _apply_validator_adjustments

        field = SimpleNamespace(name="num-devices", dest_name="num_devices")
        spec = SimpleNamespace(fields=[field])
        args = SimpleNamespace(num_devices=1)
        _apply_validator_adjustments(args, spec, {})
        assert args.num_devices == 1


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

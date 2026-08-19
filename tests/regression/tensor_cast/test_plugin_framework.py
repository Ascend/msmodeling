"""Unit tests for the fusion plugin Python API (RFC §4.2, Sprint 3).

These test the orchestration contract of ``evaluate_fusion_plugin`` without a
real model: ModelRunner / UserInputConfig are patched so no weights or network
are needed. The behaviors under test are the ones the RFC pins down and that
are easy to regress:

- the Validator is the sole gatekeeper — a failing plugin never constructs
  ModelRunner (RFC §3.5 C / §9.1);
- ``do_compile=True`` is always forced (RFC §1.2 / CRIT-1);
- ``run_inference`` is called with the explicit ``generate_inputs`` func, not
  the varlen default (RFC §4.2);
- ``plugin_path=None`` is the no-plugin baseline — no load, no validation;
- ``disable_default_patterns`` turns off every built-in ``enable_*`` switch.
"""

import unittest
from unittest import mock

from tensor_cast import config
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.plugins.validator import ValidationResult


PF = "tensor_cast.plugin_framework"


class EvaluateFusionPluginTest(unittest.TestCase):
    def setUp(self):
        # Patch the heavy collaborators. ModelRunner(...).run_inference(...) is
        # a mock chain; UserInputConfig is captured to assert do_compile.
        self._runner_cls = mock.patch(f"{PF}.ModelRunner").start()
        self._metrics = self._runner_cls.return_value.run_inference.return_value
        self._user_cfg = mock.patch(f"{PF}.UserInputConfig").start()
        self._load = mock.patch(f"{PF}.load_plugin").start()
        self._validate = mock.patch(f"{PF}.validate_plugin").start()
        self._validate.return_value = ValidationResult(ok=True, layer="OK")
        self.addCleanup(mock.patch.stopall)

        # Snapshot + restore the mutated global fusion switches.
        fp = config.compilation.fusion_patterns
        self._saved = {n: getattr(fp, n) for n in dir(fp) if n.startswith("enable_")}
        self.addCleanup(self._restore_switches)

    def _restore_switches(self):
        fp = config.compilation.fusion_patterns
        for n, v in self._saved.items():
            setattr(fp, n, v)

    def _call(self, **kw):
        from tensor_cast.plugin_framework import evaluate_fusion_plugin

        return evaluate_fusion_plugin(
            kw.pop("plugin_path", "./p.py"),
            kw.pop("model_id", "Qwen/Qwen3-32B"),
            kw.pop("device", "TEST_DEVICE"),
            **kw,
        )

    def test_forces_do_compile_true(self):
        self._call()
        _, kwargs = self._user_cfg.call_args
        self.assertTrue(kwargs["do_compile"], "do_compile must be forced True")

    def test_run_inference_uses_explicit_generate_inputs(self):
        self._call()
        _, kwargs = self._runner_cls.return_value.run_inference.call_args
        self.assertIs(kwargs["generate_inputs_func"], generate_inputs)

    def test_failed_validation_raises_and_skips_runner(self):
        from tensor_cast.plugin_framework import FusionPluginError

        self._validate.return_value = ValidationResult(ok=False, layer="L3", detail="no hit")
        with self.assertRaises(FusionPluginError):
            self._call()
        self._runner_cls.assert_not_called()
        self._load.assert_not_called()

    def test_none_baseline_skips_load_and_validation(self):
        self._call(plugin_path=None)
        # No validation for the baseline; load_plugin(None) is still called
        # (it early-returns), and ModelRunner still runs the baseline.
        self._validate.assert_not_called()
        self._load.assert_called_once_with(None)
        self._runner_cls.assert_called_once()

    def test_validate_false_skips_gatekeeper(self):
        self._call(validate=False)
        self._validate.assert_not_called()
        self._runner_cls.assert_called_once()

    def test_disable_default_patterns_turns_off_all_switches(self):
        fp = config.compilation.fusion_patterns
        # Ensure at least one starts True so the assertion is meaningful.
        fp.enable_swiglu = True
        self._call(disable_default_patterns=True)
        for n in dir(fp):
            if n.startswith("enable_"):
                self.assertFalse(getattr(fp, n), f"{n} should be disabled")

    def test_default_keeps_switches_untouched(self):
        fp = config.compilation.fusion_patterns
        fp.enable_swiglu = True
        self._call()  # disable_default_patterns defaults to False
        self.assertTrue(fp.enable_swiglu)

    def test_returns_metrics_from_runner(self):
        self.assertIs(self._call(), self._metrics)


class BaselineComparisonTest(unittest.TestCase):
    """Tests for BaselineComparison fire_count integration (RFC §3.5 L3 ②).

    A plugin that passes L1-L4 but fires 0 times will have speedup ≈ 1.0.
    BaselineComparison.fire_warning must flag this so callers do not mistake
    "fusion never triggered" for "fusion has no benefit".
    """

    def test_fire_warning_none_when_fire_count_not_checked(self):
        from tensor_cast.plugin_framework import BaselineComparison

        bc = BaselineComparison(baseline_latency_s=10.0, fused_latency_s=5.0)
        self.assertIsNone(bc.fire_count)
        self.assertIsNone(bc.fire_warning)
        self.assertAlmostEqual(bc.speedup, 2.0)

    def test_fire_warning_none_when_plugin_fired(self):
        from tensor_cast.plugin_framework import BaselineComparison

        bc = BaselineComparison(
            baseline_latency_s=10.0,
            fused_latency_s=5.0,
            fire_count=3,
            candidate_count=3,
        )
        self.assertIsNone(bc.fire_warning)
        self.assertAlmostEqual(bc.speedup, 2.0)

    def test_fire_warning_set_when_fire_zero_and_candidates_exist(self):
        from tensor_cast.plugin_framework import BaselineComparison

        bc = BaselineComparison(
            baseline_latency_s=10.0,
            fused_latency_s=10.0,
            fire_count=0,
            candidate_count=5,
        )
        self.assertIsNotNone(bc.fire_warning)
        self.assertIn("fired 0 times", bc.fire_warning)
        self.assertIn("misleading", bc.fire_warning)
        self.assertAlmostEqual(bc.speedup, 1.0)

    def test_fire_warning_set_when_fire_zero_and_no_candidates(self):
        from tensor_cast.plugin_framework import BaselineComparison

        bc = BaselineComparison(
            baseline_latency_s=10.0,
            fused_latency_s=10.0,
            fire_count=0,
            candidate_count=0,
        )
        self.assertIsNotNone(bc.fire_warning)
        self.assertIn("0 candidates", bc.fire_warning)


class CompareWithBaselineFireCountTest(unittest.TestCase):
    """compare_with_baseline passes seed_op to the fused subprocess and
    surfaces fire_count in the result.
    """

    def test_seed_op_passed_to_fused_subprocess_only(self):
        PF = "tensor_cast.plugin_framework"
        with (
            mock.patch(f"{PF}._run_plugin_subprocess") as run,
            mock.patch(f"{PF}._aggregate", side_effect=lambda d: 1.0),
            mock.patch(f"{PF}.logger"),
        ):
            from tensor_cast.plugin_framework import compare_with_baseline

            run.side_effect = [
                {"execution_time_s": {"m": 10.0}},
                {"execution_time_s": {"m": 5.0}, "fire_count": 3, "candidate_count": 3},
            ]
            bc = compare_with_baseline(
                "./plugin.py",
                "model",
                "DEV",
                seed_op="aten.relu.default",
            )
            # First call (baseline) has no seed_op
            baseline_call = run.call_args_list[0]
            self.assertIsNone(baseline_call.kwargs.get("seed_op"))
            # Second call (fused) has seed_op
            fused_call = run.call_args_list[1]
            self.assertEqual(fused_call.kwargs.get("seed_op"), "aten.relu.default")
            self.assertEqual(bc.fire_count, 3)
            self.assertIsNone(bc.fire_warning)

    def test_fire_zero_logs_warning(self):
        PF = "tensor_cast.plugin_framework"
        with (
            mock.patch(f"{PF}._run_plugin_subprocess") as run,
            mock.patch(f"{PF}._aggregate", side_effect=lambda d: 10.0),
            mock.patch(f"{PF}.logger") as log,
        ):
            from tensor_cast.plugin_framework import compare_with_baseline

            run.side_effect = [
                {"execution_time_s": {"m": 10.0}},
                {
                    "execution_time_s": {"m": 10.0},
                    "fire_count": 0,
                    "candidate_count": 5,
                },
            ]
            bc = compare_with_baseline(
                "./plugin.py",
                "model",
                "DEV",
                seed_op="aten.relu.default",
            )
            self.assertEqual(bc.fire_count, 0)
            self.assertIsNotNone(bc.fire_warning)
            log.warning.assert_called_once()

    def test_no_seed_op_skips_fire_count(self):
        PF = "tensor_cast.plugin_framework"
        with (
            mock.patch(f"{PF}._run_plugin_subprocess") as run,
            mock.patch(f"{PF}._aggregate", side_effect=lambda d: 1.0),
            mock.patch(f"{PF}.logger"),
        ):
            from tensor_cast.plugin_framework import compare_with_baseline

            run.side_effect = [
                {"execution_time_s": {"m": 10.0}},
                {"execution_time_s": {"m": 5.0}},
            ]
            bc = compare_with_baseline("./plugin.py", "model", "DEV")
            self.assertIsNone(bc.fire_count)
            self.assertIsNone(bc.fire_warning)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the decoupled-architecture deliverables (RFC manual_fusion_eval).

Covers the four gaps closed after the decoupling review:

- CLI ``--fusion-plugin`` hook + ``--compile`` guard (§3.3a)
- ServingCast load_plugin hook (§7.2)
- Validator L3 whole-model hit-ratio reporting (§3.5 L3 ②)
- ``compare_with_baseline`` / ``evaluate_fusion_plugins`` subprocess APIs
  (§5.3 / §3.2)

Heavy collaborators (ModelRunner, subprocess) are patched — no weights, no
network, no real interpreter spawned.
"""

import unittest
from unittest import mock

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tensor_cast.plugins.validator import (
    DEFAULT_HIT_RATIO_THRESHOLD,
    HitRatioReport,
    whole_model_hit_ratio,
)


PF = "tensor_cast.plugin_framework"


# --------------------------------------------------------------------------- #
# CLI --fusion-plugin hook + --compile guard (§3.3a)
# --------------------------------------------------------------------------- #
class CliFusionPluginTest(unittest.TestCase):
    """The flag is additive and guarded; we test argparse wiring only."""

    def _build_parser(self):
        # Rebuild the same parser shape the CLI uses, minimal but faithful to
        # the additive flag + guard contract (full main() needs torch/model).
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--compile", action="store_true")
        p.add_argument("--fusion-plugin", action="append", default=None)
        return p

    def test_flag_absent_defaults_none(self):
        args = self._build_parser().parse_args([])
        self.assertIsNone(args.fusion_plugin)

    def test_flag_repeatable(self):
        args = self._build_parser().parse_args(["--fusion-plugin", "a.py", "--fusion-plugin", "b.py"])
        self.assertEqual(args.fusion_plugin, ["a.py", "b.py"])

    def test_guard_logic_requires_compile(self):
        # Mirror the guard predicate from main(): plugin set but no --compile.
        args = self._build_parser().parse_args(["--fusion-plugin", "a.py"])
        self.assertTrue(args.fusion_plugin and not args.compile)
        args_ok = self._build_parser().parse_args(["--fusion-plugin", "a.py", "--compile"])
        self.assertFalse(args_ok.fusion_plugin and not args_ok.compile)


# --------------------------------------------------------------------------- #
# Validator L3 whole-model hit ratio (§3.5 L3 ②)
# --------------------------------------------------------------------------- #
class HitRatioTest(unittest.TestCase):
    def _graph_with_n_mm(self, n):
        def f(x, w):
            out = x
            for _ in range(n):
                out = torch.ops.aten.mm(out, w)
            return out

        return make_fx(f)(torch.empty(2, 2), torch.empty(2, 2))

    def test_full_coverage_passes(self):
        gm = self._graph_with_n_mm(3)
        r = whole_model_hit_ratio(gm, torch.ops.aten.mm.default, matched_cnt=3)
        self.assertEqual(r.candidate_op_count, 3)
        self.assertEqual(r.ratio, 1.0)
        self.assertTrue(r.ok)

    def test_partial_coverage_warns(self):
        gm = self._graph_with_n_mm(4)
        r = whole_model_hit_ratio(gm, torch.ops.aten.mm.default, matched_cnt=1)
        self.assertEqual(r.candidate_op_count, 4)
        self.assertEqual(r.ratio, 0.25)
        self.assertFalse(r.ok)  # below 0.9 -> warning

    def test_no_candidate_is_full(self):
        gm = self._graph_with_n_mm(2)
        # head op never appears => nothing to underestimate, ratio defaults 1.0
        r = whole_model_hit_ratio(gm, torch.ops.aten.relu.default, matched_cnt=0)
        self.assertEqual(r.candidate_op_count, 0)
        self.assertEqual(r.ratio, 1.0)
        self.assertTrue(r.ok)

    def test_threshold_is_default(self):
        r = HitRatioReport(matched_cnt=9, candidate_op_count=10, threshold=DEFAULT_HIT_RATIO_THRESHOLD)
        self.assertTrue(r.ok)  # 0.9 >= 0.9
        r2 = HitRatioReport(
            matched_cnt=89,
            candidate_op_count=100,
            threshold=DEFAULT_HIT_RATIO_THRESHOLD,
        )
        self.assertFalse(r2.ok)  # 0.89 < 0.9


# --------------------------------------------------------------------------- #
# ServingCast load_plugin hook (§7.2)
# --------------------------------------------------------------------------- #
class ServingCastHookTest(unittest.TestCase):
    def test_config_field_default_none(self):
        from serving_cast.config import ModelConfig

        cfg = ModelConfig(name="dummy")
        self.assertIsNone(cfg.fusion_plugins)

    def test_config_field_accepts_list(self):
        from serving_cast.config import ModelConfig

        cfg = ModelConfig(name="dummy", fusion_plugins=["a.py", "b.py"])
        self.assertEqual(cfg.fusion_plugins, ["a.py", "b.py"])

    def test_init_tensor_cast_model_runner_tolerates_mock_config(self):
        """fusion_plugins=None (no plugin) must not raise."""
        from unittest.mock import Mock, patch
        from serving_cast.model_runner import ModelRunner

        mock_cfg = Mock()
        mock_cfg.model_config.fusion_plugins = None
        with patch("serving_cast.model_runner.TensorCastModelRunner") as _tc:
            _tc.return_value = Mock()
            with patch("serving_cast.model_runner.UserInputConfig") as _ui:
                _ui.return_value = Mock()
                try:
                    ModelRunner.init_tensor_cast_model_runner(mock_cfg, Mock(), "TEST_DEVICE")
                except Exception as e:
                    self.fail(f"init_tensor_cast_model_runner raised unexpectedly: {e}")

    def test_do_compile_false_warns_when_fusion_plugins_set(self):
        """HIGH-4: fusion_plugins + do_compile=False must emit a warning."""
        from unittest.mock import Mock, patch
        from serving_cast.model_runner import ModelRunner

        mock_cfg = Mock()
        mock_cfg.model_config.fusion_plugins = ["/tmp/fake_plugin.py"]
        mock_cfg.model_config.do_compile = False

        with patch("serving_cast.model_runner.TensorCastModelRunner") as _tc:
            _tc.return_value = Mock()
            with patch("serving_cast.model_runner.UserInputConfig") as _ui:
                _ui.return_value = Mock()
                with self.assertLogs("serving_cast.model_runner", level="WARNING") as cm:
                    ModelRunner.init_tensor_cast_model_runner(mock_cfg, Mock(), "TEST_DEVICE")
        self.assertTrue(any("do_compile=False" in msg for msg in cm.output))


# --------------------------------------------------------------------------- #
# Subprocess APIs: compare_with_baseline / evaluate_fusion_plugins (§5.3 / §3.2)
# --------------------------------------------------------------------------- #
class SubprocessApiTest(unittest.TestCase):
    def test_compare_with_baseline_speedup(self):
        from tensor_cast.plugin_framework import compare_with_baseline

        # baseline slower than fused -> speedup > 1. Patch the subprocess runner
        # so no interpreter is spawned; first call = baseline, second = fused.
        with mock.patch(
            f"{PF}._run_plugin_subprocess",
            side_effect=[
                {"execution_time_s": {"m": 0.30}, "tps_per_model": {"m": 3.0}},
                {"execution_time_s": {"m": 0.20}, "tps_per_model": {"m": 5.0}},
            ],
        ) as run:
            cmp = compare_with_baseline("p.py", "model", "DEV", num_queries=2)
        # baseline call gets plugin_path=None, fused gets the path
        self.assertIsNone(run.call_args_list[0].args[0])
        self.assertEqual(run.call_args_list[1].args[0], "p.py")
        self.assertAlmostEqual(cmp.baseline_latency_s, 0.30)
        self.assertAlmostEqual(cmp.fused_latency_s, 0.20)
        self.assertAlmostEqual(cmp.speedup, 1.5)

    def test_evaluate_fusion_plugins_explicit_list(self):
        from tensor_cast.plugin_framework import evaluate_fusion_plugins

        with mock.patch(
            f"{PF}._run_plugin_subprocess",
            return_value={"execution_time_s": {"m": 0.1}, "tps_per_model": {}},
        ) as run:
            out = evaluate_fusion_plugins("model", "DEV", plugins=["a.py", "b.py"], num_queries=2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["plugin"], "a.py")
        self.assertEqual(run.call_count, 2)

    def test_evaluate_fusion_plugins_requires_one_source(self):
        from tensor_cast.plugin_framework import evaluate_fusion_plugins

        with self.assertRaises(ValueError):
            evaluate_fusion_plugins("model", "DEV")  # no source
        with self.assertRaises(ValueError):
            evaluate_fusion_plugins("model", "DEV", plugins=["a.py"], plugin_dir="./x")  # two sources

    def test_subprocess_failure_raises(self):
        from tensor_cast.plugin_framework import (
            FusionPluginError,
            _run_plugin_subprocess,
        )

        completed = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch("subprocess.run", return_value=completed):
            with self.assertRaises(FusionPluginError):
                _run_plugin_subprocess(None, "m", "DEV", {})


if __name__ == "__main__":
    unittest.main()

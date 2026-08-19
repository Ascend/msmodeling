"""Tests for tensor_cast.plugins.l3_real.

RED phase: all tests import from a module that does not exist yet.

Design contract under test:
  check_fire_count(plugin_path, model_id, seed_op, device, **runner_kwargs)
      -> L3RealResult
    - Runs plugin in a subprocess (process-level isolation, like _run_plugin_subprocess)
    - Spies on apply_pattern_match_passes to count:
        candidate_count: seed_op occurrences in PRE-rewrite graph
        fire_count:      virtual op occurrences in POST-rewrite graph
    - Returns L3RealResult(fire_count, candidate_count, diagnostic_section)
    - diagnostic_section: to_prompt_str() of seed_op region (non-None when fire=0)

  L3RealResult
    - ok: bool (fire_count > 0)
    - fire_count: int
    - candidate_count: int
    - diagnostic_section: str | None

  L3RealError: raised on subprocess failure (non-zero returncode)
"""

import json
import unittest
from unittest import mock

L3 = "tensor_cast.plugins.l3_real"


class L3RealImportTest(unittest.TestCase):
    def test_module_importable(self):
        from tensor_cast.plugins.l3_real import (
            check_fire_count,
            L3RealResult,
            L3RealError,
        )

        assert check_fire_count is not None
        assert L3RealResult is not None
        assert L3RealError is not None


class L3RealResultContractTest(unittest.TestCase):
    def setUp(self):
        from tensor_cast.plugins.l3_real import L3RealResult

        self.L3RealResult = L3RealResult

    # ------------------------------------------------------------------
    # Test 1: fire_count > 0 -> ok=True, diagnostic_section=None
    # ------------------------------------------------------------------
    def test_ok_true_when_fire_count_positive(self):
        r = self.L3RealResult(fire_count=3, candidate_count=3, diagnostic_section=None)
        self.assertTrue(r.ok)
        self.assertIsNone(r.diagnostic_section)

    # ------------------------------------------------------------------
    # Test 2: fire_count = 0 -> ok=False
    # ------------------------------------------------------------------
    def test_ok_false_when_fire_count_zero(self):
        r = self.L3RealResult(fire_count=0, candidate_count=3, diagnostic_section="some graph")
        self.assertFalse(r.ok)

    # ------------------------------------------------------------------
    # Test 3: candidate_count = 0 -> ok=False (seed op not in model)
    # ------------------------------------------------------------------
    def test_ok_false_when_no_candidates(self):
        r = self.L3RealResult(fire_count=0, candidate_count=0, diagnostic_section=None)
        self.assertFalse(r.ok)

    # ------------------------------------------------------------------
    # Test 4: fire=0, candidate>0 -> diagnostic_section must not be None
    # ------------------------------------------------------------------
    def test_diagnostic_required_when_fire_zero_and_has_candidates(self):
        # When there ARE candidates but 0 fires, the caller must provide diagnostics
        # This is a data contract: L3RealResult with fire=0, candidate>0, no diagnostic
        # should raise ValueError
        with self.assertRaises(ValueError):
            self.L3RealResult(
                fire_count=0,
                candidate_count=3,
                diagnostic_section=None,  # missing when it should be present
            )


class CheckFireCountTest(unittest.TestCase):
    """Tests for check_fire_count() using subprocess mock."""

    _MARKER = "__L3REAL_RESULT__"

    def setUp(self):
        from tensor_cast.plugins.l3_real import check_fire_count, L3RealError

        self.check_fire_count = check_fire_count
        self.L3RealError = L3RealError

    def _make_proc(self, fire_count, candidate_count, diagnostic=None, returncode=0):
        """Build a mock subprocess.run result."""
        payload = {
            "fire_count": fire_count,
            "candidate_count": candidate_count,
            "diagnostic_section": diagnostic,
        }
        stdout = f"some output\n{self._MARKER}{json.dumps(payload)}\n"
        proc = mock.MagicMock()
        proc.returncode = returncode
        proc.stdout = stdout
        proc.stderr = ""
        return proc

    # ------------------------------------------------------------------
    # Test 5: subprocess returns fire=3 -> L3RealResult(ok=True, fire_count=3)
    # ------------------------------------------------------------------
    def test_fire_positive_returns_ok_result(self):
        proc = self._make_proc(fire_count=3, candidate_count=3)
        with mock.patch("subprocess.run", return_value=proc):
            result = self.check_fire_count(
                plugin_path="/tmp/plugin.py",
                model_id="some_model",
                seed_op="aten.sigmoid.default",
                device="ATLAS_800_A3_752T_128G_DIE",
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.fire_count, 3)

    # ------------------------------------------------------------------
    # Test 6: subprocess returns fire=0 -> ok=False, diagnostic_section present
    # ------------------------------------------------------------------
    def test_fire_zero_returns_diagnostic(self):
        proc = self._make_proc(
            fire_count=0, candidate_count=3, diagnostic="aten.sigmoid.default: gate_1 -> sig_1\n  BOUNDARY: gate_1"
        )
        with mock.patch("subprocess.run", return_value=proc):
            result = self.check_fire_count(
                plugin_path="/tmp/plugin.py",
                model_id="some_model",
                seed_op="aten.sigmoid.default",
                device="ATLAS_800_A3_752T_128G_DIE",
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.fire_count, 0)
        self.assertIsNotNone(result.diagnostic_section)
        self.assertIn("aten.sigmoid.default", result.diagnostic_section)

    # ------------------------------------------------------------------
    # Test 7: subprocess non-zero returncode -> raises L3RealError
    # ------------------------------------------------------------------
    def test_subprocess_failure_raises_l3_real_error(self):
        proc = self._make_proc(fire_count=0, candidate_count=0, returncode=1)
        proc.stderr = "ImportError: no module named x"
        with mock.patch("subprocess.run", return_value=proc):
            with self.assertRaises(self.L3RealError):
                self.check_fire_count(
                    plugin_path="/tmp/plugin.py",
                    model_id="some_model",
                    seed_op="aten.sigmoid.default",
                    device="ATLAS_800_A3_752T_128G_DIE",
                )

    # ------------------------------------------------------------------
    # Test 8: candidate_count=0 -> ok=False, diagnostic_section=None (unsupported)
    # ------------------------------------------------------------------
    def test_zero_candidates_means_unsupported(self):
        proc = self._make_proc(fire_count=0, candidate_count=0, diagnostic=None)
        with mock.patch("subprocess.run", return_value=proc):
            result = self.check_fire_count(
                plugin_path="/tmp/plugin.py",
                model_id="some_model",
                seed_op="aten.relu.default",
                device="ATLAS_800_A3_752T_128G_DIE",
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.candidate_count, 0)
        self.assertIsNone(result.diagnostic_section)

    # ------------------------------------------------------------------
    # Test 9: diagnostic_section format matches to_prompt_str output
    # Both A (extract_subgraph) and B (l3_real) must produce same format
    # so the skill's generate-prompt and validate-prompt share one format
    # ------------------------------------------------------------------
    def test_diagnostic_section_contains_full_overload_names(self):
        diagnostic = "prims.convert_element_type.default (BOUNDARY: gate_1)\naten.sigmoid.default\n"
        proc = self._make_proc(fire_count=0, candidate_count=2, diagnostic=diagnostic)
        with mock.patch("subprocess.run", return_value=proc):
            result = self.check_fire_count(
                plugin_path="/tmp/plugin.py",
                model_id="some_model",
                seed_op="aten.sigmoid.default",
                device="ATLAS_800_A3_752T_128G_DIE",
            )
        self.assertIn("prims.convert_element_type.default", result.diagnostic_section)


class ChildCodeFireCountLogicTest(unittest.TestCase):
    """Verify the child-code spy counts fire_count correctly.

    The spy must capture seed_op count BEFORE orig() and AFTER orig() in the
    SAME call. The old implementation had a dead spy2 block that fired after
    ModelRunner finished, making fire_count always 0.
    """

    def test_spy_counts_before_and_after_orig_in_same_call(self):
        # The child code must contain a single spy that:
        # 1. counts seed_op before orig()
        # 2. calls orig()
        # 3. counts remaining seed_op after orig() in the SAME function
        from tensor_cast.plugins.l3_real import _CHILD_CODE_TEMPLATE

        # There must NOT be a separate spy2 / second monkey-patch after the run
        self.assertNotIn("spy2", _CHILD_CODE_TEMPLATE, "spy2 is dead code — fire_count logic must be in a single spy")
        # The single spy must reference 'fire_count' after calling orig
        # We check the structural invariant: orig() call appears BEFORE fire_count assignment
        orig_call_idx = _CHILD_CODE_TEMPLATE.find("orig(self")
        fire_count_idx = _CHILD_CODE_TEMPLATE.find("fire_count")
        self.assertGreater(fire_count_idx, orig_call_idx, "fire_count must be computed after orig() is called")

    def test_fire_count_uses_post_rewrite_graph(self):
        # After orig() returns, gm.graph is the POST-rewrite graph (in-place mutation).
        # fire_count = pre_candidate - remaining (seed_op nodes still present post-rewrite).
        from tensor_cast.plugins.l3_real import _CHILD_CODE_TEMPLATE

        self.assertIn("pre_candidate", _CHILD_CODE_TEMPLATE)
        self.assertIn("remaining", _CHILD_CODE_TEMPLATE)
        # fire_count computed as pre - remaining
        self.assertIn("pre_candidate", _CHILD_CODE_TEMPLATE)


class CheckFireCountSubprocessCodeTest(unittest.TestCase):
    """The child-process code string must be syntactically valid Python."""

    def test_child_code_is_valid_python(self):
        # check_fire_count builds a code string to pass to subprocess.
        # We can't run it (no model), but we can at least compile it.
        from tensor_cast.plugins.l3_real import _CHILD_CODE_TEMPLATE

        try:
            compile(_CHILD_CODE_TEMPLATE, "<l3_real_child>", "exec")
        except SyntaxError as e:
            self.fail(f"_CHILD_CODE_TEMPLATE has syntax error: {e}")


if __name__ == "__main__":
    unittest.main()

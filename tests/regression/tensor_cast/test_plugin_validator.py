"""Unit tests for the fusion plugin validator.

L1 (static / AST) checks are pure and fast. The full L1->L4 path imports and
registers a real plugin into the in-process global tables; since torch
custom_op cannot be re-registered, the integration test uses a unique virtual
op name and runs once.
"""

import textwrap
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from tensor_cast.plugins.validator import _check_static, validate_plugin


def _write(dir_path, name, body):
    p = Path(dir_path) / name
    p.write_text(textwrap.dedent(body))
    return str(p)


class ValidatorStaticTest(unittest.TestCase):
    """L1 static checks — no execution, no torch dependency."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_syntax_error_fails_l1(self):
        p = _write(self.tmp, "bad.py", "def register_all_patterns(:\n    pass\n")
        r = _check_static(p)
        self.assertFalse(r.ok)
        self.assertEqual(r.layer, "L1")

    def test_missing_entry_fails_l1(self):
        p = _write(self.tmp, "noentry.py", "VALUE = 1\n")
        r = _check_static(p)
        self.assertFalse(r.ok)
        self.assertIn("register_all_patterns", r.detail)

    def test_private_import_fails_l1(self):
        p = _write(
            self.tmp,
            "priv.py",
            """
            from tensor_cast.utils import _secret_helper
            def register_all_patterns():
                pass
            """,
        )
        r = _check_static(p)
        self.assertFalse(r.ok)
        self.assertEqual(r.layer, "L1")

    def test_valid_source_passes_l1(self):
        p = _write(
            self.tmp,
            "ok.py",
            """
            from tensor_cast.compilation.patterns import register_pattern
            def register_all_patterns():
                pass
            """,
        )
        self.assertTrue(_check_static(p).ok)

    def test_nonexistent_path_fails_l1(self):
        self.assertFalse(_check_static(str(Path(self.tmp) / "absent.py")).ok)

    def test_default_namespace_op_passes_l1(self):
        # No __plugin_namespace__ → defaults to user_fusion; op carries prefix.
        p = _write(
            self.tmp,
            "default_ns.py",
            """
            from tensor_cast.utils import register_tensor_cast_op
            @register_tensor_cast_op("user_fusion_mm_relu")
            def _meta(x, w):
                return x
            def register_all_patterns():
                pass
            """,
        )
        r = _check_static(p)
        self.assertTrue(r.ok, r.detail)
        self.assertIn("user_fusion", r.detail)

    def test_custom_namespace_op_passes_l1(self):
        p = _write(
            self.tmp,
            "custom_ns.py",
            """
            from tensor_cast.utils import register_tensor_cast_op
            __plugin_namespace__ = "my_team"
            @register_tensor_cast_op("my_team_mm_relu")
            def _meta(x, w):
                return x
            def register_all_patterns():
                pass
            """,
        )
        self.assertTrue(_check_static(p).ok)

    def test_op_without_namespace_prefix_fails_l1(self):
        # Declared op name does not carry the (default) namespace prefix.
        p = _write(
            self.tmp,
            "bad_ns.py",
            """
            from tensor_cast.utils import register_tensor_cast_op
            @register_tensor_cast_op("mm_relu")
            def _meta(x, w):
                return x
            def register_all_patterns():
                pass
            """,
        )
        r = _check_static(p)
        self.assertFalse(r.ok)
        self.assertEqual(r.layer, "L1")
        self.assertIn("namespace prefix", r.detail)

    def test_pattern_only_plugin_passes_l1(self):
        # No new virtual op declared → namespace prefix check is a no-op.
        p = _write(
            self.tmp,
            "pattern_only.py",
            """
            from tensor_cast.compilation.patterns import register_pattern
            def register_all_patterns():
                pass
            """,
        )
        self.assertTrue(_check_static(p).ok)

    def test_pattern_name_without_namespace_prefix_fails_l1(self):
        # register_pattern(name="mm_relu") without namespace prefix → L1 fail.
        p = _write(
            self.tmp,
            "bad_pat_name.py",
            """
            from tensor_cast.utils import register_tensor_cast_op
            from tensor_cast.compilation.patterns import register_pattern
            @register_tensor_cast_op("user_fusion_mm_relu")
            def _meta(x, w):
                return x
            def register_all_patterns():
                register_pattern(name="mm_relu", pattern=lambda x: x,
                                 replacement=lambda x: x, example_inputs=[x])
            """,
        )
        r = _check_static(p)
        self.assertFalse(r.ok)
        self.assertEqual(r.layer, "L1")
        self.assertIn("pattern name", r.detail)
        self.assertIn("namespace prefix", r.detail)

    def test_pattern_name_with_namespace_prefix_passes_l1(self):
        # register_pattern(name="user_fusion_mm_relu") → L1 pass.
        p = _write(
            self.tmp,
            "good_pat_name.py",
            """
            from tensor_cast.utils import register_tensor_cast_op
            from tensor_cast.compilation.patterns import register_pattern
            @register_tensor_cast_op("user_fusion_mm_relu")
            def _meta(x, w):
                return x
            def register_all_patterns():
                register_pattern(name="user_fusion_mm_relu", pattern=lambda x: x,
                                 replacement=lambda x: x, example_inputs=[x])
            """,
        )
        r = _check_static(p)
        self.assertTrue(r.ok, r.detail)

    def test_pattern_name_fstring_skips_l1_check(self):
        # f-string names are dynamic (inherently namespace-prefixed) → skipped at L1.
        p = _write(
            self.tmp,
            "fstring_pat_name.py",
            """
            from tensor_cast.utils import register_tensor_cast_op
            from tensor_cast.compilation.patterns import register_pattern
            __plugin_namespace__ = "my_team"
            @register_tensor_cast_op("my_team_mm_relu")
            def _meta(x, w):
                return x
            def register_all_patterns():
                register_pattern(name=f"{__plugin_namespace__}_mm_relu", pattern=lambda x: x,
                                 replacement=lambda x: x, example_inputs=[x])
            """,
        )
        r = _check_static(p)
        self.assertTrue(r.ok, r.detail)


class ValidatorIntegrationTest(unittest.TestCase):
    """Full L1->L4 path against a real, registrable plugin."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = self._tmp.name
        # Unique op name so re-running pytest never hits custom_op redefinition.
        self.op = f"user_fusion_mmrelu_{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        self._tmp.cleanup()

    def _make_plugin(self, op_name, with_props=True):
        props_block = (
            f"""
            @OpInvokeInfo.register_op_properties(
                torch.ops.tensor_cast.{op_name}.default
            )
            def _props(info):
                x, w = info.args
                m, k, n = x.size(0), x.size(1), w.size(1)
                props = info.get_memory_access_properties()
                props.compute_ops[x.dtype] = OpInvokeInfo.ComputeOps(
                    mma_ops=m * n * k * 2, gp_ops=m * n)
                return props
            """
            if with_props
            else ""
        )
        return _write(
            self.tmp,
            f"{op_name}.py",
            f"""
            import torch
            from tensor_cast.utils import register_tensor_cast_op
            from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
            from tensor_cast.compilation.patterns import register_pattern

            @register_tensor_cast_op("{op_name}")
            def _meta(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
                return torch.empty(x.size(0), w.size(1), dtype=x.dtype, device="meta")

            def _pattern(x, w):
                return torch.ops.aten.relu(torch.ops.aten.mm(x, w))

            def _replacement(x, w):
                return torch.ops.tensor_cast.{op_name}(x, w)
            {props_block}
            def register_all_patterns():
                ex = [torch.empty(1, 1, dtype=torch.float16, device="meta"),
                      torch.empty(1, 1, dtype=torch.float16, device="meta")]
                register_pattern(name="{op_name}_pat", pattern=_pattern,
                                 replacement=_replacement, example_inputs=ex)
            """,
        )

    def test_valid_plugin_passes_all_layers(self):
        path = self._make_plugin(self.op, with_props=True)
        r = validate_plugin(path)
        self.assertTrue(r.ok, f"expected OK, got {r.layer}: {r.detail}")
        self.assertEqual(r.layer, "OK")

    def test_missing_props_fails_l4(self):
        op = f"{self.op}_noprops"
        path = self._make_plugin(op, with_props=False)
        r = validate_plugin(path)
        self.assertFalse(r.ok)
        self.assertEqual(r.layer, "L4")

    def test_second_call_same_path_idempotent_l2_pass(self):
        """HIGH-1: validate_plugin() must be re-entrant within the same process.

        The plugin is already loaded after the first call, so _check_register()
        must return an idempotent L2 pass (not a false-positive FAIL) on the
        second call.
        """
        path = self._make_plugin(self.op, with_props=True)
        r1 = validate_plugin(path)
        self.assertTrue(r1.ok, f"first call failed: {r1.layer}: {r1.detail}")
        # Second call with same path — must not fail at L2
        r2 = validate_plugin(path)
        self.assertTrue(r2.ok, f"second call (re-entrant) failed at {r2.layer}: {r2.detail}")
        self.assertEqual(r2.layer, "OK")
        self.assertIn("idempotent", r2.detail.lower())


if __name__ == "__main__":
    unittest.main()

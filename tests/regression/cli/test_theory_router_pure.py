"""Tests for theory_router.py — template engine and routing logic."""

import unittest

from tools.perf_data_collection.grid_generator.theory_router import (
    _GRID_REGISTRY,
    _resolve_grid,
    collect_theory_generated_rows,
    generate_from_template,
    resolve_theory_pattern_name,
    resolve_complex_generator,
    get_theory_generator,
    get_default_theory_generator,
    default_complex_generators,
)


class TestCollectTheoryGeneratedRows(unittest.TestCase):
    def test_basic_collection(self):
        import tempfile
        from pathlib import Path
        from tools.perf_data_collection.grid_generator.generators import TheoryShapeRow

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "MatMulV2.csv"
            headers = [
                "Input Shapes",
                "Input Data Types",
                "Input Formats",
                "Output Shapes",
                "Output Data Types",
                "Average Duration(us)",
            ]
            source_rows = [
                {
                    "Input Shapes": '"1,5120;5120,25600"',
                    "Input Data Types": "DT_BF16;DT_BF16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": '"1,25600"',
                    "Output Data Types": "DT_BF16",
                    "Average Duration(us)": "10.0",
                }
            ]
            generated = iter(
                [
                    TheoryShapeRow([(128, 5120), (5120, 25600)], [(128, 25600)]),
                    TheoryShapeRow([(256, 5120), (5120, 25600)], [(256, 25600)]),
                ]
            )

            rows = collect_theory_generated_rows(
                headers,
                source_rows,
                generated,
                csv_path=csv_path,
                file_index=1,
                total_files=1,
                max_rows=None,
                rng=None,
            )
            self.assertEqual(len(rows), 2)
            self.assertIn("128,5120;5120,25600", rows[0]["Input Shapes"])
            self.assertEqual(rows[0]["Average Duration(us)"], "0")

    def test_max_rows_limit(self):
        import random
        import tempfile
        from pathlib import Path
        from tools.perf_data_collection.grid_generator.generators import TheoryShapeRow

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "MatMulV2.csv"
            headers = ["Input Shapes", "Output Shapes", "Average Duration(us)"]
            source_rows = [
                {
                    "Input Shapes": '"1,5120;5120,25600"',
                    "Output Shapes": '"1,25600"',
                    "Average Duration(us)": "10.0",
                }
            ]
            generated = iter(
                [TheoryShapeRow([(i * 128, 5120), (5120, 25600)], [(i * 128, 25600)]) for i in range(1, 20)]
            )

            rows = collect_theory_generated_rows(
                headers,
                source_rows,
                generated,
                csv_path=csv_path,
                file_index=1,
                total_files=1,
                max_rows=5,
                rng=random.Random(42),
            )
            self.assertEqual(len(rows), 5)


class TestGridRegistry(unittest.TestCase):
    def test_all_names_resolve(self):
        for name in _GRID_REGISTRY:
            self.assertIsInstance(_GRID_REGISTRY[name], list)
            self.assertGreater(len(_GRID_REGISTRY[name]), 0)


class TestResolveGrid(unittest.TestCase):
    def test_by_name(self):
        result = _resolve_grid("M_GRID")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_inline_list(self):
        self.assertEqual(_resolve_grid([1, 2, 3]), [1, 2, 3])

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            _resolve_grid("NONEXISTENT_GRID")

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            _resolve_grid(42)


class TestGenerateFromTemplate(unittest.TestCase):
    def test_basic_generation(self):
        pattern = self._make_pattern()
        rows = list(generate_from_template(pattern, None))
        self.assertEqual(len(rows), 2)

    def test_constraints_filter(self):
        pattern = self._make_pattern(constraints=["tokens > 200"])
        rows = list(generate_from_template(pattern, None))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].input_shapes[0], (256, 5120))

    def test_with_output_templates(self):
        pattern = self._make_pattern(outputs=["(tokens, hidden)", "(tokens, 1)"])
        rows = list(generate_from_template(pattern, None))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].output_shapes, [(128, 5120), (128, 1)])

    def test_extra_values_dtypes(self):
        pattern = self._make_pattern(
            input_dtypes=["DT_BF16"],
            input_formats=["ND"],
            output_dtypes=["DT_BF16"],
            output_formats=["ND"],
        )
        rows = list(generate_from_template(pattern, None))
        self.assertIn("Input Data Types", rows[0].extra_values)
        self.assertEqual(rows[0].extra_values["Input Data Types"], "DT_BF16")
        self.assertEqual(rows[0].extra_values["Input Formats"], "ND")
        self.assertEqual(rows[0].extra_values["Output Data Types"], "DT_BF16")
        self.assertEqual(rows[0].extra_values["Output Formats"], "ND")

    def test_empty_iters_with_constants_only(self):
        pattern = {
            "iterators": {},
            "constants": {"D": 5120},
            "inputs": ["(D,)"],
            "outputs": ["(D,)"],
        }
        rows = list(generate_from_template(pattern, None))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].input_shapes, [(5120,)])

    def test_two_iterators_product(self):
        pattern = {
            "iterators": {"tokens": [128, 256], "hidden": [4096, 5120]},
            "constants": {},
            "inputs": ["(tokens, hidden)"],
            "outputs": ["(tokens, hidden)"],
        }
        rows = list(generate_from_template(pattern, None))
        self.assertEqual(len(rows), 4)

    def test_multi_input_expression(self):
        pattern = {
            "iterators": {"seq": [1, 2]},
            "constants": {"D": 5120, "R": 64},
            "inputs": ["(seq, D)", "(seq + D, R)"],
            "outputs": ["(seq, D)"],
        }
        rows = list(generate_from_template(pattern, None))
        self.assertEqual(rows[0].input_shapes, [(1, 5120), (5121, 64)])
        self.assertEqual(rows[1].input_shapes, [(2, 5120), (5122, 64)])

    def test_max_func_in_expr(self):
        pattern = {
            "iterators": {"tokens": [1, 2048, 4096]},
            "constants": {"limit": 2048},
            "inputs": ["(max(tokens, limit), 64)"],
            "outputs": ["(max(tokens, limit), 128)"],
        }
        rows = list(generate_from_template(pattern, None))
        self.assertEqual(rows[0].input_shapes, [(2048, 64)])
        self.assertEqual(rows[1].input_shapes, [(2048, 64)])
        self.assertEqual(rows[2].input_shapes, [(4096, 64)])

    def _make_pattern(self, **overrides):
        base = {
            "iterators": {"tokens": [128, 256]},
            "constants": {"hidden": 5120},
            "inputs": ["(tokens, hidden)"],
            "outputs": ["(tokens, hidden)"],
        }
        base.update(overrides)
        return base


class TestResolveTheoryPatternName(unittest.TestCase):
    def test_direct_assignment(self):
        assignments = {"MatMulV2": "MatMulFamily"}
        self.assertEqual(
            resolve_theory_pattern_name("MatMulV2", assignments, {}),
            "MatMulFamily",
        )

    def test_alternate_resolution(self):
        assignments = {"FusedInferAttentionScore": "FIA"}
        meta = {"alternates_of": "FusedInferAttentionScore"}
        self.assertEqual(
            resolve_theory_pattern_name("BatchMatMulV2", assignments, meta),
            "FIA",
        )

    def test_elementwise_query_mode(self):
        assignments = {}
        meta = {"query_mode": "elementwise"}
        self.assertEqual(
            resolve_theory_pattern_name("Add", assignments, meta),
            "elementwise_binary",
        )

    def test_no_match(self):
        self.assertIsNone(resolve_theory_pattern_name("UnknownOp", {}, {}))


class TestResolveComplexGenerator(unittest.TestCase):
    def test_not_in_dict(self):
        result = resolve_complex_generator("nonexistent", None, {}, {})
        self.assertIsNone(result)

    def test_calls_with_model_names(self):
        called_with = []

        def fake_gen(model_names):
            called_with.append(model_names)
            return iter([])

        result = resolve_complex_generator("test_func", ["deepseek-ai/DeepSeek-V3"], {"test_func": fake_gen}, {})
        self.assertIsNotNone(result)
        rows = list(result)
        self.assertEqual(len(rows), 0)
        self.assertEqual(called_with, [["deepseek-ai/DeepSeek-V3"]])


class TestDefaultComplexGenerators(unittest.TestCase):
    def test_all_registered(self):
        generators = default_complex_generators()
        self.assertIn("_theory_grouped_matmul", generators)
        self.assertIn("_theory_dfc", generators)
        self.assertIn("_theory_fused_attention", generators)
        self.assertIn("_theory_split_qkv_rmsnorm_rope", generators)
        for func in generators.values():
            self.assertTrue(callable(func))


class TestGetDefaultTheoryGenerator(unittest.TestCase):
    def test_returns_none_for_unknown_kernel(self):
        gen = get_default_theory_generator("UnknownKernel", None, {"assignments": {}, "patterns": {}}, {})
        self.assertIsNone(gen)

    def test_returns_none_for_communication_op(self):
        config = {"assignments": {"hcom_allReduce_": "skip"}, "patterns": {}}
        op_meta = {"hcom_allReduce_": {"communication": True}}
        gen = get_default_theory_generator("hcom_allReduce_", None, config, op_meta)
        self.assertIsNone(gen)

    def test_returns_none_for_composite(self):
        config = {"assignments": {}, "patterns": {}}
        op_meta = {"FusedInferAttentionScore": {"composite": True}}
        gen = get_default_theory_generator("FusedInferAttentionScore", None, config, op_meta)
        self.assertIsNone(gen)

    def test_returns_none_for_zero_cost(self):
        config = {
            "assignments": {"TransData": "elementwise_binary"},
            "patterns": {
                "elementwise_binary": {
                    "iterators": {"tokens": [128]},
                    "constants": {"D": 5120},
                    "inputs": ["(tokens, D)"],
                    "outputs": ["(tokens, D)"],
                }
            },
        }
        op_meta = {"TransData": {"zero_cost": True}}
        gen = get_default_theory_generator("TransData", None, config, op_meta)
        self.assertIsNone(gen)

    def test_returns_generator_for_template_pattern(self):
        config = {
            "assignments": {"MatMulV2": "MatMulFamily"},
            "patterns": {
                "MatMulFamily": {
                    "iterators": {"tokens": [128]},
                    "constants": {"hidden": 5120},
                    "inputs": ["(tokens, hidden)"],
                    "outputs": ["(tokens, hidden)"],
                },
            },
        }
        op_meta = {"MatMulV2": {}}
        gen = get_default_theory_generator("MatMulV2", None, config, op_meta)
        self.assertIsNotNone(gen)
        rows = list(gen)
        self.assertEqual(len(rows), 1)


class TestGetTheoryGenerator(unittest.TestCase):
    def setUp(self):
        self.config = {
            "assignments": {"MatMulV2": "TestIdentity"},
            "patterns": {
                "TestIdentity": {
                    "iterators": {"tokens": [128]},
                    "constants": {"hidden": 5120},
                    "inputs": ["(tokens, hidden)"],
                    "outputs": ["(tokens, hidden)"],
                },
            },
        }
        self.op_meta = {"MatMulV2": {}}

    def test_basic_generator(self):
        gen = get_theory_generator(
            "MatMulV2",
            None,
            self.config,
            self.op_meta,
            complex_generators={},
            signature_cache={},
        )
        self.assertIsNotNone(gen)
        rows = list(gen)
        self.assertEqual(len(rows), 1)

    def test_zero_cost_skipped(self):
        self.op_meta["MatMulV2"]["zero_cost"] = True
        gen = get_theory_generator(
            "MatMulV2",
            None,
            self.config,
            self.op_meta,
            complex_generators={},
            signature_cache={},
        )
        self.assertIsNone(gen)

    def test_composite_skipped(self):
        self.op_meta["MatMulV2"]["composite"] = True
        gen = get_theory_generator(
            "MatMulV2",
            None,
            self.config,
            self.op_meta,
            complex_generators={},
            signature_cache={},
        )
        self.assertIsNone(gen)

    def test_communication_skipped(self):
        self.op_meta["MatMulV2"]["communication"] = True
        gen = get_theory_generator(
            "MatMulV2",
            None,
            self.config,
            self.op_meta,
            complex_generators={},
            signature_cache={},
        )
        self.assertIsNone(gen)

    def test_unknown_kernel(self):
        gen = get_theory_generator(
            "UnknownKernel",
            None,
            self.config,
            {},
            complex_generators={},
            signature_cache={},
        )
        self.assertIsNone(gen)

    def test_missing_pattern(self):
        self.config["assignments"]["MatMulV2"] = "NonExistentPattern"
        gen = get_theory_generator(
            "MatMulV2",
            None,
            self.config,
            self.op_meta,
            complex_generators={},
            signature_cache={},
        )
        self.assertIsNone(gen)


if __name__ == "__main__":
    unittest.main()

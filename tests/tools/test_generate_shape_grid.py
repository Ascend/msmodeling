import contextlib
import io
import logging
import os
import sys
import types
import unittest
import warnings
from pathlib import Path

from tools.perf_data_collection.grid_generator.config import (
    load_op_mapping_metadata,
    load_shape_grid_config,
)
from tools.perf_data_collection.grid_generator.evaluator import (
    _parse_shape_expr,
    _split_dims,
    SafeExprEval,
)
from tools.perf_data_collection.grid_generator.shape_grids import M_GRID
from tools.perf_data_collection.grid_generator.theory_router import (
    get_default_theory_generator,
)
from tools.perf_data_collection.grid_generator.utils import (
    build_shape_cell,
    build_shape_text,
    parse_shape_text,
)
from tools.perf_data_collection.memory_estimator import (
    dtype_to_bytes,
    estimate_row_memory,
    exceeds_memory_budget,
)


def _setup_transformers_compat_mock():
    """Mock 'transformers.initialization' for compatibility with older internal modules."""
    try:
        import transformers.modeling_utils

        m = types.ModuleType("transformers.initialization")
        m.no_init_weights = transformers.modeling_utils.no_init_weights
        sys.modules["transformers.initialization"] = m
    except (ImportError, AttributeError):
        pass


_setup_transformers_compat_mock()


# ── Theory mode tests (SafeExprEval + template engine) ──────────


class TestShapeGridLogic(unittest.TestCase):
    def setUp(self):
        self.vars = {"tokens": 128, "D": 5120, "tp": 4}
        self.evaluator = SafeExprEval(self.vars)

    def test_safe_eval(self):
        # Basic arithmetic
        self.assertEqual(self.evaluator.eval("D // tp"), 1280)
        self.assertEqual(self.evaluator.eval("tokens * 2"), 256)
        # Built-in funcs
        self.assertEqual(self.evaluator.eval("max(1, tokens // 256)"), 1)
        self.assertEqual(self.evaluator.eval("abs(-10)"), 10)
        # Custom align func
        self.assertEqual(self.evaluator.eval("align(tokens, 8)"), 128)
        self.assertEqual(self.evaluator.eval("align(1, 8)"), 8)
        self.assertEqual(self.evaluator.eval("align(7, 8)"), 8)
        self.assertEqual(self.evaluator.eval("align(8, 8)"), 8)
        self.assertEqual(self.evaluator.eval("align(9, 8)"), 16)
        # Dot access blocked by regex
        with self.assertRaises(ValueError):
            self.evaluator.eval("().__class__")

    def test_split_dims(self):
        self.assertEqual(_split_dims("128, 512"), ["128", "512"])
        self.assertEqual(_split_dims("max(1, D//tp), 16"), ["max(1, D//tp)", "16"])
        self.assertEqual(_split_dims("(batch, heads), 128"), ["(batch, heads)", "128"])

    def test_parse_shape_expr(self):
        self.assertEqual(
            _parse_shape_expr("(tokens, D//tp)", self.evaluator), (128, 1280)
        )
        self.assertEqual(_parse_shape_expr("(1,)", self.evaluator), (1,))
        self.assertEqual(_parse_shape_expr("()", self.evaluator), ())


# ── Shared utility tests ────


class TestParseShapeText(unittest.TestCase):
    def test_basic(self):
        result = parse_shape_text("136,7168;7168,3584")
        self.assertEqual(result, [(136, 7168), (7168, 3584)])

    def test_empty(self):
        self.assertEqual(parse_shape_text(""), [])
        self.assertEqual(parse_shape_text("N/A"), [])

    def test_roundtrip(self):
        original = "136,7168;7168,3584"
        parsed = parse_shape_text(original)
        rebuilt = build_shape_text(parsed)
        self.assertEqual(parse_shape_text(rebuilt), parsed)

    def test_build_shape_text_matches_database_style(self):
        self.assertEqual(
            build_shape_text([(136, 7168), (7168, 3584)]), "136,7168;7168,3584"
        )

    def test_build_shape_cell_keeps_database_quoting_style(self):
        self.assertEqual(build_shape_cell([(136, 7168)]), '"136,7168"')
        self.assertEqual(
            build_shape_cell([(136, 7168), (7168, 3584)]), '"136,7168;7168,3584"'
        )


class TestTheoryPadV3(unittest.TestCase):
    """Test PadV3 theory mode generation based on First Principles."""

    def test_pad_v3_alignment_logic(self):
        # Setup evaluator with sample variables
        evaluator = SafeExprEval({"tokens": 127, "D": 7168, "tp": 4})

        # Test input expression parsing (3-input structure)
        # inputs: ["(tokens, D)", "(4,)", "()"]
        input_exprs = ["(tokens, D)", "(4,)", "()"]
        parsed_inputs = [_parse_shape_expr(e, evaluator) for e in input_exprs]
        self.assertEqual(parsed_inputs, [(127, 7168), (4,), ()])

        # Test output alignment
        # outputs: ["(align(tokens, 8), D)"]
        output_exprs = ["(align(tokens, 8), D)"]
        parsed_outputs = [_parse_shape_expr(e, evaluator) for e in output_exprs]
        self.assertEqual(parsed_outputs, [(128, 7168)])

    def test_pad_v3_constraints(self):
        # tokens % 8 != 0 should allow 1, 7, 127 and block 8, 128
        evaluator_good = SafeExprEval({"tokens": 127})
        evaluator_bad = SafeExprEval({"tokens": 128})

        constraint = "tokens % 8 != 0"
        self.assertTrue(evaluator_good.eval(constraint))
        self.assertFalse(evaluator_bad.eval(constraint))

    def test_pad_v3_generation_mock(self):
        """End-to-end theory sub-logic for PadV3."""
        config = {
            "assignments": {
                "PadV3": {
                    "pattern": "PadV3",
                    "inputs": ["(tokens, D)", "(4,)", "()"],
                    "outputs": ["(align(tokens, 8), D)"],
                    "grids": {"tokens": [1, 8, 127], "D": [7168]},
                    "constraints": ["tokens % 8 != 0"],
                }
            }
        }
        # get_default_theory_generator uses model configs normally,
        # but the internal evaluator logic is what we want to check.
        # We'll mock the generator's behavior here.
        vars_list = [
            {"tokens": 1, "D": 7168},
            {"tokens": 8, "D": 7168},
            {"tokens": 127, "D": 7168},
        ]
        results = []
        for v in vars_list:
            ev = SafeExprEval(v)
            if all(ev.eval(c) for c in config["assignments"]["PadV3"]["constraints"]):
                inp = [
                    _parse_shape_expr(e, ev)
                    for e in config["assignments"]["PadV3"]["inputs"]
                ]
                out = [
                    _parse_shape_expr(e, ev)
                    for e in config["assignments"]["PadV3"]["outputs"]
                ]
                results.append((inp, out))

        # tokens=8 should be filtered out
        self.assertEqual(len(results), 2)
        # Check first result (tokens=1)
        self.assertEqual(results[0][0][0], (1, 7168))
        self.assertEqual(results[0][1][0], (8, 7168))
        # Check last result (tokens=127)
        self.assertEqual(results[1][0][0], (127, 7168))
        self.assertEqual(results[1][1][0], (128, 7168))


class ZTestMemoryEstimation(unittest.TestCase):
    """Test memory estimator and its integration with theory mode."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).parent.parent.parent
        cls.data_dir = (
            cls.root_dir
            / "tensor_cast/performance_model/profiling_database/data"
            / "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5"
        )
        cls.config_path = (
            cls.root_dir / "tools/perf_data_collection/grid_generator/config.yaml"
        )
        cls.budget_32g = 32 * 1024**3

    def test_dtype_to_bytes(self):
        self.assertEqual(dtype_to_bytes("DT_FLOAT"), 4)
        self.assertEqual(dtype_to_bytes("DT_INT8"), 1)
        self.assertEqual(dtype_to_bytes("dt_bf16"), 2)

    def test_estimate_row_memory(self):
        mem = estimate_row_memory(
            input_shapes=[(4096, 4096)],
            output_shapes=[(4096, 4096)],
            input_dtypes=["DT_FLOAT16"],
            output_dtypes=["DT_FLOAT16"],
        )
        expected = 4096 * 4096 * 2 * 2  # 2 tensors, 2 bytes each
        self.assertEqual(mem, expected)

    def test_exceeds_memory_budget(self):
        exceeded, _ = exceeds_memory_budget(
            [(131072, 131072)],
            [(131072, 131072)],
            ["DT_FLOAT16"],
            max_bytes=self.budget_32g,
        )
        self.assertTrue(exceeded)

    def test_theory_generation_dry_run(self):
        """Dry-run: verify theory generation with memory filtering for all defined kernels using dsv3/qwen332b."""
        # Suppress noise for a clean table output
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        if not self.config_path.exists():
            self.skipTest(f"Config path not found: {self.config_path}")

        config = load_shape_grid_config(self.config_path)
        op_meta = (
            load_op_mapping_metadata(self.data_dir) if self.data_dir.exists() else {}
        )

        # Test all kernels defined in the config assignments
        test_kernels = sorted(config.get("assignments", {}).keys())
        model_names = ["dsv3", "qwen332b"]
        failed_kernels = []
        all_total_rows = 0

        print(f"\n  [Dry-run] Models: {model_names} | DataDir: {self.data_dir.name}")
        print(f"    {'Operator':25} | {'Rows':>6} | {'Memory (MiB)':>22}")
        print("    " + "-" * 65)

        MOCK_DTYPES = {
            "QuantBatchMatmulV3": (["DT_INT8", "DT_INT8"], ["DT_INT32"]),
            "GroupedMatmulSwigluQuant": (["DT_INT8", "DT_INT8"], ["DT_BF16"]),
            "AscendQuantV2": (["DT_BF16"], ["DT_INT8"]),
            "DynamicQuant": (["DT_BF16"], ["DT_INT8", "DT_FLOAT"]),
        }

        for kernel in test_kernels:
            # Use redirect_stderr to catch direct prints from 3rd party libs during loading
            with contextlib.redirect_stderr(io.StringIO()):
                gen = get_default_theory_generator(kernel, model_names, config, op_meta)
                if not gen:
                    km = op_meta.get(kernel, {})
                    if (
                        km.get("zero_cost")
                        or km.get("composite")
                        or km.get("communication")
                    ):
                        continue
                    failed_kernels.append(f"{kernel} (No generator)")
                    continue
                all_rows = list(gen)

            total = len(all_rows)
            if total == 0:
                failed_kernels.append(f"{kernel} (0 rows)")
                continue

            all_total_rows += total

            # Infer dtypes natively using mock dict -> input shape fallback
            first_row = all_rows[0]
            if kernel in MOCK_DTYPES:
                input_dtypes, output_dtypes = MOCK_DTYPES[kernel]
            else:
                input_dtypes = ["DT_BF16"] * len(first_row.input_shapes)
                output_dtypes = ["DT_BF16"] * len(first_row.output_shapes)

            mems = []
            for row in all_rows:
                mem = estimate_row_memory(
                    row.input_shapes,
                    row.output_shapes,
                    input_dtypes,
                    output_dtypes or input_dtypes,
                )
                mems.append(mem)

            min_mb, max_mb = min(mems) / 1024**2, max(mems) / 1024**2
            print(f"    - {kernel:23} | {total:6d} | {min_mb:8.2f} ~ {max_mb:8.2f}")

        print("    " + "-" * 65)
        processed = len(test_kernels) - len(failed_kernels)
        print(f"    Summary: {processed}/{len(test_kernels)} processed.")
        self.assertEqual(
            len(failed_kernels),
            0,
            f"The following kernels failed to generate any rows: {failed_kernels}",
        )

    def test_m_grid_is_superset_of_baseline(self):
        """Ensure the new M_GRID is a strict superset of the baseline."""
        BASELINE_M_GRID = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            10,
            12,
            14,
            16,
            20,
            24,
            28,
            32,
            48,
            64,
            80,
            96,
            128,
            160,
            192,
            256,
            384,
            512,
            768,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
        ]

        missing = set(BASELINE_M_GRID) - set(M_GRID)
        self.assertEqual(
            missing, set(), f"M_GRID is missing values from the baseline: {missing}"
        )

    def test_quant_matmul_constraints_block_alignment(self):
        """Verify that all rows generated by the quant_matmul template meet block alignment constraints."""
        if not self.config_path.exists():
            self.skipTest("Config path not found")
        config = load_shape_grid_config(self.config_path)
        op_meta = (
            load_op_mapping_metadata(self.data_dir) if self.data_dir.exists() else {}
        )

        gen = get_default_theory_generator("QuantBatchMatmulV3", None, config, op_meta)
        if gen is None:
            self.skipTest("QuantBatchMatmulV3 generator not available")
        checked_rows = 0
        for row in gen:
            # input[0] = (M, K), output[0] = (M, N)
            if len(row.input_shapes) >= 2 and len(row.output_shapes) >= 1:
                nk_shape = row.input_shapes[1]
                if len(nk_shape) == 4:
                    checked_rows += 1
                    out_n = row.output_shapes[0][1]
                    out_k = row.input_shapes[0][1]
                    self.assertTrue(
                        out_n >= 128, f"N should be >= 128, but got {out_n}"
                    )
                    self.assertTrue(
                        out_k >= 128, f"K should be >= 128, but got {out_k}"
                    )
                    self.assertEqual(
                        out_n % 32, 0, f"N={out_n} should be divisible by block_w (32)"
                    )
                    self.assertEqual(
                        out_k % 16, 0, f"K={out_k} should be divisible by block_h (16)"
                    )
        self.assertGreater(
            checked_rows,
            0,
            "No rows matched the 4D weight shape filter — alignment assertions never ran",
        )


if __name__ == "__main__":
    unittest.main()

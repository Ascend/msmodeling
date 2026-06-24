# pylint: disable=no-name-in-module
import contextlib
import io
import logging
import os
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from unittest import mock
import warnings
from pathlib import Path

from tools.perf_data_collection.grid_generator import model_configs
from tools.perf_data_collection.grid_generator.config import (
    load_op_mapping_metadata,
    load_shape_grid_config,
)
from tools.perf_data_collection.grid_generator.generators import TheoryShapeRow
from tools.perf_data_collection.grid_generator.generators import fused_attention as fused_attention_module
from tools.perf_data_collection.grid_generator.evaluator import (
    SafeExprEval,
    _parse_shape_expr,
    _split_dims,
)
from tools.perf_data_collection.grid_generator.generators.fused_attention import (
    RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES,
    RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES,
    RUNTIME_AVG_SEQ_LEN,
    RUNTIME_BLOCK_TABLE_VALID_BLOCKS,
    RUNTIME_NUM_KEY_VALUE_HEADS,
    RUNTIME_SOURCE_PROFILE,
    _build_dense_prefill_row,
    _build_mla_decode_row,
    _build_mla_prefill_row,
    generate_fused_attention_rows,
)
from tools.perf_data_collection.grid_generator.generators.moe import (
    generate_dispatch_ffn_combine_rows,
)
from tools.perf_data_collection.grid_generator.generators.rope import (
    generate_split_qkv_rmsnorm_rope_rows,
)
from tools.perf_data_collection.grid_generator.shape_grids import M_GRID
from tools.perf_data_collection.grid_generator.theory_router import (
    collect_theory_generated_rows,
    get_default_theory_generator,
    generate_from_template,
)
from tools.perf_data_collection.grid_generator.utils import (
    align_shape_slot_count,
    build_generated_row,
    build_input_shapes_sort_key,
    build_row_template,
    build_shape_cell,
    build_shape_text,
    dedupe_generated_rows,
    extend_theory_headers,
    load_csv_template_rows,
    parse_shape_text,
    replace_csv_with_generated_rows,
    sort_generated_rows,
    zero_fill_column,
    _dedupe_key,
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

    def test_dedupe_generated_rows_ignores_duration_columns(self):
        headers = [
            "Input Shapes",
            "Input Data Types",
            "Input Formats",
            "Output Shapes",
            "Output Data Types",
            "Average Duration(us)",
            "MicroBench aiv_time(us)",
        ]
        source_rows = [
            {
                "Input Shapes": '"128,10240;128,10240;10240;10240"',
                "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16",
                "Input Formats": "ND;ND;ND;ND",
                "Output Shapes": '"128,10240;128,1;128,10240"',
                "Output Data Types": "DT_BF16;DT_FLOAT;DT_BF16",
                "Average Duration(us)": "17.840000",
                "MicroBench aiv_time(us)": "16.0",
            }
        ]
        generated_rows = [
            {
                **source_rows[0],
                "Average Duration(us)": "0",
                "MicroBench aiv_time(us)": "0",
            },
            {
                **source_rows[0],
                "Input Shapes": '"2048,384;2048,384;384;384"',
                "Output Shapes": '"2048,384;2048,1;2048,384"',
                "Average Duration(us)": "0",
                "MicroBench aiv_time(us)": "0",
            },
        ]

        deduped = dedupe_generated_rows(headers, source_rows, generated_rows)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["Input Shapes"], '"2048,384;2048,384;384;384"')
        self.assertEqual(self.evaluator.eval("align(7, 8)"), 8)
        self.assertEqual(self.evaluator.eval("align(8, 8)"), 8)
        self.assertEqual(self.evaluator.eval("align(9, 8)"), 16)
        # Dot access blocked by regex
        with self.assertRaises(ValueError):
            self.evaluator.eval("().__class__")

    def test_matmul_dedupe_uses_canonical_gemm_signature(self):
        headers = [
            "OP State",
            "Input Shapes",
            "Input Data Types",
            "Input Formats",
            "Output Shapes",
            "Output Data Types",
            "Average Duration(us)",
        ]
        source_rows = [
            {
                "OP State": "static",
                "Input Shapes": '"32,6144;2048,6144"',
                "Input Data Types": "DT_BF16;DT_BF16",
                "Input Formats": "ND;ND",
                "Output Shapes": '"32,2048"',
                "Output Data Types": "DT_BF16",
                "Average Duration(us)": "12.0",
            }
        ]
        generated_rows = [
            {
                "OP State": "dynamic",
                "Input Shapes": '"32,6144;6144,2048"',
                "Input Data Types": "DT_BF16;DT_BF16",
                "Input Formats": "ND;ND",
                "Output Shapes": '"32,2048"',
                "Output Data Types": "DT_BF16",
                "Average Duration(us)": "0",
            },
            {
                "OP State": "dynamic",
                "Input Shapes": '"48,6144;2048,6144"',
                "Input Data Types": "DT_BF16;DT_BF16",
                "Input Formats": "ND;ND",
                "Output Shapes": '"48,2048"',
                "Output Data Types": "DT_BF16",
                "Average Duration(us)": "0",
            },
        ]

        deduped = dedupe_generated_rows(headers, source_rows, generated_rows, Path("MatMulV2.csv"))

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["Input Shapes"], '"48,6144;2048,6144"')

    def test_replace_csv_keeps_original_when_permission_retry_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "review.csv"
            csv_path.write_text("A\nold\n", encoding="utf-8")
            with (
                mock.patch(
                    "tools.perf_data_collection.grid_generator.utils.os.replace",
                    side_effect=[
                        PermissionError("locked"),
                        None,
                        OSError("still locked"),
                    ],
                ),
                mock.patch("tools.perf_data_collection.grid_generator.utils.os.remove") as remove_mock,
            ):
                with self.assertRaises(OSError):
                    replace_csv_with_generated_rows(
                        csv_path,
                        ["A"],
                        [{"A": "new"}],
                        [],
                    )

            self.assertEqual(csv_path.read_text(encoding="utf-8"), "A\nold\n")
            remove_mock.assert_not_called()

    def test_dedupe_uses_microbench_profile_signature(self):
        headers = [
            "OP State",
            "Input Shapes",
            "Input Data Types",
            "Input Formats",
            "Output Shapes",
            "Output Data Types",
            "Average Duration(us)",
        ]
        source_rows = [
            {
                "OP State": "static",
                "Input Shapes": '"1024,12288;2;2"',
                "Input Data Types": "DT_BF16;INT32;INT32",
                "Input Formats": "ND;ND;ND",
                "Output Shapes": '"12288,1024"',
                "Output Data Types": "DT_BF16",
                "Average Duration(us)": "12.0",
            }
        ]
        generated_rows = [
            {
                "OP State": "dynamic",
                "Input Shapes": '"1024,12288;1;3"',
                "Input Data Types": "DT_BF16;INT32;INT32",
                "Input Formats": "ND;ND;ND",
                "Output Shapes": '"12288,1024"',
                "Output Data Types": "DT_BF16",
                "Average Duration(us)": "0",
            },
            {
                "OP State": "dynamic",
                "Input Shapes": '"512,12288;1;3"',
                "Input Data Types": "DT_BF16;INT32;INT32",
                "Input Formats": "ND;ND;ND",
                "Output Shapes": '"12288,512"',
                "Output Data Types": "DT_BF16",
                "Average Duration(us)": "0",
            },
        ]

        deduped = dedupe_generated_rows(headers, source_rows, generated_rows, Path("Transpose.csv"))

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["Input Shapes"], '"512,12288;1;3"')

    def test_split_dims(self):
        self.assertEqual(_split_dims("128, 512"), ["128", "512"])
        self.assertEqual(_split_dims("max(1, D//tp), 16"), ["max(1, D//tp)", "16"])
        self.assertEqual(_split_dims("(batch, heads), 128"), ["(batch, heads)", "128"])

    def test_parse_shape_expr(self):
        self.assertEqual(_parse_shape_expr("(tokens, D//tp)", self.evaluator), (128, 1280))
        self.assertEqual(_parse_shape_expr("(1,)", self.evaluator), (1,))
        self.assertEqual(_parse_shape_expr("()", self.evaluator), ())

    def test_template_pattern_can_override_metadata(self):
        pattern = {
            "iterators": {"tokens": [128]},
            "constants": {"hidden": 512, "rope_dim": 64},
            "inputs": [
                "(tokens, hidden + rope_dim*2)",
                "(max(tokens, 2048), rope_dim)",
                "(tokens,)",
            ],
            "outputs": ["(tokens, hidden)", "(tokens, rope_dim)"],
            "input_dtypes": ["DT_BF16", "DT_BF16", "DT_INT64"],
            "input_formats": ["ND", "ND", "ND"],
        }

        row = next(generate_from_template(pattern, None))

        self.assertEqual(row.input_shapes, [(128, 640), (2048, 64), (128,)])
        self.assertEqual(row.extra_values["Input Data Types"], "DT_BF16;DT_BF16;DT_INT64")
        self.assertEqual(row.extra_values["Input Formats"], "ND;ND;ND")

    def test_theory_rows_clear_absent_optional_input_shape_slots(self):
        headers = [
            "Input Shapes",
            "Input Data Types",
            "Input Formats",
            "Output Shapes",
            "Output Data Types",
            "Output Formats",
            "Average Duration(us)",
        ]
        source_rows = [
            {
                "Input Shapes": '"1,512;1,512;512;"',
                "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED",
                "Input Formats": "ND;ND;ND;NULL",
                "Output Shapes": '"1,512;1,1;1,512"',
                "Output Data Types": "DT_BF16;FLOAT;DT_BF16",
                "Output Formats": "ND;ND;ND",
                "Average Duration(us)": "1.0",
            }
        ]
        generated = iter(
            [
                TheoryShapeRow(
                    [(128, 512), (128, 512), (512,), (512,)],
                    [(128, 512), (128, 1), (128, 512)],
                )
            ]
        )

        rows = collect_theory_generated_rows(
            headers,
            source_rows,
            generated,
            csv_path=Path("AddRmsNormBias.csv"),
            file_index=1,
            total_files=1,
            max_rows=None,
            rng=None,
        )

        self.assertEqual(rows[0]["Input Shapes"], '"128,512;128,512;512;"')

    def test_theory_rows_keep_optional_input_shape_when_metadata_overrides_present(
        self,
    ):
        headers = [
            "Input Shapes",
            "Input Data Types",
            "Input Formats",
            "Output Shapes",
            "Output Data Types",
            "Output Formats",
            "Average Duration(us)",
        ]
        source_rows = [
            {
                "Input Shapes": '"1,512;1,512;512;"',
                "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED",
                "Input Formats": "ND;ND;ND;NULL",
                "Output Shapes": '"1,512;1,1;1,512"',
                "Output Data Types": "DT_BF16;FLOAT;DT_BF16",
                "Output Formats": "ND;ND;ND",
                "Average Duration(us)": "1.0",
            }
        ]
        generated = iter(
            [
                TheoryShapeRow(
                    [(128, 512), (128, 512), (512,), (512,)],
                    [(128, 512), (128, 1), (128, 512)],
                    extra_values={
                        "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16",
                        "Input Formats": "ND;ND;ND;ND",
                    },
                )
            ]
        )

        rows = collect_theory_generated_rows(
            headers,
            source_rows,
            generated,
            csv_path=Path("AddRmsNormBias.csv"),
            file_index=1,
            total_files=1,
            max_rows=None,
            rng=None,
        )

        self.assertEqual(rows[0]["Input Shapes"], '"128,512;128,512;512;512"')

    def test_theory_rows_cap_applies_without_rng(self):
        headers = [
            "Input Shapes",
            "Input Data Types",
            "Input Formats",
            "Output Shapes",
            "Output Data Types",
            "Output Formats",
            "Average Duration(us)",
        ]
        source_rows = [
            {
                "Input Shapes": '"1,512"',
                "Input Data Types": "DT_BF16",
                "Input Formats": "ND",
                "Output Shapes": '"1,512"',
                "Output Data Types": "DT_BF16",
                "Output Formats": "ND",
                "Average Duration(us)": "1.0",
            }
        ]
        generated = iter(
            [
                TheoryShapeRow([(1, 512)], [(1, 512)]),
                TheoryShapeRow([(2, 512)], [(2, 512)]),
                TheoryShapeRow([(3, 512)], [(3, 512)]),
            ]
        )

        rows = collect_theory_generated_rows(
            headers,
            source_rows,
            generated,
            csv_path=Path("Add.csv"),
            file_index=1,
            total_files=1,
            max_rows=2,
            rng=None,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["Input Shapes"], '"2,512"')

    def test_tensor_move_theory_rows_skip_small_unprofiled_copies(self):
        config = load_shape_grid_config(Path("tools/perf_data_collection/grid_generator/config.yaml"))
        generator = get_default_theory_generator("TensorMove", ["Qwen3-32B"], config, {})
        self.assertIsNotNone(generator)

        rows = list(generator)

        self.assertTrue(rows)
        for row in rows:
            tokens, hidden = row.input_shapes[0]
            self.assertGreaterEqual(tokens * hidden, 262144)
            self.assertEqual(row.input_shapes, row.output_shapes)


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
        self.assertEqual(build_shape_text([(136, 7168), (7168, 3584)]), "136,7168;7168,3584")
        self.assertEqual(build_shape_text([(1,), (), (2,), ()]), "1;;2;")

    def test_build_shape_cell_keeps_database_quoting_style(self):
        self.assertEqual(build_shape_cell([(136, 7168)]), '"136,7168"')
        self.assertEqual(build_shape_cell([(136, 7168), (7168, 3584)]), '"136,7168;7168,3584"')

    def test_align_shape_slot_count_keeps_extra_generated_slots(self):
        self.assertEqual(
            align_shape_slot_count([(1,), (2,)], [(1,), (2,), (3,)]),
            [(1,), (2,), (3,)],
        )
        self.assertEqual(
            align_shape_slot_count([(1,), (2,), (3,)], [(1,)]),
            [(1,), (), ()],
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
                inp = [_parse_shape_expr(e, ev) for e in config["assignments"]["PadV3"]["inputs"]]
                out = [_parse_shape_expr(e, ev) for e in config["assignments"]["PadV3"]["outputs"]]
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
        cls.config_path = cls.root_dir / "tools/perf_data_collection/grid_generator/config.yaml"
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
        """Dry-run: verify theory generation with memory filtering for all defined kernels using canonical model IDs."""
        # Suppress noise for a clean table output
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        if not self.config_path.exists():
            self.skipTest(f"Config path not found: {self.config_path}")

        config = load_shape_grid_config(self.config_path)
        op_meta = load_op_mapping_metadata(self.data_dir) if self.data_dir.exists() else {}

        # Test all kernels defined in the config assignments
        test_kernels = sorted(config.get("assignments", {}).keys())
        model_names = ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-32B"]
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
                    if config.get("assignments", {}).get(kernel) == "skip":
                        continue
                    km = op_meta.get(kernel, {})
                    if km.get("zero_cost") or km.get("composite") or km.get("communication"):
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

    def test_fused_attention_uses_stable_model_key_for_hf_id_config_names(self):
        """HF fetch success may keep cfg.name as the HF id; scene grids must still be model-specific."""
        qwen_hf_cfg = replace(model_configs.QWEN3_32B_CONFIG, name="Qwen/Qwen3-32B")
        dsv3_hf_cfg = replace(model_configs.DEEPSEEK_V3_CONFIG, name="deepseek-ai/DeepSeek-V3")

        with mock.patch.object(fused_attention_module, "resolve_configs", return_value=[qwen_hf_cfg]):
            qwen_rows = list(generate_fused_attention_rows(["Qwen/Qwen3-32B"]))
        qwen_sources = {row.extra_values[RUNTIME_SOURCE_PROFILE] for row in qwen_rows}
        self.assertIn("qwen332b_dense_prefill", qwen_sources)
        self.assertTrue(any(row.extra_values[RUNTIME_AVG_SEQ_LEN] == "4112.000000" for row in qwen_rows))

        with mock.patch.object(fused_attention_module, "resolve_configs", return_value=[dsv3_hf_cfg]):
            dsv3_rows = list(generate_fused_attention_rows(["deepseek-ai/DeepSeek-V3"]))
        dsv3_sources = {row.extra_values[RUNTIME_SOURCE_PROFILE] for row in dsv3_rows}
        self.assertIn("deepseekv3_mla_prefill", dsv3_sources)
        self.assertTrue(any(row.extra_values[RUNTIME_AVG_SEQ_LEN] == "4099.000000" for row in dsv3_rows))

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
        self.assertEqual(missing, set(), f"M_GRID is missing values from the baseline: {missing}")

    def test_dfc_rows_set_ep_size_from_expert_per_rank(self):
        """DFC generated rows must not inherit template EP Size blindly."""
        rows = list(generate_dispatch_ffn_combine_rows(["zai-org/GLM-5.1"]))
        self.assertGreater(len(rows), 0)
        for row in rows:
            expert_per_rank = row.output_shapes[1][0]
            self.assertEqual(row.extra_values["EP Size"], str(256 // expert_per_rank))

        ep32_rows = [row for row in rows if row.input_shapes[0][1] == 6144 and row.output_shapes[1] == (8,)]
        self.assertGreater(len(ep32_rows), 0)
        self.assertTrue(all(row.extra_values["EP Size"] == "32" for row in ep32_rows))

    def test_quant_matmul_constraints_block_alignment(self):
        """Verify that all rows generated by the quant_matmul template meet block alignment constraints."""
        if not self.config_path.exists():
            self.skipTest("Config path not found")
        config = load_shape_grid_config(self.config_path)
        op_meta = load_op_mapping_metadata(self.data_dir) if self.data_dir.exists() else {}

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
                    self.assertTrue(out_n >= 128, f"N should be >= 128, but got {out_n}")
                    self.assertTrue(out_k >= 128, f"K should be >= 128, but got {out_k}")
                    self.assertEqual(out_n % 32, 0, f"N={out_n} should be divisible by block_w (32)")
                    self.assertEqual(out_k % 16, 0, f"K={out_k} should be divisible by block_h (16)")
        self.assertGreater(
            checked_rows,
            0,
            "No rows matched the 4D weight shape filter — alignment assertions never ran",
        )

    def test_fia_dense_prefill_uses_cumulative_kv_lengths(self):
        row = _build_dense_prefill_row(
            scene_name="test_dense_prefill",
            batch=2,
            seq=256,
            num_heads=64,
            num_kv_heads=8,
            head_dim=128,
        )

        self.assertEqual(row.extra_values[RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES], "256,512")
        self.assertEqual(row.extra_values[RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES], "256,512")

    def test_fia_mla_decode_uses_replayable_raw_shapes(self):
        row = _build_mla_decode_row(
            scene_name="test_mla_decode",
            batch=2,
            avg_seq_len=2048,
            num_heads=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )

        self.assertEqual(row.input_shapes[0], (2, 128, 1, 512))
        self.assertEqual(row.input_shapes[14], (2, 16))
        self.assertEqual(row.input_shapes[24], (2, 128, 1, 64))
        self.assertEqual(row.output_shapes[0], (128, 2, 1, 512))
        self.assertEqual(row.extra_values[RUNTIME_BLOCK_TABLE_VALID_BLOCKS], "16,16")

    def test_fia_mla_prefill_uses_replayable_paged_rope_shapes(self):
        row = _build_mla_prefill_row(
            scene_name="test_mla_prefill",
            batch=4,
            seq=2048,
            num_heads=128,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
        )

        self.assertEqual(row.input_shapes[0], (4, 128, 2048, 512))
        self.assertEqual(row.input_shapes[14], (4, 16))
        self.assertEqual(row.input_shapes[24], (4, 128, 2048, 64))
        self.assertEqual(row.output_shapes[0], (128, 4, 2048, 512))
        self.assertEqual(
            row.extra_values[RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES],
            "2048,2048,2048,2048",
        )
        self.assertEqual(row.extra_values[RUNTIME_NUM_KEY_VALUE_HEADS], "1")
        self.assertEqual(row.extra_values[RUNTIME_BLOCK_TABLE_VALID_BLOCKS], "16,16,16,16")

    def test_fia_generated_rows_have_unique_profile_signatures(self):
        rows = list(generate_fused_attention_rows())
        signatures = {
            (
                tuple(row.input_shapes),
                row.extra_values.get("Input Data Types", ""),
                row.extra_values.get("Input Formats", ""),
                tuple(row.output_shapes),
            )
            for row in rows
        }

        self.assertEqual(len(rows), len(signatures))

    def test_split_qkv_uses_model_qkv_hidden_sizes(self):
        rows = list(generate_split_qkv_rmsnorm_rope_rows(["Qwen/Qwen3-32B"]))
        self.assertIn(
            TheoryShapeRow(
                [(1, 768), (128,)],
                [(1, 512), (1, 128), (1, 128)],
                extra_values={
                    "Input Data Types": "DT_BF16;DT_BF16",
                    "Input Formats": "ND;ND",
                    "Output Data Types": "DT_BF16;DT_BF16;DT_BF16",
                    "Output Formats": "ND;ND;ND",
                },
            ),
            rows,
        )
        for row in rows:
            input_hidden = row.input_shapes[0][1]
            q_hidden = row.output_shapes[0][1]
            kv_hidden = row.output_shapes[1][1]
            self.assertEqual(input_hidden, q_hidden + 2 * kv_hidden)
            self.assertEqual(row.output_shapes[1], row.output_shapes[2])

    def test_split_qkv_skips_mla_models(self):
        self.assertEqual(
            list(generate_split_qkv_rmsnorm_rope_rows(["deepseek-ai/DeepSeek-V3", "zai-org/GLM-5.1"])),
            [],
        )


class TestZeroFillColumn(unittest.TestCase):
    def test_duration_header(self):
        self.assertTrue(zero_fill_column("Average Duration(us)"))
        self.assertTrue(zero_fill_column("Min Duration(us)"))

    def test_latency_header(self):
        self.assertTrue(zero_fill_column("Latency(us)"))

    def test_time_header(self):
        self.assertTrue(zero_fill_column("MicroBench aiv_time(us)"))

    def test_cycles_header(self):
        self.assertTrue(zero_fill_column("Cycles"))

    def test_ratio_header(self):
        self.assertTrue(zero_fill_column("Ops/compute ratio"))

    def test_miss_header(self):
        self.assertTrue(zero_fill_column("Cache miss rate"))

    def test_utilization_header(self):
        self.assertTrue(zero_fill_column("Utilization(%)"))

    def test_non_matching_header(self):
        self.assertFalse(zero_fill_column("Input Shapes"))
        self.assertFalse(zero_fill_column("Output Shapes"))
        self.assertFalse(zero_fill_column("OP State"))


class TestBuildRowTemplate(unittest.TestCase):
    def test_keeps_keep_columns(self):
        headers = [
            "OP State",
            "Input Data Types",
            "Input Formats",
            "Output Data Types",
            "Output Formats",
            "Output Shapes",
            "Input Shapes",
            "Average Duration(us)",
        ]
        source = {
            "OP State": "static",
            "Input Data Types": "DT_BF16;DT_BF16",
            "Input Formats": "ND;ND",
            "Output Data Types": "DT_BF16",
            "Output Formats": "ND",
            "Output Shapes": '"128,5120"',
            "Input Shapes": '"128,5120;5120,128"',
            "Average Duration(us)": "12.5",
        }
        tmpl = build_row_template(headers, source)
        self.assertEqual(tmpl["OP State"], "static")
        self.assertEqual(tmpl["Output Shapes"], '"128,5120"')
        self.assertEqual(tmpl["Input Shapes"], "")

    def test_zeros_duration_columns(self):
        headers = ["Input Shapes", "Average Duration(us)", "Min Duration(us)"]
        source = {
            "Input Shapes": '"128,5120"',
            "Average Duration(us)": "12.5",
            "Min Duration(us)": "10.0",
        }
        tmpl = build_row_template(headers, source)
        self.assertEqual(tmpl["Average Duration(us)"], "0")
        self.assertEqual(tmpl["Min Duration(us)"], "0")


class TestBuildGeneratedRow(unittest.TestCase):
    def test_basic(self):
        headers = [
            "Input Shapes",
            "Output Shapes",
            "Input Data Types",
            "Average Duration(us)",
        ]
        source = {
            "Input Shapes": '"1,5120"',
            "Output Shapes": '"1,25600"',
            "Input Data Types": "DT_BF16",
            "Average Duration(us)": "10.0",
        }
        row = build_generated_row(
            headers,
            source,
            input_shapes=[(128, 5120), (5120, 25600)],
            output_shapes=[(128, 25600)],
        )
        self.assertIn("128,5120;5120,25600", row["Input Shapes"])
        self.assertIn("128,25600", row["Output Shapes"])
        self.assertEqual(row["Average Duration(us)"], "0")

    def test_with_extra_values(self):
        headers = ["Input Shapes", "Input Data Types", "Input Formats"]
        source = {
            "Input Shapes": '"1,5120"',
            "Input Data Types": "DT_BF16",
            "Input Formats": "ND",
        }
        row = build_generated_row(
            headers,
            source,
            input_shapes=[(128, 5120)],
            output_shapes=[],
            extra_values={"Input Data Types": "DT_INT8", "NewCol": "value"},
        )
        self.assertEqual(row["Input Data Types"], "DT_INT8")
        self.assertEqual(row["NewCol"], "value")


class TestExtendTheoryHeaders(unittest.TestCase):
    def test_adds_missing_headers(self):
        result = extend_theory_headers(["A", "B"], ["B", "C", "D"])
        self.assertEqual(result, ["A", "B", "C", "D"])

    def test_no_duplication(self):
        result = extend_theory_headers(["A"], ["A", "A"])
        self.assertEqual(result, ["A"])

    def test_empty_extra(self):
        result = extend_theory_headers(["A", "B"], [])
        self.assertEqual(result, ["A", "B"])


class TestSortGeneratedRows(unittest.TestCase):
    def test_sorts_by_input_shapes(self):
        rows = [
            {"Input Shapes": '"300,5120"', "val": "c"},
            {"Input Shapes": '"100,5120"', "val": "a"},
            {"Input Shapes": '"200,5120"', "val": "b"},
        ]
        sorted_rows = sort_generated_rows(rows)
        self.assertEqual(sorted_rows[0]["val"], "a")
        self.assertEqual(sorted_rows[1]["val"], "b")
        self.assertEqual(sorted_rows[2]["val"], "c")


class TestBuildInputShapesSortKey(unittest.TestCase):
    def test_single_shape(self):
        row = {"Input Shapes": '"128,5120"'}
        key = build_input_shapes_sort_key(row)
        self.assertEqual(key, ((128, 5120),))

    def test_multiple_shapes(self):
        row = {"Input Shapes": '"128,5120;5120,25600"'}
        key = build_input_shapes_sort_key(row)
        self.assertEqual(key, ((128, 5120), (5120, 25600)))


class TestDedupeKey(unittest.TestCase):
    def test_dedupe_on_non_latency_columns(self):
        headers = ["Input Shapes", "Average Duration(us)", "OP State"]
        row = {
            "Input Shapes": '"128,5120"',
            "Average Duration(us)": "12.5",
            "OP State": "static",
        }
        key = _dedupe_key(headers, row)
        self.assertIn('"128,5120"', key)
        self.assertIn("static", key)
        self.assertNotIn("12.5", key)

    def test_empty_cell(self):
        headers = ["Input Shapes"]
        row = {"Input Shapes": ""}
        key = _dedupe_key(headers, row)
        self.assertEqual(key, ("",))


class TestParseShapeTextEdgeCases(unittest.TestCase):
    def test_missing_shape_tokens(self):
        for token in ["N/A", "NA", "NULL", "NONE", "UNDEFINED"]:
            with self.subTest(token=token):
                self.assertEqual(parse_shape_text(token), [])

    def test_quoted_na(self):
        self.assertEqual(parse_shape_text('"N/A"'), [])

    def test_parenthesized_pair(self):
        result = parse_shape_text("(128, 5120)")
        self.assertEqual(result, [(128, 5120)])

    def test_mixed_semicolon_and_empty_slots(self):
        result = parse_shape_text("128,5120;;")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (128, 5120))
        self.assertEqual(result[1], ())


class TestLoadCsvTemplateRows(unittest.TestCase):
    def test_require_rows_raises_if_empty(self):
        import csv
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.csv"
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Input Shapes", "Duration(us)"])
                writer.writeheader()
            with self.assertRaises(ValueError):
                load_csv_template_rows(p, require_rows=True)

    def test_require_rows_false_returns_none_if_empty(self):
        import csv
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.csv"
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Input Shapes", "Duration(us)"])
                writer.writeheader()
            result = load_csv_template_rows(p, require_rows=False)
            self.assertIsNone(result)

    def test_loads_rows(self):
        import csv
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.csv"
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Input Shapes", "Extra"])
                writer.writeheader()
                writer.writerow({"Input Shapes": '"128,5120"', "Extra": "val"})
            headers, rows = load_csv_template_rows(p, require_rows=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Extra"], "val")

    def test_extra_headers_appended(self):
        import csv
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.csv"
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Input Shapes"])
                writer.writeheader()
                writer.writerow({"Input Shapes": '"128,5120"'})
            headers, _ = load_csv_template_rows(p, require_rows=True, extra_headers=["Runtime col", "Extra"])
            self.assertIn("Runtime col", headers)
            self.assertIn("Extra", headers)

    def test_missing_input_shapes_header(self):
        import csv
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "no_shapes.csv"
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["OP State"])
                writer.writeheader()
            result = load_csv_template_rows(p, require_rows=False)
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

"""Tests for grid_generator/config.py — config loading utilities."""

import tempfile
import unittest
from pathlib import Path

import yaml

from tools.perf_data_collection.grid_generator.config import (
    load_op_mapping_metadata,
    load_shape_grid_config,
)


class TestLoadShapeGridConfig(unittest.TestCase):
    def test_loads_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.yaml"
            p.write_text(
                "assignments:\n"
                "  MatMulV2:\n"
                "    pattern: MatMulFamily\n"
                "    models: [deepseek-ai/DeepSeek-V3, Qwen/Qwen3-32B]\n"
            )
            result = load_shape_grid_config(p)
            self.assertIn("assignments", result)
            self.assertEqual(result["assignments"]["MatMulV2"]["pattern"], "MatMulFamily")

    def test_empty_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.yaml"
            p.write_text("{}")
            result = load_shape_grid_config(p)
            self.assertEqual(result, {})


class TestLoadOpMappingMetadata(unittest.TestCase):
    def test_basic_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            mapping = {
                "operator_mappings": {
                    "aten.mm.default": {
                        "kernel_type": "MatMulV2",
                        "zero_cost": False,
                        "composite": False,
                    },
                    "aten.view.default": {
                        "kernel_type": "TransData",
                        "zero_cost": True,
                        "composite": False,
                    },
                    "tensor_cast.mla.default": {
                        "kernel_type": "FusedInferAttentionScore",
                        "composite": True,
                        "alternate_kernel_types": [
                            "BatchMatMulV2",
                            "TransposeBatchMatMul",
                        ],
                    },
                    "tensor_cast.all_reduce.default": {
                        "kernel_type": "hcom_allReduce_",
                        "category": "communication",
                        "query_mode": "hcom",
                    },
                }
            }
            with (datadir / "op_mapping.yaml").open("w") as f:
                yaml.dump(mapping, f)

            meta = load_op_mapping_metadata(datadir)
            self.assertIn("MatMulV2", meta)
            self.assertFalse(meta["MatMulV2"]["zero_cost"])
            self.assertFalse(meta["MatMulV2"]["composite"])

            self.assertIn("TransData", meta)
            self.assertTrue(meta["TransData"]["zero_cost"])

            self.assertIn("FusedInferAttentionScore", meta)
            self.assertTrue(meta["FusedInferAttentionScore"]["composite"])

            self.assertIn("BatchMatMulV2", meta)
            self.assertEqual(meta["BatchMatMulV2"]["alternates_of"], "FusedInferAttentionScore")

            self.assertIn("TransposeBatchMatMul", meta)

            self.assertIn("hcom_allReduce_", meta)
            self.assertTrue(meta["hcom_allReduce_"]["communication"])
            self.assertEqual(meta["hcom_allReduce_"]["query_mode"], "hcom")

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            meta = load_op_mapping_metadata(datadir)
            self.assertEqual(meta, {})

    def test_empty_mapping_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            with (datadir / "op_mapping.yaml").open("w") as f:
                yaml.dump({}, f)
            meta = load_op_mapping_metadata(datadir)
            self.assertEqual(meta, {})

    def test_skips_non_dict_entries(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            mapping = {
                "operator_mappings": {
                    "aten.skip": "just_a_string",
                    "aten.mm": {
                        "kernel_type": "MatMulV2",
                    },
                }
            }
            with (datadir / "op_mapping.yaml").open("w") as f:
                yaml.dump(mapping, f)

            meta = load_op_mapping_metadata(datadir)
            self.assertIn("MatMulV2", meta)
            self.assertNotIn("aten.skip", meta)

    def test_skips_no_kernel_type(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            mapping = {
                "operator_mappings": {
                    "aten.no_kt": {
                        "zero_cost": True,
                    },
                }
            }
            with (datadir / "op_mapping.yaml").open("w") as f:
                yaml.dump(mapping, f)

            meta = load_op_mapping_metadata(datadir)
            self.assertEqual(meta, {})


if __name__ == "__main__":
    unittest.main()

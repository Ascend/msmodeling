import unittest
from pathlib import Path

import yaml

CANN85_OP_MAPPING = (
    Path(__file__).resolve().parents[4]
    / "tensor_cast/performance_model/profiling_database/data"
    / "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5"
    / "op_mapping.yaml"
)


@unittest.skipIf(
    not CANN85_OP_MAPPING.exists(),
    "CANN 8.5 op_mapping.yaml not found",
)
class CompilePassOpMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CANN85_OP_MAPPING, encoding="utf-8") as f:
            full = yaml.safe_load(f)
        cls.mapping = full.get("operator_mappings", {})
        cls.torch_npu_ref = full.get("torch_npu_reference", {})

    def test_all_reduce_compile_pass_families_keep_expected_contract(self):
        """BF16/FP16 and quantized MC2 families should keep their distinct lookup contract."""
        matmul_entry = self.mapping.get("tensor_cast.matmul_all_reduce.default")
        self.assertIsNotNone(matmul_entry, "Missing matmul_all_reduce in op_mapping")
        self.assertTrue(matmul_entry.get("composite", False))
        self.assertIn("hcom_allReduce_", matmul_entry.get("sub_kernels", []))
        self.assertTrue(
            any(kernel.startswith("MatMul") for kernel in matmul_entry["sub_kernels"]),
            "matmul_all_reduce should keep at least one matmul compute kernel",
        )
        self.assertNotIn(
            "tc_input_count",
            matmul_entry,
            "matmul_all_reduce should not use quant-style tc_input_count truncation",
        )

        quant_mc2_ops = [
            "tensor_cast.static_quant_linear_all_reduce.default",
            "tensor_cast.static_quant_linear_int4_all_reduce.default",
            "tensor_cast.fp8_linear_all_reduce.default",
            "tensor_cast.mxfp4_linear_all_reduce.default",
        ]
        for op in quant_mc2_ops:
            with self.subTest(op=op):
                entry = self.mapping.get(op)
                self.assertIsNotNone(entry, f"Missing op '{op}' in op_mapping")
                self.assertTrue(entry.get("composite", False))
                self.assertIn("QuantBatchMatmulV3", entry.get("sub_kernels", []))
                self.assertIn("hcom_allReduce_", entry.get("sub_kernels", []))
                self.assertEqual(
                    entry.get("tc_input_count"),
                    2,
                    f"{op} should keep tc_input_count=2 for quant lookup truncation",
                )

    def test_mla_compile_pass_reuses_kv_rmsnorm_kernel_reference(self):
        """MLAPO should stay wired to the shipped KvRmsNormRopeCache kernel reference."""
        kv_entry = self.mapping.get("tensor_cast.kv_rmsnorm_rope_cache.default")
        self.assertIsNotNone(kv_entry, "Missing kv_rmsnorm_rope_cache in op_mapping")
        self.assertEqual(kv_entry.get("kernel_type"), "KvRmsNormRopeCache")

        mlapo_entry = self.mapping.get("tensor_cast.mlapo.default")
        self.assertIsNotNone(mlapo_entry, "Missing mlapo in op_mapping")
        self.assertTrue(mlapo_entry.get("composite", False))
        self.assertIn(kv_entry["kernel_type"], mlapo_entry.get("sub_kernels", []))

        torch_npu_entry = self.torch_npu_ref.get(kv_entry["kernel_type"])
        self.assertIsNotNone(
            torch_npu_entry,
            "KvRmsNormRopeCache should have a torch_npu_reference entry",
        )
        self.assertEqual(
            torch_npu_entry.get("microbench_api"),
            "torch_npu.npu_kv_rmsnorm_rope_cache",
        )
        self.assertIn("aclnnKvRmsNormRopeCache", torch_npu_entry.get("aclnn", []))
        self.assertIn("aclnnKvRmsNormRopeCacheV2", torch_npu_entry.get("aclnn", []))


if __name__ == "__main__":
    unittest.main()

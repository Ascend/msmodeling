"""Smoke guard for throughput_optimizer CLI nightly regressions.

Nightly coverage mapping
------------------------
test_prefix_cache_hit_rate_aggregation_valid      - already present before this change
test_prefix_cache_hit_rate_disaggregation_*_valid - already present before this change
test_vl_model_image_args                          -> TestThroughputOptimizerNightly.test_vl_model_aggregation_with_output_validation
test_vl_disagg_prefill_smoke                      -> TestThroughputOptimizerNightly.test_vl_model_disaggregation_prefill_with_output_validation
test_vl_disagg_decode_smoke                       -> TestThroughputOptimizerNightly.test_vl_model_disaggregation_decode_with_output_validation
test_vl_moe_aggregation_compile_smoke             -> TestThroughputOptimizerNightly.test_VL_MOE_model_aggregation_with_output_validation
test_prefix_cache_with_max_batched_tokens_allows_chunked_prefill -> TestThroughputOptimizerNightly
                                                                    (test_prefix_cache_hit_rate_allows_chunked_prefill_when_effective_input_exceeds_max_batched_tokens)
test_deepseek_pd_ratio_mode                       -> TestThroughputOptimizerNightly
                                                     (test_deepseek_model_pd_ratio_with_output_validation)
"""

from unittest import TestCase

import pytest

from tests.helpers.cli_runner import run_module_main

THROUGHPUT_OPTIMIZER_MODULE = "cli.inference.throughput_optimizer"


class TestThroughputOptimizerSmoke(TestCase):
    def _run_throughput_optimizer(self, args, check=True):
        result = run_module_main(THROUGHPUT_OPTIMIZER_MODULE, args)
        if check and result.returncode != 0:
            raise RuntimeError(f"throughput_optimizer failed (rc={result.returncode}): {result.stderr}")
        return result

    def test_prefix_cache_hit_rate_aggregation_valid(self):
        args = [
            "--input-length=64",
            "--output-length=16",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=1",
            "--jobs=1",
            "--tpot-limits=1000",
            "--batch-range",
            "1",
            "2",
            "--prefix-cache-hit-rate=0.5",
        ]

        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_prefix_cache_hit_rate_disaggregation_prefill_valid(self):
        args = [
            "--input-length=64",
            "--output-length=16",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=1",
            "--jobs=1",
            "--ttft-limits=1000",
            "--batch-range",
            "1",
            "2",
            "--prefix-cache-hit-rate=0.5",
            "--disagg",
        ]

        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_prefix_cache_hit_rate_disaggregation_decode_valid(self):
        args = [
            "--input-length=64",
            "--output-length=16",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=1",
            "--jobs=1",
            "--tpot-limits=1000",
            "--batch-range",
            "1",
            "2",
            "--prefix-cache-hit-rate=0.5",
            "--disagg",
        ]

        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_vl_model_image_args(self):
        """VL model aggregation with image args; guards test_vl_model_aggregation_with_output_validation."""
        args = [
            "--input-length=64",
            "--output-length=16",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "--device=TEST_DEVICE",
            "--num-devices=4",
            "--jobs=1",
            "--tpot-limits=10000",
            "--batch-range",
            "1",
            "2",
            "--image-height=224",
            "--image-width=224",
        ]
        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_vl_disagg_prefill_smoke(self):
        """VL disagg prefill; guards test_vl_model_disaggregation_prefill_with_output_validation."""
        args = [
            "--input-length=64",
            "--output-length=16",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "--device=TEST_DEVICE",
            "--num-devices=4",
            "--jobs=1",
            "--ttft-limits=10000",
            "--batch-range",
            "1",
            "2",
            "--image-height=224",
            "--image-width=224",
            "--disagg",
        ]
        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    @pytest.mark.nightly
    def test_vl_disagg_decode_smoke(self):
        """VL disagg decode; guards test_vl_model_disaggregation_decode_with_output_validation."""
        args = [
            "--input-length=64",
            "--output-length=16",
            "zai-org/GLM-4.5V",
            "--device=TEST_DEVICE",
            "--num-devices=4",
            "--jobs=1",
            "--tpot-limits=10000",
            "--image-height=224",
            "--image-width=224",
            "--disagg",
        ]
        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_vl_moe_aggregation_compile_smoke(self):
        """VL MOE + compile aggregation; guards test_VL_MOE_model_aggregation_with_output_validation.

        Uses Qwen3-VL-30B (not nightly 235B) to keep PR smoke under time budget.
        """
        args = [
            "--input-length=16",
            "--output-length=8",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "--device=TEST_DEVICE",
            "--num-devices=2",
            "--jobs=1",
            "--tpot-limits=10000",
            "--image-height=224",
            "--image-width=224",
            "--compile",
            "--quantize-linear-action=W8A8_DYNAMIC",
            "--batch-range",
            "1",
            "1",
            "--max-batched-tokens=16",
        ]
        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_prefix_cache_with_max_batched_tokens_allows_chunked_prefill(self):
        """prefix-cache hit-rate + max-batched-tokens can use chunked prefill.

        Guards test_prefix_cache_hit_rate_allows_chunked_prefill_when_effective_input_exceeds_max_batched_tokens.
        With input_length=200 and prefix_cache_hit_rate=0.5, effective_input_length=100.
        max_batched_tokens=99 < 100, so the CLI should model two prefill chunks.
        """
        args = [
            "--input-length=200",
            "--output-length=16",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=1",
            "--jobs=1",
            "--tpot-limits=1000",
            "--batch-range",
            "1",
            "2",
            "--prefix-cache-hit-rate=0.5",
            "--max-batched-tokens=99",
        ]
        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_deepseek_pd_ratio_mode(self):
        """PD-ratio optimization mode; guards DeepSeek PD-ratio nightly regression."""
        args = [
            "--input-length=64",
            "--output-length=16",
            "deepseek-ai/DeepSeek-V3.1",
            "--enable-optimize-prefill-decode-ratio",
            "--prefill-devices-per-instance=4",
            "--decode-devices-per-instance=4",
            "--device=TEST_DEVICE",
            "--jobs=1",
            "--ttft-limits=10000",
            "--tpot-limits=10000",
        ]
        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

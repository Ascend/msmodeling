import subprocess
import sys
from unittest import TestCase


class TestThroughputOptimizerSmoke(TestCase):
    def _run_throughput_optimizer(self, args, check=True):
        cmd = [sys.executable, "-m", "cli.inference.throughput_optimizer"] + args
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

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

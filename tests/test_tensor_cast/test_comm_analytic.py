import unittest

import torch

from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime


class CommAnalyticTestCase(unittest.TestCase):
    def test_all_to_all_excludes_local_chunk_from_network_bytes(self):
        x = torch.randn([16, 8], device="meta", dtype=torch.float16)
        perf_model = AnalyticPerformanceModel(TEST_DEVICE)
        with (
            Runtime(perf_model, TEST_DEVICE) as runtime,
            torch.no_grad(),
        ):
            torch.ops.tensor_cast.all_to_all(
                x,
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                0,
                [0, 1, 2, 3],
            )

        stats = runtime.event_list[0].perf_results["analytic"].statistics
        self.assertEqual(stats["total_bytes_sent"], 192)
        self.assertEqual(stats["total_bytes_received"], 192)
        self.assertEqual(stats["message_size_bytes"], 192)

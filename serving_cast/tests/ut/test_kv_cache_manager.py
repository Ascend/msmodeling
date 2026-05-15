# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

from serving_cast.config import Config
from serving_cast.kv_cache_manager import KVCacheManager  # Implementation file above


class TestKVCacheManager(unittest.TestCase):
    BLOCK_SIZE = 128  # Keep consistent with implementation
    NUM_BLOCKS = 10

    def setUp(self) -> None:
        # Create a new manager with 10 blocks before each test case
        self.mgr = KVCacheManager(num_blocks=self.NUM_BLOCKS, block_size=self.BLOCK_SIZE)
        self.mock_cfg = Mock()
        self.mock_cfg.enable_profiling = False

        self.patch_get = patch.object(Config, "get_instance")
        mock_get = self.patch_get.start()
        mock_get.return_value = self.mock_cfg

    def tearDown(self):
        self.patch_get.stop()

    # ---------- Basic functionality ----------
    def test_allocate_once(self):
        new_blocks = self.mgr.allocate_slots(request_id=1, num_new_tokens=64)
        self.assertEqual(new_blocks, [9])
        self.assertEqual(self.mgr.used_slots_in_request(1), 64)
        self.assertEqual(self.mgr.stats()["used_blocks"], 1)

    def test_allocate_exact_block(self):
        new_blocks = self.mgr.allocate_slots(request_id=2, num_new_tokens=self.BLOCK_SIZE)
        self.assertEqual(new_blocks, [9])
        self.assertEqual(self.mgr.used_slots_in_request(2), self.BLOCK_SIZE)
        self.assertEqual(self.mgr.stats()["used_blocks"], 1)

    def test_allocate_multiple_blocks(self):
        new_blocks = self.mgr.allocate_slots(request_id=3, num_new_tokens=300)
        self.assertEqual(new_blocks, [9, 8, 7])
        self.assertEqual(self.mgr.used_slots_in_request(3), 300)
        self.assertEqual(self.mgr.stats()["used_blocks"], 3)

    # ---------- Continuous allocation and reuse of tail block ----------
    def test_reuse_tail_block(self):
        # First time 64 slots
        self.mgr.allocate_slots(request_id=100, num_new_tokens=64)
        # Second time 192 slots
        new_blocks = self.mgr.allocate_slots(request_id=100, num_new_tokens=192)
        # Expected: only add 1 new block, total 2 blocks
        self.assertEqual(new_blocks, [8])  # Only allocated block 1
        self.assertEqual(self.mgr.used_slots_in_request(100), 256)
        self.assertEqual(self.mgr.stats()["used_blocks"], 2)

    # ---------- Failure scenarios ----------
    def test_insufficient_blocks(self):
        # Occupy all blocks first
        need = 10 * self.BLOCK_SIZE
        self.mgr.allocate_slots(request_id=999, num_new_tokens=need)
        self.assertEqual(self.mgr.stats()["free_blocks"], 0)
        old_state = self.mgr.stats()

        # Applying for 1 more slot should fail
        res = self.mgr.allocate_slots(request_id=998, num_new_tokens=1)
        self.assertEqual(res, None)
        new_state = self.mgr.stats()
        self.assertEqual(old_state, new_state)

    # ---------- Freeing ----------
    def test_free(self):
        self.mgr.allocate_slots(request_id=77, num_new_tokens=200)
        self.assertEqual(self.mgr.stats()["used_blocks"], 2)

        self.mgr.free(request_id=77)
        self.assertEqual(self.mgr.stats()["used_blocks"], 0)
        self.assertEqual(self.mgr.stats()["free_blocks"], 10)
        # Freeing again should not cause an error
        self.mgr.free(request_id=77)

    def test_free_partial_then_reuse(self):
        self.mgr.allocate_slots(request_id=88, num_new_tokens=50)
        self.mgr.allocate_slots(request_id=88, num_new_tokens=50)
        self.assertEqual(self.mgr.used_slots_in_request(88), 100)
        self.assertEqual(self.mgr.stats()["used_blocks"], 1)

        self.mgr.free(request_id=88)
        self.assertEqual(self.mgr.stats()["used_blocks"], 0)

        # Re-applying should start from the beginning
        new_blocks = self.mgr.allocate_slots(request_id=99, num_new_tokens=300)
        self.assertEqual(new_blocks, [9, 8, 7])

    def test_two_requests_no_reuse(self):
        # First time: request_id=1 applies for 64 slots
        new1 = self.mgr.allocate_slots(request_id=1, num_new_tokens=64)
        self.assertEqual(new1, [9])
        self.assertEqual(self.mgr.used_slots_in_request(1), 64)

        # Second time: request_id=2 applies for 192 slots
        new2 = self.mgr.allocate_slots(request_id=2, num_new_tokens=192)
        self.assertEqual(new2, [8, 7])  # Need 2 whole blocks
        self.assertEqual(self.mgr.used_slots_in_request(2), 192)

        # Overall statistics
        self.assertEqual(self.mgr.stats()["used_blocks"], 3)  # 0,1,2
        self.assertEqual(self.mgr.stats()["free_blocks"], 7)

    def test_same_request_fist_success_second_failed(self):
        # First time: request_id=1 applies for 64 slots
        new1 = self.mgr.allocate_slots(request_id=1, num_new_tokens=64)
        self.assertEqual(new1, [9])
        self.assertEqual(self.mgr.used_slots_in_request(1), 64)

        # Second time: request_id=1 applies for 200000 slots
        new2 = self.mgr.allocate_slots(request_id=1, num_new_tokens=200000)
        self.assertEqual(new2, None)
        self.assertEqual(self.mgr.used_slots_in_request(1), 64)


if __name__ == "__main__":
    unittest.main()

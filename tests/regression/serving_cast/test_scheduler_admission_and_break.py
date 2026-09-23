# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest
from unittest.mock import Mock, patch

import serving_cast.stime as stime
from serving_cast.config import Config
from serving_cast.engine import BatchScheduler
from serving_cast.kv_cache_manager import KVCacheManager
from serving_cast.request import Request, RequestState


class TestSchedulerAdmissionAndBreak(unittest.TestCase):
    BLOCK_SIZE = 128

    def setUp(self):
        stime.init_simulation()
        self.mock_cfg = Mock()
        self.mock_cfg.common_config.serving_config.max_concurrency = 100
        self.mock_cfg.common_config.serving_config.block_size = self.BLOCK_SIZE
        self.mock_cfg.common_config.serving_config.max_tokens_budget = 8192
        self.mock_cfg.common_config.model_config.enable_preprocessing_modeling = False
        self.mock_cfg.common_config.model_config.enable_kv_transfer_modeling = True
        self.mock_cfg.common_config.model_config.num_mtp_tokens = 0
        self.mock_cfg.common_config.model_config.mtp_acceptance_rate = []
        self.mock_cfg.enable_profiling = False
        self.patch_get_instance = patch.object(Config, "get_instance")
        mock_get_instance = self.patch_get_instance.start()
        mock_get_instance.return_value = self.mock_cfg

    def tearDown(self):
        self.patch_get_instance.stop()

    def _make_scheduler(self, num_blocks, block_size=None):
        block_size = block_size or self.BLOCK_SIZE
        kv_manager = KVCacheManager(num_blocks, block_size=block_size)
        model_runner = Mock()
        communication_manager = Mock()
        return BatchScheduler(model_runner, kv_manager, communication_manager)

    # ---------- #338: admission rejects physically-unschedulable requests ----------

    def test_add_rejects_request_exceeding_kv_pool(self):
        # 1 block x 128 = 128 tokens total; request needs 500032 -> unschedulable
        scheduler = self._make_scheduler(num_blocks=1)
        oversize = Request(num_input_tokens=500000, num_output_tokens=32)
        with self.assertRaises(ValueError) as ctx:
            scheduler.add(oversize)
        msg = str(ctx.exception)
        self.assertIn("exceeding total KV pool capacity", msg)
        self.assertIn("500032", msg)
        # must not be enqueued nor tracked
        self.assertNotIn(oversize.id, scheduler.requests)
        self.assertEqual(scheduler.waiting_queue, [])

    def test_add_accepts_request_within_kv_pool(self):
        # 10 blocks x 128 = 1280 tokens; request needs 562 -> schedulable
        scheduler = self._make_scheduler(num_blocks=10)
        req = Request(num_input_tokens=512, num_output_tokens=50)
        scheduler.add(req)
        self.assertIn(req.id, scheduler.requests)
        self.assertEqual(scheduler.waiting_queue, [req])

    def test_add_accepts_request_exactly_at_kv_pool_boundary(self):
        # 1 block x 128 = 128 tokens; request needs exactly 128 -> fits (boundary, not >)
        scheduler = self._make_scheduler(num_blocks=1)
        req = Request(num_input_tokens=100, num_output_tokens=28)  # 128 == capacity
        scheduler.add(req)
        self.assertIn(req.id, scheduler.requests)

    # ---------- #335: _schedule breaks (not continues) on KVS receive failure ----------

    def test_schedule_breaks_not_continues_on_kvs_receive_failure(self):
        # Force the KVS-transferring head to fail receiving remote KV. With the fix,
        # _schedule breaks and returns; with the old `continue` it would busy-loop
        # forever. The mock side_effect bounds the loop so a regression fails fast
        # instead of hanging CI.
        scheduler = self._make_scheduler(num_blocks=1)
        request = Request(num_input_tokens=200, num_output_tokens=10)
        request.need_kv_transfer = True
        request.kv_transfer_done = False
        request.state = RequestState.KVS_TRANSFERRING
        # Bypass add() (which would reject the oversize request) to directly exercise
        # the _schedule break branch on a KVS-receive failure.
        scheduler.waiting_queue.append(request)
        scheduler.requests[request.id] = request

        with patch.object(
            scheduler,
            "_receive_remote_kvs",
            side_effect=[
                False,
                False,
                False,
                AssertionError("_schedule did not break on KVS receive failure (continue busy-loop)"),
            ],
        ) as mock_receive:
            scheduler._schedule()

        # break: exactly one receive attempt, then _schedule returned without hanging
        self.assertEqual(mock_receive.call_count, 1)
        # request stays in waiting (not consumed / not scheduled into running)
        self.assertIn(request, scheduler.waiting_queue)
        self.assertEqual(scheduler.running_queue, [])


if __name__ == "__main__":
    unittest.main()

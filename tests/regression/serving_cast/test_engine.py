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
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serving_cast.engine import BatchScheduler
from serving_cast.instance import Instance
from serving_cast.kv_cache_manager import KVCacheManager
from serving_cast.request import Request, RequestState


class TestBatchScheduler(unittest.TestCase):
    @staticmethod
    def _scheduler() -> BatchScheduler:
        scheduler = object.__new__(BatchScheduler)
        scheduler.kv_manager = KVCacheManager(num_blocks=2, block_size=16)
        scheduler.waiting_queue = []
        scheduler.running_queue = []
        scheduler.requests = {}
        scheduler.max_tokens_budget = 3
        scheduler.decode_query_len = 3
        scheduler.average_decode_tokens = 1.5
        scheduler.notify = Mock()
        return scheduler

    def test_decode_seq_and_kv_progress_only_by_accepted_tokens(self):
        scheduler = self._scheduler()
        request = Request(id=1, num_input_tokens=10, num_output_tokens=10)
        request.state = RequestState.DECODING
        request.seq_len = 10
        request.num_decoded_tokens = 4
        request._expected_num_decoded_tokens = 4
        request.prepare_decode_step(query_len=3, average_tokens=1.5)
        scheduler.running_queue.append(request)
        scheduler.requests[request.id] = request

        batch = scheduler._schedule()
        self.assertEqual(request.query_len, 3)
        self.assertEqual(request.seq_len, 13)
        self.assertEqual(scheduler.kv_manager.used_slots_in_request(request.id), 1)

        scheduler._postprocess_batch(batch)

        self.assertEqual(request.num_decoded_tokens, 5)
        self.assertEqual(request.seq_len, 11)
        self.assertEqual(scheduler.kv_manager.used_slots_in_request(request.id), 1)
        self.assertEqual(request.state, RequestState.DECODING)

    def test_single_output_token_request_records_first_token_timestamp(self):
        scheduler = self._scheduler()
        request = Request(id=11, num_input_tokens=16, num_output_tokens=1)
        request.state = RequestState.PREFILLING
        request.num_current_max_new_tokens = 0
        request.leaves_client_time = 1.0
        request.arrives_server_time = 1.5
        scheduler.running_queue.append(request)
        scheduler.requests[request.id] = request
        scheduler.kv_manager.allocate_slots(request.id, 16)

        with patch("serving_cast.stime.now", return_value=2.5):
            scheduler._postprocess_batch([request])

        self.assertEqual(request.state, RequestState.DECODE_DONE)
        self.assertTrue(request.prefill_done_time_already_recorded)
        self.assertEqual(request.prefill_done_time, 2.5)
        self.assertEqual(request.decode_done_time, 2.5)
        # TTFT must be non-negative and equal to E2E for single output token
        self.assertEqual(request.client_time_to_first_token(), 1.5)
        self.assertEqual(request.server_time_to_first_token(), 1.0)
        self.assertEqual(request.serving_time(), request.client_time_to_first_token())
        self.assertEqual(request.time_per_output_token(), 0)
        self.assertNotIn(request.id, scheduler.requests)
        self.assertEqual(scheduler.running_queue, [])

    def test_kvs_transferring_request_allocates_input_kv_on_decode_side(self):
        # After PD handoff, the decode-side scheduler must run the KV-receive
        # branch: allocate num_input_tokens slots for the transferred KV before
        # the first decode step, matching the aggregated instance's accounting
        # (input tokens + one slot per generated decode token).
        scheduler = self._scheduler()
        request = Request(id=12, num_input_tokens=10, num_output_tokens=3)
        # simulate the prefill side's handoff: decode step already prepared
        request.prepare_decode_step(query_len=3, average_tokens=1.5)
        request.need_kv_transfer = True
        request.kv_transfer_done = False
        request.state = RequestState.KVS_TRANSFERRING
        scheduler.waiting_queue.append(request)
        scheduler.requests[request.id] = request

        batch = scheduler._schedule()

        self.assertTrue(request.kv_transfer_done)
        self.assertEqual(request.state, RequestState.DECODING)
        # 10 input slots (received KV) + 1 slot for the first decode step
        self.assertEqual(scheduler.kv_manager.used_slots_in_request(request.id), 11)
        self.assertIn(request, scheduler.running_queue)
        self.assertEqual(batch, [request])

    def test_duplicate_active_request_id_is_rejected_before_queue_mutation(self):
        scheduler = self._scheduler()
        first = Request(id=7)
        duplicate = Request(id=7)

        scheduler.add(first)
        scheduler.kv_manager.allocate_slots(first.id, 4)
        with self.assertRaisesRegex(ValueError, "active request ID 7 already exists"):
            scheduler.add(duplicate)

        self.assertEqual(scheduler.waiting_queue, [first])
        self.assertIs(scheduler.requests[7], first)
        self.assertEqual(scheduler.kv_manager.used_slots_in_request(7), 4)
        scheduler.notify.assert_called_once_with()

    def test_scheduling_loop_profiles_queue_snapshots(self):
        scheduler = self._scheduler()
        running_request = Request(id=8)
        waiting_request = Request(id=9)
        scheduler.running_queue = [running_request]
        scheduler.waiting_queue = [waiting_request]
        scheduler.enable_preprocessing_modeling = False
        scheduler.communication_manager = Mock()
        scheduler.model_runner = Mock()
        scheduler.model_runner.process_batch.side_effect = RuntimeError("stop scheduling loop")

        def schedule():
            scheduler.running_queue.append(waiting_request)
            scheduler.waiting_queue.remove(waiting_request)
            return [waiting_request]

        scheduler._schedule = Mock(side_effect=schedule)
        profiler = Mock()
        profiler.domain.return_value = profiler
        profiler.span_start.return_value = profiler

        with (
            patch("serving_cast.engine.Config.get_instance", return_value=SimpleNamespace(enable_profiling=True)),
            patch("serving_cast.engine.profiler_interface.is_profiling_ready", return_value=True),
            patch("serving_cast.engine.profiler_interface.Level", SimpleNamespace(INFO="INFO")),
            patch("serving_cast.engine.profiler_interface.SimProfiler", return_value=profiler),
            patch(
                "serving_cast.engine.profiler_interface.get_iter_size_info",
                return_value=[{"iter_size": 0}],
            ),
            patch("serving_cast.engine.profiler_interface.get_batch_type", return_value="Prefill"),
            patch("serving_cast.engine.profiler_interface.queue_profiler") as queue_profiler,
            self.assertRaisesRegex(RuntimeError, "stop scheduling loop"),
        ):
            scheduler._scheduling_loop()

        running_before, running_after, _ = queue_profiler.call_args_list[0].args
        waiting_before, waiting_after, _ = queue_profiler.call_args_list[1].args
        self.assertEqual(running_before, [running_request])
        self.assertEqual(running_after, [running_request, waiting_request])
        self.assertEqual(waiting_before, [waiting_request])
        self.assertEqual(waiting_after, [])
        self.assertIsNot(running_before, running_after)
        self.assertIsNot(waiting_before, waiting_after)


class TestInstanceHandleKvHandoff(unittest.TestCase):
    @staticmethod
    def _instance() -> Instance:
        instance = object.__new__(Instance)
        instance.id = 0
        engine = Mock()
        instance.load_balancer = Mock()
        instance.load_balancer.select.return_value = engine
        return instance, engine

    def test_kvs_transferring_state_is_preserved_until_kv_received(self):
        # Instance.handle must NOT eagerly transition KVS_TRANSFERRING -> DECODING:
        # BatchScheduler's receive branch (_receive_remote_kvs) only fires for
        # KVS_TRANSFERRING requests and performs the transition after allocating
        # the input KV slots on the decode side.
        instance, engine = self._instance()
        request = Request(id=21, num_input_tokens=256, num_output_tokens=3)
        request.state = RequestState.KVS_TRANSFERRING

        instance.handle(request)

        self.assertEqual(request.state, RequestState.KVS_TRANSFERRING)
        engine.handle.assert_called_once_with(request)

    def test_arrives_server_still_transitions_to_prefilling(self):
        instance, engine = self._instance()
        request = Request(id=22, num_input_tokens=16, num_output_tokens=2)
        request.state = RequestState.ARRIVES_SERVER

        instance.handle(request)

        self.assertEqual(request.state, RequestState.PREFILLING)
        engine.handle.assert_called_once_with(request)


if __name__ == "__main__":
    unittest.main()

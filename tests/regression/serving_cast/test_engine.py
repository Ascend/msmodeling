# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serving_cast.engine import BatchScheduler
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


if __name__ == "__main__":
    unittest.main()
